from __future__ import annotations

import logging
import time
from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from tqdm import tqdm

from src.training.trainer import MAETrainer

if TYPE_CHECKING:
    from src.models.mae import MaskedAutoencoder
    from src.utils.config import FKMAEConfig

logger = logging.getLogger(__name__)


class FKMAETrainer(MAETrainer):
    """Trainer for MAE pre-training on FK spectra.

    Inherits the full training loop, AMP, gradient accumulation,
    checkpointing, and scheduling from :class:`MAETrainer`. Overrides
    only per-epoch progress reporting (tqdm) and FK-specific
    visualisation.
    """

    def __init__(
        self,
        model: MaskedAutoencoder,
        config: FKMAEConfig,
        device: torch.device,
        train_loader: Any,
        val_loader: Any,
        checkpoint_dir: Path,
        run_dir: Path,
        resume_from: Path | None = None,
        argv: list[str] | None = None,
    ) -> None:
        """Initialise the FK MAE trainer.

        Args:
            model: MAE model instance.
            config: Training configuration.
            device: Torch device.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            checkpoint_dir: Directory to write checkpoints.
            run_dir: Root directory for the current run (logs, plots).
            resume_from: Optional checkpoint path to resume from.
            argv: Command-line arguments for reproducibility logging.
        """
        super().__init__(
            model=model,
            config=config,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=checkpoint_dir,
            run_dir=run_dir,
            resume_from=resume_from,
            argv=argv,
        )

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        """Run one training epoch with tqdm progress reporting.

        Args:
            epoch: 0-based epoch index.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            leave=False,
        )

        for batch_idx, batch in pbar:
            # FK dataset returns (tensor, metadata_dict) not (tensor, label)
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device, non_blocking=self.config.pin_memory)

            with self._autocast_context():
                loss, _pred, _mask = self.model(images)
                loss = loss / self.config.accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accum_step = (batch_idx + 1) % self.config.accum_steps == 0
            is_last_batch = (batch_idx + 1) == len(self.train_loader)

            if is_accum_step or is_last_batch:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip_norm,
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            total_loss += loss.item() * self.config.accum_steps
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item() * self.config.accum_steps:.4f}"})

        pbar.close()

        epoch_time = time.perf_counter() - epoch_start
        dataset_size = len(cast(Sized, self.train_loader.dataset))
        throughput = dataset_size / epoch_time if epoch_time > 0 else 0.0
        avg_loss = total_loss / num_batches
        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "train_loss": avg_loss,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
            "throughput_samples_per_sec": throughput,
        }

    def _run_visualization(self, epoch: int) -> None:
        """Generate FK-specific figures at target epochs.

        Args:
            epoch: 0-based epoch index.
        """
        viz_epochs = set(getattr(self.config, "visualization_epochs", [5, 10, 20, 30]))
        # Convert to 0-based
        viz_epochs_0 = {e - 1 for e in viz_epochs}
        is_first = epoch == self.start_epoch
        is_target = epoch in viz_epochs_0
        is_final = epoch == self.config.epochs - 1

        if not (is_first or is_target or is_final):
            return

        try:
            sample_batch = next(iter(self.val_loader))
            sample_images = sample_batch[0]
        except StopIteration:
            logger.warning("Validation loader is empty; skipping visualization.")
            return

        from src.evaluation.visualize import (
            plot_fk_reconstruction_grid,
            plot_fk_similarity_matrix,
            plot_fk_umap,
        )

        plot_dir = self.run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        if is_first:
            from src.evaluation.visualize import plot_masking_examples

            plot_masking_examples(
                self.model,
                sample_images,
                self.device,
                plot_dir / "masking_examples.png",
            )

        if is_target or is_final:
            plot_fk_reconstruction_grid(
                self.model,
                sample_images,
                self.device,
                plot_dir / f"reconstruction_epoch_{epoch + 1:03d}.png",
                seed=self.config.seed,
            )
            sil_score = plot_fk_umap(
                self.model,
                self.val_loader,
                self.device,
                plot_dir / f"umap_epoch_{epoch + 1:03d}.png",
                seed=self.config.seed,
            )
            if sil_score is not None:
                logger.info("Epoch %d | Silhouette score: %.3f", epoch + 1, sil_score)

            sim_metrics = plot_fk_similarity_matrix(
                self.model,
                self.val_loader,
                self.device,
                plot_dir / f"similarity_matrix_epoch_{epoch + 1:03d}.png",
                seed=self.config.seed,
            )
            if sim_metrics is not None:
                logger.info(
                    "Epoch %d | Similarity: intra=%.4f, inter=%.4f, contrast=%.3f",
                    epoch + 1,
                    sim_metrics["mean_intra"],
                    sim_metrics["mean_inter"],
                    sim_metrics["contrast"],
                )
