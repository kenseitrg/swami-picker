from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sized
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.training.scheduler import get_cosine_schedule_with_warmup
from src.utils.checkpoint import load_checkpoint, save_checkpoint


if TYPE_CHECKING:
    from src.models.cvt_mae import CvTMaskedAutoencoder
    from src.models.mae import MaskedAutoencoder
    from src.utils.config import MNISTConfig

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Append-only JSONL metrics logger."""

    def __init__(self, path: Path) -> None:
        """Initialise the logger.

        Args:
            path: Path to the JSONL file.
        """
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        """Append a metrics dictionary as a single JSON line.

        Args:
            metrics: Dictionary of scalar metrics.
        """
        with open(self.path, "a") as fh:
            fh.write(json.dumps(metrics) + "\n")


class MAETrainer:
    """Trainer for Masked Autoencoder (MAE) pre-training.

    Handles AMP, gradient accumulation, gradient clipping, cosine
    warmup schedule, checkpointing, and metric logging.
    """

    def __init__(
        self,
        model: MaskedAutoencoder | CvTMaskedAutoencoder,
        config: MNISTConfig,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path,
        run_dir: Path,
        resume_from: Path | None = None,
        argv: list[str] | None = None,
    ) -> None:
        """Initialise the trainer.

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
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.run_dir = run_dir
        self.argv = argv

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler() if device.type == "cuda" else None

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        self.metrics_logger = MetricsLogger(run_dir / "metrics.jsonl")

        if resume_from is not None:
            self._load_checkpoint(resume_from)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Build the AdamW optimiser.

        Returns:
            Configured ``AdamW`` instance.
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
            fused=(self.device.type == "cuda"),
        )

    def _setup_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Build the LR scheduler with warmup and cosine decay.

        Returns:
            ``LambdaLR`` instance.
        """
        steps_per_epoch = math.ceil(len(self.train_loader) / self.config.accum_steps)
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        logger.info(
            "Scheduler: %d steps/epoch, %d total steps, %d warmup steps",
            steps_per_epoch,
            total_steps,
            warmup_steps,
        )

        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=0.1,
        )

    def _autocast_context(self) -> Any:
        """Return an autocast context manager if on CUDA, else a no-op."""
        if self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def train(self) -> None:
        """Run the full training loop."""
        logger.info(
            "Starting training from epoch %d for %d epochs",
            self.start_epoch + 1,
            self.config.epochs,
        )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(self.start_epoch, self.config.epochs):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)

            max_vram_mb = 0.0
            if self.device.type == "cuda":
                max_vram = torch.cuda.max_memory_allocated(self.device)
                max_vram_mb = max_vram / (1024**2)
                torch.cuda.empty_cache()

            metrics = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                **train_metrics,
                **val_metrics,
                "max_vram_mb": max_vram_mb,
            }
            self.metrics_logger.log(metrics)

            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            self._save_checkpoint(epoch, is_best=is_best)

            logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | "
                "lr=%.2e | vram=%.1fMB | throughput=%.1f samples/s%s",
                epoch + 1,
                self.config.epochs,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
                train_metrics["lr"],
                max_vram_mb,
                train_metrics["throughput_samples_per_sec"],
                " | BEST" if is_best else "",
            )

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            self._run_visualization(epoch)

        self._plot_final_curves()

        logger.info(
            "Training complete. Best val_loss=%.6f",
            self.best_val_loss,
        )

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        """Run one training epoch.

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

        for batch_idx, (images, _labels) in enumerate(self.train_loader):
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
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip_norm,
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            # Un-scale for logging
            total_loss += loss.item() * self.config.accum_steps
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.debug(
                    "Epoch %d | Batch %d/%d | loss=%.6f | lr=%.2e",
                    epoch + 1,
                    batch_idx,
                    len(self.train_loader),
                    loss.item() * self.config.accum_steps,
                    current_lr,
                )

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

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, Any]:
        """Run validation and collect metrics.

        Args:
            epoch: 0-based epoch index (unused, kept for API symmetry).

        Returns:
            Dictionary of validation metrics.
        """
        _ = epoch
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for images, _labels in self.val_loader:
            images = images.to(self.device, non_blocking=self.config.pin_memory)
            loss, _pred, _mask = self.model(images)
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            logger.warning("Validation loader is empty; returning NaN loss.")
            return {"val_loss": float("nan")}

        avg_loss = total_loss / num_batches

        return {
            "val_loss": avg_loss,
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Persist a training checkpoint.

        Args:
            epoch: 0-based index of the just-completed epoch.
            is_best: Whether this checkpoint is the best so far.
        """
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if self.device.type == "cuda"
            else None,
        }
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "step": self.global_step,
            "seed": self.config.seed,
            "config": self.config.to_dict(),
            "metrics": {"best_val_loss": self.best_val_loss},
            "rng_state": rng_state,
            "argv": self.argv,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
        save_checkpoint(state, path, is_best=is_best)

    def _load_checkpoint(self, path: Path) -> None:
        """Restore training state from a checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = load_checkpoint(path, device=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint.get("step", 0)
        self.best_val_loss = checkpoint.get("metrics", {}).get(
            "best_val_loss", float("inf")
        )

        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state["torch"].cpu())
            cuda_rng = rng_state.get("cuda")
            if cuda_rng is not None and self.device.type == "cuda":
                torch.cuda.set_rng_state_all([s.cpu() for s in cuda_rng])

        logger.info(
            "Resumed from epoch %d, step %d",
            self.start_epoch,
            self.global_step,
        )

    def _run_visualization(self, epoch: int) -> None:
        """Generate figures at target epochs.

        Args:
            epoch: 0-based epoch index.
        """
        # 0-based indices 2 and 4 correspond to 1-based epochs 3 and 5
        plot_epochs = {2, 4}
        is_first = epoch == self.start_epoch
        is_target = epoch in plot_epochs
        is_final = epoch == self.config.epochs - 1

        if not (is_first or is_target or is_final):
            return

        # Guard against empty validation loader
        try:
            sample_images = next(iter(self.val_loader))[0]
        except StopIteration:
            logger.warning("Validation loader is empty; skipping visualization.")
            return

        # Lazy import to avoid heavy deps at module load time
        from src.evaluation.visualize import (
            plot_embedding_similarity_matrix,
            plot_masking_examples,
            plot_reconstruction_grid,
            plot_umap_embeddings,
        )

        plot_dir = self.run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        if is_first:
            plot_masking_examples(
                self.model,
                sample_images,
                self.device,
                plot_dir / "masking_examples.png",
            )

        if is_target or is_final:
            plot_reconstruction_grid(
                self.model,
                sample_images,
                self.device,
                plot_dir / f"reconstruction_epoch_{epoch + 1:03d}.png",
                seed=self.config.seed,
            )
            sil_score = plot_umap_embeddings(
                self.model,
                self.val_loader,
                self.device,
                plot_dir / f"umap_epoch_{epoch + 1:03d}.png",
                seed=self.config.seed,
            )
            if sil_score is not None:
                logger.info("Epoch %d | Silhouette score: %.3f", epoch + 1, sil_score)

            sim_metrics = plot_embedding_similarity_matrix(
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

    def _plot_final_curves(self) -> None:
        """Plot loss/LR/VRAM curves from the accumulated metrics file."""
        from src.evaluation.visualize import plot_loss_curves

        plot_dir = self.run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_loss_curves(
            self.metrics_logger.path,
            plot_dir / "loss_curves.png",
        )
