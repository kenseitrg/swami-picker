"""Training loop for Phase 4 supervised dispersion-curve picking."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sized
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.evaluation.picking_metrics import (
    compute_coverage,
    compute_curve_rmse,
    compute_presence_f1,
)
from src.evaluation.visualize_picking import plot_curve_overlays, plot_training_curves
from src.models.picking_model import inference_picks
from src.training.picking_loss import PickingLoss
from src.training.scheduler import get_cosine_schedule_with_warmup
from src.utils.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Append-only JSONL metrics logger."""

    def __init__(self, path: Path) -> None:
        """Initialize the logger.

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


class PickingTrainer:
    """Trainer for the Phase 4 supervised picking model."""

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path,
        run_dir: Path,
        resume_from: Path | None = None,
        argv: list[str] | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Picking model (U-Net or encoder-decoder).
            config: ``PickingConfig`` instance.
            device: Torch device.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            checkpoint_dir: Directory to write checkpoints.
            run_dir: Root directory for the current run.
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

        self.criterion = PickingLoss(
            pick_weight=config.loss_pick_weight,
            bce_weight=config.loss_bce_weight,
            direct_pick_weight=config.direct_pick_weight,
        )

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler() if device.type == "cuda" else None

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_rmse = float("inf")

        self.metrics_logger = MetricsLogger(run_dir / "metrics.jsonl")
        self.plots_dir = run_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        if resume_from is not None:
            self._load_checkpoint(resume_from)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Build the AdamW optimizer with optional backbone LR.

        Returns:
            Configured ``AdamW`` instance.
        """
        # The model is trained end-to-end from scratch, so all parameters
        # share the same learning rate.
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
            min_lr_ratio=0.02,
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

            is_best = val_metrics["val_rmse_pixels"] < self.best_val_rmse
            if is_best:
                self.best_val_rmse = val_metrics["val_rmse_pixels"]

            self._save_checkpoint(epoch, is_best=is_best)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | train_rmse=%.2f | "
                "val_loss=%.4f | val_rmse=%.2f | val_f1=%.3f | lr=%.2e | "
                "vram=%.1fMB | throughput=%.1f samples/s%s",
                epoch + 1,
                self.config.epochs,
                train_metrics["train_loss"],
                train_metrics["train_rmse_pixels"],
                val_metrics["val_loss"],
                val_metrics["val_rmse_pixels"],
                val_metrics["val_presence_f1"],
                train_metrics["lr"],
                max_vram_mb,
                train_metrics["throughput_samples_per_sec"],
                " | BEST" if is_best else "",
            )

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

        logger.info("Training complete. Best val_rmse=%.4f", self.best_val_rmse)

        # Final training-curve plot.
        plot_training_curves(
            self.metrics_logger.path,
            save_path=self.plots_dir / "training_curves.png",
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
        total_rmse = 0.0
        total_f1 = 0.0
        num_batches = 0
        epoch_start = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            (
                spectra,
                pick_target,
                presence_target,
                direct_mask,
                _confidence,
                cluster_embedding,
                _spectrum_id,
            ) = batch

            spectra = spectra.to(self.device, non_blocking=self.config.pin_memory)
            pick_target = pick_target.to(
                self.device, non_blocking=self.config.pin_memory
            )
            presence_target = presence_target.to(
                self.device, non_blocking=self.config.pin_memory
            )
            direct_mask = direct_mask.to(
                self.device, non_blocking=self.config.pin_memory
            )

            with self._autocast_context():
                if (
                    cluster_embedding is not None
                    and self.config.use_cluster_conditioning
                ):
                    cluster_embedding = cluster_embedding.to(
                        self.device, non_blocking=self.config.pin_memory
                    )
                    pick_logits, presence_logits = self.model(
                        spectra, cluster_embedding
                    )
                else:
                    pick_logits, presence_logits = self.model(spectra)

                loss, _ = self.criterion(
                    pick_logits,
                    presence_logits,
                    pick_target,
                    presence_target,
                    direct_mask,
                )
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

            # Logging on unscaled loss.
            total_loss += loss.item() * self.config.accum_steps
            num_batches += 1

            with torch.no_grad():
                pred_picks, pred_probs = inference_picks(pick_logits, presence_logits)
                rmse = compute_curve_rmse(
                    pred_picks, pick_target, presence_target.bool()
                )
                f1 = compute_presence_f1(pred_probs, presence_target)
                if torch.isfinite(rmse):
                    total_rmse += rmse.item()
                total_f1 += f1.item()

            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.debug(
                    "Epoch %d | Batch %d/%d | loss=%.4f | lr=%.2e",
                    epoch + 1,
                    batch_idx,
                    len(self.train_loader),
                    loss.item() * self.config.accum_steps,
                    current_lr,
                )

        dataset_size = len(cast(Sized, self.train_loader.dataset))
        epoch_time = time.perf_counter() - epoch_start
        throughput = dataset_size / epoch_time if epoch_time > 0 else 0.0

        return {
            "train_loss": total_loss / num_batches,
            "train_rmse_pixels": total_rmse / num_batches,
            "train_presence_f1": total_f1 / num_batches,
            "lr": self.optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
            "throughput_samples_per_sec": throughput,
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, Any]:
        """Run validation and collect metrics.

        Args:
            epoch: 0-based epoch index.

        Returns:
            Dictionary of validation metrics.
        """
        _ = epoch
        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0
        total_f1 = 0.0
        total_coverage = 0.0
        num_batches = 0

        all_spectra: list[torch.Tensor] = []
        all_true: list[torch.Tensor] = []
        all_pred: list[torch.Tensor] = []
        all_probs: list[torch.Tensor] = []

        for batch in self.val_loader:
            (
                spectra,
                pick_target,
                presence_target,
                direct_mask,
                _confidence,
                cluster_embedding,
                _spectrum_id,
            ) = batch

            spectra = spectra.to(self.device, non_blocking=self.config.pin_memory)
            pick_target = pick_target.to(
                self.device, non_blocking=self.config.pin_memory
            )
            presence_target = presence_target.to(
                self.device, non_blocking=self.config.pin_memory
            )
            direct_mask = direct_mask.to(
                self.device, non_blocking=self.config.pin_memory
            )

            if cluster_embedding is not None and self.config.use_cluster_conditioning:
                cluster_embedding = cluster_embedding.to(
                    self.device, non_blocking=self.config.pin_memory
                )
                pick_logits, presence_logits = self.model(spectra, cluster_embedding)
            else:
                pick_logits, presence_logits = self.model(spectra)

            loss, _ = self.criterion(
                pick_logits,
                presence_logits,
                pick_target,
                presence_target,
                direct_mask,
            )
            total_loss += loss.item()

            pred_picks, pred_probs = inference_picks(pick_logits, presence_logits)
            rmse = compute_curve_rmse(pred_picks, pick_target, presence_target.bool())
            f1 = compute_presence_f1(pred_probs, presence_target)
            coverage = compute_coverage(pred_picks)

            if torch.isfinite(rmse):
                total_rmse += rmse.item()
            total_f1 += f1.item()
            total_coverage += coverage.item()
            num_batches += 1

            # Save a few samples for visualization.
            if len(all_spectra) < 16:
                all_spectra.append(spectra.cpu())
                all_true.append(pick_target.cpu())
                all_pred.append(pred_picks.cpu())
                all_probs.append(pred_probs.cpu())

        if num_batches == 0:
            logger.warning("Validation loader is empty; returning NaN metrics.")
            return {
                "val_loss": float("nan"),
                "val_rmse_pixels": float("nan"),
                "val_presence_f1": float("nan"),
                "val_coverage": float("nan"),
            }

        avg_loss = total_loss / num_batches
        avg_rmse = total_rmse / num_batches
        avg_f1 = total_f1 / num_batches
        avg_coverage = total_coverage / num_batches

        # Visualization on a subset of validation samples.
        epoch_1based = epoch + 1
        if epoch_1based in self.config.visualization_epochs and all_spectra:
            vis_spectra = torch.cat(all_spectra, dim=0)[:6]
            vis_true = torch.cat(all_true, dim=0)[:6]
            vis_pred = torch.cat(all_pred, dim=0)[:6]
            vis_probs = torch.cat(all_probs, dim=0)[:6]
            plot_curve_overlays(
                vis_spectra,
                vis_true,
                vis_pred,
                presence_probs=vis_probs,
                save_path=self.plots_dir
                / f"curve_predictions_epoch_{epoch_1based:03d}.png",
                seed=self.config.seed,
            )

        return {
            "val_loss": avg_loss,
            "val_rmse_pixels": avg_rmse,
            "val_presence_f1": avg_f1,
            "val_coverage": avg_coverage,
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Persist a training checkpoint.

        Args:
            epoch: 0-based index of the just-completed epoch.
            is_best: Whether this checkpoint is the best so far.
        """
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "step": self.global_step,
            "seed": self.config.seed,
            "config": self.config.to_dict(),
            "metrics": {"best_val_rmse": self.best_val_rmse},
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
        self.best_val_rmse = checkpoint.get("metrics", {}).get(
            "best_val_rmse", float("inf")
        )

        rng_state = checkpoint.get("rng_state")
        restore_rng_state(rng_state)

        logger.info(
            "Resumed from epoch %d, step %d",
            self.start_epoch,
            self.global_step,
        )
