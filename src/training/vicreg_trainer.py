"""Trainer for VICReg self-supervised learning on FK spectra."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sized
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentations import FKSpectrumTransform
from src.models.vicreg import vicreg_loss
from src.training.scheduler import get_cosine_schedule_with_warmup
from src.utils.checkpoint import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from src.models.vicreg import VICReg
    from src.utils.config import VICRegConfig

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Append-only JSONL metrics logger."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(metrics) + "\n")


class VICRegTrainer:
    """Trainer for VICReg self-supervised pretraining.

    Generates two independently augmented views of each training sample,
    passes both through the encoder+projector, and optimises the VICReg
    loss (invariance + variance + covariance).

    Validation extracts encoder embeddings (before projector) and
    computes UMAP, Silhouette score, and line-based similarity metrics.
    """

    def __init__(
        self,
        model: VICReg,
        config: VICRegConfig,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Path,
        run_dir: Path,
        resume_from: Path | None = None,
        argv: list[str] | None = None,
    ) -> None:
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
        self.best_val_metric = float("inf")

        self.metrics_logger = MetricsLogger(run_dir / "metrics.jsonl")

        # Augmentation callable — applied on-the-fly during training
        self.transform = FKSpectrumTransform(
            noise_std=config.noise_std,
            intensity_jitter=config.intensity_jitter,
            freq_shift_max=config.freq_shift_max,
            waven_shift_max=config.waven_shift_max,
            freq_dropout_prob=config.freq_dropout_prob,
            freq_dropout_width=config.freq_dropout_width,
        )

        if resume_from is not None:
            self._load_checkpoint(resume_from)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
            fused=(self.device.type == "cuda"),
        )

    def _setup_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
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
        if self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.float16)
        from contextlib import nullcontext

        return nullcontext()

    def train(self) -> None:
        logger.info(
            "Starting VICReg training from epoch %d for %d epochs",
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
                max_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                torch.cuda.empty_cache()

            metrics = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                **train_metrics,
                **val_metrics,
                "max_vram_mb": max_vram_mb,
            }
            self.metrics_logger.log(metrics)

            is_best = val_metrics["val_loss"] < self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics["val_loss"]

            self._save_checkpoint(epoch, is_best=is_best)

            logger.info(
                "Epoch %d/%d | train_loss=%.6f (inv=%.4f var=%.4f cov=%.4f) | "
                "val_sil=%.3f contrast=%.3f | lr=%.2e | vram=%.1fMB | "
                "throughput=%.1f s/s%s",
                epoch + 1,
                self.config.epochs,
                train_metrics["train_loss"],
                train_metrics.get("inv_loss", 0.0),
                train_metrics.get("var_loss", 0.0),
                train_metrics.get("cov_loss", 0.0),
                val_metrics.get("silhouette", 0.0),
                val_metrics.get("contrast", 0.0),
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
            "Training complete. Best val_metric=%.6f",
            self.best_val_metric,
        )

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        total_inv = 0.0
        total_var = 0.0
        total_cov = 0.0
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
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device, non_blocking=self.config.pin_memory)

            # Generate two independently augmented views
            x1 = torch.stack(
                [self.transform(img.unsqueeze(0)).squeeze(0) for img in images]
            )
            x2 = torch.stack(
                [self.transform(img.unsqueeze(0)).squeeze(0) for img in images]
            )

            with self._autocast_context():
                z1 = self.model(x1)
                z2 = self.model(x2)
                loss, inv_loss, var_loss, cov_loss = vicreg_loss(
                    z1,
                    z2,
                    sim_weight=self.config.sim_weight,
                    var_weight=self.config.var_weight,
                    cov_weight=self.config.cov_weight,
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

            total_loss += loss.item() * self.config.accum_steps
            total_inv += inv_loss.item()
            total_var += var_loss.item()
            total_cov += cov_loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item() * self.config.accum_steps:.4f}"})

        pbar.close()

        epoch_time = time.perf_counter() - epoch_start
        dataset_size = len(cast(Sized, self.train_loader.dataset))
        throughput = dataset_size / epoch_time if epoch_time > 0 else 0.0

        return {
            "train_loss": total_loss / num_batches,
            "inv_loss": total_inv / num_batches,
            "var_loss": total_var / num_batches,
            "cov_loss": total_cov / num_batches,
            "lr": self.optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
            "throughput_samples_per_sec": throughput,
        }

    @torch.no_grad()
    def _validate(self, _epoch: int) -> dict[str, Any]:
        self.model.eval()

        all_embeddings: list[torch.Tensor] = []
        all_line_numbers: list[int] = []

        for batch in self.val_loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            metadata = batch[1] if isinstance(batch, (list, tuple)) else {}
            images = images.to(self.device, non_blocking=self.config.pin_memory)

            # Use encoder output (before projector) for evaluation
            emb = self.model.extract_embeddings(images)
            all_embeddings.append(emb.cpu())

            if isinstance(metadata, list):
                all_line_numbers.extend([m.get("line_number", -1) for m in metadata])
            elif isinstance(metadata, dict):
                all_line_numbers.extend(metadata.get("line_number", [-1] * len(images)))

        if not all_embeddings:
            return {
                "val_loss": float("inf"),
                "silhouette": 0.0,
                "contrast": 1.0,
                "intra_sim": 1.0,
                "inter_sim": 1.0,
            }

        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        line_numbers = np.array(all_line_numbers)

        # Compute silhouette if we have valid labels
        silhouette = 0.0
        try:
            from sklearn.metrics import silhouette_score

            # Need at least 2 unique labels and enough samples
            unique_labels = np.unique(line_numbers)
            if len(unique_labels) >= 2 and len(embeddings) >= len(unique_labels) * 2:
                silhouette = float(silhouette_score(embeddings, line_numbers))
        except Exception:
            pass

        # Compute similarity metrics
        from src.evaluation.visualize import _compute_similarity_metrics

        sim_metrics = _compute_similarity_metrics(embeddings, line_numbers)
        contrast = sim_metrics.get("contrast", 1.0)
        intra_sim = sim_metrics.get("mean_intra", 1.0)
        inter_sim = sim_metrics.get("mean_inter", 1.0)

        # Use negative silhouette as the "loss" for checkpointing
        # (lower is better). If silhouette computation failed, use 1/contrast.
        val_loss = -silhouette if silhouette != 0.0 else 1.0 / max(contrast, 0.01)

        return {
            "val_loss": val_loss,
            "silhouette": silhouette,
            "contrast": contrast,
            "intra_sim": intra_sim,
            "inter_sim": inter_sim,
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
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
            "metrics": {"best_val_metric": self.best_val_metric},
            "rng_state": rng_state,
            "argv": self.argv,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
        save_checkpoint(state, path, is_best=is_best)

    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = load_checkpoint(path, device=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint.get("step", 0)
        self.best_val_metric = checkpoint.get("metrics", {}).get(
            "best_val_metric", float("inf")
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
        viz_epochs = set(
            getattr(self.config, "visualization_epochs", [10, 25, 50, 100])
        )
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
            plot_fk_similarity_matrix,
            plot_fk_umap,
        )

        plot_dir = self.run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        if is_first:
            # Save a sample of augmented pairs
            self._plot_augmentation_pairs(sample_images, plot_dir)

        if is_target or is_final:
            plot_fk_umap(
                self.model,
                self.val_loader,
                self.device,
                plot_dir / f"umap_epoch_{epoch + 1:03d}.png",
                seed=self.config.seed,
            )
            plot_fk_similarity_matrix(
                self.model,
                self.val_loader,
                self.device,
                plot_dir / f"similarity_matrix_epoch_{epoch + 1:03d}.png",
                seed=self.config.seed,
            )

    @torch.no_grad()
    def _plot_augmentation_pairs(self, images: torch.Tensor, plot_dir: Path) -> None:
        """Plot a few original / augmented pairs to verify augmentation diversity."""
        import matplotlib.pyplot as plt

        from src.utils.plot_style import apply_style, save_figure

        apply_style()
        n = min(4, len(images))
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        imgs = images[:n]
        for i in range(n):
            x1 = self.transform(imgs[i].unsqueeze(0)).squeeze(0)
            x2 = self.transform(imgs[i].unsqueeze(0)).squeeze(0)

            vmin = min(imgs[i].min().item(), x1.min().item(), x2.min().item())
            vmax = max(imgs[i].max().item(), x1.max().item(), x2.max().item())

            for j, tensor in enumerate([imgs[i], x1, x2]):
                ax = axes[i, j]
                # tensor may be (1, 256, 256) or (256, 256); squeeze channel dim
                img_np = (
                    tensor.squeeze(0).numpy() if tensor.ndim == 3 else tensor.numpy()
                )
                ax.imshow(img_np, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.axis("off")
                if i == 0:
                    titles = ["Original", "Aug View 1", "Aug View 2"]
                    ax.set_title(titles[j])

        plt.tight_layout()
        save_figure(fig, plot_dir / "augmentation_pairs.png")
        plt.close(fig)
        logger.info(
            "Saved augmentation pairs to %s", plot_dir / "augmentation_pairs.png"
        )

    def _plot_final_curves(self) -> None:
        from src.evaluation.visualize import plot_loss_curves

        plot_dir = self.run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_loss_curves(
            self.metrics_logger.path,
            plot_dir / "loss_curves.png",
        )
