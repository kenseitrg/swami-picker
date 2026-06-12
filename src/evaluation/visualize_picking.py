"""Visualization utilities for Phase 4 supervised picking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.plot_style import apply_style


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a tensor or array to a CPU numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_curve_overlays(
    spectra: torch.Tensor | np.ndarray,
    true_picks: torch.Tensor | np.ndarray,
    pred_picks: torch.Tensor | np.ndarray,
    presence_probs: torch.Tensor | np.ndarray | None = None,
    metadata: list[dict[str, Any]] | None = None,
    num_samples: int = 6,
    save_path: Path | None = None,
    seed: int = 42,
) -> None:
    """Plot a grid of spectra with true and predicted dispersion curves.

    Args:
        spectra: Input spectra of shape ``(N, 1, H, W)``.
        true_picks: Ground-truth picks of shape ``(N, W)``.
        pred_picks: Predicted picks of shape ``(N, W)``.
        presence_probs: Optional presence probabilities of shape ``(N, W)``.
        metadata: Optional list of metadata dicts for physical axes.
        num_samples: Number of spectra to display.
        save_path: Optional path to save the figure.
        seed: Seed for random sample selection.
    """
    apply_style()
    rng = np.random.default_rng(seed)

    spectra = _to_numpy(spectra)
    true_picks = _to_numpy(true_picks)
    pred_picks = _to_numpy(pred_picks)
    if presence_probs is not None:
        presence_probs = _to_numpy(presence_probs)

    n = min(num_samples, spectra.shape[0])
    indices = rng.choice(spectra.shape[0], size=n, replace=False)

    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3 * ((n + 1) // 2), 6))
    axes = np.atleast_2d(axes).flatten()

    for ax, idx in zip(axes, indices, strict=False):
        spec = spectra[idx, 0]
        meta = metadata[idx] if metadata else None

        if meta is not None:
            extent = [
                meta["freq_axis_resized"][0],
                meta["freq_axis_resized"][-1],
                meta["waven_axis_resized"][-1],
                meta["waven_axis_resized"][0],
            ]
            ax.imshow(
                spec, aspect="auto", cmap="viridis", extent=extent, origin="upper"
            )
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Wavenumber (1/m)")
        else:
            ax.imshow(spec, aspect="auto", cmap="viridis", origin="upper")
            ax.set_xlabel("Frequency index")
            ax.set_ylabel("Wavenumber index")

        freqs = (
            np.arange(spec.shape[1])
            if meta is None
            else np.asarray(meta["freq_axis_resized"])
        )
        waven = (
            np.arange(spec.shape[0])
            if meta is None
            else np.asarray(meta["waven_axis_resized"])
        )

        true = true_picks[idx]
        pred = pred_picks[idx]
        true_valid = true >= 0
        pred_valid = pred >= 0

        ax.plot(
            freqs[true_valid],
            waven[true[true_valid].astype(int)],
            "r-",
            linewidth=2,
            label="True",
        )
        ax.plot(
            freqs[pred_valid],
            waven[pred[pred_valid].astype(int)],
            "g--",
            linewidth=1.5,
            label="Predicted",
        )
        title = meta.get("spectrum_id", f"Sample {idx}") if meta else f"Sample {idx}"
        ax.set_title(title)
        ax.legend()

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_training_curves(
    metrics_path: Path,
    save_path: Path | None = None,
) -> None:
    """Plot loss, RMSE, presence F1, LR, and VRAM from a JSONL metrics file.

    Args:
        metrics_path: Path to ``metrics.jsonl``.
        save_path: Optional path to save the figure.
    """
    apply_style()

    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    train_rmse: list[float] = []
    val_rmse: list[float] = []
    train_f1: list[float] = []
    val_f1: list[float] = []
    lrs: list[float] = []
    vrams: list[float] = []

    with open(metrics_path) as fh:
        for line in fh:
            row = json.loads(line)
            epochs.append(row.get("epoch", len(epochs) + 1))
            train_loss.append(row.get("train_loss", float("nan")))
            val_loss.append(row.get("val_loss", float("nan")))
            train_rmse.append(row.get("train_rmse_pixels", float("nan")))
            val_rmse.append(row.get("val_rmse_pixels", float("nan")))
            train_f1.append(row.get("train_presence_f1", float("nan")))
            val_f1.append(row.get("val_presence_f1", float("nan")))
            lrs.append(row.get("lr", float("nan")))
            vrams.append(row.get("max_vram_mb", float("nan")))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    def plot_ax(ax, y_train, y_val, label, title):
        ax.plot(epochs, y_train, label="train")
        ax.plot(epochs, y_val, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.legend()

    plot_ax(axes[0], train_loss, val_loss, "Loss", "Total Loss")
    plot_ax(axes[1], train_rmse, val_rmse, "RMSE (pixels)", "Pick RMSE")
    plot_ax(axes[2], train_f1, val_f1, "F1", "Presence F1")

    axes[3].plot(epochs, lrs)
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Learning Rate")
    axes[3].set_title("Learning Rate Schedule")

    axes[4].plot(epochs, vrams)
    axes[4].set_xlabel("Epoch")
    axes[4].set_ylabel("VRAM (MB)")
    axes[4].set_title("Peak VRAM")

    axes[5].axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_error_distribution(
    rmse_per_spectrum: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Plot a histogram of per-spectrum RMSE values.

    Args:
        rmse_per_spectrum: Array of RMSE values.
        save_path: Optional path to save the figure.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rmse_per_spectrum, bins=30, edgecolor="black")
    ax.set_xlabel("RMSE (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Spectrum Pick RMSE Distribution")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
