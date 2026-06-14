"""Visualization utilities for Phase 4 supervised picking."""

from __future__ import annotations

import json
from collections.abc import Sequence
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


def _axis_arrays(
    spec: np.ndarray, meta: dict[str, Any] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency and wavenumber arrays for a spectrum."""
    if meta is not None:
        freqs = np.asarray(meta["freq_axis_resized"])
        waven = np.asarray(meta["waven_axis_resized"])
    else:
        freqs = np.arange(spec.shape[1])
        waven = np.arange(spec.shape[0])
    return freqs, waven


def _extent_from_axes(freqs: np.ndarray, waven: np.ndarray) -> list[float]:
    """Build an ``imshow`` extent from physical axis arrays."""
    return [float(freqs[0]), float(freqs[-1]), float(waven[-1]), float(waven[0])]


def plot_curve_overlays(
    spectra: torch.Tensor | np.ndarray,
    true_picks: torch.Tensor | np.ndarray,
    pred_picks: torch.Tensor | np.ndarray,
    metadata: list[dict[str, Any]] | None = None,
    num_samples: int = 6,
    save_path: Path | None = None,
    seed: int = 42,
) -> None:
    """Plot a grid of spectra with true and predicted dispersion curves."""
    apply_style()
    rng = np.random.default_rng(seed)

    spectra = _to_numpy(spectra)
    true_picks = _to_numpy(true_picks)
    pred_picks = _to_numpy(pred_picks)

    n = min(num_samples, spectra.shape[0])
    indices = rng.choice(spectra.shape[0], size=n, replace=False)

    cols = (n + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6))
    axes = np.atleast_2d(axes).flatten()

    for ax, idx in zip(axes, indices, strict=False):
        spec = spectra[idx, 0]
        meta = metadata[idx] if metadata else None
        freqs, waven = _axis_arrays(spec, meta)
        extent = _extent_from_axes(freqs, waven)

        ax.imshow(spec, aspect="auto", cmap="viridis", extent=extent, origin="upper")

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
        ax.set_xlabel("Frequency (Hz)" if meta else "Frequency index")
        ax.set_ylabel("Wavenumber (1/m)" if meta else "Wavenumber index")
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
    """Plot loss, RMSE, presence F1, LR, and VRAM from a JSONL metrics file."""
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
    """Plot a histogram of per-spectrum RMSE values."""
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


def plot_probability_heatmap_overlay(
    spectra: torch.Tensor | np.ndarray,
    logits: torch.Tensor | np.ndarray,
    metadata: list[dict[str, Any]] | None = None,
    num_samples: int = 4,
    save_path: Path | None = None,
    seed: int = 42,
) -> None:
    """Overlay the model's pick probability heatmap on spectra.

    For each selected spectrum, this produces a two-panel figure:

    * Left: input spectrum in grayscale.
    * Right: the same spectrum with the per-column softmax probability
      distribution overlaid as a translucent ``hot`` heatmap.  No pick
      curves are drawn so the probability structure is clearly visible.
    """
    apply_style()
    rng = np.random.default_rng(seed)

    spectra = _to_numpy(spectra)
    logits = _to_numpy(logits)

    n = min(num_samples, spectra.shape[0])
    indices = rng.choice(spectra.shape[0], size=n, replace=False)

    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = np.atleast_2d(axes)
    axes = np.atleast_2d(axes)

    absent_class = logits.shape[1] - 1

    for row, idx in enumerate(indices):
        spec = spectra[idx, 0]
        meta = metadata[idx] if metadata else None
        freqs, waven = _axis_arrays(spec, meta)
        extent = _extent_from_axes(freqs, waven)

        logit = logits[idx]  # (num_classes, W)
        probs = _softmax(logit, axis=0)
        pick_probs = probs[:absent_class, :]  # (H, W)

        # Left panel: grayscale spectrum.
        ax_spec = axes[row, 0]
        ax_spec.imshow(spec, aspect="auto", cmap="gray", extent=extent, origin="upper")
        ax_spec.set_xlabel("Frequency (Hz)" if meta else "Frequency index")
        ax_spec.set_ylabel("Wavenumber (1/m)" if meta else "Wavenumber index")
        title = meta.get("spectrum_id", f"Sample {idx}") if meta else f"Sample {idx}"
        ax_spec.set_title(f"{title} — Spectrum")

        # Right panel: probability heatmap overlay.
        ax_heat = axes[row, 1]
        ax_heat.imshow(spec, aspect="auto", cmap="gray", extent=extent, origin="upper")
        # Normalize per column so the peak is always visible.
        peak = pick_probs.max(axis=0, keepdims=True)
        prob_map_norm = pick_probs / (peak + 1e-8)
        alpha_map = 0.2 + 0.6 * prob_map_norm
        im = ax_heat.imshow(
            prob_map_norm,
            aspect="auto",
            cmap="hot",
            extent=extent,
            origin="upper",
            alpha=alpha_map,
            vmin=0.0,
            vmax=1.0,
        )
        ax_heat.set_xlabel("Frequency (Hz)" if meta else "Frequency index")
        ax_heat.set_ylabel("Wavenumber (1/m)" if meta else "Wavenumber index")
        ax_heat.set_title(f"{title} — Pick Probability Heatmap")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable softmax matching PyTorch's semantics."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def torch_softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable softmax matching PyTorch's semantics."""
    return _softmax(x, axis=axis)


def plot_certainty_distributions(
    presence_probs: torch.Tensor | np.ndarray,
    true_presence: torch.Tensor | np.ndarray | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot distributions of model presence certainty."""
    apply_style()
    probs = _to_numpy(presence_probs).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(probs, bins=50, range=(0.0, 1.0), edgecolor="black")
    axes[0].set_xlabel("Presence Probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Presence Probabilities")
    axes[0].axvline(np.mean(probs), color="r", linestyle="--", label="mean")
    axes[0].legend()

    if true_presence is not None:
        true_pres = _to_numpy(true_presence).flatten().astype(bool)
        axes[1].hist(
            probs[true_pres],
            bins=50,
            range=(0.0, 1.0),
            alpha=0.6,
            label="Mode present",
            edgecolor="black",
        )
        axes[1].hist(
            probs[~true_pres],
            bins=50,
            range=(0.0, 1.0),
            alpha=0.6,
            label="Mode absent",
            edgecolor="black",
        )
        axes[1].set_xlabel("Presence Probability")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Presence Probability by Ground Truth")
        axes[1].legend()
    else:
        axes[1].axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_inference_curve_grid(
    spectra: torch.Tensor | np.ndarray,
    picks: torch.Tensor | np.ndarray,
    presence_probs: torch.Tensor | np.ndarray,
    metadata: list[dict[str, Any]] | None = None,
    indices: Sequence[int] | None = None,
    num_samples: int = 8,
    title: str = "Inference picks",
    save_path: Path | None = None,
    seed: int = 42,
) -> None:
    """Plot a grid of spectra with inferred dispersion curves.

    Unlike :func:`plot_curve_overlays`, this function does not require
    ground-truth picks.  It is intended for visualizing model inference
    outputs on the full dataset.

    Args:
        spectra: Input spectra of shape ``(N, 1, H, W)``.
        picks: Dense pick indices of shape ``(N, W)``.
        presence_probs: Presence probabilities of shape ``(N, W)``.
        metadata: Optional list of metadata dictionaries.  Used for
            physical axes and spectrum identifiers.
        indices: Optional explicit indices into *spectra* to plot.  If
            ``None``, *num_samples* random indices are chosen using
            *seed*.
        num_samples: Number of spectra to display when *indices* is not
            provided.
        title: Overall figure title.
        save_path: Optional path to save the figure.
        seed: Random seed for sample selection.
    """
    apply_style()
    rng = np.random.default_rng(seed)

    spectra = _to_numpy(spectra)
    picks = _to_numpy(picks)
    presence_probs = _to_numpy(presence_probs)

    if indices is None:
        n = min(num_samples, spectra.shape[0])
        indices = rng.choice(spectra.shape[0], size=n, replace=False)
    else:
        indices = list(indices)

    n = len(indices)
    cols = (n + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 6.4))
    axes = np.atleast_2d(axes).flatten()

    for ax, idx in zip(axes, indices, strict=False):
        spec = spectra[idx, 0]
        meta = metadata[idx] if metadata else None
        freqs, waven = _axis_arrays(spec, meta)
        extent = _extent_from_axes(freqs, waven)

        ax.imshow(spec, aspect="auto", cmap="viridis", extent=extent, origin="upper")

        pred = picks[idx]
        pred_valid = pred >= 0
        if pred_valid.any():
            ax.plot(
                freqs[pred_valid],
                waven[pred[pred_valid].astype(int)],
                "g--",
                linewidth=1.5,
                label="Predicted",
            )

        mean_prob = float(np.mean(presence_probs[idx]))
        coverage = float(np.mean(pred_valid))
        subtitle = meta.get("spectrum_id", f"Sample {idx}") if meta else f"Sample {idx}"
        ax.set_title(f"{subtitle}\ncov={coverage:.2f}, mean_p={mean_prob:.2f}")
        ax.set_xlabel("Frequency (Hz)" if meta else "Frequency index")
        ax.set_ylabel("Wavenumber (1/m)" if meta else "Wavenumber index")
        ax.legend()

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_quality_distributions(
    quality_scores: Sequence[dict[str, Any]],
    metrics: Sequence[str] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot histograms of per-spectrum quality metrics.

    Args:
        quality_scores: Sequence of quality-score records.  Each record
            must contain the requested metric keys.
        metrics: Optional list of metric names to plot.  Defaults to a
            curated set of interpretable metrics.
        save_path: Optional path to save the figure.
    """
    apply_style()

    if metrics is None:
        metrics = [
            "coverage",
            "mean_certainty",
            "smoothness",
            "monotonicity",
            "composite_score",
        ]

    data: dict[str, np.ndarray] = {}
    for metric in metrics:
        values = [float(score[metric]) for score in quality_scores if metric in score]
        data[metric] = np.array(values, dtype=np.float64)

    n_metrics = len(metrics)
    cols = (n_metrics + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 6))
    axes = np.atleast_2d(axes).flatten()

    for ax, metric in zip(axes, metrics, strict=False):
        values = data[metric]
        ax.hist(values, bins=40, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(values), color="r", linestyle="--", label="mean")
        ax.axvline(np.median(values), color="g", linestyle="--", label="median")
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.set_title(f"{metric.replace('_', ' ').title()}\nmean={np.mean(values):.3f}")
        ax.legend()

    for ax in axes[n_metrics:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_quality_scatter(
    quality_scores: Sequence[dict[str, Any]],
    x_metric: str = "mean_certainty",
    y_metric: str = "smoothness",
    color_metric: str = "composite_score",
    save_path: Path | None = None,
) -> None:
    """Scatter two quality metrics against each other, colored by a third.

    Args:
        quality_scores: Sequence of quality-score records.
        x_metric: Metric for the x-axis.
        y_metric: Metric for the y-axis.
        color_metric: Metric used for point color.
        save_path: Optional path to save the figure.
    """
    apply_style()

    x = np.array([float(score[x_metric]) for score in quality_scores], dtype=np.float64)
    y = np.array([float(score[y_metric]) for score in quality_scores], dtype=np.float64)
    c = np.array(
        [float(score.get(color_metric, 0.0)) for score in quality_scores],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(x, y, c=c, cmap="viridis", alpha=0.6, edgecolors="none", s=30)
    ax.set_xlabel(x_metric.replace("_", " ").title())
    ax.set_ylabel(y_metric.replace("_", " ").title())
    ax.set_title(
        f"{y_metric.replace('_', ' ').title()} vs {x_metric.replace('_', ' ').title()}"
    )
    plt.colorbar(scatter, ax=ax, label=color_metric.replace("_", " ").title())

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_quality_ranking(
    quality_scores: Sequence[dict[str, Any]],
    metric: str = "composite_score",
    top_n: int = 10,
    bottom_n: int = 10,
    save_path: Path | None = None,
) -> None:
    """Bar chart of the top/bottom spectra by a quality metric.

    Args:
        quality_scores: Sequence of quality-score records.  Each record
            must contain ``spectrum_id`` and *metric*.
        metric: Metric to rank by.
        top_n: Number of highest-scoring spectra to show.
        bottom_n: Number of lowest-scoring spectra to show.
        save_path: Optional path to save the figure.
    """
    apply_style()

    ranked = sorted(
        quality_scores,
        key=lambda score: float(score[metric]),
        reverse=True,
    )
    top = ranked[:top_n]
    bottom = ranked[-bottom_n:][::-1]

    top_ids = [score["spectrum_id"] for score in top]
    top_values = [float(score[metric]) for score in top]
    bottom_ids = [score["spectrum_id"] for score in bottom]
    bottom_values = [float(score[metric]) for score in bottom]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(range(len(top_ids)), top_values, color="green", alpha=0.7)
    axes[0].set_yticks(range(len(top_ids)))
    axes[0].set_yticklabels(top_ids, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(metric.replace("_", " ").title())
    axes[0].set_title(f"Top {top_n} by {metric.replace('_', ' ').title()}")

    axes[1].barh(range(len(bottom_ids)), bottom_values, color="red", alpha=0.7)
    axes[1].set_yticks(range(len(bottom_ids)))
    axes[1].set_yticklabels(bottom_ids, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(metric.replace("_", " ").title())
    axes[1].set_title(f"Bottom {bottom_n} by {metric.replace('_', ' ').title()}")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_column_error_heatmap(
    spectra: torch.Tensor | np.ndarray,
    true_picks: torch.Tensor | np.ndarray,
    pred_picks: torch.Tensor | np.ndarray,
    metadata: list[dict[str, Any]] | None = None,
    num_samples: int = 4,
    save_path: Path | None = None,
    seed: int = 42,
) -> None:
    """Plot spectra with per-column pick error highlighted."""
    apply_style()
    rng = np.random.default_rng(seed)

    spectra = _to_numpy(spectra)
    true_picks = _to_numpy(true_picks)
    pred_picks = _to_numpy(pred_picks)

    n = min(num_samples, spectra.shape[0])
    indices = rng.choice(spectra.shape[0], size=n, replace=False)

    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = np.atleast_2d(axes)
    axes = np.atleast_2d(axes)

    for row, idx in enumerate(indices):
        spec = spectra[idx, 0]
        meta = metadata[idx] if metadata else None
        freqs, waven = _axis_arrays(spec, meta)
        extent = _extent_from_axes(freqs, waven)

        true = true_picks[idx]
        pred = pred_picks[idx]
        valid = (true >= 0) & (pred >= 0)
        error = np.zeros_like(true, dtype=np.float32)
        error[valid] = np.abs(true[valid].astype(float) - pred[valid].astype(float))
        max_err = max(error.max(), 1e-6)
        normalized_error = error / max_err

        overlay = np.zeros((*spec.shape, 4), dtype=np.float32)
        for col in range(spec.shape[1]):
            overlay[:, col, 0] = 1.0
            overlay[:, col, 3] = normalized_error[col] * 0.7

        true_valid = true >= 0
        pred_valid = pred >= 0

        ax_spec = axes[row, 0]
        ax_spec.imshow(
            spec, aspect="auto", cmap="viridis", extent=extent, origin="upper"
        )
        ax_spec.plot(
            freqs[true_valid],
            waven[true[true_valid].astype(int)],
            "r-",
            linewidth=2,
            label="True",
        )
        ax_spec.plot(
            freqs[pred_valid],
            waven[pred[pred_valid].astype(int)],
            "g--",
            linewidth=1.5,
            label="Predicted",
        )
        ax_spec.set_xlabel("Frequency (Hz)" if meta else "Frequency index")
        ax_spec.set_ylabel("Wavenumber (1/m)" if meta else "Wavenumber index")
        title = meta.get("spectrum_id", f"Sample {idx}") if meta else f"Sample {idx}"
        ax_spec.set_title(f"{title} — Spectrum")
        ax_spec.legend()

        ax_err = axes[row, 1]
        ax_err.imshow(
            spec, aspect="auto", cmap="viridis", extent=extent, origin="upper"
        )
        ax_err.imshow(overlay, aspect="auto", extent=extent, origin="upper")
        ax_err.plot(
            freqs[true_valid],
            waven[true[true_valid].astype(int)],
            "c-",
            linewidth=2,
            label="True",
        )
        ax_err.plot(
            freqs[pred_valid],
            waven[pred[pred_valid].astype(int)],
            "g--",
            linewidth=1.5,
            label="Predicted",
        )
        ax_err.set_xlabel("Frequency (Hz)" if meta else "Frequency index")
        ax_err.set_ylabel("Wavenumber (1/m)" if meta else "Wavenumber index")
        ax_err.set_title(f"{title} — Pick Error Overlay (max={max_err:.1f}px)")
        ax_err.legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
