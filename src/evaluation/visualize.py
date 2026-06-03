from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from src.utils.plot_style import apply_style, save_figure

if TYPE_CHECKING:
    from torch import Tensor

    from src.models.mae import MaskedAutoencoder

try:
    import umap

    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)


def _create_masked_image(imgs: Tensor, mask: Tensor, patch_size: int) -> Tensor:
    """Create a visualisation of the masked input.

    Replaces masked patches with a neutral grey value.

    Args:
        imgs: Original images of shape ``(B, C, H, W)``.
        mask: Binary mask of shape ``(B, N)`` where ``1`` denotes a
            masked patch.
        patch_size: Size of each square patch.

    Returns:
        Masked images of shape ``(B, C, H, W)``.
    """
    B, C, H, W = imgs.shape
    p = patch_size
    n = H // p
    # Expand mask to per-pixel mask
    mask_2d = mask.reshape(B, n, n)
    mask_pixels = mask_2d.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2)
    mask_pixels = mask_pixels.unsqueeze(1)  # (B, 1, H, W)
    # Neutral grey = mean of the batch
    neutral = imgs.mean()
    return imgs * (1 - mask_pixels) + neutral * mask_pixels


def plot_reconstruction_grid(
    model: MaskedAutoencoder,
    images: Tensor,
    device: torch.device,
    save_path: Path,
    num_samples: int = 8,
    seed: int = 42,
) -> None:
    """Generate a side-by-side reconstruction grid.

    Shows input, masked input, and reconstructed output for a random
    subset of the provided images.

    Args:
        model: MAE model instance.
        images: Image tensor of shape ``(B, C, H, W)``.
        device: Device to run inference on.
        save_path: Output figure path.
        num_samples: Number of samples to display.
        seed: Random seed for sample selection.
    """
    apply_style()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=min(num_samples, len(images)), replace=False)
    imgs = images[indices].to(device)
    actual_samples = len(imgs)

    model.eval()
    with torch.no_grad():
        loss, pred, mask = model(imgs)

    recon = model.unpatchify(pred).cpu()
    masked = _create_masked_image(imgs.cpu(), mask.cpu(), model.patch_size)

    fig, axes = plt.subplots(actual_samples, 3, figsize=(6, 2 * actual_samples))
    if actual_samples == 1:
        axes = axes.reshape(1, -1)

    titles = ["Input", "Masked", "Reconstructed"]
    for row in range(actual_samples):
        # Shared vmin/vmax per row based on the original input
        row_input = imgs[row, 0].cpu().numpy()
        vmin, vmax = row_input.min(), row_input.max()
        for col, tensor in enumerate([imgs.cpu(), masked, recon]):
            ax = axes[row, col]
            img = tensor[row, 0].numpy()
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col])

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info("Saved reconstruction grid to %s", save_path)


def plot_masking_examples(
    model: MaskedAutoencoder,
    images: Tensor,
    device: torch.device,
    save_path: Path,
    num_samples: int = 4,
) -> None:
    """Visualise block masking applied to a sample batch.

    Args:
        model: MAE model instance.
        images: Image tensor of shape ``(B, C, H, W)``.
        device: Device to run inference on.
        save_path: Output figure path.
        num_samples: Number of samples to display.
    """
    apply_style()
    actual_samples = min(num_samples, len(images))
    imgs = images[:actual_samples].to(device)

    model.eval()
    with torch.no_grad():
        _loss, _pred, mask = model(imgs)

    masked = _create_masked_image(imgs.cpu(), mask.cpu(), model.patch_size)

    fig, axes = plt.subplots(2, actual_samples, figsize=(2 * actual_samples, 4))
    if actual_samples == 1:
        axes = axes.reshape(2, 1)
    for i in range(actual_samples):
        axes[0, i].imshow(imgs[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original")

        axes[1, i].imshow(masked[i, 0].numpy(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Masked")

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info("Saved masking examples to %s", save_path)


def plot_umap_embeddings(
    model: MaskedAutoencoder,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    max_samples: int = 2000,
    seed: int = 42,
) -> float | None:
    """Generate a UMAP projection of encoder embeddings coloured by label.

    Args:
        model: MAE model instance.
        loader: DataLoader yielding ``(images, labels)``.
        device: Device to run inference on.
        save_path: Output figure path.
        max_samples: Maximum number of samples to embed.

    Returns:
        Silhouette score (or ``None`` if fewer than 2 clusters).
    """
    if not _UMAP_AVAILABLE:
        logger.warning("UMAP not available; skipping embedding plot.")
        return None

    apply_style()
    model.eval()

    all_embs: list[Tensor] = []
    all_labels: list[Tensor] = []
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            embs = model.extract_embeddings(images)
            all_embs.append(embs.cpu())
            all_labels.append(labels)
            total += len(images)
            if total >= max_samples:
                break

    if not all_embs:
        logger.warning("Empty DataLoader; skipping UMAP plot.")
        return None

    embeddings = torch.cat(all_embs)[:max_samples].numpy()
    labels = torch.cat(all_labels)[:max_samples].numpy()

    n_neighbors = min(15, len(embeddings) - 1)
    if n_neighbors < 2:
        logger.warning("Too few samples for UMAP; skipping embedding plot.")
        return None

    reducer = umap.UMAP(random_state=seed, n_neighbors=n_neighbors, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)

    n_clusters = len(np.unique(labels))
    sil_score: float | None = None
    if n_clusters >= 2:
        sil_score = silhouette_score(embeddings, labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.7,
    )
    ax.set_title("UMAP of MAE Encoder Embeddings")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Digit Label")

    if sil_score is not None:
        ax.text(
            0.02,
            0.98,
            f"Silhouette: {sil_score:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info("Saved UMAP plot to %s", save_path)
    return sil_score


def plot_loss_curves(
    metrics_path: Path,
    save_path: Path,
) -> None:
    """Plot training/validation loss, LR, and VRAM over epochs.

    Reads a JSONL metrics file produced by ``MetricsLogger``.

    Args:
        metrics_path: Path to ``metrics.jsonl``.
        save_path: Output figure path.
    """
    apply_style()
    if not metrics_path.exists():
        logger.warning("Metrics file not found: %s", metrics_path)
        return

    records: list[dict[str, Any]] = []
    try:
        with open(metrics_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON line in metrics file")
    except OSError as exc:
        logger.error("Failed to read metrics file: %s", exc)
        return

    if not records:
        logger.warning("No metrics records to plot.")
        return

    epochs = [r["epoch"] for r in records]
    train_loss = [r.get("train_loss", 0.0) for r in records]
    val_loss = [r.get("val_loss", 0.0) for r in records]
    lrs = [r.get("lr", 0.0) for r in records]
    vrams = [r.get("max_vram_mb", 0.0) for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(epochs, train_loss, "-o", label="Train", markersize=4)
    axes[0, 0].plot(epochs, val_loss, "-s", label="Val", markersize=4)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].set_title("Reconstruction Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, lrs, "-o", color="C2", markersize=4)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Learning Rate")
    axes[0, 1].set_title("LR Schedule")
    axes[0, 1].set_yscale("log")

    axes[1, 0].plot(epochs, vrams, "-o", color="C3", markersize=4)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Peak VRAM (MB)")
    axes[1, 0].set_title("GPU Memory")

    throughputs = [r.get("throughput_samples_per_sec", 0.0) for r in records]
    axes[1, 1].plot(epochs, throughputs, "-o", color="C4", markersize=4)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Samples / sec")
    axes[1, 1].set_title("Throughput")

    for ax in axes.flat:
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info("Saved loss curves to %s", save_path)
