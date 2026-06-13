from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as _mpl

_mpl.use("Agg")

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from src.utils.plot_style import apply_style, save_figure


def _compute_similarity_metrics(
    embeddings: np.ndarray,
    line_numbers: np.ndarray,
) -> dict[str, float]:
    """Compute intra-line, inter-line, and contrast metrics from embeddings.

    Args:
        embeddings: Array of shape ``(N, D)``.
        line_numbers: Array of shape ``(N,)`` with integer line labels.

    Returns:
        Dictionary with ``mean_intra``, ``mean_inter``, and ``contrast``.
    """
    unique_lines = np.unique(line_numbers)
    n_lines = len(unique_lines)

    if n_lines < 2:
        return {"mean_intra": 1.0, "mean_inter": 1.0, "contrast": 1.0}

    embeddings_t = torch.from_numpy(embeddings).float()
    embeddings_norm = torch.nn.functional.normalize(embeddings_t, dim=1)

    class_embs = {int(ln): embeddings_norm[line_numbers == ln] for ln in unique_lines}

    sim_matrix = np.zeros((n_lines, n_lines), dtype=np.float64)
    for i, li in enumerate(unique_lines):
        for j, lj in enumerate(unique_lines):
            if i > j:
                sim_matrix[i, j] = sim_matrix[j, i]
                continue
            ei = class_embs[int(li)]
            ej = class_embs[int(lj)]
            sims = torch.mm(ei, ej.t())
            sim_matrix[i, j] = sims.mean().item()

    intra = float(np.diag(sim_matrix).mean())
    mask_off = ~np.eye(n_lines, dtype=bool)
    inter = float(sim_matrix[mask_off].mean())
    contrast = intra / (inter + 1e-8)

    return {"mean_intra": intra, "mean_inter": inter, "contrast": contrast}


if TYPE_CHECKING:
    from torch import Tensor

    from src.models.cvt_mae import CvTMaskedAutoencoder
    from src.models.mae import MaskedAutoencoder

    MAEType = MaskedAutoencoder | CvTMaskedAutoencoder

try:
    import umap

    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)


def _composite_reconstruction(
    imgs: Tensor, pred: Tensor, mask: Tensor, model: MAEType
) -> Tensor:
    """Build a composite image replacing only masked patches with predictions.

    Unmasked patches keep the original pixel values; masked patches use the
    model's predicted values.  This matches the standard MAE visualisation
    style (He et al., 2022).

    Args:
        imgs: Original images ``(B, C, H, W)``.
        pred: Predicted patches ``(B, N, patch_dim)``.
        mask: Binary mask ``(B, N)`` where ``1`` = masked.
        model: The MAE model (provides ``patchify`` / ``unpatchify``).

    Returns:
        Composite image ``(B, C, H, W)``.
    """
    # Convert original to patch space
    patches = model.patchify(imgs)  # (B, N, patch_dim)

    # Expand mask to match patch_dim (same dim repeated across patch_dim)
    mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)

    # Composite: masked patches ← prediction, unmasked ← original
    composite_patches = (1 - mask_expanded) * patches + mask_expanded * pred

    return model.unpatchify(composite_patches)


def _add_unmasked_outlines(
    ax, mask_2d: np.ndarray, patch_size: int, color: str = "red", linewidth: float = 1.0
) -> None:
    """Draw thin outlines around contiguous unmasked (original) patch blocks.

    Merges adjacent unmasked cells into a single bounding rectangle so
    that e.g. a 2×2 block of unmasked patches gets one outline instead
    of four.

    Args:
        ax: Matplotlib Axes to draw on.
        mask_2d: 2-D boolean/binary mask of shape ``(n, n)`` where
            ``0`` = unmasked (original) patch.
        patch_size: Size of each square patch in pixels.
        color: Outline color.
        linewidth: Width of the outline in points.
    """
    n_h, n_w = mask_2d.shape
    visited = np.zeros((n_h, n_w), dtype=bool)

    for i in range(n_h):
        for j in range(n_w):
            if mask_2d[i, j] != 0 or visited[i, j]:
                continue

            # Expand right to find maximal width for this row segment
            width = 0
            for k in range(j, n_w):
                if mask_2d[i, k] == 0 and not visited[i, k]:
                    width += 1
                else:
                    break

            # Expand down while the full width row is unmasked
            height = 0
            for k in range(i, n_h):
                if np.all(mask_2d[k, j : j + width] == 0) and not np.any(
                    visited[k, j : j + width]
                ):
                    height += 1
                else:
                    break

            visited[i : i + height, j : j + width] = True

            rect = mpatches.Rectangle(
                (j * patch_size, i * patch_size),
                width * patch_size,
                height * patch_size,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
            ax.add_patch(rect)


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
    model: MAEType,
    images: Tensor,
    device: torch.device,
    save_path: Path,
    num_samples: int = 8,
    seed: int = 42,
) -> None:
    """Generate a side-by-side reconstruction grid.

    Shows input, masked input, and composite reconstruction (original
    pixels in unmasked patches + predicted pixels in masked patches)
    for a random subset of the provided images.

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

    recon = _composite_reconstruction(imgs, pred, mask, model).cpu()
    masked = _create_masked_image(imgs.cpu(), mask.cpu(), model.patch_size)
    mask_np = mask.cpu().numpy()
    n_patches_side = int(np.sqrt(mask_np.shape[1]))

    fig, axes = plt.subplots(actual_samples, 3, figsize=(6, 2 * actual_samples))
    if actual_samples == 1:
        axes = axes.reshape(1, -1)

    titles = ["Input", "Masked", "Composite"]
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
            # Add red outlines around original patches in the composite column
            if col == 2:
                mask_2d = mask_np[row].reshape(n_patches_side, n_patches_side)
                _add_unmasked_outlines(ax, mask_2d, model.patch_size)

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info("Saved reconstruction grid to %s", save_path)


def plot_masking_examples(
    model: MAEType,
    images: Tensor,
    device: torch.device,
    save_path: Path,
    num_samples: int = 4,
) -> None:
    """Visualise block masking and composite reconstruction on a sample batch.

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
        _loss, pred, mask = model(imgs)

    recon = _composite_reconstruction(imgs, pred, mask, model).cpu()
    masked = _create_masked_image(imgs.cpu(), mask.cpu(), model.patch_size)
    mask_np = mask.cpu().numpy()
    n_patches_side = int(np.sqrt(mask_np.shape[1]))

    fig, axes = plt.subplots(3, actual_samples, figsize=(2 * actual_samples, 6))
    if actual_samples == 1:
        axes = axes.reshape(3, 1)

    titles = ["Original", "Masked", "Composite"]
    for i in range(actual_samples):
        row_input = imgs[i, 0].cpu().numpy()
        vmin, vmax = row_input.min(), row_input.max()
        for row, tensor in enumerate([imgs.cpu(), masked, recon]):
            ax = axes[row, i]
            ax.imshow(tensor[i, 0].numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[row])
            # Add red outlines around original patches in the composite row
            if row == 2:
                mask_2d = mask_np[i].reshape(n_patches_side, n_patches_side)
                _add_unmasked_outlines(ax, mask_2d, model.patch_size)

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info("Saved masking examples to %s", save_path)


def plot_umap_embeddings(
    model: MAEType,
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


def plot_embedding_similarity_matrix(
    model: MAEType,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    max_samples: int = 2000,
    seed: int = 42,
) -> dict[str, float] | None:
    """Generate a cross-digit cosine-similarity heat-map from encoder embeddings.

    For every pair of digit classes the mean cosine similarity between *all*
    embeddings of class ``i`` and *all* embeddings of class ``j`` is computed.
    The resulting matrix is visualised as a heat-map.  A compact summary
    panel reports the mean intra-class (diagonal) and inter-class
    (off-diagonal) similarities.

    Args:
        model: MAE model instance.
        loader: DataLoader yielding ``(images, labels)``.
        device: Device to run inference on.
        save_path: Output figure path.
        max_samples: Maximum number of validation samples to process.
        seed: Random seed for sample selection (deterministic subsampling).

    Returns:
        Dictionary with ``mean_intra``, ``mean_inter``, and ``contrast``
        (intra / inter), or ``None`` if no samples were found.
    """
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
        logger.warning("Empty DataLoader; skipping similarity matrix.")
        return None

    embeddings = torch.cat(all_embs)[:max_samples]  # (N, D)
    labels = torch.cat(all_labels)[:max_samples].numpy()  # (N,)

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    if n_classes < 2:
        logger.warning("Fewer than 2 classes; skipping similarity matrix.")
        return None

    # Normalise embeddings for cosine similarity (dot product of unit vectors)
    embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)

    # Build class-indexed lists
    class_embs = {int(lbl): embeddings_norm[labels == lbl] for lbl in unique_labels}

    # Compute mean pairwise cosine similarity matrix
    sim_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i, li in enumerate(unique_labels):
        for j, lj in enumerate(unique_labels):
            if i > j:
                sim_matrix[i, j] = sim_matrix[j, i]
                continue
            ei = class_embs[int(li)]  # (Ni, D)
            ej = class_embs[int(lj)]  # (Nj, D)
            # All-pairs cosine similarities: (Ni, Nj)
            sims = torch.mm(ei, ej.t())
            sim_matrix[i, j] = sims.mean().item()

    # Summary statistics
    intra = np.diag(sim_matrix).mean()
    inter = sim_matrix.mean() - intra * n_classes / (n_classes * n_classes)
    # More robust: mean of off-diagonal elements
    mask_off = ~np.eye(n_classes, dtype=bool)
    inter = sim_matrix[mask_off].mean()
    contrast = intra / (inter + 1e-8)

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1])

    ax_mat = fig.add_subplot(gs[0, 0])
    im = ax_mat.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax_mat.set_xticks(np.arange(n_classes))
    ax_mat.set_yticks(np.arange(n_classes))
    ax_mat.set_xticklabels([str(int(lbl)) for lbl in unique_labels])
    ax_mat.set_yticklabels([str(int(lbl)) for lbl in unique_labels])
    ax_mat.set_xlabel("Digit Label")
    ax_mat.set_ylabel("Digit Label")
    ax_mat.set_title("Mean Cosine Similarity Between Embeddings")

    # Annotate cells
    for i in range(n_classes):
        for j in range(n_classes):
            val = sim_matrix[i, j]
            text_color = "white" if abs(val) > 0.65 else "black"
            ax_mat.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04, label="Cosine Similarity")

    ax_summary = fig.add_subplot(gs[0, 1])
    ax_summary.axis("off")
    summary_text = (
        f"Samples: {len(embeddings)}\n"
        f"Classes: {n_classes}\n"
        f"Embed dim: {embeddings.shape[1]}\n"
        f"\n"
        f"Mean intra-class sim:\n"
        f"  {intra:.4f}\n"
        f"Mean inter-class sim:\n"
        f"  {inter:.4f}\n"
        f"Contrast (intra/inter):\n"
        f"  {contrast:.3f}"
    )
    ax_summary.text(
        0.1,
        0.5,
        summary_text,
        transform=ax_summary.transAxes,
        verticalalignment="center",
        fontfamily="monospace",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info(
        "Saved embedding similarity matrix to %s (intra=%.4f, inter=%.4f, contrast=%.3f)",
        save_path,
        intra,
        inter,
        contrast,
    )

    return {
        "mean_intra": float(intra),
        "mean_inter": float(inter),
        "contrast": float(contrast),
    }


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


def plot_fk_reconstruction_grid(
    model: MAEType,
    images: Tensor,
    device: torch.device,
    save_path: Path,
    num_samples: int = 8,
    seed: int = 42,
) -> None:
    """Generate a side-by-side reconstruction grid for FK spectra.

    Shows input, masked input, and composite reconstruction for a
    random subset of the provided FK spectra.

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
    n = min(num_samples, len(images))
    indices = rng.choice(len(images), size=n, replace=False)
    imgs = images[indices].to(device)

    model.eval()
    with torch.no_grad():
        loss, pred, mask = model(imgs)

    recon = _composite_reconstruction(imgs, pred, mask, model).cpu()
    masked = _create_masked_image(imgs.cpu(), mask.cpu(), model.patch_size)

    fig, axes = plt.subplots(n, 3, figsize=(7, 2.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    titles = ["Input", "Masked", "Composite"]
    for row in range(n):
        row_input = imgs[row, 0].cpu().numpy()
        vmin, vmax = row_input.min(), row_input.max()
        for col, tensor in enumerate([imgs.cpu(), masked, recon]):
            ax = axes[row, col]
            img = tensor[row, 0].numpy()
            im = ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col])
        # Shared colorbar per row, attached to rightmost axis
        divider = make_axes_locatable(axes[row, -1])
        cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info("Saved FK reconstruction grid to %s", save_path)


def plot_fk_umap(
    model: MAEType,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    max_samples: int = 2000,
    seed: int = 42,
    num_inset_examples: int = 4,
) -> float | None:
    """Generate a UMAP projection of FK encoder embeddings coloured by line.

    Args:
        model: MAE model instance.
        loader: DataLoader yielding ``(images, metadata_dict)``.
        device: Device to run inference on.
        save_path: Output figure path.
        max_samples: Maximum number of samples to embed.
        seed: Random seed for UMAP.
        num_inset_examples: Number of example spectra to show as insets
            from distinct UMAP neighborhoods.

    Returns:
        Silhouette score (or ``None`` if UMAP is unavailable).
    """
    if not _UMAP_AVAILABLE:
        logger.warning("UMAP not available; skipping FK UMAP plot.")
        return None

    apply_style()
    model.eval()

    all_embs: list[Tensor] = []
    all_line_numbers: list[int] = []
    all_images: list[Tensor] = []
    total = 0

    with torch.no_grad():
        for images, metadata_batch in loader:
            images = images.to(device)
            embs = model.extract_embeddings(images)
            all_embs.append(embs.cpu())
            # metadata_batch is a list of dicts when using default collate
            if isinstance(metadata_batch, list):
                all_line_numbers.extend([m["line_number"] for m in metadata_batch])
            else:
                all_line_numbers.extend(metadata_batch.get("line_number", []))
            all_images.append(images.cpu())
            total += len(images)
            if total >= max_samples:
                break

    if not all_embs:
        logger.warning("Empty DataLoader; skipping FK UMAP plot.")
        return None

    embeddings = torch.cat(all_embs)[:max_samples].numpy()
    line_numbers = np.array(all_line_numbers[:max_samples])
    images_all = torch.cat(all_images)[:max_samples]

    n_neighbors = min(15, len(embeddings) - 1)
    if n_neighbors < 2:
        logger.warning("Too few samples for UMAP; skipping FK UMAP plot.")
        return None

    reducer = umap.UMAP(random_state=seed, n_neighbors=n_neighbors, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)

    unique_lines = np.unique(line_numbers)
    n_lines = len(unique_lines)
    sil_score: float | None = None
    if n_lines >= 2:
        try:
            sil_score = silhouette_score(embeddings, line_numbers)
        except ValueError:
            sil_score = None

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = mpl.colormaps["tab20"].resampled(max(n_lines, 2))
    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=line_numbers,
        cmap=cmap,
        s=12,
        alpha=0.7,
    )
    ax.set_title("UMAP of MAE Encoder Embeddings (colored by receiver line)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Receiver Line")

    if sil_score is not None:
        ax.text(
            0.02,
            0.98,
            f"Silhouette: {sil_score:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

    # ── Inset examples from distinct neighborhoods ───────────────────
    if num_inset_examples > 0 and len(images_all) > num_inset_examples:
        # Simple k-means-like spatial partitioning for neighborhood selection
        rng = np.random.default_rng(seed)
        # Pick random points that are spatially separated
        selected_indices: list[int] = []
        candidates = list(range(len(embedding_2d)))
        rng.shuffle(candidates)
        min_dist = 0.0
        if len(candidates) > 0:
            # Use a fraction of the data range as minimum separation
            x_range = embedding_2d[:, 0].max() - embedding_2d[:, 0].min()
            y_range = embedding_2d[:, 1].max() - embedding_2d[:, 1].min()
            min_dist = 0.15 * max(x_range, y_range)

        for idx in candidates:
            if len(selected_indices) >= num_inset_examples:
                break
            point = embedding_2d[idx]
            far_enough = all(
                np.linalg.norm(point - embedding_2d[si]) >= min_dist
                for si in selected_indices
            )
            if far_enough or not selected_indices:
                selected_indices.append(idx)

        # Fill remaining slots if spatial separation failed
        for idx in candidates:
            if len(selected_indices) >= num_inset_examples:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)

        inset_size = 0.18
        inset_positions = [
            (0.02, 0.02),
            (0.80, 0.02),
            (0.02, 0.80),
            (0.80, 0.80),
        ]
        for i, idx in enumerate(selected_indices[:num_inset_examples]):
            pos = inset_positions[i % len(inset_positions)]
            inset_ax = fig.add_axes((pos[0], pos[1], inset_size, inset_size))
            img = images_all[idx, 0].numpy()
            inset_ax.imshow(img, cmap="viridis")
            inset_ax.set_title(f"Line {line_numbers[idx]}", fontsize=7)
            inset_ax.axis("off")

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info("Saved FK UMAP plot to %s", save_path)
    return sil_score


def plot_fk_similarity_matrix(
    model: MAEType,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    max_samples: int = 2000,
    seed: int = 42,
) -> dict[str, float] | None:
    """Generate a cross-line cosine-similarity heat-map from encoder embeddings.

    Args:
        model: MAE model instance.
        loader: DataLoader yielding ``(images, metadata_dict)``.
        device: Device to run inference on.
        save_path: Output figure path.
        max_samples: Maximum number of samples to process.
        seed: Random seed for sample selection.

    Returns:
        Dictionary with ``mean_intra``, ``mean_inter``, and ``contrast``,
        or ``None`` if no samples were found.
    """
    apply_style()
    model.eval()

    all_embs: list[Tensor] = []
    all_line_numbers: list[int] = []
    total = 0

    with torch.no_grad():
        for images, metadata_batch in loader:
            images = images.to(device)
            embs = model.extract_embeddings(images)
            all_embs.append(embs.cpu())
            if isinstance(metadata_batch, list):
                all_line_numbers.extend([m["line_number"] for m in metadata_batch])
            else:
                all_line_numbers.extend(metadata_batch.get("line_number", []))
            total += len(images)
            if total >= max_samples:
                break

    if not all_embs:
        logger.warning("Empty DataLoader; skipping FK similarity matrix.")
        return None

    embeddings = torch.cat(all_embs)[:max_samples]  # (N, D)
    line_numbers = np.array(all_line_numbers[:max_samples])

    unique_lines = np.unique(line_numbers)
    n_lines = len(unique_lines)
    if n_lines < 2:
        logger.warning("Fewer than 2 lines; skipping FK similarity matrix.")
        return None

    embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)
    class_embs = {int(ln): embeddings_norm[line_numbers == ln] for ln in unique_lines}

    sim_matrix = np.zeros((n_lines, n_lines), dtype=np.float64)
    for i, li in enumerate(unique_lines):
        for j, lj in enumerate(unique_lines):
            if i > j:
                sim_matrix[i, j] = sim_matrix[j, i]
                continue
            ei = class_embs[int(li)]
            ej = class_embs[int(lj)]
            sims = torch.mm(ei, ej.t())
            sim_matrix[i, j] = sims.mean().item()

    intra = np.diag(sim_matrix).mean()
    mask_off = ~np.eye(n_lines, dtype=bool)
    inter = sim_matrix[mask_off].mean()
    contrast = intra / (inter + 1e-8)

    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1])

    ax_mat = fig.add_subplot(gs[0, 0])
    im = ax_mat.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax_mat.set_xticks(np.arange(n_lines))
    ax_mat.set_yticks(np.arange(n_lines))
    line_labels = [str(int(ln)) for ln in unique_lines]
    ax_mat.set_xticklabels(line_labels, rotation=45, ha="right")
    ax_mat.set_yticklabels(line_labels)
    ax_mat.set_xlabel("Receiver Line")
    ax_mat.set_ylabel("Receiver Line")
    ax_mat.set_title("Mean Cosine Similarity Between Embeddings (by Line)")

    for i in range(n_lines):
        for j in range(n_lines):
            val = sim_matrix[i, j]
            text_color = "white" if abs(val) > 0.65 else "black"
            ax_mat.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6,
            )

    plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04, label="Cosine Similarity")

    ax_summary = fig.add_subplot(gs[0, 1])
    ax_summary.axis("off")
    summary_text = (
        f"Samples: {len(embeddings)}\n"
        f"Lines: {n_lines}\n"
        f"Embed dim: {embeddings.shape[1]}\n"
        f"\n"
        f"Mean intra-line sim:\n"
        f"  {intra:.4f}\n"
        f"Mean inter-line sim:\n"
        f"  {inter:.4f}\n"
        f"Contrast (intra/inter):\n"
        f"  {contrast:.3f}"
    )
    ax_summary.text(
        0.1,
        0.5,
        summary_text,
        transform=ax_summary.transAxes,
        verticalalignment="center",
        fontfamily="monospace",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info(
        "Saved FK similarity matrix to %s (intra=%.4f, inter=%.4f, contrast=%.3f)",
        save_path,
        intra,
        inter,
        contrast,
    )

    return {
        "mean_intra": float(intra),
        "mean_inter": float(inter),
        "contrast": float(contrast),
    }
