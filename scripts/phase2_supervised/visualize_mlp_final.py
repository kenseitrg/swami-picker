"""Generate publication-quality figures for the final MLP classifier.

Produces four figures:
1. Training metrics curves (loss, accuracy, LR, silhouette)
2. UMAP 2D projection of 128-D penultimate embeddings
3. Per-cluster example spectra (representative FK panels)
4. Cosine-similarity matrix + per-cluster silhouette scores

Usage:
    python scripts/phase2_supervised/visualize_mlp_final.py \
        --metrics experiments/phase2c-mlp-11cl-final-100ep/metrics.jsonl \
        --embeddings data/processed/mlp_embeddings_phase3.npz \
        --manifest data/processed/manifest.json \
        --output-dir experiments/phase2c-mlp-11cl-final-100ep/plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.plot_style import apply_style, save_figure


def _load_metrics(metrics_path: Path) -> list[dict]:
    records = []
    with open(metrics_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _plot_training_curves(records: list[dict], output_dir: Path) -> None:
    """Multi-panel figure: loss, accuracy, LR, silhouette."""
    import matplotlib.pyplot as plt

    apply_style()
    epochs = [r["epoch"] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel 1: Loss
    ax = axes[0, 0]
    ax.plot(epochs, [r["train_loss"] for r in records], "-", lw=1.2, label="Train")
    ax.plot(epochs, [r["val_loss"] for r in records], "-", lw=1.2, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [r["train_acc"] * 100 for r in records], "-", lw=1.2, label="Train")
    ax.plot(epochs, [r["val_acc"] * 100 for r in records], "-", lw=1.2, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training / Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, [r["lr"] for r in records], "-", color="C2", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule (Cosine with Warmup)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Panel 4: Silhouette
    ax = axes[1, 1]
    ax.plot(
        epochs, [r.get("val_silhouette", 0.0) for r in records], "-", color="C3", lw=1.2
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Validation Penultimate Silhouette")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir / "01_training_curves.png")
    plt.close(fig)
    print(f"Saved training curves to {output_dir / '01_training_curves.png'}")


def _plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    spectrum_ids: np.ndarray,
    output_dir: Path,
) -> None:
    """UMAP 2D projection colored by cluster label."""
    import matplotlib.pyplot as plt
    import umap
    from sklearn.metrics import silhouette_score

    apply_style()

    # Compute UMAP on all labeled + noise points
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    emb_2d = reducer.fit_transform(embeddings)

    # Silhouette on labeled points
    mask = labels != -1
    sil = silhouette_score(embeddings[mask], labels[mask], metric="cosine")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot noise points in grey
    noise_mask = labels == -1
    if noise_mask.sum() > 0:
        ax.scatter(
            emb_2d[noise_mask, 0],
            emb_2d[noise_mask, 1],
            c="lightgray",
            s=12,
            alpha=0.5,
            label=f"Noise ({noise_mask.sum()})",
            zorder=1,
        )

    # Plot clusters
    unique_labels = np.unique(labels[mask])
    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 2))
    for idx, lbl in enumerate(unique_labels):
        lm = labels == lbl
        ax.scatter(
            emb_2d[lm, 0],
            emb_2d[lm, 1],
            c=[cmap(idx)],
            s=20,
            alpha=0.7,
            label=f"Cluster {int(lbl)} ({lm.sum()})",
            edgecolors="none",
            zorder=2,
        )

    ax.set_title(
        f"MLP Penultimate Embeddings (128-D → UMAP 2D)\nSilhouette = {sil:.4f}"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        title="Cluster",
        title_fontsize=9,
    )

    plt.tight_layout()
    save_figure(fig, output_dir / "02_umap_embeddings.png")
    plt.close(fig)
    print(f"Saved UMAP plot to {output_dir / '02_umap_embeddings.png'}")


def _plot_cluster_examples(
    embeddings: np.ndarray,
    labels: np.ndarray,
    spectrum_ids: np.ndarray,
    manifest_path: Path,
    output_dir: Path,
) -> None:
    """Grid of representative spectra per cluster."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    apply_style()

    # Load a few spectra per cluster (closest to centroid in embedding space)
    processed_dir = manifest_path.parent / "spectra"
    unique_labels = sorted(np.unique(labels[labels != -1]))

    n_clusters = len(unique_labels)
    examples_per_cluster = 3

    fig = plt.figure(figsize=(3.5 * examples_per_cluster, 2.2 * n_clusters))
    gs = GridSpec(
        n_clusters, examples_per_cluster, figure=fig, wspace=0.15, hspace=0.25
    )

    for row, lbl in enumerate(unique_labels):
        mask = labels == lbl
        cluster_emb = embeddings[mask]
        cluster_ids = spectrum_ids[mask]

        # Find centroid and closest examples
        centroid = cluster_emb.mean(axis=0)
        dists = np.linalg.norm(cluster_emb - centroid, axis=1)
        closest_idx = np.argsort(dists)[:examples_per_cluster]

        for col, idx in enumerate(closest_idx):
            sid = cluster_ids[idx]
            npz_path = processed_dir / f"{sid}.npz"
            data = np.load(npz_path)
            try:
                tensor = np.array(data["tensor"])
            finally:
                data.close()

            if tensor.ndim == 3:
                tensor = tensor[0]

            ax = fig.add_subplot(gs[row, col])
            ax.imshow(tensor, cmap="viridis", aspect="auto", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                ax.set_ylabel(f"Cluster {int(lbl)}\n(n={mask.sum()})", fontsize=9)
            if row == 0:
                ax.set_title(f"Ex {col + 1}", fontsize=9)

    fig.suptitle(
        "Representative FK Spectra per Cluster (closest to embedding centroid)",
        fontsize=12,
        y=1.0,
    )
    plt.tight_layout()
    save_figure(fig, output_dir / "03_cluster_examples.png")
    plt.close(fig)
    print(f"Saved cluster examples to {output_dir / '03_cluster_examples.png'}")


def _plot_similarity_and_silhouette(
    embeddings: np.ndarray, labels: np.ndarray, output_dir: Path
) -> None:
    """Cosine similarity matrix + per-cluster silhouette bar chart."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    apply_style()

    mask = labels != -1
    emb_labeled = embeddings[mask]
    lab_labeled = labels[mask]

    unique_labels = np.unique(lab_labeled)
    n_classes = len(unique_labels)

    # Cosine similarity matrix
    sim = cosine_similarity(emb_labeled)
    sim_matrix = np.zeros((n_classes, n_classes))
    for i, li in enumerate(unique_labels):
        for j, lj in enumerate(unique_labels):
            if i > j:
                sim_matrix[i, j] = sim_matrix[j, i]
                continue
            sims = sim[(lab_labeled == li)[:, None] & (lab_labeled == lj)[None, :]]
            sim_matrix[i, j] = sims.mean()

    # Per-cluster silhouette
    sample_sil = silhouette_samples(emb_labeled, lab_labeled, metric="cosine")
    cluster_sil = []
    for lbl in unique_labels:
        cluster_sil.append(sample_sil[lab_labeled == lbl].mean())
    overall_sil = silhouette_score(emb_labeled, lab_labeled, metric="cosine")

    # Plot
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.3)

    # Left: similarity matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax1.set_xticks(np.arange(n_classes))
    ax1.set_yticks(np.arange(n_classes))
    ax1.set_xticklabels([str(int(lbl)) for lbl in unique_labels])
    ax1.set_yticklabels([str(int(lbl)) for lbl in unique_labels])
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Cluster")
    ax1.set_title("Mean Cosine Similarity Between Clusters")

    for i in range(n_classes):
        for j in range(n_classes):
            val = sim_matrix[i, j]
            color = "white" if abs(val) > 0.65 else "black"
            ax1.text(
                j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8
            )

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Cosine Similarity")

    # Right: per-cluster silhouette
    ax2 = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(n_classes)
    bars = ax2.barh(
        y_pos, cluster_sil, color="steelblue", edgecolor="black", linewidth=0.5
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"Cluster {int(lbl)}" for lbl in unique_labels])
    ax2.set_xlabel("Silhouette Score")
    ax2.set_title(f"Per-Cluster Silhouette (overall = {overall_sil:.4f})")
    ax2.axvline(
        x=overall_sil,
        color="red",
        linestyle="--",
        lw=1.5,
        label=f"Overall ({overall_sil:.3f})",
    )
    ax2.legend()
    ax2.set_xlim(-0.1, 1.0)
    ax2.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, cluster_sil):
        ax2.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    save_figure(fig, output_dir / "04_similarity_silhouette.png")
    plt.close(fig)
    print(
        f"Saved similarity + silhouette to {output_dir / '04_similarity_silhouette.png'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualise final MLP classifier results"
    )
    parser.add_argument(
        "--metrics", type=str, required=True, help="Path to metrics.jsonl"
    )
    parser.add_argument(
        "--embeddings", type=str, required=True, help="Path to embeddings .npz"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/manifest.json",
        help="Dataset manifest",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for figures"
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    records = _load_metrics(Path(args.metrics))
    emb_data = np.load(Path(args.embeddings))
    embeddings = np.array(emb_data["embeddings"])
    labels = np.array(emb_data["labels"])
    spectrum_ids = np.array(emb_data["spectrum_ids"])
    emb_data.close()

    n_clusters = len(np.unique(labels[labels != -1]))
    print(
        f"Loaded {len(records)} epochs, {len(embeddings)} embeddings, {n_clusters} clusters"
    )

    # Generate figures
    _plot_training_curves(records, output_dir)
    _plot_umap(embeddings, labels, spectrum_ids, output_dir)
    _plot_cluster_examples(
        embeddings, labels, spectrum_ids, Path(args.manifest), output_dir
    )
    _plot_similarity_and_silhouette(embeddings, labels, output_dir)

    print(f"\nAll figures saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
