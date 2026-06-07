from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.plot_style import apply_style

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure root logger for CLI output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_features(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and spectrum IDs from a ``.npz`` file.

    Args:
        npz_path: Path to the ``.npz`` archive.

    Returns:
        A tuple of ``(features, spectrum_ids)`` where ``features`` has shape
        ``(N, D)`` and ``spectrum_ids`` has shape ``(N,)``.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required keys are missing.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature file not found: {npz_path}")

    data = np.load(npz_path)
    try:
        features = np.array(data["features"])
        spectrum_ids = np.array(data["spectrum_ids"])
    finally:
        data.close()

    logger.info(
        "Loaded features: shape=%s, ids=%d from %s",
        features.shape,
        len(spectrum_ids),
        npz_path,
    )
    return features, spectrum_ids


def _validate_finite(features: np.ndarray) -> None:
    """Abort if the feature matrix contains non-finite values.

    Args:
        features: Feature matrix of shape ``(N, D)``.

    Raises:
        ValueError: If any value is non-finite.
    """
    if not np.all(np.isfinite(features)):
        n_bad = int(np.sum(~np.isfinite(features)))
        raise ValueError(
            f"Feature matrix contains {n_bad} non-finite values. "
            "Aborting clustering."
        )


def _plot_umap_clusters(
    umap_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    silhouette: float | None = None,
) -> None:
    """Save a 2-D UMAP scatter plot colored by HDBSCAN cluster label.

    Noise points (label ``-1``) are rendered in grey.

    Args:
        umap_2d: 2-D UMAP embedding of shape ``(N, 2)``.
        labels: Cluster labels of shape ``(N,)``.
        output_path: Destination ``.png`` path.
        silhouette: Optional Silhouette score to annotate on the plot.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = sorted(set(labels))
    has_noise = -1 in unique_labels
    if has_noise:
        unique_labels.remove(-1)

    cmap = plt.colormaps["tab20"].resampled(max(len(unique_labels), 1))

    # Plot noise first so it sits behind clusters
    if has_noise:
        noise_mask = labels == -1
        ax.scatter(
            umap_2d[noise_mask, 0],
            umap_2d[noise_mask, 1],
            c="grey",
            s=8,
            alpha=0.4,
            label="Noise",
        )

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            umap_2d[mask, 0],
            umap_2d[mask, 1],
            c=[cmap(i)],
            s=15,
            alpha=0.8,
            label=f"Cluster {lbl}",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    title = "UMAP → HDBSCAN Clustering"
    if silhouette is not None:
        title += f"\nSilhouette = {silhouette:.3f}"
    ax.set_title(title)
    ax.legend(markerscale=2, loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved cluster visualization to %s", output_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for pseudo-label clustering."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Cluster pre-extracted features with UMAP → HDBSCAN."
    )
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to ``.npz`` file with keys 'features' (N, D) and 'spectrum_ids'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/pseudo_labels.npz",
        help="Output path for clustering results ``.npz``.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="UMAP output dimensionality.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=30,
        help="HDBSCAN minimum cluster size.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="HDBSCAN min_samples.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for UMAP reproducibility.",
    )
    args = parser.parse_args(argv)

    # Optional imports with helpful error messages
    try:
        import umap
    except ImportError as exc:  # pragma: no cover
        logger.error(
            "umap-learn is required but not installed. "
            "Install it with: pip install umap-learn"
        )
        raise SystemExit(1) from exc

    try:
        import hdbscan
    except ImportError as exc:  # pragma: no cover
        logger.error(
            "hdbscan is required but not installed. "
            "Install it with: pip install hdbscan"
        )
        raise SystemExit(1) from exc

    features_path = Path(args.features)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load
    features, spectrum_ids = _load_features(features_path)
    _validate_finite(features)

    # Standardize (safety step)
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)
    logger.info(
        "Standardized features: mean≈%.2e, std≈%.2f",
        float(np.mean(features_std)),
        float(np.std(features_std)),
    )

    # UMAP
    logger.info(
        "Running UMAP (n_neighbors=%d, min_dist=%.2f, n_components=%d, seed=%d)...",
        args.n_neighbors,
        args.min_dist,
        args.n_components,
        args.random_seed,
    )
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        random_state=args.random_seed,
        metric="euclidean",
    )
    umap_embeddings = reducer.fit_transform(features_std)
    logger.info("UMAP complete: shape %s", umap_embeddings.shape)

    # HDBSCAN
    logger.info(
        "Running HDBSCAN (min_cluster_size=%d, min_samples=%d)...",
        args.min_cluster_size,
        args.min_samples,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(umap_embeddings)
    probabilities = clusterer.probabilities_

    n_noise = int(np.sum(labels == -1))
    noise_fraction = n_noise / len(labels)
    unique_clusters = sorted({int(lbl) for lbl in labels if lbl != -1})
    n_clusters = len(unique_clusters)

    logger.info("HDBSCAN found %d clusters, noise=%d (%.2f%%)", n_clusters, n_noise, noise_fraction * 100)

    # Silhouette on core (non-noise) points
    silhouette_val: float | None = None
    if n_clusters >= 2:
        core_mask = labels != -1
        if np.sum(core_mask) > 1:
            silhouette_val = float(
                silhouette_score(umap_embeddings[core_mask], labels[core_mask])
            )
            logger.info(
                "Silhouette score (core points, UMAP space): %.4f", silhouette_val
            )
    else:
        silhouette_val = -1.0

    if n_clusters == 0:
        logger.warning("HDBSCAN found 0 clusters — all points marked as noise.")
    else:
        for lbl in unique_clusters:
            size = int(np.sum(labels == lbl))
            logger.info("  Cluster %d: %d members", lbl, size)

    # Save
    np.savez_compressed(
        output_path,
        labels=labels.astype(np.int64),
        probabilities=probabilities.astype(np.float64),
        umap_embeddings=umap_embeddings.astype(np.float64),
        spectrum_ids=spectrum_ids,
        n_clusters=n_clusters,
        noise_fraction=noise_fraction,
    )
    logger.info("Saved clustering results to %s", output_path)

    # Save a JSON sidecar with cluster size distribution for easy inspection
    sidecar = {
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "cluster_sizes": {
            str(lbl): int(np.sum(labels == lbl)) for lbl in unique_clusters
        },
        "silhouette_score": silhouette_val,
        "params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "n_components": args.n_components,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "random_seed": args.random_seed,
        },
    }
    sidecar_path = output_path.with_suffix(".json")
    with open(sidecar_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)
    logger.info("Saved sidecar to %s", sidecar_path)

    # Visualization
    viz_path = output_path.with_suffix(".png")
    # Use a separate 2-D UMAP for visualization if n_components > 2
    if args.n_components > 2:
        logger.info("Fitting 2-D UMAP for visualization...")
        viz_reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            n_components=2,
            random_state=args.random_seed,
            metric="euclidean",
        )
        umap_2d = viz_reducer.fit_transform(features_std)
    else:
        umap_2d = umap_embeddings

    _plot_umap_clusters(umap_2d, labels, viz_path, silhouette_val)

    return 0


if __name__ == "__main__":
    sys.exit(main())
