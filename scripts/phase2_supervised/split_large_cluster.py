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
        npz_path: Path to the feature archive.

    Returns:
        Tuple ``(features, spectrum_ids)`` with shapes ``(N, D)`` and ``(N,)``.

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


def _load_initial_labels(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load initial cluster labels and spectrum IDs.

    Args:
        npz_path: Path to the pseudo-label archive.

    Returns:
        Tuple ``(labels, spectrum_ids)`` with shapes ``(N,)`` each.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Label file not found: {npz_path}")

    data = np.load(npz_path)
    try:
        labels = np.array(data["labels"])
        spectrum_ids = np.array(data["spectrum_ids"])
    finally:
        data.close()

    return labels, spectrum_ids


def _run_umap_hdbscan(
    features: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    min_cluster_size: int,
    min_samples: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster a feature matrix with UMAP -> HDBSCAN.

    Args:
        features: Feature matrix of shape ``(N, D)``.
        n_neighbors: UMAP ``n_neighbors``.
        min_dist: UMAP ``min_dist``.
        n_components: UMAP output dimensionality.
        min_cluster_size: HDBSCAN minimum cluster size.
        min_samples: HDBSCAN ``min_samples``.
        random_seed: Random seed for UMAP.

    Returns:
        Tuple ``(umap_embeddings, labels, probabilities)``.
    """
    try:
        import umap
        import hdbscan
    except ImportError as exc:
        raise RuntimeError(
            "umap-learn and hdbscan are required. "
            "Install with: pip install umap-learn hdbscan"
        ) from exc

    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_seed,
        metric="euclidean",
    )
    umap_embeddings = reducer.fit_transform(features_std)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(umap_embeddings)
    probabilities = clusterer.probabilities_

    return umap_embeddings, labels, probabilities


def _plot_split_summary(
    umap_2d: np.ndarray,
    initial_labels: np.ndarray,
    final_labels: np.ndarray,
    target_label: int,
    output_path: Path,
) -> None:
    """Save a before/after UMAP visualization of the cluster split.

    Args:
        umap_2d: 2-D UMAP embedding of shape ``(N, 2)`` for the target cluster.
        initial_labels: Initial labels for the target cluster spectra.
        final_labels: Final merged labels for the target cluster spectra.
        target_label: The original cluster ID that was split.
        output_path: Destination ``.png`` path.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before: all points have the same initial label
    axes[0].scatter(
        umap_2d[:, 0],
        umap_2d[:, 1],
        c="steelblue",
        s=15,
        alpha=0.7,
    )
    axes[0].set_title(f"Initial Cluster {target_label}")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    # After: colored by final merged labels
    unique_labels = sorted({int(lbl) for lbl in final_labels if lbl != -1})
    cmap = plt.colormaps["tab20"].resampled(max(len(unique_labels), 1))

    has_noise = -1 in final_labels
    if has_noise:
        noise_mask = final_labels == -1
        axes[1].scatter(
            umap_2d[noise_mask, 0],
            umap_2d[noise_mask, 1],
            c="grey",
            s=8,
            alpha=0.4,
            label="Noise",
        )

    for i, lbl in enumerate(unique_labels):
        mask = final_labels == lbl
        axes[1].scatter(
            umap_2d[mask, 0],
            umap_2d[mask, 1],
            c=[cmap(i)],
            s=15,
            alpha=0.8,
            label=f"Cluster {lbl}",
        )

    axes[1].set_title(f"After Splitting Cluster {target_label}")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].legend(markerscale=2, loc="best", fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved split summary visualization to %s", output_path)


def _merge_labels(
    initial_labels: np.ndarray,
    target_mask: np.ndarray,
    sub_labels: np.ndarray,
    target_label: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Merge sub-cluster labels back into the global label array.

    Stable clusters (non-target, non-noise) keep their original IDs.
    Sub-clusters inside the target cluster receive new contiguous IDs,
    starting from ``max(initial_labels) + 1``.

    Args:
        initial_labels: Initial global labels of shape ``(N,)``.
        target_mask: Boolean mask selecting spectra belonging to the target cluster.
        sub_labels: HDBSCAN labels for the target-cluster subset.
        target_label: Original ID of the cluster being split.

    Returns:
        Tuple of ``(merged_labels, sub_label_map)`` where ``sub_label_map``
        maps original sub-label IDs to new global IDs.
    """
    merged = initial_labels.copy()
    next_label = int(np.max(initial_labels[initial_labels != -1])) + 1

    sub_label_map: dict[int, int] = {}
    for sub_lbl in sorted({int(lbl) for lbl in sub_labels if lbl != -1}):
        sub_label_map[sub_lbl] = next_label
        next_label += 1

    # Stable clusters: leave as-is except target cluster is removed
    merged[target_mask] = -1

    # Assign new sub-cluster labels
    for sub_lbl, new_lbl in sub_label_map.items():
        sub_mask = sub_labels == sub_lbl
        merged[target_mask] = np.where(sub_mask, new_lbl, merged[target_mask])

    return merged, sub_label_map


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for hierarchical cluster splitting."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Split a dominant cluster discovered by HDBSCAN into finer sub-clusters, "
            "then merge the sub-labels back into the global label array."
        )
    )
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to ``.npz`` file with keys 'features' (N, D) and 'spectrum_ids'.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to existing ``pseudo_labels.npz`` from ``cluster_pseudo_labels.py``.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/pseudo_labels_split.npz",
        help="Output path for merged labels ``.npz``.",
    )
    parser.add_argument(
        "--target-cluster",
        type=int,
        default=None,
        help=(
            "ID of the cluster to split. If omitted, the largest cluster "
            "(by member count) is chosen automatically."
        ),
    )
    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=0.50,
        help=(
            "If --target-cluster is omitted, split the largest cluster only if it "
            "contains more than this fraction of all non-noise spectra."
        ),
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors for the sub-clustering step.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.0,
        help="UMAP min_dist for the sub-clustering step.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="UMAP output dimensionality for the sub-clustering step.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=30,
        help="HDBSCAN minimum cluster size for the sub-clustering step.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="HDBSCAN min_samples for the sub-clustering step.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for UMAP reproducibility.",
    )
    args = parser.parse_args(argv)

    features_path = Path(args.features)
    labels_path = Path(args.labels)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features, feature_ids = _load_features(features_path)
    initial_labels, label_ids = _load_initial_labels(labels_path)

    if not np.array_equal(feature_ids, label_ids):
        raise ValueError(
            "Spectrum IDs in --features and --labels do not match. "
            "Both files must be aligned row-for-row."
        )

    n_total = len(initial_labels)
    n_noise = int(np.sum(initial_labels == -1))
    core_mask = initial_labels != -1

    if args.target_cluster is not None:
        target_label = args.target_cluster
        if target_label not in initial_labels:
            raise ValueError(
                f"Target cluster {target_label} not found in initial labels."
            )
    else:
        unique_labels, counts = np.unique(initial_labels[core_mask], return_counts=True)
        largest_idx = int(np.argmax(counts))
        largest_fraction = counts[largest_idx] / int(np.sum(core_mask))
        target_label = int(unique_labels[largest_idx])
        logger.info(
            "Largest cluster is %d with %d members (%.1f%% of non-noise data).",
            target_label,
            int(counts[largest_idx]),
            largest_fraction * 100,
        )
        if largest_fraction < args.auto_threshold:
            logger.warning(
                "Largest cluster is below --auto-threshold=%.2f. "
                "Skipping split; copy labels to output unchanged.",
                args.auto_threshold,
            )
            np.savez_compressed(
                output_path,
                labels=initial_labels.astype(np.int64),
                spectrum_ids=label_ids,
                n_clusters=int(len(unique_labels)),
                noise_fraction=n_noise / n_total,
            )
            return 0

    target_mask = initial_labels == target_label
    n_target = int(np.sum(target_mask))
    logger.info(
        "Splitting cluster %d which contains %d spectra.", target_label, n_target
    )

    if n_target < args.min_cluster_size * 2:
        logger.warning(
            "Target cluster has only %d members (less than 2x min_cluster_size=%d). "
            "Splitting may produce unstable results.",
            n_target,
            args.min_cluster_size,
        )

    # Run sub-clustering only on the target cluster
    sub_features = features[target_mask]
    umap_embeddings, sub_labels, sub_probabilities = _run_umap_hdbscan(
        sub_features,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        random_seed=args.random_seed,
    )

    n_sub_noise = int(np.sum(sub_labels == -1))
    sub_clusters = sorted({int(lbl) for lbl in sub_labels if lbl != -1})
    logger.info(
        "Sub-clustering found %d sub-clusters, noise=%d (%.1f%%).",
        len(sub_clusters),
        n_sub_noise,
        n_sub_noise / n_target * 100,
    )
    for lbl in sub_clusters:
        size = int(np.sum(sub_labels == lbl))
        logger.info("  Sub-cluster %d: %d members", lbl, size)

    # Compute silhouette on the target cluster only
    silhouette_val: float | None = None
    if len(sub_clusters) >= 2:
        core_sub = sub_labels != -1
        if np.sum(core_sub) > 1:
            silhouette_val = float(
                silhouette_score(umap_embeddings[core_sub], sub_labels[core_sub])
            )
            logger.info(
                "Silhouette score inside target cluster (UMAP space): %.4f",
                silhouette_val,
            )

    # Merge labels
    merged_labels, sub_label_map = _merge_labels(
        initial_labels, target_mask, sub_labels, target_label
    )

    final_clusters = sorted({int(lbl) for lbl in merged_labels if lbl != -1})
    logger.info(
        "Merged label set: %d total clusters (was %d).",
        len(final_clusters),
        len({int(lbl) for lbl in initial_labels if lbl != -1}),
    )

    # Save
    np.savez_compressed(
        output_path,
        labels=merged_labels.astype(np.int64),
        probabilities=sub_probabilities.astype(np.float64),
        sub_umap_embeddings=umap_embeddings.astype(np.float64),
        spectrum_ids=label_ids,
        target_cluster=int(target_label),
        sub_label_map=json.dumps(sub_label_map),
        n_clusters=len(final_clusters),
        noise_fraction=int(np.sum(merged_labels == -1)) / n_total,
    )
    logger.info("Saved merged labels to %s", output_path)

    # JSON sidecar
    sidecar = {
        "target_cluster": int(target_label),
        "target_size": n_target,
        "sub_clusters": {
            str(lbl): int(np.sum(sub_labels == lbl)) for lbl in sub_clusters
        },
        "sub_noise": n_sub_noise,
        "final_cluster_count": len(final_clusters),
        "final_cluster_sizes": {
            str(lbl): int(np.sum(merged_labels == lbl)) for lbl in final_clusters
        },
        "silhouette_score": silhouette_val,
        "sub_label_map": sub_label_map,
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

    # Visualization: 2-D UMAP of target cluster, before/after
    viz_path = output_path.with_suffix(".png")
    try:
        import umap as umap_lib

        viz_reducer = umap_lib.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            n_components=2,
            random_state=args.random_seed,
            metric="euclidean",
        )
        scaler = StandardScaler()
        umap_2d = viz_reducer.fit_transform(scaler.fit_transform(sub_features))
        _plot_split_summary(
            umap_2d,
            initial_labels[target_mask],
            merged_labels[target_mask],
            target_label,
            viz_path,
        )
    except Exception:
        logger.warning("Failed to generate split summary plot.", exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
