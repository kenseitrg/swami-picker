#!/usr/bin/env python3
"""Build a Phase-3 embeddings file directly from clustering results.

When the MLP classifier produces poor embeddings (e.g. negative Silhouette),
use the raw clustering artifacts instead: cluster labels from HDBSCAN and
an embedding representation such as UMAP coordinates or standardized
descriptors.  The resulting ``.npz`` can be passed to ``prepare_session.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure root logger for CLI output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_labels(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cluster labels and spectrum IDs from a pseudo-label archive.

    Args:
        npz_path: Path to the clustering ``.npz`` file.

    Returns:
        Tuple ``(labels, spectrum_ids)`` of shape ``(N,)`` each.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Labels file not found: {npz_path}")

    data = np.load(npz_path)
    try:
        labels = np.array(data["labels"])
        spectrum_ids = np.array(data["spectrum_ids"])
    finally:
        data.close()

    logger.info(
        "Loaded labels: %d spectra, %d clusters, %d noise",
        len(labels),
        len({int(lbl) for lbl in labels if lbl != -1}),
        int(np.sum(labels == -1)),
    )
    return labels, spectrum_ids


def _load_umap_embeddings(npz_path: Path) -> np.ndarray:
    """Load UMAP embeddings from the original clustering archive.

    Args:
        npz_path: Path to the original ``pseudo_labels.npz`` file.

    Returns:
        UMAP embeddings of shape ``(N, n_components)``.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"UMAP file not found: {npz_path}")

    data = np.load(npz_path)
    try:
        embeddings = np.array(data["umap_embeddings"])
    finally:
        data.close()

    logger.info("Loaded UMAP embeddings: shape %s", embeddings.shape)
    return embeddings


def _load_features(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and spectrum IDs from a feature archive.

    Args:
        npz_path: Path to ``features_descriptors.npz`` or similar.

    Returns:
        Tuple ``(features, spectrum_ids)`` of shapes ``(N, D)`` and ``(N,)``.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature file not found: {npz_path}")

    data = np.load(npz_path)
    try:
        features = np.array(data["features"])
        spectrum_ids = np.array(data["spectrum_ids"])
    finally:
        data.close()

    logger.info("Loaded features: shape %s", features.shape)
    return features, spectrum_ids


def _align(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    target_data: np.ndarray,
) -> np.ndarray:
    """Reorder ``target_data`` so its IDs match ``source_ids``.

    Args:
        source_ids: Desired ordering of spectrum IDs.
        target_ids: Current ordering of spectrum IDs.
        target_data: Array whose rows correspond to ``target_ids``.

    Returns:
        ``target_data`` reordered to match ``source_ids``.

    Raises:
        ValueError: If the ID sets differ or duplicates exist.
    """
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_ids contains duplicates")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target_ids contains duplicates")
    if set(source_ids) != set(target_ids):
        raise ValueError(
            f"ID mismatch: source has {len(source_ids)} IDs, "
            f"target has {len(target_ids)} IDs, "
            f"intersection={len(set(source_ids) & set(target_ids))}"
        )

    target_index = {sid: idx for idx, sid in enumerate(target_ids)}
    indices = np.array([target_index[sid] for sid in source_ids], dtype=np.int64)
    return target_data[indices]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Prepare a Phase-3 embeddings file from raw clustering results."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to pseudo-labels ``.npz`` (e.g. ``pseudo_labels_split.npz``).",
    )
    parser.add_argument(
        "--embeddings-source",
        type=str,
        choices=["umap", "descriptors"],
        default="umap",
        help=(
            "Representation to use as embeddings. "
            "'umap' uses UMAP coordinates from --umap-path; "
            "'descriptors' uses standardized spectral descriptors from --features-path."
        ),
    )
    parser.add_argument(
        "--umap-path",
        type=Path,
        default=Path("data/prod/processed/pseudo_labels.npz"),
        help="Path to original clustering ``.npz`` containing 'umap_embeddings'.",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/prod/processed/features/features_descriptors.npz"),
        help="Path to feature ``.npz`` containing 'features'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ``.npz`` path for Phase 3 embeddings.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        default=True,
        help="Standardize descriptor features before saving (default: True).",
    )
    args = parser.parse_args(argv)

    labels, label_ids = _load_labels(args.labels)

    if args.embeddings_source == "umap":
        umap_ids = np.load(args.umap_path)["spectrum_ids"]
        embeddings = _load_umap_embeddings(args.umap_path)
        embeddings = _align(label_ids, np.array(umap_ids), embeddings)
    else:
        features, feature_ids = _load_features(args.features_path)
        embeddings = _align(label_ids, feature_ids, features)
        if args.standardize:
            embeddings = StandardScaler().fit_transform(embeddings)
            logger.info("Standardized descriptor features")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings.astype(np.float64),
        labels=labels.astype(np.int64),
        spectrum_ids=np.array(label_ids),
    )
    logger.info("Saved Phase 3 embeddings to %s", args.output)

    # Quick sanity report
    from sklearn.metrics import silhouette_score

    core_mask = labels != -1
    if core_mask.sum() > 1 and len({int(lbl) for lbl in labels[core_mask]}) > 1:
        sil = silhouette_score(embeddings[core_mask], labels[core_mask])
        logger.info("Silhouette score on core points: %.4f", sil)

    return 0


if __name__ == "__main__":
    sys.exit(main())
