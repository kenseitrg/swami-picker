from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.features import (
    build_pca_features,
    extract_marginal_features,
    extract_spectral_descriptors,
    standardize_features,
)

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure root logger for CLI output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_spectrum_tensor(npz_path: Path) -> np.ndarray | None:
    """Load a spectrum tensor from an ``.npz`` file.

    Args:
        npz_path: Path to the ``.npz`` file.

    Returns:
        The loaded tensor, or ``None`` if loading fails.
    """
    try:
        data = np.load(npz_path)
        try:
            tensor = np.array(data["tensor"])
        finally:
            data.close()
        return tensor
    except Exception:
        logger.warning("Failed to load tensor from %s", npz_path, exc_info=True)
        return None


def _load_spectrum_axes(json_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load resized frequency and wavenumber axes from a metadata sidecar.

    Args:
        json_path: Path to the ``.json`` sidecar file.

    Returns:
        A tuple of ``(freq_axis, waven_axis)`` as ``np.ndarray`` objects, or
        ``None`` for either if the key is missing or the file is unreadable.
    """
    try:
        with open(json_path) as fh:
            metadata: dict = json.load(fh)
        freq_axis = None
        waven_axis = None
        if "freq_axis_resized" in metadata:
            freq_axis = np.array(metadata["freq_axis_resized"], dtype=np.float64)
        if "waven_axis_resized" in metadata:
            waven_axis = np.array(metadata["waven_axis_resized"], dtype=np.float64)
        return freq_axis, waven_axis
    except Exception:
        logger.warning("Failed to load metadata from %s", json_path, exc_info=True)
        return None, None


DESCRIPTOR_NAMES = [
    "freq_centroid",
    "waven_centroid",
    "freq_bandwidth_std",
    "waven_bandwidth_std",
    "freq_iqr",
    "waven_iqr",
    "low_high_freq_energy_ratio",
    "peak_velocity_band_1",
    "peak_velocity_band_2",
    "peak_velocity_band_3",
    "peak_velocity_band_4",
    "peak_velocity_band_5",
    "peak_velocity_band_6",
    "peak_velocity_band_7",
    "peak_velocity_band_8",
    "peak_velocity_band_9",
    "peak_velocity_band_10",
    "freq_skewness",
    "waven_skewness",
    "total_energy",
]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for feature extraction."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Extract pseudo-label features from preprocessed FK spectra."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/manifest.json",
        help="Path to the dataset manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/features",
        help="Directory where feature files will be written.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of PCA components for marginals.  Auto-selected by default.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.90,
        help="Cumulative variance threshold for auto PCA component selection.",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    logger.info("Loading manifest from %s", manifest_path)
    with open(manifest_path) as fh:
        manifest: dict = json.load(fh)

    entries = manifest.get("spectra", [])
    if not entries:
        logger.error("Manifest contains no spectra entries.")
        return 1

    logger.info("Found %d spectra in manifest", len(entries))

    marginal_list: list[np.ndarray] = []
    descriptor_list: list[np.ndarray] = []
    spectrum_ids: list[str] = []

    processed_dir = manifest_path.parent / "spectra"

    for entry in entries:
        spectrum_id = entry.get("spectrum_id")
        if not spectrum_id:
            logger.warning("Skipping entry with missing spectrum_id")
            continue

        npz_path = processed_dir / f"{spectrum_id}.npz"
        json_path = processed_dir / f"{spectrum_id}.json"

        tensor = _load_spectrum_tensor(npz_path)
        if tensor is None:
            logger.warning("Skipping %s: tensor load failed", spectrum_id)
            continue

        if not np.all(np.isfinite(tensor)):
            logger.warning("Skipping %s: non-finite values in tensor", spectrum_id)
            continue

        freq_axis, waven_axis = _load_spectrum_axes(json_path)

        try:
            marginal = extract_marginal_features(tensor)
            descriptor = extract_spectral_descriptors(tensor, freq_axis, waven_axis)
        except Exception:
            logger.warning(
                "Feature extraction failed for %s", spectrum_id, exc_info=True
            )
            continue

        if not np.all(np.isfinite(descriptor)):
            logger.warning(
                "Skipping %s: non-finite values in descriptor vector", spectrum_id
            )
            continue

        marginal_list.append(marginal)
        descriptor_list.append(descriptor)
        spectrum_ids.append(spectrum_id)

    if not marginal_list:
        logger.error("No spectra were successfully processed.")
        return 1

    logger.info("Successfully processed %d spectra", len(marginal_list))

    # Path A: Marginals -> PCA
    marginal_matrix = np.stack(marginal_list, axis=0)  # shape (N, 512)
    logger.info("Marginal feature matrix shape: %s", marginal_matrix.shape)

    pca_features, pca_obj = build_pca_features(
        marginal_matrix,
        n_components=args.n_components,
        variance_threshold=args.variance_threshold,
    )
    logger.info(
        "PCA reduced marginals to shape: %s (explained variance: %.4f)",
        pca_features.shape,
        float(np.sum(pca_obj.explained_variance_ratio_)),
    )

    np.savez_compressed(
        output_dir / "features_marginal.npz",
        features=pca_features,
        spectrum_ids=np.array(spectrum_ids),
        explained_variance_ratio=pca_obj.explained_variance_ratio_,
        n_components=pca_obj.n_components_,
    )
    logger.info("Saved %s", output_dir / "features_marginal.npz")

    # Path B: Spectral descriptors -> standardize
    descriptor_matrix = np.stack(descriptor_list, axis=0)  # shape (N, ~20)
    logger.info("Descriptor feature matrix shape: %s", descriptor_matrix.shape)

    scaled_descriptors, desc_scaler = standardize_features(descriptor_matrix)

    np.savez_compressed(
        output_dir / "features_descriptors.npz",
        features=scaled_descriptors,
        spectrum_ids=np.array(spectrum_ids),
        descriptor_names=np.array(DESCRIPTOR_NAMES),
    )
    logger.info("Saved %s", output_dir / "features_descriptors.npz")

    # Feature manifest mapping spectrum_id -> row index
    feature_manifest = {
        "spectrum_id_to_index": {sid: idx for idx, sid in enumerate(spectrum_ids)},
        "count": len(spectrum_ids),
        "marginal_shape": list(pca_features.shape),
        "descriptor_shape": list(scaled_descriptors.shape),
        "n_components": int(pca_obj.n_components_),
        "explained_variance_ratio": pca_obj.explained_variance_ratio_.tolist(),
    }
    manifest_out = output_dir / "feature_manifest.json"
    with open(manifest_out, "w") as fh:
        json.dump(feature_manifest, fh, indent=2)
    logger.info("Saved %s", manifest_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
