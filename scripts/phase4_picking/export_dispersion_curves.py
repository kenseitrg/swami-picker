#!/usr/bin/env python3
"""Export inferred dispersion curves to inversion-ready CSV and JSON.

Converts the model-space predictions produced by ``run_inference.py`` into
physical units (Hz, 1/m, phase velocity) and writes one file per spectrum.
A combined ``all_dispersion_curves.csv`` and a ``manifest.json`` summary are
also produced.

Usage::

    python scripts/phase4_picking/export_dispersion_curves.py \\
        --predictions experiments/phase4-picking-seq-bilstm-v1/predictions.npz \\
        --output-dir exports/dispersion-curves \\
        --format both \\
        --model-version phase4-picking-seq-bilstm-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.transforms.coordinates import (
    dispersion_curve_to_dataframe,
    model_indices_to_physical,
)

logger = logging.getLogger(__name__)

# Column order exported in every CSV / JSON record.
_EXPORT_COLUMNS: list[str] = [
    "spectrum_id",
    "frequency_hz",
    "wavenumber_inv_m",
    "phase_velocity_m_s",
    "frequency_uncertainty_hz",
    "wavenumber_uncertainty_inv_m",
    "pick_certainty",
    "line_number",
    "point_number",
    "x_coord",
    "y_coord",
    "model_version",
]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export predicted dispersion curves to CSV/JSON.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions.npz from run_inference.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for exported dispersion curves.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "both"],
        default="both",
        help="Export format (default: both).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Model version string. Inferred from checkpoint directory if omitted.",
    )
    parser.add_argument(
        "--certainty-strategy",
        type=str,
        choices=["presence", "confidence", "uniform"],
        default="presence",
        help="Certainty source for uncertainty propagation (default: presence).",
    )
    parser.add_argument(
        "--min-certainty",
        type=float,
        default=0.0,
        help="Minimum pick certainty required for export (default: 0.0).",
    )
    parser.add_argument(
        "--skip-absent",
        action="store_true",
        help="Omit frequency columns where the model predicted no pick.",
    )
    return parser.parse_args()


def _infer_model_version(predictions_path: Path, override: str | None) -> str:
    """Resolve the model version string.

    If *override* is provided it is returned as-is.  Otherwise the function
    tries to extract a version from the parent directory name of the
    predictions file (e.g. ``phase4-picking-seq-bilstm-v1``).

    Args:
        predictions_path: Path to ``predictions.npz``.
        override: Optional explicit version string.

    Returns:
        Model version identifier.
    """
    if override is not None:
        return override

    # predictions.npz normally lives under experiments/<run_name>/predictions.npz
    run_dir = predictions_path.parent
    if run_dir.name in ("", "."):
        return "unknown"
    return run_dir.name


def _load_predictions(
    predictions_path: Path,
) -> tuple[Sequence[str], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load inference outputs and metadata sidecars.

    Args:
        predictions_path: Path to ``predictions.npz``.

    Returns:
        Tuple of ``(spectrum_ids, picks, presence_probs, metadata_list)``.

    Raises:
        FileNotFoundError: If *predictions_path* does not exist.
        ValueError: If required arrays are missing.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    data = np.load(predictions_path, allow_pickle=True)
    try:
        required_keys = {"spectrum_ids", "picks", "presence_probs", "metadata"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(
                f"predictions.npz missing required keys: {sorted(missing)}"
            )

        spectrum_ids = [str(sid) for sid in data["spectrum_ids"]]
        picks = np.asarray(data["picks"], dtype=np.int16)
        presence_probs = np.asarray(data["presence_probs"], dtype=np.float32)
        metadata = json.loads(str(data["metadata"].item()))
    finally:
        data.close()

    if picks.shape[0] != len(spectrum_ids):
        raise ValueError(
            f"Mismatch: {len(spectrum_ids)} spectrum_ids but picks shape {picks.shape}"
        )
    if presence_probs.shape != picks.shape:
        raise ValueError(
            f"picks shape {picks.shape} != presence_probs shape {presence_probs.shape}"
        )
    if len(metadata) != len(spectrum_ids):
        raise ValueError(
            f"Metadata length {len(metadata)} != spectrum count {len(spectrum_ids)}"
        )

    return spectrum_ids, picks, presence_probs, metadata


def _filter_dataframe(
    df: pd.DataFrame,
    min_certainty: float,
    skip_absent: bool,
) -> pd.DataFrame:
    """Apply certainty and presence filters to a dispersion DataFrame.

    Args:
        df: DataFrame from ``dispersion_curve_to_dataframe``.
        min_certainty: Minimum ``pick_certainty`` to retain.
        skip_absent: If ``True``, drop rows without a wavenumber pick.

    Returns:
        Filtered DataFrame.
    """
    if df.empty:
        return df

    if skip_absent and "wavenumber_inv_m" in df.columns:
        df = df.dropna(subset=["wavenumber_inv_m"]).reset_index(drop=True)

    if min_certainty > 0.0 and not df.empty:
        df = df[df["pick_certainty"] >= min_certainty].reset_index(drop=True)

    return df


def _write_csv(spectrum_id: str, df: pd.DataFrame, output_dir: Path) -> Path:
    """Write a per-spectrum CSV file.

    Args:
        spectrum_id: Spectrum identifier used in the filename.
        df: Filtered dispersion DataFrame.
        output_dir: Directory for CSV files.

    Returns:
        Path to the written CSV file.
    """
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / f"{spectrum_id}.csv"
    df = df.reindex(columns=_EXPORT_COLUMNS)
    df.to_csv(path, index=False, float_format="%.6g")
    return path


def _write_json(spectrum_id: str, df: pd.DataFrame, output_dir: Path) -> Path:
    """Write a per-spectrum JSON file.

    Args:
        spectrum_id: Spectrum identifier used in the filename.
        df: Filtered dispersion DataFrame.
        output_dir: Directory for JSON files.

    Returns:
        Path to the written JSON file.
    """
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    path = json_dir / f"{spectrum_id}.json"

    # Convert to a list of row dictionaries with native Python scalars.
    df = df.reindex(columns=_EXPORT_COLUMNS)
    rows = df.replace({np.nan: None}).to_dict(orient="records")
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=2)
    return path


def _write_combined_csv(
    output_dir: Path,
    all_frames: list[pd.DataFrame],
    model_version: str,
) -> Path:
    """Write a single combined CSV for all spectra.

    Args:
        output_dir: Export root directory.
        all_frames: List of per-spectrum DataFrames.
        model_version: Model version string.

    Returns:
        Path to the combined CSV file.
    """
    combined = (
        pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    )
    combined["model_version"] = model_version
    path = output_dir / "all_dispersion_curves.csv"
    combined = combined.reindex(columns=_EXPORT_COLUMNS + ["model_version"])
    combined.to_csv(path, index=False, float_format="%.6g")
    return path


def _write_manifest(
    output_dir: Path,
    model_version: str,
    n_spectra: int,
    n_picks: int,
    args: argparse.Namespace,
    per_spectrum_summary: list[dict[str, Any]],
) -> Path:
    """Write a manifest / metadata summary for the export run.

    Args:
        output_dir: Export root directory.
        model_version: Model version string.
        n_spectra: Number of spectra exported.
        n_picks: Total number of valid picks exported.
        args: Parsed CLI arguments.
        per_spectrum_summary: List with one summary dict per spectrum.

    Returns:
        Path to the written manifest JSON file.
    """
    manifest: dict[str, Any] = {
        "model_version": model_version,
        "predictions_source": str(args.predictions.resolve()),
        "output_directory": str(output_dir.resolve()),
        "format": args.format,
        "certainty_strategy": args.certainty_strategy,
        "min_certainty": args.min_certainty,
        "skip_absent": args.skip_absent,
        "spectrum_count": n_spectra,
        "total_picks": n_picks,
        "spectra": per_spectrum_summary,
    }
    path = output_dir / "manifest.json"
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def main() -> int:
    """Run the dispersion-curve export workflow."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    model_version = _infer_model_version(args.predictions, args.model_version)
    logger.info("Exporting dispersion curves from %s", args.predictions)
    logger.info("Model version: %s", model_version)
    logger.info("Output directory: %s", args.output_dir)

    spectrum_ids, picks, presence_probs, metadatas = _load_predictions(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    per_spectrum_summary: list[dict[str, Any]] = []
    total_picks = 0
    exported_count = 0

    for idx, spectrum_id in enumerate(spectrum_ids):
        metadata = metadatas[idx]

        physical = model_indices_to_physical(
            picks[idx],
            metadata,
            presence_probs=presence_probs[idx],
            certainty_strategy=args.certainty_strategy,
        )
        df = dispersion_curve_to_dataframe(spectrum_id, physical, metadata)
        df = _filter_dataframe(df, args.min_certainty, args.skip_absent)
        df["model_version"] = model_version

        n_valid = len(df)
        total_picks += n_valid
        exported_count += 1

        per_spectrum_summary.append(
            {
                "spectrum_id": spectrum_id,
                "n_picks": n_valid,
                "coverage": float(physical.valid_mask.mean()),
                "mean_certainty": float(
                    physical.pick_certainty[physical.valid_mask].mean()
                )
                if np.any(physical.valid_mask)
                else 0.0,
            }
        )

        if not df.empty:
            all_frames.append(df)
        if args.format in ("csv", "both"):
            _write_csv(spectrum_id, df, args.output_dir)
        if args.format in ("json", "both"):
            _write_json(spectrum_id, df, args.output_dir)

    manifest_path = _write_manifest(
        args.output_dir,
        model_version,
        exported_count,
        total_picks,
        args,
        per_spectrum_summary,
    )

    combined_path = _write_combined_csv(args.output_dir, all_frames, model_version)

    logger.info(
        "Exported %d spectra (%d total picks) to %s",
        exported_count,
        total_picks,
        args.output_dir,
    )

    print(f"\n✅ Exported {exported_count} spectra to {args.output_dir}")
    print(f"   Total picks: {total_picks}")
    print(f"   Manifest:     {manifest_path}")
    print(f"   Combined CSV: {combined_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
