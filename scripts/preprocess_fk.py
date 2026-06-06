from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Ensure project src is on path when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (
    FKPipelineConfig,
    PreprocessedSpectrum,
    load_preprocessed_spectrum,
    preprocess_spectrum,
    save_preprocessed_spectrum,
)
from src.data.segy_reader import RawSpectrum, read_spectrum_raw
from src.utils.plot_style import apply_style, save_figure
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Preprocess FK spectra from raw SEG-Y files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase1_fk_pipeline.yaml"),
        help="Path to preprocessing config YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only one file for smoke testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory from config.",
    )
    return parser.parse_args()


def _discover_sgy_files(raw_data_dir: Path) -> list[Path]:
    """Discover all ``.sgy`` files in the raw data directory, sorted."""
    files = sorted(raw_data_dir.glob("*.sgy"))
    logger.info("Discovered %d .sgy files in %s", len(files), raw_data_dir)
    return files


def _line_number_from_filename(filepath: Path) -> int | None:
    """Extract receiver line number from a filename like ``...RL5007.sgy``."""
    import re

    match = re.search(r"RL(\d+)", filepath.name)
    if match:
        return int(match.group(1))
    return None


def _determine_val_lines(files: list[Path], config: FKPipelineConfig) -> set[int]:
    """Determine which receiver lines belong to the validation split.

    If ``config.val_lines`` is non-empty, use it directly. Otherwise,
    randomly select ~10%% of unique lines (seeded by ``config.random_seed``).

    Args:
        files: List of discovered ``.sgy`` file paths.
        config: Preprocessing configuration.

    Returns:
        Set of receiver line numbers held out for validation.
    """
    if config.val_lines:
        val_set = set(config.val_lines)
        logger.info("Using explicit val_lines from config: %s", sorted(val_set))
        return val_set

    unique_lines = sorted(
        {ln for ln in (_line_number_from_filename(f) for f in files) if ln is not None}
    )
    if not unique_lines:
        logger.warning(
            "No line numbers extracted from filenames; val split will be empty."
        )
        return set()

    n_val = max(1, int(round(len(unique_lines) * 0.1)))
    rng = random.Random(config.random_seed)
    val_set = set(rng.sample(unique_lines, n_val))
    logger.info(
        "Auto-selected %d/%d lines for validation (seed=%d): %s",
        len(val_set),
        len(unique_lines),
        config.random_seed,
        sorted(val_set),
    )
    return val_set


def _process_file(
    filepath: Path,
    config: FKPipelineConfig,
    output_dir: Path,
    val_lines: set[int],
    visualization_samples: dict[str, tuple[RawSpectrum, PreprocessedSpectrum | None]]
    | None,
) -> list[dict[str, Any]]:
    """Read, preprocess, and save all spectra from a single SEG-Y file.

    Args:
        filepath: Path to the ``.sgy`` file.
        config: Preprocessing configuration.
        output_dir: Root output directory.
        val_lines: Set of line numbers designated for validation.
        visualization_samples: Optional dict to store raw+processed samples keyed by spectrum_id.

    Returns:
        List of manifest entries for the spectra in this file.
    """
    entries: list[dict[str, Any]] = []

    try:
        raw_spectra = read_spectrum_raw(filepath)
    except Exception:
        logger.exception("Validation error reading %s; skipping file.", filepath.name)
        return entries

    spectra_subdir = output_dir / "spectra"

    for spectrum_id, raw in raw_spectra.items():
        try:
            processed = preprocess_spectrum(raw, config)
        except Exception:
            logger.exception("Preprocessing failed for %s; skipping.", spectrum_id)
            continue

        save_preprocessed_spectrum(processed, spectra_subdir)

        split = "val" if raw.line_number in val_lines else "train"
        entry = {
            "spectrum_id": spectrum_id,
            "split": split,
            "npz_path": str(Path("spectra") / f"{spectrum_id}.npz"),
            "json_path": str(Path("spectra") / f"{spectrum_id}.json"),
            "line_number": raw.line_number,
            "point_number": raw.point_number,
        }
        entries.append(entry)

        if visualization_samples is not None and spectrum_id in visualization_samples:
            visualization_samples[spectrum_id] = (raw, processed)

    return entries


def _build_manifest(
    all_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the manifest dictionary with per-split stats."""
    total = len(all_entries)
    train_count = sum(1 for e in all_entries if e["split"] == "train")
    val_count = total - train_count
    return {
        "spectra": all_entries,
        "stats": {
            "total": total,
            "train": train_count,
            "val": val_count,
        },
    }


def _plot_before_after(
    samples: dict[str, tuple[RawSpectrum, PreprocessedSpectrum | None]],
    experiment_dir: Path,
) -> None:
    """Create before/after visualization panels for sampled spectra.

    Args:
        samples: Mapping from spectrum_id to (raw, processed) pairs.
        experiment_dir: Directory where the figure will be saved.
    """
    apply_style()
    n_samples = len(samples)
    if n_samples == 0:
        logger.warning("No visualization samples available; skipping plot.")
        return

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 2)

    for ax_row, (spectrum_id, (raw, processed)) in zip(
        axes, samples.items(), strict=True
    ):
        assert processed is not None  # filtered by caller

        # Original
        ax0 = ax_row[0]
        im0 = ax0.imshow(
            raw.data.T,
            aspect="auto",
            origin="lower",
            extent=[
                raw.freq_axis.min(),
                raw.freq_axis.max(),
                raw.waven_axis.min(),
                raw.waven_axis.max(),
            ],
        )
        ax0.set_title(f"Original: {spectrum_id}")
        ax0.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("Wavenumber (1/m)")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Resized
        ax1 = ax_row[1]
        im1 = ax1.imshow(processed.tensor, aspect="auto", origin="lower")
        ax1.set_title(f"Resized: {spectrum_id}")
        ax1.set_xlabel("Pixel (freq)")
        ax1.set_ylabel("Pixel (wavenumber)")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_figure(fig, experiment_dir / "before_after_comparison.png")
    plt.close(fig)


def main() -> None:
    """Run the FK preprocessing pipeline."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    config = FKPipelineConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.random_seed)

    raw_data_dir = Path(config.raw_data_dir)
    sgy_files = _discover_sgy_files(raw_data_dir)

    if not sgy_files:
        logger.error("No .sgy files found in %s; aborting.", raw_data_dir)
        sys.exit(1)

    val_lines = _determine_val_lines(sgy_files, config)

    # Select random spectrum IDs for visualization before processing.
    # We will re-read a few files after processing for visualization, but
    # to avoid re-reading large files, we randomly pick a few files now and
    # collect samples during processing.
    rng = random.Random(config.random_seed)
    vis_file_count = min(3, len(sgy_files))
    vis_files = rng.sample(sgy_files, vis_file_count)
    visualization_samples: dict[
        str, tuple[RawSpectrum, PreprocessedSpectrum | None]
    ] = {}

    # Pre-populate visualization sample keys by reading a subset of spectra
    # from the chosen files.
    for vf in vis_files:
        try:
            temp_raw = read_spectrum_raw(vf)
            # Pick up to 2 spectra per file
            sample_keys = rng.sample(sorted(temp_raw.keys()), min(2, len(temp_raw)))
            for key in sample_keys:
                if len(visualization_samples) >= 5:
                    break
                visualization_samples[key] = (temp_raw[key], None)
        except Exception:
            logger.warning(
                "Could not pre-sample from %s for visualization; skipping.", vf.name
            )

    if args.dry_run:
        sgy_files = sgy_files[:1]
        logger.info(
            "DRY-RUN mode: processing only the first file (%s).", sgy_files[0].name
        )

    all_entries: list[dict[str, Any]] = []
    per_line_counts: dict[int, int] = {}

    for file_idx, filepath in enumerate(sgy_files, start=1):
        logger.info(
            "Processing file %d/%d: %s", file_idx, len(sgy_files), filepath.name
        )

        # For dry-run, also include visualization samples if they come from this file.
        vis_store = visualization_samples if not args.dry_run or file_idx == 1 else None
        entries = _process_file(filepath, config, output_dir, val_lines, vis_store)
        all_entries.extend(entries)

        for entry in entries:
            per_line_counts[entry["line_number"]] = (
                per_line_counts.get(entry["line_number"], 0) + 1
            )

        logger.info("  → wrote %d spectra from %s", len(entries), filepath.name)

    manifest = _build_manifest(all_entries)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(
        "Wrote manifest: %d total spectra (train=%d, val=%d) to %s",
        manifest["stats"]["total"],
        manifest["stats"]["train"],
        manifest["stats"]["val"],
        manifest_path,
    )

    # Per-line counts
    for line_number in sorted(per_line_counts):
        logger.info("  Line %d: %d spectra", line_number, per_line_counts[line_number])

    # Config snapshot
    snapshot_path = output_dir / "config_snapshot.yaml"
    config.save_yaml(snapshot_path)
    logger.info("Saved config snapshot to %s", snapshot_path)

    # Visualization
    experiment_dir = (
        Path("experiments") / f"{date.today().isoformat()}_phase1-fk-pipeline"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Filter visualization_samples to those that were successfully processed.
    valid_samples = {k: v for k, v in visualization_samples.items() if v[1] is not None}
    if len(valid_samples) < 3 and len(all_entries) > 0:
        # Fallback: sample random processed spectra from manifest and load raw from disk.
        # This is a best-effort fallback for cases where pre-selected samples failed.
        fallback_ids = rng.sample(
            [e["spectrum_id"] for e in all_entries],
            min(5, len(all_entries)),
        )
        for sid in fallback_ids:
            if sid in valid_samples:
                continue
            entry = next(e for e in all_entries if e["spectrum_id"] == sid)
            # Try to locate raw file by line number
            line_num = entry["line_number"]
            raw_file = next(
                (f for f in sgy_files if _line_number_from_filename(f) == line_num),
                None,
            )
            if raw_file is None:
                continue
            try:
                raw_dict = read_spectrum_raw(raw_file)
                if sid in raw_dict:
                    proc = load_preprocessed_spectrum(sid, output_dir / "spectra")
                    valid_samples[sid] = (raw_dict[sid], proc)
            except Exception:
                logger.warning(
                    "Fallback visualization load failed for %s; skipping.", sid
                )
            if len(valid_samples) >= 5:
                break

    _plot_before_after(valid_samples, experiment_dir)
    logger.info("Preprocessing complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
