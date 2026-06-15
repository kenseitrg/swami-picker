#!/usr/bin/env python3
"""Replace model predictions with manual picks from a re-annotation session.

Reads an existing ``predictions.npz`` and one or more annotation session
directories.  For every spectrum that has a manual annotation, the script
replaces the model's ``picks`` and ``presence_probs`` with the expert's picks.
Interpolated values are preserved (``wavenumber_picks``), and the
``presence_probs`` are set to the annotation confidence values:
``1.0`` for direct picks, ``0.5`` for interpolated regions, ``0.0`` for
unpicked columns.

Usage::

    python scripts/phase4_picking/merge_manual_picks.py \\
        --predictions experiments/phase4-picking-seq-bilstm-v1/predictions.npz \\
        --session-dirs annotations/2026-06-15_review_low_quality \\
        --output experiments/phase4-picking-seq-bilstm-v1/predictions_merged.npz

The original ``predictions.npz`` is never modified; a new file is written.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.picking.annotation_io import load_annotation

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge manual re-annotations into model predictions.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the original predictions.npz file.",
    )
    parser.add_argument(
        "--session-dirs",
        type=Path,
        required=True,
        nargs="+",
        help="One or more annotation session directories with spectra/*.npz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output predictions.npz path.",
    )
    parser.add_argument(
        "--min-direct-picks",
        type=int,
        default=3,
        help="Minimum direct picks for an annotation to be accepted (default: 3).",
    )
    return parser.parse_args()


def _collect_annotations(
    session_dirs: list[Path],
    min_direct_picks: int,
) -> dict[str, dict[str, Any]]:
    """Collect manual annotations across sessions.

    Later sessions override earlier ones for the same spectrum_id.

    Args:
        session_dirs: Annotation session directories.
        min_direct_picks: Minimum direct picks required.

    Returns:
        Mapping ``spectrum_id -> {"wavenumber_picks": ..., "confidence": ...}``.
    """
    annotations: dict[str, dict[str, Any]] = {}

    for session_dir in session_dirs:
        spectra_dir = session_dir / "spectra"
        if not spectra_dir.exists():
            logger.warning("No spectra directory in %s; skipping.", session_dir)
            continue

        for npz_path in sorted(spectra_dir.glob("*.npz")):
            try:
                record = load_annotation(npz_path)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", npz_path, exc)
                continue

            n_direct = int(record.direct_mask.sum())
            if n_direct < min_direct_picks:
                logger.debug(
                    "Skipping %s: only %d direct picks (< %d)",
                    record.spectrum_id,
                    n_direct,
                    min_direct_picks,
                )
                continue

            annotations[record.spectrum_id] = {
                "wavenumber_picks": record.wavenumber_picks,
                "confidence": record.confidence,
                "n_direct": n_direct,
            }

    return annotations


def _load_predictions(
    predictions_path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load inference outputs.

    Args:
        predictions_path: Path to predictions.npz.

    Returns:
        Tuple of ``(spectrum_ids, picks, presence_probs, metadata_list)``.
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


def _print_summary(
    replaced: list[str],
    untouched: int,
    direct_counts: list[int],
) -> None:
    """Print a concise summary of the merge."""
    print("\nMerge summary:")
    print(f"  Replaced spectra:  {len(replaced)}")
    print(f"  Untouched spectra: {untouched}")
    if direct_counts:
        print(
            f"  Direct picks per replaced spectrum: "
            f"{min(direct_counts)} - {max(direct_counts)} "
            f"(mean {sum(direct_counts) / len(direct_counts):.1f})"
        )

    lines = Counter(sid.split("_")[0] for sid in replaced)
    print("\nPer-line replaced counts:")
    for line, count in sorted(lines.items()):
        print(f"    {line}: {count}")


def main() -> int:
    """Merge manual picks into model predictions."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    logger.info("Loading manual annotations from %d session(s)", len(args.session_dirs))
    annotations = _collect_annotations(args.session_dirs, args.min_direct_picks)
    if not annotations:
        logger.error("No valid manual annotations found. Aborting.")
        return 1
    logger.info("Found %d valid manual annotations", len(annotations))

    logger.info("Loading predictions from %s", args.predictions)
    spectrum_ids, picks, presence_probs, metadata = _load_predictions(args.predictions)

    replaced: list[str] = []
    direct_counts: list[int] = []
    for idx, spectrum_id in enumerate(spectrum_ids):
        if spectrum_id not in annotations:
            continue

        annotation = annotations[spectrum_id]
        picks[idx] = annotation["wavenumber_picks"]
        presence_probs[idx] = annotation["confidence"]
        replaced.append(spectrum_id)
        direct_counts.append(annotation["n_direct"])

    untouched = len(spectrum_ids) - len(replaced)
    _print_summary(replaced, untouched, direct_counts)

    # Write merged predictions.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        spectrum_ids=np.array(spectrum_ids, dtype=object),
        picks=picks.astype(np.int16),
        presence_probs=presence_probs.astype(np.float32),
        metadata=np.array(json.dumps(metadata), dtype=object),
    )
    logger.info("Saved merged predictions to %s", args.output)
    print(f"\n✅ Merged predictions saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
