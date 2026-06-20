#!/usr/bin/env python3
"""Prepare a focused re-annotation session for low-quality spectra.

Reads the ``low_quality_spectra.json`` produced by ``run_inference.py`` and
creates a minimal annotation session containing only those spectra.  The
existing model predictions (from ``annotations_for_review/spectra/*.npz``)
are copied into the session directory as starting points, so the expert can
review and correct them instead of picking from scratch.

Usage::

    python scripts/phase4_picking/prepare_review_session.py \\
        --run-dir experiments/phase4-picking-seq-bilstm-v1 \\
        --name review_low_quality \\
        --yes

Launch the app with::

    python scripts/phase3_active_learning/launch_app.py \\
        --session-dir annotations/<date>_review_low_quality
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.picking.annotation_io import create_session_manifest, save_session_manifest
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a re-annotation session for low-quality spectra.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Experiment run directory containing low_quality_spectra.json and annotations_for_review/.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="review_low_quality",
        help="Session name slug (default: review_low_quality).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotations"),
        help="Root directory for annotation sessions (default: annotations).",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/mlp_embeddings_phase3.npz"),
        help="Path to embeddings .npz used by the annotation app (default: data/processed/mlp_embeddings_phase3.npz).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    return parser.parse_args()


def _load_low_quality_spectrum_ids(run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    path = run_dir / "low_quality_spectra.json"
    if not path.exists():
        raise FileNotFoundError(f"Low-quality report not found: {path}")

    data = json.loads(path.read_text())
    spectrum_ids = list(data["spectrum_ids"])

    logger.info("Loaded %d low-quality spectrum IDs from %s", len(spectrum_ids), path)
    return spectrum_ids, data


def _load_zero_coverage_spectrum_ids(run_dir: Path) -> list[str]:
    """Find spectra where the model predicted no picks at all.

    These are not always captured by percentile-based triage (a spectrum
    with coverage=0 can still have an average composite score), so they
    are explicitly added to the review session.
    """
    predictions_path = run_dir / "predictions.npz"
    if not predictions_path.exists():
        logger.warning(
            "Predictions file not found at %s; cannot check for zero coverage.",
            predictions_path,
        )
        return []

    data = np.load(predictions_path, allow_pickle=True)
    try:
        spectrum_ids = data["spectrum_ids"]
        picks = data["picks"]
        coverage = (picks >= 0).sum(axis=1)
        zero_indices = np.nonzero(coverage == 0)[0]
        zero_ids = [str(spectrum_ids[i]) for i in zero_indices]
    finally:
        data.close()

    logger.info(
        "Found %d spectra with zero predicted coverage in %s",
        len(zero_ids),
        predictions_path,
    )
    return zero_ids


def _load_quality_scores(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "quality_scores.json"
    if not path.exists():
        raise FileNotFoundError(f"Quality scores not found: {path}")

    scores = json.loads(path.read_text())
    return {s["spectrum_id"]: s for s in scores}


def _load_cluster_map(embeddings_path: Path) -> dict[str, int]:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    data = np.load(embeddings_path, allow_pickle=True)
    try:
        ids = data["spectrum_ids"]
        labels = data["labels"]
        return {str(sid): int(lbl) for sid, lbl in zip(ids, labels)}
    finally:
        data.close()


def _print_summary(
    spectrum_ids: list[str],
    quality_scores: dict[str, dict[str, Any]],
    cluster_map: dict[str, int],
    low_quality_count: int = 0,
    zero_coverage_count: int = 0,
    added_zero_coverage: int = 0,
) -> None:
    lines = Counter(sid.split("_")[0] for sid in spectrum_ids)
    clusters = Counter(cluster_map.get(sid, -1) for sid in spectrum_ids)

    print("\nRe-annotation session summary:")
    print(f"  Spectra: {len(spectrum_ids)}")
    print(f"    - Low quality (percentile triage): {low_quality_count}")
    print(f"    - Zero predicted coverage:         {zero_coverage_count}")
    print(f"    - Zero-coverage not already in low-quality list: {added_zero_coverage}")
    print(f"  Lines:   {len(lines)}")
    print("\nPer-line counts:")
    for line, count in sorted(lines.items()):
        print(f"    {line}: {count}")

    print("\nPer-cluster counts:")
    for cluster, count in sorted(clusters.items()):
        print(f"    C{cluster}: {count}")

    composite = [quality_scores[sid]["composite_score"] for sid in spectrum_ids]
    coverage = [quality_scores[sid]["coverage"] for sid in spectrum_ids]
    smoothness = [quality_scores[sid]["smoothness"] for sid in spectrum_ids]
    print("\nQuality of selected spectra:")
    print(
        f"  Composite: {min(composite):.3f} - {max(composite):.3f} (mean {sum(composite) / len(composite):.3f})"
    )
    print(
        f"  Coverage:  {min(coverage):.3f} - {max(coverage):.3f} (mean {sum(coverage) / len(coverage):.3f})"
    )
    print(
        f"  Smoothness: {min(smoothness):.3f} - {max(smoothness):.3f} (mean {sum(smoothness) / len(smoothness):.3f})"
    )


def _confirm(prompt: str = "Proceed with creating the review session? [Y/n] ") -> bool:
    try:
        response = input(prompt).strip().lower()
    except EOFError, KeyboardInterrupt:
        return False
    return response in ("", "y", "yes")


def main() -> int:
    args = _parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    spectrum_ids, low_quality_report = _load_low_quality_spectrum_ids(args.run_dir)
    zero_coverage_ids = _load_zero_coverage_spectrum_ids(args.run_dir)

    # Merge zero-coverage spectra into the review list, preserving order
    # and deduplicating.
    seen = set(spectrum_ids)
    added_zero_coverage = 0
    for sid in zero_coverage_ids:
        if sid not in seen:
            spectrum_ids.append(sid)
            seen.add(sid)
            added_zero_coverage += 1

    if not spectrum_ids:
        raise ValueError("No low-quality or zero-coverage spectra found.")

    quality_scores = _load_quality_scores(args.run_dir)
    cluster_map = _load_cluster_map(args.embeddings)

    _print_summary(
        spectrum_ids,
        quality_scores,
        cluster_map,
        low_quality_count=len(low_quality_report.get("spectrum_ids", [])),
        zero_coverage_count=len(zero_coverage_ids),
        added_zero_coverage=added_zero_coverage,
    )

    if not args.yes and not _confirm():
        print("Aborted.")
        return 0

    # Create session directory.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session_id = f"{now}_{args.name}"
    session_dir = args.output_dir / session_id
    spectra_dir = session_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    # Copy existing annotation records as starting points.
    source_annotations_dir = args.run_dir / "annotations_for_review" / "spectra"
    copied = 0
    missing: list[str] = []
    for spectrum_id in spectrum_ids:
        src = source_annotations_dir / f"{spectrum_id}.npz"
        if src.exists():
            shutil.copy2(src, spectra_dir / f"{spectrum_id}.npz")
            copied += 1
        else:
            missing.append(spectrum_id)

    if missing:
        logger.warning(
            "Missing annotation records for %d spectra (will start empty): %s ...",
            len(missing),
            ", ".join(missing[:5]),
        )

    # Per-cluster targets are just the actual counts in this session.
    per_cluster_target = dict(Counter(cluster_map.get(sid, -1) for sid in spectrum_ids))

    # Save config snapshot.
    config: dict[str, Any] = {
        "session_id": session_id,
        "run_dir": str(args.run_dir.resolve()),
        "low_quality_thresholds": low_quality_report.get("thresholds", {}),
        "seed": args.seed,
        "embeddings_path": str(args.embeddings.resolve()),
        "source_annotations_dir": str(source_annotations_dir.resolve()),
        "spectrum_count": len(spectrum_ids),
        "copied_from_model": copied,
        "missing_starting_annotations": len(missing),
    }
    config_path = session_dir / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=False)

    # Save manifest.
    manifest = create_session_manifest(
        session_id=session_id,
        annotator=None,
        percentage=0.0,  # Not a percentage-based session.
        query_strategy="review_low_quality",
        per_cluster_target=per_cluster_target,
        spectra_ordered=spectrum_ids,
        annotations_dir=spectra_dir,
    )
    manifest_path = session_dir / "manifest.json"
    save_session_manifest(manifest, manifest_path)

    print(f"\n✅ Review session prepared: {session_dir}")
    print(
        f"   Spectra: {len(spectrum_ids)} ({copied} pre-filled from model predictions)"
    )
    print(f"   Manifest: {manifest_path}")
    print(
        f"\nLaunch the annotation app with:\n"
        f"  python scripts/phase3_active_learning/launch_app.py "
        f"--session-dir {session_dir}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
