#!/usr/bin/env python3
"""Aggregate annotation sessions into a Phase 4 training dataset.

Collects all per-spectrum ``.npz`` annotations from one or more session
directories, filters by minimum direct-pick count, and writes a single
Phase 4-ready ``.npz`` file.

Usage::

    python scripts/phase3_active_learning/export_annotations.py \\
        --session-dirs annotations/2026-06-10_iter0 \\
        --session-dirs annotations/2026-06-10_iter1 \\
        --output data/processed/phase4_training_data.npz \\
        --min-direct-picks 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.picking.annotation_io import load_annotation

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export annotations for Phase 4 training.",
    )
    parser.add_argument(
        "--session-dirs",
        type=Path,
        required=True,
        nargs="+",
        help="One or more session directories containing annotations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz path (e.g. data/processed/phase4_training_data.npz).",
    )
    parser.add_argument(
        "--min-direct-picks",
        type=int,
        default=3,
        help="Minimum number of direct picks required (default: 3).",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Include spectra whose cluster label is -1 (noise).",
    )
    return parser.parse_args()


def _load_cluster_map(embeddings_path: Path) -> dict[str, int]:
    """Build spectrum_id → cluster label mapping from embeddings file."""
    data = np.load(embeddings_path, allow_pickle=True)
    try:
        ids = data["spectrum_ids"]
        labels = data["labels"]
        return {str(sid): int(lbl) for sid, lbl in zip(ids, labels)}
    finally:
        data.close()


def _collect_annotations(
    session_dirs: list[Path],
    min_direct_picks: int,
    include_noise: bool,
    cluster_map: dict[str, int],
) -> list[dict[str, Any]]:
    """Gather valid annotations across all sessions.

    Returns a list of record dicts, each containing:
    ``spectrum_id``, ``wavenumber_picks``, ``direct_mask``,
    ``confidence``, ``cluster_label``.
    """
    # Track already-seen spectrum_ids to deduplicate across sessions.
    seen: set[str] = set()
    records: list[dict[str, Any]] = []

    for session_dir in session_dirs:
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning("No manifest in %s; skipping.", session_dir)
            continue

        with open(manifest_path) as fh:
            manifest: dict[str, Any] = json.load(fh)

        raw_dir = manifest.get("annotations_dir", "spectra")
        annotations_dir = Path(raw_dir)
        # The manifest may store a relative path from the project root
        # (e.g. "annotations/2026-06-10_test01/spectra").  Try three
        # resolution strategies and pick the first one that exists.
        candidates = [
            annotations_dir,  # as-is (absolute or CWD-relative)
            session_dir.parent / annotations_dir,  # relative to project root
            session_dir / "spectra",  # fallback default
        ]
        annotations_dir = next(
            (p for p in candidates if p.exists()),
            candidates[-1],
        )

        for npz_path in sorted(annotations_dir.glob("*.npz")):
            try:
                annotation = load_annotation(npz_path)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", npz_path, exc)
                continue

            sid = annotation.spectrum_id
            if sid in seen:
                logger.debug("Deduplicating %s from %s", sid, session_dir.name)
                continue
            seen.add(sid)

            n_direct = int(annotation.direct_mask.sum())
            if n_direct < min_direct_picks:
                logger.debug(
                    "Skipping %s: only %d direct picks (< %d)",
                    sid,
                    n_direct,
                    min_direct_picks,
                )
                continue

            cluster_label = cluster_map.get(sid, -1)
            if not include_noise and cluster_label == -1:
                logger.debug("Skipping %s: noise point.", sid)
                continue

            records.append(
                {
                    "spectrum_id": sid,
                    "wavenumber_picks": annotation.wavenumber_picks,
                    "direct_mask": annotation.direct_mask,
                    "confidence": annotation.confidence,
                    "cluster_label": cluster_label,
                }
            )

    return records


def _load_spectrum(spectrum_id: str) -> np.ndarray:
    """Load the preprocessed spectrum tensor from disk.

    Args:
        spectrum_id: Canonical spectrum identifier.

    Returns:
        Tensor of shape ``(256, 256)`` in ``float32``.

    Raises:
        FileNotFoundError: If the spectrum file is missing.
    """
    npz_path = Path(f"data/processed/spectra/{spectrum_id}.npz")
    data = np.load(npz_path)
    try:
        tensor = np.array(data["tensor"], dtype=np.float32)
    finally:
        data.close()
    return tensor


def main() -> int:
    """Execute the export workflow."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # ── Resolve cluster labels ──────────────────────────────────────
    # Use the first session's config to locate the embeddings file.
    first_session = args.session_dirs[0]
    config_path = first_session / "config.yaml"
    if not config_path.exists():
        logger.error("Config not found in %s", first_session)
        return 1

    with open(config_path) as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    embeddings_path = Path(config["embeddings_path"])
    cluster_map = _load_cluster_map(embeddings_path)

    # ── Collect annotations ─────────────────────────────────────────
    records = _collect_annotations(
        args.session_dirs,
        args.min_direct_picks,
        args.include_noise,
        cluster_map,
    )
    if not records:
        logger.error("No valid annotations found. Export aborted.")
        return 1

    logger.info(
        "Collected %d valid annotations (min_direct=%d, noise=%s)",
        len(records),
        args.min_direct_picks,
        "included" if args.include_noise else "excluded",
    )

    # ── Stack into arrays ───────────────────────────────────────────
    n = len(records)
    spectra = np.empty((n, 1, 256, 256), dtype=np.float32)
    picks = np.empty((n, 256), dtype=np.int16)
    direct_masks = np.empty((n, 256), dtype=bool)
    confidences = np.empty((n, 256), dtype=np.float32)
    cluster_labels = np.empty(n, dtype=np.int64)
    spectrum_ids = np.empty(n, dtype=object)
    metadata_list: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        sid = rec["spectrum_id"]
        try:
            tensor = _load_spectrum(sid)
        except FileNotFoundError:
            logger.warning("Spectrum %s not found; skipping.", sid)
            continue

        spectra[i, 0] = tensor
        picks[i] = rec["wavenumber_picks"]
        direct_masks[i] = rec["direct_mask"]
        confidences[i] = rec["confidence"]
        cluster_labels[i] = rec["cluster_label"]
        spectrum_ids[i] = sid

        # Load metadata JSON sidecar.
        meta_path = Path(f"data/processed/spectra/{sid}.json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
        metadata_list.append(meta)

    # ── Write output ────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        spectrum_ids=spectrum_ids,
        spectra=spectra,
        picks=picks,
        direct_masks=direct_masks,
        confidences=confidences,
        cluster_labels=cluster_labels,
        metadata=json.dumps(metadata_list),
    )
    logger.info("Exported Phase 4 dataset to %s", args.output)
    print(f"\n✅ Exported {n} spectra to {args.output}")
    print(f"   Direct picks (mean): {direct_masks.sum(axis=1).mean():.1f}")
    print(f"   Noise excluded: {not args.include_noise}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
