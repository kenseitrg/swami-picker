#!/usr/bin/env python3
"""Pre-flight script for Phase 3 active learning annotation sessions.

Loads embeddings and cluster labels, computes per-cluster annotation
budgets, prints a summary table, and — after user confirmation —
creates the session directory with a manifest and config snapshot.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

# Ensure project src is on path when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.active_learning.query import build_annotation_order, compute_annotation_budget
from src.picking.annotation_io import create_session_manifest, save_session_manifest
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an expert annotation session.",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=15.0,
        help="Percentage of each cluster to annotate (default: 15.0).",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/mlp_embeddings_phase3.npz"),
        help="Path to the embeddings .npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotations"),
        help="Root directory for annotation sessions.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="centroid_boundary",
        choices=["centroid_boundary", "random"],
        help="Query strategy (default: centroid_boundary).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="iter0",
        help="Session name slug (default: iter0).",
    )
    parser.add_argument(
        "--no-interleave",
        action="store_true",
        help="Disable round-robin interleaving (annotate cluster-by-cluster).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    return parser.parse_args()


def _print_budget_table(
    labels: np.ndarray,
    budget: dict[int, int],
) -> None:
    """Print a formatted per-cluster budget table to stdout."""
    # Compute cluster sizes (excluding noise).
    mask = labels != -1
    valid_labels = labels[mask]
    sizes = Counter(valid_labels.tolist())

    rows: list[tuple[str, str, str, str]] = []
    for cluster_id in sorted(budget):
        size = sizes.get(cluster_id, 0)
        target = budget[cluster_id]
        pct = (target / size * 100.0) if size > 0 else 0.0
        rows.append(
            (
                str(cluster_id),
                str(size),
                str(target),
                f"{pct:.1f}%",
            )
        )

    total_size = sum(sizes.get(cid, 0) for cid in budget)
    total_target = sum(budget.values())
    total_pct = (
        (total_target / total_size * 100.0) if total_size > 0 else 0.0
    )
    rows.append(
        (
            "TOTAL",
            str(total_size),
            str(total_target),
            f"{total_pct:.1f}%",
        )
    )

    col_names = ["Cluster", "Size", "Target", "% of Cluster"]
    col_widths = [max(len(r[i]) for r in rows + [col_names]) for i in range(4)]

    def fmt_row(cells: list[str] | tuple[str, ...]) -> str:
        return "  ".join(
            cell.ljust(width) for cell, width in zip(cells, col_widths)
        )

    print(fmt_row(col_names))
    print("-" * (sum(col_widths) + 3 * 4))
    for row in rows:
        print(fmt_row(row))


def _confirm(prompt: str = "Proceed with annotation? [Y/n] ") -> bool:
    """Ask the user for confirmation via stdin."""
    try:
        response = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return response in ("", "y", "yes")


def main() -> int:
    """Execute the session preparation workflow."""
    args = _parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # ── Load data ────────────────────────────────────────────────────
    if not args.embeddings.exists():
        logger.error("Embeddings file not found: %s", args.embeddings)
        return 1

    data = np.load(args.embeddings, allow_pickle=True)
    try:
        embeddings = data["embeddings"]
        labels = data["labels"]
        spectrum_ids = data["spectrum_ids"]
    except KeyError as exc:
        logger.error("Missing key in embeddings file: %s", exc)
        return 1
    finally:
        data.close()

    logger.info(
        "Loaded %d embeddings (dim=%d), %d clusters",
        len(embeddings),
        embeddings.shape[1],
        len(np.unique(labels[labels != -1])),
    )

    # ── Compute budget ───────────────────────────────────────────────
    budget = compute_annotation_budget(labels, args.percentage)
    if not budget:
        logger.error("No valid clusters found (all labels are -1).")
        return 1

    print("\nAnnotation budget:\n")
    _print_budget_table(labels, budget)
    print()

    # ── User confirmation ────────────────────────────────────────────
    if not args.yes and not _confirm():
        print("Aborted.")
        return 0

    # ── Build annotation order ───────────────────────────────────────
    order = build_annotation_order(
        embeddings,
        labels,
        spectrum_ids,
        percentage=args.percentage,
        interleave=not args.no_interleave,
        core_fraction=None,
    )
    logger.info("Annotation queue length: %d spectra", len(order))

    # ── Create session directory ─────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session_id = f"{now}_{args.name}"
    session_dir = args.output_dir / session_id
    spectra_dir = session_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Session directory: %s", session_dir)

    # ── Save config snapshot ─────────────────────────────────────────
    config = {
        "session_id": session_id,
        "percentage": args.percentage,
        "strategy": args.strategy,
        "interleave": not args.no_interleave,
        "embeddings_path": str(args.embeddings.resolve()),
        "seed": args.seed,
    }
    config_path = session_dir / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=False)
    logger.info("Saved config snapshot to %s", config_path)

    # ── Save session manifest ────────────────────────────────────────
    manifest = create_session_manifest(
        session_id=session_id,
        annotator=None,
        percentage=args.percentage,
        query_strategy=args.strategy,
        per_cluster_target=budget,
        spectra_ordered=order,
        annotations_dir=spectra_dir,
    )
    manifest_path = session_dir / "manifest.json"
    save_session_manifest(manifest, manifest_path)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n✅ Session prepared: {session_dir}")
    print(
        f"\nLaunch the annotation app with:\n"
        f"  python scripts/phase3_active_learning/launch_app.py "
        f"--session-dir {session_dir}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
