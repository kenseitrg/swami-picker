#!/usr/bin/env python3
"""Launch the tkinter annotation application for a prepared session.

Usage::

    python scripts/phase3_active_learning/launch_app.py \\
        --session-dir annotations/2026-06-10_iter0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.picking.annotation_app import AnnotationApp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the expert annotation UI.",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to the session directory created by prepare_session.py.",
    )
    return parser.parse_args()


def main() -> int:
    """Start the annotation app."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if not args.session_dir.exists():
        logging.error("Session directory not found: %s", args.session_dir)
        return 1

    manifest_path = args.session_dir / "manifest.json"
    if not manifest_path.exists():
        logging.error("Session manifest not found: %s", manifest_path)
        return 1

    app = AnnotationApp(args.session_dir)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
