from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_train_val_entries(
    manifest_path: str | Path,
    val_fraction: float,
    val_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split manifest entries into train and validation sets.

    Phase 1 val lines (``split == "val"``) are always preserved in the
    validation set. An additional ``val_fraction`` of Phase 1 train
    entries is held out deterministically.

    Args:
        manifest_path: Path to ``manifest.json``.
        val_fraction: Fraction of Phase 1 train entries to move to val.
        val_seed: Seed for deterministic shuffling.

    Returns:
        A tuple of ``(train_entries, val_entries)``.

    Raises:
        ValueError: If ``val_fraction`` is not in ``[0, 1)``.
    """
    if not (0.0 <= val_fraction < 1.0):
        msg = f"val_fraction must be in [0, 1), got {val_fraction}"
        raise ValueError(msg)

    manifest_path = Path(manifest_path)
    with open(manifest_path) as fh:
        manifest: dict[str, Any] = json.load(fh)

    all_spectra = manifest.get("spectra", [])
    phase1_val = [e for e in all_spectra if e.get("split") == "val"]
    phase1_train = [e for e in all_spectra if e.get("split") == "train"]

    # Deterministic shuffle of train entries
    rng = random.Random(val_seed)
    shuffled = phase1_train.copy()
    rng.shuffle(shuffled)

    n_val_from_train = int(len(shuffled) * val_fraction)
    extra_val = shuffled[:n_val_from_train]
    train_entries = shuffled[n_val_from_train:]
    val_entries = phase1_val + extra_val

    logger.info(
        "Split: %d train (%d phase-1 train - %d held out), "
        "%d val (%d phase-1 val + %d from train)",
        len(train_entries),
        len(phase1_train),
        n_val_from_train,
        len(val_entries),
        len(phase1_val),
        n_val_from_train,
    )
    return train_entries, val_entries
