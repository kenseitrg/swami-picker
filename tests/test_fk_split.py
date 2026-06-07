from __future__ import annotations

from pathlib import Path

import pytest

from src.data.split import create_train_val_entries

MANIFEST_PATH = Path("data/processed/manifest.json")


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Manifest not found")
def test_split_disjoint() -> None:
    """Train and val spectrum IDs must be disjoint."""
    train_entries, val_entries = create_train_val_entries(
        MANIFEST_PATH, val_fraction=0.10, val_seed=42
    )
    train_ids = {e["spectrum_id"] for e in train_entries}
    val_ids = {e["spectrum_id"] for e in val_entries}
    assert train_ids.isdisjoint(val_ids)


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Manifest not found")
def test_split_reproducibility() -> None:
    """Same seed must yield identical splits."""
    t1, v1 = create_train_val_entries(MANIFEST_PATH, val_fraction=0.10, val_seed=42)
    t2, v2 = create_train_val_entries(MANIFEST_PATH, val_fraction=0.10, val_seed=42)
    assert [e["spectrum_id"] for e in t1] == [e["spectrum_id"] for e in t2]
    assert [e["spectrum_id"] for e in v1] == [e["spectrum_id"] for e in v2]


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Manifest not found")
def test_split_preserves_phase1_val() -> None:
    """All Phase 1 val entries must remain in the validation set."""
    train_entries, val_entries = create_train_val_entries(
        MANIFEST_PATH, val_fraction=0.10, val_seed=42
    )
    val_ids = {e["spectrum_id"] for e in val_entries}
    # Phase 1 val entries are those with split == "val" in manifest
    import json

    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    phase1_val_ids = {
        e["spectrum_id"] for e in manifest["spectra"] if e.get("split") == "val"
    }
    assert phase1_val_ids.issubset(val_ids)


@pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="Manifest not found")
def test_split_size() -> None:
    """Val size should be ~120 + 10% of 1272."""
    train_entries, val_entries = create_train_val_entries(
        MANIFEST_PATH, val_fraction=0.10, val_seed=42
    )
    assert len(val_entries) >= 120  # at least phase-1 val
    # 120 + ~127 = ~247
    assert 230 <= len(val_entries) <= 260
