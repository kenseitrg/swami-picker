"""Unit tests for the Phase 4 picking dataset."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.data.picking_dataset import FKPickingDataset


@pytest.fixture
def fake_phase4_npz(tmp_path):
    """Create a minimal Phase 4 ``.npz`` for testing."""
    n = 20
    spectra = np.random.randn(n, 1, 256, 256).astype(np.float32)
    picks = np.full((n, 256), -1, dtype=np.int16)
    for i in range(n):
        picks[i, 50:200] = np.clip(
            128 + 30 * np.sin(np.linspace(0, 4 * np.pi, 150)), 0, 255
        ).astype(np.int16)

    direct_masks = picks >= 0
    # Make one spectrum fail the min_direct_picks filter.
    direct_masks[0, :] = False
    direct_masks[0, 5] = True

    confidences = np.ones((n, 256), dtype=np.float32)
    cluster_labels = np.array([i % 4 for i in range(n)], dtype=np.int64)
    spectrum_ids = np.array([f"SPEC_{i:03d}" for i in range(n)], dtype=object)
    metadata = [{"spectrum_id": f"SPEC_{i:03d}"} for i in range(n)]

    path = tmp_path / "phase4.npz"
    np.savez(
        path,
        spectra=spectra,
        picks=picks,
        direct_masks=direct_masks,
        confidences=confidences,
        cluster_labels=cluster_labels,
        spectrum_ids=spectrum_ids,
        metadata=json.dumps(metadata),
    )
    return path


def test_load_metadata_json_string(fake_phase4_npz, tmp_path):
    """Dataset must parse metadata stored as a single JSON string."""
    ds = FKPickingDataset(
        npz_path=fake_phase4_npz,
        split="train",
        val_fraction=0.2,
        val_seed=1,
        min_direct_picks=3,
    )
    assert len(ds) == 15  # 20 -> filter 1 -> train 80% of 19
    assert isinstance(ds.metadata[0], dict)


def test_train_val_disjoint(fake_phase4_npz):
    """No spectrum ID may appear in both train and val splits."""
    train = FKPickingDataset(
        npz_path=fake_phase4_npz,
        split="train",
        val_fraction=0.2,
        val_seed=42,
        min_direct_picks=3,
    )
    val = FKPickingDataset(
        npz_path=fake_phase4_npz,
        split="val",
        val_fraction=0.2,
        val_seed=42,
        min_direct_picks=3,
    )

    train_ids = set(train.spectrum_ids.tolist())
    val_ids = set(val.spectrum_ids.tolist())
    assert train_ids.isdisjoint(val_ids)


def test_min_direct_picks_filter(fake_phase4_npz):
    """Spectra below the direct-pick threshold are excluded."""
    ds = FKPickingDataset(
        npz_path=fake_phase4_npz,
        split="train",
        val_fraction=0.2,
        val_seed=42,
        min_direct_picks=3,
    )
    assert "SPEC_000" not in ds.spectrum_ids.tolist()


def test_item_shapes_and_targets(fake_phase4_npz):
    """__getitem__ returns tensors of the expected shapes and types."""
    ds = FKPickingDataset(
        npz_path=fake_phase4_npz,
        split="train",
        val_fraction=0.2,
        val_seed=42,
        min_direct_picks=3,
    )

    (
        spectrum,
        pick_target,
        presence_target,
        direct_mask,
        confidence,
        cluster_emb,
        sid,
    ) = ds[0]

    assert spectrum.shape == (1, 256, 256)
    assert pick_target.shape == (256,)
    assert presence_target.shape == (256,)
    assert direct_mask.shape == (256,)
    assert confidence.shape == (256,)
    assert cluster_emb is None
    assert isinstance(sid, str)

    # Presence target matches non-negative picks.
    valid = pick_target >= 0
    assert torch.allclose(
        presence_target[valid], torch.ones_like(presence_target[valid])
    )


def test_split_reproducibility(fake_phase4_npz):
    """Same seed yields identical splits."""
    ds1 = FKPickingDataset(
        npz_path=fake_phase4_npz,
        split="val",
        val_fraction=0.2,
        val_seed=123,
        min_direct_picks=3,
    )
    ds2 = FKPickingDataset(
        npz_path=fake_phase4_npz,
        split="val",
        val_fraction=0.2,
        val_seed=123,
        min_direct_picks=3,
    )
    assert list(ds1.spectrum_ids) == list(ds2.spectrum_ids)
