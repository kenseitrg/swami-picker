"""Unit tests for the Phase 4 picking batch collate function."""

from __future__ import annotations

import torch

from src.data.picking_collate import picking_collate


def test_picking_collate_stacks_tensors_and_ids():
    """Collate stacks tensors and preserves string spectrum IDs."""
    batch = []
    for i in range(3):
        batch.append(
            (
                torch.randn(1, 256, 256),
                torch.full((256,), -1.0),
                torch.zeros(256, dtype=torch.bool),
                torch.zeros(256),
                torch.ones(256),
                f"SPEC_{i:03d}",
            )
        )

    collated = picking_collate(batch)
    assert len(collated) == 6
    assert collated[0].shape == (3, 1, 256, 256)
    assert collated[1].shape == (3, 256)
    assert collated[5] == ["SPEC_000", "SPEC_001", "SPEC_002"]
