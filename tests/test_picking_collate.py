"""Unit tests for the Phase 4 picking batch collate function."""

from __future__ import annotations

import torch

from src.data.picking_collate import picking_collate


def test_picking_collate_with_none_embeddings():
    """Collate stacks tensors and preserves None/string fields."""
    batch = []
    for i in range(3):
        batch.append(
            (
                torch.randn(1, 256, 256),
                torch.full((256,), -1.0),
                torch.zeros(256),
                torch.zeros(256, dtype=torch.bool),
                torch.ones(256),
                None,
                f"SPEC_{i:03d}",
            )
        )

    collated = picking_collate(batch)
    assert len(collated) == 7
    assert collated[0].shape == (3, 1, 256, 256)
    assert collated[1].shape == (3, 256)
    assert collated[5] is None
    assert collated[6] == ["SPEC_000", "SPEC_001", "SPEC_002"]


def test_picking_collate_with_tensor_embeddings():
    """Collate stacks cluster embeddings when all samples provide them."""
    batch = []
    for i in range(2):
        batch.append(
            (
                torch.randn(1, 256, 256),
                torch.full((256,), -1.0),
                torch.zeros(256),
                torch.zeros(256, dtype=torch.bool),
                torch.ones(256),
                torch.randn(128),
                f"SPEC_{i:03d}",
            )
        )

    collated = picking_collate(batch)
    assert collated[5].shape == (2, 128)
