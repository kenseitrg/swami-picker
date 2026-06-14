"""Unit tests for Phase 4 picking models."""

from __future__ import annotations

import pytest
import torch

from src.models.picking_model import (
    MultiModePickingModel,
    PickingModel,
    SeqPickingModel,
    build_picking_model,
    inference_picks,
    inference_picks_multimode,
)
from src.utils.config import PickingConfig


@pytest.fixture
def batch():
    """A small input batch."""
    return torch.randn(2, 1, 256, 256)


def test_model_forward_shape(batch):
    """Model outputs logits of shape (B, num_classes, W)."""
    model = PickingModel(spectrum_height=256)
    logits = model(batch)
    assert logits.shape == (2, 257, 256)


def test_inference_argmax(batch):
    """Inference helper returns (B, W) pick indices and presence probs."""
    model = PickingModel(spectrum_height=256)
    logits = model(batch)
    pick_indices, presence_probs = inference_picks(logits)

    assert pick_indices.shape == (2, 256)
    assert presence_probs.shape == (2, 256)
    assert torch.all((pick_indices >= 0) | (pick_indices == -1))
    assert torch.all((presence_probs >= 0) & (presence_probs <= 1))


def test_absent_class_masking():
    """Absent class is mapped to -1 pick index."""
    logits = torch.full((1, 257, 4), -10.0)
    logits[:, 256, :] = 10.0  # absent class wins
    pick_indices, _ = inference_picks(logits)
    assert torch.all(pick_indices == -1)


def test_present_class_kept():
    """Present class is kept as a valid pick index."""
    logits = torch.full((1, 257, 4), -10.0)
    logits[:, 50, :] = 10.0
    pick_indices, _ = inference_picks(logits)
    assert torch.all(pick_indices == 50)


def test_build_picking_model():
    """Factory builds the picking model."""
    config = PickingConfig(spectrum_height=256)
    model = build_picking_model(config)
    assert isinstance(model, PickingModel)


def test_build_seq_picking_model():
    """Factory builds the sequence picking model."""
    config = PickingConfig(
        model_type="seq",
        spectrum_height=256,
        seq_hidden_dim=64,
        seq_layers=1,
    )
    model = build_picking_model(config)
    assert isinstance(model, SeqPickingModel)
    logits = model(torch.randn(2, 1, 256, 256))
    assert logits.shape == (2, 257, 256)


def test_build_multimode_picking_model():
    """Factory builds the multi-mode picking model."""
    config = PickingConfig(
        model_type="multimode",
        spectrum_height=256,
        num_modes=3,
        mode_hidden_dim=64,
    )
    model = build_picking_model(config)
    assert isinstance(model, MultiModePickingModel)
    logits = model(torch.randn(2, 1, 256, 256))
    assert logits.shape == (2, 3, 257, 256)


def test_inference_picks_multimode():
    """Multi-mode inference returns valid picks and presence probs."""
    logits = torch.randn(2, 3, 257, 256)
    pick_indices, presence_probs = inference_picks_multimode(logits)

    assert pick_indices.shape == (2, 256)
    assert presence_probs.shape == (2, 256)
    assert torch.all((pick_indices >= 0) | (pick_indices == -1))
    assert torch.all((presence_probs >= 0) & (presence_probs <= 1))
