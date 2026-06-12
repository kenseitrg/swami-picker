"""Unit tests for Phase 4 picking models."""

from __future__ import annotations

import pytest
import torch

from src.models.picking_model import (
    ClusterConditionalPickingModel,
    EncoderDecoderPickingModel,
    SimpleUNetPickingModel,
    build_picking_model,
    inference_picks,
)
from src.utils.config import PickingConfig


@pytest.fixture
def batch():
    """A small input batch."""
    return torch.randn(2, 1, 256, 256)


def test_unet_forward_shape(batch):
    """U-Net outputs two heatmaps of the correct shape."""
    model = SimpleUNetPickingModel()
    pick_logits, presence_logits = model(batch)

    assert pick_logits.shape == (2, 1, 256, 256)
    assert presence_logits.shape == (2, 1, 256, 256)


def test_encoder_decoder_forward_shape(batch):
    """Encoder-decoder outputs two heatmaps of the correct shape."""
    model = EncoderDecoderPickingModel()
    pick_logits, presence_logits = model(batch)

    assert pick_logits.shape == (2, 1, 256, 256)
    assert presence_logits.shape == (2, 1, 256, 256)


def test_cluster_conditional_forward_shape(batch):
    """Conditional model accepts an embedding and outputs heatmaps."""
    model = ClusterConditionalPickingModel()
    cluster_emb = torch.randn(2, 128)
    pick_logits, presence_logits = model(batch, cluster_emb)

    assert pick_logits.shape == (2, 1, 256, 256)
    assert presence_logits.shape == (2, 1, 256, 256)


def test_cluster_conditional_no_embedding(batch):
    """Conditional model falls back to zeros when no embedding is supplied."""
    model = ClusterConditionalPickingModel()
    pick_logits, presence_logits = model(batch, None)

    assert pick_logits.shape == (2, 1, 256, 256)
    assert presence_logits.shape == (2, 1, 256, 256)


def test_inference_argmax(batch):
    """Inference helper returns (B, W) pick indices."""
    model = SimpleUNetPickingModel()
    pick_logits, presence_logits = model(batch)
    pick_indices, presence_probs = inference_picks(pick_logits, presence_logits)

    assert pick_indices.shape == (2, 256)
    assert presence_probs.shape == (2, 256)
    assert torch.all((pick_indices >= 0) | (pick_indices == -1))
    assert torch.all((presence_probs >= 0) & (presence_probs <= 1))


def test_presence_masking(batch):
    """Low-presence columns are marked with -1."""
    pick_logits = torch.zeros(1, 1, 256, 256)
    presence_logits = torch.full((1, 1, 256, 256), -10.0)  # very low presence
    pick_indices, _ = inference_picks(
        pick_logits, presence_logits, presence_threshold=0.5
    )

    assert torch.all(pick_indices == -1)


def test_build_picking_model_unet():
    """Factory builds the U-Net model."""
    config = PickingConfig(backbone="unet", use_cluster_conditioning=False)
    model = build_picking_model(config)
    assert isinstance(model, SimpleUNetPickingModel)


def test_build_picking_model_conditional():
    """Factory builds the conditional model when requested."""
    config = PickingConfig(backbone="unet", use_cluster_conditioning=True)
    model = build_picking_model(config)
    assert isinstance(model, ClusterConditionalPickingModel)


def test_build_picking_model_encoder_decoder():
    """Factory builds the encoder-decoder model."""
    config = PickingConfig(backbone="encoder_decoder")
    model = build_picking_model(config)
    assert isinstance(model, EncoderDecoderPickingModel)


def test_build_picking_model_unknown_backbone():
    """Unknown backbone raises ValueError."""
    config = PickingConfig(backbone="transformer")
    with pytest.raises(ValueError):
        build_picking_model(config)
