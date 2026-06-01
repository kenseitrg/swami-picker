from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.models.mae import MaskedAutoencoder
from src.utils.config import MNISTConfig


@pytest.fixture
def small_config() -> MNISTConfig:
    """Return a small MAE config for fast unit tests."""
    return MNISTConfig(
        image_size=64,
        patch_size=16,
        in_channels=1,
        embed_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_num_heads=4,
        mask_ratio=0.75,
        use_block_masking=True,
        block_size=2,
    )


@pytest.fixture
def small_model(small_config: MNISTConfig) -> MaskedAutoencoder:
    """Return a small MAE model instantiated from ``small_config``."""
    cfg = small_config
    return MaskedAutoencoder(
        img_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_depth=cfg.decoder_depth,
        decoder_num_heads=cfg.decoder_num_heads,
        mask_ratio=cfg.mask_ratio,
        use_block_masking=cfg.use_block_masking,
        block_size=cfg.block_size,
    )


@pytest.fixture
def medium_model() -> MaskedAutoencoder:
    """Return a medium MAE model with more patches for ratio statistics."""
    return MaskedAutoencoder(
        img_size=256,
        patch_size=16,
        in_channels=1,
        embed_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_num_heads=4,
        mask_ratio=0.73,
        use_block_masking=True,
        block_size=2,
    )
