"""Unit tests for configuration dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import PickingConfig


def test_picking_config_round_trip(tmp_path: Path) -> None:
    """PickingConfig must serialize to YAML and reload identically."""
    config = PickingConfig(
        training_data_path="custom.npz",
        batch_size=8,
        base_channels=8,
        embed_dim=64,
        dropout=0.2,
        k_folds=5,
    )
    path = tmp_path / "picking.yaml"
    config.save_yaml(path)
    loaded = PickingConfig.from_yaml(path)

    assert loaded.training_data_path == "custom.npz"
    assert loaded.batch_size == 8
    assert loaded.base_channels == 8
    assert loaded.embed_dim == 64
    assert loaded.dropout == 0.2
    assert loaded.k_folds == 5
    assert loaded.betas == (0.9, 0.95)


def test_picking_config_default_loads_from_project_yaml() -> None:
    """The committed Phase 4 YAML must load without errors."""
    path = Path("configs/phase4_picking.yaml")
    config = PickingConfig.from_yaml(path)

    assert config.base_channels == 16
    assert config.embed_dim == 64
    assert config.spectrum_height == 256
    assert config.dropout == 0.5
    assert config.val_fraction > 0 and config.val_fraction < 1
    assert isinstance(config.visualization_epochs, list)


def test_picking_config_rejects_unknown_keys(tmp_path: Path) -> None:
    """Unknown keys in the YAML must raise TypeError."""
    path = tmp_path / "bad.yaml"
    path.write_text("base_channels: 8\nunknown_key: 123\n")

    with pytest.raises(TypeError):
        PickingConfig.from_yaml(path)
