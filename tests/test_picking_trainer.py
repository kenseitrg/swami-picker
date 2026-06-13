"""Smoke tests for the Phase 4 picking trainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.picking_model import PickingModel
from src.training.picking_trainer import PickingTrainer
from src.utils.config import PickingConfig


@pytest.fixture
def tiny_picking_loaders():
    """Create tiny synthetic loaders for a smoke test."""
    n = 12
    spectra = torch.randn(n, 1, 256, 256)
    picks = torch.full((n, 256), -1.0)
    picks[:, 50:200] = 100.0
    direct = (picks >= 0).bool()
    confidence = torch.ones_like(picks)
    cluster_labels = torch.zeros(n, dtype=torch.long)

    dataset = TensorDataset(
        spectra, picks, direct, confidence, cluster_labels, torch.arange(n)
    )
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [8, 4], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    return train_loader, val_loader


def test_trainer_smoke(tiny_picking_loaders, tmp_path: Path) -> None:
    """Trainer must run for two epochs and write metrics + checkpoints."""
    train_loader, val_loader = tiny_picking_loaders

    config = PickingConfig(
        base_channels=8,
        embed_dim=16,
        spectrum_height=256,
        epochs=2,
        batch_size=4,
        accum_steps=1,
        lr=1e-3,
        log_interval=1,
        visualization_epochs=[2],
        aug_enabled=False,
    )

    model = PickingModel(
        in_channels=1,
        base_channels=config.base_channels,
        embed_dim=config.embed_dim,
        spectrum_height=config.spectrum_height,
    )

    device = torch.device("cpu")
    checkpoint_dir = tmp_path / "checkpoints"
    run_dir = tmp_path / "run"

    trainer = PickingTrainer(
        model=model,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
    )
    trainer.train()

    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    lines = metrics_path.read_text().strip().split("\n")
    assert len(lines) == 2
    last = json.loads(lines[-1])
    assert "val_rmse_pixels" in last
    assert "val_presence_f1" in last

    assert (checkpoint_dir / "checkpoint_epoch_001.pt").exists()
    assert (checkpoint_dir / "checkpoint_epoch_002.pt").exists()


def test_trainer_resume(tiny_picking_loaders, tmp_path: Path) -> None:
    """Resuming from a checkpoint continues the run."""
    train_loader, val_loader = tiny_picking_loaders

    config = PickingConfig(
        base_channels=8,
        embed_dim=16,
        spectrum_height=256,
        epochs=1,
        batch_size=4,
        accum_steps=1,
        lr=1e-3,
        log_interval=1,
        visualization_epochs=[],
        aug_enabled=False,
    )

    model = PickingModel(
        in_channels=1,
        base_channels=config.base_channels,
        embed_dim=config.embed_dim,
        spectrum_height=config.spectrum_height,
    )

    device = torch.device("cpu")
    checkpoint_dir = tmp_path / "checkpoints"
    run_dir = tmp_path / "run"

    trainer = PickingTrainer(
        model=model,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
    )
    trainer.train()

    config2 = PickingConfig(
        base_channels=8,
        embed_dim=16,
        spectrum_height=256,
        epochs=2,
        batch_size=4,
        accum_steps=1,
        lr=1e-3,
        log_interval=1,
        visualization_epochs=[],
        aug_enabled=False,
    )
    model2 = PickingModel(
        in_channels=1,
        base_channels=config2.base_channels,
        embed_dim=config2.embed_dim,
        spectrum_height=config2.spectrum_height,
    )
    trainer2 = PickingTrainer(
        model=model2,
        config=config2,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        resume_from=checkpoint_dir / "checkpoint_epoch_001.pt",
    )
    trainer2.train()

    lines = (run_dir / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
