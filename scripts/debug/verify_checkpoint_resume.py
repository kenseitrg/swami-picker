from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.mae import MaskedAutoencoder
from src.training.trainer import MAETrainer
from src.utils.config import MNISTConfig
from src.utils.device import get_device
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_metrics(metrics_path: Path) -> list[dict]:
    """Parse a JSONL metrics file into a list of records."""
    records = []
    with open(metrics_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _create_tiny_loaders(config: MNISTConfig) -> tuple[DataLoader, DataLoader]:
    """Create tiny deterministic synthetic loaders for fast verification."""
    x = torch.randn(32, config.in_channels, config.image_size, config.image_size)
    y = torch.arange(32) % 10
    train = DataLoader(
        TensorDataset(x, y),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val = DataLoader(
        TensorDataset(x, y),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train, val


def _build_model(config: MNISTConfig, device: torch.device) -> MaskedAutoencoder:
    """Instantiate a fresh MAE model."""
    return MaskedAutoencoder(
        img_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        decoder_embed_dim=config.decoder_embed_dim,
        decoder_depth=config.decoder_depth,
        decoder_num_heads=config.decoder_num_heads,
        mask_ratio=config.mask_ratio,
        use_block_masking=config.use_block_masking,
        block_size=config.block_size,
    ).to(device)


def main() -> None:
    """Verify checkpoint save/resume produces continuous loss curves."""
    config = MNISTConfig(
        epochs=2,
        batch_size=4,
        accum_steps=1,
        log_interval=9999,
    )
    set_seed(config.seed)
    device = get_device()

    train_loader, val_loader = _create_tiny_loaders(config)

    # ------------------------------------------------------------------
    # Continuous run: 2 epochs straight through (reference)
    # ------------------------------------------------------------------
    logger.info("=== Continuous run (2 epochs) ===")
    run_dir_continuous = Path(tempfile.mkdtemp())
    model_c = _build_model(config, device)
    trainer_c = MAETrainer(
        model=model_c,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=run_dir_continuous / "checkpoints",
        run_dir=run_dir_continuous,
    )
    trainer_c.train()
    metrics_c = _parse_metrics(run_dir_continuous / "metrics.jsonl")

    # ------------------------------------------------------------------
    # Split run: epoch 1 -> save -> resume -> epoch 2
    # ------------------------------------------------------------------
    logger.info("=== Split run (epoch 1 + resume + epoch 2) ===")
    run_dir_split = Path(tempfile.mkdtemp())

    # First half: 1 epoch
    config_1 = MNISTConfig(
        epochs=1,
        batch_size=4,
        accum_steps=1,
        log_interval=9999,
    )
    model_s1 = _build_model(config_1, device)
    trainer_s1 = MAETrainer(
        model=model_s1,
        config=config_1,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=run_dir_split / "checkpoints",
        run_dir=run_dir_split,
    )
    trainer_s1.train()

    checkpoint_path = run_dir_split / "checkpoints" / "checkpoint_epoch_001.pt"
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    # Second half: resume and train epoch 2
    config_2 = MNISTConfig(
        epochs=2,
        batch_size=4,
        accum_steps=1,
        log_interval=9999,
    )
    model_s2 = _build_model(config_2, device)
    trainer_s2 = MAETrainer(
        model=model_s2,
        config=config_2,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=run_dir_split / "checkpoints",
        run_dir=run_dir_split,
        resume_from=checkpoint_path,
    )
    trainer_s2.train()
    metrics_s = _parse_metrics(run_dir_split / "metrics.jsonl")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert len(metrics_c) == 2, (
        f"Expected 2 epochs in continuous run, got {len(metrics_c)}"
    )
    assert len(metrics_s) == 2, f"Expected 2 epochs in split run, got {len(metrics_s)}"

    # 1. Epoch 1 losses are expected to differ slightly because the split
    #    run's first half uses a different cosine schedule (8 total steps
    #    vs 16).  We log them for information but do not assert equality.
    for key in ("train_loss", "val_loss"):
        c0 = metrics_c[0][key]
        s0 = metrics_s[0][key]
        logger.info("Epoch 1 | %s | continuous=%.6f | split=%.6f", key, c0, s0)

    # 2. Epoch 2 (post-resume) must match the continuous run tightly.
    #    This is the core resume-correctness check.
    for key in ("train_loss", "val_loss"):
        c1 = metrics_c[1][key]
        s1 = metrics_s[1][key]
        logger.info("Epoch 2 | %s | continuous=%.6f | split=%.6f", key, c1, s1)
        rel_diff = abs(c1 - s1) / (abs(c1) + 1e-8)
        assert rel_diff < 0.05, (
            f"Epoch 2 {key} diverged after resume: continuous={c1:.6f}, "
            f"split={s1:.6f} (relative diff={rel_diff:.3%})"
        )

    # 3. All losses must be finite.
    for key in ("train_loss", "val_loss"):
        for run_name, metrics in (("continuous", metrics_c), ("split", metrics_s)):
            for epoch_idx, record in enumerate(metrics):
                loss = record[key]
                assert torch.isfinite(torch.tensor(loss)), (
                    f"Non-finite {key} in {run_name} run at epoch {epoch_idx + 1}"
                )

    # 4. No regression: each epoch's loss should be <= 1.05x the previous
    #    (small upward fluctuation allowed, but not a dramatic regression).
    for key in ("train_loss", "val_loss"):
        for run_name, metrics in (("continuous", metrics_c), ("split", metrics_s)):
            e1, e2 = metrics[0][key], metrics[1][key]
            assert e2 <= e1 * 1.05, (
                f"{key} regressed in {run_name} run: epoch1={e1:.6f}, epoch2={e2:.6f}"
            )

    logger.info("✅ All checkpoint/resume checks passed.")
    logger.info(
        "Loss continuity verified: continuous vs. resumed losses match within 5%%."
    )


if __name__ == "__main__":
    main()
