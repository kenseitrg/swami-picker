"""CLI entry point for Phase 4 supervised dispersion-curve picking training."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.picking_augmentations import PickSyncTransform
from src.data.picking_collate import picking_collate
from src.data.picking_dataset import FKPickingDataset
from src.models.picking_model import build_picking_model
from src.training.picking_trainer import PickingTrainer
from src.utils.config import PickingConfig
from src.utils.device import get_device
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure root logger for CLI output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """Train the Phase 4 supervised picking model."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Train a supervised model to pick dispersion curves from FK spectra."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase4_picking.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment run name (defaults to auto-generated slug).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 epoch on a tiny subset for smoke testing.",
    )
    args = parser.parse_args(argv)

    config = PickingConfig.from_yaml(Path(args.config))

    if args.dry_run:
        config.epochs = 1
        config.batch_size = min(config.batch_size, 4)
        config.num_workers = 0
        logger.info("Dry-run mode: 1 epoch, batch_size=%d", config.batch_size)

    set_seed(config.seed)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_name = args.name or f"{datetime.now().strftime('%Y-%m-%d')}_phase4-picking"
    run_dir = Path("experiments") / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(run_dir / "config.yaml")

    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)

    transform = PickSyncTransform(
        enabled=config.aug_enabled and not args.dry_run,
        noise_std=config.aug_noise_std,
        intensity_jitter=config.aug_intensity_jitter,
        freq_shift_max=config.aug_freq_shift_max,
        waven_shift_max=config.aug_waven_shift_max,
    )

    train_ds = FKPickingDataset(
        npz_path=Path(config.training_data_path),
        split="train",
        val_fraction=config.val_fraction,
        val_seed=config.val_seed,
        min_direct_picks=config.min_direct_picks,
        transform=transform,
        k_folds=config.k_folds,
        fold_index=config.fold_index,
    )
    val_ds = FKPickingDataset(
        npz_path=Path(config.training_data_path),
        split="val",
        val_fraction=config.val_fraction,
        val_seed=config.val_seed,
        min_direct_picks=config.min_direct_picks,
        transform=None,
        k_folds=config.k_folds,
        fold_index=config.fold_index,
    )

    if args.dry_run:
        subset_size = min(32, len(train_ds))
        train_indices = list(range(subset_size))
        val_indices = list(range(min(8, len(val_ds))))
        train_ds = torch.utils.data.Subset(train_ds, train_indices)
        val_ds = torch.utils.data.Subset(val_ds, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=picking_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=picking_collate,
    )

    logger.info(
        "DataLoaders: train=%d, val=%d",
        len(train_ds),
        len(val_ds),
    )

    model = build_picking_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model parameters=%.2fM",
        n_params / 1e6,
    )

    trainer = PickingTrainer(
        model=model,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        resume_from=Path(args.resume) if args.resume else None,
        argv=sys.argv,
    )
    trainer.train()

    logger.info("Training complete. Best smoothed val_rmse=%.4f", trainer.best_val_metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
