"""Phase 2 — VICReg self-supervised pretraining on FK spectra.

Entry point for training a VICReg model after MAE failed due to
embedding collapse on homogeneous FK data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.augmentations import FKSpectrumTransform
from src.data.fk_dataset import FKDataset
from src.data.split import create_train_val_entries
from src.models.vicreg import VICReg
from src.training.vicreg_trainer import VICRegTrainer
from src.utils.config import VICRegConfig
from src.utils.device import get_device
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VICReg self-supervised pretraining on FK spectra.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_vicreg.yaml"),
        help="Path to VICReg config YAML.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume from.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (defaults to YYYY-MM-DD_phase2-vicreg).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 epoch with tiny batch for smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = VICRegConfig.from_yaml(args.config)

    if args.epochs is not None:
        config.epochs = args.epochs

    if args.dry_run:
        config.epochs = 1
        config.batch_size = 4
        config.accum_steps = 1

    set_seed(config.seed)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------
    # Experiment directory
    # ------------------------------------------------------------------
    run_name = args.name or (
        datetime.now().strftime("%Y-%m-%d") + "_phase2-vicreg"
    )
    run_dir = Path("experiments") / run_name
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config.save_yaml(run_dir / "config.yaml")
    logger.info("Run directory: %s", run_dir)
    logger.info("Config:        %s", args.config)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_entries, val_entries = create_train_val_entries(
        config.manifest_path,
        config.val_fraction,
        config.val_seed,
    )

    # Training dataset — augmentation applied inside trainer
    train_ds = FKDataset(
        manifest_path=Path(config.manifest_path),
        split="train",
        transform=None,  # augmentation done in trainer for dual views
        entries=train_entries,
    )
    val_ds = FKDataset(
        manifest_path=Path(config.manifest_path),
        split="val",
        transform=None,
        entries=val_entries,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,  # ensure consistent batch size for variance computation
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    logger.info(
        "Data: %d train / %d val samples, %d batches/epoch",
        len(train_ds),
        len(val_ds),
        len(train_loader),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = VICReg(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        projector_hidden_dim=config.projector_hidden_dim,
        projector_out_dim=config.projector_out_dim,
    )
    logger.info("Architecture: VICReg (ViT-Small encoder + projector)")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %.2fM params (%.2fM trainable)",
        n_params / 1e6,
        n_trainable / 1e6,
    )

    # ------------------------------------------------------------------
    # Trainer & training
    # ------------------------------------------------------------------
    trainer = VICRegTrainer(
        model=model,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        resume_from=args.resume,
        argv=sys.argv,
    )
    trainer.train()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("VICReg pretraining complete.")
    logger.info("Output: %s", run_dir)
    logger.info("Checkpoints: %s", checkpoint_dir)
    logger.info("Best val metric: %.6f", trainer.best_val_metric)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
