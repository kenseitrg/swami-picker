from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sized
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.augmentations import FKSpectrumTransform
from src.data.fk_dataset import FKDataset
from src.data.split import create_train_val_entries
from src.models.mae import MaskedAutoencoder
from src.training.fk_trainer import FKMAETrainer
from src.utils.config import FKMAEConfig
from src.utils.device import get_device
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Phase 2 — MAE Pretraining on FK Spectra
#
# Unsupervised masked autoencoder pretraining on real FK spectral data.
#
# Output layout (PROJECT_RULES.md §3.4):
#   experiments/
#   └── YYYY-MM-DD_phase2-fk-mae/
#       ├── config.yaml                  # resolved config snapshot
#       ├── metrics.jsonl                # line-delimited per-epoch metrics
#       ├── checkpoints/                 # model checkpoints
#       │   ├── checkpoint_epoch_NNN.pt
#       │   └── best_model.pt
#       └── plots/                       # publication-ready figures
#           ├── masking_examples.png
#           ├── reconstruction_epoch_NNN.png
#           ├── umap_epoch_NNN.png
#           └── loss_curves.png
# ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Train MAE on FK spectra.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/train_fk_mae.py\n"
            "  python scripts/train_fk_mae.py --epochs 1 --dry-run\n"
            "  python scripts/train_fk_mae.py --resume experiments/.../checkpoints/best_model.pt\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase2_fk_mae.yaml"),
        help="Path to YAML configuration (default: configs/phase2_fk_mae.yaml).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Run name slug used as the experiment subdirectory. "
            "Defaults to ISO-date + '_phase2-fk-mae'."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (useful for dry-runs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run for 2 batches then exit (quick shape/setup verification).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for Phase 2 MAE training on FK spectra."""
    args = parse_args()
    config = FKMAEConfig.from_yaml(args.config)

    if args.epochs is not None:
        config.epochs = args.epochs

    if args.dry_run:
        config.epochs = 1
        config.batch_size = 2
        config.accum_steps = 1

    set_seed(config.seed)

    # ------------------------------------------------------------------
    # Device & optimisations
    # ------------------------------------------------------------------
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------
    # Experiment directory
    # ------------------------------------------------------------------
    run_name = args.name or (datetime.now().strftime("%Y-%m-%d") + "_phase2-fk-mae")
    run_dir = Path("experiments") / run_name
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(run_dir / "config.yaml")

    logger.info("Run directory: %s", run_dir.resolve())
    logger.info("Config:        %s", args.config.resolve())

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_entries, val_entries = create_train_val_entries(
        config.manifest_path,
        config.val_fraction,
        config.val_seed,
    )

    train_transform = FKSpectrumTransform(
        noise_std=config.noise_std,
        intensity_jitter=config.intensity_jitter,
        freq_shift_max=config.freq_shift_max,
        waven_shift_max=config.waven_shift_max,
        freq_dropout_prob=config.freq_dropout_prob,
        freq_dropout_width=config.freq_dropout_width,
    )

    train_ds = FKDataset(
        manifest_path=Path(config.manifest_path),
        split="train",
        transform=train_transform,
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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    train_size = len(cast(Sized, train_loader.dataset))
    val_size = len(cast(Sized, val_loader.dataset))
    logger.info(
        "Data: %d train / %d val samples, %d batches/epoch (accum=%d, eff. batch=%d)",
        train_size,
        val_size,
        len(train_loader),
        config.accum_steps,
        config.batch_size * config.accum_steps,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MaskedAutoencoder(
        img_size=config.img_size,
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
    )
    logger.info("Architecture: ViT-MAE (FK)")

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
    trainer = FKMAETrainer(
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

    if args.dry_run:
        dry_images, _ = next(iter(train_loader))
        dry_images = dry_images[:4]
        # Create a simple dataset that yields (tensor, metadata_dict) tuples
        # so the base trainer's unpack logic works unchanged.
        dry_metadata = [{"line_number": 5007}] * len(dry_images)
        dry_ds = torch.utils.data.TensorDataset(
            dry_images, torch.zeros(len(dry_images))
        )

        # Wrap with a simple dataset that returns proper tuples
        class _DryDataset(torch.utils.data.Dataset):
            def __init__(self, images, metadata):
                self.images = images
                self.metadata = metadata

            def __len__(self):
                return len(self.images)

            def __getitem__(self, index):
                return self.images[index], self.metadata[index]

        dry_ds = _DryDataset(dry_images, dry_metadata)
        dry_loader = DataLoader(dry_ds, batch_size=2, shuffle=False)
        trainer.train_loader = dry_loader
        trainer.val_loader = dry_loader
        logger.info("DRY RUN — 2 batches per epoch, stops after 1 epoch.")

    t_start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t_start

    logger.info("Total training time: %s", _format_duration(elapsed))
    logger.info("Output: %s", run_dir.resolve())
    logger.info("Checkpoints: %s", checkpoint_dir.resolve())
    logger.info("Phase 2 FK MAE pretraining complete.")


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to ``H:MM:SS``."""
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


if __name__ == "__main__":
    main()
