from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sized
from datetime import datetime
from pathlib import Path
from typing import cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mnist_dataset import create_mnist_dataloaders
from src.models.cvt_mae import CvTMaskedAutoencoder
from src.models.mae import MaskedAutoencoder
from src.training.trainer import MAETrainer
from src.utils.config import MNISTConfig
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
# Phase 0 — MNIST Smoke Test
#
# Validates the complete MAE training stack on MNIST before
# introducing real FK spectral data.
#
# Output layout (PROJECT_RULES.md §3.4):
#   experiments/
#   └── YYYY-MM-DD_phase0-mnist_mae-small/
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
        description="Phase 0: Train MAE on MNIST (smoke test).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/train_mae.py\n"
            "  python scripts/train_mae.py --epochs 1 --dry-run\n"
            "  python scripts/train_mae.py --resume experiments/.../checkpoints/best_model.pt\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase0_mnist.yaml"),
        help="Path to YAML configuration (default: configs/phase0_mnist.yaml).",
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
            "Defaults to ISO-date + '_phase0-mnist_mae-small'."
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
    """Entry point for Phase 0 MAE training."""
    args = parse_args()
    config = MNISTConfig.from_yaml(args.config)

    # Override epochs if provided
    if args.epochs is not None:
        config.epochs = args.epochs

    # Dry-run: 1 epoch, single batch step
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
    run_name = args.name or (
        datetime.now().strftime("%Y-%m-%d") + "_phase0-mnist_mae-small"
    )
    run_dir = Path("experiments") / run_name
    checkpoint_dir = run_dir / "checkpoints"
    plot_dir = run_dir / "plots"

    run_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(run_dir / "config.yaml")

    logger.info("Run directory: %s", run_dir.resolve())
    logger.info("Config:        %s", args.config.resolve())

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader = create_mnist_dataloaders(config)
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
    if config.use_cvt:
        model = CvTMaskedAutoencoder(
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
            cvt_kernel_size=config.cvt_kernel_size,
            use_pos_embed=config.use_pos_embed,
        )
        logger.info(
            "Architecture: CvT-MAE (kernel_size=%d, pos_embed=%s)",
            config.cvt_kernel_size,
            config.use_pos_embed,
        )
    else:
        model = MaskedAutoencoder(
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
        )
        logger.info("Architecture: ViT-MAE")

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
    trainer = MAETrainer(
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

    # Dry-run: override the validation loader to yield just one batch
    # so the pipeline exercises end-to-end without waiting.
    if args.dry_run:
        # Replace both loaders with tiny subsets for fast verification
        dry_images, dry_labels = next(iter(train_loader))
        dry_ds = torch.utils.data.TensorDataset(dry_images[:4], dry_labels[:4])
        dry_loader = torch.utils.data.DataLoader(dry_ds, batch_size=2, shuffle=False)
        trainer.train_loader = dry_loader
        trainer.val_loader = dry_loader
        logger.info("DRY RUN — 2 batches per epoch, stops after 1 epoch.")

    t_start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t_start

    logger.info("Total training time: %s", _format_duration(elapsed))
    logger.info("Output: %s", run_dir.resolve())
    logger.info("Checkpoints: %s", checkpoint_dir.resolve())
    logger.info("Plots: %s", plot_dir.resolve())
    logger.info("Phase 0 MNIST smoke test complete.")


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
