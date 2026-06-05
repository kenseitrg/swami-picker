"""Regenerate reconstruction and masking plots from a checkpoint using composite visuals.

Usage:
    python scripts/regenerate_reconstruction_plots.py \
        --checkpoint experiments/2026-06-04_phase0-mnist_mae-small/checkpoints/best_model.pt \
        --config configs/phase0_mnist.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mnist_dataset import create_mnist_dataloaders
from src.evaluation.visualize import (
    plot_masking_examples,
    plot_reconstruction_grid,
)
from src.models.mae import MaskedAutoencoder
from src.utils.checkpoint import load_checkpoint
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate reconstruction and masking plots from a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint .pt file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase0_mnist.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots. Defaults to checkpoint run directory / plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = MNISTConfig.from_yaml(args.config)
    set_seed(args.seed)

    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if args.output_dir is not None:
        plot_dir = args.output_dir
    else:
        plot_dir = args.checkpoint.parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _train_loader, val_loader = create_mnist_dataloaders(config)

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
    ).to(device)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(checkpoint["model"])

    sample_images = next(iter(val_loader))[0]

    plot_masking_examples(
        model,
        sample_images,
        device,
        plot_dir / "masking_examples_validated.png",
    )

    plot_reconstruction_grid(
        model,
        sample_images,
        device,
        plot_dir / "reconstruction_validated.png",
        seed=args.seed,
    )

    logger.info("Plots saved to: %s", plot_dir)


if __name__ == "__main__":
    main()
