from __future__ import annotations

import argparse
import logging
import sys

import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mnist_dataset import create_mnist_dataloaders
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MAE on MNIST (Phase 0 smoke test)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase0_mnist.yaml"),
        help="Path to the YAML configuration file.",
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
        help="Run name slug (defaults to ISO-date + auto suffix).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for Phase 0 MAE training."""
    args = parse_args()
    config = MNISTConfig.from_yaml(args.config)
    set_seed(config.seed)

    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    logger = logging.getLogger(__name__)
    logger.info("Resolved config: %s", config.to_dict())

    run_name = (
        args.name or datetime.now().strftime("%Y-%m-%d") + "_phase0-mnist_mae-small"
    )
    run_dir = Path("experiments") / run_name
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(run_dir / "config.yaml")

    train_loader, val_loader = create_mnist_dataloaders(config)

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

    trainer.train()


if __name__ == "__main__":
    main()
