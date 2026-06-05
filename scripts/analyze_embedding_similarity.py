"""Standalone script to analyze embedding cosine-similarity from a checkpoint.

Loads a trained MAE checkpoint, extracts encoder embeddings from the MNIST
validation set, and produces:
  * A cross-digit cosine-similarity heat-map.
  * A JSON sidecar with summary statistics.

Usage:
    python scripts/analyze_embedding_similarity.py \
        --checkpoint experiments/2026-06-04_phase0-mnist_mae-small/checkpoints/best_model.pt \
        --config configs/phase0_mnist.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mnist_dataset import create_mnist_dataloaders
from src.evaluation.visualize import plot_embedding_similarity_matrix
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
        description="Analyze embedding cosine similarity from a trained MAE checkpoint.",
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
        help="Path to YAML configuration (default: configs/phase0_mnist.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the plot and JSON. Defaults to the checkpoint's run directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum validation samples to process (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sample selection.",
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

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = args.checkpoint.parent.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    _train_loader, val_loader = create_mnist_dataloaders(config)
    logger.info("Validation set: %d samples", len(cast(Sized, val_loader.dataset)))

    # Model
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

    # Load checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(checkpoint["model"])
    logger.info("Checkpoint epoch: %d", checkpoint.get("epoch", -1))

    # Generate similarity matrix
    plot_path = output_dir / "embedding_similarity_matrix.png"
    sim_metrics = plot_embedding_similarity_matrix(
        model=model,
        loader=val_loader,
        device=device,
        save_path=plot_path,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    if sim_metrics is None:
        logger.error("Failed to compute similarity matrix.")
        sys.exit(1)

    # Write JSON sidecar
    json_path = output_dir / "embedding_similarity_matrix.json"
    with open(json_path, "w") as fh:
        json.dump(sim_metrics, fh, indent=2)

    logger.info("Results:")
    logger.info("  Mean intra-class similarity: %.4f", sim_metrics["mean_intra"])
    logger.info("  Mean inter-class similarity: %.4f", sim_metrics["mean_inter"])
    logger.info("  Contrast (intra/inter):      %.3f", sim_metrics["contrast"])
    logger.info("Plot saved to:   %s", plot_path)
    logger.info("JSON saved to:   %s", json_path)


if __name__ == "__main__":
    main()
