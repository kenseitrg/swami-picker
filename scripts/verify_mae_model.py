from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.mae import MaskedAutoencoder
from src.utils.config import MNISTConfig
from src.utils.device import get_device
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Smoke-test the MAE model with dummy and real MNIST data."""
    config = MNISTConfig.from_yaml(Path("configs/phase0_mnist.yaml"))
    set_seed(config.seed)
    device = get_device()

    logger.info("Building MAE model …")
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

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2f M", n_params / 1e6)

    # ---- Dummy forward pass ------------------------------------------
    dummy = torch.randn(
        config.batch_size,
        config.in_channels,
        config.image_size,
        config.image_size,
        device=device,
    )
    loss, pred, mask = model(dummy)

    logger.info("Loss: %.4f", loss.item())
    logger.info("Pred shape: %s", tuple(pred.shape))
    logger.info("Mask shape: %s", tuple(mask.shape))
    logger.info("Masked ratio: %.2f%%", mask.sum().item() / mask.numel() * 100)

    expected_n_patches = (config.image_size // config.patch_size) ** 2
    assert tuple(pred.shape) == (
        config.batch_size,
        expected_n_patches,
        config.in_channels * config.patch_size * config.patch_size,
    )
    assert tuple(mask.shape) == (config.batch_size, expected_n_patches)

    # ---- Backward pass -----------------------------------------------
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    logger.info("Grad norm: %.4f", grad_norm.item())

    # ---- VRAM check --------------------------------------------------
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        logger.info("Peak VRAM: %.2f GB", peak)

    logger.info("✅ MAE model verification passed.")


if __name__ == "__main__":
    main()
