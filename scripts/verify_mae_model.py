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


def _check_patchify_roundtrip(model: MaskedAutoencoder) -> None:
    """Verify that patchify and unpatchify are exact inverses."""
    device = next(model.parameters()).device
    dummy = torch.randn(
        2, model.in_channels, model.img_size, model.img_size, device=device
    )
    patches = model.patchify(dummy)
    reconstructed = model.unpatchify(patches)
    assert torch.allclose(dummy, reconstructed, atol=1e-6), (
        "patchify / unpatchify round-trip failed"
    )
    logger.info("✅ patchify / unpatchify round-trip correct")


def _check_masking_correctness(model: MaskedAutoencoder) -> None:
    """Verify masking produces a consistent mask + ids_restore pair.

    For both random and block masking, the invariant
    ``mask + (ids_restore < N_keep) == 1`` must hold, meaning every
    position is either masked or kept (never both / neither).
    """
    device = next(model.parameters()).device
    B, N = 4, model.num_patches
    D = model.patch_embed.out_features
    x = torch.randn(B, N, D, device=device)

    for use_block in (False, True):
        model.use_block_masking = use_block
        if use_block:
            x_masked, mask, ids_restore = model.block_masking(x)
        else:
            x_masked, mask, ids_restore = model.random_masking(x)

        N_keep = x_masked.shape[1]
        kept_flags = (ids_restore < N_keep).float()
        invariant = mask + kept_flags
        assert torch.allclose(invariant, torch.ones_like(invariant)), (
            f"{'Block' if use_block else 'Random'} masking invariant violated: "
            f"mask + (ids_restore < N_keep) != 1"
        )
        logger.info(
            "✅ %s masking invariant holds (keep=%d/%d)",
            "Block" if use_block else "Random",
            N_keep,
            N,
        )

    # Restore original config
    model.use_block_masking = True


def _check_forward_loss(model: MaskedAutoencoder) -> None:
    """Verify forward_loss returns 0 when pred == target and > 0 otherwise."""
    device = next(model.parameters()).device
    imgs = torch.randn(
        2, model.in_channels, model.img_size, model.img_size, device=device
    )

    # Predict exactly the target patches
    target = model.patchify(imgs)
    mask = torch.ones(2, model.num_patches, device=device)
    loss_zero = model.forward_loss(imgs, target, mask)
    assert torch.isclose(loss_zero, torch.tensor(0.0, device=device), atol=1e-6), (
        f"forward_loss should be 0 when pred == target, got {loss_zero.item()}"
    )

    # Predict noise
    pred_noise = target + torch.randn_like(target) * 0.5
    loss_nonzero = model.forward_loss(imgs, pred_noise, mask)
    assert loss_nonzero.item() > 0, (
        f"forward_loss should be > 0 for noisy predictions, got {loss_nonzero.item()}"
    )

    logger.info(
        "✅ forward_loss correct: perfect reconstruction = %.6f, noisy = %.4f",
        loss_zero.item(),
        loss_nonzero.item(),
    )


def _check_forward_backward(model: MaskedAutoencoder, config: MNISTConfig) -> None:
    """Run a full forward/backward pass and check shapes."""
    device = next(model.parameters()).device
    dummy = torch.randn(
        config.batch_size,
        config.in_channels,
        config.image_size,
        config.image_size,
        device=device,
    )
    loss, pred, mask = model(dummy)

    expected_n_patches = (config.image_size // config.patch_size) ** 2
    assert tuple(pred.shape) == (
        config.batch_size,
        expected_n_patches,
        config.in_channels * config.patch_size * config.patch_size,
    )
    assert tuple(mask.shape) == (config.batch_size, expected_n_patches)

    # Mask ratio check (allow ±2% tolerance due to rounding)
    masked_ratio = mask.sum().item() / mask.numel()
    assert abs(masked_ratio - config.mask_ratio) < 0.02, (
        f"Masked ratio {masked_ratio:.2%} deviates from config {config.mask_ratio:.2%}"
    )

    logger.info("Loss: %.4f", loss.item())
    logger.info("Pred shape: %s", tuple(pred.shape))
    logger.info("Masked ratio: %.2f%%", masked_ratio * 100)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    logger.info("Grad norm: %.4f", grad_norm.item())

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        logger.info("Peak VRAM: %.2f GB", peak)


def main() -> None:
    """Comprehensive smoke-test of the MAE model."""
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
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2f M", n_params / 1e6)

    _check_patchify_roundtrip(model)
    _check_masking_correctness(model)
    _check_forward_loss(model)
    _check_forward_backward(model, config)

    logger.info("✅ All MAE model checks passed.")


if __name__ == "__main__":
    main()
