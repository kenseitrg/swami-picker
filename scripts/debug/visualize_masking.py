from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.data.mnist_dataset import create_mnist_dataloaders
from src.models.mae import MaskedAutoencoder
from src.utils.config import MNISTConfig
from src.utils.device import get_device
from src.utils.plot_style import apply_style, save_figure
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _apply_mask_to_image(
    imgs: torch.Tensor,
    mask: torch.Tensor,
    model: MaskedAutoencoder,
) -> torch.Tensor:
    """Black out masked patches in the original image for visualization.

    Args:
        imgs: Images of shape ``(B, C, H, W)``.
        mask: Binary mask of shape ``(B, N)`` where ``1`` = masked.
        model: MAE model (for patchify/unpatchify).

    Returns:
        Images with masked patches zeroed out, shape ``(B, C, H, W)``.
    """
    patches = model.patchify(imgs)  # (B, N, patch_dim)
    mask_expanded = (
        mask.unsqueeze(-1).expand_as(patches).to(patches.device)
    )  # (B, N, patch_dim)
    patches_masked = patches * (1 - mask_expanded)  # zero out masked patches
    imgs_masked = model.unpatchify(patches_masked)
    return imgs_masked


def main() -> None:
    """Visualize masking on a real MNIST sample."""
    apply_style()
    config = MNISTConfig.from_yaml(Path("configs/phase0_mnist.yaml"))
    set_seed(config.seed)
    device = get_device()

    # Load a single sample
    train_loader, _ = create_mnist_dataloaders(config)
    images, labels = next(iter(train_loader))
    sample = images[:1].to(device)  # (1, 1, 256, 256)

    # Build model for patchify/unpatchify and masking
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
        use_block_masking=True,
        block_size=config.block_size,
    ).to(device)
    model.eval()

    # ---- Get mask via encoder ----------------------------------------
    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(sample)
    mask = mask.cpu()  # (1, N)

    # ---- Create visualisations ---------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # 1. Original image
    ax = axes[0]
    ax.imshow(sample[0, 0].cpu().numpy(), cmap="gray")
    ax.set_title(f"Original (label={labels[0].item()})")
    ax.axis("off")

    # 2. Masked image (visible patches only)
    masked_img = _apply_mask_to_image(sample, mask, model)
    ax = axes[1]
    ax.imshow(masked_img[0, 0].cpu().numpy(), cmap="gray")
    ax.set_title(f"Masked ({config.mask_ratio:.0%} blacked out)")
    ax.axis("off")

    # 3. Mask overlay (1 = masked, shown as red overlay)
    ax = axes[2]
    img_np = sample[0, 0].cpu().numpy()
    ax.imshow(img_np, cmap="gray")
    # Create mask overlay: reshape mask to spatial grid
    p = config.patch_size
    n = config.image_size // p
    mask_grid = mask[0].reshape(n, n).cpu().numpy()
    # Upsample mask to full resolution for overlay
    mask_overlay = np.kron(mask_grid, np.ones((p, p)))
    ax.imshow(mask_overlay, cmap="Reds", alpha=0.4)
    ax.set_title("Mask Overlay (red = masked)")
    ax.axis("off")

    # 4. Visible patches only (unpatchified from kept patches)
    patches = model.patchify(sample)
    N_keep = latent.shape[1]
    ids_shuffle = torch.argsort(ids_restore[0])
    ids_keep = ids_shuffle[:N_keep]
    kept_patches = torch.zeros_like(patches)
    kept_patches[0, ids_keep] = patches[0, ids_keep]
    visible_only = model.unpatchify(kept_patches)
    ax = axes[3]
    ax.imshow(visible_only[0, 0].cpu().numpy(), cmap="gray")
    ax.set_title(f"Visible Patches Only ({N_keep}/{model.num_patches})")
    ax.axis("off")

    fig.suptitle(
        f"Block Masking: block_size={config.block_size}×{config.block_size}, "
        f"mask_ratio={config.mask_ratio}",
        fontsize=11,
        y=1.02,
    )

    out_dir = Path("experiments/2026-06-01_phase0-mnist_mae-visualization")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / "masking_visualization.png")
    plt.close(fig)

    logger.info("Saved masking visualization to %s", out_dir)
    logger.info(
        "Mask stats: %d/%d patches kept (%.1f%% visible)",
        N_keep,
        model.num_patches,
        N_keep / model.num_patches * 100,
    )

    # Also test random masking for comparison
    model.use_block_masking = False
    with torch.no_grad():
        _, mask_rand, ids_restore_rand = model.forward_encoder(sample)
    mask_rand = mask_rand.cpu()

    fig2, axes2 = plt.subplots(1, 4, figsize=(14, 3.5))

    axes2[0].imshow(sample[0, 0].cpu().numpy(), cmap="gray")
    axes2[0].set_title(f"Original (label={labels[0].item()})")
    axes2[0].axis("off")

    masked_img_rand = _apply_mask_to_image(sample, mask_rand, model)
    axes2[1].imshow(masked_img_rand[0, 0].cpu().numpy(), cmap="gray")
    axes2[1].set_title(f"Masked ({config.mask_ratio:.0%} blacked out)")
    axes2[1].axis("off")

    axes2[2].imshow(img_np, cmap="gray")
    mask_grid_rand = mask_rand[0].reshape(n, n).cpu().numpy()
    mask_overlay_rand = np.kron(mask_grid_rand, np.ones((p, p)))
    axes2[2].imshow(mask_overlay_rand, cmap="Reds", alpha=0.4)
    axes2[2].set_title("Mask Overlay (red = masked)")
    axes2[2].axis("off")

    N_keep_rand = int(model.num_patches * (1 - config.mask_ratio))
    ids_shuffle_rand = torch.argsort(ids_restore_rand[0])
    ids_keep_rand = ids_shuffle_rand[:N_keep_rand]
    kept_patches_rand = torch.zeros_like(patches)
    kept_patches_rand[0, ids_keep_rand] = patches[0, ids_keep_rand]
    visible_only_rand = model.unpatchify(kept_patches_rand)
    axes2[3].imshow(visible_only_rand[0, 0].cpu().numpy(), cmap="gray")
    axes2[3].set_title(f"Visible Patches Only ({N_keep_rand}/{model.num_patches})")
    axes2[3].axis("off")

    fig2.suptitle(
        f"Random Masking: mask_ratio={config.mask_ratio}",
        fontsize=11,
        y=1.02,
    )

    save_figure(fig2, out_dir / "masking_visualization_random.png")
    plt.close(fig2)

    logger.info("Saved random masking visualization to %s", out_dir)
    logger.info("✅ Masking visualization complete.")


if __name__ == "__main__":
    main()
