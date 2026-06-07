from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.patheffects as path_effects
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


def _patches_to_grid(
    patches: torch.Tensor,
    n_patches_per_side: int,
    patch_size: int,
    in_channels: int,
) -> torch.Tensor:
    """Reshape flat patches into a spatial grid image.

    Args:
        patches: Patches of shape ``(N, patch_dim)`` or ``(1, N, patch_dim)``.
        n_patches_per_side: Number of patches along each spatial axis.
        patch_size: Size of each patch in pixels.
        in_channels: Number of channels (1 for grayscale).

    Returns:
        Image tensor of shape ``(C, H, W)``.
    """
    if patches.dim() == 3:
        patches = patches[0]  # (N, patch_dim)
    N = n_patches_per_side
    p = patch_size
    C = in_channels
    # (N, C*p*p) -> (N, C, p, p)
    patches = patches.reshape(N * N, C, p, p)
    # (N, C, p, p) -> (N_patches_side, N_patches_side, C, p, p)
    grid = patches.reshape(N, N, C, p, p)
    # (N, N, C, p, p) -> (C, N*p, N*p)
    grid = grid.permute(2, 0, 3, 1, 4).reshape(C, N * p, N * p)
    return grid


def main() -> None:
    """Visualize the patchify and unpatchify transformation."""
    apply_style()
    config = MNISTConfig.from_yaml(Path("configs/phase0_mnist.yaml"))
    set_seed(config.seed)
    device = get_device()

    # Load a single sample
    train_loader, _ = create_mnist_dataloaders(config)
    images, labels = next(iter(train_loader))
    sample = images[:1].to(device)  # (1, 1, 256, 256)

    # Use MAE for patchify/unpatchify
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
    ).to(device)
    model.eval()

    with torch.no_grad():
        patches = model.patchify(sample)  # (1, 256, 256)
        reconstructed = model.unpatchify(patches)  # (1, 1, 256, 256)

    # Round-trip check
    assert torch.allclose(sample, reconstructed, atol=1e-6), (
        "patchify / unpatchify round-trip failed"
    )

    n = config.image_size // config.patch_size

    # ---- Visualise ---------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Original image
    ax = axes[0]
    img_np = sample[0, 0].cpu().numpy()
    ax.imshow(img_np, cmap="gray")
    ax.set_title(f"Original (label={labels[0].item()})")
    ax.axis("off")

    # 2. Patches as a grid (with red boundaries and patch numbers)
    ax = axes[1]
    patch_grid = _patches_to_grid(patches, n, config.patch_size, config.in_channels)
    ax.imshow(patch_grid[0].cpu().numpy(), cmap="gray")
    ax.set_title(f"Patches Grid ({n}×{n} = {n * n} patches)")

    p = config.patch_size
    img_h = config.image_size
    # Red grid lines
    for i in range(n + 1):
        ax.axhline(i * p - 0.5, color="red", linewidth=0.8, alpha=0.6)
        ax.axvline(i * p - 0.5, color="red", linewidth=0.8, alpha=0.6)

    # Patch numbers (white with black outline for readability)
    for row in range(n):
        for col in range(n):
            patch_idx = row * n + col
            x = col * p + p / 2 - 0.5
            y = row * p + p / 2 - 0.5
            ax.text(
                x,
                y,
                str(patch_idx),
                ha="center",
                va="center",
                fontsize=4,
                color="white",
                path_effects=[
                    path_effects.withStroke(linewidth=1.5, foreground="black")
                ],
            )
    ax.set_xlim(-0.5, img_h - 0.5)
    ax.set_ylim(img_h - 0.5, -0.5)
    ax.axis("off")

    # 3. Reconstructed image
    ax = axes[2]
    recon_np = reconstructed[0, 0].cpu().numpy()
    ax.imshow(recon_np, cmap="gray")
    ax.set_title("Unpatchified (Reconstructed)")
    ax.axis("off")

    # 4. Difference (should be all zero)
    ax = axes[3]
    diff = (sample - reconstructed)[0, 0].cpu().numpy()
    vmax = max(abs(diff.min()), abs(diff.max()))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Difference (max |Δ|={vmax:.2e})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Patchify / Unpatchify: {config.image_size}×{config.image_size} image → "
        f"{n * n} patches of {config.patch_size}×{config.patch_size}",
        fontsize=11,
        y=1.02,
    )

    out_dir = Path("experiments/2026-06-01_phase0-mnist_mae-visualization")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / "patchify_visualization.png")
    plt.close(fig)

    logger.info("Saved patchify visualization to %s", out_dir)
    logger.info(
        "Round-trip error: max |Δ| = %.2e (should be ~0)",
        (sample - reconstructed).abs().max().item(),
    )
    logger.info("✅ Patchify visualization complete.")


if __name__ == "__main__":
    main()
