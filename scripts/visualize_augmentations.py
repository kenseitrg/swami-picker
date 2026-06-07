from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.augmentations import FKSpectrumTransform
from src.data.fk_dataset import FKDataset
from src.utils.plot_style import apply_style, save_figure
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
        description="Visualise FK spectrum augmentations (before/after).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.json"),
        help="Path to manifest.json.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of spectra to visualise.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Gaussian noise std.",
    )
    parser.add_argument(
        "--intensity-jitter",
        type=float,
        default=0.30,
        help="Intensity jitter factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to experiments/YYYY-MM-DD_phase2-augmentation-audit/).",
    )
    return parser.parse_args()


def main() -> None:
    """Generate before/after augmentation panels."""
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir or (
        Path("experiments")
        / (datetime.now().strftime("%Y-%m-%d") + "_phase2-augmentation-audit")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = FKDataset(
        manifest_path=args.manifest,
        split="train",
        transform=None,
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)

    transform = FKSpectrumTransform(
        noise_std=args.noise_std,
        intensity_jitter=args.intensity_jitter,
    )

    apply_style()
    fig, axes = plt.subplots(len(indices), 2, figsize=(6, 3 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        tensor, metadata = ds[idx]
        original = tensor[0].numpy()
        augmented = transform(tensor.clone())[0].numpy()

        # Physical axes from metadata
        freq_axis = metadata.get("freq_axis_resized", [])
        waven_axis = metadata.get("waven_axis_resized", [])
        line_num = metadata.get("line_number", "?")
        point_num = metadata.get("point_number", "?")

        extent = None
        if len(freq_axis) >= 2 and len(waven_axis) >= 2:
            extent = [
                freq_axis[0],
                freq_axis[-1],
                waven_axis[-1],
                waven_axis[0],
            ]

        vmin = min(original.min(), augmented.min())
        vmax = max(original.max(), augmented.max())

        for col, data in enumerate([original, augmented]):
            ax = axes[row, col]
            im = ax.imshow(
                data,
                cmap="viridis",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Wavenumber (1/m)")
            if row == 0:
                title = "Original" if col == 0 else "Augmented"
                ax.set_title(title)
            ax.text(
                0.02,
                0.98,
                f"RL{line_num} P{point_num}",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = output_dir / "augmentation_before_after.png"
    save_figure(fig, save_path)
    plt.close(fig)
    logger.info("Saved augmentation visualisation to %s", save_path)


if __name__ == "__main__":
    main()
