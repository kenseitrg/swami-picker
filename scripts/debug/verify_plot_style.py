from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from matplotlib import pyplot as plt

from src.utils.plot_style import apply_style, save_figure

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Render mock figures to visually verify the project plotting style."""
    apply_style()
    out_dir = Path("/tmp/plot_style_verify")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # ── Figure 1: line plot ──────────────────────────────────────────────
    fig1, ax1 = plt.subplots()
    x = np.linspace(0, 10, 100)
    for i in range(3):
        y = np.sin(x + i) + rng.normal(0, 0.1, size=x.shape)
        ax1.plot(x, y, label=f"Series {i + 1}")
    ax1.set_title("Mock Line Plot")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.legend()
    save_figure(fig1, out_dir / "line_plot.png")
    plt.close(fig1)

    # ── Figure 2: bar plot ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots()
    categories = ["A", "B", "C", "D", "E"]
    values = rng.integers(10, 100, size=len(categories))
    bars = ax2.bar(categories, values, color="steelblue")
    ax2.set_title("Mock Bar Plot")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Value")
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_figure(fig2, out_dir / "bar_plot.png")
    plt.close(fig2)

    # ── Figure 3: imshow ─────────────────────────────────────────────────
    fig3, ax3 = plt.subplots()
    img = rng.random((64, 64))
    im = ax3.imshow(img, aspect="auto")
    ax3.set_title("Mock Image (imshow)")
    ax3.set_xlabel("Column")
    ax3.set_ylabel("Row")
    fig3.colorbar(im, ax=ax3, label="Intensity")
    save_figure(fig3, out_dir / "image_plot.png")
    plt.close(fig3)

    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
