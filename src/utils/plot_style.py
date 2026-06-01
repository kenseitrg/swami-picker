from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

_DEFAULT_STYLE: dict[str, Any] = {
    "figure.dpi": 300,
    "figure.figsize": (6, 4),
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.dpi": 300,
    "savefig.format": "png",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "image.cmap": "viridis",
    "axes.grid": False,
}


def apply_style() -> None:
    """Apply the project's unified matplotlib style."""
    plt.rcParams.update(_DEFAULT_STYLE)
    logger.debug("Applied project plotting style")


def save_figure(
    fig: Figure,
    path: Path,
    **kwargs: Any,
) -> None:
    """Save a figure in a publication-ready format.

    Args:
        fig: Matplotlib figure to save.
        path: Output file path. Extension determines format
            (e.g. ``.png``, ``.pdf``, ``.svg``).
        **kwargs: Additional keyword arguments forwarded to
            ``Figure.savefig``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    default_kwargs: dict[str, Any] = {
        "dpi": plt.rcParams["savefig.dpi"],
        "bbox_inches": "tight",
    }
    default_kwargs.update(kwargs)
    fig.savefig(path, **default_kwargs)
    logger.info("Saved figure to %s", path)
