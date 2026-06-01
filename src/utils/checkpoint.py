from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: dict[str, Any],
    path: Path,
    *,
    is_best: bool = False,
) -> None:
    """Save a training checkpoint to disk.

    The *state* dictionary should contain at minimum:
    ``model``, ``optimizer``, ``scaler``, ``scheduler``, ``epoch``,
    ``step``, ``seed``, ``config``, and ``metrics``.

    Args:
        state: Checkpoint dictionary.
        path: Destination file path.
        is_best: If ``True``, also write a copy named ``best_model.pt``
            in the same directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Saved checkpoint to %s", path)
    if is_best:
        best_path = path.parent / "best_model.pt"
        torch.save(state, best_path)
        logger.info("Saved best checkpoint to %s", best_path)


def load_checkpoint(
    path: Path,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint from disk.

    Args:
        path: Path to the checkpoint file.
        device: Device to map tensors to. Defaults to CPU.

    Returns:
        Checkpoint state dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    logger.info(
        "Loaded checkpoint from %s (epoch %d, step %d)",
        path,
        checkpoint.get("epoch", -1),
        checkpoint.get("step", -1),
    )
    return checkpoint
