from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
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
    state["rng_state"] = _get_rng_state(state.get("rng_state"))
    torch.save(state, path)
    logger.info("Saved checkpoint to %s", path)
    if is_best:
        best_path = path.parent / "best_model.pt"
        torch.save(state, best_path)
        logger.info("Saved best checkpoint to %s", best_path)


def _get_rng_state(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Capture the RNG state of ``random``, ``numpy``, and ``torch``.

    Args:
        existing: Optional existing RNG state dict to augment.

    Returns:
        Dictionary with ``random``, ``numpy``, and ``torch`` keys.
    """
    rng_state = existing if existing is not None else {}
    rng_state["random"] = random.getstate()
    rng_state["numpy"] = np.random.get_state()
    rng_state["torch"] = torch.get_rng_state()
    rng_state["cuda"] = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    return rng_state


def restore_rng_state(rng_state: dict[str, Any] | None) -> None:
    """Restore ``random``, ``numpy``, and ``torch`` RNG states.

    Args:
        rng_state: State dictionary produced by ``_get_rng_state`` or
            stored in a checkpoint.
    """
    if rng_state is None:
        return
    if "random" in rng_state:
        random.setstate(rng_state["random"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"].cpu())
    cuda_rng = rng_state.get("cuda")
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_rng])


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
