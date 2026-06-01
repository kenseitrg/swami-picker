from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(*, prefer_cuda: bool = True) -> torch.device:
    """Return the best available ``torch.device`` with graceful CPU fallback.

    Args:
        prefer_cuda: Whether to prefer CUDA when available.

    Returns:
        A ``torch.device`` object (``"cuda"`` or ``"cpu"``).
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device
