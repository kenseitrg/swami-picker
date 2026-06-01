from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Sets seeds for Python ``random``, NumPy, and PyTorch (CPU and all
    CUDA devices). Deterministic CUDA behaviour is *not* enabled by
    default because it can degrade performance; call
    ``torch.use_deterministic_algorithms(True)`` manually if required.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug("Set random seed to %d", seed)
