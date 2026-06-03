from __future__ import annotations

import math

import torch


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler with linear warmup and cosine decay.

    The learning rate starts at 0, linearly increases to the initial
    optimizer LR over ``num_warmup_steps``, then decays following a
    cosine curve down to ``min_lr_ratio`` of the initial LR.

    Args:
        optimizer: Optimiser whose learning rate will be scheduled.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as a fraction of the initial LR.

    Returns:
        A ``LambdaLR`` scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
