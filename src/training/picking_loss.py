"""Loss function for Phase 4 supervised dispersion-curve picking."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PickingLoss(nn.Module):
    """Single 257-class cross-entropy loss per frequency column.

    Each frequency column is classified into one of
    ``num_classes = spectrum_height + 1`` classes.  The last class is
    "no pick".  Direct (human-clicked) picks are weighted higher than
    interpolated picks.
    """

    def __init__(
        self,
        pick_weight: float = 1.0,
        direct_pick_weight: float = 2.0,
        absent_class: int | None = None,
    ) -> None:
        """Initialize the loss.

        Args:
            pick_weight: Global multiplier for the pick loss.
            direct_pick_weight: Multiplicative weight for direct picks
                relative to interpolated picks.
            absent_class: Index of the "no pick" class.  Defaults to the
                last class.
        """
        super().__init__()
        self.pick_weight = pick_weight
        self.direct_pick_weight = direct_pick_weight
        self.absent_class = absent_class

    def forward(
        self,
        logits: torch.Tensor,
        pick_target: torch.Tensor,
        direct_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted cross-entropy loss.

        Args:
            logits: Tensor of shape ``(B, num_classes, W)``.
            pick_target: Tensor of shape ``(B, W)`` containing wavenumber
                indices in ``[0, H-1]`` or ``-1`` / ``H`` for unpicked
                columns.
            direct_mask: Bool tensor of shape ``(B, W)`` indicating direct
                (human-clicked) picks.

        Returns:
            Tuple of ``(total_loss, loss_dict)`` where ``loss_dict``
            contains ``pick_loss``.
        """
        if self.absent_class is None:
            self.absent_class = logits.shape[1] - 1

        # Convert -1 sentinel to the absent class index.
        target = pick_target.clone()
        target[target < 0] = self.absent_class
        target = target.long()

        # Cross-entropy expects (N, C, ...) and target (N, ...).
        ce = F.cross_entropy(logits, target, reduction="none")  # (B, W)

        # Weight direct picks higher; absent columns are still graded
        # because the model must learn to predict absence explicitly.
        weights = torch.where(direct_mask, self.direct_pick_weight, 1.0)
        ce = ce * weights

        normalizer = weights.sum() + 1e-6
        pick_loss = ce.sum() / normalizer

        total_loss = self.pick_weight * pick_loss
        return total_loss, {"pick_loss": pick_loss.detach()}
