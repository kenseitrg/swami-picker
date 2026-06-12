"""Loss function for Phase 4 supervised dispersion-curve picking."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PickingLoss(nn.Module):
    """Combined pick-classification + presence loss.

    Each frequency column is treated as a 256-way classification over
    wavenumber bins.  Direct (human-clicked) picks are weighted higher
    than interpolated picks.  A separate presence head learns which
    frequency columns actually contain a visible mode.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        bce_weight: float = 0.5,
        direct_pick_weight: float = 2.0,
    ) -> None:
        """Initialize the loss.

        Args:
            l1_weight: Weight for the pick cross-entropy loss.  The name
                is kept for config compatibility; the loss is cross-entropy,
                not L1.
            bce_weight: Weight for the presence BCE loss.
            direct_pick_weight: Multiplicative weight for direct picks
                relative to interpolated picks.
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.bce_weight = bce_weight
        self.direct_pick_weight = direct_pick_weight

    def forward(
        self,
        pick_logits: torch.Tensor,
        presence_logits: torch.Tensor,
        pick_target: torch.Tensor,
        presence_target: torch.Tensor,
        direct_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the combined loss.

        Args:
            pick_logits: Tensor of shape ``(B, 1, H, W)`` where ``H`` is
                the wavenumber axis and ``W`` is the frequency axis.
            presence_logits: Tensor of shape ``(B, 1, H, W)``.
            pick_target: Tensor of shape ``(B, W)`` containing wavenumber
                indices in ``[0, H-1]`` or ``-1`` for unpicked columns.
            presence_target: Tensor of shape ``(B, W)`` with ``1.0`` where
                a pick exists and ``0.0`` otherwise.
            direct_mask: Bool tensor of shape ``(B, W)`` indicating direct
                (human-clicked) picks.

        Returns:
            Tuple of ``(total_loss, loss_dict)`` where ``loss_dict``
            contains ``pick_loss`` and ``presence_loss``.
        """
        batch_size, _, h, w = pick_logits.shape
        eps = 1e-6

        # Cross-entropy expects (N, C, ...) and target (N, ...).
        # Squeeze channel dimension: (B, H classes, W positions).
        pick_heatmap = pick_logits.squeeze(1)  # (B, H, W)

        # Mask out unpicked columns (-1) before cross_entropy.
        valid = pick_target >= 0  # (B, W)
        pick_target_clamped = pick_target.clamp(min=0).long()

        ce = F.cross_entropy(
            pick_heatmap,
            pick_target_clamped,
            reduction="none",
        )  # (B, W)

        # Weight direct picks higher; mask invalid columns.
        weights = torch.where(direct_mask, self.direct_pick_weight, 1.0)
        ce = ce * weights * presence_target * valid.float()

        normalizer = presence_target.sum() + eps
        pick_loss = ce.sum() / normalizer

        # Presence loss: one logit per frequency column.
        presence_pred = presence_logits.squeeze(1).mean(dim=1)  # (B, W)
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_pred,
            presence_target,
            reduction="mean",
        )

        total_loss = self.l1_weight * pick_loss + self.bce_weight * presence_loss

        loss_dict = {
            "pick_loss": pick_loss.detach(),
            "presence_loss": presence_loss.detach(),
        }
        return total_loss, loss_dict
