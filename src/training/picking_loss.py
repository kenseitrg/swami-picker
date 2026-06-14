"""Loss function for Phase 4 supervised dispersion-curve picking."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PickingLoss(nn.Module):
    """Cross-entropy loss per frequency column.

    Supports a single-classification head (``logits`` of shape
    ``(B, num_classes, W)``) and a multi-mode head (``logits`` of shape
    ``(B, num_modes, num_classes, W)``).  The last class is always
    "no pick".  Direct (human-clicked) picks are weighted higher than
    interpolated picks.  Optional smoothness and monotonicity terms
    encourage continuous, geophysically plausible dispersion curves.
    """

    def __init__(
        self,
        pick_weight: float = 1.0,
        direct_pick_weight: float = 2.0,
        smooth_weight: float = 0.0,
        monotonic_weight: float = 0.0,
        absent_class: int | None = None,
    ) -> None:
        """Initialize the loss.

        Args:
            pick_weight: Global multiplier for the pick loss.
            direct_pick_weight: Multiplicative weight for direct picks
                relative to interpolated picks.
            smooth_weight: Weight for the frequency-axis smoothness term.
                Set to ``0.0`` to disable.
            monotonic_weight: Weight for the soft monotonicity term.  Set
                to ``0.0`` to disable.
            absent_class: Index of the "no pick" class.  Defaults to the
                last class.
        """
        super().__init__()
        self.pick_weight = pick_weight
        self.direct_pick_weight = direct_pick_weight
        self.smooth_weight = smooth_weight
        self.monotonic_weight = monotonic_weight
        self.absent_class = absent_class

    def forward(
        self,
        logits: torch.Tensor,
        pick_target: torch.Tensor,
        direct_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted cross-entropy loss.

        Args:
            logits: Tensor of shape ``(B, num_classes, W)`` for a single
                head, or ``(B, num_modes, num_classes, W)`` for multi-mode.
            pick_target: Tensor of shape ``(B, W)`` containing wavenumber
                indices in ``[0, H-1]`` or ``-1`` / ``H`` for unpicked
                columns.
            direct_mask: Bool tensor of shape ``(B, W)`` indicating direct
                (human-clicked) picks.

        Returns:
            Tuple of ``(total_loss, loss_dict)`` where ``loss_dict``
            contains ``pick_loss`` and optional auxiliary losses.
        """
        if self.absent_class is None:
            if logits.dim() == 3:
                self.absent_class = logits.shape[1] - 1
            else:
                self.absent_class = logits.shape[2] - 1

        target = pick_target.clone()
        target[target < 0] = self.absent_class
        target = target.long()

        if logits.dim() == 3:
            pick_loss = self._single_head_loss(logits, target, direct_mask)
        else:
            pick_loss = self._multi_mode_loss(logits, target, direct_mask)

        total_loss = self.pick_weight * pick_loss
        loss_dict: dict[str, torch.Tensor] = {"pick_loss": pick_loss.detach()}

        # Use the expected single-head logits for auxiliary losses.
        aux_logits = (
            logits if logits.dim() == 3 else self._reduce_multi_mode_logits(logits)
        )

        if self.smooth_weight > 0.0:
            smooth_loss = self._smoothness_loss(aux_logits)
            total_loss = total_loss + self.smooth_weight * smooth_loss
            loss_dict["smooth_loss"] = smooth_loss.detach()

        if self.monotonic_weight > 0.0:
            mono_loss = self._monotonicity_loss(aux_logits)
            total_loss = total_loss + self.monotonic_weight * mono_loss
            loss_dict["mono_loss"] = mono_loss.detach()

        return total_loss, loss_dict

    def _single_head_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        direct_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy for a single classification head."""
        ce = F.cross_entropy(logits, target, reduction="none")
        weights = torch.where(direct_mask, self.direct_pick_weight, 1.0)
        ce = ce * weights
        normalizer = weights.sum() + 1e-6
        return ce.sum() / normalizer

    def _multi_mode_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        direct_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy for multi-mode head using best-alignment.

        For each batch element, assign the target to the mode whose
        current prediction is closest to the ground-truth picks.  This
        avoids forcing a fixed mode ordering and lets the model learn
        which mode is the fundamental.
        """
        batch_size, num_modes, num_classes, width = logits.shape
        device = logits.device

        # Per-mode cross-entropy (B, M, W).
        logits_flat = logits.reshape(batch_size * num_modes, num_classes, width)
        target_flat = (
            target.unsqueeze(1)
            .expand(-1, num_modes, -1)
            .reshape(batch_size * num_modes, width)
        )
        ce_flat = F.cross_entropy(logits_flat, target_flat, reduction="none")
        ce = ce_flat.reshape(batch_size, num_modes, width)

        # Direct-pick weights (B, W) -> (B, 1, W).
        weights = torch.where(direct_mask, self.direct_pick_weight, 1.0).unsqueeze(1)
        weighted_ce = ce * weights

        # Best-alignment: choose the mode with the lowest weighted CE per
        # sample, averaged over columns.
        per_mode_loss = weighted_ce.sum(dim=2) / (weights.sum(dim=2) + 1e-6)
        best_mode = per_mode_loss.argmin(dim=1)  # (B,)

        # Gather the loss from the best-aligned mode.
        best_ce = weighted_ce[torch.arange(batch_size, device=device), best_mode, :]
        best_weights = weights.squeeze(1)[torch.arange(batch_size, device=device)]
        normalizer = best_weights.sum() + 1e-6
        return best_ce.sum() / normalizer

    def _reduce_multi_mode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Collapse multi-mode logits to a single head for aux losses."""
        # Softmax over classes, then average across modes.
        probs = F.softmax(logits, dim=2)  # (B, M, C, W)
        return torch.log(probs.mean(dim=1) + 1e-8)  # (B, C, W)

    def _expected_pick_index(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute expected wavenumber index and presence mass.

        The pick probabilities are normalized over the pick classes so
        that the expected index is independent of the presence/absence
        probability.  The presence mass is returned separately so callers
        can mask invalid transitions.

        Args:
            logits: Tensor of shape ``(B, num_classes, W)``.

        Returns:
            Tuple of ``(expected, presence_mass)`` where both tensors
            have shape ``(B, W)``.
        """
        probs = F.softmax(logits, dim=1)  # (B, C, W)
        absent_mask = torch.ones(
            logits.shape[1], dtype=torch.bool, device=logits.device
        )
        absent_mask[self.absent_class] = False
        pick_probs = probs[:, absent_mask, :]  # (B, H, W)

        pick_mass = pick_probs.sum(dim=1, keepdim=True)  # (B, 1, W)
        pick_probs_norm = pick_probs / (pick_mass + 1e-8)

        class_indices = torch.arange(pick_probs.shape[1], device=logits.device).float()
        expected = (pick_probs_norm * class_indices.view(1, -1, 1)).sum(dim=1)
        return expected, pick_mass.squeeze(1)

    def _smoothness_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Penalize large changes in the expected wavenumber index.

        Transitions across columns with negligible presence mass are
        masked out so that the loss does not conflate presence changes
        with wavenumber jumps.

        Args:
            logits: Tensor of shape ``(B, num_classes, W)``.

        Returns:
            Scalar smoothness loss.
        """
        expected, presence_mass = self._expected_pick_index(logits)
        diff = expected[:, 1:] - expected[:, :-1]
        valid = (presence_mass[:, 1:] > 1e-3) & (presence_mass[:, :-1] > 1e-3)
        return (diff.abs() * valid.float()).sum() / (valid.sum() + 1e-8)

    def _monotonicity_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Soft penalty for non-monotonic expected pick sequences.

        Encourages the expected pick index to either increase or decrease
        monotonically along the frequency axis.  The penalty is the
        minimum of the total positive and total negative change, so a
        perfectly monotonic curve (in either direction) receives zero
        loss.  Only valid transitions (both columns have presence mass)
        contribute.

        Args:
            logits: Tensor of shape ``(B, num_classes, W)``.

        Returns:
            Scalar monotonicity loss.
        """
        expected, presence_mass = self._expected_pick_index(logits)
        diff = expected[:, 1:] - expected[:, :-1]
        valid = (presence_mass[:, 1:] > 1e-3) & (presence_mass[:, :-1] > 1e-3)

        pos_change = (F.relu(diff) * valid.float()).sum(dim=1)
        neg_change = (F.relu(-diff) * valid.float()).sum(dim=1)
        return torch.mean(torch.minimum(pos_change, neg_change))
