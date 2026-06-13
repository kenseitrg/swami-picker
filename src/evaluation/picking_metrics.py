"""Evaluation metrics for Phase 4 supervised picking."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def compute_curve_rmse(
    pred_picks: torch.Tensor,
    true_picks: torch.Tensor,
    presence_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute RMSE in pixel indices on valid (picked) columns.

    Args:
        pred_picks: Tensor of shape ``(B, W)`` containing predicted
            wavenumber indices or ``-1``.
        true_picks: Tensor of shape ``(B, W)`` containing ground-truth
            wavenumber indices or ``-1``.
        presence_mask: Optional bool tensor of shape ``(B, W)`` indicating
            columns where a ground-truth pick exists.  If ``None``, uses
            ``true_picks >= 0``.

    Returns:
        Scalar RMSE tensor.  Returns ``nan`` if no valid columns exist.
    """
    valid = (true_picks >= 0) & (pred_picks >= 0)
    if presence_mask is not None:
        valid = valid & presence_mask
    if valid.sum() == 0:
        return torch.tensor(float("nan"), device=pred_picks.device)

    error = (pred_picks.float() - true_picks.float()).abs()
    rmse = torch.sqrt((error[valid] ** 2).mean())
    return rmse


def compute_presence_f1(
    pred_presence: torch.Tensor,
    true_presence: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute presence/absence F1 score per batch element and average.

    Args:
        pred_presence: Tensor of shape ``(B, W)`` with probabilities.
        true_presence: Tensor of shape ``(B, W)`` with ``0``/``1`` labels.
        threshold: Decision threshold on ``pred_presence``.

    Returns:
        Scalar F1 tensor.
    """
    pred_binary = (pred_presence > threshold).float()
    true_binary = true_presence.float()

    tp = (pred_binary * true_binary).sum(dim=1)
    fp = (pred_binary * (1 - true_binary)).sum(dim=1)
    fn = ((1 - pred_binary) * true_binary).sum(dim=1)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    all_zero = (tp == 0) & (fp == 0) & (fn == 0)
    f1 = torch.where(all_zero, torch.tensor(1.0, device=f1.device), f1)
    return f1.mean()


def compute_coverage(picks: torch.Tensor) -> torch.Tensor:
    """Compute the fraction of frequency columns with a valid pick.

    Args:
        picks: Tensor of shape ``(B, W)`` containing indices or ``-1``.

    Returns:
        Scalar coverage tensor.
    """
    return (picks >= 0).float().mean()
