"""Evaluation metrics for Phase 4 supervised picking."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_curve_rmse(
    pred_picks: torch.Tensor,
    true_picks: torch.Tensor,
    presence_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute RMSE in pixel indices on valid (picked) columns.

    Args:
        pred_picks: Tensor of shape ``(B, W)`` containing predicted
            wavenumber indices or ``-1``.
        true_picks: Tensor of shape ``(B, W)`` containing ground-truth
            wavenumber indices or ``-1``.
        presence_mask: Bool tensor of shape ``(B, W)`` indicating columns
            where a ground-truth pick exists.

    Returns:
        Scalar RMSE tensor.  Returns ``nan`` if no valid columns exist.
    """
    valid = presence_mask & (true_picks >= 0) & (pred_picks >= 0)
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
    return f1.mean()


def compute_velocity_error(
    pred_picks: np.ndarray,
    true_picks: np.ndarray,
    metadata: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity errors between predicted and true picks.

    Args:
        pred_picks: Array of shape ``(N,)`` with predicted wavenumber
            indices or ``-1``.
        true_picks: Array of shape ``(N,)`` with true wavenumber indices
            or ``-1``.
        metadata: Metadata dictionary for one spectrum containing
            ``freq_axis_resized`` and ``waven_axis_resized``.

    Returns:
        Tuple of ``(delta_v_over_v, valid_mask)`` where ``delta_v_over_v``
        is an array of relative velocity errors and ``valid_mask`` is a
        boolean array indicating columns where both picks are valid.
    """
    freq_axis = np.asarray(metadata["freq_axis_resized"])
    waven_axis = np.asarray(metadata["waven_axis_resized"])

    valid = (pred_picks >= 0) & (true_picks >= 0)
    if not valid.any():
        return np.array([]), valid

    f_pred = freq_axis[valid]
    f_true = freq_axis[valid]
    k_pred = waven_axis[pred_picks[valid].astype(int)]
    k_true = waven_axis[true_picks[valid].astype(int)]

    # Avoid division by zero; also skip zero wavenumber.
    v_mask = (k_true > 0) & (k_pred > 0) & (f_true > 0) & (f_pred > 0)
    v_pred = f_pred[v_mask] / k_pred[v_mask]
    v_true = f_true[v_mask] / k_true[v_mask]

    delta_v_over_v = np.abs(v_pred - v_true) / v_true
    return delta_v_over_v, valid


def compute_coverage(picks: torch.Tensor) -> torch.Tensor:
    """Compute the fraction of frequency columns with a valid pick.

    Args:
        picks: Tensor of shape ``(B, W)`` containing indices or ``-1``.

    Returns:
        Scalar coverage tensor.
    """
    return (picks >= 0).float().mean()
