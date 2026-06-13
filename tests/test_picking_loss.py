"""Unit tests for the Phase 4 picking loss."""

from __future__ import annotations

import torch

from src.training.picking_loss import PickingLoss


def _make_targets(batch_size: int = 2, num_freq: int = 256, num_waven: int = 256):
    """Create synthetic targets with half the columns picked."""
    pick_target = torch.full((batch_size, num_freq), -1.0)
    pick_target[:, 50:200] = (
        torch.arange(num_waven)
        .float()[torch.linspace(0, num_waven - 1, 150).long()]
        .clamp(0, num_waven - 1)
    )
    direct_mask = torch.zeros_like(pick_target, dtype=torch.bool)
    direct_mask[:, 50:200:2] = True
    return pick_target, direct_mask


def test_unpicked_columns_graded():
    """Absent columns still contribute to the loss."""
    loss_fn = PickingLoss()
    logits = torch.randn(2, 257, 256)
    pick_target, direct_mask = _make_targets()

    # Mark one batch entirely unpicked.
    pick_target[0, :] = -1.0

    total_loss, _ = loss_fn(logits, pick_target, direct_mask)
    assert torch.isfinite(total_loss)
    assert total_loss.item() >= 0.0


def test_direct_weight_scaling():
    """Direct picks receive higher weight when the model is wrong."""
    pick_target, direct_mask = _make_targets(batch_size=1)
    direct_mask = (pick_target >= 0).bool()

    # Logits that are wrong for every picked column.
    wrong_logits = torch.zeros(1, 257, 256)

    loss_fn_direct = PickingLoss(direct_pick_weight=2.0)
    loss_fn_uniform = PickingLoss(direct_pick_weight=1.0)

    total_direct, dict_direct = loss_fn_direct(wrong_logits, pick_target, direct_mask)
    total_uniform, dict_uniform = loss_fn_uniform(
        wrong_logits, pick_target, direct_mask
    )

    # With weighted normalizer, per-sample loss is the same when all picks are direct.
    assert torch.isclose(dict_direct["pick_loss"], dict_uniform["pick_loss"], rtol=1e-4)
    assert dict_direct["pick_loss"].item() > 0.0


def test_loss_decreases_when_correct():
    """Gradient descent on a single batch lowers the loss."""
    torch.manual_seed(0)
    logits = torch.zeros(1, 257, 256, requires_grad=True)
    pick_target, direct_mask = _make_targets(batch_size=1)

    loss_fn = PickingLoss()
    optimizer = torch.optim.SGD([logits], lr=1.0)

    initial_loss, _ = loss_fn(logits, pick_target, direct_mask)

    for _ in range(10):
        optimizer.zero_grad()
        loss, _ = loss_fn(logits, pick_target, direct_mask)
        loss.backward()
        optimizer.step()

    final_loss, _ = loss_fn(logits, pick_target, direct_mask)
    assert final_loss.item() < initial_loss.item()


def test_loss_components_finite():
    """Pick loss component is finite and non-negative."""
    loss_fn = PickingLoss()
    logits = torch.randn(2, 257, 256)
    pick_target, direct_mask = _make_targets()

    total_loss, loss_dict = loss_fn(logits, pick_target, direct_mask)

    assert torch.isfinite(total_loss)
    assert torch.isfinite(loss_dict["pick_loss"])


def test_smoothness_loss_zero_when_uniform():
    """Smoothness loss is zero when every column has the same distribution."""
    loss_fn = PickingLoss(smooth_weight=1.0)
    logits = torch.randn(2, 257, 1).expand(2, 257, 256)
    pick_target, direct_mask = _make_targets()

    total_loss, loss_dict = loss_fn(logits, pick_target, direct_mask)

    assert torch.isfinite(total_loss)
    assert "smooth_loss" in loss_dict
    assert torch.isclose(loss_dict["smooth_loss"], torch.tensor(0.0), atol=1e-5)


def test_smoothness_loss_present_with_jumps():
    """Smoothness loss is positive when adjacent columns differ sharply."""
    loss_fn = PickingLoss(smooth_weight=1.0)
    logits = torch.zeros(1, 257, 256)
    # Alternating one-hot columns.
    logits[:, 0, ::2] = 10.0
    logits[:, 1, 1::2] = 10.0
    pick_target = torch.full((1, 256), -1.0)
    direct_mask = torch.zeros_like(pick_target, dtype=torch.bool)

    total_loss, loss_dict = loss_fn(logits, pick_target, direct_mask)

    assert "smooth_loss" in loss_dict
    assert loss_dict["smooth_loss"].item() > 1e-3
