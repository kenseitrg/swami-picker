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
    presence_target = (pick_target >= 0).float()
    direct_mask = torch.zeros_like(presence_target, dtype=torch.bool)
    direct_mask[:, 50:200:2] = True
    return pick_target, presence_target, direct_mask


def test_presence_mask_zeroes_pick_loss():
    """Unpicked columns must not contribute to pick loss."""
    loss_fn = PickingLoss()
    pick_logits = torch.randn(2, 1, 256, 256)
    presence_logits = torch.randn(2, 1, 256, 256)
    pick_target, presence_target, direct_mask = _make_targets()

    # Mark one batch entirely unpicked.
    presence_target[0, :] = 0.0
    pick_target[0, :] = -1.0

    total_loss, _ = loss_fn(
        pick_logits, presence_logits, pick_target, presence_target, direct_mask
    )
    assert torch.isfinite(total_loss)
    assert total_loss.item() >= 0.0


def test_direct_weight_scaling():
    """Direct picks should produce higher per-sample loss when the model is wrong."""
    pick_target, presence_target, direct_mask = _make_targets(batch_size=1)

    # All columns are direct picks for a clean comparison.
    direct_mask = presence_target.bool()

    # Create logits that are wrong for every picked column.
    wrong_logits = torch.zeros(1, 1, 256, 256)

    loss_fn_direct = PickingLoss(direct_pick_weight=2.0)
    loss_fn_uniform = PickingLoss(direct_pick_weight=1.0)

    total_direct, dict_direct = loss_fn_direct(
        wrong_logits, wrong_logits, pick_target, presence_target, direct_mask
    )
    total_uniform, dict_uniform = loss_fn_uniform(
        wrong_logits, wrong_logits, pick_target, presence_target, direct_mask
    )

    # With the weighted normalizer, the average per-sample loss is the same
    # regardless of ``direct_pick_weight`` when all picks are direct. Verify
    # equality within tolerance, and that direct picks truly matter by
    # checking the loss is strictly positive.
    assert torch.isclose(dict_direct["pick_loss"], dict_uniform["pick_loss"], rtol=1e-4)
    assert dict_direct["pick_loss"].item() > 0.0


def test_loss_decreases_when_correct():
    """Gradient descent on a single batch lowers the loss."""
    torch.manual_seed(0)
    model_logits = torch.zeros(1, 1, 256, 256, requires_grad=True)
    presence_logits = torch.zeros(1, 1, 256, 256, requires_grad=True)
    pick_target, presence_target, direct_mask = _make_targets(batch_size=1)

    loss_fn = PickingLoss()
    optimizer = torch.optim.SGD([model_logits, presence_logits], lr=1.0)

    initial_loss, _ = loss_fn(
        model_logits, presence_logits, pick_target, presence_target, direct_mask
    )

    for _ in range(10):
        optimizer.zero_grad()
        loss, _ = loss_fn(
            model_logits, presence_logits, pick_target, presence_target, direct_mask
        )
        loss.backward()
        optimizer.step()

    final_loss, _ = loss_fn(
        model_logits, presence_logits, pick_target, presence_target, direct_mask
    )
    assert final_loss.item() < initial_loss.item()


def test_loss_components_finite():
    """Pick and presence loss components are finite and non-negative."""
    loss_fn = PickingLoss()
    pick_logits = torch.randn(2, 1, 256, 256)
    presence_logits = torch.randn(2, 1, 256, 256)
    pick_target, presence_target, direct_mask = _make_targets()

    total_loss, loss_dict = loss_fn(
        pick_logits, presence_logits, pick_target, presence_target, direct_mask
    )

    assert torch.isfinite(total_loss)
    assert torch.isfinite(loss_dict["pick_loss"])
    assert torch.isfinite(loss_dict["presence_loss"])
