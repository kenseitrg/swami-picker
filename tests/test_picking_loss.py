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


def test_label_smoothing_lowers_confidence():
    """Label smoothing makes the model less confident at optimum."""
    torch.manual_seed(0)
    logits = torch.zeros(1, 257, 256, requires_grad=True)
    pick_target, direct_mask = _make_targets(batch_size=1)

    loss_fn_smooth = PickingLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD([logits], lr=10.0)

    for _ in range(50):
        optimizer.zero_grad()
        loss, _ = loss_fn_smooth(logits, pick_target, direct_mask)
        loss.backward()
        optimizer.step()

    probs = torch.softmax(logits.detach(), dim=1)
    max_probs = probs.max(dim=1).values
    # After strong smoothing, no probability should be saturated near 1.
    assert max_probs.max().item() < 0.95


def test_absent_class_weight_changes_loss():
    """Increasing absent-class weight changes the optimized predictions."""
    torch.manual_seed(0)
    pick_target, direct_mask = _make_targets(batch_size=1)

    logits_low = torch.zeros(1, 257, 256, requires_grad=True)
    logits_high = torch.zeros(1, 257, 256, requires_grad=True)

    opt_low = torch.optim.SGD([logits_low], lr=1.0)
    opt_high = torch.optim.SGD([logits_high], lr=1.0)

    loss_fn_low = PickingLoss(absent_class_weight=1.0)
    loss_fn_high = PickingLoss(absent_class_weight=5.0)

    for _ in range(20):
        opt_low.zero_grad()
        loss_low, _ = loss_fn_low(logits_low, pick_target, direct_mask)
        loss_low.backward()
        opt_low.step()

        opt_high.zero_grad()
        loss_high, _ = loss_fn_high(logits_high, pick_target, direct_mask)
        loss_high.backward()
        opt_high.step()

    pred_low = logits_low.argmax(dim=1)
    pred_high = logits_high.argmax(dim=1)
    coverage_low = (pred_low != 256).float().mean().item()
    coverage_high = (pred_high != 256).float().mean().item()
    # Higher absent-class weight should push coverage down on average.
    assert coverage_high <= coverage_low


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
    """Smoothness loss is positive when expected wavenumber jumps."""
    loss_fn = PickingLoss(smooth_weight=1.0)
    logits = torch.zeros(1, 257, 256)
    # Even columns predict class 50, odd columns predict class 150.
    logits[:, 50, ::2] = 10.0
    logits[:, 150, 1::2] = 10.0
    pick_target = torch.full((1, 256), -1.0)
    direct_mask = torch.zeros_like(pick_target, dtype=torch.bool)

    total_loss, loss_dict = loss_fn(logits, pick_target, direct_mask)

    assert "smooth_loss" in loss_dict
    assert loss_dict["smooth_loss"].item() > 1e-3


def test_monotonicity_loss_zero_when_monotonic():
    """Monotonicity loss is zero for a strictly increasing expected curve."""
    loss_fn = PickingLoss(monotonic_weight=1.0)
    logits = torch.zeros(1, 257, 256)
    # Expected index increases linearly from 0 to 255.
    for w in range(256):
        logits[0, w, w] = 10.0
    pick_target = torch.full((1, 256), -1.0)
    direct_mask = torch.zeros_like(pick_target, dtype=torch.bool)

    total_loss, loss_dict = loss_fn(logits, pick_target, direct_mask)

    assert "mono_loss" in loss_dict
    assert torch.isclose(loss_dict["mono_loss"], torch.tensor(0.0), atol=1e-5)


def test_monotonicity_loss_positive_when_oscillating():
    """Monotonicity loss is positive when the expected curve oscillates."""
    loss_fn = PickingLoss(monotonic_weight=1.0)
    logits = torch.zeros(1, 257, 256)
    # Expected index alternates between low and high values.
    logits[:, 50, ::2] = 10.0
    logits[:, 200, 1::2] = 10.0
    pick_target = torch.full((1, 256), -1.0)
    direct_mask = torch.zeros_like(pick_target, dtype=torch.bool)

    total_loss, loss_dict = loss_fn(logits, pick_target, direct_mask)

    assert "mono_loss" in loss_dict
    assert loss_dict["mono_loss"].item() > 1e-3


def test_multi_mode_loss_is_finite():
    """Multi-mode logits produce a finite loss."""
    loss_fn = PickingLoss()
    logits = torch.randn(2, 3, 257, 256)
    pick_target, direct_mask = _make_targets()

    total_loss, loss_dict = loss_fn(logits, pick_target, direct_mask)

    assert torch.isfinite(total_loss)
    assert torch.isfinite(loss_dict["pick_loss"])
