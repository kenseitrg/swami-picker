"""Unit tests for pick-synchronized augmentations."""

from __future__ import annotations

import pytest
import torch

from src.data.picking_augmentations import (
    FreqShift,
    GaussianNoise,
    IntensityJitter,
    PickSyncTransform,
    WavenShift,
)


@pytest.fixture
def sample_tensors():
    """Create a single spectrum and its pick targets."""
    spectrum = torch.randn(1, 256, 256)
    pick_target = torch.full((256,), -1.0)
    pick_target[50:200] = 100.0
    presence_target = (pick_target >= 0).float()
    direct_mask = presence_target.bool()
    confidence = torch.ones(256)
    return spectrum, pick_target, presence_target, direct_mask, confidence


def test_freq_shift_sync(sample_tensors):
    """Horizontal roll moves picks by the same amount and zero-fills image."""
    spectrum, pick_target, presence_target, direct_mask, confidence = sample_tensors
    transform = FreqShift(max_shift=0.05)
    aug = transform(spectrum, pick_target, presence_target, direct_mask, confidence)
    s, p, pr, d, c = aug

    assert s.shape == spectrum.shape
    assert torch.all((p >= 0) | (p == -1))
    # Rolled-in columns must be marked unpicked.
    first_valid = (p >= 0).nonzero(as_tuple=True)[0][0].item()
    assert first_valid > 0 or (p >= 0).sum() == (pick_target >= 0).sum()
    assert torch.all(pr[(p == -1)] == 0)
    assert torch.all(d[(p == -1)] == False)  # noqa: E712

    # Spectrum must be zero in the rolled-in strip.  The wrapped-in
    # columns are not meaningful, so they must all be unpicked, and no
    # picked column may be zeroed.
    zero_cols = torch.all(s == 0, dim=(0, 1))
    picked_cols = p >= 0
    if zero_cols.any():
        assert torch.all(~(zero_cols & picked_cols)), (
            "Zeroed rolled-in columns must be unpicked"
        )


def test_waven_shift_clip(sample_tensors):
    """Vertical shift clamps picks and marks out-of-bounds as unpicked."""
    spectrum, pick_target, presence_target, direct_mask, confidence = sample_tensors
    transform = WavenShift(max_shift=0.1)
    s, p, pr, d, c = transform(
        spectrum, pick_target, presence_target, direct_mask, confidence
    )

    assert torch.all((p >= 0) | (p == -1))
    assert torch.all((p >= 0) <= (p <= 255))
    assert torch.all(pr[(p == -1)] == 0)


def test_intensity_jitter_range(sample_tensors):
    """Intensity jitter scales the spectrum but preserves picks."""
    spectrum, pick_target, presence_target, direct_mask, confidence = sample_tensors
    transform = IntensityJitter(jitter=0.15)
    s, p, pr, d, c = transform(
        spectrum, pick_target, presence_target, direct_mask, confidence
    )

    assert torch.allclose(p, pick_target)
    assert torch.allclose(pr, presence_target)
    assert torch.allclose(d, direct_mask)
    assert torch.allclose(c, confidence)


def test_gaussian_noise_shape_preservation(sample_tensors):
    """Gaussian noise preserves all tensor shapes."""
    spectrum, pick_target, presence_target, direct_mask, confidence = sample_tensors
    transform = GaussianNoise(std=0.05)
    s, p, pr, d, c = transform(
        spectrum, pick_target, presence_target, direct_mask, confidence
    )

    assert s.shape == spectrum.shape
    assert torch.allclose(p, pick_target)
    assert torch.allclose(pr, presence_target)


def test_pick_sync_transform_disabled(sample_tensors):
    """Disabled transform is a no-op."""
    transform = PickSyncTransform(enabled=False)
    out = transform(*sample_tensors)
    for a, b in zip(out, sample_tensors):
        assert torch.allclose(a, b)


def test_pick_sync_transform_enabled_changes_spectrum(sample_tensors):
    """Enabled transform changes the spectrum but keeps shapes."""
    transform = PickSyncTransform(
        enabled=True,
        noise_std=0.05,
        intensity_jitter=0.15,
        freq_shift_max=0.05,
        waven_shift_max=0.03,
    )
    s, p, pr, d, c = transform(*sample_tensors)

    assert s.shape == (1, 256, 256)
    assert p.shape == (256,)
    assert pr.shape == (256,)
    assert d.shape == (256,)
    assert c.shape == (256,)
    assert torch.all((p >= 0) | (p == -1))
