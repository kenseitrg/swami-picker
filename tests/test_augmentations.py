from __future__ import annotations

import pytest
import torch

from src.data.augmentations import FKSpectrumTransform


def test_gaussian_noise_shape_preservation() -> None:
    """Output shape must match input shape."""
    transform = FKSpectrumTransform(noise_std=0.01, intensity_jitter=0.0)
    tensor = torch.randn(1, 256, 256)
    out = transform(tensor)
    assert out.shape == (1, 256, 256)


def test_intensity_jitter_range() -> None:
    """Intensity scale factor must be within [1-jitter, 1+jitter]."""
    jitter = 0.15
    transform = FKSpectrumTransform(noise_std=0.0, intensity_jitter=jitter)
    tensor = torch.ones(1, 256, 256)
    out = transform(tensor)
    scale = out.mean().item()
    assert 1.0 - jitter <= scale <= 1.0 + jitter


def test_augmentation_determinism() -> None:
    """Same torch seed must produce identical output."""
    transform = FKSpectrumTransform(noise_std=0.01, intensity_jitter=0.15)
    tensor = torch.randn(1, 256, 256)

    torch.manual_seed(42)
    out1 = transform(tensor.clone())

    torch.manual_seed(42)
    out2 = transform(tensor.clone())

    assert torch.allclose(out1, out2)


def test_no_augmentation_on_val() -> None:
    """Transform=None should return raw tensor."""
    from pathlib import Path

    from src.data.fk_dataset import FKDataset

    manifest = Path("data/processed/manifest.json")
    if not manifest.exists():
        pytest.skip("Manifest not found")

    ds = FKDataset(manifest, split="val", transform=None)
    tensor, _metadata = ds[0]
    assert tensor.shape == (1, 256, 256)
    # Just verify it loaded without augmentation errors
    assert not torch.isnan(tensor).any()
