"""Pick-synchronized augmentations for Phase 4 supervised picking.

Unlike plain spectrum augmentations, these transforms update both the
image and the ground-truth pick target consistently.
"""

from __future__ import annotations

import torch


def _roll_with_fill(
    tensor: torch.Tensor, shift: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll a tensor along *dim* and return a mask of valid positions.

    Args:
        tensor: Input tensor of any shape.
        shift: Number of positions to roll (positive = forward).
        dim: Dimension along which to roll.

    Returns:
        A tuple of ``(rolled_tensor, valid_mask)`` where ``valid_mask``
        has the same shape as *tensor* and is ``True`` for positions that
        did not wrap around from the other side.
    """
    rolled = torch.roll(tensor, shifts=shift, dims=dim)
    valid = torch.ones_like(tensor, dtype=torch.bool)

    if shift > 0:
        # First ``shift`` positions are filled from the end.
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(None, shift)
        valid[tuple(slices)] = False
    elif shift < 0:
        # Last ``|shift|`` positions are filled from the start.
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(shift, None)
        valid[tuple(slices)] = False

    return rolled, valid


class FreqShift:
    """Horizontal frequency shift: rolls the spectrum and its picks together."""

    def __init__(self, max_shift: float = 0.05) -> None:
        """Initialize frequency shift augmentation.

        Args:
            max_shift: Maximum shift as a fraction of the 256 frequency
                columns. The actual shift is drawn uniformly from
                ``[-max_shift, +max_shift]``.
        """
        self.max_shift = max_shift

    def __call__(
        self,
        spectrum: torch.Tensor,
        pick_target: torch.Tensor,
        presence_target: torch.Tensor,
        direct_mask: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a random horizontal shift.

        Args:
            spectrum: Tensor of shape ``(1, 256, 256)``.
            pick_target: Tensor of shape ``(256,)`` with pick indices.
            presence_target: Tensor of shape ``(256,)``.
            direct_mask: Tensor of shape ``(256,)``.
            confidence: Tensor of shape ``(256,)``.

        Returns:
            Augmented tuple with the same shapes.
        """
        if self.max_shift == 0.0:
            return spectrum, pick_target, presence_target, direct_mask, confidence

        size = spectrum.shape[2]
        max_pixels = int(round(self.max_shift * size))
        if max_pixels == 0:
            return spectrum, pick_target, presence_target, direct_mask, confidence

        shift = int(torch.randint(-max_pixels, max_pixels + 1, (1,)).item())
        if shift == 0:
            return spectrum, pick_target, presence_target, direct_mask, confidence

        spectrum, _ = _roll_with_fill(spectrum, shift, dim=2)
        pick_target, valid = _roll_with_fill(pick_target, shift, dim=0)
        presence_target, _ = _roll_with_fill(presence_target, shift, dim=0)
        direct_mask, _ = _roll_with_fill(direct_mask, shift, dim=0)
        confidence, _ = _roll_with_fill(confidence, shift, dim=0)

        # Mark rolled-in columns as unpicked.
        pick_target = pick_target.where(valid, torch.tensor(-1.0))
        presence_target = presence_target.where(valid, torch.tensor(0.0))
        direct_mask = direct_mask.where(valid, torch.tensor(False))
        confidence = confidence.where(valid, torch.tensor(0.0))

        return spectrum, pick_target, presence_target, direct_mask, confidence


class WavenShift:
    """Vertical wavenumber shift: moves all picks up or down."""

    def __init__(self, max_shift: float = 0.03) -> None:
        """Initialize wavenumber shift augmentation.

        Args:
            max_shift: Maximum shift as a fraction of the 256 wavenumber
                rows. The actual shift is drawn uniformly from
                ``[-max_shift, +max_shift]``.
        """
        self.max_shift = max_shift

    def __call__(
        self,
        spectrum: torch.Tensor,
        pick_target: torch.Tensor,
        presence_target: torch.Tensor,
        direct_mask: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a random vertical shift.

        Args:
            spectrum: Tensor of shape ``(1, 256, 256)``.
            pick_target: Tensor of shape ``(256,)`` with pick indices.
            presence_target: Tensor of shape ``(256,)``.
            direct_mask: Tensor of shape ``(256,)``.
            confidence: Tensor of shape ``(256,)``.

        Returns:
            Augmented tuple with the same shapes.
        """
        if self.max_shift == 0.0:
            return spectrum, pick_target, presence_target, direct_mask, confidence

        size = spectrum.shape[1]
        max_pixels = int(round(self.max_shift * size))
        if max_pixels == 0:
            return spectrum, pick_target, presence_target, direct_mask, confidence

        shift = int(torch.randint(-max_pixels, max_pixels + 1, (1,)).item())
        if shift == 0:
            return spectrum, pick_target, presence_target, direct_mask, confidence

        spectrum, _ = _roll_with_fill(spectrum, shift, dim=1)

        valid = torch.ones_like(pick_target, dtype=torch.bool)
        shifted = pick_target + float(shift)
        shifted = torch.clamp(shifted, 0.0, float(size - 1))

        # Any pick that would have been pushed out of bounds becomes unpicked.
        out_of_bounds = (pick_target + float(shift) < 0.0) | (
            pick_target + float(shift) > float(size - 1)
        )
        valid = valid & ~out_of_bounds

        pick_target = shifted.where(valid, torch.tensor(-1.0))
        presence_target = presence_target.where(valid, torch.tensor(0.0))
        direct_mask = direct_mask.where(valid, torch.tensor(False))
        confidence = confidence.where(valid, torch.tensor(0.0))

        return spectrum, pick_target, presence_target, direct_mask, confidence


class IntensityJitter:
    """Multiplicative intensity scaling (does not move picks)."""

    def __init__(self, jitter: float = 0.15) -> None:
        """Initialize intensity jitter.

        Args:
            jitter: Relative scale factor. The spectrum is multiplied by
                ``U(1 - jitter, 1 + jitter)``.
        """
        self.jitter = jitter

    def __call__(
        self,
        spectrum: torch.Tensor,
        pick_target: torch.Tensor,
        presence_target: torch.Tensor,
        direct_mask: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random intensity scaling."""
        if self.jitter == 0.0:
            return spectrum, pick_target, presence_target, direct_mask, confidence

        scale = torch.empty(1).uniform_(1.0 - self.jitter, 1.0 + self.jitter).item()
        spectrum = spectrum * scale
        return spectrum, pick_target, presence_target, direct_mask, confidence


class GaussianNoise:
    """Additive Gaussian noise (does not move picks)."""

    def __init__(self, std: float = 0.05) -> None:
        """Initialize Gaussian noise augmentation.

        Args:
            std: Standard deviation of the noise in normalized amplitude
                units.
        """
        self.std = std

    def __call__(
        self,
        spectrum: torch.Tensor,
        pick_target: torch.Tensor,
        presence_target: torch.Tensor,
        direct_mask: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random Gaussian noise."""
        if self.std == 0.0:
            return spectrum, pick_target, presence_target, direct_mask, confidence

        noise = torch.randn_like(spectrum) * self.std
        spectrum = spectrum + noise
        return spectrum, pick_target, presence_target, direct_mask, confidence


class PickSyncTransform:
    """Composed pick-synchronized augmentation pipeline."""

    def __init__(
        self,
        enabled: bool = True,
        noise_std: float = 0.05,
        intensity_jitter: float = 0.15,
        freq_shift_max: float = 0.05,
        waven_shift_max: float = 0.03,
    ) -> None:
        """Initialize the composed augmentation pipeline.

        Args:
            enabled: If ``False``, the transform is a no-op.
            noise_std: Std of additive Gaussian noise.
            intensity_jitter: Relative intensity scale range.
            freq_shift_max: Maximum horizontal shift fraction.
            waven_shift_max: Maximum vertical shift fraction.
        """
        self.enabled = enabled
        if enabled:
            self.transforms: list = [
                IntensityJitter(intensity_jitter),
                GaussianNoise(noise_std),
                FreqShift(freq_shift_max),
                WavenShift(waven_shift_max),
            ]
        else:
            self.transforms = []

    def __call__(
        self,
        spectrum: torch.Tensor,
        pick_target: torch.Tensor,
        presence_target: torch.Tensor,
        direct_mask: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply all enabled augmentations in order."""
        for transform in self.transforms:
            spectrum, pick_target, presence_target, direct_mask, confidence = transform(
                spectrum, pick_target, presence_target, direct_mask, confidence
            )
        return spectrum, pick_target, presence_target, direct_mask, confidence
