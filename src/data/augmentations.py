from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class FKSpectrumTransform:
    """On-the-fly augmentation pipeline for FK spectra.

    Applies a cascade of physically motivated augmentations to increase
    training diversity and prevent the MAE from learning a trivial
    "average spectrum" reconstruction.

    Augmentations applied in order:
        1. Frequency-axis shift (horizontal) — simulates velocity variation
        2. Wavenumber-axis shift (vertical) — simulates array-aperture variation
        3. Frequency-band dropout — simulates noisy / aliased frequency bands
        4. Intensity jitter — simulates source amplitude variation
        5. Gaussian noise — simulates data quality variability

    All operations are deterministic when the global torch seed is fixed.

    Args:
        noise_std: Standard deviation of additive Gaussian noise.
        intensity_jitter: Maximum relative intensity scale factor.
            The scale is drawn from ``U(1 - jitter, 1 + jitter)``.
        freq_shift_max: Max frequency shift as fraction of image width
            (horizontal axis). Drawn from ``U(-max, +max)``.
        waven_shift_max: Max wavenumber shift as fraction of image height
            (vertical axis). Drawn from ``U(-max, +max)``.
        freq_dropout_prob: Probability of applying a frequency-band dropout.
        freq_dropout_width: Width of dropped band as fraction of image width.
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        intensity_jitter: float = 0.30,
        freq_shift_max: float = 0.0,
        waven_shift_max: float = 0.0,
        freq_dropout_prob: float = 0.0,
        freq_dropout_width: float = 0.05,
    ) -> None:
        """Initialise the transform pipeline.

        Args:
            noise_std: Standard deviation of additive Gaussian noise.
            intensity_jitter: Maximum relative intensity scale factor.
            freq_shift_max: Max horizontal (frequency) shift as fraction of width.
            waven_shift_max: Max vertical (wavenumber) shift as fraction of height.
            freq_dropout_prob: Probability of frequency-band dropout.
            freq_dropout_width: Width of dropped band as fraction of image width.
        """
        self.noise_std = noise_std
        self.intensity_jitter = intensity_jitter
        self.freq_shift_max = freq_shift_max
        self.waven_shift_max = waven_shift_max
        self.freq_dropout_prob = freq_dropout_prob
        self.freq_dropout_width = freq_dropout_width

    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply the augmentation pipeline.

        Args:
            tensor: Input tensor of shape ``(1, H, W)`` where H=W=256.

        Returns:
            Augmented tensor of the same shape.
        """
        # 1. Frequency shift (horizontal axis, dim=2)
        if self.freq_shift_max > 0.0:
            shift_px = int(
                torch.empty(1)
                .uniform_(-self.freq_shift_max, self.freq_shift_max)
                .item()
                * tensor.shape[2]
            )
            if shift_px != 0:
                tensor = torch.roll(tensor, shifts=shift_px, dims=2)
                if shift_px > 0:
                    tensor[:, :, :shift_px] = 0.0
                else:
                    tensor[:, :, shift_px:] = 0.0

        # 2. Wavenumber shift (vertical axis, dim=1)
        if self.waven_shift_max > 0.0:
            shift_px = int(
                torch.empty(1)
                .uniform_(-self.waven_shift_max, self.waven_shift_max)
                .item()
                * tensor.shape[1]
            )
            if shift_px != 0:
                tensor = torch.roll(tensor, shifts=shift_px, dims=1)
                if shift_px > 0:
                    tensor[:, :shift_px, :] = 0.0
                else:
                    tensor[:, shift_px:, :] = 0.0

        # 3. Frequency-band dropout
        if self.freq_dropout_prob > 0.0 and torch.rand(1).item() < self.freq_dropout_prob:
            band_width = max(1, int(tensor.shape[2] * self.freq_dropout_width))
            start = torch.randint(0, tensor.shape[2] - band_width + 1, (1,)).item()
            tensor[:, :, start:start + band_width] = 0.0

        # 4. Intensity jitter
        if self.intensity_jitter > 0.0:
            scale = (
                torch.empty(1)
                .uniform_(
                    1.0 - self.intensity_jitter,
                    1.0 + self.intensity_jitter,
                )
                .item()
            )
            tensor = tensor * scale

        # 5. Gaussian noise
        if self.noise_std > 0.0:
            noise = torch.randn_like(tensor) * self.noise_std
            tensor = tensor + noise

        return tensor
