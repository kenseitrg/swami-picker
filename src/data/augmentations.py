from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class FKSpectrumTransform:
    """On-the-fly augmentation pipeline for FK spectra.

    Applies intensity jitter followed by Gaussian noise. All operations
    are deterministic when the global torch seed is fixed.

    Args:
        noise_std: Standard deviation of additive Gaussian noise.
        intensity_jitter: Maximum relative intensity scale factor.
            The scale is drawn from ``U(1 - jitter, 1 + jitter)``.
    """

    def __init__(self, noise_std: float = 0.01, intensity_jitter: float = 0.15) -> None:
        """Initialise the transform.

        Args:
            noise_std: Standard deviation of additive Gaussian noise.
            intensity_jitter: Maximum relative intensity scale factor.
        """
        self.noise_std = noise_std
        self.intensity_jitter = intensity_jitter

    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply the augmentation pipeline.

        Args:
            tensor: Input tensor of shape ``(1, 256, 256)``.

        Returns:
            Augmented tensor of the same shape.
        """
        # Intensity jitter
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

        # Gaussian noise
        if self.noise_std > 0.0:
            noise = torch.randn_like(tensor) * self.noise_std
            tensor = tensor + noise

        return tensor
