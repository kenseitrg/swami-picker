"""Supervised picking models for Phase 4.

The models consume raw FK spectra ``(B, 1, 256, 256)`` and produce a
single output of shape ``(B, num_classes, W)`` where ``num_classes`` is
``spectrum_height + 1`` (one logit per wavenumber bin plus one
"no pick" class).  This forces the model to make an explicit decision
for every frequency column instead of delegating presence detection to
a separate head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two convolutional layers with ReLU activations and optional dropout."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            dropout: Dropout probability applied between the two convolutions.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convolutions with ReLU and optional dropout.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of shape ``(B, out_channels, H, W)``.
        """
        x = F.relu(self.conv1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.relu(self.conv2(x))
        return x


class PickingModel(nn.Module):
    """Compact encoder-decoder for dispersion-curve picking.

    Outputs one logit per (wavenumber_bin, no_pick) class for each
    frequency column.  The absence class is the last index.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        embed_dim: int = 64,
        spectrum_height: int = 256,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the model.

        Args:
            in_channels: Number of input channels (1 for FK spectra).
            base_channels: Width of the first encoder feature map.
            embed_dim: Bottleneck feature dimension.
            spectrum_height: Number of wavenumber bins; determines the
                number of pick classes (``spectrum_height + 1``).
            dropout: Dropout probability inside each ConvBlock.
        """
        super().__init__()
        self.spectrum_height = spectrum_height
        self.num_classes = spectrum_height + 1

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(
            base_channels * 2, embed_dim, dropout=dropout
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(
            embed_dim, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(
            base_channels * 4, base_channels * 2, dropout=dropout
        )
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(
            base_channels * 2, base_channels, dropout=dropout
        )

        # Final classification head.
        # The decoder output is (B, base_channels, H, W).  We want to
        # produce (B, num_classes, W).  Each frequency column is treated
        # as a feature vector of length base_channels * H.
        self.feature_conv = nn.Conv2d(
            base_channels, base_channels, kernel_size=1
        )
        self.classifier = nn.Conv1d(
            base_channels * spectrum_height,
            self.num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass.

        Args:
            x: Input tensor of shape ``(B, 1, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes, W)`` where the last class
            is the "no pick" class.
        """
        b, _, h, w = x.shape
        assert h == self.spectrum_height, (
            f"Input height {h} does not match spectrum_height {self.spectrum_height}"
        )

        # Encoder
        e1 = self.enc1(x)  # (B, base_channels, H, W)
        e2 = self.enc2(self.pool1(e1))  # (B, base_channels*2, H/2, W/2)

        # Bottleneck
        btl = self.bottleneck(self.pool2(e2))  # (B, embed_dim, H/4, W/4)

        # Decoder
        d2 = self.up2(btl)  # (B, base_channels*2, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)  # (B, base_channels, H, W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, base_channels, H, W)

        # Classifier
        feats = F.relu(self.feature_conv(d1))  # (B, base_channels, H, W)
        feats = feats.reshape(b, -1, w)  # (B, base_channels * H, W)
        logits = self.classifier(feats)  # (B, num_classes, W)
        return logits


def build_picking_model(config) -> nn.Module:
    """Factory helper that builds a model from a ``PickingConfig``.

    Args:
        config: A ``PickingConfig`` instance.

    Returns:
        An uninitialized picking model.
    """
    return PickingModel(
        in_channels=1,
        base_channels=config.base_channels,
        embed_dim=config.embed_dim,
        spectrum_height=config.spectrum_height,
        dropout=config.dropout,
    )


def inference_picks(
    logits: torch.Tensor,
    absent_class: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert model logits to pick indices and presence probabilities.

    Args:
        logits: Tensor of shape ``(B, num_classes, W)``.
        absent_class: Index of the "no pick" class.  Defaults to the
            last class.

    Returns:
        Tuple of ``(pick_indices, presence_probs)`` where ``pick_indices``
        has shape ``(B, W)`` (with ``-1`` for absent columns) and
        ``presence_probs`` has shape ``(B, W)``.
    """
    if absent_class is None:
        absent_class = logits.shape[1] - 1

    probs = F.softmax(logits, dim=1)
    pick_indices = logits.argmax(dim=1)  # (B, W)
    presence_probs = 1.0 - probs[:, absent_class, :]  # (B, W)

    pick_indices = pick_indices.where(
        pick_indices != absent_class,
        torch.tensor(-1, device=logits.device, dtype=pick_indices.dtype),
    )
    return pick_indices, presence_probs
