"""Supervised picking models for Phase 4.

The models consume raw FK spectra ``(B, 1, 256, 256)`` and produce two
outputs:

* ``pick_logits`` of shape ``(B, 1, 256, 256)``: a heatmap interpreted
  as one logit per (wavenumber, frequency) pixel.  For each frequency
  column we take ``argmax`` over the wavenumber axis to obtain the pick.
* ``presence_logits`` of shape ``(B, 1, 256, 256)``: a heatmap whose
  column-wise mean gives the probability that a mode is visible at that
  frequency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two convolutional layers with ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convolutions with ReLU.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of shape ``(B, out_channels, H, W)``.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class SimpleUNetPickingModel(nn.Module):
    """Lightweight U-Net for dispersion-curve picking.

    The network has three encoder blocks, a bottleneck, and three decoder
    blocks with skip connections.  Two 1x1 convolutional heads produce the
    pick heatmap and the presence heatmap.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        embed_dim: int = 128,
    ) -> None:
        """Initialize the U-Net picking model.

        Args:
            in_channels: Number of input channels (1 for FK spectra).
            base_channels: Width of the first encoder feature map.
            embed_dim: Bottleneck feature dimension.  Kept for API
                compatibility with the conditional variant.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, embed_dim)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(embed_dim, embed_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(embed_dim * 2, embed_dim)
        self.up2 = nn.ConvTranspose2d(
            embed_dim, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Heads
        self.pick_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.presence_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full forward pass.

        Args:
            x: Input tensor of shape ``(B, 1, 256, 256)``.

        Returns:
            Tuple of ``(pick_logits, presence_logits)``, each of shape
            ``(B, 1, 256, 256)``.
        """
        # Encoder
        e1 = self.enc1(x)  # (B, base_channels, 256, 256)
        e2 = self.enc2(self.pool1(e1))  # (B, base_channels*2, 128, 128)
        e3 = self.enc3(self.pool2(e2))  # (B, embed_dim, 64, 64)

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))  # (B, embed_dim, 32, 32)

        # Decoder with skip connections
        d3 = self.up3(b)  # (B, embed_dim, 64, 64)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # (B, embed_dim, 64, 64)
        d2 = self.up2(d3)  # (B, base_channels*2, 128, 128)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, base_channels*2, 128, 128)
        d1 = self.up1(d2)  # (B, base_channels, 256, 256)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, base_channels, 256, 256)

        pick_logits = self.pick_head(d1)  # (B, 1, 256, 256)
        presence_logits = self.presence_head(d1)  # (B, 1, 256, 256)

        return pick_logits, presence_logits


class EncoderDecoderPickingModel(nn.Module):
    """Simpler encoder-decoder baseline with no skip connections.

    This variant has fewer parameters and is useful as a lower-capacity
    baseline if the U-Net overfits on a small annotated set.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        embed_dim: int = 128,
    ) -> None:
        """Initialize the encoder-decoder picking model.

        Args:
            in_channels: Number of input channels.
            base_channels: Width of the first encoder feature map.
            embed_dim: Bottleneck feature dimension.
        """
        super().__init__()
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim, base_channels * 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, kernel_size=2, stride=2
            ),
            nn.ReLU(),
        )

        self.pick_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.presence_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full forward pass.

        Args:
            x: Input tensor of shape ``(B, 1, 256, 256)``.

        Returns:
            Tuple of ``(pick_logits, presence_logits)``, each of shape
            ``(B, 1, 256, 256)``.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        pick_logits = self.pick_head(x)
        presence_logits = self.presence_head(x)
        return pick_logits, presence_logits


class ClusterConditionalPickingModel(nn.Module):
    """U-Net picking model conditioned on a cluster embedding vector.

    The 128-D cluster embedding is broadcast to the bottleneck spatial
    resolution and concatenated with the encoder features before the
    decoder.  This provides a learned prior per spectral type.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        embed_dim: int = 128,
        cluster_embed_dim: int = 128,
    ) -> None:
        """Initialize the conditional picking model.

        Args:
            in_channels: Number of input channels.
            base_channels: Width of the first encoder feature map.
            embed_dim: Bottleneck feature dimension.
            cluster_embed_dim: Dimensionality of the conditioning vector.
        """
        super().__init__()
        self.unet = SimpleUNetPickingModel(
            in_channels=in_channels,
            base_channels=base_channels,
            embed_dim=embed_dim,
        )
        self.cluster_embed_dim = cluster_embed_dim

        # Project cluster embedding and fuse into the bottleneck.
        self.cluster_proj = nn.Linear(cluster_embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        cluster_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the conditional forward pass.

        Args:
            x: Input tensor of shape ``(B, 1, 256, 256)``.
            cluster_embedding: Optional conditioning vector of shape
                ``(B, cluster_embed_dim)``.  If ``None``, the model falls
                back to the unconditional U-Net behavior (zeros).

        Returns:
            Tuple of ``(pick_logits, presence_logits)``, each of shape
            ``(B, 1, 256, 256)``.
        """
        if cluster_embedding is None:
            cluster_embedding = torch.zeros(
                x.shape[0], self.cluster_embed_dim, device=x.device, dtype=x.dtype
            )

        # Use the U-Net encoder/bottleneck manually so we can inject the
        # cluster embedding before the decoder.
        e1 = self.unet.enc1(x)
        e2 = self.unet.enc2(self.unet.pool1(e1))
        e3 = self.unet.enc3(self.unet.pool2(e2))
        b = self.unet.bottleneck(self.unet.pool3(e3))

        _, _, h, w = b.shape
        cluster_feat = self.cluster_proj(cluster_embedding)  # (B, embed_dim)
        cluster_feat = cluster_feat.view(-1, self.unet.embed_dim, 1, 1).expand(
            -1, -1, h, w
        )
        b = b + cluster_feat

        d3 = self.unet.up3(b)
        d3 = self.unet.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.unet.up2(d3)
        d2 = self.unet.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.unet.up1(d2)
        d1 = self.unet.dec1(torch.cat([d1, e1], dim=1))

        pick_logits = self.unet.pick_head(d1)
        presence_logits = self.unet.presence_head(d1)
        return pick_logits, presence_logits


def build_picking_model(config) -> nn.Module:
    """Factory helper that builds a model from a ``PickingConfig``.

    Args:
        config: A ``PickingConfig`` instance.

    Returns:
        An uninitialized picking model.

    Raises:
        ValueError: If ``config.backbone`` is not supported.
    """
    base_channels = config.base_channels
    embed_dim = config.embed_dim

    if config.backbone == "unet":
        if config.use_cluster_conditioning:
            return ClusterConditionalPickingModel(
                base_channels=base_channels, embed_dim=embed_dim
            )
        return SimpleUNetPickingModel(base_channels=base_channels, embed_dim=embed_dim)

    if config.backbone == "encoder_decoder":
        if config.use_cluster_conditioning:
            raise ValueError("encoder_decoder does not support cluster conditioning")
        return EncoderDecoderPickingModel(
            base_channels=base_channels, embed_dim=embed_dim
        )

    msg = f"Unknown backbone: {config.backbone}"
    raise ValueError(msg)


def inference_picks(
    pick_logits: torch.Tensor,
    presence_logits: torch.Tensor,
    presence_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert model logits to pick indices and presence probabilities.

    Args:
        pick_logits: Tensor of shape ``(B, 1, H, W)`` where ``H`` is the
            wavenumber axis and ``W`` is the frequency axis.
        presence_logits: Tensor of shape ``(B, 1, H, W)``.
        presence_threshold: Threshold on the column-wise presence
            probability below which a pick is marked as ``-1``.

    Returns:
        Tuple of ``(pick_indices, presence_probs)`` where ``pick_indices``
        has shape ``(B, W)`` and ``presence_probs`` has shape ``(B, W)``.
    """
    heatmap = pick_logits.squeeze(1)  # (B, H, W)
    pick_indices = heatmap.argmax(dim=1)  # (B, W)

    presence_heatmap = presence_logits.squeeze(1)  # (B, H, W)
    presence_probs = torch.sigmoid(presence_heatmap.mean(dim=1))  # (B, W)

    pick_indices = pick_indices.where(
        presence_probs > presence_threshold,
        torch.tensor(-1, device=pick_indices.device, dtype=pick_indices.dtype),
    )
    return pick_indices, presence_probs
