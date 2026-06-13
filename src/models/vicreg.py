"""VICReg: Variance-Invariance-Covariance Regularization for self-supervised learning.

Adapted from Bardes et al., "VICReg: Variance-Invariance-Covariance
Regularization for Self-Supervised Learning", ICLR 2022.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mae import TransformerBlock

if TYPE_CHECKING:
    from torch import Tensor


class VICReg(nn.Module):
    """VICReg self-supervised learning model for FK spectra.

    Architecture:
        Input ──► Encoder (ViT-Small) ──► Mean Pool ──► Projector MLP ──► Embedding

    The encoder mirrors the ViT-Small from Phase 0 MAE. The projector is a
    3-layer MLP with BatchNorm + ReLU between layers (no BN after final layer).

    Args:
        img_size: Spatial resolution of input images (square).
        patch_size: Size of each square patch.
        in_channels: Number of input channels.
        embed_dim: Encoder embedding dimension.
        depth: Number of encoder Transformer blocks.
        num_heads: Number of attention heads in the encoder.
        mlp_ratio: MLP hidden dim ratio.
        projector_hidden_dim: Hidden dimension of the projector MLP.
        projector_out_dim: Output dimension of the projector MLP.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        projector_hidden_dim: int = 2048,
        projector_out_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the VICReg model."""
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        # ---- Encoder (identical to MAE encoder) ----------------------
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ---- Projector -----------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(projector_hidden_dim, projector_out_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise model parameters."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x: Tensor) -> Tensor:
        """Split an image tensor into non-overlapping patches.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            Patches of shape ``(B, N, patch_dim)``.
        """
        B, C, H, W = x.shape
        p = self.patch_size
        n = H // p
        x = x.reshape(B, C, n, p, n, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, n * n, C * p * p)
        return x

    def forward_encoder(self, x: Tensor) -> Tensor:
        """Run the encoder on a batch of images.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            Per-patch encoder outputs of shape ``(B, N, embed_dim)``.
        """
        x = self.patchify(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass: encode, pool, and project.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Projected embeddings of shape ``(B, projector_out_dim)``.
        """
        x = self.forward_encoder(x)
        x = x.mean(dim=1)  # (B, embed_dim)
        z = self.projector(x)  # (B, projector_out_dim)
        return z

    def extract_embeddings(self, x: Tensor) -> Tensor:
        """Extract mean-pooled encoder embeddings (before projector).

        Used for evaluation and downstream clustering. The projector is
        discarded at inference time — the encoder output is the actual
        representation.

        Args:
            x: Input images of shape ``(B, C, H, W)``.

        Returns:
            Embeddings of shape ``(B, embed_dim)``.
        """
        x = self.forward_encoder(x)
        return x.mean(dim=1)  # (B, embed_dim)


def vicreg_loss(
    z1: Tensor,
    z2: Tensor,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the VICReg loss.

    L = λ · s(Z₁, Z₂) + µ · [v(Z₁) + v(Z₂)] + ν · [c(Z₁) + c(Z₂)]

    Args:
        z1: Embeddings from view 1, shape ``(B, D)``.
        z2: Embeddings from view 2, shape ``(B, D)``.
        sim_weight: Weight λ for the invariance (similarity) term.
        var_weight: Weight µ for the variance term.
        cov_weight: Weight ν for the covariance term.

    Returns:
        A tuple of ``(total_loss, inv_loss, var_loss, cov_loss)``.
    """
    # ---- Invariance: MSE between the two views ----------------------
    inv_loss = F.mse_loss(z1, z2)

    # ---- Variance: hinge loss keeping std ≥ 1 per dimension --------
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))

    # ---- Covariance: off-diagonal regularization --------------------
    def _covariance_loss(z: Tensor) -> Tensor:
        """Sum of squared off-diagonal covariance elements."""
        num_features = z.size(1)
        z_centered = z - z.mean(dim=0)
        # Use the unbiased estimator but without Bessel correction for speed;
        # the absolute scale is absorbed into cov_weight anyway.
        cov = (z_centered.T @ z_centered) / z.size(0)
        # Sum of squared off-diagonal elements, normalised by feature count
        off_diag = cov.pow(2).sum() - cov.diag().pow(2).sum()
        return off_diag / num_features

    cov_loss = _covariance_loss(z1) + _covariance_loss(z2)

    total_loss = sim_weight * inv_loss + var_weight * var_loss + cov_weight * cov_loss

    return total_loss, inv_loss, var_loss, cov_loss
