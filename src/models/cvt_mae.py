"""Convolutional Vision Transformer (CvT) based Masked Autoencoder.

Adapts the CvT architecture (Wu et al., 2021) for MAE-style self-supervised
pre-training.  The encoder uses depth-wise convolutional projections for
Q, K, V before self-attention, operating on the full spatial token grid.
Masked patches are replaced with a learnable mask token prior to encoding,
maintaining the MAE reconstruction objective while leveraging CvT's
local inductive bias.

The masking strategy follows ConvMAE (Gao et al., 2022): the mask is
computed on the full token grid, masked positions are filled with a
learnable token, and the CvT encoder processes the complete grid.  After
encoding, only the visible tokens are passed to the lightweight decoder,
which adds mask tokens and reconstructs all patches.

References:
    Wu, H., et al. (2021). CvT: Introducing Convolutions to Vision
        Transformers. arXiv:2103.15808.
    Gao, P., et al. (2022). ConvMAE: Masked Convolution Meets Masked
        Autoencoders. arXiv:2205.03892.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.models.mae import TransformerBlock

if TYPE_CHECKING:
    from torch import Tensor


class CvTAttention(nn.Module):
    """Multi-head self-attention with depth-wise convolutional projections.

    Each of Q, K, V is first projected through a depth-wise separable
    convolution over the spatial token grid, then reshaped back to the
    sequence domain for standard dot-product attention.  This introduces
    local spatial context into the attention mechanism while preserving
    the global receptive field of Transformers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the attention module.

        Args:
            embed_dim: Token embedding dimension.
            num_heads: Number of attention heads.
            kernel_size: Size of the depth-wise convolution kernel.
            dropout: Dropout probability for attention weights and output.

        Raises:
            ValueError: If ``embed_dim`` is not divisible by ``num_heads``.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size ({kernel_size}) must be odd to preserve "
                "spatial dimensions with padding='same' equivalence."
            )
        padding = kernel_size // 2
        self.conv_proj_q = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=kernel_size,
                padding=padding,
                groups=embed_dim,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
        )
        self.conv_proj_k = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=kernel_size,
                padding=padding,
                groups=embed_dim,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
        )
        self.conv_proj_v = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=kernel_size,
                padding=padding,
                groups=embed_dim,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
        )

        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, N, D)`` where ``N = h * w``.
            h: Height of the token grid in patch units.
            w: Width of the token grid in patch units.

        Returns:
            Output tensor of shape ``(B, N, D)``.
        """
        B, N, D = x.shape

        # Reshape to spatial grid for conv projections
        x_spatial = x.transpose(1, 2).reshape(B, D, h, w)

        # Depth-wise conv projections
        q = self.conv_proj_q(x_spatial).reshape(B, D, -1).transpose(1, 2)
        k = self.conv_proj_k(x_spatial).reshape(B, D, -1).transpose(1, 2)
        v = self.conv_proj_v(x_spatial).reshape(B, D, -1).transpose(1, 2)

        # Linear projections and multi-head split:
        # (B, N, D) -> (B, num_heads, N, head_dim)
        q = self.proj_q(q).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(k).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(v).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class CvTBlock(nn.Module):
    """CvT Transformer block with convolutional projection attention.

    Uses pre-norm residual connections: the attention and MLP sub-layers
    are each preceded by layer normalisation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the block.

        Args:
            embed_dim: Token embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
            kernel_size: Convolution kernel size for attention projections.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CvTAttention(embed_dim, num_heads, kernel_size, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, N, D)``.
            h: Height of the token grid in patch units.
            w: Width of the token grid in patch units.

        Returns:
            Output tensor of shape ``(B, N, D)``.
        """
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x))
        return x


class CvTMaskedAutoencoder(nn.Module):
    """CvT-based Masked Autoencoder for self-supervised learning.

    Unlike the standard ViT-based MAE, the CvT encoder processes **all**
    tokens (no token dropping during encoding).  Masked patches are
    replaced with a learnable mask token *before* the encoder, so the
    CvT blocks see a corrupted but spatially regular grid.  After the
    encoder, only visible tokens are gathered for the decoder, which
    reconstructs all patches.  This design is inspired by ConvMAE's
    strategy of masking the spatial grid for convolutional stages.
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
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mask_ratio: float = 0.75,
        use_block_masking: bool = True,
        block_size: int = 2,
        cvt_kernel_size: int = 3,
        use_pos_embed: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the CvT-MAE.

        Args:
            img_size: Spatial resolution of input images (square).
            patch_size: Size of each square patch.
            in_channels: Number of input channels.
            embed_dim: Encoder embedding dimension.
            depth: Number of encoder CvT blocks.
            num_heads: Number of attention heads in the encoder.
            mlp_ratio: MLP hidden dim ratio.
            decoder_embed_dim: Decoder embedding dimension.
            decoder_depth: Number of decoder Transformer blocks.
            decoder_num_heads: Number of attention heads in the decoder.
            mask_ratio: Fraction of patches to mask.
            use_block_masking: If ``True``, use block-wise masking.
            block_size: Side length of masking blocks (in patch units).
            cvt_kernel_size: Kernel size for depth-wise conv projections.
            use_pos_embed: Whether to add learned positional embeddings.
                The CvT paper claims they can be safely removed.
            dropout: Dropout probability.

        Raises:
            ValueError: If ``img_size`` is not divisible by ``patch_size``.
        """
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

        self.mask_ratio = mask_ratio
        self.use_block_masking = use_block_masking
        self.block_size = block_size
        self.decoder_num_heads = decoder_num_heads

        # Learnable mask token in the embedding space
        self.encoder_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---- Encoder ---------------------------------------------------
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        else:
            self.register_buffer(
                "pos_embed", torch.zeros(1, self.num_patches, embed_dim)
            )

        self.encoder_blocks = nn.ModuleList(
            [
                CvTBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    cvt_kernel_size,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ---- Decoder (lightweight, standard Transformer) ---------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise model parameters."""
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.encoder_mask_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
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

    def unpatchify(self, x: Tensor) -> Tensor:
        """Reassemble patches into an image tensor.

        Args:
            x: Patches of shape ``(B, N, patch_dim)``.

        Returns:
            Image tensor of shape ``(B, C, H, W)``.
        """
        B, N, _ = x.shape
        p = self.patch_size
        n = self.img_size // p
        C = self.in_channels
        x = x.reshape(B, n, n, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, C, self.img_size, self.img_size)
        return x

    def _generate_random_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate a random per-sample mask.

        Args:
            batch_size: Number of samples in the batch.
            num_patches: Total number of patches.
            device: Torch device.

        Returns:
            A tuple of ``(ids_keep, mask, ids_restore)``.
        """
        len_keep = int(num_patches * (1 - self.mask_ratio))
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones(batch_size, num_patches, device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def _generate_block_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate a block-wise mask.

        Args:
            batch_size: Number of samples in the batch.
            num_patches: Total number of patches (must be a perfect square).
            device: Torch device.

        Returns:
            A tuple of ``(ids_keep, mask, ids_restore)``.

        Raises:
            ValueError: If ``num_patches`` is not a perfect square or if
                ``block_size`` does not divide ``sqrt(num_patches)``.
        """
        sqrt_N = math.isqrt(num_patches)
        if sqrt_N * sqrt_N != num_patches:
            raise ValueError(f"num_patches={num_patches} must be a perfect square")
        if sqrt_N % self.block_size != 0:
            raise ValueError(
                f"block_size={self.block_size} must divide sqrt_N={sqrt_N}"
            )

        n_blocks = sqrt_N // self.block_size
        num_blocks_total = n_blocks * n_blocks
        patches_per_block = self.block_size * self.block_size
        len_keep_blocks = int(num_blocks_total * (1 - self.mask_ratio))

        noise = torch.rand(batch_size, num_blocks_total, device=device)
        block_ids_shuffle = torch.argsort(noise, dim=1)

        # Build patch-level shuffle and restore indices
        patch_grid = torch.arange(num_patches, device=device).reshape(sqrt_N, sqrt_N)
        patch_grid_blocks = patch_grid.reshape(
            n_blocks, self.block_size, n_blocks, self.block_size
        )
        patch_grid_blocks = patch_grid_blocks.permute(0, 2, 1, 3).reshape(
            num_blocks_total, patches_per_block
        )
        patch_grid_blocks = patch_grid_blocks.unsqueeze(0).expand(batch_size, -1, -1)

        patch_shuffled = torch.gather(
            patch_grid_blocks,
            dim=1,
            index=block_ids_shuffle.unsqueeze(-1).expand(-1, -1, patches_per_block),
        )
        ids_shuffle_flat = patch_shuffled.reshape(batch_size, num_patches)
        ids_restore = torch.argsort(ids_shuffle_flat, dim=1)

        ids_keep = ids_shuffle_flat[:, : len_keep_blocks * patches_per_block]

        mask = torch.ones(batch_size, num_patches, device=device)
        mask[:, : len_keep_blocks * patches_per_block] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward_encoder(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode an image batch with spatial masking.

        Masked patches are replaced with a learnable mask token before
        being fed to the CvT encoder, which processes the full token grid.
        After encoding, only the visible tokens are gathered for the
        decoder.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            A tuple of:
            - **latent**: Encoded visible tokens of shape
              ``(B, N_keep, embed_dim)``.
            - **mask**: Binary mask of shape ``(B, N)`` (``1`` = masked).
            - **ids_restore**: Restoration indices of shape ``(B, N)``.
        """
        # 1. Patchify and embed
        x = self.patchify(x)  # (B, N, patch_dim)
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # 2. Generate mask
        if self.use_block_masking:
            ids_keep, mask, ids_restore = self._generate_block_mask(
                x.shape[0], x.shape[1], x.device
            )
        else:
            ids_keep, mask, ids_restore = self._generate_random_mask(
                x.shape[0], x.shape[1], x.device
            )

        # 3. Replace masked positions with learnable mask token
        mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
        x = x * (1 - mask_expanded) + self.encoder_mask_token * mask_expanded

        # 4. Add positional embeddings
        x = x + self.pos_embed  # (B, N, embed_dim)

        # 5. CvT encoder (full grid)
        h = w = self.img_size // self.patch_size
        for block in self.encoder_blocks:
            x = block(x, h, w)
        x = self.encoder_norm(x)  # (B, N, embed_dim)

        # 6. Gather visible tokens for decoder
        x_visible = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]),
        )

        return x_visible, mask, ids_restore

    def forward_decoder(self, x: Tensor, ids_restore: Tensor) -> Tensor:
        """Decode latent tokens back to patch predictions.

        This is the standard MAE decoder: visible tokens are projected,
        mask tokens are appended, and the sequence is restored to the
        original spatial order before running lightweight Transformer
        blocks.

        Args:
            x: Encoded visible tokens of shape ``(B, N_keep, embed_dim)``.
            ids_restore: Restoration indices of shape ``(B, N)``.

        Returns:
            Predicted patches of shape ``(B, N, patch_dim)``.
        """
        x = self.decoder_embed(x)  # (B, N_keep, decoder_embed_dim)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = torch.cat([x, mask_tokens], dim=1)  # (B, N, dec_dim)

        # Restore original order
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]),
        )

        # Add decoder positional embeddings
        x_ = x_ + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x_ = block(x_)
        x_ = self.decoder_norm(x_)
        x_ = self.decoder_pred(x_)  # (B, N, patch_dim)

        return x_

    def forward_loss(self, imgs: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        """Compute MSE reconstruction loss on masked patches only.

        Args:
            imgs: Ground-truth images of shape ``(B, C, H, W)``.
            pred: Predicted patches of shape ``(B, N, patch_dim)``.
            mask: Binary mask of shape ``(B, N)`` where ``1`` denotes
                a masked patch.

        Returns:
            Scalar loss tensor.
        """
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / (mask.sum() + 1e-5)
        return loss

    def forward(self, imgs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass: encode, decode, and compute loss.

        Args:
            imgs: Input images of shape ``(B, C, H, W)``.

        Returns:
            A tuple of:
            - **loss**: Scalar MSE loss on masked patches.
            - **pred**: Predicted patches of shape ``(B, N, patch_dim)``.
            - **mask**: Binary mask of shape ``(B, N)``.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def extract_embeddings(self, imgs: Tensor) -> Tensor:
        """Extract mean-pooled encoder embeddings without masking.

        Runs the full patch embedding + CvT encoder stack on *all*
        patches and returns the mean across the patch dimension.

        Args:
            imgs: Input images of shape ``(B, C, H, W)``.

        Returns:
            Embedding tensor of shape ``(B, embed_dim)``.
        """
        x = self.patchify(imgs)  # (B, N, patch_dim)
        x = self.patch_embed(x)  # (B, N, embed_dim)
        x = x + self.pos_embed  # (B, N, embed_dim)

        h = w = self.img_size // self.patch_size
        for block in self.encoder_blocks:
            x = block(x, h, w)
        x = self.encoder_norm(x)  # (B, N, embed_dim)

        return x.mean(dim=1)  # (B, embed_dim)
