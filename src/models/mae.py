from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


class TransformerBlock(nn.Module):
    """A single pre-norm Transformer encoder block.

    Uses multi-head self-attention followed by an MLP, both with
    residual connections and layer normalisation applied before the
    sub-layer (pre-norm).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the block.

        Args:
            embed_dim: Token embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, N, D)``.

        Returns:
            Output tensor of shape ``(B, N, D)``.
        """
        x_norm = self.norm1(x)
        x = (
            x
            + self.attn(
                x_norm,
                x_norm,
                x_norm,
                need_weights=False,
            )[0]
        )
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder (MAE) for self-supervised representation learning.

    The model follows the standard MAE architecture: a ViT-style encoder
    that processes only unmasked patch tokens, and a lightweight decoder
    that reconstructs the full set of patches from the latent
    representation plus learnable mask tokens.

    Both random and block-wise masking strategies are supported.
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
        dropout: float = 0.0,
    ) -> None:
        """Initialise the MAE.

        Args:
            img_size: Spatial resolution of input images (square).
            patch_size: Size of each square patch.
            in_channels: Number of input channels.
            embed_dim: Encoder embedding dimension.
            depth: Number of encoder Transformer blocks.
            num_heads: Number of attention heads in the encoder.
            mlp_ratio: MLP hidden dim ratio.
            decoder_embed_dim: Decoder embedding dimension.
            decoder_depth: Number of decoder Transformer blocks.
            decoder_num_heads: Number of attention heads in the decoder.
            mask_ratio: Fraction of patches to mask.
            use_block_masking: If ``True``, use block-wise masking;
                otherwise random masking.
            block_size: Side length of masking blocks (in patch units).
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

        # ---- Encoder ---------------------------------------------------
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ---- Decoder ---------------------------------------------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_embed_dim, decoder_num_heads, mlp_ratio, dropout
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise model parameters."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

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
            Patches of shape ``(B, N, patch_dim)`` where
            ``N = (H / p) * (W / p)`` and ``patch_dim = C * p * p``.
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

    def random_masking(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply random per-token masking.

        Args:
            x: Token tensor of shape ``(B, N, D)``.

        Returns:
            A tuple of:
            - **x_masked**: Visible tokens of shape
              ``(B, N_keep, D)``.
            - **mask**: Binary mask of shape ``(B, N)`` where ``1``
              denotes a masked (removed) token.
            - **ids_restore**: Index tensor of shape ``(B, N)`` used
              to restore the original token order.
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def block_masking(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply block-wise masking.

        Masks contiguous 2-D blocks of tokens rather than individual
        tokens.  This preserves more local structure in the visible
        context.

        Args:
            x: Token tensor of shape ``(B, N, D)``.  ``N`` must be a
                perfect square.

        Returns:
            Same return signature as :meth:`random_masking`.

        Raises:
            ValueError: If ``N`` is not a perfect square or if
                ``block_size`` does not divide ``sqrt(N)``.
        """
        B, N, D = x.shape
        sqrt_N = math.isqrt(N)
        if sqrt_N * sqrt_N != N:
            raise ValueError(f"N={N} must be a perfect square")
        if sqrt_N % self.block_size != 0:
            raise ValueError(
                f"block_size={self.block_size} must divide sqrt_N={sqrt_N}"
            )

        n_blocks = sqrt_N // self.block_size
        num_blocks = n_blocks * n_blocks
        patches_per_block = self.block_size * self.block_size

        # Reshape tokens into a 2-D spatial grid, then into blocks
        x_2d = x.reshape(B, sqrt_N, sqrt_N, D)
        x_blocks = x_2d.reshape(
            B, n_blocks, self.block_size, n_blocks, self.block_size, D
        )
        x_blocks = x_blocks.permute(0, 1, 3, 2, 4, 5)
        x_blocks = x_blocks.reshape(B, num_blocks, patches_per_block, D)

        # Shuffle blocks
        len_keep_blocks = int(num_blocks * (1 - self.mask_ratio))
        noise = torch.rand(B, num_blocks, device=x.device)
        block_ids_shuffle = torch.argsort(noise, dim=1)

        # Gather kept blocks
        block_ids_keep = block_ids_shuffle[:, :len_keep_blocks]
        x_kept = torch.gather(
            x_blocks,
            dim=1,
            index=block_ids_keep.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, patches_per_block, D),
        )
        x_masked = x_kept.reshape(B, len_keep_blocks * patches_per_block, D)

        # ids_restore at patch level
        patch_grid = torch.arange(N, device=x.device).reshape(sqrt_N, sqrt_N)
        patch_grid_blocks = patch_grid.reshape(
            n_blocks, self.block_size, n_blocks, self.block_size
        )
        patch_grid_blocks = patch_grid_blocks.permute(0, 2, 1, 3).reshape(
            num_blocks, patches_per_block
        )
        patch_grid_blocks = patch_grid_blocks.unsqueeze(0).expand(B, -1, -1)

        patch_shuffled = torch.gather(
            patch_grid_blocks,
            dim=1,
            index=block_ids_shuffle.unsqueeze(-1).expand(-1, -1, patches_per_block),
        )
        ids_shuffle_flat = patch_shuffled.reshape(B, N)
        ids_restore = torch.argsort(ids_shuffle_flat, dim=1)

        # Mask: 1 = masked, 0 = visible
        mask = torch.ones(B, N, device=x.device)
        mask[:, : len_keep_blocks * patches_per_block] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode an image batch, masking patches in the process.

        Args:
            x: Image tensor of shape ``(B, C, H, W)``.

        Returns:
            A tuple of:
            - **latent**: Encoded visible tokens of shape
              ``(B, N_keep, embed_dim)``.
            - **mask**: Binary mask of shape ``(B, N)``.
            - **ids_restore**: Restoration indices of shape ``(B, N)``.
        """
        x = self.patchify(x)  # (B, N, patch_dim)
        x = self.patch_embed(x)  # (B, N, embed_dim)
        x = x + self.pos_embed  # (B, N, embed_dim)

        if self.use_block_masking:
            x_masked, mask, ids_restore = self.block_masking(x)
        else:
            x_masked, mask, ids_restore = self.random_masking(x)

        for block in self.encoder_blocks:
            x_masked = block(x_masked)
        latent = self.encoder_norm(x_masked)

        return latent, mask, ids_restore

    def forward_decoder(self, x: Tensor, ids_restore: Tensor) -> Tensor:
        """Decode latent tokens back to patch predictions.

        Args:
            x: Encoded visible tokens of shape
                ``(B, N_keep, embed_dim)``.
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

        Runs the full patch embedding + positional embedding + encoder
        stack on *all* patches (no masking) and returns the mean across
        the patch dimension.

        Args:
            imgs: Input images of shape ``(B, C, H, W)``.

        Returns:
            Embedding tensor of shape ``(B, embed_dim)``.
        """
        x = self.patchify(imgs)  # (B, N, patch_dim)
        x = self.patch_embed(x)  # (B, N, embed_dim)
        x = x + self.pos_embed  # (B, N, embed_dim)

        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)  # (B, N, embed_dim)

        return x.mean(dim=1)  # (B, embed_dim)
