from __future__ import annotations

import math

import pytest
import torch

from src.models.mae import MaskedAutoencoder


class TestPatchifyUnpatchify:
    def test_patchify_unpatchify_roundtrip_1ch(
        self, small_model: MaskedAutoencoder
    ) -> None:
        """1-channel patchify/unpatchify is an exact inverse."""
        model = small_model
        dummy = torch.randn(2, model.in_channels, model.img_size, model.img_size)
        patches = model.patchify(dummy)
        reconstructed = model.unpatchify(patches)
        assert torch.allclose(dummy, reconstructed, atol=1e-6)

    def test_patchify_unpatchify_roundtrip_3ch(self) -> None:
        """3-channel patchify/unpatchify is an exact inverse."""
        model = MaskedAutoencoder(
            img_size=64,
            patch_size=16,
            in_channels=3,
            embed_dim=64,
            depth=2,
            num_heads=4,
            decoder_embed_dim=32,
            decoder_depth=1,
            decoder_num_heads=4,
        )
        dummy = torch.randn(2, 3, 64, 64)
        patches = model.patchify(dummy)
        reconstructed = model.unpatchify(patches)
        assert torch.allclose(dummy, reconstructed, atol=1e-6)

    def test_patchify_unpatchify_roundtrip_min_size(self) -> None:
        """Patchify/unpatchify works for the minimum 32x32 image size."""
        model = MaskedAutoencoder(
            img_size=32,
            patch_size=16,
            in_channels=1,
            embed_dim=64,
            depth=2,
            num_heads=4,
            decoder_embed_dim=32,
            decoder_depth=1,
            decoder_num_heads=4,
        )
        dummy = torch.randn(1, 1, 32, 32)
        patches = model.patchify(dummy)
        reconstructed = model.unpatchify(patches)
        assert torch.allclose(dummy, reconstructed, atol=1e-6)


class TestRandomMasking:
    def test_random_masking_invariant(self, small_model: MaskedAutoencoder) -> None:
        """mask + (ids_restore < N_keep) == 1 for every position."""
        model = small_model
        model.use_block_masking = False
        B, N = 8, model.num_patches
        D = model.patch_embed.out_features
        x = torch.randn(B, N, D)

        x_masked, mask, ids_restore = model.random_masking(x)

        N_keep = x_masked.shape[1]
        kept_flags = (ids_restore < N_keep).float()
        invariant = mask + kept_flags
        assert torch.allclose(invariant, torch.ones_like(invariant))

    def test_random_masking_ratio(self, medium_model: MaskedAutoencoder) -> None:
        """Masked fraction is within ±2% of the configured mask_ratio."""
        model = medium_model
        model.use_block_masking = False
        B, N = 32, model.num_patches
        D = model.patch_embed.out_features
        x = torch.randn(B, N, D)

        _, mask, _ = model.random_masking(x)
        masked_ratio = mask.sum().item() / mask.numel()
        assert abs(masked_ratio - model.mask_ratio) < 0.02

    def test_random_masking_ids_restore_order(
        self, small_model: MaskedAutoencoder
    ) -> None:
        """Gathering shuffled tokens by ids_restore recovers the original order."""
        model = small_model
        model.use_block_masking = False
        B, N = 4, model.num_patches
        D = model.patch_embed.out_features
        x = torch.randn(B, N, D)

        _, _, ids_restore = model.random_masking(x)

        # ids_restore is the inverse permutation of ids_shuffle.
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        token_ids = (
            torch.arange(N, device=x.device).unsqueeze(-1).unsqueeze(0).expand(B, -1, D)
        )
        shuffled = torch.gather(
            token_ids, dim=1, index=ids_shuffle.unsqueeze(-1).expand(-1, -1, D)
        )
        restored = torch.gather(
            shuffled, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )
        assert torch.equal(restored, token_ids)


class TestBlockMasking:
    def test_block_masking_invariant(self, small_model: MaskedAutoencoder) -> None:
        """mask + (ids_restore < N_keep) == 1 for every position under block masking."""
        model = small_model
        model.use_block_masking = True
        B, N = 8, model.num_patches
        D = model.patch_embed.out_features
        x = torch.randn(B, N, D)

        x_masked, mask, ids_restore = model.block_masking(x)

        N_keep = x_masked.shape[1]
        kept_flags = (ids_restore < N_keep).float()
        invariant = mask + kept_flags
        assert torch.allclose(invariant, torch.ones_like(invariant))

    def test_block_masking_contiguous_blocks(
        self, small_model: MaskedAutoencoder
    ) -> None:
        """Masked patches form contiguous blocks rather than scattered tokens."""
        model = small_model
        model.use_block_masking = True
        B, N = 4, model.num_patches
        D = model.patch_embed.out_features
        x = torch.randn(B, N, D)

        _, mask, _ = model.block_masking(x)

        sqrt_N = int(math.isqrt(N))
        n_blocks = sqrt_N // model.block_size
        mask_2d = mask.reshape(B, sqrt_N, sqrt_N)
        mask_blocks = mask_2d.reshape(
            B, n_blocks, model.block_size, n_blocks, model.block_size
        )
        mask_blocks = mask_blocks.permute(0, 1, 3, 2, 4)
        block_var = mask_blocks.reshape(B, n_blocks, n_blocks, -1).var(dim=-1)
        assert torch.allclose(block_var, torch.zeros_like(block_var), atol=1e-6)

    def test_block_masking_ratio(self, medium_model: MaskedAutoencoder) -> None:
        """Masked fraction under block masking is within ±2% of config."""
        model = medium_model
        model.use_block_masking = True
        B, N = 32, model.num_patches
        D = model.patch_embed.out_features
        x = torch.randn(B, N, D)

        _, mask, _ = model.block_masking(x)
        masked_ratio = mask.sum().item() / mask.numel()
        assert abs(masked_ratio - model.mask_ratio) < 0.02

    @pytest.mark.parametrize("block_size", [2, 4])
    def test_block_masking_various_block_sizes(self, block_size: int) -> None:
        """Block masking is correct for multiple block_size values."""
        model = MaskedAutoencoder(
            img_size=128,
            patch_size=16,
            in_channels=1,
            embed_dim=64,
            depth=2,
            num_heads=4,
            decoder_embed_dim=32,
            decoder_depth=1,
            decoder_num_heads=4,
            mask_ratio=0.75,
            use_block_masking=True,
            block_size=block_size,
        )
        B, N = 4, model.num_patches
        D = model.patch_embed.out_features
        x = torch.randn(B, N, D)

        x_masked, mask, ids_restore = model.block_masking(x)

        # Invariant
        N_keep = x_masked.shape[1]
        kept_flags = (ids_restore < N_keep).float()
        invariant = mask + kept_flags
        assert torch.allclose(invariant, torch.ones_like(invariant))

        # Contiguous blocks
        sqrt_N = int(math.isqrt(N))
        n_blocks = sqrt_N // block_size
        mask_2d = mask.reshape(B, sqrt_N, sqrt_N)
        mask_blocks = mask_2d.reshape(B, n_blocks, block_size, n_blocks, block_size)
        mask_blocks = mask_blocks.permute(0, 1, 3, 2, 4)
        block_var = mask_blocks.reshape(B, n_blocks, n_blocks, -1).var(dim=-1)
        assert torch.allclose(block_var, torch.zeros_like(block_var), atol=1e-6)

        # Ratio
        masked_ratio = mask.sum().item() / mask.numel()
        assert abs(masked_ratio - model.mask_ratio) < 0.02


class TestForwardLoss:
    def test_forward_loss_zero_when_perfect(
        self, small_model: MaskedAutoencoder
    ) -> None:
        """Loss is exactly zero when predictions match targets and mask is all-ones."""
        model = small_model
        imgs = torch.randn(2, model.in_channels, model.img_size, model.img_size)
        target = model.patchify(imgs)
        mask = torch.ones(2, model.num_patches)
        loss = model.forward_loss(imgs, target, mask)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_forward_loss_nonzero_when_noisy(
        self, small_model: MaskedAutoencoder
    ) -> None:
        """Loss is strictly positive for noisy predictions."""
        model = small_model
        imgs = torch.randn(2, model.in_channels, model.img_size, model.img_size)
        target = model.patchify(imgs)
        pred = target + torch.randn_like(target) * 0.5
        mask = torch.ones(2, model.num_patches)
        loss = model.forward_loss(imgs, pred, mask)
        assert loss.item() > 0

    def test_forward_loss_no_division_by_zero_all_zeros_mask(
        self, small_model: MaskedAutoencoder
    ) -> None:
        """No division by zero when mask is all-zeros; loss is finite and zero."""
        model = small_model
        imgs = torch.randn(2, model.in_channels, model.img_size, model.img_size)
        target = model.patchify(imgs)
        mask = torch.zeros(2, model.num_patches)
        loss = model.forward_loss(imgs, target, mask)
        assert torch.isfinite(loss)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestModelForward:
    def test_forward_output_shapes(self, small_model: MaskedAutoencoder) -> None:
        """The forward method returns a scalar loss and tensors of expected shapes."""
        model = small_model
        model.eval()
        B = 3
        imgs = torch.randn(B, model.in_channels, model.img_size, model.img_size)
        loss, pred, mask = model(imgs)

        assert loss.ndim == 0
        expected_n_patches = (model.img_size // model.patch_size) ** 2
        expected_patch_dim = model.in_channels * model.patch_size * model.patch_size
        assert tuple(pred.shape) == (B, expected_n_patches, expected_patch_dim)
        assert tuple(mask.shape) == (B, expected_n_patches)

    def test_forward_encoder_processes_fewer_tokens(
        self, small_model: MaskedAutoencoder
    ) -> None:
        """The encoder latent contains fewer tokens than the total number of patches."""
        model = small_model
        imgs = torch.randn(2, model.in_channels, model.img_size, model.img_size)
        latent, mask, ids_restore = model.forward_encoder(imgs)
        assert latent.shape[1] < model.num_patches

    def test_forward_backward_no_nan_inf(self, small_model: MaskedAutoencoder) -> None:
        """Gradients flow through the full model without NaN or Inf."""
        model = small_model
        model.train()
        imgs = torch.randn(2, model.in_channels, model.img_size, model.img_size)
        loss, pred, mask = model(imgs)
        assert torch.isfinite(loss)
        loss.backward()
        has_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                has_grad = True
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
        assert has_grad, "No parameters received gradients"


class TestArchitectureValidation:
    def test_init_raises_when_img_size_not_divisible_by_patch_size(self) -> None:
        """Constructor raises ValueError when img_size is not divisible by patch_size."""
        with pytest.raises(ValueError, match="must be divisible by patch_size"):
            MaskedAutoencoder(img_size=100, patch_size=16)

    def test_block_masking_raises_when_num_patches_not_perfect_square(
        self, small_model: MaskedAutoencoder
    ) -> None:
        """block_masking raises ValueError when token count is not a perfect square."""
        model = small_model
        x = torch.randn(2, 15, 128)  # 15 is not a perfect square
        with pytest.raises(ValueError, match="must be a perfect square"):
            model.block_masking(x)

    def test_block_masking_raises_when_block_size_does_not_divide_sqrt(
        self, small_model: MaskedAutoencoder
    ) -> None:
        """block_masking raises ValueError when block_size does not divide sqrt(N)."""
        model = small_model
        original = model.block_size
        model.block_size = 3  # sqrt(16) = 4, 3 does not divide 4
        x = torch.randn(2, 16, 128)
        with pytest.raises(ValueError, match="must divide sqrt_N"):
            model.block_masking(x)
        model.block_size = original
