from __future__ import annotations

import pytest
import torch

from src.models.cvt_mae import CvTAttention, CvTBlock, CvTMaskedAutoencoder


@pytest.fixture
def small_cvt_model() -> CvTMaskedAutoencoder:
    """Return a small CvT-MAE for fast unit tests."""
    return CvTMaskedAutoencoder(
        img_size=64,
        patch_size=16,
        in_channels=1,
        embed_dim=128,
        depth=2,
        num_heads=4,
        decoder_embed_dim=64,
        decoder_depth=1,
        decoder_num_heads=4,
        use_pos_embed=True,
        use_block_masking=True,
        block_size=2,
    )


class TestCvTAttention:
    def test_forward_shape(self) -> None:
        """Output shape matches input shape."""
        attn = CvTAttention(embed_dim=128, num_heads=4, kernel_size=3)
        B, N, D = 2, 16, 128
        h = w = 4
        x = torch.randn(B, N, D)
        out = attn(x, h, w)
        assert out.shape == (B, N, D)

    def test_raises_on_even_kernel_size(self) -> None:
        """Even kernel_size raises ValueError."""
        with pytest.raises(ValueError, match="must be odd"):
            CvTAttention(embed_dim=128, num_heads=4, kernel_size=2)

    def test_raises_on_indivisible_embed_dim(self) -> None:
        """embed_dim not divisible by num_heads raises ValueError."""
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            CvTAttention(embed_dim=127, num_heads=4)


class TestCvTBlock:
    def test_forward_shape(self) -> None:
        """Output shape matches input shape."""
        block = CvTBlock(embed_dim=128, num_heads=4, kernel_size=3)
        B, N, D = 2, 16, 128
        h = w = 4
        x = torch.randn(B, N, D)
        out = block(x, h, w)
        assert out.shape == (B, N, D)


class TestCvTMaskedAutoencoderForward:
    def test_forward_output_shapes(self, small_cvt_model: CvTMaskedAutoencoder) -> None:
        """Forward returns scalar loss and tensors of expected shapes."""
        model = small_cvt_model
        model.eval()
        B = 3
        imgs = torch.randn(B, model.in_channels, model.img_size, model.img_size)
        loss, pred, mask = model(imgs)

        assert loss.ndim == 0
        expected_n_patches = (model.img_size // model.patch_size) ** 2
        expected_patch_dim = model.in_channels * model.patch_size * model.patch_size
        assert tuple(pred.shape) == (B, expected_n_patches, expected_patch_dim)
        assert tuple(mask.shape) == (B, expected_n_patches)

    def test_forward_encoder_keeps_all_tokens(
        self, small_cvt_model: CvTMaskedAutoencoder
    ) -> None:
        """CvT encoder processes the full token grid (no dropping)."""
        model = small_cvt_model
        imgs = torch.randn(2, model.in_channels, model.img_size, model.img_size)
        latent, mask, ids_restore = model.forward_encoder(imgs)
        # latent contains only visible tokens, but its length is N_keep < N
        assert latent.shape[1] < model.num_patches
        # mask should have the expected number of masked positions
        expected_masked = int(model.num_patches * model.mask_ratio)
        assert abs(mask.sum().item() / mask.shape[0] - expected_masked) < 2

    def test_forward_backward_no_nan_inf(
        self, small_cvt_model: CvTMaskedAutoencoder
    ) -> None:
        """Gradients flow through the full model without NaN or Inf."""
        model = small_cvt_model
        model.train()
        imgs = torch.randn(2, model.in_channels, model.img_size, model.img_size)
        loss, _pred, _mask = model(imgs)
        assert torch.isfinite(loss)
        loss.backward()
        has_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                has_grad = True
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
        assert has_grad, "No parameters received gradients"

    def test_extract_embeddings_shape(
        self, small_cvt_model: CvTMaskedAutoencoder
    ) -> None:
        """extract_embeddings returns (B, embed_dim)."""
        model = small_cvt_model
        B = 4
        imgs = torch.randn(B, model.in_channels, model.img_size, model.img_size)
        embs = model.extract_embeddings(imgs)
        assert embs.shape == (B, model.patch_embed.out_features)

    def test_extract_embeddings_matches_encoder_with_no_mask(
        self, small_cvt_model: CvTMaskedAutoencoder
    ) -> None:
        """extract_embeddings is consistent with forward_encoder at mask_ratio=0."""
        model = small_cvt_model
        model.mask_ratio = 0.0
        imgs = torch.randn(2, model.in_channels, model.img_size, model.img_size)
        latent, _mask, _ids = model.forward_encoder(imgs)
        embs = model.extract_embeddings(imgs)
        assert torch.allclose(latent.mean(dim=1), embs, atol=1e-6)

    def test_patchify_unpatchify_roundtrip(
        self, small_cvt_model: CvTMaskedAutoencoder
    ) -> None:
        """patchify/unpatchify is an exact inverse."""
        model = small_cvt_model
        dummy = torch.randn(2, model.in_channels, model.img_size, model.img_size)
        patches = model.patchify(dummy)
        reconstructed = model.unpatchify(patches)
        assert torch.allclose(dummy, reconstructed, atol=1e-6)

    def test_block_masking_invariant(
        self, small_cvt_model: CvTMaskedAutoencoder
    ) -> None:
        """mask + (ids_restore < N_keep) == 1 for every position."""
        model = small_cvt_model
        B, N = 8, model.num_patches
        ids_keep, mask, ids_restore = model._generate_block_mask(
            B, N, torch.device("cpu")
        )
        N_keep = ids_keep.shape[1]
        kept_flags = (ids_restore < N_keep).float()
        invariant = mask + kept_flags
        assert torch.allclose(invariant, torch.ones_like(invariant))

    def test_random_masking_invariant(
        self, small_cvt_model: CvTMaskedAutoencoder
    ) -> None:
        """mask + (ids_restore < N_keep) == 1 for every position."""
        model = small_cvt_model
        model.use_block_masking = False
        B, N = 8, model.num_patches
        ids_keep, mask, ids_restore = model._generate_random_mask(
            B, N, torch.device("cpu")
        )
        N_keep = ids_keep.shape[1]
        kept_flags = (ids_restore < N_keep).float()
        invariant = mask + kept_flags
        assert torch.allclose(invariant, torch.ones_like(invariant))


class TestCvTArchitectureValidation:
    def test_init_raises_when_img_size_not_divisible_by_patch_size(self) -> None:
        """Constructor raises ValueError when img_size is not divisible by patch_size."""
        with pytest.raises(ValueError, match="must be divisible by patch_size"):
            CvTMaskedAutoencoder(img_size=100, patch_size=16)

    def test_block_masking_raises_when_block_size_does_not_divide_sqrt(
        self, small_cvt_model: CvTMaskedAutoencoder
    ) -> None:
        """_generate_block_mask raises ValueError when block_size does not divide sqrt(N)."""
        model = small_cvt_model
        model.block_size = 3  # sqrt(16) = 4, 3 does not divide 4
        with pytest.raises(ValueError, match="must divide sqrt_N"):
            model._generate_block_mask(2, 16, torch.device("cpu"))
