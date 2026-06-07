# Model Change Log

Accumulated architecture–metric history for the swami-picker project.
Every modification to model architecture, loss formulation, or data augmentation must be logged here.

## Log Format

| Date | Model Version | Architecture Delta | Baseline Metric | New Metric | Metric Delta |
|------|---------------|--------------------|-----------------|------------|--------------|
| YYYY-MM-DD | git-short-hash or semantic | Brief description of what changed | Previous best metric value | Metric after change | New − Baseline |

## Entries

| 2026-06-05 | phase0-mnist-fast-decoder | batch_size=32, accum=1 (no grad-accum overhead); decoder simplified: depth=2, num_heads=4 (was 8, 8) | val_loss=0.271, Silhouette=0.18, contrast=3.701 | val_loss=0.266, Silhouette=0.168, contrast=2.959 | val_loss=-0.005, Silhouette=-0.012, contrast=-0.742 |
| 2026-06-05 | phase0-mnist-mae-small | MNIST MAE smoke test: ViT-small encoder (embed_dim=384, depth=12, num_heads=6), lightweight decoder (depth=4, embed_dim=256), block masking 75%, 5 epochs | N/A (initial run) | val_loss=0.271, VRAM=777MB, Silhouette=0.18 | N/A |

| 2026-06-05 | phase0-mnist-cvt-mae | CvT-MAE hybrid: ViT-small encoder replaced with CvT blocks (depth-wise conv projections for Q/K/V, kernel_size=3). Encoder processes full token grid with learnable mask tokens for masked positions; only visible tokens passed to decoder. Block masking 75%, 5 epochs. | val_loss=0.271, Silhouette=0.18, contrast=3.701 (ViT-MAE baseline) | val_loss=0.198, Silhouette=0.032, contrast=1.072, VRAM=2047MB | val_loss=-0.073, Silhouette=-0.148, contrast=-2.629 |

| 2026-06-07 | phase2-fk-mae-v1 | Phase 0 ViT-MAE transferred to FK data. Block masking 75%, Gaussian noise (std=0.01) + intensity jitter (±15%). Val split: 120 phase-1 val + 10% random from train (~247 total). Epochs=30. | N/A | val_loss=0.084, Silhouette=−0.322, contrast=1.078 | ❌ Embedding collapse |
| 2026-06-07 | phase2-fk-mae-v2 | Random masking 50%, same augmentation as v1. Epochs=30. | v1: val_loss=0.084, Silhouette=−0.322, contrast=1.078 | val_loss=0.075, Silhouette≈−0.3, contrast≈1.08 | val_loss −0.009, Silhouette no change — still collapsed |
| 2026-06-07 | phase2-fk-mae-v3 | Aggressive aug (noise=0.15, jitter=0.50, freq/waven shift, band dropout), random masking 25%, 100 epochs planned (stopped at 54), min_lr=1e-6. | v2: val_loss=0.075, collapse | val_loss=0.095 (epoch 54), still collapsed | ❌ MAE fundamentally unsuitable for FK spectra |
| 2026-06-07 | phase2-vicreg-v1 | (Planned) VICReg self-supervised learning. ViT-Small encoder (no decoder) + projector MLP. Same aggressive augmentations. | N/A | TBD | N/A |

<!-- Append new entries above this line -->
