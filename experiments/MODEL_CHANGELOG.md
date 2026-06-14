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
| 2026-06-07 | phase2-vicreg-v1 | VICReg self-supervised learning. ViT-Small encoder (no decoder) + projector MLP (2048-d). Loss: λ=25, µ=25, ν=1. Batch=16, LR=3e-4, 50 epochs. Same aggressive augmentations. | N/A | Silhouette=−0.252, contrast=1.036, loss=37.0 | ❌ Fuzzy-ball collapse — variance hinge still active after 50 epochs |

| 2026-06-07 | phase2c-clustering-v1 | Option C clustering pipeline: spectral descriptors (20-D) → UMAP(5D, md=0.0) → HDBSCAN. Two-step hierarchical clustering: initial 5 clusters → re-cluster dominant cluster (940 spectra) → 11 merged clusters (12% noise). | N/A (first successful clustering) | Silhouette=0.60 (step 1), 0.46 (step 2), 11 clusters, noise=12% | ✅ First successful FK spectrum clustering — pseudo-labels ready for Stage-1 training |

| 2026-06-13 | phase4-picking-v2 | Refactored to single 257-class head (256 wavenumber bins + 1 absent class). Compact U-Net: base_channels=8, embed_dim=64, dropout=0.3 (~0.59M params). K-fold CV (5 folds). Smoothed val RMSE checkpoint selection (5-epoch moving average). Grayscale probability heatmaps without pick overlays. | N/A | TBD | N/A |
| 2026-06-13 | phase4-picking-v2.1 | Final architecture: base_channels=16, embed_dim=64, dropout=0.5 (~2.3M params). Added expected-value frequency-axis smoothness loss (weight=0.05). Disabled pick-synchronized shifts (too strong for 150 training spectra). Added min_val_coverage safeguard for checkpoint selection. Final run `phase4-picking-v2-final`: best val RMSE=3.46 px, smoothed=3.77 px, val F1=0.934, coverage=0.468, no mode jumps. | v2: best val RMSE=6.73 px, smoothed=6.93 px, F1=0.921 | best val RMSE=3.46 px, smoothed=3.77 px, F1=0.934 | val RMSE=-3.27 px, smoothed=-3.16 px, F1=+0.013 |

| 2026-06-14 | phase5-coordinate-transform-v1 | Implemented matched forward/inverse coordinate transform pair in `src/transforms/coordinates.py`. Supports Hz/1/m conversion with first-order uncertainty propagation from pick certainty, inference-to-annotation bridge, relative smoothness quality score, and DataFrame export. 33 unit/integration tests passing. | N/A | round-trip wavenumber RMSE < 0.5 px on linear axes; real-metadata integration verified | ✅ Coordinate-transform infrastructure ready |
| 2026-06-14 | phase4-inference-v1 | Implemented `scripts/phase4_picking/run_inference.py`. Runs trained v2.1 model over all 1,392 spectra, saves `predictions.npz` with picks + presence probabilities, generates `quality_scores.json` and `low_quality_spectra.json`, and optionally exports annotation records for review in the picking app. | N/A | 1,392 spectra in ~8.8 s (~158 spectra/s); composite score mean=0.853, range=[0.729, 0.950] | ✅ Full-dataset inference complete |

| 2026-06-14 | phase4-picking-seq-bilstm-v1 | **New default for Phase 4 picking.** Added `SeqPickingModel`: U-Net decoder output reshaped to `(B, C*H, W)` and processed by a 2-layer BiLSTM (`seq_hidden_dim=128`) with a residual skip. Updated `configs/phase4_picking.yaml` and `PickingConfig.model_type` to default to `seq`/`bilstm`. | v2.1: val RMSE=3.46 px, smoothed=3.77 px, F1=0.934 | `phase4-picking-seq-bilstm-v1`: val RMSE≈1.94 px, F1≈0.93, coverage≈0.52 | val RMSE≈-1.5 px, F1≈-0.004 |
| 2026-06-14 | phase4-picking-seq-reg-v1 | Regularization experiment on top of the BiLSTM default: `weight_decay=0.10`, `label_smoothing=0.1`. | baseline: val RMSE=1.94 px, F1=0.932, coverage=0.516 | `phase4-picking-seq-reg-v1`: val RMSE=1.76 px, F1=0.932, coverage=0.485 | val RMSE=-0.18 px, F1=0.000, coverage=-0.031 |
| 2026-06-14 | phase4-picking-seq-reg-coverage-v1 | Added `absent_class_weight=0.8` to reg-v1 to raise coverage. | reg-v1: val RMSE=1.76 px, F1=0.932, coverage=0.485 | `phase4-picking-seq-reg-coverage-v1`: val RMSE=1.94 px, F1=0.939, coverage=0.480 | val RMSE=+0.18 px; visible mode jumps — reverted |
| 2026-06-14 | phase4-picking-inference-triage-v1 | Replaced hard-coded `--quality-threshold` in `run_inference.py` with percentile-based triage (`needs_review_from_batch`). Defaults: composite p5, coverage p5, smoothness p5. On `phase4-picking-seq-bilstm-v1` full-dataset inference this flags ~129/1,392 spectra (~9.3%) for manual review. | hard threshold: 0 spectra below 0.5 | percentile rule: ~129 flagged (~9.3%) | data-adaptive review queue |

<!-- Append new entries above this line -->
