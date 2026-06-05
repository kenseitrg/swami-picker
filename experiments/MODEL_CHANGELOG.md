# Model Change Log

Accumulated architecture–metric history for the swami-picker project.
Every modification to model architecture, loss formulation, or data augmentation must be logged here.

## Log Format

| Date | Model Version | Architecture Delta | Baseline Metric | New Metric | Metric Delta |
|------|---------------|--------------------|-----------------|------------|--------------|
| YYYY-MM-DD | git-short-hash or semantic | Brief description of what changed | Previous best metric value | Metric after change | New − Baseline |

## Entries

| 2026-06-05 | phase0-mnist-mae-small | MNIST MAE smoke test: ViT-small encoder (embed_dim=384, depth=12, num_heads=6), lightweight decoder (depth=4, embed_dim=256), block masking 75%, 5 epochs | N/A (initial run) | val_loss=0.271, VRAM=777MB, Silhouette=0.18 | N/A |

<!-- Append new entries above this line -->
