# swami-picker

A human-in-the-loop deep-learning pipeline to extract fundamental-mode dispersion curves from noisy FK (frequency-wavenumber) spectra for surface-wave inversion.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Status](#project-status)
3. [Repository Layout](#repository-layout)
4. [Installation](#installation)
5. [End-to-End Pipeline on a New Dataset](#end-to-end-pipeline-on-a-new-dataset)
   - [Phase 1: Preprocess raw SEG-Y FK spectra](#phase-1-preprocess-raw-seg-y-fk-spectra)
   - [Phase 2: Build pseudo-labels and embeddings](#phase-2-build-pseudo-labels-and-embeddings)
   - [Phase 3: Active annotation](#phase-3-active-annotation)
   - [Phase 4: Train the picking model](#phase-4-train-the-picking-model)
   - [Phase 5: Run inference and export dispersion curves](#phase-5-run-inference-and-export-dispersion-curves)
6. [Visual Quality Checks](#visual-quality-checks)
7. [Development](#development)
8. [Hardware Notes](#hardware-notes)

---

## Overview

Surface-wave dispersion curves are a key input to near-surface velocity inversion, but automatically picking the fundamental mode from FK spectra is hard:

- No ground-truth labels exist.
- Higher-order modes and noise often dominate.
- Spectra are visually homogeneous (shared dark background + diagonal mode bands).
- Coordinates must be transformed back to physical units (Hz and 1/m) for inversion software.

**swami-picker** solves this with a reproducible, modular pipeline:

1. **Preprocessing** — read raw SEG-Y traces, normalize, resize to `256×256`, and store reversible metadata sidecars.
2. **Pseudo-label clustering** — extract physics-informed spectral descriptors, reduce with UMAP, cluster with HDBSCAN, and train a lightweight classifier to obtain separable 128-D embeddings.
3. **Active learning** — use the embeddings to select representative spectra for expert annotation via a tkinter picking app.
4. **Supervised picking** — train a compact U-Net + BiLSTM that predicts a dense `256`-element dispersion curve from a raw spectrum.
5. **Coordinate transform & export** — map model picks to Hz / 1 / m, propagate uncertainty, and export CSV/JSON for inversion.

---

## Project Status

The pipeline is implemented end-to-end and has been run on a 1,392-spectrum field dataset:

| Phase | Status | Key Output |
|-------|--------|------------|
| Phase 0 — MNIST MAE smoke test | ✅ Complete | Verified training stack |
| Phase 1 — FK preprocessing | ✅ Complete | `data/processed/manifest.json`, `data/processed/spectra/*.npz` |
| Phase 2 — Pseudo-label clustering | ✅ Complete | `data/processed/mlp_embeddings_phase3.npz`, 11 merged clusters |
| Phase 3 — Active annotation | ✅ Complete | 188 initial + 129 review annotations |
| Phase 4 — Supervised picking | ✅ Complete | `phase4-picking-seq-bilstm-v1`, val RMSE≈1.94 px, F1≈0.93 |
| Phase 5 — Inversion export | ✅ Complete | CSV/JSON dispersion curves, quality triage |

Self-supervised MAE and VICReg were tested exhaustively and abandoned because FK spectra are too homogeneous for unsupervised representation learning. The working path uses **weakly supervised pseudo-labels**.

---

## Repository Layout

```text
swami-picker/
├── configs/                  # YAML experiment configs
├── data/                     # Raw & processed data (gitignored)
│   └── processed/            # manifest.json, spectra/*.npz + .json
├── experiments/              # Training logs & checkpoints (gitignored)
│   └── MODEL_CHANGELOG.md    # Architecture–metric history (tracked)
├── scripts/                  # Executable pipeline scripts
│   ├── phase1_pipeline/
│   ├── phase2_supervised/
│   ├── phase3_active_learning/
│   ├── phase4_picking/
│   └── debug/
├── src/                      # Library code
│   ├── data/                 # SEG-Y reader, datasets, augmentations
│   ├── models/               # MAE, VICReg, classifiers, picking model
│   ├── training/             # Trainers, losses, schedulers
│   ├── evaluation/           # Metrics, features, visualizations
│   ├── picking/              # Annotation app, interpolation, I/O
│   ├── transforms/           # Model ↔ physical coordinate transforms
│   └── utils/                # Seed, device, config, plotting style
├── tests/                    # 250+ unit / integration tests
├── todos/                    # Phase-by-phase planning docs
├── PROJECT_PLAN.md           # Full research plan
├── PROJECT_RULES.md          # Coding conventions
├── pyproject.toml            # Python dependencies
└── README.md                 # This file
```

---

## Installation

Requires **Python ≥3.14** and a CUDA-capable GPU is recommended (tested on RTX 3060, 6 GB).

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project and dev dependencies
pip install -e ".[dev]"

# Verify the test suite
pytest tests/
```

**Using `uv`:**

```bash
# Create and sync the environment (Python ≥3.14)
uv sync

# Run commands inside the managed environment
uv run pytest tests/
uv run python scripts/phase1_pipeline/preprocess_fk.py --dry-run

# Or activate the uv-managed venv manually
source .venv/bin/activate
pytest tests/
```

All scripts assume the repository root as the working directory so that relative paths (`data/processed/...`, `configs/...`) resolve correctly.

---

## End-to-End Pipeline on a New Dataset

These steps assume you have raw SEG-Y FK spectra in `data/*.sgy`. Replace file names with your own configs and run names.

### Phase 1: Preprocess raw SEG-Y FK spectra

```bash
python scripts/phase1_pipeline/preprocess_fk.py \
    --config configs/phase1_fk_pipeline.yaml
```

Outputs:

- `data/processed/manifest.json` — list of all spectra with train/val split.
- `data/processed/spectra/<spectrum_id>.npz` — preprocessed `256×256` tensor.
- `data/processed/spectra/<spectrum_id>.json` — reversible metadata sidecar.
- `experiments/<date>_phase1-fk-pipeline/before_after_comparison.png` — quality check.

To smoke-test on one file:

```bash
python scripts/phase1_pipeline/preprocess_fk.py --dry-run
```

To inspect the preprocessed data programmatically:

```bash
python scripts/phase1_pipeline/verify_data_pipeline.py \
    --manifest data/processed/manifest.json
```

---

### Phase 2: Build pseudo-labels and embeddings

The production path uses 20-D physics-informed spectral descriptors → UMAP → HDBSCAN → MLP classifier.

1. **Extract features** (two paths: marginals + PCA, and spectral descriptors):

```bash
python scripts/phase2_supervised/extract_pseudo_label_features.py \
    --manifest data/processed/manifest.json \
    --output-dir data/processed/features
```

Outputs:

- `data/processed/features/features_marginal.npz`
- `data/processed/features/features_descriptors.npz`

2. **Cluster descriptors** to obtain pseudo-labels:

```bash
python scripts/phase2_supervised/cluster_pseudo_labels.py \
    --features data/processed/features/features_descriptors.npz \
    --output data/processed/pseudo_labels.npz \
    --n-neighbors 15 \
    --min-dist 0.0 \
    --n-components 5 \
    --min-cluster-size 30 \
    --min-samples 10
```

Outputs:

- `data/processed/pseudo_labels.npz` — labels, probabilities, UMAP embeddings.
- `data/processed/pseudo_labels.json` — cluster size summary + Silhouette score.
- `data/processed/pseudo_labels.png` — UMAP scatter plot.

If HDBSCAN produces one dominant cluster, run the **two-step hierarchical merge** (see `PROJECT_PLAN.md` §16.4 and `todos/phase2_fk_mae_pretraining.md` §16.9). The final merged labels used in production are saved as `pseudo_labels_merged.npz`.

3. **Train the MLP classifier** on pseudo-labels:

```bash
python scripts/phase2_supervised/train_pseudo_label_classifier.py \
    --config configs/phase2_supervised_mlp_final.yaml \
    --labels data/processed/pseudo_labels_merged.npz \
    --name phase2c-mlp-final
```

Outputs in `experiments/phase2c-mlp-final/`:

- `checkpoints/best_model.pt`
- `config.yaml`
- `metrics.jsonl`
- UMAP / similarity plots.

4. **Extract 128-D embeddings** for all spectra (input to active learning):

```bash
python scripts/phase2_supervised/extract_mlp_embeddings.py \
    --checkpoint experiments/phase2c-mlp-final/checkpoints/best_model.pt \
    --manifest data/processed/manifest.json \
    --output data/processed/mlp_embeddings_phase3.npz
```

---

### Phase 3: Active annotation

1. **Prepare an annotation session** using the embeddings:

```bash
python scripts/phase3_active_learning/prepare_session.py \
    --embeddings data/processed/mlp_embeddings_phase3.npz \
    --percentage 15.0 \
    --name iter0
```

This prints a per-cluster budget table and asks for confirmation. It then creates `annotations/<date>_iter0/`.

2. **Launch the annotation app**:

```bash
python scripts/phase3_active_learning/launch_app.py \
    --session-dir annotations/<date>_iter0
```

Hotkeys in the app:

- `Click` — add pick, `Right-click` — remove nearest pick.
- `Space` / `z` — next / previous spectrum.
- `q` / `w` — jump to previous / next cluster.
- `v` — snap picks to nearest positive local maxima.
- `x` — clear all picks.
- `s` — save, `Esc` — quit.

3. **Export annotations** to Phase 4 training format:

```bash
python scripts/phase3_active_learning/export_annotations.py \
    --session-dirs annotations/<date>_iter0 \
    --output data/processed/phase4_training_data.npz \
    --min-direct-picks 3
```

---

### Phase 4: Train the picking model

```bash
python scripts/phase4_picking/train_picking_model.py \
    --config configs/phase4_picking.yaml \
    --name phase4-picking-my-run
```

The default config trains the U-Net + BiLSTM model with 5-fold cross-validation. Outputs in `experiments/phase4-picking-my-run/`:

- `checkpoints/best_model.pt`
- `config.yaml`
- `metrics.jsonl`
- `plots/` — curve overlays, probability heatmaps, training curves.

For a quick smoke test:

```bash
python scripts/phase4_picking/train_picking_model.py \
    --config configs/phase4_picking.yaml \
    --dry-run
```

---

### Phase 5: Run inference and export dispersion curves

1. **Run inference** on the full dataset:

```bash
python scripts/phase4_picking/run_inference.py \
    --checkpoint experiments/phase4-picking-my-run/checkpoints/best_model.pt \
    --manifest data/processed/manifest.json \
    --export-annotations
```

Outputs in the run directory:

- `predictions.npz` — picks + presence probabilities for all spectra.
- `quality_scores.json` — per-spectrum quality metrics.
- `low_quality_spectra.json` — spectra flagged for manual review.
- `annotations_for_review/spectra/*.npz` — model picks loaded by the annotation app.

2. **Review low-quality spectra** (optional):

```bash
python scripts/phase4_picking/prepare_review_session.py \
    --run-dir experiments/phase4-picking-my-run \
    --name review_low_quality \
    --yes

python scripts/phase3_active_learning/launch_app.py \
    --session-dir annotations/<date>_review_low_quality
```

3. **Merge manual corrections** back into predictions:

```bash
python scripts/phase4_picking/merge_manual_picks.py \
    --predictions experiments/phase4-picking-my-run/predictions.npz \
    --session-dirs annotations/<date>_review_low_quality \
    --output experiments/phase4-picking-my-run/predictions_merged.npz
```

4. **Export dispersion curves** in physical units:

```bash
python scripts/phase4_picking/export_dispersion_curves.py \
    --predictions experiments/phase4-picking-my-run/predictions_merged.npz \
    --output-dir exports/dispersion-curves \
    --format both
```

Outputs:

- `exports/dispersion-curves/csv/<spectrum_id>.csv`
- `exports/dispersion-curves/json/<spectrum_id>.json`
- `exports/dispersion-curves/all_dispersion_curves.csv`
- `exports/dispersion-curves/manifest.json`

---

## Visual Quality Checks

Quality-control figures are generated automatically, but several scripts help inspect results at each step.

| Step | Script | What it shows |
|------|--------|---------------|
| Preprocessing | `experiments/<date>_phase1-fk-pipeline/before_after_comparison.png` | Original vs. resized spectra |
| Preprocessing | `scripts/debug/visualize_augmentations.py` | Augmented variants of random spectra |
| Clustering | `data/processed/pseudo_labels.png` | UMAP → HDBSCAN clusters |
| Classifier | `scripts/phase2_supervised/visualize_mlp_final.py` | Embedding UMAP, similarity matrix |
| Picking training | `experiments/<run>/plots/training_curves.png` | Loss, RMSE, F1, LR, VRAM |
| Picking training | `experiments/<run>/plots/curve_predictions_epoch_*.png` | True vs. predicted curves |
| Picking inference | `scripts/phase4_picking/visualize_inference_results.py` | Quality distributions + best/worst examples |
| Picking inference | `experiments/<run>/plots/inference_*.png` | Composite/confidence/smoothness rankings |

Example: visualize full-dataset inference results.

```bash
python scripts/phase4_picking/visualize_inference_results.py \
    --predictions experiments/phase4-picking-my-run/predictions_merged.npz \
    --quality-scores experiments/phase4-picking-my-run/quality_scores.json \
    --manifest data/processed/manifest.json \
    --num-examples 8
```

Example: inspect augmentations before training.

```bash
python scripts/debug/visualize_augmentations.py \
    --manifest data/processed/manifest.json \
    --n-samples 5
```

---

## Development

Code quality is enforced with **Ruff** (lint + format) and **ty** (static type check). Run before every commit:

```bash
ruff check .
ruff format .
ty check .
pytest tests/
```

**With `uv`:**

```bash
uv run ruff check .
uv run ruff format .
uv run ty check .
uv run pytest tests/
```

Key conventions:

- Google-style docstrings.
- Type hints on all public signatures.
- Tensor shapes documented in docstrings.
- Config-driven experiments; no magic numbers in training scripts.
- All logs and checkpoints under `experiments/`.

Architecture or hyperparameter changes must be logged in `experiments/MODEL_CHANGELOG.md`.

---

## Hardware Notes

The pipeline is optimized for consumer GPUs:

- Automatic Mixed Precision (AMP) is used in all training loops.
- Micro-batch size stays small; effective batch size is scaled via gradient accumulation.
- `torch.backends.cudnn.benchmark = True` is enabled for fixed-size inputs.
- Peak VRAM is logged every epoch. The picking model fits in ~4.5 GB at batch size 16.

If you run out of memory:

- Reduce `--batch-size` in training scripts.
- Increase `accum_steps` in the config instead of increasing batch size.
- Reduce `base_channels` or `seq_hidden_dim` in `configs/phase4_picking.yaml`.

---

---

*Last updated: 2026-06-15*
