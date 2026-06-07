# Phase 2: Self-Supervised Pretraining on FK Spectra — Implementation TODO

> **Status:** ❌ MAE exhausted — switching to **VICReg**  
> **Depends on:** Phase 0 (✅), Phase 1 (✅)  
> **Hardware target:** RTX 3060, 6 GB VRAM
> **Lessons learned:** MAE reconstruction objective fails for homogeneous FK spectra (embedding collapse in all 3 experiments). VICReg is the primary alternative.

---

## 0. Architectural Decisions & Reuse Strategy

### Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **MAE approach** | ❌ Abandoned after 3 experiments | All variants (block/random masking, 75%-25% ratios, aggressive augmentations) produced embedding collapse (Silhouette < 0, contrast < 1.1). Reconstruction loss is fundamentally insufficient for homogeneous FK data. |
| **New approach** | **VICReg** (Variance-Invariance-Covariance Regularization) | Explicitly prevents collapse via variance regularization. No reconstruction head. Works at batch_size=2. ~150 lines of code. |
| **Encoder architecture** | Re-use ViT-Small encoder from Phase 0 (without decoder head) | Already proven to have sufficient capacity. VICReg uses encoder + projector MLP instead of encoder + decoder. |
| **Config strategy** | New `VICRegConfig` dataclass (or extend `FKMAEConfig`) | Separate config from MAE. Requires: embedding_dim, projector_hidden_dims, loss weights (λ, µ, ν), augmentation params. |
| **Augmentation scope** | Same aggressive augmentations from MAE v3 | freq_shift, waven_shift, freq_dropout, noise, intensity jitter — all transfer directly. VICReg *requires* at least 2 different augmented views per sample. |
| **Validation split** | Same as Phase 2 (120 phase-1 val + 10% random from train) | No change needed. |
| **Trainer pattern** | New `VICRegTrainer` | No decoder, no masking, no reconstruction loss. Different loss computation (invariance + variance + covariance). |
| **Embedding extraction** | Projector output (or encoder output before projector) | VICReg uses encoder + projector; typically the projector output is used as the embedding, but encoder output before projection also works. |

### Reuse from Phase 0 (no changes)

- `src/models/mae.py` — `MaskedAutoencoder` (including `extract_embeddings`)
- `src/training/scheduler.py` — cosine warmup schedule
- `src/utils/seed.py`, `src/utils/device.py`, `src/utils/checkpoint.py`, `src/utils/plot_style.py`
- AMP pattern, gradient accumulation logic, checkpoint save/load format

### Reuse from Phase 1 (minor extensions)

- `src/data/fk_dataset.py` — extend to accept a `transform` callable and support programmatic val-split expansion
- `src/data/preprocessing.py` — `load_preprocessed_spectrum()` used as-is

### New Components

- `src/data/augmentations.py` — FK-specific on-the-fly transforms
- `src/utils/config.py` — `FKMAEConfig` dataclass
- `src/training/fk_trainer.py` — `FKMAETrainer(MAETrainer)` with tqdm and FK visuals
- `configs/phase2_fk_mae.yaml` — resolved config for first run
- `scripts/train_fk_mae.py` — CLI entry point

---

## 1. Configuration

### 1.1 FKMAEConfig dataclass (`src/utils/config.py`)

Create `FKMAEConfig` with the following groups. All model hyperparameters mirror the Phase 0 ViT-MAE baseline so that the smoke-test validation transfers directly.

**Data**
- `manifest_path: str = "data/processed/manifest.json"`
- `val_fraction: float = 0.10` — fraction of *train-line* spectra to hold out for val
- `val_seed: int = 42` — seed for the random sub-split (independent of training seed)

**Model** (identical to Phase 0 baseline)
- `img_size: int = 256`, `patch_size: int = 16`, `in_channels: int = 1`
- `embed_dim: int = 384`, `depth: int = 12`, `num_heads: int = 6`, `mlp_ratio: float = 4.0`
- `decoder_embed_dim: int = 256`, `decoder_depth: int = 4`, `decoder_num_heads: int = 8`
- `mask_ratio: float = 0.75`, `use_block_masking: bool = True`, `block_size: int = 2`

**Augmentation**
- `noise_std: float = 0.05` — std for Gaussian noise (in normalized amplitude units)
- `intensity_jitter: float = 0.30` — relative scale factor, i.e. `tensor * U(1±jitter)`

**Training**
- `batch_size: int = 2`, `accum_steps: int = 8` → eff. batch 16 (conservative for 6 GB)
- `epochs: int = 30`
- `lr: float = 5e-5`, `weight_decay: float = 0.05`, `betas: tuple = (0.9, 0.95)`
- `warmup_ratio: float = 0.1`, `grad_clip_norm: float = 1.0`
- `seed: int = 42`

**System**
- `num_workers: int = 4`, `pin_memory: bool = True`

**Logging**
- `log_interval: int = 50`
- `visualization_epochs: list[int] = [5, 10, 20, 30]` — 1-based epochs for UMAP/recon plots

**Serialization:** implement `to_dict()`, `from_yaml()`, `save_yaml()` same pattern as existing configs.

### 1.2 Default config YAML

`configs/phase2_fk_mae.yaml` — populate with the default values above. Comment every section so the file is self-documenting.

---

## 2. Data Augmentation

### 2.1 Design principles

- Augmentations are **on-the-fly** (applied in `__getitem__`, not during preprocessing).
- They operate on the already-normalized tensor of shape `(1, 256, 256)`.
- Only the **training** split receives augmentation; validation sees raw preprocessed spectra.
- Each augmentation must be **deterministic given a seed** so that unit tests can assert invariants.

### 2.2 Augmentation module (`src/data/augmentations.py`)

Implement a callable `FKSpectrumTransform` (or plain function) that composes the enabled augmentations. The callable accepts and returns a `Tensor` of shape `(1, 256, 256)`.

**Gaussian noise:** `tensor + torch.randn_like(tensor) * std`

**Intensity jitter:** `tensor * scale` where `scale ~ U(1 - jitter, 1 + jitter)`.

Apply order: intensity jitter → noise.

### 2.3 Integration with FKDataset

`FKDataset.__init__` already accepts `transform: Callable | None`. Pass the augmentation callable when building the **train** dataset only.

```python
train_ds = FKDataset(
    manifest_path=config.manifest_path,
    split="train",
    transform=build_augmentation(config),
    val_fraction=config.val_fraction,
    val_seed=config.val_seed,
)
```

---

## 3. Validation Split Expansion

### 3.1 Strategy

Phase 1 manifest contains:
- 1,272 entries with `split: "train"` (lines other than 5115, 5259)
- 120 entries with `split: "val"` (lines 5115, 5259)

Goal: val set = 120 + ~127 = ~247 spectra.

Approach: add a helper `create_train_val_entries(manifest_path, val_fraction, val_seed)` that:
1. Loads the manifest.
2. Separates entries into `phase1_val` (existing `split == "val"`) and `phase1_train` (existing `split == "train"`).
3. Uses `random.Random(val_seed)` to shuffle `phase1_train` deterministically.
4. Moves the first `val_fraction` of `phase1_train` into `val_entries`.
5. Returns `(train_entries, val_entries)`.

`FKDataset` then accepts an optional `entries: list[dict] | None` parameter. If `entries` is provided, it is used directly instead of reading the manifest. This keeps split logic testable and outside the dataset class.

### 3.2 Leakage safety

- No spectrum may appear in both train and val.
- The sub-split must be reproducible (`val_seed`).
- Unit test: assert `set(train_ids).isdisjoint(set(val_ids))`.

---

## 4. Training Infrastructure

### 4.1 FKMAETrainer (`src/training/fk_trainer.py`)

Subclass `MAETrainer` and override the minimal surface area:

**`_train_epoch`** — wrap the batch loop with `tqdm` for per-epoch progress reporting. Keep all existing logic (AMP, grad-accum, clipping, scheduler stepping). Log batch loss to tqdm description.

**`_run_visualization`** — replace MNIST-specific visualization with FK-specific:
- Reconstruction grid: show original / masked / reconstructed FK spectra.
- UMAP of validation embeddings: color points by `line_number` from metadata (reveals geographic structure) instead of digit labels. Overlay a few example spectra per UMAP neighborhood/cluster as inset thumbnails to verify that nearby points are visually similar.
- Similarity matrix: compute intra-line vs inter-line cosine similarity and contrast ratio.

**`_validate`** — extend to also compute and log embedding statistics (optional for first iteration; can be deferred if it adds VRAM pressure).

All other methods (`_setup_optimizer`, `_setup_scheduler`, `_save_checkpoint`, `_load_checkpoint`, `train`) are inherited unchanged.

### 4.2 Visualization adaptations (`src/evaluation/visualize.py`)

Add FK-specific plotting functions (keep MNIST ones intact):

- `plot_fk_reconstruction_grid(...)` — 3-panel layout: original spectrum, masked, composite reconstruction. Use `imshow` with a sequential colormap (e.g., `viridis` or `plasma`). Add colorbar. No need for patch outlines (FK spectra don't have sharp edges like digits).
- `plot_fk_umap(...)` — UMAP of embeddings with points colored by `line_number`. Include Silhouette score annotation. Optionally sample 3–4 representative spectra from distinct UMAP neighborhoods and render them as small inset panels or a companion figure. This provides qualitative evidence that clusters are visually distinct (e.g., different receiver lines or geologic settings produce separable spectral signatures).
- `plot_fk_similarity_matrix(...)` — intra-line vs inter-line mean cosine similarity, contrast ratio.

All functions follow the same API contract as Phase 0: accept `save_path: Path | None`, be runnable headless, seed random sampling.

---

## 5. Training Script (`scripts/train_fk_mae.py`)

Mirror the structure of `scripts/train_mae.py` with FK-specific adaptations:

1. Parse args: `--config`, `--resume`, `--name`, `--epochs`, `--dry-run`.
2. Load `FKMAEConfig` from YAML.
3. `set_seed(config.seed)`.
4. Call `get_device()`, enable `cudnn.benchmark`.
5. Create run directory: `experiments/YYYY-MM-DD_phase2-fk-mae/`.
6. Save resolved config snapshot.
7. **Build datasets:**
   ```python
   train_entries, val_entries = create_train_val_entries(
       config.manifest_path, config.val_fraction, config.val_seed
   )
   train_ds = FKDataset(
       manifest_path=config.manifest_path,
       split="train",
       transform=build_augmentation(config),
       entries=train_entries,
   )
   val_ds = FKDataset(
       manifest_path=config.manifest_path,
       split="val",
       transform=None,
       entries=val_entries,
   )
   ```
8. Build DataLoaders (`num_workers=4`, `pin_memory=True`).
9. Instantiate `MaskedAutoencoder` (same as Phase 0).
10. Instantiate `FKMAETrainer`.
11. Run `trainer.train()`.
12. Log total time and output paths.

Dry-run mode: 1 epoch, 2 batches, tiny subset.

---

## 6. Evaluation & Visualization Schedule

| Epoch | Action |
|-------|--------|
| 1 | Masking example panel (first batch) |
| 5 | Reconstruction grid + UMAP + similarity matrix |
| 10 | Reconstruction grid + UMAP + similarity matrix |
| 20 | Reconstruction grid + UMAP + similarity matrix |
| 30 | Reconstruction grid + UMAP + similarity matrix + final loss curves |

All plots saved to `experiments/<run>/plots/` as high-DPI PNG per PROJECT_RULES §8.

Metrics logged to `metrics.jsonl` every epoch: `epoch`, `train_loss`, `val_loss`, `lr`, `max_vram_mb`, `throughput_samples_per_sec`.

---

## 7. Testing

### 7.1 Unit tests (`tests/test_augmentations.py`)

- `test_gaussian_noise_shape_preservation` — output shape == input shape `(1, 256, 256)`.
- `test_intensity_jitter_range` — verify scale factor is within `[1-jitter, 1+jitter]`.
- `test_augmentation_determinism` — same seed → same output.
- `test_no_augmentation_on_val` — FKDataset with `transform=None` returns raw tensor.

### 7.1.1 Visualization audit

Write a script `scripts/visualize_augmentations.py` (or a pytest fixture) that:
1. Loads 5 random spectra from the training split (seeded).
2. Applies the full augmentation pipeline to each.
3. Produces a before/after figure panel for each spectrum:
   - Left: original preprocessed spectrum (no augmentation).
   - Right: augmented spectrum.
   - Shared colorbar and axis labels (Hz, 1/m) from metadata.
4. Saves as high-DPI PNG to `experiments/YYYY-MM-DD_phase2-augmentation-audit/`.

**Purpose:** Manual sanity check that augmentation preserves FK mode structures and does not introduce artifacts (e.g., excessive noise drowning dispersion curves, intensity jitter creating non-physical negative values if clipping is not applied).

### 7.2 Unit tests (`tests/test_fk_split.py`)

- `test_split_disjoint` — train and val spectrum IDs are disjoint.
- `test_split_reproducibility` — same seed yields identical splits.
- `test_split_preserves_phase1_val` — all Phase 1 val entries remain in val.
- `test_split_size` — val size ≈ 120 + 0.10 × 1272.

### 7.3 Smoke test

Run `scripts/train_fk_mae.py --dry-run`:
- Completes 1 epoch without errors.
- Checkpoint file is written.
- VRAM stays under 4.5 GB.
- Plots directory is created (may be empty in dry-run).

### 7.4 Checkpoint resume test

- Run for 2 epochs, save checkpoint.
- Resume from checkpoint, run 2 more epochs.
- Assert loss continuity (< 1 % relative divergence at first batch).

---

## 8. Success Criteria Gate

Before declaring Phase 2 complete and moving to Phase 3, the following must hold:

| Check | Target | How to Verify |
|-------|--------|---------------|
| Training stability | Val loss decreases monotonically (or at least trends down) over 30 epochs | `metrics.jsonl` inspection + loss curve plot |
| No NaN/Inf | No NaN or Inf in any logged metric | Assert in smoke test + manual log review |
| VRAM ceiling | Peak < 4.5 GB | `max_vram_mb` logged every epoch |
| Checkpoint integrity | Save/load/resume produces < 1 % loss divergence | Resume smoke test |
| Embedding structure | UMAP shows visually separable structure (not a single blob) | Manual review of UMAP plots |
| Intra/inter contrast | Cosine similarity contrast > 1.5 (lower bar than MNIST's 3.70 due to unlabeled, noisier data) | Similarity matrix plot |
| Code quality | `ruff check .` and `ruff format .` pass; `ty .` passes | CI / manual run |
| Model change log | Entry appended to `experiments/MODEL_CHANGELOG.md` | Manual review |

If **all pass** → update `MODEL_CHANGELOG.md`, freeze encoder weights, proceed to Phase 3 (embedding extraction + clustering).

If **loss does not converge** → check LR (may need 1e-4 instead of 5e-5), check augmentation strength (too much noise can drown signal), verify data normalization range.

If **VRAM > 5.5 GB** → reduce `batch_size` to 1 and increase `accum_steps` to 16.

If **UMAP is a single blob** → revisit masking ratio (try 0.70), check that normalization hasn't collapsed dynamic range, or consider adding frequency-shift augmentation for diversity.

---

## 9. Implementation Order

Recommended sequence to minimize interdependencies:

1. `FKMAEConfig` + `configs/phase2_fk_mae.yaml`
2. `create_train_val_entries()` helper + tests
3. `src/data/augmentations.py` + tests
4. `FKDataset` extension for `entries` parameter
5. `src/training/fk_trainer.py` (subclass + tqdm + FK visualization)
6. `src/evaluation/visualize.py` FK plotting functions
7. `scripts/train_fk_mae.py`
8. Smoke test + full 30-epoch run
9. Model change log entry

---

## 10. Model Change Tracking

Before the first training run, append a new section to `experiments/MODEL_CHANGELOG.md`:

| Date | Model Version | Architecture Delta | Baseline Metric | New Metric | Metric Delta |
|------|---------------|--------------------|-----------------|------------|--------------|
| 2026-06-07 | phase2-fk-mae-v1 | Phase 0 ViT-MAE transferred to FK data. Aug: Gaussian noise (std=0.01) + intensity jitter (±15%). Val split: 120 phase-1 val + 10% random from train lines. Epochs=30. | N/A | TBD | N/A |

After the run completes, fill in the metric columns with the best val_loss, Silhouette, and contrast values.

---

---

## 11. Experiment Log

### v1 — 2026-06-07: Baseline (block masking 75%, 30 epochs)

| Setting | Value |
|---------|-------|
| `mask_ratio` | 0.75 |
| `use_block_masking` | true |
| `noise_std` | 0.05 |
| `intensity_jitter` | 0.30 |
| `epochs` | 30 |
| `min_lr` | 5e-6 (min_lr_ratio=0.1) |

**Outcome:** ❌ Embedding collapse. Silhouette = −0.322, contrast = 1.078 (intra=0.876, inter=0.812).
UMAP: ring structure with mixed lines. Loss plateaued at ~0.08 by epoch 10.

### v2 — 2026-06-07: Random masking 50% (30 epochs)

| Setting | Value |
|---------|-------|
| `mask_ratio` | 0.50 |
| `use_block_masking` | false |
| `noise_std` | 0.05 |
| `intensity_jitter` | 0.30 |
| `epochs` | 30 |
| `min_lr` | 5e-6 |

**Outcome:** ❌ Marginal improvement. Val loss 0.075 (vs 0.084 in v1), but Silhouette still negative,
contrast ~1.08. Embedding collapse persists.

### v3 — 2026-06-07: Aggressive aug + 25% masking + 100 epochs (stopped at epoch 54)

| Setting | Value |
|---------|-------|
| `mask_ratio` | 0.25 |
| `use_block_masking` | false |
| `noise_std` | 0.15 |
| `intensity_jitter` | 0.50 |
| `freq_shift_max` | 0.10 |
| `waven_shift_max` | 0.05 |
| `freq_dropout_prob` | 0.30 |
| `epochs` | 100 (stopped at epoch 54) |
| `min_lr` | 1e-6 (min_lr_ratio=0.02) |

**Outcome:** ❌ No meaningful improvement at epoch 50. Val loss = 0.0945 at epoch 54
(still higher than v2's 0.0747 at epoch 30). Embedding collapse persists — Silhouette
still negative, contrast ~1.08. MAE definitively fails for FK spectra.

**Diagnosis:** The pixel-level reconstruction objective is fundamentally unsuitable for
FK spectra. All samples share the same global structure (dark field + diagonal dispersion
modes), making the "average spectrum" a low-loss reconstruction strategy regardless of
masking ratio or augmentation strength. Encoder embeddings collapse because the model
does not need to distinguish samples to minimize the reconstruction loss.

---

## 12. VICReg Implementation — Completed but Failed

### Implementation Summary

All files were created and a 50-epoch run was executed:
- `src/models/vicreg.py` — VICReg model + loss (verified correct vs paper)
- `src/training/vicreg_trainer.py` — Trainer with dual-view augmentation
- `src/utils/config.py` — `VICRegConfig` dataclass
- `configs/phase2_vicreg.yaml` — Config with λ=25, µ=25, ν=1, batch=16, LR=3e-4
- `scripts/train_vicreg.py` — CLI entry point

### v4 — 2026-06-07: VICReg (50 epochs, batch=16)

| Setting | Value |
|---------|-------|
| `batch_size` | 16 |
| `projector_hidden_dim` | 2048 |
| `projector_out_dim` | 2048 |
| `sim_weight` | 25.0 |
| `var_weight` | 25.0 |
| `cov_weight` | 1.0 |
| `epochs` | 50 |
| `lr` | 3e-4 |

**Outcome:** ❌ Embedding collapse persists. Silhouette = −0.252, contrast = 1.036 at epoch 50.
Loss plateaued at ~37.0 (inv=0.007, var=1.30, cov=4.39). UMAP shows a fuzzy ball with no
line-based clustering. Similarity matrix: intra=0.895, inter=0.864.

**Key finding:** The variance hinge loss (`var_loss ≈ 1.30`) is *still active* after 50 epochs,
meaning many embedding dimensions have std < 1. The model cannot push variance above the
threshold because all samples map to the same small region of embedding space. The
covariance term cannot decorrelate features because there is no signal to decorrelate.

---

## 13. Synthesis: Why All Self-Supervised Methods Failed

### The Core Problem

FK spectra are **too homogeneous**. All 1,145 training samples share the same global
structure: a dark background with diagonal dispersion mode bands. The differences between
spectra from different receiver lines are extremely subtle — essentially noise in the
exact position and amplitude of the modes.

| Method | Collapse Mode | Why It Failed |
|--------|--------------|---------------|
| **MAE** | Exact collapse (all embeddings ≈ identical) | Reconstruction loss minimized by predicting "average spectrum" for every masked patch. No need to distinguish samples. |
| **VICReg** | Fuzzy-ball collapse (all embeddings in small region) | Variance hinge cannot push std ≥ 1 because the optimal representation for all samples is the same small region. Invariance + covariance have no signal to work with. |

**Signal-to-noise ratio is too low** for any self-supervised objective to extract
distinguishing features. The model learns the shared structure perfectly but never
learns what makes spectra different.

### Evidence Summary

| Experiment | Best Silhouette | Best Contrast | Verdict |
|-----------|-----------------|---------------|---------|
| MAE v1 (block 75%) | −0.322 | 1.078 | ❌ Collapse |
| MAE v2 (random 50%) | ~−0.30 | ~1.08 | ❌ Collapse |
| MAE v3 (aggressive aug, 25%) | ~−0.30 | ~1.08 | ❌ Collapse |
| VICReg v4 (batch=16, 50 ep) | −0.252 | 1.036 | ❌ Collapse |

All four experiments converge to the same result: **the data itself does not contain
enough distinguishing signal for unsupervised representation learning.**

---

## 14. Next Steps — Decision Tree

### Option A: BYOL (Low effort, low expected reward)

Bootstrap Your Own Latent uses an EMA target network + predictor. It doesn't rely on
batch statistics like VICReg, so it might escape the fuzzy-ball minimum. But BYOL also
lacks negative pairs — it only pushes augmented views of the *same* sample together.
If VICReg's explicit variance term couldn't prevent collapse, BYOL's implicit mechanism
is unlikely to succeed either.

**Effort:** ~1 hour (model + trainer already exist, just add EMA target + predictor)
**Expected outcome:** 20% chance of marginal improvement (Silhouette > −0.1)

### Option B: SimCLR / Contrastive Learning (Medium effort, medium reward)

Uses explicit negative pairs: push *different* samples apart in embedding space while
pulling augmented views of the same sample together. This directly addresses collapse.

**Problem:** SimCLR needs batch sizes of 256+ for stable negative sampling. With 6GB VRAM
and our model size (30M params), we'd need gradient accumulation (batch=4, accum=64).
This makes training very slow (~2 hours for 100 epochs).

**Effort:** ~2 hours (implement InfoNCE loss, memory bank or large-batch accum)
**Expected outcome:** 50% chance of meaningful improvement (Silhouette > 0.0)

### Option C: Supervised Pretraining with Pseudo-Labels (Medium effort, high reward) — **RECOMMENDED**

Skip self-supervised entirely. Use weak supervision:
1. Extract simple features from raw spectra (e.g., PCA 50 components, or spectral descriptors)
2. Cluster with HDBSCAN to get pseudo-labels
3. Train a supervised ViT classifier (cross-entropy) to predict pseudo-labels
4. Use the classifier's penultimate layer as embeddings for Phase 3

The cross-entropy loss **explicitly forces the model to discriminate between clusters**,
which is exactly what self-supervised methods failed to do.

**Effort:** ~1 hour (HDBSCAN clustering + supervised classifier training)
**Expected outcome:** 70% chance of Silhouette > 0.1, contrast > 1.5

### Option D: Classical Feature Extraction (Low effort, guaranteed baseline)

Skip learned embeddings entirely:
1. Flatten spectra → PCA (50-200 components) → UMAP → HDBSCAN clustering
2. Or use spectral descriptors: peak frequencies, mode bandwidths, energy distribution
3. Proceed directly to Phase 3 (active learning) with classical features

This provides a **guaranteed working baseline** even if all learned approaches fail.
The downside is that classical features may miss subtle patterns that a neural network
could theoretically learn.

**Effort:** ~30 minutes
**Expected outcome:** Silhouette unknown, but clusters will form. Quality depends on
feature engineering.

---

## 15. Recommendation

**Primary:** Implement **Option C** (supervised pretraining with pseudo-labels). It is
the most principled next step because it gives the model an explicit discrimination
signal that self-supervised methods lack.

**Fallback:** If Option C fails after a 30-epoch test, proceed immediately to **Option D**
(classical features) and move to Phase 3. Do not spend more time on representation
learning — the bottleneck is the data, not the algorithm.

**Do not pursue:** Option A (BYOL) unless Option C fails and the user explicitly wants
to exhaust all deep-learning approaches before falling back to classical methods.

---

## 16. Implementation Order (Option C)

1. Extract raw spectra → flatten → PCA (50-200 components)
2. HDBSCAN clustering on PCA embeddings → pseudo-labels
3. Create `FKClassifierConfig` + `configs/phase2_supervised.yaml`
4. Implement `src/models/fk_classifier.py` — ViT-Small + classification head
5. Implement `src/training/supervised_trainer.py` — standard cross-entropy training
6. Train 30 epochs, evaluate Silhouette / contrast on pseudo-labels
7. If Silhouette > 0.1, train full 100 epochs and proceed to Phase 3
8. If Silhouette < 0.1, fall back to Option D (classical features)

