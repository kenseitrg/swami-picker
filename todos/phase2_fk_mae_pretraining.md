# Phase 2: MAE Pretraining on FK Spectra — Implementation TODO

> **Status:** Ready to start  
> **Depends on:** Phase 0 (✅), Phase 1 (✅)  
> **Hardware target:** RTX 3060, 6 GB VRAM  
> **Epoch budget (first run):** 30

---

## 0. Architectural Decisions & Reuse Strategy

### Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Encoder architecture** | Re-use `MaskedAutoencoder` (ViT-Small) from Phase 0 as-is | FK spectra are already 256×256 single-channel; identical input shape. No need to change patch size, embed_dim, depth, or decoder. |
| **Config strategy** | New `FKMAEConfig` dataclass | Keeps Phase 0 configs untouched; avoids conditional fields for MNIST vs FK. |
| **Augmentation scope (initial)** | Gaussian/Poisson noise + intensity jitter only | Per user direction. Frequency/wavenumber shift and block masking deferred to a later iteration. |
| **Validation split** | Phase 1 val lines (120 spectra) + 10 % random from train lines (~127) | Increases val size to ~247 for more robust convergence signal while preserving geographic holdout. |
| **Trainer pattern** | Subclass `MAETrainer` → `FKMAETrainer` | Core loop (AMP, grad-accum, clipping, scheduler) is identical. Override only visualization and add tqdm. Phase 0 script continues to work unchanged. |
| **Embedding extraction** | Mean-pooled encoder output (as-is) | Phase 0 achieved contrast 3.70 with mean pooling. Revisit CLS token only if Phase 3 clustering under-performs. |

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

## 11. Open Questions to Revisit

- **Normalization ablation:** After this run, compare min-max vs z-score on reconstruction loss (carried over from Phase 1 TODO).
- **Frequency/wavenumber shift augmentation:** Defer to Phase 2 iteration 2; requires careful coordinate tracking if we want to use shifted spectra for supervised picking later.
- **Effective batch size:** Current plan is eff. batch 16 (`batch=2, accum=8`). If convergence is slow, try eff. batch 32 (`batch=2, accum=16`) on a short run and compare.
- **Encoder freezing for Phase 3:** Decide whether to freeze the entire encoder or only the earlier layers during prototype clustering. Document the decision when Phase 3 TODO is written.
