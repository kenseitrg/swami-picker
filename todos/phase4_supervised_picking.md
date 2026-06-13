# Phase 4: Supervised Fine-Tuning & Dispersion Curve Picking — TODO

> **Status:** Implementation in progress — core library complete, scripts pending.  
> **Depends on:** Phase 3 (✅ annotations collected)  
> **Goal:** Train a supervised model that predicts a dense `(256,)` dispersion-curve pick from a raw `(1, 256, 256)` FK spectrum.  
> **Tests:** 196 passing (Phase 4 modules: 32; visualization: 8; full suite).

---

## 0. Inventory of Existing Artifacts

| Artifact | Path | Shape / Description |
|----------|------|---------------------|
| Annotated spectra | `data/processed/phase4_training_data.npz` | `(188, 1, 256, 256)` float32 |
| Pick indices | same | `(188, 256)` int16, `-1` = no pick |
| Direct pick masks | same | `(188, 256)` bool, `True` = human clicked |
| Confidences | same | `(188, 256)` float32 |
| Cluster labels | same | `(188,)`, labels `0..3, 5..11` (label `4` missing) |
| Metadata | same | Stored as **JSON string** — needs parsing/fixing |

**Data-quality notes (locked until more annotations arrive):**
- Mean direct picks ≈ 5.7/spectrum, coverage ≈ 53% — lower than the Phase 3 target of ≥8 direct picks.
- Missing cluster `4` means the model will have no examples for that cluster; plan for zero-shot / cluster-conditional fallback if cluster `4` appears at inference.
- Metadata is a JSON string in the `.npz`; the Phase 4 loader must parse it or we must re-export with `metadata` as a list of dicts.

---

## 1. Architectural Decisions (Locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Input** | Raw preprocessed spectrum `(1, 256, 256)` | Spatial relationships between frequency and wavenumber must be preserved. |
| **Output representation** | Two heads: (a) `(256,)` wavenumber classification, (b) `(256,)` presence probability | Per-column cross-entropy over 256 wavenumber bins; presence mask handles "no visible mode" columns. |
| **Backbone** | Lightweight U-Net with skip connections (`SimpleUNetPickingModel`) | Fits on RTX 3060; encoder-decoder variant available as baseline. |
| **Cluster conditioning** | Optional `ClusterConditionalPickingModel` | Broadcast-adds projected 128-D cluster embedding into U-Net bottleneck. |
| **Augmentation** | **Pick-synchronized** transforms only | `FreqShift` and `WavenShift` move both image and picks consistently; intensity jitter and Gaussian noise leave picks unchanged. |
| **Validation split** | Stratified by cluster label + 10% hold-out | Cluster-stratified to ensure all represented clusters appear in validation. |
| **Loss** | Wavenumber cross-entropy (weighted by direct/interpolated) + BCE on presence mask | Cross-entropy respects the ordinal heatmap structure; BCE learns where a mode is visible. |
| **Coordinate transform** | Re-use Phase 5 helpers from `src/transforms/` (pending) | Model outputs pixel indices; inversion needs Hz and 1/m. |

---

## 2. Configuration & Scaffolding

### 2.1 Create `src/utils/config.py::PickingConfig` ✅

Implemented. Locked fields after reviewer feedback:

- Removed misleading `num_classes`; replaced with `spectrum_height: int = 256`.
- Removed unused `aug_hflip` (dispersion curves are not symmetric).
- Removed unused `backbone_lr` (model trained end-to-end from scratch).
- Renamed `loss_l1_weight` → `loss_pick_weight` to match cross-entropy semantics.
- Added `cluster_embed_dim: int = 128` for conditional model factory.

**Data**
- `training_data_path: str = "data/processed/phase4_training_data.npz"`
- `val_fraction: float = 0.10`
- `val_seed: int = 42`
- `min_direct_picks: int = 3` — filter spectra below this threshold
- `use_cluster_conditioning: bool = False`
- `cluster_embedding_path: str | None = None` — path to `mlp_embeddings_phase3.npz` if conditioning used

**Model**
- `backbone: str = "unet"` — `"unet"` or `"encoder_decoder"`
- `base_channels: int = 32`
- `embed_dim: int = 128` — bottleneck / conditioning vector
- `spectrum_height: int = 256` — wavenumber bins (must match input height)

**Augmentation (pick-synchronized)**
- `aug_enabled: bool = True`
- `aug_noise_std: float = 0.05`
- `aug_intensity_jitter: float = 0.15`
- `aug_freq_shift_max: float = 0.05` — horizontal shift, picks shift with image
- `aug_waven_shift_max: float = 0.03` — vertical shift, picks shift with image

**Training**
- `batch_size: int = 16`
- `accum_steps: int = 1`
- `epochs: int = 100`
- `lr: float = 1e-3`
- `weight_decay: float = 0.05`
- `betas: tuple = (0.9, 0.95)`
- `warmup_ratio: float = 0.1`
- `grad_clip_norm: float = 1.0`
- `loss_pick_weight: float = 1.0` — weight for wavenumber cross-entropy loss
- `loss_bce_weight: float = 0.5`
- `direct_pick_weight: float = 2.0` — weight direct picks higher than interpolated
- `seed: int = 42`

**System**
- `num_workers: int = 4`
- `pin_memory: bool = True`
- `log_interval: int = 10`

**Logging**
- `visualization_epochs: list[int] = [10, 25, 50, 75, 100]`

Implement `to_dict()`, `from_yaml()`, `save_yaml()` following existing patterns.

### 2.2 Create `configs/phase4_picking.yaml` ✅

Created with all locked defaults and section comments.

### 2.3 Create run-directory layout ✅

`PickingTrainer` creates:

```
experiments/YYYY-MM-DD_phase4-picking-<name>/
├── config.yaml
├── metrics.jsonl
├── checkpoints/
│   ├── checkpoint_epoch_*.pt
│   └── best_model.pt
├── plots/
│   ├── curve_predictions_epoch_*.png
│   ├── probability_heatmaps_epoch_*.png
│   ├── certainty_distributions_epoch_*.png
│   └── training_curves.png
└── logs/
```

---

## 3. Pick-Synchronized Data Pipeline ✅

### 3.1 Create `src/data/picking_dataset.py` ✅

**Class `FKPickingDataset`**

```python
class FKPickingDataset(Dataset):
    def __init__(
        self,
        npz_path: Path,
        split: str = "train",          # "train" or "val"
        val_fraction: float = 0.10,
        val_seed: int = 42,
        min_direct_picks: int = 3,
        transform: Callable | None = None,
        cluster_embeddings: dict[str, ndarray] | None = None,
    )
```

Responsibilities:
1. Load `.npz` with `allow_pickle=True`.
2. Parse metadata JSON string into a list of dicts (or fail gracefully if already a list).
3. Filter spectra with fewer than `min_direct_picks` direct picks.
4. Build stratified train/val split by `cluster_labels` using `val_seed`.
5. Return tuple `(spectrum, pick_target, presence_target, direct_mask, confidence, cluster_embedding_or_none, spectrum_id)`.

**Target construction:**
- `pick_target`: `(256,)` float32. `-1` regions become `NaN` or a masked value. Valid picks in `[0, 255]`.
- `presence_target`: `(256,)` float32, `1.0` where `pick != -1`, `0.0` otherwise.
- `direct_mask`: `(256,)` bool.
- `confidence`: `(256,)` float32.

### 3.2 Create `src/data/picking_augmentations.py` ✅

**Class `PickSyncTransform`**

Augmentations must update both the spectrum **and** the pick target consistently.

| Transform | Image effect | Pick effect |
|-----------|--------------|-------------|
| `FreqShift` | `torch.roll(..., dims=2)` | Shift pick indices horizontally; wrap or fill `-1` for rolled-in columns |
| `WavenShift` | `torch.roll(..., dims=1)` | Add shift to all pick indices; clip to `[0, 255]`, mark out-of-bounds as unpicked |
| `IntensityJitter` | Scale amplitude | No change |
| `GaussianNoise` | Add noise | No change |

Important: when a shift wraps a column into the image, the corresponding pick is ambiguous. Safer strategy: shift with zero-fill and mark the filled columns as unpicked (`presence = 0`, `pick = NaN`).

Return augmented `(spectrum, pick_target, presence_target, direct_mask, confidence)`.

### 3.3 Unit tests (`tests/test_picking_dataset.py`, `tests/test_picking_augmentations.py`) ✅

- `test_load_metadata_json_string` — metadata JSON string parsed.
- `test_train_val_disjoint` — no spectrum in both splits.
- `test_min_direct_picks_filter` — spectra below threshold excluded.
- `test_item_shapes_and_targets` — returns expected tensors and presence target.
- `test_split_reproducibility` — same seed yields identical splits.
- `test_freq_shift_sync` — after horizontal roll, picks roll by same amount; rolled-in columns marked unpicked.
- `test_waven_shift_clip` — vertical shift clamps picks to `[0, 255]` and marks OOB as unpicked.
- `test_intensity_jitter_range` / `test_gaussian_noise_shape_preservation` — non-pick-changing augmentations.
- `test_pick_sync_transform_disabled` / `test_pick_sync_transform_enabled_changes_spectrum` — composed transform behavior.

---

## 4. Model Architecture ✅

### 4.1 Create `src/models/picking_model.py` ✅

**Primary: `SimpleUNetPickingModel`** (Option A)

```
Input (B, 1, 256, 256)
    │
    ├── Encoder ──► (B, 128, 32, 32)
    │   Conv(1→32) → ReLU → Down
    │   Conv(32→64) → ReLU → Down
    │   Conv(64→128) → ReLU → Down
    │
    ├── Decoder ──► (B, 32, 256, 256)
    │   Up → Conv(128→64) → ReLU
    │   Up → Conv(64→32) → ReLU
    │   Up → Conv(32→32) → ReLU
    │
    ├── Pick head ──► (B, 256, 256) logit heatmap
    │   Conv(32→1, k=1)
    │
    └── Presence head ──► (B, 1, 256, 256)
        Conv(32→1, k=1)
```

Forward returns:
- `pick_logits`: `(B, 256, 256)` — one logit per (frequency, wavenumber)
- `presence_logits`: `(B, 256, 256)` — one logit per frequency column

At inference:
- `pick_idx = argmax(pick_logits, dim=-1)` → `(B, 256)`
- `presence_prob = sigmoid(presence_logits.mean(dim=-1))` → `(B, 256)`
- Final pick: `pick_idx` where `presence_prob > threshold`, else `-1`

**Alternative: `EncoderDecoderPickingModel`**

Simpler 3-layer encoder-decoder as described in PROJECT_PLAN.md §4 Option A. Implement if U-Net overfits on 188 samples.

**Cluster-conditional variant (Option B): `ClusterConditionalPickingModel`**

- Load 128-D MLP cluster embedding for the spectrum.
- Broadcast to spatial grid `(B, 128, 64, 64)` and concatenate to encoder features.
- Keep as fallback, not default.

### 4.2 Model unit tests (`tests/test_picking_model.py`) ✅

- `test_unet_forward_shape`
- `test_encoder_decoder_forward_shape`
- `test_cluster_conditional_forward_shape`
- `test_cluster_conditional_no_embedding` — falls back to zero conditioning.
- `test_inference_argmax`
- `test_presence_masking`
- `test_build_picking_model_*` — factory covers U-Net, conditional, encoder-decoder, unknown backbone.

---

## 5. Loss Function ✅

### 5.1 Create `src/training/picking_loss.py` ✅

**Class `PickingLoss`**

Combines:
1. **Pick classification loss** (per-frequency cross-entropy over 256 wavenumber bins):
   ```python
   ce = F.cross_entropy(
       pick_logits.permute(0, 2, 1),   # (B, 256 freq, 256 waven)
       pick_target.long(),             # (B, 256)
       reduction="none",
   )
   ce = ce * presence_target * (direct_mask * direct_weight + (1 - direct_mask))
   pick_loss = ce.sum() / (presence_target.sum() + eps)
   ```
   This treats each frequency column as a 256-way classification. Target is the picked wavenumber index; `-1` columns are masked out by `presence_target`.

2. **Presence BCE loss**:
   ```python
   presence_loss = F.binary_cross_entropy_with_logits(
       presence_logits.squeeze(1), presence_target
   )
   ```

3. **Total**:
   ```python
   loss = pick_loss + loss_bce_weight * presence_loss
   ```

**Rationale:** Cross-entropy per column is a cleaner spatial fit than Smooth L1 on indices because it respects the ordinal nature of wavenumber bins and naturally gives uncertainty. If experiment shows regression is better, add `SmoothL1Loss` on normalized coordinates as an optional head.

### 5.2 Loss tests (`tests/test_picking_loss.py`) ✅

- `test_loss_decreases_when_correct`
- `test_presence_mask_zeroes_pick_loss`
- `test_direct_weight_scaling` — weighted normalizer ensures effective direct-pick weight is exact.
- `test_loss_components_finite`

---

## 6. Training Infrastructure ✅

### 6.1 Create `src/training/picking_trainer.py` ✅

**Class `PickingTrainer`**

Pattern after `PseudoLabelTrainer`:
- AMP + GradScaler
- Gradient accumulation
- Cosine warmup scheduler
- Gradient clipping
- Checkpointing (best by validation RMSE, not accuracy)
- Metrics logging to JSONL

**Metrics to log every epoch:**
- `train_loss`, `val_loss`
- `train_rmse_pixels`, `val_rmse_pixels` — RMSE on valid (presence=1) columns
- `train_presence_f1`, `val_presence_f1`
- `lr`, `max_vram_mb`, `throughput_samples_per_sec`

**Visualization epochs:** generates three plot sets on validation set:
- curve predictions overlay (true vs. predicted picks)
- probability heatmap overlays
- presence certainty distributions

### 6.2 Create `scripts/phase4_picking/train_picking_model.py` ✅

Implemented. Supports:
- Args: `--config`, `--resume`, `--name`, `--dry-run`
- Loads config, sets seed, gets device, enables `cudnn.benchmark` on CUDA
- Builds `FKPickingDataset` train/val with stratified split
- Loads optional cluster embeddings for conditional model
- Builds model via `build_picking_model()` and logs parameter count
- Instantiates `PickingTrainer` and runs training
- Saves config snapshot to run directory
- Dry-run mode: 1 epoch on a 32-sample train / 8-sample val subset

### 6.3 Unit / smoke tests ✅

- `tests/test_picking_trainer.py` — smoke run for 2 epochs and resume test.
- `tests/test_picking_collate.py` — custom collate handles `None` cluster embeddings and string spectrum IDs.
- `scripts/phase4_picking/train_picking_model.py --dry-run` completed successfully on RTX 3060 (~4.5 GB VRAM).

---

## 7. Evaluation & Metrics ✅

### 7.1 Create `src/evaluation/picking_metrics.py` ✅

Functions:
- `compute_curve_rmse(pred_picks, true_picks, presence_mask)` — RMSE in pixel indices on valid columns
- `compute_presence_f1(pred_presence, true_presence)` — threshold 0.5
- `compute_velocity_error(pred_picks, true_picks, metadata)` — convert indices to Hz/1/m, compute `|V_pred - V_true| / V_true`
- `compute_coverage(picks)` — fraction of frequency columns with a pick

### 7.2 Create `src/evaluation/visualize_picking.py` ✅

- `plot_curve_overlays` — grid of spectra with red=true, green=pred curves.
- `plot_training_curves` — loss, RMSE, presence F1, LR, VRAM.
- `plot_probability_heatmap_overlay` — spectrum with translucent softmax probability heatmap overlay.
- `plot_certainty_distributions` — histograms of presence probabilities, split by ground truth.
- `plot_column_error_heatmap` — spectrum with red overlay proportional to per-column pick error.
- `plot_error_distribution` — histogram of per-spectrum RMSE values.
- `torch_softmax` — numerically stable numpy softmax helper.

All plots use `src/utils/plot_style.py`, headless, publication-ready.

### 7.3 Visualization tests (`tests/test_visualize_picking.py`) ✅

8 tests covering each plot function and the softmax helper.

---

## 8. Coordinate Transform Integration (Phase 5 Prep) ⏳

### 8.1 Create `src/transforms/coordinates.py`

Implement the matched pair required by PROJECT_RULES §4.2 and PROJECT_PLAN.md §5:

- `model_indices_to_physical(picks_model, metadata) -> list[tuple[float, float, float, float]]`
  - Input: `picks_model` `(256,)` int array of wavenumber indices per frequency column.
  - Output: `(frequency_hz, wavenumber_inv_m, uncertainty_freq, uncertainty_waven)` tuples.
  - Only for columns where `pick != -1`.

- `physical_picks_to_model_indices(f_hz, k_inv_m, metadata) -> tuple[int, int]`
  - Forward transform for round-trip unit tests.

### 8.2 Tests (`tests/test_coordinate_transform.py`)

- `test_round_trip_linear_axes` — synthetic grid, forward → inverse → compare, RMSE < 1 pixel.
- `test_round_trip_log_freq` — if log freq transform added later.
- `test_uncertainty_propagation` — verify first-order uncertainty scaling.

---

## 9. Inference & Export Script ⏳

### 9.1 Create `scripts/phase4_picking/run_inference.py`

Args:
- `--checkpoint`: path to trained model checkpoint
- `--manifest`: path to `data/processed/manifest.json`
- `--output`: path to output `.npz`
- `--batch-size`: default 32
- `--presence-threshold`: default 0.5

Actions:
1. Load model from checkpoint, set eval mode.
2. Iterate over **all** 1,392 spectra in manifest.
3. For each spectrum, output `(256,)` pick indices + `(256,)` presence probabilities.
4. Save:
   ```
   predictions.npz:
     spectrum_ids: (N,)
     picks: (N, 256) int16
     presence_probs: (N, 256) float32
     metadata: list[dict]
   ```

### 9.2 Create `scripts/phase4_picking/export_dispersion_curves.py`

Args:
- `--predictions`: path to `predictions.npz`
- `--output-dir`: directory for CSV/JSON exports
- `--format`: `"csv"`, `"json"`, or `"geopsy"`

Output one file per spectrum with columns:
`spectrum_id, frequency_hz, wavenumber_inv_m, phase_velocity_m_s, presence_prob, model_version`

---

## 10. Experiment Matrix

Run these sequentially (each ~30–60 min on RTX 3060):

| Run | Architecture | Augmentation | Cluster Conditioning | Purpose |
|-----|--------------|--------------|----------------------|---------|
| `picking-unet-v1` | U-Net | noise + intensity | No | Primary candidate |
| `picking-unet-v2` | U-Net | + freq/waven shift | No | Test shift-aware aug |
| `picking-encdec-v1` | Encoder-decoder | noise + intensity | No | Lower-capacity baseline |
| `picking-unet-cond-v1` | U-Net | noise + intensity | Yes | Fallback if clusters vary |

For each run, append to `experiments/MODEL_CHANGELOG.md` with:
- `model_version`
- `architecture_delta`
- `baseline_metric` / `new_metric` / `metric_delta`

---

## 11. Success Criteria Gate

Before declaring Phase 4 complete:

| Check | Target | How to Verify |
|-------|--------|---------------|
| Training stability | Val loss decreases, no NaN/Inf | `metrics.jsonl` |
| Picking RMSE (model space) | < 3 pixels on valid columns | `val_rmse_pixels` |
| Presence/absence F1 | > 0.85 | `val_presence_f1` |
| Overfitting gap | `train_rmse - val_rmse` < 2 pixels | Loss curves |
| Visual sanity | Predicted curves follow visible mode energy | Curve overlay plots |
| VRAM peak | < 4.5 GB | `max_vram_mb` |
| Coordinate round-trip | RMSE < 1 pixel equivalent | `tests/test_coordinate_transform.py` |
| Code quality | `ruff check .`, `ruff format .`, `ty .` pass | CI / manual |
| Model change log | Entry appended | `experiments/MODEL_CHANGELOG.md` |

If **all pass** → freeze best model, run inference on all 1,392 spectra, export dispersion curves, proceed to Phase 5 (full coordinate transform / inversion export) or close Phase 4.

If **RMSE > 5 pixels** → increase annotation count, try cluster conditioning, or add a transformer-style non-local block.

If **overfitting gap > 3 pixels** → reduce model capacity, increase dropout, or add heavier augmentation.

---

## 12. Implementation Order

1. **`src/utils/config.py::PickingConfig`** + `configs/phase4_picking.yaml`
2. **`src/data/picking_dataset.py`** + tests (parse metadata string, stratified split)
3. **`src/data/picking_augmentations.py`** + tests (pick-synchronized shifts)
4. **`src/models/picking_model.py`** + tests (U-Net + optional conditioning)
5. **`src/training/picking_loss.py`** + tests
6. **`src/training/picking_trainer.py`**
7. **`src/evaluation/picking_metrics.py`** + `src/evaluation/visualize_picking.py`
8. **`src/transforms/coordinates.py`** + tests
9. **`scripts/phase4_picking/train_picking_model.py`**
10. **`scripts/phase4_picking/run_inference.py`**
11. **`scripts/phase4_picking/export_dispersion_curves.py`**
12. **Smoke test + full run** ⏳
13. **Update `experiments/MODEL_CHANGELOG.md`** ⏳

---

## 13. Known Issues to Fix Before Training

| Issue | Location | Fix |
|-------|----------|-----|
| Metadata stored as JSON string | `data/processed/phase4_training_data.npz["metadata"]` | Parse in `FKPickingDataset` or re-export from Phase 3 as list of dicts. |
| Low direct-pick count | Phase 3 annotations | Accept for v1; consider a second annotation pass if model underfits. |
| Missing cluster `4` | `phase4_training_data.npz["cluster_labels"]` | Document in MODEL_CHANGELOG; cluster-conditional model must handle unseen cluster IDs. |

---

## 14. Model Change Tracking Template

Append to `experiments/MODEL_CHANGELOG.md` before the first run:

```
| 2026-06-12 | phase4-picking-v1 | Phase 4 core library implemented. U-Net picking model on 188 annotated spectra. Two heads: 256-class wavenumber logits + presence logit. Pick-synchronized augmentation. Visualizations: curve overlays, probability heatmaps, certainty distributions. | N/A | N/A | N/A |
```

After the first training run, fill in best `val_rmse_pixels`, `val_presence_f1`, and velocity error.

---

*Last updated: 2026-06-12*
