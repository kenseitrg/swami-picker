# Phase 4: Supervised Fine-Tuning & Dispersion Curve Picking — TODO

> **Status:** Implementation complete — **BiLSTM sequence model is now the default**. `phase4-picking-seq-bilstm-v1` reached val RMSE≈1.94 px and val F1≈0.93, improving on the previous v2.1 single-head default (val RMSE=3.46 px, F1=0.934). The older single-head architecture and the multi-mode experiment are kept as alternatives but are no longer the default. Full-dataset inference completed on all 1,392 spectra.  
> **Depends on:** Phase 3 (✅ annotations collected)  
> **Goal:** Train a supervised model that predicts a dense `(256,)` dispersion-curve pick from a raw `(1, 256, 256)` FK spectrum.  
> **Tests:** 33 coordinate-transform tests + 11 FK dataset tests passing; full Phase 4 suite passing.

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
| **Output representation** | **Single 257-class head** `(B, 257, W)`: 256 wavenumber bins + 1 explicit "no pick" class | Removes separate presence head, preventing the model from being too conservative by hiding behind a presence gate. Every column must choose a wavenumber or absence. |
| **Backbone** | Compact U-Net with skip connections (`PickingModel`) | `base_channels=16`, `embed_dim=64`, ~2.3M params. Larger capacity than the initial v2 refactor because the small dataset still needs enough model capacity to capture multi-mode context; strong dropout (`0.5`) controls overfitting. |
| **Sequence head (default)** | **BiLSTM over the frequency axis** (`SeqPickingModel`) | The U-Net decoder output is reshaped to `(B, C*H, W)` and processed by a 2-layer BiLSTM (`seq_hidden_dim=128`) with a residual skip. This gives the model explicit frequency-axis context and is the current default after `phase4-picking-seq-bilstm-v1`. |
| **Multi-mode head** | Optional `MultiModePickingModel` | Predicts `num_modes` parallel 257-class sequences and selects the smoothest path at inference. Kept as an experimental alternative, but the current best-alignment loss suffers from assignment degeneracy, so it is **not** the default. |
| **Regularization experiments** | Label smoothing + higher weight decay tested, then reverted | `phase4-picking-seq-reg-v1` (`label_smoothing=0.1`, `weight_decay=0.10`) slightly improved val RMSE but did not reduce presence overfit. `phase4-picking-seq-reg-coverage-v1` (`absent_class_weight=0.8`) introduced visible mode jumps; the baseline BiLSTM hyperparameters are retained as default. |
| **Manual review triage** | Percentile-based quality guards | Implemented in `needs_review_from_batch()`. Default flags spectra in the bottom 5% of composite score, bottom 5% of coverage, or bottom 5% of smoothness. On BiLSTM inference this flags ~129/1,392 spectra (~9.3%) for expert review. The old hard-coded `--quality-threshold` is deprecated but still supported. |
| **Cluster conditioning** | ❌ **Removed** for v2 | Single-head model makes cluster conditioning harder to integrate; revisit only if needed. |
| **Augmentation** | **Pick-synchronized** transforms only | `FreqShift` and `WavenShift` move both image and picks consistently; intensity jitter and Gaussian noise leave picks unchanged. |
| **Validation split** | **K-fold cross-validation** (`k_folds=5`) + stratified by cluster label | Larger, more robust validation sets (~38 spectra) instead of a single tiny 10% hold-out. |
| **Checkpoint selection** | **Smoothed validation RMSE** (5-epoch moving average) | Avoids selecting an overfitted spike due to noisy small validation sets. |
| **Loss** | Single cross-entropy over 257 classes; direct picks weighted ×2 | Simpler objective; absence is just another class, so the model is graded on coverage and accuracy jointly. |
| **Coordinate transform** | ✅ Implemented in `src/transforms/coordinates.py` | Model outputs pixel indices; inversion needs Hz and 1/m. |

---

## 2. Configuration & Scaffolding

### 2.1 Create `src/utils/config.py::PickingConfig` ✅

Implemented. Locked fields after reviewer feedback:

- Removed misleading `num_classes`; replaced with `spectrum_height: int = 256`.
- Removed unused `aug_hflip` (dispersion curves are not symmetric).
- Removed unused `backbone_lr` (model trained end-to-end from scratch).
- Renamed `loss_l1_weight` → `loss_pick_weight` to match cross-entropy semantics.
- ❌ Removed `backbone`, `use_cluster_conditioning`, `cluster_embedding_path`, `loss_bce_weight` after single-head refactor.
- Added `k_folds: int = 1` and `fold_index: int = 0` for cross-validation.
- Added `dropout: float = 0.5` inside conv blocks.
- Added `loss_smooth_weight: float` for frequency-axis smoothness.
- Added `min_val_coverage: float` to avoid selecting collapsed checkpoints.
- Added `early_stopping_patience: int = 15`.
- Added `smooth_window: int = 5` for moving-average checkpoint selection.

**Data**
- `training_data_path: str = "data/processed/phase4_training_data.npz"`
- `val_fraction: float = 0.10` (used only when `k_folds == 1`)
- `val_seed: int = 42`
- `min_direct_picks: int = 3` — filter spectra below this threshold
- `k_folds: int = 5` — number of CV folds
- `fold_index: int = 0` — which fold to use as validation

**Model**
- `model_type: str = "seq"` — default architecture: U-Net + BiLSTM over frequency
- `base_channels: int = 16`
- `embed_dim: int = 64`
- `spectrum_height: int = 256` — wavenumber bins (must match input height)
- `dropout: float = 0.5`
- `seq_hidden_dim: int = 128`
- `seq_layers: int = 2`
- `seq_type: str = "bilstm"`
- `num_downsample: int = 2`

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
- `loss_pick_weight: float = 1.0`
- `direct_pick_weight: float = 2.0`
- `loss_smooth_weight: float = 0.05`
- `min_val_coverage: float = 0.05`
- `early_stopping_patience: int = 15`
- `smooth_window: int = 5`
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
        k_folds: int = 1,
        fold_index: int = 0,
    )
```

Responsibilities:
1. Load `.npz` with `allow_pickle=True`.
2. Parse metadata JSON string into a list of dicts (or fail gracefully if already a list).
3. Filter spectra with fewer than `min_direct_picks` direct picks.
4. Build stratified train/val split by `cluster_labels` using `val_seed`, or use k-fold CV when `k_folds > 1`.
5. Return tuple `(spectrum, pick_target, direct_mask, confidence, cluster_label, spectrum_id)`.

**Target construction:**
- `pick_target`: `(256,)` float32. `-1` for unpicked columns. Valid picks in `[0, 255]`.
- `direct_mask`: `(256,)` bool.
- `confidence`: `(256,)` float32.
- `cluster_label`: scalar int.

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
- `test_kfold_disjoint` — k-fold train/val indices are disjoint.
- `test_min_direct_picks_filter` — spectra below threshold excluded.
- `test_item_shapes_and_targets` — returns expected tensors.
- `test_split_reproducibility` — same seed yields identical splits.
- `test_freq_shift_sync` — after horizontal roll, picks roll by same amount; rolled-in columns marked unpicked.
- `test_waven_shift_clip` — vertical shift clamps picks to `[0, 255]` and marks OOB as unpicked.
- `test_intensity_jitter_range` / `test_gaussian_noise_shape_preservation` — non-pick-changing augmentations.
- `test_pick_sync_transform_disabled` / `test_pick_sync_transform_enabled_changes_spectrum` — composed transform behavior.

---

## 4. Model Architecture ✅

### 4.1 Create `src/models/picking_model.py` ✅

**`PickingModel`** — single 257-class head.

```
Input (B, 1, 256, 256)
    │
    ├── Encoder ──► (B, 16, 128, 128)
    │   Conv(1→8) → ReLU → Down
    │   Conv(8→16) → ReLU → Down
    │
    ├── Bottleneck ──► (B, 64, 64, 64)
    │   Conv(16→64) → ReLU → Down
    │
    ├── Decoder ──► (B, 8, 256, 256)
    │   Up → Conv(64+16→16) → ReLU
    │   Up → Conv(16+8→8) → ReLU
    │
    └── Classifier ──► (B, 257, 256)
        Conv1d(base_channels * H, 257, k=1)
```

Forward returns:
- `logits`: `(B, 257, W)` — one logit per (wavenumber class / absent class, frequency column)

At inference (`inference_picks`):
- `pick_idx = argmax(logits, dim=1)` → `(B, W)`
- `presence_prob = 1 - softmax(logits)[:, absent_class, :]` → `(B, W)`
- Final pick: `pick_idx` where `pick_idx != absent_class`, else `-1`

### 4.2 Model unit tests (`tests/test_picking_model.py`) ✅

- `test_model_forward_shape`
- `test_inference_argmax`
- `test_absent_class_masking`
- `test_present_class_kept`
- `test_build_picking_model`

---

## 5. Loss Function ✅

### 5.1 Create `src/training/picking_loss.py` ✅

**Class `PickingLoss`**

Single cross-entropy over 257 classes per frequency column:

```python
target = pick_target.clone()
target[target < 0] = absent_class
ce = F.cross_entropy(logits, target.long(), reduction="none")
weights = torch.where(direct_mask, direct_pick_weight, 1.0)
pick_loss = (ce * weights).sum() / (weights.sum() + eps)
```

Absence is the last class; every column is graded, so the model cannot avoid coverage by suppressing presence.

### 5.2 Loss tests (`tests/test_picking_loss.py`) ✅

- `test_unpicked_columns_graded`
- `test_direct_weight_scaling`
- `test_loss_decreases_when_correct`
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

**Best checkpoint selection:**
- Maintain a 5-epoch moving average of `val_rmse_pixels`.
- Save `best_model.pt` only when the smoothed metric improves.
- Early stopping counts epochs without smoothed improvement.

**Visualization epochs:** generates three plot sets on validation set:
- curve predictions overlay (true vs. predicted picks)
- probability heatmap overlays (grayscale spectrum + `hot` probability overlay, no pick curves)
- presence certainty distributions

### 6.2 Create `scripts/phase4_picking/train_picking_model.py` ✅

Implemented. Supports:
- Args: `--config`, `--resume`, `--name`, `--dry-run`
- Loads config, sets seed, gets device, enables `cudnn.benchmark` on CUDA
- Builds `FKPickingDataset` train/val with k-fold CV support
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
- `plot_probability_heatmap_overlay` — grayscale spectrum with translucent `hot` probability heatmap overlay (no pick curves).
- `plot_certainty_distributions` — histograms of presence probabilities, split by ground truth.
- `plot_column_error_heatmap` — spectrum with red overlay proportional to per-column pick error.
- `plot_error_distribution` — histogram of per-spectrum RMSE values.
- `torch_softmax` — numerically stable numpy softmax helper.

All plots use `src/utils/plot_style.py`, headless, publication-ready.

### 7.3 Visualization tests (`tests/test_visualize_picking.py`) ✅

8 tests covering each plot function and the softmax helper.

---

## 8. Coordinate Transform Integration ✅

### 8.1 Create `src/transforms/coordinates.py` ✅

Implemented the matched pair required by PROJECT_RULES §4.2 and PROJECT_PLAN.md §5:

- `model_indices_to_physical(picks, metadata, presence_probs=..., confidence=..., certainty_strategy="presence") -> PhysicalPicks`
  - Maps dense `(256,)` wavenumber-index picks to Hz / 1/m using the resized physical axes in metadata.
  - Propagates **first-order uncertainty**: base 0.5-pixel quantization error scaled by inverse pick certainty (presence probability or confidence), multiplied by local bin width.
  - Absent picks (`-1`) produce `NaN` wavenumber values/uncertainties.
  - Certainties are conservative bounds, not calibrated 1σ Gaussian errors.

- `physical_picks_to_model_indices(f_hz, k_inv_m, metadata) -> list[tuple[int, int]]`
  - Forward transform from physical units to sparse model indices.

- `physical_picks_to_dense_model_indices(...)` → dense `(256,)` array via PCHIP interpolation.

- `inference_to_annotation_record(...)` → converts model predictions into `AnnotationRecord` objects loadable by the existing picking app.

- `compute_spectrum_quality_score(...)` → composite score (coverage + certainty + relative smoothness + monotonicity) for triaging spectra for manual re-annotation.

- `dispersion_curve_to_dataframe(...)` → exports physical picks with Hz, 1/m, phase velocity, uncertainties, and geographic metadata.

### 8.2 Tests (`tests/test_coordinate_transform.py`) ✅

33 tests covering:
- Metadata validation (missing keys, non-monotonic axes, descending axes).
- Forward transform shapes, values, absent-pick handling, uncertainty scaling with presence probability.
- Inverse transform sparse/dense mappings, out-of-bounds dropping, NaN handling.
- Round-trip accuracy on linear and log-spaced axes, including non-identity picks.
- Inference-to-annotation bridge.
- Quality scoring: physical-picks branch, trend/steep-tail tolerance, spike penalty, weight validation.
- DataFrame export including integration against real `RL5007_50071009.json` metadata.

---

## 9. Inference & Export Script ✅

### 9.1 Create `scripts/phase4_picking/run_inference.py` ✅

Implemented. Args:
- `--checkpoint`: path to trained model checkpoint (required).
- `--manifest`: path to `data/processed/manifest.json` (default).
- `--config`: optional path to model config YAML; if omitted, config is restored from the checkpoint.
- `--output`: output `.npz` path (defaults to `<checkpoint-run>/predictions.npz`).
- `--batch-size`: default 32.
- `--num-workers`: DataLoader workers (default 4).
- `--quality-threshold`: composite score threshold for flagging low-quality spectra (default 0.5).
- `--confidence-threshold`: minimum presence probability to mark a pick as direct in exported annotations (default 0.5).
- `--export-annotations`: export `.npz` annotation records for review in the picking app.
- `--seed`: reproducibility seed.

Actions:
1. Loads model from checkpoint in eval mode.
2. Iterates over **all 1,392 spectra** in the manifest (`split=None`).
3. Outputs `(256,)` pick indices + `(256,)` presence probabilities per spectrum.
4. Saves:
   ```
   predictions.npz:
     spectrum_ids: (1392,)
     picks: (1392, 256) int16
     presence_probs: (1392, 256) float32
     metadata: JSON string of list[dict]
   ```
5. Saves `quality_scores.json` with per-spectrum composite metrics.
6. Saves `low_quality_spectra.json` listing spectra below `--quality-threshold`.
7. Optional: exports `annotations_for_review/spectra/<spectrum_id>.npz` ready for the existing picking app.

**End-to-end run:** 1,392 spectra processed in ~8.8 s at ~158 spectra/s. Composite scores range 0.729–0.950 (mean 0.853). With default threshold 0.5, no spectra flagged; the score distribution is useful for ranking rather than hard filtering.

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

| Run | Config | Model | Augmentation | Notes |
|-----|--------|-------|--------------|-------|
| `phase4-picking-seq-bilstm-v1` | `configs/phase4_picking.yaml` | **U-Net + BiLSTM (default)**, base=16, embed=64, dropout=0.5 | noise + intensity | New default; val RMSE≈1.94 px, F1≈0.93 |
| `phase4-picking-seq-reg-v1` | `configs/phase4_picking_seq_regularized.yaml` | U-Net + BiLSTM | noise + intensity + label smoothing 0.1 + weight decay 0.10 | Slightly better RMSE, same presence overfit |
| `phase4-picking-seq-reg-coverage-v1` | `configs/phase4_picking_seq_reg_coverage.yaml` | U-Net + BiLSTM | as reg-v1 + `absent_class_weight=0.8` | Worse: visible mode jumps; not promoted |
| `phase4-picking-v2-singlehead` | `configs/phase4_picking.yaml` (model_type=`picking`) | Single 257-class head, base=16, embed=64, dropout=0.5 | noise + intensity | Previous v2.1 default; kept for reference |

Manual review triage defaults (see `scripts/phase4_picking/run_inference.py`):
- `--review-composite-percentile 5`
- `--review-coverage-percentile 5`
- `--review-smoothness-percentile 5`
| `phase4-picking-v2-shifts` | `configs/phase4_picking_shifts.yaml` | Single 257-class head | Shifts disabled (shift aug hurt metrics on the small dataset) | Kept for reference |
| `phase4-picking-multimode-v1` | `configs/phase4_picking_multimode.yaml` | Multi-mode head | noise + intensity | Experimental; assignment degeneracy prevents convergence |

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
| Picking RMSE (model space) | < 2 pixels on valid columns | Smoothed `val_rmse_pixels` (BiLSTM default reaches ~1.94 px) |
| Manual review queue | ~5–10% of spectra flagged | `low_quality_spectra.json` from percentile-based triage |
| Presence/absence F1 | > 0.85 | `val_presence_f1` |
| Coverage | Predicted coverage within 10% of true coverage | `val_coverage` |
| Overfitting gap | `train_rmse - val_rmse` < 2 pixels | Loss curves |
| Visual sanity | Predicted curves follow visible mode energy | Curve overlay plots |
| VRAM peak | < 4.5 GB | `max_vram_mb` |
| Coordinate round-trip | RMSE < 1 pixel equivalent | `tests/test_coordinate_transform.py` |
| Code quality | `ruff check .`, `ruff format .`, `ty .` pass | CI / manual |
| Model change log | Entry appended | `experiments/MODEL_CHANGELOG.md` |

If **all pass** → freeze best model, run inference on all 1,392 spectra, export dispersion curves, proceed to Phase 5 (full coordinate transform / inversion export) or close Phase 4.

If **RMSE > 5 pixels** → increase annotation count, try a slightly larger model, or add a transformer-style non-local block.

If **overfitting gap > 3 pixels** → reduce model capacity, increase dropout, or add heavier augmentation.

If **coverage is much lower than true coverage** → the single-head model is still too conservative; revisit class weighting or add an explicit coverage penalty.

---

## 12. Implementation Order

1. ✅ **`src/utils/config.py::PickingConfig`** + `configs/phase4_picking.yaml` + `configs/phase4_picking_shifts.yaml`
2. ✅ **`src/data/picking_dataset.py`** + tests (k-fold CV support)
3. ✅ **`src/data/picking_augmentations.py`** + tests
4. ✅ **`src/models/picking_model.py`** + tests (single 257-class head, compact U-Net)
5. ✅ **`src/training/picking_loss.py`** + tests (single cross-entropy)
6. ✅ **`src/training/picking_trainer.py`** + tests (smoothed metric selection)
7. ✅ **`src/evaluation/picking_metrics.py`** + `src/evaluation/visualize_picking.py`
8. ✅ **`src/transforms/coordinates.py`** + tests
9. ✅ **`scripts/phase4_picking/train_picking_model.py`**
10. ✅ **`scripts/phase4_picking/run_inference.py`**
11. ⏳ **`scripts/phase4_picking/export_dispersion_curves.py`**
12. **Smoke test + full run** ✅
13. **Update `experiments/MODEL_CHANGELOG.md`** ✅

---

## 13. Known Issues to Fix Before Training

| Issue | Location | Fix |
|-------|----------|-----|
| Metadata stored as JSON string | `data/processed/phase4_training_data.npz["metadata"]` | Parse in `FKPickingDataset` or re-export from Phase 3 as list of dicts. ✅ Fixed in `FKPickingDataset._load_npz`. |
| Low direct-pick count | Phase 3 annotations | Accept for v2; consider a second annotation pass if model underfits. |
| Missing cluster `4` | `phase4_training_data.npz["cluster_labels"]` | Document in MODEL_CHANGELOG; single-head model has no cluster conditioning, so this is less critical. |

---

## 14. Model Change Tracking Template

Append to `experiments/MODEL_CHANGELOG.md` before the first run:

```
| 2026-06-12 | phase4-picking-v1 | Phase 4 core library implemented. U-Net picking model on 188 annotated spectra. Two heads: 256-class wavenumber logits + presence logit. Pick-synchronized augmentation. Visualizations: curve overlays, probability heatmaps, certainty distributions. | N/A | N/A | N/A |
| 2026-06-13 | phase4-picking-v2 | Refactored to single 257-class head (256 bins + absent class). Compact U-Net: base_channels=8, embed_dim=64, dropout=0.3 (~0.59M params). K-fold CV (5 folds). Smoothed val RMSE checkpoint selection. Grayscale probability heatmaps. | N/A | N/A | N/A |
| 2026-06-13 | phase4-picking-v2.1 | Final single-head architecture: base_channels=16, embed_dim=64, dropout=0.5 (~2.3M params). Added expected-value frequency-axis smoothness loss (weight=0.05). Disabled pick-synchronized shifts; kept noise + intensity jitter. Coverage safeguard on checkpoint selection. Run `phase4-picking-v2-final`: best val RMSE=3.46 px, smoothed=3.77 px, F1=0.934. | N/A | N/A | N/A |
| 2026-06-14 | phase4-picking-seq-bilstm-v1 | **New default.** Added `SeqPickingModel`: U-Net decoder output reshaped to `(B, C*H, W)` and processed by a 2-layer BiLSTM (`seq_hidden_dim=128`) with residual skip. `configs/phase4_picking.yaml` and `PickingConfig` now default to `model_type="seq"`, `seq_type="bilstm"`. | v2.1: val RMSE=3.46 px, F1=0.934 | val RMSE≈1.94 px, F1≈0.93 | val RMSE≈-1.5 px |
```

After each run, fill in best smoothed `val_rmse_pixels`, `val_presence_f1`, and velocity error.

---

*Last updated: 2026-06-14*
