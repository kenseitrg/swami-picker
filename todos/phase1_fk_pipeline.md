# Phase 1: FK Data Pipeline & Preprocessing — TODO

> **Status: Implementation complete.** All modules implemented, 77 tests passing, ruff + ty clean. Ready for full pipeline run.

## Goal
Transform raw SEG-Y FK spectra into a model-ready dataset with full metadata provenance, enabling reversible coordinate transforms in Phase 5.

## Summary of Changes
| File | Purpose | Tests |
|------|---------|-------|
| `src/data/segy_reader.py` | IBM float decoder, trace header parser, `RawSpectrum` dataclass, `read_spectrum_raw()` | `tests/test_segy_reader.py` (14 tests) |
| `src/data/preprocessing.py` | Normalization, resize, clip, metadata assembly, save/load | `tests/test_preprocessing.py` (19 tests) |
| `src/data/fk_dataset.py` | PyTorch `FKDataset` with train/val split support | `tests/test_fk_dataset.py` (10 tests) |
| `scripts/preprocess_fk.py` | CLI entry script with `--dry-run`, visualization, manifest generation | Smoke-tested on RL5007 |
| `src/utils/config.py` | Extended with `FKPipelineConfig` dataclass | — |
| `configs/phase1_fk_pipeline.yaml` | Default preprocessing configuration | — |

---

## 0. Architectural Decisions (Locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Normalization** | Per-spectrum min-max to `[-1, 1]`; z-score retained as config flag for ablation | Raw amplitudes are all-positive (`[0.33, 1.0]`). Min-max preserves relative contrast. Ablate against z-score in a later smoke test. |
| **Train/val split** | Hold out **entire receiver lines** (~10% of lines) | Geophysical generalization matters more than IID balance. A model that has never seen spectra from RL5115 or RL5259 is a stronger validation signal. |
| **Storage format** | Individual `.npz` tensor + `.json` sidecar per spectrum | PROJECT_RULES §4.1 mandates JSON sidecars. Inspectability outweighs I/O overhead for ~1,450 spectra. |
| **Output resolution** | `256×256` bilinear | Matches Phase 0 MAE config exactly. Patch size `16×16` → 256 tokens. |
| **Spectrum orientation** | `(wavenumber, freq)` → model input `(1, 256, 256)` | Transposed from raw `(freq, wavenumber)`. Frequency on horizontal axis (fast dimension, x-axis in imshow), wavenumber on vertical axis (slow dimension, y-axis). Matches seismic processing display convention. |

---

## 1. SEG-Y Reader Architecture

### 1.1 Trace Header Mapping
Each trace in the SEG-Y file carries the following custom headers (big-endian):

| Field | Bytes | Type | Decoder | Notes |
|-------|-------|------|---------|-------|
| Elevation | 41–44 | 4-byte signed int | `/ 100.0` | Receiver elevation in meters |
| X coordinate | 81–84 | 4-byte signed int | `/ 100.0` | Receiver X in meters |
| Y coordinate | 85–88 | 4-byte signed int | `/ 100.0` | Receiver Y in meters |
| Station number | 201–204 | 4-byte signed int | — | Composite: first 4 digits = line, last 4 digits = point |
| Frequency | 229–232 | IBM Float-32 | `ibm2ieee()` | Horizontal axis value of this trace (Hz) |
| Min wavenumber | 233–236 | IBM Float-32 | `ibm2ieee()` | Start of vertical axis (1/m); expected `0.0` |
| Max wavenumber | 237–240 | IBM Float-32 | `ibm2ieee()` | End of vertical axis (1/m); expected `0.08` |

### 1.2 File Structure Assumptions
- Each `.sgy` file corresponds to one receiver line (e.g., `RL5007`).
- Traces are **not guaranteed** to be grouped by station in file order.
- Each station contains **262 traces** × **400 samples** = a single FK spectrum.
- Frequency values range `0.0 – 15.93 Hz` with uniform `~0.061 Hz` step.
- Wavenumber axis is linear `0.0 – 0.08 1/m` with `0.0002 1/m` step.

### 1.3 Reader Design
Implement a pure reader (no preprocessing) in `src/data/segy_reader.py`:

- **`read_spectrum_raw(filepath) → dict[str, RawSpectrum]`**: Parse all traces, group by station number, sort each group by frequency ascending, and assemble `(freq_bins, wavenumber_bins)` arrays.
- **Validation gates** (fail fast with descriptive `ValueError`):
  - Every station must have monotonically increasing frequency values.
  - `kmin` must be `0.0` (within float tolerance); `kmax` must be consistent across all traces in file.
  - All stations in a file must share the same `kmax`.
  - Sample count must be exactly `400` for every trace.
- **`ibm2ieee()`** must be implemented as a standalone, unit-tested helper.
- **Memory discipline**: Use `segyio` trace iteration; do not load the entire file into memory as a dense `(tracecount, samples)` array.

### 1.4 RawSpectrum Dataclass
```
RawSpectrum:
  - data: ndarray[float32] of shape (262, 400)
  - station_number: int
  - line_number: int   (derived: station_number // 10000)
  - point_number: int  (derived: station_number % 10000)
  - freq_axis: ndarray[float32] of shape (262,)   — from trace headers
  - waven_axis: ndarray[float32] of shape (400,)  — computed from kmin/kmax/samples
  - elevation: float
  - x_coord: float
  - y_coord: float
  - source_file: str
```

---

## 2. Preprocessing Pipeline Architecture

### 2.1 Pipeline Stages (Reversible)
Each `RawSpectrum` flows through a deterministic, ordered pipeline:

1. **Amplitude normalization** (configurable)
   - Default: `minmax`: `(x - min) / (max - min) * 2 - 1` → `[-1, 1]`
   - Optional: `zscore`: `(x - μ) / (σ + 1e-6)`
   - Store `norm_params` dict in metadata for inverse.

2. **Axis transpose** (from raw SEG-Y to display convention)
   - Raw SEG-Y layout: `(freq_bins, wavenumber_bins)` = `(262, 400)`
   - Transpose to `(wavenumber_bins, freq_bins)` = `(400, 262)` so frequency is on the horizontal axis.
   - This matches seismic processing convention (freq on x, wavenumber on y).

3. **Resize to 256×256**
   - Use `torch.nn.functional.interpolate` with `mode='bilinear', align_corners=False`.
   - Input shape `(400, 262)`, output shape `(256, 256)`.
   - Store `resize_factors` (`256/400`, `256/262`) in metadata. These correspond to `[waven_scale, freq_scale]`.

4. **Dynamic range clipping**
   - `np.clip(x, -3, 3)` after z-score; no-op for min-max (already bounded).
   - Store `clipping_bounds` in metadata.

5. **Metadata assembly**
   - Produce a JSON-serializable metadata dict containing all fields required by PROJECT_RULES §4.1 and §5.1.

### 2.2 PreprocessedSpectrum Dataclass
```
PreprocessedSpectrum:
  - tensor: ndarray[float32] of shape (256, 256)
  - metadata: FKMetadata dict
```

### 2.3 FKMetadata Schema (JSON sidecar)
```json
{
  "spectrum_id": "RL5007_50071009",
  "original_shape": [262, 400],
  "resize_factors": [0.6400, 0.9771],  # [waven_scale, freq_scale] = [256/400, 256/262]
  "freq_axis_original": [0.0, 0.0610, ..., 15.9302],
  "waven_axis_original": [0.0, 0.0002, ..., 0.08],
  "freq_axis_resized": [0.0, ..., 15.9302],   // interpolated axis values
  "waven_axis_resized": [0.0, ..., 0.08],     // interpolated axis values
  "norm_method": "minmax",
  "norm_params": {"min": 0.3353, "max": 0.9998, "mu": 0.6008, "sigma": 0.1234},
  "clipping_bounds": [-3, 3],
  "elevation": 98.30,
  "x_coord": 470510.70,
  "y_coord": 6933223.30,
  "station_number": 50071009,
  "line_number": 5007,
  "point_number": 1009,
  "source_file": "04_09_SWAMI_raw_spect_decim8_RL5007.sgy"
}
```

### 2.4 Design Constraints
- **No mutation of raw data**. Write all outputs to `data/processed/`.
- **Deterministic**. Same source file + same config = identical byte output.
- **Parallelizable**. Each source file can be processed independently.

---

## 3. Dataset & PyTorch Integration

### 3.1 File Layout on Disk
```
data/processed/
├── manifest.json              # dataset manifest: list of all spectra with train/val flag
├── config_snapshot.yaml       # resolved preprocessing config
└── spectra/
    ├── RL5007_50071009.npz    # tensor only (compressed)
    ├── RL5007_50071009.json   # metadata sidecar
    ├── RL5007_50071017.npz
    ├── RL5007_50071017.json
    └── ...
```

### 3.2 `FKDataset` (PyTorch Dataset)
- Reads from the `manifest.json` and loads `.npz` + `.json` pairs on demand.
- Returns `(tensor, metadata)` where `tensor` has shape `(1, 256, 256)`.
- Supports `split='train'` and `split='val'` filtering.
- **No augmentation in Phase 1** — augmentation belongs in the training loop (Phase 2) and must be config-driven.

### 3.3 DataLoader Configuration
- `batch_size`: determined by Phase 2 config; Phase 1 only validates shapes.
- `num_workers`: `4` (CPU-bound I/O).
- `pin_memory=True` for GPU transfer.

---

## 4. Configuration & Scripts

### 4.1 Config Schema
Create `configs/phase1_fk_pipeline.yaml`:

```yaml
# Phase 1: FK Data Pipeline Config
raw_data_dir: "data"
output_dir: "data/processed"
normalization: "minmax"      # or "zscore"
clip_bounds: [-3.0, 3.0]     # applied after normalization
output_size: [256, 256]
interpolation_mode: "bilinear"
align_corners: false

# Train/val split: hold out entire lines
val_lines: [5115, 5259]      # ~10% of 26 lines; exact lines TBD after file inventory
random_seed: 42
```

### 4.2 Entry Script
`scripts/preprocess_fk.py`: argument parsing + invocation of pipeline. Must:
- Log total spectra found, per-line counts, and split sizes.
- Save `config_snapshot.yaml` into `output_dir`.
- Write `manifest.json`.
- Produce a **before/after visualization** of 3–5 random spectra saved to `experiments/YYYY-MM-DD_phase1-fk-pipeline/`.

---

## 5. Testing Plan

### 5.1 Unit Tests (`tests/test_fk_pipeline/`)
- **`test_ibm2ieee`**: Round-trip or known-value tests for IBM float decoder.
- **`test_segy_reader`**: Mock SEG-Y binary (or use a truncated real file) asserting correct trace grouping, sorting, and shape.
- **`test_metadata_completeness`**: Every preprocessed spectrum must contain all required keys.
- **`test_train_val_disjoint`**: No station appears in both splits.

### 5.2 Smoke Test
- Run `scripts/preprocess_fk.py` on a **single source file**.
- Verify output shapes `(256, 256)`, metadata JSON validity, and manifest creation.

### 5.3 Visualization Audit
- Generate a figure panel for each of 5 sampled spectra:
  - **Original** `(262, 400)` with physical axis labels (Hz, 1/m)
  - **Resized** `(256, 256)` with pixel axis labels
  - **Normalized** histogram before/after
- Save to experiment directory as high-DPI PNG (per PROJECT_RULES §8).

---

## 6. Integration with Phase 0 Artifacts

- Re-use `src/utils/seed.py`, `src/utils/device.py`, `src/utils/plot_style.py`.
- Extend `src/utils/config.py` if new config dataclasses are needed (e.g., `FKPipelineConfig`).
- Follow PROJECT_RULES §2.5: document tensor shapes in docstrings.

---

## 7. Success Criteria Gate

| Check | Target | Status | How to Verify |
|-------|--------|--------|---------------|
| Reader correctness | IBM decoder accurate, headers parsed, spectra grouped | ✅ | 14 segy_reader tests pass |
| Preprocessing correctness | Normalization, resize, clip, metadata all correct | ✅ | 19 preprocessing tests pass |
| Dataset integration | `FKDataset` loads splits, returns `(1, 256, 256)` tensors | ✅ | 10 dataset tests pass |
| Shape consistency | Every tensor is `(256, 256)` | ✅ | Verified: sample `(256, 256)`, `float32` |
| Metadata completeness | 100% of files contain all required keys | ✅ | `test_metadata_completeness` |
| Train/val disjointness | Zero overlap | ✅ | `test_split_no_overlap` |
| Coordinate axis validity | `freq_axis_original` monotonic, `waven_axis_original` linear | ✅ | `test_frequency_monotonic` + `test_wavenumber_axis_bounds` |
| Code quality | ruff + ty clean | ✅ | `ruff check .` and `ty check` pass |
| All spectra extracted | ~1,450 `.npz` + `.json` pairs | ✅ | **1,392 spectra** (1,272 train, 120 val) across 25 receiver lines |
| Visual sanity | Before/after panels show expected smoothing, no artifacts | ✅ | Manual review: FK mode structures preserved, normalization range `[-0.98, 0.99]` |

---

## 8. Pending Decisions to Revisit

- [x] **Full pipeline run**: Completed — 1,392 spectra across 25 receiver lines (9×48 + 16×60). Train=1,272, val=120 (lines 5115, 5259).
- [ ] **Ablate normalization**: After Phase 2 smoke test, compare min-max vs. z-score on reconstruction loss.
- [ ] **Val line selection**: Lines 5115 and 5259 are held out. Revisit geographically diverse coverage after plotting receiver line map from X/Y coordinates.
- [ ] **Augmentation**: Random frequency/wavenumber shift, block masking, noise injection — defined in Phase 2 config, not Phase 1.
