# Phase 5: Coordinate Transformation & Inversion Export — TODO

> **Status:** Core implementation complete. Matched forward/inverse coordinate transforms, full-dataset inference, and CSV/JSON export are implemented and tested. Remaining work is validation, optional inversion-format converters, documentation, and final sign-off.
> **Depends on:** Phase 4 (✅ trained model + predictions.npz)
> **Goal:** Reliably convert model-space picks to physical units (Hz, 1/m, phase velocity), validate round-trip accuracy, and deliver inversion-ready dispersion curves with uncertainty estimates.

---

## 0. Inventory of Existing Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Matched coordinate-transform pair | `src/transforms/coordinates.py` | ✅ Implemented |
| Coordinate-transform tests | `tests/test_coordinate_transform.py` | ✅ 33 tests passing |
| Full-dataset inference script | `scripts/phase4_picking/run_inference.py` | ✅ Run on 1,392 spectra |
| CSV/JSON export script | `scripts/phase4_picking/export_dispersion_curves.py` | ✅ Implemented |
| Inference outputs | `experiments/phase4-picking-seq-bilstm-v1/predictions.npz` | ✅ Available |
| Quality triage | `quality_scores.json` + `low_quality_spectra.json` | ✅ Generated |
| Model changelog entry | `experiments/MODEL_CHANGELOG.md` | ✅ Drafted; finalize after validation |

---

## 1. Architectural Decisions (Locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Coordinate basis | Use `freq_axis_resized` / `waven_axis_resized` from metadata | Avoids double interpolation through the original grid; makes round-trip tests exact up to quantization. |
| Uncertainty model | First-order propagation: `0.5 px * local_bin_width / certainty` | Conservative bounds derived from pixel quantization and model certainty; not a calibrated σ. |
| Interpolation for dense inverse | PCHIP (monotone cubic) with linear fallback | Preserves local extrema and avoids cubic-spline overshoot on sparse geophysical picks. |
| Baseline export formats | CSV + JSON | Generic, human-readable, easy to reformat for any inversion package. |
| Optional formats | Geopsy `.disp`, Dinver `.dat` | Out of scope until a specific inversion workflow is chosen. |

---

## 2. Remaining Implementation Tasks

### 2.1 Validation Against Original-Resolution Manual Picks
- [ ] Select 20–50 spectra with high-confidence manual picks on **original-resolution** data.
- [ ] Run the full pipeline: original → preprocess → model pick → inverse transform.
- [ ] Compute and log:
  - Coordinate RMSE in Hz / 1/m and pixel-equivalent.
  - Velocity error `ΔV/V = |(f_pred/k_pred) - (f_manual/k_manual)| / (f_manual/k_manual)`.
  - Coverage and presence/absence agreement.
- [ ] Target: RMSE < 1 pixel equivalent, `ΔV/V` < 0.05 on linear axes.

### 2.2 End-to-End Export Integration Tests
- [ ] Add `tests/test_export_dispersion_curves.py`:
  - Smoke test: run `export_dispersion_curves.py --format both` on a tiny synthetic `predictions.npz`.
  - Verify `manifest.json`, `all_dispersion_curves.csv`, per-spectrum CSV/JSON exist and match expected schema.
  - Verify `--min-certainty` and `--skip-absent` filters reduce row counts correctly.
- [ ] Add regression test for `_EXPORT_COLUMNS` ordering and NaN handling in JSON output.

### 2.3 Optional Inversion-Specific Converters
- [ ] Confirm whether downstream inversion software can consume the generic CSV directly.
- [ ] If **Geopsy** is required: implement a `.disp` writer (typically frequency + velocity pairs, one file per spectrum).
- [ ] If **Dinver** is required: implement a `.dat` writer (check exact column/header convention).
- [ ] Place converters under `src/transforms/inversion_formats.py` and add thin CLI flags to `export_dispersion_curves.py`.

### 2.4 Batch / Config-Driven Export
- [ ] Add an optional `--config` path to `export_dispersion_curves.py` so export settings can be versioned in YAML.
- [ ] Save a resolved config snapshot next to `manifest.json` for reproducibility.

### 2.5 Quality-Control Summary Figures
- [ ] Generate a single-page summary plot after export:
  - Histogram of composite quality scores.
  - Coverage vs. mean certainty scatter.
  - Map view of flagged spectra (X/Y coords colored by review reason).
  - Example curves from top/bottom quality deciles.
- [ ] Save to `exports/dispersion-curves/quality_summary.png`.

### 2.6 Documentation
- [ ] Write a short `docs/coordinate_transform.md` explaining the forward/inverse math, uncertainty propagation, and example CLI usage.
- [ ] Document CSV/JSON export schema and how to reformat for common inversion packages.

### 2.7 Reproducibility & Versioning
- [ ] Save a hash of `src/transforms/coordinates.py` and the model checkpoint path into the export manifest.
- [ ] Add `coordinate_transform_version` field to `manifest.json`.

---

## 3. Testing Plan

| Test | Location | Target |
|------|----------|--------|
| Round-trip on synthetic grids | `tests/test_coordinate_transform.py` | ✅ Already passing (< 0.5 px on linear axes) |
| Round-trip on log-spaced axes | `tests/test_coordinate_transform.py` | ✅ Already passing (< 1.0 px) |
| Real-metadata integration | `tests/test_coordinate_transform.py` | ✅ Verified against `RL5007_50071009.json` |
| End-to-end export smoke | `tests/test_export_dispersion_curves.py` | ⏳ Add |
| Manual-pick validation | New notebook or script under `scripts/phase5/` | ⏳ Add |
| Optional format converters | `tests/test_inversion_formats.py` | ⏳ Add if converters are implemented |

---

## 4. Success Criteria Gate

Before declaring Phase 5 complete:

| Check | Target | How to Verify |
|-------|--------|---------------|
| Round-trip RMSE (model ↔ physical) | < 1 pixel equivalent | `tests/test_coordinate_transform.py` |
| Velocity error on manual picks | `ΔV/V` < 0.05 | Validation script/notebook |
| Export schema stable | Column order and NaN handling unchanged | Regression tests |
| All exports produced | CSV + JSON for all 1,392 spectra | `manifest.json` |
| Low-quality spectra triaged | ~5–10% flagged for review | `quality_scores.json` |
| Code quality | `ruff check .`, `ruff format .`, `ty .` pass | CI / manual |
| Model changelog finalized | Phase 5 entry filled with validation metrics | `experiments/MODEL_CHANGELOG.md` |

If **all pass** → Phase 5 is complete and the pipeline is ready for production inversion runs.

If **round-trip RMSE > 1 px** → inspect axis ordering in metadata and resize-factor application.

If **`ΔV/V` > 0.05** → verify that original-resolution manual picks were transformed consistently with the preprocessing pipeline.

---

## 5. Model Change Tracking

Finalize the existing Phase 5 entry in `experiments/MODEL_CHANGELOG.md` after validation:

```
| 2026-06-15 | phase5-coordinate-transform-v1 | Matched forward/inverse coordinate transform pair. CSV/JSON export of 1,392 spectra. First-order uncertainty propagation from pick certainty. | N/A | round-trip RMSE=?, ΔV/V=? | ✅ Phase 5 complete |
```

---

## 6. Known Issues & Notes

- **Uncertainty values are conservative bounds**, not calibrated Gaussian errors. If inversion software expects true σ, a separate calibration step on manual picks is needed.
- **Geopsy `.disp` / Dinver `.dat` converters are not implemented.** They should be added only when the target inversion package is confirmed.
- **Metadata JSON string in `predictions.npz`** is already parsed correctly by `export_dispersion_curves.py`.

*Last updated: 2026-06-15*
