## Review: Visualization & Embedding Extraction Code

**Scope:** `src/evaluation/visualize.py`, `src/models/mae.py`, `src/training/trainer.py`, `src/utils/plot_style.py`  
**Date:** 2026-06-03  
**Reviewer:** sub-agent

---

### 1. `extract_embeddings()` — `src/models/mae.py` (lines 384–403)

**Correct**
- Runs the encoder **without masking**: the body is identical to `forward_encoder` minus the masking branch (`random_masking` / `block_masking`).
- **Positional embeddings are added**: `x = x + self.pos_embed` (line 390), matching `forward_encoder` (line 305).
- **Mean-pooling is correct**: `return x.mean(dim=1)` (line 397) pools over the patch dimension `N`, yielding shape `(B, embed_dim)`.
- Verified by direct code comparison with `forward_encoder` (lines 299–312).

**Note**
- No unit tests cover `extract_embeddings`. A simple shape/consistency test (e.g. output shape `(B, embed_dim)`, no gradients, deterministic for same input) is missing.

---

### 2. `plot_reconstruction_grid()` — `src/evaluation/visualize.py` (lines 53–89)

**Correct**
- Calls `model.eval()` + `torch.no_grad()` before inference (lines 65–66).
- Uses `model.unpatchify(pred)` to turn decoder patch predictions back into image tensors (line 68).
- `_create_masked_image()` correctly expands the 1-D mask `(B, N)` to pixel resolution `(B, 1, H, W)` via `repeat_interleave(patch_size, ...)` (lines 31–35).
- Handles the `num_samples == 1` edge case by reshaping the axes array (lines 70–71).

**Fix**
- **Misleading per-panel intensity scaling** (lines 75–76):
  ```python
  vmin, vmax = img.min(), img.max()
  ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
  ```
  Each panel (Input, Masked, Reconstructed) gets its own independent contrast stretch. This makes reconstruction quality impossible to judge visually because the "Reconstructed" panel is always auto-scaled to fill its own dynamic range.  
  **Location:** `src/evaluation/visualize.py`, lines 75–76.  
  **Resolution:** Use a shared `vmin/vmax` per row (e.g. based on the original input image) so that amplitude differences are faithfully represented.

---

### 3. `plot_masking_examples()` — `src/evaluation/visualize.py` (lines 92–121)

**Fix**
- **Crash when `num_samples == 1`** (lines 100, 104, 108, 112):
  ```python
  fig, axes = plt.subplots(2, num_samples, ...)
  axes[0, i].imshow(...)
  ```
  When `num_samples == 1`, `plt.subplots(2, 1)` returns a 1-D `ndarray` of shape `(2,)`. Indexing with `axes[0, i]` raises `IndexError: too many indices for array`.  
  **Location:** `src/evaluation/visualize.py`, line 100.  
  **Resolution:** Add the same `squeeze=False` or reshape guard used in `plot_reconstruction_grid`:
  ```python
  if num_samples == 1:
      axes = axes.reshape(2, -1)
  ```

**Note**
- Does not accept a `seed` parameter (unlike `plot_reconstruction_grid`). If the input batch order changes, the examples are not reproducible. Add a `seed` argument and use `rng.choice` or deterministic slicing.

---

### 4. `plot_umap_embeddings()` — `src/evaluation/visualize.py` (lines 124–193)

**Correct**
- **Embedding extraction pipeline is correct**: calls `model.extract_embeddings(images)` inside `torch.no_grad()` (line 154), runs on GPU, and immediately moves results to CPU (`embs.cpu()`, line 155) — good VRAM hygiene.
- **Silhouette score is computed on the high-dimensional embeddings** (line 166), not on the 2-D UMAP projection. This is the statistically sound choice.
- **Labels are correctly threaded** to both the scatter plot (`c=labels`, line 172) and `silhouette_score(embeddings, labels)`, line 166).
- **UMAP availability is handled gracefully**: returns `None` and logs a warning when `umap` is not installed (lines 131–134).
- Correctly truncates to `max_samples` (lines 159–160).

**Note**
- Sampling from the DataLoader is not seeded inside this function. Reproducibility depends on the caller ensuring a deterministic loader or fixed seed.

---

### 5. `plot_loss_curves()` — `src/evaluation/visualize.py` (lines 196–248)

**Correct**
- Parses `metrics.jsonl` line-by-line with `json.loads` (lines 204–206).
- All four requested panels are present and correctly mapped:
  - Top-left: Train/Val MSE loss (lines 212–217)
  - Top-right: Learning rate (log scale) (lines 219–223)
  - Bottom-left: Peak VRAM (lines 225–229)
  - Bottom-right: Throughput (lines 231–235)
- Uses `get(..., 0.0)` for optional keys (`max_vram_mb`, `throughput_samples_per_sec`), so missing fields do not crash the plotter.

**Note**
- `ax.set_xticks(epochs)` (line 240) on every axis can produce unreadably crowded tick labels for long training runs (>50 epochs). Consider tick thinning or `MaxNLocator`.

---

### 6. Integration in `MAETrainer._run_visualization()` — `src/training/trainer.py` (lines 334–369)

**Correct**
- **Lazy imports** are performed inside the method (lines 348–352), avoiding heavy import-time dependencies.
- **UMAP absence is handled**: `plot_umap_embeddings` returns `None` when UMAP is missing; the caller checks `if sil_score is not None` before logging (lines 366–367).
- **Plots are saved to the run directory**: `plot_dir = self.run_dir / "plots"` (line 354).
- Generation epochs are `is_first`, `is_target` (0-based epochs 2 and 4, i.e. 1-based epochs 3 and 5), and `is_final`. This is a reasonable cadence.

**Note**
- **Redundant `apply_style()` call**: `_run_visualization` calls `apply_style()` (line 357) and then every plotting function calls it again internally. Harmless but unnecessary.
- **Repeated `next(iter(self.val_loader))`** (lines 360, 363) spawns a fresh DataLoader iterator each time. With `num_workers > 0` this incurs worker-process creation overhead. Caching a single validation batch would be more efficient.
- **Confusing comment** (line 336): `# epochs 3 and 5 (0-based)` should read `# 0-based indices 2 and 4 -> 1-based epochs 3 and 5`.

---

### 7. Performance Issues

**Correct / Acceptable**
- No unnecessary GPU→CPU transfers were found. Embeddings are moved to CPU immediately after extraction (line 155). Reconstruction tensors are moved to CPU right after `unpatchify` (line 68).
- No redundant forward passes or duplicate model evaluations inside the plotting functions.

**Suggestion**
- In `_run_visualization`, caching the validation batch (`sample_images`) between `is_target` and `is_final` would avoid a second `next(iter(...))` call when both conditions are true (e.g. if `epochs == 3` and the total run is 3 epochs).

---

### 8. Figures & Style Adherence (`PROJECT_RULES.md` §8)

**Correct**
- **All figures call `apply_style()`** (`plot_reconstruction_grid`, `plot_masking_examples`, `plot_umap_embeddings`, `plot_loss_curves`).
- **All figures are saved headlessly** — no `plt.show()` anywhere. Each function ends with `save_figure(...)` + `plt.close(fig)`.
- **Saved to the correct run directory** (`self.run_dir / "plots"`).
- **High-resolution PNG exports**: `savefig.dpi` is 300 in `_DEFAULT_STYLE` (`src/utils/plot_style.py`, line 18).

**Note**
- `plot_reconstruction_grid` violates §8 "Clarity over embellishment" because independent per-panel `vmin/vmax` scaling makes reconstruction errors invisible. A shared intensity scale per row is needed for publication-quality comparison.

---

## Summary Table

| # | Item | Location | Classification |
|---|------|----------|----------------|
| 1 | `extract_embeddings` encoder path, pos-embed, mean-pool | `src/models/mae.py:384-403` | **Correct** |
| 2 | Independent `vmin/vmax` per panel hides reconstruction quality | `src/evaluation/visualize.py:75-76` | **Fix** |
| 3 | `plot_masking_examples` crashes when `num_samples=1` | `src/evaluation/visualize.py:100` | **Fix** |
| 4 | UMAP pipeline, Silhouette on high-dim embeddings, labels | `src/evaluation/visualize.py:124-193` | **Correct** |
| 5 | All four loss-curve panels present and parsed correctly | `src/evaluation/visualize.py:196-248` | **Correct** |
| 6 | Lazy imports, UMAP handling, run-dir save | `src/training/trainer.py:334-369` | **Correct** |
| 7 | No harmful CPU/GPU transfer overhead | — | **Correct** |
| 8 | `apply_style()`, headless save, run-directory | `src/utils/plot_style.py` + callers | **Correct** |

**No blockers found.** The two fixes above should be applied before the code is considered fully robust.
