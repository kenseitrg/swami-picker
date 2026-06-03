# Review: Visualization & Trainer Code Quality

**Scope:** `src/evaluation/visualize.py`, `src/evaluation/__init__.py`, `src/training/trainer.py`, `src/models/mae.py`
**Date:** 2026-06-03
**Tools:** `ruff check`, `ty check`, manual static analysis, runtime edge-case verification

---

## 1. Type Hints

**Correct**
- All new public functions in `src/evaluation/visualize.py` carry full annotations (`model: MaskedAutoencoder`, `images: Tensor`, `save_path: Path`, etc.).
- `plot_umap_embeddings` correctly returns `float | None` (`src/evaluation/visualize.py:182`).
- `extract_embeddings` in `src/models/mae.py:430` is annotated `-> Tensor` and documents input/output shapes per §2.5.
- No `nn.Module` annotations appear where `MaskedAutoencoder` is the actual expectation.

**Suggestion**
- `MAETrainer._autocast_context` (`src/training/trainer.py:142`) lacks a return type annotation. It returns either `torch.amp.autocast` or `nullcontext()`, so the return type should be `AbstractContextManager[Any]` (or at minimum `Any`).

---

## 2. Lazy Imports

**Correct**
- `MAETrainer._run_visualization` (`src/training/trainer.py:333-338`) and `_plot_final_curves` (`src/training/trainer.py:365`) both import from `src.evaluation.visualize` lazily inside the method bodies. This is appropriate: it defers the heavy dependencies (`numpy`, `sklearn`, `umap`) until the first visualization epoch rather than paying the import cost at trainer module-load time.
- No other module-level imports in the reviewed files should be made lazy.

**Note**
- `src/training/trainer.py:29` imports `apply_style` from `src.utils.plot_style` at module level. `plot_style.py` in turn imports `matplotlib.pyplot`, so **matplotlib is already loaded when the trainer module is imported**. The lazy import therefore does not fully achieve its goal of deferring *all* visualization dependencies, but it still successfully defers `sklearn` and `umap`, which are the heaviest.

---

## 3. `__init__.py` Files

**Correct**
- `src/evaluation/__init__.py` starts with `from __future__ import annotations` (line 1).
- `__all__` is explicitly declared and contains exactly the four public symbols re-exported from `visualize.py`:
  - `plot_loss_curves`
  - `plot_masking_examples`
  - `plot_reconstruction_grid`
  - `plot_umap_embeddings`

---

## 4. Unused Imports

**Correct**
- `ruff check .` passes cleanly across all four files.
- Manual AST walk confirmed no dead imports at module level in any of the reviewed files.

---

## 5. Error Handling

### BLOCKER: `_validate` crashes on empty `val_loader`
- **File:** `src/training/trainer.py:303-311`
- **Issue:** If `self.val_loader` yields no batches, `num_batches` remains `0` and the division `avg_loss = total_loss / num_batches` raises `ZeroDivisionError`.
- **Evidence:** Runtime test with `DataLoader(TensorDataset(torch.empty(0, ...), torch.empty(0,)))` reproduced the crash.

### BLOCKER: `_run_visualization` crashes on empty `val_loader`
- **File:** `src/training/trainer.py:348`, `src/training/trainer.py:353`
- **Issue:** `next(iter(self.val_loader))[0]` raises `StopIteration` when the loader is empty.
- **Evidence:** Runtime test reproduced `StopIteration`.

### FIX: `plot_loss_curves` does not guard against missing or malformed metrics file
- **File:** `src/evaluation/visualize.py:249-252`
- **Issue:** `open(metrics_path)` raises `FileNotFoundError` if the file is missing; `json.loads(line)` raises `JSONDecodeError` on malformed lines; missing keys (e.g. `val_loss`) raise `KeyError`.
- **Evidence:** Runtime tests confirmed all three exception types.
- **Recommendation:** Wrap the file read in a `try/except (FileNotFoundError, json.JSONDecodeError)` and log a warning, or assert existence up front.

### FIX: `plot_umap_embeddings` crashes on empty `DataLoader`
- **File:** `src/evaluation/visualize.py:212-214`
- **Issue:** If `loader` yields no batches, `torch.cat(all_embs)` raises `ValueError: expected a non-empty list of Tensors`.
- **Evidence:** Runtime test with empty `DataLoader` reproduced the crash.

### FIX: `plot_masking_examples` crashes when `num_samples == 1`
- **File:** `src/evaluation/visualize.py:117-118`
- **Issue:** `plt.subplots(2, 1)` with default `squeeze=True` returns a 1-D `(2,)` array. The subsequent `axes[0, i]` indexing raises `IndexError: too many indices for array`.
- **Evidence:** `python -c "import matplotlib.pyplot as plt; fig, axes = plt.subplots(2, 1); axes[0, 0]"` reproduces the error.
- **Recommendation:** Add the same reshape guard used in `plot_reconstruction_grid`:
  ```python
  if num_samples == 1:
      axes = axes.reshape(2, 1)
  ```

---

## 6. Ruff / `ty` Compliance

**Correct**
- `ruff check src/evaluation/visualize.py src/evaluation/__init__.py src/training/trainer.py src/models/mae.py` → **All checks passed.**
- `ty check` on the same files → **All checks passed.**

**Note**
- While `ty` is clean, the missing return type on `_autocast_context` (see §1) is something a stricter configuration might flag in future.

---

## 7. Docstrings

**Correct**
- Google-style docstrings are used consistently across all public functions, classes, and methods.
- `Args:`, `Returns:`, and `Raises:` blocks are present where required.
- Tensor shapes are documented in the §2.5 format (e.g. `(B, C, H, W)`, `(B, N, D)`) in:
  - `src/models/mae.py` (`patchify`, `unpatchify`, `random_masking`, `block_masking`, `forward_encoder`, `forward_decoder`, `forward_loss`, `forward`, `extract_embeddings`)
  - `src/evaluation/visualize.py` (`_create_masked_image`, `plot_reconstruction_grid`, `plot_masking_examples`, `plot_umap_embeddings`)

**Suggestion**
- `extract_embeddings` docstring says the output shape is `(B, embed_dim)` (`src/models/mae.py:440`). The class does **not** store `self.embed_dim`; the value is only available as `self.patch_embed.out_features`. It would be clearer to write `(B, patch_embed.out_features)` or add `self.embed_dim = embed_dim` in `__init__` for self-documentation.

---

## 8. Logging

**Correct**
- No `print()` statements exist in any of the reviewed files.
- Log levels are appropriate:
  - `INFO` for saved-figure confirmations and epoch summaries.
  - `WARNING` for skipped UMAP when the package is unavailable (`src/evaluation/visualize.py:189`).
  - `DEBUG` for batch-level metrics in `trainer.py`.

---

## Summary Table

| Finding | File | Line(s) | Class |
|---------|------|---------|-------|
| `_validate` ZeroDivisionError on empty val_loader | `trainer.py` | 303-311 | **BLOCKER** |
| `_run_visualization` StopIteration on empty val_loader | `trainer.py` | 348, 353 | **BLOCKER** |
| `plot_loss_curves` unhandled FileNotFoundError / JSONDecodeError / KeyError | `visualize.py` | 249-252 | **FIX** |
| `plot_umap_embeddings` crash on empty DataLoader | `visualize.py` | 212-214 | **FIX** |
| `plot_masking_examples` IndexError when `num_samples == 1` | `visualize.py` | 117-118 | **FIX** |
| `_autocast_context` missing return type | `trainer.py` | 142 | SUGGESTION |
| `extract_embeddings` docstring references unset `embed_dim` | `mae.py` | 440 | SUGGESTION |
| `apply_style` module-level import undermines lazy import | `trainer.py` | 29 | NOTE |
| All public functions fully typed | various | — | CORRECT |
| `__init__.py` has `from __future__ import annotations` | `__init__.py` | 1 | CORRECT |
| Exports complete in `__all__` | `__init__.py` | 11-16 | CORRECT |
| Ruff & `ty` pass cleanly | all | — | CORRECT |
| Google-style docstrings with shapes | all | — | CORRECT |
| No print statements | all | — | CORRECT |
