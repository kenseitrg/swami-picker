# Review: Visualization Integration in Training Loop

**Scope:** `src/training/trainer.py`, `src/evaluation/visualize.py`, `src/models/mae.py`, `configs/phase0_mnist.yaml`, `todos/phase0_mnist.md`  
**Date:** 2026-06-03

---

## 1. DataLoader Exhaustion — `next(iter(self.val_loader))`

**Finding:** The DataLoader is **not exhausted**. Each call to `iter(self.val_loader)` instantiates a brand-new `_BaseDataLoaderIter`, so every `next(iter(...))` simply returns the first batch again. Because the test loader uses `shuffle=False` (`src/data/mnist_dataset.py:76`), the returned batch is deterministic (the first 8 MNIST test images in dataset order).

**Evidence:**
- `src/training/trainer.py:336` — `sample_images = next(iter(self.val_loader))[0]` (first epoch)
- `src/training/trainer.py:341` — same pattern (target / final epochs)
- `src/data/mnist_dataset.py:76` — `test_loader = DataLoader(..., shuffle=False, ...)`

**Verdict:** Not a bug, but the same batch is reused every time.  
**Classification:** SUGGESTION — Consider caching one sample batch in the trainer (e.g. `self._sample_batch = next(iter(self.val_loader))[0].cpu()`) so that `_run_visualization` does not reconstruct an iterator on every target epoch. This also guarantees the exact same pixels are shown across epochs, making visual comparisons cleaner.

---

## 2. Epoch Counting vs. TODO Requirement

**Finding:** The epoch logic is **correct** for the 5-epoch config and matches the TODO.

**Evidence:**
- `src/training/trainer.py:325` — `plot_epochs = {2, 4}` (0-based)
- `configs/phase0_mnist.yaml:24` — `epochs: 5`
- Loop runs 0,1,2,3,4.
  - Epoch 0 (1-based 1): `is_first` → masking examples
  - Epoch 2 (1-based 3): `is_target` → reconstruction + UMAP
  - Epoch 4 (1-based 5): `is_target` and `is_final` → reconstruction + UMAP
- `todos/phase0_mnist.md` step 6: "Reconstruction grid (epoch 3 & 5)"

**Verdict:** Reconstruction and UMAP run exactly at 1-based epochs 3 and 5.  
**Classification:** CORRECT.

---

## 3. Checkpointing Order — `_save_checkpoint` Before `_run_visualization`

**Finding:** Checkpointing **before** visualization is the **safer** ordering. `save_checkpoint()` (`src/utils/checkpoint.py:24`) calls `torch.save(state, path)`, which blocks until the file is fully written and flushed. If `_run_visualization` subsequently crashes (OOM, plotting error, etc.), the checkpoint on disk remains intact.

**Evidence:**
- `src/training/trainer.py:173` — `self._save_checkpoint(epoch, is_best=is_best)`
- `src/training/trainer.py:189` — `self._run_visualization(epoch)`
- `src/utils/checkpoint.py:24` — `torch.save(state, path)` is synchronous.

**Verdict:** No risk of checkpoint corruption. The opposite order (viz before save) would risk losing the epoch’s checkpoint if visualization failed.  
**Classification:** CORRECT.

---

## 4. Memory During Visualization — `extract_embeddings` + UMAP

**Finding:** For the Phase-0 config, OOM is **extremely unlikely**.

**Evidence:**
- `src/evaluation/visualize.py:213-222` — `plot_umap_embeddings` streams batches, immediately moving embeddings to CPU (`embs.cpu()`). Only one batch of images lives on the GPU at a time.
- `configs/phase0_mnist.yaml` — `batch_size: 8`, `image_size: 256`, `embed_dim: 384`
- `max_samples: 2000` (`src/evaluation/visualize.py:183`) caps the total embeddings to ~3 MB (2000 × 384 × 4 bytes).
- UMAP’s memory footprint on 2000 samples is well below 1 GB.
- `src/training/trainer.py:161-163` — `torch.cuda.empty_cache()` is called after validation and before visualization.

**Edge case:** If a user later increases `embed_dim` (e.g. to 768) and keeps `max_samples=2000`, embeddings grow to only ~6 MB. If `max_samples` itself is raised significantly, UMAP memory grows super-linearly.

**Classification:** SUGGESTION — Add a runtime guard in `plot_umap_embeddings` that `max_samples >= n_neighbors` (UMAP defaults to 15). Passing `max_samples < 15` would cause a UMAP runtime error. A simple `max_samples = max(max_samples, n_neighbors)` or assertion would harden the code for future configs.

---

## 5. CPU Fallback

**Finding:** Visualization **works on CPU** without modification.

**Evidence:**
- All three viz functions accept a generic `device: torch.device` and move tensors with `.to(device)`.
- `src/evaluation/visualize.py:139` — `imgs = images[indices].to(device)`
- `src/evaluation/visualize.py:186` — `images = images.to(device)`
- UMAP operates on NumPy arrays on CPU.
- The trainer guards every CUDA-specific call (`torch.cuda.reset_peak_memory_stats`, `torch.cuda.empty_cache`, `torch.cuda.get_rng_state_all`) behind `if self.device.type == "cuda"`.

**Verdict:** No CUDA-specific code in the viz path.  
**Classification:** CORRECT.

---

## 6. Reproducibility

**Finding:** Most random operations are seeded, but **UMAP uses a hard-coded `random_state=42`** instead of the config seed.

**Evidence:**
- `src/utils/seed.py` — `set_seed(config.seed)` is called in `scripts/train_mae.py:37` before DataLoader creation.
- Test loader has `shuffle=False`, so `next(iter(self.val_loader))` is deterministic.
- `src/evaluation/visualize.py:135` — `rng = np.random.default_rng(seed)` (seed comes from `self.config.seed` via `_run_visualization`).
- `src/evaluation/visualize.py:217` — `umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)` hardcodes `42`.
- `plot_masking_examples` uses deterministic slicing (`images[:num_samples]`), no RNG.

**Impact:** If a user changes `seed` in the YAML config, the UMAP layout will still be identical to any other run because UMAP ignores the config seed. This undermines the reproducibility contract.

**Classification:** FIX — Pass `seed=self.config.seed` into `plot_umap_embeddings` and forward it to `umap.UMAP(random_state=seed, ...)`.  
**Secondary note:** The model’s masking (`torch.rand` inside `forward_encoder`) depends on the global RNG state, which is seeded and restored on resume. Because the number of training steps per epoch is fixed for a given config, mask patterns during visualization are reproducible across fresh runs.

---

## 7. File Overwriting

**Finding:** Per-epoch filenames are **unique** and safe from collision.

**Evidence:**
- `src/training/trainer.py:343` — `f"reconstruction_epoch_{epoch + 1:03d}.png"` → e.g. `reconstruction_epoch_003.png`
- `src/training/trainer.py:345` — `f"umap_epoch_{epoch + 1:03d}.png"` → e.g. `umap_epoch_003.png`
- Zero-padding to 3 digits handles up to epoch 999.

**Edge case on resume:**
- `src/training/trainer.py:327` — `is_first = epoch == self.start_epoch`
- If training resumes from epoch 3, `start_epoch = 3`, so `is_first` is `True` on the resumed first epoch. `masking_examples.png` is written again, overwriting the file from the original run.

**Classification:** SUGGESTION — Either cache `masking_examples.png` and skip rewriting when the file already exists, or rename it to include the epoch (`masking_examples_epoch_{epoch+1:03d}.png`) so resume runs do not clobber prior outputs.

---

## 8. Model State — Eval Mode During Visualization

**Finding:** Running visualization after `_validate` is **safe**.

**Evidence:**
- `src/training/trainer.py:288` — `_validate` calls `self.model.eval()`.
- `src/evaluation/visualize.py:102, 144, 206` — `plot_masking_examples`, `plot_reconstruction_grid`, and `plot_umap_embeddings` each redundantly call `model.eval()`.
- `src/evaluation/visualize.py:208` — `extract_embeddings` is invoked inside a `torch.no_grad()` context and after `model.eval()`.
- `extract_embeddings` (`src/models/mae.py:403-420`) does not use dropout or batch normalization, so eval mode is not strictly required for correctness, but the current code ensures it.
- `src/training/trainer.py:234` — `_train_epoch` calls `self.model.train()` at the start of the next epoch, so there is no train/eval state leak.

**Verdict:** No risk of model state corruption or unintended gradient accumulation.  
**Classification:** CORRECT.

---

## Summary Table

| # | Topic | Classification | Location |
|---|-------|----------------|----------|
| 1 | DataLoader exhaustion | SUGGESTION | `trainer.py:336,341` |
| 2 | Epoch counting | CORRECT | `trainer.py:325-330` |
| 3 | Checkpoint before viz | CORRECT | `trainer.py:173,189` |
| 4 | Memory / OOM safeguard | SUGGESTION | `visualize.py:217` |
| 5 | CPU fallback | CORRECT | `visualize.py` (device-agnostic) |
| 6 | Reproducibility (UMAP seed) | FIX | `visualize.py:217` |
| 7 | File overwriting on resume | SUGGESTION | `trainer.py:327,339` |
| 8 | Model eval mode | CORRECT | `trainer.py:288`, `visualize.py:102,144,206` |

---

## Overall Assessment

The integration is **solid and safe** for the Phase-0 smoke test. No blockers were found. The only actionable item is the hard-coded UMAP `random_state=42`, which should consume `config.seed` to respect the user’s reproducibility setting. The remaining items are defensive suggestions (cache sample batch, guard `max_samples >= n_neighbors`, avoid overwriting masking examples on resume) that improve robustness but do not prevent correct operation.
