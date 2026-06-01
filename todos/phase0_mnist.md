# Phase 0: MNIST Smoke Test — TODO

## Goal
Validate the complete MAE training stack on a clean dataset before introducing FK spectral complexity.

---

## 1. Project Scaffolding

- [x] Create `src/` subdirectories: `models/`, `data/`, `training/`, `utils/`, `evaluation/`
- [x] Create `configs/`, `scripts/`, `tests/`, `experiments/` directories
- [x] Add `__init__.py` files where needed for package imports
- [x] Create `experiments/MODEL_CHANGELOG.md` with header template

---

## 2. Configuration & Utilities

- [x] Define `MNISTConfig` dataclass (image size, patch size, mask ratio, batch/accum, LR, epochs, seed)
- [x] Write `src/utils/seed.py`: `set_seed(seed: int)` helper setting `torch`, `numpy`, `random`, `torch.cuda`
- [x] Write `src/utils/device.py`: `get_device()` returning `torch.device` with CPU fallback
- [x] Write `src/utils/checkpoint.py`: `save_checkpoint()` and `load_checkpoint()` using state-dict dicts
- [x] Write `src/utils/plot_style.py`: unified matplotlib style sheet (fonts, palette, line weights, figure size) for publication-ready figures
- [x] Save a sample config to `configs/phase0_mnist.yaml`

---

## 3. Data Pipeline

- [x] Write `src/data/mnist_dataset.py`: torchvision MNIST download → resize 256×256 bilinear → normalize to zero-mean
- [x] Configure DataLoader with `batch_size=8`, `num_workers=0` (MNIST is small), `pin_memory=True`
- [x] Verify output tensor shape: `(B, 1, 256, 256)`

---

## 4. MAE Model Skeleton

### 4.1 Patching
- [x] Implement `patchify(x: Tensor) -> Tensor`: `(B, C, H, W) → (B, N, patch_dim)` where `N = (H/p)×(W/p)`
- [x] Implement `unpatchify(x: Tensor) -> Tensor`: reverse mapping

### 4.2 Masking
- [x] Implement `random_masking(x: Tensor, mask_ratio: float) -> (x_masked, mask, ids_restore)`
- [x] Implement `block_masking(x: Tensor, mask_ratio: float, block_size: int = 2) -> ...` (2×2 patch groups)
- [x] Default to **block masking** for this phase

### 4.3 Encoder
- [x] Build ViT-style encoder: `nn.Linear` patch embedding + positional embeddings + Transformer blocks
- [x] Config: `embed_dim=384`, `depth=12`, `num_heads=6`, `mlp_ratio=4`
- [x] Encoder must process **only unmasked tokens** to save compute/VRAM

### 4.4 Decoder
- [x] Lightweight decoder: re-add mask tokens → small Transformer (e.g., `depth=4`, `embed_dim=256`) → linear head to `patch_dim`
- [x] Decoder reconstructs **all patches**, loss computed only on masked ones

### 4.5 Forward Pass
- [x] `MAE.forward(x)` → `loss, pred, mask` using MSE on masked patches

---

## 5. Training Loop

### 5.1 Setup
- [ ] Instantiate model, move to device
- [ ] Optimizer: `AdamW(..., fused=True)`, LR `1e-4`, betas `(0.9, 0.95)`, weight decay `0.05`
- [ ] Scheduler: 10% linear warmup → cosine decay to 10% of peak
- [ ] AMP: `torch.amp.autocast("cuda", dtype=torch.float16)` + `GradScaler()`
- [ ] Gradient clipping: `nn.utils.clip_grad_norm_(..., max_norm=1.0)`

### 5.2 Loop Structure
```
for epoch in epochs:
    for batch in loader:
        with autocast(): loss = model(batch)
        scaler.scale(loss / accum_steps).backward()
        if (step + 1) % accum_steps == 0:
            clip_grad_norm_()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
```
- [ ] Log loss, LR, and VRAM peak every epoch

### 5.3 Validation
- [ ] Run reconstruction loss on held-out test set every epoch
- [ ] Track best validation loss for checkpointing

---

## 6. Visualization & Embedding Checks

All figures must use `src/utils/plot_style.py`, be reproducible (seeded sampling), and saved as high-DPI PNG or vector graphics to the experiment run directory.

- [ ] **Reconstruction grid** (epoch 3 & 5): input / masked / reconstructed side-by-side; before/after panels demonstrating the masking transformation clearly
- [ ] **UMAP/t-SNE** of encoder outputs on test set, colored by digit label; demonstrate 10 visually distinct clusters without supervision
- [ ] **Loss & LR curves**: training and validation loss, learning rate schedule, and VRAM usage over epochs on a single multi-panel figure
- [ ] **Masking visual examples**: show block masking (2×2) applied to a sample batch to make the masking strategy unambiguous
- [ ] Log Silhouette score (approximate) as sanity metric; include it as an annotation on the embedding plot

---

## 7. Checkpointing & Resume

- [ ] Save checkpoint dict with: `model`, `optimizer`, `scaler`, `scheduler`, `epoch`, `step`, `seed`, `config`
- [ ] Implement resume: load checkpoint → restore all states → verify loss curve continuity for 2 steps
- [ ] Verify no regression in loss after resume

---

## 8. VRAM & Performance Profiling

- [ ] Log `torch.cuda.max_memory_allocated()` after every epoch
- [ ] Log throughput: samples/sec averaged over epoch
- [ ] **Target:** Peak VRAM < 4.5 GB
- [ ] If VRAM > 5.5 GB: reduce patch size to 8 or disable gradient accumulation as fallback

---

## 9. Success Criteria Gate

| Check | Target | Status |
|-------|--------|--------|
| Loss curve monotonic decrease | Low MSE by epoch 5, no NaN/Inf | ⬜ |
| Reconstruction quality | Digits recognizable, sharp edges | ⬜ |
| Embedding clusters | 10 UMAP clusters | ⬜ |
| VRAM peak | < 4.5 GB | ⬜ |
| Checkpoint resume | Identical loss curve | ⬜ |

- [ ] **If ALL pass** → approve Phase 0, update `MODEL_CHANGELOG.md`, proceed to Phase 1
- [ ] **If ANY fail** → diagnose, document failure mode, fix, re-run

---

## 10. Documentation

- [ ] Add docstrings to all public functions/classes (Google style)
- [ ] Run `ruff check .` and `ruff format .` — must pass
- [ ] Run `ty check .` — must pass
