# Project Plan: Self-Supervised FK Spectrum Analysis & Robust Dispersion Picking

## 🎯 1. Project Overview
| Aspect | Description |
|--------|-------------|
| **Objective** | Develop a robust, human-in-the-loop pipeline to extract fundamental-mode dispersion curves from noisy, highly variable FK spectra for downstream inversion. |
| **Key Challenges** | No ground-truth labels, higher-order modes often dominate, rapid near-surface variability, limited compute (RTX 3060, 6GB VRAM), coordinate system mismatch between model and inversion software. |
| **Proposed Solution** | Two-stage self-supervised learning: (1) Masked Autoencoder (MAE) pretraining for physics-aware embeddings, (2) Prototype-based clustering → active expert labeling → supervised fine-tuning. |
| **Expected Output** | Automated picking model with uncertainty estimates, exportable dispersion curves in original physical units, reusable embedding encoder for future datasets. |

---

## 🔬 Phase 0: Architecture Verification (MNIST Smoke Test)
**Purpose:** Validate the complete MAE training stack (data pipeline, patching, masking, AMP, gradient accumulation, optimizer, scheduler, and reconstruction head) using a clean, well-understood dataset before introducing FK spectral complexity.

### Configuration
| Component | Setting | Rationale |
|-----------|---------|-----------|
| **Dataset** | `torchvision.datasets.MNIST` | Download automatically, 60k train / 10k test |
| **Resize** | `256×256` (bilinear, single-channel) | Matches exact FK pipeline dimensions; validates scaling & memory footprint |
| **Patch Size** | `16×16` | Yields 256 tokens, identical to FK config |
| **Mask Ratio** | `0.75` (block masking 2×2) | Tests high-masking logic & gradient flow |
| **Batch Size** | `8` + gradient accumulation (`accum=4`) → eff. 32 | Validates VRAM ceiling under realistic load |
| **Epochs** | `5` | Fast convergence expected; catches bugs early |
| **Optimizer** | `AdamW (fused=True)`, LR `1e-4`, warmup 10%, cosine decay | Mirrors Stage 1 config exactly |

### Evaluation & Success Criteria
| Check | Metric | Target |
|-------|--------|--------|
| **Training Stability** | Loss curve monotonic decrease | low MSE by epoch 5, no NaN/Inf |
| **Reconstruction Quality** | Visual side-by-side (input vs masked vs reconstructed) | Digits recognizable, edges sharp, no checkerboard artifacts |
| **Embedding Structure** | UMAP/t-SNE of encoder outputs (colored by digit label) | 10 distinct clusters emerge without supervision |
| **Embedding Separability** | Mean cosine-similarity contrast (intra-class / inter-class) | > 2.5 (required for downstream clustering) |
| **VRAM Profile** | `torch.cuda.max_memory_allocated()` | < 4.5 GB (leaves headroom for FK data + augmentations) |
| **Pipeline Integrity** | Checkpoint save/load + resume | Identical loss curve on resume |

### Execution Steps
1. Download MNIST, apply resize + normalization pipeline identical to FK preprocessing
2. Run 5-epoch training with exact AMP/fused/accumulation flags planned for Stage 1
3. Generate reconstruction grid + UMAP plot after epoch 3 & 5
4. Log VRAM, throughput (samples/sec), and loss history
5. ✅ **Proceed to Phase 1 only if all success criteria are met**

### Architecture Exploration: CvT-MAE
A Convolutional Vision Transformer (CvT) variant was implemented and evaluated as a potential encoder replacement. The CvT-MAE uses depth-wise convolutional projections for Q/K/V in self-attention and processes the **full token grid** (no token dropping during encoding), with masked patches replaced by a learnable mask token before the encoder.

| Metric | ViT-MAE Baseline | CvT-MAE | Assessment |
|--------|-----------------|---------|------------|
| Best Val Loss | 0.271 | **0.198** | ✅ Better reconstruction |
| Silhouette Score | **0.18** | 0.032 | ❌ Catastrophic collapse |
| Intra/Inter Contrast | **3.70** | 1.07 | ❌ Embeddings not separable |
| VRAM Peak | **777 MB** | 2047 MB | ❌ 2.6× overhead |
| Throughput | ~60 s/s | ~73 s/s | ~ Parity |

**Diagnosis:** The CvT encoder's depth-wise convolutions mix the learned mask token with neighboring visible patches, creating a "homogenizing" effect that destroys class-discriminative signal in the embeddings. The reconstruction objective improves, but the embedding space becomes nearly uniform (all digits have ~0.82 cosine similarity).

**Decision:** ❌ **CvT-MAE is not viable for this pipeline.** Phase 3 prototype clustering and active learning depend on embeddings with high intra/inter contrast. The ViT-MAE baseline is retained as the architecture of record for Phase 1 FK pretraining.

⚠️ **Failure Triggers:** 
- Loss NaN/Inf → check AMP scaling, LR warmup, or patch divisibility
- VRAM > 5.5 GB → reduce patch size to 8 or disable gradient accumulation
- No UMAP clusters → verify masking isn't hiding all signal, check cosine similarity head
- **Embedding contrast < 2.0** → architecture is unsuitable for downstream clustering; revert to ViT-MAE

---

## 📦 Phase 1: FK Data Pipeline & Preprocessing
### Input Format
- 2D FK spectra: `[Batch, 1, Wavenumber_Bins, Freq_Bins]` (single-channel)
  - Axis 0 (vertical) = wavenumber, Axis 1 (horizontal) = frequency
  - Tensor is transposed from the raw SEG-Y layout: (freq, waven) → (waven, freq)
  - Frequency on the horizontal axis matches seismic processing convention
- Target resolution: `256×256` (downsample/interpolate if necessary)

### Preprocessing Steps (With Metadata Tracking)
```python
# Per-spectrum metadata dictionary (MUST be saved alongside processed tensor)
metadata = {
    "original_shape": (freq_orig, waven_orig),      # e.g., (262, 400) — raw SEG-Y order
    "freq_axis_original": freq_vals_original,        # Hz, linear
    "waven_axis_original": waven_vals_original,      # 1/m
    "resize_factors": (waven_scale, freq_scale),     # 256/waven_orig, 256/freq_orig
    # Transposed before resize: tensor shape (waven, freq) → resize_factors[0] for wavenumber axis
    "amplitude_normalization": {"mu": x.mean(), "sigma": x.std()},
    "clipping_bounds": (-3, 3),
    "spectrum_id": "site_001_shot_042"
}
```
1. Per-spectrum amplitude normalization: (x - μ) / (σ + 1e-6) → store μ, σ for inverse
2. Resize to 256×256: torch.nn.functional.interpolate(..., mode='bilinear', align_corners=False) → store scale factors
3. Dynamic range clipping: np.clip(x, -3, 3) → store bounds for uncertainty propagation
4. Save metadata as JSON sidecar or embed in HDF5 dataset attributes

### Augmentation Strategy (Applied On-The-Fly)

| Augmentation | Purpose | Coordinate Impact |
|--------------|---------|-------------------|
| Random frequency/wavenumber shift (±5%) | Velocity variation robustness | Track shift offsets for inverse transform |
| Block masking / random crop | Simulate array aperture gaps | Masking is model-space only; no inverse needed |
| Gaussian/Poisson noise injection | Data quality variability | None |
| Intensity jitter (±15%) | Source amplitude invariance | None (amplitude not used for picking coordinates) |

### Dataset Split
* Pretraining: 100% (unsupervised)
* Validation/Clustering: 10% held-out subset
* Expert Annotation: 50–200 spectra per active learning iteration

---

## 🧠 Phase 2: Self-Supervised Pretraining (Unsupervised)

### ⚠️ MAE Experiment — Failed (Abandoned)

Three MAE variants were tested. **All produced embedding collapse** (Silhouette < 0,
contrast < 1.1), conclusively demonstrating that the pixel-level reconstruction
objective is unsuitable for FK spectra:

| Experiment | Setting | Val Loss | Silhouette | Verdict |
|-----------|---------|----------|------------|---------|
| v1 (baseline) | Block masking 75%, 30 epochs | 0.084 | −0.322 | ❌ Collapse |
| v2 (random) | Random masking 50%, 30 epochs | 0.075 | ~−0.3 | ❌ Collapse |
| v3 (aggressive) | Aggressive aug, masking 25%, 100 ep (stopped @54) | 0.095 | ~−0.3 | ❌ Collapse |

**Root cause:** All FK spectra share the same global structure (dark field + diagonal
dispersion bands). The MAE can trivially minimize MSE by predicting the "average
spectrum" for every masked patch, never needing to distinguish samples. Encoder
embeddings become degenerate.

---

### Primary Approach: VICReg (Current)

**Paper:** Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for
Self-Supervised Learning", ICLR 2022.

VICReg replaces the reconstruction objective with three explicit regularizers that
force the embedding space to be informative:
1. **Invariance** — two augmented views of the same spectrum have similar embeddings
2. **Variance** — each embedding dimension has std ≥ 1 across the batch (hinge loss)
3. **Covariance** — embedding dimensions are decorrelated (off-diagonal → 0)

**Why this should work for FK (where MAE failed):**
- No reconstruction head → model cannot cheat by predicting average pixels
- Variance term explicitly prevents dimensional collapse (MAE's failure mode)
- Invariance term forces the model to learn *what stays the same* across augmentations
  — i.e., the underlying spectrum identity, not the noise
- No decoder needed → faster training, lower VRAM

#### Architecture
```
Input ─── Aug A ──► Encoder (ViT-Small, shared) ──► Projector MLP ──► Z₁
         Aug B ──► Encoder (ViT-Small, shared) ──► Projector MLP ──► Z₂

L = λ · MSE(Z₁, Z₂)                    (invariance)
  + µ · Σⱼ max(0, 1 − √Var(Zⱼ))        (variance, per-feature)
  + ν · Σᵢ≠ⱼ C[i,j]² / D              (covariance, C = cov matrix of Z)
```

#### Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Encoder | ViT-Small (same as Phase 0, no decoder head) | Reuse proven architecture |
| Projector | 3× MLP: 384 → 2048 → 2048 → 2048 + BN + ReLU | Standard VICReg design |
| Embedding dim | 2048 | Matches projector output |
| λ (sim_weight) | 25.0 | VICReg paper defaults |
| µ (var_weight) | 25.0 | VICReg paper defaults |
| ν (cov_weight) | 1.0 | VICReg paper defaults |
| Batch size | 16 (or batch=2 + accum=8) | Larger batches stabilize variance estimate |
| Optimizer | AdamW, LR=3e-4, cosine decay, 10% warmup | Standard schedule |
| Augmentations | Same aggressive pipeline as MAE v3 | Already developed and tested |
| Epochs | 100 | Matches MAE v3 schedule |

#### Success Criteria
| Metric | Target | How to Verify |
|--------|--------|---------------|
| Silhouette score | > 0.1 | Logged at visualization epochs |
| Intra/inter contrast | > 1.5 | Similarity matrix plot |
| UMAP structure | Separable by line_number | Manual plot review |
| VRAM | < 4.5 GB | `max_vram_mb` logged |

---

### Fallback: BYOL (if VICReg Silhouette < 0.1)

**Paper:** Grill et al., "Bootstrap Your Own Latent", NeurIPS 2020.

Uses an online–target network pair (EMA update of target). The online network must
predict the target's embedding of a differently augmented view, creating a stable
learning signal. Works at very small batch sizes.

#### Architecture
```
View 1 ──► Online encoder ──► Projector ──► Predictor ──► Prediction q
View 2 ──► Target encoder ──► Projector ──────────────────► Target z (stop-grad)
                   ▲ EMA: θ_target ← τ·θ_target + (1−τ)·θ_online
Loss = 2 − 2·(q·z) / (||q||·||z||)  (negative cosine similarity)
```

---

### Final Fallback: Classical Feature Extraction

If all learned approaches fail:
- Flatten spectra → PCA (50–200 components) → HDBSCAN clustering
- Or use spectral descriptors (peak frequencies, mode bandwidth, energy distribution)
- This sacrifices the learned representation benefits but provides a working pipeline

---

## 🔍 Phase 3: Clustering & Active Learning (Labeling Strategy)
### Embedding Extraction
- Freeze MAE encoder, extract `[B, embed_dim]` vectors for full dataset
- L2-normalize embeddings for cosine-based clustering

### Prototype-Based Clustering
| Component | Configuration |
|-----------|---------------|
| Initial prototypes | `K_init = 50` |
| Assignment | Cosine similarity → softmax with temperature `τ` |
| Regularization | Entropy penalty on cluster usage + consistency loss across augmentations |
| Pruning | Remove prototypes with rolling usage `<5%`; merge highly correlated centroids |
| Temperature | Anneal `τ: 0.1 → 0.05` over 20 epochs |

### 📌 Expert Labeling Budget & Active Learning Strategy
- **Target**: **A few high-quality picks per *active* cluster** (parameter defined by user as a max percentage of cluster examples)
- **Labeling Schedule**:
  - Iteration 0: 5 diverse/core samples per cluster
  - Iterations 1–3: Add 2–5 uncertain/boundary samples per cluster based on model entropy
  - Stop when validation RMSE plateaus or cluster coverage >90%
- **Query Strategy**: Hybrid sampling (core centroids + high uncertainty + prototype boundaries) with similarity deduplication
- **Expected Efficiency**: ~80% of maximum performance gain achieved at ~8 labels/cluster; diminishing returns beyond ~15/cluster

### Human-in-the-Loop Interface
- **Display**: Top 3–5 representative spectra per cluster + UMAP neighborhood
- **Annotation**: Minimal click-based pick (frequency-wavenumber pairs) + confidence flag
- **Query Strategy**: Active sampling based on assignment entropy, prototype confidence, and reconstruction error

---

## 🎯 Phase 4: Supervised Fine-Tuning & Picking
### Model Adaptation
- Freeze MAE encoder weights
- Attach lightweight picking head: `Conv2D → AdaptivePool → MLP` (outputs 1D dispersion curve with heatmap)
- Optional: Add Monte Carlo dropout for uncertainty quantification

### Training Setup
| Parameter | Value |
|-----------|-------|
| Optimizer | `AdamW` |
| LR | `5e-5` (encoder), `1e-4` (head) |
| Loss | Smooth L1 (curve regression) + BCE (presence/absence mask) |
| Epochs | 15–30 (few-shot convergence) |
| Regularization | Dropout `0.2`, early stopping on validation picking RMSE |

---

## 🔄 Phase 5: Coordinate Transformation & Inversion Export
**Critical:** Model outputs picks in *normalized, resized model space*. Inversion software requires picks in *original physical units* (Hz, 1/m). This phase handles the reversible mapping.

### 5.1 Inverse Transformation Pipeline
```python
def model_to_original_coords(picks_model, metadata):
    """
    picks_model: [(k_model_idx, f_model_idx), ...] in [0, 255] pixel indices
    metadata: dict from Phase 1 preprocessing
    Returns: [(f_hz, k_inv_m), uncertainty_transformed, ...]

    Note: The model operates on the transposed tensor (shape waven×freq),
    so pick coordinates are [wavenumber_idx, freq_idx].
    """
    picks = np.array(picks_model)

    # 1. Denormalize pixel indices to [0, 1] model-space coordinates
    k_norm = picks[:, 0] / 255.0  # wavenumber is axis 0 (vertical)
    f_norm = picks[:, 1] / 255.0  # frequency is axis 1 (horizontal)

    # 2. Reverse resize scaling
    # Note: metadata["resize_factors"] = [waven_scale, freq_scale]
    # (wavenumber maps to original axis 1, frequency maps to original axis 0)
    k_resized = k_norm * metadata["original_shape"][1]  # inverse: 256/waven_orig → waven_orig
    f_resized = f_norm * metadata["original_shape"][0]  # inverse: 256/freq_orig → freq_orig

    # 3. Reverse axis transformations
    if metadata.get("freq_transform") == "log10":
        f_original = 10**f_resized - 1e-8  # reverse log10(f + eps)
    else:
        f_original = f_resized  # linear axis

    k_original = k_resized  # wavenumber typically linear

    # 4. Propagate uncertainty (first-order error propagation)
    if "uncertainty_model" in metadata:
        unc_waven = metadata["uncertainty_model"]["k_std"] * metadata["resize_factors"][0]
        unc_freq = metadata["uncertainty_model"]["f_std"] * metadata["resize_factors"][1]
        if metadata.get("freq_transform") == "log10":
            unc_freq = unc_freq * np.log(10) * (10**f_resized)
        return list(zip(f_original, k_original, unc_freq, unc_waven))

    return list(zip(f_original, k_original))
```

### 5.2 Round-Trip Validation Protocol
Before exporting to inversion software, validate coordinate mapping:
1. Select 20–50 spectra with high-confidence manual picks on *original-resolution* data
2. Run full pipeline: original → preprocess → model pick → inverse transform
3. Compute error metrics:
   - **Coordinate RMSE**: `sqrt(mean((f_pred - f_manual)² + (k_pred - k_manual)²))`
   - **Velocity error**: `ΔV/V = |(f_pred/k_pred) - (f_manual/k_manual)| / (f_manual/k_manual)`
   - **Target**: RMSE < 1 pixel equivalent, ΔV/V < 0.05

### 5.3 Export Format for Inversion Software
- Structured JSON/CSV with fields: `spectrum_id`, `frequency_hz`, `wavenumber_inv_m`, `phase_velocity_m_s`, `uncertainty`, `mode_flag`, `confidence`, `source_resolution`, `model_version`
- Compatibility: Export converters for Geopsy `.disp`, Dinver `.dat`, or generic CSV
- Fallback: Hybrid picking flag for edge cases requiring manual adjustment on original grid

---

## 📊 Evaluation & Validation Metrics
| Phase | Metric | Target (Rough estimate) |
|-------|--------|--------|
| MNIST Verification | MSE loss, UMAP cluster count, VRAM peak | < 0.02, 10 clusters, < 4.5 GB |
| MAE Pretraining | PSNR / SSIM (masked regions) | > 25 dB / > 0.85 |
| Embeddings | Approx. Silhouette (val set) | > 0.35 |
| Clustering | Prototype usage entropy, HDBSCAN stability | Active K stabilizes ±5 |
| Picking (model space) | RMSE vs expert picks, F1 on mode detection | < 0.15 ΔV/V, > 0.9 F1 |
| Coordinate Transform | Round-trip RMSE, velocity error ΔV/V | < 1 pixel equiv., < 0.05 |

---

## 💻 Software & Hardware Requirements
### Stack
- **Framework**: PyTorch ≥2.0, `timm`, `einops`, `torchvision`
- **Analysis**: `scikit-learn`, `umap-learn`, `hdbscan`, `numpy`, `matplotlib`
- **Tracking**: Weights & Biases or TensorBoard
- **Interface**: Streamlit / Gradio (for expert annotation)
- **Export**: `pandas`, `json`, `h5py` for inversion software compatibility

### RTX 3060 Optimization Checklist
- [ ] Enable `fused=True` in AdamW
- [ ] Use `torch.amp.autocast` + `GradScaler`
- [ ] Gradient accumulation (`micro_batch=1`, `accum=16`)
- [ ] `torch.backends.cudnn.benchmark = True`
- [ ] Monitor VRAM: `nvidia-smi` / `torch.cuda.max_memory_allocated()`

### Reproducibility
- Fixed seeds (`torch`, `numpy`, `random`)
- Save augmentation configs + preprocessing metadata **with each spectrum**
- Export training logs, checkpoint hashes, and environment YAML
- For each change of model architecture, data augmentation or hyperparameters log metric
- **Version coordinate transform functions** alongside model weights

---
