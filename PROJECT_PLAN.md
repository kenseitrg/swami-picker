# Project Plan: Self-Supervised FK Spectrum Analysis & Robust Dispersion Picking

## 🎯 1. Project Overview
| Aspect | Description |
|--------|-------------|
| **Objective** | Develop a robust, human-in-the-loop pipeline to extract fundamental-mode dispersion curves from noisy, highly variable FK spectra for downstream inversion. |
| **Key Challenges** | No ground-truth labels, higher-order modes often dominate, rapid near-surface variability, limited compute (RTX 3060, 6GB VRAM), coordinate system mismatch between model and inversion software. |
| **Proposed Solution** | Representation learning pipeline: (1) Weakly-supervised pretraining with pseudo-labels (clustering → classifier), (2) Prototype-based clustering → active expert labeling → supervised fine-tuning. MAE/VICReg self-supervised approaches were exhausted and abandoned due to embedding collapse on homogeneous FK data. |
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

## 🧠 Phase 2: Representation Learning (Unsupervised → Weakly Supervised)

### Status: Self-Supervised Approaches Exhausted — Both Failed

**Four experiments conclusively demonstrate that FK spectra are too homogeneous for
self-supervised representation learning.**

| Experiment | Method | Epochs | Best Silhouette | Best Contrast | Verdict |
|-----------|--------|--------|-----------------|---------------|---------|
| v1 | MAE (block 75%) | 30 | −0.322 | 1.078 | ❌ Collapse |
| v2 | MAE (random 50%) | 30 | ~−0.30 | ~1.08 | ❌ Collapse |
| v3 | MAE (aggressive aug, 25%) | 54 | ~−0.30 | ~1.08 | ❌ Collapse |
| v4 | VICReg (batch=16) | 50 | −0.252 | 1.036 | ❌ Collapse |

**Root cause:** All 1,145 FK spectra share the same global structure (dark field +
diagonal dispersion bands). The differences between receiver lines are extremely
subtle — essentially noise in mode position/amplitude. The signal-to-noise ratio is
too low for any self-supervised objective to extract distinguishing features.

| Method | Collapse Mode | Why It Failed |
|--------|--------------|---------------|
| **MAE** | Exact collapse | Reconstruction loss minimized by predicting "average spectrum" for every masked patch |
| **VICReg** | Fuzzy-ball collapse | Variance hinge cannot push std ≥ 1 because all samples map to the same small region. Covariance has no signal to decorrelate. |

---

### Option C: Supervised Pretraining with Pseudo-Labels (IMPLEMENTED ✅)

**Approach:** Skip self-supervised learning. Use weak supervision via **2-step hierarchical clustering** of engineered features:
1. Extract physics-informed spectral descriptors (20 features per spectrum)
2. **Step 1:** UMAP(5D, min_dist=0.0) → HDBSCAN to obtain initial pseudo-labels
3. **Step 2:** Re-cluster the dominant cluster with the same pipeline to split it into sub-clusters
4. Merge sub-clusters with the stable core → balanced 11-cluster label set
5. Train a lightweight MLP classifier (cross-entropy) on the merged pseudo-labels
6. Use the classifier's penultimate layer as embeddings for Phase 3

**Why this works where self-supervised failed:**
- Cross-entropy loss **explicitly forces the model to discriminate between clusters**
- The model receives a direct "push different samples apart" signal
- Even if pseudo-labels are noisy, the classifier must learn to separate them
- 2-step hierarchical clustering prevents a single dominant cluster from collapsing the label space

#### Clustering Front-End
```
Raw spectrum (256×256)
    │
    ├──► Energy marginals (sum along each axis) → concatenate 512-D → PCA(10)
    │     └── Optional Path A: 10 PCA components
    │
    └──► Spectral descriptors (20 physics-informed features)
            └── Path B (winner): centroids, bandwidths, energy ratios,
                 peak velocities, skewness, total energy
                     │
                     ▼
              StandardScaler
                     │
                     ▼
              UMAP(5D, n_neighbors=15, min_dist=0.0)
                     │
                     ▼
              HDBSCAN(min_cluster_size=30, min_samples=10, eom)
                     │
                     ├──► Stable core (K-1 clusters)
                     └──► Dominant cluster (if > 50% of data)
                              │
                              ▼
                      Re-run UMAP → HDBSCAN
                              │
                              ▼
                      Sub-clusters → merge
```

**Winning feature set:** Physics-informed spectral descriptors (20-D), standardized.
- Frequency/wavenumber centroids & bandwidths
- Energy-weighted IQR
- Low/High frequency energy ratio
- Peak velocities at 10 frequency bands
- Frequency/wavenumber skewness
- Total energy

#### Architecture for Stage-1 Classifier
```
Input (features: 20-D or raw: 1×256×256)
    │
    ├──► MLP(20→256→128→K)   [for feature input]
    └──► CNN(→128→GAP→256→K) [for raw spectra]
    │
    └──► Penultimate layer → Embedding for Phase 3
```

#### Stage-2 Pseudo-Label Expansion
After Stage-1 training, run inference on noise points (-1). Accept pseudo-labels where
softmax confidence ≥ 0.90 and cluster size after addition ≥ 5. Retrain on expanded set.

#### Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Features | Spectral descriptors (20-D) | Best separation (Silhouette 0.60) |
| Clustering | 2-step hierarchical UMAP→HDBSCAN | Splits dominant cluster |
| Label set | 11 clusters, 12% noise | Balanced sizes (36–260) |
| Classifier | MLP(20→256→128→K) | Fast, low VRAM, interpretable |
| Loss | Cross-entropy | Explicit discrimination signal |
| Batch size | 32 | Features are tiny (20-D) |
| Optimizer | AdamW(fused=True), LR=1e-4, cosine decay | Proven stable |
| Epochs | 30 (smoke test) → 100 (full) | Quick validation |

#### Success Criteria
| Metric | Target | How to Verify |
|--------|--------|---------------|
| Training accuracy | > 60% (above random for 11 clusters) | Logged per epoch |
| Validation accuracy | Within 10% of training accuracy | Check for overfitting |
| Silhouette score (penultimate) | > 0.10 | Logged at visualization epochs |
| Intra/inter contrast | > 1.5 | Similarity matrix plot |
| VRAM | < 4.5 GB | `max_vram_mb` logged |
| Pseudo-label purity | > 85% agreement (Stage 1 ↔ Stage 3) | Confusion matrix diagonal |

---

### Option D: Classical Feature Extraction (Guaranteed Baseline)

If Option C fails after a 30-epoch test, fall back immediately to classical methods:
1. Flatten spectra → PCA (50–200 components) → UMAP → HDBSCAN clustering
2. Or use spectral descriptors: peak frequencies, mode bandwidths, energy distribution
3. Proceed directly to Phase 3 (active learning) with classical features

This provides a **guaranteed working baseline**. The tradeoff is that classical features
may miss subtle patterns, but they will produce separable clusters for downstream picking.

---

### Abandoned Approaches (Documented for Reference)

#### MAE (Masking Ratios 75% → 25%)
Pixel-level reconstruction objective. Failed because "average spectrum" prediction
minimizes MSE without learning distinguishing features.

#### VICReg (Variance-Invariance-Covariance)
Bardes et al., ICLR 2022. Failed because explicit variance regularization cannot
overcome the fundamental lack of distinguishing signal in the data. Variance hinge
remained active (std < 1) after 50 epochs because all samples collapsed to the same
small region of embedding space.

#### BYOL (Not Pursued)
Grill et al., NeurIPS 2020. EMA-based self-supervised method. Not pursued because
it lacks negative pairs and is unlikely to succeed where VICReg's stronger explicit
regularization failed.

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
