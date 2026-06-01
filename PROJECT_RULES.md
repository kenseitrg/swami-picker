# Project Rules: swami-picker

Coding conventions, architectural guidelines, and quality standards for the self-supervised FK spectrum analysis project.

---

## 1. Python Style & Quality

### 1.1 Tooling
- **Formatter/Linter**: Ruff (configured in `pyproject.toml`). Run `ruff check .` and `ruff format .` before every commit.
- **Line length**: 100 characters.
- **Import style**: `isort` compatible. Group: stdlib → third-party → first-party. Use absolute imports within the package.

### 1.2 Type Hints
- **Mandatory** on all public function signatures, class attributes, and dataclass fields.
- Use `from __future__ import annotations` to enable PEP 563 postponed annotations (allows modern syntax like `list[int]` instead of `List[int]`).
- Prefer `collections.abc` generics (`Sequence`, `Mapping`, `Callable`) over concrete types in parameter annotations.
- Use `torch.Tensor` for tensor parameters. Document expected shapes in docstrings (see §2.5).

### 1.3 Static Checks Before Commit
- Run **`ruff check .`** and **`ruff format .`** before every commit.
- Run **`ty .`** as a static type-checking pass before pushing or merging.
- Both commands must pass cleanly; do not silence warnings unless justified in a code-review comment.

### 1.3 Docstrings
- **Google style** for all public modules, classes, methods, and functions.
- Every function must have a one-line summary. Complex functions require an `Args:` and `Returns:` block.
- If a function raises exceptions, document them in a `Raises:` block.

### 1.4 Error Handling
- No bare `except:` clauses. Always catch specific exception types.
- Validate tensor shapes and device placement at function entry points when ambiguity would cause silent failures.
- Use `logging` (not `print`) for all runtime diagnostics. Log levels:
  - `INFO`: training epoch summaries, checkpoint saves
  - `DEBUG`: batch-level metrics, shape traces
  - `WARNING`: skipped samples, fallback code paths
  - `ERROR`: hard stops with context

---

## 2. PyTorch & Deep Learning Conventions

### 2.1 Reproducibility
- Every script entry point must call a centralized `set_seed(seed: int)` helper that sets:
  ```python
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  # deterministic flags are optional (warn about perf cost)
  ```
- Save the seed, full command-line arguments, and a hash of the config file alongside every checkpoint.

### 2.2 Device Management
- Do **not** hardcode `.cuda()` or `.to("cuda")` deep in model code.
- Prefer passing a `device: torch.device` parameter, or use a utility like `get_device()` that falls back to CPU gracefully.
- Keep `.to(device)` calls close to the data-loading boundary, not scattered inside forward passes.

### 2.3 Automatic Mixed Precision (AMP)
- Use `torch.amp.autocast("cuda", dtype=torch.float16)` + `torch.amp.GradScaler()` for all training loops.
- Wrap only the forward pass and loss computation in `autocast`; keep optimizer step outside.
- Always use `scaler.scale(loss).backward()` and `scaler.step(optimizer); scaler.update()`.

### 2.4 Memory & VRAM Discipline (RTX 3060, 6 GB)
- Profile peak VRAM with `torch.cuda.max_memory_allocated()` after every epoch. Log it.
- Clear unused tensors explicitly: `del loss, outputs; torch.cuda.empty_cache()` when appropriate (especially after validation).
- Gradient accumulation is the default scaling knob for effective batch size, not increasing micro-batch size.
- Prefer `torch.backends.cudnn.benchmark = True` for fixed-input-size training; disable if input sizes vary.
- No in-place operations (`relu(inplace=True)`, `add_()`) on tensors that participate in gradient computation unless proven safe.

### 2.5 Tensor Shape Documentation
Every function accepting or returning tensors must document shapes in the docstring using the format:
```python
"""Brief summary.

Args:
    x: Input tensor of shape (B, C, H, W).
    mask: Boolean tensor of shape (B, num_patches).

Returns:
    Reconstructed tensor of shape (B, C, H, W).
"""
```

### 2.6 Checkpointing
- Save checkpoints as dictionaries with **at minimum**:
  ```python
  {
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "scaler": scaler.state_dict(),
      "scheduler": scheduler.state_dict(),
      "epoch": epoch,
      "step": global_step,
      "seed": seed,
      "config": config_dict,   # full config snapshot
      "metrics": { ... },      # current best metrics
  }
  ```
- Never rely on `torch.save(model, path)` (breaks on class moves).
- Resume logic must restore the exact same training state, including RNG if possible.

---

## 3. Research Reproducibility & Experiment Tracking

### 3.1 Configuration as Code
- No magic numbers in training scripts. All hyperparameters live in a typed config (dataclass or Pydantic `BaseModel`).
- Configs are versioned in git (e.g., `configs/phase0_mnist.yaml`).
- When running an experiment, save a **snapshot** of the resolved config next to the checkpoint.

### 3.2 Logging
- Use a single experiment logger (Weights & Biases or TensorBoard) for all metrics.
- Log hyperparameters once at startup.
- Log learning rate, loss, and VRAM at every step/epoch.
- Log visual outputs (reconstruction grids, UMAP plots) no more than once per epoch to avoid bloat.

### 3.3 Model-Change Tracking
Every modification to model architecture, loss formulation, or data augmentation must be logged together with the resulting metric delta.
- **Required fields per experiment**:
  - `model_version`: semantic or git-short-hash identifier
  - `architecture_delta`: brief description of what changed (e.g., "added 2-layer ConvNet decoder", "switched masking from random to 2×2 block")
  - `baseline_metric`: the best metric from the previous stable version
  - `new_metric`: the best metric after the change
  - `metric_delta`: computed difference (`new_metric − baseline_metric`)
- Store these entries in the experiment logger (as a summary table or `wandb` config) **and** append a line to `experiments/MODEL_CHANGELOG.md`.
- If a change produces a regression (negative delta) after a full training run, document the hypothesis for why before reverting or iterating.

### 3.4 Experiment Logs Directory
All training logs, profiler traces, and logger artifacts are written to a dedicated top-level directory:
```
experiments/
├── YYYY-MM-DD_experiment-name/     # one sub-folder per run
│   ├── config.yaml                 # resolved config snapshot
│   ├── metrics.jsonl               # line-delimited metrics (backup)
│   ├── checkpoints/                # model checkpoints
│   └── logs/                       # text logs & profiler traces
└── MODEL_CHANGELOG.md              # accumulated architecture-metric history
```
- Never write logs directly into `src/` or `scripts/`.
- The run folder name must include an ISO date prefix and a short human-readable slug (e.g., `2026-06-01_phase0-mnist_mae-small`).

### 3.3 Data Versioning
- Preprocessed datasets must carry a manifest or hash file listing source files and preprocessing parameters.
- Never mutate raw data in place. Write processed outputs to a separate directory with a naming scheme that includes the preprocessing config hash.

---

## 4. Data & Metadata Handling

### 4.1 Metadata is Non-Negotiable
Every preprocessed spectrum **must** have an associated metadata dictionary (see Phase 1 of `PROJECT_PLAN.md`).

- Store as JSON sidecar (`spectrum_001.json`) or HDF5 attributes.
- Required fields:
  - `original_shape`, `resize_factors`
  - `freq_axis_original`, `waven_axis_original`
  - `amplitude_normalization` (`mu`, `sigma`)
  - `clipping_bounds`
  - `spectrum_id`
- The inverse transform function (Phase 5) is only valid if metadata is complete. Validate keys before transform.

### 4.2 Coordinate Transform Safety
- Forward and inverse transforms must be implemented as a matched pair in the same module.
- Unit-test the round-trip on synthetic data: `original → preprocess → inverse → compare`.
- Propagate uncertainty through transforms using first-order error propagation (document the math in comments).

---

## 5. Project Structure

```
swami-picker/
├── configs/              # YAML/JSON experiment configs
├── data/                 # Raw & processed data (gitignored)
├── experiments/          # Training logs, checkpoints, profiler traces (gitignored)
│   └── MODEL_CHANGELOG.md # Accumulated architecture-metric history (tracked in git)
├── src/
│   ├── models/           # MAE encoder, decoder, picking head
│   ├── data/             # Datasets, transforms, augmentation
│   ├── training/         # Training loops, optimizers, schedulers
│   ├── clustering/       # Embedding extraction, prototype clustering
│   ├── picking/          # Supervised fine-tuning & inference
│   ├── transforms/       # Coordinate transforms (model ↔ original)
│   ├── utils/            # Seed, device, logging, checkpointing helpers
│   └── evaluation/       # Metrics, UMAP/Silhouette, round-trip validation
├── scripts/              # Executable training & evaluation scripts
├── notebooks/            # Exploratory analysis (kept minimal)
├── tests/                # Unit tests (round-trip transforms, data loading)
└── PROJECT_RULES.md      # This file
```

- Keep scripts thin: argument parsing + a call to a library function in `src/`.
- Do not put business logic in notebooks. Use notebooks only for visualization and ad-hoc exploration.

---

## 6. Testing

- **Unit tests** for:
  - Coordinate round-trip transforms (must hold exactly for synthetic grids).
  - Data augmentation invariants (e.g., amplitude jitter preserves shape).
  - Checkpoint save/load resume (loss curve identical for 2 steps).
- **Smoke tests** for every training script: run for 2 iterations on a tiny subset to catch shape mismatches before a full epoch.

---

## 8. Visualization & Publication Quality

All data transformations, model behaviour, and results must be accompanied by clear visual evidence. Figures are a first-class deliverable intended for eventual publication.

- **Default to a figure**: whenever a script produces metrics, distributions, or spatial/spectral data, it must also emit a corresponding chart, graph, or illustrative example.
- **Clarity over embellishment**: every plot must accurately represent the underlying data. Avoid misleading axis scaling, truncated colour bars, or ambiguous legends.
- **Demonstrate transformations**: for every preprocessing step (resizing, masking, normalization, coordinate transforms), provide before/after panels that make the effect unambiguous.
- **Consistent style**: establish a single matplotlib style sheet (e.g., `src/utils/plot_style.py`) and apply it to all figures. Define uniform colour palettes, font sizes, line weights, and figure dimensions so that panels from different phases are visually compatible.
- **High-resolution exports**: save publication-ready figures as vector graphics (PDF/SVG) or high-DPI PNG (≥300 dpi) in the experiment run directory. Embed concise, informative captions in filenames or adjacent text files.
- **Reproducible plotting**: plotting code must accept a `save_path: Path | None` argument and be runnable headless (no `plt.show()` blocking in scripts). Random or sampled examples must be seeded so the exact same figure can be regenerated from a checkpoint.

---

## 7. Review & Iteration

These rules are a living document. Propose changes via:
1. Edit `PROJECT_RULES.md` in a feature branch.
2. Open a review thread explaining the motivation.
3. Merge only after at least one round of discussion.

---

*Last updated: 2026-06-01*
