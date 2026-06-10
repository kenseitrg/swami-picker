# Phase 3: Active Learning & Human-in-the-Loop Annotation — TODO

> **Status:** Planning  
> **Depends on:** Phase 2c (✅ completed — MLP classifier trained, 11 merged HDBSCAN clusters locked, 128-D embeddings extracted for all 1,392 spectra)  
> **Goal:** Collect expert dispersion-curve picks on a strategically chosen subset of spectra, producing dense `(256,)` ground-truth arrays for Phase 4 supervised picking model.

---

## 0. Inventory of Existing Artifacts

Before any new code is written, verify that the following artifacts are available and loadable. These are the inputs to Phase 3.

| Artifact | Path | Shape / Description |
|----------|------|---------------------|
| MLP embeddings (all spectra) | `data/processed/mlp_embeddings_phase3.npz` | `(1392, 128)` — penultimate layer |
| Spectrum IDs | same `.npz` | `(1392,)` — string IDs |
| Cluster labels (11 merged) | same `.npz` | `(1392,)` — integers `0..10`, `-1` = noise |
| Pseudo-label provenance | `experiments/2026-06-07_phase2c-descriptor-umap5-mindist0/pseudo_labels_merged.npz` | probabilities, original labels |
| Preprocessed spectra | `data/processed/spectra/*.npz` + `.json` | `(1, 256, 256)` tensors + metadata |
| Manifest | `data/processed/manifest.json` | train/val split, file paths |

**Verify:** Load embeddings + labels, confirm 11 non-noise clusters, print cluster size distribution.

---

## 1. Architectural Decisions (Locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Clustering backend** | Re-use existing 11 merged HDBSCAN clusters | User confirmed. No prototype clustering needed. |
| **Annotation UI framework** | tkinter + matplotlib canvas | User confirmed. Simpler than Streamlit/Gradio for dense keyboard-driven image annotation. No web server needed. |
| **Ground-truth format** | Dense `(256,)` `int16` array per spectrum | User confirmed. One wavenumber index per frequency column. Sentinel `-1` = "not picked / no mode visible here". |
| **Interpolation method** | PCHIP (monotone cubic) with linear fallback | PCHIP preserves local extrema and avoids cubic-spline overshoot on sparse geophysical picks. Fallback to linear if < 3 points. |
| **Distinguish direct vs. interpolated** | Save alongside a `bool` mask `(256,)` | Phase 4 can weight direct picks higher, or mask out interpolated regions during early training. |
| **Query strategy** | Two-phase per cluster: (1) centroid-near samples, (2) boundary/far samples | Maximizes information per annotation: core samples define the cluster archetype, boundaries capture diversity. |
| **Storage** | One `.npz` per annotation session + JSON manifest | Atomic saves, versioned sessions, easy to diff/merge. |

---

## 2. Annotation Data Model

### 2.1 Per-Spectrum Annotation Structure

```
AnnotationRecord:
  - spectrum_id: str
  - annotator: str | None        # e.g. "expert_01"
  - session_id: str              # e.g. "2026-06-10_iter0"
  - timestamp: str (ISO-8601)
  - version: int                 # incremented on re-edit

  # Pick data
  - wavenumber_picks: ndarray[int16] of shape (256,)
      # One index per frequency column (0..255).
      # -1  = "no pick at this frequency" (user skipped or deleted).
      # 0..255 = wavenumber index of the picked mode.
  - direct_mask: ndarray[bool] of shape (256,)
      # True  = user explicitly clicked this frequency.
      # False = spline-interpolated or unpicked (-1).
  - confidence: ndarray[float32] of shape (256,)
      # Optional per-frequency confidence (0..1).
      # Default 1.0 for direct picks, 0.5 for interpolated, 0.0 for -1.
```

### 2.2 Session-Level Manifest

```json
{
  "session_id": "2026-06-10_iter0",
  "created": "2026-06-10T14:30:00Z",
  "annotator": "expert_01",
  "percentage_per_cluster": 15.0,
  "total_target": 208,
  "per_cluster_target": {
    "0": 12, "1": 18, ..., "10": 9
  },
  "spectra_ordered": ["RL5007_50071009", "RL5007_50071017", ...],
  "query_strategy": "centroid_then_boundary",
  "annotations_dir": "annotations/2026-06-10_iter0/"
}
```

### 2.3 File Layout

```
annotations/
├── 2026-06-10_iter0/
│   ├── manifest.json
│   ├── config.yaml              # snapshot of percentage + strategy
│   └── spectra/
│       ├── RL5007_50071009.npz
│       ├── RL5007_50071017.npz
│       └── ...
```

Each `.npz` contains: `wavenumber_picks`, `direct_mask`, `confidence`, `spectrum_id`, `timestamp`.

---

## 3. Active Learning Query Strategy

Since clusters are fixed, the problem reduces to: **which spectra within each cluster should be annotated?**

### 3.1 Ranking Function per Cluster

For each cluster `c` with `N_c` spectra:

1. Compute cluster centroid in 128-D embedding space (mean of L2-normalized embeddings).
2. For each spectrum in the cluster, compute cosine distance to centroid: `dist = 1 - cos_sim`.
3. Sort by `dist` ascending.
4. The ranked list has two zones:
   - **Core zone** (first `ceil(N_c * pct / 100 * 0.6)`): closest to centroid — representative archetypes.
   - **Boundary zone** (remaining `ceil(N_c * pct / 100 * 0.4)`): farthest from centroid — capture diversity, edge cases.

If `pct` is small (e.g., 5%), bias even more toward core samples (80/20 split) to ensure stable archetypes first.

### 3.2 Global Ordering

Interleave clusters to prevent annotator fatigue. Do **not** annotate all of cluster 0, then cluster 1, etc.

```
Round-robin interleaving:
  cluster_0[0], cluster_1[0], ..., cluster_10[0],
  cluster_0[1], cluster_1[1], ..., cluster_10[1],
  ...
```

This ensures the expert sees spectral diversity early, which helps calibrate their picking strategy.

### 3.3 Parameter: Percentage per Cluster

The UI (or a pre-flight CLI script) accepts a percentage `p ∈ [1, 100]`.

**Computed totals displayed to user:**
- Per-cluster: `target_c = ceil(N_c * p / 100)`
- Total: `sum(target_c)`
- Estimated time: `total * t_per_spectrum` (assume 30–60 s per spectrum for calibration)

Example at `p = 10%` with cluster sizes `[36, 52, 48, 260, ...]`:
- Targets: `[4, 6, 5, 26, ...]`
- Total: ~139 spectra

Example at `p = 20%`:
- Total: ~278 spectra

The user can adjust `p` until the total feels manageable before starting the session.

### 3.4 Required Module: `src/active_learning/query.py`

**Functions (no implementation details, just signatures):**

- `rank_spectra_for_cluster(embeddings, labels, cluster_id, strategy="centroid_boundary") -> ndarray[int]`
  - Returns indices into the global spectra array, ordered by annotation priority.
- `build_annotation_order(embeddings, labels, percentage, interleave=True) -> list[str]`
  - Returns ordered list of `spectrum_id`s.
- `compute_annotation_budget(labels, percentage) -> dict[str, int]`
  - Returns per-cluster targets and total.

---

## 4. Spline Interpolation Module

### 4.1 Problem

The expert clicks sparse `(freq_idx, waven_idx)` points on the spectrum. We need a smooth curve `wavenumber = f(frequency)` defined on all 256 frequency indices.

### 4.2 Design

**Input:** List of `(f_i, w_i)` tuples where `f_i` are distinct frequency indices (0..255) and `w_i` are wavenumber indices (0..255).

**Output:**
- `wavenumber_picks`: `(256,)` int16 array
- `direct_mask`: `(256,)` bool array (True at `f_i`)

**Algorithm:**
1. If `< 2` points: return all `-1`, `direct_mask` all False. (Insufficient for interpolation.)
2. If `2` points: linear interpolation between them; everything else `-1`.
3. If `≥ 3` points: sort by `f_i`. Fit `scipy.interpolate.PchipInterpolator(freqs, wavenumbers)`.
4. Evaluate interpolator on `np.arange(256)`.
5. Clip interpolated values to `[0, 255]` and round to `int16`.
6. Outside the `[min(f_i), max(f_i)]` range: set `-1` (extrapolation is dangerous for dispersion curves).

**Why PCHIP:** Unlike cubic splines, PCHIP is monotone-preserving. If the expert picks points that trend upward, PCHIP will not overshoot downward between them — critical for physical data like dispersion curves.

### 4.3 Required Module: `src/picking/interpolation.py`

**Functions:**
- `interpolate_picks(picks: list[tuple[int, int]]) -> tuple[ndarray[int16], ndarray[bool]]`
- `add_pick(existing_picks, freq_idx, waven_idx) -> list[tuple[int, int]]`
- `remove_pick(existing_picks, freq_idx) -> list[tuple[int, int]]`
- `delete_picks_at_location(existing_picks, freq_idx, tol=1) -> list[tuple[int, int]]`  # for "delete nearby"

### 4.4 Unit Tests

- `test_linear_two_points` — linear interpolation between 2 points, everything else `-1`.
- `test_pchip_three_points` — PCHIP curve passes through all 3 points, monotonicity preserved.
- `test_out_of_range_unpicked` — values outside `[min_f, max_f]` are `-1`.
- `test_clip_to_bounds` — interpolated values clipped to `[0, 255]`.
- `test_remove_pick` — removing a point updates direct_mask and triggers re-interpolation.

---

## 5. Human-in-the-Loop Interface (tkinter Application)

### 5.1 UI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Session: iter0  |  Cluster: 3/11  |  Spectrum: 42/208  |  [Save]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                         ┌──────────────────────┐                   │
│                         │   Spectrum Display   │                   │
│                         │   (256×256 imshow)   │                   │
│                         │   viridis colormap   │                   │
│                         │   physical axes      │                   │
│                         │   (Hz, 1/m)          │                   │
│                         └──────────────────────┘                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Pick Curve Overlay: red line = current interpolated picks  │   │
│  │  Blue dots = directly clicked points                        │   │
│  │  Grey dashed = unpicked regions (-1)                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│  Coverage: 15% (locked)  |  Total to pick: 208  |  Done: 71/208   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Cluster | Total | Target | Done | Remaining                │   │
│  │  --------|-------|--------|------|----------                │   │
│  │  0       | 36    | 6      | 2    | 4                       │   │
│  │  1       | 52    | 8      | 5    | 3                       │   │
│  │  ...     | ...   | ...    | ...  | ...                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  Progress: ████████░░░░ 34% (71/208 annotated)                     │
├─────────────────────────────────────────────────────────────────────┤
│  Hotkeys: Space=Next  Z=Prev  Q=PrevCluster  W=NextCluster        │
│           D=DeleteAtCursor  Click=AddPick  RightClick=RemovePick  │
│           ↑/↓=NudgePick  X=ClearSpectrum  S=Save  Esc=Quit       │
├─────────────────────────────────────────────────────────────────────┤
│  Location Map: small scatter plot of all spectra (X/Y coords)     │
│                current spectrum highlighted in red                │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Main Canvas: Spectrum Display

- **Backend:** `matplotlib.backends.backend_tkagg.FigureCanvasTkAgg`
- **Image:** `imshow` of the spectrum tensor `(256, 256)` with `extent` set from metadata's physical axes (`freq_axis_resized`, `waven_axis_resized`).
- **Overlays:**
  - Red line: current interpolated picks.
  - Blue dots: directly clicked points.
  - Vertical crosshair: follows mouse X (frequency), snaps to nearest column.
- **Interaction:**
  - **Left click:** Add a pick at `(freq_idx, waven_idx)` where `waven_idx` is derived from the click Y position. Recompute spline. Update overlay.
  - **Right click:** Remove the pick at the nearest frequency index. Recompute spline.
  - **Mouse move:** Update crosshair X position; display current frequency and wavenumber in a status bar.

### 5.3 Percentage Parameter & Budget Display

The annotation percentage is **locked at pre-flight time** by `prepare_session.py`. The UI does **not** allow changing it — doing so mid-session would invalidate the annotation queue and create state-synchronization problems.

The UI **displays** the locked budget read-only in a compact panel:

```
Session: iter0  |  Strategy: centroid_boundary  |  Coverage: 15%

Cluster | Total | To Pick | Annotated | Remaining
--------|-------|---------|-----------|----------
0       | 36    | 6       | 2         | 4
1       | 52    | 8       | 5         | 3
...     | ...   | ...     | ...       | ...
-----------------------------------------------
TOTAL   | 1225  | 208     | 71        | 137
```

This gives the expert immediate awareness of progress and which clusters still need work, without allowing destructive configuration changes.

### 5.4 Hotkey Design

All hotkeys are global to the application window (bound via `tk.bind`).

| Key | Action | Details |
|-----|--------|---------|
| `Space` | Next spectrum | Save current annotation (if dirty), load next spectrum in queue. |
| `z` | Previous spectrum | Save current, load previous. |
| `q` | Jump to previous cluster | Skip to the next spectrum in the queue that belongs to the previous cluster (wraps). Useful if the expert realizes they need to calibrate on a different cluster type. |
| `w` | Jump to next cluster | Same, forward direction. |
| `d` | Delete pick at cursor | Remove the pick at the frequency column nearest to the mouse X position. If no pick exists within `tol=1` column, do nothing. Recompute spline. |
| `x` | Clear all picks on current spectrum | Reset to empty (all `-1`). Confirmation dialog if > 3 picks exist. |
| `s` | Manual save | Force-write the current annotation to disk immediately. |
| `Esc` | Quit | Prompt for save if unsaved changes exist. |
| `↑` / `↓` | Nudge last pick | Move the most recently added pick up/down by 1 wavenumber index. Useful for fine-tuning. |

**Rationale for key choices:**
- `Space` and `z` are on opposite hands — fast forward/backward without hand travel.
- `q`/`w` are adjacent — intuitive cluster navigation.
- `d` is under the left hand (home row) — common delete action.

### 5.5 Auto-Save & Dirty State

- A `dirty` flag tracks whether the current spectrum has unsaved changes.
- On `Space`/`z`/`q`/`w`: if dirty, auto-save before navigating.
- Auto-save writes to `annotations/<session>/spectra/<spectrum_id>.npz` without confirmation.
- Manual `s` triggers the same save path but flashes a "Saved" label for 1 second.
- On quit (`Esc`): if dirty, show `tk.messagebox.askyesnocancel` — Yes=save & quit, No=quit without save, Cancel=return to app.

### 5.6 Metadata Display & Location Map

A sidebar or status bar shows:
- `spectrum_id`
- `line_number`, `point_number` (from metadata)
- `cluster_id`
- Physical axis ranges (Hz, 1/m)
- Current cursor position in physical units

**Location Map:** A small `matplotlib` subplot (e.g., 200×200 px) embedded in the tkinter UI shows a scatter plot of all 1,392 spectra using their `x_coord` / `y_coord` from metadata. Each dot is colored by cluster. The current spectrum is highlighted with a red ring marker. This gives the expert immediate spatial context — e.g., "I'm annotating spectra near the southern edge of the survey." The map is non-interactive (view only).

### 5.7 Required Module: `src/picking/annotation_app.py`

**Class:** `AnnotationApp(tk.Tk)`

**Responsibilities:**
- Load session config and annotation queue.
- Initialize matplotlib figure + canvas.
- Bind hotkeys and mouse events.
- Manage current annotation state (sparse picks → spline → dense array).
- Read/write `.npz` annotation files.
- Track dirty flag and auto-save.

**No business logic in the class:**
- Spline logic → `src/picking/interpolation.py`
- Query strategy → `src/active_learning/query.py`
- Spectrum loading → `src/data/fk_dataset.py` (re-use)

---

## 6. Pre-Flight CLI Script

Before launching the tkinter app, a CLI script prepares the session.

### 6.1 `scripts/phase3_active_learning/prepare_session.py`

**Args:**
- `--percentage`: float, e.g. `15.0`
- `--embeddings`: path to `mlp_embeddings_phase3.npz`
- `--output-dir`: where to create `annotations/<session>/`
- `--strategy`: `"centroid_boundary"` (default) or `"random"`
- `--name`: session name slug

**Actions:**
1. Load embeddings, labels, spectrum IDs.
2. Compute per-cluster targets with `compute_annotation_budget()`.
3. Print a table to stdout:
   ```
   Cluster | Size | Target | % of Cluster
   --------|------|--------|-------------
   0       | 36   | 4      | 11.1%
   1       | 52   | 6      | 11.5%
   ...
   TOTAL   | 1225 | 139    | 11.3%
   ```
4. Prompt user: "Proceed with annotation? [Y/n]"
5. If yes:
   - Run `build_annotation_order()`.
   - Create session directory.
   - Write `manifest.json` + `config.yaml`.
   - Print the command to launch the app.

### 6.2 `scripts/phase3_active_learning/launch_app.py`

**Args:**
- `--session-dir`: path to session directory created by `prepare_session.py`

**Actions:**
1. Load session manifest.
2. Initialize `AnnotationApp`.
3. Start `tk.mainloop()`.

---

## 7. Export to Phase 4 Training Format

After one or more annotation sessions, a script aggregates annotations into a Phase 4-ready dataset.

### 7.1 `scripts/phase3_active_learning/export_annotations.py`

**Args:**
- `--session-dirs`: one or more annotation session directories
- `--output`: path to output `.npz`
- `--min-direct-picks`: minimum number of direct picks for a spectrum to be included (default 3)

**Output format:**
```
phase4_training_data.npz:
  - spectrum_ids: ndarray[str] of shape (N,)
  - spectra: ndarray[float32] of shape (N, 1, 256, 256)
  - picks: ndarray[int16] of shape (N, 256)       # -1 = no pick
  - direct_masks: ndarray[bool] of shape (N, 256)  # True = directly picked
  - cluster_labels: ndarray[int] of shape (N,)     # 0..10
  - metadata: list[dict]                           # JSON-serializable per spectrum
```

**Filtering:**
- Spectra with fewer than `min-direct-picks` direct picks are excluded (too much interpolation is unreliable).
- Noise points (`cluster_label == -1`) may be optionally included or excluded via a flag.

### 7.2 Phase 4 Compatibility

The Phase 4 model (U-Net / encoder-decoder) will consume:
- Input: `spectra` `(B, 1, 256, 256)`
- Target: `picks` `(B, 256)` — for each frequency column, the index of the dispersion mode
- Loss mask: `direct_masks` can be used to weight direct picks higher in the loss function

---

## 8. Testing Plan

### 8.1 Unit Tests (`tests/test_interpolation.py`)

- `test_pchip_monotonicity` — monotonic input points produce monotonic output.
- `test_two_point_linear` — exactly 2 points → linear segment, rest `-1`.
- `test_single_point_no_interp` — 1 point → no interpolation possible, all `-1`.
- `test_out_of_bounds_unpicked` — indices outside `[min_f, max_f]` are `-1`.
- `test_add_remove_idempotent` — add then remove a pick → back to original state.

### 8.2 Unit Tests (`tests/test_query_strategy.py`)

- `test_core_samples_closer_than_boundary` — first half of ranked list has smaller centroid distance than second half.
- `test_interleaving_preserves_cluster_balance` — round-robin order visits each cluster equally often at the start.
- `test_budget_computation` — `ceil(N * pct / 100)` logic is correct, total sums properly.
- `test_disjointness` — no spectrum appears twice in the annotation order.

### 8.3 Integration Tests

- **Smoke test:** Launch app, click 3 points on a spectrum, press Space, verify `.npz` file exists and contains expected arrays.
- **Resume test:** Close app mid-session, relaunch with same `--session-dir`, verify it resumes at the same spectrum with picks intact (no versioning, simple overwrite).
- **Export test:** Run `export_annotations.py` on a test session, verify output shapes and that `min-direct-picks` filter works.

### 8.4 Manual UI Audit

- Verify physical axis labels match metadata (Hz horizontal, 1/m vertical).
- Verify spline curve visually follows clicked points without overshoot.
- Verify hotkeys respond without focus issues.
- Test on a real spectrum from each of the 11 clusters.

---

## 9. Success Criteria Gate

Before declaring Phase 3 complete and moving to Phase 4:

| Check | Target | How to Verify |
|-------|--------|---------------|
| Annotation coverage | ≥ 10% of each cluster annotated (or user-defined percentage met) | `manifest.json` counts |
| Direct pick density | Mean ≥ 8 direct picks per annotated spectrum | `export_annotations.py` statistics |
| Spline sanity | Visual inspection: curves follow visible mode energy without wild oscillations | Manual review of 20 random spectra |
| Data export | Phase 4 `.npz` loads cleanly, shapes correct | Load + assert shapes |
| Resume reliability | Close and reopen app → exact same state | Integration test |
| Code quality | `ruff check .`, `ruff format .`, `ty .` all pass | CI / manual run |
| Hotkey coverage | All documented hotkeys implemented and tested | Manual + smoke test |

If **all pass** → update `experiments/MODEL_CHANGELOG.md`, proceed to Phase 4 (supervised picking model training).

If **coverage too low** → increase percentage or run a second annotation session.

If **spline produces artifacts** → switch interpolation method (PCHIP → linear) and re-annotate affected spectra.

---

## 10. Implementation Order

Recommended sequence to minimize interdependencies and enable early testing:

1. **`src/picking/interpolation.py`** + `tests/test_interpolation.py`
   - Core algorithm, well-defined interface, no UI dependencies.

2. **`src/active_learning/query.py`** + `tests/test_query_strategy.py`
   - Deterministic, testable offline.

3. **`src/picking/annotation_io.py`**
   - Save/load `.npz` annotation files, session manifest management.
   - Depends only on `numpy` + `pathlib`.

4. **`scripts/phase3_active_learning/prepare_session.py`**
   - CLI pre-flight. Validate query + interpolation modules are wired correctly.

5. **`src/picking/annotation_app.py`**
   - tkinter + matplotlib scaffolding.
   - Start with display + click-to-pick only.
   - Add spline overlay.
   - Add hotkeys one at a time.
   - Add auto-save last.

6. **`scripts/phase3_active_learning/launch_app.py`**
   - Thin wrapper around `AnnotationApp`.

7. **`scripts/phase3_active_learning/export_annotations.py`**
   - Aggregate `.npz` files into Phase 4 training format.

8. **Smoke test + manual audit**
   - Annotate 5–10 spectra end-to-end.
   - Verify export pipeline.

9. **Full annotation session**
   - Lock percentage, generate order, begin labeling.

10. **Export + MODEL_CHANGELOG.md update**
    - Document annotation statistics, proceed to Phase 4.

---

## 11. Resolved Design Decisions

| Question | Decision |
|----------|----------|
| Display cluster labels to expert? | **Yes** — cluster ID is shown in the UI header. No bias concern; may actually help the expert calibrate expectations per spectral type. |
| Multiple modes at same frequency? | **Phase 3 scope = fundamental mode only.** The expert picks the lowest-velocity (lowest wavenumber) visible mode. Higher-order modes are future work. |
| Re-annotation versioning? | **No versioning.** Overwriting an existing annotation is fine — no backups, no `.npz.bak`. Simple overwrite on save. |
| "Mode not visible" vs "skipped" sentinel? | **Deferred.** Both map to `-1`. If needed later, add `-2` = "mode absent". |
| Geographic info display? | **Yes, as a map.** A small subplot shows all spectra as dots (from X/Y metadata), with the current spectrum highlighted in red. Helps the expert understand spatial context and coverage. |

---

## 12. Model Change Tracking

Before the first annotation session, append to `experiments/MODEL_CHANGELOG.md`:

```
| 2026-06-10 | phase3-active-learning-v1 | Phase 3 initiated. 11 HDBSCAN clusters used as annotation groups. Query strategy: centroid-boundary 60/40 split. Interpolation: PCHIP. UI: tkinter + matplotlib. | N/A | N/A | N/A |
```

After the session completes, fill in:
- Total spectra annotated
- Mean direct picks per spectrum
- Percentage coverage per cluster
- Any interpolation method changes

---

## 13. Known Issues & Deferred Fixes

| Issue | Severity | Location | Description | Proposed Fix |
|-------|----------|----------|-------------|--------------|
| Hardcoded relative spectrum path | **Warning** | `src/picking/annotation_app.py:188` | `load_preprocessed_spectrum` uses `Path("data/processed/spectra")`. If the app is launched from any directory other than the project root, all spectrum loads fail with `FileNotFoundError`. | Resolve the path relative to the session directory or store it as an absolute path in the session config. |
| Relative `annotations_dir` in manifest | **Warning** | `src/picking/annotation_app.py:72` | `annotations_dir` is stored as a relative string in `manifest.json`. Launching from a different CWD causes annotations to be saved to the wrong location. | Resolve `annotations_dir` relative to the manifest's parent directory at load time, or store as absolute path. |

*Last updated: 2026-06-10*
