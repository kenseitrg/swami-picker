from __future__ import annotations

import logging
import tkinter as tk
import tkinter.messagebox as messagebox
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.data.preprocessing import load_preprocessed_spectrum
from src.picking.annotation_io import (
    AnnotationRecord,
    compute_confidence,
    load_annotation,
    save_annotation,
)
from src.picking.interpolation import (
    add_pick,
    delete_picks_at_location,
    interpolate_picks,
    snap_picks_to_maxima,
)

logger = logging.getLogger(__name__)

_GRID_SIZE = 256


class AnnotationApp(tk.Tk):
    """tkinter + matplotlib annotation interface for dispersion picking.

    The main window shows a single FK spectrum with physical axis labels.
    The expert clicks on the image to place picks; a PCHIP spline
    interpolates sparse clicks into a dense dispersion curve shown as a
    red line.  Directly clicked points are shown as blue dots.

    Hotkeys
    -------
    ``Space`` — save current and load next spectrum.
    ``z`` — save current and load previous spectrum.
    ``q`` — jump backward to the nearest spectrum from a different cluster.
    ``w`` — jump forward to the nearest spectrum from a different cluster.
    ``d`` — delete the pick nearest to the current mouse X position.
    ``x`` — clear all picks on the current spectrum (with confirmation).
    ``s`` — force-save current annotation.
    ``v`` — snap all direct picks to the nearest positive local
    maximum and recompute the interpolated curve.
    ``↑`` / ``↓`` — nudge the most recently added pick up/down by one
    wavenumber index.
    ``Esc`` — quit (prompts to save if unsaved).

    Mouse
    -----
    Left click — add a pick at the clicked frequency/wavenumber.
    Right click — remove the nearest pick (within 2 columns).
    """

    def __init__(self, session_dir: Path) -> None:
        super().__init__()
        self.session_dir = Path(session_dir)

        # ── Session state ─────────────────────────────────────────────
        self.manifest = self._load_manifest()
        self.queue: list[str] = self.manifest["spectra_ordered"]
        self.annotations_dir = Path(self.manifest["annotations_dir"])
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        self.cluster_map = self._build_cluster_map()

        # Per-spectrum mutable state.
        self.current_idx = 0
        self.picks: list[tuple[int, int]] = []
        self.dirty = False
        self.spectrum_tensor: np.ndarray | None = None
        self.spectrum_meta: dict[str, Any] | None = None
        self._hover_freq_idx: int | None = None

        # ── tkinter UI ────────────────────────────────────────────────
        self.title("swami-picker — Annotation")
        self.geometry("900x850")
        self._build_ui()

        # Window close handler.
        self.protocol("WM_DELETE_WINDOW", self._quit)

        # ── Load first spectrum ───────────────────────────────────────
        if self.queue:
            self._load_current_spectrum()
        else:
            self.status_label.config(text="Annotation queue is empty.")

    # ──────────────────────────────────────────────────────────────────
    #  Setup / helpers
    # ──────────────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict[str, Any]:
        """Load and return the session manifest."""
        from src.picking.annotation_io import load_session_manifest

        return load_session_manifest(self.session_dir / "manifest.json")

    def _build_cluster_map(self) -> dict[str, int]:
        """Map spectrum_id → cluster label from the embeddings file."""
        config_path = self.session_dir / "config.yaml"
        with open(config_path) as fh:
            config: dict[str, Any] = yaml.safe_load(fh)

        embeddings_path = Path(config["embeddings_path"])
        data = np.load(embeddings_path, allow_pickle=True)
        try:
            ids = data["spectrum_ids"]
            lbls = data["labels"]
            return {str(sid): int(lbl) for sid, lbl in zip(ids, lbls)}
        finally:
            data.close()

    def _build_ui(self) -> None:
        """Assemble tkinter widgets and matplotlib canvas."""
        # Top info bar.
        self.info_frame = tk.Frame(self)
        self.info_frame.pack(fill=tk.X, padx=8, pady=(8, 0))

        self.session_label = tk.Label(
            self.info_frame, text="", font=("TkDefaultFont", 10, "bold")
        )
        self.session_label.pack(side=tk.LEFT)

        self.progress_label = tk.Label(self.info_frame, text="")
        self.progress_label.pack(side=tk.RIGHT)

        # Matplotlib canvas.
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        widget = self.canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)
        self.canvas.draw()

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        # Bottom status / hotkey bar.
        self.status_frame = tk.Frame(self)
        self.status_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.status_label = tk.Label(self.status_frame, text="", anchor=tk.W)
        self.status_label.pack(fill=tk.X)

        self.hotkey_label = tk.Label(
            self.status_frame,
            text=(
                "Space=Next  Z=Prev  Q=PrevCluster  W=NextCluster  "
                "D=Delete  V=SnapToMax  X=Clear  S=Save  Esc=Quit  ↑↓=Nudge"
            ),
            fg="gray",
        )
        self.hotkey_label.pack(fill=tk.X)

        # Keyboard bindings.
        self.bind("<space>", lambda _e: self._navigate(1))
        self.bind("z", lambda _e: self._navigate(-1))
        self.bind("q", lambda _e: self._jump_cluster(-1))
        self.bind("w", lambda _e: self._jump_cluster(1))
        self.bind("d", lambda _e: self._delete_at_cursor())
        self.bind("v", lambda _e: self._snap_picks())
        self.bind("x", lambda _e: self._clear_spectrum())
        self.bind("s", lambda _e: self._save_current())
        self.bind("<Escape>", lambda _e: self._quit())
        self.bind("<Up>", lambda _e: self._nudge_pick(-1))
        self.bind("<Down>", lambda _e: self._nudge_pick(1))

        # Ensure keyboard focus.
        self.focus_set()
        widget.focus_set()

    # ──────────────────────────────────────────────────────────────────
    #  Spectrum / annotation loading
    # ──────────────────────────────────────────────────────────────────

    def _load_current_spectrum(self) -> None:
        """Load the current spectrum and any existing annotation."""
        spectrum_id = self.queue[self.current_idx]

        try:
            spectrum = load_preprocessed_spectrum(
                spectrum_id, Path("data/processed/spectra")
            )
            self.spectrum_tensor = spectrum.tensor
            self.spectrum_meta = spectrum.metadata
        except FileNotFoundError:
            logger.error("Spectrum file not found: %s", spectrum_id)
            self.spectrum_tensor = None
            self.spectrum_meta = None
            self.picks = []
            self.dirty = False
            self._update_display()
            self._update_info()
            return

        # Load existing annotation if present.
        annotation_path = self.annotations_dir / f"{spectrum_id}.npz"
        if annotation_path.exists():
            try:
                record = load_annotation(annotation_path)
                self.picks = [
                    (int(f), int(record.wavenumber_picks[f]))
                    for f in np.flatnonzero(record.direct_mask)
                ]
                self.dirty = False
            except Exception as exc:
                logger.warning(
                    "Failed to load annotation for %s: %s", spectrum_id, exc
                )
                self.picks = []
                self.dirty = False
        else:
            self.picks = []
            self.dirty = False

        self._update_display()
        self._update_info()

    # ──────────────────────────────────────────────────────────────────
    #  Display updates
    # ──────────────────────────────────────────────────────────────────

    def _update_display(self) -> None:
        """Redraw the spectrum and pick overlay."""
        self.ax.clear()

        if self.spectrum_tensor is None or self.spectrum_meta is None:
            self.canvas.draw()
            return

        meta = self.spectrum_meta
        freq_axis = np.array(meta["freq_axis_resized"])
        waven_axis = np.array(meta["waven_axis_resized"])
        extent = (
            float(freq_axis[0]),
            float(freq_axis[-1]),
            float(waven_axis[-1]),
            float(waven_axis[0]),
        )

        # origin="upper" places array row 0 at the visual top,
        # matching seismic convention (low wavenumber / low velocity
        # at the top of the plot).
        self.ax.imshow(
            self.spectrum_tensor,
            extent=extent,
            origin="upper",
            cmap="viridis",
            aspect="auto",
        )
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Wavenumber (1/m)")
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])

        if self.picks:
            wavenumber_picks, direct_mask = interpolate_picks(self.picks)

            valid = wavenumber_picks != -1
            if valid.any():
                freqs = np.arange(_GRID_SIZE)[valid]
                wavens = wavenumber_picks[valid]
                self.ax.plot(
                    freq_axis[freqs],
                    waven_axis[wavens],
                    "r-",
                    linewidth=2,
                    alpha=0.85,
                    label="Interpolated",
                )

            # Plot direct picks using the original sparse coordinates
            # (``wavenumber_picks`` may contain -1 for single-point
            # cases where interpolation is not yet possible).
            pick_dict = dict(self.picks)
            direct_freqs = np.array(list(pick_dict.keys()), dtype=np.int64)
            direct_wavens = np.array(list(pick_dict.values()), dtype=np.int64)
            if len(direct_freqs) > 0:
                self.ax.scatter(
                    freq_axis[direct_freqs],
                    waven_axis[direct_wavens],
                    c="blue",
                    s=35,
                    zorder=5,
                    label="Direct picks",
                )

            self.ax.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def _update_info(self) -> None:
        """Update info labels and title."""
        if not self.queue:
            return

        spectrum_id = self.queue[self.current_idx]
        cluster = self.cluster_map.get(spectrum_id, -1)
        progress = f"{self.current_idx + 1} / {len(self.queue)}"

        self.session_label.config(
            text=f"{spectrum_id}  |  Cluster {cluster}"
        )
        self.progress_label.config(text=progress)
        dirty_flag = "  ● unsaved" if self.dirty else ""
        self.status_label.config(
            text=f"Direct picks: {len(self.picks)}{dirty_flag}"
        )
        self.title(
            f"swami-picker — {spectrum_id} (C{cluster}) [{progress}]"
        )

    def _update_hover_status(self, freq_idx: int, waven_idx: int) -> None:
        """Append hover coordinates to the status bar."""
        if not self.queue:
            return
        dirty_flag = "  ● unsaved" if self.dirty else ""
        self.status_label.config(
            text=(
                f"Direct picks: {len(self.picks)}{dirty_flag}  |  "
                f"Hover: freq={freq_idx}  waven={waven_idx}"
            )
        )

    # ──────────────────────────────────────────────────────────────────
    #  Interaction handlers
    # ──────────────────────────────────────────────────────────────────

    def _on_click(self, event: Any) -> None:
        """Handle mouse click on the matplotlib canvas."""
        if event.inaxes != self.ax or self.spectrum_meta is None:
            return

        meta = self.spectrum_meta
        freq_axis = np.array(meta["freq_axis_resized"])
        waven_axis = np.array(meta["waven_axis_resized"])

        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        freq_idx = self._physical_to_index(xdata, freq_axis[0], freq_axis[-1])
        waven_idx = self._physical_to_index(
            ydata, waven_axis[0], waven_axis[-1]
        )

        if event.button == 1:  # Left click — add
            self.picks = add_pick(self.picks, freq_idx, waven_idx)
            self.dirty = True
            logger.debug("Added pick at (%d, %d)", freq_idx, waven_idx)
        elif event.button in (2, 3):  # Middle / right click — remove nearest
            self.picks = delete_picks_at_location(
                self.picks, freq_idx, tol=2
            )
            self.dirty = True
            logger.debug("Removed pick near freq %d", freq_idx)

        self._update_display()
        self._update_info()

    def _on_hover(self, event: Any) -> None:
        """Track mouse position for hover display and 'd' hotkey."""
        if event.inaxes != self.ax or self.spectrum_meta is None:
            self._hover_freq_idx = None
            self._update_info()
            return

        meta = self.spectrum_meta
        freq_axis = np.array(meta["freq_axis_resized"])
        waven_axis = np.array(meta["waven_axis_resized"])

        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            self._hover_freq_idx = None
            self._update_info()
            return

        freq_idx = self._physical_to_index(xdata, freq_axis[0], freq_axis[-1])
        waven_idx = self._physical_to_index(
            ydata, waven_axis[0], waven_axis[-1]
        )
        self._hover_freq_idx = freq_idx
        self._update_hover_status(freq_idx, waven_idx)

    @staticmethod
    def _physical_to_index(
        value: float, vmin: float, vmax: float
    ) -> int:
        """Map a physical coordinate to a pixel index in ``[0, 255]``."""
        if vmax == vmin:
            return 0
        idx = int(round((value - vmin) / (vmax - vmin) * (_GRID_SIZE - 1)))
        return max(0, min(_GRID_SIZE - 1, idx))

    # ──────────────────────────────────────────────────────────────────
    #  Keyboard actions
    # ──────────────────────────────────────────────────────────────────

    def _navigate(self, delta: int) -> None:
        """Save current and move to next / previous spectrum."""
        new_idx = self.current_idx + delta
        if 0 <= new_idx < len(self.queue):
            self._go_to(new_idx)

    def _go_to(self, idx: int) -> None:
        """Save if dirty and load the spectrum at *idx*."""
        if self.dirty:
            self._save_current()
        self.current_idx = idx
        self._load_current_spectrum()

    def _jump_cluster(self, direction: int) -> None:
        """Jump to the nearest spectrum belonging to a different cluster."""
        if not self.queue:
            return
        current_cluster = self.cluster_map.get(
            self.queue[self.current_idx], -1
        )
        idx = self.current_idx + direction
        while 0 <= idx < len(self.queue):
            if self.cluster_map.get(self.queue[idx], -1) != current_cluster:
                self._go_to(idx)
                return
            idx += direction

    def _delete_at_cursor(self) -> None:
        """Delete the pick nearest to the current hover position."""
        if not self.picks:
            return

        if self._hover_freq_idx is not None:
            self.picks = delete_picks_at_location(
                self.picks, self._hover_freq_idx, tol=2
            )
        else:
            # Fallback: remove the highest-frequency pick.
            self.picks = self.picks[:-1]

        self.dirty = True
        self._update_display()
        self._update_info()

    def _nudge_pick(self, delta: int) -> None:
        """Nudge the most recently added pick up/down."""
        if not self.picks:
            return
        last_freq, last_waven = self.picks[-1]
        new_waven = max(0, min(_GRID_SIZE - 1, last_waven + delta))
        self.picks = add_pick(self.picks, last_freq, new_waven)
        self.dirty = True
        self._update_display()
        self._update_info()

    def _clear_spectrum(self) -> None:
        """Reset picks after optional confirmation."""
        if len(self.picks) > 3:
            if not messagebox.askyesno("Confirm", "Clear all picks?"):
                return
        self.picks = []
        self.dirty = True
        self._update_display()
        self._update_info()

    def _snap_picks(self) -> None:
        """Snap all picks to the nearest positive local maxima."""
        if not self.picks or self.spectrum_tensor is None:
            return
        snapped = snap_picks_to_maxima(self.picks, self.spectrum_tensor)
        if snapped != self.picks:
            self.picks = snapped
            self.dirty = True
            logger.debug("Snapped %d picks to maxima", len(self.picks))
            self._update_display()
            self._update_info()

    def _save_current(self) -> None:
        """Persist the current annotation to disk."""
        if not self.queue:
            return

        spectrum_id = self.queue[self.current_idx]

        if not self.picks:
            # No picks — remove any existing annotation file.
            annotation_path = self.annotations_dir / f"{spectrum_id}.npz"
            if annotation_path.exists():
                annotation_path.unlink()
                logger.info("Removed empty annotation for %s", spectrum_id)
            self.dirty = False
            self._update_info()
            return

        wavenumber_picks, direct_mask = interpolate_picks(self.picks)
        confidence = compute_confidence(wavenumber_picks, direct_mask)

        # Try to read existing version number.
        annotation_path = self.annotations_dir / f"{spectrum_id}.npz"
        version = 1
        if annotation_path.exists():
            try:
                existing = load_annotation(annotation_path)
                version = existing.version + 1
            except Exception:
                pass

        record = AnnotationRecord(
            spectrum_id=spectrum_id,
            wavenumber_picks=wavenumber_picks,
            direct_mask=direct_mask,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            version=version,
        )
        save_annotation(record, annotation_path)
        self.dirty = False
        self._update_info()
        logger.info("Saved annotation for %s (v%d)", spectrum_id, version)

    def _quit(self) -> None:
        """Exit with save prompt if unsaved changes exist."""
        if self.dirty:
            result = messagebox.askyesnocancel(
                "Unsaved Changes", "Save before quitting?"
            )
            if result is True:  # Yes
                self._save_current()
            elif result is None:  # Cancel
                return
            # No → quit without saving.

        self.destroy()

    # ──────────────────────────────────────────────────────────────────
    #  Public entry point
    # ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the tkinter main loop."""
        self.mainloop()
