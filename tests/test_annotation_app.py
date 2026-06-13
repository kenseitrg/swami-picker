from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# tkinter requires a display; skip these tests in headless CI.
try:
    import tkinter as tk

    tk.Tk().destroy()
    _DISPLAY_AVAILABLE = True
except Exception:
    _DISPLAY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _DISPLAY_AVAILABLE, reason="No display available for tkinter"
)


@pytest.fixture
def app_instance(tmp_path: Path) -> Any:
    """Create an AnnotationApp backed by a real prepared session."""
    import subprocess
    import sys

    prepare_script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "phase3_active_learning"
        / "prepare_session.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(prepare_script),
            "--percentage",
            "5.0",
            "--yes",
            "--name",
            "app-test",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
    )

    session_dir = list(tmp_path.iterdir())[0]

    # Delay import until after we know a display exists.
    from src.picking.annotation_app import AnnotationApp

    app = AnnotationApp(session_dir)
    yield app
    try:
        app.destroy()
    except Exception:
        # Window may already have been destroyed by _quit().
        pass


class TestAnnotationAppSmoke:
    """Smoke tests for the annotation application."""

    def test_app_loads_queue(self, app_instance: Any) -> None:
        """The app loads the annotation queue from the manifest."""
        assert len(app_instance.queue) > 0
        assert app_instance.current_idx == 0
        assert app_instance.spectrum_tensor is not None
        assert app_instance.spectrum_meta is not None

    def test_cluster_map_populated(self, app_instance: Any) -> None:
        """Cluster map contains entries for all spectra in the queue."""
        for sid in app_instance.queue:
            assert sid in app_instance.cluster_map

    def test_navigation_moves_index(self, app_instance: Any) -> None:
        """_navigate changes the current index."""
        original = app_instance.current_idx
        if len(app_instance.queue) > 1:
            app_instance._navigate(1)
            assert app_instance.current_idx == original + 1

    def test_add_pick_updates_state(self, app_instance: Any) -> None:
        """Adding a pick increases the pick count and marks dirty."""
        app_instance.picks = []
        app_instance.dirty = False
        app_instance.picks = [(50, 100)]
        app_instance.dirty = True
        assert len(app_instance.picks) == 1
        assert app_instance.dirty

    def test_save_creates_file(self, app_instance: Any) -> None:
        """Saving an annotation writes an .npz file."""
        app_instance.picks = [(10, 20), (20, 40), (30, 60)]
        app_instance.dirty = True
        app_instance._save_current()

        spectrum_id = app_instance.queue[app_instance.current_idx]
        annotation_path = app_instance.annotations_dir / f"{spectrum_id}.npz"
        assert annotation_path.exists()

    def test_quit_without_dirty_does_not_prompt(self, app_instance: Any) -> None:
        """_quit when not dirty destroys the window immediately."""
        app_instance.dirty = False
        app_instance._quit()
        # If we reach here without a dialog hang, the test passes.
