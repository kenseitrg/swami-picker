from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.picking.annotation_io import save_annotation, AnnotationRecord

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "phase3_active_learning"
    / "export_annotations.py"
)


@pytest.fixture
def fake_session(tmp_path: Path) -> Path:
    """Create a session directory with a few fake annotations."""
    session_dir = tmp_path / "2026-06-10_test"
    spectra_dir = session_dir / "spectra"
    spectra_dir.mkdir(parents=True)

    # Create session config with absolute embeddings path.
    config = {
        "session_id": "2026-06-10_test",
        "embeddings_path": str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / "mlp_embeddings_phase3.npz"
        ),
    }
    with open(session_dir / "config.yaml", "w") as fh:
        import yaml

        yaml.safe_dump(config, fh)

    # Create manifest.
    manifest = {
        "annotations_dir": str(spectra_dir),
        "per_cluster_target": {"0": 3},
    }
    with open(session_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh)

    # Create 3 fake annotations.
    for i, sid in enumerate(["s0", "s1", "s2"]):
        picks = np.full(256, -1, dtype=np.int16)
        mask = np.zeros(256, dtype=bool)
        # Vary number of direct picks.
        n_picks = [2, 5, 8][i]
        for j in range(n_picks):
            freq = 10 + j * 20
            waven = 50 + j * 10
            picks[freq] = waven
            mask[freq] = True

        record = AnnotationRecord(
            spectrum_id=sid,
            wavenumber_picks=picks,
            direct_mask=mask,
            confidence=np.where(mask, 1.0, 0.0).astype(np.float32),
            timestamp="2026-06-10T12:00:00Z",
        )
        save_annotation(record, spectra_dir / f"{sid}.npz")

    # Create fake spectrum files.
    for sid in ["s0", "s1", "s2"]:
        npz_path = Path("data/processed/spectra") / f"{sid}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, tensor=np.ones((256, 256), dtype=np.float32))
        with open(npz_path.with_suffix(".json"), "w") as fh:
            json.dump({"spectrum_id": sid}, fh)

    return session_dir


class TestExportAnnotations:
    """Integration tests for export_annotations.py."""

    def test_min_direct_picks_filter(self, tmp_path: Path, fake_session: Path) -> None:
        """Only spectra with >= min_direct_picks are included."""
        output = tmp_path / "exported.npz"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--session-dirs",
                str(fake_session),
                "--output",
                str(output),
                "--min-direct-picks",
                "5",
                "--include-noise",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output.exists()

        data = np.load(output, allow_pickle=True)
        spectrum_ids = data["spectrum_ids"]
        assert len(spectrum_ids) == 2  # s1 (5), s2 (8)
        assert "s0" not in spectrum_ids

    def test_full_export(self, tmp_path: Path, fake_session: Path) -> None:
        """All valid spectra are exported with correct shapes."""
        output = tmp_path / "exported.npz"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--session-dirs",
                str(fake_session),
                "--output",
                str(output),
                "--min-direct-picks",
                "2",
                "--include-noise",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        data = np.load(output, allow_pickle=True)
        assert data["spectra"].shape == (3, 1, 256, 256)
        assert data["picks"].shape == (3, 256)
        assert data["direct_masks"].shape == (3, 256)
        assert data["confidences"].shape == (3, 256)
        assert data["cluster_labels"].shape == (3,)
        assert len(data["spectrum_ids"]) == 3
