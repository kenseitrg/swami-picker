"""Unit tests for scripts/phase4_picking/export_dispersion_curves.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "phase4_picking"
    / "export_dispersion_curves.py"
)


def _make_metadata(spectrum_id: str = "RL5007_50071009") -> dict:
    """Build a minimal valid metadata dictionary."""
    return {
        "spectrum_id": spectrum_id,
        "original_shape": [262, 400],
        "resize_factors": [256 / 400, 256 / 262],
        "freq_axis_original": np.linspace(0.0, 16.0, 262).tolist(),
        "waven_axis_original": np.linspace(0.0, 0.08, 400).tolist(),
        "freq_axis_resized": np.linspace(0.0, 16.0, 256).astype(np.float32).tolist(),
        "waven_axis_resized": np.linspace(0.0, 0.08, 256).astype(np.float32).tolist(),
        "norm_method": "minmax",
        "norm_params": {"min": 0.0, "max": 1.0, "mu": 0.5, "sigma": 0.3},
        "clipping_bounds": [-1.0, 1.0],
        "elevation": 100.0,
        "x_coord": 470510.7,
        "y_coord": 6933223.3,
        "station_number": 50071009,
        "line_number": 5007,
        "point_number": 1009,
        "source_file": "test.sgy",
    }


def _make_predictions(tmp_path: Path, n_spectra: int = 2) -> Path:
    """Create a tiny predictions.npz file for testing."""
    spectrum_ids = [f"RL5007_500710{9 + i:02d}" for i in range(n_spectra)]
    picks = np.full((n_spectra, 256), -1, dtype=np.int16)
    presence_probs = np.full((n_spectra, 256), 0.0, dtype=np.float32)

    # First spectrum: valid picks with varying certainty.
    picks[0, 50:150] = np.arange(100, dtype=np.int16)
    presence_probs[0, 50:150] = np.linspace(0.5, 1.0, 100, dtype=np.float32)

    # Second spectrum: all absent.
    picks[1, :] = -1
    presence_probs[1, :] = 0.0

    metadatas = [_make_metadata(sid) for sid in spectrum_ids]

    path = tmp_path / "predictions.npz"
    np.savez_compressed(
        path,
        spectrum_ids=np.array(spectrum_ids, dtype=object),
        picks=picks,
        presence_probs=presence_probs,
        metadata=np.array(json.dumps(metadatas), dtype=object),
    )
    return path


def _run(tmp_path: Path, predictions: Path, *, extra: list[str] | None = None) -> Path:
    """Invoke the export script and return the output directory."""
    output_dir = tmp_path / "export"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--predictions",
        str(predictions),
        "--output-dir",
        str(output_dir),
    ]
    if extra:
        cmd.extend(extra)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed:\n{result.stdout}\n{result.stderr}"
    return output_dir


class TestExportFormats:
    """Tests for CSV/JSON export output."""

    def test_csv_export(self, tmp_path: Path) -> None:
        """CSV format produces one CSV per spectrum plus combined CSV."""
        predictions = _make_predictions(tmp_path)
        output_dir = _run(tmp_path, predictions, extra=["--format", "csv"])

        csv_dir = output_dir / "csv"
        assert csv_dir.exists()
        files = sorted(csv_dir.glob("*.csv"))
        assert len(files) == 2

        df = pd.read_csv(files[0])
        assert set(df.columns) >= {
            "spectrum_id",
            "frequency_hz",
            "wavenumber_inv_m",
            "phase_velocity_m_s",
            "frequency_uncertainty_hz",
            "wavenumber_uncertainty_inv_m",
            "pick_certainty",
            "line_number",
            "point_number",
            "x_coord",
            "y_coord",
        }
        assert len(df) == 100
        assert (df["spectrum_id"] == "RL5007_50071009").all()

        combined = output_dir / "all_dispersion_curves.csv"
        assert combined.exists()
        combined_df = pd.read_csv(combined)
        assert "model_version" in combined_df.columns

    def test_json_export(self, tmp_path: Path) -> None:
        """JSON format produces one JSON array per spectrum."""
        predictions = _make_predictions(tmp_path)
        output_dir = _run(tmp_path, predictions, extra=["--format", "json"])

        json_dir = output_dir / "json"
        assert json_dir.exists()
        files = sorted(json_dir.glob("*.json"))
        assert len(files) == 2

        with open(files[0]) as fh:
            rows = json.load(fh)
        assert isinstance(rows, list)
        assert len(rows) == 100
        assert rows[0]["spectrum_id"] == "RL5007_50071009"
        assert "phase_velocity_m_s" in rows[0]

    def test_both_export(self, tmp_path: Path) -> None:
        """Both format emits CSV and JSON files."""
        predictions = _make_predictions(tmp_path)
        output_dir = _run(tmp_path, predictions, extra=["--format", "both"])

        assert (output_dir / "csv").exists()
        assert (output_dir / "json").exists()
        assert len(list((output_dir / "csv").glob("*.csv"))) == 2
        assert len(list((output_dir / "json").glob("*.json"))) == 2


class TestExportFilters:
    """Tests for certainty and presence filters."""

    def test_skip_absent(self, tmp_path: Path) -> None:
        """--skip-absent removes rows where no pick was predicted."""
        predictions = _make_predictions(tmp_path)
        output_dir = _run(
            tmp_path,
            predictions,
            extra=["--format", "csv", "--skip-absent"],
        )

        df = pd.read_csv(output_dir / "csv" / "RL5007_50071010.csv")
        assert df.empty

        df0 = pd.read_csv(output_dir / "csv" / "RL5007_50071009.csv")
        assert not df0["wavenumber_inv_m"].isna().any()

    def test_min_certainty(self, tmp_path: Path) -> None:
        """--min-certainty drops low-certainty picks."""
        predictions = _make_predictions(tmp_path)
        output_dir = _run(
            tmp_path,
            predictions,
            extra=["--format", "csv", "--min-certainty", "0.9"],
        )

        df = pd.read_csv(output_dir / "csv" / "RL5007_50071009.csv")
        assert (df["pick_certainty"] >= 0.9).all()
        assert len(df) < 100


class TestManifestAndMetadata:
    """Tests for manifest content and model-version inference."""

    def test_manifest_written(self, tmp_path: Path) -> None:
        """A manifest.json is produced with the expected keys."""
        predictions = _make_predictions(tmp_path)
        output_dir = _run(tmp_path, predictions)

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as fh:
            manifest = json.load(fh)

        assert manifest["format"] == "both"
        assert manifest["certainty_strategy"] == "presence"
        assert manifest["spectrum_count"] == 2
        assert manifest["total_picks"] == 100
        assert len(manifest["spectra"]) == 2
        assert "coverage" in manifest["spectra"][0]
        assert "combined_csv" not in manifest

    def test_model_version_override(self, tmp_path: Path) -> None:
        """Explicit --model-version appears in outputs."""
        predictions = _make_predictions(tmp_path)
        version = "test-version-123"
        output_dir = _run(
            tmp_path,
            predictions,
            extra=["--model-version", version, "--format", "csv"],
        )

        with open(output_dir / "manifest.json") as fh:
            manifest = json.load(fh)
        assert manifest["model_version"] == version

        combined = pd.read_csv(output_dir / "all_dispersion_curves.csv")
        assert (combined["model_version"] == version).all()

    def test_model_version_inference(self, tmp_path: Path) -> None:
        """Model version is inferred from the predictions parent directory."""
        run_dir = tmp_path / "my-model-v2"
        run_dir.mkdir()
        predictions = _make_predictions(run_dir)
        output_dir = _run(tmp_path, predictions, extra=["--format", "csv"])

        with open(output_dir / "manifest.json") as fh:
            manifest = json.load(fh)
        assert manifest["model_version"] == "my-model-v2"


class TestErrors:
    """Tests for error handling."""

    def test_missing_predictions_file(self, tmp_path: Path) -> None:
        """Missing predictions file causes a non-zero exit."""
        output_dir = tmp_path / "export"
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--predictions",
            str(tmp_path / "does_not_exist.npz"),
            "--output-dir",
            str(output_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert result.returncode != 0
        assert "not found" in result.stderr.lower()

    def test_invalid_predictions_missing_key(self, tmp_path: Path) -> None:
        """Predictions missing required keys cause a non-zero exit."""
        path = tmp_path / "bad.npz"
        np.savez_compressed(path, spectrum_ids=np.array(["x"], dtype=object))
        output_dir = tmp_path / "export"
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--predictions",
            str(path),
            "--output-dir",
            str(output_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert result.returncode != 0
        assert "missing required keys" in result.stderr.lower()
