from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phase3_active_learning" / "prepare_session.py"


class TestPrepareSessionIntegration:
    """Integration tests for the prepare_session CLI."""

    def test_session_creation(self, tmp_path: Path) -> None:
        """Running the script creates a valid session directory."""
        output_dir = tmp_path / "annotations"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--percentage", "5.0",
                "--yes",
                "--name", "test-session",
                "--output-dir", str(output_dir),
                "--seed", "123",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Find the created session directory.
        session_dirs = list(output_dir.iterdir())
        assert len(session_dirs) == 1
        session_dir = session_dirs[0]

        # Config snapshot.
        config_path = session_dir / "config.yaml"
        assert config_path.exists()
        with open(config_path) as fh:
            config = yaml.safe_load(fh)
        assert config["percentage"] == 5.0
        assert config["strategy"] == "centroid_boundary"
        assert config["interleave"] is True
        assert config["seed"] == 123

        # Manifest.
        manifest_path = session_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        assert manifest["session_id"].endswith("_test-session")
        assert manifest["query_strategy"] == "centroid_boundary"
        assert manifest["percentage_per_cluster"] == 5.0
        assert "total_target" in manifest
        assert "per_cluster_target" in manifest
        assert isinstance(manifest["spectra_ordered"], list)
        assert len(manifest["spectra_ordered"]) == manifest["total_target"]
        assert manifest["total_target"] > 0

        # Spectra sub-directory.
        spectra_dir = session_dir / "spectra"
        assert spectra_dir.exists()

    def test_no_interleave_option(self, tmp_path: Path) -> None:
        """--no-interleave sets interleave=False in config."""
        output_dir = tmp_path / "annotations"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--percentage", "5.0",
                "--yes",
                "--name", "no-interleave",
                "--output-dir", str(output_dir),
                "--no-interleave",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        session_dir = list(output_dir.iterdir())[0]
        with open(session_dir / "config.yaml") as fh:
            config = yaml.safe_load(fh)
        assert config["interleave"] is False

    def test_missing_embeddings_file(self, tmp_path: Path) -> None:
        """Non-existent embeddings file causes a non-zero exit."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--embeddings", str(tmp_path / "missing.npz"),
                "--yes",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "not found" in result.stderr.lower()
