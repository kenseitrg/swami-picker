from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from src.picking.annotation_io import (
    AnnotationRecord,
    compute_confidence,
    create_session_manifest,
    load_annotation,
    load_session_manifest,
    save_annotation,
    save_session_manifest,
)


class TestAnnotationRecord:
    """Tests for :class:`AnnotationRecord` validation."""

    def _valid_record(self) -> AnnotationRecord:
        return AnnotationRecord(
            spectrum_id="RL5007_50071009",
            wavenumber_picks=np.full(256, -1, dtype=np.int16),
            direct_mask=np.zeros(256, dtype=bool),
            confidence=np.zeros(256, dtype=np.float32),
            timestamp="2026-06-10T14:30:00Z",
        )

    def test_valid_record(self) -> None:
        """A correctly shaped record constructs without error."""
        record = self._valid_record()
        assert record.spectrum_id == "RL5007_50071009"

    def test_wrong_shape_raises(self) -> None:
        """Mismatched array shape raises ValueError."""
        with pytest.raises(ValueError, match="wavenumber_picks must have shape"):
            AnnotationRecord(
                spectrum_id="test",
                wavenumber_picks=np.array([1, 2, 3], dtype=np.int16),
                direct_mask=np.zeros(256, dtype=bool),
                confidence=np.zeros(256, dtype=np.float32),
                timestamp="now",
            )

    def test_wrong_dtype_raises(self) -> None:
        """Mismatched array dtype raises ValueError."""
        with pytest.raises(ValueError, match="wavenumber_picks must be int16"):
            AnnotationRecord(
                spectrum_id="test",
                wavenumber_picks=np.full(256, -1, dtype=np.int32),
                direct_mask=np.zeros(256, dtype=bool),
                confidence=np.zeros(256, dtype=np.float32),
                timestamp="now",
            )


class TestComputeConfidence:
    """Tests for :func:`compute_confidence`."""

    def test_all_unpicked(self) -> None:
        """All -1 picks yield zero confidence."""
        picks = np.full(256, -1, dtype=np.int16)
        mask = np.zeros(256, dtype=bool)
        conf = compute_confidence(picks, mask)
        assert_array_equal(conf, np.zeros(256, dtype=np.float32))

    def test_direct_picks(self) -> None:
        """Direct picks get confidence 1.0."""
        picks = np.full(256, -1, dtype=np.int16)
        mask = np.zeros(256, dtype=bool)
        picks[10] = 50
        mask[10] = True
        conf = compute_confidence(picks, mask)
        assert conf[10] == 1.0
        assert (conf[~mask] == 0.0).all()

    def test_interpolated_region(self) -> None:
        """Interpolated (non-direct, non--1) regions get 0.5."""
        picks = np.full(256, -1, dtype=np.int16)
        mask = np.zeros(256, dtype=bool)
        # Simulate an interpolated region between freq 10 and 20.
        picks[10] = 50
        mask[10] = True
        picks[15] = 55
        picks[20] = 60
        mask[20] = True
        conf = compute_confidence(picks, mask)
        assert conf[10] == 1.0
        assert conf[20] == 1.0
        assert conf[15] == 0.5
        assert (conf[picks == -1] == 0.0).all()


class TestSaveLoadAnnotation:
    """Round-trip tests for annotation I/O."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Save and load produce an equivalent record."""
        original = AnnotationRecord(
            spectrum_id="RL5007_50071009",
            wavenumber_picks=np.full(256, -1, dtype=np.int16),
            direct_mask=np.zeros(256, dtype=bool),
            confidence=np.zeros(256, dtype=np.float32),
            timestamp="2026-06-10T14:30:00Z",
            version=1,
        )
        path = tmp_path / "test.npz"
        save_annotation(original, path)
        loaded = load_annotation(path)

        assert loaded.spectrum_id == original.spectrum_id
        assert_array_equal(loaded.wavenumber_picks, original.wavenumber_picks)
        assert_array_equal(loaded.direct_mask, original.direct_mask)
        assert_array_equal(loaded.confidence, original.confidence)
        assert loaded.timestamp == original.timestamp
        assert loaded.version == original.version

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_annotation(tmp_path / "missing.npz")

    def test_save_creates_directories(self, tmp_path: Path) -> None:
        """Saving to a nested path creates parent directories."""
        nested = tmp_path / "a" / "b" / "test.npz"
        record = AnnotationRecord(
            spectrum_id="x",
            wavenumber_picks=np.full(256, -1, dtype=np.int16),
            direct_mask=np.zeros(256, dtype=bool),
            confidence=np.zeros(256, dtype=np.float32),
            timestamp="now",
        )
        save_annotation(record, nested)
        assert nested.exists()

    def test_load_missing_key(self, tmp_path: Path) -> None:
        """Corrupted npz (missing key) raises KeyError."""
        path = tmp_path / "bad.npz"
        np.savez_compressed(path, spectrum_id=np.array("x", dtype=object))
        with pytest.raises(KeyError, match="Missing key"):
            load_annotation(path)


class TestSessionManifest:
    """Tests for session manifest helpers."""

    def test_create_manifest_structure(self) -> None:
        """Manifest contains all required fields."""
        manifest = create_session_manifest(
            session_id="2026-06-10_iter0",
            annotator="expert_01",
            percentage=15.0,
            query_strategy="centroid_boundary",
            per_cluster_target={0: 12, 1: 18},
            spectra_ordered=["s0", "s1"],
            annotations_dir=Path("annotations/iter0"),
        )
        assert manifest["session_id"] == "2026-06-10_iter0"
        assert manifest["annotator"] == "expert_01"
        assert manifest["percentage_per_cluster"] == 15.0
        assert manifest["total_target"] == 30
        assert manifest["per_cluster_target"] == {"0": 12, "1": 18}
        assert manifest["spectra_ordered"] == ["s0", "s1"]
        assert manifest["query_strategy"] == "centroid_boundary"
        assert "annotations_dir" in manifest
        assert "created" in manifest

    def test_round_trip_json(self, tmp_path: Path) -> None:
        """Save and load manifest round-trip correctly."""
        manifest = create_session_manifest(
            session_id="iter0",
            annotator=None,
            percentage=10.0,
            query_strategy="random",
            per_cluster_target={0: 5},
            spectra_ordered=["a", "b"],
            annotations_dir=tmp_path,
        )
        path = tmp_path / "manifest.json"
        save_session_manifest(manifest, path)
        loaded = load_session_manifest(path)
        assert loaded == manifest

    def test_load_missing_manifest(self, tmp_path: Path) -> None:
        """Loading a non-existent manifest raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_session_manifest(tmp_path / "missing.json")
