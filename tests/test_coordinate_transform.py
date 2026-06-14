"""Unit tests for src.transforms.coordinates."""

from __future__ import annotations

import numpy as np
import pytest

from src.picking.annotation_io import AnnotationRecord
from src.transforms.coordinates import (
    compute_spectrum_quality_score,
    dispersion_curve_to_dataframe,
    inference_to_annotation_record,
    model_indices_to_physical,
    physical_picks_to_dense_model_indices,
    physical_picks_to_model_indices,
    round_trip_error,
    validate_metadata,
)


def _make_metadata(
    n_freq: int = 256,
    n_waven: int = 256,
    freq_min: float = 0.0,
    freq_max: float = 16.0,
    waven_min: float = 0.0,
    waven_max: float = 0.08,
    log_freq: bool = False,
) -> dict:
    """Build synthetic metadata matching the Phase 1 schema."""
    if log_freq:
        # Resized axis stores Hz values directly; build via log spacing.
        freq_resized = (
            np.logspace(np.log10(freq_min + 1e-6), np.log10(freq_max + 1e-6), n_freq)
            - 1e-6
        )
    else:
        freq_resized = np.linspace(freq_min, freq_max, n_freq)
    waven_resized = np.linspace(waven_min, waven_max, n_waven)

    return {
        "spectrum_id": "RL5007_50071009",
        "original_shape": [262, 400],
        "resize_factors": [256 / 400, 256 / 262],
        "freq_axis_original": np.linspace(0.0, 16.0, 262).tolist(),
        "waven_axis_original": np.linspace(0.0, 0.08, 400).tolist(),
        "freq_axis_resized": freq_resized.astype(np.float32).tolist(),
        "waven_axis_resized": waven_resized.astype(np.float32).tolist(),
        "norm_method": "minmax",
        "norm_params": {"min": 0.0, "max": 1.0, "mu": 0.5, "sigma": 0.3},
        "clipping_bounds": [-1.0, 1.0],
        "elevation": 100.0,
        "x_coord": 470000.0,
        "y_coord": 6933000.0,
        "station_number": 50071009,
        "line_number": 5007,
        "point_number": 1009,
        "source_file": "test.sgy",
    }


class TestValidateMetadata:
    """Metadata validation tests."""

    def test_missing_required_key_raises(self) -> None:
        """Missing keys should raise ValueError."""
        metadata = _make_metadata()
        del metadata["freq_axis_resized"]
        with pytest.raises(ValueError, match="Metadata missing required keys"):
            validate_metadata(metadata)

    def test_non_monotonic_freq_axis_raises(self) -> None:
        """Non-monotonic axes must be rejected."""
        metadata = _make_metadata()
        metadata["freq_axis_resized"] = np.sin(np.linspace(0, 2 * np.pi, 256)).tolist()
        with pytest.raises(
            ValueError, match="freq_axis_resized must be strictly monotonic"
        ):
            validate_metadata(metadata)

    def test_non_monotonic_waven_axis_raises(self) -> None:
        """Non-monotonic wavenumber axes must be rejected."""
        metadata = _make_metadata()
        metadata["waven_axis_resized"] = np.cos(np.linspace(0, 2 * np.pi, 256)).tolist()
        with pytest.raises(
            ValueError, match="waven_axis_resized must be strictly monotonic"
        ):
            validate_metadata(metadata)

    def test_descending_axes_warn(self, caplog) -> None:
        """Descending axes should trigger a warning but pass validation."""
        metadata = _make_metadata()
        metadata["freq_axis_resized"] = np.linspace(16.0, 0.0, 256).tolist()
        metadata["waven_axis_resized"] = np.linspace(0.08, 0.0, 256).tolist()
        with caplog.at_level("WARNING"):
            validate_metadata(metadata)
        assert "descending" in caplog.text


class TestModelIndicesToPhysical:
    """Forward transform tests."""

    def test_out_of_bounds_wavenumber_raises(self) -> None:
        """Valid picks outside the wavenumber axis must raise an error."""
        metadata = _make_metadata()
        picks = np.full(256, -1, dtype=np.int16)
        picks[10] = 300  # axis length is 256
        with pytest.raises(ValueError, match="wavenumber indices exceed axis length"):
            model_indices_to_physical(picks, metadata)

    def test_nan_certainty_raises(self) -> None:
        """NaN certainty values must be rejected."""
        metadata = _make_metadata()
        picks = np.full(256, 100, dtype=np.int16)
        nan_probs = np.full(256, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="Certainty array contains NaN"):
            model_indices_to_physical(picks, metadata, presence_probs=nan_probs)

    def test_shape_and_values(self) -> None:
        """Output arrays have expected shape and physical values."""
        metadata = _make_metadata()
        picks = np.full(256, -1, dtype=np.int16)
        picks[50] = 100
        picks[200] = 200

        physical = model_indices_to_physical(picks, metadata)

        assert physical.frequency_hz.shape == (256,)
        assert physical.wavenumber_inv_m.shape == (256,)
        assert physical.valid_mask.sum() == 2

        expected_f_50 = np.asarray(metadata["freq_axis_resized"], dtype=np.float64)[50]
        expected_k_100 = np.asarray(metadata["waven_axis_resized"], dtype=np.float64)[
            100
        ]
        assert physical.frequency_hz[50] == pytest.approx(expected_f_50, abs=1e-6)
        assert physical.wavenumber_inv_m[50] == pytest.approx(expected_k_100, abs=1e-6)
        assert np.isnan(physical.wavenumber_inv_m[0])

    def test_absent_picks_are_nan(self) -> None:
        """Unpicked columns produce NaN wavenumber and wavenumber uncertainty."""
        metadata = _make_metadata()
        picks = np.full(256, -1, dtype=np.int16)
        picks[10] = 10

        physical = model_indices_to_physical(picks, metadata)

        assert np.isnan(physical.wavenumber_inv_m[0])
        assert np.isnan(physical.wavenumber_uncertainty_inv_m[0])
        assert not np.isnan(physical.frequency_hz[0])
        assert not np.isnan(physical.frequency_uncertainty_hz[0])

    def test_uncertainty_scales_with_presence_probability(self) -> None:
        """Lower presence probability increases coordinate uncertainty."""
        metadata = _make_metadata()
        picks = np.full(256, 100, dtype=np.int16)
        high_conf = np.ones(256, dtype=np.float32)
        low_conf = np.full(256, 0.1, dtype=np.float32)

        physical_high = model_indices_to_physical(
            picks, metadata, presence_probs=high_conf
        )
        physical_low = model_indices_to_physical(
            picks, metadata, presence_probs=low_conf
        )

        assert np.all(
            physical_low.wavenumber_uncertainty_inv_m
            > physical_high.wavenumber_uncertainty_inv_m
        )
        assert np.all(
            physical_low.frequency_uncertainty_hz
            > physical_high.frequency_uncertainty_hz
        )

    def test_uniform_certainty_strategy(self) -> None:
        """Uniform strategy gives identical uncertainty for all columns."""
        metadata = _make_metadata()
        picks = np.full(256, 100, dtype=np.int16)
        physical = model_indices_to_physical(
            picks, metadata, certainty_strategy="uniform"
        )

        assert np.allclose(
            physical.wavenumber_uncertainty_inv_m,
            physical.wavenumber_uncertainty_inv_m[0],
        )

    def test_confidence_strategy_requires_confidence(self) -> None:
        """strategy='confidence' without confidence input raises."""
        metadata = _make_metadata()
        picks = np.full(256, 100, dtype=np.int16)
        with pytest.raises(
            ValueError, match="strategy='confidence' requires confidence"
        ):
            model_indices_to_physical(picks, metadata, certainty_strategy="confidence")

    def test_invalid_certainty_strategy(self) -> None:
        """Unknown certainty strategy raises."""
        metadata = _make_metadata()
        picks = np.full(256, 100, dtype=np.int16)
        with pytest.raises(ValueError, match="Unknown certainty strategy"):
            model_indices_to_physical(picks, metadata, certainty_strategy="magic")

    def test_picks_length_mismatch_raises(self) -> None:
        """picks length must match freq_axis_resized length."""
        metadata = _make_metadata()
        picks = np.full(128, -1, dtype=np.int16)
        with pytest.raises(ValueError, match="does not match freq_axis_resized length"):
            model_indices_to_physical(picks, metadata)


class TestPhysicalToModel:
    """Inverse transform tests."""

    def test_physical_picks_to_sparse_indices(self) -> None:
        """Physical values map to the expected model indices."""
        metadata = _make_metadata()
        freq_axis = np.asarray(metadata["freq_axis_resized"], dtype=np.float64)
        waven_axis = np.asarray(metadata["waven_axis_resized"], dtype=np.float64)

        f_hz = np.array([freq_axis[50], freq_axis[100]])
        k_inv_m = np.array([waven_axis[75], waven_axis[150]])

        sparse = physical_picks_to_model_indices(f_hz, k_inv_m, metadata)

        assert sparse == [(50, 75), (100, 150)]

    def test_out_of_bounds_dropped(self) -> None:
        """Physical values outside the axis range are dropped."""
        metadata = _make_metadata()
        f_hz = np.array([-1.0, 100.0])
        k_inv_m = np.array([0.04, 0.04])

        sparse = physical_picks_to_model_indices(f_hz, k_inv_m, metadata)
        assert sparse == []

    def test_nan_dropped(self) -> None:
        """NaN physical values are dropped."""
        metadata = _make_metadata()
        f_hz = np.array([8.0, np.nan])
        k_inv_m = np.array([0.04, 0.04])

        sparse = physical_picks_to_model_indices(f_hz, k_inv_m, metadata)
        assert len(sparse) == 1

    def test_dense_inverse_interpolates(self) -> None:
        """Dense inverse uses interpolation between sparse picks."""
        metadata = _make_metadata()
        freq_axis = np.asarray(metadata["freq_axis_resized"], dtype=np.float64)
        waven_axis = np.asarray(metadata["waven_axis_resized"], dtype=np.float64)

        f_hz = np.array([freq_axis[50], freq_axis[150]])
        k_inv_m = np.array([waven_axis[50], waven_axis[150]])

        dense = physical_picks_to_dense_model_indices(f_hz, k_inv_m, metadata)

        assert dense[50] == 50
        assert dense[150] == 150
        assert np.all(dense[:50] == -1)
        assert np.all(dense[150 + 1 :] == -1)
        # Interpolated region should be monotonic between the two picks.
        inside = dense[50:151]
        assert np.all(np.diff(inside.astype(np.float64)) >= -0.5)


class TestRoundTrip:
    """Round-trip accuracy tests."""

    def test_round_trip_linear_axes(self) -> None:
        """Round-trip on uniform linear axes is within quantization tolerance."""
        metadata = _make_metadata()
        picks = np.full(256, -1, dtype=np.int16)
        picks[50:200:10] = np.arange(50, 200, 10, dtype=np.int16)

        physical = model_indices_to_physical(picks, metadata)
        recovered = physical_picks_to_dense_model_indices(
            physical.frequency_hz,
            physical.wavenumber_inv_m,
            metadata,
        )

        valid = picks >= 0
        assert np.allclose(picks[valid], recovered[valid], atol=1)

        waven_rmse, coverage = round_trip_error(picks, metadata)
        assert waven_rmse <= 0.5
        assert coverage == 1.0

    def test_round_trip_non_identity_picks(self) -> None:
        """Round-trip with constant wavenumber index catches the freq-rmse bug."""
        metadata = _make_metadata()
        picks = np.full(256, -1, dtype=np.int16)
        picks[50:200:10] = 100

        waven_rmse, coverage = round_trip_error(picks, metadata)
        assert waven_rmse <= 0.5
        assert coverage == 1.0

    def test_round_trip_non_uniform_axes(self) -> None:
        """Round-trip on log-spaced frequency axis is still accurate."""
        metadata = _make_metadata(log_freq=True)
        picks = np.full(256, -1, dtype=np.int16)
        picks[50:200:10] = np.arange(50, 200, 10, dtype=np.int16)

        waven_rmse, coverage = round_trip_error(picks, metadata)
        assert waven_rmse <= 1.0
        assert coverage >= 0.95

    def test_round_trip_no_valid_picks(self) -> None:
        """All-absent picks return NaN RMSE."""
        metadata = _make_metadata()
        picks = np.full(256, -1, dtype=np.int16)
        waven_rmse, coverage = round_trip_error(picks, metadata)
        assert np.isnan(waven_rmse)
        assert np.isnan(coverage)


class TestInferenceToAnnotation:
    """Inference → annotation bridge tests."""

    def test_high_confidence_becomes_direct_pick(self) -> None:
        """High-confidence predictions are marked direct with full confidence."""
        picks = np.full(256, -1, dtype=np.int16)
        picks[10] = 20
        picks[20] = 40
        probs = np.zeros(256, dtype=np.float32)
        probs[10] = 0.9
        probs[20] = 0.4

        record = inference_to_annotation_record(
            "test_id", picks, probs, confidence_threshold=0.5
        )

        assert isinstance(record, AnnotationRecord)
        assert record.wavenumber_picks[10] == 20
        assert record.direct_mask[10]
        assert record.confidence[10] == pytest.approx(0.9)
        assert not record.direct_mask[20]
        assert record.wavenumber_picks[20] == -1

    def test_absent_picks_are_not_direct(self) -> None:
        """Model absent class (-1) is never marked as a direct pick."""
        picks = np.full(256, -1, dtype=np.int16)
        probs = np.ones(256, dtype=np.float32)

        record = inference_to_annotation_record("test_id", picks, probs)
        assert not np.any(record.direct_mask)

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched shapes raise ValueError."""
        picks = np.full(256, -1, dtype=np.int16)
        probs = np.zeros(128, dtype=np.float32)
        with pytest.raises(ValueError, match="does not match presence_probs shape"):
            inference_to_annotation_record("test_id", picks, probs)


class TestRealMetadataIntegration:
    """Integration tests against actual preprocessed spectrum metadata."""

    def test_real_metadata_round_trip(self) -> None:
        """Round-trip transform works with a real Phase 1 metadata file."""
        import json
        from pathlib import Path

        real_meta_path = Path("data/processed/spectra/RL5007_50071009.json")
        if not real_meta_path.exists():
            pytest.skip("Real metadata file not available")

        with open(real_meta_path) as fh:
            metadata = json.load(fh)

        picks = np.full(256, -1, dtype=np.int16)
        picks[50:200:10] = np.arange(50, 200, 10, dtype=np.int16)

        physical = model_indices_to_physical(picks, metadata)
        waven_rmse, coverage = round_trip_error(picks, metadata)

        assert physical.frequency_hz[0] == pytest.approx(0.0, abs=1e-3)
        assert physical.frequency_hz[-1] == pytest.approx(15.93, abs=1e-2)
        assert waven_rmse <= 1.0
        assert coverage >= 0.95

    def test_real_metadata_export_has_velocity(self) -> None:
        """DataFrame export from real metadata includes finite velocities."""
        import json
        from pathlib import Path

        real_meta_path = Path("data/processed/spectra/RL5007_50071009.json")
        if not real_meta_path.exists():
            pytest.skip("Real metadata file not available")

        with open(real_meta_path) as fh:
            metadata = json.load(fh)

        picks = np.full(256, -1, dtype=np.int16)
        picks[100:150] = 100

        physical = model_indices_to_physical(picks, metadata)
        df = dispersion_curve_to_dataframe("RL5007_50071009", physical, metadata)

        assert not df.empty
        assert np.all(np.isfinite(df["phase_velocity_m_s"].values))
        assert "line_number" in df.columns
        assert "x_coord" in df.columns


class TestQualityAndExport:
    """Quality scoring and export tests."""

    def test_quality_score_higher_for_better_curve(self) -> None:
        """A high-certainty, smooth, complete curve scores better."""
        good_picks = np.arange(256, dtype=np.int16)
        good_probs = np.ones(256, dtype=np.float32)

        bad_picks = np.full(256, -1, dtype=np.int16)
        bad_picks[::10] = 100
        bad_probs = np.zeros(256, dtype=np.float32)
        bad_probs[::10] = 0.3

        good_score = compute_spectrum_quality_score(good_picks, good_probs)
        bad_score = compute_spectrum_quality_score(bad_picks, bad_probs)

        assert good_score["composite_score"] > bad_score["composite_score"]
        assert good_score["coverage"] == 1.0
        assert bad_score["coverage"] == pytest.approx(0.1, abs=0.02)

    def test_quality_score_physical_picks_branch(self) -> None:
        """The physical_picks branch uses propagated uncertainty."""
        metadata = _make_metadata()
        picks = np.arange(256, dtype=np.int16)
        low_conf = np.full(256, 0.1, dtype=np.float32)
        high_conf = np.ones(256, dtype=np.float32)

        physical_low = model_indices_to_physical(
            picks, metadata, presence_probs=low_conf
        )
        physical_high = model_indices_to_physical(
            picks, metadata, presence_probs=high_conf
        )

        score_low = compute_spectrum_quality_score(picks, physical_picks=physical_low)
        score_high = compute_spectrum_quality_score(picks, physical_picks=physical_high)

        assert score_low["uncertainty_penalty"] > score_high["uncertainty_penalty"]
        assert score_low["effective_certainty"] < score_high["effective_certainty"]

    def test_quality_score_smooth_but_wrong_curve(self) -> None:
        """A smooth but physically flat/wrong curve still gets a smoothness bonus."""
        wrong_picks = np.full(256, 100, dtype=np.int16)
        right_picks = np.arange(256, dtype=np.int16)
        probs = np.ones(256, dtype=np.float32)

        wrong_score = compute_spectrum_quality_score(wrong_picks, probs)
        right_score = compute_spectrum_quality_score(right_picks, probs)

        # The wrong curve is perfectly smooth and monotonic, so its composite
        # score can be competitive; this documents the known limitation.
        assert wrong_score["smoothness"] == 1.0
        assert wrong_score["monotonicity"] == 1.0
        assert right_score["smoothness"] == 1.0

    def test_quality_score_weights_validated(self) -> None:
        """Negative weights are rejected."""
        picks = np.arange(256, dtype=np.int16)
        with pytest.raises(ValueError, match="coverage_weight must be non-negative"):
            compute_spectrum_quality_score(picks, coverage_weight=-1.0)

    def test_dataframe_export(self) -> None:
        """DataFrame export contains one row per valid pick."""
        metadata = _make_metadata()
        picks = np.full(256, -1, dtype=np.int16)
        picks[50] = 100
        picks[51] = 101

        physical = model_indices_to_physical(picks, metadata)
        df = dispersion_curve_to_dataframe("test_id", physical, metadata)

        assert len(df) == 2
        assert "frequency_hz" in df.columns
        assert "wavenumber_inv_m" in df.columns
        assert "phase_velocity_m_s" in df.columns
        assert "line_number" in df.columns
        assert df["line_number"].iloc[0] == 5007

    def test_dataframe_export_empty(self) -> None:
        """Empty picks produce an empty DataFrame."""
        metadata = _make_metadata()
        picks = np.full(256, -1, dtype=np.int16)
        physical = model_indices_to_physical(picks, metadata)
        df = dispersion_curve_to_dataframe("test_id", physical)
        assert df.empty
