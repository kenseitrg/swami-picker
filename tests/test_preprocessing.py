from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.preprocessing import (
    FKPipelineConfig,
    _interpolate_axis,
    _spectrum_id,
    clip_spectrum,
    load_preprocessed_spectrum,
    normalize_spectrum,
    preprocess_spectrum,
    resize_spectrum,
    save_preprocessed_spectrum,
)
from src.data.segy_reader import RawSpectrum


def make_raw_spectrum(
    data: np.ndarray | None = None,
    station_number: int = 50071009,
    source_file: str = "04_09_SWAMI_raw_spect_decim8_RL5007.sgy",
) -> RawSpectrum:
    """Factory for a ``RawSpectrum`` with sensible defaults.

    Args:
        data: Optional 2-D amplitude array. Defaults to random ``(262, 400)``.
        station_number: Full 8-digit station identifier.
        source_file: Originating SEG-Y filename.

    Returns:
        A ``RawSpectrum`` instance.
    """
    if data is None:
        rng = np.random.default_rng(42)
        data = rng.random((262, 400), dtype=np.float32)
    freq_axis = np.linspace(0.0, 15.93, 262, dtype=np.float32)
    waven_axis = np.linspace(0.0, 0.08, 400, dtype=np.float32)
    return RawSpectrum(
        data=data,
        station_number=station_number,
        line_number=station_number // 10000,
        point_number=station_number % 10000,
        freq_axis=freq_axis,
        waven_axis=waven_axis,
        elevation=98.30,
        x_coord=470510.70,
        y_coord=6933223.30,
        source_file=source_file,
    )


class TestNormalizeSpectrum:
    """Tests for ``normalize_spectrum``."""

    def test_minmax_range(self) -> None:
        """Minmax normalization must map data to ``[-1, 1]``."""
        rng = np.random.default_rng(1)
        data = rng.random((50, 50), dtype=np.float32)
        norm, params = normalize_spectrum(data, "minmax")
        assert norm.min() == pytest.approx(-1.0, abs=1e-6)
        assert norm.max() == pytest.approx(1.0, abs=1e-6)
        assert params["min"] == pytest.approx(float(data.min()), abs=1e-6)
        assert params["max"] == pytest.approx(float(data.max()), abs=1e-6)

    def test_zscore_statistics(self) -> None:
        """Z-score normalization must yield mean ~0 and std ~1."""
        rng = np.random.default_rng(2)
        data = rng.random((100, 100), dtype=np.float32) * 10.0 + 5.0
        norm, params = normalize_spectrum(data, "zscore")
        assert pytest.approx(float(norm.mean()), abs=1e-5) == 0.0
        assert pytest.approx(float(norm.std()), abs=1e-5) == 1.0
        assert params["mu"] == pytest.approx(float(data.mean()), abs=1e-5)
        assert params["sigma"] == pytest.approx(float(data.std()), abs=1e-5)

    def test_constant_array_minmax(self) -> None:
        """Minmax on a constant array must return zeros without crashing."""
        data = np.ones((10, 10), dtype=np.float32) * 5.0
        norm, params = normalize_spectrum(data, "minmax")
        assert np.allclose(norm, 0.0)
        assert params["min"] == 5.0
        assert params["max"] == 5.0

    def test_unsupported_method_raises(self) -> None:
        """An unsupported normalization string must raise ``ValueError``."""
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            normalize_spectrum(np.zeros((5, 5)), "unknown")


class TestResizeSpectrum:
    """Tests for ``resize_spectrum``."""

    def test_output_shape(self) -> None:
        """Resizing ``(262, 400)`` to ``(256, 256)`` must yield the target shape."""
        data = np.random.default_rng(3).random((262, 400), dtype=np.float32)
        resized, factors = resize_spectrum(data, (256, 256))
        assert resized.shape == (256, 256)
        assert factors == pytest.approx((256 / 262, 256 / 400))

    def test_resize_preserves_dtype(self) -> None:
        """Resized output must remain ``float32``."""
        data = np.ones((100, 100), dtype=np.float32)
        resized, _ = resize_spectrum(data, (50, 50))
        assert resized.dtype == np.float32


class TestClipSpectrum:
    """Tests for ``clip_spectrum``."""

    def test_clip_bounds(self) -> None:
        """Clipping must enforce the requested bounds."""
        data = np.array([[-5.0, 0.0, 5.0]], dtype=np.float32)
        clipped, bounds = clip_spectrum(data, (-3.0, 3.0))
        assert np.all(clipped >= -3.0)
        assert np.all(clipped <= 3.0)
        assert bounds == (-3.0, 3.0)


class TestInterpolateAxis:
    """Tests for ``_interpolate_axis``."""

    def test_length_and_endpoints(self) -> None:
        """Interpolated axis must have the requested length and span."""
        axis = np.linspace(0.0, 10.0, 11, dtype=np.float32)
        new = _interpolate_axis(axis, 21)
        assert new.shape == (21,)
        assert new[0] == pytest.approx(0.0, abs=1e-6)
        assert new[-1] == pytest.approx(10.0, abs=1e-6)


class TestPreprocessSpectrum:
    """Tests for the full ``preprocess_spectrum`` pipeline."""

    @pytest.fixture
    def config(self) -> FKPipelineConfig:
        """Return a default preprocessing config."""
        return FKPipelineConfig()

    def test_output_tensor_shape(self, config: FKPipelineConfig) -> None:
        """The output tensor must have shape ``(256, 256)``."""
        raw = make_raw_spectrum()
        out = preprocess_spectrum(raw, config)
        assert out.tensor.shape == (256, 256)
        assert out.tensor.dtype == np.float32

    def test_metadata_completeness(self, config: FKPipelineConfig) -> None:
        """Metadata must contain every key required by the FK schema."""
        raw = make_raw_spectrum()
        out = preprocess_spectrum(raw, config)
        required_keys = {
            "spectrum_id",
            "original_shape",
            "resize_factors",
            "freq_axis_original",
            "waven_axis_original",
            "freq_axis_resized",
            "waven_axis_resized",
            "norm_method",
            "norm_params",
            "clipping_bounds",
            "elevation",
            "x_coord",
            "y_coord",
            "station_number",
            "line_number",
            "point_number",
            "source_file",
        }
        assert required_keys.issubset(set(out.metadata.keys()))

    def test_metadata_values(self, config: FKPipelineConfig) -> None:
        """Key metadata fields must match the raw spectrum."""
        raw = make_raw_spectrum()
        out = preprocess_spectrum(raw, config)
        assert out.metadata["original_shape"] == [262, 400]
        assert out.metadata["norm_method"] == "minmax"
        assert out.metadata["station_number"] == raw.station_number
        assert out.metadata["line_number"] == raw.line_number
        assert out.metadata["point_number"] == raw.point_number
        assert out.metadata["source_file"] == raw.source_file
        assert out.metadata["spectrum_id"] == _spectrum_id(raw)

    def test_zscore_pipeline(self) -> None:
        """Using z-score normalization must update the metadata accordingly."""
        config = FKPipelineConfig(normalization="zscore")
        raw = make_raw_spectrum()
        out = preprocess_spectrum(raw, config)
        assert out.metadata["norm_method"] == "zscore"
        # After z-score + resize + clip, values should be within clip bounds
        assert out.tensor.min() >= config.clip_bounds[0]
        assert out.tensor.max() <= config.clip_bounds[1]

    def test_spectrum_id_from_non_rl_filename(self) -> None:
        """Filenames without an RL prefix must fall back to the stem."""
        raw = make_raw_spectrum(source_file="my_data.sgy")
        assert _spectrum_id(raw) == "my_data_50071009"


class TestSaveLoad:
    """Tests for ``save_preprocessed_spectrum`` and ``load_preprocessed_spectrum``."""

    def test_roundtrip(self) -> None:
        """Saving and loading must restore the original tensor and metadata."""
        config = FKPipelineConfig()
        raw = make_raw_spectrum()
        processed = preprocess_spectrum(raw, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            save_preprocessed_spectrum(processed, out_dir)

            loaded = load_preprocessed_spectrum(
                processed.metadata["spectrum_id"],
                out_dir,
            )

        assert loaded.tensor.shape == processed.tensor.shape
        assert np.allclose(loaded.tensor, processed.tensor)
        assert loaded.metadata == processed.metadata

    def test_load_missing_raises(self) -> None:
        """Loading a missing spectrum must raise ``FileNotFoundError``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_preprocessed_spectrum("missing_id", Path(tmpdir))


class TestFKPipelineConfig:
    """Tests for ``FKPipelineConfig`` serialization."""

    def test_to_dict_roundtrip(self) -> None:
        """``to_dict`` must capture all fields."""
        cfg = FKPipelineConfig(
            normalization="zscore",
            val_lines=[5115, 5259],
        )
        d = cfg.to_dict()
        assert d["normalization"] == "zscore"
        assert d["val_lines"] == [5115, 5259]

    def test_save_load_yaml(self) -> None:
        """Saving to YAML and loading back must restore the config exactly."""
        original = FKPipelineConfig(
            raw_data_dir="/custom/raw",
            output_dir="/custom/out",
            normalization="zscore",
            clip_bounds=(-4.0, 4.0),
            output_size=(128, 128),
            val_lines=[1000, 2000],
            random_seed=123,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            original.save_yaml(path)
            loaded = FKPipelineConfig.from_yaml(path)

        assert loaded.raw_data_dir == original.raw_data_dir
        assert loaded.output_dir == original.output_dir
        assert loaded.normalization == original.normalization
        assert loaded.clip_bounds == original.clip_bounds
        assert loaded.output_size == original.output_size
        assert loaded.val_lines == original.val_lines
        assert loaded.random_seed == original.random_seed

    def test_from_yaml_missing_file(self) -> None:
        """Loading a missing YAML file must raise ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            FKPipelineConfig.from_yaml(Path("/nonexistent/config.yaml"))

    def test_from_yaml_unknown_key(self) -> None:
        """YAML with unexpected keys must raise ``TypeError``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yaml"
            with open(path, "w") as fh:
                fh.write("unknown_key: 123\n")
            with pytest.raises(TypeError, match="Unexpected config keys"):
                FKPipelineConfig.from_yaml(path)
