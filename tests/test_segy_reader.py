from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.segy_reader import RawSpectrum, ibm2ieee, read_spectrum_raw


class TestIbm2Ieee:
    """Unit tests for the IBM Float-32 to IEEE-754 converter."""

    def test_zero(self) -> None:
        """All-zero bytes must decode to 0.0."""
        assert ibm2ieee(b"\x00\x00\x00\x00") == 0.0

    def test_positive_one(self) -> None:
        """IBM encoding of +1.0 must decode correctly."""
        assert ibm2ieee(b"\x41\x10\x00\x00") == 1.0

    def test_negative_one(self) -> None:
        """IBM encoding of -1.0 must decode correctly."""
        assert ibm2ieee(b"\xc1\x10\x00\x00") == -1.0

    def test_one_half(self) -> None:
        """IBM encoding of 0.5 must decode correctly."""
        assert ibm2ieee(b"\x40\x80\x00\x00") == 0.5

    def test_positive_small(self) -> None:
        """A small positive IBM float must decode correctly."""
        # 0.06103515625 ≈ 1/16.384, observed as the frequency step in the data
        assert ibm2ieee(b"\x40\x10\x00\x00") == pytest.approx(0.0625)

    def test_invalid_length_raises(self) -> None:
        """Passing fewer than 4 bytes must raise ValueError."""
        with pytest.raises(ValueError, match="expected 4 bytes"):
            ibm2ieee(b"\x00\x00\x00")


class TestReadSpectrumRaw:
    """Integration tests against real SEG-Y data."""

    @pytest.fixture
    def sample_sgy(self) -> Path:
        """Return the path to a known SEG-Y test file."""
        return Path("data/04_09_SWAMI_raw_spect_decim8_RL5007.sgy")

    def test_returns_expected_number_of_spectra(self, sample_sgy: Path) -> None:
        """RL5007 contains 48 stations, each with 262 traces."""
        spectra = read_spectrum_raw(sample_sgy)
        assert len(spectra) == 48

    def test_spectrum_shapes(self, sample_sgy: Path) -> None:
        """Every spectrum must have shape (262, 400) and matching axes."""
        spectra = read_spectrum_raw(sample_sgy)
        for spec in spectra.values():
            assert isinstance(spec, RawSpectrum)
            assert spec.data.shape == (262, 400)
            assert spec.data.dtype == np.float32
            assert spec.freq_axis.shape == (262,)
            assert spec.freq_axis.dtype == np.float32
            assert spec.waven_axis.shape == (400,)
            assert spec.waven_axis.dtype == np.float32

    def test_frequency_monotonic(self, sample_sgy: Path) -> None:
        """Frequency axis must be strictly increasing for every spectrum."""
        spectra = read_spectrum_raw(sample_sgy)
        for spec in spectra.values():
            assert np.all(np.diff(spec.freq_axis) > 0)

    def test_wavenumber_axis_bounds(self, sample_sgy: Path) -> None:
        """Wavenumber axis must start at 0.0 and end near the observed kmax."""
        spectra = read_spectrum_raw(sample_sgy)
        for spec in spectra.values():
            assert spec.waven_axis[0] == 0.0
            assert spec.waven_axis[-1] == pytest.approx(0.08, abs=1e-5)

    def test_station_number_decomposition(self, sample_sgy: Path) -> None:
        """``line_number`` and ``point_number`` must partition ``station_number``."""
        spectra = read_spectrum_raw(sample_sgy)
        for spec in spectra.values():
            assert spec.line_number == spec.station_number // 10000
            assert spec.point_number == spec.station_number % 10000

    def test_spectrum_id_format(self, sample_sgy: Path) -> None:
        """Keys must follow the ``RL####_station_number`` convention."""
        spectra = read_spectrum_raw(sample_sgy)
        for sid, spec in spectra.items():
            assert sid.startswith("RL5007_")
            assert sid == f"RL5007_{spec.station_number}"

    def test_source_file_set(self, sample_sgy: Path) -> None:
        """``source_file`` must store the originating file name."""
        spectra = read_spectrum_raw(sample_sgy)
        for spec in spectra.values():
            assert spec.source_file == sample_sgy.name

    def test_file_not_found(self) -> None:
        """A missing file must raise ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            read_spectrum_raw("data/nonexistent_file.sgy")
