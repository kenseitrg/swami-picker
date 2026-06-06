from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import segyio

logger = logging.getLogger(__name__)


def ibm2ieee(ibm_bytes: bytes) -> float:
    """Convert a 4-byte IBM Float-32 value to IEEE-754 float.

    The IBM format uses a 1-bit sign, 7-bit exponent (base-16, bias 64),
    and 24-bit mantissa.  The value is interpreted as
    ``(-1)**sign * (mantissa / 2**24) * 16**(exponent - 64)``.
    A zero mantissa always yields ``0.0`` regardless of exponent.

    Args:
        ibm_bytes: Exactly 4 big-endian bytes.

    Returns:
        The equivalent Python ``float`` (IEEE-754).
    """
    if len(ibm_bytes) != 4:
        msg = f"ibm2ieee expected 4 bytes, got {len(ibm_bytes)}"
        raise ValueError(msg)

    n = int.from_bytes(ibm_bytes, "big")
    sign = (n >> 31) & 0x1
    exponent = (n >> 24) & 0x7F
    mantissa = n & 0x00FFFFFF

    if mantissa == 0:
        return 0.0

    frac = mantissa / (2**24)
    value = math.ldexp(frac, 4 * (exponent - 64))
    return -value if sign else value


@dataclass(frozen=True)
class RawSpectrum:
    """A single 2-D FK spectrum extracted from a SEG-Y file.

    Attributes:
        data: Spectrum amplitudes of shape ``(262, 400)``.
        station_number: Full 8-digit station identifier.
        line_number: Receiver line (first 4 digits of ``station_number``).
        point_number: Point number (last 4 digits of ``station_number``).
        freq_axis: Frequency values in Hz, shape ``(262,)``, sorted ascending.
        waven_axis: Wavenumber values in 1/m, shape ``(400,)``.
        elevation: Surface elevation from trace header.
        x_coord: X coordinate from trace header.
        y_coord: Y coordinate from trace header.
        source_file: Name of the originating SEG-Y file.
    """

    data: np.ndarray
    station_number: int
    line_number: int
    point_number: int
    freq_axis: np.ndarray
    waven_axis: np.ndarray
    elevation: float
    x_coord: float
    y_coord: float
    source_file: str


def _parse_trace_header(header_buf: bytearray) -> dict[str, Union[int, float]]:
    """Parse a 240-byte SEG-Y trace header using the SWAMI field mapping.

    Args:
        header_buf: 240-byte trace header buffer.

    Returns:
        Dictionary with parsed header fields.
    """
    return {
        "elevation": int.from_bytes(header_buf[40:44], "big", signed=True) / 100.0,
        "x_coord": int.from_bytes(header_buf[80:84], "big", signed=True) / 100.0,
        "y_coord": int.from_bytes(header_buf[84:88], "big", signed=True) / 100.0,
        "station_number": int.from_bytes(header_buf[200:204], "big", signed=True),
        "frequency": ibm2ieee(bytes(header_buf[228:232])),
        "kmin": ibm2ieee(bytes(header_buf[232:236])),
        "kmax": ibm2ieee(bytes(header_buf[236:240])),
    }


def _extract_rl_prefix(filename: str) -> str:
    """Extract the ``RL####`` prefix from a SEG-Y filename.

    Args:
        filename: File name (e.g. ``04_09_SWAMI_raw_spect_decim8_RL5007.sgy``).

    Returns:
        The ``RL####`` token, or the stem if no token is found.
    """
    match = re.search(r"RL\d+", filename)
    if match:
        return match.group(0)
    return Path(filename).stem


def read_spectrum_raw(filepath: Union[Path, str]) -> dict[str, RawSpectrum]:
    """Read all FK spectra from a SEG-Y file.

    Traces are grouped by station number, sorted by frequency, and assembled
    into 2-D arrays of shape ``(262, 400)``.

    Args:
        filepath: Path to the ``.sgy`` file.

    Returns:
        Mapping from ``spectrum_id`` (e.g. ``"RL5007_50071009"``) to
        ``RawSpectrum``.

    Raises:
        ValueError: If any validation gate fails (monotonic frequencies,
            inconsistent ``kmax``, wrong sample count, or mismatched trace
            counts per station).
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        msg = f"File not found: {filepath}"
        raise FileNotFoundError(msg)

    rl_prefix = _extract_rl_prefix(filepath.name)
    logger.info("Reading SEG-Y file: %s (prefix=%s)", filepath.name, rl_prefix)

    stations: dict[int, list[dict]] = {}
    file_kmax: float | None = None
    expected_samples: int | None = None

    with segyio.open(str(filepath), ignore_geometry=True) as f:
        for trace_index, trace in enumerate(f.trace):
            header = f.header[trace_index]
            hdr = _parse_trace_header(header.buf)

            # Validate sample count
            sample_count = trace.shape[0]
            if expected_samples is None:
                expected_samples = sample_count
            if sample_count != expected_samples:
                msg = (
                    f"Trace {trace_index}: sample count {sample_count} does not match "
                    f"expected {expected_samples}"
                )
                raise ValueError(msg)

            station = int(hdr["station_number"])
            freq = float(hdr["frequency"])
            kmin = float(hdr["kmin"])
            kmax = float(hdr["kmax"])

            # Validate kmin is zero
            if abs(kmin) > 1e-6:
                msg = (
                    f"Trace {trace_index} (station {station}): "
                    f"kmin {kmin} is not 0.0 within tolerance"
                )
                raise ValueError(msg)

            # Validate consistent kmax across file
            if file_kmax is None:
                file_kmax = kmax
            elif abs(kmax - file_kmax) > 1e-6:
                msg = (
                    f"Trace {trace_index} (station {station}): "
                    f"kmax {kmax} differs from file kmax {file_kmax}"
                )
                raise ValueError(msg)

            if station not in stations:
                stations[station] = []
            stations[station].append(
                {
                    "trace": trace.astype(np.float32, copy=True),
                    "frequency": freq,
                    "elevation": float(hdr["elevation"]),
                    "x_coord": float(hdr["x_coord"]),
                    "y_coord": float(hdr["y_coord"]),
                }
            )

    if expected_samples is None:
        msg = f"No traces found in {filepath}"
        raise ValueError(msg)

    assert file_kmax is not None  # guaranteed because expected_samples is not None

    # Validation: every station must have the same number of traces
    trace_counts = {st: len(traces) for st, traces in stations.items()}
    unique_counts = set(trace_counts.values())
    if len(unique_counts) != 1:
        msg = f"Inconsistent trace counts per station: {unique_counts}"
        raise ValueError(msg)

    expected_traces = unique_counts.pop()
    logger.info(
        "Found %d stations, each with %d traces, sample count=%d",
        len(stations),
        expected_traces,
        expected_samples,
    )

    result: dict[str, RawSpectrum] = {}
    for station, traces in stations.items():
        # Sort by frequency ascending
        traces.sort(key=lambda t: t["frequency"])

        freq_axis = np.array([t["frequency"] for t in traces], dtype=np.float32)

        # Validate monotonic frequency
        if not np.all(np.diff(freq_axis) > 0):
            msg = f"Station {station}: frequency values are not strictly monotonically increasing"
            raise ValueError(msg)

        data = np.stack([t["trace"] for t in traces], axis=0)
        if data.shape != (expected_traces, expected_samples):
            msg = (
                f"Station {station}: assembled data shape {data.shape} does not match "
                f"expected ({expected_traces}, {expected_samples})"
            )
            raise ValueError(msg)

        waven_axis = np.linspace(
            0.0, float(file_kmax), expected_samples, dtype=np.float32
        )

        spectrum_id = f"{rl_prefix}_{station}"
        result[spectrum_id] = RawSpectrum(
            data=data,
            station_number=station,
            line_number=station // 10000,
            point_number=station % 10000,
            freq_axis=freq_axis,
            waven_axis=waven_axis,
            elevation=traces[0]["elevation"],
            x_coord=traces[0]["x_coord"],
            y_coord=traces[0]["y_coord"],
            source_file=filepath.name,
        )

    logger.info("Assembled %d spectra from %s", len(result), filepath.name)
    return result
