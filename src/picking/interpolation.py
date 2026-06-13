from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _validate_indices(picks: list[tuple[int, int]]) -> None:
    """Validate that all pick indices lie within the model grid.

    Args:
        picks: List of ``(frequency_index, wavenumber_index)`` pairs.

    Raises:
        ValueError: If any index is outside ``[0, 255]``.
    """
    for freq_idx, waven_idx in picks:
        if not (0 <= freq_idx <= 255):
            msg = f"frequency index {freq_idx} out of bounds [0, 255]"
            raise ValueError(msg)
        if not (0 <= waven_idx <= 255):
            msg = f"wavenumber index {waven_idx} out of bounds [0, 255]"
            raise ValueError(msg)


def interpolate_picks(
    picks: list[tuple[int, int]],
) -> tuple[NDArray[np.int16], NDArray[np.bool_]]:
    """Convert sparse expert picks into a dense dispersion curve.

    Uses PCHIP (monotone cubic) interpolation when at least three points
    are provided, linear interpolation for exactly two points, and returns
    all ``-1`` for fewer than two points.  Values outside the span of the
    clicked frequencies are set to ``-1`` to avoid dangerous extrapolation.

    Args:
        picks: List of ``(frequency_index, wavenumber_index)`` pairs.
            ``frequency_index`` maps to the horizontal axis (0..255) and
            ``wavenumber_index`` maps to the vertical axis (0..255).

    Returns:
        A tuple of ``(wavenumber_picks, direct_mask)`` where
        ``wavenumber_picks`` is an ``int16`` array of shape ``(256,)``
        giving the wavenumber index for each frequency column (``-1`` = no
        pick), and ``direct_mask`` is a ``bool`` array of shape ``(256,)``
        that is ``True`` at frequencies where the expert explicitly clicked.

    Raises:
        ValueError: If any index is out of bounds.
    """
    _validate_indices(picks)

    n_grid = 256
    wavenumber_picks = np.full(n_grid, -1, dtype=np.int16)
    direct_mask = np.zeros(n_grid, dtype=np.bool_)

    if not picks:
        return wavenumber_picks, direct_mask

    # Deduplicate by frequency (keep last occurrence) and sort.
    unique: dict[int, int] = {}
    for freq_idx, waven_idx in picks:
        unique[freq_idx] = waven_idx
    sorted_items = sorted(unique.items(), key=lambda item: item[0])
    freqs = np.array([f for f, _ in sorted_items], dtype=np.int16)
    wavens = np.array([w for _, w in sorted_items], dtype=np.float64)

    direct_mask[freqs] = True

    if len(freqs) < 2:
        # Interpolation is not possible, but the direct pick itself is
        # valid data.  Store it so that save/load round-trips correctly.
        wavenumber_picks[freqs] = np.rint(np.clip(wavens, 0, n_grid - 1)).astype(
            np.int16
        )
        return wavenumber_picks, direct_mask

    freq_grid = np.arange(n_grid, dtype=np.float64)
    min_f, max_f = int(freqs.min()), int(freqs.max())
    inside = (freq_grid >= min_f) & (freq_grid <= max_f)

    if len(freqs) == 2:
        # Linear interpolation between the two clicked points.
        wavenumber_picks[inside] = np.interp(
            freq_grid[inside], freqs.astype(np.float64), wavens
        )
    else:
        # PCHIP interpolation — preserves monotonicity and avoids overshoot.
        interp = PchipInterpolator(freqs.astype(np.float64), wavens)
        interpolated = interp(freq_grid[inside])
        interpolated = np.clip(interpolated, 0, n_grid - 1)
        wavenumber_picks[inside] = interpolated

    # Round, clip, and cast.  Linear interpolation may produce floats.
    wavenumber_picks[inside] = np.rint(
        np.clip(wavenumber_picks[inside], 0, n_grid - 1)
    ).astype(np.int16)
    # Ensure unpicked regions are exactly -1.
    wavenumber_picks[~inside] = -1

    return wavenumber_picks, direct_mask


def add_pick(
    existing_picks: list[tuple[int, int]],
    freq_idx: int,
    waven_idx: int,
) -> list[tuple[int, int]]:
    """Add or replace a pick at the given frequency.

    If a pick already exists at *freq_idx* it is replaced by the new
    wavenumber index.

    Args:
        existing_picks: Current sparse picks.
        freq_idx: Frequency column index (0..255).
        waven_idx: Wavenumber row index (0..255).

    Returns:
        A new list of picks sorted by frequency index.

    Raises:
        ValueError: If either index is out of bounds.
    """
    if not (0 <= freq_idx <= 255):
        msg = f"frequency index {freq_idx} out of bounds [0, 255]"
        raise ValueError(msg)
    if not (0 <= waven_idx <= 255):
        msg = f"wavenumber index {waven_idx} out of bounds [0, 255]"
        raise ValueError(msg)

    filtered = [p for p in existing_picks if p[0] != freq_idx]
    filtered.append((freq_idx, waven_idx))
    return sorted(filtered, key=lambda p: p[0])


def remove_pick(
    existing_picks: list[tuple[int, int]],
    freq_idx: int,
) -> list[tuple[int, int]]:
    """Remove the pick at the exact frequency index.

    Args:
        existing_picks: Current sparse picks.
        freq_idx: Frequency column index to remove.

    Returns:
        A new list with the pick removed (unchanged if not present).
    """
    return [p for p in existing_picks if p[0] != freq_idx]


def snap_picks_to_maxima(
    picks: list[tuple[int, int]],
    spectrum: np.ndarray,
) -> list[tuple[int, int]]:
    """Snap each pick to the nearest positive local maximum in its column.

    For every pick at frequency ``f`` the 1-D amplitude profile
    ``spectrum[:, f]`` is scanned for positive local maxima (peaks with
    amplitude ``> 0``).  The pick is moved to the nearest such maximum.
    If no positive maximum exists in the column the pick is left
    unchanged.

    Args:
        picks: Current sparse picks.
        spectrum: Spectrum array of shape ``(256, 256)``.

    Returns:
        New picks with wavenumber indices snapped to nearest maxima.
    """
    _validate_indices(picks)
    snapped: list[tuple[int, int]] = []
    for freq_idx, waven_idx in picks:
        col = spectrum[:, freq_idx]
        peaks, _ = find_peaks(col, height=0.0)
        if len(peaks) == 0:
            snapped.append((freq_idx, waven_idx))
            continue
        nearest_peak = peaks[np.argmin(np.abs(peaks - waven_idx))]
        snapped.append((freq_idx, int(nearest_peak)))
    return snapped


def delete_picks_at_location(
    existing_picks: list[tuple[int, int]],
    freq_idx: int,
    tol: int = 1,
) -> list[tuple[int, int]]:
    """Remove picks near the specified frequency.

    Any pick whose frequency index is within *tol* columns of *freq_idx*
    (inclusive) is removed.

    Args:
        existing_picks: Current sparse picks.
        freq_idx: Centre frequency index.
        tol: Tolerance in columns (default 1).

    Returns:
        A new list with nearby picks removed.
    """
    return [p for p in existing_picks if abs(p[0] - freq_idx) > tol]
