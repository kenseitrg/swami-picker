from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from src.picking.interpolation import (
    add_pick,
    delete_picks_at_location,
    interpolate_picks,
    remove_pick,
    snap_picks_to_maxima,
)


class TestInterpolatePicks:
    """Tests for :func:`interpolate_picks`."""

    def test_empty_picks(self) -> None:
        """Empty pick list yields all -1 and all-False mask."""
        picks, mask = interpolate_picks([])
        assert picks.shape == (256,)
        assert mask.shape == (256,)
        assert picks.dtype == np.int16
        assert mask.dtype == bool
        assert_array_equal(picks, np.full(256, -1, dtype=np.int16))
        assert not mask.any()

    def test_single_point_no_interp(self) -> None:
        """One point cannot define a curve, but the direct pick is stored."""
        picks, mask = interpolate_picks([(50, 100)])
        assert picks[50] == 100  # direct pick preserved for round-trip
        assert (picks[:50] == -1).all()
        assert (picks[51:] == -1).all()
        assert mask[50]
        assert mask.sum() == 1

    def test_two_point_linear(self) -> None:
        """Exactly two points produce a linear segment between them."""
        picks, mask = interpolate_picks([(10, 20), (30, 60)])
        # Linear: slope = (60-20)/(30-10) = 2.0
        assert picks[10] == 20
        assert picks[30] == 60
        assert picks[20] == 40  # midpoint
        # Outside the span should be -1
        assert picks[0] == -1
        assert picks[31] == -1
        assert picks[9] == -1
        assert mask[10]
        assert mask[30]
        assert mask.sum() == 2

    def test_pchip_three_points(self) -> None:
        """PCHIP passes through three points."""
        picks, mask = interpolate_picks([(10, 50), (50, 100), (90, 80)])
        assert picks[10] == 50
        assert picks[50] == 100
        assert picks[90] == 80
        assert mask[10]
        assert mask[50]
        assert mask[90]
        assert mask.sum() == 3

    def test_pchip_monotonicity(self) -> None:
        """Monotonic input points produce monotonic PCHIP output."""
        picks, _mask = interpolate_picks([(0, 10), (50, 100), (100, 200)])
        inside = picks[0:101]
        # Remove -1 entries (shouldn't be any inside the span)
        valid = inside[inside != -1]
        diffs = np.diff(valid)
        assert (diffs >= 0).all(), "PCHIP output should be non-decreasing"

    def test_out_of_bounds_unpicked(self) -> None:
        """Indices outside [min_f, max_f] remain -1."""
        picks, _mask = interpolate_picks([(50, 100), (60, 120)])
        assert picks[49] == -1
        assert picks[61] == -1
        assert picks[50] != -1
        assert picks[60] != -1

    def test_clip_to_bounds(self) -> None:
        """Interpolated values are clipped to [0, 255]."""
        # Extreme valid points; the interpolation path must never escape
        # the model grid even if numerical rounding occurs.
        picks, _mask = interpolate_picks([(0, 0), (128, 255), (255, 0)])
        assert picks.min() >= 0
        assert picks.max() <= 255

    def test_deduplicate_by_frequency(self) -> None:
        """Duplicate frequencies keep the last occurrence."""
        picks, mask = interpolate_picks([(10, 20), (10, 30)])
        # Only one point remains → no interpolation, but direct pick stored.
        assert mask[10]
        assert mask.sum() == 1
        assert picks[10] == 30  # last value kept

    def test_sorted_output_independent_of_input_order(self) -> None:
        """Shuffled input yields the same interpolated curve."""
        ordered = [(10, 20), (50, 100), (90, 80)]
        shuffled = [(50, 100), (10, 20), (90, 80)]
        p1, m1 = interpolate_picks(ordered)
        p2, m2 = interpolate_picks(shuffled)
        assert_array_equal(p1, p2)
        assert_array_equal(m1, m2)

    def test_invalid_frequency_raises(self) -> None:
        """Out-of-bounds frequency index raises ValueError."""
        with pytest.raises(ValueError, match="frequency index.*out of bounds"):
            interpolate_picks([(256, 10)])

    def test_invalid_wavenumber_raises(self) -> None:
        """Out-of-bounds wavenumber index raises ValueError."""
        with pytest.raises(ValueError, match="wavenumber index.*out of bounds"):
            interpolate_picks([(10, 256)])


class TestAddPick:
    """Tests for :func:`add_pick`."""

    def test_add_new_pick(self) -> None:
        """Adding a new pick appends and sorts."""
        result = add_pick([(10, 20)], 30, 40)
        assert result == [(10, 20), (30, 40)]

    def test_replace_existing_pick(self) -> None:
        """Adding at an existing frequency replaces the wavenumber."""
        result = add_pick([(10, 20), (30, 40)], 10, 99)
        assert result == [(10, 99), (30, 40)]

    def test_add_pick_sorts(self) -> None:
        """Result is always sorted by frequency."""
        result = add_pick([(30, 40)], 10, 20)
        assert result == [(10, 20), (30, 40)]

    def test_invalid_index_raises(self) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            add_pick([], 256, 10)
        with pytest.raises(ValueError, match="out of bounds"):
            add_pick([], 10, -1)


class TestRemovePick:
    """Tests for :func:`remove_pick`."""

    def test_remove_existing(self) -> None:
        result = remove_pick([(10, 20), (30, 40)], 10)
        assert result == [(30, 40)]

    def test_remove_missing_is_noop(self) -> None:
        result = remove_pick([(10, 20)], 99)
        assert result == [(10, 20)]


class TestDeletePicksAtLocation:
    """Tests for :func:`delete_picks_at_location`."""

    def test_delete_within_tol(self) -> None:
        """Picks within tol (inclusive) are removed."""
        result = delete_picks_at_location([(10, 20), (11, 30), (20, 40)], 10, tol=1)
        assert result == [(20, 40)]

    def test_delete_exact_match(self) -> None:
        result = delete_picks_at_location([(10, 20)], 10, tol=0)
        assert result == []

    def test_no_match_is_noop(self) -> None:
        result = delete_picks_at_location([(10, 20)], 50, tol=1)
        assert result == [(10, 20)]


class TestSnapPicksToMaxima:
    """Tests for :func:`snap_picks_to_maxima`."""

    @pytest.fixture
    def simple_spectrum(self) -> np.ndarray:
        """A 256×256 spectrum with a clear diagonal ridge."""
        spec = np.full((256, 256), -0.5, dtype=np.float32)
        for f in range(256):
            w = int(50 + f * 0.5)
            if 0 <= w < 256:
                # Create a sharp positive peak with neighbouring dips.
                spec[w, f] = 1.0
                if w > 0:
                    spec[w - 1, f] = 0.3
                if w < 255:
                    spec[w + 1, f] = 0.3
        return spec

    def test_snap_to_ridge_peak(self, simple_spectrum: np.ndarray) -> None:
        """A pick near the ridge snaps to the exact peak index."""
        # At f=100 the ridge peak is at w=100.
        picks = [(100, 95)]  # slightly above peak
        snapped = snap_picks_to_maxima(picks, simple_spectrum)
        assert snapped == [(100, 100)]

    def test_snap_no_positive_maxima(self) -> None:
        """If no positive maxima exist the pick is unchanged."""
        spec = np.full((256, 256), -0.5, dtype=np.float32)
        picks = [(50, 100)]
        snapped = snap_picks_to_maxima(picks, spec)
        assert snapped == picks

    def test_snap_multiple_picks(self, simple_spectrum: np.ndarray) -> None:
        """All picks in the list are snapped independently."""
        picks = [(100, 95), (200, 160)]
        snapped = snap_picks_to_maxima(picks, simple_spectrum)
        # f=100 peak at w=100, f=200 peak at w=150.
        assert snapped[0] == (100, 100)
        assert snapped[1] == (200, 150)

    def test_snap_unchanged_if_already_peak(self, simple_spectrum: np.ndarray) -> None:
        """A pick already on a maximum stays put."""
        picks = [(100, 100)]
        snapped = snap_picks_to_maxima(picks, simple_spectrum)
        assert snapped == picks

    def test_snap_invalid_index_raises(self) -> None:
        """Out-of-bounds pick indices raise ValueError."""
        spec = np.zeros((256, 256), dtype=np.float32)
        with pytest.raises(ValueError, match="out of bounds"):
            snap_picks_to_maxima([(256, 10)], spec)


class TestAddRemoveIdempotent:
    """Round-trip tests."""

    def test_add_then_remove(self) -> None:
        """Add then remove returns to the original state."""
        original = [(10, 20), (30, 40)]
        added = add_pick(original, 50, 60)
        removed = remove_pick(added, 50)
        assert removed == original

    def test_remove_then_add_same_value(self) -> None:
        original = [(10, 20), (30, 40)]
        removed = remove_pick(original, 10)
        added = add_pick(removed, 10, 20)
        assert added == original
