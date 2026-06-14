"""Model-space ↔ physical-coordinate transforms for dispersion picks.

This module implements the matched forward/inverse transform pair required by
PROJECT_RULES §4.2 and PROJECT_PLAN.md §5.  The model operates on transposed
FK spectra where axis 0 is wavenumber and axis 1 is frequency, so a dense pick
array ``picks[f_idx] = k_idx`` maps each frequency column to one wavenumber
bin (or ``-1`` for "no pick").

The canonical transform uses the resized physical axes stored in every
preprocessed spectrum's metadata (``freq_axis_resized`` and
``waven_axis_resized``).  This avoids double interpolation through the original
grid and makes round-trip tests exact up to quantization.

Uncertainty propagation is first-order: a model pick index has a base
quantization uncertainty of 0.5 pixel, scaled by the inverse of the pick
certainty (presence probability or confidence).  The pixel uncertainty is then
multiplied by the local physical bin width to yield Hz / 1/m uncertainties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.picking.annotation_io import AnnotationRecord
from src.picking.interpolation import interpolate_picks

logger = logging.getLogger(__name__)

# Sentinel for unpicked columns in model space.
_NO_PICK: np.int16 = np.int16(-1)

# Required metadata keys for coordinate transforms.
_REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "spectrum_id",
        "original_shape",
        "resize_factors",
        "freq_axis_original",
        "waven_axis_original",
        "freq_axis_resized",
        "waven_axis_resized",
    }
)

# Default certainty/uncertainty hyperparameters.
_DEFAULT_CERTAINTY_FLOOR: float = 0.1
_DEFAULT_MAX_UNCERTAINTY_MULTIPLIER: float = 10.0


@dataclass
class PhysicalPicks:
    """Physical-unit representation of a dispersion-curve pick.

    All arrays have length ``W`` (the number of frequency columns, 256 by
    default).  ``wavenumber_inv_m`` and ``wavenumber_uncertainty_inv_m`` are
    ``NaN`` wherever the model did not predict a pick.  ``frequency_hz`` and
    ``frequency_uncertainty_hz`` are always defined because the frequency
    column is fixed by the model grid.

    Attributes:
        frequency_hz: Frequency values in Hz for each column.
        wavenumber_inv_m: Wavenumber values in 1/m for picked columns, else NaN.
        frequency_uncertainty_hz: Frequency uncertainty in Hz for each column.
        wavenumber_uncertainty_inv_m: Wavenumber uncertainty in 1/m for picked
            columns, else NaN.
        pick_certainty: Per-column certainty in ``[0, 1]``.
        valid_mask: ``True`` for columns with a valid wavenumber pick.
    """

    frequency_hz: NDArray[np.float64]
    wavenumber_inv_m: NDArray[np.float64]
    frequency_uncertainty_hz: NDArray[np.float64]
    wavenumber_uncertainty_inv_m: NDArray[np.float64]
    pick_certainty: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Validate that *metadata* contains the keys required for transforms.

    Args:
        metadata: Preprocessed spectrum metadata dictionary.

    Raises:
        ValueError: If required keys are missing or axes are malformed.
    """
    missing = _REQUIRED_METADATA_KEYS - metadata.keys()
    if missing:
        msg = f"Metadata missing required keys: {sorted(missing)}"
        raise ValueError(msg)

    for key in ("freq_axis_resized", "waven_axis_resized"):
        axis = np.asarray(metadata[key], dtype=np.float64)
        if axis.ndim != 1:
            msg = f"Metadata '{key}' must be 1-D, got shape {axis.shape}"
            raise ValueError(msg)
        if axis.size < 2:
            msg = f"Metadata '{key}' must have length >= 2, got {axis.size}"
            raise ValueError(msg)
        if not np.all(np.isfinite(axis)):
            msg = f"Metadata '{key}' contains non-finite values"
            raise ValueError(msg)

    freq_axis = np.asarray(metadata["freq_axis_resized"], dtype=np.float64)
    waven_axis = np.asarray(metadata["waven_axis_resized"], dtype=np.float64)

    # Monotonicity is required for lookup and interpolation.
    if not (np.all(np.diff(freq_axis) > 0) or np.all(np.diff(freq_axis) < 0)):
        msg = "freq_axis_resized must be strictly monotonic"
        raise ValueError(msg)
    if not (np.all(np.diff(waven_axis) > 0) or np.all(np.diff(waven_axis) < 0)):
        msg = "waven_axis_resized must be strictly monotonic"
        raise ValueError(msg)

    # Both axes should be ascending for the lookup helpers; if descending,
    # transforms still work because we use np.interp/searchsorted with care,
    # but mixed conventions in metadata are a sign of trouble.
    if freq_axis[0] > freq_axis[-1]:
        logger.warning(
            "freq_axis_resized is descending; transforms assume ascending axes"
        )
    if waven_axis[0] > waven_axis[-1]:
        logger.warning(
            "waven_axis_resized is descending; transforms assume ascending axes"
        )


def _load_axis(metadata: dict[str, Any], key: str) -> NDArray[np.float64]:
    """Load a 1-D axis from metadata and ensure it is ascending.

    Args:
        metadata: Spectrum metadata.
        key: Metadata key for the axis.

    Returns:
        Ascending 1-D float64 array.
    """
    axis = np.asarray(metadata[key], dtype=np.float64)
    if axis.size > 1 and axis[0] > axis[-1]:
        axis = axis[::-1]
    return axis


def _bin_widths(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute local bin widths for an ascending axis.

    Uses the gradient, which for uniform axes equals the constant spacing and
    for non-uniform axes gives the local spacing.

    Args:
        axis: Ascending 1-D axis.

    Returns:
        Array of bin widths with the same shape as *axis*.
    """
    widths = np.gradient(axis)
    return np.abs(widths)


def _ensure_certainty(
    certainty: NDArray[np.float32] | None,
    length: int,
    presence_probs: NDArray[np.float32] | None,
    confidence: NDArray[np.float32] | None,
    strategy: str,
) -> NDArray[np.float32]:
    """Resolve a per-column certainty array from the provided inputs.

    Args:
        certainty: Pre-computed certainty array, if available.
        length: Expected array length.
        presence_probs: Model presence probabilities, if available.
        confidence: Per-column confidence, if available.
        strategy: How to derive certainty from *presence_probs* / *confidence*.

    Returns:
        Float32 certainty array clipped to ``[0, 1]``.

    Raises:
        ValueError: If no certainty source is provided for a non-"uniform"
            strategy, or if shapes mismatch.
    """
    if certainty is not None:
        arr = np.asarray(certainty, dtype=np.float32)
    elif strategy == "presence":
        if presence_probs is None:
            msg = "strategy='presence' requires presence_probs"
            raise ValueError(msg)
        arr = np.asarray(presence_probs, dtype=np.float32)
    elif strategy == "confidence":
        if confidence is None:
            msg = "strategy='confidence' requires confidence"
            raise ValueError(msg)
        arr = np.asarray(confidence, dtype=np.float32)
    elif strategy == "uniform":
        arr = np.ones(length, dtype=np.float32)
    else:
        msg = f"Unknown certainty strategy: {strategy}"
        raise ValueError(msg)

    if arr.shape != (length,):
        msg = f"Certainty array must have shape ({length},), got {arr.shape}"
        raise ValueError(msg)

    return np.clip(arr, 0.0, 1.0)


def _sigma_multiplier(
    certainty: NDArray[np.float32],
    certainty_floor: float,
    max_multiplier: float,
) -> NDArray[np.float64]:
    """Convert certainty to a pixel-uncertainty multiplier.

    ``sigma_pixels = 0.5 * multiplier``.  A certainty of 1.0 gives the base
    quantization multiplier of 1.0; lower certainty increases the multiplier
    up to *max_multiplier*.

    Args:
        certainty: Per-column certainty in ``[0, 1]``.
        certainty_floor: Minimum certainty used to avoid division by zero.
        max_multiplier: Cap on the multiplier.

    Returns:
        Multiplier array.
    """
    clipped = np.clip(certainty, certainty_floor, 1.0)
    multiplier = 1.0 / clipped
    return np.clip(multiplier, 1.0, max_multiplier).astype(np.float64)


def model_indices_to_physical(
    picks: NDArray[np.int16] | NDArray[np.int64],
    metadata: dict[str, Any],
    *,
    presence_probs: NDArray[np.float32] | None = None,
    confidence: NDArray[np.float32] | None = None,
    certainty: NDArray[np.float32] | None = None,
    certainty_strategy: str = "presence",
    certainty_floor: float = _DEFAULT_CERTAINTY_FLOOR,
    max_uncertainty_multiplier: float = _DEFAULT_MAX_UNCERTAINTY_MULTIPLIER,
) -> PhysicalPicks:
    """Convert dense model-space picks to physical units with uncertainty.

    Args:
        picks: Dense pick array of shape ``(W,)`` where ``W`` is the number of
            frequency columns (typically 256).  ``picks[f] = k`` gives the
            wavenumber index for frequency column ``f``; ``-1`` means no pick.
        metadata: Preprocessed spectrum metadata.
        presence_probs: Optional model presence probabilities of shape ``(W,)``
            in ``[0, 1]``.  Used as the default certainty source.
        confidence: Optional per-column confidence of shape ``(W,)`` in
            ``[0, 1]``.  Used when *certainty_strategy* is ``"confidence"``.
        certainty: Optional pre-computed certainty of shape ``(W,)``.  If
            provided, takes precedence over *presence_probs* / *confidence*.
        certainty_strategy: How to derive certainty when *certainty* is not
            given: ``"presence"`` (default), ``"confidence"``, or ``"uniform"``.
        certainty_floor: Minimum certainty value to avoid infinite uncertainty.
        max_uncertainty_multiplier: Maximum multiplier for the pixel
            uncertainty, limiting uncertainty for very low-certainty picks.

    Returns:
        ``PhysicalPicks`` containing frequencies, wavenumbers, uncertainties,
        and certainties.

    Raises:
        ValueError: If metadata is invalid or array shapes mismatch.
    """
    validate_metadata(metadata)

    picks_arr = np.asarray(picks)
    if picks_arr.ndim != 1:
        msg = f"picks must be 1-D, got shape {picks_arr.shape}"
        raise ValueError(msg)

    n_cols = picks_arr.shape[0]
    freq_axis = _load_axis(metadata, "freq_axis_resized")
    waven_axis = _load_axis(metadata, "waven_axis_resized")

    if freq_axis.size != n_cols:
        msg = (
            f"picks length {n_cols} does not match freq_axis_resized "
            f"length {freq_axis.size}"
        )
        raise ValueError(msg)

    valid_mask = picks_arr >= 0
    k_idx = np.where(valid_mask, picks_arr, 0).astype(np.int64)
    k_idx = np.clip(k_idx, 0, waven_axis.size - 1)

    frequency_hz = freq_axis.copy()
    wavenumber_inv_m = np.full(n_cols, np.nan, dtype=np.float64)
    wavenumber_inv_m[valid_mask] = waven_axis[k_idx[valid_mask]]

    valid_strategies = {"presence", "confidence", "uniform"}
    if certainty_strategy not in valid_strategies:
        msg = f"Unknown certainty strategy: {certainty_strategy}"
        raise ValueError(msg)

    has_certainty_source = (
        certainty is not None or presence_probs is not None or confidence is not None
    )
    if certainty_strategy == "presence" and not has_certainty_source:
        logger.warning(
            "No certainty source provided; falling back to uniform certainty. "
            "Pass presence_probs, confidence, or certainty to enable uncertainty propagation."
        )
        certainty_strategy = "uniform"

    cert = _ensure_certainty(
        certainty,
        n_cols,
        presence_probs,
        confidence,
        certainty_strategy,
    )
    multiplier = _sigma_multiplier(
        cert,
        certainty_floor,
        max_uncertainty_multiplier,
    )

    freq_widths = _bin_widths(freq_axis)
    waven_widths = _bin_widths(waven_axis)

    frequency_uncertainty_hz = 0.5 * freq_widths * multiplier
    wavenumber_uncertainty_inv_m = np.full(n_cols, np.nan, dtype=np.float64)
    wavenumber_uncertainty_inv_m[valid_mask] = (
        0.5 * waven_widths[k_idx[valid_mask]] * multiplier[valid_mask]
    )

    return PhysicalPicks(
        frequency_hz=frequency_hz,
        wavenumber_inv_m=wavenumber_inv_m,
        frequency_uncertainty_hz=frequency_uncertainty_hz,
        wavenumber_uncertainty_inv_m=wavenumber_uncertainty_inv_m,
        pick_certainty=cert,
        valid_mask=valid_mask,
    )


def physical_picks_to_model_indices(
    frequency_hz: NDArray[np.float64] | NDArray[np.float32],
    wavenumber_inv_m: NDArray[np.float64] | NDArray[np.float32],
    metadata: dict[str, Any],
) -> list[tuple[int, int]]:
    """Convert physical (Hz, 1/m) picks to sparse model indices.

    Values outside the resized axis range or ``NaN`` are dropped.

    Args:
        frequency_hz: Array of frequency values in Hz.
        wavenumber_inv_m: Array of wavenumber values in 1/m.
        metadata: Preprocessed spectrum metadata.

    Returns:
        List of ``(frequency_index, wavenumber_index)`` tuples sorted by
        frequency index.  Indices are in ``[0, 255]``.
    """
    validate_metadata(metadata)

    f_hz = np.asarray(frequency_hz, dtype=np.float64)
    k_inv_m = np.asarray(wavenumber_inv_m, dtype=np.float64)
    if f_hz.shape != k_inv_m.shape:
        msg = (
            f"frequency_hz shape {f_hz.shape} does not match "
            f"wavenumber_inv_m shape {k_inv_m.shape}"
        )
        raise ValueError(msg)

    freq_axis = _load_axis(metadata, "freq_axis_resized")
    waven_axis = _load_axis(metadata, "waven_axis_resized")

    f_min, f_max = float(freq_axis[0]), float(freq_axis[-1])
    k_min, k_max = float(waven_axis[0]), float(waven_axis[-1])

    finite = np.isfinite(f_hz) & np.isfinite(k_inv_m)
    in_range = (
        finite
        & (f_hz >= f_min)
        & (f_hz <= f_max)
        & (k_inv_m >= k_min)
        & (k_inv_m <= k_max)
    )

    f_valid = f_hz[in_range]
    k_valid = k_inv_m[in_range]

    # np.interp for ascending axis gives the continuous index, which we round.
    f_idx_float = np.interp(
        f_valid, freq_axis, np.arange(freq_axis.size, dtype=np.float64)
    )
    k_idx_float = np.interp(
        k_valid, waven_axis, np.arange(waven_axis.size, dtype=np.float64)
    )

    f_idx = np.rint(np.clip(f_idx_float, 0, freq_axis.size - 1)).astype(np.int64)
    k_idx = np.rint(np.clip(k_idx_float, 0, waven_axis.size - 1)).astype(np.int64)

    # Deduplicate by frequency index, keeping the last wavenumber index.
    unique: dict[int, int] = {}
    for fi, ki in zip(f_idx.tolist(), k_idx.tolist()):
        unique[int(fi)] = int(ki)

    return sorted(unique.items(), key=lambda item: item[0])


def physical_picks_to_dense_model_indices(
    frequency_hz: NDArray[np.float64] | NDArray[np.float32],
    wavenumber_inv_m: NDArray[np.float64] | NDArray[np.float32],
    metadata: dict[str, Any],
) -> NDArray[np.int16]:
    """Convert physical (Hz, 1/m) picks to a dense interpolated model array.

    After mapping physical picks to sparse model indices, this function uses
    PCHIP interpolation (via ``src.picking.interpolation``) to produce a
    dense ``(W,)`` array of wavenumber indices, with ``-1`` for extrapolated
    or unpicked regions.

    Args:
        frequency_hz: Array of frequency values in Hz.
        wavenumber_inv_m: Array of wavenumber values in 1/m.
        metadata: Preprocessed spectrum metadata.

    Returns:
        Dense wavenumber-index array of shape ``(W,)``.
    """
    sparse = physical_picks_to_model_indices(frequency_hz, wavenumber_inv_m, metadata)
    dense, _ = interpolate_picks(sparse)
    return dense


def round_trip_error(
    picks: NDArray[np.int16] | NDArray[np.int64],
    metadata: dict[str, Any],
) -> tuple[float, float]:
    """Compute forward → inverse round-trip error in pixel indices.

    Args:
        picks: Dense model pick array of shape ``(W,)``.
        metadata: Preprocessed spectrum metadata.

    Returns:
        Tuple of ``(frequency_rmse, wavenumber_rmse)`` in pixel indices.  Only
        valid pick columns are evaluated.  Returns ``(NaN, NaN)`` if no valid
        picks exist.
    """
    physical = model_indices_to_physical(picks, metadata)
    valid = physical.valid_mask
    if not np.any(valid):
        return float("nan"), float("nan")

    recovered = physical_picks_to_dense_model_indices(
        physical.frequency_hz,
        physical.wavenumber_inv_m,
        metadata,
    )

    freq_cols = np.arange(picks.shape[0], dtype=np.int64)
    freq_error = (freq_cols[valid] - recovered[valid]).astype(np.float64)
    waven_error = (picks[valid] - recovered[valid]).astype(np.float64)

    freq_rmse = float(np.sqrt(np.mean(freq_error**2)))
    waven_rmse = float(np.sqrt(np.mean(waven_error**2)))
    return freq_rmse, waven_rmse


def inference_to_annotation_record(
    spectrum_id: str,
    picks: NDArray[np.int16] | NDArray[np.int64],
    presence_probs: NDArray[np.float32],
    confidence_threshold: float = 0.5,
) -> AnnotationRecord:
    """Convert model inference output into an ``AnnotationRecord`` for re-annotation.

    High-confidence model picks are marked as direct picks so the expert can
    review and adjust them in the existing picking application.  Low-confidence
    columns are left unpicked (``-1``) to force manual labeling.

    Args:
        spectrum_id: Canonical spectrum identifier.
        picks: Dense model pick array of shape ``(W,)``.
        presence_probs: Model presence probabilities of shape ``(W,)``.
        confidence_threshold: Minimum presence probability for a model pick to
            be marked as a direct pick.

    Returns:
        ``AnnotationRecord`` ready to be saved and loaded by the annotation app.

    Raises:
        ValueError: If array shapes mismatch.
    """
    from datetime import datetime, timezone

    picks_arr = np.asarray(picks, dtype=np.int16)
    probs_arr = np.asarray(presence_probs, dtype=np.float32)

    if picks_arr.shape != probs_arr.shape:
        msg = (
            f"picks shape {picks_arr.shape} does not match "
            f"presence_probs shape {probs_arr.shape}"
        )
        raise ValueError(msg)
    if picks_arr.ndim != 1:
        msg = f"picks must be 1-D, got shape {picks_arr.shape}"
        raise ValueError(msg)

    valid = picks_arr >= 0
    direct_mask = valid & (probs_arr >= confidence_threshold)

    wavenumber_picks = np.where(direct_mask, picks_arr, _NO_PICK)
    confidence = probs_arr.copy()
    confidence[~direct_mask] = 0.0

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return AnnotationRecord(
        spectrum_id=spectrum_id,
        wavenumber_picks=wavenumber_picks,
        direct_mask=direct_mask,
        confidence=confidence,
        timestamp=timestamp,
        version=1,
    )


def compute_spectrum_quality_score(
    picks: NDArray[np.int16] | NDArray[np.int64],
    presence_probs: NDArray[np.float32] | None = None,
    physical_picks: PhysicalPicks | None = None,
    coverage_weight: float = 0.3,
    certainty_weight: float = 0.4,
    smoothness_weight: float = 0.3,
) -> dict[str, float]:
    """Compute scalar quality metrics for an inferred dispersion curve.

    Lower scores indicate poorer quality and are useful for selecting spectra
    that need manual re-annotation.

    Args:
        picks: Dense model pick array of shape ``(W,)``.
        presence_probs: Optional model presence probabilities of shape ``(W,)``.
        physical_picks: Optional ``PhysicalPicks`` for uncertainty-aware metrics.
        coverage_weight: Weight for the coverage term.
        certainty_weight: Weight for the mean certainty term.
        smoothness_weight: Weight for the curve smoothness term.

    Returns:
        Dictionary with ``coverage``, ``mean_certainty``, ``smoothness``,
        and ``composite_score`` (higher is better, range roughly ``[0, 1]``).
    """
    picks_arr = np.asarray(picks)
    n_cols = picks_arr.shape[0]
    valid = picks_arr >= 0
    coverage = float(np.mean(valid)) if n_cols > 0 else 0.0

    if presence_probs is not None:
        probs_arr = np.asarray(presence_probs, dtype=np.float32)
        mean_certainty = float(np.mean(probs_arr[valid])) if np.any(valid) else 0.0
    elif physical_picks is not None:
        mean_certainty = (
            float(np.mean(physical_picks.pick_certainty[valid]))
            if np.any(valid)
            else 0.0
        )
    else:
        mean_certainty = float(np.mean(valid)) if n_cols > 0 else 0.0

    # Smoothness: fraction of adjacent valid picks with <= 1 pixel difference.
    if np.sum(valid) >= 2:
        valid_indices = np.nonzero(valid)[0]
        diffs = np.abs(np.diff(picks_arr[valid_indices]).astype(np.float64))
        smoothness = float(np.mean(diffs <= 1.0))
    else:
        smoothness = 0.0

    total_weight = coverage_weight + certainty_weight + smoothness_weight
    composite = (
        coverage_weight * coverage
        + certainty_weight * mean_certainty
        + smoothness_weight * smoothness
    ) / (total_weight + 1e-8)

    return {
        "coverage": coverage,
        "mean_certainty": mean_certainty,
        "smoothness": smoothness,
        "composite_score": composite,
    }


def dispersion_curve_to_dataframe(
    spectrum_id: str,
    physical: PhysicalPicks,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Export physical picks to a pandas ``DataFrame`` for inversion software.

    Args:
        spectrum_id: Spectrum identifier.
        physical: ``PhysicalPicks`` from ``model_indices_to_physical``.
        metadata: Optional metadata dictionary; if provided, ``line_number``,
            ``point_number``, ``x_coord``, and ``y_coord`` are added to each
            row when available.

    Returns:
        ``DataFrame`` with one row per valid pick.  Columns include
        ``spectrum_id``, ``frequency_hz``, ``wavenumber_inv_m``,
        ``phase_velocity_m_s``, ``frequency_uncertainty_hz``,
        ``wavenumber_uncertainty_inv_m``, ``pick_certainty``, plus optional
        geographic columns.
    """
    valid = physical.valid_mask
    if not np.any(valid):
        return pd.DataFrame()

    frequency = physical.frequency_hz[valid]
    wavenumber = physical.wavenumber_inv_m[valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = np.where(
            np.abs(wavenumber) > 1e-12,
            frequency / wavenumber,
            np.nan,
        )

    rows: dict[str, Any] = {
        "spectrum_id": [spectrum_id] * int(np.sum(valid)),
        "frequency_hz": frequency,
        "wavenumber_inv_m": wavenumber,
        "phase_velocity_m_s": velocity,
        "frequency_uncertainty_hz": physical.frequency_uncertainty_hz[valid],
        "wavenumber_uncertainty_inv_m": physical.wavenumber_uncertainty_inv_m[valid],
        "pick_certainty": physical.pick_certainty[valid],
    }

    if metadata is not None:
        for key in ("line_number", "point_number", "x_coord", "y_coord"):
            if key in metadata:
                rows[key] = [metadata[key]] * int(np.sum(valid))

    return pd.DataFrame(rows)
