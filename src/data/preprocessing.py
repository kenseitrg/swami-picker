from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.data.segy_reader import RawSpectrum
from src.utils.config import FKPipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedSpectrum:
    """A single preprocessed FK spectrum ready for model input.

    Attributes:
        tensor: Preprocessed amplitude array of shape ``(256, 256)``.
        metadata: JSON-serializable provenance dictionary.
    """

    tensor: np.ndarray
    metadata: dict[str, Any]


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


def _spectrum_id(raw: RawSpectrum) -> str:
    """Build the canonical spectrum id from a ``RawSpectrum``.

    Args:
        raw: Raw spectrum dataclass.

    Returns:
        Spectrum identifier string (e.g. ``"RL5007_50071009"``).
    """
    prefix = _extract_rl_prefix(raw.source_file)
    return f"{prefix}_{raw.station_number}"


def _interpolate_axis(axis: np.ndarray, new_length: int) -> np.ndarray:
    """Linearly interpolate a 1-D axis to a new length.

    Args:
        axis: Original monotonic axis of shape ``(N,)``.
        new_length: Desired output length.

    Returns:
        Interpolated axis of shape ``(new_length,)``.
    """
    old_indices = np.arange(len(axis), dtype=np.float32)
    new_indices = np.linspace(0, len(axis) - 1, new_length, dtype=np.float32)
    return np.interp(new_indices, old_indices, axis).astype(np.float32)


def normalize_spectrum(
    data: np.ndarray,
    method: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize spectrum amplitudes.

    Supports ``"minmax"`` (scale to ``[-1, 1]``) and ``"zscore"``
    (zero-mean, unit-variance).

    Args:
        data: Input array of arbitrary shape.
        method: Normalization method — ``"minmax"`` or ``"zscore"``.

    Returns:
        A tuple of ``(normalized_data, norm_params)`` where ``norm_params``
        contains ``min``, ``max``, ``mu``, and ``sigma``.

    Raises:
        ValueError: If ``method`` is not supported.
    """
    mu = float(np.mean(data))
    sigma = float(np.std(data))
    dmin = float(np.min(data))
    dmax = float(np.max(data))

    norm_params = {
        "min": dmin,
        "max": dmax,
        "mu": mu,
        "sigma": sigma,
    }

    if method == "minmax":
        denom = dmax - dmin
        if denom < 1e-12:
            logger.warning("Constant array in minmax normalization; returning zeros.")
            normalized = np.zeros_like(data, dtype=np.float32)
        else:
            normalized = (data - dmin) / denom * 2.0 - 1.0
    elif method == "zscore":
        denom = sigma + 1e-6
        normalized = (data - mu) / denom
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return normalized.astype(np.float32, copy=False), norm_params


def resize_spectrum(
    data: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Resize a 2-D spectrum via bilinear interpolation.

    Args:
        data: Input array of shape ``(H, W)``.
        output_size: Target size ``(H_out, W_out)``.

    Returns:
        A tuple of ``(resized_data, resize_factors)`` where
        ``resize_factors`` is ``(H_out / H, W_out / W)``.
    """
    h_in, w_in = data.shape
    h_out, w_out = output_size

    # Convert to torch tensor with batch/channel dims: (1, 1, H, W)
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(
        tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    resized_np = resized.squeeze(0).squeeze(0).numpy()

    resize_factors = (float(h_out) / float(h_in), float(w_out) / float(w_in))
    return resized_np, resize_factors


def clip_spectrum(
    data: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Clip spectrum values to a fixed dynamic range.

    Args:
        data: Input array of arbitrary shape.
        bounds: Lower and upper clipping bounds.

    Returns:
        A tuple of ``(clipped_data, clipping_bounds)``.
    """
    clipped = np.clip(data, bounds[0], bounds[1])
    return clipped, bounds


def preprocess_spectrum(
    raw: RawSpectrum,
    config: FKPipelineConfig,
) -> PreprocessedSpectrum:
    """Run the full FK preprocessing pipeline on a single raw spectrum.

    Stages (in order):
        1. Amplitude normalization (minmax or zscore).
        2. Resize to ``config.output_size`` via bilinear interpolation.
        3. Dynamic range clipping.
        4. Metadata assembly.

    Args:
        raw: Raw spectrum from the SEG-Y reader.
        config: Preprocessing configuration.

    Returns:
        A ``PreprocessedSpectrum`` containing the processed tensor and
        JSON-serializable metadata.

    Raises:
        ValueError: If ``config.normalization`` is unsupported.
    """
    data = raw.data.astype(np.float32, copy=False)
    original_shape = data.shape

    # 1. Normalization
    data, norm_params = normalize_spectrum(data, config.normalization)

    # 2. Resize
    data, resize_factors = resize_spectrum(data, config.output_size)

    # 3. Clipping
    data, clipping_bounds = clip_spectrum(data, config.clip_bounds)

    # Axis interpolation for resized axes
    freq_resized = _interpolate_axis(raw.freq_axis, config.output_size[0])
    waven_resized = _interpolate_axis(raw.waven_axis, config.output_size[1])

    metadata: dict[str, Any] = {
        "spectrum_id": _spectrum_id(raw),
        "original_shape": list(original_shape),
        "resize_factors": list(resize_factors),
        "freq_axis_original": raw.freq_axis.tolist(),
        "waven_axis_original": raw.waven_axis.tolist(),
        "freq_axis_resized": freq_resized.tolist(),
        "waven_axis_resized": waven_resized.tolist(),
        "norm_method": config.normalization,
        "norm_params": norm_params,
        "clipping_bounds": list(clipping_bounds),
        "elevation": float(raw.elevation),
        "x_coord": float(raw.x_coord),
        "y_coord": float(raw.y_coord),
        "station_number": int(raw.station_number),
        "line_number": int(raw.line_number),
        "point_number": int(raw.point_number),
        "source_file": str(raw.source_file),
    }

    return PreprocessedSpectrum(tensor=data, metadata=metadata)


def save_preprocessed_spectrum(
    spectrum: PreprocessedSpectrum,
    output_dir: Path,
) -> None:
    """Save a preprocessed spectrum to disk.

    Writes an ``.npz`` file containing the tensor and a sidecar ``.json``
    file containing the metadata.

    Args:
        spectrum: Preprocessed spectrum to persist.
        output_dir: Directory where files will be written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectrum_id = spectrum.metadata["spectrum_id"]
    npz_path = output_dir / f"{spectrum_id}.npz"
    json_path = output_dir / f"{spectrum_id}.json"

    np.savez_compressed(npz_path, tensor=spectrum.tensor)
    with open(json_path, "w") as fh:
        json.dump(spectrum.metadata, fh, indent=2)

    logger.info(
        "Saved preprocessed spectrum %s to %s and %s",
        spectrum_id,
        npz_path.name,
        json_path.name,
    )


def load_preprocessed_spectrum(
    spectrum_id: str,
    output_dir: Path,
) -> PreprocessedSpectrum:
    """Load a preprocessed spectrum from disk.

    Args:
        spectrum_id: Identifier of the spectrum to load.
        output_dir: Directory containing the ``.npz`` and ``.json`` files.

    Returns:
        The reconstructed ``PreprocessedSpectrum``.

    Raises:
        FileNotFoundError: If either the ``.npz`` or ``.json`` file is missing.
    """
    output_dir = Path(output_dir)
    npz_path = output_dir / f"{spectrum_id}.npz"
    json_path = output_dir / f"{spectrum_id}.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    with np.load(npz_path) as npz:
        tensor = npz["tensor"]

    with open(json_path) as fh:
        metadata: dict[str, Any] = json.load(fh)

    logger.debug("Loaded preprocessed spectrum %s from %s", spectrum_id, output_dir)
    return PreprocessedSpectrum(tensor=tensor, metadata=metadata)
