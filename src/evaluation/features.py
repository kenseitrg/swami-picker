from __future__ import annotations

import logging
import numpy as np
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _ensure_2d(spectrum: np.ndarray) -> np.ndarray:
    """Ensure the spectrum is 2-D, squeezing a leading channel dim if present.

    Args:
        spectrum: Input array of shape ``(1, 256, 256)`` or ``(256, 256)``.

    Returns:
        2-D array of shape ``(256, 256)``.
    """
    if spectrum.ndim == 3 and spectrum.shape[0] == 1:
        return spectrum[0]
    return spectrum


def extract_marginal_features(spectrum: np.ndarray) -> np.ndarray:
    """Extract 1-D marginal energy distributions from a preprocessed spectrum.

    Sums amplitudes along each spatial axis to collapse the 2-D structure
    into two 1-D energy curves.  This forces the feature representation to
    focus on spectral energy distribution rather than shared 2-D background.

    Args:
        spectrum: Preprocessed spectrum of shape ``(1, 256, 256)`` or
            ``(256, 256)``.

    Returns:
        Concatenated marginal vector of shape ``(512,)`` with dtype
        ``float64``.
    """
    sp = _ensure_2d(spectrum)
    if sp.shape != (256, 256):
        raise ValueError(f"Expected spectrum shape (256, 256), got {sp.shape}")

    # axis 0 = wavenumber (vertical); axis 1 = frequency (horizontal)
    e_freq = np.sum(sp, axis=0, dtype=np.float64)  # shape (256,)
    e_waven = np.sum(sp, axis=1, dtype=np.float64)  # shape (256,)
    return np.concatenate([e_freq, e_waven])


def build_pca_features(
    features: np.ndarray,
    n_components: int | None = None,
    variance_threshold: float = 0.90,
) -> tuple[np.ndarray, PCA]:
    """Standardize features and reduce dimensionality with PCA.

    Args:
        features: Feature matrix of shape ``(N, D)``.
        n_components: Target number of PCA components.  If ``None``, the
            minimum number of components explaining at least
            ``variance_threshold`` cumulative variance is chosen.
        variance_threshold: Cumulative variance threshold used when
            ``n_components`` is ``None``.

    Returns:
        A tuple of ``(transformed_features, pca)`` where
        ``transformed_features`` has shape ``(N, n_components)``.
    """
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)

    if n_components is None:
        # Fit a full PCA to inspect explained variance ratios
        pca_full = PCA()
        pca_full.fit(standardized)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumsum, variance_threshold)) + 1
        logger.info(
            "Auto-selected %d components for %.2f%% cumulative variance",
            n_components,
            variance_threshold * 100,
        )

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(standardized)
    return transformed, pca


def _energy_weighted_iqr(energy: np.ndarray, axis: np.ndarray) -> float:
    """Compute the energy-weighted interquartile range along an axis.

    Args:
        energy: 1-D energy distribution of shape ``(L,)``.
        axis: Axis values of shape ``(L,)``.

    Returns:
        The IQR (75th percentile − 25th percentile) of the axis weighted by
        the energy distribution.
    """
    total = float(np.sum(energy))
    if total <= 0:
        return float(axis[-1] - axis[0])

    # Normalized CDF of energy
    cdf = np.cumsum(energy) / total

    # Find axis values at 25th and 75th percentiles of the CDF
    p25 = float(np.interp(0.25, cdf, axis))
    p75 = float(np.interp(0.75, cdf, axis))
    return p75 - p25


def extract_spectral_descriptors(
    spectrum: np.ndarray,
    freq_axis: np.ndarray | None = None,
    waven_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Extract physics-informed spectral descriptors from a preprocessed spectrum.

    The descriptors encode geophysical properties such as average velocity,
    layer sharpness, energy partitioning, and dispersion-curve shape.

    Args:
        spectrum: Preprocessed spectrum of shape ``(1, 256, 256)`` or
            ``(256, 256)``.
        freq_axis: Physical frequency values for each column index, shape
            ``(256,)``.  If ``None``, uses ``np.arange(256)``.
        waven_axis: Physical wavenumber values for each row index, shape
            ``(256,)``.  If ``None``, uses ``np.arange(256)``.

    Returns:
        Descriptor vector of shape ``(20,)`` with dtype ``float64``.
    """
    sp = _ensure_2d(spectrum)
    if sp.shape != (256, 256):
        raise ValueError(f"Expected spectrum shape (256, 256), got {sp.shape}")

    if freq_axis is None:
        freq_axis = np.arange(256, dtype=np.float64)
    if waven_axis is None:
        waven_axis = np.arange(256, dtype=np.float64)

    # 1-D marginals
    e_freq = np.sum(sp, axis=0, dtype=np.float64)  # shape (256,)
    e_waven = np.sum(sp, axis=1, dtype=np.float64)  # shape (256,)

    # Guard against all-zero energy
    freq_sum = float(np.sum(e_freq))
    waven_sum = float(np.sum(e_waven))
    if freq_sum <= 0:
        freq_sum = 1.0
    if waven_sum <= 0:
        waven_sum = 1.0

    # 1. Frequency centroid
    freq_centroid = float(np.sum(freq_axis * e_freq) / freq_sum)

    # 2. Wavenumber centroid
    waven_centroid = float(np.sum(waven_axis * e_waven) / waven_sum)

    # 3. Frequency bandwidth (std)
    freq_var = float(np.sum(e_freq * (freq_axis - freq_centroid) ** 2) / freq_sum)
    freq_bw = float(np.sqrt(max(freq_var, 0.0)))

    # 4. Wavenumber bandwidth (std)
    waven_var = float(np.sum(e_waven * (waven_axis - waven_centroid) ** 2) / waven_sum)
    waven_bw = float(np.sqrt(max(waven_var, 0.0)))

    # 5. Frequency IQR
    freq_iqr = _energy_weighted_iqr(e_freq, freq_axis)

    # 6. Wavenumber IQR
    waven_iqr = _energy_weighted_iqr(e_waven, waven_axis)

    # 7. Low/High frequency energy ratio
    f_mid = float(freq_axis[len(freq_axis) // 2])
    low_mask = freq_axis < f_mid
    high_mask = ~low_mask
    low_energy = float(np.sum(e_freq[low_mask]))
    high_energy = float(np.sum(e_freq[high_mask]))
    if high_energy <= 0:
        lh_ratio = 0.0
    else:
        lh_ratio = low_energy / high_energy

    # 8–17. Peak velocities at 10 evenly-spaced frequency bands
    n_bands = 10
    band_edges = np.linspace(0, len(freq_axis), n_bands + 1, dtype=int)
    peak_velocities = np.zeros(n_bands, dtype=np.float64)
    for i in range(n_bands):
        start = band_edges[i]
        end = band_edges[i + 1]
        if start >= end:
            peak_velocities[i] = 0.0
            continue
        band_slice = sp[:, start:end]  # shape (256, band_width)
        flat_idx = int(np.argmax(band_slice))
        k_idx, rel_f_idx = divmod(flat_idx, band_slice.shape[1])
        f_idx = start + rel_f_idx
        f_peak = float(freq_axis[f_idx])
        k_peak = float(waven_axis[k_idx])
        eps = 1e-12
        peak_velocities[i] = f_peak / max(k_peak, eps)

    # 18. Frequency skewness
    freq_skew = float(skew(e_freq))

    # 19. Wavenumber skewness
    waven_skew = float(skew(e_waven))

    # 20. Total energy
    total_energy = float(np.sum(sp))

    descriptors = np.array(
        [
            freq_centroid,
            waven_centroid,
            freq_bw,
            waven_bw,
            freq_iqr,
            waven_iqr,
            lh_ratio,
            *peak_velocities.tolist(),
            freq_skew,
            waven_skew,
            total_energy,
        ],
        dtype=np.float64,
    )
    return descriptors


def standardize_features(
    features: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    """Fit a ``StandardScaler`` and transform features.

    Args:
        features: Feature matrix of shape ``(N, D)``.

    Returns:
        A tuple of ``(scaled_features, scaler)`` where ``scaled_features``
        has zero mean and unit variance per column.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler
