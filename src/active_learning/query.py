from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_annotation_budget(
    labels: NDArray[np.integer],
    percentage: float,
) -> dict[int, int]:
    """Compute per-cluster annotation targets.

    Noise points (label ``-1``) are excluded from every cluster.

    Args:
        labels: Array of cluster labels of shape ``(N,)``.
        percentage: Percentage of each cluster to annotate (0–100).

    Returns:
        Mapping ``cluster_id -> target_count``.

    Raises:
        ValueError: If *percentage* is not in ``(0, 100]``.
    """
    if not (0.0 < percentage <= 100.0):
        msg = f"percentage must be in (0, 100], got {percentage}"
        raise ValueError(msg)

    mask = labels != -1
    valid_labels = labels[mask]
    unique, counts = np.unique(valid_labels, return_counts=True)

    budget: dict[int, int] = {}
    for cluster_id, count in zip(unique.tolist(), counts.tolist()):
        target = math.ceil(count * percentage / 100.0)
        budget[int(cluster_id)] = min(target, count)

    return budget


def _l2_normalize(embeddings: NDArray[np.floating]) -> NDArray[np.float64]:
    """L2-normalise embeddings row-wise.

    Args:
        embeddings: Array of shape ``(N, D)``.

    Returns:
        Normalised array of shape ``(N, D)`` with dtype ``float64``.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero — zero vectors stay zero.
    safe_norms = np.where(norms == 0, 1.0, norms)
    return (embeddings / safe_norms).astype(np.float64)


def rank_spectra_for_cluster(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    cluster_id: int,
    percentage: float = 100.0,
    core_fraction: float | None = None,
) -> NDArray[np.int64]:
    """Return global indices of spectra to annotate within a cluster.

    The ranking is based on cosine distance to the cluster centroid in
    L2-normalised embedding space.  The top portion (core) are the
    closest samples — representative archetypes.  The remainder
    (boundary) are the farthest samples — capturing diversity and edge
    cases.

    Args:
        embeddings: Array of shape ``(N, D)``.
        labels: Array of cluster labels of shape ``(N,)``.
        cluster_id: Target cluster identifier.
        percentage: Fraction of the cluster to select (0–100).
        core_fraction: Fraction of the selected quota devoted to core
            (centroid-near) samples.  If ``None``, defaults to ``0.8``
            when *percentage* < 10, otherwise ``0.6``.

    Returns:
        Global array indices into the original dataset, ordered
        core-first then boundary.

    Raises:
        ValueError: If *percentage* is not in ``(0, 100]``.
    """
    if not (0.0 < percentage <= 100.0):
        msg = f"percentage must be in (0, 100], got {percentage}"
        raise ValueError(msg)

    if core_fraction is None:
        core_fraction = 0.8 if percentage < 10.0 else 0.6

    member_mask = labels == cluster_id
    member_indices = np.flatnonzero(member_mask)
    n_members = len(member_indices)

    if n_members == 0:
        return np.array([], dtype=np.int64)

    target = min(n_members, math.ceil(n_members * percentage / 100.0))
    core_count = max(1, int(round(target * core_fraction)))
    boundary_count = target - core_count

    # Normalise embeddings and compute centroid.
    norm_emb = _l2_normalize(embeddings)
    cluster_emb = norm_emb[member_indices]
    centroid = cluster_emb.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)

    # Cosine distance = 1 - cosine_similarity.
    cos_sims = cluster_emb @ centroid_norm
    distances = 1.0 - cos_sims

    # Sort by distance ascending.
    sort_order = np.argsort(distances, kind="mergesort")
    sorted_members = member_indices[sort_order]

    core_indices = sorted_members[:core_count]

    # Boundary: farthest samples, excluding any already in core.
    if boundary_count <= 0:
        return core_indices

    # If target exceeds member count, just return all members.
    if core_count + boundary_count >= n_members:
        return sorted_members

    boundary_indices = sorted_members[-boundary_count:]
    return np.concatenate([core_indices, boundary_indices])


def build_annotation_order(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    spectrum_ids: NDArray,
    percentage: float,
    interleave: bool = True,
    core_fraction: float | None = None,
) -> list[str]:
    """Build the ordered list of spectrum IDs to annotate.

    Args:
        embeddings: Array of shape ``(N, D)``.
        labels: Array of cluster labels of shape ``(N,)``.
        spectrum_ids: Array of spectrum identifiers of shape ``(N,)``.
        percentage: Percentage of each cluster to annotate.
        interleave: If ``True``, round-robin interleave clusters so
            the expert sees diversity early.  If ``False``, annotate
            one cluster fully before moving to the next.
        core_fraction: Passed through to
            :func:`rank_spectra_for_cluster`.

    Returns:
        Ordered list of spectrum ID strings.

    Raises:
        ValueError: If array lengths do not match or *percentage* is
            invalid.
    """
    if not (embeddings.shape[0] == labels.shape[0] == spectrum_ids.shape[0]):
        msg = (
            f"Array length mismatch: embeddings {embeddings.shape[0]}, "
            f"labels {labels.shape[0]}, spectrum_ids {spectrum_ids.shape[0]}"
        )
        raise ValueError(msg)

    budget = compute_annotation_budget(labels, percentage)
    if not budget:
        logger.warning("No valid clusters found (all labels are -1).")
        return []

    cluster_orders: dict[int, NDArray[np.int64]] = {}
    for cluster_id in sorted(budget):
        indices = rank_spectra_for_cluster(
            embeddings,
            labels,
            cluster_id,
            percentage=percentage,
            core_fraction=core_fraction,
        )
        cluster_orders[cluster_id] = indices

    if not interleave:
        ordered_indices = np.concatenate(
            [cluster_orders[cid] for cid in sorted(cluster_orders)]
        )
        return [str(spectrum_ids[i]) for i in ordered_indices]

    # Round-robin interleaving.
    cluster_ids = sorted(cluster_orders)
    pointers: dict[int, int] = {cid: 0 for cid in cluster_ids}
    result: list[str] = []

    while True:
        made_progress = False
        for cid in cluster_ids:
            arr = cluster_orders[cid]
            ptr = pointers[cid]
            if ptr < len(arr):
                result.append(str(spectrum_ids[arr[ptr]]))
                pointers[cid] = ptr + 1
                made_progress = True
        if not made_progress:
            break

    return result
