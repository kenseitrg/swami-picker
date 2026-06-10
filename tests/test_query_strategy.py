from __future__ import annotations


import numpy as np
import pytest

from src.active_learning.query import (
    build_annotation_order,
    compute_annotation_budget,
    rank_spectra_for_cluster,
)


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return synthetic (embeddings, labels, spectrum_ids)."""
    # Cluster 0: three vectors with clear distance ordering to centroid.
    # v0 closest to centroid, v1 middle, v2 farthest.
    cluster0 = np.array(
        [
            [0.9, 0.1, 0.0],   # closest after normalisation
            [1.0, 0.0, 0.0],   # middle
            [0.5, 0.5, 0.707],  # farthest (≈ [1,1,1] normalised)
        ],
        dtype=np.float32,
    )
    # Cluster 1: two vectors.
    cluster1 = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    # Noise point.
    noise = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)

    embeddings = np.concatenate([cluster0, cluster1, noise], axis=0)
    labels = np.array([0, 0, 0, 1, 1, -1], dtype=np.int64)
    spectrum_ids = np.array(["s0", "s1", "s2", "s3", "s4", "s5"])
    return embeddings, labels, spectrum_ids


class TestComputeAnnotationBudget:
    """Tests for :func:`compute_annotation_budget`."""

    def test_basic_budget(self) -> None:
        """Budget uses ceil and respects cluster sizes."""
        labels = np.array([0, 0, 0, 0, 1, 1, 1, -1], dtype=np.int64)
        budget = compute_annotation_budget(labels, 50.0)
        assert budget[0] == 2  # ceil(4 * 0.5)
        assert budget[1] == 2  # ceil(3 * 0.5)
        assert -1 not in budget

    def test_total_does_not_exceed_valid_count(self) -> None:
        """Sum of targets never exceeds number of non-noise samples."""
        labels = np.array([0, 0, 1, 1, -1], dtype=np.int64)
        budget = compute_annotation_budget(labels, 100.0)
        assert sum(budget.values()) == 4

    def test_noise_excluded(self) -> None:
        """Noise points (-1) do not appear in any cluster budget."""
        labels = np.array([-1, -1, 0, 0], dtype=np.int64)
        budget = compute_annotation_budget(labels, 100.0)
        assert budget == {0: 2}

    def test_percentage_bounds(self) -> None:
        """Invalid percentage raises ValueError."""
        labels = np.array([0, 0], dtype=np.int64)
        with pytest.raises(ValueError, match="percentage must be in"):
            compute_annotation_budget(labels, 0.0)
        with pytest.raises(ValueError, match="percentage must be in"):
            compute_annotation_budget(labels, 101.0)
        with pytest.raises(ValueError, match="percentage must be in"):
            compute_annotation_budget(labels, -5.0)

    def test_all_noise_returns_empty(self) -> None:
        """All-noise label array yields empty budget."""
        labels = np.array([-1, -1, -1], dtype=np.int64)
        budget = compute_annotation_budget(labels, 50.0)
        assert budget == {}


class TestRankSpectraForCluster:
    """Tests for :func:`rank_spectra_for_cluster`."""

    def test_core_samples_closer_than_boundary(self, synthetic_data) -> None:
        """Core zone mean distance to centroid is smaller than boundary."""
        embeddings, labels, _spectrum_ids = synthetic_data
        indices = rank_spectra_for_cluster(
            embeddings, labels, cluster_id=0, percentage=100.0, core_fraction=0.6
        )
        # With 3 members and 100%, all are selected. core=2, boundary=1.
        assert len(indices) == 3
        # Compute distances manually.
        from src.active_learning.query import _l2_normalize

        norm_emb = _l2_normalize(embeddings)
        members = norm_emb[labels == 0]
        centroid = members.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
        dists = 1.0 - members @ centroid_norm

        core_idx = indices[:2]
        bound_idx = indices[2:]
        core_dists = dists[np.isin(np.flatnonzero(labels == 0), core_idx)]
        bound_dists = dists[np.isin(np.flatnonzero(labels == 0), bound_idx)]
        assert core_dists.mean() < bound_dists.mean()

    def test_small_percentage_biases_core(self, synthetic_data) -> None:
        """At 5% coverage the default core fraction is 0.8."""
        embeddings, labels, _spectrum_ids = synthetic_data
        indices = rank_spectra_for_cluster(
            embeddings, labels, cluster_id=0, percentage=5.0
        )
        # 3 members, 5% → ceil(0.15) = 1 target.
        # core_fraction default = 0.8 → core_count = 1, boundary = 0.
        assert len(indices) == 1

    def test_large_percentage_default_core_fraction(self, synthetic_data) -> None:
        """At 15% coverage the default core fraction is 0.6."""
        embeddings, labels, _spectrum_ids = synthetic_data
        indices = rank_spectra_for_cluster(
            embeddings, labels, cluster_id=0, percentage=15.0
        )
        # 3 members, 15% → ceil(0.45) = 1 target.
        # core_fraction default = 0.6 → core_count = 1, boundary = 0.
        assert len(indices) == 1

    def test_cluster_target_capped_at_size(self, synthetic_data) -> None:
        """percentage=100 selects every member regardless of core/boundary split."""
        embeddings, labels, _spectrum_ids = synthetic_data
        indices = rank_spectra_for_cluster(
            embeddings, labels, cluster_id=0, percentage=100.0
        )
        assert len(indices) == 3  # cluster 0 has 3 members

    def test_empty_cluster_returns_empty(self, synthetic_data) -> None:
        """Requesting a non-existent cluster yields an empty array."""
        embeddings, labels, _spectrum_ids = synthetic_data
        indices = rank_spectra_for_cluster(
            embeddings, labels, cluster_id=99, percentage=100.0
        )
        assert len(indices) == 0
        assert indices.dtype == np.int64

    def test_noise_not_included(self, synthetic_data) -> None:
        """Noise points are never selected."""
        embeddings, labels, spectrum_ids = synthetic_data
        indices = rank_spectra_for_cluster(
            embeddings, labels, cluster_id=0, percentage=100.0
        )
        # The noise point is at global index 5.
        assert 5 not in indices
        # Verify spectrum_ids mapping.
        selected_ids = [spectrum_ids[i] for i in indices]
        assert "s5" not in selected_ids

    def test_percentage_bounds(self, synthetic_data) -> None:
        """Invalid percentage raises ValueError."""
        embeddings, labels, _spectrum_ids = synthetic_data
        with pytest.raises(ValueError, match="percentage must be in"):
            rank_spectra_for_cluster(embeddings, labels, 0, 0.0)
        with pytest.raises(ValueError, match="percentage must be in"):
            rank_spectra_for_cluster(embeddings, labels, 0, 101.0)

    def test_explicit_core_fraction_override(self, synthetic_data) -> None:
        """Explicit core_fraction overrides the default."""
        embeddings, labels, _spectrum_ids = synthetic_data
        # At 100% with core_fraction=1.0, all selected samples are from core.
        indices = rank_spectra_for_cluster(
            embeddings, labels, cluster_id=0, percentage=100.0, core_fraction=1.0
        )
        # 3 members, target=3, core=3, boundary=0.
        assert len(indices) == 3


class TestBuildAnnotationOrder:
    """Tests for :func:`build_annotation_order`."""

    def test_interleaving_preserves_cluster_balance(self, synthetic_data) -> None:
        """Round-robin visits each cluster in the first few positions."""
        embeddings, labels, spectrum_ids = synthetic_data
        order = build_annotation_order(
            embeddings, labels, spectrum_ids, percentage=100.0, interleave=True
        )
        # Non-noise IDs: s0,s1,s2 (cluster 0) and s3,s4 (cluster 1).
        assert len(order) == 5
        # First two should be from different clusters (one from each).
        # We know the ordering within each cluster, but interleaving means
        # the first element is cluster 0's first pick, second is cluster 1's.
        assert order[0] in {"s0", "s1", "s2"}
        assert order[1] in {"s3", "s4"}

    def test_no_interleave_concatenates_clusters(self, synthetic_data) -> None:
        """Without interleaving, clusters appear contiguously."""
        embeddings, labels, spectrum_ids = synthetic_data
        order = build_annotation_order(
            embeddings, labels, spectrum_ids, percentage=100.0, interleave=False
        )
        # Cluster 0 first (id=0), then cluster 1.
        cluster0_ids = {"s0", "s1", "s2"}
        cluster1_ids = {"s3", "s4"}
        # All cluster 0 items should appear before cluster 1 items.
        last_c0 = max(i for i, sid in enumerate(order) if sid in cluster0_ids)
        first_c1 = min(i for i, sid in enumerate(order) if sid in cluster1_ids)
        assert last_c0 < first_c1

    def test_disjointness(self, synthetic_data) -> None:
        """No spectrum appears twice in the annotation order."""
        embeddings, labels, spectrum_ids = synthetic_data
        order = build_annotation_order(
            embeddings, labels, spectrum_ids, percentage=100.0, interleave=True
        )
        assert len(order) == len(set(order))

    def test_full_coverage(self, synthetic_data) -> None:
        """percentage=100 includes all non-noise spectra."""
        embeddings, labels, spectrum_ids = synthetic_data
        order = build_annotation_order(
            embeddings, labels, spectrum_ids, percentage=100.0, interleave=True
        )
        non_noise = {"s0", "s1", "s2", "s3", "s4"}
        assert set(order) == non_noise

    def test_length_mismatch_raises(self, synthetic_data) -> None:
        """Mismatched array lengths raise ValueError."""
        embeddings, labels, spectrum_ids = synthetic_data
        with pytest.raises(ValueError, match="Array length mismatch"):
            build_annotation_order(
                embeddings[:-1], labels, spectrum_ids, percentage=50.0
            )

    def test_all_noise_returns_empty(self) -> None:
        """All-noise label array yields empty order."""
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        labels = np.array([-1], dtype=np.int64)
        spectrum_ids = np.array(["s0"])
        order = build_annotation_order(embeddings, labels, spectrum_ids, 100.0)
        assert order == []

    def test_order_respects_percentage(self, synthetic_data) -> None:
        """Partial percentage selects only the requested quota."""
        embeddings, labels, spectrum_ids = synthetic_data
        # Cluster 0 has 3 members; cluster 1 has 2.
        order = build_annotation_order(
            embeddings, labels, spectrum_ids, percentage=50.0, interleave=True
        )
        # ceil(3*0.5)=2 from cluster 0, ceil(2*0.5)=1 from cluster 1.
        assert len(order) == 3
        assert set(order).issubset({"s0", "s1", "s2", "s3", "s4"})
