"""Unit tests for Phase 4 visualization utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.visualize_picking import (
    plot_certainty_distributions,
    plot_curve_overlays,
    plot_error_distribution,
    plot_column_error_heatmap,
    plot_inference_curve_grid,
    plot_probability_heatmap_overlay,
    plot_quality_distributions,
    plot_quality_ranking,
    plot_quality_scatter,
    plot_training_curves,
    torch_softmax,
)


@pytest.fixture
def sample_data():
    """Create synthetic spectra, picks, and logits."""
    rng = np.random.default_rng(0)
    n, h, w = 8, 256, 256
    spectra = rng.standard_normal((n, 1, h, w)).astype(np.float32)
    true_picks = np.full((n, w), -1, dtype=np.int16)
    true_picks[:, 50:200] = rng.integers(50, 200, size=(n, 150))
    pred_picks = true_picks.copy()
    pred_picks[:, 50:200] += rng.integers(-10, 11, size=(n, 150))
    pred_picks = np.clip(pred_picks, 0, h - 1)

    logits = rng.standard_normal((n, h + 1, w)).astype(np.float32)
    presence_probs = rng.random((n, w)).astype(np.float32)
    presence_targets = (true_picks >= 0).astype(np.float32)
    return (
        spectra,
        true_picks,
        pred_picks,
        logits,
        presence_probs,
        presence_targets,
    )


def test_plot_curve_overlays_creates_file(sample_data, tmp_path: Path):
    """Curve overlay figure is written to disk."""
    spectra, true_picks, pred_picks, _, _, _ = sample_data
    save_path = tmp_path / "curves.png"
    plot_curve_overlays(spectra, true_picks, pred_picks, save_path=save_path, seed=1)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_training_curves_creates_file(tmp_path: Path):
    """Training-curve figure is written from a JSONL file."""
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "val_loss": 0.9,
            "train_rmse_pixels": 5.0,
            "val_rmse_pixels": 4.5,
            "train_presence_f1": 0.8,
            "val_presence_f1": 0.82,
            "lr": 1e-4,
            "max_vram_mb": 1024,
        },
        {
            "epoch": 2,
            "train_loss": 0.9,
            "val_loss": 0.8,
            "train_rmse_pixels": 4.5,
            "val_rmse_pixels": 4.0,
            "train_presence_f1": 0.82,
            "val_presence_f1": 0.85,
            "lr": 5e-5,
            "max_vram_mb": 1024,
        },
        {
            "epoch": 3,
            "train_loss": 0.8,
            "val_loss": 0.7,
            "train_rmse_pixels": 4.0,
            "val_rmse_pixels": 3.5,
            "train_presence_f1": 0.85,
            "val_presence_f1": 0.87,
            "lr": 1e-5,
            "max_vram_mb": 1024,
        },
    ]
    with open(metrics_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    save_path = tmp_path / "curves.png"
    plot_training_curves(metrics_path, save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_error_distribution_creates_file(tmp_path: Path):
    """Error distribution figure is written to disk."""
    save_path = tmp_path / "errors.png"
    plot_error_distribution(np.random.randn(50).astype(np.float32), save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_torch_softmax_sums_to_one():
    """Softmax helper produces a valid probability distribution."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    s = torch_softmax(x, axis=0)
    np.testing.assert_allclose(s.sum(axis=0), 1.0, atol=1e-6)
    assert np.all(s >= 0)


def test_plot_probability_heatmap_overlay_creates_file(sample_data, tmp_path: Path):
    """Probability heatmap overlay figure is written to disk."""
    spectra, _, _, logits, _, _ = sample_data
    save_path = tmp_path / "heatmap.png"
    plot_probability_heatmap_overlay(
        spectra,
        logits,
        save_path=save_path,
        seed=2,
    )
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_certainty_distributions_creates_file(sample_data, tmp_path: Path):
    """Certainty distribution figure is written to disk."""
    _, _, _, _, presence_probs, presence_targets = sample_data
    save_path = tmp_path / "certainty.png"
    plot_certainty_distributions(
        presence_probs, true_presence=presence_targets, save_path=save_path
    )
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_certainty_distributions_without_ground_truth(sample_data, tmp_path: Path):
    """Certainty distribution works without ground-truth presence."""
    _, _, _, _, presence_probs, _ = sample_data
    save_path = tmp_path / "certainty_no_gt.png"
    plot_certainty_distributions(presence_probs, save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_column_error_heatmap_creates_file(sample_data, tmp_path: Path):
    """Column error heatmap figure is written to disk."""
    spectra, true_picks, pred_picks, _, _, _ = sample_data
    save_path = tmp_path / "error_overlay.png"
    plot_column_error_heatmap(
        spectra, true_picks, pred_picks, save_path=save_path, seed=3
    )
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_inference_curve_grid_creates_file(sample_data, tmp_path: Path):
    """Inference curve grid figure is written to disk without ground truth."""
    spectra, _, pred_picks, _, presence_probs, _ = sample_data
    save_path = tmp_path / "inference_grid.png"
    metadata = [
        {"spectrum_id": f"spec_{i}", "freq_axis_resized": np.linspace(0, 10, 256)}
        for i in range(spectra.shape[0])
    ]
    for meta in metadata:
        meta["waven_axis_resized"] = np.linspace(0, 0.08, 256)
    plot_inference_curve_grid(
        spectra,
        pred_picks,
        presence_probs,
        metadata=metadata,
        num_samples=4,
        save_path=save_path,
        seed=4,
    )
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_quality_distributions_creates_file(tmp_path: Path):
    """Quality metric distribution figure is written to disk."""
    rng = np.random.default_rng(5)
    quality_scores = [
        {
            "spectrum_id": f"spec_{i}",
            "coverage": float(rng.random()),
            "mean_certainty": float(rng.random()),
            "smoothness": float(rng.random()),
            "monotonicity": float(rng.random()),
            "composite_score": float(rng.random()),
        }
        for i in range(50)
    ]
    save_path = tmp_path / "quality_distributions.png"
    plot_quality_distributions(quality_scores, save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_quality_scatter_creates_file(tmp_path: Path):
    """Quality scatter figure is written to disk."""
    rng = np.random.default_rng(6)
    quality_scores = [
        {
            "spectrum_id": f"spec_{i}",
            "mean_certainty": float(rng.random()),
            "smoothness": float(rng.random()),
            "composite_score": float(rng.random()),
        }
        for i in range(30)
    ]
    save_path = tmp_path / "quality_scatter.png"
    plot_quality_scatter(quality_scores, save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_quality_ranking_creates_file(tmp_path: Path):
    """Quality ranking bar chart is written to disk."""
    quality_scores = [
        {"spectrum_id": f"spec_{i}", "composite_score": float(i) / 20.0}
        for i in range(20)
    ]
    save_path = tmp_path / "quality_ranking.png"
    plot_quality_ranking(
        quality_scores,
        metric="composite_score",
        top_n=5,
        bottom_n=5,
        save_path=save_path,
    )
    assert save_path.exists()
    assert save_path.stat().st_size > 0
