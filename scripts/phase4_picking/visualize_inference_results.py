"""Visualize full-dataset Phase 4 inference results.

Produces publication-quality figures showing:
    * Distributions of per-spectrum quality metrics.
    * Scatter relationships between metrics (e.g. confidence vs. smoothness).
    * Best and worst example spectra by confidence, smoothness, and
      composite quality score.

Outputs are written to the run's plots directory so they can be inspected
alongside the training curves.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.fk_dataset import FKDataset
from src.evaluation.visualize_picking import (
    plot_inference_curve_grid,
    plot_quality_distributions,
    plot_quality_ranking,
    plot_quality_scatter,
)
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure root logger for CLI output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_predictions(predictions_path: Path) -> dict[str, Any]:
    """Load inference predictions from an ``.npz`` file.

    Args:
        predictions_path: Path to ``predictions.npz``.

    Returns:
        Dictionary with ``spectrum_ids``, ``picks``, ``presence_probs``,
        and ``metadatas``.
    """
    data = np.load(predictions_path, allow_pickle=True)
    spectrum_ids = data["spectrum_ids"].tolist()
    metadatas = json.loads(data["metadata"].item())
    return {
        "spectrum_ids": spectrum_ids,
        "picks": data["picks"],
        "presence_probs": data["presence_probs"],
        "metadatas": metadatas,
    }


def _load_quality_scores(quality_path: Path) -> list[dict[str, Any]]:
    """Load quality scores from JSON.

    Args:
        quality_path: Path to ``quality_scores.json``.

    Returns:
        List of quality-score records.
    """
    with open(quality_path) as fh:
        return json.load(fh)


def _load_spectra(manifest_path: Path, spectrum_ids: list[str]) -> np.ndarray:
    """Load preprocessed spectra from the manifest in the given order.

    Args:
        manifest_path: Path to ``data/processed/manifest.json``.
        spectrum_ids: Ordered list of spectrum identifiers.

    Returns:
        Array of spectra with shape ``(N, 1, 256, 256)``.
    """
    dataset = FKDataset(manifest_path=manifest_path, split=None)
    id_to_index = {
        entry["spectrum_id"]: idx for idx, entry in enumerate(dataset.entries)
    }

    spectra: list[np.ndarray] = []
    for spectrum_id in spectrum_ids:
        idx = id_to_index.get(spectrum_id)
        if idx is None:
            msg = f"Spectrum {spectrum_id} not found in manifest"
            raise ValueError(msg)
        tensor, _ = dataset[idx]
        spectra.append(tensor.numpy())

    return np.stack(spectra, axis=0).astype(np.float32)


def _rank_indices(
    quality_scores: list[dict[str, Any]],
    metric: str,
    descending: bool = True,
    n: int = 8,
) -> list[int]:
    """Return indices of the top/bottom spectra by a metric.

    Args:
        quality_scores: Quality-score records.
        metric: Metric to rank by.
        descending: If ``True``, return highest values first.
        n: Number of indices to return.

    Returns:
        List of integer indices into *quality_scores*.
    """
    indexed = list(enumerate(quality_scores))
    indexed.sort(key=lambda item: float(item[1][metric]), reverse=descending)
    return [idx for idx, _ in indexed[:n]]


def _compute_summary_stats(
    quality_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute summary statistics for quality metrics.

    Args:
        quality_scores: Quality-score records.

    Returns:
        Dictionary of summary statistics.
    """
    metrics = [
        "coverage",
        "mean_certainty",
        "effective_certainty",
        "uncertainty_penalty",
        "smoothness",
        "monotonicity",
        "composite_score",
    ]
    stats: dict[str, Any] = {"count": len(quality_scores)}
    for metric in metrics:
        values = np.array(
            [float(score[metric]) for score in quality_scores if metric in score],
            dtype=np.float64,
        )
        stats[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    """Generate inference-result visualizations."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Visualize full-dataset Phase 4 inference results."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="experiments/phase4-picking-v2-final/predictions.npz",
        help="Path to predictions.npz from run_inference.py.",
    )
    parser.add_argument(
        "--quality-scores",
        type=str,
        default=None,
        help="Path to quality_scores.json. Defaults to sibling of predictions.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/manifest.json",
        help="Path to data/processed/manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots. Defaults to predictions' plots/ directory.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Number of best/worst examples to show per metric.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for any non-deterministic sample selection.",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        logger.error("Predictions file not found: %s", predictions_path)
        return 1

    quality_path = (
        Path(args.quality_scores)
        if args.quality_scores
        else predictions_path.parent / "quality_scores.json"
    )
    if not quality_path.exists():
        logger.error("Quality scores file not found: %s", quality_path)
        return 1

    output_dir = (
        Path(args.output_dir) if args.output_dir else predictions_path.parent / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading predictions from %s", predictions_path)
    predictions = _load_predictions(predictions_path)
    spectrum_ids = predictions["spectrum_ids"]
    picks = predictions["picks"]
    presence_probs = predictions["presence_probs"]
    metadatas = predictions["metadatas"]

    logger.info("Loading quality scores from %s", quality_path)
    quality_scores = _load_quality_scores(quality_path)

    if len(quality_scores) != len(spectrum_ids):
        logger.error(
            "Mismatch: %d quality scores vs %d predictions",
            len(quality_scores),
            len(spectrum_ids),
        )
        return 1

    # Ensure quality scores are in the same order as the predictions.
    id_to_quality = {score["spectrum_id"]: score for score in quality_scores}
    ordered_scores = [id_to_quality[sid] for sid in spectrum_ids]

    logger.info("Loading spectra from %s", args.manifest)
    spectra = _load_spectra(Path(args.manifest), spectrum_ids)

    # Quality metric distributions.
    logger.info("Plotting quality metric distributions")
    plot_quality_distributions(
        ordered_scores,
        save_path=output_dir / "inference_quality_distributions.png",
    )

    # Scatter relationships.
    logger.info("Plotting confidence vs smoothness")
    plot_quality_scatter(
        ordered_scores,
        x_metric="mean_certainty",
        y_metric="smoothness",
        color_metric="composite_score",
        save_path=output_dir / "inference_confidence_vs_smoothness.png",
    )

    logger.info("Plotting coverage vs composite score")
    plot_quality_scatter(
        ordered_scores,
        x_metric="coverage",
        y_metric="composite_score",
        color_metric="mean_certainty",
        save_path=output_dir / "inference_coverage_vs_composite.png",
    )

    # Rankings.
    logger.info("Plotting composite score ranking")
    plot_quality_ranking(
        ordered_scores,
        metric="composite_score",
        top_n=args.num_examples,
        bottom_n=args.num_examples,
        save_path=output_dir / "inference_composite_ranking.png",
    )

    # Best/worst example grids.
    for metric, label in (
        ("mean_certainty", "Confidence"),
        ("smoothness", "Smoothness"),
        ("composite_score", "Composite Score"),
    ):
        best_indices = _rank_indices(
            ordered_scores, metric, descending=True, n=args.num_examples
        )
        worst_indices = _rank_indices(
            ordered_scores, metric, descending=False, n=args.num_examples
        )

        logger.info("Plotting best examples by %s", metric)
        plot_inference_curve_grid(
            spectra,
            picks,
            presence_probs,
            metadata=metadatas,
            indices=best_indices,
            title=f"Best {args.num_examples} by {label}",
            save_path=output_dir / f"inference_best_by_{metric}.png",
            seed=args.seed,
        )

        logger.info("Plotting worst examples by %s", metric)
        plot_inference_curve_grid(
            spectra,
            picks,
            presence_probs,
            metadata=metadatas,
            indices=worst_indices,
            title=f"Worst {args.num_examples} by {label}",
            save_path=output_dir / f"inference_worst_by_{metric}.png",
            seed=args.seed,
        )

    # Summary statistics.
    stats = _compute_summary_stats(ordered_scores)
    stats_path = output_dir / "inference_quality_summary.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Saved summary statistics to %s", stats_path)

    logger.info("Visualization complete. Outputs in %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
