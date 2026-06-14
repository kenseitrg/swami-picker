"""Run inference for the Phase 4 supervised picking model on the full dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.fk_dataset import FKDataset
from src.models.picking_model import build_picking_model, inference_picks
from src.transforms.coordinates import (
    compute_spectrum_quality_score,
    inference_to_annotation_record,
    model_indices_to_physical,
    needs_review_from_batch,
)
from src.utils.config import PickingConfig
from src.utils.device import get_device
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure root logger for CLI output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class _InferenceDataset(Dataset[tuple[torch.Tensor, dict[str, Any], str]]):
    """Wrapper around ``FKDataset`` returning (tensor, metadata, spectrum_id)."""

    def __init__(self, base: FKDataset) -> None:
        """Initialize the wrapper.

        Args:
            base: An ``FKDataset`` instance.
        """
        self.base = base

    def __len__(self) -> int:
        """Return the number of spectra."""
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any], str]:
        """Return tensor, metadata, and spectrum id.

        Args:
            index: Index into the dataset.

        Returns:
            Tuple of ``(tensor, metadata, spectrum_id)``.
        """
        tensor, metadata = self.base[index]
        return tensor, metadata, metadata["spectrum_id"]


def _collate_inference(
    batch: list[tuple[torch.Tensor, dict[str, Any], str]],
) -> tuple[torch.Tensor, list[dict[str, Any]], list[str]]:
    """Collate inference batch.

    Args:
        batch: List of tuples from ``_InferenceDataset.__getitem__``.

    Returns:
        Tuple of ``(stacked_tensors, metadata_list, spectrum_ids)``.
    """
    tensors, metadatas, spectrum_ids = zip(*batch)
    return torch.stack(tensors, dim=0), list(metadatas), list(spectrum_ids)


def _load_model_for_inference(
    checkpoint_path: Path,
    config: PickingConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained model from checkpoint in eval mode.

    Args:
        checkpoint_path: Path to the checkpoint ``.pt`` file.
        config: Model configuration.
        device: Device to load the model onto.

    Returns:
        Model in evaluation mode with loaded weights.
    """
    model = build_picking_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    logger.info(
        "Loaded model checkpoint from %s (epoch %d)",
        checkpoint_path,
        checkpoint.get("epoch", -1) + 1,
    )
    return model


def _build_dataset(manifest_path: Path) -> _InferenceDataset:
    """Build an inference dataset over all spectra in the manifest.

    Args:
        manifest_path: Path to ``data/processed/manifest.json``.

    Returns:
        Inference dataset wrapper.
    """
    base = FKDataset(
        manifest_path=manifest_path,
        split=None,  # Load all spectra regardless of train/val split.
    )
    return _InferenceDataset(base)


def _save_predictions(
    output_path: Path,
    spectrum_ids: Sequence[str],
    picks: np.ndarray,
    presence_probs: np.ndarray,
    metadatas: Sequence[dict[str, Any]],
) -> None:
    """Save inference outputs to an ``.npz`` file.

    Args:
        output_path: Destination ``.npz`` path.
        spectrum_ids: Spectrum identifiers of shape ``(N,)``.
        picks: Dense pick indices of shape ``(N, W)``.
        presence_probs: Presence probabilities of shape ``(N, W)``.
        metadatas: List of metadata dictionaries (saved as JSON string).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        spectrum_ids=np.array(spectrum_ids, dtype=object),
        picks=picks.astype(np.int16),
        presence_probs=presence_probs.astype(np.float32),
        metadata=np.array(json.dumps(metadatas), dtype=object),
    )
    logger.info("Saved predictions to %s", output_path)


def _save_quality_report(
    output_dir: Path,
    spectrum_ids: Sequence[str],
    picks: np.ndarray,
    presence_probs: np.ndarray,
    metadatas: Sequence[dict[str, Any]],
    quality_threshold: float | None,
    review_composite_percentile: float = 10.0,
    review_coverage_percentile: float = 5.0,
    review_smoothness_percentile: float = 10.0,
) -> tuple[Path, Path]:
    """Save quality scores and a list of low-quality spectra for re-annotation.

    Args:
        output_dir: Directory for output files.
        spectrum_ids: Spectrum identifiers of shape ``(N,)``.
        picks: Dense pick indices of shape ``(N, W)``.
        presence_probs: Presence probabilities of shape ``(N, W)``.
        metadatas: Metadata dictionaries for coordinate transforms.
        quality_threshold: Composite-score threshold below which a spectrum
            is flagged for manual review.

    Returns:
        Tuple of ``(quality_report_path, low_quality_list_path)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_scores: list[dict[str, Any]] = []
    low_quality_ids: list[str] = []

    for idx, spectrum_id in enumerate(spectrum_ids):
        metadata = metadatas[idx]
        physical = model_indices_to_physical(
            picks[idx], metadata, presence_probs=presence_probs[idx]
        )
        score = compute_spectrum_quality_score(
            picks[idx],
            presence_probs=presence_probs[idx],
            physical_picks=physical,
        )
        record: dict[str, Any] = {
            "spectrum_id": spectrum_id,
            **score,
        }
        quality_scores.append(record)

    # Use percentile-based triage; legacy threshold is ignored unless explicitly set.
    if quality_threshold is not None:
        low_quality_ids = [
            spectrum_ids[idx]
            for idx, score in enumerate(quality_scores)
            if score["composite_score"] < quality_threshold
        ]
    else:
        low_quality_ids, thresholds = needs_review_from_batch(
            [
                {"spectrum_id": sid, **score}
                for sid, score in zip(spectrum_ids, quality_scores)
            ],
            composite_percentile=review_composite_percentile,
            coverage_percentile=review_coverage_percentile,
            smoothness_percentile=review_smoothness_percentile,
        )
        for key, value in thresholds.items():
            logger.info("Review threshold %s: %s", key, value)

    report_path = output_dir / "quality_scores.json"
    with open(report_path, "w") as fh:
        json.dump(quality_scores, fh, indent=2)
    logger.info("Saved quality report to %s", report_path)

    low_quality_path = output_dir / "low_quality_spectra.json"
    with open(low_quality_path, "w") as fh:
        report: dict[str, Any] = {
            "count": len(low_quality_ids),
            "spectrum_ids": low_quality_ids,
        }
        if quality_threshold is not None:
            report["threshold"] = quality_threshold
        else:
            report["thresholds"] = thresholds
        json.dump(report, fh, indent=2)
    logger.info(
        "Flagged %d spectra for manual review",
        len(low_quality_ids),
    )
    return report_path, low_quality_path


def _export_annotation_records(
    output_dir: Path,
    spectrum_ids: Sequence[str],
    picks: np.ndarray,
    presence_probs: np.ndarray,
    confidence_threshold: float,
) -> Path:
    """Export high-confidence predictions as annotation records for review.

    Each record is written to ``annotations/<run_name>/spectra/`` and can be
    opened directly in the existing picking app.

    Args:
        output_dir: Directory for annotation files.
        spectrum_ids: Spectrum identifiers of shape ``(N,)``.
        picks: Dense pick indices of shape ``(N, W)``.
        presence_probs: Presence probabilities of shape ``(N, W)``.
        confidence_threshold: Minimum presence probability for a column to be
            marked as a direct pick.

    Returns:
        Path to the annotations directory.
    """
    from src.picking.annotation_io import save_annotation

    annotations_dir = output_dir / "annotations_for_review" / "spectra"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for idx, spectrum_id in enumerate(spectrum_ids):
        record = inference_to_annotation_record(
            spectrum_id=spectrum_id,
            picks=picks[idx],
            presence_probs=presence_probs[idx],
            confidence_threshold=confidence_threshold,
        )
        save_annotation(record, annotations_dir / f"{spectrum_id}.npz")

    logger.info(
        "Exported %d annotation records to %s",
        len(spectrum_ids),
        annotations_dir,
    )
    return annotations_dir


def main(argv: list[str] | None = None) -> int:
    """Run Phase 4 inference on the full FK dataset."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Run Phase 4 picking model inference on all spectra."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (best_model.pt).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/manifest.json",
        help="Path to data/processed/manifest.json.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config YAML. If omitted, uses config from checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz path. Defaults to experiments/<checkpoint-run>/predictions.npz.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help="Deprecated. Use --review-composite-percentile instead.",
    )
    parser.add_argument(
        "--review-composite-percentile",
        type=float,
        default=5.0,
        help="Flag spectra with composite score below this percentile (default: 5).",
    )
    parser.add_argument(
        "--review-coverage-percentile",
        type=float,
        default=5.0,
        help="Flag spectra with coverage below this percentile (default: 5).",
    )
    parser.add_argument(
        "--review-smoothness-percentile",
        type=float,
        default=5.0,
        help="Flag spectra with smoothness below this percentile (default: 5).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum presence probability to mark a pick as direct in exported annotations.",
    )
    parser.add_argument(
        "--export-annotations",
        action="store_true",
        help="Export annotation records (.npz) for review in the picking app.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return 1

    # Resolve config: prefer CLI override, fall back to checkpoint config.
    if args.config is not None:
        config = PickingConfig.from_yaml(Path(args.config))
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = PickingConfig.from_dict(checkpoint["config"])

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    # Resolve output path.
    if args.output is not None:
        output_path = Path(args.output)
    else:
        run_dir = checkpoint_path.parents[1]
        output_path = run_dir / "predictions.npz"

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Manifest:   %s", manifest_path)
    logger.info("Output:     %s", output_path)
    logger.info("Device:     %s", device)

    model = _load_model_for_inference(checkpoint_path, config, device)
    dataset = _build_dataset(manifest_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate_inference,
    )

    all_picks: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_ids: list[str] = []
    all_metadatas: list[dict[str, Any]] = []

    total = len(dataset)
    processed = 0
    start_time = datetime.now(timezone.utc)

    with torch.no_grad():
        for spectra, metadatas, spectrum_ids in loader:
            spectra = spectra.to(device, non_blocking=(device.type == "cuda"))

            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(spectra)
            else:
                logits = model(spectra)

            pred_picks, presence_probs = inference_picks(logits)

            all_picks.append(pred_picks.cpu().numpy())
            all_probs.append(presence_probs.cpu().numpy())
            all_ids.extend(spectrum_ids)
            all_metadatas.extend(metadatas)

            processed += len(spectrum_ids)
            if processed % 200 == 0 or processed == total:
                logger.info("Processed %d / %d spectra", processed, total)

    picks_arr = np.concatenate(all_picks, axis=0).astype(np.int16)
    probs_arr = np.concatenate(all_probs, axis=0).astype(np.float32)

    _save_predictions(
        output_path=output_path,
        spectrum_ids=all_ids,
        picks=picks_arr,
        presence_probs=probs_arr,
        metadatas=all_metadatas,
    )

    report_path, low_quality_path = _save_quality_report(
        output_dir=output_dir,
        spectrum_ids=all_ids,
        picks=picks_arr,
        presence_probs=probs_arr,
        metadatas=all_metadatas,
        quality_threshold=args.quality_threshold,
        review_composite_percentile=args.review_composite_percentile,
        review_coverage_percentile=args.review_coverage_percentile,
        review_smoothness_percentile=args.review_smoothness_percentile,
    )

    if args.export_annotations:
        _export_annotation_records(
            output_dir=output_dir,
            spectrum_ids=all_ids,
            picks=picks_arr,
            presence_probs=probs_arr,
            confidence_threshold=args.confidence_threshold,
        )

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info(
        "Inference complete: %d spectra in %.1f s (%.2f spectra/s)",
        total,
        elapsed,
        total / elapsed if elapsed > 0 else 0.0,
    )
    logger.info("Outputs:")
    logger.info("  Predictions:     %s", output_path)
    logger.info("  Quality report:  %s", report_path)
    logger.info("  Low-quality IDs: %s", low_quality_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
