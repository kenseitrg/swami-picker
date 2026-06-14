from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.pseudo_label_classifier import MLPClassifier, ShallowCNNClassifier
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure root logger for CLI output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class FeatureDataset(Dataset):
    """Simple feature dataset for inference."""

    def __init__(self, features: np.ndarray) -> None:
        self.features = torch.from_numpy(features).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.features[index]


class SpectrumDataset(Dataset):
    """Raw spectrum dataset for inference."""

    def __init__(self, processed_dir: Path, spectrum_ids: list[str]) -> None:
        self.processed_dir = processed_dir
        self.spectrum_ids = spectrum_ids

    def __len__(self) -> int:
        return len(self.spectrum_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        sid = self.spectrum_ids[index]
        npz_path = self.processed_dir / f"{sid}.npz"
        data = np.load(npz_path)
        try:
            tensor = np.array(data["tensor"])
        finally:
            data.close()
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, ...]
        return torch.from_numpy(tensor).float()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for pseudo-label expansion.

    Loads a trained classifier checkpoint, predicts on all spectra (including
    those previously marked as noise by HDBSCAN), and accepts new pseudo-labels
    where the classifier confidence exceeds a threshold.
    """
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Expand pseudo-labels using a trained classifier."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained classifier checkpoint (.pt).",
    )
    parser.add_argument(
        "--pseudo-labels",
        type=str,
        default="data/processed/pseudo_labels.npz",
        help="Original HDBSCAN pseudo-labels file.",
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        default="data/processed/features/features_marginal.npz",
        help="Feature file (required for MLP classifier).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/manifest.json",
        help="Dataset manifest (required for CNN classifier).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.90,
        help="Minimum softmax confidence to accept a pseudo-label.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size after expansion; reject labels that would create smaller clusters.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/pseudo_labels_expanded.npz",
        help="Output path for expanded pseudo-labels.",
    )
    args = parser.parse_args(argv)

    device = get_device()
    checkpoint_path = Path(args.checkpoint)
    pseudo_labels_path = Path(args.pseudo_labels)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load original pseudo-labels
    data = np.load(pseudo_labels_path)
    try:
        original_labels = np.array(data["labels"])
        spectrum_ids = np.array(data["spectrum_ids"])
    finally:
        data.close()

    logger.info(
        "Loaded %d pseudo-labels from %s", len(original_labels), pseudo_labels_path
    )

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config_dict = checkpoint.get("config", {})
    use_features = config_dict.get("use_features", True)

    # Reconstruct model
    if use_features:
        feature_path = Path(args.feature_path)
        feat_data = np.load(feature_path)
        try:
            all_features = np.array(feat_data["features"])
            feat_spectrum_ids = np.array(feat_data["spectrum_ids"])
        finally:
            feat_data.close()

        id_to_row = {sid: i for i, sid in enumerate(feat_spectrum_ids)}
        aligned_features = []
        aligned_indices = []
        for i, sid in enumerate(spectrum_ids):
            if sid in id_to_row:
                aligned_features.append(all_features[id_to_row[sid]])
                aligned_indices.append(i)

        feat_matrix = np.stack(aligned_features, axis=0)
        input_dim = feat_matrix.shape[1]

        core_labels = original_labels[original_labels != -1]
        if len(core_labels) == 0:
            logger.error("HDBSCAN produced zero clusters (all noise). Cannot expand.")
            return 1
        num_classes = int(np.max(core_labels) + 1)

        mlp_hidden = config_dict.get("mlp_hidden_dims", [256, 128])
        mlp_dropout = config_dict.get("mlp_dropout", 0.2)
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=mlp_hidden,
            num_classes=num_classes,
            dropout=mlp_dropout,
        )
        infer_ds = FeatureDataset(feat_matrix)
    else:
        manifest_path = Path(args.manifest)
        processed_dir = manifest_path.parent / "spectra"

        core_labels = original_labels[original_labels != -1]
        if len(core_labels) == 0:
            logger.error("HDBSCAN produced zero clusters (all noise). Cannot expand.")
            return 1
        num_classes = int(np.max(core_labels) + 1)

        cnn_dropout = config_dict.get("cnn_dropout", 0.2)
        model = ShallowCNNClassifier(
            in_channels=1,
            num_classes=num_classes,
            dropout=cnn_dropout,
        )
        infer_ds = SpectrumDataset(processed_dir, spectrum_ids.tolist())

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # Run inference
    infer_loader = DataLoader(
        infer_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_confidences: list[float] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in infer_loader:
            x = batch.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            confidences_batch, predictions_batch = torch.max(probs, dim=1)
            all_confidences.extend(confidences_batch.cpu().numpy().tolist())
            all_predictions.extend(predictions_batch.cpu().numpy().tolist())

    # Scatter predictions back to full spectrum_ids length
    if use_features:
        aligned_confidences = np.array(all_confidences)
        aligned_predictions = np.array(all_predictions)
        confidences = np.zeros(len(spectrum_ids), dtype=np.float64)
        predictions = np.zeros(len(spectrum_ids), dtype=np.int64)
        for j, global_idx in enumerate(aligned_indices):
            confidences[global_idx] = aligned_confidences[j]
            predictions[global_idx] = aligned_predictions[j]
    else:
        confidences = np.array(all_confidences)
        predictions = np.array(all_predictions)

    # Expand labels: keep original core labels, try to recover noise
    expanded_labels = original_labels.copy()
    noise_mask = original_labels == -1

    # First pass: mark candidates
    candidate_mask = noise_mask & (confidences >= args.confidence_threshold)
    n_candidates = int(np.sum(candidate_mask))
    logger.info(
        "Found %d noise points with confidence >= %.2f",
        n_candidates,
        args.confidence_threshold,
    )

    # Second pass: reject labels that would create clusters smaller than min_cluster_size
    for lbl in sorted(set(predictions[candidate_mask])):
        lbl_mask = candidate_mask & (predictions == lbl)
        # Count how many already have this label in core
        core_count = int(np.sum(expanded_labels == lbl))
        new_count = int(np.sum(lbl_mask))
        if core_count + new_count < args.min_cluster_size:
            logger.warning(
                "Rejecting %d candidate labels for cluster %d: would create size %d < %d",
                new_count,
                lbl,
                core_count + new_count,
                args.min_cluster_size,
            )
            candidate_mask[lbl_mask] = False

    expanded_labels[candidate_mask] = predictions[candidate_mask]
    n_accepted = int(np.sum(candidate_mask))
    n_remaining_noise = int(np.sum(expanded_labels == -1))

    logger.info(
        "Accepted %d new pseudo-labels. Remaining noise: %d (%.1f%%)",
        n_accepted,
        n_remaining_noise,
        100 * n_remaining_noise / len(expanded_labels),
    )

    # Save expanded labels
    np.savez_compressed(
        output_path,
        labels=expanded_labels.astype(np.int64),
        original_labels=original_labels.astype(np.int64),
        predictions=predictions.astype(np.int64),
        confidences=confidences.astype(np.float64),
        spectrum_ids=spectrum_ids,
        n_accepted=n_accepted,
        n_remaining_noise=n_remaining_noise,
        confidence_threshold=args.confidence_threshold,
    )
    logger.info("Saved expanded pseudo-labels to %s", output_path)

    # JSON sidecar with stats
    sidecar = {
        "n_spectra": len(spectrum_ids),
        "n_original_noise": int(np.sum(original_labels == -1)),
        "n_accepted": n_accepted,
        "n_remaining_noise": n_remaining_noise,
        "confidence_threshold": args.confidence_threshold,
        "min_cluster_size": args.min_cluster_size,
        "cluster_counts": {
            str(lbl): int(np.sum(expanded_labels == lbl))
            for lbl in sorted(set(expanded_labels))
            if lbl != -1
        },
    }
    with open(output_path.with_suffix(".json"), "w") as fh:
        json.dump(sidecar, fh, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
