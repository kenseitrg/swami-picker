from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.pseudo_label_classifier import MLPClassifier, ShallowCNNClassifier
from src.training.pseudo_label_trainer import PseudoLabelTrainer
from src.utils.config import PseudoLabelConfig
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


class FeatureDataset(Dataset):
    """PyTorch Dataset for feature-vector inputs with pseudo-labels.

    Args:
        features: Feature matrix of shape ``(N, D)``.
        labels: Integer labels of shape ``(N,)``.
        spectrum_ids: Array of spectrum identifier strings of shape ``(N,)``.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        spectrum_ids: np.ndarray,
    ) -> None:
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.spectrum_ids = spectrum_ids

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class SpectrumDataset(Dataset):
    """PyTorch Dataset for raw spectrum inputs with pseudo-labels.

    Loads ``.npz`` tensors on demand from the processed spectra directory.

    Args:
        manifest_path: Path to the dataset manifest JSON.
        labels_dict: Mapping from ``spectrum_id`` -> integer pseudo-label.
    """

    def __init__(self, manifest_path: Path, labels_dict: dict[str, int]) -> None:
        self.processed_dir = manifest_path.parent / "spectra"
        self.labels_dict = labels_dict
        self.spectrum_ids = sorted(labels_dict.keys())

    def __len__(self) -> int:
        return len(self.spectrum_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sid = self.spectrum_ids[idx]
        npz_path = self.processed_dir / f"{sid}.npz"
        data = np.load(npz_path)
        try:
            tensor = np.array(data["tensor"])
        finally:
            data.close()

        # Ensure shape (1, 256, 256)
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, ...]
        x = torch.from_numpy(tensor).float()
        y = torch.tensor(self.labels_dict[sid], dtype=torch.long)
        return x, y


def _load_pseudo_labels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pseudo-labels and spectrum IDs from a clustering output file.

    Args:
        path: Path to the ``.npz`` file produced by ``cluster_pseudo_labels.py``.

    Returns:
        Tuple of ``(labels, probabilities, spectrum_ids)``.
    """
    data = np.load(path)
    try:
        labels = np.array(data["labels"])
        probabilities = np.array(data["probabilities"])
        spectrum_ids = np.array(data["spectrum_ids"])
    finally:
        data.close()
    return labels, probabilities, spectrum_ids


def _load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and spectrum IDs.

    Args:
        path: Path to a ``.npz`` feature file.

    Returns:
        Tuple of ``(features, spectrum_ids)``.
    """
    data = np.load(path)
    try:
        features = np.array(data["features"])
        spectrum_ids = np.array(data["spectrum_ids"])
    finally:
        data.close()
    return features, spectrum_ids


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for pseudo-label classifier training."""
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Train a supervised classifier on HDBSCAN pseudo-labels."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase2_supervised.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment run name (defaults to auto-generated slug).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Override pseudo-labels path (defaults to config value).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 epoch on a tiny subset for smoke testing.",
    )
    args = parser.parse_args(argv)

    config = PseudoLabelConfig.from_yaml(Path(args.config))

    if args.dry_run:
        config.epochs = 1
        config.batch_size = min(config.batch_size, 4)
        dry_run_subset: int = 64

    set_seed(config.seed)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_name = args.name or f"{datetime.now().strftime('%Y-%m-%d')}_phase2c-supervised"
    run_dir = Path("experiments") / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(run_dir / "config.yaml")

    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)

    # Load pseudo-labels
    pseudo_path = Path(args.labels) if args.labels else Path(config.pseudo_labels_path)
    labels, probabilities, spectrum_ids = _load_pseudo_labels(pseudo_path)
    logger.info("Loaded %d pseudo-labels from %s", len(labels), pseudo_path)

    # Discard noise (-1 labels) for Stage-1 training
    core_mask = labels != -1
    n_noise = int(np.sum(~core_mask))
    logger.info(
        "Discarding %d noise points (%.1f%%) for Stage-1 training",
        n_noise,
        100 * n_noise / len(labels),
    )

    core_labels = labels[core_mask]
    core_spectrum_ids = spectrum_ids[core_mask]

    if len(core_labels) == 0:
        logger.error(
            "HDBSCAN produced zero core clusters (all -1). "
            "Cannot train classifier. Aborting."
        )
        return 1

    logger.info(
        "Stage-1 training set: %d spectra, %d clusters",
        len(core_labels),
        len(set(core_labels)),
    )

    # Build dataset
    if config.use_features:
        feature_path = Path(config.feature_path)
        all_features, all_spectrum_ids_feat = _load_features(feature_path)

        # Align features with pseudo-labels by spectrum_id
        id_to_row = {sid: i for i, sid in enumerate(all_spectrum_ids_feat)}
        aligned_features = []
        aligned_labels = []
        for sid, lbl in zip(core_spectrum_ids, core_labels):
            if sid in id_to_row:
                aligned_features.append(all_features[id_to_row[sid]])
                aligned_labels.append(lbl)
            else:
                logger.warning("Spectrum %s not found in feature file; skipping", sid)

        if not aligned_features:
            logger.error("No spectra could be aligned between labels and features.")
            return 1

        feat_matrix = np.stack(aligned_features, axis=0)
        label_array = np.array(aligned_labels, dtype=np.int64)

        # Dry-run: subset to tiny slice for quick smoke test
        if args.dry_run:
            dry_limit = min(dry_run_subset, len(feat_matrix))
            feat_matrix = feat_matrix[:dry_limit]
            label_array = label_array[:dry_limit]
            logger.info(
                "Dry-run mode: subset to %d samples", dry_limit
            )

        # Train/val split (10% stratified hold-out)
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            np.arange(len(label_array)),
            test_size=0.10,
            random_state=config.seed,
            stratify=label_array,
        )
        train_ds = FeatureDataset(
            feat_matrix[train_idx],
            label_array[train_idx],
            core_spectrum_ids[train_idx],
        )
        val_ds = FeatureDataset(
            feat_matrix[val_idx],
            label_array[val_idx],
            core_spectrum_ids[val_idx],
        )

        num_classes = int(np.max(label_array) + 1)
        model = MLPClassifier(
            input_dim=feat_matrix.shape[1],
            hidden_dims=config.mlp_hidden_dims,
            num_classes=num_classes,
            dropout=config.mlp_dropout,
        )
        logger.info(
            "MLP classifier: input_dim=%d, hidden=%s, num_classes=%d",
            feat_matrix.shape[1],
            config.mlp_hidden_dims,
            num_classes,
        )
    else:
        # CNN on raw spectra
        labels_dict = {
            sid: int(lbl) for sid, lbl in zip(core_spectrum_ids, core_labels)
        }
        manifest_path = Path(config.manifest_path)
        all_ids = list(labels_dict.keys())

        # Dry-run: subset to tiny slice for quick smoke test
        if args.dry_run:
            dry_limit = min(dry_run_subset, len(all_ids))
            all_ids = all_ids[:dry_limit]
            labels_dict = {sid: labels_dict[sid] for sid in all_ids}
            logger.info(
                "Dry-run mode: subset to %d samples", dry_limit
            )
        from sklearn.model_selection import train_test_split

        train_ids, val_ids = train_test_split(
            all_ids,
            test_size=0.10,
            random_state=config.seed,
            stratify=[labels_dict[sid] for sid in all_ids],
        )
        train_ds = SpectrumDataset(
            manifest_path, {sid: labels_dict[sid] for sid in train_ids}
        )
        val_ds = SpectrumDataset(
            manifest_path, {sid: labels_dict[sid] for sid in val_ids}
        )

        num_classes = int(np.max(core_labels) + 1)
        model = ShallowCNNClassifier(
            in_channels=1,
            num_classes=num_classes,
            dropout=config.cnn_dropout,
        )
        logger.info(
            "CNN classifier: num_classes=%d",
            num_classes,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    logger.info(
        "DataLoaders: train=%d, val=%d",
        len(train_ds),
        len(val_ds),
    )

    trainer = PseudoLabelTrainer(
        model=model,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        resume_from=Path(args.resume) if args.resume else None,
        argv=sys.argv,
    )
    trainer.train()

    logger.info("Training complete. Best val_acc=%.4f", trainer.best_val_acc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
