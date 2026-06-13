"""PyTorch dataset for Phase 4 supervised dispersion-curve picking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FKPickingDataset(Dataset):
    """Dataset for supervised FK dispersion-curve picking.

    Loads annotated spectra from the ``.npz`` produced by
    ``export_annotations.py`` and returns tuples suitable for training a
    model that predicts a dense ``(256,)`` wavenumber pick plus a presence
    mask.
    """

    def __init__(
        self,
        npz_path: Path,
        split: str = "train",
        val_fraction: float = 0.10,
        val_seed: int = 42,
        min_direct_picks: int = 3,
        transform: Callable | None = None,
        cluster_embeddings: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Initialize the picking dataset.

        Args:
            npz_path: Path to the Phase 4 ``.npz`` file.
            split: ``"train"`` or ``"val"``.
            val_fraction: Fraction of data to hold out for validation.
            val_seed: Seed for the stratified split.
            min_direct_picks: Minimum number of direct picks required for a
                spectrum to be included.
            transform: Optional pick-synchronized augmentation callable.
            cluster_embeddings: Optional mapping from spectrum_id to a cluster
                embedding vector (used when ``use_cluster_conditioning`` is on).
        """
        self.npz_path = Path(npz_path)
        if split not in {"train", "val"}:
            msg = f"split must be 'train' or 'val', got '{split}'"
            raise ValueError(msg)
        self.split = split
        self.val_fraction = val_fraction
        self.val_seed = val_seed
        self.min_direct_picks = min_direct_picks
        self.transform = transform
        self.cluster_embeddings = cluster_embeddings

        (
            spectra,
            picks,
            direct_masks,
            confidences,
            cluster_labels,
            spectrum_ids,
            metadata,
        ) = self._load_npz()

        # Filter by minimum direct-pick count.
        direct_counts = direct_masks.sum(axis=1)
        keep = direct_counts >= min_direct_picks
        keep_indices = np.nonzero(keep)[0]

        if len(keep_indices) == 0:
            msg = f"No spectra have >= {min_direct_picks} direct picks"
            raise ValueError(msg)

        spectra = spectra[keep]
        picks = picks[keep]
        direct_masks = direct_masks[keep]
        confidences = confidences[keep]
        cluster_labels = cluster_labels[keep]
        spectrum_ids = spectrum_ids[keep]
        metadata = [metadata[i] for i in keep_indices.tolist()]

        # Stratified split by cluster label.
        indices = np.arange(len(spectrum_ids))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=val_seed,
            stratify=cluster_labels,
        )
        selected = train_idx if split == "train" else val_idx

        self.spectra = torch.from_numpy(spectra[selected]).float()
        self.picks = torch.from_numpy(picks[selected]).long()
        self.direct_masks = torch.from_numpy(direct_masks[selected]).bool()
        self.confidences = torch.from_numpy(confidences[selected]).float()
        self.cluster_labels = torch.from_numpy(cluster_labels[selected]).long()
        self.spectrum_ids = spectrum_ids[selected]
        self.metadata = [metadata[i] for i in selected.tolist()]

        logger.info(
            "FKPickingDataset initialized: path=%s split='%s' samples=%d",
            self.npz_path,
            split,
            len(self),
        )

    def _load_npz(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        """Load arrays and parse metadata from the ``.npz`` file.

        Returns:
            Tuple of ``(spectra, picks, direct_masks, confidences,
            cluster_labels, spectrum_ids, metadata_list)``.

        Raises:
            FileNotFoundError: If ``npz_path`` does not exist.
            KeyError: If a required array is missing.
            ValueError: If metadata cannot be parsed.
        """
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.npz_path}")

        with np.load(self.npz_path, allow_pickle=True) as data:
            spectra = data["spectra"]
            picks = data["picks"]
            direct_masks = data["direct_masks"]
            confidences = data["confidences"]
            cluster_labels = data["cluster_labels"]
            spectrum_ids = data["spectrum_ids"]

            raw_metadata = data["metadata"]
            # The metadata array may be stored as a single JSON string (0-d)
            # or as an object array of dicts. Handle both.
            if raw_metadata.ndim == 0:
                metadata = json.loads(raw_metadata.item())
            else:
                metadata = [raw_metadata.item(i) for i in range(raw_metadata.shape[0])]
                metadata = [
                    m if isinstance(m, dict) else json.loads(m) for m in metadata
                ]

        return (
            spectra,
            picks,
            direct_masks,
            confidences,
            cluster_labels,
            spectrum_ids,
            metadata,
        )

    def __len__(self) -> int:
        """Return the number of spectra in the split."""
        return len(self.spectra)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        str,
    ]:
        """Return a single training example.

        Args:
            index: Index into the split.

        Returns:
            Tuple of ``(spectrum, pick_target, presence_target, direct_mask,
            confidence, cluster_embedding, spectrum_id)``. ``pick_target``
            uses ``-1`` for unpicked columns; ``presence_target`` is ``1.0``
            where a pick exists and ``0.0`` otherwise.
        """
        spectrum = self.spectra[index]
        pick_target = self.picks[index].float()
        direct_mask = self.direct_masks[index]
        confidence = self.confidences[index]
        spectrum_id = str(self.spectrum_ids[index])

        presence_target = (pick_target >= 0).float()

        if self.transform is not None:
            spectrum, pick_target, presence_target, direct_mask, confidence = (
                self.transform(
                    spectrum, pick_target, presence_target, direct_mask, confidence
                )
            )

        cluster_embedding: torch.Tensor | None = None
        if self.cluster_embeddings is not None:
            vec = self.cluster_embeddings.get(spectrum_id)
            if vec is not None:
                cluster_embedding = torch.from_numpy(vec).float()

        return (
            spectrum,
            pick_target,
            presence_target,
            direct_mask,
            confidence,
            cluster_embedding,
            spectrum_id,
        )
