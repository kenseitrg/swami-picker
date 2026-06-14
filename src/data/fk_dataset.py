from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from src.data.preprocessing import load_preprocessed_spectrum

logger = logging.getLogger(__name__)


class FKDataset(Dataset):
    """PyTorch Dataset for preprocessed FK spectra.

    Loads spectra on demand from a manifest produced by the preprocessing
    pipeline. Supports train/val splits and optional tensor transforms.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str | None = "train",
        transform: Callable | None = None,
        entries: list[dict[str, Any]] | None = None,
    ) -> None:
        """Load preprocessed FK spectra from manifest.

        Args:
            manifest_path: Path to manifest.json produced by preprocessing script.
            split: ``"train"``, ``"val"``, or ``None``.  If ``None``, all
                spectra in the manifest are loaded regardless of split.
            transform: Optional callable applied to the tensor.
            entries: Optional pre-filtered entry list. When provided,
                ``split`` is ignored and these entries are used directly.
                This allows programmatic train/val splits (e.g. the
                expanded validation set from :func:`create_train_val_entries`).
        """
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform

        if entries is not None:
            self.entries = entries
        else:
            if split is not None and split not in {"train", "val"}:
                msg = f"split must be 'train', 'val', or None, got '{split}'"
                raise ValueError(msg)

            with open(self.manifest_path) as fh:
                manifest: dict[str, Any] = json.load(fh)

            all_entries = manifest.get("spectra", [])
            if split is None:
                self.entries = list(all_entries)
            else:
                self.entries = [
                    entry for entry in all_entries if entry.get("split") == split
                ]

        # Resolve base directory from manifest path so npz/json paths are relative to it.
        self.base_dir = self.manifest_path.parent

        logger.info(
            "FKDataset initialized: split='%s', entries=%d, manifest=%s",
            split,
            len(self.entries),
            self.manifest_path,
        )

    def __len__(self) -> int:
        """Return the number of spectra in the split."""
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        """Return a single spectrum as a tensor and its metadata.

        Args:
            index: Index into the filtered split entries.

        Returns:
            A tuple of ``(tensor, metadata)`` where ``tensor`` has shape
            ``(1, 256, 256)`` and dtype ``float32``.
        """
        entry = self.entries[index]
        spectrum_id = entry["spectrum_id"]

        # Resolve directory containing the .npz/.json files relative to manifest dir.
        spectrum_dir = self.base_dir / Path(entry["npz_path"]).parent

        spectrum = load_preprocessed_spectrum(spectrum_id, spectrum_dir)

        tensor = torch.from_numpy(spectrum.tensor).unsqueeze(0).float()

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, spectrum.metadata
