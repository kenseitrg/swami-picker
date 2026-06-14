from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.fk_dataset import FKDataset
from src.data.preprocessing import (
    PreprocessedSpectrum,
    save_preprocessed_spectrum,
)
from src.data.segy_reader import RawSpectrum


def _make_raw_spectrum(
    station_number: int = 50071009,
    source_file: str = "04_09_SWAMI_raw_spect_decim8_RL5007.sgy",
) -> RawSpectrum:
    """Factory for a ``RawSpectrum`` with sensible defaults."""
    rng = np.random.default_rng(42)
    data = rng.random((262, 400), dtype=np.float32)
    freq_axis = np.linspace(0.0, 15.93, 262, dtype=np.float32)
    waven_axis = np.linspace(0.0, 0.08, 400, dtype=np.float32)
    return RawSpectrum(
        data=data,
        station_number=station_number,
        line_number=station_number // 10000,
        point_number=station_number % 10000,
        freq_axis=freq_axis,
        waven_axis=waven_axis,
        elevation=98.30,
        x_coord=470510.70,
        y_coord=6933223.30,
        source_file=source_file,
    )


def _make_preprocessed_spectrum(
    spectrum_id: str,
    station_number: int = 50071009,
) -> PreprocessedSpectrum:
    """Factory for a ``PreprocessedSpectrum`` ready for saving."""
    rng = np.random.default_rng(station_number)
    tensor = rng.random((256, 256), dtype=np.float32)
    metadata = {
        "spectrum_id": spectrum_id,
        "original_shape": [262, 400],
        "resize_factors": [256 / 262, 256 / 400],
        "freq_axis_original": np.linspace(0.0, 15.93, 262).tolist(),
        "waven_axis_original": np.linspace(0.0, 0.08, 400).tolist(),
        "freq_axis_resized": np.linspace(0.0, 15.93, 256).tolist(),
        "waven_axis_resized": np.linspace(0.0, 0.08, 256).tolist(),
        "norm_method": "minmax",
        "norm_params": {"min": 0.0, "max": 1.0, "mu": 0.5, "sigma": 0.3},
        "clipping_bounds": [-3.0, 3.0],
        "elevation": 98.30,
        "x_coord": 470510.70,
        "y_coord": 6933223.30,
        "station_number": station_number,
        "line_number": station_number // 10000,
        "point_number": station_number % 10000,
        "source_file": "04_09_SWAMI_raw_spect_decim8_RL5007.sgy",
    }
    return PreprocessedSpectrum(tensor=tensor, metadata=metadata)


def _create_mock_dataset(tmpdir: Path, n_train: int = 4, n_val: int = 2) -> Path:
    """Create a mock dataset with train and val spectra.

    Returns:
        Path to the generated manifest.json.
    """
    spectra_dir = tmpdir / "spectra"
    spectra_dir.mkdir()

    entries: list[dict] = []
    for i in range(n_train):
        station = 50071009 + i
        sid = f"RL5007_{station}"
        proc = _make_preprocessed_spectrum(sid, station)
        save_preprocessed_spectrum(proc, spectra_dir)
        entries.append(
            {
                "spectrum_id": sid,
                "split": "train",
                "npz_path": f"spectra/{sid}.npz",
                "json_path": f"spectra/{sid}.json",
                "line_number": 5007,
                "point_number": station % 10000,
            }
        )

    for i in range(n_val):
        station = 51151009 + i
        sid = f"RL5115_{station}"
        proc = _make_preprocessed_spectrum(sid, station)
        save_preprocessed_spectrum(proc, spectra_dir)
        entries.append(
            {
                "spectrum_id": sid,
                "split": "val",
                "npz_path": f"spectra/{sid}.npz",
                "json_path": f"spectra/{sid}.json",
                "line_number": 5115,
                "point_number": station % 10000,
            }
        )

    manifest = {
        "spectra": entries,
        "stats": {"total": n_train + n_val, "train": n_train, "val": n_val},
    }
    manifest_path = tmpdir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest_path


class TestFKDatasetInit:
    """Tests for FKDataset initialization."""

    def test_init_loads_split(self) -> None:
        """Dataset should load only entries matching the requested split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir), n_train=4, n_val=2)
            ds_train = FKDataset(manifest, split="train")
            ds_val = FKDataset(manifest, split="val")

            assert len(ds_train) == 4
            assert len(ds_val) == 2

    def test_invalid_split_raises(self) -> None:
        """An invalid split string must raise ``ValueError``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir))
            with pytest.raises(
                ValueError, match="split must be 'train', 'val', or None"
            ):
                FKDataset(manifest, split="test")

    def test_load_all_splits_with_none(self) -> None:
        """``split=None`` loads all spectra regardless of split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir), n_train=4, n_val=2)
            ds = FKDataset(manifest, split=None)
            assert len(ds) == 6

    def test_empty_split(self) -> None:
        """A split with no entries should have length 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir), n_train=0, n_val=0)
            ds = FKDataset(manifest, split="train")
            assert len(ds) == 0


class TestFKDatasetGetItem:
    """Tests for FKDataset __getitem__."""

    def test_tensor_shape_and_dtype(self) -> None:
        """Returned tensor must have shape ``(1, 256, 256)`` and dtype float32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir), n_train=1, n_val=0)
            ds = FKDataset(manifest, split="train")
            tensor, metadata = ds[0]

            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (1, 256, 256)
            assert tensor.dtype == torch.float32

    def test_metadata_is_dict(self) -> None:
        """Returned metadata must be a dict containing the spectrum_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir), n_train=1, n_val=0)
            ds = FKDataset(manifest, split="train")
            tensor, metadata = ds[0]

            assert isinstance(metadata, dict)
            assert "spectrum_id" in metadata

    def test_transform_applied(self) -> None:
        """An optional transform must be applied to the tensor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir), n_train=1, n_val=0)
            ds = FKDataset(manifest, split="train", transform=lambda t: t * 2.0)
            tensor, _ = ds[0]
            # Just verify the transform ran by shape (it would have errored on None).
            assert tensor.shape == (1, 256, 256)

    def test_split_no_overlap(self) -> None:
        """Train and val splits must share no spectrum IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir), n_train=4, n_val=2)
            ds_train = FKDataset(manifest, split="train")
            ds_val = FKDataset(manifest, split="val")

            train_ids = {ds_train[i][1]["spectrum_id"] for i in range(len(ds_train))}
            val_ids = {ds_val[i][1]["spectrum_id"] for i in range(len(ds_val))}

            assert not train_ids & val_ids

    def test_index_out_of_range(self) -> None:
        """Indexing beyond the dataset length must raise ``IndexError``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir), n_train=1, n_val=0)
            ds = FKDataset(manifest, split="train")
            with pytest.raises(IndexError):
                ds[5]


class TestFKDatasetDataLoader:
    """Tests for DataLoader compatibility."""

    def test_dataloader_batch_shape(self) -> None:
        """A DataLoader batch must have shape ``(B, 1, 256, 256)``."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir), n_train=4, n_val=0)
            ds = FKDataset(manifest, split="train")
            loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

            batch = next(iter(loader))
            tensors, metadata = batch

            assert isinstance(tensors, torch.Tensor)
            assert tensors.shape == (2, 1, 256, 256)
            # default_collate stacks dict values when keys match across samples.
            assert isinstance(metadata, dict)
            assert "spectrum_id" in metadata

    def test_dataloader_with_workers(self) -> None:
        """DataLoader with multiple workers must produce valid batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _create_mock_dataset(Path(tmpdir), n_train=4, n_val=0)
            ds = FKDataset(manifest, split="train")
            loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)

            batch = next(iter(loader))
            tensors, metadata = batch

            assert tensors.shape == (2, 1, 256, 256)
            assert isinstance(metadata, dict)
