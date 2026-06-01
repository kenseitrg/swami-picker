from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

if TYPE_CHECKING:
    from src.utils.config import MNISTConfig


def _build_transform(image_size: int) -> transforms.Compose:
    """Build the preprocessing pipeline for MNIST.

    The pipeline converts PIL images to tensors, resizes to the target
    resolution using bilinear interpolation, and normalises to zero mean
    and unit variance using the standard MNIST dataset statistics.  This
    matches the per-spectrum z-score normalisation planned for the FK
    data pipeline.

    Args:
        image_size: Target spatial resolution (square).

    Returns:
        A ``torchvision.transforms.Compose`` object.
    """
    # Standard MNIST statistics (computed over the full training set).
    _MNIST_MEAN = 0.1307
    _MNIST_STD = 0.3081

    return transforms.Compose(
        [
            transforms.ToTensor(),  # (H, W) -> (1, H, W) in [0, 1]
            transforms.Resize(
                image_size,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.Normalize(mean=[_MNIST_MEAN], std=[_MNIST_STD]),
        ]
    )


def create_mnist_dataloaders(
    config: MNISTConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create training and test DataLoaders for MNIST.

    MNIST is downloaded automatically if not already present under
    ``data/raw/mnist/``.

    Args:
        config: Phase-0 configuration dataclass.

    Returns:
        A ``(train_loader, test_loader)`` tuple.
    """
    transform = _build_transform(config.image_size)
    data_root = "data/raw/mnist"

    train_set = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, test_loader


def verify_batch_shape(loader: DataLoader) -> tuple[int, ...]:
    """Extract and return the shape of the first batch from a DataLoader.

    Args:
        loader: PyTorch DataLoader.

    Returns:
        Shape tuple of the first image batch.
    """
    images, _ = next(iter(loader))
    return tuple(images.shape)
