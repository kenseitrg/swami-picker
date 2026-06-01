from __future__ import annotations

import logging
import sys
from collections.abc import Sized
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mnist_dataset import create_mnist_dataloaders, verify_batch_shape
from src.utils.config import MNISTConfig
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Smoke-test the Phase-0 MNIST data pipeline."""
    config = MNISTConfig.from_yaml(Path("configs/phase0_mnist.yaml"))
    set_seed(config.seed)

    logger.info("Creating MNIST dataloaders …")
    train_loader, test_loader = create_mnist_dataloaders(config)

    logger.info("Train set size: %d", len(cast(Sized, train_loader.dataset)))
    logger.info("Test set size:  %d", len(cast(Sized, test_loader.dataset)))

    train_shape = verify_batch_shape(train_loader)
    test_shape = verify_batch_shape(test_loader)

    logger.info("Train batch shape: %s", train_shape)
    logger.info("Test batch shape:  %s", test_shape)

    expected = (
        config.batch_size,
        config.in_channels,
        config.image_size,
        config.image_size,
    )
    assert train_shape == expected, f"Train shape {train_shape} != expected {expected}"
    assert test_shape == expected, f"Test shape {test_shape} != expected {expected}"

    # Quick value-range sanity check (should be ~zero mean / unit variance)
    images, labels = next(iter(train_loader))
    logger.info(
        "Batch value range: [%.3f, %.3f]", images.min().item(), images.max().item()
    )
    logger.info("Label sample: %s", labels[:8].tolist())

    logger.info("✅ Data pipeline verification passed.")


if __name__ == "__main__":
    main()
