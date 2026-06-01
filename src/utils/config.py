from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MNISTConfig:
    """Configuration for the Phase 0 MNIST smoke test.

    All hyperparameters mirror the settings planned for Stage 1 FK
    pre-training so that this smoke test is a realistic validation of
    the full training stack.
    """

    # Data
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 1

    # Masking
    mask_ratio: float = 0.75
    use_block_masking: bool = True
    block_size: int = 2

    # Model
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    decoder_embed_dim: int = 256
    decoder_depth: int = 4
    decoder_num_heads: int = 8

    # Training
    batch_size: int = 8
    accum_steps: int = 4
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0

    # Reproducibility
    seed: int = 42

    # System
    num_workers: int = 0
    pin_memory: bool = True

    # Logging
    log_interval: int = 50

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_yaml(cls, path: Path) -> MNISTConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            A populated ``MNISTConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            TypeError: If the YAML contains unexpected keys.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        # YAML lists become Python lists; convert ``betas`` to a tuple.
        if "betas" in raw and isinstance(raw["betas"], list):
            raw["betas"] = tuple(raw["betas"])

        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw) - known
        if unknown:
            raise TypeError(f"Unexpected config keys: {unknown}")

        return cls(**raw)

    def save_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(
                self.to_dict(),
                fh,
                default_flow_style=False,
                sort_keys=False,
            )
