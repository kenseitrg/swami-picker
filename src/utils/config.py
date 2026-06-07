from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FKPipelineConfig:
    """Configuration for the Phase 1 FK spectrum preprocessing pipeline.

    All parameters that affect the preprocessed output are captured here
    so that the preprocessing step is fully reproducible.
    """

    # Directories
    raw_data_dir: str = "data"
    output_dir: str = "data/processed"

    # Preprocessing
    normalization: str = "minmax"  # "minmax" or "zscore"
    clip_bounds: tuple[float, float] = (-3.0, 3.0)
    output_size: tuple[int, int] = (256, 256)
    interpolation_mode: str = "bilinear"
    align_corners: bool = False

    # Splits
    val_lines: list[int] = field(default_factory=list)

    # Reproducibility
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_yaml(cls, path: Path) -> FKPipelineConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            A populated ``FKPipelineConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            TypeError: If the YAML contains unexpected keys.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        # YAML lists become Python lists; convert tuples back.
        for key in ("clip_bounds", "output_size"):
            if key in raw and isinstance(raw[key], list):
                raw[key] = tuple(raw[key])

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


@dataclass
class FKMAEConfig:
    """Configuration for Phase 2: MAE pretraining on FK spectra.

    Model hyperparameters mirror the Phase 0 ViT-MAE baseline so that
    the smoke-test validation transfers directly.
    """

    # Data
    manifest_path: str = "data/processed/manifest.json"
    val_fraction: float = 0.10
    val_seed: int = 42

    # Model
    img_size: int = 256
    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    decoder_embed_dim: int = 256
    decoder_depth: int = 4
    decoder_num_heads: int = 8
    mask_ratio: float = 0.75
    use_block_masking: bool = True
    block_size: int = 2

    # Augmentation
    noise_std: float = 0.01
    intensity_jitter: float = 0.15
    freq_shift_max: float = 0.0
    waven_shift_max: float = 0.0
    freq_dropout_prob: float = 0.0
    freq_dropout_width: float = 0.05

    # Training
    batch_size: int = 2
    accum_steps: int = 8
    epochs: int = 30
    lr: float = 5e-5
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    seed: int = 42

    # System
    num_workers: int = 4
    pin_memory: bool = True

    # Logging
    log_interval: int = 50
    visualization_epochs: list[int] = field(default_factory=lambda: [5, 10, 20, 30])

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_yaml(cls, path: Path) -> FKMAEConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            A populated ``FKMAEConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            TypeError: If the YAML contains unexpected keys.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

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
    use_cvt: bool = False
    cvt_kernel_size: int = 3
    use_pos_embed: bool = True

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
