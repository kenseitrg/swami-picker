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
class VICRegConfig:
    """Configuration for VICReg self-supervised pretraining on FK spectra."""

    # Data
    manifest_path: str = "data/processed/manifest.json"
    val_fraction: float = 0.10
    val_seed: int = 42

    # Model (encoder = same ViT-Small as Phase 0)
    img_size: int = 256
    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    projector_hidden_dim: int = 2048
    projector_out_dim: int = 2048

    # VICReg loss weights
    sim_weight: float = 25.0
    var_weight: float = 25.0
    cov_weight: float = 1.0

    # Augmentation (same aggressive pipeline as MAE v3)
    noise_std: float = 0.15
    intensity_jitter: float = 0.50
    freq_shift_max: float = 0.10
    waven_shift_max: float = 0.05
    freq_dropout_prob: float = 0.30
    freq_dropout_width: float = 0.08

    # Training
    batch_size: int = 16
    accum_steps: int = 1
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    seed: int = 42

    # System
    num_workers: int = 0
    pin_memory: bool = True

    # Logging
    log_interval: int = 50
    visualization_epochs: list[int] = field(default_factory=lambda: [10, 25, 50, 100])

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "VICRegConfig":
        """Load configuration from a YAML file."""
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
        """Save configuration to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(
                self.to_dict(),
                fh,
                default_flow_style=False,
                sort_keys=False,
            )


@dataclass
class PseudoLabelConfig:
    """Configuration for supervised pseudo-label classifier training (Option C).

    Supports both MLP (feature-input) and CNN (raw-spectrum-input) classifiers.
    """

    # Data
    manifest_path: str = "data/processed/manifest.json"
    pseudo_labels_path: str = "data/processed/pseudo_labels.npz"
    feature_path: str = "data/processed/features/features_marginal.npz"
    use_features: bool = True  # True = MLP on features, False = CNN on raw spectra

    # Model (MLP)
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    mlp_dropout: float = 0.2

    # Model (CNN)
    cnn_dropout: float = 0.2
    cnn_embed_dim: int = 128  # 128=GAP-direct, 256=original MLP head

    # Augmentation (applied only when use_features=False / CNN mode)
    augment_noise_std: float = 0.05
    augment_intensity_jitter: float = 0.15
    augment_freq_shift_max: float = 0.0
    augment_waven_shift_max: float = 0.0
    augment_freq_dropout_prob: float = 0.0
    augment_freq_dropout_width: float = 0.05

    # Class imbalance handling
    use_class_weights: bool = False  # True = balanced weights in CrossEntropyLoss

    # Training
    batch_size: int = 32
    accum_steps: int = 1
    epochs: int = 30
    lr: float = 1e-4
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_yaml(cls, path: Path) -> PseudoLabelConfig:
        """Load configuration from a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        if "betas" in raw and isinstance(raw["betas"], list):
            raw["betas"] = tuple(raw["betas"])
        if "mlp_hidden_dims" in raw and isinstance(raw["mlp_hidden_dims"], list):
            raw["mlp_hidden_dims"] = list(raw["mlp_hidden_dims"])

        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw) - known
        if unknown:
            raise TypeError(f"Unexpected config keys: {unknown}")

        return cls(**raw)

    def save_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(
                self.to_dict(),
                fh,
                default_flow_style=False,
                sort_keys=False,
            )


@dataclass
class PickingConfig:
    """Configuration for Phase 4: supervised dispersion-curve picking.

    The model consumes raw preprocessed FK spectra and outputs a dense
    ``(256,)`` wavenumber pick plus a presence mask. All parameters that
    affect training or inference are captured here for reproducibility.
    """

    # Data
    training_data_path: str = "data/processed/phase4_training_data.npz"
    val_fraction: float = 0.10
    val_seed: int = 42
    min_direct_picks: int = 3
    k_folds: int = 1  # 1 = simple train/val split; >1 = k-fold cross-validation
    fold_index: int = 0  # which fold to use as validation when k_folds > 1

    # Model
    model_type: str = "picking"  # "picking" | "seq" | "multimode"
    base_channels: int = 8
    embed_dim: int = 64
    spectrum_height: int = 256  # must match model input height (wavenumber bins)
    dropout: float = 0.3  # dropout inside conv blocks

    # Sequence model (used when model_type == "seq")
    seq_hidden_dim: int = 128
    seq_layers: int = 2
    seq_type: str = "bilstm"  # "bilstm" | "conv1d"

    # Multi-mode model (used when model_type == "multimode")
    num_modes: int = 3
    mode_hidden_dim: int = 128

    # U-Net depth: 2 or 3 downsample stages
    num_downsample: int = 2

    # Augmentation (pick-synchronized)
    aug_enabled: bool = True
    aug_noise_std: float = 0.05
    aug_intensity_jitter: float = 0.15
    aug_freq_shift_max: float = 0.05
    aug_waven_shift_max: float = 0.03

    # Training
    batch_size: int = 16
    accum_steps: int = 1
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    loss_pick_weight: float = 1.0
    direct_pick_weight: float = 2.0
    loss_smooth_weight: float = 0.0  # weight for frequency-axis smoothness loss
    loss_monotonic_weight: float = 0.0  # weight for soft monotonicity loss
    min_val_coverage: float = 0.05  # minimum predicted val coverage for best checkpoint
    early_stopping_patience: int = 15
    smooth_window: int = 5  # epochs for moving-average val metric smoothing
    seed: int = 42

    # System
    num_workers: int = 4
    pin_memory: bool = True
    log_interval: int = 10

    # Logging
    visualization_epochs: list[int] = field(
        default_factory=lambda: [10, 25, 50, 75, 100]
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PickingConfig:
        """Load configuration from a plain dictionary.

        Args:
            data: Dictionary of configuration values.

        Returns:
            A populated ``PickingConfig`` instance.

        Raises:
            TypeError: If the dictionary contains unexpected keys.
        """
        raw = dict(data)
        if "betas" in raw and isinstance(raw["betas"], list):
            raw["betas"] = tuple(raw["betas"])

        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw) - known
        if unknown:
            raise TypeError(f"Unexpected config keys: {unknown}")

        return cls(**raw)

    @classmethod
    def from_yaml(cls, path: Path) -> PickingConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            A populated ``PickingConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            TypeError: If the YAML contains unexpected keys.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        return cls.from_dict(raw)

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
