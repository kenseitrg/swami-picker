from __future__ import annotations

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Lightweight MLP classifier for pseudo-label training.

    Operates on pre-extracted feature vectors (e.g. PCA-reduced marginals or
    spectral descriptors) and outputs logits for K pseudo-label clusters.

    Args:
        input_dim: Dimensionality of the input feature vector.
        hidden_dims: List of hidden layer widths.  Default ``[256, 128]``.
        num_classes: Number of output clusters (K).
        dropout: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Input feature tensor of shape ``(B, input_dim)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        return self.net(x)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the penultimate-layer embedding for downstream clustering.

        Args:
            x: Input feature tensor of shape ``(B, input_dim)``.

        Returns:
            Embedding tensor of shape ``(B, hidden_dims[-1])``.
        """
        # Run all layers except the final Linear
        *blocks, _ = self.net
        return nn.Sequential(*blocks)(x)


class ShallowCNNClassifier(nn.Module):
    """Shallow CNN classifier operating directly on raw spectra.

    Uses a small stack of 2-D convolutions with global average pooling,
    followed by an MLP classifier head.  Suitable for learning features
    from the raw 256×256 spectra when engineered features are insufficient.

    Args:
        in_channels: Number of input channels (default 1 for greyscale spectra).
        num_classes: Number of output clusters (K).
        dropout: Dropout probability in the classifier head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (B, 128, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 128)
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits from a raw spectrum.

        Args:
            x: Input tensor of shape ``(B, in_channels, 256, 256)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        x = self.features(x)
        return self.classifier(x)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the penultimate-layer embedding.

        Args:
            x: Input tensor of shape ``(B, in_channels, 256, 256)``.

        Returns:
            Embedding tensor of shape ``(B, 256)``.
        """
        x = self.features(x)
        x = nn.Flatten()(x)  # (B, 128)
        # Run classifier up to the last Linear
        for layer in self.classifier:
            if (
                isinstance(layer, nn.Linear)
                and layer.out_features == self.classifier[-1].out_features
            ):
                break
            x = layer(x)
        return x
