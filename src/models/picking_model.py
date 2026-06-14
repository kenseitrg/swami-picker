"""Supervised picking models for Phase 4.

The models consume raw FK spectra ``(B, 1, 256, 256)`` and produce a
single output of shape ``(B, num_classes, W)`` where ``num_classes`` is
``spectrum_height + 1`` (one logit per wavenumber bin plus one
"no pick" class).  This forces the model to make an explicit decision
for every frequency column instead of delegating presence detection to
a separate head.

Several architectures are available:

* :class:`PickingModel` — compact U-Net with a 1-D conv classifier.
* :class:`SeqPickingModel` — U-Net followed by a BiLSTM or 1-D CNN over
  the frequency axis.
* :class:`MultiModePickingModel` — predicts logits for multiple
  dispersion modes (fundamental + overtones) and selects the most
  likely fundamental-mode path during inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two convolutional layers with ReLU activations and optional dropout."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            dropout: Dropout probability applied between the two convolutions.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convolutions with ReLU and optional dropout.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of shape ``(B, out_channels, H, W)``.
        """
        x = F.relu(self.conv1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.relu(self.conv2(x))
        return x


class _UNetBackbone(nn.Module):
    """Compact U-Net encoder/decoder used by all picking models."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        embed_dim: int,
        dropout: float = 0.0,
        num_downsample: int = 2,
    ) -> None:
        """Initialize the U-Net backbone.

        Args:
            in_channels: Number of input channels.
            base_channels: Width of the first encoder feature map.
            embed_dim: Bottleneck feature dimension.
            dropout: Dropout probability inside each ConvBlock.
            num_downsample: Number of pooling/upsampling stages.  Default 2.
        """
        super().__init__()
        if num_downsample not in (2, 3):
            msg = f"num_downsample must be 2 or 3, got {num_downsample}"
            raise ValueError(msg)

        self.num_downsample = num_downsample
        self.enc1 = ConvBlock(in_channels, base_channels, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)

        if num_downsample == 3:
            self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout)
            self.pool3 = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(base_channels * 4, embed_dim, dropout=dropout)
            self.up3 = nn.ConvTranspose2d(
                embed_dim, base_channels * 4, kernel_size=2, stride=2
            )
            self.dec3 = ConvBlock(base_channels * 8, base_channels * 4, dropout=dropout)
            self.enc2_to_bottleneck = nn.Sequential(
                ConvBlock(base_channels * 2, base_channels * 4, dropout=dropout),
                nn.MaxPool2d(2),
            )
        else:
            self.enc3 = None
            self.pool3 = None
            self.bottleneck = ConvBlock(base_channels * 2, embed_dim, dropout=dropout)
            self.up3 = None
            self.dec3 = None
            self.enc2_to_bottleneck = nn.Identity()

        self.up2 = nn.ConvTranspose2d(
            embed_dim if num_downsample == 2 else base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, dropout=dropout)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(base_channels * 2, base_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder + decoder.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Decoder output of shape ``(B, base_channels, H, W)``.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        if self.num_downsample == 3:
            assert self.enc3 is not None
            assert self.pool3 is not None
            assert self.up3 is not None
            assert self.dec3 is not None
            e3 = self.enc3(self.pool2(e2))
            btl = self.bottleneck(self.pool3(e3))
            d3 = self.up3(btl)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2_input = self.up2(d3)
        else:
            btl = self.bottleneck(self.pool2(e2))
            d2_input = self.up2(btl)

        d2 = self.dec2(torch.cat([d2_input, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return d1


class PickingModel(nn.Module):
    """Compact encoder-decoder for dispersion-curve picking.

    Outputs one logit per (wavenumber_bin, no_pick) class for each
    frequency column.  The absence class is the last index.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        embed_dim: int = 64,
        spectrum_height: int = 256,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the model.

        Args:
            in_channels: Number of input channels (1 for FK spectra).
            base_channels: Width of the first encoder feature map.
            embed_dim: Bottleneck feature dimension.
            spectrum_height: Number of wavenumber bins; determines the
                number of pick classes (``spectrum_height + 1``).
            dropout: Dropout probability inside each ConvBlock.
        """
        super().__init__()
        self.spectrum_height = spectrum_height
        self.num_classes = spectrum_height + 1

        self.backbone = _UNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            embed_dim=embed_dim,
            dropout=dropout,
            num_downsample=2,
        )

        self.feature_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.classifier = nn.Conv1d(
            base_channels * spectrum_height,
            self.num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass.

        Args:
            x: Input tensor of shape ``(B, 1, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes, W)`` where the last class
            is the "no pick" class.
        """
        b, _, h, w = x.shape
        if h != self.spectrum_height:
            msg = (
                f"Input height {h} does not match spectrum_height "
                f"{self.spectrum_height}"
            )
            raise ValueError(msg)

        d1 = self.backbone(x)
        feats = F.relu(self.feature_conv(d1))
        feats = feats.reshape(b, -1, w)
        logits = self.classifier(feats)
        return logits


class SeqPickingModel(nn.Module):
    """U-Net + frequency-axis sequence model.

    After the U-Net decoder, a per-column feature vector is extracted
    and processed by a 1-D sequence model along the frequency axis
    before classification.  This gives the model explicit access to
    predictions in neighboring columns.

    The sequence model can be either a stack of 1-D convolutions or a
    bidirectional LSTM.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        embed_dim: int = 64,
        spectrum_height: int = 256,
        dropout: float = 0.0,
        seq_hidden_dim: int = 128,
        seq_layers: int = 2,
        seq_type: str = "bilstm",
        num_downsample: int = 2,
    ) -> None:
        """Initialize the model.

        Args:
            in_channels: Number of input channels.
            base_channels: Width of the first encoder feature map.
            embed_dim: Bottleneck feature dimension.
            spectrum_height: Number of wavenumber bins.
            dropout: Dropout probability inside conv blocks and sequence
                model.
            seq_hidden_dim: Hidden dimension of the BiLSTM or 1-D CNN
                feature projection.
            seq_layers: Number of BiLSTM layers or 1-D CNN blocks.
            seq_type: ``"bilstm"`` or ``"conv1d"``.
            num_downsample: Number of U-Net downsample stages (2 or 3).
        """
        super().__init__()
        if seq_type not in {"bilstm", "conv1d"}:
            msg = f"seq_type must be 'bilstm' or 'conv1d', got '{seq_type}'"
            raise ValueError(msg)

        self.spectrum_height = spectrum_height
        self.num_classes = spectrum_height + 1
        self.seq_type = seq_type

        self.backbone = _UNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            embed_dim=embed_dim,
            dropout=dropout,
            num_downsample=num_downsample,
        )

        self.feature_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1)

        column_feature_dim = base_channels * spectrum_height

        if seq_type == "bilstm":
            self.seq_pre = nn.Linear(column_feature_dim, seq_hidden_dim)
            self.seq = nn.LSTM(
                input_size=seq_hidden_dim,
                hidden_size=seq_hidden_dim // 2,
                num_layers=seq_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if seq_layers > 1 else 0.0,
            )
            self.seq_post = nn.Linear(seq_hidden_dim, column_feature_dim)
        else:
            layers: list[nn.Module] = []
            in_dim = column_feature_dim
            for _ in range(seq_layers):
                layers.extend(
                    [
                        nn.Conv1d(
                            in_dim,
                            seq_hidden_dim,
                            kernel_size=3,
                            padding=1,
                        ),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = seq_hidden_dim
            layers.append(nn.Conv1d(in_dim, column_feature_dim, kernel_size=1))
            self.seq = nn.Sequential(*layers)
            self.seq_pre = nn.Identity()
            self.seq_post = nn.Identity()

        self.classifier = nn.Conv1d(
            column_feature_dim,
            self.num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass.

        Args:
            x: Input tensor of shape ``(B, 1, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes, W)``.
        """
        b, _, h, w = x.shape
        if h != self.spectrum_height:
            msg = (
                f"Input height {h} does not match spectrum_height "
                f"{self.spectrum_height}"
            )
            raise ValueError(msg)

        d1 = self.backbone(x)
        feats = F.relu(self.feature_conv(d1))  # (B, C, H, W)
        feats = feats.reshape(b, -1, w)  # (B, C*H, W)

        if self.seq_type == "bilstm":
            # LSTM expects (B, W, F).
            x_seq = self.seq_pre(feats.permute(0, 2, 1))
            x_seq, _ = self.seq(x_seq)
            x_seq = self.seq_post(x_seq).permute(0, 2, 1)
        else:
            x_seq = self.seq(feats)

        # Residual connection helps preserve spatial classification signal.
        logits = self.classifier(feats + x_seq)
        return logits


class MultiModePickingModel(nn.Module):
    """Multi-mode dispersion-curve picking model.

    The model predicts a stack of ``num_modes`` independent mode
    sequences.  During inference, the fundamental mode is selected as
    the smoothest valid path from the top mode candidates at each
    frequency column.  This decouples detection of multiple modes from
    the single-fundamental assumption.

    At training time, only the mode that best matches the ground-truth
    target receives gradients (best-alignment assignment).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        embed_dim: int = 64,
        spectrum_height: int = 256,
        dropout: float = 0.0,
        num_modes: int = 3,
        mode_hidden_dim: int = 128,
        num_downsample: int = 2,
    ) -> None:
        """Initialize the model.

        Args:
            in_channels: Number of input channels.
            base_channels: Width of the first encoder feature map.
            embed_dim: Bottleneck feature dimension.
            spectrum_height: Number of wavenumber bins.
            dropout: Dropout probability inside conv blocks.
            num_modes: Number of modes to predict (fundamental + overtones).
            mode_hidden_dim: Hidden dimension of the per-mode projection.
            num_downsample: Number of U-Net downsample stages (2 or 3).
        """
        super().__init__()
        self.spectrum_height = spectrum_height
        self.num_classes = spectrum_height + 1
        self.num_modes = num_modes

        self.backbone = _UNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            embed_dim=embed_dim,
            dropout=dropout,
            num_downsample=num_downsample,
        )

        self.feature_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1)

        column_feature_dim = base_channels * spectrum_height

        self.mode_proj = nn.Sequential(
            nn.Linear(column_feature_dim, mode_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mode_hidden_dim, mode_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mode_heads = nn.ModuleList(
            nn.Linear(mode_hidden_dim, self.num_classes) for _ in range(num_modes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass.

        Args:
            x: Input tensor of shape ``(B, 1, H, W)``.

        Returns:
            Logits of shape ``(B, num_modes, num_classes, W)``.
        """
        b, _, h, w = x.shape
        if h != self.spectrum_height:
            msg = (
                f"Input height {h} does not match spectrum_height "
                f"{self.spectrum_height}"
            )
            raise ValueError(msg)

        d1 = self.backbone(x)  # (B, C, H, W)
        feats = F.relu(self.feature_conv(d1))
        feats = feats.reshape(b, -1, w)  # (B, C*H, W)

        # Process each column independently through the mode projection.
        x_proj = self.mode_proj(feats.permute(0, 2, 1))  # (B, W, hidden)

        # Each mode head produces (B, W, num_classes).
        mode_logits = torch.stack(
            [head(x_proj) for head in self.mode_heads], dim=1
        )  # (B, num_modes, W, num_classes)

        # Rearrange to (B, num_modes, num_classes, W) for consistency.
        return mode_logits.permute(0, 1, 3, 2)


def build_picking_model(config) -> nn.Module:
    """Factory helper that builds a model from a ``PickingConfig``.

    Args:
        config: A ``PickingConfig`` instance.

    Returns:
        An uninitialized picking model.
    """
    model_type = getattr(config, "model_type", "picking")

    if model_type == "picking":
        return PickingModel(
            in_channels=1,
            base_channels=config.base_channels,
            embed_dim=config.embed_dim,
            spectrum_height=config.spectrum_height,
            dropout=config.dropout,
        )

    if model_type == "seq":
        return SeqPickingModel(
            in_channels=1,
            base_channels=config.base_channels,
            embed_dim=config.embed_dim,
            spectrum_height=config.spectrum_height,
            dropout=config.dropout,
            seq_hidden_dim=getattr(config, "seq_hidden_dim", 128),
            seq_layers=getattr(config, "seq_layers", 2),
            seq_type=getattr(config, "seq_type", "bilstm"),
            num_downsample=getattr(config, "num_downsample", 2),
        )

    if model_type == "multimode":
        return MultiModePickingModel(
            in_channels=1,
            base_channels=config.base_channels,
            embed_dim=config.embed_dim,
            spectrum_height=config.spectrum_height,
            dropout=config.dropout,
            num_modes=getattr(config, "num_modes", 3),
            mode_hidden_dim=getattr(config, "mode_hidden_dim", 128),
            num_downsample=getattr(config, "num_downsample", 2),
        )

    msg = f"Unknown model_type: {model_type}"
    raise ValueError(msg)


def inference_picks(
    logits: torch.Tensor,
    absent_class: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert model logits to pick indices and presence probabilities.

    Args:
        logits: Tensor of shape ``(B, num_classes, W)``.
        absent_class: Index of the "no pick" class.  Defaults to the
            last class.

    Returns:
        Tuple of ``(pick_indices, presence_probs)`` where ``pick_indices``
        has shape ``(B, W)`` (with ``-1`` for absent columns) and
        ``presence_probs`` has shape ``(B, W)``.
    """
    if absent_class is None:
        absent_class = logits.shape[1] - 1

    probs = F.softmax(logits, dim=1)
    pick_indices = logits.argmax(dim=1)  # (B, W)
    presence_probs = 1.0 - probs[:, absent_class, :]  # (B, W)

    pick_indices = pick_indices.where(
        pick_indices != absent_class,
        torch.tensor(-1, device=logits.device, dtype=pick_indices.dtype),
    )
    return pick_indices, presence_probs


def inference_picks_multimode(
    logits: torch.Tensor,
    absent_class: int | None = None,
    smoothness_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert multi-mode logits to fundamental-mode picks.

    For each spectrum, the model predicts ``num_modes`` candidate modes.
    This function selects the single smoothest valid path across all
    modes using a lightweight dynamic-programming formulation.

    Args:
        logits: Tensor of shape ``(B, num_modes, num_classes, W)``.
        absent_class: Index of the "no pick" class.  Defaults to last.
        smoothness_weight: Weight for the transition penalty in the DP
            path selection.

    Returns:
        Tuple of ``(pick_indices, presence_probs)`` with shapes
        ``(B, W)`` each.
    """
    if absent_class is None:
        absent_class = logits.shape[2] - 1

    batch_size, num_modes, num_classes, width = logits.shape
    device = logits.device

    probs = F.softmax(logits, dim=2)  # (B, M, C, W)
    presence_probs_all = 1.0 - probs[:, :, absent_class, :]  # (B, M, W)
    pick_probs_all = probs[:, :, :absent_class, :]  # (B, M, H, W)

    picks_out = torch.full(
        (batch_size, width),
        -1,
        dtype=torch.long,
        device=device,
    )
    presence_out = torch.zeros(batch_size, width, device=device)

    for b in range(batch_size):
        pick_probs_b = pick_probs_all[b]  # (M, H, W)
        presence_probs_b = presence_probs_all[b]  # (M, W)

        # Per-column top mode and top index.
        col_mode = presence_probs_b.argmax(dim=0)  # (W,)
        # pick_probs_b is (M, H, W).  For each column, pick the mode with
        # the highest presence probability and then the most likely index
        # within that mode.
        col_idx = torch.stack(
            [pick_probs_b[col_mode[w], :, w].argmax() for w in range(width)]
        )  # (W,)

        # DP over columns to pick the smoothest connected path.
        best_mode = int(col_mode[0].item())
        best_idx = int(col_idx[0].item())
        picks_out[b, 0] = best_idx
        presence_out[b, 0] = presence_probs_b[best_mode, 0]

        for t in range(1, width):
            prev_idx = int(picks_out[b, t - 1].item())
            best_cost = float("inf")
            best_m = best_mode

            for m in range(num_modes):
                # Local score: negative log-probability at this column.
                k_t = int(col_idx[t].item())
                local_score = -torch.log(pick_probs_b[m, k_t, t] + 1e-8).item()
                transition = (
                    smoothness_weight
                    * (k_t - prev_idx) ** 2
                    / max(num_classes - 1, 1) ** 2
                )
                cost = local_score + transition
                if cost < best_cost:
                    best_cost = cost
                    best_m = m

            picks_out[b, t] = int(col_idx[t].item())
            presence_out[b, t] = presence_probs_b[best_m, t]

    return picks_out, presence_out
