# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Network architectures for EMG gesture classification.

Contains:
1. TDS building blocks (copied from emg2pose) — self-contained, no cross-project deps
2. Emg2PoseTdsGestureArchitecture — hybrid model using full TDS backbone + LSTM classifier
3. Original DiscreteGesturesArchitecture (from emg_nature) for baseline comparison
"""

import collections
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# TDS Building Blocks (from emg2pose/networks.py)
# =============================================================================

class Permute(nn.Module):
    """Permute the dimensions of the input tensor."""

    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        assert len(from_dims) == len(to_dims), \
            "Same number of from- and to- dimensions should be specified"
        self.from_dims = from_dims
        self.to_dims = to_dims
        self._permute_idx: list[int] = [from_dims.index(d) for d in to_dims]

    def get_inverse_permute(self) -> "Permute":
        return Permute(from_dims=self.to_dims, to_dims=self.from_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)


class BatchNorm1d(nn.Module):
    """Wrapper around nn.BatchNorm1d except in NTC format"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.permute_forward = Permute("NTC", "NCT")
        self.bn = nn.BatchNorm1d(*args, **kwargs)
        self.permute_back = Permute("NCT", "NTC")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.permute_back(self.bn(self.permute_forward(inputs)))


class Conv1dBlock(nn.Module):
    """A 1D convolution with padding so the input and output lengths match."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

        layers = {}
        layers["conv1d"] = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=0,
        )
        if norm_type == "batch":
            layers["norm"] = BatchNorm1d(out_channels)
        layers["relu"] = nn.ReLU(inplace=True)
        layers["dropout"] = nn.Dropout(dropout)

        self.conv = nn.Sequential(
            *[layers[key] for key in layers if layers[key] is not None]
        )
        if norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type == "layer":
            x = self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSConv2dBlock(nn.Module):
    """2D temporal convolution block (Hannun et al. 2019)."""

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        assert kernel_width % 2, "kernel_width must be odd."
        self.conv2d = nn.Conv2d(
            in_channels=channels, out_channels=channels,
            kernel_size=(1, kernel_width),
            dilation=(1, 1), stride=(1, 1), padding=(0, 0),
            groups=1, bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels * width)
        self.channels = channels
        self.width = width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, C, T = inputs.shape  # BCT
        x = inputs.reshape(B, self.channels, self.width, T)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(B, C, -1)  # BcwT -> BCT
        T_out = x.shape[-1]
        x = x + inputs[..., -T_out:]
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSFullyConnectedBlock(nn.Module):
    """Fully connected block (Hannun et al. 2019)."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x = x.swapaxes(-1, -2)  # BCT -> BTC
        x = self.fc_block(x)
        x = x.swapaxes(-1, -2)  # BTC -> BCT
        x += inputs
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSConvEncoder(nn.Module):
    """Time depth-separable convolutional encoder (Hannun et al. 2019)."""

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        self.kernel_width = kernel_width
        self.num_blocks = len(block_channels)
        assert len(block_channels) > 0
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            assert num_features % channels == 0
            tds_conv_blocks.extend([
                TDSConv2dBlock(channels, feature_width, kernel_width),
                TDSFullyConnectedBlock(num_features),
            ])
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)


class TdsStage(nn.Module):
    """Stage of TDS blocks preceded by a sub-sampling conv."""

    def __init__(
        self,
        in_channels: int = 16,
        in_conv_kernel_width: int = 5,
        in_conv_stride: int = 1,
        num_blocks: int = 1,
        channels: int = 8,
        feature_width: int = 2,
        kernel_width: int = 1,
        out_channels: int | None = None,
    ):
        super().__init__()
        layers: collections.OrderedDict[str, nn.Module] = collections.OrderedDict()
        C = channels * feature_width
        self.out_channels = out_channels

        if in_conv_kernel_width > 0:
            layers["conv1dblock"] = Conv1dBlock(
                in_channels, C, kernel_size=in_conv_kernel_width,
                stride=in_conv_stride,
            )
        elif in_channels != C:
            raise ValueError(
                f"in_channels ({in_channels}) must equal channels * feature_width"
                f" ({channels} * {feature_width}) if in_conv_kernel_width <= 0."
            )

        layers["tds_block"] = TDSConvEncoder(
            num_features=C,
            block_channels=[channels] * num_blocks,
            kernel_width=kernel_width,
        )

        if out_channels is not None:
            self.linear_layer = nn.Linear(channels * feature_width, out_channels)

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = self.layers(x)
        if self.out_channels is not None:
            x = self.linear_layer(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TdsNetwork(nn.Module):
    """Full TDS network (Hannun et al. 2019) for EMG encoding."""

    def __init__(
        self, conv_blocks: Sequence[Conv1dBlock], tds_stages: Sequence[TdsStage]
    ):
        super().__init__()
        self.layers = nn.Sequential(*conv_blocks, *tds_stages)
        self.left_context = self._get_left_context(conv_blocks, tds_stages)
        self.right_context = 0

    def forward(self, x):
        return self.layers(x)

    def _get_left_context(self, conv_blocks, tds_stages) -> int:
        left, stride = 0, 1
        for conv_block in conv_blocks:
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride
        for tds_stage in tds_stages:
            conv_block = tds_stage.layers.conv1dblock
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride
            tds_block = tds_stage.layers.tds_block
            for _ in range(tds_block.num_blocks):
                left += (tds_block.kernel_width - 1) * stride
        return left


def build_tds_network() -> TdsNetwork:
    """Build the standard TDS network as defined in emg2pose config/network/tds.yaml."""
    return TdsNetwork(
        conv_blocks=[
            Conv1dBlock(16, 256, kernel_size=11, stride=5),
            Conv1dBlock(256, 256, kernel_size=5, stride=2),
        ],
        tds_stages=[
            TdsStage(
                in_channels=256, in_conv_kernel_width=17, in_conv_stride=4,
                num_blocks=2, channels=16, feature_width=16, kernel_width=9,
            ),
            TdsStage(
                in_channels=256, in_conv_kernel_width=9, in_conv_stride=2,
                num_blocks=2, channels=16, feature_width=16, kernel_width=5,
                out_channels=64,
            ),
        ],
    )


# =============================================================================
# Transfer Learning Model
# =============================================================================

class Emg2PoseTdsGestureArchitecture(nn.Module):
    """
    Gesture classification using pre-trained emg2pose TDS backbone.

    Pipeline:
        raw EMG (B, 16, T) @2000Hz
          → TDS encoder → (B, 64, T/80) @25Hz
          → interpolate to 50Hz → (B, 64, T/40)
          → 1-layer LSTM(64→128) → Linear(128→9)
          → (B, 9, T_out) @50Hz

    Exposes left_context and stride so DiscreteGesturesModule._step()
    correctly slices targets via targets[:, :, left_context::stride].

    No Reinhard compression — raw EMG is fed directly to the TDS,
    consistent with how the TDS was pre-trained.
    """

    def __init__(
        self,
        encoder: TdsNetwork | None = None,
        lstm_hidden: int = 128,
        output_channels: int = 9,
    ):
        super().__init__()
        if encoder is None:
            encoder = build_tds_network()
        self.encoder = encoder

        # These are used by DiscreteGesturesModule._step() to slice targets
        # stride = 2000Hz / 50Hz = 40, left_context from TDS calculation
        self.left_context = encoder.left_context  # 1790
        self.stride = 40

        self.lstm = nn.LSTM(
            input_size=64,  # TDS output dim
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,    # no dropout for single-layer LSTM
        )
        self.classifier = nn.Linear(lstm_hidden, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16, T) raw EMG at 2000Hz
        Returns:
            (B, 9, T_out) gesture logits at 50Hz, aligned with
            targets[:, :, left_context::stride]
        """
        # 1. TDS encoding: (B, 16, T) → (B, 64, T_feat) @25Hz
        features = self.encoder(x)

        # 2. Compute target output length
        T = x.shape[-1]
        target_len = (T - self.left_context - 1) // self.stride + 1

        # 3. Interpolate: 25Hz → target_len @50Hz
        features = F.interpolate(
            features, size=target_len, mode='linear', align_corners=False
        )  # (B, 64, target_len)

        # 4. LSTM + classify
        x = features.permute(0, 2, 1)       # (B, target_len, 64)
        x, _ = self.lstm(x)                  # (B, target_len, 128)
        x = self.classifier(x)               # (B, target_len, 9)
        x = x.permute(0, 2, 1)               # (B, 9, target_len)
        return x

    def load_encoder_from_ckpt(self, ckpt_path: str, strict: bool = False):
        """
        Load TDS encoder weights from an emg2pose checkpoint.

        The emg2pose LightningModule stores the TDS network at:
            state_dict['model.network.*']
        We remap to:
            state_dict['encoder.*']
        """
        import logging
        logger = logging.getLogger(__name__)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        # emg2pose: model.network.layers.X.* → our: encoder.layers.X.*
        src_prefix = 'model.network.'
        dst_prefix = 'encoder.'

        encoder_sd = {}
        for k, v in state_dict.items():
            if k.startswith(src_prefix):
                new_k = dst_prefix + k[len(src_prefix):]
                encoder_sd[new_k] = v

        if not encoder_sd:
            # Maybe the checkpoint uses a different structure — try without prefix
            # Some checkpoints might store TDS directly under 'network.'
            src_prefix = 'network.'
            for k, v in state_dict.items():
                if k.startswith(src_prefix):
                    new_k = dst_prefix + k[len(src_prefix):]
                    encoder_sd[new_k] = v

        if not encoder_sd:
            logger.warning(
                f"No encoder weights found in {ckpt_path}. "
                f"Available keys (first 10): {list(state_dict.keys())[:10]}"
            )

        missing, unexpected = self.load_state_dict(encoder_sd, strict=strict)
        if missing:
            logger.warning(f"Missing encoder keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected encoder keys ({len(unexpected)}): {unexpected[:5]}...")
        logger.info(
            f"Loaded {len(encoder_sd) - len(missing)} encoder params "
            f"from {ckpt_path}"
        )


# =============================================================================
# Original emg_nature Baseline (for comparison)
# =============================================================================

class ReinhardCompression(nn.Module):
    """Reinhard dynamic range compression on raw EMG."""

    def __init__(self, range: float = 64.0, midpoint: float = 32.0):
        super().__init__()
        self.range = range
        self.midpoint = midpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.range * x / (self.midpoint + torch.abs(x))


class DiscreteGesturesArchitecture(nn.Module):
    """Original emg_nature gesture model: Conv1d + 3-layer LSTM."""

    def __init__(
        self,
        input_channels: int = 16,
        conv_output_channels: int = 512,
        kernel_width: int = 21,
        stride: int = 10,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 3,
        output_channels: int = 9,
    ) -> None:
        super().__init__()
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.left_context = kernel_width - 1
        self.stride = stride

        self.compression = ReinhardCompression(range=64.0, midpoint=32.0)
        self.conv_layer = nn.Conv1d(
            input_channels, conv_output_channels,
            kernel_size=kernel_width, stride=stride,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.post_conv_layer_norm = nn.LayerNorm(normalized_shape=conv_output_channels)
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.post_lstm_layer_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
        self.projection = nn.Linear(lstm_hidden_size, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.compression(inputs)
        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.post_conv_layer_norm(x)
        x, _ = self.lstm(x)
        x = self.post_lstm_layer_norm(x)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        return x
