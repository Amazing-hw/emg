# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Method 1: Conformer Encoder for EMG Gesture Classification

"""
Network architectures for EMG gesture classification.

Contains:
1. Conformer building blocks (ConvolutionModule, ConformerBlock, etc.)
2. ConformerGestureArchitecture — Conformer encoder + LSTM/Linear classifier head
3. TDS building blocks + Emg2PoseTdsGestureArchitecture (for baseline comparison)
4. Original DiscreteGesturesArchitecture (from emg_nature) for baseline comparison
"""

import collections
import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# Conformer Building Blocks
# Reference: Gulati et al. "Conformer: Convolution-augmented Transformer
#            for Speech Recognition" (Interspeech 2020)
# =============================================================================

class Swish(nn.Module):
    """Swish activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=1)
        return out * torch.sigmoid(gate)


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module.
    Pointwise → GLU → DepthwiseConv1d → BatchNorm → Swish → Pointwise → Dropout
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, 2 * d_model, kernel_size=1, stride=1, padding=0,
        )
        self.glu = GLU(d_model)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=1, stride=1, padding=0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, D)
        return residual + x


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for self-attention in Conformer."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.1)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:length]


class MultiHeadedSelfAttention(nn.Module):
    """
    Multi-headed self-attention with relative positional encoding.
    Uses pre-norm (LayerNorm before attention).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.layer_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.layer_norm(x)

        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return residual + attn_output


class FeedForwardModule(nn.Module):
    """Conformer feed-forward module with Macaron-net half-step scaling."""

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        half_step: bool = False,
    ):
        super().__init__()
        self.half_step = half_step
        d_ff = d_model * expansion_factor
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.scale = 0.5 if half_step else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return residual + self.scale * x


class ConformerBlock(nn.Module):
    """
    A single Conformer block with Macaron-net structure:
    FFN (half) → MHSA → Conv → FFN (half) → LayerNorm
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(
            d_model, ff_expansion_factor, dropout, half_step=True,
        )
        self.mhsa = MultiHeadedSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(
            d_model, ff_expansion_factor, dropout, half_step=True,
        )
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.mhsa(x, mask)
        x = self.conv(x)
        x = self.ffn2(x)
        x = self.final_layer_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """
    Conformer encoder for EMG signal processing.

    Replaces the TDS backbone with:
    1. Conv1d subsampling (16 channels → d_model features, ~40x temporal reduction)
    2. N stacked ConformerBlocks
    3. Linear projection to output_dim

    Pipeline:
        raw EMG (B, 16, T) @2000Hz
          → Conv1d subsampling → (B, d_model, T//stride) @50Hz
          → ConformerBlocks → (B, T_feat, d_model)
          → Linear → (B, d_model, T_feat)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        output_dim: int = 64,
        subsampling_stride: int = 40,
    ):
        super().__init__()
        self.d_model = d_model
        self.subsampling_stride = subsampling_stride

        # Subsampling: two Conv1d layers to go from 2000Hz to ~50Hz
        # stride1=5, stride2=8 → total 40x subsampling
        stride1 = 5
        stride2 = subsampling_stride // stride1  # 40 // 5 = 8

        self.conv_subsample = nn.Sequential(
            nn.Conv1d(16, d_model, kernel_size=11, stride=stride1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=9, stride=stride2, padding=0),
            nn.ReLU(inplace=True),
        )

        self.linear_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                ff_expansion_factor=ff_expansion_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(d_model, output_dim)

        # Calculate left_context for target alignment
        # Conv1: kernel=11, stride=5 → left = (11-1)=10
        # Conv2: with cumulative stride=5, kernel=9 → left += (9-1)*5 = 40
        # Total: 50 samples at 2000Hz
        left_conv1 = (11 - 1) * 1    # = 10
        left_conv2 = (9 - 1) * 5     # = 40
        self.left_context = left_conv1 + left_conv2  # 50 samples @2000Hz
        self.stride = subsampling_stride  # 40

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16, T) raw EMG at 2000Hz
        Returns:
            (B, D_out, T_feat) features at ~50Hz
        """
        # Convolutional subsampling
        x = self.conv_subsample(x)  # (B, d_model, T_feat)
        x = x.transpose(1, 2)       # (B, T_feat, d_model)
        x = self.linear_proj(x)
        x = self.dropout(x)

        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)        # (B, T_feat, output_dim)
        x = x.transpose(1, 2)          # (B, output_dim, T_feat)
        return x


# =============================================================================
# Conformer Gesture Classification Model
# =============================================================================

class ConformerGestureArchitecture(nn.Module):
    """
    Gesture classification using Conformer encoder.

    Pipeline:
        raw EMG (B, 16, T) @2000Hz
          → ConformerEncoder → (B, 64, T_feat) @50Hz
          → 1-layer LSTM(64→128) → Linear(128→9)
          → (B, 9, T_out) @50Hz

    Attributes:
        left_context: int — samples to skip in target alignment
        stride: int — temporal stride for target alignment
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        output_dim: int = 64,
        lstm_hidden: int = 128,
        num_classes: int = 9,
        use_lstm_head: bool = True,
        subsampling_stride: int = 40,
    ):
        super().__init__()
        self.encoder = ConformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            conv_kernel_size=conv_kernel_size,
            ff_expansion_factor=ff_expansion_factor,
            dropout=dropout,
            output_dim=output_dim,
            subsampling_stride=subsampling_stride,
        )
        self.left_context = self.encoder.left_context
        self.stride = self.encoder.stride
        self.output_dim = output_dim
        self.use_lstm_head = use_lstm_head

        if use_lstm_head:
            self.lstm = nn.LSTM(
                input_size=output_dim,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            self.classifier = nn.Linear(lstm_hidden, num_classes)
        else:
            self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16, T) raw EMG at 2000Hz
        Returns:
            (B, num_classes, T_out) gesture logits at 50Hz
        """
        T = x.shape[-1]
        # For very long sequences, use chunked inference to avoid O(T^2) attention OOM
        # Training max is 16000 samples; chunk if test sequence > 32000
        max_len = 32000
        if T > max_len and not self.training:
            return self._chunked_forward(x, chunk_size=16000)

        features = self.encoder(x)  # (B, output_dim, T_feat)
        target_len = (T - self.left_context - 1) // self.stride + 1

        # Interpolate if needed
        if features.shape[-1] != target_len:
            features = F.interpolate(
                features, size=target_len, mode='linear', align_corners=False,
            )

        if self.use_lstm_head:
            x = features.permute(0, 2, 1)       # (B, target_len, output_dim)
            x, _ = self.lstm(x)                  # (B, target_len, lstm_hidden)
            x = self.classifier(x)               # (B, target_len, num_classes)
            x = x.permute(0, 2, 1)               # (B, num_classes, target_len)
        else:
            x = self.classifier(features.transpose(1, 2))  # (B, T, num_classes)
            x = x.transpose(1, 2)                           # (B, num_classes, T)
        return x

    @torch.no_grad()
    def _chunked_forward(self, x, chunk_size=16000):
        """
        Process long test sequences in non-overlapping chunks to avoid OOM.
        Each chunk is processed independently; outputs are concatenated.
        """
        B, C, T = x.shape
        total_target_len = (T - self.left_context - 1) // self.stride + 1
        feat_per_sample = self.stride  # 40

        # Pre-allocate output
        all_logits = torch.zeros(B, 9, total_target_len, device=x.device, dtype=torch.float32)

        # Process non-overlapping chunks
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk = x[:, :, chunk_start:chunk_end]
            chunk_T = chunk.shape[-1]

            if chunk_T < self.left_context + self.stride:
                continue

            # Forward
            features = self.encoder(chunk)
            chunk_target_len = (chunk_T - self.left_context - 1) // self.stride + 1
            if features.shape[-1] != chunk_target_len:
                features = F.interpolate(
                    features, size=chunk_target_len, mode='linear', align_corners=False,
                )

            if self.use_lstm_head:
                c = features.permute(0, 2, 1)
                c, _ = self.lstm(c)
                logits = self.classifier(c).permute(0, 2, 1)
            else:
                logits = self.classifier(features.transpose(1, 2)).transpose(1, 2)

            # Output position
            out_start = (chunk_start + self.left_context) // feat_per_sample
            out_end = out_start + logits.shape[-1]
            out_end = min(out_end, total_target_len)
            logits_slice = logits[:, :, :out_end - out_start]
            all_logits[:, :, out_start:out_end] = logits_slice

        return all_logits


# =============================================================================
# TDS Building Blocks (from emg2pose — kept for baseline comparison)
# =============================================================================

class Permute(nn.Module):
    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        self._permute_idx: list[int] = [from_dims.index(d) for d in to_dims]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)


class BatchNorm1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.permute_forward = Permute("NTC", "NCT")
        self.bn = nn.BatchNorm1d(*args, **kwargs)
        self.permute_back = Permute("NCT", "NTC")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.permute_back(self.bn(self.permute_forward(inputs)))


class Conv1dBlock(nn.Module):
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
    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
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
        B, C, T = inputs.shape
        x = inputs.reshape(B, self.channels, self.width, T)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(B, C, -1)
        T_out = x.shape[-1]
        x = x + inputs[..., -T_out:]
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSFullyConnectedBlock(nn.Module):
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
        x = x.swapaxes(-1, -2)
        x = self.fc_block(x)
        x = x.swapaxes(-1, -2)
        x += inputs
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSConvEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        self.kernel_width = kernel_width
        self.num_blocks = len(block_channels)
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            tds_conv_blocks.extend([
                TDSConv2dBlock(channels, feature_width, kernel_width),
                TDSFullyConnectedBlock(num_features),
            ])
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)


class TdsStage(nn.Module):
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
        layers_map: collections.OrderedDict[str, nn.Module] = collections.OrderedDict()
        C = channels * feature_width
        self.out_channels = out_channels
        if in_conv_kernel_width > 0:
            layers_map["conv1dblock"] = Conv1dBlock(
                in_channels, C, kernel_size=in_conv_kernel_width,
                stride=in_conv_stride,
            )
        layers_map["tds_block"] = TDSConvEncoder(
            num_features=C,
            block_channels=[channels] * num_blocks,
            kernel_width=kernel_width,
        )
        if out_channels is not None:
            self.linear_layer = nn.Linear(channels * feature_width, out_channels)
        self.layers = nn.Sequential(layers_map)

    def forward(self, x):
        x = self.layers(x)
        if self.out_channels is not None:
            x = self.linear_layer(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TdsNetwork(nn.Module):
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
            conv_block_item = tds_stage.layers.conv1dblock
            left += (conv_block_item.kernel_size - 1) * stride
            stride *= conv_block_item.stride
            tds_block = tds_stage.layers.tds_block
            for _ in range(tds_block.num_blocks):
                left += (tds_block.kernel_width - 1) * stride
        return left


def build_tds_network() -> TdsNetwork:
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


class Emg2PoseTdsGestureArchitecture(nn.Module):
    """Original TDS-based gesture classification (baseline)."""

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
        self.left_context = encoder.left_context
        self.stride = 40
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True, dropout=0.0,
        )
        self.classifier = nn.Linear(lstm_hidden, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        T = x.shape[-1]
        target_len = (T - self.left_context - 1) // self.stride + 1
        features = F.interpolate(
            features, size=target_len, mode='linear', align_corners=False,
        )
        x = features.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.classifier(x)
        x = x.permute(0, 2, 1)
        return x

    def load_encoder_from_ckpt(self, ckpt_path: str, strict: bool = False):
        import logging
        logger = logging.getLogger(__name__)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        src_prefix = 'model.network.'
        dst_prefix = 'encoder.'
        encoder_sd = {}
        for k, v in state_dict.items():
            if k.startswith(src_prefix):
                new_k = dst_prefix + k[len(src_prefix):]
                encoder_sd[new_k] = v
        if not encoder_sd:
            src_prefix = 'network.'
            for k, v in state_dict.items():
                if k.startswith(src_prefix):
                    new_k = dst_prefix + k[len(src_prefix):]
                    encoder_sd[new_k] = v
        missing, unexpected = self.load_state_dict(encoder_sd, strict=strict)
        if missing:
            logger.warning(f"Missing encoder keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected encoder keys ({len(unexpected)}): {unexpected[:5]}...")
        logger.info(f"Loaded {len(encoder_sd) - len(missing)} encoder params from {ckpt_path}")


# =============================================================================
# Original emg_nature Baseline
# =============================================================================

class ReinhardCompression(nn.Module):
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
