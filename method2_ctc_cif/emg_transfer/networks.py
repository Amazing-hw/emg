# Method 2: CTC and CIF Sequence Modeling for EMG Gesture Classification

"""
Network architectures supporting CTC loss and CIF alignment.

Contains:
1. ConformerEncoder (shared backbone)
2. CtcGestureArchitecture — Conformer + CTC head for event sequence recognition
3. CifGestureArchitecture — Conformer + CIF head for automatic event boundary detection
4. TDS + original baselines for comparison
"""

import collections
import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# Conformer Building Blocks (shared with Method 1)
# =============================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=1)
        return out * torch.sigmoid(gate)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, stride=1, padding=0)
        self.glu = GLU(d_model)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
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
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1, half_step: bool = False):
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
    def __init__(self, d_model: int = 256, num_heads: int = 4, conv_kernel_size: int = 15,
                 ff_expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ff_expansion_factor, dropout, half_step=True)
        self.mhsa = MultiHeadedSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ff_expansion_factor, dropout, half_step=True)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ffn1(x)
        x = self.mhsa(x, mask)
        x = self.conv(x)
        x = self.ffn2(x)
        x = self.final_layer_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder shared by CTC and CIF models."""

    def __init__(self, d_model: int = 256, num_layers: int = 8, num_heads: int = 4,
                 conv_kernel_size: int = 15, ff_expansion_factor: int = 4,
                 dropout: float = 0.1, output_dim: int = 256, subsampling_stride: int = 40):
        super().__init__()
        self.d_model = d_model
        self.subsampling_stride = subsampling_stride
        stride1 = 5
        stride2 = subsampling_stride // stride1

        self.conv_subsample = nn.Sequential(
            nn.Conv1d(16, d_model, kernel_size=11, stride=stride1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=9, stride=stride2, padding=0),
            nn.ReLU(inplace=True),
        )
        self.linear_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model=d_model, num_heads=num_heads,
                           conv_kernel_size=conv_kernel_size,
                           ff_expansion_factor=ff_expansion_factor,
                           dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, output_dim)
        left_conv1 = (11 - 1) * 1
        left_conv2 = (9 - 1) * 5
        self.left_context = left_conv1 + left_conv2
        self.stride = subsampling_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_subsample(x)
        x = x.transpose(1, 2)
        x = self.linear_proj(x)
        x = self.dropout(x)
        for block in self.conformer_blocks:
            x = block(x)
        x = self.output_proj(x)
        return x  # (B, T_feat, output_dim)


# =============================================================================
# CTC Gesture Model
# =============================================================================

class CtcGestureArchitecture(nn.Module):
    """
    Gesture classification using CTC loss.

    Pipeline:
        raw EMG (B, 16, T) @2000Hz
          → ConformerEncoder → (B, T_feat, d_model) @50Hz
          → Linear(d_model → num_classes + 1) → (B, T_feat, 10)
            (9 gestures + 1 blank)

    Output: per-frame log-probabilities for CTC decoding

    Attributes:
        left_context: int — samples to skip
        stride: int — temporal stride
        blank_id: int — index of blank token (always 9)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        encoder_output_dim: int = 256,
        num_classes: int = 9,
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
            output_dim=encoder_output_dim,
            subsampling_stride=subsampling_stride,
        )
        self.left_context = self.encoder.left_context
        self.stride = self.encoder.stride
        self.num_classes = num_classes
        self.blank_id = num_classes  # blank is the last class

        # CTC head: project encoder output to num_classes + 1 (incl. blank)
        self.ctc_head = nn.Linear(encoder_output_dim, num_classes + 1)

        # For computing log-probabilities required by CTCLoss
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16, T) raw EMG at 2000Hz
        Returns:
            log_probs: (B, T_feat, num_classes+1) log-probabilities for CTC
            input_lengths: (B,) valid frame counts
        """
        features = self.encoder(x)  # (B, T_feat, encoder_output_dim)
        logits = self.ctc_head(features)  # (B, T_feat, num_classes+1)
        log_probs = self.log_softmax(logits)  # (B, T_feat, num_classes+1)
        return log_probs

    def get_input_lengths(self, x: torch.Tensor) -> torch.Tensor:
        """Compute valid output lengths based on input duration."""
        T = x.shape[-1]
        target_len = (T - self.left_context - 1) // self.stride + 1
        return torch.full((x.shape[0],), target_len, dtype=torch.long, device=x.device)


# =============================================================================
# CIF (Continuous Integrate-and-Fire) Gesture Model
# =============================================================================

class CIFHead(nn.Module):
    """
    Continuous Integrate-and-Fire module.

    Encoder每帧预测 α_t ∈ [0,1] 作为"权重"。
    当累积权重达到1时，fire一个事件token。

    Returns:
        event_tokens: (B, U, d_model) — fired event representations
        alpha: (B, T) — per-frame weights (for analysis/visualization)
        num_events: (B,) — number of fired events per sample
    """

    def __init__(self, d_model: int, num_classes: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.num_classes = num_classes
        self.alpha_proj = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, encoder_output: torch.Tensor) -> dict:
        """
        Args:
            encoder_output: (B, T, D)
        Returns:
            dict with 'logits', 'alpha', 'num_events', 'event_lengths'
        """
        B, T, D = encoder_output.shape

        # Per-frame weight
        alpha = torch.sigmoid(self.alpha_proj(encoder_output)).squeeze(-1)  # (B, T)

        # Integrate and fire
        event_tokens_list = []
        num_events_list = []
        alpha_list = []

        for b in range(B):
            tokens, boundaries, n_events = self._integrate_and_fire(
                encoder_output[b], alpha[b]
            )
            event_tokens_list.append(tokens)
            num_events_list.append(n_events)
            alpha_list.append(alpha[b])

        # Pad event tokens to same length
        max_events = max(num_events_list) if num_events_list else 1
        max_events = max(max_events, 1)

        padded_tokens = torch.zeros(
            B, max_events, D, device=encoder_output.device, dtype=encoder_output.dtype,
        )
        for b, tokens in enumerate(event_tokens_list):
            if tokens.shape[0] > 0:
                padded_tokens[b, :tokens.shape[0]] = tokens

        num_events = torch.tensor(num_events_list, device=encoder_output.device)
        alpha_out = torch.stack(alpha_list) if alpha_list else alpha

        # Classify each event
        logits = self.classifier(padded_tokens)  # (B, max_events, num_classes)

        return {
            "logits": logits,
            "alpha": alpha_out,
            "num_events": num_events,
        }

    def _integrate_and_fire(self, h: torch.Tensor, alpha: torch.Tensor):
        """
        Args:
            h: (T, D) encoder features
            alpha: (T,) per-frame weights
        Returns:
            tokens: (U, D) fired event representations
            boundaries: list of (start, end) frame indices
            num_events: int
        """
        T = alpha.shape[0]
        tokens = []
        boundaries = []
        accumulator = 0.0
        start_idx = 0
        weighted_sum = torch.zeros_like(h[0])

        for t in range(T):
            a = alpha[t].item()
            if accumulator + a >= 1.0:
                # Fire: take partial weight from current frame
                remaining = 1.0 - accumulator
                if remaining > self.eps:
                    weighted_sum = weighted_sum + h[t] * (remaining / a)
                # Normalize and store
                tokens.append(weighted_sum)
                boundaries.append((start_idx, t + 1))

                # Reset accumulator with excess from current frame
                accumulator = (accumulator + a) - 1.0
                if accumulator > self.eps:
                    weighted_sum = h[t] * (accumulator / a)
                else:
                    weighted_sum = torch.zeros_like(h[0])
                start_idx = t + 1
            else:
                accumulator += a
                weighted_sum = weighted_sum + h[t]

        # Handle remaining accumulator (partial last event)
        if accumulator > 0.3 and len(tokens) > 0:
            # Append to last event if substantial
            pass

        if len(tokens) == 0:
            tokens = torch.zeros(1, h.shape[-1], device=h.device, dtype=h.dtype)
            boundaries = [(0, T)]

        return torch.stack(tokens), boundaries, len(tokens)


class CifGestureArchitecture(nn.Module):
    """
    Gesture classification using CIF alignment.

    Pipeline:
        raw EMG (B, 16, T) @2000Hz
          → ConformerEncoder → (B, T_feat, d_model) @50Hz
          → CIFHead → event tokens → classifier → (B, U, num_classes)

    CIF automatically learns event boundaries, replacing manual pulse windows.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        encoder_output_dim: int = 256,
        num_classes: int = 9,
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
            output_dim=encoder_output_dim,
            subsampling_stride=subsampling_stride,
        )
        self.left_context = self.encoder.left_context
        self.stride = self.encoder.stride
        self.cif_head = CIFHead(encoder_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.encoder(x)  # (B, T_feat, D)
        return self.cif_head(features)


# =============================================================================
# TDS + Baselines (same as Method 1, for comparison)
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, norm_type: Literal["layer", "batch", "none"] = "layer",
                 dropout: float = 0.0):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        layers = {}
        layers["conv1d"] = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        if norm_type == "batch":
            layers["norm"] = BatchNorm1d(out_channels)
        layers["relu"] = nn.ReLU(inplace=True)
        layers["dropout"] = nn.Dropout(dropout)
        self.conv = nn.Sequential(*[layers[key] for key in layers if layers[key] is not None])
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
        self.conv2d = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=(1, kernel_width), dilation=(1, 1),
                                stride=(1, 1), padding=(0, 0), groups=1, bias=True)
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
        self.fc_block = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(inplace=True), nn.Linear(num_features, num_features))
        self.layer_norm = nn.LayerNorm(num_features)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.swapaxes(-1, -2)
        x = self.fc_block(x)
        x = x.swapaxes(-1, -2)
        x += inputs
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSConvEncoder(nn.Module):
    def __init__(self, num_features: int, block_channels: Sequence[int] = (24, 24, 24, 24), kernel_width: int = 32) -> None:
        super().__init__()
        self.kernel_width = kernel_width
        self.num_blocks = len(block_channels)
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            tds_conv_blocks.extend([TDSConv2dBlock(channels, feature_width, kernel_width), TDSFullyConnectedBlock(num_features)])
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)


class TdsStage(nn.Module):
    def __init__(self, in_channels: int = 16, in_conv_kernel_width: int = 5, in_conv_stride: int = 1,
                 num_blocks: int = 1, channels: int = 8, feature_width: int = 2,
                 kernel_width: int = 1, out_channels: int | None = None):
        super().__init__()
        layers_map: collections.OrderedDict[str, nn.Module] = collections.OrderedDict()
        C = channels * feature_width
        self.out_channels = out_channels
        if in_conv_kernel_width > 0:
            layers_map["conv1dblock"] = Conv1dBlock(in_channels, C, kernel_size=in_conv_kernel_width, stride=in_conv_stride)
        layers_map["tds_block"] = TDSConvEncoder(num_features=C, block_channels=[channels] * num_blocks, kernel_width=kernel_width)
        if out_channels is not None:
            self.linear_layer = nn.Linear(channels * feature_width, out_channels)
        self.layers = nn.Sequential(layers_map)
    def forward(self, x):
        x = self.layers(x)
        if self.out_channels is not None:
            x = self.linear_layer(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TdsNetwork(nn.Module):
    def __init__(self, conv_blocks: Sequence[Conv1dBlock], tds_stages: Sequence[TdsStage]):
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
        conv_blocks=[Conv1dBlock(16, 256, kernel_size=11, stride=5), Conv1dBlock(256, 256, kernel_size=5, stride=2)],
        tds_stages=[
            TdsStage(in_channels=256, in_conv_kernel_width=17, in_conv_stride=4, num_blocks=2, channels=16, feature_width=16, kernel_width=9),
            TdsStage(in_channels=256, in_conv_kernel_width=9, in_conv_stride=2, num_blocks=2, channels=16, feature_width=16, kernel_width=5, out_channels=64),
        ],
    )


class Emg2PoseTdsGestureArchitecture(nn.Module):
    def __init__(self, encoder: TdsNetwork | None = None, lstm_hidden: int = 128, output_channels: int = 9):
        super().__init__()
        if encoder is None:
            encoder = build_tds_network()
        self.encoder = encoder
        self.left_context = encoder.left_context
        self.stride = 40
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden, num_layers=1, batch_first=True, dropout=0.0)
        self.classifier = nn.Linear(lstm_hidden, output_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        T = x.shape[-1]
        target_len = (T - self.left_context - 1) // self.stride + 1
        features = F.interpolate(features, size=target_len, mode='linear', align_corners=False)
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
        for src_prefix in ['model.network.', 'network.']:
            dst_prefix = 'encoder.'
            encoder_sd = {dst_prefix + k[len(src_prefix):]: v for k, v in state_dict.items() if k.startswith(src_prefix)}
            if encoder_sd:
                break
        missing, unexpected = self.load_state_dict(encoder_sd, strict=strict)
        logger.info(f"Loaded {len(encoder_sd) - len(missing)} encoder params from {ckpt_path}")


class ReinhardCompression(nn.Module):
    def __init__(self, range: float = 64.0, midpoint: float = 32.0):
        super().__init__()
        self.range = range
        self.midpoint = midpoint
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.range * x / (self.midpoint + torch.abs(x))


class DiscreteGesturesArchitecture(nn.Module):
    def __init__(self, input_channels: int = 16, conv_output_channels: int = 512, kernel_width: int = 21,
                 stride: int = 10, lstm_hidden_size: int = 512, lstm_num_layers: int = 3, output_channels: int = 9) -> None:
        super().__init__()
        self.left_context = kernel_width - 1
        self.stride = stride
        self.compression = ReinhardCompression(range=64.0, midpoint=32.0)
        self.conv_layer = nn.Conv1d(input_channels, conv_output_channels, kernel_size=kernel_width, stride=stride)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.post_conv_layer_norm = nn.LayerNorm(normalized_shape=conv_output_channels)
        self.lstm = nn.LSTM(input_size=conv_output_channels, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True, dropout=0.1)
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
