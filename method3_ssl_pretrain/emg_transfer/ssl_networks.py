# Method 3: Self-Supervised Pre-training Networks
#
# Implements wav2vec 2.0 and HuBERT style architectures for EMG.

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# Conformer Blocks (same as Method 1/2)
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
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = GLU(d_model)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                         padding=(kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x).transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x).transpose(1, 2)
        return residual + x


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(self.d_head)
        self.layer_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.layer_norm(x)
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return residual + self.dropout(self.out_proj(out))


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1, half_step=False):
        super().__init__()
        d_ff = d_model * expansion_factor
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.scale = 0.5 if half_step else 1.0
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        return residual + self.scale * self.dropout2(self.fc2(self.dropout1(self.swish(self.fc1(x)))))


class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, conv_kernel_size=15, ff_expansion_factor=4, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ff_expansion_factor, dropout, half_step=True)
        self.mhsa = MultiHeadedSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ff_expansion_factor, dropout, half_step=True)
        self.final_layer_norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        x = self.ffn1(x)
        x = self.mhsa(x, mask)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.final_layer_norm(x)


# =============================================================================
# Feature Encoder: Raw EMG → latent features
# =============================================================================

class EmgFeatureEncoder(nn.Module):
    """
    Lightweight CNN encoder that converts raw EMG to latent features.

    Raw EMG (B, 16, T) @2000Hz → z (B, T_feat, d_encoder) @50Hz

    Uses grouped convolutions to preserve channel structure.
    """

    def __init__(self, in_channels=16, out_dim=512, subsample_factor=40):
        super().__init__()
        self.subsample_factor = subsample_factor
        # Multi-layer conv subsampling (BCT format → use LayerNorm on transposed)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, 256, kernel_size=10, stride=5, padding=0),
            nn.Conv1d(256, 512, kernel_size=8, stride=4, padding=0),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=0),
        ])
        self.conv_norms = nn.ModuleList([
            nn.LayerNorm(256),
            nn.LayerNorm(512),
            nn.LayerNorm(512),
        ])
        self.gelu = nn.GELU()
        # Final projection after flattening channels
        self.proj = nn.Linear(512, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)

        # Compute left context for downstream alignment
        # Conv1: (10-1)*1 = 9
        # Conv2: (8-1)*5 = 35
        # Conv3: (4-1)*20 = 60
        # Total: 104 samples @2000Hz
        self.left_context = (10-1)*1 + (8-1)*5 + (4-1)*20  # = 104

    def forward(self, x):
        # x: (B, 16, T)
        for conv, norm in zip(self.conv_layers, self.conv_norms):
            x = conv(x)                             # (B, C, T)
            x = norm(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm expects (B, T, C)
            x = self.gelu(x)                        # (B, C, T)
        x = x.transpose(1, 2)  # (B, T_feat, 512)
        x = self.proj(x)
        x = self.layer_norm(x)  # (B, T_feat, out_dim) — BTC format, LayerNorm works
        x = self.dropout(x)
        return x  # (B, T_feat, out_dim)


# =============================================================================
# Wav2Vec 2.0 Model
# =============================================================================

class GumbelVectorQuantizer(nn.Module):
    """
    Gumbel softmax quantization for wav2vec 2.0.

    Two groups of codebooks, each with V codewords.
    Uses straight-through estimator during training.
    """

    def __init__(self, input_dim=512, num_groups=2, num_vars=320,
                 temperature=1.0, temperature_decay=0.999995):
        super().__init__()
        self.num_groups = num_groups
        self.num_vars = num_vars
        self.temperature = temperature
        self.temperature_decay = temperature_decay

        # One linear projection per group + one weight matrix per group
        self.projections = nn.ModuleList([
            nn.Linear(input_dim, num_vars) for _ in range(num_groups)
        ])
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(1, num_vars, input_dim // num_groups))
            for _ in range(num_groups)
        ])
        nn.init.xavier_uniform_(self.codebooks[0])
        nn.init.xavier_uniform_(self.codebooks[1])

    def forward(self, z):
        """
        Args:
            z: (B, T, D) latent features
        Returns:
            q: (B, T, D) quantized features
            perplexity: scalar
        """
        B, T, D = z.shape
        d_g = D // self.num_groups  # 256

        quantized_parts = []
        perplexities = []

        for g in range(self.num_groups):
            # Project to codebook space
            logits = self.projections[g](z)  # (B, T, num_vars)

            # Gumbel softmax
            if self.training:
                # Straight-through: use hard assignment in forward, soft in backward
                hard = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
                soft = F.softmax(logits / self.temperature, dim=-1)
                probs = (hard - soft).detach() + soft
            else:
                # At test time, take argmax
                idx = logits.argmax(dim=-1)
                probs = F.one_hot(idx, self.num_vars).float()

            # Weighted sum of codebook entries
            q_g = torch.matmul(probs, self.codebooks[g].squeeze(0))  # (B, T, d_g)
            quantized_parts.append(q_g)

            # Perplexity
            avg_probs = soft.mean(dim=[0, 1]) if self.training else probs.mean(dim=[0, 1])
            perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())
            perplexities.append(perplexity)

        q = torch.cat(quantized_parts, dim=-1)  # (B, T, D)

        # Decay temperature
        if self.training:
            self.temperature = max(self.temperature * self.temperature_decay, 0.5)

        return q, sum(perplexities) / len(perplexities)


class Wav2Vec2Model(nn.Module):
    """
    wav2vec 2.0 style self-supervised model for EMG.

    Architecture:
        Raw EMG → FeatureEncoder → z (latent)
                                   ↓ Quantizer → q (target for contrastive loss)
                                   ↓ Masking
        masked z → Conformer ContextEncoder → c (context representations)

    Loss: contrastive loss between c_t and q_t for masked time steps.
    """

    def __init__(
        self,
        feature_dim=512,
        context_dim=256,
        num_conformer_layers=8,
        num_heads=4,
        conv_kernel_size=15,
        dropout=0.1,
        num_quantizer_groups=2,
        num_quantizer_vars=320,
        mask_prob=0.5,
        mask_length=5,  # consecutive frames to mask (5 * 20ms = 100ms at 50Hz)
    ):
        super().__init__()
        self.feature_encoder = EmgFeatureEncoder(
            in_channels=16, out_dim=feature_dim, subsample_factor=40,
        )
        self.left_context = self.feature_encoder.left_context  # 104

        self.quantizer = GumbelVectorQuantizer(
            input_dim=feature_dim, num_groups=num_quantizer_groups,
            num_vars=num_quantizer_vars,
        )

        # Context encoder: Conformer on top of masked latent features
        self.context_proj = nn.Linear(feature_dim, context_dim)
        self.context_dropout = nn.Dropout(dropout)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model=context_dim, num_heads=num_heads,
                           conv_kernel_size=conv_kernel_size, dropout=dropout)
            for _ in range(num_conformer_layers)
        ])
        # Final projection: context → quantized space (for contrastive loss)
        self.final_proj = nn.Linear(context_dim, feature_dim)

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.num_negatives = 100

    def apply_mask(self, z):
        """
        Mask random spans of the latent feature sequence.
        Args:
            z: (B, T, D)
        Returns:
            masked_z: (B, T, D) with masked positions zeroed
            mask: (B, T) boolean mask (True = masked)
        """
        B, T, D = z.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=z.device)

        # Select num_masked time steps as mask starts
        num_masked = int(T * self.mask_prob / self.mask_length)
        num_masked = max(num_masked, 1)

        for b in range(B):
            starts = torch.randperm(T)[:num_masked]
            for s in starts:
                end = min(s + self.mask_length, T)
                mask[b, s:end] = True

        # Apply mask (zero out masked positions)
        masked_z = z.clone()
        masked_z[mask] = 0.0

        return masked_z, mask

    def sample_negatives(self, z, mask, num_neg=100):
        """
        Sample negative examples from unmasked time steps.
        Args:
            z: (B, T, D)
            mask: (B, T)
        Returns:
            negs: (B, T, num_neg, D) negative samples for each position
        """
        B, T, D = z.shape
        negs = torch.zeros(B, T, num_neg, D, device=z.device, dtype=z.dtype)

        for b in range(B):
            # Get indices of unmasked positions
            unmasked = (~mask[b]).nonzero(as_tuple=False).squeeze(-1)
            if len(unmasked) == 0:
                continue
            for t in range(T):
                if len(unmasked) >= num_neg:
                    idx = unmasked[torch.randperm(len(unmasked))[:num_neg]]
                else:
                    idx = unmasked[torch.randint(0, len(unmasked), (num_neg,))]
                negs[b, t] = z[b, idx]

        return negs

    def forward(self, x):
        """
        Args:
            x: (B, 16, T) raw EMG
        Returns:
            c: (B, T_feat, context_dim) context representations
            q: (B, T_feat, feature_dim) quantized targets
            mask: (B, T_feat) boolean mask
        """
        # Feature encoding
        z = self.feature_encoder(x)  # (B, T_feat, feature_dim)

        # Quantize (for targets)
        q, perplexity = self.quantizer(z)

        # Mask latent features
        masked_z, mask = self.apply_mask(z)

        # Context encoding
        c = self.context_proj(masked_z)
        c = self.context_dropout(c)
        for block in self.conformer_blocks:
            c = block(c)
        c = self.final_proj(c)  # (B, T_feat, feature_dim)

        return c, q, mask, perplexity


# =============================================================================
# HuBERT Model
# =============================================================================

class HubertModel(nn.Module):
    """
    HuBERT-style self-supervised model.

    Instead of quantization + contrastive loss, uses:
    1. K-means clustering to generate discrete pseudo-labels
    2. Masked prediction of cluster IDs

    Simpler and more stable than wav2vec 2.0.
    """

    def __init__(
        self,
        feature_dim=512,
        context_dim=256,
        num_conformer_layers=8,
        num_heads=4,
        conv_kernel_size=15,
        dropout=0.1,
        num_clusters=500,
        mask_prob=0.5,
        mask_length=5,
    ):
        super().__init__()
        self.feature_encoder = EmgFeatureEncoder(
            in_channels=16, out_dim=feature_dim, subsample_factor=40,
        )
        self.left_context = self.feature_encoder.left_context

        self.context_proj = nn.Linear(feature_dim, context_dim)
        self.context_dropout = nn.Dropout(dropout)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model=context_dim, num_heads=num_heads,
                           conv_kernel_size=conv_kernel_size, dropout=dropout)
            for _ in range(num_conformer_layers)
        ])

        # Classification head for predicting cluster IDs
        self.cluster_head = nn.Linear(context_dim, num_clusters)
        self.num_clusters = num_clusters

        self.mask_prob = mask_prob
        self.mask_length = mask_length

    def apply_mask(self, z):
        B, T, D = z.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=z.device)
        num_masked = max(int(T * self.mask_prob / self.mask_length), 1)
        for b in range(B):
            starts = torch.randperm(T)[:num_masked]
            for s in starts:
                end = min(s + self.mask_length, T)
                mask[b, s:end] = True
        masked_z = z.clone()
        masked_z[mask] = 0.0
        return masked_z, mask

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, 16, T) raw EMG
            return_features: if True, return raw features before masking (for clustering)
        Returns:
            logits: (B, T_feat, num_clusters) cluster predictions
            mask: (B, T_feat) boolean mask
            features: (B, T_feat, context_dim) — only if return_features=True
        """
        z = self.feature_encoder(x)

        if return_features:
            # Return unmasked features for K-means clustering
            c = self.context_proj(z)
            c = self.context_dropout(c)
            for block in self.conformer_blocks:
                c = block(c)
            logits = self.cluster_head(c)
            return logits, None, c

        masked_z, mask = self.apply_mask(z)
        c = self.context_proj(masked_z)
        c = self.context_dropout(c)
        for block in self.conformer_blocks:
            c = block(c)
        logits = self.cluster_head(c)
        return logits, mask, z
