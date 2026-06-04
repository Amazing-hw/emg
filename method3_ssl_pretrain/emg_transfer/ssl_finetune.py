# Method 3: Fine-tuning after SSL Pre-training

import logging
import torch
import torch.nn.functional as F
from torch import nn

from emg_transfer.ssl_networks import EmgFeatureEncoder, ConformerBlock

log = logging.getLogger(__name__)


class SslGestureArchitecture(nn.Module):
    """
    Gesture classification model that fine-tunes a pre-trained SSL encoder.

    Accepts either a wav2vec 2.0 or HuBERT pre-trained feature encoder
    and context encoder, adds a gesture classification head.

    Pipeline:
        raw EMG (B, 16, T) @2000Hz
          → FeatureEncoder (pre-trained) → z @50Hz
          → ContextEncoder (pre-trained Conformer) → c @50Hz
          → Classification Head → (B, num_classes, T_feat) @50Hz
    """

    def __init__(
        self,
        feature_encoder: EmgFeatureEncoder | None = None,
        num_conformer_layers: int = 8,
        context_dim: int = 256,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        num_classes: int = 9,
        lstm_hidden: int = 128,
        use_lstm_head: bool = True,
    ):
        super().__init__()
        if feature_encoder is None:
            feature_encoder = EmgFeatureEncoder(in_channels=16, out_dim=512, subsample_factor=40)
        self.feature_encoder = feature_encoder

        self.left_context = self.feature_encoder.left_context  # 104
        self.stride = self.feature_encoder.subsample_factor  # 40

        # Context encoder: Conformer blocks (same as during pre-training)
        self.context_proj = nn.Linear(512, context_dim)
        self.context_dropout = nn.Dropout(dropout)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model=context_dim, num_heads=num_heads,
                           conv_kernel_size=conv_kernel_size, dropout=dropout)
            for _ in range(num_conformer_layers)
        ])

        # Classification head
        self.use_lstm_head = use_lstm_head
        if use_lstm_head:
            self.lstm = nn.LSTM(
                input_size=context_dim, hidden_size=lstm_hidden,
                num_layers=1, batch_first=True, dropout=0.0,
            )
            self.classifier = nn.Linear(lstm_hidden, num_classes)
        else:
            self.classifier = nn.Linear(context_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 16, T) raw EMG
        Returns:
            (B, num_classes, T_out) gesture logits
        """
        # Feature encoding
        z = self.feature_encoder(x)  # (B, T_feat, 512)

        # Context encoding
        c = self.context_proj(z)
        c = self.context_dropout(c)
        for block in self.conformer_blocks:
            c = block(c)  # (B, T_feat, context_dim)

        # Compute target length
        T = x.shape[-1]
        target_len = (T - self.left_context - 1) // self.stride + 1

        if c.shape[1] != target_len:
            c = F.interpolate(
                c.transpose(1, 2), size=target_len, mode='linear', align_corners=False,
            ).transpose(1, 2)

        # Classification
        if self.use_lstm_head:
            c, _ = self.lstm(c)  # (B, T_feat, lstm_hidden)
        out = self.classifier(c)  # (B, T_feat, num_classes)
        return out.transpose(1, 2)  # (B, num_classes, T_feat)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        num_classes: int = 9,
        lstm_hidden: int = 128,
        use_lstm_head: bool = True,
        freeze_feature_encoder: bool = False,
    ) -> "SslGestureArchitecture":
        """
        Load a pre-trained SSL model and create a fine-tuning architecture.

        Args:
            pretrained_path: Path to pre-trained checkpoint
            num_classes: Number of gesture classes
            lstm_hidden: LSTM hidden size for classification head
            use_lstm_head: Whether to use LSTM or Linear head
            freeze_feature_encoder: If True, freeze the feature encoder during fine-tuning
        Returns:
            SslGestureArchitecture ready for fine-tuning
        """
        log.info(f"Loading pre-trained model from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location='cpu')

        # Extract state dict and model config
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Remove 'model.' prefix if present
        clean_sd = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_sd[k[6:]] = v
            else:
                clean_sd[k] = v

        # Infer architecture from state dict keys
        feature_dim = 512
        context_dim = 256 if 'context_proj.weight' in clean_sd else 512
        num_layers = sum(1 for k in clean_sd if k.startswith('conformer_blocks'))

        if num_layers == 0:
            log.warning("Could not infer num_layers from checkpoint, defaulting to 8")
            num_layers = 8

        # Create feature encoder
        feature_encoder = EmgFeatureEncoder(in_channels=16, out_dim=feature_dim, subsample_factor=40)

        # Create model
        model = cls(
            feature_encoder=feature_encoder,
            num_conformer_layers=num_layers,
            context_dim=context_dim,
            num_classes=num_classes,
            lstm_hidden=lstm_hidden,
            use_lstm_head=use_lstm_head,
        )

        # Load pre-trained weights (ignore classifier head)
        # Map keys
        model_sd = model.state_dict()
        matched_keys = []
        for k, v in clean_sd.items():
            if k in model_sd and k not in ['lstm.weight_ih_l0', 'lstm.weight_hh_l0',
                                              'lstm.bias_ih_l0', 'lstm.bias_hh_l0',
                                              'classifier.weight', 'classifier.bias']:
                if model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    matched_keys.append(k)

        model.load_state_dict(model_sd)
        log.info(f"Loaded {len(matched_keys)} pre-trained parameters")

        if freeze_feature_encoder:
            model.feature_encoder.requires_grad_(False)
            log.info("Feature encoder frozen")

        return model

    def load_pretrained(self, ckpt_path: str):
        """
        Load SSL pre-trained weights into this model instance.
        Used by the training pipeline when pretrained_encoder_ckpt is set.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Remove 'model.' prefix
        clean_sd = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_sd[k[6:]] = v
            else:
                clean_sd[k] = v

        # Filter out classification head params
        exclude_keys = {'lstm.weight_ih_l0', 'lstm.weight_hh_l0',
                        'lstm.bias_ih_l0', 'lstm.bias_hh_l0',
                        'classifier.weight', 'classifier.bias',
                        'cluster_head.weight', 'cluster_head.bias',
                        'final_proj.weight', 'final_proj.bias',
                        'quantizer.codebooks', 'quantizer.projections'}

        model_sd = self.state_dict()
        matched, skipped = 0, 0
        for k, v in clean_sd.items():
            if k in model_sd and not any(ex in k for ex in exclude_keys):
                if model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    matched += 1
                else:
                    skipped += 1

        self.load_state_dict(model_sd)
        log.info(f"Loaded {matched} SSL pre-trained parameters (skipped {skipped} shape mismatch)")
