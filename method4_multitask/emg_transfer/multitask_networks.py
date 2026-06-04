# Method 4: Multi-Task Learning Networks
#
# Shared encoder → dual heads for gesture classification + joint angle regression.

import torch
import torch.nn.functional as F
from torch import nn

# Re-use Conformer building blocks
from emg_transfer.networks import (
    ConformerBlock, ConformerEncoder, ConformerGestureArchitecture,
)


class MultiTaskGesturePoseArchitecture(nn.Module):
    """
    Multi-task model: shared Conformer encoder with two heads.

    Head 1 (Gesture): gesture classification (BCE loss, 9 classes)
    Head 2 (Joint Angle): joint angle regression (MAE loss, 20 DOF)

    Training data:
      - emg_nature: has gesture labels, no joint angles → L_gesture only
      - emg2pose: has joint angles, no gesture labels → L_joint only
        (optionally use weak gesture labels inferred from joint angles)

    The shared encoder learns features useful for both tasks,
    which acts as a regularizer and provides richer supervision.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        output_dim: int = 256,
        num_gesture_classes: int = 9,
        num_joint_angles: int = 20,
        lstm_hidden: int = 128,
        use_lstm_head: bool = True,
        subsampling_stride: int = 40,
    ):
        super().__init__()

        # Shared Conformer encoder
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

        # === Gesture Classification Head ===
        self.use_lstm_head = use_lstm_head
        if use_lstm_head:
            self.gesture_lstm = nn.LSTM(
                input_size=output_dim, hidden_size=lstm_hidden,
                num_layers=1, batch_first=True, dropout=0.0,
            )
            self.gesture_classifier = nn.Linear(lstm_hidden, num_gesture_classes)
        else:
            self.gesture_classifier = nn.Linear(output_dim, num_gesture_classes)

        # === Joint Angle Regression Head ===
        self.joint_angle_head = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_joint_angles),
        )

    def forward(self, x: torch.Tensor, task: str = "gesture"):
        """
        Args:
            x: (B, 16, T) raw EMG
            task: 'gesture', 'joint', or 'both'
        Returns:
            If task='gesture':
                gesture_logits: (B, num_classes, T_feat)
            If task='joint':
                joint_angles: (B, num_joint_angles, T_feat)
            If task='both':
                (gesture_logits, joint_angles)
        """
        features = self.encoder(x)  # (B, output_dim, T_feat)
        T = x.shape[-1]
        target_len = (T - self.left_context - 1) // self.stride + 1
        if features.shape[-1] != target_len:
            features = F.interpolate(
                features, size=target_len, mode='linear', align_corners=False,
            )

        if task == "gesture":
            return self._gesture_forward(features)
        elif task == "joint":
            return self._joint_forward(features)
        elif task == "both":
            return self._gesture_forward(features), self._joint_forward(features)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _gesture_forward(self, features):
        if self.use_lstm_head:
            x = features.permute(0, 2, 1)  # (B, T, D)
            x, _ = self.gesture_lstm(x)
            x = self.gesture_classifier(x)
            x = x.permute(0, 2, 1)  # (B, C, T)
        else:
            x = self.gesture_classifier(features.transpose(1, 2)).transpose(1, 2)
        return x

    def _joint_forward(self, features):
        x = features.permute(0, 2, 1)  # (B, T, D)
        x = self.joint_angle_head(x)
        x = x.permute(0, 2, 1)  # (B, 20, T)
        return x


class JointAngleToGestureMapper:
    """
    Weak supervision: infer approximate gesture labels from joint angles.

    Maps joint angle configurations to gesture classes using heuristic thresholds.
    Used to create weak gesture labels on emg2pose data where only joint angles exist.
    """

    # Joint angle index mapping (from emg2pose constants.py)
    THUMB_CMC_FE = 0    # Thumb CMC flexion/extension
    THUMB_CMC_AA = 1    # Thumb CMC abduction/adduction
    THUMB_MCP_FE = 2    # Thumb MCP flexion/extension
    THUMB_IP_FE = 3     # Thumb IP flexion/extension
    INDEX_MCP_AA = 4    # Index MCP abduction/adduction
    INDEX_MCP_FE = 5    # Index MCP flexion/extension
    INDEX_PIP_FE = 6    # Index PIP flexion/extension
    MIDDLE_MCP_AA = 8
    MIDDLE_MCP_FE = 9
    MIDDLE_PIP_FE = 10

    # Thresholds (in radians, approximate)
    THRESH_CMC_FE_OPEN = 0.3       # thumb CMC extension → thumb_up
    THRESH_CMC_FE_CLOSE = -0.2     # thumb CMC flexion → thumb_down
    THRESH_CMC_AA_OUT = 0.3        # thumb abduction → thumb_out
    THRESH_CMC_AA_IN = -0.2        # thumb adduction → thumb_in
    THRESH_IP_FE = 0.5             # thumb IP flexion → thumb_click
    THRESH_MCP_FE_PRESS = 0.4      # finger MCP flexion → press

    @classmethod
    def infer_gesture_from_angles(cls, joint_angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert joint angles to weak gesture labels (pulse matrix format).

        Args:
            joint_angles: (B, 20, T) joint angles in radians
        Returns:
            gesture_targets: (B, 9, T) binary pulse matrix (weak labels)
            confidence: (B, 9, T) confidence scores for each label
        """
        B, _, T = joint_angles.shape
        gesture_targets = torch.zeros(B, 9, T, device=joint_angles.device)
        confidence = torch.zeros(B, 9, T, device=joint_angles.device)

        # Thumb gestures (indices 4-8)
        thumb_cmc_fe = joint_angles[:, cls.THUMB_CMC_FE, :]  # (B, T)
        thumb_cmc_aa = joint_angles[:, cls.THUMB_CMC_AA, :]
        thumb_ip_fe = joint_angles[:, cls.THUMB_IP_FE, :]

        # thumb_up (5): CMC_FE > threshold + CMC_AA > 0
        thumb_up_mask = (thumb_cmc_fe > cls.THRESH_CMC_FE_OPEN) & (thumb_cmc_aa > 0.15)
        gesture_targets[:, 8, :] = thumb_up_mask.float()
        confidence[:, 8, :] = thumb_up_mask.float() * 0.6  # moderate confidence

        # thumb_down (5): CMC_FE < negative threshold
        thumb_down_mask = thumb_cmc_fe < cls.THRESH_CMC_FE_CLOSE
        gesture_targets[:, 5, :] = thumb_down_mask.float()
        confidence[:, 5, :] = thumb_down_mask.float() * 0.7

        # thumb_in (6): CMC_AA < negative threshold
        thumb_in_mask = thumb_cmc_aa < cls.THRESH_CMC_AA_IN
        gesture_targets[:, 6, :] = thumb_in_mask.float()
        confidence[:, 6, :] = thumb_in_mask.float() * 0.6

        # thumb_out (7): CMC_AA > threshold
        thumb_out_mask = thumb_cmc_aa > cls.THRESH_CMC_AA_OUT
        gesture_targets[:, 7, :] = thumb_out_mask.float()
        confidence[:, 7, :] = thumb_out_mask.float() * 0.6

        # thumb_click (4): IP_FE > threshold
        thumb_click_mask = thumb_ip_fe > cls.THRESH_IP_FE
        gesture_targets[:, 4, :] = thumb_click_mask.float()
        confidence[:, 4, :] = thumb_click_mask.float() * 0.5  # lower confidence

        # Index finger gestures (indices 0-1)
        index_mcp_fe = joint_angles[:, cls.INDEX_MCP_FE, :]
        index_press_mask = index_mcp_fe > cls.THRESH_MCP_FE_PRESS
        gesture_targets[:, 0, :] = index_press_mask.float()
        confidence[:, 0, :] = index_press_mask.float() * 0.7

        # index_release: when MCP_FE drops back after being pressed
        # Use a simple finite difference to detect release
        # (approximated as: was pressed, now not pressed)
        index_release = torch.diff(
            (~index_press_mask).float(), dim=-1, prepend=torch.zeros(B, 1, device=joint_angles.device)
        ) > 0
        gesture_targets[:, 1, :] = index_release.float()
        confidence[:, 1, :] = index_release.float() * 0.5

        # Middle finger gestures (indices 2-3)
        middle_mcp_fe = joint_angles[:, cls.MIDDLE_MCP_FE, :]
        middle_press_mask = middle_mcp_fe > cls.THRESH_MCP_FE_PRESS
        gesture_targets[:, 2, :] = middle_press_mask.float()
        confidence[:, 2, :] = middle_press_mask.float() * 0.7

        middle_release = torch.diff(
            (~middle_press_mask).float(), dim=-1, prepend=torch.zeros(B, 1, device=joint_angles.device)
        ) > 0
        gesture_targets[:, 3, :] = middle_release.float()
        confidence[:, 3, :] = middle_release.float() * 0.5

        return gesture_targets, confidence
