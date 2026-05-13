"""
Finger-specific channel scoring using emg2pose pre-trained model.

Key insight: the emg2pose model predicts 20 DOF joint angles mapped to 5 fingers.
By computing per-finger gradient attribution, we can determine which EMG channels
are most important for predicting each finger's movement.

Joint-to-finger mapping (from emg2pose.constants):
  Thumb:  joints 0-3  (CMC_FE, CMC_AA, MCP_FE, IP_FE)
  Index:  joints 4-7  (MCP_AA, MCP_FE, PIP_FE, DIP_FE)
  Middle: joints 8-11 (MCP_AA, MCP_FE, PIP_FE, DIP_FE)
  Ring:   joints 12-15
  Pinky:  joints 16-19

For our 2 scenarios:
  Scenario A (thumb):       thumb joints 0-3
  Scenario B (index_middle): index joints 4-7 + middle joints 8-11
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data_utils import NUM_CHANNELS, EMG_SAMPLE_RATE

# Finger-to-joint mapping
FINGER_JOINTS = {
    "thumb":  [0, 1, 2, 3],
    "index":  [4, 5, 6, 7],
    "middle": [8, 9, 10, 11],
    "ring":   [12, 13, 14, 15],
    "pinky":  [16, 17, 18, 19],
}

# Scenario to finger mapping
SCENARIO_FINGERS = {
    "thumb":        ["thumb"],
    "index_middle": ["index", "middle"],
}

JOINT_NAMES = [
    "THUMB_CMC_FE", "THUMB_CMC_AA", "THUMB_MCP_FE", "THUMB_IP_FE",
    "INDEX_MCP_AA", "INDEX_MCP_FE", "INDEX_PIP_FE", "INDEX_DIP_FE",
    "MIDDLE_MCP_AA", "MIDDLE_MCP_FE", "MIDDLE_PIP_FE", "MIDDLE_DIP_FE",
    "RING_MCP_AA", "RING_MCP_FE", "RING_PIP_FE", "RING_DIP_FE",
    "PINKY_MCP_AA", "PINKY_MCP_FE", "PINKY_PIP_FE", "PINKY_DIP_FE",
]


def build_tds_backbone_from_ckpt(ckpt_path: str, device: str = "cpu") -> nn.Module:
    """
    Build TDS backbone and load pre-trained weights from emg2pose checkpoint.

    Uses the TDS building blocks from emg_transfer.networks.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "emg_transfer"))
    from emg_transfer.networks import build_tds_network

    tds = build_tds_network()
    tds.num_channels = 16

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Load TDS backbone weights (model.network.layers.X.*)
    tds_sd = {}
    prefix = "model.network."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            tds_sd[new_k] = v

    if not tds_sd:
        prefix = "network."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                tds_sd[new_k] = v

    tds.load_state_dict(tds_sd, strict=True)
    tds.to(device)
    tds.eval()
    return tds


def build_emg2pose_decoder_from_ckpt(ckpt_path: str, device: str = "cpu") -> nn.Module:
    """
    Build the emg2pose decoder (LSTM + MLP) and load weights.

    Decoder structure from tracking_vemg2pose:
      LSTM(input=84, hidden=512, num_layers=2)
      MLP: Linear(512, 20)

    Input is concatenation of [TDS_features(64), previous_state(20)] = 84.
    For our purposes, we feed zeros as the state input.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Build LSTM
    lstm = nn.LSTM(input_size=84, hidden_size=512, num_layers=2, batch_first=True)
    # Build MLP output
    mlp = nn.Linear(512, 20)

    # Load weights
    prefix = "model.decoder."
    lstm_sd = {}
    mlp_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix + "lstm."):
            lstm_sd[k[len(prefix + "lstm."):]] = v
        elif k.startswith(prefix + "mlp_out.1."):
            mlp_sd[k[len(prefix + "mlp_out.1."):]] = v

    lstm.load_state_dict(lstm_sd, strict=True)
    mlp.load_state_dict(mlp_sd, strict=True)

    class Decoder(nn.Module):
        def __init__(self, lstm, mlp):
            super().__init__()
            self.lstm = lstm
            self.mlp = mlp

        def forward(self, tds_features):
            # tds_features: (B, 64, T_feat) → (B, T_feat, 64)
            x = tds_features.permute(0, 2, 1)
            B, T, _ = x.shape
            # Concatenate zero initial state: (B, T_feat, 84)
            state_input = torch.zeros(B, T, 20, device=x.device, dtype=x.dtype)
            x = torch.cat([x, state_input], dim=-1)
            x, _ = self.lstm(x)  # (B, T_feat, 512)
            x = self.mlp(x)  # (B, T_feat, 20)
            return x.permute(0, 2, 1)  # (B, 20, T_feat)

    decoder = Decoder(lstm, mlp)
    decoder.to(device)
    decoder.eval()
    return decoder


def compute_finger_channel_scores(
    ckpt_path: str,
    dataloader: torch.utils.data.DataLoader,
    scenario: str = "thumb",
    num_batches: int = 100,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """
    Compute per-finger per-channel importance scores using gradient attribution
    through the full emg2pose model (TDS backbone + decoder).

    For each finger relevant to the scenario, computes:
      Score_finger[c] = mean(||∂(Σ joint_preds_finger) / ∂EMG_c||²)

    Returns:
        dict mapping finger_name → np.ndarray(16,) of channel importance scores
    """
    print(f"\n  Building emg2pose model from {ckpt_path}...")
    tds = build_tds_backbone_from_ckpt(ckpt_path, device)
    decoder = build_emg2pose_decoder_from_ckpt(ckpt_path, device)

    fingers = SCENARIO_FINGERS.get(scenario, ["thumb"])
    print(f"  Scenario '{scenario}' → fingers: {fingers}")

    # Accumulate gradients per finger
    grad_sq_sums = {f: np.zeros(NUM_CHANNELS) for f in fingers}
    n_samples = 0

    print(f"  Running gradient attribution on {num_batches} batches...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="  Finger grad", total=num_batches)):
        if batch_idx >= num_batches:
            break

        emg = batch["emg"].to(device)  # (B, 16, T)
        if emg.dim() == 2:
            emg = emg.unsqueeze(0)

        emg.requires_grad_(True)

        # Forward through TDS backbone
        features = tds(emg)  # (B, 64, T/80)

        # Forward through decoder
        joint_preds = decoder(features)  # (B, 20, T_feat)

        # For each finger, compute gradient of summed joint predictions
        for finger in fingers:
            joint_idxs = FINGER_JOINTS[finger]
            # Sum predictions for this finger's joints over time
            finger_pred_sum = joint_preds[:, joint_idxs, :].sum()

            grads = torch.autograd.grad(
                finger_pred_sum, emg,
                create_graph=False, retain_graph=True
            )[0]  # (B, 16, T)

            # Accumulate squared gradients per channel
            for b in range(grads.shape[0]):
                for c in range(NUM_CHANNELS):
                    grad_sq_sums[finger][c] += (grads[b, c] ** 2).sum().item()
                if finger == fingers[0]:
                    n_samples += 1  # count once

        emg.requires_grad_(False)

    # Normalize
    for finger in fingers:
        if n_samples > 0:
            grad_sq_sums[finger] /= n_samples
        mx = grad_sq_sums[finger].max()
        if mx > 0:
            grad_sq_sums[finger] /= mx

    return grad_sq_sums


def compute_finger_channel_scores_tds_only(
    ckpt_path: str,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 100,
    device: str = "cuda",
) -> np.ndarray:
    """
    Simplified version: compute gradient of TDS output features w.r.t. input channels.
    This is scenario-independent — just shows which channels drive the TDS most.

    Returns:
        np.ndarray(16,) — per-channel importance for the TDS feature extractor.
    """
    print(f"\n  Building TDS backbone from {ckpt_path}...")
    tds = build_tds_backbone_from_ckpt(ckpt_path, device)

    grad_sq_sum = np.zeros(NUM_CHANNELS)
    n_samples = 0

    print(f"  Running TDS-only gradient attribution...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="  TDS grad", total=num_batches)):
        if batch_idx >= num_batches:
            break

        emg = batch["emg"].to(device)
        if emg.dim() == 2:
            emg = emg.unsqueeze(0)

        emg.requires_grad_(True)
        features = tds(emg)  # (B, 64, T/80)

        # Gradient of L2 norm of all TDS features w.r.t. input
        feat_norm = (features ** 2).sum()
        grads = torch.autograd.grad(feat_norm, emg, create_graph=False)[0]

        for b in range(grads.shape[0]):
            for c in range(NUM_CHANNELS):
                grad_sq_sum[c] += (grads[b, c] ** 2).sum().item()
            n_samples += 1

        emg.requires_grad_(False)

    if n_samples > 0:
        grad_sq_sum /= n_samples
    if grad_sq_sum.max() > 0:
        grad_sq_sum /= grad_sq_sum.max()

    return grad_sq_sum


def analyze_decoder_weights_direct(ckpt_path: str) -> dict[str, np.ndarray]:
    """
    Direct weight analysis of the decoder's final linear projection.

    The decoder MLP has weight (20, 512) and bias (20,).
    Each row corresponds to one joint angle output.

    We extract the L2 norm of weights per joint, then sum per finger.
    This tells us which joints the decoder allocates most capacity to.

    Returns:
        dict with:
          - "per_joint_weight": (20,) — L2 norm of each joint's weight row
          - "per_finger_weight": {finger: float} — sum of norms per finger
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Find the MLP output weight
    mlp_key = "model.decoder.mlp_out.1.weight"  # (20, 512)
    if mlp_key not in state_dict:
        # Try alternative
        for k in state_dict:
            if "mlp_out" in k and "weight" in k:
                mlp_key = k
                break

    weight = state_dict[mlp_key].numpy()  # (20, 512)
    per_joint_norm = np.sqrt((weight ** 2).sum(axis=1))  # (20,)

    per_finger = {}
    for finger, idxs in FINGER_JOINTS.items():
        per_finger[finger] = float(per_joint_norm[idxs].sum())

    return {
        "per_joint_weight": per_joint_norm.tolist(),
        "per_joint_names": JOINT_NAMES,
        "per_finger_weight": per_finger,
    }


def compute_finger_scores(
    ckpt_path: str,
    dataloader: torch.utils.data.DataLoader | None = None,
    scenario: str = "both",
    num_batches: int = 50,
    device: str = "cuda",
) -> dict:
    """
    Master function: compute finger-specific channel scores using all available methods.

    Methods:
      1. Direct decoder weight analysis (fast, requires only checkpoint)
      2. Gradient attribution with full model (slow, requires dataloader + GPU)
      3. TDS-only gradient attribution (medium)

    Returns:
        Comprehensive dict with per-finger channel scores.
    """
    results = {}

    # Method 1: Direct weight analysis (always available)
    print("\n[Finger Analysis] Method 1: Decoder weight analysis...")
    dw = analyze_decoder_weights_direct(ckpt_path)
    results["decoder_weights"] = dw
    for finger, w in dw["per_finger_weight"].items():
        print(f"  {finger}: weight_norm = {w:.2f}")

    # Method 2: Gradient attribution (needs dataloader)
    if dataloader is not None and torch.cuda.is_available():
        print("\n[Finger Analysis] Method 2: Full model gradient attribution...")
        try:
            if scenario == "both":
                for scen in ["thumb", "index_middle"]:
                    finger_scores = compute_finger_channel_scores(
                        ckpt_path, dataloader, scen, num_batches, device
                    )
                    results[f"gradient_{scen}"] = {
                        finger: scores.tolist()
                        for finger, scores in finger_scores.items()
                    }
                    for finger, scores in finger_scores.items():
                        top3 = np.argsort(scores)[::-1][:3]
                        print(f"  {scen}/{finger}: top channels = {top3.tolist()}")
            else:
                finger_scores = compute_finger_channel_scores(
                    ckpt_path, dataloader, scenario, num_batches, device
                )
                results["gradient"] = {
                    finger: scores.tolist()
                    for finger, scores in finger_scores.items()
                }
                for finger, scores in finger_scores.items():
                    top3 = np.argsort(scores)[::-1][:3]
                    print(f"  {finger}: top channels = {top3.tolist()}")
        except Exception as e:
            print(f"  Gradient attribution failed: {e}")
            import traceback
            traceback.print_exc()

    # Method 3: TDS-only gradient
    if dataloader is not None and torch.cuda.is_available():
        print("\n[Finger Analysis] Method 3: TDS-only gradient...")
        try:
            tds_scores = compute_finger_channel_scores_tds_only(
                ckpt_path, dataloader, num_batches, device
            )
            results["tds_gradient"] = tds_scores.tolist()
            top3 = np.argsort(tds_scores)[::-1][:3]
            print(f"  TDS top channels: {top3.tolist()}")
        except Exception as e:
            print(f"  TDS gradient failed: {e}")

    return results
