"""
Channel scoring methods for 4-channel EMG selection.

Six complementary methods, all producing per-channel scores (16,):

  Method 1 — SNR (Signal-to-Noise Ratio)
  Method 2 — Fisher Discriminability
  Method 3 — Mutual Information
  Method 4 — TDS First-Layer Weight Norm
  Method 5 — Gradient Saliency
  Method 6 — Leave-One-Channel-Out CLER Drop

All methods implement: compute(...) → np.ndarray of shape (16,)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .data_utils import (
    NUM_CHANNELS,
    EMG_SAMPLE_RATE,
    SCENARIO_GESTURES,
    collect_scenario_event_data,
)

# =============================================================================
# Base class
# =============================================================================


class BaseScoringMethod(ABC):
    """Base class for a channel scoring method."""

    name: str = "base"

    @abstractmethod
    def compute(
        self,
        event_data: dict | None = None,
        scenario: str = "thumb",
    ) -> np.ndarray:
        """Return per-channel scores, shape (16,). Higher = better channel."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# Method 1: SNR
# =============================================================================


class SNRScoring(BaseScoringMethod):
    """
    Signal-to-Noise Ratio scoring.

    For each channel c:
      P_signal(c)   = mean variance of EMG_c in pre-event window [t-200ms, t]
      P_baseline(c) = mean variance of EMG_c in quiet window [t-500ms, t-300ms]
      Score(c) = 10 * log10(P_signal / P_baseline)

    Higher SNR → channel has stronger EMG activation during gestures vs. rest.
    """

    name = "snr"

    def __init__(
        self,
        signal_window_ms: tuple[float, float] = (-200.0, 0.0),
        baseline_window_ms: tuple[float, float] = (-500.0, -300.0),
    ):
        self.signal_before = -signal_window_ms[0]
        self.signal_after = signal_window_ms[1]
        self.baseline_before = -baseline_window_ms[0]
        self.baseline_after = -baseline_window_ms[1]  # relative to event time

    def compute(
        self,
        event_data: dict | None = None,
        scenario: str = "thumb",
    ) -> np.ndarray:
        if event_data is None:
            raise ValueError("SNRScoring requires event_data")

        scores = np.zeros(NUM_CHANNELS)
        gesture_names = SCENARIO_GESTURES[scenario]
        n_gestures = 0

        for name in gesture_names:
            signal_segs = event_data["signal"].get(name)
            if signal_segs is None or signal_segs.ndim < 3:
                continue
            # signal_segs: (N_events, 16, T_signal)
            signal_var = np.var(signal_segs, axis=-1).mean(axis=0)  # (16,)

            baseline_segs = event_data["baseline"].get(name)
            if baseline_segs is None or baseline_segs.ndim < 3:
                continue
            baseline_var = np.var(baseline_segs, axis=-1).mean(axis=0)  # (16,)

            # Avoid division by zero
            baseline_var = np.maximum(baseline_var, 1e-12)
            snr = 10.0 * np.log10(signal_var / baseline_var)
            scores += snr
            n_gestures += 1

        if n_gestures > 0:
            scores /= n_gestures
        return scores


# =============================================================================
# Method 2: Fisher Discriminability
# =============================================================================


class FisherScoring(BaseScoringMethod):
    """
    Fisher F-score (between-class variance / within-class variance).

    Feature per event: mean EMG value of each channel in the pre-event window.

    F(c) = Σ_g N_g * (μ_g,c - μ_c)²  /  Σ_g N_g * σ²_g,c

    Higher F → better separation between gesture classes on this channel.
    """

    name = "fisher"

    def __init__(self, window_ms: tuple[float, float] = (-200.0, 0.0)):
        self.window_before = -window_ms[0]
        self.window_after = window_ms[1]

    def _extract_features(
        self, event_data: dict, scenario: str
    ) -> tuple[dict[str, np.ndarray], int]:
        """
        Extract per-event mean EMG features.

        Returns:
            features: {gesture_name: np.ndarray(N_events, 16)}
        """
        features = {}
        gesture_names = SCENARIO_GESTURES[scenario]
        for name in gesture_names:
            segs = event_data["signal"].get(name)
            if segs is None or segs.ndim < 3 or segs.shape[0] == 0:
                continue
            # segs: (N, 16, T) → mean over T → (N, 16)
            features[name] = np.mean(segs, axis=-1)
        return features

    def compute(
        self,
        event_data: dict | None = None,
        scenario: str = "thumb",
    ) -> np.ndarray:
        if event_data is None:
            raise ValueError("FisherScoring requires event_data")

        features = self._extract_features(event_data, scenario)
        if not features:
            return np.zeros(NUM_CHANNELS)

        # Grand mean across all events
        all_features = np.concatenate(list(features.values()), axis=0)  # (N_total, 16)
        grand_mean = np.mean(all_features, axis=0)  # (16,)

        between = np.zeros(NUM_CHANNELS)
        within = np.zeros(NUM_CHANNELS)

        for name, feats in features.items():
            N_g = feats.shape[0]
            class_mean = np.mean(feats, axis=0)  # (16,)
            between += N_g * (class_mean - grand_mean) ** 2
            within += N_g * np.var(feats, axis=0)  # sum of per-class variance

        within = np.maximum(within, 1e-12)
        scores = between / within
        return scores


# =============================================================================
# Method 3: Mutual Information
# =============================================================================


class MutualInfoScoring(BaseScoringMethod):
    """
    Mutual Information between discretized mean EMG and gesture identity.

    MI(c) = Σ_bin Σ_g p(bin, g) * log(p(bin, g) / (p(bin) * p(g)))

    Higher MI → channel carries more information about which gesture occurred.
    """

    name = "mutual_info"

    def __init__(
        self,
        window_ms: tuple[float, float] = (-200.0, 0.0),
        num_bins: int = 20,
    ):
        self.window_before = -window_ms[0]
        self.window_after = window_ms[1]
        self.num_bins = num_bins

    def compute(
        self,
        event_data: dict | None = None,
        scenario: str = "thumb",
    ) -> np.ndarray:
        if event_data is None:
            raise ValueError("MutualInfoScoring requires event_data")

        gesture_names = SCENARIO_GESTURES[scenario]

        # Collect per-event mean EMG and gesture label
        all_feats = []  # list of (N_g, 16)
        all_labels = []  # list of (N_g,) label indices
        for g_idx, name in enumerate(gesture_names):
            segs = event_data["signal"].get(name)
            if segs is None or segs.ndim < 3 or segs.shape[0] == 0:
                continue
            feats = np.mean(segs, axis=-1)  # (N, 16)
            all_feats.append(feats)
            all_labels.append(np.full(feats.shape[0], g_idx))

        if not all_feats:
            return np.zeros(NUM_CHANNELS)

        X = np.concatenate(all_feats, axis=0)  # (N_total, 16)
        y = np.concatenate(all_labels, axis=0)  # (N_total,)

        scores = np.zeros(NUM_CHANNELS)
        n_gestures = len(gesture_names)

        for c in range(NUM_CHANNELS):
            x_c = X[:, c]
            # Discretize into bins using percentile edges
            bin_edges = np.percentile(x_c, np.linspace(0, 100, self.num_bins + 1))
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                continue
            # digitize returns bin index 1..len(edges)-1, 0 for below, len(edges) for above
            x_disc = np.digitize(x_c, bin_edges[1:-1])  # inner edges → 0..num_bins-1
            # Clip to valid range
            x_disc = np.clip(x_disc, 0, len(bin_edges) - 2)
            num_bins_actual = len(bin_edges) - 1

            # Joint histogram
            joint = np.zeros((num_bins_actual, n_gestures))
            for i in range(len(x_disc)):
                joint[x_disc[i], y[i]] += 1
            joint /= len(x_disc)

            # Marginals
            p_bin = joint.sum(axis=1)  # P(bin)
            p_g = joint.sum(axis=0)  # P(gesture)

            # MI = Σ p(b,g) * log(p(b,g) / (p(b) * p(g)))
            mi = 0.0
            for b in range(joint.shape[0]):
                if p_bin[b] == 0:
                    continue
                for g in range(joint.shape[1]):
                    p_joint = joint[b, g]
                    if p_joint == 0:
                        continue
                    mi += p_joint * np.log(p_joint / (p_bin[b] * p_g[g]))
            scores[c] = mi

        return scores


# =============================================================================
# Method 4: TDS First-Layer Weight Norm
# =============================================================================


class WeightNormScoring(BaseScoringMethod):
    """
    Score channels by the L2 norm of weights in the first TDS Conv1d layer.

    The pre-trained TDS has Conv1d(16→256, k=11) with weights (256, 16, 11).
    Score(c) = Σ_j ||W[j, c, :]||²  — sum of squared L2 norms per input channel.

    This is scenario-independent (same for thumb and index_middle).
    """

    name = "weight_norm"

    def __init__(self, checkpoint_path: str | Path | None = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._cached_scores: np.ndarray | None = None

    def _load_weights(self) -> np.ndarray:
        """Extract first Conv1d weight tensor from checkpoint."""
        if self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required for WeightNormScoring")

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Search for the first conv weight: encoder.layers.0.conv.conv1d.weight
        # or model.network.layers.0.conv.conv1d.weight
        for key in state_dict:
            if "layers.0.conv" in key and "weight" in key:
                w = state_dict[key].numpy()  # (256, 16, 11) or (256, 11) depending on format
                break
        else:
            # Fallback: search more broadly
            for key in state_dict:
                if "conv1d" in key and "weight" in key and "layers.0" in key:
                    w = state_dict[key].numpy()
                    break
            else:
                raise KeyError(
                    "Could not find first Conv1d weight in checkpoint. "
                    f"Available keys (first 10): {list(state_dict.keys())[:10]}"
                )

        # Ensure shape is (out_channels, in_channels, kernel_size)
        if w.ndim == 2:
            # Conv1d weight stored transposed
            w = w.reshape(w.shape[0], NUM_CHANNELS, -1)
        return w

    def compute(
        self,
        event_data: dict | None = None,
        scenario: str = "thumb",
    ) -> np.ndarray:
        if self._cached_scores is not None:
            return self._cached_scores

        w = self._load_weights()  # (256, 16, 11)
        # Sum of squares across output channels and kernel taps
        scores = np.sum(w**2, axis=(0, 2))  # (16,)
        # Normalize by max
        scores = scores / np.max(scores)
        self._cached_scores = scores
        return scores


# =============================================================================
# Method 5: Gradient Saliency
# =============================================================================


class SaliencyScoring(BaseScoringMethod):
    """
    Input-gradient saliency: mean squared gradient norm of loss w.r.t. each input channel.

    Requires a trained 16-channel model. Runs inference + backward on validation windows.
    Score(c) = mean ||∂Loss/∂Input_c||² across validation samples.
    """

    name = "saliency"

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        dataloader: torch.utils.data.DataLoader | None = None,
        num_batches: int = 200,
        device: str = "cuda",
    ):
        self.model = model
        self.dataloader = dataloader
        self.num_batches = num_batches
        self.device = device

    def compute(
        self,
        event_data: dict | None = None,
        scenario: str = "thumb",
    ) -> np.ndarray:
        if self.model is None or self.dataloader is None:
            raise ValueError(
                "SaliencyScoring requires model and dataloader. "
                "Set them before calling compute()."
            )

        self.model.to(self.device)
        self.model.eval()

        grad_sq_sum = np.zeros(NUM_CHANNELS)
        n_samples = 0

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Saliency", total=self.num_batches)):
            if batch_idx >= self.num_batches:
                break

            emg = batch["emg"].to(self.device)  # (B, 16, T)
            targets = batch["targets"].to(self.device)  # (B, 9, T)
            # Slice targets as in training
            stride = getattr(self.model, "stride", 40)
            left = getattr(self.model, "left_context", 1790)
            targets = targets[:, :, left::stride]

            emg.requires_grad_(True)

            preds = self.model(emg)
            loss = loss_fn(preds, targets)

            grads = torch.autograd.grad(loss, emg, create_graph=False, retain_graph=False)[0]
            # grads: (B, 16, T) — same shape as emg

            # Mean squared gradient per channel per sample
            for b in range(grads.shape[0]):
                for c in range(NUM_CHANNELS):
                    grad_sq_sum[c] += (grads[b, c] ** 2).sum().item()
                n_samples += 1

            emg.requires_grad_(False)

        if n_samples > 0:
            grad_sq_sum /= n_samples

        # Normalize
        if grad_sq_sum.max() > 0:
            grad_sq_sum /= grad_sq_sum.max()

        return grad_sq_sum


# =============================================================================
# Method 6: Leave-One-Channel-Out CLER Drop
# =============================================================================


class AblationScoring(BaseScoringMethod):
    """
    Leave-one-channel-out (LOO) ablation: zero out each channel and measure CLER increase.

    Score(c) = CLER(model with channel c zeroed) - CLER(full model)
    Higher CLER increase → channel is more important.

    Note: This is the most expensive method (16 inference passes over the test set).
    """

    name = "ablation"

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        test_data: list[dict] | None = None,
        device: str = "cuda",
    ):
        self.model = model
        self.test_data = test_data  # list of dicts with "emg", "targets", "prompts", "timestamps"
        self.device = device

    def _compute_cler_single(
        self, emg: torch.Tensor, targets: torch.Tensor, prompts, timestamps
    ) -> float:
        """Compute CLER for a single test sample."""
        from emg_transfer.cler import compute_cler

        self.model.eval()
        with torch.no_grad():
            preds = self.model(emg.unsqueeze(0).to(self.device))
            preds_prob = torch.sigmoid(preds).squeeze(0).cpu().numpy()

        stride = getattr(self.model, "stride", 40)
        left = getattr(self.model, "left_context", 1790)
        times_sliced = timestamps[left::stride]

        return compute_cler(preds_prob, times_sliced, prompts)

    def compute(
        self,
        event_data: dict | None = None,
        scenario: str = "thumb",
    ) -> np.ndarray:
        if self.model is None or self.test_data is None:
            raise ValueError("AblationScoring requires model and test_data")

        self.model.to(self.device)
        self.model.eval()

        # Baseline CLER (all channels)
        baseline_cler = 0.0
        n_samples = 0
        print("  Computing baseline CLER (all 16 channels)...")
        for sample in tqdm(self.test_data, desc="Baseline CLER"):
            try:
                cler = self._compute_cler_single(
                    sample["emg"], sample["targets"],
                    sample["prompts"], sample["timestamps"]
                )
                baseline_cler += cler
                n_samples += 1
            except Exception as e:
                print(f"  Skipping sample: {e}")

        if n_samples > 0:
            baseline_cler /= n_samples
        print(f"  Baseline CLER: {baseline_cler:.4f}")

        # Per-channel ablation
        scores = np.zeros(NUM_CHANNELS)
        for c in range(NUM_CHANNELS):
            print(f"  Ablating channel {c}...")
            cler_sum = 0.0
            for sample in tqdm(self.test_data, desc=f"  Ch {c}"):
                try:
                    emg_zeroed = sample["emg"].clone()
                    emg_zeroed[c, :] = 0.0
                    cler = self._compute_cler_single(
                        emg_zeroed, sample["targets"],
                        sample["prompts"], sample["timestamps"]
                    )
                    cler_sum += cler
                except Exception:
                    continue
            if n_samples > 0:
                scores[c] = cler_sum / n_samples - baseline_cler

        return scores  # Higher = more important


# =============================================================================
# Factory
# =============================================================================


def get_method(name: str, **kwargs) -> BaseScoringMethod:
    """Create a scoring method by name."""
    registry = {
        "snr": SNRScoring,
        "fisher": FisherScoring,
        "mutual_info": MutualInfoScoring,
        "weight_norm": WeightNormScoring,
        "saliency": SaliencyScoring,
        "ablation": AblationScoring,
    }
    if name not in registry:
        raise ValueError(f"Unknown method '{name}'. Choose from {list(registry.keys())}")
    return registry[name](**kwargs)
