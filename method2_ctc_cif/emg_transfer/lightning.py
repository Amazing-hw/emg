# Method 2: CTC and CIF Sequence Modeling — Lightning Modules

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

from emg_transfer.cler import compute_cler
from emg_transfer.constants import GestureType
from torchmetrics.classification import MulticlassAccuracy

log = logging.getLogger(__name__)


# =============================================================================
# Finger State Mask Generator (shared)
# =============================================================================

class FingerStateMaskGenerator(torch.nn.Module):
    def __init__(self, lpad: int = 0, rpad: int = 0) -> None:
        super().__init__()
        self.lpad = lpad
        self.rpad = rpad
        self.INDEX_FINGER = 0
        self.MIDDLE_FINGER = 1

    def forward(self, gesture_labels: torch.Tensor) -> torch.Tensor:
        batch_size, _, time_steps = gesture_labels.shape
        finger_masks = torch.zeros((batch_size, 2, time_steps), device=gesture_labels.device, dtype=torch.float32)
        for b in range(batch_size):
            self._process_finger(gesture_labels[b], finger_masks[b],
                                 press_channel=GestureType.index_press.value,
                                 release_channel=GestureType.index_release.value,
                                 output_channel=self.INDEX_FINGER, time_steps=time_steps)
            self._process_finger(gesture_labels[b], finger_masks[b],
                                 press_channel=GestureType.middle_press.value,
                                 release_channel=GestureType.middle_release.value,
                                 output_channel=self.MIDDLE_FINGER, time_steps=time_steps)
        return finger_masks

    def _process_finger(self, gesture_labels, finger_masks, press_channel,
                        release_channel, output_channel, time_steps) -> None:
        press_signal = gesture_labels[press_channel]
        release_signal = gesture_labels[release_channel]
        zero_tensor = torch.zeros(1, device=gesture_labels.device)
        press_diff = torch.diff(press_signal, n=1, prepend=zero_tensor)
        release_diff = torch.diff(release_signal, n=1, prepend=zero_tensor)
        press_onsets = torch.nonzero(press_diff > 0, as_tuple=True)[0]
        release_onsets = torch.nonzero(release_diff > 0, as_tuple=True)[0]
        if press_onsets.numel() == 0 or release_onsets.numel() == 0:
            return
        for press_idx in press_onsets:
            future_releases = release_onsets[release_onsets > press_idx]
            if future_releases.numel() == 0:
                release_idx = torch.tensor(time_steps - 1, device=finger_masks.device)
            else:
                release_idx = future_releases[0]
            start_idx = torch.clamp(press_idx - self.lpad, min=0)
            end_idx = torch.clamp(release_idx + self.rpad + 1, max=time_steps)
            finger_masks[output_channel, start_idx:end_idx] = 1.0


# =============================================================================
# Base Module
# =============================================================================

class BaseLightningModule(pl.LightningModule):
    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network = network
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx: int | None = None) -> torch.Tensor:
        return self._step(batch, stage="test")


# =============================================================================
# CTC Gesture Module
# =============================================================================

class CtcGestureModule(BaseLightningModule):
    """
    Gesture classification using CTC loss.

    Pipeline:
        1. Pulse window labels → event sequence labels
        2. Encoder + CTC head → frame-level log-probabilities
        3. CTC loss with blank token
        4. Greedy CTC decoding for validation
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        lr_scheduler_milestones: list[int],
        lr_scheduler_factor: float,
        warmup_start_factor: float,
        warmup_end_factor: float,
        warmup_total_epochs: int,
        gradient_clip_val: float,
        freeze_backbone_epochs: int = 5,
        backbone_lr_ratio: float = 0.02,
    ) -> None:
        super().__init__(network=network, optimizer=optimizer)
        self.blank_id = network.blank_id
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, reduction='mean', zero_infinity=True)

        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.backbone_lr_ratio = backbone_lr_ratio
        self.val_accuracy = MulticlassAccuracy(num_classes=9)

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        return self.network(emg)

    def _pulse_to_event_sequence(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert pulse window targets to event sequence for CTC.

        For each gesture class, find the first active frame in each pulse
        and treat it as an event occurrence. Events are sorted by time.

        Args:
            targets: (B, 9, T) pulse matrix
        Returns:
            event_seqs: (B, max_events) padded event class indices
            event_lengths: (B,) number of events per sample
        """
        B, C, T = targets.shape
        event_seqs_list = []
        event_lengths = []

        for b in range(B):
            events = []
            for c in range(C):
                pulse = targets[b, c]
                # Find rising edges (pulse start)
                diff = torch.diff(pulse, prepend=torch.zeros(1, device=pulse.device))
                onset_indices = torch.where(diff > 0)[0]
                for idx in onset_indices:
                    events.append((idx.item(), c))
            # Sort by time
            events.sort(key=lambda x: x[0])
            seq = torch.tensor([e[1] for e in events], dtype=torch.long, device=targets.device)
            event_seqs_list.append(seq)
            event_lengths.append(len(seq))

        max_len = max(event_lengths) if event_lengths else 1
        max_len = max(max_len, 1)
        padded_seqs = torch.zeros(B, max_len, dtype=torch.long, device=targets.device)
        for b, seq in enumerate(event_seqs_list):
            if len(seq) > 0:
                padded_seqs[b, :len(seq)] = seq

        return padded_seqs, torch.tensor(event_lengths, dtype=torch.long, device=targets.device)

    def _ctc_decode(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> list[list[int]]:
        """
        Greedy CTC decoding.
        Args:
            log_probs: (B, T, 10) log-probabilities
            input_lengths: (B,) valid frame counts
        Returns:
            decoded: list of list of class indices (no blanks, no repeats)
        """
        B = log_probs.shape[0]
        decoded = []
        for b in range(B):
            T_valid = input_lengths[b].item()
            best_path = log_probs[b, :T_valid].argmax(dim=-1)  # (T_valid,)
            # Collapse: remove blanks and consecutive repeats
            collapsed = []
            prev = self.blank_id
            for t in range(T_valid):
                token = best_path[t].item()
                if token != self.blank_id and token != prev:
                    collapsed.append(token)
                prev = token
            decoded.append(collapsed)
        return decoded

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str = "train") -> float:
        emg = batch["emg"]
        targets = batch["targets"]  # (B, 9, T) pulse matrix

        # Convert pulse targets to event sequences for CTC
        event_seqs, event_lengths = self._pulse_to_event_sequence(targets)

        # Forward
        log_probs = self.forward(emg)  # (B, T_feat, num_classes+1)
        input_lengths = self.network.get_input_lengths(emg)

        # CTC loss
        # CTCLoss expects: log_probs (T, B, C), targets (B, S)
        log_probs_ctc = log_probs.transpose(0, 1)  # (T_feat, B, num_classes+1)
        loss = self.ctc_loss(log_probs_ctc, event_seqs, input_lengths, event_lengths)
        self.log(f"{stage}_loss", loss, sync_dist=True)

        if stage == "val":
            # Decode and compute accuracy
            decoded = self._ctc_decode(log_probs, input_lengths)
            # Compute per-event accuracy against ground truth event sequences
            all_correct = 0
            all_total = 0
            for b in range(len(decoded)):
                gt_seq = event_seqs[b, :event_lengths[b]].tolist()
                pred_seq = decoded[b]
                for i in range(min(len(gt_seq), len(pred_seq))):
                    if gt_seq[i] == pred_seq[i]:
                        all_correct += 1
                all_total += max(len(gt_seq), len(pred_seq))
            if all_total > 0:
                acc = all_correct / all_total
                self.log("val_accuracy", acc, sync_dist=True, prog_bar=True)

        if stage == "test":
            # Use original BCE-based CLER evaluation
            prompts = batch["prompts"][0]
            # Convert log_probs to BCE-style probabilities
            probs = torch.exp(log_probs[:, :, :9])  # (B, T, 9) — drop blank
            probs_np = probs.squeeze(0).detach().cpu().numpy().T  # (9, T)
            times = batch["timestamps"][0]
            times = times[self.network.left_context :: self.network.stride]
            if len(times) != probs_np.shape[1]:
                min_len = min(len(times), probs_np.shape[1])
                times = times[:min_len]
                probs_np = probs_np[:, :min_len]
            cler = compute_cler(probs_np, times, prompts)
            self.log("test_cler", cler, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_train_epoch_start(self):
        if hasattr(self.network, 'encoder'):
            if self.current_epoch < self.freeze_backbone_epochs:
                self.network.encoder.requires_grad_(False)
                self.network.encoder.eval()
                log.info(f"Epoch {self.current_epoch}: encoder FROZEN")
            elif self.current_epoch == self.freeze_backbone_epochs:
                self.network.encoder.requires_grad_(True)
                self.network.encoder.train()
                log.info(f"Epoch {self.current_epoch}: encoder UNFROZEN")

    def configure_optimizers(self):
        if hasattr(self.network, 'encoder'):
            backbone_lr = self.hparams.learning_rate * self.backbone_lr_ratio
            backbone_params = list(self.network.encoder.parameters())
            head_params = [p for n, p in self.network.named_parameters() if not n.startswith('encoder')]
            param_groups = [
                {'params': head_params, 'lr': self.hparams.learning_rate},
                {'params': backbone_params, 'lr': backbone_lr},
            ]
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams.warmup_start_factor,
                                                    end_factor=self.hparams.warmup_end_factor,
                                                    total_iters=self.hparams.warmup_total_epochs)
        multistep = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_scheduler_milestones,
                                                          gamma=self.hparams.lr_scheduler_factor)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, multistep])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
                "gradient_clip_val": self.hparams.gradient_clip_val}


# =============================================================================
# CIF Gesture Module
# =============================================================================

class CifGestureModule(BaseLightningModule):
    """
    Gesture classification using CIF alignment.

    CIF learns event boundaries automatically, replacing manual pulse windows.
    Loss: CrossEntropy on fired events + length regularization.
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        lr_scheduler_milestones: list[int],
        lr_scheduler_factor: float,
        warmup_start_factor: float,
        warmup_end_factor: float,
        warmup_total_epochs: int,
        gradient_clip_val: float,
        freeze_backbone_epochs: int = 5,
        backbone_lr_ratio: float = 0.02,
        quantity_loss_weight: float = 0.1,
    ) -> None:
        super().__init__(network=network, optimizer=optimizer)
        self.quantity_loss_weight = quantity_loss_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.backbone_lr_ratio = backbone_lr_ratio

    def forward(self, emg: torch.Tensor) -> dict:
        return self.network(emg)

    def _pulse_to_event_targets(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert pulse matrix to event class target for CIF.
        Each event (rising edge) becomes a target class label.

        Returns:
            target_classes: (B, max_events) class indices for each event
            target_lengths: (B,) number of events
        """
        B, C, T = targets.shape
        all_classes = []
        all_lengths = []
        for b in range(B):
            events = []
            for c in range(C):
                diff = torch.diff(targets[b, c], prepend=torch.zeros(1, device=targets.device))
                onset_indices = torch.where(diff > 0)[0]
                for _ in onset_indices:
                    events.append(c)
            events_sorted = sorted(events)  # Not time-sorted, just grouped
            if len(events_sorted) == 0:
                events_sorted = [0]  # dummy
            all_classes.append(torch.tensor(events_sorted, dtype=torch.long, device=targets.device))
            all_lengths.append(len(events_sorted))

        max_events = max(all_lengths) if all_lengths else 1
        padded = torch.full((B, max_events), -100, dtype=torch.long, device=targets.device)
        for b in range(B):
            padded[b, :all_lengths[b]] = all_classes[b]
        return padded, torch.tensor(all_lengths, dtype=torch.long, device=targets.device)

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str = "train") -> float:
        emg = batch["emg"]
        targets = batch["targets"]

        # Convert pulse targets to event class targets
        target_classes, target_lengths = self._pulse_to_event_targets(targets)

        output = self.forward(emg)  # dict with logits, alpha, num_events
        logits = output["logits"]  # (B, max_events, num_classes)
        num_events = output["num_events"]  # (B,)
        alpha = output["alpha"]  # (B, T)

        # Cross-entropy loss on events
        B, U, C = logits.shape
        logits_flat = logits.reshape(B * U, C)
        targets_flat = target_classes.reshape(B * U)
        ce_loss = self.ce_loss(logits_flat, targets_flat)

        # Quantity loss: encourage correct number of events
        quantity_loss = F.l1_loss(
            num_events.float(), target_lengths.float()
        ) * self.quantity_loss_weight

        loss = ce_loss + quantity_loss
        self.log(f"{stage}_loss", loss, sync_dist=True)
        self.log(f"{stage}_ce_loss", ce_loss, sync_dist=True)
        self.log(f"{stage}_quantity_loss", quantity_loss, sync_dist=True)

        if stage == "val":
            # Accuracy: argmax per event
            preds = logits.argmax(dim=-1)  # (B, max_events)
            correct = 0
            total = 0
            for b in range(B):
                valid = target_classes[b] != -100
                pred_b = preds[b]
                gt_b = target_classes[b]
                n_valid = valid.sum().item()
                if n_valid > 0:
                    correct += (pred_b[valid] == gt_b[valid]).sum().item()
                    total += n_valid
            if total > 0:
                self.log("val_accuracy", correct / total, sync_dist=True, prog_bar=True)

        return loss

    def on_train_epoch_start(self):
        if hasattr(self.network, 'encoder'):
            if self.current_epoch < self.freeze_backbone_epochs:
                self.network.encoder.requires_grad_(False)
                self.network.encoder.eval()
            elif self.current_epoch == self.freeze_backbone_epochs:
                self.network.encoder.requires_grad_(True)
                self.network.encoder.train()

    def configure_optimizers(self):
        if hasattr(self.network, 'encoder'):
            backbone_lr = self.hparams.learning_rate * self.backbone_lr_ratio
            backbone_params = list(self.network.encoder.parameters())
            head_params = [p for n, p in self.network.named_parameters() if not n.startswith('encoder')]
            param_groups = [
                {'params': head_params, 'lr': self.hparams.learning_rate},
                {'params': backbone_params, 'lr': backbone_lr},
            ]
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams.warmup_start_factor,
                                                    end_factor=self.hparams.warmup_end_factor,
                                                    total_iters=self.hparams.warmup_total_epochs)
        multistep = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_scheduler_milestones,
                                                          gamma=self.hparams.lr_scheduler_factor)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, multistep])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
                "gradient_clip_val": self.hparams.gradient_clip_val}


# =============================================================================
# Original BCE Module (for baseline comparison in this project)
# =============================================================================

class DiscreteGesturesModule(BaseLightningModule):
    """Original BCE-based gesture module (baseline for comparison)."""

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer,
                 learning_rate: float, lr_scheduler_milestones: list[int],
                 lr_scheduler_factor: float, warmup_start_factor: float,
                 warmup_end_factor: float, warmup_total_epochs: int,
                 gradient_clip_val: float, freeze_backbone_epochs: int = 0,
                 backbone_lr_ratio: float = 0.02) -> None:
        super().__init__(network=network, optimizer=optimizer)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        output_freq = 2000 // network.stride
        rpad = round(0.040 * output_freq)
        self.mask_generator = FingerStateMaskGenerator(lpad=0, rpad=rpad)
        self.val_accuracy = MulticlassAccuracy(num_classes=9)
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.backbone_lr_ratio = backbone_lr_ratio

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        return self.network(emg)

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str = "train") -> float:
        emg = batch["emg"]
        targets = batch["targets"]
        targets = targets[:, :, self.network.left_context :: self.network.stride]
        release_mask = self.mask_generator(targets)
        mask = torch.ones_like(targets)
        mask[:, [GestureType.index_release.value, GestureType.middle_release.value], :] = release_mask
        preds = self.forward(emg)
        loss = self.loss_fn(preds, targets)
        loss = (loss * mask).sum() / mask.sum()
        self.log(f"{stage}_loss", loss, sync_dist=True)
        if stage == "val":
            output_freq = 2000 // self.network.stride
            w_start = round(0.050 * output_freq)
            w_end = round(0.150 * output_freq)
            probs = torch.sigmoid(preds)
            y = targets.to(torch.int32)
            y_class, y_hat_class = [], []
            for b in range(y.shape[0]):
                y_diff = torch.diff(y[b], axis=0)
                indices = torch.argwhere(y_diff == 1)
                for index in indices:
                    start = max(index[0] - w_start, 0)
                    end = min(index[0] + w_end, y.shape[1])
                    y_hat = probs[b, start:end, :]
                    col = y_hat.argmax() % y_hat.shape[1]
                    y_hat_class.append(col)
                    y_class.append(index[1])
            if len(y_class) > 0:
                y_class = torch.stack(y_class).long()
                y_hat_class = torch.stack(y_hat_class).long()
                self.val_accuracy.update(y_hat_class, y_class)
                self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        if stage == "test":
            prompts = batch["prompts"][0]
            times = batch["timestamps"][0]
            preds_prob = nn.Sigmoid()(preds)
            preds_prob = preds_prob.squeeze(0).detach().cpu().numpy()
            times = times[self.network.left_context :: self.network.stride]
            cler = compute_cler(preds_prob, times, prompts)
            self.log("test_cler", cler, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_start(self):
        if hasattr(self.network, 'encoder'):
            if self.current_epoch < self.freeze_backbone_epochs:
                self.network.encoder.requires_grad_(False)
                self.network.encoder.eval()
            elif self.current_epoch == self.freeze_backbone_epochs:
                self.network.encoder.requires_grad_(True)
                self.network.encoder.train()

    def configure_optimizers(self):
        if hasattr(self.network, 'encoder'):
            backbone_lr = self.hparams.learning_rate * self.backbone_lr_ratio
            backbone_params = list(self.network.encoder.parameters())
            head_params = [p for n, p in self.network.named_parameters() if not n.startswith('encoder')]
            param_groups = [{'params': head_params, 'lr': self.hparams.learning_rate},
                            {'params': backbone_params, 'lr': backbone_lr}]
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams.warmup_start_factor,
                                                    end_factor=self.hparams.warmup_end_factor,
                                                    total_iters=self.hparams.warmup_total_epochs)
        multistep = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_scheduler_milestones,
                                                          gamma=self.hparams.lr_scheduler_factor)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, multistep])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
                "gradient_clip_val": self.hparams.gradient_clip_val}


# =============================================================================
# Wrist and Handwriting modules (for completeness)
# =============================================================================

class WristModule(BaseLightningModule):
    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        super().__init__(network=network, optimizer=optimizer)
        self.loss_fn = torch.nn.L1Loss(reduction="mean")

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str = "train") -> torch.Tensor:
        emg = batch["emg"]
        wrist_angles = batch["wrist_angles"]
        preds = self.forward(emg)
        wrist_angles = wrist_angles[:, :, self.network.left_context :: self.network.stride]
        preds = preds[:, :, 1:]
        labels = torch.diff(wrist_angles, dim=2)
        loss = self.loss_fn(preds, labels)
        self.log(f"{stage}_loss", loss, sync_dist=True)
        return loss
