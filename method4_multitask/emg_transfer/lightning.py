# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

import pytorch_lightning as pl
import torch

from emg_transfer.cler import compute_cler
from emg_transfer.constants import GestureType
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

log = logging.getLogger(__name__)


class BaseLightningModule(pl.LightningModule):
    """Child classes should implement _step."""

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network = network
        self.optimizer = optimizer

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        return self.network(emg)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(
        self, batch, batch_idx, dataloader_idx: int | None = None
    ) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return self.optimizer(self.parameters())


class WristModule(BaseLightningModule):
    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        super().__init__(network=network, optimizer=optimizer)
        self.loss_fn = torch.nn.L1Loss(reduction="mean")

    def _step(
        self, batch: Mapping[str, torch.Tensor], stage: str = "train"
    ) -> torch.Tensor:
        emg = batch["emg"]
        wrist_angles = batch["wrist_angles"]
        preds = self.forward(emg)
        wrist_angles = wrist_angles[
            :, :, self.network.left_context :: self.network.stride
        ]
        preds = preds[:, :, 1:]
        labels = torch.diff(wrist_angles, dim=2)
        loss = self.loss_fn(preds, labels)
        self.log(f"{stage}_loss", loss, sync_dist=True)
        mae_deg_s = np.rad2deg(loss.item()) * 50
        self.log(f"{stage}_mae_deg_per_sec", mae_deg_s, sync_dist=True)
        return loss


class FingerStateMaskGenerator(torch.nn.Module):
    """
    Generate finger state masks based on press and release event labels.
    """

    def __init__(self, lpad: int = 0, rpad: int = 0) -> None:
        super().__init__()
        self.lpad = lpad
        self.rpad = rpad
        self.INDEX_FINGER = 0
        self.MIDDLE_FINGER = 1

    def forward(self, gesture_labels: torch.Tensor) -> torch.Tensor:
        batch_size, _, time_steps = gesture_labels.shape
        finger_masks = torch.zeros(
            (batch_size, 2, time_steps),
            device=gesture_labels.device,
            dtype=torch.float32,
        )
        for b in range(batch_size):
            self._process_finger(
                gesture_labels[b], finger_masks[b],
                press_channel=GestureType.index_press.value,
                release_channel=GestureType.index_release.value,
                output_channel=self.INDEX_FINGER, time_steps=time_steps,
            )
            self._process_finger(
                gesture_labels[b], finger_masks[b],
                press_channel=GestureType.middle_press.value,
                release_channel=GestureType.middle_release.value,
                output_channel=self.MIDDLE_FINGER, time_steps=time_steps,
            )
        return finger_masks

    def _process_finger(
        self, gesture_labels, finger_masks, press_channel, release_channel,
        output_channel, time_steps,
    ) -> None:
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


class DiscreteGesturesModule(BaseLightningModule):
    """
    PyTorch Lightning module for discrete gesture classification.

    Supports transfer learning via freeze_backbone_epochs:
      - Epochs 0..(freeze_backbone_epochs-1): backbone frozen, only head trains
      - Epochs freeze_backbone_epochs..max: full fine-tuning

    Metric windows (w_start, w_end, rpad) are computed dynamically from
    self.network.stride, so both the original model (stride=10, 200Hz) and
    the transfer model (stride=40, 50Hz) work correctly.
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
        freeze_backbone_epochs: int = 0,
        backbone_lr_ratio: float = 0.02,
    ) -> None:
        super().__init__(network=network, optimizer=optimizer)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        # Dynamic rpad: original model stride=10 → 40ms needs 8 steps, rpad=7
        #               transfer model stride=40 → 40ms needs 2 steps, rpad=1~2
        output_freq = 2000 // network.stride
        rpad = round(0.040 * output_freq)
        self.mask_generator = FingerStateMaskGenerator(lpad=0, rpad=rpad)
        self.val_accuracy = MulticlassAccuracy(num_classes=9)

        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.backbone_lr_ratio = backbone_lr_ratio

    def on_train_epoch_start(self):
        """Freeze/unfreeze encoder based on current epoch."""
        if hasattr(self.network, 'encoder'):
            if self.current_epoch < self.freeze_backbone_epochs:
                self.network.encoder.requires_grad_(False)
                self.network.encoder.eval()
                log.info(f"Epoch {self.current_epoch}: encoder FROZEN")
            elif self.current_epoch == self.freeze_backbone_epochs:
                self.network.encoder.requires_grad_(True)
                self.network.encoder.train()
                log.info(f"Epoch {self.current_epoch}: encoder UNFROZEN")

    def get_metrics(self, phase: str, domain: str | None = None) -> Any:
        return self.val_accuracy

    def collect_metric(
        self, logits, target, phase, domain=None,
    ) -> Any:
        device = logits.device

        # Dynamic windows based on output frequency
        output_freq = 2000 // self.network.stride
        w_start = round(0.050 * output_freq)   # 50ms before event
        w_end = round(0.150 * output_freq)       # 150ms after event

        probs = torch.sigmoid(logits)
        y = target.to(torch.int32)
        y_class = []
        y_hat_class = []

        for batch in range(y.shape[0]):
            y_diff = torch.diff(y[batch], axis=0)
            indices = torch.argwhere(y_diff == 1)
            for index in indices:
                start = max(index[0] - w_start, 0)
                end = min(index[0] + w_end, y.shape[1])
                y_hat = probs[batch, start:end, :]
                flattened_index = y_hat.argmax()
                _, cols = y_hat.shape
                col = flattened_index % cols
                y_hat_class.append(col)
                y_class.append(index[1])

        if len(y_class) > 0:
            y_class = torch.stack(y_class).long().to(device)
            y_hat_class = torch.stack(y_hat_class).long().to(device)
        else:
            y_class = torch.zeros(1, dtype=torch.int64, device=device)
            y_hat_class = torch.zeros(1, dtype=torch.int64, device=device)

        metric_value = self.get_metrics(phase, domain).update(y_hat_class, y_class)
        self.log(f"{phase}_accuracy", self.val_accuracy,
                 on_step=False, on_epoch=True, sync_dist=True)
        return metric_value

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str = "train") -> float:
        emg = batch["emg"]
        targets = batch["targets"]
        targets = targets[:, :, self.network.left_context :: self.network.stride]
        release_mask = self.mask_generator(targets)
        mask = torch.ones_like(targets)
        mask[
            :, [GestureType.index_release.value, GestureType.middle_release.value], :
        ] = release_mask

        preds = self.forward(emg)

        loss = self.loss_fn(preds, targets)
        loss = (loss * mask).sum() / mask.sum()
        self.log(f"{stage}_loss", loss, sync_dist=True)

        if stage == "val":
            self.collect_metric(
                preds.permute(0, 2, 1),
                targets.permute(0, 2, 1),
                phase=stage,
            )

        if stage == "test":
            prompts = batch["prompts"][0]
            times = batch["timestamps"][0]
            preds_prob = nn.Sigmoid()(preds)
            preds_prob = preds_prob.squeeze(0).detach().cpu().numpy()
            times = times[self.network.left_context :: self.network.stride]
            cler = compute_cler(preds_prob, times, prompts)
            self.log("test_cler", cler, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer with optional differential learning rates."""
        if hasattr(self.network, 'encoder'):
            # Differential LR: backbone gets 1/50 of head LR
            backbone_lr = self.hparams.learning_rate * self.backbone_lr_ratio
            backbone_params = list(self.network.encoder.parameters())
            head_params = [
                p for n, p in self.network.named_parameters()
                if not n.startswith('encoder')
            ]
            param_groups = [
                {'params': head_params, 'lr': self.hparams.learning_rate},
                {'params': backbone_params, 'lr': backbone_lr},
            ]
            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate
            )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.hparams.warmup_start_factor,
            end_factor=self.hparams.warmup_end_factor,
            total_iters=self.hparams.warmup_total_epochs,
        )
        multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.hparams.lr_scheduler_milestones,
            gamma=self.hparams.lr_scheduler_factor,
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [warmup_scheduler, multistep_scheduler]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
            "gradient_clip_val": self.hparams.gradient_clip_val,
        }


class HandwritingModule(BaseLightningModule):
    """
    Handwriting module — requires handwriting_utils from emg_nature.
    This is included for completeness but not used in gesture transfer.
    """

    def __init__(
        self, network, optimizer, lr_scheduler, decoder,
    ) -> None:
        super().__init__(network=network, optimizer=optimizer)
        self.lr_scheduler = lr_scheduler
        # Lazy import — these modules come from emg_nature, not copied into emg_transfer
        from generic_neuromotor_interface.handwriting_utils import (
            CharacterErrorRates, charset as _charset,
        )
        self._charset = _charset

        self.ctc_loss = nn.CTCLoss(
            blank=self._charset().null_class, zero_infinity=True,
        )
        self.decoder = decoder
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })
        torch.autograd.set_detect_anomaly(True)

    def _step(self, batch, stage="train"):
        emg = batch["emg"]
        prompts = batch["prompts"]
        emg_lengths = batch["emg_lengths"]
        target_lengths = batch["prompt_lengths"]
        N = len(emg_lengths)
        emissions, slc = self.forward(emg)
        emission_lengths = self.network.compute_time_downsampling(
            emg_lengths=emg_lengths, slc=slc,
        )
        loss = self.ctc_loss(
            log_probs=emissions.movedim(0, 1),
            targets=prompts,
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )
        self.log(f"{stage}_loss", loss, sync_dist=True)
        predictions = self.decoder.decode_batch(
            emissions=emissions.movedim(0, 1).detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )
        metrics = self.metrics[f"{stage}_metrics"]
        prompts_np = prompts.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = prompts_np[i, : target_lengths_np[i]]
            metrics.update(
                prediction=self.decoder._charset.labels_to_str(predictions[i]),
                target=self.decoder._charset.labels_to_str(target),
            )
        return loss

    def on_train_epoch_end(self):
        self._on_epoch_end(stage="train")

    def on_validation_epoch_end(self):
        self._on_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self._on_epoch_end(stage="test")

    def _on_epoch_end(self, stage):
        metrics = self.metrics[f"{stage}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def configure_optimizers(self):
        self.optimizer = self.optimizer(self.parameters())
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[sched(self.optimizer) for sched in self.lr_scheduler["schedules"]],
                    milestones=self.lr_scheduler["milestones"],
                ),
                "interval": self.lr_scheduler["interval"],
            },
        }
