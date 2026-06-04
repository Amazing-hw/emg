# Method 4: Multi-Task Lightning Module

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from emg_transfer.lightning import BaseLightningModule, FingerStateMaskGenerator
from emg_transfer.constants import GestureType
from emg_transfer.cler import compute_cler
from torchmetrics.classification import MulticlassAccuracy

log = logging.getLogger(__name__)


class MultiTaskGestureModule(BaseLightningModule):
    """
    Multi-task learning: gesture classification + joint angle regression.

    Uses shared encoder with two task heads.
    Trains on mixed batches:
      - emg_nature batches → gesture BCELoss
      - emg2pose batches → joint angle MAE + weak gesture BCE (optional)
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
        joint_loss_weight: float = 0.1,
        weak_gesture_loss_weight: float = 0.05,
    ) -> None:
        super().__init__(network=network, optimizer=optimizer)
        self.gesture_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.joint_loss_fn = torch.nn.L1Loss(reduction="mean")

        output_freq = 2000 // network.stride
        rpad = round(0.040 * output_freq)
        self.mask_generator = FingerStateMaskGenerator(lpad=0, rpad=rpad)
        self.val_accuracy = MulticlassAccuracy(num_classes=9)

        self.joint_loss_weight = joint_loss_weight
        self.weak_gesture_loss_weight = weak_gesture_loss_weight
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.backbone_lr_ratio = backbone_lr_ratio

    def forward(self, emg: torch.Tensor, task: str = "gesture"):
        return self.network(emg, task=task)

    def _gesture_step(self, batch, stage):
        emg = batch["emg"]
        targets = batch["targets"]
        targets_aligned = targets[:, :, self.network.left_context :: self.network.stride]

        # Generate finger state mask
        release_mask = self.mask_generator(targets_aligned)
        mask = torch.ones_like(targets_aligned)
        mask[:, [GestureType.index_release.value, GestureType.middle_release.value], :] = release_mask

        preds = self.forward(emg, task="gesture")
        loss = self.gesture_loss_fn(preds, targets_aligned)
        loss = (loss * mask).sum() / mask.sum()

        self.log(f"{stage}/gesture_loss", loss, sync_dist=True)

        if stage == "val":
            self._compute_accuracy(preds, targets_aligned)
        if stage == "test":
            self._compute_cler(batch, preds)

        return loss

    def _joint_step(self, batch, stage):
        emg = batch["emg"]
        joint_angles = batch["joint_angles"]  # (B, 20, T)
        joint_aligned = joint_angles[:, :, self.network.left_context :: self.network.stride]

        joint_preds = self.forward(emg, task="joint")
        joint_loss = self.joint_loss_fn(joint_preds, joint_aligned)
        self.log(f"{stage}/joint_loss", joint_loss, sync_dist=True)

        total_loss = self.joint_loss_weight * joint_loss

        # Optional weak gesture supervision
        if "weak_targets" in batch:
            weak_targets = batch["weak_targets"]
            weak_confidence = batch.get("weak_confidence", None)
            if weak_targets.dim() == 3:
                weak_targets = weak_targets[:, :, self.network.left_context :: self.network.stride]
                if weak_confidence is not None and weak_confidence.dim() == 3:
                    weak_confidence = weak_confidence[:, :, self.network.left_context :: self.network.stride]

            gesture_preds = self.forward(emg, task="gesture")
            w_loss = self.gesture_loss_fn(gesture_preds, weak_targets)

            if weak_confidence is not None:
                w_loss = (w_loss * weak_confidence).sum() / (weak_confidence.sum() + 1e-8)
            else:
                w_loss = w_loss.mean()

            total_loss = total_loss + self.weak_gesture_loss_weight * w_loss
            self.log(f"{stage}/weak_gesture_loss", w_loss, sync_dist=True)

        return total_loss

    def _step(self, batch, stage: str = "train") -> float:
        # Check if this is a gesture batch or joint angle batch
        if isinstance(batch, tuple):
            batch, task_type = batch
        elif "task" in batch:
            task_type = batch["task"]
        elif "joint_angles" in batch:
            task_type = "joint"
        else:
            task_type = "gesture"

        if task_type == "gesture":
            loss = self._gesture_step(batch, stage)
        elif task_type == "joint":
            loss = self._joint_step(batch, stage)
        else:
            # Mixed: compute both if data has both labels
            loss = self._gesture_step(batch, stage)
            if "joint_angles" in batch:
                loss = loss + self._joint_step(batch, stage)

        return loss

    def _compute_accuracy(self, preds, targets):
        output_freq = 2000 // self.network.stride
        w_start = round(0.050 * output_freq)
        w_end = round(0.150 * output_freq)
        probs = torch.sigmoid(preds)
        y = targets.to(torch.int32)
        y_class, y_hat_class = [], []
        device = preds.device
        for b in range(y.shape[0]):
            y_diff = torch.diff(y[b], axis=0)
            indices = torch.argwhere(y_diff == 1)
            for index in indices:
                start_idx = max(index[0] - w_start, 0)
                end_idx = min(index[0] + w_end, y.shape[1])
                y_hat = probs[b, start_idx:end_idx, :]
                col = y_hat.argmax() % y_hat.shape[1]
                y_hat_class.append(col)
                y_class.append(index[1])
        if len(y_class) > 0:
            y_class = torch.stack(y_class).long().to(device)
            y_hat_class = torch.stack(y_hat_class).long().to(device)
            self.val_accuracy.update(y_hat_class, y_class)
            self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, sync_dist=True)

    def _compute_cler(self, batch, preds):
        prompts = batch["prompts"][0]
        times = batch["timestamps"][0]
        preds_prob = nn.Sigmoid()(preds).squeeze(0).detach().cpu().numpy()
        times = times[self.network.left_context :: self.network.stride]
        if len(times) > preds_prob.shape[1]:
            times = times[:preds_prob.shape[1]]
        elif len(times) < preds_prob.shape[1]:
            preds_prob = preds_prob[:, :len(times)]
        cler = compute_cler(preds_prob, times, prompts)
        self.log("test_cler", cler, on_step=False, on_epoch=True, sync_dist=True)

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

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=self.hparams.warmup_start_factor,
            end_factor=self.hparams.warmup_end_factor,
            total_iters=self.hparams.warmup_total_epochs,
        )
        multistep = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hparams.lr_scheduler_milestones,
            gamma=self.hparams.lr_scheduler_factor,
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, multistep])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
            "gradient_clip_val": self.hparams.gradient_clip_val,
        }
