# Method 3: Self-Supervised Pre-training Modules

import logging
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn

from sklearn.cluster import MiniBatchKMeans

log = logging.getLogger(__name__)


# =============================================================================
# Wav2Vec 2.0 Pre-training Module
# =============================================================================

class Wav2Vec2PretrainModule(pl.LightningModule):
    """
    wav2vec 2.0 pre-training for EMG.

    Loss = Contrastive loss + Diversity loss

    Contrastive: distinguish the true quantized latent from distractors
    Diversity: encourage uniform codebook usage
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 5e-4,
        warmup_steps: int = 10000,
        total_steps: int = 100000,
        contrastive_temp: float = 0.1,
        diversity_weight: float = 0.1,
        num_negatives: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.contrastive_temp = contrastive_temp
        self.diversity_weight = diversity_weight
        self.num_negatives = num_negatives

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        emg = batch["emg"]  # (B, 16, T)

        c, q, mask, perplexity = self.model(emg)

        # Only compute loss on masked positions
        masked_c = c[mask]  # (num_masked, feature_dim)
        masked_q = q[mask]  # (num_masked, feature_dim)

        if masked_c.shape[0] == 0:
            return torch.tensor(0.0, device=emg.device, requires_grad=True)

        # Sample negatives from unmasked positions in the same batch
        # Use all unmasked positions as negatives
        unmasked_q = q[~mask]  # (num_unmasked, feature_dim)
        if unmasked_q.shape[0] < self.num_negatives:
            # Pad with random if not enough negatives
            extra = torch.randn(
                self.num_negatives - unmasked_q.shape[0], q.shape[-1],
                device=q.device, dtype=q.dtype,
            )
            negatives = torch.cat([unmasked_q, extra], dim=0)[:self.num_negatives]
        else:
            idx = torch.randperm(unmasked_q.shape[0])[:self.num_negatives]
            negatives = unmasked_q[idx]  # (num_neg, feature_dim)

        # Contrastive loss
        # For each masked position: positive = q_t, negatives = random unmasked q
        pos_sim = F.cosine_similarity(masked_c, masked_q, dim=-1) / self.contrastive_temp  # (M,)
        neg_sim = torch.matmul(masked_c, negatives.T) / self.contrastive_temp  # (M, K)

        # InfoNCE-style loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (M, 1+K)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        # Diversity loss
        diversity_loss = (1.0 - perplexity / self.model.quantizer.num_vars)

        total_loss = contrastive_loss + self.diversity_weight * diversity_loss

        self.log("pretrain_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("contrastive_loss", contrastive_loss, sync_dist=True)
        self.log("diversity_loss", diversity_loss, sync_dist=True)
        self.log("perplexity", perplexity, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01,
        )
        # Linear warmup + cosine decay
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / max(self.hparams.warmup_steps, 1)
            else:
                progress = (step - self.hparams.warmup_steps) / max(
                    self.hparams.total_steps - self.hparams.warmup_steps, 1
                )
                return max(0.5 * (1 + np.cos(np.pi * progress)), 0.01)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# =============================================================================
# HuBERT Pre-training Module
# =============================================================================

class HubertPretrainModule(pl.LightningModule):
    """
    HuBERT-style iterative masked prediction pre-training.

    Phase 1: Extract features → K-means clustering → cluster IDs as targets
    Phase 2: Mask input → predict cluster IDs at masked positions
    Repeat 2-3 iterations with improved features.

    The K-means model is trained externally and cluster labels are pre-computed.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 5e-4,
        warmup_steps: int = 10000,
        total_steps: int = 100000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        emg = batch["emg"]  # (B, 16, T)
        cluster_targets = batch.get("cluster_ids", None)

        if cluster_targets is None:
            # First iteration: use the model to get features and cluster on-the-fly
            logits, mask, _ = self.model(emg)
            # Use argmax as pseudo-target
            cluster_targets = logits.argmax(dim=-1).detach()

            # Mask the targets that aren't in masked positions
            valid_targets = cluster_targets.clone()
            valid_targets[~mask] = -100  # ignore non-masked positions
            cluster_targets = valid_targets
        else:
            logits, mask, _ = self.model(emg)
            # Apply mask to targets
            valid_targets = cluster_targets.clone()
            valid_targets[~mask] = -100
            cluster_targets = valid_targets

        # Cross-entropy on masked positions
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            cluster_targets.reshape(-1),
            ignore_index=-100,
        )

        self.log("pretrain_loss", loss, prog_bar=True, sync_dist=True)

        # Accuracy on masked positions
        if mask.sum() > 0:
            pred = logits[mask].argmax(dim=-1)
            target = cluster_targets[mask]
            valid = target != -100
            if valid.sum() > 0:
                acc = (pred[valid] == target[valid]).float().mean()
                self.log("masked_accuracy", acc, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01,
        )
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / max(self.hparams.warmup_steps, 1)
            else:
                progress = (step - self.hparams.warmup_steps) / max(
                    self.hparams.total_steps - self.hparams.warmup_steps, 1
                )
                return max(0.5 * (1 + np.cos(np.pi * progress)), 0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# =============================================================================
# K-means clustering utility for HuBERT
# =============================================================================

def run_kmeans_clustering(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_clusters: int = 500,
    n_samples: int = 50000,
    device: str = "cuda",
) -> MiniBatchKMeans:
    """
    Extract features from the model and run K-means clustering.

    Args:
        model: HubertModel (in eval mode)
        dataloader: DataLoader for unlabeled EMG data
        num_clusters: Number of K-means clusters
        n_samples: Max number of feature vectors to use for clustering
        device: Device to run on
    Returns:
        kmeans: Fitted MiniBatchKMeans model
    """
    model.eval()
    features_list = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            emg = batch["emg"].to(device)
            _, _, feats = model(emg, return_features=True)
            # feats: (B, T, context_dim)
            feats = feats.reshape(-1, feats.shape[-1]).cpu().numpy()
            features_list.append(feats)
            total_samples += feats.shape[0]
            if total_samples >= n_samples:
                break

    features = np.concatenate(features_list, axis=0)[:n_samples]

    log.info(f"Running K-means clustering on {features.shape[0]} samples, {num_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, batch_size=4096, max_iter=100, random_state=42,
        n_init=3,
    )
    kmeans.fit(features)
    log.info(f"K-means completed. Inertia: {kmeans.inertia_:.2f}")

    return kmeans


def assign_cluster_labels(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    kmeans: MiniBatchKMeans,
    device: str = "cuda",
) -> dict:
    """
    Assign cluster IDs to all data samples using the fitted K-means model.
    Stores results that can be loaded during HuBERT training.

    Returns:
        Mapping from sample index to cluster ID tensor
    """
    model.eval()
    all_cluster_ids = []

    with torch.no_grad():
        for batch in dataloader:
            emg = batch["emg"].to(device)
            _, _, feats = model(emg, return_features=True)
            B, T, D = feats.shape
            feats_flat = feats.reshape(-1, D).cpu().numpy()
            cluster_ids = kmeans.predict(feats_flat).reshape(B, T)
            all_cluster_ids.append(torch.from_numpy(cluster_ids))

    return {"cluster_ids": all_cluster_ids}
