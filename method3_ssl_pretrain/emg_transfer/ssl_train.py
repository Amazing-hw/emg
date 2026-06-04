# Method 3: SSL Pre-training Entry Point

"""
Usage:
    # Wav2Vec 2.0 pre-training:
    python -m emg_transfer.ssl_train method=wav2vec2

    # HuBERT pre-training (iteration 1):
    python -m emg_transfer.ssl_train method=hubert

    # Fine-tuning on gesture data:
    python -m emg_transfer.train  (uses standard train.py with ssl_finetune config)
"""

import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

log = logging.getLogger(__name__)


def train_ssl(config: DictConfig):
    """Main SSL pre-training entry point."""
    log.info(f"Config:\n{OmegaConf.to_yaml(config)}")
    pl.seed_everything(config.seed, workers=True)

    # Create dataloaders
    from emg_transfer.ssl_data import create_ssl_dataloader

    train_loader = create_ssl_dataloader(
        data_dir=config.data_dir,
        split="train",
        batch_size=config.batch_size,
        window_length=config.window_length,
        stride=config.stride,
        max_files=config.max_files,
        max_windows_per_file=config.max_windows_per_file,
        num_workers=config.num_workers,
    )

    val_loader = create_ssl_dataloader(
        data_dir=config.data_dir,
        split="val",
        batch_size=config.batch_size,
        window_length=config.window_length,
        stride=config.stride,
        max_files=config.max_files,
        max_windows_per_file=config.max_windows_per_file // 4,
        num_workers=config.num_workers,
    )

    log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    if config.method == "wav2vec2":
        from emg_transfer.ssl_networks import Wav2Vec2Model
        from emg_transfer.ssl_pretrain import Wav2Vec2PretrainModule

        model = Wav2Vec2Model(
            feature_dim=config.feature_dim,
            context_dim=config.context_dim,
            num_conformer_layers=config.num_conformer_layers,
            num_heads=config.num_heads,
            conv_kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
            num_quantizer_groups=config.num_quantizer_groups,
            num_quantizer_vars=config.num_quantizer_vars,
            mask_prob=config.mask_prob,
            mask_length=config.mask_length,
        )
        module = Wav2Vec2PretrainModule(
            model=model,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            total_steps=len(train_loader) * config.max_epochs,
            contrastive_temp=config.contrastive_temp,
            diversity_weight=config.diversity_weight,
        )

    elif config.method == "hubert":
        from emg_transfer.ssl_networks import HubertModel
        from emg_transfer.ssl_pretrain import HubertPretrainModule

        model = HubertModel(
            feature_dim=config.feature_dim,
            context_dim=config.context_dim,
            num_conformer_layers=config.num_conformer_layers,
            num_heads=config.num_heads,
            conv_kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
            num_clusters=config.num_clusters,
            mask_prob=config.mask_prob,
            mask_length=config.mask_length,
        )
        module = HubertPretrainModule(
            model=model,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            total_steps=len(train_loader) * config.max_epochs,
        )

    else:
        raise ValueError(f"Unknown method: {config.method}")

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f"{config.method}-{{epoch:02d}}-{{pretrain_loss:.4f}}",
        monitor="pretrain_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor()

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=10,
    )

    trainer.fit(module, train_loader, val_loader)
    log.info(f"Pre-training completed! Best model: {checkpoint_callback.best_model_path}")

    return checkpoint_callback.best_model_path


@hydra.main(config_path="../config/ssl", config_name="wav2vec2_pretrain", version_base="1.1")
def cli(config: DictConfig):
    train_ssl(config)


if __name__ == "__main__":
    cli()
