#!/bin/bash
# Method 3: SSL Pretrain + Finetune
source E:/software/anaconda/anaconda/etc/profile.d/conda.sh
conda activate neuromotor
cd D:/emg/method3_ssl_pretrain

echo "=== Method 3: HuBERT SSL Pretrain ==="
echo "Started: $(date)"

# Step 1: HuBERT pre-training
python -m emg_transfer.ssl_train --config-name hubert_pretrain 2>&1 | tee ssl_pretrain_output.log
echo "SSL pre-training done: $(date)"

# Find best checkpoint
BEST_CKPT=$(find . -path "*/ssl_checkpoints/*.ckpt" -type f 2>/dev/null | head -1)
echo "Best checkpoint: $BEST_CKPT"

# Step 2: Fine-tuning
echo "=== Method 3: Fine-tuning ==="
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/discrete_gestures_transfer.yaml')
cfg.pretrained_encoder_ckpt = '$BEST_CKPT'
cfg.seed = 0
OmegaConf.save(cfg, 'config/discrete_gestures_transfer.yaml')
"
python -m emg_transfer.train 2>&1 | tee ssl_finetune_output.log
echo "Method 3 done: $(date)"
