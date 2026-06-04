#!/bin/bash
# Sequential training: Method 2 → Method 3 (SSL pretrain + finetune) → Method 4

source E:/software/anaconda/anaconda/etc/profile.d/conda.sh
conda activate neuromotor

set -e  # Exit on error

echo "============================================"
echo "Starting sequential training pipeline"
echo "Started at: $(date)"
echo "============================================"

# --- Method 2: CTC ---
echo ""
echo "=== Method 2: CTC ==="
echo "Started at: $(date)"

cd D:/emg/method2_ctc_cif
if [ ! -f "training_output.log" ] || ! grep -q "trainer.fit completed" training_output.log 2>/dev/null; then
    echo "Running Method 2 (CTC)..."
    python -m emg_transfer.train 2>&1 | tee training_output.log
    echo "Method 2 completed at: $(date)"
else
    echo "Method 2 already completed, skipping."
fi

# --- Method 3: SSL Pretrain + Finetune ---
echo ""
echo "=== Method 3: SSL Pretrain + Finetune ==="
echo "Started at: $(date)"

cd D:/emg/method3_ssl_pretrain

# Step 1: HuBERT pre-training
echo "--- Method 3 Step 1: HuBERT pre-training ---"
python -m emg_transfer.ssl_train --config-name hubert_pretrain 2>&1 | tee ssl_pretrain_output.log
echo "HuBERT pre-training completed at: $(date)"

# Step 2: Find best checkpoint
BEST_CKPT=$(ls -t ssl_checkpoints/hubert-*.ckpt 2>/dev/null | head -1)
if [ -z "$BEST_CKPT" ]; then
    # Fallback: find last.ckpt
    BEST_CKPT=$(find ssl_checkpoints -name "last.ckpt" 2>/dev/null | head -1)
fi
echo "Best SSL checkpoint: $BEST_CKPT"

# Step 3: Fine-tuning on gesture data
echo "--- Method 3 Step 2: Fine-tuning ---"
# Update config with pre-trained checkpoint path
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/discrete_gestures_transfer.yaml')
cfg.pretrained_encoder_ckpt = '$BEST_CKPT'
OmegaConf.save(cfg, 'config/discrete_gestures_transfer.yaml')
"
python -m emg_transfer.train 2>&1 | tee ssl_finetune_output.log
echo "Method 3 fine-tuning completed at: $(date)"

# --- Method 4: Multi-task ---
echo ""
echo "=== Method 4: Multi-task ==="
echo "Started at: $(date)"

cd D:/emg/method4_multitask
python -m emg_transfer.train --config-name multitask 2>&1 | tee training_output.log
echo "Method 4 completed at: $(date)"

echo ""
echo "============================================"
echo "All methods completed!"
echo "Finished at: $(date)"
echo "============================================"
