#!/bin/bash
# Method 4: Multi-task Learning
source E:/software/anaconda/anaconda/etc/profile.d/conda.sh
conda activate neuromotor
cd D:/emg/method4_multitask

echo "=== Method 4: Multi-task Gesture + Joint Angle ==="
echo "Started: $(date)"

python -m emg_transfer.train --config-name multitask 2>&1 | tee training_output.log
echo "Method 4 done: $(date)"
