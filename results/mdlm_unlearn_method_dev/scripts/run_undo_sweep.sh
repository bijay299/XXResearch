#!/bin/bash
# UNDO alpha sweep. Measured 8s/step at batch 1, 32.9 GiB/GPU, 106 GB CPU RAM.
# 600 steps ~ 80 min/arm. alpha controls the damage/robustness-vs-compute tradeoff in UNDO.
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
export OMP_NUM_THREADS=60
# `--num_gpus 2` makes the deepspeed launcher OVERWRITE CUDA_VISIBLE_DEVICES with 0,1, which
# silently moved the job onto the two low-index GPUs that other users' jobs take by default.
# `--include localhost:2,3` is the only form that actually pins the physical devices.
for al in 0.3 0.6; do
  lab="undo_a${al/./p}"
  [ -f "adapters/$lab/report.json" ] && { echo "[sweep] skip $lab"; continue; }
  echo "[sweep] alpha=$al -> $lab"
  /data/bijaypandey/llada_smoke/venv_llada/bin/deepspeed --include localhost:2,3 \
    --master_port 29577 \
    scripts/train_undo.py --alpha "$al" --beta 0.1 --steps 300 \
    --probe-every 50 --save-every 300 --batch 1 \
    --out "/data/bijaypandey/mdlm_unlearn/adapters/$lab" \
    > "outputs/log_$lab.txt" 2>&1
done
echo "[sweep] done"
