#!/bin/bash
# U34 -- the BENIGN rung of the threat ladder (Step 2).
#
# WHY. Every attack in this project trains on images of the TARGET concept: a deliberate adversary.
# ResAlign (NeurIPS 2025, arXiv:2507.16302) -- the only published method that demonstrates
# resistance -- reports a different, gentler threat: fine-tuning on DreamBench++/DiffusionDB,
# downstream data that does NOT contain the forget concept, where the capability leaks back as a
# SIDE EFFECT of ordinary use. Until this rung exists we cannot place our numbers beside theirs:
# they answer "does erasure survive ordinary downstream use?" and we answer "does it survive a
# deliberate attack?" Different questions.
#
# WHAT IT MEASURES. Fine-tune the erased model on non-target concepts, then ask whether the target
# came back -- having never been trained on.
#
# NOT BUDGET-MATCHED TO THE OTHER RUNGS, ON PURPOSE. 200 steps, not 25: a downstream fine-tune is a
# training run, not a 25-step poke, and matching the attack budget would understate this threat.
# The benign rung must therefore be read as its own column, never interpolated into the capacity
# ladder.
#
# VALIDITY. Cannot be gated on target-prompt rise -- the target was never trained on. A cell is
# valid only if the fine-tune ACTUALLY TRAINED (probe loss fell AND weights moved past the fp32
# floor). A run that changed nothing is INCONCLUSIVE, not evidence of robustness.
#
# Usage: ./run_u34_benign_rung.sh <arm> [<arm> ...]
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn
[ $# -ge 1 ] || { echo "usage: $0 <arm> [<arm> ...]" >&2; exit 1; }
ARMS=("$@")
TAG=$(printf '%s' "${ARMS[0]}" | tr -c 'A-Za-z0-9' '_')
source scripts/gpu_reserve.sh
gpu_reserve_hold "u34_${TAG}" 240 >/dev/null || { echo "U34 ABORT: no reservation (rc=${GPU_RESERVE_RC:-?})" >&2; exit 1; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "U34 ABORT: no CVD" >&2; exit 1; }
echo "U34 reserved GPU $GPU_RESERVED CVD=$CUDA_VISIBLE_DEVICES for ${ARMS[*]}" >&2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
L="logs/u34_benign_${TAG}.log"
source scripts/run_guard.sh
rg_init "U34_${TAG}" "$L"

for ARM in "${ARMS[@]}"; do
  U="adapters/$ARM/unet.pt"
  [ -s "$U" ] || { echo "SKIP $ARM: no checkpoint" >>"$L"; continue; }
  for SD in 0 1 6; do
    K="U34_${ARM}_BENIGN_sd${SD}_s200"
    [ -f "outputs/benign_${K}.json" ] && continue
    echo "=== $K $(date -Is)" >>"$L"
    rg_cell venv_t2i/bin/python scripts/benign_finetune.py --concept church \
      --unet "$U" --finetune-mode full --steps 200 --lr 1e-5 --n-images 32 \
      --seed "$SD" --label "$K" --validate-reload
    rg_require_loadable "outputs/benign_${K}.json" json
  done
  echo "=== ARM DONE $ARM $(date -Is)" >>"$L"
done
rg_finish
echo "U34 DONE ${ARMS[*]} $(date -Is)" >>"$L"
