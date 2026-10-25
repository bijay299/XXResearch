#!/bin/bash
# FINDING56 S5a recorded exactly this as the missing measurement, and as NOT DONE:
#
#   "Distinguishing them needs a TI attack on a non-meta arm at MATCHED UTILITY -- the ablation sits
#    at retain 0.9417, far above lambda=0.5's 0.6333, so the present control is not utility-matched
#    and part of the asymmetry could be a utility effect rather than a trap effect."
#
# GIFT beta=1.3 sits at retain8 0.6083, against the Meta-Unlearning lambda=0.5 arm's 0.6333 -- a gap
# of 0.025, far closer than the 0.308 gap to the ablation. And TI is comfortably VALID on GIFT
# beta=1.0 (2/2, T 0.333 -> 1.000). So:
#   TI valid on GIFT b1.3 too  -> utility is NOT the explanation, and the meta trigger genuinely
#                                 obstructs embedding-space optimisation (FINDING56 S5a reading 1);
#   TI gate-fails on b1.3 too  -> low utility IS sufficient to break TI, and the meta-arm asymmetry
#                                 is explained away (reading 2). Either way FINDING56 S5a gets its
#                                 answer instead of staying open.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
run() {  # $1 arm  $2 steps  $3 lr  $4 seed  $5 gpu
  K="H81TM_$1_s$2_lr$3_sd$4"
  mkdir ".labels/$K" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$5
  [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept church \
    --unet "adapters/$1/unet.pt" --steps "$2" --lr "$3" --seed "$4" --label "$K" \
    >>logs/ti_matched.log 2>&1
}
( run gift_church_b1.3 2000 1e-2 0 0; run gift_church_b1.3 3000 2e-2 6 0
  echo "TIM A DONE $(date -Is)" >>logs/ti_matched.log ) &
( run gift_church_b1.3 3000 1e-2 1 1; run gift_church_b1.35 2000 1e-2 0 1
  echo "TIM B DONE $(date -Is)" >>logs/ti_matched.log ) &
wait
