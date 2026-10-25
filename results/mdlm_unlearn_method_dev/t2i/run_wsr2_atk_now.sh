#!/bin/bash
# The bounded retry cleared K1 and one arm is genuinely interesting:
#   wsr2_church_lr1    target 0.1333  retain8 0.9417  <- identical to the FRP teacher: anchor is a no-op
#   wsr2_church_lr10   target 0.0667  retain8 0.9000  <- DEEPER erasure than the teacher, retain ABOVE
#                                                        the mandate's 0.85 floor
# The mandate's bar is retain8 >= 0.85 AND max-over-valid H <= 0.40. lr10 clears the retain half with
# better erasure than FRP, so the attack result alone decides it. Attacking now on the free GPU
# rather than waiting for the lam_reg=100 build -- this is the decisive measurement.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
mkdir -p .labels logs
for A in wsr2_church_lr10 wsr2_church_lr1; do
  for SD in 0 1 6; do
    K="H83B_${A}_sd${SD}_lr1e-5"
    mkdir ".labels/$K" 2>/dev/null || continue
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --label "$K" >>logs/wsr2_atk.log 2>&1
  done
done
echo "WSR2 ATK-NOW DONE $(date -Is)" >>logs/wsr2.log
