#!/bin/bash
# ANOMALY: TI FAILED THE GATE on the meta arm, twice, with T actually FALLING (0.208->0.083,
# 0.083->0.000) -- while full-UNet relearning reversed the same arm 6/6 to 1.000. Per FINDING20 a
# gate failure is NOT resistance, so this must be pushed: either TI becomes valid and we learn
# whether the meta trigger blocks embedding-space attacks specifically, or it cannot be made valid
# and that is reported as an adversary we could not validate on this arm.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for cfg in "2000 1e-2 0" "3000 1e-2 1" "3000 2e-2 6"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3
    L=H80T_ti_meta_lam05_s${ST}_lr${LR}_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept church \
      --unet adapters/h80_meta_church_lam0.5/unet.pt --steps $ST --lr $LR --seed $SD --label "$L" \
      >>logs/h80ti_a.log 2>&1
  done
  echo "H80TI A DONE $(date -Is)" >>logs/h80ti_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  # CONTROL: same TI configs on the no-meta ABLATION. If TI gate-fails there too, the failure is
  # about the distillation wrapper or the arm, not the meta trigger.
  for cfg in "2000 1e-2 0" "3000 1e-2 1"; do
    set -- $cfg; ST=$1; LR=$2; SD=$3
    L=H80T_ti_abl_lam00_s${ST}_lr${LR}_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept church \
      --unet adapters/h80_meta_church_lam0.0/unet.pt --steps $ST --lr $LR --seed $SD --label "$L" \
      >>logs/h80ti_b.log 2>&1
  done
  echo "H80TI B DONE $(date -Is)" >>logs/h80ti_b.log ) &
wait
