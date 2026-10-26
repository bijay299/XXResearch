#!/bin/bash
# lam=1.0 and 2.0 fail K2 (retain8 0.1167) -- the meta objective destroyed the model AS IS, so no
# robustness number is admissible for them. lam=0.5 passes K2 at 0.6333 (still below the 0.85
# mandate floor) and lam=0.0 is the no-meta ablation. Attack both: the ablation is the control that
# says whether any difference is the meta term or the distillation.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for spec in "0 1e-5" "1 1e-5" "6 1e-5" "0 5e-5" "1 5e-5" "0 1e-4"; do
    set -- $spec; SD=$1; LR=$2
    L=H80A_meta_lam05_sd${SD}_lr${LR}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/h80_meta_church_lam0.5/unet.pt --steps 300 --lr "$LR" --seed "$SD" \
      --n-images 16 --label "$L" >>logs/h80atk_a.log 2>&1
  done
  echo "H80ATK A DONE $(date -Is)" >>logs/h80atk_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  for spec in "0 1e-5" "1 1e-5" "6 1e-5"; do
    set -- $spec; SD=$1; LR=$2
    L=H80A_abl_lam00_sd${SD}_lr${LR}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet adapters/h80_meta_church_lam0.0/unet.pt --steps 300 --lr "$LR" --seed "$SD" \
      --n-images 16 --label "$L" >>logs/h80atk_b.log 2>&1
  done
  for SD in 0 1; do
    L=H80A_ti_meta_lam05_sd${SD}
    [ -f "outputs/atk_${L}.json" ] && continue
    venv_t2i/bin/python scripts/attack_ti.py --concept church \
      --unet adapters/h80_meta_church_lam0.5/unet.pt --steps 1000 --lr 5e-3 --seed $SD \
      --label "$L" >>logs/h80atk_b.log 2>&1
  done
  echo "H80ATK B DONE $(date -Is)" >>logs/h80atk_b.log ) &
wait
