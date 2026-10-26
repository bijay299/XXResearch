#!/bin/bash
# H80: Meta-Unlearning port. lam_meta sweep on the phase's best erased arm.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
( export CUDA_VISIBLE_DEVICES=0
  for LAM in 1.0 2.0; do
    ARM=h80_meta_church_lam${LAM}
    [ -f "adapters/$ARM/unet.pt" ] || venv_t2i/bin/python scripts/meta_unlearn.py \
      --teacher adapters/frp_church_r3_l05/unet.pt --concept church --steps 1500 --lr 1e-5 \
      --inner-alpha 1e-5 --lam-meta $LAM --n-images 64 --seed 0 --label "$ARM" \
      --out "adapters/$ARM" >>logs/h80_a.log 2>&1
    [ -f "outputs/gen_${ARM}.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$ARM" \
      --unet "adapters/$ARM/unet.pt" --target church >>logs/h80_a.log 2>&1
  done
  echo "H80 A DONE $(date -Is)" >>logs/h80_a.log ) &
( export CUDA_VISIBLE_DEVICES=1
  ARM=h80_meta_church_lam0.5
  [ -f "adapters/$ARM/unet.pt" ] || venv_t2i/bin/python scripts/meta_unlearn.py \
    --teacher adapters/frp_church_r3_l05/unet.pt --concept church --steps 1500 --lr 1e-5 \
    --inner-alpha 1e-5 --lam-meta 0.5 --n-images 64 --seed 0 --label "$ARM" \
    --out "adapters/$ARM" >>logs/h80_b.log 2>&1
  [ -f "outputs/gen_${ARM}.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$ARM" \
    --unet "adapters/$ARM/unet.pt" --target church >>logs/h80_b.log 2>&1
  # ABLATION: lam_meta = 0 is the meta term switched OFF -- distillation only. If the lam>0 arms
  # differ from this, the difference is the meta objective and not the distillation.
  ARM=h80_meta_church_lam0.0
  [ -f "adapters/$ARM/unet.pt" ] || venv_t2i/bin/python scripts/meta_unlearn.py \
    --teacher adapters/frp_church_r3_l05/unet.pt --concept church --steps 1500 --lr 1e-5 \
    --inner-alpha 1e-5 --lam-meta 0.0 --n-images 64 --seed 0 --label "$ARM" \
    --out "adapters/$ARM" >>logs/h80_b.log 2>&1
  [ -f "outputs/gen_${ARM}.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$ARM" \
    --unet "adapters/$ARM/unet.pt" --target church >>logs/h80_b.log 2>&1
  echo "H80 B DONE $(date -Is)" >>logs/h80_b.log ) &
wait
