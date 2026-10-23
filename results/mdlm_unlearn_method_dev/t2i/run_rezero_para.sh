#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/bijaypandey/hf_home
LOG=logs/rezero_para.log
L=DIAG_H66_frp_parachute_sd1_lr1e-5_keepckpt
venv_t2i/bin/python scripts/attack_relearn.py --concept parachute \
  --unet adapters/frp_parachute_r5_l05/unet.pt --steps 300 --lr 1e-5 --seed 1 --n-images 16 \
  --keep-checkpoint --label "$L" >>$LOG 2>&1
venv_t2i/bin/python scripts/frp_mask_refill.py --arm frp_parachute_r5_l05 \
  --attacked "attacked/$L/unet.pt" --label parachute_r5 >>$LOG 2>&1
venv_t2i/bin/python scripts/rezero_test.py --arm frp_parachute_r5_l05 \
  --attacked "attacked/$L/unet.pt" --label parachute_r5 >>$LOG 2>&1
venv_t2i/bin/python scripts/gen_eval.py --label REZERO_frp_parachute_r5 \
  --unet adapters/_rezero_parachute_r5/unet.pt --target parachute >>$LOG 2>&1
echo "REZERO PARA DONE $(date -Is)" >>$LOG
