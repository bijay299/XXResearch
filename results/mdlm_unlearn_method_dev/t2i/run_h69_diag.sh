#!/bin/bash
# GPU 0: PREREG_H66 S5a mask-refill diagnostic -- ONE re-run of the highest-recovery FRP attack
#        with --keep-checkpoint, labelled DIAG so it is never pooled into the frozen grid.
# GPU 1: PREREG_H69 norm-matched frag (beta=0.07245 so realised norm matches retain_fisher's).
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
mkdir -p logs attacked

( export CUDA_VISIBLE_DEVICES=0
  L=DIAG_H66_frp_church_sd0_lr1e-5_keepckpt
  venv_t2i/bin/python scripts/attack_relearn.py --concept church \
    --unet adapters/frp_church_r3_l05/unet.pt --steps 300 --lr 1e-5 --seed 0 --n-images 16 \
    --keep-checkpoint --label "$L" >>logs/h66_diag_refill.log 2>&1
  venv_t2i/bin/python scripts/frp_mask_refill.py --arm frp_church_r3_l05 \
    --attacked "attacked/$L/unet.pt" --label church_r3 >>logs/h66_diag_refill.log 2>&1
  echo "DIAG DONE $(date -Is)" >>logs/h66_diag_refill.log ) &

( export CUDA_VISIBLE_DEVICES=1
  TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
  ARM=h69v2_fragnorm_a0.3_church
  venv_t2i/bin/python scripts/undo_t2i.py --teacher-te "$TE" --concept church \
    --alpha 0.3 --beta 0.07245 --protect frag --fisher-batches 16 --steps 6000 --lr 1e-5 \
    --n-images 300 --forget-frac 0.2 --seed 0 --log-every 3000 \
    --label "$ARM" --out "adapters/$ARM" >>logs/h69_fragnorm_church.log 2>&1
  venv_t2i/bin/python scripts/gen_eval.py --label "$ARM" --unet "adapters/$ARM/unet.pt" \
    --target church >>logs/h69_fragnorm_church.log 2>&1
  echo "H69 DONE $(date -Is)" >>logs/h69_fragnorm_church.log ) &
wait
echo "BOTH DONE $(date -Is)" >>logs/h69_diag_chain.log
