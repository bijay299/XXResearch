#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/parachute_unlearned
ARM=h69v2_fragnorm_a0.3_parachute
LOG=logs/h69_fragnorm_parachute.log
venv_t2i/bin/python scripts/undo_t2i.py --teacher-te "$TE" --concept parachute \
  --alpha 0.3 --beta 0.07245 --protect frag --fisher-batches 16 --steps 6000 --lr 1e-5 \
  --n-images 300 --forget-frac 0.2 --seed 0 --log-every 3000 \
  --label "$ARM" --out "adapters/$ARM" >>$LOG 2>&1
venv_t2i/bin/python scripts/gen_eval.py --label "$ARM" --unet "adapters/$ARM/unet.pt" \
  --target parachute >>$LOG 2>&1
echo "H69 PARA DONE $(date -Is)" >>$LOG
