#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
ROOT=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local'])")
TE="$ROOT/parachute_unlearned"
# H49: GENERALISATION. Every T2I robustness number in this phase rests on ONE concept (church).
# This repeats the whole pipeline on PARACHUTE with official AdvUnlearn weights:
#   1. verify the official erasure works on this concept
#   2. attack it (does the H46 reversal replicate on a second concept?)
#   3. the composite at alpha=0.3 (does H47's retain 0.867 / H 0.533 replicate?)
# Chosen over re-implementing Meta-Unlearning/IMMA/GIFT because those are bi-level defences with
# no released weights, and this session has direct evidence that my re-implementations of harder
# methods fail (adversarial erasure: 3 attempts, never erased) -- and a WEAK baseline is worse
# than none, since it flatters the treatment for the wrong reason.
[ -f outputs/gen_advunlearn_parachute.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py \
  --label advunlearn_parachute --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 \
  > outputs/log_advunlearn_parachute.txt 2>&1
for sd in 0 1 3; do
  lab="H49_advunlearn_parachute_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --text-encoder "$TE" --concept parachute \
    --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
arm=h49_para_a0.3
o=adapters/$arm
[ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher-te "$TE" --concept parachute --alpha 0.3 --beta 0.1 --protect retain_fisher \
  --steps 6000 --log-every 3000 --n-images 300 --forget-frac 0.20 --seed 0 --label $arm --out $o \
  > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
for sd in 0 1 3; do
  lab="${arm}_parachute_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
    --concept parachute --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
rm -f $o/unet.pt
echo "[h49] done"
