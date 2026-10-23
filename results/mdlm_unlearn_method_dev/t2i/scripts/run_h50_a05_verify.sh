#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/church_unlearned')")
# H50: H48 just destroyed the alpha=0.3 composite's robustness number -- under lr 5e-5 and 1e-4 it
# is FULLY reversed (H -> 1.000, 3/3 gate PASS), where lr 1e-5 with 3 seeds had read 0.533.
# The alpha=0.5 arm's 0.267 was measured under exactly the same too-weak conditions, so it must
# get the same treatment before it is believed. Rebuilt with the determinism fix (the Fisher and
# timestep draws are now seeded; without that the same config gave retain 0.867 vs 0.783).
arm=h50_adv_a0.5
o=adapters/$arm
[ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher-te "$TE" --concept church --alpha 0.5 --beta 0.1 --protect retain_fisher \
  --steps 6000 --log-every 3000 --n-images 300 --forget-frac 0.20 --seed 0 --label $arm --out $o \
  > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
for spec in "0|5e-5" "1|5e-5" "0|1e-4" "1|1e-4" "6|1e-5"; do
  IFS='|' read -r sd lr <<< "$spec"
  lab="${arm}_church_sd${sd}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
    --concept church --steps 300 --lr $lr --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_attacks_progress.txt
done
echo "[h50] done"
