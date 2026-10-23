#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1
TE=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local']+'/church_unlearned')")
# H56: does the frozen-probe loss-removal statistic discriminate ANYTHING in T2I?
# H55 found the attacker removes only 6.5-7.1% (official arm) and 9.3-9.9% (composite) of
# available loss while FULLY restoring the concept. If a genuinely BROKEN arm -- one the gate
# consistently refuses, retain ~0.05 -- shows a similar figure, the statistic is uninformative
# here and FINDING32's trainability reasoning loses its instrument entirely. If it shows ~0%,
# the statistic discriminates BROKENNESS but still not ROBUSTNESS, which is a narrower but real
# conclusion. Either way this is the measurement that settles what the statistic can support.
arm=h56_broken_a0.7
o=adapters/$arm
[ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher-te "$TE" --concept church --alpha 0.7 --beta 0.1 --protect retain_fisher \
  --steps 6000 --log-every 3000 --n-images 300 --forget-frac 0.20 --seed 0 --label $arm --out $o \
  > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
for sd in 0 1; do
  lab="H56_broken_church_sd${sd}_lr1e-5"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
    --concept church --steps 300 --lr 1e-5 --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] (frozen|${lab})" outputs/log_atk_$lab.txt >> outputs/log_h55_probe.txt
done
rm -f $o/unet.pt
echo "[h56] done"
