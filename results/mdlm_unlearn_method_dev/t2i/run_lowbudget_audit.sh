#!/bin/bash
# Closes an outstanding audit gap: ERRATUM_r2 S6 lists "re-measure the voided low-budget
# attacker-cost curve (2/50 steps), ~0.5 GPU-h, NOT STARTED". The original curve was voided by the
# text-encoder evaluation bug. FINDING49 covered low budgets on the FRP arm only; this does the
# OFFICIAL AdvUnlearn arm, which is where the voided claim lived.
set -u
cd /data/bijaypandey/t2i_unlearn
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/bijaypandey/hf_home
TE=$(grep local= ADVUNLEARN_PIN.txt | cut -d= -f2)/church_unlearned
for S in 2 10 25 50 100; do
  L=LOWB_advunlearn_church_s${S}
  [ -f "outputs/atk_${L}.json" ] && continue
  venv_t2i/bin/python scripts/attack_relearn.py --concept church --text-encoder "$TE" \
    --steps $S --lr 1e-5 --seed 0 --n-images 16 --label "$L" >>logs/lowbudget.log 2>&1
done
echo "LOWBUDGET DONE $(date -Is)" >>logs/lowbudget.log
