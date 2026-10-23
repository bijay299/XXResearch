#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1; shift
ROOT=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local'])")
TE="$ROOT/tench_unlearned"
# H62: is tench's 0.733 plateau REAL or seed noise? It is the one concept of five that did not
# fully reverse under corrected measurement (0.733/0.733/0.467 across seeds 0/1/6). Either it is a
# genuine concept-level difference -- the only case where the attack does not fully succeed, which
# would be the most interesting single datapoint in the phase -- or it is noise in a 15-sample
# metric. More seeds, plus a higher-lr point, settle it. Direction note: adding seeds can only
# RAISE max-over-valid, so this test can only weaken the "tench resists somewhat" reading.
for spec in "$@"; do
  IFS='|' read -r sd lr <<< "$spec"
  lab="H62_tench_sd${sd}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --text-encoder "$TE" --concept tench \
    --steps 300 --lr $lr --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_h62_tench.txt
done
echo "[h62] done"
