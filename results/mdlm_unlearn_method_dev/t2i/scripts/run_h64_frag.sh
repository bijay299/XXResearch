#!/bin/bash
set -u
cd /data/bijaypandey/t2i_unlearn
source env.sh
PY=./venv_t2i/bin/python
G=$1; AL=$2
ROOT=$($PY -c "d=dict(l.split('=',1) for l in open('ADVUNLEARN_PIN.txt').read().split('\n') if '=' in l);print(d['local'])")
TE="$ROOT/church_unlearned"
# H64: the FRP-principled allocation -- damage forget-critical AND spare retain-critical, the
# intersection neither of my earlier schemes implemented. Grounded in arXiv 2608.25429, whose FRP
# achieves dES +0.059 vs RMU's +0.644 on exactly this principle in the LLM setting.
# Full attack budget from the start (9 attacks), per the lesson H62 taught: a sub-ceiling number
# from 3 seeds at one lr has been wrong four times in this project.
arm=h64_frag_a${AL}
o=adapters/$arm
[ -f outputs/undo_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/undo_t2i.py \
  --teacher-te "$TE" --concept church --alpha $AL --beta 0.1 --protect frag \
  --steps 6000 --log-every 3000 --n-images 300 --forget-frac 0.20 --fisher-batches 16 \
  --seed 0 --label $arm --out $o > outputs/log_$arm.txt 2>&1
[ -f outputs/gen_${arm}.json ] || CUDA_VISIBLE_DEVICES=$G $PY scripts/gen_eval.py --label $arm \
  --unet $o/unet.pt --text-encoder "$TE" --n-per-prompt 5 --steps 25 --seed 0 >> outputs/log_$arm.txt 2>&1
ret=$($PY -c "import json;d=json.load(open('outputs/gen_${arm}.json'));F={'church','parachute','garbage_truck','van_gogh'};pc=d['per_concept'];r=[v['clip_acc'] for k,v in pc.items() if k not in F];print(round(sum(r)/len(r),4))" 2>/dev/null || echo 0)
echo "[h64] $arm church=$($PY -c "import json;print(json.load(open('outputs/gen_${arm}.json'))['per_concept']['church']['clip_acc'])" 2>/dev/null) retain8=$ret" >> outputs/log_h64_frag.txt
for spec in "0|1e-5" "1|1e-5" "2|1e-5" "3|1e-5" "6|1e-5" "0|5e-5" "1|5e-5" "6|5e-5" "0|1e-4"; do
  IFS='|' read -r sd lr <<< "$spec"
  lab="H64_frag_a${AL}_sd${sd}_lr${lr}"
  [ -f outputs/atk_${lab}.json ] && continue
  CUDA_VISIBLE_DEVICES=$G $PY scripts/attack_relearn.py --unet $o/unet.pt --text-encoder "$TE" \
    --concept church --steps 300 --lr $lr --n-images 16 --eval-n 5 --seed $sd --label $lab \
    > outputs/log_atk_$lab.txt 2>&1
  grep -E "^\[atk\] ${lab}" outputs/log_atk_$lab.txt >> outputs/log_h64_frag.txt
done
rm -f $o/unet.pt
echo "[h64-$AL] done"
