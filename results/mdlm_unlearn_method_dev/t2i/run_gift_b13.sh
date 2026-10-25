#!/bin/bash
# The cliff is now bracketed and there IS usable room inside it:
#   beta=1.0  target 0.4667  retain8 0.7833  K1 PASS
#   beta=1.2  target 0.3333  retain8 0.6667  K1 PASS   <- deeper erasure, still usable
#   beta=1.5  target 0.1333  retain8 0.5750  K1 FAIL   <- FRP-depth erasure, just misses the floor
#   beta=2.0  target 0.0000  retain8 0.2750  K1 FAIL
# So the deepest usable GIFT arm lies in (1.2, 1.5). beta=1.3 and 1.35 locate it. This matters
# because it is GIFT's genuine best shot: if an arm there reaches FRP-like depth while clearing K1,
# it is the fairest operating point to attack, and FINDING58's headline should be measured there
# too rather than only at the pre-declared primary.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p .labels logs
for B in 1.3 1.35; do
  L="gift_church_b${B}"
  mkdir ".labels/$L" 2>/dev/null || continue
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/erase_gift.py --concept church \
    --beta "$B" --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- NOT a kill-rule result" >>logs/gift_b13.log; continue
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L as-is: target=$T retain8=$R" >>logs/gift_b13.log
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$L retain8=$R below K1 -- not attacked" >>logs/gift_b13.log; continue; }
  for SD in 0 1 6; do
    K="H81D_${L}_sd${SD}_lr1e-5"
    mkdir ".labels/$K" 2>/dev/null || continue
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py \
      --concept church --unet "adapters/$L/unet.pt" --seed "$SD" --lr 1e-5 --label "$K" \
      >>logs/gift_b13_atk.log 2>&1
  done
done
echo "GIFT B13 DONE $(date -Is)" >>logs/gift_b13.log
