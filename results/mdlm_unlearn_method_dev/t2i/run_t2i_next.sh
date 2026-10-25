#!/bin/bash
# t2i priority. H80's honest follow-up, which FINDING56 section 6 named: a finer lambda_meta sweep
# between 0.5 and 1.0. The Meta-Unlearning trap costs retain8 0.9417 -> 0.6333 at lam=0.5 and
# collapses to 0.1167 at lam>=1.0 (PREREG_H80 K2 fail). If a usable trigger exists at all it lives
# in that gap. At lam=0.5 the trap bought ZERO robustness over its own lam=0 ablation, so the
# prediction is that nothing usable is there -- but that gap is exactly where it would have to be.
#
# TWO BUGS IN THE PREVIOUS VERSION OF THIS SCRIPT, both fixed here:
#
# (1) Its GPU guard waited on "train_lam.py --arm mdu", which stopped matching once the AFC lam=4
#     run moved from TRAINING to EVALUATION. All four builds then launched onto GPUs already holding
#     a 16.5 GiB eval and died with CUDA OOM (meta_unlearn needs ~22.7 GiB; 39.4 GiB total). The
#     guard now waits on ANY of my python processes on the target GPU, by GPU index.
#
# (2) Far worse: the K2 check read retain8 through a bare try/except that printed '0' on failure, so
#     four CRASHED builds were logged as "retain8=0 below K2 -- self-destruct fired on normal use".
#     That is a scientific conclusion manufactured from a missing file. A build failure and a
#     measured-low retain are now distinguished explicitly and a missing output is reported as
#     BUILD FAILED, never as a K2 result.
set -u
cd /data/bijaypandey/t2i_unlearn
export HF_HOME=/data/bijaypandey/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p .labels logs
claim() { mkdir ".labels/$1" 2>/dev/null; }

# wait until the named GPU has no process of mine on it
gpu_free() {
  local g=$1 pids
  pids=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader 2>/dev/null \
         | awk -F, -v idx="$g" 'NR>0')
  # simpler and sufficient: compare used memory on that index against a small threshold
  local used
  used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  [ -n "$used" ] && [ "$used" -lt 1000 ]
}
wait_for_gpu() { while ! gpu_free "$1"; do sleep 60; done; }

build() {  # $1 lam_meta  $2 gpu
  local LAM=$1 GPU=$2 L
  L="h80_meta_church_lam${LAM}"
  claim "$L" || return 0
  export CUDA_VISIBLE_DEVICES=$GPU
  if [ ! -f "adapters/$L/unet.pt" ]; then
    venv_t2i/bin/python scripts/meta_unlearn.py \
      --teacher adapters/frp_church_r3_l05/unet.pt \
      --concept church --lam-meta "$LAM" --label "$L" --out "adapters/$L" \
      >>"logs/$L.log" 2>&1
  fi
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- no unet.pt. NOT a K2 result; see logs/$L.log" >>logs/t2i_next.log
    return 1
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py \
    --label "$L" --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
}

wait_for_gpu 1
( for LAM in 0.6 0.8; do build "$LAM" 1; done; echo "T2I A DONE $(date -Is)" >>logs/t2i_next.log ) &
( wait_for_gpu 0
  for LAM in 0.7 0.9; do build "$LAM" 0; done
  echo "T2I B DONE $(date -Is)" >>logs/t2i_next.log ) &
wait

# Attack only arms that BUILT and then cleared the pre-registered K2 floor. A missing gen output is
# a build/eval failure and is reported as such -- never as "retain8=0".
for LAM in 0.6 0.7 0.8 0.9; do
  A="h80_meta_church_lam${LAM}"
  if [ ! -f "outputs/gen_$A.json" ]; then
    echo "$A NO EVAL OUTPUT -- build or eval failed, no K2 verdict possible" >>logs/t2i_next.log
    continue
  fi
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$A.json'))['active_target_metrics']['legacy_retain8'])")
  if ! awk -v r="$R" 'BEGIN{exit !(r>=0.60)}'; then
    echo "$A retain8=$R MEASURED below K2 0.60 -- self-destruct fires on normal use, not attacked" \
      >>logs/t2i_next.log
    continue
  fi
  echo "$A retain8=$R clears K2 -- attacking on the frozen grid" >>logs/t2i_next.log
  for SD in 0 1 6; do
    K="H80F_${A}_sd${SD}_lr1e-5"
    claim "$K" || continue
    export CUDA_VISIBLE_DEVICES=1
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py \
      --concept church --unet "adapters/$A/unet.pt" --seed "$SD" --lr 1e-5 --label "$K" \
      >>logs/t2i_atk.log 2>&1
  done
done
echo "T2I ATK DONE $(date -Is)" >>logs/t2i_next.log
