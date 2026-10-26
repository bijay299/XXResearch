#!/bin/bash
# THIS IS NOT ANOTHER REVERSAL TEST. It tests the ONE POSITIVE SIGNAL in the entire phase.
#
# FINDING56 S5a: Meta-Unlearning's trigger is the only thing measured here that obstructed an
# adversary's OPTIMISATION -- Textual Inversion gate-failed 3/3 on its arm with T FALLING, while TI
# succeeded on GIFT at LOWER retained utility (0.6000 vs 0.6333), which eliminated the utility
# confound. Everything else in this phase is a reversal; that is the single effect worth asking a
# generalisation question about.
#
# So: build the Meta-Unlearning arm on BOTH new models (teachers = the FRP arms built there), then
# run BOTH adversaries on each:
#   * TEXTUAL INVERSION -- does the obstruction transfer? New information either way.
#       transfers  -> a real, narrow, architecture-independent mechanism, the phase's one asset;
#       does not   -> the positive signal was checkpoint-specific and FINDING56 S5a must say so.
#   * FULL-UNET relearning -- the control, expected at 1.000, included so the TI result cannot be
#       confused with general resistance.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
mkdir -p .labels logs
run_one() {  # $1 tag  $2 model  $3 variant  $4 gpu  $5 revision
  local TAG=$1 M=$2 VAR=$3 G=$4 REV=$5
  local TEACH="adapters/frp_${TAG}_church_r3_l05/unet.pt"
  local L="meta_${TAG}_church_lam0.5"
  [ -f "$TEACH" ] || { echo "$L SKIPPED -- teacher missing" >>logs/meta_cm.log; return 0; }
  mkdir ".labels/$L" 2>/dev/null || return 0
  export CUDA_VISIBLE_DEVICES=$G BASE_MODEL="$M" BASE_VARIANT="$VAR" BASE_REV="$REV"
  FREE=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  [ "$FREE" -lt 10 ] && { echo "$L SKIPPED -- ${FREE}G free" >>logs/meta_cm.log; return 0; }
  [ -f "adapters/$L/unet.pt" ] || venv_t2i/bin/python scripts/meta_unlearn.py --concept church \
    --teacher "$TEACH" --lam-meta 0.5 --label "$L" --out "adapters/$L" >>"logs/$L.log" 2>&1
  if [ ! -f "adapters/$L/unet.pt" ]; then
    echo "$L BUILD FAILED -- NOT a kill-rule result; see logs/$L.log" >>logs/meta_cm.log; return 1
  fi
  [ -f "outputs/gen_$L.json" ] || venv_t2i/bin/python scripts/gen_eval.py --label "$L" \
    --unet "adapters/$L/unet.pt" --target church >>"logs/$L.log" 2>&1
  R=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['legacy_retain8'])" 2>/dev/null || echo 0)
  T=$(venv_t2i/bin/python -c "
import json
print('%.4f' % json.load(open('outputs/gen_$L.json'))['active_target_metrics']['target_suppression_acc'])" 2>/dev/null || echo 0)
  echo "$L ($M) as-is: target=$T retain8=$R" >>logs/meta_cm.log
  awk -v r="$R" 'BEGIN{exit !(r>=0.60)}' || { echo "$L retain8=$R below K1 -- not attacked" >>logs/meta_cm.log; return 0; }
  # the question: does the TI obstruction transfer?
  for cfg in "2000 1e-2 0" "3000 2e-2 1"; do
    set -- $cfg
    local K="H88T_${L}_s$1_lr$2_sd$3"
    mkdir ".labels/$K" 2>/dev/null || continue
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_ti.py --concept church \
      --unet "adapters/$L/unet.pt" --steps "$1" --lr "$2" --seed "$3" --label "$K" >>logs/meta_cm_atk.log 2>&1
  done
  # the control: full-UNet, expected 1.000
  for SD in 0 1; do
    local K="H88F_${L}_sd${SD}_lr1e-5"
    mkdir ".labels/$K" 2>/dev/null || continue
    [ -f "outputs/atk_${K}.json" ] || venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$L/unet.pt" --seed "$SD" --lr 1e-5 --label "$K" >>logs/meta_cm_atk.log 2>&1
  done
  echo "$L both adversaries done $(date -Is)" >>logs/meta_cm.log
}
( run_one bksdm nota-ai/bk-sdm-small "" 0 572238db7ed3; echo "MCM A DONE $(date -Is)" >>logs/meta_cm.log ) &
( run_one sd15 runwayml/stable-diffusion-v1-5 fp16 1 451f4fe16113bff5a5d2269ed5ad43b0592e9a14; echo "MCM B DONE $(date -Is)" >>logs/meta_cm.log ) &
wait
