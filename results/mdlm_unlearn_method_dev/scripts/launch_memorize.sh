#!/bin/bash
# =============================================================================
# launch_memorize.sh -- THE preserved launch script for the v2 memorization
# backbone (theta_0). The previous backbone's hyperparameters are unrecoverable
# because no launch script was kept; this file is the single source of truth for
# the resolved configuration and is sha256'd into every report.json it produces.
#
# Usage:
#   launch_memorize.sh calib                 # cheap t_low scan, GPU 2 + 3
#   launch_memorize.sh seeds                 # the 3 qualified backbones
#   launch_memorize.sh one <gpu> <seed>      # a single qualified backbone
#
# HARD CONSTRAINTS enforced here and in launch_py.sh:
#   * GPUs 2 and 3 ONLY, at most 2 concurrently (GPUs 0-1 belong to another user)
#   * every run goes through attempt.sh so it lands in ATTEMPTS.tsv with measured
#     GPU-seconds and CPU-seconds; nothing bypasses the ledger
#   * caches on /data via env.sh (root filesystem is 100% full)
#   * no model-weight downloads (HF_HUB_OFFLINE=1); model pinned at
#     /data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct
# =============================================================================
set -u
W=/data/bijaypandey/mdlm_unlearn
source $W/env.sh

# ---- RESOLVED CONFIGURATION (the thing that was lost last time) -------------
STEPS=2400          # 2400 steps x batch 4 / 128 facts = 75 passes over the set
BATCH=4
LR=2e-4
RANK=32             # lora_alpha = 2*rank = 64, dropout 0.0, bias none
TLOW=0.7            # masking-rate floor, t ~ U(0.7, 1); calibrated, see docs/BACKBONE_BUILD.md
ANSW=1.0            # answer-word upweighting OFF -> the pinned loss formula verbatim
NORM=pinned         # pinned LLaDA SFT normalization: sum(tok/ans_len)/batch
CANVAS=genlen       # training canvas EOS-padded to the decoder's gen_length and fully noised
CLIP=1.0
THRESH=0.95
SEEDS=(20261020 20261021 20261022)
TIMEOUT=7200
# -----------------------------------------------------------------------------

gpu_free () {  # refuse to contend with another user's job
  local g=$1
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 1000 ]; then
    echo "REFUSING: GPU $g already has ${used} MiB in use -- not contending" >&2
    return 1
  fi
  return 0
}

# per-call overrides (used only by the calibration scan); empty = use the resolved
# configuration above. Assignments are explicit, never prefixed onto the function
# call, because bash makes prefix assignments to a *function* persist.
OV_STEPS=""; OV_TLOW=""; OV_ANSW=""; OV_LR=""; OV_CANVAS=""

run_one () {   # run_one <gpu> <seed> <phase> <label> <mode> [extra args...]
  local g=$1 seed=$2 phase=$3 label=$4 mode=$5; shift 5
  local steps=${OV_STEPS:-$STEPS} tlow=${OV_TLOW:-$TLOW}
  local answ=${OV_ANSW:-$ANSW} lr=${OV_LR:-$LR} canvas=${OV_CANVAS:-$CANVAS}
  local out rep
  if [ "$mode" = calibrate ]; then
    out=$W/tmp/calib_${label}; rep=$W/outputs/calib_${label}_report.json
  else
    out=$W/adapters/memorized_v2_s${seed}; rep=$W/outputs/memorize_v2_s${seed}_report.json
  fi
  local cmd="$W/scripts/launch_py.sh $g $phase $label $TIMEOUT memorize_v2.py \
--seed $seed --steps $steps --batch $BATCH --lr $lr --rank $RANK --t-low $tlow \
--answer-weight $answ --norm $NORM --canvas $canvas --grad-clip $CLIP --threshold $THRESH \
--mode $mode --out $out --report $rep $*"
  export MDLM_LAUNCH_CMD="$cmd"
  export MDLM_ATTEMPT_LABEL="$label"
  echo "[launch] gpu=$g seed=$seed mode=$mode -> $out"
  echo "[cmd] $cmd"
  $cmd
}

case "${1:-}" in
  calib)
    # t_low scan: the one hyperparameter the audit implicates (1980/2400 steps of the
    # previous run had loss=0 and gnorm=0 yet recall was only 0.797 -- train states
    # too far from the fully-masked generation condition). Short runs, 32 facts/split.
    gpu_free 2 || exit 92; gpu_free 3 || exit 92
    OV_STEPS=600
    OV_CANVAS=exact
    OV_TLOW=0.0; run_one 2 20261020 calib tlow000 calibrate --qual-subset 32 & P2=$!
    OV_TLOW=0.7; run_one 3 20261020 calib tlow070 calibrate --qual-subset 32 & P3=$!
    wait $P2; R2=$?; wait $P3; R3=$?
    OV_STEPS=""; OV_TLOW=""; OV_CANVAS=""
    echo "[calib] tlow000 rc=$R2  tlow070 rc=$R3"
    ;;
  calibone)
    # calibone <gpu> <label> <steps> <tlow> <answ> <lr> <canvas>
    gpu_free "$2" || exit 92
    OV_STEPS=$4; OV_TLOW=$5; OV_ANSW=$6; OV_LR=$7; OV_CANVAS=$8
    run_one "$2" 20261020 calib "$3" calibrate --qual-subset 32
    RC=$?; OV_STEPS=""; OV_TLOW=""; OV_ANSW=""; OV_LR=""; OV_CANVAS=""
    echo "[calibone] $3 rc=$RC"; exit $RC
    ;;
  calibpair)
    # calibpair <label_a> <steps_a> <tlow_a> <answ_a> <lr_a> <label_b> ... <lr_b>
    # two calibration runs, one on GPU 2 and one on GPU 3 (never more than 2 at once).
    gpu_free 2 || exit 92; gpu_free 3 || exit 92
    shift
    LA=$1; OV_STEPS=$2; OV_TLOW=$3; OV_ANSW=$4; OV_LR=$5; OV_CANVAS=$6
    run_one 2 20261020 calib "$LA" calibrate --qual-subset 32 & P2=$!
    LB=$7; OV_STEPS=$8; OV_TLOW=$9; OV_ANSW=${10}; OV_LR=${11}; OV_CANVAS=${12}
    run_one 3 20261020 calib "$LB" calibrate --qual-subset 32 & P3=$!
    wait $P2; R2=$?; wait $P3; R3=$?
    OV_STEPS=""; OV_TLOW=""; OV_ANSW=""; OV_LR=""; OV_CANVAS=""; OV_CANVAS=""
    echo "[calibpair] $LA rc=$R2  $LB rc=$R3"
    ;;
  seeds)
    # 3 independent memorized starting checkpoints = the variance component of the
    # evaluation protocol. At most 2 concurrent: seeds 1+2 together, then seed 3.
    gpu_free 2 || exit 92; gpu_free 3 || exit 92
    run_one 2 "${SEEDS[0]}" memorize v2_s"${SEEDS[0]}" qualify & P2=$!
    run_one 3 "${SEEDS[1]}" memorize v2_s"${SEEDS[1]}" qualify & P3=$!
    wait $P2; R1=$?; wait $P3; R2=$?
    gpu_free 2 || exit 92
    run_one 2 "${SEEDS[2]}" memorize v2_s"${SEEDS[2]}" qualify; R3=$?
    echo "[seeds] ${SEEDS[0]} rc=$R1  ${SEEDS[1]} rc=$R2  ${SEEDS[2]} rc=$R3"
    [ "$R1" = 0 ] && [ "$R2" = 0 ] && [ "$R3" = 0 ]
    ;;
  one)
    gpu_free "$2" || exit 92
    run_one "$2" "$3" memorize v2_s"$3" qualify
    ;;
  smoke)
    gpu_free "$2" || exit 92
    OV_STEPS=6; run_one "$2" 1 smoke v2_smoke calibrate --qual-subset 2
    ;;
  *)
    echo "usage: $0 {calib|calibone <gpu> <label> <steps> <tlow> <answ> <lr> <canvas>|calibpair <6 args> <6 args>|seeds|one <gpu> <seed>|smoke <gpu>}" >&2; exit 2;;
esac
