#!/bin/bash
# =============================================================================
# launch_tofu.sh -- THE preserved launch script for the TOFU backbones.
# Single source of truth for the resolved configuration; sha256'd into every
# report.json it produces. Same contract as launch_memorize.sh, which built the
# synthetic backbone; that script and memorize_v2.py are UNTOUCHED (their sha256
# is recorded inside the three delivered synthetic backbone reports).
#
# Usage:
#   launch_tofu.sh floor                      # pristine-base floor, all 800 probes
#   launch_tofu.sh smoke <gpu>                # 6 steps, 4 probes/split -- wiring check
#   launch_tofu.sh calibpair <A...> <B...>    # two calibration runs, GPU 2 + 3
#   launch_tofu.sh calibone <gpu> <label> <steps> <canvas> <lr> <subset>
#   launch_tofu.sh backbones                  # the two production runs, GPU 2 + 3
#   launch_tofu.sh one <gpu> <split> <label>  # a single production run
#
# HARD CONSTRAINTS enforced here and in launch_py.sh:
#   * GPUs 2 and 3 ONLY, at most 2 concurrently; pre-check and re-queue rather
#     than contend (the p045 OOM in the matched batch was caused by a foreign
#     process appearing after the pre-check, and was re-queued, not forced)
#   * every run through attempt.sh -> ATTEMPTS.tsv with measured GPU/CPU seconds
#   * caches on /data via env.sh (root filesystem is full)
#   * no model-weight downloads (HF_HUB_OFFLINE=1)
# =============================================================================
set -u
W=/data/bijaypandey/mdlm_unlearn
source $W/env.sh

# ---- RESOLVED CONFIGURATION ------------------------------------------------
# Scale: 800 QA is 6.25x the 128-fact synthetic set. The synthetic backbone used
# 1200 steps x batch 4 / 128 = 37.5 passes. Holding passes constant would need
# 7500 steps; CALIBRATED instead (see docs/BACKBONE_TOFU.md), because TOFU answers
# are 34 tokens rather than one word and the loss has far more signal per example.
STEPS=12000         # SET BY THE CALIBRATION SCAN (docs/BACKBONE_TOFU.md section 4.3):
                    # distinctive recall on 64 probes/split went 1500 -> 0.325/0.417,
                    # 6000 -> 0.622/0.660, 12000 -> 0.789/0.827. 6000 clears the 0.60 gate
                    # by only 0.022 on a 64-probe subset (SE ~0.04), which is a coin flip
                    # on the full 400; 12000 clears it by 0.19. Still rising at 12000 but
                    # with clearly diminishing returns.
                    # Both runs use the SAME step budget. For retain_ref_tofu that is 120
                    # passes over its 400 QA versus 60 for theta0_tofu -- deliberately, and
                    # it makes the ignorance test STRONGER: if forget recall stays at the
                    # pristine floor after 120 passes of retain-only training, nothing
                    # about the forget split is learnable from retain.
BATCH=4
LR=2e-4             # as the synthetic backbone
RANK=32             # lora_alpha = 2*rank = 64, dropout 0.0, bias none
CANVAS=genlen       # the answer region is EOS-padded to gen_length(96) and the WHOLE
                    # region is noised, so training sees the fully-masked canvas the
                    # decoder starts from. Decisive on the synthetic set (0.844 -> 1.000)
                    # and expected to matter MORE here: a TOFU answer has variable
                    # length, so the model must also learn where to stop.
TLOW=0.0            # t ~ U(0,1): the pinned recipe verbatim
NORM=pinned         # pinned LLaDA SFT normalization: sum(tok/ans_len)/batch
CLIP=1.0
TIMEOUT=14400
# Thresholds are declared in docs/BACKBONE_TOFU.md and justified against the MEASURED
# pristine-base floor, not against a round number. Passed explicitly so the resolved
# value is in the launch command recorded in every report.
TH_RECALL=0.60      # memorization gate, both splits
TH_IGN=0.15         # retain-reference IGNORANCE ceiling on the forget split
# ---------------------------------------------------------------------------

gpu_free () {   # refuse to contend with another user's job rather than queueing on it
  local g=$1 used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 1000 ]; then
    echo "REFUSING: GPU $g already has ${used} MiB in use -- not contending" >&2
    return 1
  fi
  return 0
}

OV_STEPS=""; OV_CANVAS=""; OV_LR=""; OV_SUBSET=""; OV_MODE=""

run_train () {   # run_train <gpu> <split> <label> [extra args...]
  local g=$1 split=$2 label=$3; shift 3
  local steps=${OV_STEPS:-$STEPS} canvas=${OV_CANVAS:-$CANVAS} lr=${OV_LR:-$LR}
  local mode=${OV_MODE:-qualify}
  local out rep
  if [ "$mode" = calibrate ]; then
    out=$W/tmp/tofu_calib_${label}; rep=$W/outputs/tofu_calib_${label}_report.json
  else
    out=$W/adapters/${label}; rep=$W/outputs/${label}_report.json
  fi
  local sub=""
  [ -n "${OV_SUBSET}" ] && sub="--qual-subset ${OV_SUBSET}"
  local cmd="$W/scripts/launch_py.sh $g tofu $label $TIMEOUT memorize_tofu.py \
--train-split $split --steps $steps --batch $BATCH --lr $lr --rank $RANK \
--canvas $canvas --t-low $TLOW --norm $NORM --grad-clip $CLIP --seed 20261101 \
--threshold $TH_RECALL --ignorance-ceiling $TH_IGN --mode $mode $sub \
--out $out --report $rep $*"
  export MDLM_LAUNCH_CMD="$cmd"
  export MDLM_ATTEMPT_LABEL="$label"
  echo "[launch] gpu=$g split=$split mode=$mode steps=$steps canvas=$canvas -> $out"
  echo "[cmd] $cmd"
  $cmd
}

case "${1:-}" in
  floor)
    # The pristine base, all 800 probes, under the IDENTICAL evaluator the gates use.
    # This is the number every threshold is justified against.
    gpu_free 2 || exit 92
    $W/scripts/launch_py.sh 2 tofu floor_base $TIMEOUT eval_tofu.py \
      --adapter none --label pristine_base --out $W/outputs/tofu_floor_base.json
    ;;
  smoke)
    gpu_free "$2" || exit 92
    OV_STEPS=6; OV_MODE=calibrate; OV_SUBSET=4
    run_train "$2" all tofu_smoke
    ;;
  calibone)
    # calibone <gpu> <label> <steps> <canvas> <lr> <subset-per-split>
    gpu_free "$2" || exit 92
    OV_STEPS=$4; OV_CANVAS=$5; OV_LR=$6; OV_SUBSET=$7; OV_MODE=calibrate
    run_train "$2" all "$3"
    RC=$?; echo "[calibone] $3 rc=$RC"; exit $RC
    ;;
  calibpair)
    # calibpair <labelA> <stepsA> <canvasA> <lrA> <subA>  <labelB> ... <subB>
    gpu_free 2 || exit 92; gpu_free 3 || exit 92
    shift
    OV_MODE=calibrate
    OV_STEPS=$2; OV_CANVAS=$3; OV_LR=$4; OV_SUBSET=$5
    run_train 2 all "$1" & P2=$!
    OV_STEPS=$7; OV_CANVAS=$8; OV_LR=$9; OV_SUBSET=${10}
    run_train 3 all "$6" & P3=$!
    wait $P2; R2=$?; wait $P3; R3=$?
    echo "[calibpair] $1 rc=$R2   $6 rc=$R3"
    ;;
  backbones)
    # THE TWO PRODUCTION RUNS, one per GPU, never more than 2 concurrent.
    #   theta0_tofu      -- SFT on all 800 QA; must recall BOTH splits
    #   retain_ref_tofu  -- SFT on the 400 RETAIN QA only; must recall retain and
    #                       must NOT recall forget (the ignorance gate the synthetic
    #                       set failed at 1.000 because surname->answer was learnable
    #                       from retain alone)
    gpu_free 2 || exit 92; gpu_free 3 || exit 92
    run_train 2 all    theta0_tofu     & P2=$!
    run_train 3 retain retain_ref_tofu & P3=$!
    wait $P2; R2=$?; wait $P3; R3=$?
    echo "[backbones] theta0_tofu rc=$R2   retain_ref_tofu rc=$R3"
    [ "$R2" = 0 ] && [ "$R3" = 0 ]
    ;;
  one)
    gpu_free "$2" || exit 92
    run_train "$2" "$3" "$4"
    ;;
  *)
    echo "usage: $0 {floor|smoke <gpu>|calibone <gpu> <label> <steps> <canvas> <lr> <subset>|calibpair <5 args> <5 args>|backbones|one <gpu> <split> <label>}" >&2
    exit 2;;
esac
