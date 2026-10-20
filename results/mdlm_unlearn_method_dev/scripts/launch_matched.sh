#!/bin/bash
# =============================================================================
# launch_matched.sh -- THE preserved launch script for the matched qualified-baseline
# comparison. Single source of truth for the resolved configuration AND for the
# declared hyperparameter grid; sha256'd into docs/BASELINE_MATCHED.md.
#
# Usage:
#   launch_matched.sh smoke <gpu>        # 6 steps, 4 probes/split -- wiring check
#   launch_matched.sh reference          # the memorized reference row (no training)
#   launch_matched.sh sweep              # the full declared grid, 2 GPUs at a time
#   launch_matched.sh one <gpu> <arm> <hp> <label>
#
# HARD CONSTRAINTS enforced here and in launch_py.sh:
#   * GPUs 2 and 3 ONLY, at most 2 concurrently
#   * every run through attempt.sh -> ATTEMPTS.tsv with measured GPU/CPU seconds
#   * caches on /data via env.sh (root filesystem is full)
#   * no model-weight downloads (HF_HUB_OFFLINE=1)
# =============================================================================
set -u
W=/data/bijaypandey/mdlm_unlearn
source $W/env.sh

# ---- MATCHED RECIPE (identical for all five arms; see run_matched.py MATCHED) ----
STEPS=150          # identical step budget for every arm
BACKBONE=$W/adapters/memorized_v2_s20261101     # 1.000/1.000, save/reload max|dlogit|=0
TIMEOUT=5400

# ---- THE DECLARED GRID (pre-registered; written into the report BEFORE any run) ----
# Each arm gets a tuning opportunity on its own principal hyperparameter. Three points
# per arm, one order of magnitude apart where the parameter is a scale, so the grid is
# wide enough that a badly-chosen default cannot be mistaken for a method failure.
#
#   mdu_t1  tau  0.5 1.0 2.0   anchor temperature (1.0 = the published MDU tau=1)
#   emdu    tau  0.5 1.0 2.0   anchor temperature
#   npo     beta 0.05 0.2 0.5  (0.2 = the published default in MDU App. B)
#   sc_tf   eps  1e-5 1e-3 1e-1  exclusion-target leak budget
#   excl    eps  1e-5 1e-3 1e-1  exclusion-target leak budget
#
# NOTE, DECLARED: the brief names "SC-TF's temperature" as its principal hyperparameter.
# The pinned sc_tf_loss_llada takes NO temperature -- it is fx_forget_loss_llada, whose
# only knob is the exclusion-target eps. Adding a temperature would mean editing the
# PINNED package, which is forbidden. We therefore sweep eps for SC-TF and record this
# substitution rather than inventing a knob.
GRID_MDU="0.5 1.0 2.0"
GRID_EMDU="0.5 1.0 2.0"
GRID_NPO="0.05 0.2 0.5"
GRID_SCTF="1e-5 1e-3 1e-1"
GRID_EXCL="1e-5 1e-3 1e-1"
# ----------------------------------------------------------------------------------

gpu_free () {   # refuse to contend with another user's job rather than queueing on it
  local g=$1 used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 1000 ]; then
    echo "REFUSING: GPU $g already has ${used} MiB in use -- not contending" >&2
    return 1
  fi
  return 0
}

hp_tag () { echo "$1" | tr '.-' 'pm'; }

run_one () {   # run_one <gpu> <arm> <hp> <label> [extra args...]
  local g=$1 arm=$2 hp=$3 label=$4; shift 4
  local rep=$W/outputs/matched_${label}.json
  local hparg=""
  [ "$arm" != reference ] && hparg="--hp $hp"
  echo "[launch] gpu=$g arm=$arm hp=$hp label=$label -> $rep"
  $W/scripts/launch_py.sh "$g" matched "$label" $TIMEOUT run_matched.py \
    --arm "$arm" $hparg --steps $STEPS --memo "$BACKBONE" --label "$label" \
    --report "$rep" "$@"
}

pair () {   # pair "<arm> <hp> <label>" "<arm> <hp> <label>"  -- never more than 2 at once
  gpu_free 2 || return 92
  gpu_free 3 || return 92
  # shellcheck disable=SC2086
  run_one 2 $1 & local P2=$!
  # shellcheck disable=SC2086
  run_one 3 $2 & local P3=$!
  wait $P2; local R2=$?
  wait $P3; local R3=$?
  echo "[pair] ($1) rc=$R2   ($2) rc=$R3"
}

case "${1:-}" in
  smoke)
    gpu_free "$2" || exit 92
    $W/scripts/launch_py.sh "$2" smoke matched_smoke 1800 run_matched.py \
      --arm excl --hp 1e-5 --steps 6 --memo "$BACKBONE" --label smoke \
      --report $W/outputs/matched_smoke.json --eval-limit 4 --save-adapter 0
    ;;
  reference)
    gpu_free 2 || exit 92
    run_one 2 reference 0 reference --save-adapter 0
    ;;
  sweep)
    # 15 runs = 5 arms x 3 grid points, two at a time. Deterministic order, so the
    # ATTEMPTS.tsv rows can be read against this file.
    RC=0; JOBS=()
    for t in $GRID_MDU;  do JOBS+=("mdu_t1 $t mdu_tau$(hp_tag $t)");   done
    for t in $GRID_EMDU; do JOBS+=("emdu $t emdu_tau$(hp_tag $t)");    done
    for b in $GRID_NPO;  do JOBS+=("npo $b npo_beta$(hp_tag $b)");     done
    for e in $GRID_SCTF; do JOBS+=("sc_tf $e sctf_eps$(hp_tag $e)");   done
    for e in $GRID_EXCL; do JOBS+=("excl $e excl_eps$(hp_tag $e)");    done
    echo "[sweep] ${#JOBS[@]} runs:"; printf '   %s\n' "${JOBS[@]}"
    i=0
    while [ $i -lt ${#JOBS[@]} ]; do
      if [ $((i + 1)) -lt ${#JOBS[@]} ]; then
        pair "${JOBS[$i]}" "${JOBS[$((i+1))]}" || RC=$?
        i=$((i + 2))
      else
        gpu_free 2 || exit 92
        # shellcheck disable=SC2086
        run_one 2 ${JOBS[$i]} || RC=$?
        i=$((i + 1))
      fi
    done
    echo "[sweep] done rc=$RC"; exit $RC
    ;;
  one)
    gpu_free "$2" || exit 92
    run_one "$2" "$3" "$4" "$5"
    ;;
  *)
    echo "usage: $0 {smoke <gpu>|reference|sweep|one <gpu> <arm> <hp> <label>}" >&2
    exit 2;;
esac
