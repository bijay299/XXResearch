#!/bin/bash
# Aggregate -> parameter movement -> figures -> image grids -> report + slides.
#
# Required products are produced into an ISOLATED staging directory and
# published only after every required stage has succeeded, so a half-failed run
# cannot leave a mixture of fresh and stale files that looks complete. Optional
# output formats are declared optional and reported as absent rather than
# silently skipped.
#
#   bash scripts/seq/run_analysis.sh 29
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/exp_config.sh"
source "${PILOT_VENV}/bin/activate"
cd "${PILOT_REPO_ROOT}"

SEED="${1:?usage: run_analysis.sh <seed>}"
MODELS="${SEQ_ROOT}/models/seed${SEED}"
ANA="${SEQ_ROOT}/analysis/seed${SEED}"
# Overridable so the CPU test suite can publish into a sandbox rather than
# overwriting the committed results tree.
REPO_OUT="${SEQ_REPO_OUT_OVERRIDE:-${PILOT_REPO_ROOT}/results/seq${SEED}}"

# ONE shared path policy (exp_config.sh) -- the same function run_seed.sh uses.
# These two scripts previously disagreed for seed 17: run_seed.sh wrote
# eval_seed17 while this script read eval/.
EVALR="$(seq_resolve_eval_root "$SEED")" || exit 1
echo "seed ${SEED}: analysing ${EVALR}"

for f in "$SEQ_EVAL_MANIFEST" "$SEQ_CKPT_CONTRACT" "$SEQ_GEN_SETTINGS_CONTRACT" \
         "$SEQ_VALIDATOR"; do
    [ -s "$f" ] || { echo "FATAL: required contract/validator missing: $f" >&2; exit 1; }
done
# Frozen expected identity from CONFIGURATION, not from the manifest being
# validated.
MANIFEST_SHA="${SEQ_EVAL_MANIFEST_SHA:?SEQ_EVAL_MANIFEST_SHA must be set}"
# This script validates EVALUATIONS only; the completed-step policy belongs to
# training validation, is decided per artifact in run_seed.sh, and is not
# consulted here. It used to be computed per SEED at this point and never used.

ckpt_sha() {   # ckpt_sha <name>
    local f="${MODELS}/$1/delta.bin"
    [ -s "$f" ] || return 1
    sha256sum "$f" | cut -d" " -f1
}

# Refuse to analyse a seed whose evaluations do not validate. A short, stale or
# malformed detections.jsonl would otherwise be aggregated into rates that look
# finished.
INVALID=()
for ck in M0 MA MAB MAB_L2 MAC MAC_L2; do
    extra=()
    if [ "$ck" = "M0" ]; then
        # Same explicit policy as the launcher: declared reuse plus base-model
        # identity, because M0 applies no delta.
        extra=(--allow_shared_images --base_model_contract "$SEQ_BASE_MODEL_CONTRACT")
    else
        # Bind each evaluation to the checkpoint now on disk. The analysis path
        # needs this as much as the launch path: aggregating an evaluation that
        # was generated from a different same-named checkpoint would silently
        # mix evidence from two models.
        if ! sha="$(ckpt_sha "$ck")"; then
            echo "    ${ck}: delta.bin missing or empty; cannot bind evaluation" >&2
            INVALID+=("$ck"); continue
        fi
        extra=(--expect_sha "$sha")
    fi
    if ! CUDA_VISIBLE_DEVICES="" python "$SEQ_VALIDATOR" eval "$ck" \
            --eval_root "$EVALR" --manifest "$SEQ_EVAL_MANIFEST" \
            --expect_manifest_sha "$MANIFEST_SHA" \
            --expect_gen_settings "$SEQ_GEN_SETTINGS_CONTRACT" \
            --require_images "${extra[@]}" 2>&1 | sed 's/^/    /'; then
        INVALID+=("$ck")
    fi
done
if [ "${#INVALID[@]}" -gt 0 ]; then
    echo "FATAL: seed ${SEED} evaluations do not validate: ${INVALID[*]}" >&2
    echo "Re-run scripts/seq/run_seed.sh ${SEED} before aggregating." >&2
    exit 1
fi
echo "all 6 evaluations validated"

# Stage into an isolated directory; publish only on full success.
STAGE="$(mktemp -d "${SEQ_ROOT}/analysis/.stage_seed${SEED}.XXXXXX")" || exit 1
cleanup() { [ -n "${STAGE:-}" ] && [ -d "$STAGE" ] && rm -rf "$STAGE"; }
trap cleanup EXIT
mkdir -p "$STAGE/figures" "$STAGE/grids"

req() {   # req <label> <cmd...> : a required stage; any failure aborts
    local label="$1"; shift
    echo "--- ${label}"
    if ! "$@"; then
        echo "FATAL: required stage failed: ${label}" >&2
        echo "Nothing was published; ${REPO_OUT} and ${ANA} are unchanged." >&2
        exit 1
    fi
}

echo "=== 1/5 aggregate (required) ==="
req "aggregate" python scripts/seq/aggregate.py --eval_root "$EVALR" \
    --out_dir "$STAGE" --training_seed "$SEED"

echo "=== 2/5 parameter movement (required) ==="
req "parameter movement" python scripts/seq/param_movement.py \
    --models_root "$MODELS" --base_model_dir "$SEQ_BASE_MODEL" \
    --out "${STAGE}/param_movement.json"

echo "=== 3/5 figures (required) ==="
req "figures" python scripts/seq/make_figures.py \
    --contrasts "${STAGE}/contrasts.json" --out_dir "${STAGE}/figures"

echo "=== 4/5 image grids (required) ==="
req "image grids" python scripts/seq/make_grids.py \
    --eval_root "$EVALR" --out_dir "${STAGE}/grids"

echo "=== 5/5 report + slide outline (required) ==="
req "report" python scripts/seq/make_report.py \
    --contrasts "${STAGE}/contrasts.json" --models_root "$MODELS" \
    --calib "${SEQ_ROOT}/calib/calibration_report.json" \
    --reload_check "${SEQ_ROOT}/eval/_reload_check/report.json" \
    --anchor_cache "${SEQ_ANCHOR_ROOT}/Horses/cache_meta.json" \
    --movement "${STAGE}/param_movement.json" \
    --out "${STAGE}/RESULTS.md" --slides "${STAGE}/SLIDE_OUTLINE.md" \
    --training_seed "$SEED"

# Every required product must exist and be non-empty before anything publishes.
REQUIRED=(per_image.csv rates.csv contrasts.json summary.md RESULTS.md
          SLIDE_OUTLINE.md param_movement.json
          figures/fig1_historical_trajectory.png
          figures/fig2_tradeoff.png
          figures/fig3_checkpoint_category_heatmap.png
          grids/blinded/annotation_sheet.csv grids/blinded/README.md)
MISSING=()
for f in "${REQUIRED[@]}"; do [ -s "${STAGE}/${f}" ] || MISSING+=("$f"); done
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "FATAL: required product(s) absent or empty: ${MISSING[*]}" >&2
    echo "Nothing was published." >&2
    exit 1
fi

# ---- publish: required products copied with hard failure on error -----------
mkdir -p "$ANA" "${REPO_OUT}/figures" "${REPO_OUT}/train" "${REPO_OUT}/grids"
cp -r "${STAGE}/." "${ANA}/" || { echo "FATAL: publish to ${ANA} failed" >&2; exit 1; }

pub() {   # pub <src> <dst> : a required copy
    cp "$1" "$2" || { echo "FATAL: required copy failed: $1 -> $2" >&2; exit 1; }
}
for f in contrasts.json rates.csv summary.md RESULTS.md SLIDE_OUTLINE.md \
         param_movement.json per_image.csv; do
    pub "${STAGE}/${f}" "${REPO_OUT}/${f}"
done
for f in fig1_historical_trajectory fig2_tradeoff fig3_checkpoint_category_heatmap; do
    pub "${STAGE}/figures/${f}.png" "${REPO_OUT}/figures/${f}.png"
done
pub "${STAGE}/grids/blinded/annotation_sheet.csv" "${REPO_OUT}/grids/annotation_sheet.csv"
pub "${STAGE}/grids/blinded/README.md"            "${REPO_OUT}/grids/README.md"

# Training reports: required, one per trained checkpoint.
for n in MA MAB MAB_L2 MAC MAC_L2; do
    pub "${MODELS}/${n}/train_report.json" "${REPO_OUT}/train/${n}.json"
done
# Detection reports: required, one per evaluated checkpoint.
for ck in M0 MA MAB MAB_L2 MAC MAC_L2; do
    pub "${EVALR}/${ck}/detect_report.json" "${REPO_OUT}/train/detect_${ck}.json"
done

# ---- optional output formats: explicitly optional, reported if absent -------
OPTIONAL_ABSENT=()
for f in fig1_historical_trajectory fig2_tradeoff fig3_checkpoint_category_heatmap; do
    for ext in pdf svg; do
        if [ -s "${STAGE}/figures/${f}.${ext}" ]; then
            cp "${STAGE}/figures/${f}.${ext}" "${REPO_OUT}/figures/${f}.${ext}" \
                || OPTIONAL_ABSENT+=("${f}.${ext} (copy failed)")
        else
            OPTIONAL_ABSENT+=("${f}.${ext}")
        fi
    done
done

echo
echo "analysis dir : $ANA"
echo "repo copy    : $REPO_OUT"
if [ "${#OPTIONAL_ABSENT[@]}" -gt 0 ]; then
    echo "optional formats ABSENT (not an error): ${OPTIONAL_ABSENT[*]}"
else
    echo "optional formats: all present"
fi
du -sh "$REPO_OUT"
echo "=== ANALYSIS seed ${SEED} COMPLETE (all required stages validated) ==="
