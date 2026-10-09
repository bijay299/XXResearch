#!/bin/bash
# Aggregate -> parameter movement -> figures -> image grids -> report + slides.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/exp_config.sh"
source "${PILOT_VENV}/bin/activate"
cd "${PILOT_REPO_ROOT}"

SEED="${1:?usage: run_analysis.sh <seed>}"
MODELS="${SEQ_ROOT}/models/seed${SEED}"
ANA="${SEQ_ROOT}/analysis/seed${SEED}"
REPO_OUT="${PILOT_REPO_ROOT}/results/seq${SEED}"

# The evaluation root MUST follow the seed. run_seed.sh writes seed 17 to
# eval/ (the original layout) and every later seed to eval_seed<N>/. This was
# previously hard-coded to eval/, so `run_analysis.sh 29` would have analysed
# seed 17's images and labelled the output seed 29.
if [ "$SEED" = "17" ]; then EVALR="${SEQ_ROOT}/eval"; else EVALR="${SEQ_ROOT}/eval_seed${SEED}"; fi
[ -d "$EVALR" ] || { echo "FATAL: eval root $EVALR does not exist" >&2; exit 1; }

# Refuse to analyse an incomplete seed: a short detections.jsonl would otherwise
# be aggregated into rates that look finished.
EXPECTED_IMAGES="${SEQ_EXPECTED_IMAGES:-280}"
short=()
for ck in M0 MA MAB MAB_L2 MAC MAC_L2; do
    n=$(wc -l < "${EVALR}/${ck}/detections.jsonl" 2>/dev/null || echo 0)
    [ "$n" -eq "$EXPECTED_IMAGES" ] || short+=("${ck}:${n}/${EXPECTED_IMAGES}")
done
if [ "${#short[@]}" -gt 0 ]; then
    echo "FATAL: seed ${SEED} is not completely evaluated: ${short[*]}" >&2
    echo "Re-run scripts/seq/run_seed.sh ${SEED} before aggregating." >&2
    exit 1
fi
mkdir -p "$ANA" "$REPO_OUT"
echo "seed ${SEED}: analysing ${EVALR}"

echo "=== 1/5 aggregate ==="
python scripts/seq/aggregate.py --eval_root "$EVALR" \
    --out_dir "$ANA" --training_seed "$SEED" || exit 1

echo "=== 2/5 parameter movement ==="
python scripts/seq/param_movement.py --models_root "$MODELS" \
    --base_model_dir "$SEQ_BASE_MODEL" --out "${ANA}/param_movement.json"

echo "=== 3/5 figures ==="
python scripts/seq/make_figures.py --contrasts "${ANA}/contrasts.json" \
    --out_dir "${ANA}/figures" || exit 1

echo "=== 4/5 image grids (labelled + blinded) ==="
python scripts/seq/make_grids.py --eval_root "$EVALR" \
    --out_dir "${ANA}/grids"

echo "=== 5/5 report + slide outline ==="
python scripts/seq/make_report.py \
    --contrasts "${ANA}/contrasts.json" --models_root "$MODELS" \
    --calib "${SEQ_ROOT}/calib/calibration_report.json" \
    --reload_check "${SEQ_ROOT}/eval/_reload_check/report.json" \
    --anchor_cache "${SEQ_ANCHOR_ROOT}/Horses/cache_meta.json" \
    --movement "${ANA}/param_movement.json" \
    --out "${ANA}/RESULTS.md" --slides "${ANA}/SLIDE_OUTLINE.md" \
    --training_seed "$SEED" || exit 1

# --- copy the small artefacts into the repo ---------------------------------
mkdir -p "${REPO_OUT}/figures" "${REPO_OUT}/train" "${REPO_OUT}/grids"
cp "${ANA}"/{contrasts.json,rates.csv,summary.md,RESULTS.md,SLIDE_OUTLINE.md} "$REPO_OUT/" 2>/dev/null
cp "${ANA}/param_movement.json" "$REPO_OUT/" 2>/dev/null
cp "${ANA}/per_image.csv" "$REPO_OUT/" 2>/dev/null
cp "${ANA}/figures/"*.png "${ANA}/figures/"*.pdf "${ANA}/figures/"*.svg "${REPO_OUT}/figures/" 2>/dev/null
cp "${ANA}/grids/blinded/annotation_sheet.csv" "${ANA}/grids/blinded/README.md" "${REPO_OUT}/grids/" 2>/dev/null
for n in MA MAB MAB_L2 MAC MAC_L2; do
    [ -f "${MODELS}/${n}/train_report.json" ] && cp "${MODELS}/${n}/train_report.json" "${REPO_OUT}/train/${n}.json"
done
for ck in M0 MA MAB MAB_L2 MAC MAC_L2; do
    [ -f "${EVALR}/${ck}/detect_report.json" ] && \
        cp "${EVALR}/${ck}/detect_report.json" "${REPO_OUT}/train/detect_${ck}.json"
done

echo
echo "analysis dir : $ANA"
echo "repo copy    : $REPO_OUT"
du -sh "$REPO_OUT"
