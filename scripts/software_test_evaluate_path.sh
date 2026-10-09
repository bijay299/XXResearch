#!/bin/bash
# ---------------------------------------------------------------------------
# software_test_evaluate_path.sh
#
# SOFTWARE TEST of Evaluation/UnlearnCanvas/evaluate.py. NOT AN EXPERIMENT.
# Produces NO scientific result. Produces NO UA / IRA / CRA.
#
# The real UnlearnCanvas classifiers cannot be obtained on this host
# (docs/ASSETS.md), so this test builds SYNTHETIC, RANDOMLY-INITIALISED
# classifier heads with the shapes evaluate.py expects, purely to confirm that
# the evaluation code path executes: checkpoint loading, label indexing,
# per-image bookkeeping, summary computation and the Excel export.
#
# Any accuracy printed by this test is MEANINGLESS NOISE.
#
# Isolation guarantees:
#   * synthetic weights are written to  $PILOT_DATA_ROOT/software_tests/...
#     and NEVER to the benchmark path   $PILOT_DATA_ROOT/Checkpoints/Classifiers/...
#   * results are written to            $PILOT_DATA_ROOT/software_tests/...
#     and NEVER to the experiment output root
#   * the script refuses to run if the real classifiers are present, so it can
#     never shadow a genuine evaluation.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/pilot_config.sh"

GPU="${1:-}"
[[ -z "$GPU" ]] && { echo "Usage: $0 <idle-gpu-index>" >&2; exit 2; }

if [[ -f "${CUIG_UNLEARNCANVAS_CLASSIFIER_DIR}/style_classifier.pth" ]]; then
    echo "Real UnlearnCanvas classifiers are present at ${CUIG_UNLEARNCANVAS_CLASSIFIER_DIR}." >&2
    echo "Refusing to run the synthetic software test; run a real evaluation instead." >&2
    exit 1
fi

GPU_PROBE="$(bash "${HERE}/gpu_select.sh" || true)"
if ! grep -qE "^GPU ${GPU} .*-> IDLE" <<<"$GPU_PROBE"; then
    echo "ERROR: GPU ${GPU} is not idle. Refusing to launch." >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES="$GPU"

TEST_ROOT="${PILOT_DATA_ROOT}/software_tests/evaluate_path"
SYN_CLS="${TEST_ROOT}/synthetic_classifiers_DO_NOT_USE_FOR_RESULTS"
OUT="${TEST_ROOT}/metrics"
# The mechanics-check images are reused only as pixel input; their content is
# irrelevant to what this test checks.
IMAGES="${PILOT_DATA_ROOT}/mechanics_check/Results/Abstractionism/images"
mkdir -p "$SYN_CLS" "$OUT"

[[ -d "$IMAGES" ]] || { echo "ERROR: run scripts/mechanics_check.sh first (no images at $IMAGES)" >&2; exit 4; }

# shellcheck disable=SC1091
source "${PILOT_VENV}/bin/activate"

echo "### Building SYNTHETIC classifier heads (random weights, correct shapes) ###"
python - "$SYN_CLS" "${REPO_ROOT}" <<'EOF'
import sys, torch, timm
syn_dir, repo_root = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{repo_root}/Evaluation/UnlearnCanvas")
import constants

for task, n in (("style", len(constants.STYLES_AVAILABLE)),
                ("object", len(constants.OBJECTS_AVAILABLE))):
    # Same construction evaluate.py::load_task_classifier performs.
    m = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True)
    m.head = torch.nn.Linear(1024, n)
    torch.save({"model_state_dict": m.state_dict()}, f"{syn_dir}/{task}_classifier.pth")
    print(f"  wrote SYNTHETIC {task}_classifier.pth  (head out_features={n})")
EOF

echo
echo "### Running evaluate.py over the 4 mechanics-check images ###"
cd "${REPO_ROOT}/Evaluation/UnlearnCanvas"
# Subsets chosen so the (style x object x seed) grid exactly covers the 4 images
# that exist, leaving no 'image not found' warnings.
python evaluate.py \
    --input_dir "$IMAGES" \
    --output_dir "$OUT" \
    --eval_classifier_dir "$SYN_CLS" \
    --unlearn '["Abstractionism"]' \
    --retain '["Crayon"]' \
    --cross_retain '["Architectures", "Horses"]' \
    --seed 188 \
    2>&1 | tee "${TEST_ROOT}/evaluate.log"

echo
echo "### Asserting expected artefacts exist ###"
python - "$OUT" "${TEST_ROOT}/software_test_report.json" <<'EOF'
import json, os, sys, glob
out, report_path = sys.argv[1], sys.argv[2]
checks = {}
for f in ("style_results.json", "object_results.json", "summary.json"):
    checks[f] = os.path.exists(os.path.join(out, f))
xlsx = glob.glob(os.path.join(out, "*_table.xlsx"))
checks["xlsx_report"] = bool(xlsx)

summary = {}
if checks["summary.json"]:
    summary = json.load(open(os.path.join(out, "summary.json")))

style_res = json.load(open(os.path.join(out, "style_results.json"))) if checks["style_results.json"] else {}
n_scored = sum(style_res.get("eval_count", {}).values())

rep = {
    "TEST_KIND": "SOFTWARE TEST OF evaluate.py CODE PATH",
    "IS_EXPERIMENT": False,
    "classifiers": "SYNTHETIC, RANDOMLY INITIALISED - accuracies below are meaningless noise",
    "artefacts_written": checks,
    "images_scored_style_task": n_scored,
    "summary_keys_present": sorted(summary.keys()),
    "meaningless_summary_values": {k: summary.get(k) for k in ("UA", "IRA", "CRA")},
    "code_path_exercised": [
        "load_task_classifier (timm backbone + Linear head + model_state_dict)",
        "STYLES_AVAILABLE.index / OBJECTS_AVAILABLE.index label lookup",
        "_record_prediction img_path.split('images/') bookkeeping",
        "normalize_and_prune_results",
        "compute_unlearncanvas_summary",
        "write_excel_report",
    ],
    "PASS": all(checks.values()) and n_scored > 0,
}
json.dump(rep, open(report_path, "w"), indent=2)
print(json.dumps(rep, indent=2))
raise SystemExit(0 if rep["PASS"] else 1)
EOF

echo
echo "### SOFTWARE TEST complete. No scientific result produced. UA/IRA/CRA remain UNAVAILABLE. ###"
