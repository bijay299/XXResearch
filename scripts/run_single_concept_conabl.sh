#!/bin/bash
# ---------------------------------------------------------------------------
# run_single_concept_conabl.sh
#
# Direct, single-GPU, SINGLE-CONCEPT ConAbl run on UnlearnCanvas.
#
# Adapted from upstream CUIG:
#   BashScripts/Independent/Style/Base/ConAbl.sh
# which is a Slurm *submission generator*: it loops over 12 styles and emits one
# sbatch job each. This host has no Slurm and we deliberately want exactly one
# concept on exactly one GPU, so the sbatch wrapper is removed and the three
# stages (train / sample / evaluate) are executed inline.
#
# TRAINING HYPERPARAMETERS ARE UNCHANGED from the upstream script:
#   --iterations 1000  --num_anchor_images 200  --num_anchor_prompts 200
#   --scale_lr --hflip --noaug --enable_xformers_memory_efficient_attention
#   concept_type=style, anchor dataset/prompts = style/Laion,
#   everything else left at upstream argparse defaults
#   (lr 2e-6 pre-scaling, anchor_batch_size 4, parameter_group kv-xattn,
#    lr_scheduler constant, lr_warmup_steps 500, seed 42, epochs 1).
#
# DELIBERATE DEVIATIONS (see docs/DEVIATIONS.md):
#   1. No sbatch; single concept instead of a 12-style loop.
#   2. Accelerate config: gpu_ids 'all' instead of '0'. Upstream's gpu_ids:'0'
#      makes `accelerate launch` OVERWRITE CUDA_VISIBLE_DEVICES with "0", which
#      would silently move the job onto physical GPU 0 regardless of our pin.
#      Verified on this host. 'all' makes accelerate respect our pin. Every
#      training-relevant field (mixed_precision 'no', distributed_type 'NO',
#      num_processes 1) is identical to upstream.
#   3. Assets and outputs live on /data (the filesystem holding /home is full).
#
# Usage:
#   bash scripts/run_single_concept_conabl.sh --style Abstractionism [--gpu N]
#   bash scripts/run_single_concept_conabl.sh --style Abstractionism --stage train
#
# Stages: all (default) | train | sample | evaluate
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/pilot_config.sh"

STYLE=""
GPU=""
STAGE="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --style) STYLE="$2"; shift 2 ;;
        --gpu)   GPU="$2";   shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -z "$STYLE" ]] && { echo "ERROR: --style is required (e.g. --style Abstractionism)" >&2; exit 2; }

# --- Concept split: taken verbatim from upstream Independent/Style/Base ------
# (identical to Evaluation/UnlearnCanvas/constants.py XLSX_ALL_* splits)
RETAIN_STYLES=(Blossom_Season Rust Crayon Fauvism Superstring Red_Blue_Ink \
               Gorgeous_Love French Joy Greenfield Expressionism Impressionism)
RETAIN_OBJECTS=(Architectures Butterfly Flame Flowers Horses Human Sea Trees)

to_json() { local j="["; for i in "$@"; do j+="\"${i}\","; done; printf '%s' "${j%,}]"; }

UNLEARN_JSON="$(to_json "$STYLE")"
RETAIN_STYLES_JSON="$(to_json "${RETAIN_STYLES[@]}")"
RETAIN_OBJECTS_JSON="$(to_json "${RETAIN_OBJECTS[@]}")"
STYLES_SUBSET_JSON="$(to_json "$STYLE" "${RETAIN_STYLES[@]}")"

# --- Asset gate: refuse to produce numbers from unverified assets ------------
# Presence on disk is not enough. A classifier with a permuted class order, or a
# generator that was never fine-tuned on UnlearnCanvas, is shape-identical to the
# real thing and would yield confident, wrong UA/IRA/CRA. Only
# scripts/validate_assets.py may open this gate, by writing a receipt.
RECEIPT="${PILOT_REPO_ROOT}/results/assets/validation_receipt.json"
if ! python3 - "$RECEIPT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(f"ERROR: no asset validation receipt at {p}", file=sys.stderr)
    print("       run: python scripts/validate_assets.py --gpu <idle> --update-manifest",
          file=sys.stderr)
    raise SystemExit(1)
v = json.loads(p.read_text()).get("verdict", {})
if not v.get("assets_verified"):
    print(f"ERROR: benchmark assets are NOT verified (overall={v.get('overall')}).", file=sys.stderr)
    for k in ("label_mapping_confirmed", "heldout_accuracy_pass", "untouched_generator_pass"):
        print(f"         {k}: {v.get(k)}", file=sys.stderr)
    print("       UA/IRA/CRA must stay UNAVAILABLE. See docs/ASSETS.md.", file=sys.stderr)
    raise SystemExit(1)
print("Asset validation receipt OK: assets verified.")
PY
then
    echo "Refusing to run the baseline on unverified assets." >&2
    exit 5
fi

# --- GPU selection: must be explicit and verified immediately before launch --
if [[ -z "$GPU" ]]; then
    echo "### No --gpu given; probing for an idle GPU ###"
    GPU="$(bash "${HERE}/gpu_select.sh" --pick)" || {
        echo "ERROR: no idle GPU available. Not launching." >&2
        echo "Training is waiting for an available device." >&2
        exit 3
    }
fi
echo "### Re-verifying GPU ${GPU} is still idle immediately before launch ###"
# Capture first, then match. Piping into `grep -q` under `set -o pipefail` makes
# grep close the pipe on its first match, killing gpu_select.sh with SIGPIPE and
# turning a successful match into a false "no longer idle".
GPU_PROBE="$(bash "${HERE}/gpu_select.sh" || true)"
echo "$GPU_PROBE"
if ! grep -qE "^GPU ${GPU} .*-> IDLE" <<<"$GPU_PROBE"; then
    echo "ERROR: GPU ${GPU} is no longer idle. Refusing to launch." >&2
    exit 3
fi

GPU_UUID="$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader -i "$GPU" | xargs)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" | xargs)"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "### Pinned to GPU ${GPU} (${GPU_NAME}, ${GPU_UUID}) via CUDA_VISIBLE_DEVICES ###"

# --- Paths (mirror upstream's Independent/Style/Base/ConAbl layout) ----------
TRAIN_DIR="${REPO_ROOT}/UnlearningMethods/ConAbl"
EVAL_DIR="${REPO_ROOT}/Evaluation/UnlearnCanvas"
ACCEL_CFG="${PILOT_REPO_ROOT}/configs/accelerate_single_visible_gpu.yaml"
ANCHOR_DATASET_DIR="${PILOT_DATA_ROOT}/anchor_datasets/style/Laion"
ANCHOR_PROMPT_PATH="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/style/Laion.txt"

RUN_ROOT="${OUTPUT_ROOT}/Independent/Style/Base/ConAbl"
OUTPUT_DIR="${RUN_ROOT}/Models/${STYLE}"
RESULT_DIR="${RUN_ROOT}/Results/${STYLE}"
# NOTE: the "images" path segment is REQUIRED. Evaluation/UnlearnCanvas/evaluate.py
# does `img_path.split("images/")[1]` in _record_prediction, so an input directory
# without an "images/" segment raises IndexError. Upstream relies on this too.
SAMPLE_OUTPUT_DIR="${RESULT_DIR}/images"
METRICS_OUTPUT_DIR="${RESULT_DIR}/metrics"
LOG_DIR="${RUN_ROOT}/logs/${STYLE}"
mkdir -p "$OUTPUT_DIR" "$SAMPLE_OUTPUT_DIR" "$METRICS_OUTPUT_DIR" "$LOG_DIR" "$ANCHOR_DATASET_DIR"

MANIFEST="${RESULT_DIR}/run_manifest.json"

# --- Peak-memory sampler: tracks only OUR processes on the pinned GPU --------
# The GPU is shared, so total memory.used would mix in other users' jobs.
PEAK_FILE="$(mktemp "${TMPDIR}/peak_mem_XXXXXX")"
echo 0 > "$PEAK_FILE"
sample_peak_mem() {
    local gpu_uuid="$1" peak_file="$2" parent="$3"
    while kill -0 "$parent" 2>/dev/null; do
        local total=0
        while IFS=, read -r uuid pid used; do
            uuid="$(echo "$uuid" | xargs)"; pid="$(echo "$pid" | xargs)"
            used="$(echo "$used" | tr -dc '0-9')"
            [[ "$uuid" != "$gpu_uuid" || -z "$pid" || -z "$used" ]] && continue
            # Count only PIDs in our own process tree.
            if ps -o ppid=,pid= -p "$pid" >/dev/null 2>&1; then
                local owner; owner="$(ps -o user= -p "$pid" 2>/dev/null | xargs)"
                [[ "$owner" == "$(id -un)" ]] && total=$(( total + used ))
            fi
        done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
        local prev; prev="$(cat "$peak_file" 2>/dev/null || echo 0)"
        (( total > prev )) && echo "$total" > "$peak_file"
        sleep 3
    done
}
sample_peak_mem "$GPU_UUID" "$PEAK_FILE" $$ &
SAMPLER_PID=$!
cleanup() { kill "$SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# shellcheck disable=SC1091
source "${PILOT_VENV}/bin/activate"

TRAIN_SECONDS=""; SAMPLE_SECONDS=""; EVAL_SECONDS=""
RUN_START="$(date -Is)"

# =========================== TRAIN ==========================================
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    echo
    echo "############ TRAIN: unlearning style '${STYLE}' ############"
    cd "$TRAIN_DIR"
    train_args=(
        --anchor_target_concepts "${STYLE}"
        --concept_type "style"
        --output_dir "${OUTPUT_DIR}"
        --base_model_dir "${CUIG_UNLEARNCANVAS_GENERATOR_DIR}"
        --iterations 1000
        --num_anchor_images 200
        --num_anchor_prompts 200
        --anchor_dataset_dirs "${ANCHOR_DATASET_DIR}"
        --anchor_prompt_paths "${ANCHOR_PROMPT_PATH}"
        --scale_lr
        --hflip
        --noaug
        --enable_xformers_memory_efficient_attention
    )
    t0=$SECONDS
    accelerate launch --config_file "${ACCEL_CFG}" train_conabl.py "${train_args[@]}" \
        2>&1 | tee "${LOG_DIR}/train.log"
    TRAIN_SECONDS=$(( SECONDS - t0 ))
    echo "### TRAIN done in ${TRAIN_SECONDS}s -> ${OUTPUT_DIR}/delta.bin ###"
fi

# =========================== SAMPLE =========================================
if [[ "$STAGE" == "all" || "$STAGE" == "sample" ]]; then
    echo
    echo "############ SAMPLE: 13 styles x 8 objects x 5 seeds ############"
    cd "$EVAL_DIR"
    t0=$SECONDS
    python sample.py \
        --unet_ckpt_path "${OUTPUT_DIR}/delta.bin" \
        --output_dir "${SAMPLE_OUTPUT_DIR}" \
        --styles_subset "${STYLES_SUBSET_JSON}" \
        --objects_subset "${RETAIN_OBJECTS_JSON}" \
        --pipeline_dir "${CUIG_UNLEARNCANVAS_GENERATOR_DIR}" \
        2>&1 | tee "${LOG_DIR}/sample.log"
    SAMPLE_SECONDS=$(( SECONDS - t0 ))
    echo "### SAMPLE done in ${SAMPLE_SECONDS}s -> ${SAMPLE_OUTPUT_DIR} ###"
fi

# =========================== EVALUATE =======================================
if [[ "$STAGE" == "all" || "$STAGE" == "evaluate" ]]; then
    echo
    echo "############ EVALUATE: UnlearnCanvas style/object classifiers ############"
    if [[ ! -f "${CUIG_UNLEARNCANVAS_CLASSIFIER_DIR}/style_classifier.pth" ]]; then
        echo "ERROR: UnlearnCanvas classifiers not found at ${CUIG_UNLEARNCANVAS_CLASSIFIER_DIR}" >&2
        echo "       UA / IRA / CRA cannot be computed. See docs/ASSETS.md." >&2
        exit 4
    fi
    cd "$EVAL_DIR"
    t0=$SECONDS
    python evaluate.py \
        --input_dir "${SAMPLE_OUTPUT_DIR}" \
        --output_dir "${METRICS_OUTPUT_DIR}" \
        --eval_classifier_dir "${CUIG_UNLEARNCANVAS_CLASSIFIER_DIR}" \
        --unlearn "${UNLEARN_JSON}" \
        --retain "${RETAIN_STYLES_JSON}" \
        --cross_retain "${RETAIN_OBJECTS_JSON}" \
        2>&1 | tee "${LOG_DIR}/evaluate.log"
    EVAL_SECONDS=$(( SECONDS - t0 ))
    echo "### EVALUATE done in ${EVAL_SECONDS}s -> ${METRICS_OUTPUT_DIR} ###"
fi

# =========================== MANIFEST =======================================
PEAK_MIB="$(cat "$PEAK_FILE" 2>/dev/null || echo 0)"
UPSTREAM_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
CKPT="${OUTPUT_DIR}/delta.bin"
python - "$MANIFEST" <<EOF
import json, os, sys
m = {
  "run_started": "${RUN_START}",
  "run_finished": "$(date -Is)",
  "host": "$(hostname)",
  "stage": "${STAGE}",
  "concept_type": "style",
  "target_concept": "${STYLE}",
  "upstream_cuig_commit": "${UPSTREAM_COMMIT}",
  "gpu": {"index": "${GPU}", "name": "${GPU_NAME}", "uuid": "${GPU_UUID}",
          "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}"},
  "generator_dir": "${CUIG_UNLEARNCANVAS_GENERATOR_DIR}",
  "classifier_dir": "${CUIG_UNLEARNCANVAS_CLASSIFIER_DIR}",
  "timings_seconds": {"train": "${TRAIN_SECONDS}", "sample": "${SAMPLE_SECONDS}",
                      "evaluate": "${EVAL_SECONDS}"},
  "peak_gpu_mem_mib_own_procs": ${PEAK_MIB:-0},
  "checkpoint": {"path": "${CKPT}",
                 "exists": os.path.exists("${CKPT}"),
                 "bytes": os.path.getsize("${CKPT}") if os.path.exists("${CKPT}") else None},
  "outputs": {"images": "${SAMPLE_OUTPUT_DIR}", "metrics": "${METRICS_OUTPUT_DIR}",
              "logs": "${LOG_DIR}"},
}
with open(sys.argv[1], "w") as f:
    json.dump(m, f, indent=2)
print(json.dumps(m, indent=2))
EOF
echo
echo "### Manifest: ${MANIFEST} ###"
