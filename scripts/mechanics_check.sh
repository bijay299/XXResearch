#!/bin/bash
# ---------------------------------------------------------------------------
# mechanics_check.sh
#
# SMALL SUBSTITUTE-CONFIGURATION MECHANICS CHECK. NOT A BASELINE.
# NOT a reproduction of any published result.
#
# Purpose: prove the CUIG ConAbl code path executes end-to-end on this host:
#   1. base model loads (diffusers pipeline)
#   2. minimal anchor-image generation runs
#   3. a handful of optimizer steps run with finite losses
#   4. delta.bin is written, and reloads into a fresh UNet
#   5. a few images generate from the reloaded checkpoint
#
# It uses base Stable Diffusion v1.5 as a STAND-IN generator because the
# UnlearnCanvas style50 generator could not be obtained on this host
# (see docs/ASSETS.md). Consequently:
#   - the model has NOT been fine-tuned on UnlearnCanvas styles,
#   - UA / IRA / CRA are NOT computed here and remain UNAVAILABLE,
#   - every timing/memory number applies ONLY to this substitute setup.
#
# Outputs land under $PILOT_DATA_ROOT/mechanics_check/, deliberately separate
# from the experiment output root, so substitute runs can never be mistaken for
# benchmark results.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/pilot_config.sh"

GPU=""
STYLE="Abstractionism"
ITERS=8
N_ANCHOR=16
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)   GPU="$2";      shift 2 ;;
        --style) STYLE="$2";    shift 2 ;;
        --iters) ITERS="$2";    shift 2 ;;
        --anchors) N_ANCHOR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

SUBSTITUTE_GENERATOR="${PILOT_DATA_ROOT}/Generators_substitute/sd-v1-5"
[[ -f "${SUBSTITUTE_GENERATOR}/model_index.json" ]] || {
    echo "ERROR: substitute generator missing at ${SUBSTITUTE_GENERATOR}" >&2; exit 4; }

CHECK_ROOT="${PILOT_DATA_ROOT}/mechanics_check"
OUTPUT_DIR="${CHECK_ROOT}/Models/${STYLE}"
# "images" segment required by evaluate.py's path handling; kept for consistency.
SAMPLE_DIR="${CHECK_ROOT}/Results/${STYLE}/images"
LOG_DIR="${CHECK_ROOT}/logs/${STYLE}"
ANCHOR_DIR="${CHECK_ROOT}/anchor_datasets/style/Laion"
ANCHOR_PROMPTS="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/style/Laion.txt"
mkdir -p "$OUTPUT_DIR" "$SAMPLE_DIR" "$LOG_DIR" "$ANCHOR_DIR"

# --- GPU selection, verified immediately before launch ----------------------
if [[ -z "$GPU" ]]; then
    GPU="$(bash "${HERE}/gpu_select.sh" --pick)" || {
        echo "ERROR: no idle GPU. Mechanics check is waiting for an available device." >&2
        exit 3
    }
fi
# Capture first, then match. Piping into `grep -q` under `set -o pipefail` makes
# grep close the pipe on its first match, killing gpu_select.sh with SIGPIPE and
# turning a successful match into a false "not idle".
GPU_PROBE="$(bash "${HERE}/gpu_select.sh" || true)"
echo "$GPU_PROBE"
if ! grep -qE "^GPU ${GPU} .*-> IDLE" <<<"$GPU_PROBE"; then
    echo "ERROR: GPU ${GPU} is not idle. Refusing to launch." >&2
    exit 3
fi
GPU_UUID="$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader -i "$GPU" | xargs)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" | xargs)"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "### MECHANICS CHECK pinned to GPU ${GPU} (${GPU_NAME}, ${GPU_UUID}) ###"
echo "### Substitute generator: ${SUBSTITUTE_GENERATOR} (base SD v1.5, NOT UnlearnCanvas) ###"

# --- Peak memory sampler over our own processes only ------------------------
PEAK_FILE="$(mktemp "${TMPDIR}/mech_peak_XXXXXX")"; echo 0 > "$PEAK_FILE"
ME="$(id -un)"
(
  while :; do
    tot=0
    while IFS=, read -r uuid pid used; do
        uuid="$(echo "$uuid" | xargs)"; pid="$(echo "$pid" | xargs)"
        used="$(echo "$used" | tr -dc '0-9')"
        [[ "$uuid" != "$GPU_UUID" || -z "$pid" || -z "$used" ]] && continue
        owner="$(ps -o user= -p "$pid" 2>/dev/null | xargs)"
        [[ "$owner" == "$ME" ]] && tot=$(( tot + used ))
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
    prev="$(cat "$PEAK_FILE" 2>/dev/null || echo 0)"
    (( tot > prev )) && echo "$tot" > "$PEAK_FILE"
    sleep 2
  done
) & SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null || true' EXIT

# shellcheck disable=SC1091
source "${PILOT_VENV}/bin/activate"

# =============== STAGE 1+2+3: anchor generation + optimizer steps ===========
echo
echo "############ STAGE A: anchor generation + ${ITERS} optimizer steps ############"
cd "${REPO_ROOT}/UnlearningMethods/ConAbl"
t0=$SECONDS
# Hyperparameters here are deliberately MINIMAL (not the released config):
# upstream uses --iterations 1000 --num_anchor_images 200 --num_anchor_prompts 200.
accelerate launch \
    --config_file "${PILOT_REPO_ROOT}/configs/accelerate_single_visible_gpu.yaml" \
    train_conabl.py \
    --anchor_target_concepts "${STYLE}" \
    --concept_type "style" \
    --output_dir "${OUTPUT_DIR}" \
    --base_model_dir "${SUBSTITUTE_GENERATOR}" \
    --iterations "${ITERS}" \
    --epochs 4 \
    --num_anchor_images "${N_ANCHOR}" \
    --num_anchor_prompts "${N_ANCHOR}" \
    --anchor_dataset_dirs "${ANCHOR_DIR}" \
    --anchor_prompt_paths "${ANCHOR_PROMPTS}" \
    --scale_lr \
    --hflip \
    --noaug \
    --enable_xformers_memory_efficient_attention \
    --overwrite_existing_ckpt \
    2>&1 | tee "${LOG_DIR}/train.log"
TRAIN_SECONDS=$(( SECONDS - t0 ))
echo "### STAGE A done in ${TRAIN_SECONDS}s ###"

# =============== STAGE 4: checkpoint save/reload verification ===============
echo
echo "############ STAGE B: checkpoint save/reload ############"
t0=$SECONDS
python "${HERE}/verify_checkpoint.py" \
    --delta "${OUTPUT_DIR}/delta.bin" \
    --base_model_dir "${SUBSTITUTE_GENERATOR}" \
    --out "${LOG_DIR}/checkpoint_verify.json" \
    2>&1 | tee "${LOG_DIR}/checkpoint_verify.log"
CKPT_SECONDS=$(( SECONDS - t0 ))

# =============== STAGE 5: generate a few images from the checkpoint =========
echo
echo "############ STAGE C: sample 4 images from reloaded checkpoint ############"
cd "${REPO_ROOT}/Evaluation/UnlearnCanvas"
t0=$SECONDS
python sample.py \
    --unet_ckpt_path "${OUTPUT_DIR}/delta.bin" \
    --output_dir "${SAMPLE_DIR}" \
    --styles_subset "[\"${STYLE}\", \"Crayon\"]" \
    --objects_subset '["Architectures", "Horses"]' \
    --seed 188 \
    --pipeline_dir "${SUBSTITUTE_GENERATOR}" \
    2>&1 | tee "${LOG_DIR}/sample.log"
SAMPLE_SECONDS=$(( SECONDS - t0 ))
echo "### STAGE C done in ${SAMPLE_SECONDS}s ###"

# =============== Record ====================================================
PEAK_MIB="$(cat "$PEAK_FILE" 2>/dev/null || echo 0)"
UPSTREAM_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
N_IMG="$(find "$SAMPLE_DIR" -maxdepth 1 -name '*.jpg' | wc -l)"

python - "${CHECK_ROOT}/mechanics_check_report.json" <<EOF
import json, os, re, sys

log = open("${LOG_DIR}/train.log", errors="replace").read()
losses = [float(x) for x in re.findall(r"'loss': ([0-9eE.+-]+)", log)]
if not losses:
    losses = [float(x) for x in re.findall(r"loss=([0-9eE.+-]+)", log)]
import math
finite = all(math.isfinite(v) for v in losses) if losses else None

verify = {}
vp = "${LOG_DIR}/checkpoint_verify.json"
if os.path.exists(vp):
    verify = json.load(open(vp))

rep = {
  "RUN_KIND": "MECHANICS CHECK ON SUBSTITUTE CONFIGURATION",
  "IS_BASELINE": False,
  "IS_REPRODUCTION": False,
  "warning": ("Generator is base Stable Diffusion v1.5, NOT the UnlearnCanvas "
              "style50 generator. No UnlearnCanvas classifiers available. "
              "UA/IRA/CRA are UNAVAILABLE. Timings/memory apply only to this "
              "substitute configuration."),
  "finished": "$(date -Is)",
  "host": "$(hostname)",
  "upstream_cuig_commit": "${UPSTREAM_COMMIT}",
  "gpu": {"index": "${GPU}", "name": "${GPU_NAME}", "uuid": "${GPU_UUID}",
          "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}"},
  "substitute_generator": "${SUBSTITUTE_GENERATOR}",
  "config": {"target_style": "${STYLE}", "optimizer_steps": ${ITERS},
             "num_anchor_images": ${N_ANCHOR}, "num_anchor_prompts": ${N_ANCHOR},
             "parameter_group": "kv-xattn (upstream default)",
             "note": "released config is iterations=1000, anchors=200; reduced here on purpose"},
  "timings_seconds": {"train_incl_anchor_generation": ${TRAIN_SECONDS},
                      "checkpoint_verify": ${CKPT_SECONDS},
                      "sample_4_images": ${SAMPLE_SECONDS}},
  "peak_gpu_mem_mib_own_procs": ${PEAK_MIB},
  "losses": {"n_logged": len(losses), "all_finite": finite,
             "first": losses[0] if losses else None,
             "last": losses[-1] if losses else None,
             "min": min(losses) if losses else None,
             "max": max(losses) if losses else None},
  "checkpoint_verify": verify,
  "images_generated": ${N_IMG},
  "image_dir": "${SAMPLE_DIR}",
  "metrics_UA_IRA_CRA": "UNAVAILABLE - UnlearnCanvas classifiers not obtainable on this host",
}
json.dump(rep, open(sys.argv[1], "w"), indent=2)
print(json.dumps(rep, indent=2))
EOF

echo
echo "### MECHANICS CHECK report: ${CHECK_ROOT}/mechanics_check_report.json ###"
echo "### REMINDER: substitute configuration. Not a baseline. UA/IRA/CRA unavailable. ###"
