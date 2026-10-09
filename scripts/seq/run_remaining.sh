#!/bin/bash
# ---------------------------------------------------------------------------
# Finish the seed-17 core: MAC + MAC_L2 training, then evaluation generation and
# detection for every checkpoint that still needs it.
#
# Two GPUs are used as independent slots. Each launch re-verifies that its device
# is genuinely idle (fail-closed selector) immediately beforehand, and nothing
# belonging to another user is ever touched.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/exp_config.sh"
source "${PILOT_VENV}/bin/activate"
cd "${PILOT_REPO_ROOT}"

MODELS="${SEQ_ROOT}/models/seed17"
LOGS="${SEQ_ROOT}/logs"
GA="${GPU_A:-2}"
GB="${GPU_B:-3}"

wait_for() { while [ ! -f "$1" ]; do sleep 15; done; }

verify_gpu() {   # verify_gpu <idx> ; returns 1 unless that GPU is IDLE
    local probe
    probe="$(GPU_SAMPLES=3 GPU_SAMPLE_INTERVAL=1 bash "${HERE}/../gpu_select.sh" || true)"
    grep -qE "^GPU $1 .*-> IDLE" <<<"$probe"
}

train_one() {    # train_one <gpu> <name> <target> <anchor_dir> <prompts> <l2>
    local gpu="$1" name="$2" target="$3" adir="$4" apath="$5" l2="$6"
    if ! verify_gpu "$gpu"; then
        echo "[skip] GPU $gpu not idle; not launching $name" >&2
        return 1
    fi
    echo "[launch] $name on GPU $gpu (l2sp=$l2)"
    CUDA_VISIBLE_DEVICES="$gpu" accelerate launch --config_file "$SEQ_ACCEL_CFG" \
        scripts/seq/train_request.py \
        --name "$name" --parent_name MA --anchor_name "$7" --target "$target" \
        --anchor_dataset_dir "$adir" --anchor_prompt_path "$apath" \
        --base_model_dir "$SEQ_BASE_MODEL" --output_dir "${MODELS}/${name}" \
        --unet_ckpt "${MODELS}/MA/delta.bin" \
        --iterations "$SEQ_ITERATIONS" --epochs "$SEQ_EPOCHS" --seed 17 \
        --l2sp_weight "$l2" --report "${MODELS}/${name}/train_report.json" \
        > "${LOGS}/train_${name}.log" 2>&1
    echo "[done] $name exit=$?"
}

eval_one() {     # eval_one <gpu> <checkpoint>
    local gpu="$1" ck="$2" ckpt_arg=()
    [ "$ck" != "M0" ] && ckpt_arg=(--unet_ckpt "${MODELS}/${ck}/delta.bin")
    if [ -f "${SEQ_ROOT}/eval/${ck}/detections.jsonl" ]; then
        echo "[skip] $ck already evaluated"; return 0
    fi
    if ! verify_gpu "$gpu"; then
        echo "[skip] GPU $gpu not idle; not evaluating $ck" >&2; return 1
    fi
    echo "[eval] $ck on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/seq/generate_eval_images.py \
        --manifest "${SEQ_ROOT}/eval_manifest.json" \
        --base_model_dir "$SEQ_BASE_MODEL" "${ckpt_arg[@]}" \
        --checkpoint_name "$ck" --out_dir "${SEQ_ROOT}/eval/${ck}/images" \
        --report "${SEQ_ROOT}/eval/${ck}/image_report.json" \
        > "${LOGS}/gen_${ck}.log" 2>&1 || { echo "[FAIL] gen $ck" >&2; return 1; }
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/seq/detect.py \
        --image_report "${SEQ_ROOT}/eval/${ck}/image_report.json" \
        --checkpoint_name "$ck" --out "${SEQ_ROOT}/eval/${ck}/detections.jsonl" \
        --report "${SEQ_ROOT}/eval/${ck}/detect_report.json" \
        > "${LOGS}/detect_${ck}.log" 2>&1 || { echo "[FAIL] detect $ck" >&2; return 1; }
    echo "[done] eval $ck"
}

echo "=== waiting for MAB / MAB_L2 ==="
wait_for "${MODELS}/MAB/train_report.json"
wait_for "${MODELS}/MAB_L2/train_report.json"
echo "=== MAB and MAB_L2 complete ==="

echo "=== training MAC and MAC_L2 ==="
train_one "$GA" MAC    sandwich "$SEQ_ANCHOR_FLOWERS" "$SEQ_PROMPTS_FLOWERS" 0 flower &
P1=$!
train_one "$GB" MAC_L2 sandwich "$SEQ_ANCHOR_FLOWERS" "$SEQ_PROMPTS_FLOWERS" "$SEQ_L2SP_WEIGHT" flower &
P2=$!
wait $P1 $P2
echo "=== second-request training complete ==="

echo "=== evaluation: 2 slots ==="
( for ck in MA MAB MAC;     do eval_one "$GA" "$ck"; done ) &
E1=$!
( for ck in MAB_L2 MAC_L2;  do eval_one "$GB" "$ck"; done ) &
E2=$!
wait $E1 $E2

echo "=== ALL STAGES DONE ==="
for ck in M0 MA MAB MAB_L2 MAC MAC_L2; do
    n=$(wc -l < "${SEQ_ROOT}/eval/${ck}/detections.jsonl" 2>/dev/null || echo 0)
    echo "  ${ck}: ${n} scored images"
done
