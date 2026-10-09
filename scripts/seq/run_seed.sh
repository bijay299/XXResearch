#!/bin/bash
# ---------------------------------------------------------------------------
# Run one complete training-seed core: MA, then MAB/MAB_L2, then MAC/MAC_L2,
# then evaluation for each new checkpoint.
#
# Shares the anchor caches and the frozen evaluation manifest with every other
# seed, and reuses the M0 evaluation (M0 does not depend on the training seed).
# Each launch re-verifies its GPU is genuinely idle first.
#
#   bash scripts/seq/run_seed.sh 29
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/exp_config.sh"
source "${PILOT_VENV}/bin/activate"
cd "${PILOT_REPO_ROOT}"

SEED="${1:?usage: run_seed.sh <seed>}"
MODELS="${SEQ_ROOT}/models/seed${SEED}"
EVALR="${SEQ_ROOT}/eval_seed${SEED}"
LOGS="${SEQ_ROOT}/logs/seed${SEED}"
mkdir -p "$MODELS" "$EVALR" "$LOGS"
GA="${GPU_A:-2}"; GB="${GPU_B:-3}"

verify_gpu() {
    local probe
    probe="$(GPU_SAMPLES=3 GPU_SAMPLE_INTERVAL=1 bash "${HERE}/../gpu_select.sh" || true)"
    grep -qE "^GPU $1 .*-> IDLE" <<<"$probe"
}

train() {   # train <gpu> <name> <anchor_name> <target> <adir> <aprompts> <l2> [parent]
    local gpu="$1" name="$2" an="$3" tgt="$4" adir="$5" ap="$6" l2="$7" parent="${8:-}"
    local extra=()
    [ -n "$parent" ] && extra=(--unet_ckpt "${MODELS}/${parent}/delta.bin")
    if [ -f "${MODELS}/${name}/train_report.json" ]; then echo "[skip] $name"; return 0; fi
    if ! verify_gpu "$gpu"; then echo "[skip] GPU $gpu not idle for $name" >&2; return 1; fi
    echo "[train] seed${SEED} ${name} on GPU ${gpu} (l2sp=${l2})"
    CUDA_VISIBLE_DEVICES="$gpu" accelerate launch --config_file "$SEQ_ACCEL_CFG" \
        scripts/seq/train_request.py \
        --name "$name" --parent_name "${parent:-M0}" --anchor_name "$an" --target "$tgt" \
        --anchor_dataset_dir "$adir" --anchor_prompt_path "$ap" \
        --base_model_dir "$SEQ_BASE_MODEL" --output_dir "${MODELS}/${name}" \
        "${extra[@]}" --iterations "$SEQ_ITERATIONS" --epochs "$SEQ_EPOCHS" \
        --seed "$SEED" --l2sp_weight "$l2" \
        --report "${MODELS}/${name}/train_report.json" \
        > "${LOGS}/train_${name}.log" 2>&1
    echo "[done] ${name} exit=$?"
}

evaluate() {   # evaluate <gpu> <name>
    local gpu="$1" ck="$2"
    [ -f "${EVALR}/${ck}/detections.jsonl" ] && { echo "[skip] eval $ck"; return 0; }
    if ! verify_gpu "$gpu"; then echo "[skip] GPU $gpu not idle for eval $ck" >&2; return 1; fi
    echo "[eval] seed${SEED} ${ck} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/seq/generate_eval_images.py \
        --manifest "${SEQ_ROOT}/eval_manifest.json" --base_model_dir "$SEQ_BASE_MODEL" \
        --unet_ckpt "${MODELS}/${ck}/delta.bin" --checkpoint_name "$ck" \
        --out_dir "${EVALR}/${ck}/images" --report "${EVALR}/${ck}/image_report.json" \
        > "${LOGS}/gen_${ck}.log" 2>&1 || return 1
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/seq/detect.py \
        --image_report "${EVALR}/${ck}/image_report.json" --checkpoint_name "$ck" \
        --out "${EVALR}/${ck}/detections.jsonl" --report "${EVALR}/${ck}/detect_report.json" \
        > "${LOGS}/detect_${ck}.log" 2>&1 || return 1
    echo "[done] eval ${ck}"
}

# M0 is seed-independent: reuse the existing evaluation.
ln -sfn "${SEQ_ROOT}/eval/M0" "${EVALR}/M0"

train "$GA" MA horse cat "$SEQ_ANCHOR_HORSES" "$SEQ_PROMPTS_HORSES" 0
[ -f "${MODELS}/MA/delta.bin" ] || { echo "FATAL: MA missing" >&2; exit 1; }

train "$GA" MAB    horse dog "$SEQ_ANCHOR_HORSES" "$SEQ_PROMPTS_HORSES" 0 MA & P1=$!
train "$GB" MAB_L2 horse dog "$SEQ_ANCHOR_HORSES" "$SEQ_PROMPTS_HORSES" "$SEQ_L2SP_WEIGHT" MA & P2=$!
wait $P1 $P2

train "$GA" MAC    flower sandwich "$SEQ_ANCHOR_FLOWERS" "$SEQ_PROMPTS_FLOWERS" 0 MA & P3=$!
train "$GB" MAC_L2 flower sandwich "$SEQ_ANCHOR_FLOWERS" "$SEQ_PROMPTS_FLOWERS" "$SEQ_L2SP_WEIGHT" MA & P4=$!
wait $P3 $P4

( for ck in MA MAB MAC;    do evaluate "$GA" "$ck"; done ) & E1=$!
( for ck in MAB_L2 MAC_L2; do evaluate "$GB" "$ck"; done ) & E2=$!
wait $E1 $E2

echo "=== SEED ${SEED} DONE ==="
for ck in M0 MA MAB MAB_L2 MAC MAC_L2; do
    echo "  ${ck}: $(wc -l < "${EVALR}/${ck}/detections.jsonl" 2>/dev/null || echo 0) scored"
done
