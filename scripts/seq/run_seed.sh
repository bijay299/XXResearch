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

EXPECTED_IMAGES="${SEQ_EXPECTED_IMAGES:-280}"
FAILED=()           # names of stages that did not complete

verify_gpu() {
    local probe
    probe="$(GPU_SAMPLES=3 GPU_SAMPLE_INTERVAL=1 bash "${HERE}/../gpu_select.sh" || true)"
    grep -qE "^GPU $1 .*-> IDLE" <<<"$probe"
}

# ---- completion sentinels -------------------------------------------------
# Both are deliberately stricter than "the output path exists". train_request.py
# writes its report last, but a kill during that write leaves truncated JSON;
# detect.py opens detections.jsonl with mode "w" and appends per batch, so a
# kill leaves a SHORT but non-empty file. Existence-only guards would make such
# a partial artifact permanent and let the final summary call the seed complete.
train_complete() {   # train_complete <name>
    local r="${MODELS}/$1/train_report.json"
    [ -s "$r" ] || return 1
    [ -s "${MODELS}/$1/delta.bin" ] || return 1
    python - "$r" <<'PY' >/dev/null 2>&1 || return 1
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("finished_utc") and d.get("child_checkpoint", {}).get("sha256")
PY
    return 0
}

eval_complete() {    # eval_complete <checkpoint>
    local f="${EVALR}/$1/detections.jsonl"
    [ -s "$f" ] || return 1
    [ "$(wc -l < "$f")" -eq "$EXPECTED_IMAGES" ]
}

train() {   # train <gpu> <name> <anchor_name> <target> <adir> <aprompts> <l2> [parent]
    local gpu="$1" name="$2" an="$3" tgt="$4" adir="$5" ap="$6" l2="$7" parent="${8:-}"
    local extra=() rc=0
    [ -n "$parent" ] && extra=(--unet_ckpt "${MODELS}/${parent}/delta.bin")
    # Completion sentinel: a parseable report AND the delta it describes. A report
    # truncated by a kill, or a report with no weights beside it, is NOT complete.
    if train_complete "$name"; then echo "[skip] $name (already complete)"; return 0; fi
    if [ -e "${MODELS}/${name}/train_report.json" ]; then
        echo "[warn] ${name}: partial/unparseable train_report.json; retraining" >&2
    fi
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
    rc=$?   # capture BEFORE any other command overwrites $?
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL] ${name} exit=${rc} (see ${LOGS}/train_${name}.log)" >&2
        return "$rc"
    fi
    if ! train_complete "$name"; then
        echo "[FAIL] ${name} exited 0 but left no usable report/delta" >&2
        return 1
    fi
    echo "[done] ${name} exit=0"
    return 0
}

evaluate() {   # evaluate <gpu> <name>
    local gpu="$1" ck="$2" n=0
    # Guard on the expected ROW COUNT, never on mere existence (see the note on
    # the sentinels above): a short detections.jsonl is a partial run, not a
    # finished one, and must be re-run rather than silently reused.
    if eval_complete "$ck"; then
        echo "[skip] eval $ck (${EXPECTED_IMAGES} scored)"; return 0
    fi
    if [ -e "${EVALR}/${ck}/detections.jsonl" ]; then
        n=$(wc -l < "${EVALR}/${ck}/detections.jsonl" 2>/dev/null || echo 0)
        echo "[warn] ${ck}: partial detections.jsonl (${n}/${EXPECTED_IMAGES});" \
             "quarantining and re-running" >&2
        mv -f "${EVALR}/${ck}/detections.jsonl" \
              "${EVALR}/${ck}/detections.jsonl.partial.$(date +%s)" || return 1
    fi
    if [ ! -s "${MODELS}/${ck}/delta.bin" ]; then
        echo "[FAIL] eval ${ck}: ${MODELS}/${ck}/delta.bin missing or empty" >&2
        return 1
    fi
    if ! verify_gpu "$gpu"; then echo "[skip] GPU $gpu not idle for eval $ck" >&2; return 1; fi
    echo "[eval] seed${SEED} ${ck} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/seq/generate_eval_images.py \
        --manifest "${SEQ_ROOT}/eval_manifest.json" --base_model_dir "$SEQ_BASE_MODEL" \
        --unet_ckpt "${MODELS}/${ck}/delta.bin" --checkpoint_name "$ck" \
        --out_dir "${EVALR}/${ck}/images" --report "${EVALR}/${ck}/image_report.json" \
        > "${LOGS}/gen_${ck}.log" 2>&1 \
        || { echo "[FAIL] generate ${ck} (see ${LOGS}/gen_${ck}.log)" >&2; return 1; }
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/seq/detect.py \
        --image_report "${EVALR}/${ck}/image_report.json" --checkpoint_name "$ck" \
        --out "${EVALR}/${ck}/detections.jsonl" --report "${EVALR}/${ck}/detect_report.json" \
        > "${LOGS}/detect_${ck}.log" 2>&1 \
        || { echo "[FAIL] detect ${ck} (see ${LOGS}/detect_${ck}.log)" >&2; return 1; }
    if ! eval_complete "$ck"; then
        n=$(wc -l < "${EVALR}/${ck}/detections.jsonl" 2>/dev/null || echo 0)
        echo "[FAIL] eval ${ck}: detector exited 0 but scored ${n}/${EXPECTED_IMAGES}" >&2
        return 1
    fi
    echo "[done] eval ${ck}"
    return 0
}

# M0 is seed-independent: reuse the existing evaluation.
ln -sfn "${SEQ_ROOT}/eval/M0" "${EVALR}/M0"

train "$GA" MA horse cat "$SEQ_ANCHOR_HORSES" "$SEQ_PROMPTS_HORSES" 0
train_complete MA || { echo "FATAL: MA did not complete; no child can be trusted" >&2; exit 1; }

# Each background child's status is collected explicitly: `wait pidA pidB` only
# reports the status of the LAST pid, so a first-child failure would vanish.
train "$GA" MAB    horse dog "$SEQ_ANCHOR_HORSES" "$SEQ_PROMPTS_HORSES" 0 MA & P1=$!
train "$GB" MAB_L2 horse dog "$SEQ_ANCHOR_HORSES" "$SEQ_PROMPTS_HORSES" "$SEQ_L2SP_WEIGHT" MA & P2=$!
wait "$P1" || FAILED+=("train:MAB")
wait "$P2" || FAILED+=("train:MAB_L2")

train "$GA" MAC    flower sandwich "$SEQ_ANCHOR_FLOWERS" "$SEQ_PROMPTS_FLOWERS" 0 MA & P3=$!
train "$GB" MAC_L2 flower sandwich "$SEQ_ANCHOR_FLOWERS" "$SEQ_PROMPTS_FLOWERS" "$SEQ_L2SP_WEIGHT" MA & P4=$!
wait "$P3" || FAILED+=("train:MAC")
wait "$P4" || FAILED+=("train:MAC_L2")

# A subshell loop must not stop at the first failure, but it must still report
# one: propagate a non-zero status out of each lane.
( rc=0; for ck in MA MAB MAC;    do evaluate "$GA" "$ck" || rc=1; done; exit $rc ) & E1=$!
( rc=0; for ck in MAB_L2 MAC_L2; do evaluate "$GB" "$ck" || rc=1; done; exit $rc ) & E2=$!
wait "$E1" || FAILED+=("eval-lane-A")
wait "$E2" || FAILED+=("eval-lane-B")

# ---- final verification: completeness is re-derived from the artifacts, not
# ---- inferred from the fact that the script reached this line.
echo "=== SEED ${SEED} verification ==="
INCOMPLETE=()
for ck in MA MAB MAB_L2 MAC MAC_L2; do
    train_complete "$ck" || INCOMPLETE+=("train:${ck}")
done
for ck in M0 MA MAB MAB_L2 MAC MAC_L2; do
    n=$(wc -l < "${EVALR}/${ck}/detections.jsonl" 2>/dev/null || echo 0)
    if [ "$n" -eq "$EXPECTED_IMAGES" ]; then
        echo "  ${ck}: ${n}/${EXPECTED_IMAGES} scored  OK"
    else
        echo "  ${ck}: ${n}/${EXPECTED_IMAGES} scored  INCOMPLETE"
        INCOMPLETE+=("eval:${ck}")
    fi
done
PARTIALS=$(find "$EVALR" -name 'detections.jsonl.partial.*' 2>/dev/null | wc -l)
[ "$PARTIALS" -gt 0 ] && echo "  note: ${PARTIALS} quarantined partial detection file(s) present"

if [ "${#FAILED[@]}" -gt 0 ] || [ "${#INCOMPLETE[@]}" -gt 0 ]; then
    echo "=== SEED ${SEED} NOT COMPLETE ===" >&2
    [ "${#FAILED[@]}" -gt 0 ]     && echo "  stages that failed:     ${FAILED[*]}" >&2
    [ "${#INCOMPLETE[@]}" -gt 0 ] && echo "  artifacts incomplete:   ${INCOMPLETE[*]}" >&2
    echo "  Do NOT aggregate or report this seed until these are resolved." >&2
    exit 1
fi
echo "=== SEED ${SEED} DONE (all stages verified complete) ==="
