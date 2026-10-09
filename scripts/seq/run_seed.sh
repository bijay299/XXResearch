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
#
# Completion is decided by scripts/seq/validate_stage.py, which reads the
# artifacts and checks them against the frozen contracts AND against the
# settings of the request being run. Existence and row counts are not evidence:
# the earlier shell sentinels accepted a delta.bin holding plain text, a
# detections.jsonl of 280 identical rows, and one of 280 malformed lines.
#
# Nothing is ever deleted. Artifacts that fail validation are quarantined
# alongside a reason file, and the script exits non-zero if any required stage
# did not complete.
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/exp_config.sh"
source "${PILOT_VENV}/bin/activate"
cd "${PILOT_REPO_ROOT}"

SEED="${1:?usage: run_seed.sh <seed>}"
MODELS="${SEQ_ROOT}/models/seed${SEED}"

# ONE shared path policy (defined in exp_config.sh), so this script and
# run_analysis.sh can never address different directories for the same seed.
EVALR="$(seq_resolve_eval_root "$SEED")" || exit 1
LOGS="${SEQ_ROOT}/logs/seed${SEED}"
mkdir -p "$MODELS" "$EVALR" "$LOGS"
GA="${GPU_A:-2}"; GB="${GPU_B:-3}"
FAILED=()

echo "seed ${SEED}: models=${MODELS}"
echo "seed ${SEED}: eval  =${EVALR}"

for f in "$SEQ_EVAL_MANIFEST" "$SEQ_CKPT_CONTRACT" "$SEQ_GEN_SETTINGS_CONTRACT" \
         "$SEQ_VALIDATOR"; do
    [ -s "$f" ] || { echo "FATAL: required contract/validator missing: $f" >&2; exit 1; }
done
# The frozen expected identity comes from CONFIGURATION, never from the
# manifest being validated: reading the claimed digest out of that same file
# would make content tampering invisible.
MANIFEST_SHA="${SEQ_EVAL_MANIFEST_SHA:?SEQ_EVAL_MANIFEST_SHA must be set}"
STEPS_EVIDENCE="$(seq_steps_evidence "$SEED")"
echo "seed ${SEED}: manifest identity ${MANIFEST_SHA:0:12}..., steps evidence=${STEPS_EVIDENCE}"
if [ "$STEPS_EVIDENCE" = "legacy_optional" ]; then
    echo "seed ${SEED}: declared LEGACY (SEQ_LEGACY_TRAIN_SEEDS) -- training completion" \
         "will be reported UNVERIFIED where no step counter exists; structural" \
         "validation still applies and nothing is retrained on that basis."
fi

# sha256 of a checkpoint currently on disk, so an evaluation can be bound to it.
ckpt_sha() {   # ckpt_sha <name>
    local f="${MODELS}/$1/delta.bin"
    [ -s "$f" ] || return 1
    sha256sum "$f" | cut -d" " -f1
}

# The idle-GPU selector is a seam so the CPU test suite can substitute a mock.
# It defaults to the real fail-closed selector; an override is only ever set by
# the test harness, never in a real run.
GPU_SELECT="${GPU_SELECT:-${HERE}/../gpu_select.sh}"
[ -x "$GPU_SELECT" ] || [ -f "$GPU_SELECT" ] || {
    echo "FATAL: GPU selector not found: $GPU_SELECT" >&2; exit 1; }

verify_gpu() {
    local probe
    probe="$(GPU_SAMPLES=3 GPU_SAMPLE_INTERVAL=1 bash "$GPU_SELECT" || true)"
    grep -qE "^GPU $1 .*-> IDLE" <<<"$probe"
}

# --- request table: the settings each checkpoint must have been trained with.
#     Passed to the validator so a stale report from a DIFFERENT configuration
#     cannot satisfy the completion check.
req_target() { case "$1" in MA) echo cat;;  MAB|MAB_L2) echo dog;;  MAC|MAC_L2) echo sandwich;; esac; }
req_anchor() { case "$1" in MA|MAB|MAB_L2) echo horse;; MAC|MAC_L2) echo flower;; esac; }
req_parent() { case "$1" in MA) echo M0;; *) echo MA;; esac; }
req_l2sp()   { case "$1" in *_L2) echo "$SEQ_L2SP_WEIGHT";; *) echo 0;; esac; }

validate_train() {   # validate_train <name>
    CUDA_VISIBLE_DEVICES="" python "$SEQ_VALIDATOR" train "$1" \
        --models_root "$MODELS" --contract "$SEQ_CKPT_CONTRACT" \
        --expect_seed "$SEED" --expect_parent "$(req_parent "$1")" \
        --expect_target "$(req_target "$1")" --expect_anchor "$(req_anchor "$1")" \
        --expect_l2sp "$(req_l2sp "$1")" --expect_steps "$SEQ_ITERATIONS" \
        --steps_evidence "$STEPS_EVIDENCE" \
        --write_marker
}

validate_eval() {    # validate_eval <checkpoint>
    local extra=() sha
    if [ "$1" = "M0" ]; then
        # M0 is one evaluation set reused by every seed, and it applies no
        # delta: its identity is the BASE MODEL, asserted via the base-model
        # contract. Both facts are declared, not left as an exception.
        extra=(--allow_shared_images --base_model_contract "$SEQ_BASE_MODEL_CONTRACT")
    else
        # Bind the images to the checkpoint on disk RIGHT NOW. Without this, an
        # evaluation generated from an older same-named checkpoint validates clean.
        if ! sha="$(ckpt_sha "$1")"; then
            echo "[FAIL] eval $1: ${MODELS}/$1/delta.bin missing or empty, so the" \
                 "evaluation cannot be bound to a checkpoint" >&2
            return 1
        fi
        extra=(--expect_sha "$sha")
    fi
    CUDA_VISIBLE_DEVICES="" python "$SEQ_VALIDATOR" eval "$1" \
        --eval_root "$EVALR" --manifest "$SEQ_EVAL_MANIFEST" \
        --expect_manifest_sha "$MANIFEST_SHA" \
        --expect_gen_settings "$SEQ_GEN_SETTINGS_CONTRACT" \
        --require_images "${extra[@]}" --write_marker
}

# Move a failed artifact aside with a reason. Evidence is preserved, never
# deleted: a re-run must not destroy the record of what went wrong.
quarantine() {   # quarantine <path> <reason>
    local p="$1" why="$2" dest
    [ -e "$p" ] || return 0
    dest="${p}.invalid.$(date +%Y%m%dT%H%M%S)"
    mv -f "$p" "$dest" || return 1
    printf '%s\n' "$why" > "${dest}.reason.txt"
    echo "[quarantine] $(basename "$p") -> $(basename "$dest")" >&2
}

train() {   # train <gpu> <name> <anchor_name> <target> <adir> <aprompts> <l2> [parent]
    local gpu="$1" name="$2" an="$3" tgt="$4" adir="$5" ap="$6" l2="$7" parent="${8:-}"
    local extra=() rc=0
    [ -n "$parent" ] && extra=(--unet_ckpt "${MODELS}/${parent}/delta.bin")

    if validate_train "$name" >/dev/null 2>&1; then
        echo "[skip] $name (validated complete)"; return 0
    fi
    if [ -e "${MODELS}/${name}/train_report.json" ]; then
        echo "[warn] ${name}: existing output does not validate; retraining" >&2
        validate_train "$name" 2>&1 | sed 's/^/        /' >&2
        quarantine "${MODELS}/${name}/train_report.json" \
                   "failed validate_stage.py train at $(date -Is); retrained after this"
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
    if ! validate_train "$name" 2>&1 | sed 's/^/        /'; then
        echo "[FAIL] ${name} exited 0 but its artifacts do not validate" >&2
        # Quarantine the freshly produced invalid report too (see the note in
        # evaluate). The delta.bin is left in place: it is the only copy of
        # whatever the run actually produced, and the report beside it is what
        # would otherwise make it look complete.
        quarantine "${MODELS}/${name}/train_report.json" \
                   "produced by this run but failed validate_stage.py train at $(date -Is)"
        return 1
    fi
    echo "[done] ${name} exit=0, validated"
    return 0
}

evaluate() {   # evaluate <gpu> <name>
    local gpu="$1" ck="$2" rc=0
    if validate_eval "$ck" >/dev/null 2>&1; then
        echo "[skip] eval $ck (validated complete)"; return 0
    fi
    if [ -e "${EVALR}/${ck}/detections.jsonl" ]; then
        echo "[warn] ${ck}: existing detections do not validate; re-running" >&2
        validate_eval "$ck" 2>&1 | sed 's/^/        /' >&2
        quarantine "${EVALR}/${ck}/detections.jsonl" \
                   "failed validate_stage.py eval at $(date -Is); re-run after this"
        quarantine "${EVALR}/${ck}/detect_report.json" "companion of the above"
    fi
    if ! CUDA_VISIBLE_DEVICES="" python "$SEQ_VALIDATOR" train "$ck" \
            --models_root "$MODELS" --contract "$SEQ_CKPT_CONTRACT" \
            --expect_seed "$SEED" --expect_parent "$(req_parent "$ck")" \
            --expect_target "$(req_target "$ck")" --expect_anchor "$(req_anchor "$ck")" \
            --expect_l2sp "$(req_l2sp "$ck")" --expect_steps "$SEQ_ITERATIONS" \
            --steps_evidence "$STEPS_EVIDENCE" \
            >/dev/null 2>&1; then
        echo "[FAIL] eval ${ck}: its training checkpoint does not validate; " \
             "refusing to evaluate an unverified checkpoint" >&2
        return 1
    fi
    if ! verify_gpu "$gpu"; then echo "[skip] GPU $gpu not idle for eval $ck" >&2; return 1; fi

    echo "[eval] seed${SEED} ${ck} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/seq/generate_eval_images.py \
        --manifest "$SEQ_EVAL_MANIFEST" --base_model_dir "$SEQ_BASE_MODEL" \
        --unet_ckpt "${MODELS}/${ck}/delta.bin" --checkpoint_name "$ck" \
        --out_dir "${EVALR}/${ck}/images" --report "${EVALR}/${ck}/image_report.json" \
        > "${LOGS}/gen_${ck}.log" 2>&1
    rc=$?
    [ "$rc" -ne 0 ] && { echo "[FAIL] generate ${ck} exit=${rc} (${LOGS}/gen_${ck}.log)" >&2; return "$rc"; }

    CUDA_VISIBLE_DEVICES="$gpu" python scripts/seq/detect.py \
        --image_report "${EVALR}/${ck}/image_report.json" --checkpoint_name "$ck" \
        --out "${EVALR}/${ck}/detections.jsonl" --report "${EVALR}/${ck}/detect_report.json" \
        > "${LOGS}/detect_${ck}.log" 2>&1
    rc=$?
    [ "$rc" -ne 0 ] && { echo "[FAIL] detect ${ck} exit=${rc} (${LOGS}/detect_${ck}.log)" >&2; return "$rc"; }

    if ! validate_eval "$ck" 2>&1 | sed 's/^/        /'; then
        echo "[FAIL] eval ${ck}: stages exited 0 but artifacts do not validate" >&2
        # Quarantine the freshly produced invalid artifact as well, not only a
        # stale one found at entry: leaving it in place would make the next run
        # rediscover the same failure, and an operator inspecting the directory
        # could mistake it for a usable result.
        quarantine "${EVALR}/${ck}/detections.jsonl" \
                   "produced by this run but failed validate_stage.py eval at $(date -Is)"
        quarantine "${EVALR}/${ck}/detect_report.json" "companion of the above"
        return 1
    fi
    echo "[done] eval ${ck}, validated"
    return 0
}

# M0 is seed-independent: reuse the existing evaluation rather than regenerate.
if [ -d "${SEQ_ROOT}/eval/M0" ] && [ "$(readlink -f "${EVALR}")" != "$(readlink -f "${SEQ_ROOT}/eval")" ]; then
    ln -sfn "${SEQ_ROOT}/eval/M0" "${EVALR}/M0"
fi

# The MA call itself is guarded, not only its artifacts: a failed launch and a
# launch that produced unusable output are different faults and both are fatal.
train "$GA" MA horse cat "$SEQ_ANCHOR_HORSES" "$SEQ_PROMPTS_HORSES" 0
MA_RC=$?
if [ "$MA_RC" -ne 0 ]; then
    echo "FATAL: MA training stage returned ${MA_RC}; no child can be trusted" >&2; exit 1
fi
if ! validate_train MA 2>&1 | sed 's/^/        /'; then
    echo "FATAL: MA artifacts do not validate; no child can be trusted" >&2; exit 1
fi

# `wait p1 p2` reports only the LAST pid's status, so each child is waited on
# separately and its status recorded.
train "$GA" MAB    horse dog "$SEQ_ANCHOR_HORSES" "$SEQ_PROMPTS_HORSES" 0 MA & P1=$!
train "$GB" MAB_L2 horse dog "$SEQ_ANCHOR_HORSES" "$SEQ_PROMPTS_HORSES" "$SEQ_L2SP_WEIGHT" MA & P2=$!
wait "$P1" || FAILED+=("train:MAB")
wait "$P2" || FAILED+=("train:MAB_L2")

train "$GA" MAC    flower sandwich "$SEQ_ANCHOR_FLOWERS" "$SEQ_PROMPTS_FLOWERS" 0 MA & P3=$!
train "$GB" MAC_L2 flower sandwich "$SEQ_ANCHOR_FLOWERS" "$SEQ_PROMPTS_FLOWERS" "$SEQ_L2SP_WEIGHT" MA & P4=$!
wait "$P3" || FAILED+=("train:MAC")
wait "$P4" || FAILED+=("train:MAC_L2")

# A lane must not stop at its first failure, but must still report one.
( rc=0; for ck in MA MAB MAC;    do evaluate "$GA" "$ck" || rc=1; done; exit $rc ) & E1=$!
( rc=0; for ck in MAB_L2 MAC_L2; do evaluate "$GB" "$ck" || rc=1; done; exit $rc ) & E2=$!
wait "$E1" || FAILED+=("eval-lane-A")
wait "$E2" || FAILED+=("eval-lane-B")

# ---- final verification: re-derived from the artifacts, never inferred from
# ---- the fact that the script reached this line.
echo "=== SEED ${SEED} verification ==="
INCOMPLETE=()
for ck in MA MAB MAB_L2 MAC MAC_L2; do
    if validate_train "$ck" >/dev/null 2>&1; then echo "  train ${ck}: VALID"
    else echo "  train ${ck}: INVALID"; INCOMPLETE+=("train:${ck}"); fi
done
for ck in M0 MA MAB MAB_L2 MAC MAC_L2; do
    if validate_eval "$ck" >/dev/null 2>&1; then echo "  eval  ${ck}: VALID"
    else echo "  eval  ${ck}: INVALID"; INCOMPLETE+=("eval:${ck}"); fi
done
Q=$(find "$EVALR" "$MODELS" -name '*.invalid.*' -not -name '*.reason.txt' 2>/dev/null | wc -l)
[ "$Q" -gt 0 ] && echo "  note: ${Q} quarantined artifact(s) retained for inspection"

if [ "${#FAILED[@]}" -gt 0 ] || [ "${#INCOMPLETE[@]}" -gt 0 ]; then
    echo "=== SEED ${SEED} NOT COMPLETE ===" >&2
    [ "${#FAILED[@]}" -gt 0 ]     && echo "  stages that failed:   ${FAILED[*]}" >&2
    [ "${#INCOMPLETE[@]}" -gt 0 ] && echo "  artifacts invalid:    ${INCOMPLETE[*]}" >&2
    echo "  Do NOT aggregate or report this seed until these are resolved." >&2
    exit 1
fi
echo "=== SEED ${SEED} DONE (all stages validated) ==="
