#!/bin/bash
# ---------------------------------------------------------------------------
# SD-1.5 exploratory sequential object-erasure pilot - shared configuration.
#
# This is a SEPARATE experiment from the (blocked) UnlearnCanvas baseline. It
# intentionally uses base Stable Diffusion v1.5 as its backbone, and a public
# COCO object detector as its evaluator. Nothing here produces UA/IRA/CRA and
# nothing here is a reproduction of a published result.
# ---------------------------------------------------------------------------

export PILOT_REPO_ROOT="${PILOT_REPO_ROOT:-/home/bijaypandey/FINMLResearch}"
export PILOT_DATA_ROOT="${PILOT_DATA_ROOT:-/data/bijaypandey/cuig_pilot}"
export SEQ_ROOT="${SEQ_ROOT:-${PILOT_DATA_ROOT}/seq_pilot}"

export CUIG_REPO_ROOT="${CUIG_REPO_ROOT:-${PILOT_DATA_ROOT}/CUIG}"
export REPO_ROOT="${CUIG_REPO_ROOT}"
export PILOT_VENV="${PILOT_VENV:-${PILOT_DATA_ROOT}/venv-cuig}"

# M0: the untouched backbone for this experiment (already downloaded).
export SEQ_BASE_MODEL="${SEQ_BASE_MODEL:-${PILOT_DATA_ROOT}/Generators_substitute/sd-v1-5}"

# Caches off the full root filesystem.
export HF_HOME="${HF_HOME:-${PILOT_DATA_ROOT}/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${PILOT_DATA_ROOT}/torch_home}"
export TMPDIR="${TMPDIR:-/data/bijaypandey/tmp}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

# Accelerate config keeps gpu_ids: all so our CUDA_VISIBLE_DEVICES pin survives.
export SEQ_ACCEL_CFG="${PILOT_REPO_ROOT}/configs/accelerate_single_visible_gpu.yaml"

# --- Fixed training protocol ------------------------------------------------
export SEQ_ITERATIONS="${SEQ_ITERATIONS:-1000}"   # upstream sequential uses 2000
export SEQ_ANCHOR_IMAGES=200
export SEQ_ANCHOR_PROMPTS=200
export SEQ_BATCH=4                                 # anchor_batch_size default
# 200 anchor images / batch 4 = 50 optimizer steps per epoch. Upstream's default
# --epochs 1 would therefore stop at 50 steps, far short of SEQ_ITERATIONS. We
# raise the epoch cap so the step budget is what actually binds; training still
# breaks at exactly SEQ_ITERATIONS optimizer steps.
export SEQ_EPOCHS="${SEQ_EPOCHS:-25}"              # 25*50 = 1250 >= 1000
export SEQ_L2SP_WEIGHT=25000
export SEQ_PARAM_GROUP="kv-xattn"                  # upstream default, held fixed

# --- Anchor caches (generated once from untouched M0, shared by all arms) ----
export SEQ_ANCHOR_SEED="${SEQ_ANCHOR_SEED:-17}"
export SEQ_ANCHOR_ROOT="${SEQ_ROOT}/anchor_caches"
export SEQ_ANCHOR_HORSES="${SEQ_ANCHOR_ROOT}/Horses"
export SEQ_ANCHOR_FLOWERS="${SEQ_ANCHOR_ROOT}/Flowers"
export SEQ_PROMPTS_HORSES="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/object/Horses.txt"
export SEQ_PROMPTS_FLOWERS="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/object/Flowers.txt"

# --- Evaluation generation settings (identical at every checkpoint) ---------
export SEQ_EVAL_STEPS=30
export SEQ_EVAL_GUIDANCE=7.5
export SEQ_EVAL_RES=512
export SEQ_EVAL_SEEDS="101 202 303 404"
export SEQ_EVAL_CATEGORIES="cat dog sandwich horse bird chair bicycle"
export SEQ_DET_THRESHOLDS="0.3 0.5 0.7"

# ---------------------------------------------------------------------------
# ONE shared evaluation-path policy, used by run_seed.sh and run_analysis.sh.
#
# History: seed 17 was evaluated into `eval/` before the multi-seed layout
# existed; every later seed goes to `eval_seed<N>/`. The two scripts had drifted
# apart -- run_seed.sh wrote `eval_seed17` while run_analysis.sh read `eval` --
# so a seed-17 re-run and its analysis addressed different directories. Both now
# call this function and neither chooses a layout on its own.
#
# The legacy directory is PRESERVED, never migrated or deleted. When both
# layouts exist for the same seed the situation is ambiguous and is REFUSED:
# picking one by existence is how stale evidence gets analysed as if it were new.
#
#   seq_eval_root <seed>   -> prints the directory, or fails with a message
# ---------------------------------------------------------------------------
export SEQ_LEGACY_EVAL_SEED="${SEQ_LEGACY_EVAL_SEED:-17}"

seq_eval_root() {
    local seed="${1:?seq_eval_root <seed>}"
    local canon="${SEQ_ROOT}/eval_seed${seed}"
    local legacy="${SEQ_ROOT}/eval"

    if [ "$seed" != "$SEQ_LEGACY_EVAL_SEED" ]; then
        printf '%s\n' "$canon"; return 0
    fi
    local has_c=0 has_l=0
    [ -d "$canon" ]  && has_c=1
    [ -d "$legacy" ] && has_l=1
    if [ "$has_c" = 1 ] && [ "$has_l" = 1 ]; then
        cat >&2 <<MSG
FATAL: ambiguous evaluation layout for seed ${seed}.
  legacy    : ${legacy}
  canonical : ${canon}
Both exist, so which one holds the current evidence cannot be decided here.
Resolve it deliberately -- neither directory will be chosen automatically and
neither will be deleted by this script. Either move the stale one aside, or set
SEQ_EVAL_ROOT_OVERRIDE to the directory you mean.
MSG
        return 1
    fi
    if [ "$has_l" = 1 ]; then
        echo "note: seed ${seed} uses the legacy evaluation directory ${legacy}" >&2
        printf '%s\n' "$legacy"; return 0
    fi
    printf '%s\n' "$canon"
}

# Resolve, honouring an explicit operator override.
seq_resolve_eval_root() {
    local seed="${1:?seq_resolve_eval_root <seed>}"
    if [ -n "${SEQ_EVAL_ROOT_OVERRIDE:-}" ]; then
        echo "note: SEQ_EVAL_ROOT_OVERRIDE in effect: ${SEQ_EVAL_ROOT_OVERRIDE}" >&2
        printf '%s\n' "$SEQ_EVAL_ROOT_OVERRIDE"; return 0
    fi
    seq_eval_root "$seed"
}

# Frozen evaluation manifest, and the generation settings every checkpoint must
# share. Validation compares a stage's recorded settings against these, so a
# checkpoint generated under different settings cannot pass as complete.
export SEQ_EVAL_MANIFEST="${SEQ_EVAL_MANIFEST:-${SEQ_ROOT}/eval_manifest.json}"
export SEQ_GEN_SETTINGS_CONTRACT="${SEQ_GEN_SETTINGS_CONTRACT:-${PILOT_REPO_ROOT}/configs/generation_settings.json}"
export SEQ_CKPT_CONTRACT="${SEQ_CKPT_CONTRACT:-${PILOT_REPO_ROOT}/configs/checkpoint_contract.json}"
export SEQ_VALIDATOR="${SEQ_VALIDATOR:-${PILOT_REPO_ROOT}/scripts/seq/validate_stage.py}"

# ---------------------------------------------------------------------------
# EXPLICIT legacy-evidence policy, scoped to SAVED ARTIFACTS, not seed numbers.
#
# The saved pilot (seeds 17 and 29, commit a9de625) was trained before
# train_request.py recorded optimizer_steps_completed, and the completion check
# in use at the time was circular. Those ten reports therefore carry NO evidence
# of training completion. They are still validated STRUCTURALLY in full.
#
# History of this knob. It used to be SEQ_LEGACY_TRAIN_SEEDS="17 29" -- a list
# of training-seed NUMBERS. That is not a property of saved evidence: a brand-new
# run using seed 17 or 29 inherited the same exception and could pass completion
# validation with no counter at all. The proposed new unregularised trajectories
# use exactly those two seeds, so the exception would have covered the very runs
# whose completion has to be proven.
#
# The exception is now bound to the EXACT saved artifacts by CONTENT: a
# train_report.json sha256 together with its delta.bin sha256, both recorded with
# explicit provenance in the registry below. Therefore:
#
#   * a newly produced run can never match (new report bytes, new checkpoint
#     bytes), so its counter is REQUIRED regardless of seed or directory label;
#   * a saved report that was edited, or a checkpoint swapped under a legacy
#     NAME, breaks the match and is held to the strict policy;
#   * a registered saved artifact keeps completion UNVERIFIED and is never
#     retrained, moved, replaced or backfilled on that basis;
#   * a counter that is PRESENT BUT SHORT fails in both policies.
#
# Setting SEQ_LEGACY_ARTIFACT_REGISTRY to the EMPTY STRING removes every
# exception: the counter is then required everywhere, the saved pilot included.
# Note the `${VAR-default}` form (not `${VAR:-default}`): an explicitly empty
# value is honoured rather than silently replaced by the default, which is what
# the seed-list version got wrong.
export SEQ_LEGACY_ARTIFACT_REGISTRY="${SEQ_LEGACY_ARTIFACT_REGISTRY-${PILOT_REPO_ROOT}/configs/legacy_training_artifacts.json}"
export SEQ_LEGACY_POLICY="${SEQ_LEGACY_POLICY:-${PILOT_REPO_ROOT}/scripts/seq/legacy_artifact_policy.py}"
export SEQ_BASE_MODEL_CONTRACT="${SEQ_BASE_MODEL_CONTRACT:-${PILOT_REPO_ROOT}/configs/base_model_contract.json}"

# The frozen expected identity of the evaluation manifest. Validation recomputes
# the manifest's content digest and compares it against THIS value, rather than
# against the digest field inside the manifest being validated.
export SEQ_EVAL_MANIFEST_SHA="${SEQ_EVAL_MANIFEST_SHA:-0020c81c4a4dd3580a6574c91b4aea1a7e349e9e89fc845ed8c2077d47564892}"

# seq_steps_evidence_full <models_root> <checkpoint>
#   -> "<policy>|<reason>" where policy is counter | legacy_optional
#
# The decision reads the ARTIFACT on disk and matches its content identity
# against the frozen registry. It takes no seed number and no directory label.
# Fail-closed: anything unexpected yields the strict `counter` policy.
seq_steps_evidence_full() {
    local mroot="${1:?seq_steps_evidence_full <models_root> <checkpoint>}"
    local ck="${2:?seq_steps_evidence_full <models_root> <checkpoint>}"
    local out=""
    if [ -n "${SEQ_LEGACY_ARTIFACT_REGISTRY}" ] && [ -f "$SEQ_LEGACY_POLICY" ]; then
        out="$(CUDA_VISIBLE_DEVICES="" python "$SEQ_LEGACY_POLICY" decide \
                 --models_root "$mroot" --checkpoint "$ck" \
                 --registry "$SEQ_LEGACY_ARTIFACT_REGISTRY" 2>/dev/null)"
    fi
    case "$out" in
        legacy_optional\|*) printf '%s\n' "$out" ;;
        counter\|*)         printf '%s\n' "$out" ;;
        *) printf 'counter|no usable legacy-artifact registry decision; the completed-step counter is required\n' ;;
    esac
}

# Just the policy word, for callers that do not want the reason.
seq_steps_evidence() {   # seq_steps_evidence <models_root> <checkpoint>
    local line
    line="$(seq_steps_evidence_full "$1" "$2")"
    printf '%s\n' "${line%%|*}"
}

# Is this checkpoint's delta.bin a REGISTERED saved artifact? Exit 0 = yes, and
# it must not be retrained over, moved or replaced whatever state its report is
# in. Keyed on the checkpoint alone: an edited report loses the legacy
# completed-step exception, but the saved checkpoint beside it is still saved
# evidence and overwriting it would destroy the pilot.
seq_registered_checkpoint() {   # seq_registered_checkpoint <models_root> <checkpoint>
    [ -n "${SEQ_LEGACY_ARTIFACT_REGISTRY}" ] || return 1
    [ -f "$SEQ_LEGACY_POLICY" ] || return 1
    CUDA_VISIBLE_DEVICES="" python "$SEQ_LEGACY_POLICY" protect \
        --models_root "$1" --checkpoint "$2" \
        --registry "$SEQ_LEGACY_ARTIFACT_REGISTRY" >/dev/null 2>&1
}
