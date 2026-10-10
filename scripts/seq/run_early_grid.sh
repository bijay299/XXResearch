#!/bin/bash
# ---------------------------------------------------------------------------
# AMENDMENT 01: the fixed EARLY DUMP GRID at optimizer steps 10,20,...,90.
#
#   bash scripts/seq/run_early_grid.sh [stage]
#       stage: all (default) | preflight | smoke | train | dev | bridge
#              | select | test
#
# STATUS: this runner exists so the amendment can be CPU-verified before it is
# decided. It performs no GPU work until it is invoked, and the PI's amendment
# decision is recorded separately from the resource clarification -- having the
# GPUs free is not an approval of this protocol change.
#
# What it is. The same diagnostic, same selection rules, driven over a finer and
# EARLIER dump grid, because the completed 100-step scan put each seed's match
# point below step 100 where that grid has no point to offer.
#
# The ONLY two things that change in training, both declared before the run:
#
#   1. the STOPPING LIMIT      1000 -> 100 optimizer steps
#   2. the SAVE CADENCE        every 100 -> every 10 optimizer steps
#
# Everything that determines the first 100 updates is preserved: the saved MA
# parent (bound by hash), the training seed, the constant learning rate and its
# scaling, batch size, gradient accumulation, optimizer and all its moments,
# gradient clipping, precision, resolution, augmentation, parameter group,
# `l2sp_weight 0`, and the anchor set. The stopping limit is harmless ONLY
# because the learning-rate schedule is `constant`, which diffusers builds
# without reference to the horizon; bridge_check.py asserts that rather than
# trusting it.
#
# Upstream recomputes epochs = ceil(iterations / steps_per_epoch), so epochs_cap
# and epoch_capacity_steps change as CONSEQUENCES of the stopping limit. They are
# declared, not hidden.
#
# Comparability with the original run is established by MEASUREMENT at step 100,
# the one step both runs write: settings, weights and development behaviour
# (scripts/seq/bridge_check.py). The bridge carries a stop condition declared in
# advance, and this runner refuses to evaluate the frozen test set unless the
# bridge HELD.
#
# The original run is untouched: separate output root, and the saved pilot's MA
# and MAB_L2 are read-only and bound by hash as before.
#
# Resource policy (PI, 2026-10-10): no GPU-hour ceiling. Work starts only on
# verified idle GPUs, waits rather than pre-empting, and never touches another
# user's job. Usage is recorded, not capped.
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The shared configuration first, so the profile below is expressed in terms of
# the same SEQ_ROOT and the same measured cost model the original run used --
# rather than re-deriving those paths and risking a divergence.
source "${HERE}/exp_config.sh"
# The bridge check loads checkpoints with torch, so it needs the pilot
# interpreter exactly as the runner does.
# shellcheck disable=SC1091
[ -f "${PILOT_VENV}/bin/activate" ] && source "${PILOT_VENV}/bin/activate"

# The original run is where the bridge and the reused estimates come from. Taken
# BEFORE the profile overrides SEQ_DIAG_ROOT.
ORIGINAL_ROOT="${SEQ_EG_ORIGINAL_ROOT:-${SEQ_DIAG_ROOT}}"
BRIDGE_STEP="${SEQ_EG_BRIDGE_STEP:-100}"
BRIDGE_TOL_PP="${SEQ_EG_BRIDGE_TOL_PP:-5.0}"

# --- the amendment's profile -------------------------------------------------
# Exported, so run_diagnostic.sh's own `source exp_config.sh` -- which uses
# ${VAR:-default} throughout -- keeps these rather than the defaults.
export SEQ_DIAG_ROOT="${SEQ_EG_ROOT:-${SEQ_ROOT}/diag_v2_early_grid}"
export SEQ_DIAG_ITERATIONS="${SEQ_EG_ITERATIONS:-100}"     # declared change 1
export SEQ_DIAG_DUMP_EVERY="${SEQ_EG_DUMP_EVERY:-10}"      # declared change 2
# Step 100 is SCANNED as the bridge to the original run but is NOT offered for
# matching: the amendment's grid is 10..90 and was declared as such.
export SEQ_DIAG_MATCH_STEPS="${SEQ_EG_MATCH_STEPS:-10,20,30,40,50,60,70,80,90}"
# The FIXED reference evaluations: the original run's verified MA and MAB_L2
# development evaluations, read-only. This rerun does NOT regenerate them, so
# the suppression target it matches to cannot move.
export SEQ_DIAG_REFERENCE_DEV_ROOT="${SEQ_EG_REFERENCE_DEV_ROOT:-${ORIGINAL_ROOT}/eval_dev}"
# The PUBLISHED, independently verified record of those evaluations. Binding to
# it is what catches a reference whose measurements were rewritten while its
# checkpoint and manifest still look right.
export SEQ_DIAG_REFERENCE_SLOTS_INDEX="${SEQ_EG_REFERENCE_SLOTS_INDEX:-${PILOT_REPO_ROOT}/results/audit_v1/diag_v1_evidence/development_slots_index.json}"
# The DECLARED match target per seed, from the original run's published
# development measurements. A reference that no longer measures this has moved
# the target and the bridge refuses.
EG_DECLARED_L2_PP="${SEQ_EG_DECLARED_L2_PP:-17=30.0,29=33.75}"
RUNNER="${SEQ_DIAG_RUNNER:-${HERE}/run_diagnostic.sh}"
BRIDGE_OUT="${SEQ_DIAG_ROOT}/bridge_check.json"
STAGE="${1:-all}"

say() { printf '%s\n' "$*"; }
banner() {
    say ""
    say "##############################################################"
    say "# AMENDMENT 01 -- early dump grid ${SEQ_DIAG_MATCH_STEPS}"
    say "# output root : ${SEQ_DIAG_ROOT}"
    say "# original run: ${ORIGINAL_ROOT}  (read-only)"
    say "# training     : ${SEQ_DIAG_ITERATIONS} steps, dumps every ${SEQ_DIAG_DUMP_EVERY}"
    say "#                (declared changes: stopping limit and save cadence)"
    say "# bridge       : step ${BRIDGE_STEP}, stop if |Delta suppression| > ${BRIDGE_TOL_PP} pp"
    say "##############################################################"
}

delegate() { bash "$RUNNER" "$1"; }

# The bridge is RECOMPUTED, never read as a stored verdict.
#
# The defect this closes: `bridge_held` used to read the `verdict` string out of
# bridge_check.json. A bridge computed while the settings were right stayed
# "BRIDGE_HELD" on disk after the learning rate was changed, and `test` was
# delegated on the strength of it. A verdict is only evidence about the bytes it
# was computed from, so every gate below recomputes it and then re-binds the
# record to the artifacts now on disk.
stage_bridge() {
    say ""
    say "=============================================================="
    say "BRIDGE CHECK (CPU; no GPU time) -- original run vs this rerun"
    say "=============================================================="
    CUDA_VISIBLE_DEVICES="" python "${HERE}/bridge_check.py" \
        --original_root "$ORIGINAL_ROOT" --rerun_root "$SEQ_DIAG_ROOT" \
        --reference_dev_root "$SEQ_DIAG_REFERENCE_DEV_ROOT" \
        --registry "$SEQ_LEGACY_ARTIFACT_REGISTRY" \
        --expect_dev_manifest_sha "$SEQ_DEV_MANIFEST_SHA" \
        --expect_slots_index "$SEQ_DIAG_REFERENCE_SLOTS_INDEX" \
        --expect_l2_suppression_pp "$EG_DECLARED_L2_PP" \
        --seeds "$SEQ_DIAG_SEEDS" --bridge_step "$BRIDGE_STEP" \
        --tolerance_pp "$BRIDGE_TOL_PP" \
        --declared_rerun_iterations "$SEQ_DIAG_ITERATIONS" \
        --declared_rerun_cadence "$SEQ_DIAG_DUMP_EVERY" \
        --contract "$SEQ_CKPT_CONTRACT" --out "$BRIDGE_OUT"
}

# Recompute, then verify the record binds to what is on disk. Both must pass.
fresh_bridge_holds() {
    local what="$1"
    say ""
    say "[gate] ${what} requires a FRESHLY COMPUTED bridge; recomputing now"
    if ! stage_bridge; then
        say "[gate] the recomputed bridge does not hold; ${what} is refused" >&2
        return 1
    fi
    if ! CUDA_VISIBLE_DEVICES="" python "${HERE}/bridge_check.py" \
             --verify "$BRIDGE_OUT" --original_root "$ORIGINAL_ROOT" \
             --rerun_root "$SEQ_DIAG_ROOT" --out "$BRIDGE_OUT" \
             | sed 's/^/    /'; then
        say "[gate] the bridge record does not bind to the artifacts on disk;" \
            "${what} is refused" >&2
        return 1
    fi
    return 0
}

selection_binds() {
    [ -s "${SEQ_DIAG_ROOT}/selection.json" ] || return 1
    CUDA_VISIBLE_DEVICES="" python "${HERE}/select_matched_dump.py" \
        --verify "${SEQ_DIAG_ROOT}/selection.json" | sed 's/^/    /'
}

case "$STAGE" in
    preflight|smoke|train|dev) banner; delegate "$STAGE" ;;
    bridge) banner; stage_bridge ;;
    select)
        banner
        # Selection is gated on the bridge too: choosing a checkpoint from a
        # grid whose comparability has not been established would make the
        # decision meaningless, and a stale bridge is no evidence at all.
        fresh_bridge_holds "selection" || exit 3
        delegate select
        ;;
    test)
        banner
        # The frozen test set stays separated from everything that chose the
        # checkpoint. Three independent gates, every one recomputed now:
        #   1. the bridge, recomputed and re-bound to the artifacts on disk;
        #   2. the selection record, re-bound to the development inputs it read;
        #   3. both seeds matched.
        # And selection can no longer run once test output exists, so this is
        # one-way: no automatic reselection after test access.
        if ! fresh_bridge_holds "the frozen test set"; then
            say "STOP: the frozen test set is NOT evaluated." >&2
            say "See ${BRIDGE_OUT}. If the bridge failed, that failure IS the" >&2
            say "result and no test evaluation follows." >&2
            exit 3
        fi
        say ""
        say "[gate] re-binding the selection record to its development inputs"
        if ! selection_binds; then
            say "STOP: the selection record is missing or no longer describes" >&2
            say "the development inputs it was made from. The frozen test set" >&2
            say "is NOT evaluated." >&2
            exit 3
        fi
        delegate test
        ;;
    all)
        banner
        delegate preflight || exit 1
        if [ "${SEQ_DIAG_SKIP_SMOKE:-0}" != "1" ]; then
            delegate smoke || { say "=== STOPPED: the dump mechanism failed its smoke test ===" >&2; exit 1; }
        fi
        delegate train || { say "=== STOPPED: training incomplete ===" >&2; exit 1; }
        delegate dev   || { say "=== STOPPED: development evaluation incomplete ===" >&2; exit 1; }
        if ! fresh_bridge_holds "selection"; then
            say ""
            say "=== STOPPED AT THE BRIDGE, BY THE CONDITION DECLARED IN ADVANCE ==="
            say "The early grid is not trusted to locate a match point, so no"
            say "selection is made and the frozen test set is NOT evaluated."
            say "This is a result. See ${BRIDGE_OUT}."
            exit 3
        fi
        if ! delegate select; then
            say ""
            say "=== STOPPED AT SELECTION, BY THE PRESPECIFIED RULE ==="
            say "The frozen test set was NOT evaluated."
            exit 3
        fi
        say ""
        say "Development matching succeeded on both seeds AND the bridge held."
        say "The frozen test set is a SEPARATE, explicit invocation:"
        say "    bash scripts/seq/run_early_grid.sh test"
        say "Detector results are PROVISIONAL until the blinded human annotation"
        say "is complete. No final scientific conclusion follows from them alone."
        ;;
    *)
        say "unknown stage '${STAGE}'" >&2
        say "stages: all | preflight | smoke | train | dev | bridge | select | test" >&2
        exit 2
        ;;
esac
