#!/bin/bash
# ---------------------------------------------------------------------------
# AMENDMENT 01, unattended. Runs the whole authorised sequence with nobody
# watching, and survives the operator's workstation disconnecting.
#
#   setsid nohup bash scripts/seq/run_amendment_unattended.sh &
#
# What it does, in order, stopping only where the protocol says to stop:
#
#   1. takes an exclusive LOCK, so a second invocation cannot double-launch
#      onto the same output root;
#   2. waits for GPUs that are VERIFIED unoccupied -- the fail-closed selector,
#      where an unreadable reading is not availability -- and never touches
#      another user's processes;
#   3. runs `run_early_grid.sh all` (preflight, smoke, train, dev, bridge,
#      select). Validated stages are skipped, so an interrupted run resumes;
#   4. ONLY if the freshly recomputed bridge held and both seeds matched,
#      invokes `run_early_grid.sh test` -- the separate, explicitly authorised
#      frozen-test evaluation, once;
#   5. runs the CPU analysis and builds the blinded human-audit packet;
#   6. writes a STATUS file throughout: RUNNING / QUEUED / BLOCKED /
#      STOPPED_BY_RULE / COMPLETED, each with a timestamp and a reason.
#
# A prespecified stop (bridge fails, or either seed does not match) is a RESULT.
# This script records it and exits; it never widens a tolerance, refines the
# grid, changes seeds, retrains L2, or retries a completed draw to get a
# different answer.
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${HERE}/../.." && pwd)"

# Queue politely and for a long time: the runner's own capacity probe waits
# this long per attempt, and the outer loop keeps trying.
export SEQ_DIAG_WAIT_SECONDS="${SEQ_DIAG_WAIT_SECONDS:-1800}"
export SEQ_DIAG_WAIT_POLL_SECONDS="${SEQ_DIAG_WAIT_POLL_SECONDS:-120}"
OUTER_MAX_SECONDS="${AMD_QUEUE_MAX_SECONDS:-86400}"   # 24 h of polite queuing
OUTER_SLEEP="${AMD_QUEUE_SLEEP:-300}"

source "${HERE}/exp_config.sh" >/dev/null
EG_ROOT="${SEQ_EG_ROOT:-${SEQ_ROOT}/diag_v2_early_grid}"
LOGDIR="${EG_ROOT}/logs"
mkdir -p "$LOGDIR"
LOG="${LOGDIR}/unattended.log"
STATUS="${EG_ROOT}/STATUS.json"
LOCK="${SEQ_ROOT}/.amendment01.lock"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
START_UTC="$(date -Is)"

log() { printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG"; }

set_status() {   # set_status <state> <phase> <reason...>
    local state="$1" phase="$2"; shift 2
    CUDA_VISIBLE_DEVICES="" python - "$STATUS" "$state" "$phase" "$*" \
        "$COMMIT" "$START_UTC" "$$" "$EG_ROOT" "$LOG" <<'PY'
import datetime, json, os, sys
out, state, phase, reason, commit, started, pid, root, log = sys.argv[1:10]
prev = {}
if os.path.exists(out):
    try:
        prev = json.load(open(out))
    except Exception:
        prev = {}
hist = prev.get("history", [])
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
if not hist or hist[-1].get("state") != state or hist[-1].get("phase") != phase:
    hist.append({"utc": now, "state": state, "phase": phase, "reason": reason})
json.dump({
    "experiment": "Amendment 01 - fixed early dump grid 10..90",
    "state": state, "phase": phase, "reason": reason,
    "updated_utc": now, "started_utc": started,
    "execution_commit": commit, "supervisor_pid": int(pid),
    "output_root": root, "log": log,
    "states": "RUNNING | QUEUED | BLOCKED | STOPPED_BY_RULE | COMPLETED",
    "history": hist,
}, open(out, "w"), indent=2)
PY
    log "[status] ${state} (${phase}) ${*}"
}

# ---------------------------------------------------------------- the lock
exec 9>"$LOCK" || { echo "cannot open lock $LOCK" >&2; exit 2; }
if ! flock -n 9; then
    echo "ALREADY RUNNING: another supervisor holds $LOCK. Refusing to launch a" >&2
    echo "second copy onto ${EG_ROOT}. See ${STATUS} and ${LOG}." >&2
    exit 3
fi
printf '%s\n' "$$" >&9

log "=============================================================="
log "AMENDMENT 01 unattended supervisor starting"
log "  commit      : ${COMMIT}"
log "  pid         : $$"
log "  output root : ${EG_ROOT}"
log "  lock        : ${LOCK}"
log "  usage record: ${SEQ_DIAG_LEDGER}"
set_status RUNNING startup "supervisor started under lock"

# ------------------------------------------------------- capacity, verified
# Immediately before assignment, and again by the runner's own preflight. An
# UNKNOWN reading is never availability (the selector fails closed), and no
# other user's process is ever signalled.
capacity_now() {
    local probe idle
    probe="$(GPU_SAMPLES=6 GPU_SAMPLE_INTERVAL=2 bash scripts/gpu_select.sh 2>&1)"
    idle="$(printf '%s\n' "$probe" | grep -cE '^GPU [0-9]+ .*-> IDLE$')"
    printf '%s\n' "$probe" >> "$LOG"
    log "[capacity] ${idle} verified-idle device(s)"
    [ "$idle" -ge 2 ]
}

waited=0
while ! capacity_now; do
    if [ "$waited" -ge "$OUTER_MAX_SECONDS" ]; then
        set_status BLOCKED capacity \
            "no 2 verified-idle GPUs after ${waited}s of polite polling; nothing was launched and no other user's job was touched"
        exit 4
    fi
    set_status QUEUED capacity \
        "waiting for 2 verified-idle GPUs (${waited}s elapsed, polling every ${OUTER_SLEEP}s); other users' jobs are never pre-empted"
    sleep "$OUTER_SLEEP"
    waited=$((waited + OUTER_SLEEP))
done

# --------------------------------------------------------- the main sequence
set_status RUNNING early_grid "preflight, smoke, train, dev, bridge, select"
log "--- run_early_grid.sh all ---"
bash scripts/seq/run_early_grid.sh all >> "$LOG" 2>&1
RC_ALL=$?
log "[all] exit=${RC_ALL}"

if [ "$RC_ALL" -eq 3 ]; then
    set_status STOPPED_BY_RULE early_grid \
        "a prespecified stopping condition was met (the step-100 bridge did not hold, or a seed did not match on the development set). The frozen test set was NOT evaluated. This is a result; no tolerance is widened and no grid is refined."
    log "see ${EG_ROOT}/bridge_check.json and ${EG_ROOT}/selection.json"
    exit 0
fi
if [ "$RC_ALL" -ne 0 ]; then
    set_status BLOCKED early_grid \
        "run_early_grid.sh all failed with exit ${RC_ALL}; everything produced so far is preserved and nothing was selected or tested"
    exit 5
fi

# -------------------------------------------- the conditional frozen test
# Authorised in advance, and gated on evidence recomputed at this moment: the
# bridge is recomputed inside the test stage, the selection record is re-bound
# to the development inputs it used, and both seeds must have matched.
PROCEED="$(CUDA_VISIBLE_DEVICES="" python -c '
import json,sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("False"); raise SystemExit(0)
print(d.get("proceed_to_frozen_test"))' "${EG_ROOT}/selection.json" 2>/dev/null)"
if [ "$PROCEED" != "True" ]; then
    set_status STOPPED_BY_RULE selection \
        "development matching did not succeed on both seeds, so the planned paired test does not run. This is a result."
    exit 0
fi

set_status RUNNING frozen_test "evaluating the frozen test set ONCE"
log "--- run_early_grid.sh test ---"
bash scripts/seq/run_early_grid.sh test >> "$LOG" 2>&1
RC_TEST=$?
log "[test] exit=${RC_TEST}"
if [ "$RC_TEST" -eq 3 ]; then
    set_status STOPPED_BY_RULE frozen_test \
        "the frozen test set was refused at its gates (bridge recomputation or selection re-binding failed). Nothing was evaluated."
    exit 0
fi
if [ "$RC_TEST" -ne 0 ]; then
    set_status BLOCKED frozen_test \
        "the frozen-test evaluation failed with exit ${RC_TEST}; its outputs are preserved for inspection and no analysis was run on a partial set"
    exit 6
fi

# ------------------------------------------------------------- CPU analysis
set_status RUNNING analysis "prescribed CPU analysis of the frozen test set"
log "--- diag_analysis.py --set test ---"
CUDA_VISIBLE_DEVICES="" python scripts/seq/diag_analysis.py \
    --diag_root "$EG_ROOT" --set test \
    --out "${EG_ROOT}/analysis_test.json" >> "$LOG" 2>&1
RC_AN=$?
log "[analysis] exit=${RC_AN}"
if [ "$RC_AN" -ne 0 ]; then
    set_status BLOCKED analysis \
        "the frozen test set was evaluated and is preserved, but the analysis exited ${RC_AN}; the measurements are intact and the analysis can be re-run on CPU"
    exit 7
fi

# --------------------------------------------------- blinded audit packet
set_status RUNNING annotation_packet "building the blinded human-audit packet (labels left blank)"
log "--- build_diag_annotation_packet.py ---"
CUDA_VISIBLE_DEVICES="" python scripts/seq/build_diag_annotation_packet.py \
    --eval_root "${EG_ROOT}/eval_test" --seeds "$SEQ_DIAG_SEEDS" \
    --selection "${EG_ROOT}/selection.json" \
    --out_root "${EG_ROOT}/annotation_v2" \
    --repo_out "results/audit_v1/annotation_packet_v2" >> "$LOG" 2>&1
RC_PK=$?
log "[packet] exit=${RC_PK}"

if [ "$RC_PK" -ne 0 ]; then
    set_status COMPLETED annotation_packet \
        "frozen test evaluated once and analysed; the blinded packet builder exited ${RC_PK} and can be re-run on CPU. Detector results are PROVISIONAL until the blinded human annotation is complete."
    exit 0
fi

set_status COMPLETED done \
    "early grid trained, development scan complete, bridge held, both seeds matched, frozen test evaluated ONCE, CPU analysis written, blinded audit packet built with labels blank. Detector results are PROVISIONAL until the human annotation is complete."
log "ALL AUTHORISED WORK COMPLETE"
log "  selection : ${EG_ROOT}/selection.json"
log "  bridge    : ${EG_ROOT}/bridge_check.json"
log "  analysis  : ${EG_ROOT}/analysis_test.json"
log "  packet    : ${EG_ROOT}/annotation_v2"
exit 0
