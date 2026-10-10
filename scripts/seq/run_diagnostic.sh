#!/bin/bash
# ---------------------------------------------------------------------------
# Matched-effectiveness diagnostic.
#
# RESOURCE POLICY (PI, 2026-10-10): the four-GPU-hour ceiling, the frozen-test
# reserve withholding, and every budget-derived cutoff are WITHDRAWN. No
# replacement cap is invented here. What gates work now is VERIFIED GPU
# AVAILABILITY -- see acquire_gpus, which polls the real fail-closed selector,
# waits rather than pre-empting, and never touches another user's job. Usage is
# still recorded, stage by stage, for provenance and efficiency.
#
#   bash scripts/seq/run_diagnostic.sh [stage]
#       stage: all (default) | preflight | train | dev | select | test
#
# What it does, in the order the protocol freezes:
#
#   0 preflight  CPU. Verify the commit, the contracts, BOTH frozen manifest
#                identities, the frozen bootstrap grouping, and the exact saved
#                artifacts being reused (MA and MAB_L2 per seed, by hash against
#                configs/legacy_training_artifacts.json). Initialise the budget
#                ledger. Verify idle GPUs with the real fail-closed selector.
#   1 train      Two NEW unregularised dog trajectories, seeds 17 and 29,
#                parent = that seed's saved MA, 1000 steps, dumps every 100.
#   2 dev        Development set, dog only, 80 pairs x 12 checkpoints x 2 seeds
#                = 1,920 images. Selection only.
#   3 select     Gates (>=30 pp suppression, <=60% residue) and the <=5 pp match
#                against each seed's own L2 endpoint. STOPS here unless BOTH
#                seeds match -- declaring infeasibility is a result.
#   4 test       The frozen 3,360-image test set: MA, L2, selected-U per seed.
#                Evaluated ONCE. No test-based reselection, ever.
#
# Analysis is a SEPARATE CPU step (scripts/seq/diag_analysis.py) so that no
# confirmatory number is produced by the same process that spends GPU time.
#
# Accounting. Every GPU stage runs through `charge`, which now RECORDS rather
# than gates: it opens a reservation (so an interrupted stage is still
# reconciled), runs the stage, and records the wall time it actually occupied
# WHETHER OR NOT IT SUCCEEDED. Device-hours, so two GPUs for 30 minutes is 1.0
# GPU-hour. Nothing is refused for want of budget and no runtime cutoff is
# derived from a cost estimate.
#
# Watchdog. There is deliberately NO default time limit on a stage. A stage that
# needs longer than estimated now simply takes longer. If an operator wants a
# watchdog they set SEQ_STAGE_WATCHDOG_SECONDS explicitly; it then applies
# per invocation of `bounded`, which is stated plainly because the retired
# budget-derived bound did NOT bound a whole evaluation slot -- generation and
# detection each received the full slot allowance, so a slot could occupy twice
# it. That defect goes away with the cutoff rather than being re-tuned.
#
# The saved pilot is read-only here. Nothing is retrained, moved or rewritten,
# and no other user's job is ever touched.
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/exp_config.sh"
source "${PILOT_VENV}/bin/activate"
cd "${PILOT_REPO_ROOT}"

STAGE="${1:-all}"
DIAG="${SEQ_DIAG_ROOT}"
LEDGER="${SEQ_DIAG_LEDGER}"
MODELS="${DIAG}/models"
EVAL_DEV="${DIAG}/eval_dev"
EVAL_TEST="${DIAG}/eval_test"
LOGS="${DIAG}/logs"
mkdir -p "$DIAG" "$MODELS" "$EVAL_DEV" "$EVAL_TEST" "$LOGS"

# Per-stage estimates. INFORMATIONAL ONLY since the ceiling was retired: they
# are recorded beside the actual occupancy so that estimate-vs-actual stays
# reportable and the cost model can be improved. They admit nothing and bound
# nothing. The measured model (scripts/seq/diag_cost_model.py) supplies them
# when present.
load_estimates() {
    EST_TRAIN="${SEQ_EST_TRAIN:-0.2924}"
    EST_DEV_SLOT="${SEQ_EST_DEV_SLOT:-0.0397}"
    EST_TEST_SLOT="${SEQ_EST_TEST_SLOT:-0.2246}"
    EST_SMOKE="${SEQ_EST_SMOKE:-0.0300}"
    EST_SOURCE="built-in defaults derived from the completed run"
    if [ -s "${SEQ_DIAG_COST_MODEL:-}" ]; then
        local j
        j="$(CUDA_VISIBLE_DEVICES="" python -c '
import json, sys
m = json.load(open(sys.argv[1]))["estimates_gpu_hours"]
print(m["train_trajectory"], m["development_slot"], m["frozen_test_slot"])
' "$SEQ_DIAG_COST_MODEL" 2>/dev/null)" && [ -n "$j" ] && {
            read -r EST_TRAIN EST_DEV_SLOT EST_TEST_SLOT <<<"$j"
            EST_SOURCE="$SEQ_DIAG_COST_MODEL"
        }
    fi
    export EST_TRAIN EST_DEV_SLOT EST_TEST_SLOT EST_SMOKE
}
load_estimates

GPU_SELECT="${GPU_SELECT:-${HERE}/../gpu_select.sh}"
COMMIT="$(git -C "$PILOT_REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
read -r -a SEEDS <<<"$SEQ_DIAG_SEEDS"
DUMP_STEPS=()
for ((s = SEQ_DIAG_DUMP_EVERY; s <= SEQ_DIAG_ITERATIONS; s += SEQ_DIAG_DUMP_EVERY)); do
    DUMP_STEPS+=("$s")
done
DUMP_CSV="$(IFS=,; echo "${DUMP_STEPS[*]}")"
# Which of the scanned dumps are OFFERED FOR MATCHING. Normally all of them.
# The early-grid amendment scans one extra dump (step 100) purely as a bridge to
# the original run and must not match on it, so the two lists are separable --
# and the difference is printed in preflight rather than left implicit.
MATCH_CSV="${SEQ_DIAG_MATCH_STEPS:-$DUMP_CSV}"
# Where the FIXED reference evaluations (MA, MAB_L2) live. Empty = this run
# generates its own, which is what the original run did. When set, the two
# reference arms are READ-ONLY from there and are NOT regenerated here: the
# match target must not be able to move when candidates are rescored.
REF_DEV="${SEQ_DIAG_REFERENCE_DEV_ROOT:-}"

say()  { printf '%s\n' "$*"; }
head1() { say ""; say "=============================================================="; say "$*"; say "=============================================================="; }
fail() { say "FATAL: $*" >&2; exit 1; }

# --- ordinary process cleanup -------------------------------------------------
# Retained independently of the retired budget cutoff: an interrupt must not
# leave a trainer or a generator holding a device. Only DESCENDANTS of this run
# are signalled -- never `kill 0`, which would reach whatever launched us, and
# never anything belonging to another user.
kill_tree() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child"; done
    kill "$pid" 2>/dev/null
}
on_interrupt() {
    trap - INT TERM
    say "" >&2
    say "[cleanup] interrupted: stopping this run's own GPU children." >&2
    local pid
    for pid in $(jobs -p 2>/dev/null); do kill_tree "$pid"; done
    say "[cleanup] the next preflight will reconcile the usage record for" \
        "whatever was in flight." >&2
    exit 130
}
trap on_interrupt INT TERM

# --- budget ------------------------------------------------------------------
budget() { CUDA_VISIBLE_DEVICES="" python "$SEQ_BUDGET" "$@"; }

jfield() {   # jfield <json-line> <key>
    printf '%s' "$1" | CUDA_VISIBLE_DEVICES="" python -c '
import json, sys
line = [l for l in sys.stdin.read().splitlines() if l.strip().startswith("{")][-1]
print(json.loads(line)[sys.argv[1]])' "$2"
}

# charge <stage> <gpus> <gpu_index> <est_gpu_hours> <reserve:0|1> -- cmd...
#
# RECORDING, not gating. `budget admit` on a recording-only ledger never
# refuses and returns no wall-clock bound; it writes a durable OPEN RESERVATION
# only so that a stage interrupted between start and finish is still reconciled
# instead of vanishing from the usage record. `budget settle` then records the
# ACTUAL occupancy, success or failure alike. If this shell dies in between, the
# reservation stays on the books and the next admission reconciles it
# conservatively.
#
# The `res` argument is vestigial: the frozen-test reserve was withdrawn with
# the ceiling. It is still passed through so the recorded entries keep saying
# which stage was confirmatory.
charge() {
    local stage="$1" gpus="$2" gidx="$3" est="$4" res="$5"; shift 5
    [ "${1:-}" = "--" ] && shift
    local resflag=() adm rid t0 t1 rc mypid
    [ "$res" = "1" ] && resflag=(--draws_on_reserve)
    # Capture the owning pid OUTSIDE the command substitution. $BASHPID inside
    # "$(...)" is the substitution's own short-lived subshell, which exits the
    # moment admit returns -- so the reservation would look orphaned to the very
    # next admission and be reconciled away while its stage was still running.
    mypid="$BASHPID"
    adm="$(budget admit --ledger "$LEDGER" --stage "$stage" --need "$est" \
             --gpus "$gpus" --gpu_index "$gidx" --pid "$mypid" \
             "${resflag[@]}" 2>&1)"
    if [ $? -ne 0 ]; then
        # Only a broken/missing ledger or a ledger still in the retired CEILING
        # mode can get here. Either way nothing was launched.
        say "$adm" >&2
        say "[usage] could not open a usage record for ${stage}; nothing was" \
            "launched. The ledger must exist and be in recording_only mode" \
            "(scripts/seq/gpu_budget.py retire)." >&2
        return 97
    fi
    say "$adm"
    rid="$(jfield "$adm" reservation_id)"
    # No STAGE_TIMEOUT is derived from the estimate. `bounded` applies a limit
    # only if an operator set SEQ_STAGE_WATCHDOG_SECONDS explicitly.
    t0=$(date +%s)
    "$@"
    rc=$?
    t1=$(date +%s)
    budget settle --ledger "$LEDGER" --id "$rid" \
        --wall_seconds "$((t1 - t0))" --exit_code "$rc" >/dev/null || {
            say "[usage] could not record ${stage}; see the ledger" >&2; return 98; }
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        say "[usage] ${stage} was stopped by the explicit watchdog" \
            "(SEQ_STAGE_WATCHDOG_SECONDS=${SEQ_STAGE_WATCHDOG_SECONDS:-unset});" \
            "the occupancy it used is recorded" >&2
    fi
    return "$rc"
}

# Run a GPU child. By default there is NO time limit: the GPU-hour ceiling that
# used to supply one is retired, and inventing a replacement would reintroduce
# exactly the risk the old bound carried -- killing legitimate work because an
# estimate was low.
#
# SEQ_STAGE_WATCHDOG_SECONDS, if an operator sets it, applies PER CALL, not per
# stage: an evaluation slot calls `bounded` for generation and again for
# detection, so the slot as a whole can occupy up to twice the value. That was
# the defect in the retired budget-derived bound and it is stated here rather
# than papered over; with the cap withdrawn there is nothing to enforce, and the
# honest limit is the one the operator set.
#
# `timeout` without --foreground puts the child in its own process group and
# signals the group, so accelerate's workers are stopped too rather than left
# holding the device. That is ordinary cleanup and is retained.
bounded() {
    if [ -n "${SEQ_STAGE_WATCHDOG_SECONDS:-}" ]; then
        timeout --kill-after=60 "${SEQ_STAGE_WATCHDOG_SECONDS}" "$@"
    else
        "$@"
    fi
}

# --- GPU capacity ------------------------------------------------------------
# Start only on verified idle capacity. On a host with no scheduler this poll of
# the real fail-closed selector IS the queue; it never pre-empts anyone.
IDLE_GPUS=()
acquire_gpus() {   # acquire_gpus <n>
    local want="$1" waited=0 probe line
    while :; do
        probe="$(GPU_SAMPLES=6 GPU_SAMPLE_INTERVAL=2 bash "$GPU_SELECT" 2>&1)" || true
        IDLE_GPUS=()
        while read -r line; do
            [[ "$line" =~ ^GPU\ ([0-9]+)\ .*-\>\ IDLE$ ]] && IDLE_GPUS+=("${BASH_REMATCH[1]}")
        done <<<"$probe"
        say "$probe" | sed 's/^/    /'
        if [ "${#IDLE_GPUS[@]}" -ge "$want" ]; then
            say "[capacity] RUNNING: ${want} verified-idle GPU(s) acquired: ${IDLE_GPUS[*]:0:$want}"
            return 0
        fi
        if [ "$waited" -ge "${SEQ_DIAG_WAIT_SECONDS}" ]; then
            say "[capacity] QUEUED/BLOCKED: need ${want} idle GPU(s), found" \
                "${#IDLE_GPUS[@]} after ${waited}s of waiting." >&2
            say "[capacity] This host has no scheduler; other users' jobs are" \
                "never pre-empted. Re-run when capacity frees, or set" \
                "SEQ_DIAG_WAIT_SECONDS to wait longer." >&2
            return 1
        fi
        say "[capacity] QUEUED: ${#IDLE_GPUS[@]}/${want} idle; waiting" \
            "${SEQ_DIAG_WAIT_POLL_SECONDS}s (waited ${waited}s of ${SEQ_DIAG_WAIT_SECONDS}s)"
        sleep "${SEQ_DIAG_WAIT_POLL_SECONDS}"
        waited=$((waited + SEQ_DIAG_WAIT_POLL_SECONDS))
    done
}

# --- reused saved artifacts --------------------------------------------------
saved_ckpt() { printf '%s\n' "${SEQ_ROOT}/models/seed${1}/${2}/delta.bin"; }
sha_of()     { sha256sum "$1" | cut -d" " -f1; }

# The dump/endpoint a diagnostic evaluation slot is generated from.
ckpt_path_for() {   # ckpt_path_for <seed> <slot>
    local seed="$1" slot="$2"
    case "$slot" in
        MA|MAB_L2) saved_ckpt "$seed" "$slot" ;;
        U_step*)   printf '%s\n' "${MODELS}/seed${seed}/U/delta-${slot#U_step}" ;;
        *) return 1 ;;
    esac
}

# =============================================================== 0. preflight
stage_preflight() {
    head1 "0. PREFLIGHT (CPU only, no GPU time charged)"
    say "commit            : ${COMMIT}"
    say "diagnostic root   : ${DIAG}"
    say "seeds             : ${SEEDS[*]}"
    say "dump schedule     : every ${SEQ_DIAG_DUMP_EVERY} to ${SEQ_DIAG_ITERATIONS} (${DUMP_CSV})"
    say "offered for match : ${MATCH_CSV}"
    if [ "$MATCH_CSV" != "$DUMP_CSV" ]; then
        say "                    (the scanned dumps not in this list are measured"
        say "                     but cannot be selected; see the amendment)"
    fi
    if [ -n "$(git -C "$PILOT_REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
        say "[warn] working tree is NOT clean; the execution commit above does not" \
            "fully describe what will run" >&2
    else
        say "working tree      : clean"
    fi

    for f in "$SEQ_DEV_MANIFEST" "$SEQ_TEST_MANIFEST" "$SEQ_GEN_SETTINGS_CONTRACT" \
             "$SEQ_CKPT_CONTRACT" "$SEQ_VALIDATOR" "$SEQ_BOOTSTRAP_GROUPING" \
             "$SEQ_LEGACY_ARTIFACT_REGISTRY" "$SEQ_BUDGET"; do
        [ -s "$f" ] || fail "required file missing or empty: $f"
    done

    say ""
    say "--- frozen prompt-set identities (recomputed from content) ---"
    CUDA_VISIBLE_DEVICES="" python "$SEQ_VALIDATOR" manifest dev \
        --manifest "$SEQ_DEV_MANIFEST" --expect_manifest_sha "$SEQ_DEV_MANIFEST_SHA" \
        --expect_records 80 --expect_prompt_texts 20 || fail "dev manifest identity"
    CUDA_VISIBLE_DEVICES="" python "$SEQ_VALIDATOR" manifest test \
        --manifest "$SEQ_TEST_MANIFEST" --expect_manifest_sha "$SEQ_TEST_MANIFEST_SHA" \
        --expect_records 560 --expect_prompt_texts 140 || fail "test manifest identity"

    say ""
    say "--- frozen bootstrap grouping ---"
    CUDA_VISIBLE_DEVICES="" python - "$SEQ_BOOTSTRAP_GROUPING" <<'PY' || fail "bootstrap grouping"
import hashlib, json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
stored = d["grouping_sha256"]
body = {k: v for k, v in d.items() if k != "grouping_sha256"}
actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
print(f"grouping identity {actual[:16]}… (stored {stored[:16]}…) "
      f"{'OK' if actual == stored else 'MISMATCH'}")
print(f"  unit={d['resampling_unit']['definition']}  draws={d['n_draws']}  "
      f"rng={d['rng']['bootstrap_rng_seed']}")
sys.exit(0 if actual == stored else 1)
PY

    say ""
    say "--- reused saved artifacts, bound by hash (read-only) ---"
    # Each reuse target must be THE registered artifact for its own
    # (seed, slot) -- not merely some registered artifact. Accepting any
    # registered digest would let seed 29's MA stand in for seed 17's, which is
    # exactly the cross-seed mix-up this experiment must not make.
    local bad=0
    for seed in "${SEEDS[@]}"; do
        for slot in MA MAB_L2; do
            local p; p="$(saved_ckpt "$seed" "$slot")"
            if [ ! -s "$p" ]; then say "  MISSING seed${seed}/${slot}: $p" >&2; bad=1; continue; fi
            if CUDA_VISIBLE_DEVICES="" python - "$p" "$SEQ_LEGACY_ARTIFACT_REGISTRY" \
                   "seed${seed}/${slot}" "$slot" <<'PY'
import hashlib, json, sys
from pathlib import Path
ck, reg_p, want_id, slot = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
h = hashlib.sha256(Path(ck).read_bytes()).hexdigest()
arts = json.loads(reg_p.read_text())["artifacts"]
hit = next((a for a in arts if a.get("id") == want_id), None)
if hit is None:
    print(f"  {want_id}: {h[:12]}…  NOT a registered saved artifact "
          f"(no registry entry with this id)")
    sys.exit(1)
if hit.get("checkpoint") != slot or hit.get("delta_sha256") != h:
    other = next((a["id"] for a in arts if a.get("delta_sha256") == h), None)
    print(f"  {want_id}: {h[:12]}…  NOT a registered saved artifact: the "
          f"registry records {str(hit.get('delta_sha256'))[:12]}… for this id"
          + (f", and these bytes are {other} instead" if other else ""))
    sys.exit(1)
print(f"  {want_id}: {h[:12]}…  REGISTERED saved artifact (reused, never rewritten)")
PY
            then :; else bad=1; fi
        done
    done
    [ "$bad" -eq 0 ] || fail "the artifacts this diagnostic must reuse are not the registered saved ones"

    if [ -n "$REF_DEV" ]; then
        say ""
        say "--- FIXED reference evaluations (read-only; never regenerated here) ---"
        say "reference dev root: ${REF_DEV}"
        local refbad=0 rb="${DIAG}/reference_bindings.json"
        : > "${LOGS}/reference_verify.log"
        for seed in "${SEEDS[@]}"; do
            for slot in MA MAB_L2; do
                local rp; rp="$(saved_ckpt "$seed" "$slot")"
                [ -s "$rp" ] || { say "  MISSING checkpoint for seed${seed}/${slot}" >&2; refbad=1; continue; }
                local rsha; rsha="$(sha_of "$rp")"
                if CUDA_VISIBLE_DEVICES="" python "$SEQ_VALIDATOR" eval \
                       "seed${seed}_${slot}" --eval_root "$REF_DEV" \
                       --manifest "$SEQ_DEV_MANIFEST" \
                       --expect_manifest_sha "$SEQ_DEV_MANIFEST_SHA" \
                       --expect_gen_settings "$SEQ_GEN_SETTINGS_CONTRACT" \
                       --require_images --expect_sha "$rsha" \
                       >> "${LOGS}/reference_verify.log" 2>&1; then
                    say "  seed${seed}_${slot}: VALID fixed reference" \
                        "(generated by ${rsha:0:12}., the registered saved artifact)"
                else
                    say "  seed${seed}_${slot}: INVALID as a fixed reference" \
                        "(see ${LOGS}/reference_verify.log)" >&2
                    refbad=1
                fi
            done
        done
        local idxflag=()
        [ -n "${SEQ_DIAG_REFERENCE_SLOTS_INDEX:-}" ] && \
            idxflag=(--expect_slots_index "$SEQ_DIAG_REFERENCE_SLOTS_INDEX")
        CUDA_VISIBLE_DEVICES="" python "${HERE}/bind_reference_evals.py" \
            --reference_dev_root "$REF_DEV" --seeds "${SEEDS[*]}" \
            --registry "$SEQ_LEGACY_ARTIFACT_REGISTRY" \
            --expect_manifest_sha "$SEQ_DEV_MANIFEST_SHA" \
            "${idxflag[@]}" --out "$rb" \
            | sed 's/^/  /' || refbad=1
        [ "$refbad" -eq 0 ] || fail "the fixed reference evaluations are not usable; nothing is selected or generated against a moving target"
    fi

    say ""
    say "--- per-stage estimates (INFORMATIONAL; they gate nothing) ---"
    say "source: ${EST_SOURCE}"
    say "  train trajectory : ${EST_TRAIN} GPU-h"
    say "  development slot : ${EST_DEV_SLOT} GPU-h"
    say "  frozen-test slot : ${EST_TEST_SLOT} GPU-h"
    say "These are recorded beside the actual occupancy so estimate-vs-actual"
    say "stays reportable. No stage is refused and no stage is time-limited"
    say "because of them."
    say ""
    say "--- GPU-hour USAGE RECORD (the EXPERIMENT's, not this output directory's) ---"
    say "ledger: ${LEDGER}"
    if [ ! -s "$LEDGER" ]; then
        budget init --ledger "$LEDGER" --commit "$COMMIT" || fail "ledger init"
    fi
    # Explicitly disable the retired enforcement on whatever ledger we found,
    # rather than leaving a 4.0-ceiling ledger in place and hoping no stage ever
    # trips it. Idempotent, and it stamps what the record held at the switch.
    budget retire --ledger "$LEDGER" \
        --authority "PI, 2026-10-10" \
        --reason "Compute is bounded by verified GPU availability, not by an hour cap. The four-GPU-hour ceiling, the frozen-test reserve withholding and all budget-derived cutoffs are withdrawn; no replacement cap is imposed." \
        || fail "could not retire budget enforcement on ${LEDGER}"
    # Record anything a crash or interruption left unaccounted.
    budget reconcile --ledger "$LEDGER" || true
    budget report --ledger "$LEDGER"
    say ""
    say "--- GPU capacity (real fail-closed probe) ---"
    acquire_gpus 2 || return 1
    return 0
}

# ============================================================== 0b. GPU smoke
# A tiny REAL trajectory that proves the periodic-dump mechanism on real
# hardware before the budget is committed to two full ones. 20 steps with dumps
# every 10 costs ~0.02 GPU-h; discovering after 0.46 GPU-h that upstream wrote
# no dumps would cost twenty times that. Charged to the ledger like any other
# GPU work, and written into its own throwaway directory.
stage_smoke() {
    head1 "0b. GPU SMOKE TEST: does the dump mechanism actually produce dumps?"
    local out="${DIAG}/smoke/seed${SEEDS[0]}_U20" gpu="${IDLE_GPUS[0]}" rc=0
    local parent; parent="$(saved_ckpt "${SEEDS[0]}" MA)"
    if [ -s "${out}/train_report.json" ] && \
       CUDA_VISIBLE_DEVICES="" python -c "
import json,sys
d=json.load(open(sys.argv[1])).get('dumps') or {}
sys.exit(0 if d.get('complete') and d.get('n_present')==2 else 1)" \
          "${out}/train_report.json" 2>/dev/null; then
        say "[skip] smoke test already passed"; return 0
    fi
    rm -rf "$out"; mkdir -p "$out"
    smoke_run() {
        CUDA_VISIBLE_DEVICES="$gpu" bounded accelerate launch --config_file "$SEQ_ACCEL_CFG" \
            scripts/seq/train_request.py \
            --name U --parent_name MA --anchor_name horse --target dog \
            --anchor_dataset_dir "$SEQ_ANCHOR_HORSES" \
            --anchor_prompt_path "$SEQ_PROMPTS_HORSES" \
            --base_model_dir "$SEQ_BASE_MODEL" --output_dir "$out" \
            --unet_ckpt "$parent" --iterations 20 --epochs "$SEQ_EPOCHS" \
            --seed "${SEEDS[0]}" --l2sp_weight 0 --checkpoint_every 10 \
            --report "${out}/train_report.json" \
            > "${LOGS}/smoke_train.log" 2>&1
    }
    charge "smoke:dump_mechanism" 1 "$gpu" "$EST_SMOKE" 0 -- smoke_run || rc=$?
    if [ "$rc" -ne 0 ]; then
        say "[FAIL] smoke training exit=${rc} (${LOGS}/smoke_train.log)" >&2
        tail -20 "${LOGS}/smoke_train.log" | sed 's/^/    /' >&2
        return 1
    fi
    CUDA_VISIBLE_DEVICES="" python - "${out}/train_report.json" "$out" <<'PY' || return 1
import json, sys
from pathlib import Path
rep = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
d = rep.get("dumps") or {}
on_disk = sorted(p.name for p in out.glob("delta-*"))
print(f"    report dumps : {d.get('n_present')}/{d.get('n_expected')} "
      f"complete={d.get('complete')} distinct={d.get('all_digests_distinct')}")
print(f"    files on disk: {on_disk}")
print(f"    steps completed: {(rep.get('runtime') or {}).get('optimizer_steps_completed')}")
ok = (d.get("complete") and d.get("n_present") == 2
      and d.get("all_digests_distinct")
      and on_disk == ["delta-10", "delta-20"]
      and (rep.get("runtime") or {}).get("optimizer_steps_completed") == 20)
print("    SMOKE: " + ("PASS -- periodic dumps work on real hardware" if ok
                       else "FAIL -- the dump mechanism did not behave as required"))
sys.exit(0 if ok else 1)
PY
    say "[smoke] removing the throwaway trajectory; its evidence is the report above"
    return 0
}

# =================================================================== 1. train
train_one() {   # train_one <seed> <gpu>
    local seed="$1" gpu="$2" out="${MODELS}/seed${seed}/U" parent rc
    parent="$(saved_ckpt "$seed" MA)"
    mkdir -p "$out"
    say "[train] seed${seed} U on GPU ${gpu} (unregularised, parent = saved MA)"
    CUDA_VISIBLE_DEVICES="$gpu" bounded accelerate launch --config_file "$SEQ_ACCEL_CFG" \
        scripts/seq/train_request.py \
        --name U --parent_name MA --anchor_name horse --target dog \
        --anchor_dataset_dir "$SEQ_ANCHOR_HORSES" \
        --anchor_prompt_path "$SEQ_PROMPTS_HORSES" \
        --base_model_dir "$SEQ_BASE_MODEL" --output_dir "$out" \
        --unet_ckpt "$parent" --iterations "$SEQ_DIAG_ITERATIONS" \
        --epochs "$SEQ_EPOCHS" --seed "$seed" --l2sp_weight 0 \
        --checkpoint_every "$SEQ_DIAG_DUMP_EVERY" \
        --report "${out}/train_report.json" \
        > "${LOGS}/train_U_seed${seed}.log" 2>&1
    rc=$?
    [ "$rc" -ne 0 ] && say "[FAIL] train seed${seed} exit=${rc} (${LOGS}/train_U_seed${seed}.log)" >&2
    return "$rc"
}

validate_trajectory() {   # validate_trajectory <seed>
    local seed="$1" out="${MODELS}/seed${seed}/U" ok=0
    CUDA_VISIBLE_DEVICES="" python "$SEQ_VALIDATOR" train U \
        --models_root "${MODELS}/seed${seed}" \
        --parent_models_root "${SEQ_ROOT}/models/seed${seed}" \
        --contract "$SEQ_CKPT_CONTRACT" \
        --expect_seed "$seed" --expect_parent MA --expect_target dog \
        --expect_anchor horse --expect_l2sp 0 \
        --expect_steps "$SEQ_DIAG_ITERATIONS" --steps_evidence counter \
        --write_marker 2>&1 | sed 's/^/    /' || ok=1
    # The dumps are arms of the comparison, so each one is validated against the
    # same checkpoint contract as a final checkpoint, and the schedule must be
    # complete. A silently short scan is not the full fixed scan that was frozen.
    CUDA_VISIBLE_DEVICES="" python - "${out}/train_report.json" "$SEQ_CKPT_CONTRACT" \
        "$DUMP_CSV" <<'PY' || ok=1
import hashlib, json, sys
from pathlib import Path
rep = json.loads(Path(sys.argv[1]).read_text())
contract = json.loads(Path(sys.argv[2]).read_text())
want = [int(x) for x in sys.argv[3].split(",")]
d = rep.get("dumps") or {}
bad = []
if not d:
    bad.append("train_report records no 'dumps' block")
if sorted(d.get("expected_steps") or []) != want:
    bad.append(f"dump schedule {d.get('expected_steps')} != frozen {want}")
if d.get("missing_steps"):
    bad.append(f"missing dumps at steps {d['missing_steps']}")
if not d.get("all_digests_distinct"):
    bad.append("two dumps have identical content: the trajectory did not move")
import torch
for row in d.get("dumps") or []:
    p = Path(row["path"])
    if not p.is_file():
        bad.append(f"dump {row['step']} absent at {p}"); continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != row["sha256"]:
        bad.append(f"dump {row['step']} digest changed since it was written"); continue
    obj = torch.load(p, map_location="cpu", weights_only=False)
    top = contract.get("top_level_key", "unet")
    sd = obj.get(top) if isinstance(obj, dict) else None
    if not isinstance(sd, dict):
        bad.append(f"dump {row['step']} has no {top!r} state dict"); continue
    if len(sd) != contract["n_tensors"]:
        bad.append(f"dump {row['step']}: {len(sd)} tensors != {contract['n_tensors']}")
    tot = 0
    for k, spec in contract["tensors"].items():
        t = sd.get(k)
        if t is None:
            bad.append(f"dump {row['step']}: tensor {k} missing"); break
        if list(t.shape) != spec["shape"]:
            bad.append(f"dump {row['step']}: {k} shape {tuple(t.shape)}"); break
        tot += t.numel()
        if not bool(torch.isfinite(t).all()):
            bad.append(f"dump {row['step']}: {k} has NaN/Inf"); break
    else:
        if tot != contract["total_elements"]:
            bad.append(f"dump {row['step']}: {tot} params != {contract['total_elements']}")
if bad:
    print("    DUMPS INVALID:")
    for m in bad[:10]:
        print(f"      - {m}")
    sys.exit(1)
print(f"    dumps: {d['n_present']}/{d['n_expected']} present, all validate "
      f"against the checkpoint contract, all digests distinct")
PY
    return "$ok"
}

stage_train() {
    head1 "1. TRAIN two new unregularised dog trajectories"
    local pids=() seeds_run=() i=0 rc=0
    for seed in "${SEEDS[@]}"; do
        local out="${MODELS}/seed${seed}/U"
        if validate_trajectory "$seed" >/dev/null 2>&1; then
            say "[skip] seed${seed} U already trained and validated"; continue
        fi
        if [ -e "${out}/train_report.json" ]; then
            local dest="${out}.invalid.$(date +%Y%m%dT%H%M%S)"
            mv -f "$out" "$dest" && printf '%s\n' \
                "failed validation at $(date -Is); preserved whole, retrained into a fresh directory" \
                > "${dest}.reason.txt"
            say "[quarantine] seed${seed} U -> $(basename "$dest")" >&2
            mkdir -p "$out"
        fi
        local gpu="${IDLE_GPUS[$i]}"
        # 1000 steps measured at ~0.83 s/step in the pilot -> ~0.23 GPU-h, plus
        # ten periodic saves. Estimate generously; the ACTUAL time is charged.
        charge "train:U_seed${seed}" 1 "$gpu" "$EST_TRAIN" 0 -- train_one "$seed" "$gpu" &
        pids+=("$!"); seeds_run+=("$seed")
        i=$((i + 1))
    done
    # wait on EXACT child pids: `wait a b` reports only the last one's status.
    for ((j = 0; j < ${#pids[@]}; j++)); do
        if ! wait "${pids[$j]}"; then
            say "[FAIL] trajectory seed${seeds_run[$j]} did not complete" >&2; rc=1
        fi
    done
    say ""
    for seed in "${SEEDS[@]}"; do
        say "--- validating trajectory seed${seed} ---"
        validate_trajectory "$seed" || { say "[FAIL] seed${seed} trajectory invalid" >&2; rc=1; }
    done
    budget report --ledger "$LEDGER"
    return "$rc"
}

# ============================================================ 2/4. evaluation
# evaluate_slot <manifest> <manifest_sha> <eval_root> <seed> <slot> <gpu> <n_images>
evaluate_slot() {
    local man="$1" msha="$2" root="$3" seed="$4" slot="$5" gpu="$6" n="$7"
    local name="seed${seed}_${slot}" ck rc
    ck="$(ckpt_path_for "$seed" "$slot")" || { say "[FAIL] unknown slot ${slot}" >&2; return 1; }
    [ -s "$ck" ] || { say "[FAIL] ${name}: checkpoint absent: $ck" >&2; return 1; }
    local sha; sha="$(sha_of "$ck")"

    if CUDA_VISIBLE_DEVICES="" python "$SEQ_VALIDATOR" eval "$name" \
           --eval_root "$root" --manifest "$man" --expect_manifest_sha "$msha" \
           --expect_gen_settings "$SEQ_GEN_SETTINGS_CONTRACT" \
           --require_images --expect_sha "$sha" >/dev/null 2>&1; then
        say "[skip] eval ${name} (validated complete)"; return 0
    fi
    say "[eval] ${name} on GPU ${gpu} (${n} images, checkpoint ${sha:0:12}…)"
    CUDA_VISIBLE_DEVICES="$gpu" bounded python scripts/seq/generate_eval_images.py \
        --manifest "$man" --expect_manifest_sha "$msha" \
        --expect_gen_settings "$SEQ_GEN_SETTINGS_CONTRACT" \
        --base_model_dir "$SEQ_BASE_MODEL" --unet_ckpt "$ck" \
        --checkpoint_name "$name" --out_dir "${root}/${name}/images" \
        --report "${root}/${name}/image_report.json" \
        > "${LOGS}/gen_${name}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 3 ]; then
        say "[FAIL] ${name}: image reuse REFUSED for want of valid provenance" \
            "(${LOGS}/gen_${name}.log). Nothing was overwritten." >&2
        return 3
    fi
    [ "$rc" -ne 0 ] && { say "[FAIL] generate ${name} exit=${rc}" >&2; return "$rc"; }

    CUDA_VISIBLE_DEVICES="$gpu" bounded python scripts/seq/detect.py \
        --image_report "${root}/${name}/image_report.json" --checkpoint_name "$name" \
        --out "${root}/${name}/detections.jsonl" \
        --report "${root}/${name}/detect_report.json" \
        > "${LOGS}/detect_${name}.log" 2>&1
    rc=$?
    [ "$rc" -ne 0 ] && { say "[FAIL] detect ${name} exit=${rc}" >&2; return "$rc"; }

    if ! CUDA_VISIBLE_DEVICES="" python "$SEQ_VALIDATOR" eval "$name" \
             --eval_root "$root" --manifest "$man" --expect_manifest_sha "$msha" \
             --expect_gen_settings "$SEQ_GEN_SETTINGS_CONTRACT" \
             --require_images --expect_sha "$sha" --write_marker 2>&1 | sed 's/^/    /'; then
        say "[FAIL] ${name}: stages exited 0 but artifacts do not validate" >&2
        return 1
    fi
    return 0
}

# run_lane <manifest> <sha> <root> <gpu> <est_per_slot> <reserve> <seed:slot ...>
run_lane() {
    local man="$1" msha="$2" root="$3" gpu="$4" est="$5" res="$6"; shift 6
    local rc=0 item seed slot n
    n=$(CUDA_VISIBLE_DEVICES="" python -c "
import json,sys; print(len(json.load(open(sys.argv[1]))['records']))" "$man")
    for item in "$@"; do
        seed="${item%%:*}"; slot="${item##*:}"
        charge "eval:$(basename "$root"):seed${seed}_${slot}" 1 "$gpu" "$est" "$res" -- \
            evaluate_slot "$man" "$msha" "$root" "$seed" "$slot" "$gpu" "$n" || rc=1
    done
    return "$rc"
}

stage_dev() {
    head1 "2. DEVELOPMENT set (dog only, selection only)"
    local slots=() seed slot
    for seed in "${SEEDS[@]}"; do
        if [ -z "$REF_DEV" ]; then
            for slot in MA MAB_L2; do slots+=("${seed}:${slot}"); done
        fi
        for st in "${DUMP_STEPS[@]}"; do slots+=("${seed}:U_step${st}"); done
    done
    if [ -n "$REF_DEV" ]; then
        say "slots: ${#slots[@]} (per seed: ${#DUMP_STEPS[@]} dumps only)"
        say "the reference arms are NOT regenerated here: they are read-only from"
        say "  ${REF_DEV}"
        say "so the suppression target cannot move while candidates are scored."
    else
        say "slots: ${#slots[@]} (per seed: MA + L2 endpoint + ${#DUMP_STEPS[@]} dumps)"
    fi
    # Split across the two acquired GPUs; each lane is sequential on its device.
    local lane_a=() lane_b=() k=0
    for s in "${slots[@]}"; do
        if [ $((k % 2)) -eq 0 ]; then lane_a+=("$s"); else lane_b+=("$s"); fi
        k=$((k + 1))
    done
    local rc=0 pa pb
    run_lane "$SEQ_DEV_MANIFEST" "$SEQ_DEV_MANIFEST_SHA" "$EVAL_DEV" \
        "${IDLE_GPUS[0]}" "$EST_DEV_SLOT" 0 "${lane_a[@]}" & pa=$!
    run_lane "$SEQ_DEV_MANIFEST" "$SEQ_DEV_MANIFEST_SHA" "$EVAL_DEV" \
        "${IDLE_GPUS[1]}" "$EST_DEV_SLOT" 0 "${lane_b[@]}" & pb=$!
    wait "$pa" || { say "[FAIL] dev lane A" >&2; rc=1; }
    wait "$pb" || { say "[FAIL] dev lane B" >&2; rc=1; }
    budget report --ledger "$LEDGER"
    return "$rc"
}

stage_select() {
    head1 "3. SELECTION on the development set (CPU; no GPU time)"
    local refflag=()
    [ -n "$REF_DEV" ] && refflag=(--reference_dev_root "$REF_DEV")
    # Selection is refused once the frozen test output exists: re-deciding the
    # compared checkpoint with test evidence in hand is test-based selection
    # whatever the intent, so it must not be reachable by re-running a stage.
    CUDA_VISIBLE_DEVICES="" python scripts/seq/select_matched_dump.py \
        --dev_root "$EVAL_DEV" --seeds "${SEEDS[*]}" --dump_steps "$MATCH_CSV" \
        "${refflag[@]}" --refuse_if_test_evaluated "$EVAL_TEST" \
        --out "${DIAG}/selection.json"
}

stage_test() {
    head1 "4. FROZEN TEST set (evaluated ONCE; no reselection)"
    [ -s "${DIAG}/selection.json" ] || fail "no selection.json; run the select stage first"
    # The decision must still describe the artifacts on disk. A selection made
    # from development records that have since changed is not a decision about
    # this experiment.
    say "--- re-binding the selection to the development inputs it used ---"
    CUDA_VISIBLE_DEVICES="" python scripts/seq/select_matched_dump.py \
        --verify "${DIAG}/selection.json" | sed 's/^/    /' \
        || fail "the selection record does not match the development inputs it was made from"
    local proceed; proceed=$(CUDA_VISIBLE_DEVICES="" python -c "
import json,sys; print(json.load(open(sys.argv[1]))['proceed_to_frozen_test'])" \
        "${DIAG}/selection.json")
    if [ "$proceed" != "True" ]; then
        say "STOP: development matching did not succeed for both seeds, so the" >&2
        say "planned paired test does NOT run. This is a result, not a failure." >&2
        CUDA_VISIBLE_DEVICES="" python -c "
import json,sys
d=json.load(open(sys.argv[1]))
for s,r in (d.get('stop_reason') or {}).items(): print(f'  seed {s}: {r}')" \
            "${DIAG}/selection.json" >&2
        return 2
    fi
    local slots=() seed sel
    for seed in "${SEEDS[@]}"; do
        sel=$(CUDA_VISIBLE_DEVICES="" python -c "
import json,sys
d=json.load(open(sys.argv[1]))['per_seed'][sys.argv[2]]['selected']
print(d['step'])" "${DIAG}/selection.json" "$seed")
        say "seed ${seed}: selected U step ${sel}"
        slots+=("${seed}:MA" "${seed}:MAB_L2" "${seed}:U_step${sel}")
    done
    local lane_a=() lane_b=() k=0
    for s in "${slots[@]}"; do
        if [ $((k % 2)) -eq 0 ]; then lane_a+=("$s"); else lane_b+=("$s"); fi
        k=$((k + 1))
    done
    local rc=0 pa pb
    run_lane "$SEQ_TEST_MANIFEST" "$SEQ_TEST_MANIFEST_SHA" "$EVAL_TEST" \
        "${IDLE_GPUS[0]}" "$EST_TEST_SLOT" 1 "${lane_a[@]}" & pa=$!
    run_lane "$SEQ_TEST_MANIFEST" "$SEQ_TEST_MANIFEST_SHA" "$EVAL_TEST" \
        "${IDLE_GPUS[1]}" "$EST_TEST_SLOT" 1 "${lane_b[@]}" & pb=$!
    wait "$pa" || { say "[FAIL] test lane A" >&2; rc=1; }
    wait "$pb" || { say "[FAIL] test lane B" >&2; rc=1; }
    budget report --ledger "$LEDGER"
    return "$rc"
}

# ===================================================================== driver
case "$STAGE" in
    preflight) stage_preflight ;;
    smoke)     stage_preflight && stage_smoke ;;
    train)     stage_preflight && stage_train ;;
    dev)       stage_preflight && stage_dev ;;
    select)    stage_select ;;
    test)      stage_preflight && stage_test ;;
    all)
        stage_preflight || exit 1
        if [ "${SEQ_DIAG_SKIP_SMOKE:-0}" != "1" ]; then
            stage_smoke || { say "=== DIAGNOSTIC STOPPED: the dump mechanism failed its smoke test; no full trajectory was started ===" >&2; budget report --ledger "$LEDGER"; exit 1; }
        fi
        stage_train     || { say "=== DIAGNOSTIC STOPPED: training incomplete ===" >&2; budget report --ledger "$LEDGER"; exit 1; }
        stage_dev       || { say "=== DIAGNOSTIC STOPPED: development evaluation incomplete ===" >&2; budget report --ledger "$LEDGER"; exit 1; }
        if stage_select; then
            stage_test || { say "=== DIAGNOSTIC STOPPED: test evaluation incomplete ===" >&2; budget report --ledger "$LEDGER"; exit 1; }
        else
            say ""
            say "=== STOPPED AT SELECTION, BY THE PRESPECIFIED RULE ==="
            say "The frozen test set was NOT evaluated and its reserve was not spent."
            budget report --ledger "$LEDGER"
            exit 3
        fi
        head1 "DIAGNOSTIC COMPUTE COMPLETE"
        say "Next, on CPU and as a separate step:"
        say "  python scripts/seq/diag_analysis.py --diag_root ${DIAG}"
        say "Detector results are PROVISIONAL until the blinded human annotation"
        say "is complete. No final scientific conclusion follows from them alone."
        budget report --ledger "$LEDGER"
        ;;
    *) fail "unknown stage '${STAGE}' (all|preflight|train|dev|select|test)" ;;
esac
