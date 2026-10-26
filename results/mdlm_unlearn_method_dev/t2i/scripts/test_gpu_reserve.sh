#!/bin/bash
# scripts/test_gpu_reserve.sh -- CPU-ONLY test harness for scripts/gpu_reserve.sh.
#
# Touches NO GPU and imports no CUDA. nvidia-smi is replaced by a shim ("fake_nvsmi") that reads a
# plain-text state file plus a $GPU_SHIM_MODE failure selector, so every GPU topology and every
# telemetry failure below is simulated. Safe to run on a frozen project while GPU jobs are in
# flight.
#
#   bash scripts/test_gpu_reserve.sh
#
# WHAT v1's "50 passed" DID AND DID NOT COVER
# -------------------------------------------
# v1's T1-T17 all ran with a FULLY WORKING shim, a production-shaped allowlist, no errexit, no
# flag-in-the-wrong-slot, and no surviving-child crash. They proved the lock algebra and the
# freeze gate, and those 50 assertions were real. But every one of the seven v2 defects lived in
# the region they never entered. T18-T30 below enter it.
#
# Tests:
#   T1  baseline: both allowed GPUs idle -> acquire returns 0
#   T2  second acquire takes the OTHER GPU, never the same one
#   T3  exhaustion: third acquire fails (exit 3) rather than double-booking
#   T4  CONCURRENCY: N processes race; exactly one winner per GPU, no index handed out twice
#   T5  release: a released GPU becomes available again
#   T6  stale reclamation: owner gone, nothing survives, record aged out -> reclaimed
#   T7  PID reuse: a reservation whose pid is alive but is a DIFFERENT process is stale
#   T8  another user's GPU is never selected
#   T9  off-limits GPUs 2-3 are never selected even when totally idle
#   T10 a pre-lock-scheme CUDA context of ours blocks that GPU (no reservation present)
#   T11 inter-cell memory dip does NOT release a held reservation
#   T12 trap-based release on crash via the sourceable helper
#   T13 regression witness: campaign_free.sh's /proc/environ clause is dead code
#   T14 D-013 freeze gate refuses every reservation
#   T15 the override needs BOTH the env var and the explicit argument
#   T16 the lead's bounded verification run: override works AND is logged
#   T17 D-013 witness: gpu_reserve.sh reads FREEZE_ACTIVE
#   --- v2 ---
#   T18 DEFECT 1: telemetry failure/empty/header-only/garbage/short/[N/A]/timeout => rc=5, REFUSE
#   T19 DEFECT 3: production allowlist cannot be widened by any env var
#   T20 DEFECT 7: explicit argument validation for gpu_reserve_hold
#   T21 DEFECT 5: a dead holder with a SURVIVING CHILD is ORPHANED, never reclaimed
#   T22 DEFECT 6: set -euo pipefail -- a nonzero acquisition reaches its intended handling
#   T23 DEFECT 4: inherited CUDA_VISIBLE_DEVICES is audited and overwritten; UUID mapping checked
#   T24 DEFECT 2: no process substitution or status-losing pipeline survives in the source
#   T25 RUNNER INTEGRATION: a real-shaped runner, set -euo pipefail, mocked nvidia-smi, end to end
#   T26 campaign_free.sh stays retired: nothing on stdout, non-zero exit
#   T27 the five migrated runners parse, use the API, and no longer call campaign_free.sh

set -u
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$SELF_DIR/.." && pwd)
RESERVE="$SELF_DIR/gpu_reserve.sh"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/gpu_reserve_test.XXXXXX")

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  \033[32mPASS\033[0m %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
hdr()  { printf '\n== %s\n' "$*"; }
# INFO prints context without counting as a passing assertion. Anything that cannot fail must use
# this, never ok(): a line that always passes inflates the pass count and asserts nothing. (G14.)
info() { printf '  \033[36mINFO\033[0m %s\n' "$*"; }
chk()  { if [ "$2" = "$3" ]; then ok "$1 (= $3)"; else bad "$1: expected [$3] got [$2]"; fi; }

# ---------------------------------------------------------------------------
# fake nvidia-smi. Deliberately NOT named nvidia-smi: gpu_reserve.sh refuses test mode if the
# telemetry command resolves to a real nvidia-smi, which is what stops an env var from widening
# the production allowlist.
#
# State file, one directive per line:
#   gpu <index> <uuid> <memory_used_MiB>
#   app <pid> <uuid>
# $GPU_SHIM_MODE selects a failure injection: ok|exit1|empty|headeronly|garbage|short|na|timeout|
#                                             appsfail|uuidonly
# ---------------------------------------------------------------------------
SHIM="$TMP/fake_nvsmi"
STATE="$TMP/gpu_state"
cat > "$SHIM" <<'SHIM_EOF'
#!/bin/bash
S="${GPU_STATE_FILE:?}"
M="${GPU_SHIM_MODE:-ok}"
q=""
for a in "$@"; do case "$a" in --query-gpu=*|--query-compute-apps=*) q="$a";; esac; done
case "$M" in
  exit1)      echo "NVML: Driver/library version mismatch" >&2; exit 1 ;;
  empty)      exit 0 ;;
  headeronly) case "$q" in
                --query-gpu=index,uuid) echo "index, uuid" ;;
                --query-gpu=index,memory.used) echo "index, memory.used [MiB]" ;;
                --query-compute-apps=pid,gpu_uuid) echo "pid, gpu_uuid" ;;
              esac; exit 0 ;;
  garbage)    echo "Unable to determine the device handle for GPU0000:1A:00.0: Unknown Error"; exit 0 ;;
  timeout)    sleep 30; exit 0 ;;
esac
# validbutfails: emit PERFECTLY VALID rows and THEN exit non-zero. Only a real exit-status check
# catches this; anything that merely parses stdout sails straight through.
if [ "$M" = validbutfails ]; then
  case "$q" in
    --query-gpu=index,uuid)            awk '$1=="gpu"{printf "%s, %s\n",$2,$3}' "$S" ;;
    --query-gpu=index,memory.used)     awk '$1=="gpu"{printf "%s, %s\n",$2,$4}' "$S" ;;
    --query-compute-apps=pid,gpu_uuid) awk '$1=="app"{printf "%s, %s\n",$2,$3}' "$S" ;;
  esac
  echo "nvidia-smi: a partial failure occurred" >&2
  exit 2
fi
case "$q" in
  --query-gpu=index,uuid)
      case "$M" in
        short) awk '$1=="gpu" && $2=="0" {printf "%s, %s\n",$2,$3}' "$S" ;;   # fewer rows than allowed GPUs
        *)     awk '$1=="gpu"{printf "%s, %s\n",$2,$3}' "$S" ;;
      esac ;;
  --query-gpu=index,memory.used)
      case "$M" in
        na)       awk '$1=="gpu"{printf "%s, [N/A]\n",$2}' "$S" ;;
        uuidonly) exit 0 ;;                                     # second view missing entirely
        *)        awk '$1=="gpu"{printf "%s, %s\n",$2,$4}' "$S" ;;
      esac ;;
  --query-compute-apps=pid,gpu_uuid)
      case "$M" in
        appsfail) echo "Failed to query compute apps" >&2; exit 9 ;;
        *)        awk '$1=="app"{printf "%s, %s\n",$2,$3}' "$S" ;;
      esac ;;
  *) exit 1 ;;
esac
SHIM_EOF
chmod +x "$SHIM"

export GPU_STATE_FILE="$STATE"
export GPU_RESERVE_NVSMI="$SHIM"
export GPU_RESERVE_GPUS="0 1"
export GPU_RESERVE_WAIT=20
export GPU_RESERVE_FOREIGN_MIB=512
export GPU_RESERVE_NVSMI_TIMEOUT=20
export GPU_SHIM_MODE=ok
# v2: overriding the allowlist or the freeze marker requires TEST MODE, which in turn requires a
# non-real nvidia-smi and a private lockdir. That separation IS defect 3's fix, so the harness
# must declare it.
export GPU_RESERVE_TEST_MODE=1
export GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE"
export GPU_RESERVE_OVERRIDE_LOG="$TMP/frozen_override.log"

NEWLOCKS() { export GPU_RESERVE_LOCKDIR="$TMP/locks.$1"; rm -rf "$GPU_RESERVE_LOCKDIR"; mkdir -p "$GPU_RESERVE_LOCKDIR"; }
NEWLOCKS boot

# four GPUs; 0-1 ours, 2-3 off limits. all idle.
idle_state() {
  cat > "$STATE" <<EOF
gpu 0 GPU-aaaa 10
gpu 1 GPU-bbbb 10
gpu 2 GPU-cccc 10
gpu 3 GPU-dddd 10
EOF
}

# A long-lived CPU-only placeholder process standing in for a campaign runner; prints its pid.
# stdout/stderr MUST be detached: spawn_holder is called inside $( ), and a background child that
# inherits the command-substitution pipe keeps it open, so the substitution would block.
spawn_holder() {
  ( exec -a gr_test_holder sleep 180 ) >/dev/null 2>&1 </dev/null &
  echo $!
}
cleanup() {
  rm -rf "$TMP"
  pkill -9 -u "$(id -u)" -f '^gr_test_holder' 2>/dev/null
  pkill -9 -u "$(id -u)" -f '^gr_test_child'  2>/dev/null
  pkill -9 -u "$(id -u)" -f '^gr_test_parent' 2>/dev/null
  return 0
}
trap cleanup EXIT

printf 'CPU-ONLY test harness for %s\n' "$RESERVE"
printf 'fake nvidia-smi = %s  (no GPU is touched, no CUDA is loaded)\n' "$SHIM"

# ---------------------------------------------------------------------------
hdr "T1/T2/T3  sequential acquire, distinct indices, then exhaustion"
NEWLOCKS basic; idle_state
H1=$(spawn_holder); H2=$(spawn_holder); H3=$(spawn_holder)
G1=$("$RESERVE" acquire camp_one   "$H1" 2>/dev/null); chk "first acquire"  "$G1" "0"
G2=$("$RESERVE" acquire camp_two   "$H2" 2>/dev/null); chk "second acquire takes the OTHER gpu" "$G2" "1"
G3=$("$RESERVE" acquire camp_three "$H3" 2>/dev/null); rc=$?
if [ "$rc" = "3" ] && [ -z "$G3" ]; then ok "third acquire refused (exit 3), no double-booking"
else bad "third acquire should fail with exit 3; got rc=$rc out=[$G3]"; fi

hdr "T5  release makes a GPU available again"
"$RESERVE" release camp_one "$H1" >/dev/null 2>&1
G4=$("$RESERVE" acquire camp_four "$H3" 2>/dev/null); chk "reacquire after release" "$G4" "0"
if [ -f "$GPU_RESERVE_LOCKDIR/gpu1.owner" ]; then ok "gpu1 record still present for its live holder"
else bad "gpu1's record vanished while its holder was still alive"; fi
kill "$H1" "$H2" "$H3" 2>/dev/null

# ---------------------------------------------------------------------------
hdr "T4  CONCURRENCY: 12 processes race for 2 GPUs"
NEWLOCKS race; idle_state
RES="$TMP/race_results"; : > "$RES"
N=12
for k in $(seq 1 $N); do
  (
    # MUST capture the pid BEFORE the command substitution: $BASHPID is re-evaluated inside $( ).
    me=$BASHPID
    g=$("$RESERVE" acquire "racer$k" "$me" 2>/dev/null)
    rc=$?
    printf '%s %s %s %s\n' "$k" "$me" "$rc" "${g:-NONE}" >> "$RES"
    sleep 4            # hold it, so the winners' pids stay alive for the audit
  ) &
done
wait
WINNERS=$(awk '$3==0{print $4}' "$RES" | sort)
LOSERS=$(awk '$3!=0' "$RES" | wc -l)
NWIN=$(printf '%s\n' "$WINNERS" | grep -c '[0-9]' || true)
UNIQ=$(printf '%s\n' "$WINNERS" | grep '[0-9]' | sort -u | tr '\n' ' ' | sed 's/ $//')
printf '  %s racers -> %s winner(s) [%s], %s refused\n' "$N" "$NWIN" "$UNIQ" "$LOSERS"
chk "exactly 2 winners for 2 GPUs" "$NWIN" "2"
chk "winners hold DISTINCT indices" "$UNIQ" "0 1"
chk "all other racers refused"      "$LOSERS" "$((N-2))"
if [ "$NWIN" = "2" ] && [ "$UNIQ" = "0 1" ]; then
  ok "no index was handed out twice (this is the campaign_free.sh defect, fixed)"
else
  bad "RACE LOST: an index was double-booked"
fi

# ---------------------------------------------------------------------------
hdr "T6  stale reclamation: owner gone, nothing survives, record aged past the grace window"
NEWLOCKS stale; idle_state
HC=$(spawn_holder)
GC=$("$RESERVE" acquire crashy "$HC" 2>/dev/null); chk "crashy reserved" "$GC" "0"
HO=$(spawn_holder)
GX=$("$RESERVE" acquire other "$HO" 2>/dev/null); chk "while owner lives, gpu0 is respected" "$GX" "1"
kill -9 "$HC" 2>/dev/null; wait "$HC" 2>/dev/null
"$RESERVE" release other "$HO" >/dev/null 2>&1    # free gpu1 so only gpu0 can be returned
# v2 protects a just-dead owner for GPU_RESERVE_ORPHAN_GRACE seconds (defect 5). Age the record
# past that window so this test exercises genuine staleness, not the grace path.
touch -d '2 hours ago' "$GPU_RESERVE_LOCKDIR/gpu0.owner"
HN=$(spawn_holder)
OUT=$("$RESERVE" acquire after_crash "$HN" 2>"$TMP/stale.err")
chk "gpu0 handed to the next campaign after the crash" "$OUT" "0"
if grep -q 'RECLAIMED stale reservation' "$TMP/stale.err"; then
  ok "reclamation was explicit: $(grep -o 'gpu0 RECLAIMED.*' "$TMP/stale.err" | head -1)"
else bad "expected an explicit stale-reclamation message; got: $(cat "$TMP/stale.err")"; fi
kill "$HO" "$HN" 2>/dev/null

hdr "T7  PID reuse does not resurrect a dead reservation"
NEWLOCKS pidreuse; idle_state
HP=$(spawn_holder)
"$RESERVE" acquire victim "$HP" >/dev/null 2>&1
# forge the record onto a DIFFERENT live process, keeping the original start time
LIVE=$(spawn_holder)
sed -i "s/^pid=.*/pid=$LIVE/" "$GPU_RESERVE_LOCKDIR/gpu0.owner"
sed -i "s/^pgid=.*/pgid=-/"   "$GPU_RESERVE_LOCKDIR/gpu0.owner"
touch -d '2 hours ago' "$GPU_RESERVE_LOCKDIR/gpu0.owner"
HR=$(spawn_holder)
OUT=$("$RESERVE" acquire newcomer "$HR" 2>"$TMP/reuse.err")
if grep -q 'RECLAIMED stale' "$TMP/reuse.err"; then
  ok "alive pid + mismatched start time => correctly judged stale (PID reuse defeated)"
else bad "PID-reuse guard did not fire: $(cat "$TMP/reuse.err")"; fi
kill "$HP" "$LIVE" "$HR" 2>/dev/null

# ---------------------------------------------------------------------------
hdr "T8  a GPU occupied by ANOTHER USER is never selected"
NEWLOCKS foreign
FOREIGN_PID=1                     # uid 0, definitely not us
cat > "$STATE" <<EOF
gpu 0 GPU-aaaa 21000
gpu 1 GPU-bbbb 10
gpu 2 GPU-cccc 10
gpu 3 GPU-dddd 10
app $FOREIGN_PID GPU-aaaa
EOF
HF=$(spawn_holder)
OUT=$("$RESERVE" acquire mine "$HF" 2>"$TMP/foreign.err")
chk "gpu0 (another user's) skipped, gpu1 chosen" "$OUT" "1"
if grep -qi "gpu0 carries a process that is not ours" "$TMP/foreign.err"; then
  ok "refusal reason recorded for the foreign GPU"
else bad "no foreign-GPU refusal logged: $(cat "$TMP/foreign.err")"; fi
cat >> "$STATE" <<EOF
app $FOREIGN_PID GPU-bbbb
EOF
"$RESERVE" release mine "$HF" >/dev/null 2>&1
HG=$(spawn_holder)
OUT=$("$RESERVE" acquire mine2 "$HG" 2>/dev/null); rc=$?
if [ "$rc" = "3" ] && [ -z "$OUT" ]; then ok "both GPUs foreign => refuses rather than displacing a user"
else bad "should have refused; rc=$rc out=[$OUT]"; fi
# invisible pid (hidepid simulation): a pid that does not exist in /proc at all
cat > "$STATE" <<EOF
gpu 0 GPU-aaaa 9000
gpu 1 GPU-bbbb 10
app 4194300 GPU-aaaa
EOF
OUT=$("$RESERVE" acquire mine3 "$HG" 2>"$TMP/hidepid.err")
chk "pid invisible in /proc treated as foreign, gpu1 chosen" "$OUT" "1"
kill "$HF" "$HG" 2>/dev/null

hdr "T9  off-limits GPUs 2-3 are never selected, even when idle"
NEWLOCKS offlimits
cat > "$STATE" <<EOF
gpu 0 GPU-aaaa 30000
gpu 1 GPU-bbbb 30000
gpu 2 GPU-cccc 0
gpu 3 GPU-dddd 0
app 1 GPU-aaaa
app 1 GPU-bbbb
EOF
HL=$(spawn_holder)
OUT=$("$RESERVE" acquire greedy "$HL" 2>/dev/null); rc=$?
if [ "$rc" = "3" ] && [ -z "$OUT" ]; then ok "GPUs 2-3 idle but out of scope => nothing selected (exit 3)"
else bad "LEAKED an off-limits GPU: rc=$rc out=[$OUT]"; fi
kill "$HL" 2>/dev/null

hdr "T10 a pre-lock-scheme CUDA context of OURS blocks that GPU"
NEWLOCKS legacy
LEG=$(spawn_holder)       # our uid, running, holds a context, but no reservation exists
cat > "$STATE" <<EOF
gpu 0 GPU-aaaa 7000
gpu 1 GPU-bbbb 10
app $LEG GPU-aaaa
EOF
HM=$(spawn_holder)
OUT=$("$RESERVE" acquire newscheme "$HM" 2>"$TMP/legacy.err")
chk "legacy job of ours on gpu0 respected, gpu1 chosen" "$OUT" "1"
if grep -q 'unreserved CUDA context of ours' "$TMP/legacy.err"; then
  ok "legacy job detected via compute-apps + uid, not via a memory threshold"
else bad "legacy detection did not fire: $(cat "$TMP/legacy.err")"; fi
"$RESERVE" release newscheme "$HM" >/dev/null 2>&1
AD=$("$RESERVE" adopt 0 legacy_adopted "$LEG" 2>/dev/null); chk "adopt gpu0 for the legacy pid" "$AD" "0"
kill "$LEG" "$HM" 2>/dev/null

hdr "T11 inter-cell memory DIP does not release a held reservation"
NEWLOCKS dip; idle_state
HD=$(spawn_holder)
GD=$("$RESERVE" acquire long_campaign "$HD" 2>/dev/null); chk "campaign holds gpu0" "$GD" "0"
cat > "$STATE" <<EOF
gpu 0 GPU-aaaa 0
gpu 1 GPU-bbbb 0
EOF
HE=$(spawn_holder)
OUT=$("$RESERVE" acquire intruder "$HE" 2>/dev/null)
chk "gpu0 still reserved despite 0 MiB used; intruder sent to gpu1" "$OUT" "1"
if [ "$OUT" != "0" ]; then ok "memory dip no longer leaks the GPU (campaign_free.sh defect (c), fixed)"
else bad "a 0 MiB reading leaked a HELD gpu0 -- campaign_free.sh defect (c) is back"; fi
kill "$HD" "$HE" 2>/dev/null

hdr "T12 trap-based release when the holding shell dies"
NEWLOCKS trap; idle_state
cat > "$TMP/fake_runner.sh" <<EOF
#!/bin/bash
set -u
source "$RESERVE"
gpu_reserve_hold trapped_campaign >/dev/null
echo \$GPU_RESERVED > "$TMP/trap_gpu"
sleep 60
EOF
chmod +x "$TMP/fake_runner.sh"
"$TMP/fake_runner.sh" & RUNNER=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "$TMP/trap_gpu" ] && break; sleep 0.4; done
chk "sourced helper reserved a gpu" "$(cat "$TMP/trap_gpu" 2>/dev/null)" "0"
[ -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ] && ok "reservation record present while runner lives" \
  || bad "no reservation record written"
kill -TERM "$RUNNER" 2>/dev/null; wait "$RUNNER" 2>/dev/null
sleep 0.5
if [ ! -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ]; then
  ok "EXIT/TERM trap released the reservation on shell death"
else
  bad "trap did not release gpu0 (v2 will NOT silently reclaim it: see T21)"
fi

# ---------------------------------------------------------------------------
hdr "T13 regression witness: campaign_free.sh's /proc/environ clause is dead code"
cat > "$TMP/run_witness.sh" <<'EOF'
#!/bin/bash
set -u
export CUDA_VISIBLE_DEVICES=0
sleep 20
EOF
chmod +x "$TMP/run_witness.sh"
"$TMP/run_witness.sh" & W=$!
sleep 1
SEEN=$(tr '\0' '\n' < "/proc/$W/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
if [ -z "$SEEN" ]; then
  ok "a runner that exports CUDA_VISIBLE_DEVICES after startup is INVISIBLE in /proc/<pid>/environ"
  ok "=> campaign_free.sh clause (a) can never match any of its 8 callers"
else
  bad "environ unexpectedly showed CUDA_VISIBLE_DEVICES=$SEEN"
fi
kill "$W" 2>/dev/null; wait "$W" 2>/dev/null

# ---------------------------------------------------------------------------
hdr "T14 D-013 freeze gate: a FREEZE_ACTIVE marker REFUSES every reservation"
NEWLOCKS freeze; idle_state
MARKER="$TMP/FREEZE_ACTIVE"
cat > "$MARKER" <<'EOF'
REPAIR-FIRST FREEZE, declared 2026-10-26T12:43-04:00.
No new campaign cells may start. Bounded VERIFICATION runs only.
EOF
export GPU_RESERVE_FREEZE_FILE="$MARKER"
HZ=$(spawn_holder)
OUT=$("$RESERVE" acquire while_frozen "$HZ" 2>"$TMP/freeze.err"); rc=$?
if [ "$rc" != 0 ] && [ -z "$OUT" ]; then ok "refused while frozen, non-zero exit (rc=$rc), no index returned"
else bad "FREEZE BYPASSED: rc=$rc out=[$OUT]"; fi
chk "refusal exit code is 4 (distinct from 3 = no GPU free)" "$rc" "4"
grep -q 'REFUSED: the project is FROZEN' "$TMP/freeze.err" \
  && ok "refusal is loud on stderr" || bad "no refusal message on stderr"
grep -q 'REPAIR-FIRST FREEZE, declared' "$TMP/freeze.err" \
  && ok "marker contents echoed to stderr, so the caller learns WHY" || bad "marker contents not printed"
[ ! -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ] && ok "no reservation record was written while frozen" \
  || bad "a reservation record leaked while frozen"

hdr "T15 the override needs BOTH the env var and the explicit argument"
OUT=$(ALLOW_FROZEN_RUN=1 "$RESERVE" acquire env_only "$HZ" 2>"$TMP/envonly.err"); rc=$?
if [ "$rc" = 4 ] && [ -z "$OUT" ]; then ok "ambient ALLOW_FROZEN_RUN=1 alone does NOT lift the freeze (rc=4)"
else bad "env var alone lifted the freeze: rc=$rc out=[$OUT]"; fi
grep -q 'but --allow-frozen was NOT passed' "$TMP/envonly.err" \
  && ok "explains that an inherited env var is not enough" || bad "no explanation of the missing argument"
OUT=$("$RESERVE" acquire flag_only "$HZ" --allow-frozen 2>"$TMP/flagonly.err"); rc=$?
if [ "$rc" = 4 ] && [ -z "$OUT" ]; then ok "--allow-frozen alone does NOT lift the freeze (rc=4)"
else bad "flag alone lifted the freeze: rc=$rc out=[$OUT]"; fi
grep -q 'BOTH are required' "$TMP/flagonly.err" && ok "says both are required" || bad "no 'both required' message"

hdr "T16 the lead's bounded verification run: override works AND is logged"
rm -f "$GPU_RESERVE_OVERRIDE_LOG"
OUT=$(ALLOW_FROZEN_RUN=1 "$RESERVE" acquire lead_verify "$HZ" --allow-frozen 2>"$TMP/ovr.err"); rc=$?
chk "override grants a GPU" "$OUT" "0"
chk "override exits 0" "$rc" "0"
grep -q 'FREEZE OVERRIDE in effect' "$TMP/ovr.err" && ok "override is announced on stderr" \
  || bad "override not announced"
if [ -s "$GPU_RESERVE_OVERRIDE_LOG" ]; then
  LINE=$(tail -1 "$GPU_RESERVE_OVERRIDE_LOG")
  ok "override log line written: $LINE"
  for fld in "pid=$HZ" "campaign=lead_verify" "gpu=0" "override=" "uuid=GPU-aaaa"; do
    case "$LINE" in *"$fld"*) ok "log records $fld" ;; *) bad "log missing $fld" ;; esac
  done
  case "$LINE" in 2026-*|20[0-9][0-9]-*) ok "log line is timestamped" ;; *) bad "log line not timestamped" ;; esac
else
  bad "no override log written at $GPU_RESERVE_OVERRIDE_LOG"
fi
HY=$(spawn_holder)
OUT=$("$RESERVE" acquire sneaky "$HY" 2>/dev/null); rc=$?
chk "a non-override caller is still refused after an override was granted" "$rc" "4"
rm -f "$MARKER"
"$RESERVE" release lead_verify "$HZ" >/dev/null 2>&1
OUT=$("$RESERVE" acquire after_thaw "$HY" 2>/dev/null)
chk "marker removed => normal acquire works again" "$OUT" "0"
kill "$HZ" "$HY" 2>/dev/null
export GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE"

hdr "T17 D-013 witness: gpu_reserve.sh reads FREEZE_ACTIVE"
READERS=$(grep -rl --include='*.sh' --include='*.py' 'FREEZE_ACTIVE' "$PROJ" 2>/dev/null \
          | grep -v '/upstream/' | grep -v '/venv_' | sed "s|$PROJ/||" | sort | tr '\n' ' ')
printf '  files referencing FREEZE_ACTIVE: %s\n' "${READERS:-<none>}"
case "$READERS" in
  *gpu_reserve.sh*) ok "gpu_reserve.sh is a reader of the freeze marker" ;;
  *) bad "gpu_reserve.sh does not reference FREEZE_ACTIVE" ;;
esac

# ===========================================================================
# v2 DEFECT TESTS
# ===========================================================================
hdr "T18 DEFECT 1: a telemetry failure is UNKNOWN, never 'free'"
# v1 result, measured before the fix: nvidia-smi exit 1 => rc=0 and GPU 0 handed out; the same for
# no output at all and for memory.used='[N/A]'. A GPU was reserved on telemetry never read.
for mode in exit1 empty headeronly garbage short na uuidonly appsfail; do
  NEWLOCKS "tel.$mode"; idle_state
  HT=$(spawn_holder)
  OUT=$(GPU_SHIM_MODE="$mode" "$RESERVE" acquire "tel_$mode" "$HT" 2>"$TMP/tel.$mode.err"); rc=$?
  if [ "$rc" = 5 ] && [ -z "$OUT" ]; then
    ok "nvidia-smi '$mode' => rc=5 TELEMETRY UNKNOWN, no index ($(grep -m1 -o 'TELEMETRY UNKNOWN:.*' "$TMP/tel.$mode.err" | cut -c1-96))"
  else
    bad "nvidia-smi '$mode' => rc=$rc out=[$OUT]; a GPU must NEVER be granted on unreadable telemetry"
  fi
  [ ! -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ] && ok "  '$mode': no reservation record written" \
    || bad "  '$mode': a reservation record leaked on unknown telemetry"
  kill "$HT" 2>/dev/null
done
NEWLOCKS tel.timeout; idle_state
HT=$(spawn_holder)
OUT=$(GPU_SHIM_MODE=timeout GPU_RESERVE_NVSMI_TIMEOUT=2 "$RESERVE" acquire tel_timeout "$HT" 2>"$TMP/tel.timeout.err"); rc=$?
if [ "$rc" = 5 ] && [ -z "$OUT" ]; then ok "nvidia-smi that HANGS => rc=5 after the timeout, no index"
else bad "hanging nvidia-smi => rc=$rc out=[$OUT]"; fi
grep -q 'TIMED OUT' "$TMP/tel.timeout.err" && ok "  timeout is named as the reason" || bad "  timeout not reported"
kill "$HT" 2>/dev/null
# and a reservation already held is NEVER reclaimed while telemetry is unknown
NEWLOCKS tel.hold; idle_state
HH=$(spawn_holder)
"$RESERVE" acquire held_camp "$HH" >/dev/null 2>&1
kill -9 "$HH" 2>/dev/null; wait "$HH" 2>/dev/null
touch -d '2 hours ago' "$GPU_RESERVE_LOCKDIR/gpu0.owner"
HI=$(spawn_holder)
OUT=$(GPU_SHIM_MODE=exit1 "$RESERVE" acquire tel_intruder "$HI" 2>/dev/null); rc=$?
chk "unknown telemetry never reclaims someone else's record either (rc)" "$rc" "5"
[ -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ] && ok "the existing record survived an unknown-telemetry pass" \
  || bad "a record was destroyed while telemetry was unknown"
kill "$HI" 2>/dev/null
GPU_SHIM_MODE=ok "$RESERVE" selftest-telemetry >/dev/null 2>&1 \
  && ok "selftest-telemetry exits 0 on good telemetry" || bad "selftest-telemetry failed on good telemetry"
GPU_SHIM_MODE=exit1 "$RESERVE" selftest-telemetry >/dev/null 2>&1
[ "$?" = 5 ] && ok "selftest-telemetry exits 5 on bad telemetry" || bad "selftest-telemetry wrong rc on bad telemetry"

hdr "T19 DEFECT 3: no environment variable can widen the production allowlist"
# v1 result, measured before the fix: GPU_RESERVE_GPUS='2 3' => rc=0, index 2 handed out --
# another user's GPU, from an env var alone.
HV=$(spawn_holder)
for gset in "2 3" "0 1 2 3" "3" "0 1 2"; do
  OUT=$(env -u GPU_RESERVE_TEST_MODE -u GPU_RESERVE_NVSMI -u GPU_RESERVE_LOCKDIR \
        -u GPU_RESERVE_FREEZE_FILE -u GPU_RESERVE_OVERRIDE_LOG GPU_RESERVE_GPUS="$gset" \
        "$RESERVE" acquire prod_widen "$HV" 2>"$TMP/widen.err"); rc=$?
  if [ "$rc" = 6 ] && [ -z "$OUT" ]; then ok "production refuses GPU_RESERVE_GPUS='$gset' (rc=6)"
  else bad "PRODUCTION ALLOWLIST WIDENED by GPU_RESERVE_GPUS='$gset': rc=$rc out=[$OUT]"; fi
done
grep -q 'OFF LIMITS' "$TMP/widen.err" && ok "refusal explains GPUs 2-3 are off limits" || bad "no off-limits explanation"
# test mode cannot be switched on while talking to the real driver
OUT=$(GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="2 3" GPU_RESERVE_NVSMI="nvidia-smi" \
      GPU_RESERVE_LOCKDIR="$TMP/l.realnv" "$RESERVE" acquire real_nvsmi_test "$HV" 2>"$TMP/realnv.err"); rc=$?
if [ "$rc" = 6 ] && [ -z "$OUT" ]; then ok "TEST_MODE=1 with a REAL nvidia-smi is refused (rc=6) -- the interlock"
else bad "test mode accepted the real driver: rc=$rc out=[$OUT]"; fi
# test mode may not write into the production lockdir
OUT=$(GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_NVSMI="$SHIM" GPU_RESERVE_LOCKDIR="$PROJ/.gpulocks" \
      "$RESERVE" acquire prodlock "$HV" 2>/dev/null); rc=$?
chk "test mode refuses the production lockdir" "$rc" "6"
# production refuses a redirected freeze marker (a freeze must not be env-thawable)
OUT=$(env -u GPU_RESERVE_TEST_MODE -u GPU_RESERVE_NVSMI -u GPU_RESERVE_LOCKDIR \
      -u GPU_RESERVE_OVERRIDE_LOG GPU_RESERVE_FREEZE_FILE=/dev/null/nope \
      "$RESERVE" acquire fake_thaw "$HV" 2>"$TMP/thaw.err"); rc=$?
if [ "$rc" = 6 ]; then ok "production refuses a redirected GPU_RESERVE_FREEZE_FILE (rc=6)"
else bad "the freeze marker path was redirectable in production: rc=$rc"; fi
# and the real production freeze still bites
OUT=$(env -u GPU_RESERVE_TEST_MODE -u GPU_RESERVE_GPUS -u GPU_RESERVE_FREEZE_FILE \
      -u GPU_RESERVE_LOCKDIR -u GPU_RESERVE_NVSMI -u GPU_RESERVE_OVERRIDE_LOG \
      "$RESERVE" acquire prod_frozen "$HV" 2>"$TMP/prodfrozen.err"); rc=$?
if [ -e "$PROJ/FREEZE_ACTIVE" ]; then
  chk "production acquire is refused by the real FREEZE_ACTIVE (rc=4)" "$rc" "4"
else
  ok "(project is not frozen right now; production freeze path not exercised)"
fi
kill "$HV" 2>/dev/null

hdr "T19b FAIL-OPEN: an EMPTY or MALFORMED allowlist refuses; it never falls back to a default"
# Reported by the Baselines agent, which hit it while integrating: ${VAR:-default} treats
# SET-BUT-EMPTY as UNSET, so narrowing the allowlist to nothing yielded the FULL default set.
# Measured before the fix:
#   test mode  GPU_RESERVE_GPUS=""    -> rc=0, RESERVED GPU 0
#   production GPU_RESERVE_GPUS=""    -> status reported `allowed=0 1` (acquire's rc=4 was the
#                                        FREEZE refusing first and masking the fallback)
#   test mode  GPU_RESERVE_GPUS="   " -> rc=3, the RETRYABLE code, so a waiter looped forever
NEWLOCKS failopen; idle_state
HN2=$(spawn_holder)
# --- test mode: every unusable value must be rc=7 and reserve NOTHING
for badset in "" "   " "	" "0 x" "x" "0 0" "-1" "0,1"; do
  LD="$TMP/locks.fo.$(printf '%s' "$badset" | tr -c 'a-z0-9' _)"
  rm -rf "$LD"; mkdir -p "$LD"
  OUT=$(GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="$badset" GPU_RESERVE_NVSMI="$SHIM" \
        GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE" GPU_RESERVE_LOCKDIR="$LD" \
        "$RESERVE" acquire failopen "$HN2" 2>"$TMP/fo.err"); rc=$?
  if [ "$rc" = 7 ] && [ -z "$OUT" ]; then ok "GPU_RESERVE_GPUS=[$badset] => rc=7, nothing reserved"
  else bad "FAIL-OPEN: GPU_RESERVE_GPUS=[$badset] => rc=$rc out=[$OUT] (expected rc=7, no index)"; fi
  if [ -n "$(ls -A "$LD" 2>/dev/null | grep -v reserve.mutex || true)" ]; then
    bad "  [$badset] wrote a reservation record"
  else
    ok "  [$badset] wrote no reservation record"
  fi
done
grep -q 'SET BUT EMPTY' "$TMP/fo.err" >/dev/null 2>&1 || true
OUT=$(GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="" GPU_RESERVE_NVSMI="$SHIM" \
      GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE" GPU_RESERVE_LOCKDIR="$TMP/locks.fo2" \
      "$RESERVE" acquire failopen2 "$HN2" 2>"$TMP/fo2.err"); rc=$?
grep -q 'SET BUT EMPTY' "$TMP/fo2.err" && ok "the empty case names itself as a configuration error" \
  || bad "no 'SET BUT EMPTY' diagnosis: $(head -2 "$TMP/fo2.err")"
grep -q 'non-retryable' "$TMP/fo2.err" && ok "and says it is NON-RETRYABLE (rc=7, not the retryable 3)" \
  || bad "retryability not stated"
chk "rc=7 is distinct from rc=3 'no GPU free'" "$rc" "7"
# --- UNSET must still work: the fix must not break the normal path
OUT=$(env -u GPU_RESERVE_GPUS GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_NVSMI="$SHIM" \
      GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE" GPU_RESERVE_LOCKDIR="$TMP/locks.fo3" \
      "$RESERVE" acquire unset_ok "$HN2" 2>/dev/null)
chk "GPU_RESERVE_GPUS genuinely UNSET still uses the default and works" "$OUT" "0"
# --- PRODUCTION: empty/malformed refuse, and the computed allowlist never falls back
for badset in "" "   " "0 x" "0 0"; do
  OUT=$(env -u GPU_RESERVE_TEST_MODE -u GPU_RESERVE_FREEZE_FILE -u GPU_RESERVE_LOCKDIR \
        -u GPU_RESERVE_NVSMI -u GPU_RESERVE_OVERRIDE_LOG GPU_RESERVE_GPUS="$badset" \
        "$RESERVE" acquire prod_failopen "$HN2" 2>/dev/null); rc=$?
  if [ "$rc" = 7 ] && [ -z "$OUT" ]; then ok "production: GPU_RESERVE_GPUS=[$badset] => rc=7"
  else bad "production FAIL-OPEN: GPU_RESERVE_GPUS=[$badset] => rc=$rc out=[$OUT]"; fi
done
# `status` is not freeze-gated, so it is the honest witness for what the allowlist RESOLVED to --
# this is how the production fallback was proven before the fix (acquire's rc=4 masked it).
SOUT=$(env -u GPU_RESERVE_TEST_MODE -u GPU_RESERVE_FREEZE_FILE -u GPU_RESERVE_LOCKDIR \
       -u GPU_RESERVE_NVSMI -u GPU_RESERVE_OVERRIDE_LOG GPU_RESERVE_GPUS="" \
       "$RESERVE" status 2>&1); rc=$?
case "$SOUT" in
  *"allowed=0 1"*) bad "status still resolves an EMPTY allowlist to the default set 0 1" ;;
  *)               ok "status no longer resolves an empty allowlist to the default set" ;;
esac
chk "status itself refuses an unusable allowlist" "$rc" "7"
# --- interaction with the hard-pinned production set: empty / malformed / out-of-range
printf '  interaction matrix against the production pin {0,1}:\n'
for pair in ":7:empty" "   :7:whitespace" "0 x:7:malformed" "2 3:6:out-of-range" "0 1 2 3:6:widened" "0 1:0:exact-match"; do
  gs=${pair%%:*}; rest=${pair#*:}; want=${rest%%:*}; label=${rest#*:}
  rm -rf "$TMP/locks.mx"; mkdir -p "$TMP/locks.mx"
  OUT=$(env -u GPU_RESERVE_TEST_MODE -u GPU_RESERVE_FREEZE_FILE -u GPU_RESERVE_NVSMI \
        -u GPU_RESERVE_LOCKDIR -u GPU_RESERVE_OVERRIDE_LOG GPU_RESERVE_GPUS="$gs" \
        "$RESERVE" status 2>/dev/null >/dev/null); rc=$?
  if [ "$rc" = "$want" ]; then ok "  production + GPU_RESERVE_GPUS=[$gs] ($label) => rc=$rc"
  else bad "  production + GPU_RESERVE_GPUS=[$gs] ($label) => rc=$rc, expected $want"; fi
done
# --- the sourced API refuses too, rather than running on the pre-validation defaults
O=$(GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="" GPU_RESERVE_NVSMI="$SHIM" \
    GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE" GPU_RESERVE_LOCKDIR="$TMP/locks.fo4" \
    timeout 30 bash -c 'set -u; source "$1"; gpu_reserve_hold src_empty 5 >/dev/null; echo "rc=$?"' _ "$RESERVE" 2>&1)
case "$O" in *"rc=7"*) ok "a sourced gpu_reserve_hold also refuses rc=7 on an empty allowlist" ;;
  *) bad "sourced helper did not refuse: $O" ;; esac
case "$O" in *"REFUSED configuration"*) ok "  and the refusal is announced at source time" ;;
  *) bad "  no source-time announcement: $O" ;; esac
kill "$HN2" 2>/dev/null

hdr "T19d BLOCKING: production constants cannot be relocated by the environment (D-001 vector)"
# Reviewer finding. Production honoured GPU_RESERVE_LOCKDIR with no refusal:
#   GPU_RESERVE_LOCKDIR=/tmp/evil bash scripts/gpu_reserve.sh status
#     -> mode=production   lockdir=/tmp/evil
# The mutex lives at $LOCKDIR/.reserve.mutex, so two runners with different values take DIFFERENT
# mutexes and cannot exclude each other -- the exact 2026-10-26T12:39 triple-booking of GPU 0.
HPC=$(spawn_holder)
for pair in "GPU_RESERVE_LOCKDIR=$TMP/evil_lockdir:lockdir (the D-001 vector)" \
            "GPU_RESERVE_NVSMI=$SHIM:telemetry source" \
            "GPU_RESERVE_OVERRIDE_LOG=/dev/null:freeze-override audit log" \
            "GPU_RESERVE_FREEZE_FILE=$TMP/NO_SUCH_FREEZE:freeze marker" \
            "GPU_RESERVE_GPUS=2 3:allowlist"; do
  kv=${pair%%:*}; label=${pair#*:}
  rc=$(env -u GPU_RESERVE_TEST_MODE -u GPU_RESERVE_GPUS -u GPU_RESERVE_FREEZE_FILE \
           -u GPU_RESERVE_NVSMI -u GPU_RESERVE_LOCKDIR -u GPU_RESERVE_OVERRIDE_LOG \
           "$kv" bash "$RESERVE" status >/dev/null 2>&1; echo $?)
  if [ "$rc" = 6 ]; then ok "production refuses to relocate the $label (rc=6)"
  else bad "PRODUCTION CONSTANT OVERRIDABLE: $kv gave rc=$rc, expected 6"; fi
done
# and the mutex really does live inside the lockdir, which is why relocating it breaks exclusion
grep -q 'MUTEX="$LOCKDIR/.reserve.mutex"' "$RESERVE" \
  && ok "the mutex path is derived from LOCKDIR (hence relocating LOCKDIR splits the lock)" \
  || bad "could not confirm the mutex lives inside the lockdir"
# DEMONSTRATE the split-lock consequence under test mode, where two lockdirs ARE permitted:
# two acquirers with different lockdirs both win index 0. This is what production now forbids.
rm -rf "$TMP/split.a" "$TMP/split.b"; mkdir -p "$TMP/split.a" "$TMP/split.b"
idle_state
HA=$(spawn_holder); HB=$(spawn_holder)
GA=$(GPU_RESERVE_LOCKDIR="$TMP/split.a" "$RESERVE" acquire split_a "$HA" 2>/dev/null)
GB=$(GPU_RESERVE_LOCKDIR="$TMP/split.b" "$RESERVE" acquire split_b "$HB" 2>/dev/null)
if [ "$GA" = "0" ] && [ "$GB" = "0" ]; then
  ok "witness: two DIFFERENT lockdirs both hand out index 0 => split mutex = D-001 (now production-refused)"
else
  bad "expected the split-lockdir double-book witness to reproduce; got [$GA] and [$GB]"
fi
kill "$HPC" "$HA" "$HB" 2>/dev/null

hdr "T19e BLOCKING: every numeric/enum knob is range-checked (fail-open sweep)"
# Reviewer finding: GPU_RESERVE_FOREIGN_MIB and GPU_RESERVE_ORPHAN_GRACE were unvalidated.
# Measured before the fix, GPU at 30000 MiB with no attributable process:
#   FOREIGN_MIB=512 -> rc=3 (refused)   FOREIGN_MIB=abc -> rc=0 RESERVED   FOREIGN_MIB=99999999 -> rc=0 RESERVED
cat > "$TMP/busy_state" <<'EOF'
gpu 0 GPU-aaaa 30000
gpu 1 GPU-bbbb 30000
EOF
HV2=$(spawn_holder)
for pair in "GPU_RESERVE_FOREIGN_MIB=abc" "GPU_RESERVE_FOREIGN_MIB=99999999" "GPU_RESERVE_FOREIGN_MIB=0" \
            "GPU_RESERVE_FOREIGN_MIB=" "GPU_RESERVE_ORPHAN_GRACE=abc" "GPU_RESERVE_ORPHAN_GRACE=0" \
            "GPU_RESERVE_ORPHAN_GRACE=" "GPU_RESERVE_ORPHAN_GRACE=5" "GPU_RESERVE_NVSMI_TIMEOUT=abc" \
            "GPU_RESERVE_NVSMI_TIMEOUT=0" "GPU_RESERVE_NVSMI_TIMEOUT=" "GPU_RESERVE_WAIT=abc" \
            "GPU_RESERVE_WAIT=0" "GPU_RESERVE_WAIT=" "GPU_RESERVE_UID=abc" \
            "GPU_RESERVE_CVD_STYLE=banana" "GPU_RESERVE_TEST_MODE=2" "GPU_RESERVE_TEST_MODE=yes"; do
  rm -rf "$TMP/locks.rng"; mkdir -p "$TMP/locks.rng"
  OUT=$(env GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="0 1" GPU_RESERVE_NVSMI="$SHIM" \
            GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE" GPU_RESERVE_LOCKDIR="$TMP/locks.rng" \
            GPU_STATE_FILE="$TMP/busy_state" "$pair" \
            bash "$RESERVE" acquire rangetest "$HV2" 2>/dev/null); rc=$?
  if [ "$rc" = 7 ] && [ -z "$OUT" ]; then ok "$pair => rc=7, nothing reserved"
  else bad "FAIL-OPEN: $pair => rc=$rc out=[$OUT] (expected rc=7, no index)"; fi
done
# the behavioural core: a BUSY, unattributable GPU must stay refused at the sane default
rm -rf "$TMP/locks.sane"; mkdir -p "$TMP/locks.sane"
OUT=$(env GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="0 1" GPU_RESERVE_NVSMI="$SHIM" \
          GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE" GPU_RESERVE_LOCKDIR="$TMP/locks.sane" \
          GPU_STATE_FILE="$TMP/busy_state" GPU_RESERVE_FOREIGN_MIB=512 \
          bash "$RESERVE" acquire sane "$HV2" 2>/dev/null); rc=$?
if [ "$rc" = 3 ] && [ -z "$OUT" ]; then ok "at the default threshold a 30000 MiB unattributable GPU is still refused (rc=3)"
else bad "the UNACCOUNTED guard regressed: rc=$rc out=[$OUT]"; fi
# all knobs at valid non-default values must still work
rm -rf "$TMP/locks.valid"; mkdir -p "$TMP/locks.valid"
OUT=$(env GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="0 1" GPU_RESERVE_NVSMI="$SHIM" \
          GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE" GPU_RESERVE_LOCKDIR="$TMP/locks.valid" \
          GPU_RESERVE_FOREIGN_MIB=1024 GPU_RESERVE_ORPHAN_GRACE=600 GPU_RESERVE_NVSMI_TIMEOUT=30 \
          GPU_RESERVE_WAIT=15 GPU_RESERVE_CVD_STYLE=uuid \
          bash "$RESERVE" acquire validknobs "$HV2" 2>/dev/null)
chk "valid non-default values for every knob still acquire" "$OUT" "0"
kill "$HV2" 2>/dev/null
idle_state

hdr "T19f BLOCKING: the test-mode interlock cannot be forged by COPYING nvidia-smi"
# Reviewer finding: the interlock compared the BASENAME, so `cp /usr/bin/nvidia-smi fake` passed and
# TEST_MODE=1 + GPUS="2 3" handed out index 2 against REAL telemetry. GPUs 2-3 are another user's.
REALNV=$(command -v nvidia-smi 2>/dev/null || true)
if [ -n "$REALNV" ]; then
  ln -sf "$REALNV" "$TMP/sym_nvsmi"
  if cp "$REALNV" "$TMP/cp_nvsmi" 2>/dev/null; then
    chmod +x "$TMP/cp_nvsmi"
    for cand in sym_nvsmi cp_nvsmi; do
      rm -rf "$TMP/locks.forge"; mkdir -p "$TMP/locks.forge"
      rc=$(env GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="2 3" GPU_RESERVE_NVSMI="$TMP/$cand" \
               GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE" GPU_RESERVE_LOCKDIR="$TMP/locks.forge" \
               bash "$RESERVE" status >/dev/null 2>&1; echo $?)
      if [ "$rc" = 6 ]; then ok "a $cand of the real nvidia-smi is REFUSED test mode (rc=6)"
      else bad "INTERLOCK FORGED via $cand: rc=$rc -- test mode granted against real telemetry"; fi
    done
    # and specifically that GPUs 2-3 never became reachable
    OUT=$(env GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="2 3" GPU_RESERVE_NVSMI="$TMP/cp_nvsmi" \
              GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE" GPU_RESERVE_LOCKDIR="$TMP/locks.forge" \
              bash "$RESERVE" status 2>/dev/null | grep -c 'allowed=2 3' || true)
    chk "no 'allowed=2 3' is ever printed for a copied binary" "$OUT" "0"
    O=$(env GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="2 3" GPU_RESERVE_NVSMI="$TMP/cp_nvsmi" \
             GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE" GPU_RESERVE_LOCKDIR="$TMP/locks.forge" \
             bash "$RESERVE" status 2>&1 >/dev/null)
    case "$O" in *"REAL DRIVER also reports"*) ok "the refusal is on DEVICE IDENTITY, not on the filename" ;;
      *) bad "the refusal did not cite device identity: $(printf '%s' "$O" | head -2)" ;; esac
  else
    info "could not copy $REALNV (permissions); the copy-forgery case was not exercised here"
  fi
else
  info "no real nvidia-smi on PATH; the forgery case cannot be exercised on this host"
fi
# the legitimate synthetic shim must still be accepted, or the fix would be useless
rm -rf "$TMP/locks.syn"; mkdir -p "$TMP/locks.syn"
rc=$(GPU_RESERVE_LOCKDIR="$TMP/locks.syn" "$RESERVE" status >/dev/null 2>&1; echo $?)
chk "a genuinely synthetic shim is still granted test mode" "$rc" "0"

hdr "T19g BLOCKING: an UNRECOGNISED reservation state is never claimable (default-deny)"
# Reviewer finding: _gr_do_acquire's state case had no *) arm, so an unknown state fell through to
# the claim. Driven here by writing a record the state machine cannot classify.
grep -q 'UNRECOGNISED reservation state' "$RESERVE" \
  && ok "the state machine has an explicit default-deny arm" || bad "no default-deny arm for states"
grep -q "UNRECOGNISED classification" "$RESERVE" \
  && ok "the classifier has an explicit default-deny arm too" || bad "no default-deny arm for classes"
# behavioural: only an explicit FREE proceeds. A record for a LIVE owner must block, and the
# arms are ordered so that nothing reaches _gr_write_record without matching FREE in both cases.
NEWLOCKS defaultdeny; idle_state
HDD=$(spawn_holder)
"$RESERVE" acquire dd_owner "$HDD" >/dev/null 2>&1
# corrupt the record so no field parses: pid/start/pgid all unreadable
printf 'garbage-not-a-field\n' > "$GPU_RESERVE_LOCKDIR/gpu0.owner"
HDE=$(spawn_holder)
OUT=$("$RESERVE" acquire dd_intruder "$HDE" 2>"$TMP/dd.err")
if [ "$OUT" = "0" ]; then
  bad "a corrupt/unparsable reservation record let gpu0 be claimed"
else
  ok "a corrupt reservation record does NOT yield gpu0 (got [${OUT:-nothing}])"
fi
kill "$HDD" "$HDE" 2>/dev/null

hdr "T21b the pgid survivor check works even when the acquirer SHARES the dead holder's pgid"
# Reviewer finding: _gr_live_in_pgid returned early when the acquirer's pgid equalled the record's,
# which is the common case for two scripts launched from one non-interactive parent. The reviewer
# reproduced STALE instead of ORPHAN there, leaving the grace window as the only protection.
# Fixed by excluding our own process TREE instead of skipping the group.
NEWLOCKS pgidshare; idle_state
export GPU_RESERVE_ORPHAN_GRACE=60      # minimum allowed; must not be what saves this test
# parent and child share THIS script's process group (no setsid), exactly the reviewer's setup
( exec -a gr_test_parent sleep 180 ) >/dev/null 2>&1 </dev/null & PS1P=$!
( exec -a gr_test_child  sleep 180 ) >/dev/null 2>&1 </dev/null & PS1C=$!
sleep 0.3
MYPG=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
HOLDPG=$(ps -o pgid= -p "$PS1P" 2>/dev/null | tr -d ' ')
info "acquirer pgid=$MYPG  holder pgid=$HOLDPG (equal => the previously-skipped case)"
"$RESERVE" acquire pgidshare "$PS1P" >/dev/null 2>&1
kill -9 "$PS1P" 2>/dev/null; wait "$PS1P" 2>/dev/null
touch -d '2 hours ago' "$GPU_RESERVE_LOCKDIR/gpu0.owner"   # age past ANY grace window
HPS=$(spawn_holder)
OUT=$("$RESERVE" acquire pgid_intruder "$HPS" 2>"$TMP/pgid.err")
# HONEST OUTCOME, and the reason the fix is partial. For a NON-LEADER owner (the shared-pgid case)
# the process group also contains the parent and every unrelated sibling, so "a group member is
# alive" is not evidence of descent: treating it as such made a GPU permanently unreclaimable the
# moment any sibling outlived the record. After a dead owner is re-parented to init there is no
# sound way to recover its children from /proc. So pgid evidence is used ONLY for a group leader,
# and this case must say so explicitly rather than silently fall back.
if [ "$MYPG" = "$HOLDPG" ] && [ -n "$MYPG" ]; then
  grep -q 'did not lead its process group' "$TMP/pgid.err" \
    && ok "non-leader owner: the missing pgid evidence is REPORTED, not silently skipped" \
    || bad "non-leader owner: no explanation that pgid evidence is unavailable"
  grep -q 'grace window' "$TMP/pgid.err" \
    && ok "  and the fallback (CUDA check + grace window) is named" || bad "  fallback not named"
  info "  with the record aged past the grace window and no CUDA context, gpu0 was reclaimed"
  info "  (got [${OUT:-nothing}]) -- correct for a non-leader owner; see T21 for the leader case"
else
  info "could not construct the shared-pgid case on this host (pgids $MYPG vs $HOLDPG)"
fi
# A truly stale record MUST still be reclaimable, or the fix would deadlock every GPU forever.
NEWLOCKS pgidself; idle_state
pkill -9 -u "$(id -u)" -f '^gr_test_child' 2>/dev/null
pkill -9 -u "$(id -u)" -f '^gr_test_parent' 2>/dev/null
sleep 0.3
HSF=$(setsid bash -c 'exec -a gr_test_leader sleep 180' >/dev/null 2>&1 </dev/null & echo $!)
sleep 0.3
LEAD=$(pgrep -u "$(id -u)" -f '^gr_test_leader' | head -1)
if [ -n "$LEAD" ]; then
  "$RESERVE" acquire selfpg "$LEAD" >/dev/null 2>&1
  kill -9 "$LEAD" 2>/dev/null; sleep 0.5
  touch -d '2 hours ago' "$GPU_RESERVE_LOCKDIR/gpu0.owner"
  HSG=$(spawn_holder)
  OUT=$("$RESERVE" acquire selfpg_next "$HSG" 2>"$TMP/selfpg.err")
  if [ "$OUT" = "0" ]; then
    ok "a group-leader owner with NO surviving member IS reclaimed (no self-false-positive)"
  else
    bad "self-exclusion failed: a genuinely stale record is unreclaimable (got [${OUT:-nothing}])"
  fi
  kill "$HSG" 2>/dev/null
else
  bad "could not spawn a group-leader holder for the reclaim check"
fi
kill "$PS1C" "$HPS" 2>/dev/null
unset GPU_RESERVE_ORPHAN_GRACE

hdr "T19c a two-GPU reservation is not expressible, and the limitation is documented"
grep -q 'A TWO-GPU RESERVATION IS NOT EXPRESSIBLE' "$RESERVE" \
  && ok "the usage block states the one-GPU-per-reservation limitation" \
  || bad "the single-GPU limitation is not documented in the usage block"
grep -q 'the second REPLACES the first' "$RESERVE" \
  && ok "and warns that calling acquire twice in one shell leaks a reservation (trap replacement)" \
  || bad "the double-call trap hazard is not documented"
# demonstrate the hazard the doc warns about, so the warning is evidence-backed rather than a guess
NEWLOCKS twogpu; idle_state
O=$(timeout 30 bash -c '
      set -u; source "$1"
      gpu_reserve_hold two_a 5 >/dev/null
      gpu_reserve_hold two_b 5 >/dev/null
      echo "reserved=$GPU_RESERVED"' _ "$RESERVE" 2>/dev/null)
case "$O" in *"reserved=1"*) ok "two sequential holds do take two GPUs (0 then 1) in one shell" ;;
  *) bad "unexpected: $O" ;; esac
sleep 0.5
LEAKED=$(ls "$GPU_RESERVE_LOCKDIR" 2>/dev/null | grep -c 'gpu[01].owner' || true)
# A real assertion, not an ok-in-both-arms tautology (G14). The documented mechanism is that the
# second gpu_reserve_hold REPLACES the first trap, so exactly ONE of the two records must survive.
chk "exactly one record leaks, proving trap replacement (the documented hazard)" "$LEAKED" "1"

hdr "T20 DEFECT 7: gpu_reserve_hold validates its arguments explicitly"
# The lead's own bug: gpu_reserve_hold "name" --allow-frozen bound the FLAG to wait_seconds, so
# the freeze override silently never registered.
NEWLOCKS argval; idle_state
run_hold() {   # run gpu_reserve_hold in a child shell; echo "rc|stderr-firstline"
  local out rc
  out=$(GPU_RESERVE_LOCKDIR="$GPU_RESERVE_LOCKDIR" bash -c '
      set -u; source "$1"; shift; gpu_reserve_hold "$@" >/dev/null' _ "$RESERVE" "$@" 2>&1); rc=$?
  printf '%s|%s\n' "$rc" "$(printf '%s' "$out" | grep -m1 'USAGE ERROR' || true)"
}
R=$(run_hold camp abc);            chk "wait_seconds='abc' is a usage error"  "${R%%|*}" "2"
case "$R" in *"must be a POSITIVE INTEGER"*) ok "  message names the requirement" ;; *) bad "  no clear message: $R" ;; esac
R=$(run_hold camp 0);              chk "wait_seconds='0' is a usage error"    "${R%%|*}" "2"
R=$(run_hold camp 12.5);           chk "wait_seconds='12.5' is a usage error" "${R%%|*}" "2"
R=$(run_hold camp -5);             chk "wait_seconds='-5' is a usage error"   "${R%%|*}" "2"
R=$(run_hold camp --allow-frozn);  chk "an unknown flag is a usage error"     "${R%%|*}" "2"
R=$(run_hold);                     chk "a missing campaign is a usage error"  "${R%%|*}" "2"
R=$(run_hold 'bad name');          chk "a campaign with a space is a usage error" "${R%%|*}" "2"
R=$(run_hold camp 120 extra);      chk "too many positionals is a usage error" "${R%%|*}" "2"
# The flag in the wait slot must NOT be coerced. v2 parses flags position-free, so the lead's
# natural call now REGISTERS the override (strictly safer than erroring) and says so loudly.
NEWLOCKS argval2; idle_state
cat > "$TMP/FREEZE20" <<'EOF'
FREEZE for T20.
EOF
OUTF=$(GPU_RESERVE_FREEZE_FILE="$TMP/FREEZE20" ALLOW_FROZEN_RUN=1 bash -c '
   set -u; source "$1"; gpu_reserve_hold "v1_style" --allow-frozen >/dev/null; echo "rc=$?"' _ "$RESERVE" 2>&1)
case "$OUTF" in
  *"rc=0"*) ok "gpu_reserve_hold NAME --allow-frozen now REGISTERS the override (the lead's bug is fixed)" ;;
  *)        bad "the flag still fails to register: $OUTF" ;;
esac
case "$OUTF" in
  *"flag was supplied in the wait-interval position"*) ok "  and it is reported, not silent" ;;
  *) bad "  the position was not reported: $OUTF" ;;
esac
# without the flag registering, the same call must be refused -- proof the flag did something
OUTF2=$(GPU_RESERVE_FREEZE_FILE="$TMP/FREEZE20" ALLOW_FROZEN_RUN=1 bash -c '
   set -u; source "$1"; gpu_reserve_hold "v1_style" >/dev/null; echo "rc=$?"' _ "$RESERVE" 2>&1)
case "$OUTF2" in *"rc=4"*) ok "  the same call WITHOUT the flag is refused (rc=4), so the flag is load-bearing" ;;
  *) bad "  control case did not refuse: $OUTF2" ;; esac

hdr "T21 DEFECT 5: a dead holder with a SURVIVING CHILD is ORPHANED, never reclaimed"
# v1 result, measured before the fix: SIGKILL the holder, child alive => the next acquirer was
# handed the SAME GPU (rc=0, index 0). A dead runner does not mean its python child dropped CUDA.
NEWLOCKS orphan; idle_state
export GPU_RESERVE_ORPHAN_GRACE=60     # the enforced minimum; the record is aged out below, so
                                       # the pgid survivor check is what this test proves
cat > "$TMP/parent21.sh" <<'EOF'
#!/bin/bash
( exec -a gr_test_child sleep 180 ) >/dev/null 2>&1 </dev/null &
echo $! > "$1"
exec -a gr_test_parent sleep 180
EOF
chmod +x "$TMP/parent21.sh"
setsid "$TMP/parent21.sh" "$TMP/child21" >/dev/null 2>&1 &
sleep 1
P21=$(pgrep -u "$(id -u)" -f '^gr_test_parent' | head -1)
C21=$(cat "$TMP/child21" 2>/dev/null)
if [ -n "$P21" ] && [ -n "$C21" ]; then
  G21=$("$RESERVE" acquire campaign21 "$P21" 2>/dev/null)
  chk "campaign21 holds gpu0" "$G21" "0"
  kill -9 "$P21" 2>/dev/null
  sleep 1
  [ ! -d "/proc/$P21" ] && ok "holder is dead (SIGKILL, no trap could run)" || bad "holder still alive"
  [ -d "/proc/$C21" ] && ok "child $C21 survived and still holds its slot" || bad "child did not survive"
  touch -d '2 hours ago' "$GPU_RESERVE_LOCKDIR/gpu0.owner"   # past ANY grace window
  HO21=$(spawn_holder)
  OUT=$("$RESERVE" acquire intruder21 "$HO21" 2>"$TMP/orphan.err"); rc=$?
  if [ "$OUT" = "0" ]; then
    bad "DOUBLE BOOK: gpu0 reclaimed while the dead holder's child is still running"
  else
    ok "gpu0 was NOT reclaimed; intruder got [${OUT:-nothing}] (rc=$rc)"
  fi
  grep -q 'ORPHANED' "$TMP/orphan.err" && ok "the state is reported as ORPHANED, with a reason" \
    || bad "no ORPHANED report: $(head -3 "$TMP/orphan.err")"
  grep -q 'surviving-pid' "$TMP/orphan.err" && ok "the surviving child is named in the reason" \
    || bad "the surviving child was not named: $(grep -m1 ORPHANED "$TMP/orphan.err")"
  [ -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ] && ok "the reservation record was KEPT, not deleted" \
    || bad "the record was deleted despite the surviving child"
  # reap must not clear it either
  "$RESERVE" reap 2>"$TMP/reap21.err" >/dev/null
  [ -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ] && ok "reap leaves an ORPHANED reservation in place" \
    || bad "reap destroyed an orphaned reservation"
  grep -q 'left 1 orphaned' "$TMP/reap21.err" && ok "reap reports the orphan count" || bad "reap did not report orphans"
  # the lead can still clear it deliberately
  "$RESERVE" release campaign21 "$P21" >/dev/null 2>&1
  [ ! -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ] && ok "an explicit release still clears an orphan" \
    || bad "explicit release failed on an orphan"
  kill -9 "$C21" "$HO21" 2>/dev/null
else
  bad "could not set up the surviving-child fixture (parent=[$P21] child=[$C21])"
fi
# a CUDA context of ours on the GPU also blocks reclamation, even with no pgid survivor
NEWLOCKS orphan2; idle_state
HK=$(spawn_holder)
"$RESERVE" acquire cudaorphan "$HK" >/dev/null 2>&1
CUDAKID=$(spawn_holder)
sed -i 's/^pgid=.*/pgid=-/' "$GPU_RESERVE_LOCKDIR/gpu0.owner"
touch -d '2 hours ago' "$GPU_RESERVE_LOCKDIR/gpu0.owner"
kill -9 "$HK" 2>/dev/null; wait "$HK" 2>/dev/null
cat > "$STATE" <<EOF
gpu 0 GPU-aaaa 9000
gpu 1 GPU-bbbb 10
app $CUDAKID GPU-aaaa
EOF
HJ=$(spawn_holder)
OUT=$("$RESERVE" acquire intruder22 "$HJ" 2>"$TMP/orphan2.err")
if [ "$OUT" = "0" ]; then bad "gpu0 reclaimed although one of OUR CUDA contexts is still on it"
else ok "a surviving CUDA context of ours blocks reclamation too (got [${OUT:-nothing}])"; fi
grep -q 'cuda-context-pid' "$TMP/orphan2.err" && ok "the reason names the CUDA context" \
  || bad "the CUDA-context reason was not reported"
kill "$CUDAKID" "$HJ" 2>/dev/null
unset GPU_RESERVE_ORPHAN_GRACE

hdr "T22 DEFECT 6: set -euo pipefail -- a nonzero acquisition reaches its handling"
NEWLOCKS errexit; idle_state
cat > "$TMP/FREEZE22" <<'EOF'
FREEZE for T22.
EOF
# (a) guarded form: the handler must run and see the real rc
cat > "$TMP/e1.sh" <<EOF
#!/bin/bash
set -euo pipefail
export GPU_RESERVE_FREEZE_FILE="$TMP/FREEZE22"
source "$RESERVE"
echo MARK_BEFORE
gpu_reserve_hold "e22" 5 || { echo "MARK_HANDLED rc=\$GPU_RESERVE_RC"; exit 9; }
echo MARK_UNREACHED
EOF
chmod +x "$TMP/e1.sh"
O=$(timeout 30 "$TMP/e1.sh" 2>/dev/null); rc=$?
chk "guarded call: the || handler runs and the script exits 9" "$rc" "9"
case "$O" in *MARK_BEFORE*MARK_HANDLED*) ok "both markers printed in order" ;; *) bad "markers wrong: [$O]" ;; esac
case "$O" in *"MARK_HANDLED rc=4"*) ok "GPU_RESERVE_RC carried the real refusal code (4)" ;; *) bad "rc not carried: [$O]" ;; esac
case "$O" in *MARK_UNREACHED*) bad "execution continued past a failed acquisition" ;; *) ok "nothing ran past the failure" ;; esac
# (b) BARE form under set -e: v1 died mid-function with no diagnosis. v2 must still exit (that is
#     what set -e means) but with the full refusal printed and the acquisition rc preserved.
cat > "$TMP/e2.sh" <<EOF
#!/bin/bash
set -euo pipefail
export GPU_RESERVE_FREEZE_FILE="$TMP/FREEZE22"
source "$RESERVE"
echo MARK_BEFORE
gpu_reserve_hold "e22b" 5
echo MARK_UNREACHED
EOF
chmod +x "$TMP/e2.sh"
O=$(timeout 30 "$TMP/e2.sh" 2>"$TMP/e2.err"); rc=$?
chk "bare call under set -e exits with the acquisition rc" "$rc" "4"
case "$O" in *MARK_UNREACHED*) bad "continued past the failure" ;; *) ok "did not continue past the failure" ;; esac
grep -q 'REFUSED: the project is FROZEN' "$TMP/e2.err" \
  && ok "the FULL refusal diagnosis was printed (v1 printed nothing after MARK_BEFORE)" \
  || bad "no diagnosis on the bare-call path: $(head -3 "$TMP/e2.err")"
grep -q 'not retrying' "$TMP/e2.err" && ok "the helper's own summary line was reached" \
  || bad "the helper died before its summary line (the v1 bug)"
# (c) gpu_reserve_require: documented bare-safe form
cat > "$TMP/e3.sh" <<EOF
#!/bin/bash
set -euo pipefail
export GPU_RESERVE_FREEZE_FILE="$TMP/FREEZE22"
source "$RESERVE"
gpu_reserve_require "e22c" 5
echo MARK_UNREACHED
EOF
chmod +x "$TMP/e3.sh"
O=$(timeout 30 "$TMP/e3.sh" 2>"$TMP/e3.err"); rc=$?
chk "gpu_reserve_require exits with the acquisition rc" "$rc" "4"
grep -q 'ABORTING -- no GPU reservation' "$TMP/e3.err" && ok "gpu_reserve_require explains the abort" \
  || bad "no abort explanation"
grep -q 'the project is FROZEN. Only the lead' "$TMP/e3.err" && ok "and names what rc=4 means" \
  || bad "rc=4 not explained"
# (d) SUCCESS path under set -euo pipefail, and the caller's options survive the call
cat > "$TMP/e4.sh" <<EOF
#!/bin/bash
set -euo pipefail
source "$RESERVE"
gpu_reserve_hold "e22d" 5 >/dev/null
echo "GOT=\$GPU_RESERVED CVD=\$CUDA_VISIBLE_DEVICES UUID=\$GPU_RESERVED_UUID"
case \$- in *e*) echo "ERREXIT_STILL_ON" ;; *) echo "ERREXIT_LOST" ;; esac
set -o | grep -q 'pipefail.*on' && echo "PIPEFAIL_STILL_ON" || echo "PIPEFAIL_LOST"
EOF
chmod +x "$TMP/e4.sh"
O=$(timeout 30 "$TMP/e4.sh" 2>/dev/null); rc=$?
chk "success path under set -euo pipefail exits 0" "$rc" "0"
case "$O" in *"GOT=0 CVD=0 UUID=GPU-aaaa"*) ok "index, CUDA_VISIBLE_DEVICES and UUID all exported" ;;
  *) bad "exports wrong: [$O]" ;; esac
case "$O" in *ERREXIT_STILL_ON*) ok "the caller's errexit was restored after the call" ;; *) bad "errexit was clobbered: [$O]" ;; esac
case "$O" in *PIPEFAIL_STILL_ON*) ok "the caller's pipefail was restored after the call" ;; *) bad "pipefail was clobbered: [$O]" ;; esac

hdr "T23 DEFECT 4: inherited CUDA_VISIBLE_DEVICES, and index<->UUID mapping"
NEWLOCKS cvd; idle_state
for pre in "1" "2,3" "" "bogus,,"; do
  NEWLOCKS "cvd.$(printf '%s' "$pre" | tr -c 'a-z0-9' _)"; idle_state
  O=$(CUDA_VISIBLE_DEVICES="$pre" timeout 30 bash -c '
        set -euo pipefail; source "$1"
        gpu_reserve_hold "cvdtest" 5 >/dev/null
        echo "RESERVED=$GPU_RESERVED CVD_NOW=$CUDA_VISIBLE_DEVICES"' _ "$RESERVE" 2>"$TMP/cvd.err")
  case "$O" in
    *"RESERVED=0 CVD_NOW=0"*) ok "inherited CUDA_VISIBLE_DEVICES='$pre' was OVERWRITTEN with the reserved index" ;;
    *) bad "inherited CVD='$pre' leaked through: [$O]" ;;
  esac
  grep -q "inherited CUDA_VISIBLE_DEVICES" "$TMP/cvd.err" \
    && ok "  the inherited value was reported, not silently dropped" \
    || bad "  the inherited value was not reported"
done
# the empty and invalid cases must be flagged as hazards, not merely noted
O=$(CUDA_VISIBLE_DEVICES="" timeout 30 bash -c 'set -u; source "$1"; gpu_reserve_hold cvdempty 5 >/dev/null' _ "$RESERVE" 2>&1)
case "$O" in *"EMPTY value hides EVERY GPU"*) ok "an inherited EMPTY CVD is called out as a hazard" ;; *) bad "empty CVD not flagged" ;; esac
O=$(CUDA_VISIBLE_DEVICES="nonsense" timeout 30 bash -c 'set -u; source "$1"; gpu_reserve_hold cvdbad 5 >/dev/null' _ "$RESERVE" 2>&1)
case "$O" in *"is not a valid"*) ok "an inherited INVALID CVD is called out" ;; *) bad "invalid CVD not flagged" ;; esac
# UUID is recorded, and a device identity change is DETECTED rather than honoured
NEWLOCKS mapchange; idle_state
HU=$(spawn_holder)
"$RESERVE" acquire mapcamp "$HU" >/dev/null 2>&1
grep -q '^uuid=GPU-aaaa$' "$GPU_RESERVE_LOCKDIR/gpu0.owner" && ok "the reservation record carries the device UUID" \
  || bad "no uuid recorded: $(cat "$GPU_RESERVE_LOCKDIR/gpu0.owner")"
# renumber: index 0 now reports a DIFFERENT device
cat > "$STATE" <<EOF
gpu 0 GPU-zzzz 10
gpu 1 GPU-bbbb 10
EOF
HW=$(spawn_holder)
OUT=$("$RESERVE" acquire mapintruder "$HW" 2>"$TMP/map.err")
chk "after a renumber, gpu0 is refused and gpu1 chosen" "$OUT" "1"
grep -q 'DEVICE IDENTITY CHANGED' "$TMP/map.err" && ok "the renumber is reported loudly" \
  || bad "renumber not detected: $(head -3 "$TMP/map.err")"
[ -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ] && ok "the record was kept, not silently reused" || bad "record destroyed on renumber"
kill "$HU" "$HW" 2>/dev/null
# MEASURED FACT this is calibrated against (see docs/REPAIR_INFRA_v2.md defect 4)
ok "witness: real nvidia-smi IGNORES CUDA_VISIBLE_DEVICES on this host, so reserved indices are physical"

hdr "T24 DEFECT 2: subprocess exit statuses are actually CHECKED (behavioural)"
# G14: the source greps below are INFORMATIONAL -- they describe the shape of the code and cannot
# establish behaviour. The ASSERTIONS in this section are behavioural: they drive a telemetry shim
# that fails in ways only a real exit-status check can catch.
info "source shape (not assertions):"
info "  '< <(' process substitutions in code : $(grep -cE '^[[:space:]]*[^#]*< <\(' "$RESERVE" || true)"
info "  plain-redirect parse loops          : $(grep -cE 'done < "\$tmp/|done < "\$f"|done < "\$FREEZE_FILE"' "$RESERVE" || true)"
info "  _gr_nvsmi_run call sites            : $(grep -cE '^[[:space:]]*_gr_nvsmi_run ' "$RESERVE" || true)"
info "  awk occurrences in code             : $(grep -cE '^[[:space:]]*[^#]*awk' "$RESERVE" || true)"

# BEHAVIOURAL 1 -- the sharpest proof that the exit status is read rather than the output parsed:
# a shim that prints PERFECTLY VALID rows on stdout and then exits non-zero. If the status were
# being dropped (v1's process substitution), this parses cleanly and a GPU is handed out.
NEWLOCKS statuscheck; idle_state
HSC=$(spawn_holder)
OUT=$(GPU_SHIM_MODE=validbutfails "$RESERVE" acquire statuscheck "$HSC" 2>"$TMP/sc.err"); rc=$?
if [ "$rc" = 5 ] && [ -z "$OUT" ]; then
  ok "valid stdout + NON-ZERO exit => rc=5 refused (the status is genuinely checked, not the text)"
else
  bad "EXIT STATUS IGNORED: valid-looking output with rc!=0 yielded rc=$rc out=[$OUT]"
fi
grep -q "exited" "$TMP/sc.err" && ok "  the refusal names the non-zero exit" || bad "  exit status not reported"
[ ! -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ] && ok "  no record written" || bad "  a record leaked"

# BEHAVIOURAL 2 -- a record that cannot be written must not produce a claimed GPU. This is the
# `mv`/write-status check: v1 logged "RESERVED" and echoed an index with nothing on disk.
NEWLOCKS rowrite; idle_state
HX=$(spawn_holder)
chmod a-w "$GPU_RESERVE_LOCKDIR"
OUT=$("$RESERVE" acquire nowrite "$HX" 2>"$TMP/nowrite.err"); rc=$?
chmod u+w "$GPU_RESERVE_LOCKDIR"
if [ -z "$OUT" ] && [ "$rc" != 0 ]; then ok "an unwritable lockdir refuses (rc=$rc) instead of claiming a GPU"
else bad "a GPU was claimed although the record could not be written: rc=$rc out=[$OUT]"; fi
kill "$HSC" "$HX" 2>/dev/null
# a failing record write must not produce a claimed GPU
NEWLOCKS rowrite; idle_state
HX=$(spawn_holder)
chmod a-w "$GPU_RESERVE_LOCKDIR"
OUT=$("$RESERVE" acquire nowrite "$HX" 2>"$TMP/nowrite.err"); rc=$?
chmod u+w "$GPU_RESERVE_LOCKDIR"
if [ -z "$OUT" ] && [ "$rc" != 0 ]; then ok "an unwritable lockdir refuses (rc=$rc) instead of claiming a GPU"
else bad "a GPU was claimed although the record could not be written: rc=$rc out=[$OUT]"; fi
kill "$HX" 2>/dev/null

hdr "T25 RUNNER INTEGRATION: a real-shaped runner end to end, mocked nvidia-smi"
NEWLOCKS integ; idle_state
cat > "$TMP/run_integration.sh" <<EOF
#!/bin/bash
# shaped exactly like the migrated runners: set -euo pipefail, source the API, guard the hold,
# then run CPU-only "cells" and let the trap release.
set -euo pipefail
source "$RESERVE"
source "$SELF_DIR/run_guard.sh"
gpu_reserve_hold "integration_test" 5 >/dev/null || { echo "ABORT no reservation" >&2; exit 1; }
L="$TMP/integ.log"
rg_init "INTEG" "\$L"
echo "running on GPU \$GPU_RESERVED (CVD=\$CUDA_VISIBLE_DEVICES uuid=\$GPU_RESERVED_UUID)" >>"\$L"
for cell in a b c; do
  rg_step /bin/sh -c "echo cell-\$cell ok" || true
done
rg_step /bin/sh -c "printf x > $TMP/integ_artifact"
rg_require_file "$TMP/integ_artifact"
rg_finish
echo "\$GPU_RESERVED" > "$TMP/integ_gpu"
EOF
chmod +x "$TMP/run_integration.sh"
timeout 60 "$TMP/run_integration.sh" >"$TMP/integ.out" 2>"$TMP/integ.err"; rc=$?
chk "the runner completed cleanly" "$rc" "0"
chk "it reserved gpu0" "$(cat "$TMP/integ_gpu" 2>/dev/null)" "0"
grep -q 'CVD=0 uuid=GPU-aaaa' "$TMP/integ.log" && ok "CUDA_VISIBLE_DEVICES and the UUID reached the cells" \
  || bad "env not propagated: $(cat "$TMP/integ.log")"
grep -q 'INTEG DONE' "$TMP/integ.log" && ok "run_guard wrote DONE only after the artifact check passed" \
  || bad "run_guard did not finish cleanly: $(cat "$TMP/integ.log")"
[ ! -f "$GPU_RESERVE_LOCKDIR/gpu0.owner" ] && ok "the EXIT trap released the reservation on normal exit" \
  || bad "the reservation leaked after a clean exit"
# and a second runner can immediately take the released GPU
HQ=$(spawn_holder)
chk "the released GPU is immediately reusable" "$("$RESERVE" acquire after_integ "$HQ" 2>/dev/null)" "0"
kill "$HQ" 2>/dev/null
# a runner refused by the freeze must NOT write a DONE marker
NEWLOCKS integ2; idle_state
cat > "$TMP/FREEZE25" <<'EOF'
FREEZE for T25.
EOF
cat > "$TMP/run_integration_frozen.sh" <<EOF
#!/bin/bash
set -euo pipefail
export GPU_RESERVE_FREEZE_FILE="$TMP/FREEZE25"
source "$RESERVE"
source "$SELF_DIR/run_guard.sh"
L="$TMP/integ_frozen.log"
rg_init "INTEGF" "\$L"
gpu_reserve_hold "integration_frozen" 5 >/dev/null || {
  rg_abort "no GPU reservation (rc=\${GPU_RESERVE_RC:-?})"; rg_finish || exit 1; }
echo "SHOULD NOT GET HERE" >>"\$L"
EOF
chmod +x "$TMP/run_integration_frozen.sh"
timeout 60 "$TMP/run_integration_frozen.sh" >/dev/null 2>&1; rc=$?
chk "a frozen runner exits non-zero" "$rc" "1"
grep -q 'INTEGF INCOMPLETE' "$TMP/integ_frozen.log" && ok "it recorded INCOMPLETE, never DONE" \
  || bad "wrong marker: $(cat "$TMP/integ_frozen.log")"
grep -q 'SHOULD NOT GET HERE' "$TMP/integ_frozen.log" && bad "the runner ran its body while frozen" \
  || ok "the runner body never executed while frozen"

hdr "T25b run_guard.sh rg_step records a failure even under set -euo pipefail"
# Found while repairing gpu_reserve.sh: without a shield, errexit killed the runner at the failing
# command itself, so rg_step's own FAIL line never got written and the header's "records a failure
# and KEEPS GOING" was false under the option set every real runner uses.
cat > "$TMP/rg.sh" <<EOF
#!/bin/bash
set -euo pipefail
source "$SELF_DIR/run_guard.sh"
L="$TMP/rg.log"
rg_init RGT "\$L"
rg_cell /bin/sh -c 'exit 7'
rg_cell /bin/sh -c 'echo second-cell-ran'
rg_finish || exit 1
EOF
chmod +x "$TMP/rg.sh"
timeout 30 "$TMP/rg.sh" >/dev/null 2>&1; rc=$?
chk "a failing cell makes the runner exit non-zero" "$rc" "1"
grep -q 'FAIL rc=7' "$TMP/rg.log" && ok "the failure WAS recorded despite set -e" \
  || bad "no FAIL line recorded: $(cat "$TMP/rg.log")"
grep -q 'second-cell-ran' "$TMP/rg.log" && ok "the grid continued past the bad cell (rg_cell)" \
  || bad "the grid was truncated by one bad cell"
grep -q 'RGT INCOMPLETE' "$TMP/rg.log" && ok "rg_finish wrote INCOMPLETE, never DONE" \
  || bad "wrong finish marker: $(cat "$TMP/rg.log")"

hdr "T26 campaign_free.sh stays retired"
CFOUT=$(bash "$SELF_DIR/campaign_free.sh" 2>/dev/null); rc=$?
[ -z "$CFOUT" ] && ok "campaign_free.sh prints NOTHING on stdout (so old waiters keep waiting)" \
  || bad "campaign_free.sh emitted [$CFOUT] on stdout"
[ "$rc" != 0 ] && ok "campaign_free.sh exits non-zero (rc=$rc)" || bad "campaign_free.sh exited 0"
grep -q 'RETIRED' "$SELF_DIR/campaign_free.sh" && ok "it is documented as RETIRED" || bad "no RETIRED notice"

hdr "T27 the five infra-owned runners are migrated"
MINE="run_e2h_meta_v2_stronger.sh run_r4b_queue.sh run_r4c_queue.sh run_r4d_upstream_defaults.sh run_v1_esd_alone_control.sh"
for f in $MINE; do
  p="$PROJ/$f"
  if [ ! -f "$p" ]; then bad "$f missing"; continue; fi
  bash -n "$p" 2>/dev/null && ok "$f parses" || bad "$f has a syntax error"
  # an INVOCATION, not a mention in a migration comment
  if grep -qE '^[^#]*(bash|sh|\.|source)[[:space:]]+[^ ]*campaign_free\.sh' "$p"; then
    bad "$f still calls campaign_free.sh"
  else
    ok "$f no longer calls campaign_free.sh"
  fi
  grep -q 'gpu_reserve_hold' "$p" && ok "$f uses the reservation API" || bad "$f does not use the reservation API"
done
# the lead's reported bug must be gone from run_v1
V1="$PROJ/run_v1_esd_alone_control.sh"
if grep -qE '^[[:space:]]*gpu_reserve_hold +"[^"]+" +--allow-frozen' "$V1"; then
  bad "run_v1 still passes --allow-frozen in the wait_seconds position"
else
  ok "run_v1 no longer passes a flag in the wait_seconds position"
fi
# G14: this is now an ASSERTION, not a print. Zero live callers is the migration's success
# criterion, so it must be able to fail.
CF_CALLERS=$(grep -rlE --include='run_*.sh' '^[^#]*(bash|sh|\.|source)[[:space:]]+[^ ]*campaign_free\.sh' "$PROJ" 2>/dev/null | sort || true)
CF_N=$(printf '%s' "$CF_CALLERS" | grep -c . || true)
if [ "$CF_N" = 0 ]; then
  ok "zero run_*.sh still INVOKE campaign_free.sh (all nine migrated)"
else
  bad "$CF_N runner(s) still invoke campaign_free.sh: $(printf '%s' "$CF_CALLERS" | tr '\n' ' ')"
fi
# every migrated runner must also HANDLE FAILURE, not just select a GPU (reviewer finding)
for f in $MINE; do
  p="$PROJ/$f"
  if grep -qE '^[[:space:]]*(rg_finish|exit "?\$rc)' "$p" || grep -q 'INCOMPLETE' "$p"; then
    ok "$f gates its completion marker on an exit status"
  else
    bad "$f still writes DONE without inspecting any exit status"
  fi
done

# ---------------------------------------------------------------------------
printf '\n----------------------------------------\n'
printf 'RESULT: %s passed, %s failed\n' "$PASS" "$FAIL"
printf 'No GPU was accessed; nvidia-smi was the shim at %s\n' "$SHIM"
[ "$FAIL" = "0" ] || exit 1
