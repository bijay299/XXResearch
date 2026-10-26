#!/bin/bash
# scripts/test_gpu_reserve.sh -- CPU-ONLY test harness for scripts/gpu_reserve.sh.
#
# Touches NO GPU and imports no CUDA. nvidia-smi is replaced by a shim that reads a plain-text
# state file, so every GPU topology below (including "another user is on GPU 1") is simulated.
# Safe to run on a frozen project while GPU jobs are in flight.
#
#   bash scripts/test_gpu_reserve.sh
#
# Tests:
#   T1  baseline: both allowed GPUs idle -> acquire returns 0
#   T2  second acquire takes the OTHER GPU, never the same one
#   T3  exhaustion: third acquire fails (exit 3) rather than double-booking
#   T4  CONCURRENCY: N processes race; exactly one winner per GPU, no index handed out twice
#   T5  release: a released GPU becomes available again
#   T6  stale reclamation: owner killed (simulated crash) -> next acquirer reclaims
#   T7  PID reuse: a reservation whose pid is alive but is a DIFFERENT process is stale
#   T8  another user's GPU is never selected
#   T9  off-limits GPUs 2-3 are never selected even when totally idle
#   T10 a pre-lock-scheme CUDA context of ours blocks that GPU (no reservation present)
#   T11 inter-cell memory dip does NOT release a held reservation (the campaign_free.sh (c) bug)
#   T12 trap-based release on crash via the sourceable helper
#   T13 regression witness: campaign_free.sh's /proc/environ clause is dead code

set -u
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESERVE="$SELF_DIR/gpu_reserve.sh"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/gpu_reserve_test.XXXXXX")
# cleanup trap is installed below, once spawn_holder/cleanup are defined

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  \033[32mPASS\033[0m %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
hdr()  { printf '\n== %s\n' "$*"; }
chk()  { if [ "$2" = "$3" ]; then ok "$1 (= $3)"; else bad "$1: expected [$3] got [$2]"; fi; }

# ---------------------------------------------------------------------------
# fake nvidia-smi. State file format, one directive per line:
#   gpu <index> <uuid> <memory_used_MiB>
#   app <pid> <uuid>
# ---------------------------------------------------------------------------
SHIM="$TMP/nvidia-smi"
STATE="$TMP/gpu_state"
cat > "$SHIM" <<'SHIM_EOF'
#!/bin/bash
# fake nvidia-smi: answers the three query forms gpu_reserve.sh uses, from $GPU_STATE_FILE
S="${GPU_STATE_FILE:?}"
q=""
for a in "$@"; do case "$a" in --query-gpu=*|--query-compute-apps=*) q="$a";; esac; done
case "$q" in
  --query-gpu=index,uuid)
      awk '$1=="gpu"{printf "%s, %s\n",$2,$3}' "$S" ;;
  --query-gpu=index,memory.used)
      awk '$1=="gpu"{printf "%s, %s\n",$2,$4}' "$S" ;;
  --query-compute-apps=pid,gpu_uuid)
      awk '$1=="app"{printf "%s, %s\n",$2,$3}' "$S" ;;
  *) exit 1 ;;
esac
SHIM_EOF
chmod +x "$SHIM"

export GPU_STATE_FILE="$STATE"
export GPU_RESERVE_NVSMI="$SHIM"
export GPU_RESERVE_GPUS="0 1"
export GPU_RESERVE_WAIT=20
export GPU_RESERVE_FOREIGN_MIB=512
# Point the freeze marker at a path that does not exist, so T1-T13 exercise the lock itself and
# are not short-circuited by the project's real FREEZE_ACTIVE. T14-T16 drive it deliberately.
export GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE"
export GPU_RESERVE_OVERRIDE_LOG="$TMP/frozen_override.log"

NEWLOCKS() { export GPU_RESERVE_LOCKDIR="$TMP/locks.$1"; rm -rf "$GPU_RESERVE_LOCKDIR"; mkdir -p "$GPU_RESERVE_LOCKDIR"; }

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
# inherits the command-substitution pipe keeps it open, so the substitution would block until the
# placeholder exits.
spawn_holder() {
  ( exec -a gr_test_holder sleep 120 ) >/dev/null 2>&1 </dev/null &
  echo $!
}
# placeholders are reparented grandchildren, so reap them by name, not by pid list
cleanup() { rm -rf "$TMP"; pkill -9 -u "$(id -u)" -f '^gr_test_holder' 2>/dev/null; return 0; }
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
kill "$H1" "$H2" "$H3" 2>/dev/null

# ---------------------------------------------------------------------------
hdr "T4  CONCURRENCY: 12 processes race for 2 GPUs"
NEWLOCKS race; idle_state
RES="$TMP/race_results"; : > "$RES"
N=12
for k in $(seq 1 $N); do
  (
    # Each racer is its own long-lived process and reserves on its own behalf.
    # MUST capture the pid BEFORE the command substitution: $BASHPID is re-evaluated inside
    # $( ), so passing it inline would name the transient substitution subshell instead.
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
hdr "T6  stale-lock reclamation after a simulated runner crash"
NEWLOCKS stale; idle_state
HC=$(spawn_holder)
GC=$("$RESERVE" acquire crashy "$HC" 2>/dev/null); chk "crashy reserved" "$GC" "0"
HO=$(spawn_holder)
GX=$("$RESERVE" acquire other "$HO" 2>/dev/null); chk "while owner lives, gpu0 is respected" "$GX" "1"
kill -9 "$HC" 2>/dev/null; wait "$HC" 2>/dev/null
"$RESERVE" release other "$HO" >/dev/null 2>&1    # free gpu1 so only gpu0 can be returned
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
HR=$(spawn_holder)
"$RESERVE" release '' "$HP" >/dev/null 2>&1 || true
OUT=$("$RESERVE" acquire newcomer "$HR" 2>"$TMP/reuse.err")
if grep -q 'RECLAIMED stale' "$TMP/reuse.err"; then
  ok "alive pid + mismatched start time => correctly judged stale (PID reuse defeated)"
else bad "PID-reuse guard did not fire: $(cat "$TMP/reuse.err")"; fi
kill "$HP" "$LIVE" "$HR" 2>/dev/null

# ---------------------------------------------------------------------------
hdr "T8  a GPU occupied by ANOTHER USER is never selected"
NEWLOCKS foreign
# pid 2 is init-adjacent / not ours; use a pid that exists but is uid 0, plus a
# nonexistent pid, to cover both "wrong uid" and "invisible under hidepid".
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
# now make BOTH ours-allowed GPUs foreign -> must refuse entirely, not steal
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
# and 'adopt' can retroactively give that legacy job a proper reservation
"$RESERVE" release newscheme "$HM" >/dev/null 2>&1
AD=$("$RESERVE" adopt 0 legacy_adopted "$LEG" 2>/dev/null); chk "adopt gpu0 for the legacy pid" "$AD" "0"
kill "$LEG" "$HM" 2>/dev/null

hdr "T11 inter-cell memory DIP does not release a held reservation"
NEWLOCKS dip; idle_state
HD=$(spawn_holder)
GD=$("$RESERVE" acquire long_campaign "$HD" 2>/dev/null); chk "campaign holds gpu0" "$GD" "0"
# simulate the gap between campaign cells: context torn down, memory back to ~0
cat > "$STATE" <<EOF
gpu 0 GPU-aaaa 0
gpu 1 GPU-bbbb 0
EOF
HE=$(spawn_holder)
OUT=$("$RESERVE" acquire intruder "$HE" 2>/dev/null)
chk "gpu0 still reserved despite 0 MiB used; intruder sent to gpu1" "$OUT" "1"
if [ "$OUT" != "0" ]; then ok "memory dip no longer leaks the GPU (campaign_free.sh defect (c), fixed)"; fi
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
  # even if the trap missed it, the record must at least be judged stale
  HS=$(spawn_holder)
  OUT=$("$RESERVE" acquire post_kill "$HS" 2>"$TMP/trap2.err")
  if [ "$OUT" = "0" ]; then ok "trap missed but stale-reclaim recovered gpu0 (belt and braces)"
  else bad "gpu0 leaked after the holder died: $(cat "$TMP/trap2.err")"; fi
  kill "$HS" 2>/dev/null
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
  for fld in "pid=$HZ" "campaign=lead_verify" "gpu=0" "override="; do
    case "$LINE" in *"$fld"*) ok "log records $fld" ;; *) bad "log missing $fld" ;; esac
  done
  case "$LINE" in 2026-*|20[0-9][0-9]-*) ok "log line is timestamped" ;; *) bad "log line not timestamped" ;; esac
else
  bad "no override log written at $GPU_RESERVE_OVERRIDE_LOG"
fi
# and the freeze must still bite for everyone who does not pass both
HY=$(spawn_holder)
OUT=$("$RESERVE" acquire sneaky "$HY" 2>/dev/null); rc=$?
chk "a non-override caller is still refused after an override was granted" "$rc" "4"
# lifting the marker restores normal operation
rm -f "$MARKER"
"$RESERVE" release lead_verify "$HZ" >/dev/null 2>&1
OUT=$("$RESERVE" acquire after_thaw "$HY" 2>/dev/null)
chk "marker removed => normal acquire works again" "$OUT" "0"
kill "$HZ" "$HY" 2>/dev/null
export GPU_RESERVE_FREEZE_FILE="$TMP/NO_SUCH_FREEZE"

hdr "T17 D-013 witness: nothing in the project reads FREEZE_ACTIVE except gpu_reserve.sh"
PROJ=$(cd "$SELF_DIR/.." && pwd)
READERS=$(grep -rl --include='*.sh' --include='*.py' 'FREEZE_ACTIVE' "$PROJ" 2>/dev/null \
          | grep -v '/upstream/' | grep -v '/venv_' | sed "s|$PROJ/||" | sort | tr '\n' ' ')
printf '  files referencing FREEZE_ACTIVE: %s\n' "${READERS:-<none>}"
case "$READERS" in
  *gpu_reserve.sh*) ok "gpu_reserve.sh is now a reader of the freeze marker" ;;
  *) bad "gpu_reserve.sh does not reference FREEZE_ACTIVE" ;;
esac

# ---------------------------------------------------------------------------
printf '\n----------------------------------------\n'
printf 'RESULT: %s passed, %s failed\n' "$PASS" "$FAIL"
printf 'No GPU was accessed; nvidia-smi was the shim at %s\n' "$SHIM"
[ "$FAIL" = "0" ] || exit 1
