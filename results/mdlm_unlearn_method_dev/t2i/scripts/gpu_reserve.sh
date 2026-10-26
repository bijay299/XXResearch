#!/bin/bash
# scripts/gpu_reserve.sh -- EXCLUSIVE per-GPU reservation held for an ENTIRE campaign.
#
# v2 (2026-10-26). v1's lock algebra was correct and is unchanged in substance; v2 repairs seven
# defects that all lived OUTSIDE what v1's 50-test suite exercised. See docs/REPAIR_INFRA_v2.md.
#
# WHY THIS EXISTS
# ---------------
# scripts/campaign_free.sh was check-then-select: it READ GPU state and RETURNED an index while
# holding no lock. Two runners that read before either wrote both got the same index. Observed
# live on 2026-10-26: run_r2_meta_official.sh (12:39:35), run_r1c_eval.sh (12:39:29) and
# run_r4d_upstream_defaults.sh (12:39:32) each independently ran campaign_free.sh, each was told
# "0", and all three ran on GPU 0 (3 pids, 15.3 GB) while GPU 1 sat idle at 34 MiB.
# campaign_free.sh is RETIRED and must stay retired; it returns nothing on stdout by design.
#
# THE FIX: an explicit reservation, not an inference from observed state. A reservation is a
# record on disk, created inside a flock-protected critical section, naming the owner process.
#
# ===========================================================================================
# PUBLIC API CONTRACT  (stable; the Baselines agent migrates run_r1*.sh / run_r2_meta_official.sh
#                       against exactly this)
# ===========================================================================================
#
# FORM A -- sourced helper (PREFERRED). The runner holds the reservation for its own lifetime and
# releases it from an EXIT/INT/TERM/HUP trap. The helper also EXPORTS CUDA_VISIBLE_DEVICES itself,
# so a runner that forgets to export cannot inherit a stale value (defect 4).
#
#     source scripts/gpu_reserve.sh
#     gpu_reserve_hold "r1_imma_official" || { echo "ABORT: no reservation" >&2; exit 1; }
#     # on success: $GPU_RESERVED (nvidia-smi/NVML index), $GPU_RESERVED_UUID,
#     #             $CUDA_VISIBLE_DEVICES already exported for you.
#
#   Signature:  gpu_reserve_hold <campaign> [wait_seconds] [--allow-frozen]
#     <campaign>      required; [A-Za-z0-9._-]+
#     [wait_seconds]  optional; a POSITIVE INTEGER, the retry interval while every allowed GPU is
#                     busy. Default 120. A non-integer here (notably a FLAG -- the bug that bit the
#                     lead in run_v1_esd_alone_control.sh) is a hard USAGE ERROR, rc=2, never
#                     silently coerced.
#     --allow-frozen  optional, position-free. Needed, TOGETHER with ALLOW_FROZEN_RUN=1 in the
#                     environment, to acquire while FREEZE_ACTIVE exists.
#
#   ERREXIT CONTRACT (defect 6). Real runners use `set -euo pipefail`. gpu_reserve_hold shields
#   its own body from the caller's errexit/pipefail, so a refusal always produces a full diagnosis
#   and a meaningful return code instead of killing the shell halfway through the function.
#   But a BARE `gpu_reserve_hold foo` under `set -e` still terminates the caller when it returns
#   non-zero -- that is what set -e means. Therefore ALWAYS use one of:
#       gpu_reserve_hold "camp" || { <your abort path>; exit 1; }     # handle it yourself
#       gpu_reserve_require "camp"                                    # abort for me, loudly
#   $GPU_RESERVE_RC always carries the last acquisition rc, even on the errexit path.
#
# FORM B -- subshell call style, for a runner keeping the old `G=$(...)` shape. The owner pid MUST
# be passed explicitly, because inside $( ) this script's parent is a transient subshell:
#     while :; do G=$(bash scripts/gpu_reserve.sh acquire my_campaign $$) && break
#                 rc=$?; [ "$rc" = 3 ] || exit "$rc"; sleep 120; done
#     trap 'bash scripts/gpu_reserve.sh release my_campaign $$' EXIT INT TERM HUP
#     export CUDA_VISIBLE_DEVICES=$G
#   Note `$$`, NOT `$BASHPID`: $BASHPID is re-evaluated inside $( ) and would name the transient
#   substitution subshell, whose immediate exit makes the reservation look stale at once.
#   Note also that only rc=3 is worth sleeping on; see the exit codes below.
#
# SUBCOMMANDS
#     acquire <campaign> [owner_pid] [--allow-frozen]   print index, exit 0
#     release <campaign> [owner_pid]                    drop matching reservations
#     status                                            table of all allowed GPUs
#     adopt <index> <campaign> <pid>                    reserve for an ALREADY-running job
#     reap                                              reclaim provably-stale records only
#     selftest-telemetry                                print the telemetry verdict and exit
#
# EXIT CODES (stable -- callers switch on these)
#     0  reserved
#     2  usage / bad argument / dead owner pid
#     3  no allowed GPU available right now        <-- the ONLY code worth retrying
#     4  refused: project is FROZEN
#     5  refused: GPU TELEMETRY UNKNOWN (nvidia-smi failed/empty/malformed/timed out)
#     6  refused: prohibited GPU set requested (environment tried to widen the allowlist)
#     7  refused: the GPU allowlist itself is unusable -- GPU_RESERVE_GPUS set but EMPTY,
#        whitespace-only, non-numeric, or containing a duplicate. A CONFIGURATION error, kept
#        deliberately distinct from 3 so a waiter loop does not sleep on it forever.
#    10  cannot create/open the lock directory or mutex
#    11  could not take the reservation mutex within GPU_RESERVE_WAIT
#
# ONE GPU PER RESERVATION -- A TWO-GPU RESERVATION IS NOT EXPRESSIBLE.
#   `acquire` returns exactly ONE index and a record covers exactly one device. There is no
#   multi-GPU reservation (no DDP across 0 and 1), by design and by omission: nothing in this
#   project needs it. Do NOT emulate one by calling gpu_reserve_hold twice in one shell -- each
#   call installs its own EXIT/INT/TERM/HUP trap and the second REPLACES the first, so one
#   reservation would leak until it aged out of the orphan window. If multi-GPU is ever needed it
#   must be built deliberately (one record naming several indices, acquired and released
#   atomically under the same mutex), not looped.
#
# ===========================================================================================
# GUARANTEES
#   * Mutual exclusion: scan-and-claim runs under flock(2) on a mutex file.
#   * Telemetry is FAIL-CLOSED (defect 1). "I could not read the GPUs" is UNKNOWN, never "free".
#     Absent, empty, short, non-numeric, [N/A] or timed-out nvidia-smi output refuses acquisition
#     with rc=5. v1 read a total nvidia-smi failure as "0 MiB used, no processes" = FREE.
#   * Every subprocess status is checked (defect 2). nvidia-smi is run into a temp file with its
#     rc captured; no `< <(cmd)` process substitution and no pipeline hides a failure.
#   * The production allowlist {0,1} cannot be widened by the environment (defect 3). Honouring
#     GPU_RESERVE_GPUS requires test mode, and test mode is refused while the telemetry source is
#     a real nvidia-smi, so no env var can point production at GPUs 2-3.
#   * Index<->UUID mapping is recorded and re-validated (defect 4); an inherited
#     CUDA_VISIBLE_DEVICES is audited, reported, and overwritten rather than trusted.
#   * A dead owner with SURVIVING WORK is ORPHANED, not stale (defect 5): the reservation is kept
#     and the GPU refused if any process of the owner's process group is alive, or any CUDA
#     context of our uid sits on that GPU, or the record is younger than the orphan grace window.
#   * Normal release: `release` on the happy path plus an EXIT/INT/TERM/HUP trap.
#   * Other users are never displaced: GPUs 2-3 are not in the allowed set, and an allowed GPU
#     carrying any compute process we cannot prove is ours is refused.
#   * D-013 FREEZE GATE: while FREEZE_ACTIVE exists, acquisition is refused (rc=4) unless BOTH
#     ALLOW_FROZEN_RUN=1 and --allow-frozen are given; every override is logged. In production the
#     marker PATH is hard-wired, so GPU_RESERVE_FREEZE_FILE cannot be used to fake a thaw.
#
# ENV (production ignores or refuses the ones marked TEST-ONLY)
#   GPU_RESERVE_LOCKDIR        reservation directory      (TEST-ONLY; production is hard-wired)
#   GPU_RESERVE_GPUS           TEST-ONLY allowed indices. Production is hard-wired to "0 1" and
#                              REFUSES (rc=6) any other value. SET-BUT-EMPTY, whitespace-only,
#                              non-numeric or duplicated is rc=7 in BOTH modes -- it is never
#                              treated as "unset" and never falls back to a default.
#   GPU_RESERVE_NVSMI          nvidia-smi command         (default "nvidia-smi")
#   GPU_RESERVE_NVSMI_TIMEOUT  seconds per nvidia-smi call (default 20; 0 disables the timeout)
#   GPU_RESERVE_FOREIGN_MIB    unattributable MiB above which an allowed GPU is refused (default 512)
#   GPU_RESERVE_UID            uid considered "ours"      (default: id -u)
#   GPU_RESERVE_WAIT           seconds to wait for the mutex (default 30)
#   GPU_RESERVE_ORPHAN_GRACE   seconds a record is protected after its owner dies (default 120)
#   GPU_RESERVE_FREEZE_FILE    TEST-ONLY freeze marker path; production path is hard-wired
#   GPU_RESERVE_CVD_STYLE      "index" (default) or "uuid" -- what CUDA_VISIBLE_DEVICES is set to
#   GPU_RESERVE_TEST_MODE=1    enable test mode; refused unless GPU_RESERVE_NVSMI is NOT a real
#                              nvidia-smi and GPU_RESERVE_LOCKDIR is outside the production dir.

set -u

# ---------------------------------------------------------------------------
# THE PROJECT ROOT IS A LITERAL. D-060, and it was the root of the whole class.
#
# This line used to read:
#     PROJECT="${PROJECT:-/data/bijaypandey/t2i_unlearn}"
# and the three "hard-wired production constants" below were derived FROM IT. So every one of the
# D-047 production refusals -- which compare a requested value against _GR_PROD_* -- was comparing
# the request against a constant the same environment had already moved. One variable defeated the
# lot. Measured before this fix, with the project FROZEN:
#     PROJECT=<scratch>/evil bash scripts/gpu_reserve.sh status
#       mode=production   lockdir=<scratch>/evil/.gpulocks
#       FREEZE: none (<scratch>/evil/FREEZE_ACTIVE absent)
#   and acquire then returned rc=3 (no GPU free) instead of rc=4 (frozen) -- the freeze gate was
#   passed outright, the mutex was relocated (reintroducing the D-001 triple-booking, because two
#   runners with different PROJECT values take different mutexes and cannot exclude each other),
#   and the freeze-override audit log was redirected so an override would go unrecorded.
# It also falsified this file's own claim, below, to be "a sweep of EVERY variable the file reads":
# the sweep covered the GPU_RESERVE_* knobs and missed the variable the knobs' baselines came from.
#
# The root is therefore a LITERAL with no environment fallback, and an attempt to set PROJECT is
# REFUSED in _gr_mode_init with the same non-retryable code (6) as the other production overrides,
# rather than ignored -- an ignored override is indistinguishable from an honoured one at the call
# site, which is how this survived. Nothing in the project sets PROJECT; `grep -rn '^\s*\(export
# \)\?PROJECT=' --include='*.sh'` matched only the old line itself.
# ---------------------------------------------------------------------------
_GR_PROD_ROOT="/data/bijaypandey/t2i_unlearn"

# ---------------------------------------------------------------------------
# HARD-WIRED PRODUCTION CONSTANTS. Not read from the environment, by design, and now anchored to
# _GR_PROD_ROOT (a literal) rather than to anything a caller can set.
# GPUs 2 and 3 belong to other users of this shared host and are OFF LIMITS.
# ---------------------------------------------------------------------------
_GR_PROD_GPUS="0 1"
_GR_PROD_LOCKDIR="$_GR_PROD_ROOT/.gpulocks"
_GR_PROD_FREEZE="$_GR_PROD_ROOT/FREEZE_ACTIVE"

_GR_PROD_NVSMI="nvidia-smi"
_GR_PROD_OVERRIDE_LOG="$_GR_PROD_ROOT/logs/frozen_override.log"

ALLOW_FROZEN_FLAG=0   # set only by an explicit --allow-frozen argument

_gr_log() { printf '[gpu_reserve] %s\n' "$*" >&2; }

_gr_now() { date -Is 2>/dev/null || date 2>/dev/null || printf 'unknown-time\n'; }

# ===========================================================================
# ENVIRONMENT SWEEP (reviewer finding, 2026-10-26).
#
# After the empty-allowlist fail-open was fixed, a reviewer found the SAME pattern in two more
# knobs, so this is now a sweep of EVERY variable the file reads rather than per-variable patches.
# Measured before the fix, with a GPU sitting at 30000 MiB and no attributable process:
#     GPU_RESERVE_FOREIGN_MIB=512       -> rc=3  (correct: refused)
#     GPU_RESERVE_FOREIGN_MIB=abc       -> rc=0, RESERVED gpu0   <-- tramples another user
#     GPU_RESERVE_FOREIGN_MIB=99999999  -> rc=0, RESERVED gpu0   <-- threshold above any real card
#     GPU_RESERVE_ORPHAN_GRACE=abc|0|-5 -> grace window silently deleted; orphan reclaimed
# `abc` reached `[ "$mem" -ge "$FOREIGN_MIB" ]`, where bash evaluates a non-numeric operand as 0,
# so "30000 >= 0" became false and the GPU read as free. Numeric-looking-but-absurd was just as
# bad: a threshold above total board memory can never trip.
#
# RULE APPLIED TO EVERY KNOB BELOW: a value that is present but unusable is a NON-RETRYABLE
# CONFIGURATION ERROR (rc=7). Never silently repaired, never defaulted over, never coerced to 0.
# Ranges are bounded on BOTH sides, because an out-of-range-high threshold disables the check it
# is supposed to perform, which is the fail-open in numeric clothing.
# ===========================================================================
_GR_ENV_RC=0

# _gr_need_int <varname> <present:"set"|""> <raw> <default> <min> <max> <what>
# echoes the accepted value; RETURNS 7 and echoes the default on refusal (an assignment inside
# $( ) would be lost in the subshell, so every call site appends `|| _GR_ENV_RC=7`).
# <present> MUST be passed as ${VAR+set}: inside a function "$3" is always set (possibly to ""),
# so set-ness cannot be recovered from the value alone. Getting this wrong made every UNSET knob
# report itself as "SET BUT EMPTY" -- caught immediately by re-running the blocking-item probes.
_gr_need_int() {
  local name="$1" present="$2" raw="$3" def="$4" min="$5" max="$6" what="$7"
  if [ -z "$present" ]; then printf '%s\n' "$def"; return 0; fi    # genuinely unset: use default
  if [ -z "$raw" ]; then
    _gr_log "REFUSED: $name is SET BUT EMPTY. $what must be an integer in [$min..$max];"
    _gr_log "         an empty value is a configuration error, not a request for the default."
    printf '%s\n' "$def"; return 7
  fi
  case "$raw" in
    ''|*[!0-9]*)
      _gr_log "REFUSED: $name='$raw' is not a non-negative integer. $what must be an integer in"
      _gr_log "         [$min..$max]. Refusing with rc=7 rather than coercing it: bash evaluates a"
      _gr_log "         non-numeric operand in [ -ge ] as 0, which silently DISABLES the check."
      printf '%s\n' "$def"; return 7 ;;
  esac
  if [ "$raw" -lt "$min" ] || [ "$raw" -gt "$max" ]; then
    _gr_log "REFUSED: $name='$raw' is outside [$min..$max]. $what."
    _gr_log "         Both bounds are enforced: an out-of-range-HIGH value disables the check it"
    _gr_log "         exists to perform, which is a fail-open in numeric clothing."
    printf '%s\n' "$def"; return 7
  fi
  printf '%s\n' "$raw"
}

_gr_env_init() {
  _GR_ENV_RC=0

  NVSMI_TIMEOUT=$(_gr_need_int GPU_RESERVE_NVSMI_TIMEOUT "${GPU_RESERVE_NVSMI_TIMEOUT+set}" "${GPU_RESERVE_NVSMI_TIMEOUT-}" 20 1 600 \
    "seconds allowed per nvidia-smi call (0 is refused: disabling the timeout re-opens the hang that makes telemetry unreadable)") || _GR_ENV_RC=7

  # Upper bound deliberately far below a 40 GB card (~40960 MiB): a threshold at or above board
  # capacity means "never refuse an unattributable GPU", i.e. the guard is off.
  FOREIGN_MIB=$(_gr_need_int GPU_RESERVE_FOREIGN_MIB "${GPU_RESERVE_FOREIGN_MIB+set}" "${GPU_RESERVE_FOREIGN_MIB-}" 512 1 4096 \
    "MiB of unattributable memory above which an allowed GPU is refused; our cards are 40 GB, so anything above 4096 would never trip") || _GR_ENV_RC=7

  MUTEX_WAIT=$(_gr_need_int GPU_RESERVE_WAIT "${GPU_RESERVE_WAIT+set}" "${GPU_RESERVE_WAIT-}" 30 1 3600 \
    "seconds to wait for the reservation mutex") || _GR_ENV_RC=7

  # Raised from 120s to 900s on the reviewer's point that a cold model load can exceed two minutes,
  # so a just-orphaned child may not be in nvidia-smi yet. The grace window is now a backstop, not
  # the primary protection -- that is the pgid survivor scan, fixed below.
  ORPHAN_GRACE=$(_gr_need_int GPU_RESERVE_ORPHAN_GRACE "${GPU_RESERVE_ORPHAN_GRACE+set}" "${GPU_RESERVE_ORPHAN_GRACE-}" 900 60 86400 \
    "seconds a reservation is protected after its owner dies; below 60 is refused because a cold model load alone can exceed it") || _GR_ENV_RC=7

  if [ -n "${GPU_RESERVE_UID+x}" ]; then
    case "${GPU_RESERVE_UID}" in
      ''|*[!0-9]*) _gr_log "REFUSED: GPU_RESERVE_UID='${GPU_RESERVE_UID}' is not a uid."; _GR_ENV_RC=7; MYUID=$(id -u) ;;
      *) MYUID="$GPU_RESERVE_UID" ;;
    esac
  else
    MYUID=$(id -u) || { _gr_log "REFUSED: cannot determine our own uid"; _GR_ENV_RC=7; MYUID=-1; }
  fi

  case "${GPU_RESERVE_CVD_STYLE-index}" in
    index|uuid) CVD_STYLE="${GPU_RESERVE_CVD_STYLE-index}" ;;
    *) _gr_log "REFUSED: GPU_RESERVE_CVD_STYLE='${GPU_RESERVE_CVD_STYLE-}' must be 'index' or 'uuid'."
       _GR_ENV_RC=7; CVD_STYLE=index ;;
  esac

  case "${GPU_RESERVE_TEST_MODE-0}" in
    0|1) ;;
    *) _gr_log "REFUSED: GPU_RESERVE_TEST_MODE='${GPU_RESERVE_TEST_MODE-}' must be 0 or 1."
       _gr_log "         Anything else is a configuration error, not 'not test mode'."
       _GR_ENV_RC=7 ;;
  esac

  return "$_GR_ENV_RC"
}

# ===========================================================================
# DEFECT 3: MODE SELECTION. Production's allowlist, lockdir and freeze marker are constants.
# Test mode may override them, but ONLY when the telemetry source is demonstrably not the real
# driver tool and the lockdir is not the production one. An env var therefore cannot widen
# production onto GPUs 2-3: setting GPU_RESERVE_TEST_MODE=1 while talking to the real nvidia-smi
# is itself refused.
# ===========================================================================
# ---------------------------------------------------------------------------
# UNFORGEABLE TEST-MODE INTERLOCK (reviewer finding, 2026-10-26).
#
# The first interlock compared the telemetry command's BASENAME to "nvidia-smi". That is forgeable
# by copying the binary, and the reviewer demonstrated it. Reproduced here before the fix:
#     cp /usr/bin/nvidia-smi $TMP/copied_nvsmi
#     GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="2 3" GPU_RESERVE_NVSMI=$TMP/copied_nvsmi status
#       -> mode=test   allowed=2 3
# i.e. test mode against REAL telemetry, with GPUs 2-3 (another user's, in use) in the allowlist.
# A symlink was caught (readlink -f resolved it); a plain copy was not. The name is not the thing.
#
# THE THING is whether the configured telemetry source describes THIS MACHINE'S GPUS. So the
# interlock now compares DEVICE IDENTITIES: test mode is granted only if no UUID reported by the
# configured source also appears in the real driver's enumeration. A copied binary reports the real
# UUIDs and is refused; a shim reporting GPU-aaaa/GPU-bbbb overlaps nothing and is allowed. This
# cannot be forged by copying, renaming, wrapping or PATH games, because any source that can see the
# real devices is by definition describing them.
#
# If the real driver is unreachable (no nvidia-smi, driver down) there are no real GPUs to protect,
# so test mode is allowed and the decision is logged.
# ---------------------------------------------------------------------------
_gr_nvsmi_is_real() {           # cheap first filter only; NOT the interlock
  local head p real
  head=${NVSMI%% *}
  case "$head" in
    */*) p="$head" ;;
    *)   p=$(command -v -- "$head" 2>/dev/null) || return 1 ;;
  esac
  real=$(readlink -f -- "$p" 2>/dev/null) || real="$p"
  [ -n "$real" ] || real="$p"
  case "${real##*/}" in nvidia-smi|nvidia-smi.*) return 0 ;; esac
  return 1
}

# Returns 0 = synthetic (safe for test mode), 1 = describes real devices (refuse).
_gr_nvsmi_is_synthetic() {
  local tmp rc realnv u
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/gpu_reserve_interlock.XXXXXX") || {
    _gr_log "REFUSED: cannot create a temp dir to verify the test-mode interlock"; return 1; }

  realnv=$(command -v -- "$_GR_PROD_NVSMI" 2>/dev/null) || realnv=""
  if [ -z "$realnv" ]; then
    rm -rf "$tmp"
    _gr_log "test-mode interlock: no real '$_GR_PROD_NVSMI' on PATH, so there are no real GPUs to"
    _gr_log "        protect. Test mode allowed."
    return 0
  fi
  # Deliberately the SAME query form the rest of this file depends on (index,uuid), not a fourth
  # one: any shim that can serve gpu_reserve.sh at all can serve the interlock, so the interlock
  # cannot be dodged by a shim that simply does not implement its query. The uuid is field 2.
  "$realnv" --query-gpu=index,uuid --format=csv,noheader >"$tmp/real" 2>/dev/null
  rc=$?
  if [ "$rc" != 0 ] || [ ! -s "$tmp/real" ]; then
    rm -rf "$tmp"
    _gr_log "test-mode interlock: the real driver did not enumerate any GPU (rc=$rc), so there are"
    _gr_log "        no real devices to protect. Test mode allowed."
    return 0
  fi
  $NVSMI --query-gpu=index,uuid --format=csv,noheader >"$tmp/cand" 2>/dev/null
  rc=$?
  if [ "$rc" != 0 ]; then
    rm -rf "$tmp"
    # A source that cannot enumerate ANYTHING cannot be describing this machine's real GPUs, so
    # there is nothing for the interlock to protect and test mode is allowed. This is safe, not
    # lenient: acquisition still has to pass the telemetry gate, which will refuse with rc=5
    # because the same query just failed. Refusing here instead would mask every telemetry-failure
    # path behind rc=6 and make the fail-closed behaviour untestable.
    _gr_log "test-mode interlock: the configured telemetry source enumerated nothing (rc=$rc), so"
    _gr_log "        it cannot be describing real devices. Test mode allowed; acquisition will"
    _gr_log "        still refuse on unreadable telemetry (rc=5)."
    return 0
  fi
  # normalise the real UUIDs into a lookup string once (no pipeline, no process substitution).
  # Rows are "index, uuid"; take field 2.
  local realset="" idx
  while IFS=, read -r idx u || [ -n "$idx" ]; do
    u=${u//[[:space:]]/}
    [ -n "$u" ] && realset="$realset $u"
  done < "$tmp/real"
  while IFS=, read -r idx u || [ -n "$idx" ]; do
    u=${u//[[:space:]]/}
    [ -n "$u" ] || continue
    if [ "$realset" != "${realset/ $u/}" ]; then
      rm -rf "$tmp"
      _gr_log "REFUSED: GPU_RESERVE_TEST_MODE=1 but the configured telemetry source reports device"
      _gr_log "         UUID $u, which THIS MACHINE'S REAL DRIVER also reports."
      _gr_log "         That means it is describing real GPUs -- a copied, renamed or wrapped"
      _gr_log "         nvidia-smi, not a shim. Test mode may never run against real devices,"
      _gr_log "         because it is what allows GPUs 2-3 (another user's) into the allowlist."
      _gr_log "         A real harness reports synthetic UUIDs that overlap nothing."
      return 1
    fi
  done < "$tmp/cand"
  rm -rf "$tmp"
  _gr_log "test-mode interlock: configured telemetry reports no UUID known to the real driver"
  _gr_log "        -> genuinely synthetic. Test mode allowed."
  return 0
}

_GR_MODE=production
_GR_MODE_REASON=""

# ---------------------------------------------------------------------------
# FAIL-OPEN DEFECT, found by the Baselines agent 2026-10-26, fixed here.
#
# Both modes used `${GPU_RESERVE_GPUS:-<default>}`, whose `:-` treats SET-BUT-EMPTY exactly like
# UNSET. So a caller that deliberately narrowed the allowlist to nothing silently received the FULL
# default set instead. Measured before the fix:
#     test mode,   GPU_RESERVE_GPUS=""     -> rc=0, RESERVED GPU 0
#     production,  GPU_RESERVE_GPUS=""     -> `status` reported `allowed=0 1` (fallback);
#                                             acquire returned 4 only because FREEZE_ACTIVE
#                                             happened to refuse first and masked it
#     test mode,   GPU_RESERVE_GPUS="   "  -> rc=3, i.e. the RETRYABLE "no GPU free" code, so a
#                                             waiter loop would sleep forever on a config error
# A guard that falls back to a permissive default when its own configuration is unusable is worse
# than no guard, because it is cited as protection. `${VAR+set}` below distinguishes the three
# states, and an empty or malformed allowlist is now a hard, NON-RETRYABLE refusal (rc=7).
#
# ONE GPU PER RESERVATION. The allowlist is the set of indices this process may CHOOSE FROM; a
# reservation always covers exactly ONE index. Multi-GPU (e.g. DDP across 0 and 1) is NOT
# expressible, by design and by omission -- nothing in this project needs it. Do not simulate it by
# calling acquire twice: each call installs its own release trap in the same shell and the second
# trap REPLACES the first, so one of the two reservations would leak until it aged out. If multi-GPU
# is ever needed it is a deliberate feature (a record listing several indices, released atomically),
# not a loop over acquire.
# ---------------------------------------------------------------------------
_gr_allowlist_from_env() {
  # echoes the requested list; returns 0 = use it, 1 = unset (use the default), 7 = refuse
  if [ -z "${GPU_RESERVE_GPUS+set}" ]; then return 1; fi      # genuinely UNSET
  local raw="$GPU_RESERVE_GPUS" norm=()
  # shellcheck disable=SC2206
  norm=( $raw )                                               # word-split; drops pure whitespace
  if [ "${#norm[@]}" = 0 ]; then
    _gr_log "REFUSED: GPU_RESERVE_GPUS is SET BUT EMPTY ('$raw')."
    _gr_log "         An empty allowlist cannot be satisfied, so there is nothing to fall back to:"
    _gr_log "         this is a configuration error, not 'all GPUs are busy'. Refusing with rc=7"
    _gr_log "         (non-retryable) rather than quietly using the default set {$_GR_PROD_GPUS}."
    _gr_log "         To allow no GPUs, do not call acquire at all."
    return 7
  fi
  local g seen=""
  for g in "${norm[@]}"; do
    case "$g" in ''|*[!0-9]*)
      _gr_log "REFUSED: GPU_RESERVE_GPUS contains '$g', which is not a GPU index. The whole"
      _gr_log "         allowlist is rejected (rc=7); a malformed guard is never partially honoured."
      return 7 ;;
    esac
    case " $seen " in *" $g "*)
      _gr_log "REFUSED: GPU_RESERVE_GPUS lists index $g more than once ('$raw'); rc=7."
      return 7 ;;
    esac
    seen="$seen $g"
  done
  printf '%s\n' "${norm[*]}"
  return 0
}

_gr_mode_init() {
  ALLOWED_GPUS="$_GR_PROD_GPUS"
  LOCKDIR="$_GR_PROD_LOCKDIR"
  FREEZE_FILE="$_GR_PROD_FREEZE"
  NVSMI="$_GR_PROD_NVSMI"
  OVERRIDE_LOG="$_GR_PROD_OVERRIDE_LOG"

  # D-060: PROJECT is refused, in BOTH modes, before anything else.
  # It is checked first because it used to be the SOURCE of the constants every other check in this
  # function compares against -- so a PROJECT that reached those checks made all of them vacuous.
  # Refused rather than ignored: the whole lesson of D-047 is that silently dropping an override
  # looks identical to honouring it from the call site. Test mode is not exempt, because test mode
  # has its own GPU_RESERVE_LOCKDIR/FREEZE_FILE/OVERRIDE_LOG knobs and never needs PROJECT.
  if [ -n "${PROJECT+x}" ] && [ "${PROJECT:-}" != "$_GR_PROD_ROOT" ]; then
    _gr_log "REFUSED: PROJECT='${PROJECT:-}' tried to move the project root. It is a LITERAL,"
    _gr_log "         hard-wired to $_GR_PROD_ROOT, because the lockdir, the freeze marker and the"
    _gr_log "         freeze-override audit log are all anchored to it. Setting it relocated the"
    _gr_log "         mutex (two runners with different values cannot exclude each other -- the"
    _gr_log "         D-001 triple-booking), moved the freeze marker so the project read as thawed,"
    _gr_log "         and redirected the audit log so an override left no record. rc=6."
    _gr_log "         Mocked harnesses set GPU_RESERVE_TEST_MODE=1 with GPU_RESERVE_LOCKDIR."
    return 6
  fi

  # Validate every scalar knob first (ranges, enums, set-but-empty). rc=7 is non-retryable.
  _gr_env_init || return 7

  # Validate the requested allowlist BEFORE anything else, in BOTH modes, so an unusable
  # configuration can never reach a code path that falls back to a permissive default.
  local _gr_req _gr_areq_rc
  _gr_req=$(_gr_allowlist_from_env); _gr_areq_rc=$?
  case "$_gr_areq_rc" in
    7) return 7 ;;
    1) _gr_req="" ;;          # unset: each mode applies its own default below
  esac

  if [ "${GPU_RESERVE_TEST_MODE:-0}" = 1 ]; then
    # The telemetry source must be configured BEFORE the interlock can inspect it.
    NVSMI="${GPU_RESERVE_NVSMI:-$_GR_PROD_NVSMI}"
    if [ -z "${GPU_RESERVE_LOCKDIR:-}" ]; then
      _gr_log "REFUSED: test mode requires GPU_RESERVE_LOCKDIR (never the production lockdir)."
      return 6
    fi
    if [ "$GPU_RESERVE_LOCKDIR" = "$_GR_PROD_LOCKDIR" ]; then
      _gr_log "REFUSED: test mode may not write into the production lockdir $_GR_PROD_LOCKDIR."
      return 6
    fi
    # Cheap first filter: an obvious real nvidia-smi by name/symlink.
    if _gr_nvsmi_is_real; then
      _gr_log "REFUSED: GPU_RESERVE_TEST_MODE=1 but GPU_RESERVE_NVSMI ('$NVSMI') resolves to a"
      _gr_log "         REAL nvidia-smi. Test mode may never run against the live driver."
      return 6
    fi
    # The actual interlock: device IDENTITY, which copying the binary cannot forge.
    if ! _gr_nvsmi_is_synthetic; then
      return 6
    fi
    _GR_MODE=test
    _GR_MODE_REASON="synthetic nvidia-smi '$NVSMI', lockdir $GPU_RESERVE_LOCKDIR"
    LOCKDIR="$GPU_RESERVE_LOCKDIR"
    ALLOWED_GPUS="${_gr_req:-$_GR_PROD_GPUS}"
    FREEZE_FILE="${GPU_RESERVE_FREEZE_FILE:-$_GR_PROD_FREEZE}"
    OVERRIDE_LOG="${GPU_RESERVE_OVERRIDE_LOG:-$_GR_PROD_OVERRIDE_LOG}"
  else
    # ---- production. Refuse, do not merely ignore, any attempt to redefine the allowlist.
    # (A set-but-empty or malformed value never reaches here: it was refused above with rc=7.)
    if [ -n "$_gr_req" ] && [ "$_gr_req" != "$_GR_PROD_GPUS" ]; then
      _gr_log "REFUSED: GPU_RESERVE_GPUS='$GPU_RESERVE_GPUS' tried to change the production"
      _gr_log "         allowlist. Production is hard-wired to {$_GR_PROD_GPUS}; GPUs 2 and 3"
      _gr_log "         belong to other users of this shared host and are OFF LIMITS."
      _gr_log "         No environment variable can widen this. (Mocked harnesses set"
      _gr_log "         GPU_RESERVE_TEST_MODE=1 with a shim nvidia-smi instead.)"
      return 6
    fi
    if [ -n "${GPU_RESERVE_FREEZE_FILE:-}" ] && [ "$GPU_RESERVE_FREEZE_FILE" != "$_GR_PROD_FREEZE" ]; then
      _gr_log "REFUSED: GPU_RESERVE_FREEZE_FILE='$GPU_RESERVE_FREEZE_FILE' tried to redirect the"
      _gr_log "         freeze marker in production. The marker path is hard-wired to"
      _gr_log "         $_GR_PROD_FREEZE so the gate cannot be faked away by an env var."
      return 6
    fi
    # ---- BLOCKING: D-001 WAS REACHABLE THROUGH THE LOCKDIR (reviewer finding, 2026-10-26).
    # Production previously did `if [ -n "$GPU_RESERVE_LOCKDIR" ]; then LOCKDIR=...; fi`, with no
    # refusal. Measured before this fix:
    #     GPU_RESERVE_LOCKDIR=/tmp/evil_lockdir bash scripts/gpu_reserve.sh status
    #       -> mode=production   lockdir=/tmp/evil_lockdir
    # The mutex lives at $LOCKDIR/.reserve.mutex, so two runners started with DIFFERENT values
    # take DIFFERENT mutexes and cannot exclude each other -- which is precisely the
    # 2026-10-26T12:39 triple-booking of GPU 0 that this entire file exists to prevent. The
    # reservation records live there too, so neither runner can even see the other's claim. An
    # env var must never be able to relocate a mutex: a lock at a caller-chosen path is not a lock.
    if [ -n "${GPU_RESERVE_LOCKDIR:-}" ] && [ "$GPU_RESERVE_LOCKDIR" != "$_GR_PROD_LOCKDIR" ]; then
      _gr_log "REFUSED: GPU_RESERVE_LOCKDIR='$GPU_RESERVE_LOCKDIR' tried to relocate the reservation"
      _gr_log "         directory in production. It is hard-wired to $_GR_PROD_LOCKDIR."
      _gr_log "         The mutex lives inside it, so two runners with different values would take"
      _gr_log "         DIFFERENT mutexes and could not exclude each other -- reintroducing the"
      _gr_log "         exact triple-booking of GPU 0 observed on 2026-10-26T12:39 (D-001)."
      _gr_log "         Mocked harnesses set GPU_RESERVE_TEST_MODE=1 with a synthetic nvidia-smi."
      return 6
    fi
    if [ -n "${GPU_RESERVE_NVSMI:-}" ] && [ "$GPU_RESERVE_NVSMI" != "$_GR_PROD_NVSMI" ]; then
      _gr_log "REFUSED: GPU_RESERVE_NVSMI='$GPU_RESERVE_NVSMI' tried to replace the telemetry source"
      _gr_log "         in production. Pointing production at a fake nvidia-smi would make a BUSY"
      _gr_log "         GPU read as free and trample our own running campaign."
      return 6
    fi
    if [ -n "${GPU_RESERVE_OVERRIDE_LOG:-}" ] && [ "$GPU_RESERVE_OVERRIDE_LOG" != "$_GR_PROD_OVERRIDE_LOG" ]; then
      _gr_log "REFUSED: GPU_RESERVE_OVERRIDE_LOG='$GPU_RESERVE_OVERRIDE_LOG' tried to redirect the"
      _gr_log "         freeze-override AUDIT log in production. Redirecting it to /dev/null would"
      _gr_log "         let a frozen-project override happen with no record. Hard-wired to"
      _gr_log "         $_GR_PROD_OVERRIDE_LOG."
      return 6
    fi
  fi

  # Final, mode-independent backstop. Belt and braces for defects 3 and the fail-open above: the
  # computed allowlist must be non-empty, numeric, and (in production) a subset of {0,1}. This runs
  # whatever path set ALLOWED_GPUS, so a future edit that reintroduces a permissive fallback still
  # cannot produce an empty or out-of-range allowlist.
  local g n=0
  for g in $ALLOWED_GPUS; do n=$((n+1)); done
  if [ "$n" = 0 ]; then
    _gr_log "REFUSED: the computed GPU allowlist is EMPTY. Nothing can be reserved, and this is a"
    _gr_log "         configuration error, not a busy-GPU condition. rc=7 (non-retryable)."
    return 7
  fi
  for g in $ALLOWED_GPUS; do
    case "$g" in ''|*[!0-9]*) _gr_log "REFUSED: non-numeric GPU index '$g' in the allowed set"; return 7 ;; esac
    if [ "$_GR_MODE" = production ]; then
      case " $_GR_PROD_GPUS " in *" $g "*) ;; *) _gr_log "REFUSED: gpu$g outside production {$_GR_PROD_GPUS}"; return 6 ;; esac
    fi
  done
  MUTEX="$LOCKDIR/.reserve.mutex"
  return 0
}

# ---------------------------------------------------------------------------
# process identity: pid + start time, so a recycled PID cannot inherit a dead reservation.
# /proc/<pid>/stat: field 1 pid, 2 comm, 3 state, 4 ppid, 5 pgrp, ..., 22 starttime.
# comm may contain spaces and parentheses, so cut after the LAST ')': what remains starts at
# field 3, making starttime its 20th token and pgrp its 3rd.
# DEFECT 2: no pipeline here. v1 used `printf | awk`, whose status reflected only awk.
# ---------------------------------------------------------------------------
_GR_STAT_TOK=()
_gr_read_stat() {
  local pid="$1" st rest
  _GR_STAT_TOK=()
  [ -r "/proc/$pid/stat" ] || return 1
  IFS= read -r st < "/proc/$pid/stat" || return 1
  rest=${st##*\) }
  [ "$rest" != "$st" ] || return 1          # no ')' found => not a stat line we understand
  read -r -a _GR_STAT_TOK <<<"$rest" || return 1
  [ "${#_GR_STAT_TOK[@]}" -ge 20 ] || return 1
  return 0
}

_gr_starttime() {
  local v
  _gr_read_stat "$1" || return 1
  v=${_GR_STAT_TOK[19]}
  case "$v" in ''|*[!0-9]*) return 1 ;; esac
  printf '%s\n' "$v"
}

_gr_pgid() {
  local v
  _gr_read_stat "$1" || return 1
  v=${_GR_STAT_TOK[2]}
  case "$v" in ''|*[!0-9]*) return 1 ;; esac
  printf '%s\n' "$v"
}

# /proc/<pid>/stat field 6 = session id; after cutting at the last ')' it is token 3 (index 3).
_gr_sid() {
  local v
  _gr_read_stat "$1" || return 1
  v=${_GR_STAT_TOK[3]}
  case "$v" in ''|*[!0-9]*) return 1 ;; esac
  printf '%s\n' "$v"
}

# ---------------------------------------------------------------------------
# D-049/R3: THE LIVENESS LOCK -- the actual close for the reservation residual.
#
# What was wrong before. Ownership liveness was decided from /proc: owner pid + start time, and
# for descendants a process-GROUP scan that _gr_live_in_pgid refuses to trust unless the owner
# LED its group (gpu_reserve.sh:615, `return 2`). A runner started the ordinary way -- `bash
# run_r1_imma_official.sh` from any non-interactive parent -- INHERITS its parent's group, so
# pgid != pid, the scan returns 2, and the only remaining protection is the 900 s grace window.
# Adding `setsid` to two launchers does not fix that: it is a CALLER-SIDE CONVENTION that the
# reservation cannot verify, cannot enforce, and that any other invocation path silently omits.
# A guarantee that depends on how the caller happened to spell the command is not a guarantee.
#
# What replaces it. A reservation now also owns an flock(2) on a dedicated hold file, taken by
# the LONG-LIVED runner shell on a file descriptor. An flock lives on the open file description,
# so it is shared with every process that inherits the fd -- across fork AND across execve -- and
# the kernel releases it only when the LAST such descriptor is closed, i.e. when the owner and
# every descendant holding it have exited. A would-be acquirer therefore does not have to infer
# descent from /proc at all: it asks the kernel "can I take this lock?", and a refusal is positive
# proof that something from that reservation is still alive.
#
# This is strictly stronger than the pgid scan on every axis that mattered:
#   * independent of group/session leadership, so no `setsid` convention is required
#   * survives re-parenting to init, which is what destroyed the ppid chain
#   * survives SIGKILL of the owner (nothing runs in the owner; the kernel does the accounting)
#   * holds from BEFORE the record is published until after the work exits, so a job that has not
#     yet created a CUDA context is already protected (the delayed-init case)
#   * cannot be forged by an env var: the path is validated to live inside $LOCKDIR
#
# MEASURED, on this host, before writing this code (bash 5.1.16, util-linux flock 2.37.2):
#   owner took `exec {FD}>>hold; flock -x $FD`, spawned a child shell that did `exec sleep 300`
#   (so the surviving process is an EXECED BINARY, not a bash fork), then the owner was SIGKILLed.
#   With the owner's /proc entry gone and the grandchild alive, `flock -n -x hold -c true`
#   returned 1 (BUSY). The lock had been inherited through fork and through execve and was still
#   held by the grandchild. That is the guarantee the pgid scan could not give.
#
# Residual, stated plainly: the fd is inherited by EVERYTHING the runner spawns, so a process that
# the runner deliberately leaves running (a stray `nohup`) keeps the reservation alive. That is the
# safe direction -- it refuses a GPU rather than trampling one -- and `holders` names the pids so an
# operator can see exactly why, with `release` as the deliberate override.
# ---------------------------------------------------------------------------

# The hold path is caller-supplied (the runner shell must own the fd), so it is VALIDATED, never
# trusted: it must be $LOCKDIR/.hold.<digits>.<campaign-charset>. Otherwise a caller could name a
# file that is permanently locked (GPU unreclaimable for ever) or never locked (no protection).
_gr_hold_path_ok() {
  local p="$1" camp="$2" base
  [ -n "$p" ] && [ "$p" != "-" ] || return 1
  case "$p" in "$LOCKDIR"/*) ;; *) return 1 ;; esac
  base=${p#"$LOCKDIR"/}
  case "$base" in */*) return 1 ;; esac            # no nesting, no traversal
  case "$base" in .hold.*) ;; *) return 1 ;; esac
  local rest=${base#.hold.}
  local pidpart=${rest%%.*}
  case "$pidpart" in ''|*[!0-9]*) return 1 ;; esac
  [ "$rest" != "$pidpart" ] || return 1            # a campaign part must be present
  local camppart=${rest#"$pidpart".}
  case "$camppart" in ''|*[!A-Za-z0-9._-]*) return 1 ;; esac
  [ -z "$camp" ] || [ "$camppart" = "$camp" ] || return 1
  return 0
}

# Can we take the hold lock right now? 0 = free (nothing from that reservation is alive),
# 1 = BUSY (a live owner or descendant holds it), 2 = cannot tell (no flock(1), missing file).
# `flock -n -x <file> -c true` opens, tries, runs true, and closes -- it never leaves a lock behind.
# The file is NOT created here: a missing hold file must read as "cannot tell", not as "free".
_gr_hold_free() {
  local p="$1" rc
  [ -n "$p" ] && [ "$p" != "-" ] || return 2
  [ -e "$p" ] || return 2
  command -v flock >/dev/null 2>&1 || return 2
  flock -n -x "$p" -c true >/dev/null 2>&1; rc=$?
  case "$rc" in
    0) return 0 ;;
    1) return 1 ;;
    *) return 2 ;;        # flock itself failed (permissions, ENOLCK): never call that "free"
  esac
}

# Which pids currently hold the hold file open? DIAGNOSTIC ONLY -- the allow/refuse decision is
# made by the kernel in _gr_hold_free, never by this scan.
#
# PERFORMANCE DEFECT I INTRODUCED AND THEN MEASURED (2026-10-26). The first version walked
# /proc/<pid>/fd/* in bash and ran `readlink` per descriptor: on this host that is 2,247 forks,
# and it is reached from _gr_reservation_state, which runs INSIDE the flock-held critical section
# during acquire. It stalled a measurement run for tens of seconds per call while holding the
# reservation mutex -- i.e. a diagnostic string was blocking the thing it was describing. A single
# `find -lname` does the same job in ~149 ms with one fork, and is wrapped in `timeout` so a
# pathological /proc can never hold the mutex either. If it cannot answer, the caller still has
# the kernel's verdict; only the list of pids is lost, and it says so.
_gr_holders() {
  local p="$1" out=""
  [ -n "$p" ] && [ -e "$p" ] || return 1
  if command -v find >/dev/null 2>&1; then
    if command -v timeout >/dev/null 2>&1; then
      out=$(timeout 10 find /proc/[0-9]*/fd -lname "$p" -printf '%h\n' 2>/dev/null \
            | sed 's#^/proc/##; s#/fd$##' | sort -un | tr '\n' ' ') || out=""
    else
      out=$(find /proc/[0-9]*/fd -lname "$p" -printf '%h\n' 2>/dev/null \
            | sed 's#^/proc/##; s#/fd$##' | sort -un | tr '\n' ' ') || out=""
    fi
  fi
  out=${out% }
  [ -n "$out" ] || return 1
  printf '%s\n' "$out"
}

_gr_proc_uid() { stat -c %u "/proc/$1" 2>/dev/null; }

# is this pid alive AND the same process that took the reservation?
_gr_owner_alive() {
  local pid="$1" want="$2" now
  [ -n "$pid" ] || return 1
  [ -d "/proc/$pid" ] || return 1
  now=$(_gr_starttime "$pid") || return 1
  if [ -z "$want" ] || [ "$want" = "-" ]; then return 0; fi   # legacy/adopted record
  [ "$now" = "$want" ]
}

# DEFECT 5: is ANY process of the dead owner's process group still alive? A campaign runner's
# python children inherit its pgid, so this is how a surviving CUDA child is detected even before
# nvidia-smi lists it.
#
# REVIEWER FINDING 2026-10-26: the first version SKIPPED the whole check when the asking process
# shared the dead holder's pgid (`[ "$want" != "$mypg" ] || return 1`), because a non-interactive
# parent shell gives its children its own process group and we must not mistake OURSELVES for the
# dead campaign's survivor. The reviewer reproduced the consequence: in that configuration the
# record was judged STALE rather than ORPHAN, leaving the grace window as the sole protection --
# and a cold model load can outlast it.
#
# Fixed in two parts.
#
# (1) Do not skip the group; EXCLUDE OUR OWN PROCESS TREE from it. A survivor must not be us, an
#     ancestor of ours, or a descendant of ours. One /proc pass builds pid -> (ppid, pgid);
#     ancestors from walking ppid upward, descendants from a breadth-first walk down.
#
# (2) Use the pgid as evidence of DESCENT only when the dead owner was its group's LEADER
#     (recorded pgid == recorded owner pid). That is the only case in which "member of this group"
#     really means "started by the owner". When the owner merely INHERITED its parent's group --
#     the normal case for a script launched from a non-interactive parent -- the group also
#     contains the parent and every unrelated sibling, so treating members as survivors would make
#     a GPU permanently unreclaimable the moment any sibling outlived the record. (Found while
#     testing part (1): a sibling placeholder blocked reclamation forever.)
#
# RESIDUAL LIMITATION, stated rather than hidden: for a NON-leader owner there is no sound way to
# identify its orphaned children from /proc after it dies, because they are re-parented to init and
# the ppid chain to the owner is gone. In that case protection falls to the CUDA-context check plus
# the grace window -- which is why the grace default was raised from 120s to 900s, enough for a
# cold model load. Runners that want the strong guarantee should be launched with `setsid`, which
# makes the runner a group leader; gpu_reserve.sh logs which regime applies.
_gr_live_in_pgid() {
  local want="$1" me="$2" owner="${3:-}" d pid
  [ -n "$want" ] && [ "$want" != "-" ] || return 1
  case "$want" in *[!0-9]*) return 1 ;; esac
  case "$me" in ''|*[!0-9]*) me="" ;; esac
  # pgid is evidence of descent only for a group LEADER (see (2) above).
  if [ -z "$owner" ] || [ "$want" != "$owner" ]; then return 2; fi

  local -A _pp=() _pg=()
  local -a members=()
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    _gr_read_stat "$pid" || continue
    _pp[$pid]="${_GR_STAT_TOK[1]}"          # ppid  (/proc stat field 4)
    _pg[$pid]="${_GR_STAT_TOK[2]}"          # pgrp  (/proc stat field 5)
    [ "${_GR_STAT_TOK[2]}" = "$want" ] && members+=("$pid")
  done
  [ "${#members[@]}" -gt 0 ] || return 1

  # ---- our own tree: ancestors (walk up) ...
  local -A mine=()
  if [ -n "$me" ]; then
    local cur="$me" hops=0
    while [ -n "$cur" ] && [ "$cur" != 0 ] && [ "$hops" -lt 256 ]; do
      mine[$cur]=1
      cur="${_pp[$cur]:-}"
      hops=$((hops+1))
    done
    # ... and descendants (walk down over the children map)
    local -a queue=("$me")
    local head child
    while [ "${#queue[@]}" -gt 0 ]; do
      head="${queue[0]}"; queue=("${queue[@]:1}")
      for child in "${!_pp[@]}"; do
        if [ "${_pp[$child]}" = "$head" ] && [ -z "${mine[$child]:-}" ]; then
          mine[$child]=1; queue+=("$child")
        fi
      done
    done
  fi

  for pid in "${members[@]}"; do
    [ -z "${mine[$pid]:-}" ] || continue    # that is us or our own tree, not a survivor
    printf '%s\n' "$pid"
    return 0
  done
  return 1
}

_gr_record() { printf '%s\n' "$LOCKDIR/gpu$1.owner"; }

# DEFECT 2: no `sed | head` pipeline; status is the loop's own.
_gr_read_field() {
  local f="$1" k="$2" line
  [ -r "$f" ] || return 1
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in "$k="*) printf '%s\n' "${line#"$k="}"; return 0 ;; esac
  done < "$f"
  return 1
}

# ===========================================================================
# DEFECT 1 + 2: GPU TELEMETRY, FAIL-CLOSED.
#
# v1 ran each nvidia-smi query inside a process substitution -- `while read ... done < <($NVSMI
# ... 2>/dev/null)` -- so the query's exit status was unobservable, and a total failure simply left
# the arrays empty. _gr_classify then defaulted memory to 0 and saw no processes, i.e. "FREE", and
# handed out a GPU on the strength of telemetry it had never actually read.
#
# v2 runs each query into a temp file with its rc captured, then VALIDATES:
#   * rc must be 0 (124 from `timeout` is reported as a timeout)
#   * the index,uuid and index,memory.used queries must be non-empty
#   * EVERY allowed index must appear in BOTH, with a plausible UUID and an integer memory value
#   * "[N/A]", "[Not Supported]", headers and garbage are malformed, not zero
# The compute-apps query is allowed to be empty (that legitimately means "no CUDA processes"), but
# must still exit 0, and every line it does print must parse.
# Any violation sets _GR_TELEMETRY=UNKNOWN with a reason, and acquisition refuses with rc=5.
# ===========================================================================
declare -A _GR_MEM=()
declare -A _GR_APPS=()
declare -A _GR_UUID2IDX=()
declare -A _GR_IDX2UUID=()
_GR_TELEMETRY=UNKNOWN
_GR_TELEMETRY_WHY="not probed"

_gr_tmpdir() {
  local d
  d=$(mktemp -d "${TMPDIR:-/tmp}/gpu_reserve_probe.XXXXXX" 2>/dev/null) || return 1
  printf '%s\n' "$d"
}

# $1 = output file, rest = nvidia-smi args. Returns the query's real exit status.
_gr_nvsmi_run() {
  local out="$1"; shift
  local rc
  if [ "$NVSMI_TIMEOUT" != 0 ] && command -v timeout >/dev/null 2>&1; then
    timeout -k 5 "$NVSMI_TIMEOUT" $NVSMI "$@" >"$out" 2>"$out.err"
    rc=$?
  else
    $NVSMI "$@" >"$out" 2>"$out.err"
    rc=$?
  fi
  return "$rc"
}

_gr_telemetry_fail() {
  _GR_TELEMETRY=UNKNOWN
  _GR_TELEMETRY_WHY="$*"
  _gr_log "TELEMETRY UNKNOWN: $*"
  return 5
}

_gr_probe() {
  local tmp rc line idx uuid mem pid i n
  _GR_MEM=(); _GR_APPS=(); _GR_UUID2IDX=(); _GR_IDX2UUID=()
  _GR_TELEMETRY=UNKNOWN; _GR_TELEMETRY_WHY="not probed"

  tmp=$(_gr_tmpdir) || { _gr_telemetry_fail "cannot create a temp dir for the nvidia-smi probe"; return 5; }

  # ---- query 1: index,uuid -------------------------------------------------
  _gr_nvsmi_run "$tmp/uuid" --query-gpu=index,uuid --format=csv,noheader
  rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    rm -rf "$tmp"; _gr_telemetry_fail "'$NVSMI --query-gpu=index,uuid' TIMED OUT after ${NVSMI_TIMEOUT}s (rc=$rc)"; return 5
  fi
  if [ "$rc" != 0 ]; then
    local e1; e1=$(head -c 200 "$tmp/uuid.err" 2>/dev/null) || e1=""
    rm -rf "$tmp"; _gr_telemetry_fail "'$NVSMI --query-gpu=index,uuid' exited $rc: ${e1:-<no stderr>}"; return 5
  fi
  if [ ! -s "$tmp/uuid" ]; then
    rm -rf "$tmp"; _gr_telemetry_fail "'$NVSMI --query-gpu=index,uuid' printed NOTHING (empty output is not 'no GPUs busy')"; return 5
  fi
  n=0
  while IFS=, read -r idx uuid || [ -n "$idx" ]; do
    idx=${idx//[[:space:]]/}; uuid=${uuid//[[:space:]]/}
    [ -n "$idx$uuid" ] || continue
    n=$((n+1))
    case "$idx" in ''|*[!0-9]*)
      rm -rf "$tmp"; _gr_telemetry_fail "malformed index,uuid row (index '$idx' is not a number): header-only or garbage output"; return 5 ;;
    esac
    case "$uuid" in
      GPU-*|MIG-*) ;;
      *) rm -rf "$tmp"; _gr_telemetry_fail "malformed UUID '$uuid' for gpu$idx (expected GPU-... or MIG-...)"; return 5 ;;
    esac
    _GR_UUID2IDX[$uuid]=$idx
    _GR_IDX2UUID[$idx]=$uuid
  done < "$tmp/uuid"
  if [ "$n" = 0 ]; then
    rm -rf "$tmp"; _gr_telemetry_fail "index,uuid query produced no parsable rows"; return 5
  fi

  # ---- query 2: index,memory.used -----------------------------------------
  _gr_nvsmi_run "$tmp/mem" --query-gpu=index,memory.used --format=csv,noheader,nounits
  rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    rm -rf "$tmp"; _gr_telemetry_fail "'$NVSMI --query-gpu=index,memory.used' TIMED OUT after ${NVSMI_TIMEOUT}s (rc=$rc)"; return 5
  fi
  if [ "$rc" != 0 ]; then
    local e2; e2=$(head -c 200 "$tmp/mem.err" 2>/dev/null) || e2=""
    rm -rf "$tmp"; _gr_telemetry_fail "'$NVSMI --query-gpu=index,memory.used' exited $rc: ${e2:-<no stderr>}"; return 5
  fi
  if [ ! -s "$tmp/mem" ]; then
    rm -rf "$tmp"; _gr_telemetry_fail "'$NVSMI --query-gpu=index,memory.used' printed NOTHING"; return 5
  fi
  while IFS=, read -r idx mem || [ -n "$idx" ]; do
    idx=${idx//[[:space:]]/}; mem=${mem//[[:space:]]/}
    [ -n "$idx$mem" ] || continue
    case "$idx" in ''|*[!0-9]*)
      rm -rf "$tmp"; _gr_telemetry_fail "malformed index,memory.used row (index '$idx' is not a number)"; return 5 ;;
    esac
    # "[N/A]" / "[Not Supported]" are UNKNOWN. v1 coerced them to 0, i.e. to "free".
    case "$mem" in ''|*[!0-9]*)
      rm -rf "$tmp"; _gr_telemetry_fail "gpu$idx reported memory.used='$mem', which is not a number -- refusing to read it as 0 MiB"; return 5 ;;
    esac
    _GR_MEM[$idx]=$mem
  done < "$tmp/mem"

  # ---- query 3: compute apps (empty is legitimate; a failure is not) --------
  _gr_nvsmi_run "$tmp/apps" --query-compute-apps=pid,gpu_uuid --format=csv,noheader
  rc=$?
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    rm -rf "$tmp"; _gr_telemetry_fail "'$NVSMI --query-compute-apps' TIMED OUT after ${NVSMI_TIMEOUT}s (rc=$rc)"; return 5
  fi
  if [ "$rc" != 0 ]; then
    local e3; e3=$(head -c 200 "$tmp/apps.err" 2>/dev/null) || e3=""
    rm -rf "$tmp"; _gr_telemetry_fail "'$NVSMI --query-compute-apps' exited $rc -- cannot tell whether a GPU is in use: ${e3:-<no stderr>}"; return 5
  fi
  while IFS=, read -r pid uuid || [ -n "$pid" ]; do
    pid=${pid//[[:space:]]/}; uuid=${uuid//[[:space:]]/}
    [ -n "$pid$uuid" ] || continue
    case "$pid" in ''|*[!0-9]*)
      rm -rf "$tmp"; _gr_telemetry_fail "malformed compute-apps row (pid '$pid' is not a number)"; return 5 ;;
    esac
    idx=${_GR_UUID2IDX[$uuid]:-}
    if [ -z "$idx" ]; then
      rm -rf "$tmp"; _gr_telemetry_fail "compute-apps names gpu_uuid '$uuid', which the index,uuid query never listed -- the two views disagree"; return 5
    fi
    _GR_APPS[$idx]="${_GR_APPS[$idx]:-} $pid"
  done < "$tmp/apps"

  # ---- every ALLOWED index must be fully covered by both views -------------
  for i in $ALLOWED_GPUS; do
    if [ -z "${_GR_IDX2UUID[$i]:-}" ]; then
      rm -rf "$tmp"; _gr_telemetry_fail "allowed gpu$i is absent from the index,uuid query (only [${!_GR_IDX2UUID[*]}] reported) -- truncated telemetry"; return 5
    fi
    if [ -z "${_GR_MEM[$i]:-}" ]; then
      rm -rf "$tmp"; _gr_telemetry_fail "allowed gpu$i is absent from the memory.used query (only [${!_GR_MEM[*]}] reported) -- truncated telemetry"; return 5
    fi
  done

  rm -rf "$tmp"
  _GR_TELEMETRY=OK
  _GR_TELEMETRY_WHY="$(printf 'gpus=[%s] via %s' "${!_GR_IDX2UUID[*]}" "$NVSMI")"
  return 0
}

# ===========================================================================
# DEFECT 4: inherited CUDA_VISIBLE_DEVICES, and index<->UUID mapping.
#
# MEASURED FACT (this host, 2026-10-26): nvidia-smi IGNORES CUDA_VISIBLE_DEVICES -- with CVD
# unset, "1", "2,3" and "bogus", `nvidia-smi --query-gpu=index,uuid` returned the identical four
# rows. So the NVML indices this script reserves are PHYSICAL and are not renumbered by an
# inherited CVD. The review's premise that "nvidia-smi index N and the physical device may differ"
# does NOT hold here, and v2 does not pretend to fix a renumbering that does not occur.
#
# What IS a real hazard, and is fixed:
#   (a) a CUDA CLIENT (torch) does honour CVD, so a runner that forgets to export it inherits
#       whatever its parent had and lands on a device nobody reserved. gpu_reserve_hold therefore
#       exports CUDA_VISIBLE_DEVICES itself and reports what it displaced.
#   (b) an inherited empty CVD means "no GPUs visible", which makes a job fall back to CPU or die
#       confusingly; an invalid one is equally silent. Both are reported loudly.
#   (c) an index is only a NAME for a device. v2 records the reserved device's UUID and
#       re-validates index->UUID on every operation, so a renumbering (driver reload, GPU reset,
#       MIG reconfiguration) is DETECTED rather than silently honoured.
# ===========================================================================
_gr_cvd_classify() {
  local v="$1"
  if [ -z "$v" ]; then printf 'EMPTY\n'; return 0; fi
  case "$v" in
    GPU-*|MIG-*)  printf 'UUID_LIST\n'  ;;
    *[!0-9,]*)    printf 'INVALID\n'    ;;   # anything but digits and commas
    *,,*|,*|*,)   printf 'INVALID\n'    ;;   # malformed index list
    *)            printf 'INDEX_LIST\n' ;;
  esac
  return 0
}

_gr_cvd_audit() {
  local kind
  if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    _gr_log "inherited CUDA_VISIBLE_DEVICES: <unset> (clean)"
    return 0
  fi
  kind=$(_gr_cvd_classify "$CUDA_VISIBLE_DEVICES")
  case "$kind" in
    EMPTY)
      _gr_log "WARNING inherited CUDA_VISIBLE_DEVICES='' -- an EMPTY value hides EVERY GPU from"
      _gr_log "        CUDA. Any job that kept it would silently find 0 devices. It will be"
      _gr_log "        OVERWRITTEN with the reserved device." ;;
    INVALID)
      _gr_log "WARNING inherited CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' is not a valid"
      _gr_log "        index or UUID list. It will be OVERWRITTEN with the reserved device." ;;
    *)
      _gr_log "inherited CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' ($kind) -- note that"
      _gr_log "        nvidia-smi ignores it (measured), so reserved indices stay physical; it"
      _gr_log "        will be OVERWRITTEN with the reserved device." ;;
  esac
  return 0
}

# ---------------------------------------------------------------------------
# Classify an allowed GPU: FREE | FOREIGN | OURS_UNLOCKED | UNACCOUNTED
# Called only for GPUs carrying no live reservation, and only after a SUCCESSFUL probe.
# ---------------------------------------------------------------------------
_gr_classify() {
  local i="$1" pid uid ours=0 foreign=0 accounted=0 mem
  if [ "$_GR_TELEMETRY" != OK ]; then echo UNKNOWN; return; fi
  for pid in ${_GR_APPS[$i]:-}; do
    uid=$(_gr_proc_uid "$pid")
    if [ -z "$uid" ]; then
      # nvidia-smi lists this pid as holding a CUDA context yet we cannot see it in /proc:
      # hidepid, i.e. another user's. Refuse -- never displace.
      _gr_log "gpu$i: compute pid $pid not visible in /proc -> treating as another user's"
      foreign=1
    elif [ "$uid" = "$MYUID" ]; then
      ours=1; accounted=1
    else
      foreign=1
    fi
  done
  [ "$foreign" = 1 ] && { echo FOREIGN; return; }
  [ "$ours" = 1 ] && { echo OURS_UNLOCKED; return; }
  mem=${_GR_MEM[$i]:-}
  case "$mem" in ''|*[!0-9]*) echo UNKNOWN; return ;; esac   # never coerce to 0 (defect 1)
  if [ "$accounted" = 0 ] && [ "$mem" -ge "$FOREIGN_MIB" ]; then echo UNACCOUNTED; return; fi
  echo FREE
}

# ===========================================================================
# DEFECT 5: reservation state for one GPU (no locking; callers hold the mutex).
#   echoes: FREE | "HELD <pid> <campaign>" | "ORPHAN <pid> <campaign> <why>" | "STALE <pid> <campaign>"
#
# v1 had only HELD/STALE: owner pid dead => STALE => record deleted => GPU handed out. Verified
# live: SIGKILL the holder while its child survives and the next acquirer was given the same GPU.
# A dead runner does not mean the work stopped -- a python child keeps its CUDA context, and may
# not even be listed by nvidia-smi yet. ORPHAN keeps the record and refuses the GPU.
# ===========================================================================
_gr_reservation_state() {
  local i="$1" rec pid start camp pgid sid hold uuid age mtime nowts survivor
  rec=$(_gr_record "$i")
  [ -f "$rec" ] || { echo FREE; return; }
  pid=$(_gr_read_field "$rec" pid) || pid=""
  start=$(_gr_read_field "$rec" start) || start=""
  camp=$(_gr_read_field "$rec" campaign) || camp=""
  pgid=$(_gr_read_field "$rec" pgid) || pgid=""
  sid=$(_gr_read_field "$rec" sid) || sid=""
  hold=$(_gr_read_field "$rec" holdpath) || hold=""
  uuid=$(_gr_read_field "$rec" uuid) || uuid=""

  # DEFECT 4: an index is only a NAME for a device. If this record was taken against a different
  # UUID than index $i now reports, the driver renumbered (reload / reset / MIG) and the record no
  # longer describes this device. Refuse the GPU; never silently honour or reclaim it.
  if [ "$_GR_TELEMETRY" = OK ] && [ -n "$uuid" ] && [ "$uuid" != "-" ]; then
    if [ "${_GR_IDX2UUID[$i]:-}" != "$uuid" ]; then
      echo "MAPCHANGED $pid $camp reserved-uuid-$uuid-now-${_GR_IDX2UUID[$i]:-absent}"; return
    fi
  fi

  if _gr_owner_alive "$pid" "$start"; then echo "HELD $pid $camp"; return; fi

  # ---- owner is gone. Is anything it started still running, or still on the GPU? ----
  #
  # FIRST and AUTHORITATIVE: the liveness lock. If the kernel will not give us the flock, a
  # process holding an inherited descriptor from this reservation is still alive -- owner, child,
  # grandchild or deeper, forked or execed, re-parented to init or not, CUDA context created or
  # not yet. No /proc inference and no leadership precondition is involved, so this arm fires for
  # the ordinary non-leader runner that _gr_live_in_pgid has to refuse (see its rc=2 at the
  # `[ "$want" != "$owner" ]` guard). This is the arm that closes the residual.
  local holdrc=2 holders=""
  if _gr_hold_path_ok "$hold" "$camp"; then
    _gr_hold_free "$hold"; holdrc=$?
    if [ "$holdrc" = 1 ]; then
      holders=$(_gr_holders "$hold") || holders="unreadable"
      echo "ORPHAN $pid $camp liveness-lock-held-by-pids-[${holders// /,}]"; return
    fi
  elif [ -n "$hold" ] && [ "$hold" != "-" ]; then
    # A record naming a hold path we refuse to accept is a record we cannot reason about.
    # Default-deny: keep the reservation rather than reclaim on a path we would not honour.
    echo "ORPHAN $pid $camp holdpath-rejected-'$hold'"; return
  fi

  # Pass OUR OWN PID (we exclude our own process tree) and the OWNER pid (pgid counts as evidence
  # of descent only if the owner led its group). rc=2 means "no pgid evidence available".
  # Retained as a SECOND line of defence for legacy records (holdpath=- , written before the
  # liveness lock existed) and for the gap where flock(1) is unavailable.
  local pgrc
  survivor=$(_gr_live_in_pgid "$pgid" "$$" "$pid"); pgrc=$?
  if [ "$pgrc" = 0 ] && [ -n "$survivor" ]; then
    echo "ORPHAN $pid $camp surviving-pid-$survivor-in-pgid-$pgid"; return
  fi
  if [ "$pgrc" = 2 ] && [ "$holdrc" != 0 ]; then
    # Only reachable for a record with NO usable liveness lock. With a lock present and free we
    # have POSITIVE proof that owner and descendants are gone, so group leadership is irrelevant.
    _gr_log "gpu$i: owner pid $pid did not lead its process group (pgid=$pgid, sid=${sid:--}) AND"
    _gr_log "        this record carries no usable liveness lock (holdpath='${hold:--}'), so its"
    _gr_log "        orphaned children cannot be identified. Falling back to the CUDA-context check"
    _gr_log "        and the ${ORPHAN_GRACE}s grace window. This is the LEGACY path: records written"
    _gr_log "        by gpu_reserve_hold now always carry a lock and do not reach it."
  fi
  if [ "$_GR_TELEMETRY" = OK ]; then
    local p u
    for p in ${_GR_APPS[$i]:-}; do
      u=$(_gr_proc_uid "$p")
      if [ -n "$u" ] && [ "$u" = "$MYUID" ]; then
        echo "ORPHAN $pid $camp cuda-context-pid-$p-still-on-gpu$i"; return
      fi
    done
  else
    # We cannot see the GPU. Never reclaim a reservation on the strength of telemetry we failed
    # to read -- that is defect 1 wearing a different hat.
    echo "ORPHAN $pid $camp telemetry-unknown-cannot-confirm-idle"; return
  fi
  # ---- grace window: a just-dead owner's child may not be in nvidia-smi yet ----
  mtime=$(stat -c %Y "$rec" 2>/dev/null) || mtime=""
  nowts=$(date +%s 2>/dev/null) || nowts=""
  if [ -n "$mtime" ] && [ -n "$nowts" ]; then
    age=$((nowts - mtime))
    if [ "$age" -ge 0 ] && [ "$age" -lt "$ORPHAN_GRACE" ]; then
      echo "ORPHAN $pid $camp within-${ORPHAN_GRACE}s-grace-age-${age}s"; return
    fi
  fi
  echo "STALE $pid $camp"
}

_gr_write_record() {
  local i="$1" camp="$2" pid="$3" hold="${4:-}" rec tmp start pgid sid uuid
  rec=$(_gr_record "$i"); tmp="$rec.tmp.$$"
  start=$(_gr_starttime "$pid") || start="-"
  pgid=$(_gr_pgid "$pid") || pgid="-"
  sid=$(_gr_sid "$pid") || sid="-"
  uuid=${_GR_IDX2UUID[$i]:--}
  [ -n "$hold" ] || hold="-"
  # DEFECT 2: the write and the publish are both checked. A full disk previously produced a
  # truncated or absent record that would read as FREE on the next pass.
  if ! {
    printf 'pid=%s\n'       "$pid"
    printf 'start=%s\n'     "${start:--}"
    printf 'pgid=%s\n'      "${pgid:--}"
    printf 'sid=%s\n'       "${sid:--}"
    printf 'holdpath=%s\n'  "$hold"
    printf 'campaign=%s\n'  "$camp"
    printf 'gpu=%s\n'       "$i"
    printf 'uuid=%s\n'      "$uuid"
    printf 'uid=%s\n'       "$MYUID"
    printf 'host=%s\n'      "$(hostname 2>/dev/null || echo unknown)"
    printf 'acquired=%s\n'  "$(_gr_now)"
  } > "$tmp"; then
    _gr_log "FATAL could not write the reservation record $tmp (disk full? permissions?)"
    rm -f "$tmp" 2>/dev/null
    return 10
  fi
  if ! mv -f "$tmp" "$rec"; then
    _gr_log "FATAL could not publish the reservation record to $rec"
    rm -f "$tmp" 2>/dev/null
    return 10
  fi
  return 0
}

# ---------------------------------------------------------------------------
# critical section: everything below runs with flock held on $MUTEX (FD 9).
# ---------------------------------------------------------------------------
_gr_with_mutex() {
  local rc
  mkdir -p "$LOCKDIR" || { _gr_log "FATAL cannot create $LOCKDIR"; return 10; }
  : > "$MUTEX" 2>/dev/null || true
  exec 9>>"$MUTEX" || { _gr_log "FATAL cannot open mutex $MUTEX"; return 10; }
  if ! flock -w "$MUTEX_WAIT" 9; then
    _gr_log "FATAL could not take reservation mutex within ${MUTEX_WAIT}s"
    exec 9>&-
    return 11
  fi
  "$@"
  rc=$?
  flock -u 9 || _gr_log "WARNING flock -u failed (rc=$?); closing the fd releases it anyway"
  exec 9>&-
  return "$rc"
}

# ---------------------------------------------------------------------------
# D-013: FREEZE GATE.
# A freeze marker that nothing reads is theatre. FREEZE_ACTIVE existed at the repo root from
# 12:43 on 2026-10-26, yet no *.sh or *.py in the project referenced the string. Reservation is
# the one choke point every GPU job must pass, so the gate lives here.
#   * refusal is LOUD: non-zero exit (4), and the marker's own text goes to stderr
#   * the check runs INSIDE the critical section, so a freeze written microseconds after a
#     would-be acquirer started still refuses it
#   * the override needs BOTH `ALLOW_FROZEN_RUN=1` AND the explicit `--allow-frozen` argument: a
#     profile can set an env var but cannot add an argument to someone else's command line
#   * in production the marker PATH is a constant, so GPU_RESERVE_FREEZE_FILE cannot fake a thaw
#   * every override is appended to $OVERRIDE_LOG
# `adopt` is deliberately exempt: it records a reservation for a job ALREADY running and starts no
# new work. `release`/`reap`/`status` are likewise unaffected.
# ---------------------------------------------------------------------------
_gr_freeze_gate() {
  local camp="$1" owner="$2" l
  [ -e "$FREEZE_FILE" ] || return 0        # not frozen

  if [ "$ALLOW_FROZEN_FLAG" != 1 ] || [ "${ALLOW_FROZEN_RUN:-}" != 1 ]; then
    _gr_log "REFUSED: the project is FROZEN. No reservation will be granted."
    _gr_log "marker: $FREEZE_FILE"
    _gr_log "---- marker contents ----"
    if [ -r "$FREEZE_FILE" ]; then
      while IFS= read -r l || [ -n "$l" ]; do _gr_log "| $l"; done < "$FREEZE_FILE"
    else
      _gr_log "| (marker exists but is not readable -- still refusing)"
    fi
    _gr_log "-------------------------"
    if [ "$ALLOW_FROZEN_FLAG" = 1 ] && [ "${ALLOW_FROZEN_RUN:-}" != 1 ]; then
      _gr_log "--allow-frozen was passed but ALLOW_FROZEN_RUN=1 is not set; BOTH are required."
    elif [ "${ALLOW_FROZEN_RUN:-}" = 1 ]; then
      _gr_log "ALLOW_FROZEN_RUN=1 is set in the environment but --allow-frozen was NOT passed."
      _gr_log "An inherited env var alone never lifts the freeze -- pass the argument explicitly."
    fi
    _gr_log "A bounded verification run by the implementation lead must pass BOTH:"
    _gr_log "  ALLOW_FROZEN_RUN=1 bash scripts/gpu_reserve.sh acquire <campaign> \$\$ --allow-frozen"
    return 4
  fi
  _GR_OVERRIDE_USED=1
  _gr_log "FREEZE OVERRIDE in effect for '$camp' (pid $owner) -- this is logged."
  return 0
}

_gr_log_override() {
  local camp="$1" owner="$2" gpu="$3"
  mkdir -p "$(dirname "$OVERRIDE_LOG")" 2>/dev/null || _gr_log "WARNING cannot create $(dirname "$OVERRIDE_LOG")"
  if ! printf '%s\toverride=ALLOW_FROZEN_RUN+--allow-frozen\tpid=%s\tcampaign=%s\tgpu=%s\tuuid=%s\tmarker=%s\tuid=%s\n' \
      "$(_gr_now)" "$owner" "$camp" "$gpu" "${_GR_IDX2UUID[$gpu]:--}" "$FREEZE_FILE" "$MYUID" >> "$OVERRIDE_LOG"; then
    _gr_log "WARNING could not append to the override log $OVERRIDE_LOG"
    return 0
  fi
  _gr_log "override recorded in $OVERRIDE_LOG"
}

_gr_do_acquire() {
  local camp="$1" owner="$2" hold="${3:-}" i st rest cls rc
  _GR_OVERRIDE_USED=0
  # Guard against the classic caller error: an owner pid that is already dead. $BASHPID is
  # RE-EVALUATED inside $( ), so `G=$(gpu_reserve.sh acquire c "$BASHPID")` records the transient
  # command-substitution subshell, which exits at once. ($$ is NOT re-evaluated and is correct.)
  if [ -z "$owner" ] || [ ! -d "/proc/$owner" ]; then
    _gr_log "FATAL owner pid '${owner:-<empty>}' is not a running process. Pass the campaign"
    _gr_log "      runner's own pid: use \$\$ (not \$BASHPID, which \$( ) re-evaluates)."
    return 2
  fi
  # ---- VERIFY the liveness lock rather than take the caller's word for it. ----------------
  # gpu_reserve_hold flocks $hold in the long-lived runner shell BEFORE calling acquire, so by the
  # time we get here the lock must already be HELD. We check that, because a record that merely
  # NAMES a hold path would promise a protection that does not exist: every later reclamation
  # decision trusts this field, so an unlocked path would read as "no descendant alive" for ever.
  # Refusing here is what makes the guarantee auditable instead of aspirational.
  local holdverified=0
  if [ -n "$hold" ] && [ "$hold" != "-" ]; then
    if ! _gr_hold_path_ok "$hold" "$camp"; then
      _gr_log "FATAL liveness-lock path '$hold' is not acceptable. It must be"
      _gr_log "      \$LOCKDIR/.hold.<ownerpid>.<campaign> inside $LOCKDIR, so that no caller can"
      _gr_log "      point the reclamation check at a file it controls."
      return 2
    fi
    _gr_hold_free "$hold"; local _hrc=$?
    case "$_hrc" in
      1) holdverified=1 ;;                     # BUSY = the caller really is holding it. Good.
      0) _gr_log "FATAL the liveness-lock file '$hold' is NOT LOCKED. acquire was called with a"
         _gr_log "      hold path whose flock the caller does not hold, so a surviving child could"
         _gr_log "      not be detected later. Use gpu_reserve_hold, which takes the lock first."
         return 2 ;;
      *) _gr_log "FATAL cannot test the liveness lock '$hold' (flock(1) missing or unusable)."
         _gr_log "      Refusing rather than recording a protection we cannot verify."
         return 2 ;;
    esac
  fi

  # D-013 freeze gate, inside the mutex-held critical section.
  _gr_freeze_gate "$camp" "$owner" || return $?

  # DEFECT 1: fail closed. No telemetry => no reservation.
  if ! _gr_probe; then
    _gr_log "REFUSED: cannot establish GPU state, so no GPU can be called free."
    _gr_log "reason: $_GR_TELEMETRY_WHY"
    _gr_log "This is deliberate: an unreadable nvidia-smi used to look like '0 MiB used, no"
    _gr_log "processes', i.e. FREE. Fix the driver/telemetry, then retry."
    return 5
  fi
  _gr_cvd_audit

  for i in $ALLOWED_GPUS; do
    st=$(_gr_reservation_state "$i")
    case "$st" in
      HELD*)
        rest=${st#HELD }
        _gr_log "gpu$i reserved by pid ${rest%% *} (${rest#* }) -- skipping"
        continue ;;
      MAPCHANGED*)
        rest=${st#MAPCHANGED }
        _gr_log "gpu$i DEVICE IDENTITY CHANGED since its reservation (${rest#* }). The device"
        _gr_log "        behind index $i is not the one that was reserved -- REFUSED. Investigate"
        _gr_log "        a driver reload / GPU reset / MIG change before reusing this index."
        continue ;;
      ORPHAN*)
        rest=${st#ORPHAN }
        _gr_log "gpu$i ORPHANED: owner pid ${rest%% *} is dead but work may still be running"
        _gr_log "        (${rest#* }). The reservation is KEPT and gpu$i is REFUSED -- a dead"
        _gr_log "        runner does not mean its python child released CUDA. Clear it"
        _gr_log "        deliberately with: $0 release <campaign> <pid>"
        continue ;;
      STALE*)
        rest=${st#STALE }
        _gr_log "gpu$i RECLAIMED stale reservation: owner pid ${rest%% *} (${rest#* }) is dead,"
        _gr_log "        no process of its group survives, no CUDA context of ours is on gpu$i,"
        _gr_log "        and the record is older than the ${ORPHAN_GRACE}s grace window."
        rm -f "$(_gr_record "$i")" || { _gr_log "could not remove the stale record; skipping gpu$i"; continue; } ;;
      FREE) ;;                        # the only state from which a claim may proceed
      # DEFAULT-DENY (reviewer finding, 2026-10-26). This case previously had no *) arm, so an
      # UNRECOGNISED reservation state fell straight through to the claim below. In a file whose
      # whole purpose is exclusion, "I do not know what this record means" must never mean "take
      # it". Any future state added to _gr_reservation_state is refused here until it is handled
      # explicitly, which is the safe direction for the omission.
      *)
        _gr_log "gpu$i: UNRECOGNISED reservation state '$st' -- REFUSED."
        _gr_log "        This is a default-deny arm: an unhandled state is never claimable. If a"
        _gr_log "        new state was added to _gr_reservation_state, handle it explicitly above."
        continue ;;
    esac
    cls=$(_gr_classify "$i")
    case "$cls" in
      FREE) ;;                        # the only classification from which a claim may proceed
      FOREIGN)       _gr_log "gpu$i carries a process that is not ours -- REFUSED"; continue ;;
      OURS_UNLOCKED) _gr_log "gpu$i has an unreserved CUDA context of ours (pre-lock-scheme job:${_GR_APPS[$i]:-}) -- REFUSED"; continue ;;
      UNACCOUNTED)   _gr_log "gpu$i holds ${_GR_MEM[$i]:-?} MiB we cannot attribute to our own pids -- REFUSED"; continue ;;
      UNKNOWN)       _gr_log "gpu$i state is UNKNOWN -- REFUSED (never treated as free)"; continue ;;
      # DEFAULT-DENY, same reasoning as above: only an explicit FREE proceeds.
      *)
        _gr_log "gpu$i: UNRECOGNISED classification '$cls' -- REFUSED (default-deny)."
        continue ;;
    esac
    # DEFECT 4: refuse to reserve an index whose device identity we do not know, so the record
    # always carries a UUID that later passes/fails the MAPCHANGED check above.
    if [ -z "${_GR_IDX2UUID[$i]:-}" ]; then
      _gr_log "gpu$i has no UUID in telemetry -- REFUSED (an index with no device identity cannot"
      _gr_log "        be re-validated later)"; continue
    fi
    _gr_write_record "$i" "$camp" "$owner" "$hold"; rc=$?
    if [ "$rc" != 0 ]; then
      _gr_log "gpu$i: reservation record could not be written -- NOT claiming it"
      return "$rc"
    fi
    # Report the supervision regime that actually applies, verified from /proc rather than assumed
    # from how the command was spelled. This is the line to read when judging whether a run was
    # protected: "liveness-lock" is the strong regime and does not depend on leadership at all.
    local _o_start _o_pgid _o_sid _regime
    _o_start=$(_gr_starttime "$owner") || _o_start="?"
    _o_pgid=$(_gr_pgid "$owner")       || _o_pgid="?"
    _o_sid=$(_gr_sid "$owner")         || _o_sid="?"
    if [ "$holdverified" = 1 ]; then _regime="liveness-lock (kernel-enforced; descendants covered)"
    elif [ "$_o_pgid" = "$owner" ];  then _regime="pgid-leader scan + ${ORPHAN_GRACE}s grace (no liveness lock)"
    else _regime="GRACE WINDOW ONLY (${ORPHAN_GRACE}s) -- owner leads neither its group nor a lock"
    fi
    _gr_log "gpu$i RESERVED for '$camp' by pid $owner (uuid ${_GR_IDX2UUID[$i]:--})"
    _gr_log "        owner identity: pid=$owner start=$_o_start pgid=$_o_pgid sid=$_o_sid"
    _gr_log "        group-leader=$([ "$_o_pgid" = "$owner" ] && echo yes || echo no) session-leader=$([ "$_o_sid" = "$owner" ] && echo yes || echo no)"
    _gr_log "        supervision: $_regime"
    [ "${_GR_OVERRIDE_USED:-0}" = 1 ] && _gr_log_override "$camp" "$owner" "$i"
    printf '%s\n' "$i"
    return 0
  done
  _gr_log "no allowed GPU available (allowed: $ALLOWED_GPUS)"
  return 3
}

_gr_do_release() {
  local camp="$1" owner="$2" i rec pid cp hp n=0
  for i in $ALLOWED_GPUS; do
    rec=$(_gr_record "$i"); [ -f "$rec" ] || continue
    pid=$(_gr_read_field "$rec" pid) || pid=""
    cp=$(_gr_read_field "$rec" campaign) || cp=""
    hp=$(_gr_read_field "$rec" holdpath) || hp=""
    if { [ -z "$owner" ] || [ "$pid" = "$owner" ]; } && { [ -z "$camp" ] || [ "$cp" = "$camp" ]; }; then
      # Warn, but still obey: an explicit `release` is the deliberate operator override, and the
      # refusal-under-live-descendants policy lives in gpu_reserve_drop where the fd can be closed.
      if _gr_hold_path_ok "$hp" "$cp"; then
        _gr_hold_free "$hp" || _gr_log "WARNING gpu$i's liveness lock is still HELD by pid(s)" \
          "[$(_gr_holders "$hp" || echo '?')] -- releasing anyway because release was asked for"
      fi
      if rm -f "$rec"; then _gr_log "gpu$i released (campaign '$cp', pid $pid)"; n=$((n+1))
      else _gr_log "WARNING could not remove $rec"; fi
      if _gr_hold_path_ok "$hp" "$cp" && _gr_hold_free "$hp"; then
        rm -f "$hp" 2>/dev/null || _gr_log "WARNING could not remove the hold file $hp"
      fi
    fi
  done
  [ "$n" -gt 0 ] || _gr_log "nothing to release for campaign='$camp' pid='$owner'"
  return 0
}

_gr_do_reap() {
  local i st rest n=0 o=0
  _gr_probe || _gr_log "WARNING telemetry is UNKNOWN; reap will refuse to reclaim anything"
  for i in $ALLOWED_GPUS; do
    st=$(_gr_reservation_state "$i")
    case "$st" in
      ORPHAN*) rest=${st#ORPHAN }
        _gr_log "gpu$i ORPHANED (not reaped): dead owner pid ${rest%% *}, ${rest#* }"; o=$((o+1)) ;;
      MAPCHANGED*) rest=${st#MAPCHANGED }
        _gr_log "gpu$i device identity changed (not reaped): ${rest#* }"; o=$((o+1)) ;;
      STALE*)  rest=${st#STALE }
        if rm -f "$(_gr_record "$i")"; then
          _gr_log "gpu$i reaped stale reservation of dead pid ${rest%% *} (${rest#* })"; n=$((n+1))
        fi ;;
    esac
  done
  _gr_log "reaped $n stale reservation(s); left $o orphaned reservation(s) in place"
  return 0
}

_gr_do_adopt() {
  local i="$1" camp="$2" pid="$3" st rc
  case "$i" in ''|*[!0-9]*) _gr_log "adopt: '$i' is not a GPU index"; return 2 ;; esac
  case " $ALLOWED_GPUS " in *" $i "*) ;; *) _gr_log "gpu$i is not in the allowed set ($ALLOWED_GPUS)"; return 2 ;; esac
  if ! _gr_probe; then
    _gr_log "adopt REFUSED: telemetry UNKNOWN ($_GR_TELEMETRY_WHY); cannot record a UUID for gpu$i"
    return 5
  fi
  st=$(_gr_reservation_state "$i")
  case "$st" in
    HELD*)   _gr_log "gpu$i already reserved: $st"; return 4 ;;
    ORPHAN*) _gr_log "gpu$i holds an ORPHANED reservation: $st -- release it explicitly first"; return 4 ;;
    MAPCHANGED*) _gr_log "gpu$i device identity changed since its reservation: $st -- refusing"; return 4 ;;
  esac
  [ -d "/proc/$pid" ] || { _gr_log "pid $pid is not running -- refusing to adopt"; return 2; }
  _gr_write_record "$i" "$camp" "$pid"; rc=$?
  [ "$rc" = 0 ] || return "$rc"
  _gr_log "gpu$i adopted for already-running '$camp' pid $pid"
  printf '%s\n' "$i"
}

_gr_do_status() {
  local i st cls
  _gr_probe || true
  printf 'mode=%s%s\n' "$_GR_MODE" "${_GR_MODE_REASON:+ ($_GR_MODE_REASON)}"
  printf 'telemetry=%s  %s\n' "$_GR_TELEMETRY" "$_GR_TELEMETRY_WHY"
  printf '%-5s %-12s %-14s %-26s %-9s %s\n' GPU RESERVATION CLASS CAMPAIGN/PID MEM_MiB UUID
  for i in $ALLOWED_GPUS; do
    st=$(_gr_reservation_state "$i")
    cls=$(_gr_classify "$i")
    case "$st" in
      FREE)  printf '%-5s %-12s %-14s %-26s %-9s %s\n' "$i" free "$cls" - "${_GR_MEM[$i]:-?}" "${_GR_IDX2UUID[$i]:-?}" ;;
      HELD*) printf '%-5s %-12s %-14s %-26s %-9s %s\n' "$i" held "$cls" \
               "$(_gr_read_field "$(_gr_record "$i")" campaign)/$(_gr_read_field "$(_gr_record "$i")" pid)" \
               "${_GR_MEM[$i]:-?}" "${_GR_IDX2UUID[$i]:-?}" ;;
      ORPHAN*) printf '%-5s %-12s %-14s %-26s %-9s %s\n' "$i" ORPHAN "$cls" \
               "$(_gr_read_field "$(_gr_record "$i")" campaign)/$(_gr_read_field "$(_gr_record "$i")" pid)" \
               "${_GR_MEM[$i]:-?}" "${_GR_IDX2UUID[$i]:-?}" ;;
      MAPCHANGED*) printf '%-5s %-12s %-14s %-26s %-9s %s\n' "$i" MAPCHANGED "$cls" \
               "$(_gr_read_field "$(_gr_record "$i")" campaign)/$(_gr_read_field "$(_gr_record "$i")" pid)" \
               "${_GR_MEM[$i]:-?}" "${_GR_IDX2UUID[$i]:-?}" ;;
      STALE*) printf '%-5s %-12s %-14s %-26s %-9s %s\n' "$i" STALE "$cls" \
               "$(_gr_read_field "$(_gr_record "$i")" campaign)/$(_gr_read_field "$(_gr_record "$i")" pid)" \
               "${_GR_MEM[$i]:-?}" "${_GR_IDX2UUID[$i]:-?}" ;;
    esac
  done
  printf 'allowed=%s  off-limits=everything else  lockdir=%s\n' "$ALLOWED_GPUS" "$LOCKDIR"
  if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    printf 'inherited CUDA_VISIBLE_DEVICES: <unset>\n'
  else
    printf "inherited CUDA_VISIBLE_DEVICES: '%s' (%s)\n" "$CUDA_VISIBLE_DEVICES" "$(_gr_cvd_classify "$CUDA_VISIBLE_DEVICES")"
  fi
  if [ -e "$FREEZE_FILE" ]; then
    printf 'FREEZE: ACTIVE (%s) -- acquire will REFUSE without ALLOW_FROZEN_RUN=1 + --allow-frozen\n' "$FREEZE_FILE"
  else
    printf 'FREEZE: none (%s absent)\n' "$FREEZE_FILE"
  fi
  [ "$_GR_TELEMETRY" = OK ] || return 5
  return 0
}

_gr_do_selftest_telemetry() {
  if _gr_probe; then printf 'TELEMETRY OK: %s\n' "$_GR_TELEMETRY_WHY"; return 0; fi
  printf 'TELEMETRY UNKNOWN: %s\n' "$_GR_TELEMETRY_WHY"
  return 5
}

# ===========================================================================
# sourceable helper: hold for the life of THIS shell, release via trap.
#
# DEFECT 6 -- errexit/pipefail safety. v1's body did `g=$(... acquire ...); rc=$?`. Under the
# caller's `set -e` an UNGUARDED call died ON THAT LINE, so rc=$? and every diagnostic after it
# never ran: the shell exited with the raw acquire status and no explanation. Verified: a runner
# with `set -euo pipefail` calling `gpu_reserve_hold c 5` while frozen exited 4 having printed
# nothing after "BEFORE". v2 shields the body, so the full refusal is always printed and a
# meaningful rc is always produced; $GPU_RESERVE_RC records it even on the errexit path.
#
# DEFECT 7 -- explicit argument validation. v1 took "$2" as wait_seconds and "$3" as the flag, so
# `gpu_reserve_hold "name" --allow-frozen` bound the FLAG to wait_seconds: the override silently
# did not register and `sleep --allow-frozen` would have been the first symptom. That is the bug
# the lead hit in run_v1_esd_alone_control.sh. v2 parses flags position-free and requires
# wait_seconds to be a positive integer.
# ===========================================================================
# Close the liveness-lock descriptor WITHOUT redirecting the shell.
#
# SECOND INSTANCE OF THE SAME BUG, and it is the reason this helper exists rather than an inline
# close. `exec {FD}>&- 2>/dev/null` applies that `2>/dev/null` to the SHELL, not to the exec, so
# every close site silenced the runner's stderr permanently. Measured: with it inline on the
# FAILURE path, gpu_reserve_require printed its full freeze diagnosis and then NOTHING -- no
# "ABORTING -- no GPU reservation", no explanation of rc=4 -- because stderr had been sent to
# /dev/null by the close two lines earlier. Two suite assertions caught it
# ("no abort explanation", "rc=4 not explained"). I had already fixed the OPEN site and missed
# the three CLOSE sites; a control that silences the diagnosis of its own failure is exactly the
# shape this repair cycle is supposed to remove.
# The fd is checked through /proc instead, so nothing needs to be suppressed.
#
# CLOSE, NEVER `flock -u`. THIS IS AN INVARIANT, not a style choice, and it is measured in
# scripts/test_gpu_reserve.sh (T35). An flock lives on the OPEN FILE DESCRIPTION, which fork and
# execve SHARE, so:
#     closing our descriptor  -> the lock survives while any inheritor still holds one  (correct)
#     flock -u on that fd     -> the lock is released FOR EVERY SHARER, instantly        (disaster)
# Measured both ways on this host:
#     parent CLOSES its fd, child alive  -> LOCK STILL HELD
#     parent runs flock -u, child alive  -> LOCK FREE          <- another job could take the GPU
# So no code path may ever `flock -u` the hold descriptor. The only `flock -u` in this file is on
# fd 9, the reservation MUTEX, whose lifetime is deliberately NOT shared with anything.
_gr_close_hold_fd() {
  [ -n "${_GR_HOLD_FD:-}" ] || return 0
  if [ -e "/proc/$$/fd/$_GR_HOLD_FD" ]; then
    eval "exec ${_GR_HOLD_FD}>&-"
  fi
  _GR_HOLD_FD=""
  return 0
}

gpu_reserve_hold() {
  local _gr_rc=0 camp="" wait_s="" flag=0 a g
  # ---- REFUSE a second reservation in the same shell. ---------------------------------------
  # The header has warned since v2 that two acquires in one shell leak one of them, because the
  # second EXIT trap replaces the first. It is no longer only a warning: with a liveness lock the
  # second call would also overwrite _GR_HOLD_FD, leaving the FIRST descriptor open for the life
  # of the shell -- so the first GPU would stay locked and unreclaimable even after the runner
  # exited. run_r1_imma_official.sh does exactly this (stage 2 prep, then stage 4 smoke, with no
  # drop in between), so this is a live call shape and not a hypothetical. Refusing with a pointer
  # to gpu_reserve_drop is the only safe answer.
  if [ -n "${_GR_HOLD_FD:-}" ]; then
    _gr_log "REFUSED: this shell already holds a GPU reservation (gpu ${GPU_RESERVED:-?}, lock"
    _gr_log "         ${_GR_HOLD_PATH:-?}). A second reservation in one shell would replace the"
    _gr_log "         release trap and orphan the first liveness-lock descriptor, pinning that GPU"
    _gr_log "         for the life of the shell. Release the first one explicitly:"
    _gr_log "           gpu_reserve_drop '<campaign>'   # then acquire again"
    GPU_RESERVE_RC=2; export GPU_RESERVE_RC
    return 2
  fi
  _GR_HOLD_PATH=""
  # Save the caller's errexit/pipefail WITHOUT a command substitution: `shopt -po errexit pipefail`
  # exits non-zero whenever either option is off, and inside $( ) it reports the subshell's own
  # (already-cleared) errexit, so capturing it both lied and tripped the caller's set -e.
  local _gr_had_e=0 _gr_had_pf=0
  case $- in *e*) _gr_had_e=1 ;; esac
  if shopt -qo pipefail 2>/dev/null; then _gr_had_pf=1; fi
  set +e
  set +o pipefail 2>/dev/null || true

  # ---- argument parsing / validation (defect 7) ----
  local flag_slot=0
  for a in "$@"; do
    case "$a" in
      --allow-frozen)
        flag=1
        # The lead's bug: `gpu_reserve_hold "name" --allow-frozen`. v1 bound this flag to
        # wait_seconds, so the override silently never reached the freeze gate. v2 parses flags
        # position-free so the INTENT is honoured -- but says so out loud, because a silently
        # accepted odd call shape is how the original defect hid.
        [ -n "$camp" ] && [ -z "$wait_s" ] && flag_slot=1 ;;
      --*|-*)
        _gr_log "USAGE ERROR: unknown option '$a'"
        _gr_log "  gpu_reserve_hold <campaign> [wait_seconds] [--allow-frozen]"
        _gr_rc=2; break ;;
      *)
        if   [ -z "$camp" ];   then camp="$a"
        elif [ -z "$wait_s" ]; then wait_s="$a"
        else
          _gr_log "USAGE ERROR: too many positional arguments (unexpected '$a')"
          _gr_log "  gpu_reserve_hold <campaign> [wait_seconds] [--allow-frozen]"
          _gr_rc=2; break
        fi ;;
    esac
  done

  if [ "$_gr_rc" = 0 ] && [ -z "$camp" ]; then
    _gr_log "USAGE ERROR: <campaign> is required."
    _gr_log "  gpu_reserve_hold <campaign> [wait_seconds] [--allow-frozen]"
    _gr_rc=2
  fi
  if [ "$_gr_rc" = 0 ]; then
    case "$camp" in
      *[!A-Za-z0-9._-]*)
        _gr_log "USAGE ERROR: campaign '$camp' must match [A-Za-z0-9._-]+ (it is written into the"
        _gr_log "             reservation record and the override log)."
        _gr_rc=2 ;;
    esac
  fi
  if [ "$_gr_rc" = 0 ] && [ -n "$wait_s" ]; then
    case "$wait_s" in
      0|*[!0-9]*)
        _gr_log "USAGE ERROR: wait_seconds must be a POSITIVE INTEGER, got '$wait_s'."
        _gr_log "  gpu_reserve_hold <campaign> [wait_seconds] [--allow-frozen]"
        case "$wait_s" in -*|--*)
          _gr_log "  '$wait_s' looks like a FLAG supplied in the wait-interval position. Flags are"
          _gr_log "  position-free, so write:  gpu_reserve_hold \"$camp\" $wait_s"
          _gr_log "  or, with an explicit interval:  gpu_reserve_hold \"$camp\" 120 $wait_s" ;;
        esac
        _gr_log "  This is refused rather than coerced: v1 bound a flag here silently, and a"
        _gr_log "  --allow-frozen that lands in this slot never reaches the freeze gate."
        _gr_rc=2 ;;
    esac
  fi
  if [ "$_gr_rc" = 0 ] && [ "$flag_slot" = 1 ]; then
    _gr_log "NOTE: the --allow-frozen flag was supplied in the wait-interval position"
    _gr_log "      (gpu_reserve_hold '$camp' --allow-frozen). Flags are position-free in v2, so it"
    _gr_log "      IS honoured -- v1 bound it to wait_seconds and the override silently never"
    _gr_log "      registered. The unambiguous form is:"
    _gr_log "        gpu_reserve_hold '$camp' 120 --allow-frozen"
  fi
  [ -n "$wait_s" ] || wait_s=120

  # A configuration refused at source time can never be satisfied by acquiring.
  if [ "$_gr_rc" = 0 ] && [ "${_GR_INIT_RC:-0}" != 0 ]; then
    _gr_log "gpu_reserve_hold: the reservation configuration was refused when this file was"
    _gr_log "        sourced (rc=${_GR_INIT_RC}). Refusing without attempting to acquire."
    _gr_rc="$_GR_INIT_RC"
  fi

  # ---- take the LIVENESS LOCK in THIS shell, before acquiring. ------------------------------
  # Order is the whole point. The lock is held from before the reservation record exists until
  # after the last descendant exits, so there is no window in which a published record promises a
  # protection that is not yet in force. The fd lives in this long-lived shell, so every process
  # the runner goes on to spawn -- child, grandchild, execed python -- inherits it automatically
  # and the kernel keeps the GPU reserved for exactly as long as any of them lives.
  #
  # GPU_RESERVE_HOLD_FD is deliberately NOT exported as a number children are expected to act on;
  # it is recorded only so gpu_reserve_drop can close the right descriptor.
  if [ "$_gr_rc" = 0 ]; then
    _GR_HOLD_PATH="$LOCKDIR/.hold.$$.$camp"
    # NOTE, and it cost a full test run to find: `exec {FD}>>file 2>/dev/null` does NOT silence
    # just this command. `exec` with redirections and no command applies them to the SHELL, so the
    # `2>/dev/null` would redirect the runner's stderr PERMANENTLY -- swallowing the CVD audit, the
    # RESERVED line, every freeze refusal and every diagnosis for the rest of the run. Measured:
    # 12 assertions in scripts/test_gpu_reserve.sh went from pass to fail on exactly that, all of
    # them "the explanation was not printed". So: no redirection on the exec, and the file is
    # pre-created with a plain command whose failure can be caught normally.
    if ! mkdir -p "$LOCKDIR" 2>/dev/null; then
      _gr_log "FATAL cannot create $LOCKDIR for the liveness lock"
      _gr_rc=10
    elif ! : >>"$_GR_HOLD_PATH" 2>/dev/null; then
      _gr_log "FATAL cannot create the liveness-lock file $_GR_HOLD_PATH (permissions? disk full?)"
      _gr_rc=10
    elif ! exec {_GR_HOLD_FD}>>"$_GR_HOLD_PATH"; then
      _gr_log "FATAL cannot open the liveness-lock file $_GR_HOLD_PATH"
      _gr_rc=10
    elif ! flock -x -w 30 "$_GR_HOLD_FD"; then
      # Our own pid + campaign names this file, so a conflict means a previous run with this very
      # pid still has live descendants. Refusing is correct: we must not reuse its lock.
      _gr_log "FATAL could not take the liveness lock $_GR_HOLD_PATH within 30s."
      _gr_log "      Another process still holds it -- see: $0 holders <index>"
      _gr_close_hold_fd
      _gr_rc=11
    fi
  fi

  if [ "$_gr_rc" = 0 ]; then
    while :; do
      # $$ (not $BASHPID) is correct here: $$ is not re-evaluated inside $( ).
      if [ "$flag" = 1 ]; then
        g=$("${BASH_SOURCE[0]}" acquire "$camp" "$$" "$_GR_HOLD_PATH" --allow-frozen)
      else
        g=$("${BASH_SOURCE[0]}" acquire "$camp" "$$" "$_GR_HOLD_PATH")
      fi
      _gr_rc=$?
      [ "$_gr_rc" = 0 ] && break
      # Only "no GPU available right now" (3) is worth waiting out. A freeze refusal (4), unknown
      # telemetry (5), a prohibited GPU set (6), an unusable allowlist (7), a bad owner pid (2) or
      # a mutex failure (10/11) will never resolve by sleeping. Keeping 7 out of 3 is the point of
      # having it: an empty allowlist used to surface as the retryable 3 and loop forever.
      if [ "$_gr_rc" = 3 ]; then
        _gr_log "gpu_reserve_hold: every allowed GPU is busy; retrying in ${wait_s}s"
        sleep "$wait_s" || { _gr_log "sleep interrupted -- giving up"; _gr_rc=3; break; }
        continue
      fi
      _gr_log "gpu_reserve_hold: acquire failed with rc=$_gr_rc -- not retrying (see above)"
      break
    done
  fi

  if [ "$_gr_rc" = 0 ]; then
    GPU_RESERVED="$g"
    GPU_RESERVED_UUID=$("${BASH_SOURCE[0]}" _uuid_of "$g" 2>/dev/null) || GPU_RESERVED_UUID=""
    export GPU_RESERVED GPU_RESERVED_UUID
    # DEFECT 4: set CUDA_VISIBLE_DEVICES here, so a runner cannot inherit a stale value by
    # forgetting to export one. "index" is the default because it is what every existing runner
    # and every venv in this project has actually been run with; "uuid" is stronger (immune to
    # renumbering) but has never been verified on a real GPU here, so it is opt-in.
    case "$CVD_STYLE" in
      uuid) if [ -n "$GPU_RESERVED_UUID" ]; then CUDA_VISIBLE_DEVICES="$GPU_RESERVED_UUID"
            else _gr_log "WARNING GPU_RESERVE_CVD_STYLE=uuid but no UUID was recorded; using the index"
                 CUDA_VISIBLE_DEVICES="$GPU_RESERVED"; fi ;;
      *)    CUDA_VISIBLE_DEVICES="$GPU_RESERVED" ;;
    esac
    export CUDA_VISIBLE_DEVICES
    # The teardown goes through gpu_reserve_drop (a function IN THIS SHELL), not straight to the
    # external `release`. It has to: only this shell can close the liveness-lock descriptor, and
    # until that descriptor is closed we cannot test whether anything ELSE still holds it.
    #
    # v2's trap ran `release` unconditionally, which on SIGTERM deleted the record while the
    # python child was still training -- handing the GPU to the next acquirer underneath live
    # work, i.e. the very failure this file exists to prevent, reintroduced on the interrupt path.
    _GR_HOLD_CAMP="$camp"
    trap 'gpu_reserve_drop "$_GR_HOLD_CAMP" >/dev/null' EXIT
    # Signals must also TERMINATE, not just run a handler and fall through to the next line. v2
    # trapped INT/TERM/HUP to the same release command, so a SIGINT ran the handler and then
    # CARRIED ON executing the runner with its reservation deleted. Exiting 128+signo both stops
    # the runner and gives the conventional non-zero status that a caller can detect (test g).
    trap 'gpu_reserve_drop "$_GR_HOLD_CAMP" >/dev/null; trap - EXIT; exit 130' INT
    trap 'gpu_reserve_drop "$_GR_HOLD_CAMP" >/dev/null; trap - EXIT; exit 143' TERM
    trap 'gpu_reserve_drop "$_GR_HOLD_CAMP" >/dev/null; trap - EXIT; exit 129' HUP
    printf '%s\n' "$GPU_RESERVED"
  else
    # Never leave a locked descriptor or a stray hold file behind on a failed acquisition: a hold
    # file that stays locked by this shell would make the NEXT attempt from the same pid refuse.
    _gr_close_hold_fd
    [ -n "${_GR_HOLD_PATH:-}" ] && rm -f "$_GR_HOLD_PATH" 2>/dev/null
    _GR_HOLD_PATH=""
  fi

  GPU_RESERVE_RC="$_gr_rc"; export GPU_RESERVE_RC
  # Restore the caller's shell options exactly as they were, then return. A non-zero return under
  # a restored errexit terminates the caller -- which is what `set -e` means and is why the API
  # contract says to guard the call or use gpu_reserve_require. The point of the shield is that by
  # now the FULL refusal has already been printed and $GPU_RESERVE_RC is set, which is precisely
  # what v1 failed to do: it died inside the function at `g=$(... acquire ...)`.
  if [ "$_gr_had_pf" = 1 ]; then set -o pipefail 2>/dev/null || true; fi
  if [ "$_gr_had_e" = 1 ]; then set -e; fi
  return "$_gr_rc"
}

# Abort-for-me variant: safe to call bare under `set -euo pipefail`, because it either succeeds or
# exits deliberately with the acquisition rc after printing a full explanation.
gpu_reserve_require() {
  # `|| rc=$?` is deliberate: in a || list the caller's errexit is suppressed, so this function
  # reaches its own diagnosis instead of being killed the instant gpu_reserve_hold returns non-zero.
  local rc=0
  gpu_reserve_hold "$@" || rc=$?
  [ "$rc" = 0 ] && return 0
  _gr_log "gpu_reserve_require: ABORTING -- no GPU reservation (rc=$rc)."
  case "$rc" in
    2) _gr_log "  rc=2  usage error or a dead owner pid. Fix the call, do not retry." ;;
    3) _gr_log "  rc=3  every allowed GPU is busy and the wait was interrupted." ;;
    4) _gr_log "  rc=4  the project is FROZEN. Only the lead may run, with ALLOW_FROZEN_RUN=1 and --allow-frozen." ;;
    5) _gr_log "  rc=5  GPU telemetry is UNKNOWN. Fix nvidia-smi; a GPU is never assumed free." ;;
    6) _gr_log "  rc=6  a prohibited GPU set was requested. GPUs 2-3 are off limits." ;;
    7) _gr_log "  rc=7  the GPU allowlist is unusable (GPU_RESERVE_GPUS empty/malformed). Fix the"
       _gr_log "        configuration; this will never resolve by waiting." ;;
   10|11) _gr_log "  rc=$rc lockdir/mutex failure. Check $LOCKDIR." ;;
  esac
  exit "$rc"
}

# Teardown. Correct order, and the order is load-bearing:
#   1. close OUR liveness-lock descriptor -- while it is open every lock test says BUSY, so we
#      cannot distinguish "a child is still running" from "I am still running";
#   2. ask the kernel whether anything ELSE still holds the lock;
#   3. only if nothing does, delete the reservation record and the hold file.
# If descendants ARE still alive we deliberately leave the record in place. It then reads as
# ORPHAN to every other acquirer (owner dead, lock held) and is reclaimed automatically once the
# last descendant exits -- no grace window, no timer, no manual step. Normal completion and an
# interrupted run therefore take the SAME path and differ only in what the kernel reports.
gpu_reserve_drop() {
  local camp="${1:-${_GR_HOLD_CAMP:-}}" hold="${_GR_HOLD_PATH:-}" rc=0 holders=""
  _gr_close_hold_fd
  if [ -n "$hold" ]; then
    _gr_hold_free "$hold"; rc=$?
    if [ "$rc" = 1 ]; then
      holders=$(_gr_holders "$hold") || holders="<not readable>"
      _gr_log "NOT releasing the reservation for '$camp': the liveness lock $hold is STILL HELD"
      _gr_log "        by pid(s) [$holders] -- work this runner started is still alive. The record"
      _gr_log "        is KEPT so no other job can be given this GPU underneath it; it reclaims"
      _gr_log "        itself automatically when the last of those processes exits."
      trap - EXIT INT TERM HUP
      return 0
    fi
  fi
  "${BASH_SOURCE[0]}" release "$camp" "$$"
  [ -n "$hold" ] && rm -f "$hold" 2>/dev/null
  _GR_HOLD_PATH=""
  trap - EXIT INT TERM HUP
}

# ---------------------------------------------------------------------------
# dispatch (skipped when sourced)
# ---------------------------------------------------------------------------
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  cmd="${1:-status}"; shift 2>/dev/null || true
  # extract the explicit freeze override argument from anywhere in the remaining args
  _args=()
  for _a in "$@"; do
    case "$_a" in
      --allow-frozen) ALLOW_FROZEN_FLAG=1 ;;
      --*) _gr_log "unknown option '$_a'"; exit 2 ;;
      *) _args+=("$_a") ;;
    esac
  done
  set -- ${_args[@]+"${_args[@]}"}

  _gr_mode_init || exit $?

  case "$cmd" in
    acquire) _gr_with_mutex _gr_do_acquire "${1:?acquire <campaign> [owner_pid] [hold_path]}" "${2:-$PPID}" "${3:-}" ;;
    release) _gr_with_mutex _gr_do_release "${1:-}" "${2:-$PPID}" ;;
    holders)
      _i="${1:?holders <index>}"
      _h=$(_gr_read_field "$(_gr_record "$_i")" holdpath) || _h=""
      if [ -z "$_h" ] || [ "$_h" = "-" ]; then printf 'gpu%s: no liveness lock recorded\n' "$_i"; exit 1; fi
      _gr_hold_free "$_h"; _hr=$?
      printf 'gpu%s holdpath=%s\n' "$_i" "$_h"
      case "$_hr" in
        0) printf 'lock=FREE (owner and all descendants have exited)\n' ;;
        1) printf 'lock=HELD by pids: %s\n' "$(_gr_holders "$_h" || echo '<not readable>')" ;;
        *) printf 'lock=UNTESTABLE\n' ;;
      esac ;;
    reap)    _gr_with_mutex _gr_do_reap ;;
    adopt)   _gr_with_mutex _gr_do_adopt "${1:?adopt <index> <campaign> <pid>}" "${2:?}" "${3:?}" ;;
    status)  _gr_with_mutex _gr_do_status ;;
    selftest-telemetry) _gr_do_selftest_telemetry ;;
    # internal: print the UUID recorded for an index, for gpu_reserve_hold's export
    _uuid_of)
      _i="${1:?}"
      _u=$(_gr_read_field "$(_gr_record "$_i")" uuid) || _u=""
      [ -n "$_u" ] && [ "$_u" != "-" ] || exit 1
      printf '%s\n' "$_u" ;;
    *) _gr_log "unknown subcommand '$cmd'; expected acquire|release|reap|adopt|status|holders|selftest-telemetry"; exit 2 ;;
  esac
else
  # sourced: initialise the mode so LOCKDIR/ALLOWED_GPUS/FREEZE_FILE exist for the helpers.
  # A refused configuration is REMEMBERED, so gpu_reserve_hold refuses immediately with the same
  # code instead of proceeding with the defaults that were assigned before validation ran.
  _GR_INIT_RC=0
  _gr_mode_init || _GR_INIT_RC=$?
  if [ "$_GR_INIT_RC" != 0 ]; then
    _gr_log "WARNING gpu_reserve.sh sourced with a REFUSED configuration (rc=$_GR_INIT_RC);"
    _gr_log "        gpu_reserve_hold will refuse with that code. No reservation is possible."
  fi
fi
