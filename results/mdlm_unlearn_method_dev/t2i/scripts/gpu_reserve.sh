#!/bin/bash
# scripts/gpu_reserve.sh -- EXCLUSIVE per-GPU reservation held for an ENTIRE campaign.
#
# WHY THIS EXISTS
# ---------------
# scripts/campaign_free.sh is check-then-select: it READS GPU state and RETURNS an index while
# holding no lock. Two runners that read before either writes both get the same index. Observed
# live on 2026-10-26: run_r2_meta_official.sh (12:39:35), run_r1c_eval.sh (12:39:29) and
# run_r4d_upstream_defaults.sh (12:39:32) each independently ran campaign_free.sh, each was told
# "0", and all three ran on GPU 0 (3 pids, 15.3 GB) while GPU 1 sat idle at 34 MiB.
#
# campaign_free.sh has three independent defects, all of which this script avoids:
#   (a) read-then-return with no lock held -> the race above.
#   (b) it reads /proc/<pid>/environ for CUDA_VISIBLE_DEVICES. That is a snapshot taken at
#       execve(2) and is NEVER updated. Every caller does `export CUDA_VISIBLE_DEVICES=$G`
#       AFTER startup, so clause (a) of campaign_free.sh can never match any of its own callers:
#       it is dead code. Verified empirically.
#   (c) a memory threshold cannot distinguish "free" from "between campaign cells". A campaign
#       dips to ~0-1.4 GiB while loading the next cell, which reads as free under any threshold
#       low enough to be useful.
#
# THE FIX: an explicit reservation, not an inference from observed state. A reservation is a
# record on disk, created inside a flock-protected critical section, naming the owner process.
# It is therefore immune to (a) by construction, does not consult /proc environ at all (b), and
# survives inter-cell memory dips because it is not a memory reading (c).
#
# GUARANTEES
#   * Mutual exclusion: the scan-and-claim transaction runs under flock(2) on a mutex file, so
#     exactly one process can claim a given GPU, no matter how many race.
#   * Crash safety: a reservation is validated by (owner pid alive) AND (owner pid start time
#     unchanged). A crashed owner's reservation is therefore detected as stale and reclaimed by
#     the next acquirer. Recording the start time defeats PID reuse: a recycled PID does not
#     resurrect a dead reservation.
#   * Normal release: `release` on the happy path, plus an EXIT/INT/TERM/HUP trap installed by
#     the sourceable helper, so a reservation is dropped promptly rather than waiting to go stale.
#   * Other users are never displaced: GPUs 2-3 are not in the allowed set at all, and an allowed
#     GPU carrying any compute process we cannot prove is ours is refused.
#   * Pre-lock-scheme jobs of ours are respected: an allowed GPU carrying a CUDA context owned by
#     our uid but covered by no reservation is refused (and reported as OURS_UNLOCKED), so this
#     script will not land on top of a legacy campaign that predates reservations.
#
# USAGE
#   Preferred (runner holds the reservation and releases it automatically):
#       source scripts/gpu_reserve.sh
#       gpu_reserve_hold "r4d_upstream_defaults"   # sets $GPU_RESERVED, installs release trap
#       export CUDA_VISIBLE_DEVICES=$GPU_RESERVED
#
#   Drop-in for the existing `G=$(bash scripts/campaign_free.sh|head -1)` call style -- the owner
#   pid MUST be passed explicitly, because inside $( ) this script's parent is a transient
#   subshell that exits immediately:
#       while :; do G=$(bash scripts/gpu_reserve.sh acquire my_campaign $$) && break; sleep 120; done
#       trap 'bash scripts/gpu_reserve.sh release my_campaign $$' EXIT INT TERM HUP
#       export CUDA_VISIBLE_DEVICES=$G
#
#   Subcommands:
#       acquire <campaign> [owner_pid]   print reserved index, exit 0; exit 3 if none available
#       release <campaign> [owner_pid]   release reservations owned by that pid/campaign
#       status                           human-readable table of all allowed GPUs
#       adopt <index> <campaign> <pid>   retroactively reserve a GPU for an already-running job
#       reap                             reclaim stale reservations only, claim nothing
#
# ENV OVERRIDES (the test harness uses these; production needs none)
#   GPU_RESERVE_LOCKDIR   reservation directory          (default $PROJECT/.gpulocks)
#   GPU_RESERVE_GPUS      allowed indices, space-sep     (default "0 1" -- 2 and 3 are OFF LIMITS)
#   GPU_RESERVE_NVSMI     nvidia-smi command             (default "nvidia-smi")
#   GPU_RESERVE_FOREIGN_MIB  memory on an allowed GPU that we cannot attribute to our own live
#                            pids, above which the GPU is refused as possibly another user's
#                            (default 512)
#   GPU_RESERVE_UID       uid considered "ours"          (default: id -u)
#   GPU_RESERVE_WAIT      seconds to wait for the mutex  (default 30)

set -u

PROJECT="${PROJECT:-/data/bijaypandey/t2i_unlearn}"
LOCKDIR="${GPU_RESERVE_LOCKDIR:-$PROJECT/.gpulocks}"
ALLOWED_GPUS="${GPU_RESERVE_GPUS:-0 1}"
NVSMI="${GPU_RESERVE_NVSMI:-nvidia-smi}"
FOREIGN_MIB="${GPU_RESERVE_FOREIGN_MIB:-512}"
MYUID="${GPU_RESERVE_UID:-$(id -u)}"
MUTEX_WAIT="${GPU_RESERVE_WAIT:-30}"
MUTEX="$LOCKDIR/.reserve.mutex"
FREEZE_FILE="${GPU_RESERVE_FREEZE_FILE:-$PROJECT/FREEZE_ACTIVE}"
OVERRIDE_LOG="${GPU_RESERVE_OVERRIDE_LOG:-$PROJECT/logs/frozen_override.log}"
ALLOW_FROZEN_FLAG=0   # set only by an explicit --allow-frozen argument

_gr_log() { printf '[gpu_reserve] %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# process identity: pid + start time, so a recycled PID cannot inherit a
# dead reservation. Field 22 of /proc/<pid>/stat is starttime in clock ticks.
# comm (field 2) may contain spaces and parentheses, so cut after the LAST ')'.
# ---------------------------------------------------------------------------
_gr_starttime() {
  local pid="$1" st rest
  st=$(cat "/proc/$pid/stat" 2>/dev/null) || return 1
  rest=${st##*\) }                       # drop "pid (comm) "
  # rest now begins at field 3 (state); starttime is field 22 => 20th token here
  printf '%s\n' "$rest" | awk '{print $20}'
}

_gr_proc_uid() { stat -c %u "/proc/$1" 2>/dev/null; }

# is this pid alive AND the same process that took the reservation?
_gr_owner_alive() {
  local pid="$1" want="$2" now
  [ -n "$pid" ] || return 1
  [ -d "/proc/$pid" ] || return 1
  now=$(_gr_starttime "$pid") || return 1
  # no recorded start time (legacy/adopted record): fall back to liveness alone
  if [ -z "$want" ] || [ "$want" = "-" ]; then return 0; fi
  [ "$now" = "$want" ]
}

_gr_record() { printf '%s\n' "$LOCKDIR/gpu$1.owner"; }

_gr_read_field() { sed -n "s/^$2=//p" "$1" 2>/dev/null | head -1; }

# ---------------------------------------------------------------------------
# GPU facts, via exactly three nvidia-smi invocation forms so a test shim can
# emulate them. Populates _GR_MEM[idx] and _GR_APPS[idx]="pid pid ...".
# ---------------------------------------------------------------------------
declare -A _GR_MEM=()
declare -A _GR_APPS=()
declare -A _GR_UUID2IDX=()

_gr_probe() {
  local line idx uuid mem pid
  _GR_MEM=(); _GR_APPS=(); _GR_UUID2IDX=()
  while IFS=, read -r idx uuid; do
    idx=$(printf '%s' "$idx" | tr -d ' '); uuid=$(printf '%s' "$uuid" | tr -d ' ')
    [ -n "$idx" ] && _GR_UUID2IDX[$uuid]=$idx
  done < <($NVSMI --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)

  while IFS=, read -r idx mem; do
    idx=$(printf '%s' "$idx" | tr -d ' '); mem=$(printf '%s' "$mem" | tr -d ' ')
    [ -n "$idx" ] && _GR_MEM[$idx]=${mem:-0}
  done < <($NVSMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)

  while IFS=, read -r pid uuid; do
    pid=$(printf '%s' "$pid" | tr -d ' '); uuid=$(printf '%s' "$uuid" | tr -d ' ')
    [ -n "$pid" ] || continue
    idx=${_GR_UUID2IDX[$uuid]:-}
    [ -n "$idx" ] || continue
    _GR_APPS[$idx]="${_GR_APPS[$idx]:-} $pid"
  done < <($NVSMI --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)
}

# Classify an allowed GPU. Echoes one of:
#   FREE | FOREIGN | OURS_UNLOCKED | UNACCOUNTED
# Called only for GPUs that carry no live reservation.
_gr_classify() {
  local i="$1" pid uid ours=0 foreign=0 accounted=0 mem
  for pid in ${_GR_APPS[$i]:-}; do
    uid=$(_gr_proc_uid "$pid")
    if [ -z "$uid" ]; then
      # nvidia-smi still lists this pid as holding a CUDA context, yet we cannot
      # see it in /proc. That means hidepid: it is another user's. Refuse --
      # never displace. (Deliberately conservative: a GPU we cannot fully
      # account for is never selected.)
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
  mem=${_GR_MEM[$i]:-0}
  case "$mem" in ''|*[!0-9]*) mem=0 ;; esac
  if [ "$accounted" = 0 ] && [ "$mem" -ge "$FOREIGN_MIB" ]; then echo UNACCOUNTED; return; fi
  echo FREE
}

# ---------------------------------------------------------------------------
# reservation state for one GPU (no locking; callers hold the mutex)
#   echoes: "FREE" | "HELD <pid> <campaign>" | "STALE <pid> <campaign>"
# ---------------------------------------------------------------------------
_gr_reservation_state() {
  local i="$1" rec pid start camp
  rec=$(_gr_record "$i")
  [ -f "$rec" ] || { echo FREE; return; }
  pid=$(_gr_read_field "$rec" pid)
  start=$(_gr_read_field "$rec" start)
  camp=$(_gr_read_field "$rec" campaign)
  if _gr_owner_alive "$pid" "$start"; then echo "HELD $pid $camp"; else echo "STALE $pid $camp"; fi
}

_gr_write_record() {
  local i="$1" camp="$2" pid="$3" rec tmp start
  rec=$(_gr_record "$i"); tmp="$rec.tmp.$$"
  start=$(_gr_starttime "$pid" 2>/dev/null || echo -)
  {
    echo "pid=$pid"
    echo "start=${start:--}"
    echo "campaign=$camp"
    echo "gpu=$i"
    echo "uid=$MYUID"
    echo "host=$(hostname 2>/dev/null || echo unknown)"
    echo "acquired=$(date -Is 2>/dev/null)"
  } > "$tmp"
  mv -f "$tmp" "$rec"            # atomic publish
}

# ---------------------------------------------------------------------------
# critical section: everything below runs with flock held on $MUTEX (FD 9).
# ---------------------------------------------------------------------------
_gr_with_mutex() {
  mkdir -p "$LOCKDIR" || { _gr_log "FATAL cannot create $LOCKDIR"; return 10; }
  : > "$MUTEX" 2>/dev/null || true
  exec 9>>"$MUTEX" || { _gr_log "FATAL cannot open mutex $MUTEX"; return 10; }
  if ! flock -w "$MUTEX_WAIT" 9; then
    _gr_log "FATAL could not take reservation mutex within ${MUTEX_WAIT}s"
    return 11
  fi
  "$@"
  local rc=$?
  flock -u 9
  exec 9>&-
  return $rc
}

# ---------------------------------------------------------------------------
# D-013: FREEZE GATE.
# A freeze marker that nothing reads is theatre. FREEZE_ACTIVE existed at the repo root from
# 12:43 on 2026-10-26, yet no *.sh or *.py in the project referenced the string -- the freeze was
# held only by the lead having SIGTERM'd the runner loops. Any runner started by hand, by a stale
# cron, or by a resumed waiter loop would have taken a GPU and proceeded.
#
# Reservation is the one choke point every GPU job must pass, so the gate lives here.
#   * refusal is LOUD: non-zero exit, and the marker's own text goes to stderr
#   * the check runs INSIDE the critical section (see _gr_do_acquire, called under the mutex), so
#     a freeze written microseconds after a would-be acquirer started still refuses it. Checking
#     before taking the lock would leave exactly the check-then-act window this file exists to close
#   * the override needs BOTH `ALLOW_FROZEN_RUN=1` AND the explicit `--allow-frozen` argument.
#     Requiring the argument is what stops an ambient `export ALLOW_FROZEN_RUN=1` inherited from a
#     profile, a parent shell, or a resumed script from silently lifting the freeze: a profile can
#     set an env var but cannot add an argument to someone else's command line
#   * every override is appended to $OVERRIDE_LOG with timestamp, pid, campaign and GPU granted
# `adopt` is deliberately exempt: it records a reservation for a job that is ALREADY running and
# starts no new work. `release`/`reap`/`status` are likewise unaffected.
# ---------------------------------------------------------------------------
_gr_freeze_gate() {
  local camp="$1" owner="$2"
  [ -e "$FREEZE_FILE" ] || return 0        # not frozen

  if [ "$ALLOW_FROZEN_FLAG" != 1 ] || [ "${ALLOW_FROZEN_RUN:-}" != 1 ]; then
    _gr_log "REFUSED: the project is FROZEN. No reservation will be granted."
    _gr_log "marker: $FREEZE_FILE"
    _gr_log "---- marker contents ----"
    while IFS= read -r l || [ -n "$l" ]; do _gr_log "| $l"; done < "$FREEZE_FILE"
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
  mkdir -p "$(dirname "$OVERRIDE_LOG")" 2>/dev/null
  printf '%s\toverride=ALLOW_FROZEN_RUN+--allow-frozen\tpid=%s\tcampaign=%s\tgpu=%s\tmarker=%s\tuid=%s\n' \
    "$(date -Is 2>/dev/null)" "$owner" "$camp" "$gpu" "$FREEZE_FILE" "$MYUID" >> "$OVERRIDE_LOG"
  _gr_log "override recorded in $OVERRIDE_LOG"
}

_gr_do_acquire() {
  local camp="$1" owner="$2" i st rest cls
  _GR_OVERRIDE_USED=0
  # Guard against the classic caller error: passing an owner pid that is already dead.
  # $BASHPID is RE-EVALUATED inside $( ), so `G=$(gpu_reserve.sh acquire c "$BASHPID")` records
  # the transient command-substitution subshell, which exits at once -- the reservation would be
  # reclaimed as stale immediately and every racer would win the same GPU. ($$ is NOT
  # re-evaluated in subshells and is therefore the correct thing to pass.) Fail loudly.
  if [ -z "$owner" ] || [ ! -d "/proc/$owner" ]; then
    _gr_log "FATAL owner pid '${owner:-<empty>}' is not a running process. Pass the campaign"
    _gr_log "      runner's own pid: use \$\$ (not \$BASHPID, which \$( ) re-evaluates)."
    return 2
  fi
  # D-013: freeze gate, evaluated here so it is inside the mutex-held critical section.
  _gr_freeze_gate "$camp" "$owner" || return $?
  _gr_probe
  for i in $ALLOWED_GPUS; do
    st=$(_gr_reservation_state "$i")
    case "$st" in
      HELD*)
        rest=${st#HELD }
        _gr_log "gpu$i reserved by pid ${rest%% *} (${rest#* }) -- skipping"
        continue ;;
      STALE*)
        rest=${st#STALE }
        _gr_log "gpu$i RECLAIMED stale reservation: owner pid ${rest%% *} (${rest#* }) is dead"
        rm -f "$(_gr_record "$i")" ;;
    esac
    cls=$(_gr_classify "$i")
    case "$cls" in
      FOREIGN)       _gr_log "gpu$i carries a process that is not ours -- REFUSED"; continue ;;
      OURS_UNLOCKED) _gr_log "gpu$i has an unreserved CUDA context of ours (pre-lock-scheme job:${_GR_APPS[$i]:-}) -- REFUSED"; continue ;;
      UNACCOUNTED)   _gr_log "gpu$i holds ${_GR_MEM[$i]:-?} MiB we cannot attribute to our own pids -- REFUSED"; continue ;;
    esac
    _gr_write_record "$i" "$camp" "$owner"
    _gr_log "gpu$i RESERVED for '$camp' by pid $owner"
    [ "${_GR_OVERRIDE_USED:-0}" = 1 ] && _gr_log_override "$camp" "$owner" "$i"
    echo "$i"
    return 0
  done
  _gr_log "no allowed GPU available (allowed: $ALLOWED_GPUS)"
  return 3
}

_gr_do_release() {
  local camp="$1" owner="$2" i rec pid cp n=0
  for i in $ALLOWED_GPUS; do
    rec=$(_gr_record "$i"); [ -f "$rec" ] || continue
    pid=$(_gr_read_field "$rec" pid); cp=$(_gr_read_field "$rec" campaign)
    if { [ -z "$owner" ] || [ "$pid" = "$owner" ]; } && { [ -z "$camp" ] || [ "$cp" = "$camp" ]; }; then
      rm -f "$rec"; _gr_log "gpu$i released (campaign '$cp', pid $pid)"; n=$((n+1))
    fi
  done
  [ "$n" -gt 0 ] || _gr_log "nothing to release for campaign='$camp' pid='$owner'"
  return 0
}

_gr_do_reap() {
  local i st rest n=0
  for i in $ALLOWED_GPUS; do
    st=$(_gr_reservation_state "$i")
    case "$st" in STALE*)
      rest=${st#STALE }
      rm -f "$(_gr_record "$i")"
      _gr_log "gpu$i reaped stale reservation of dead pid ${rest%% *} (${rest#* })"; n=$((n+1)) ;;
    esac
  done
  _gr_log "reaped $n stale reservation(s)"
  return 0
}

_gr_do_adopt() {
  local i="$1" camp="$2" pid="$3" st
  case " $ALLOWED_GPUS " in *" $i "*) ;; *) _gr_log "gpu$i is not in the allowed set"; return 2 ;; esac
  st=$(_gr_reservation_state "$i")
  case "$st" in HELD*) _gr_log "gpu$i already reserved: $st"; return 4 ;; esac
  [ -d "/proc/$pid" ] || { _gr_log "pid $pid is not running -- refusing to adopt"; return 2; }
  _gr_write_record "$i" "$camp" "$pid"
  _gr_log "gpu$i adopted for already-running '$camp' pid $pid"
  echo "$i"
}

_gr_do_status() {
  local i st cls
  _gr_probe
  printf '%-5s %-14s %-10s %-26s %s\n' GPU RESERVATION CLASS CAMPAIGN/PID MEM_MiB
  for i in $ALLOWED_GPUS; do
    st=$(_gr_reservation_state "$i")
    cls=$(_gr_classify "$i")
    case "$st" in
      FREE)  printf '%-5s %-14s %-10s %-26s %s\n' "$i" free     "$cls" -                                 "${_GR_MEM[$i]:-?}" ;;
      HELD*) printf '%-5s %-14s %-10s %-26s %s\n' "$i" held     "$cls" "$(_gr_read_field "$(_gr_record "$i")" campaign)/$(_gr_read_field "$(_gr_record "$i")" pid)" "${_GR_MEM[$i]:-?}" ;;
      STALE*)printf '%-5s %-14s %-10s %-26s %s\n' "$i" STALE    "$cls" "$(_gr_read_field "$(_gr_record "$i")" campaign)/$(_gr_read_field "$(_gr_record "$i")" pid)" "${_GR_MEM[$i]:-?}" ;;
    esac
  done
  printf 'allowed=%s  off-limits=everything else  lockdir=%s\n' "$ALLOWED_GPUS" "$LOCKDIR"
  if [ -e "$FREEZE_FILE" ]; then
    printf 'FREEZE: ACTIVE (%s) -- acquire will REFUSE without ALLOW_FROZEN_RUN=1 + --allow-frozen\n' "$FREEZE_FILE"
  else
    printf 'FREEZE: none (%s absent)\n' "$FREEZE_FILE"
  fi
}

# ---------------------------------------------------------------------------
# sourceable helper: hold for the life of THIS shell, release via trap
# ---------------------------------------------------------------------------
gpu_reserve_hold() {
  local camp="${1:?gpu_reserve_hold <campaign> [wait_seconds] [--allow-frozen]}" every="${2:-120}" g rc
  local extra="${3:-}"
  while :; do
    # $$ (not $BASHPID) is correct here: $$ is not re-evaluated inside $( ).
    g=$("${BASH_SOURCE[0]}" acquire "$camp" "$$" ${extra:+$extra}); rc=$?
    [ "$rc" = 0 ] && break
    # only "no GPU available right now" (3) is worth waiting out. A freeze refusal (4), a bad
    # owner pid (2) or a mutex failure (10/11) will never resolve by sleeping.
    if [ "$rc" != 3 ]; then
      _gr_log "gpu_reserve_hold: acquire failed with rc=$rc -- not retrying"
      return "$rc"
    fi
    sleep "$every"
  done
  GPU_RESERVED="$g"
  # shellcheck disable=SC2064
  trap "'${BASH_SOURCE[0]}' release '$camp' $$ >/dev/null 2>&1" EXIT INT TERM HUP
  export GPU_RESERVED
  printf '%s\n' "$GPU_RESERVED"
}

gpu_reserve_drop() { "${BASH_SOURCE[0]}" release "${1:-}" "$$"; trap - EXIT INT TERM HUP; }

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
      *) _args+=("$_a") ;;
    esac
  done
  set -- ${_args[@]+"${_args[@]}"}
  case "$cmd" in
    acquire) _gr_with_mutex _gr_do_acquire "${1:?acquire <campaign> [owner_pid]}" "${2:-$PPID}" ;;
    release) _gr_with_mutex _gr_do_release "${1:-}" "${2:-$PPID}" ;;
    reap)    _gr_with_mutex _gr_do_reap ;;
    adopt)   _gr_with_mutex _gr_do_adopt "${1:?adopt <index> <campaign> <pid>}" "${2:?}" "${3:?}" ;;
    status)  _gr_with_mutex _gr_do_status ;;
    *) _gr_log "unknown subcommand '$cmd'; expected acquire|release|reap|adopt|status"; exit 2 ;;
  esac
fi
