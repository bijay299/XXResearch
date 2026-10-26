#!/bin/bash
# scripts/run_guard.sh -- sourceable guard so a runner cannot report success after a failure.
# Gate condition G2 / audit defect D-012.
#
# WHY THIS EXISTS
# ---------------
# 104 of the project's run_*.sh scripts end with `echo "<ID> DONE $(date -Is)" >>"$L"` and never
# inspect an exit code anywhere. A cell that crashed, OOM'd, or aborted on low disk therefore left
# a log whose last line says DONE. run_e2h_meta_v2_stronger.sh was the worst case: its low-disk
# abort used `break 2`, which fell straight through to the DONE line and exited 0.
#
# SCOPE OF THE DAMAGE, MEASURED (2026-10-26) -- bounded, and stated so nobody over-reacts:
#   * No Python in scripts/ parses the string DONE.
#   * No runner gates on a previous DONE marker.
#   * No published finding cites a log DONE marker as evidence of completion.
# So a false DONE never mechanically drove a decision. The hazard is a HUMAN reading a log tail
# and concluding a campaign finished. That is why the 104 historical markers are left in place --
# rewriting runners that already ran would not change what their logs said at the time, and would
# churn 104 files whose exact bytes are the record of how historical arms were produced. They are
# instead declared non-evidence (see docs/AUDIT_REPAIR_REGISTER_v1.md, G2). Completion is
# established from ARTIFACTS, never from a log line.
#
# USAGE
#   source scripts/run_guard.sh
#   rg_init "E2H" "$L"                      # campaign id + log path
#   rg_step venv_t2i/bin/python scripts/attack_relearn.py --label "$K" ...
#   rg_require_file "outputs/atk_${K}.json" # an artifact that must exist for the cell to count
#   rg_finish                               # writes DONE only if clean; else INCOMPLETE + exit 1
#
# rg_step records a failure and KEEPS GOING, so one bad cell does not silently truncate a grid;
# rg_abort records a failure and stops immediately. Either way rg_finish refuses to say DONE.
set -u

RG_ID="${RG_ID:-run}"
RG_LOG="${RG_LOG:-/dev/stderr}"
RG_FAILS=0
RG_FAIL_LIST=""

rg_init() {
  RG_ID="${1:?rg_init <campaign_id> <logfile>}"
  RG_LOG="${2:?rg_init <campaign_id> <logfile>}"
  RG_FAILS=0
  RG_FAIL_LIST=""
  # SUBSHELL-SAFE FAILURE LEDGER (found while testing rg_metric, 2026-10-26).
  # A shell variable cannot carry a failure out of a command substitution. rg_metric has to be
  # called as `V=$(rg_metric ...)` to yield its value, so every RG_FAILS++ it performed happened
  # in a subshell and VANISHED: the FAIL line reached the log, the COUNT did not, and rg_finish
  # would then have written DONE for a run whose metric could not be read. Measured: 4 induced
  # failures, only 2 counted. The ledger is a FILE, so a failure recorded anywhere -- subshell,
  # pipeline, command substitution -- is still there when rg_finish looks.
  RG_FAIL_FILE="${RG_FAIL_FILE:-${RG_LOG}.failures}"
  : > "$RG_FAIL_FILE" 2>/dev/null || RG_FAIL_FILE=""
  echo "=== $RG_ID START $(date -Is) pid=$$" >>"$RG_LOG"
}

# The one place a failure is recorded. Writes to BOTH the in-shell counter (fast path, same shell)
# and the on-disk ledger (survives subshells). rg_finish reads the ledger.
_rg_record() {
  local detail="$*"
  RG_FAILS=$((RG_FAILS + 1))
  RG_FAIL_LIST="${RG_FAIL_LIST}
  $detail"
  [ -n "${RG_FAIL_FILE:-}" ] && printf '%s\n' "$detail" >>"$RG_FAIL_FILE" 2>/dev/null
  return 0
}

# How many failures have been recorded, counting the ledger so subshell failures are included.
rg_failcount() {
  local n=0
  if [ -n "${RG_FAIL_FILE:-}" ] && [ -f "$RG_FAIL_FILE" ]; then
    n=$(grep -c . "$RG_FAIL_FILE" 2>/dev/null) || n=0
  fi
  [ "${RG_FAILS:-0}" -gt "$n" ] && n="$RG_FAILS"
  printf '%s' "$n"
}

# Run a command; on non-zero exit, record it and continue.
#
# ERREXIT NOTE (found while repairing gpu_reserve.sh, 2026-10-26). Real runners use
# `set -euo pipefail`. Without the shield below, `"$@" >>"$RG_LOG" 2>&1` is itself the failing
# command, so errexit killed the runner BEFORE `local rc=$?` ran -- the FAIL line was never
# written and the header's promise that rg_step "records a failure and KEEPS GOING" was false
# under exactly the option set the runners use. The shield guarantees the recording happens.
# rg_step still RETURNS the command's rc, so a caller under set -e that wants to continue must
# write `rg_step ... || true` (or use rg_cell, below). rg_finish is what refuses to say DONE.
rg_step() {
  local _rg_had_e=0 _rg_had_pf=0
  case $- in *e*) _rg_had_e=1 ;; esac
  if shopt -qo pipefail 2>/dev/null; then _rg_had_pf=1; fi
  set +e
  set +o pipefail 2>/dev/null || true
  "$@" >>"$RG_LOG" 2>&1
  local rc=$?
  if [ "$_rg_had_pf" = 1 ]; then set -o pipefail 2>/dev/null || true; fi
  if [ "$_rg_had_e" = 1 ]; then set -e; fi
  if [ "$rc" -ne 0 ]; then
    _rg_record "rc=$rc  $*"
    echo "FAIL rc=$rc :: $*" >>"$RG_LOG"
  fi
  return "$rc"
}

# Grid-safe variant: records the failure and ALWAYS returns 0, so one bad cell cannot truncate a
# grid even when the runner uses `set -e`. rg_finish still refuses to write DONE.
rg_cell() {
  rg_step "$@" || true
  return 0
}

# Assert an artifact the cell was supposed to produce. Catches "exit 0 but wrote nothing",
# and the truncation class (D-028): a zero-byte or absent file is not a result.
rg_require_file() {
  local f="${1:?rg_require_file <path>}"
  if [ ! -s "$f" ]; then
    _rg_record "missing-or-empty  $f"
    echo "FAIL missing-or-empty artifact :: $f" >>"$RG_LOG"
    return 1
  fi
  return 0
}

# Like rg_require_file, but takes a GLOB and requires at least one NON-EMPTY match.
#
# WHY THIS EXISTS. Several runners cannot predict their artifact's exact filename, because the
# python side formats it. scripts/r4b_uce_upstream_vs_port.py builds
# `outputs/r4b_uce_lamb{a.lamb}.json` where a.lamb is a FLOAT, so the shell loop value `1000`
# becomes the file `r4b_uce_lamb1000.0.json`. A hard-coded rg_require_file would have reported
# INCOMPLETE on a cell that actually succeeded -- a guard that cries wolf gets switched off, which
# is how guards die. r4c/r4d are worse: their config tag changed shape between versions
# (`r4c_uce_lamb0.5.json` then `r4c_uce_church_lam0.5_es1.0_ps1.0.json`), so only a glob is stable.
#
# A glob that matches nothing is still a FAILURE. That is the safe direction: if the artifact name
# drifts again the runner says INCOMPLETE rather than DONE, and someone looks.
rg_require_glob() {
  local pat="${1:?rg_require_glob <glob>}" f hit=0
  for f in $pat; do
    [ -s "$f" ] && hit=1 && break
  done
  if [ "$hit" = 0 ]; then
    _rg_record "no-match-or-empty  $pat"
    echo "FAIL no non-empty artifact matches :: $pat" >>"$RG_LOG"
    return 1
  fi
  echo "ok artifact :: $f" >>"$RG_LOG"
  return 0
}

# Record a deliberate abort (low disk, refused reservation, operator stop) as a FAILURE,
# so it can never be confused with completion.
rg_abort() {
  _rg_record "abort  ${*:-unspecified}"
  echo "ABORT :: ${*:-unspecified}" >>"$RG_LOG"
}

# ===========================================================================
# TRUTHFUL MEASUREMENT HANDLING (repair A v3, task 2).
#
# Three defect classes are closed here, all of them cases where the ABSENCE of a measurement was
# turned into a VALUE and then reasoned about:
#
#  (1) MISSING-AS-ZERO. The idiom `X=$(python -c '...json.load...') 2>/dev/null || echo 0` and its
#      relatives appear 33 times in this project's run_*.sh. The historical ~104 scripts are
#      byte-frozen and stay as they are, but the resumed path must never do it: a crashed reader
#      and a genuine 0.0 are not the same event, and the kill-rule text "retain8=0.0000 ... below
#      K1 -- destroys the model" is a claim about a model, printed from a json.load that crashed.
#      rg_require_metric refuses to produce a number it did not read.
#
#  (2) EXISTENCE-AS-VALIDITY. `[ -f ]` and even `[ -s ]` accept a truncated file.
#      adapters/utrap_church_l1.0k4/unet.pt is 511,229,952 bytes -- non-empty, passes every guard
#      in the project -- and torch.load on it raises
#        RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central
#        directory
#      (measured 2026-10-26). rg_require_loadable actually opens the artifact.
#
#  (3) MARKER-AS-COMPLETION. A manifest written before its artifacts are validated says a run
#      finished when it did not. rg_manifest validates first and writes atomically.
#
# Atomicity throughout: same-directory temp file, flush, fsync, os.replace. os.replace is atomic
# within a filesystem, so a reader sees either the old file or the new one -- never a half-written
# one, which is defect class (2) being manufactured by our own writer.
# ===========================================================================

# The interpreter is resolved to an ABSOLUTE path, anchored to this file's own location.
#
# It used to be the relative `venv_t2i/bin/python`, which works only while the caller's working
# directory happens to be the project root. Found while testing the run_v2 gates from a scratch
# directory: every rg_require_loadable and every rg_metric failed with
#     run_guard.sh: line 274: venv_t2i/bin/python: No such file or directory
# and was recorded as "artifact exists but does NOT LOAD". The direction is safe -- it records
# failures rather than passing -- but it is a FALSE failure that reads exactly like a real
# truncated artifact, which is its own kind of untruthful reporting, and it would have made the
# guard look broken the first time any runner was launched from elsewhere.
_rg_self=$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null) || _rg_self="${BASH_SOURCE[0]}"
_rg_root=$(cd "$(dirname "$_rg_self")/.." 2>/dev/null && pwd -P) || _rg_root="."
RG_PY="${RG_PY:-$_rg_root/venv_t2i/bin/python}"
if [ ! -x "$RG_PY" ]; then
  printf '[run_guard] WARNING interpreter %s is not executable; loadability and metric checks\n' "$RG_PY" >&2
  printf '[run_guard]         will FAIL CLOSED (recorded as failures), not be skipped.\n' >&2
fi

# rg_atomic_write <dest> -- reads the payload from STDIN and publishes it atomically.
#
# The payload is captured by the SHELL, not by python. The obvious spelling,
#   "$RG_PY" - "$dest" <<'RGPY' ... sys.stdin.buffer.read() ... RGPY
# cannot work: `python -` takes its PROGRAM from stdin, so the heredoc IS stdin and the payload
# read comes back empty. Measured while testing this function -- it wrote a ZERO-BYTE file and
# returned 0, i.e. it manufactured exactly the truncated-artifact class it was written to prevent,
# and reported success. python is therefore used only for fsync/os.replace, via -c.
rg_atomic_write() {
  local dest="${1:?rg_atomic_write <dest>}" d tmp rc
  d=$(dirname "$dest")
  mkdir -p "$d" || return 1
  tmp=$(mktemp "$d/.rgtmp.XXXXXX") || return 1      # SAME directory => os.replace is atomic
  if ! cat > "$tmp"; then
    rm -f "$tmp"; echo "FAIL atomic write: could not buffer the payload for $dest" >>"$RG_LOG"
    return 1
  fi
  # An empty payload is a bug in the caller, never a legitimate result file. Refuse it here rather
  # than publish a zero-byte artifact that every [ -s ] guard downstream would then reject anyway.
  if [ ! -s "$tmp" ]; then
    rm -f "$tmp"
    echo "FAIL atomic write: refusing to publish an EMPTY payload to $dest" >>"$RG_LOG"
    return 1
  fi
  "$RG_PY" -c '
import os, sys
tmp, dest = sys.argv[1], sys.argv[2]
fd = os.open(tmp, os.O_RDONLY)
try: os.fsync(fd)                      # the bytes, before the rename
finally: os.close(fd)
os.replace(tmp, dest)                  # atomic within a filesystem
d = os.path.dirname(os.path.abspath(dest)) or "."
dfd = os.open(d, os.O_RDONLY)
try: os.fsync(dfd)                     # the rename itself, so it survives a crash
finally: os.close(dfd)
' "$tmp" "$dest"; rc=$?
  if [ "$rc" -ne 0 ]; then
    rm -f "$tmp" 2>/dev/null
    echo "FAIL atomic write: could not publish $dest (rc=$rc)" >>"$RG_LOG"
    return 1
  fi
  return 0
}

# rg_require_loadable <path> [kind]  -- kind: auto|torch|json|image. Existence is not validity.
# Records a FAILURE (so rg_finish cannot say DONE) and returns non-zero.
rg_require_loadable() {
  local f="${1:?rg_require_loadable <path> [kind]}" kind="${2:-auto}" out rc
  if [ ! -s "$f" ]; then
    _rg_record "missing-or-empty  $f"
    echo "FAIL missing-or-empty artifact :: $f" >>"$RG_LOG"
    return 1
  fi
  out=$("$RG_PY" - "$f" "$kind" 2>&1 <<'RGPY'
import json, os, sys
p, kind = sys.argv[1], sys.argv[2]
if kind == "auto":
    e = os.path.splitext(p)[1].lower()
    kind = {".pt": "torch", ".pth": "torch", ".bin": "torch", ".safetensors": "torch",
            ".json": "json", ".png": "image", ".jpg": "image", ".jpeg": "image"}.get(e, "bytes")
if kind == "torch":
    import torch
    sd = torch.load(p, map_location="cpu")
    if hasattr(sd, "keys"):
        n = sum(getattr(v, "numel", lambda: 0)() for v in sd.values())
        # A checkpoint of zero tensors loads fine and means nothing. Non-finite values are the
        # quiet version of the same problem: the file is readable and the numbers are garbage.
        if len(sd) == 0:
            raise SystemExit("loaded but contains NO tensors")
        bad = [k for k, v in sd.items()
               if hasattr(v, "isfinite") and not bool(v.isfinite().all())]
        if bad:
            raise SystemExit("non-finite tensors: %s%s" % (bad[:5], " ..." if len(bad) > 5 else ""))
        print("torch ok: %d entries, %d params" % (len(sd), n))
    else:
        print("torch ok: %s" % type(sd).__name__)
elif kind == "json":
    with open(p) as fh: d = json.load(fh)
    print("json ok: %s with %d top-level keys"
          % (type(d).__name__, len(d) if hasattr(d, "__len__") else 0))
elif kind == "image":
    from PIL import Image
    im = Image.open(p); im.load()
    print("image ok: %s %s" % (im.mode, im.size))
else:
    with open(p, "rb") as fh:
        if not fh.read(1): raise SystemExit("file is empty")
    print("bytes ok: %d" % os.path.getsize(p))
RGPY
  ); rc=$?
  if [ "$rc" -ne 0 ]; then
    _rg_record "UNLOADABLE  $f :: $(printf '%s' "$out" | tail -2 | tr '\n' ' ')"
    echo "FAIL artifact exists but does NOT LOAD :: $f" >>"$RG_LOG"
    printf '%s\n' "$out" | sed 's/^/    /' >>"$RG_LOG"
    return 1
  fi
  echo "ok loadable :: $f :: $(printf '%s' "$out" | tail -1)" >>"$RG_LOG"
  return 0
}

# rg_metric <json> <dotted.key.path> -- prints the value ONLY if it is present and a finite
# number. On any failure it prints NOTHING, records a failure and returns non-zero. It must never
# print a fallback: a caller that substitutes 0 here is the defect this function exists to remove.
# Optional <min> <max> bound the plausible range; a metric outside it is malformed, not extreme.
rg_metric() {
  local f="${1:?rg_metric <json> <dotted.path> [min] [max]}" key="${2:?}" lo="${3:-}" hi="${4:-}"
  local out rc
  out=$("$RG_PY" - "$f" "$key" "$lo" "$hi" 2>&1 <<'RGPY'
import json, math, sys
f, key, lo, hi = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    with open(f) as fh: d = json.load(fh)
except FileNotFoundError:
    raise SystemExit("ABSENT: %s does not exist" % f)
except Exception as e:
    raise SystemExit("UNREADABLE: %s: %s: %s" % (f, type(e).__name__, e))
cur, walked = d, []
for part in key.split("."):
    walked.append(part)
    if not isinstance(cur, dict):
        raise SystemExit("NOT-A-DICT at '%s' in %s" % (".".join(walked), f))
    if part not in cur:
        raise SystemExit("MISSING FIELD '%s' in %s (available: %s)"
                         % (".".join(walked), f, sorted(cur)[:12]))
    cur = cur[part]
if isinstance(cur, bool) or not isinstance(cur, (int, float)):
    raise SystemExit("NON-NUMERIC '%s' = %r (%s) in %s" % (key, cur, type(cur).__name__, f))
if not math.isfinite(cur):
    raise SystemExit("NON-FINITE '%s' = %r in %s" % (key, cur, f))
if lo and cur < float(lo): raise SystemExit("OUT OF RANGE '%s' = %r < %s" % (key, cur, lo))
if hi and cur > float(hi): raise SystemExit("OUT OF RANGE '%s' = %r > %s" % (key, cur, hi))
sys.stdout.write(repr(float(cur)))
RGPY
  ); rc=$?
  if [ "$rc" -ne 0 ]; then
    _rg_record "BAD-METRIC  $f :: $key :: $out"
    echo "FAIL metric unusable :: $f :: $key :: $out" >>"$RG_LOG"
    return 1
  fi
  printf '%s' "$out"
  return 0
}

# rg_ratio_ok <json> <numerator.path> <denominator.path>
# Validates a DENOMINATOR before anything divides by it. A rate computed over zero samples is not
# 0.0, it is undefined, and printing it as 0.0000 is how "destroys the model" gets claimed about a
# run that scored nothing at all.
rg_ratio_ok() {
  local f="${1:?}" num="${2:?}" den="${3:?}" d
  d=$(rg_metric "$f" "$den") || return 1
  if "$RG_PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "$d" 2>/dev/null; then
    rg_metric "$f" "$num" >/dev/null || return 1
    return 0
  fi
  _rg_record "ZERO-DENOMINATOR  $f :: $den = $d"
  echo "FAIL denominator is zero/absent, so '$num/$den' is UNDEFINED, not 0.0 :: $f" >>"$RG_LOG"
  return 1
}

# rg_reuse_ok <artifact> <sidecar.json> <kind> [key=expected ...]
# The ONLY sanctioned way to decide that an existing artifact may be reused. All three must hold:
#   hash identity   -- the sidecar's recorded sha256 equals the artifact's sha256 NOW
#   config identity -- every key=expected pair matches the sidecar
#   loadability     -- the artifact actually opens
# Returns 0 = safe to reuse, 1 = recompute. It records NO failure: "must recompute" is a normal
# outcome. Silence is deliberate here and loudness is deliberate in rg_require_loadable.
rg_reuse_ok() {
  local art="${1:?rg_reuse_ok <artifact> <sidecar> <kind> [k=v ...]}" side="${2:?}" kind="${3:-auto}"
  shift 3
  [ -s "$art" ]  || { echo "reuse: no artifact $art" >>"$RG_LOG"; return 1; }
  [ -s "$side" ] || { echo "reuse: no sidecar $side -> recompute $art" >>"$RG_LOG"; return 1; }
  local why
  why=$("$RG_PY" - "$art" "$side" "$@" 2>&1 <<'RGPY'
import hashlib, json, sys
art, side = sys.argv[1], sys.argv[2]
pairs = sys.argv[3:]
try:
    with open(side) as fh: s = json.load(fh)
except Exception as e:
    raise SystemExit("sidecar unreadable: %s: %s" % (type(e).__name__, e))
h = hashlib.sha256()
with open(art, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
got = h.hexdigest()
rec = s.get("sha256") or s.get("artifact_sha256") or s.get("unet_sha256")
if not rec:
    raise SystemExit("sidecar records no sha256 for %s" % art)
if rec != got:
    raise SystemExit("HASH MISMATCH: on disk %s, sidecar %s" % (got[:16], rec[:16]))
for p in pairs:
    k, _, want = p.partition("=")
    if k not in s:
        raise SystemExit("config key '%s' absent from the sidecar" % k)
    if str(s[k]) != want:
        raise SystemExit("config MISMATCH '%s': sidecar %r, required %r" % (k, s[k], want))
print("hash+config identical")
RGPY
  ) || { echo "reuse: REJECTED $art :: $why" >>"$RG_LOG"; return 1; }
  # Loadability last, because it is the expensive check -- but it is NOT optional. A matching hash
  # over a truncated file is a matching hash over a truncated file.
  local before="$RG_FAILS" blist="$RG_FAIL_LIST" bledger=""
  [ -n "${RG_FAIL_FILE:-}" ] && [ -f "$RG_FAIL_FILE" ] && bledger=$(cat "$RG_FAIL_FILE")
  if ! rg_require_loadable "$art" "$kind" >/dev/null 2>&1; then
    # "recompute" is not a run failure, so the LEDGER must be rewound too, not just the counter.
    RG_FAILS="$before"; RG_FAIL_LIST="$blist"
    if [ -n "${RG_FAIL_FILE:-}" ]; then
      if [ -n "$bledger" ]; then printf '%s\n' "$bledger" > "$RG_FAIL_FILE"; else : > "$RG_FAIL_FILE"; fi
    fi
    echo "reuse: REJECTED $art :: hash+config matched but the artifact does NOT LOAD" >>"$RG_LOG"
    return 1
  fi
  echo "reuse: ACCEPTED $art :: $why, loads" >>"$RG_LOG"
  return 0
}

# rg_manifest <path> <artifact>[:kind] ...
# Publishes a completion manifest ONLY after every listed artifact validates, and only if no
# failure has been recorded. Written atomically. This is the artifact-level counterpart of
# rg_finish: rg_finish gates the LOG marker, rg_manifest gates the MACHINE-READABLE one.
rg_manifest() {
  local dest="${1:?rg_manifest <path> <artifact>[:kind] ...}"; shift
  local a f kind bad=0 entries=""
  for a in "$@"; do
    f="${a%%:*}"; kind="auto"
    case "$a" in *:*) kind="${a##*:}" ;; esac
    if rg_require_loadable "$f" "$kind"; then
      entries="$entries $f"
    else
      bad=1
    fi
  done
  if [ "$bad" != 0 ] || [ "$(rg_failcount)" != 0 ]; then
    echo "MANIFEST WITHHELD :: $dest ($(rg_failcount) recorded failure(s)); the run is NOT complete" >>"$RG_LOG"
    rm -f "$dest" 2>/dev/null
    return 1
  fi
  # ---- PUBLISH, WITH THE STATUS ACTUALLY CHECKED. -------------------------------------------
  # D-061. This was previously a pipeline --
  #     "$RG_PY" - "$dest" "$RG_ID" $entries <<'RGPY' ... RGPY | rg_atomic_write "$dest"
  #     echo "manifest published :: $dest" >>"$RG_LOG"
  #     return 0
  # -- whose exit status was never captured. Neither the generator's failure nor the publish's
  # failure could be seen: it logged "manifest published" and returned 0 regardless. And because
  # run_r1c_eval.sh calls it inside an `if`, which suppresses errexit, rg_finish then went on to
  # write DONE with NO MANIFEST ON DISK. That is the one requirement this primitive exists to
  # enforce -- "publish a completion manifest only after all required artifacts validate" --
  # failing inside the enforcer.
  #
  # The pipeline is gone: the JSON is generated to a temp file whose status is checked, then
  # published, then READ BACK AND PARSED. A manifest that cannot be re-read is not a manifest, and
  # saying so costs one json.load.
  local gen; gen=$(mktemp "${TMPDIR:-/tmp}/rg_manifest.XXXXXX") || {
    _rg_record "manifest  $dest :: cannot create a temp file"
    echo "FAIL manifest: cannot create a temp file for $dest" >>"$RG_LOG"; return 1; }
  # shellcheck disable=SC2086
  if ! "$RG_PY" - "$dest" "$RG_ID" $entries > "$gen" <<'RGPY'
import datetime, hashlib, json, os, sys
dest, rid = sys.argv[1], sys.argv[2]
arts = []
for p in sys.argv[3:]:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    arts.append({"path": p, "bytes": os.path.getsize(p), "sha256": h.hexdigest()})
json.dump({"campaign": rid,
           "status": "complete",
           "written_at": datetime.datetime.now().astimezone().isoformat(),
           "validation": "every artifact below was opened and parsed, not merely stat()ed",
           "artifacts": arts}, sys.stdout, indent=1)
RGPY
  then
    rm -f "$gen" 2>/dev/null
    _rg_record "manifest  $dest :: generator failed"
    echo "FAIL manifest: the generator failed; nothing published to $dest" >>"$RG_LOG"
    return 1
  fi
  if ! rg_atomic_write "$dest" < "$gen"; then
    rm -f "$gen" 2>/dev/null
    _rg_record "manifest  $dest :: atomic publish failed"
    echo "FAIL manifest: the atomic publish failed; $dest was NOT written" >>"$RG_LOG"
    return 1
  fi
  rm -f "$gen" 2>/dev/null
  # Read it back. "I wrote it" and "it is there and parses" are different claims.
  if ! rg_require_loadable "$dest" json >/dev/null 2>&1; then
    _rg_record "manifest  $dest :: published but does not parse"
    echo "FAIL manifest: $dest was published but does NOT parse -- withdrawing it" >>"$RG_LOG"
    rm -f "$dest" 2>/dev/null
    return 1
  fi
  echo "manifest published and re-read :: $dest" >>"$RG_LOG"
  return 0
}

# The only place a DONE marker may be written.
rg_finish() {
  local n; n=$(rg_failcount)
  if [ "$n" -ne 0 ]; then
    echo "$RG_ID INCOMPLETE $(date -Is) -- $n failure(s):${RG_FAIL_LIST}" >>"$RG_LOG"
    [ -n "${RG_FAIL_FILE:-}" ] && [ -f "$RG_FAIL_FILE" ] && sed 's/^/  ledger: /' "$RG_FAIL_FILE" >>"$RG_LOG"
    echo "$RG_ID INCOMPLETE: $n failure(s); see $RG_LOG" >&2
    return 1
  fi
  echo "$RG_ID DONE $(date -Is)" >>"$RG_LOG"
  return 0
}
