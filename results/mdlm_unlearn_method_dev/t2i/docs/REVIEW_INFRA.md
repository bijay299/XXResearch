# REVIEW_INFRA.md — independent review of the v2 infrastructure repairs

Reviewer: independent (did not build this code). Date 2026-10-26.
Scope: `scripts/gpu_reserve.sh`, `scripts/test_gpu_reserve.sh`, `scripts/run_guard.sh`,
`scripts/campaign_free.sh`, and the nine migrated runners. Builder's write-up
`docs/REPAIR_INFRA_v2.md` was read but **not** taken as evidence; every claim below is
settled against the code or against a test I ran myself.

**No GPU was used.** No CUDA, no `run_*.sh` execution. All dynamic evidence comes from a
shim `nvidia-smi` I wrote myself (independent of the builder's, at
`<scratch>/rv/smi`) plus CPU-only placeholder processes. `FREEZE_ACTIVE` was not removed,
no process was killed outside my own fixtures, nothing was committed.

---

## 0. The suite, re-run by me

```
$ bash scripts/test_gpu_reserve.sh
...
RESULT: 215 passed, 0 failed
No GPU was accessed; nvidia-smi was the shim at .../fake_nvsmi
(exit status 0)
```

**215 passed / 0 failed** is what I observe, matching the documented count. Two of those
215 are not assertions:

| site | problem |
|---|---|
| `test_gpu_reserve.sh:609-613` | `if [ "$LEAKED" -gt 0 ]; then ok "...LEAKED"; else ok "both released"; fi` — **both branches call `ok`**. This test cannot fail. It asserts "either a record leaked or it did not". |
| `test_gpu_reserve.sh:320` | `if [ "$OUT" != "0" ]; then ok ...; fi` — no `bad` arm; a regression here silently reduces the PASS count instead of failing. (Harmless in practice: the preceding `chk` at :319 covers it.) |

Everything else has a real failure arm (19 `ok`-only call sites vs 103 `bad` sites; the two
above are the only cannot-fail ones, found by parsing every `if` block).

### The current perimeter

The harness pins, at `test_gpu_reserve.sh:112-124`, for **every** test:
`GPU_RESERVE_TEST_MODE=1`, a shim `NVSMI`, `GPU_RESERVE_GPUS="0 1"`,
`GPU_RESERVE_FOREIGN_MIB=512`, `GPU_RESERVE_NVSMI_TIMEOUT=20`, `GPU_RESERVE_WAIT=20`, a
harness-owned `GPU_RESERVE_LOCKDIR` (reset by `NEWLOCKS`), and a redirected freeze marker.
`GPU_RESERVE_ORPHAN_GRACE=1` is exported at :660 and correctly `unset` at :726.

v1's perimeter was "working shim, `GPU_RESERVE_GPUS="0 1"`, no `set -e`, childless victim".
v2 genuinely leaves that perimeter: test mode is now *contrasted* with production
(T19/T19b), errexit shapes are exercised (T22), and the orphan victim has a surviving child
in its own process group (T21, via `setsid` at :668). That is real progress.

**What still lies outside it**, and where every finding below lives:

1. **Production mode with a substituted telemetry source or lockdir.** The suite only ever
   asks "does *test mode* refuse the production lockdir / the real driver". It never asks
   "does *production* refuse a non-production lockdir or an arbitrary `NVSMI`". It does not.
2. **Env vars other than `GPU_RESERVE_GPUS`.** `FOREIGN_MIB`, `ORPHAN_GRACE`, `UID`,
   `WAIT`, `CVD_STYLE`, `NVSMI`, `LOCKDIR` are never varied off their pinned values; three
   of them are unvalidated and two of those are fail-open.
3. **A dead holder whose survivor shares the *acquirer's* process group** (the non-`setsid`
   case — i.e. a runner relaunched from the same parent script).
4. **The runners' own non-reservation logic.** T27 asserts only `bash -n` parses, absence of
   `campaign_free.sh`, and presence of the API string. It inspects no exit-status handling,
   no `DONE` marker, and no conditional around the reservation call.

---

## 1. Telemetry fail-closed — **VERIFIED, with two residual shapes**

The code is honest about this one. `_gr_probe` (`gpu_reserve.sh:440-542`) runs each of the
three queries into a temp file via `_gr_nvsmi_run` (:420-431) with `rc=$?` captured
immediately, treats 124/137 as a timeout, requires non-empty output for queries 1-2,
rejects a non-numeric index (:465, :495), a non-`GPU-`/`MIG-` UUID (:468-470), a
non-numeric `memory.used` (:499-501, the `[N/A]`→0 coercion), a non-numeric compute-apps
pid (:518), a `gpu_uuid` the index view never listed (:522-524), and any allowed index
missing from **either** view (:529-536). `_gr_classify:621` refuses to coerce a
non-numeric memory to 0. I reproduced the documented rc=5 for all nine shim modes.

I then tried to construct a shape that slips through. Nine attempts, my own shim:

```
G1 duplicate index rows (40000 then 5 MiB)                  rc=0  index=[0]   <-- slips
G2 two indices, SAME uuid, foreign pid on it                rc=0  index=[0]   <-- slips
G3 uuid containing a comma                                  rc=0  index=[0]   (benign)
G4 baseline all idle (control)                              rc=0  index=[0]
G5 apps pid absent from /proc (hidepid-like)                rc=0  index=[1]   (fail-closed: gpu0 FOREIGN)
G6 our own live pid on gpu0                                 rc=0  index=[1]   (correct)
G7 memory.used='1,024'                                      rc=5  index=[]
G8 memory.used='0x7000'                                     rc=5  index=[]
G9 40000 MiB attributed to uid-0 pid                        rc=3  index=[]    (correct)
```

* **G2 is a genuine residual fail-open.** `_GR_UUID2IDX` (`:472`) is a many-to-one map with
  **no collision check**: if two indices report the same UUID string, the compute-apps rows
  for that UUID are attributed only to the *last* index, and the other index reads as FREE
  even with another user's process on the device. The parser detects a *missing* index but
  not a *duplicated* UUID. Not a shape a healthy driver emits; a truncated/garbled UUID
  column is the plausible route. **Severity: low-medium. Fix is two lines.**
* **G1**: duplicate index rows are accepted last-wins rather than refused, so contradictory
  telemetry resolves silently to the later row. Same class, same fix site.

Neither is "nvidia-smi failed and we called it free" — on that specific claim the repair
holds. But see §3: the *threshold* against which memory is judged is unvalidated, and that
is a much easier way to turn a busy GPU into a free one.

## 2. Subprocess exit statuses — **VERIFIED in `gpu_reserve.sh`; DEFECT FOUND in the runners**

In `gpu_reserve.sh` I found no surviving `< <(...)` process substitution and no pipeline
whose status is discarded with consequence. Checked individually: the three
`_gr_nvsmi_run` sites, `_gr_read_stat` (`:319-329`, builtins only, no `printf|awk`),
`_gr_read_field` (`:380-387`, plain redirect), `_gr_write_record` (`:695-714`, both the
write *and* the `mv` publish checked → rc=10), `_gr_with_mutex` (`:721-736`, `exec 9>>` and
`flock -w` checked, `flock -u` logged), `_gr_tmpdir`, the `head -c 200` stderr captures, and
`sleep "$wait_s"` at `:1086`. `_gr_proc_uid` (`:347`) is unchecked but empty output is
handled as "not visible" and classified FOREIGN — fail-closed, confirmed by G5.

**But the test that is supposed to prove this is a grep test.** `T24`
(`test_gpu_reserve.sh:845-862`) counts occurrences of `< <(`, of `_gr_nvsmi_run`, of
`rc=$?`, of `awk`, and asserts a string `if ! mv -f` is present. That asserts *the source
matches a pattern*, not *every status is checked*. It would pass unchanged if a new
unchecked pipeline were added tomorrow in any other shape. Its one behavioural assertion
(the unwritable-lockdir case at :863-872) is good and I reproduced it.

**DEFECT, and it is in a migrated runner.** `run_e2h_meta_v2_stronger.sh:55`:

```
FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
[ "$FREE" -lt 5 ] && { echo "ABORT: ${FREE}GiB free" >>"$L"; INCOMPLETE=1; break 2; }
```

`df`'s status is lost in the pipeline. If `df` fails, `FREE=""`, and `[ "" -lt 5 ]` is a
`[`-level error (status 2), so the `&&` branch **does not run**: the low-disk abort
silently does not fire and the grid proceeds with disk state unknown. Reproduced:

```
FREE=[]
ABORT did NOT fire -> the loop continues with disk state UNKNOWN
```

Three more runners never inspect a status at all — see §9.

**Latent structural fail-open in `gpu_reserve.sh:819-843`.** The `case "$st"` over
`_gr_reservation_state`'s output has arms for `HELD*`, `MAPCHANGED*`, `ORPHAN*`, `STALE*`
and **no `FREE)` arm and no `*)` default**. Any unrecognised state — an empty string from a
killed subshell, or a new state name added later — falls straight through to `_gr_classify`
and can be claimed. This is the "unknown treated as free" class surviving in the control
flow after being eliminated from the telemetry. Not currently reachable (every branch of
`_gr_reservation_state` echoes one of the five known strings); it is a one-line hardening.

## 3. Production allowlist cannot be widened — **CLAIM UNSUPPORTED (as stated)**

The rc matrix is exactly as claimed. My own run, production mode, my own shim:

```
GPUS=[2 3     ] rc=6   GPUS=[0 1 2 3 ] rc=6   GPUS=[         ] rc=7
GPUS=[   (ws) ] rc=7   GPUS=[abc     ] rc=7   GPUS=[-1      ] rc=7
GPUS=[0 0     ] rc=7   GPUS=[0 1 1   ] rc=7   GPUS=[1 0     ] rc=6
GPUS=[00 1    ] rc=6   GPUS=[0 1     ] rc=4  (the real FREEZE_ACTIVE)
```

7 for config errors, 6 for prohibited sets, 4 for the freeze, 2 for usage (§6). **A config
error can never return the retryable 3**: `_gr_allowlist_from_env` (`:205-234`) validates
*before* mode selection in both modes, `:296-306` is a mode-independent backstop, and
`gpu_reserve_hold`'s wait loop retries **only** on 3 (`:1084-1090`). `_GR_INIT_RC`
(`:1189-1194`, consumed at `:1064-1068`) means a sourced-with-bad-config helper refuses with
the same non-retryable code instead of running on pre-validation defaults. This part is
well built and the `${VAR+set}` fix at `:207` is correct.

**But "NO env var can widen it" is false.** The only thing standing between
`GPU_RESERVE_GPUS="2 3"` and physical GPU 2 is `_gr_nvsmi_is_real` (`:165-176`), and that
detector is **name-based only**: it `readlink -f`s the command and compares the *basename*
to `nvidia-smi`. A hard copy of the real binary under any other name defeats it. Measured
(the copy was **not executed** — only the detector was called on its path):

```
detector: smi_link (symlink to /usr/bin/nvidia-smi) => REAL   (readlink -f catches it)
detector: smi_copy (cp of /usr/bin/nvidia-smi)      => NOT REAL  <== INTERLOCK BYPASSED
```

and the widening itself, end to end:

```
GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="2 3" GPU_RESERVE_NVSMI=<shim> GPU_RESERVE_LOCKDIR=<tmp>
  -> acquire rc=0, INDEX HANDED OUT = 2, CUDA_VISIBLE_DEVICES exported as 2
```

So a three-variable combination reaches another user's GPU. It is deliberate, not
accidental, and the header does document that test mode overrides the allowlist — but the
guarantee as written in `gpu_reserve.sh:99-101` and in `REPAIR_INFRA_v2.md` ("no
environment variable can point production at GPUs 2-3") overstates what the code enforces.
It should say: *production* cannot be widened; *test mode* can, and test mode is gated only
by the telemetry command's filename.

**Two production env vars are unvalidated and fail OPEN.** Production refuses
`GPU_RESERVE_GPUS` and `GPU_RESERVE_FREEZE_FILE` but silently honours everything else.

*(a) `GPU_RESERVE_FOREIGN_MIB`* — set unvalidated at `:146`, used at `:622` as
`[ "$mem" -ge "$FOREIGN_MIB" ]`. A non-numeric value makes `[` error, which reads as false,
which yields `FREE`. Both GPUs at 30 000 MiB with no attributable process:

```
FOREIGN_MIB=[512      ] rc=3  index=[]   (correct: UNACCOUNTED, refused)
FOREIGN_MIB=[<unset>  ] rc=3  index=[]
FOREIGN_MIB=[abc      ] rc=0  index=[0]  <-- 30 GB of foreign memory read as FREE
FOREIGN_MIB=[99999999 ] rc=0  index=[0]  <-- same, via a plausible-looking number
FOREIGN_MIB=[0        ] rc=3  index=[]
```

This is defect 1 intact in the threshold rather than in the telemetry, and it is reachable
in production.

*(b) `GPU_RESERVE_ORPHAN_GRACE`* — `:149`, used at `:680`. Same `[`-error pattern:

```
ORPHAN_GRACE=[120 ] fresh record, childless SIGKILLed victim => ORPHAN   (correct)
ORPHAN_GRACE=[abc ] => STALE   <-- grace window silently gone
ORPHAN_GRACE=[0   ] => STALE
```

*(c) `GPU_RESERVE_UID`* (`:147`) is **fail-closed** either way — pointing it at another uid
makes their processes `OURS_UNLOCKED` (refused) and pointing it at nothing makes ours
`FOREIGN` (refused). No finding.

**(d) The most serious one: `GPU_RESERVE_LOCKDIR` is honoured in production, unrefused**
(`:287`). Verified:

```
$ env -u GPU_RESERVE_TEST_MODE GPU_RESERVE_NVSMI=<my shim> GPU_RESERVE_LOCKDIR=<tmp> gpu_reserve.sh status
mode=production
telemetry=OK  gpus=[1 0] via <my shim>          <-- production accepted a fabricated telemetry source
allowed=0 1  lockdir=<tmp>                      <-- the MUTEX moved out of the project
```

Two production runners with different `GPU_RESERVE_LOCKDIR` values take different mutexes
and write different record sets, so they cannot see each other's reservations. That is the
original 2026-10-26 triple-booking failure, reachable from one stray exported variable,
against the guarantee the whole file exists to provide. Production hard-pins the allowlist
and the freeze path; it should hard-pin the lockdir and require `_gr_nvsmi_is_real` for the
same reason. **This is my primary gate-blocking finding.**

## 4. ORPHAN state — **VERIFIED, but the pgid arm has a hole, and 120 s is not defensible here**

The mechanism is real and the test is real. `_gr_reservation_state` (`:635-685`) keeps the
record and refuses when (a) any process of the dead owner's pgid lives (`:658-661`), or
(b) any CUDA context of our uid sits on that GPU (`:662-669`), or (c) telemetry is UNKNOWN
(`:670-674`, the right call), or (d) the record is younger than the grace window
(`:675-683`). `_gr_do_reap` (`:886-904`) refuses to reap ORPHAN and MAPCHANGED. T21 passes
for the right reason — it uses `setsid` (`:668`) so the victim has its own process group,
and sets grace to 1 s (`:660`) so the grace path cannot be what passes. I reproduced the
whole fixture. SIGKILL leaks the record by design; SIGTERM releases it:

```
SIGTERM  records-after=0  RELEASED
SIGKILL  records-after=1  LEAKED (orphan window applies)   <-- intended
```

**DEFECT: the pgid arm is inoperative when the acquirer shares the dead holder's pgid.**
`_gr_live_in_pgid:368` is `[ "$want" != "$mypg" ] || return 1`. The comment explains why
(a non-interactive parent gives its children its own group, so we must not mistake
ourselves for the survivor) — but the consequence is that the survivor check is skipped
entirely in exactly that case. Reproduced, holder SIGKILLed, child alive in the same group
as the asking process, grace neutralised to isolate the arm:

```
record pgid=1784782  my pgid=1784782
state after SIGKILL of holder (child alive, SAME pgid, not in nvidia-smi yet): STALE
```

T21 cannot see this because `setsid` puts its victim in a separate group. The exposed
case is a runner relaunched from the same non-interactive parent script — which is the
shape of `run_r4b/r4c/r4d_*.sh`.

**On the 120 s grace.** The builder calls it "a judgement call, not a measurement", and
that is honest. My assessment: in the *separate-pgid* case 120 s is a redundant backstop
(the pgid arm already holds) and its value barely matters. In the *same-pgid* case above it
is the **only** remaining protection, and then 120 s is too short for this project: the
window that must be covered is "child alive but has not yet created a CUDA context", and a
`venv_t2i` python loading fp32 SD-1.4 from `/data` routinely spends minutes before its first
CUDA allocation. Either side of the boundary: **under** 120 s the GPU is correctly refused;
**over** 120 s, with the pgid arm skipped and no CUDA context yet visible, the record goes
STALE and the GPU is handed to a second job while the first is still coming up. Fix the
pgid skip (compare the *record's* pgid against the live holder, or record a session id)
rather than tuning the constant.

## 5. `gpu_reserve_hold` under `set -euo pipefail` — **VERIFIED**

Four runners use `set -euo pipefail`: `run_r1_imma_official.sh:52`,
`run_r1b_imma_train.sh:80`, `run_r1c_eval.sh:40`, `run_r2_meta_official.sh:52`. The other
five use `set -u` only. I tested the exact call shapes those four use, with a freeze marker
present so acquisition returns 4:

```
gpu_reserve_hold "c" 120 >/dev/null || { echo "HANDLER rc=$?"; exit 1; }
   -> BEFORE | HANDLER rc=4           (handler reached, full diagnosis printed first)
gpu_reserve_hold "c" 120 --allow-frozen >/dev/null || { ... }
   -> BEFORE | HANDLER rc=4           (flag alone does not lift the freeze — correct)
gpu_reserve_hold "c" >/dev/null || die "no res"
   -> BEFORE | DIE: no res
gpu_reserve_require "c" 120
   -> BEFORE | (exits; AFTER never printed)
gpu_reserve_hold "c" --allow-frozen >/dev/null; echo "BARE reached"    [not a real shape]
   -> BEFORE | (shell terminated at the bare call; "BARE reached" never printed)
```

The shield is implemented correctly and for the right reason: `case $- in *e*)` plus
`shopt -qo pipefail` at `:994-998` rather than a command substitution (the comment at
`:991-993` explains why `$(shopt -po ...)` both lied and tripped errexit — that reasoning
checks out), restored at `:1120-1121`, with `GPU_RESERVE_RC` exported at `:1114` before the
restore so it survives the errexit path. The bare-call behaviour is `set -e` working as
specified and is documented at `:42-49`. All four real callers guard. **No defect.**

## 6. Position-free `--allow-frozen` — **VERIFIED; the deviation is safe**

All three positions register, and only with the env var:

```
flag-in-wait-slot, NO env    -> refused, 0 override-log lines
flag-in-wait-slot, env SET   -> rc=0, 1 override line
flag 3rd position, env SET   -> rc=0, 1 override line
flag FIRST, env SET          -> rc=0, 1 override line
NO flag, env SET             -> refused, 0 override lines
bad wait_s '120x'            -> rc=2
unknown flag '--bogus'       -> rc=2
too many positionals         -> rc=2
```

and the odd shape is loud, not silent:

```
[gpu_reserve] NOTE: the --allow-frozen flag was supplied in the wait-interval position
[gpu_reserve] FREEZE OVERRIDE in effect for 'camp' (pid ...) -- this is logged.
```

**Judgement: honouring it rather than erroring is safe**, because the flag alone grants
nothing — the freeze gate (`:757`) requires `ALLOW_FROZEN_FLAG=1` **and**
`ALLOW_FROZEN_RUN=1`, and every override is appended to `logs/frozen_override.log`
(`:782-791`). Erroring would have been defensible too, but the deviation does not weaken
the gate, and the `NOTE` preserves the thing that mattered (the shape stops being silent).

**Can the override register by accident? Yes — but through the env var, not the flag.**
`run_v1_esd_alone_control.sh:41` hardcodes `--allow-frozen`. Its comment at :37-39 says
`ALLOW_FROZEN_RUN` "is NOT exported here on purpose, so merely executing this file can
never lift the freeze by itself" — true only while the variable is absent from the
environment. One `export ALLOW_FROZEN_RUN=1` in an interactive session makes every
subsequent `bash run_v1_esd_alone_control.sh` bypass the freeze silently from the
operator's point of view. It is logged and announced on stderr, so it is auditable rather
than invisible. Worth a line in the runner telling the lead to use a one-shot
`ALLOW_FROZEN_RUN=1 bash ...` prefix and never `export`. **Low severity.**

## 7. `run_guard.sh` — **VERIFIED for the guard and for `run_v1`; CLAIM UNSUPPORTED project-wide**

Every behaviour I tested holds, under `set -euo pipefail`:

```
rg_step false   -> returned nonzero and the caller is STILL ALIVE under set -e   (shield at :57-65)
rg_cell false   -> rc=0                                                          (:77-80)
rg_require_file <zero-byte>  -> rejected       (-s test at :86, not -f)
rg_require_file <absent>     -> rejected
rg_require_file <1 byte>     -> accepted
rg_finish after 5 failures   -> rc=1, log says "K INCOMPLETE ... 5 failure(s)", DONE count = 0
rg_finish on a clean run     -> rc=0, exactly 1 DONE line
```

`rg_finish` (`:106-114`) is the only `DONE` writer **inside `run_guard.sh`**, and
`run_v1_esd_alone_control.sh` uses it correctly end to end: `rg_init` (:64) → `rg_cell`
(:67) → `rg_require_file` (:74) → `rg_finish` (:91) as the final command, so the script's
exit status is the guard's verdict. `grep DONE run_v1_esd_alone_control.sh` matches only a
comment. That runner is clean.

**But the guard has exactly one caller.** `grep -rln run_guard.sh` over the project returns
`run_v1_esd_alone_control.sh`, `run_guard.sh` itself, and `test_gpu_reserve.sh`. All eight
other migrated runners write a `DONE` line directly — `run_r1_imma_official.sh:282`,
`run_r1b_imma_train.sh:395`, `run_r1c_eval.sh:174`, `run_r2_meta_official.sh:606`,
`run_r4b_queue.sh:26`, `run_r4c_queue.sh:23`, `run_r4d_upstream_defaults.sh:27`,
`run_e2h_meta_v2_stronger.sh:99`. The four `set -euo pipefail` ones are defensible (errexit
makes the `DONE` unreachable after a failure). The other four are not — see §9.
`run_guard.sh`'s own header is candid that the 104 historical markers are deliberately left
and declared non-evidence; the gap is that the newly-touched runners were not brought in.

## 8. `campaign_free.sh` retirement — **VERIFIED**

Zero live callers project-wide. My own scan for non-comment invocations:

```
$ grep -rnE '^[^#]*(bash|sh|source|\.)[[:space:]]+[^ ]*campaign_free\.sh' --include='*.sh' --include='*.py' .
scripts/test_gpu_reserve.sh:961    (the retirement test itself)
```

All nine runners reference it only in comments (I read each match). And it cannot hand one
out if called anyway — it prints the diagnostic to **stderr** (`:47-58`), delegates to
`gpu_reserve.sh status >&2`, and exits 1 with an empty stdout:

```
$ bash scripts/campaign_free.sh            # with a mocked nvidia-smi
stdout=[]  rc=1          head -1 of stdout = []
```

which is the correct shape for the legacy `G=$(... | head -1); [ -n "$G" ] || sleep` waiters.
No other handout script exists (`scripts/` contains only `campaign_free.sh`,
`gpu_reserve.sh`, `test_gpu_reserve.sh`); the `pick_gpus.sh` referenced in
`docs/STATUS_CURRENT_v2.md:93` is **absent from the tree** and has no callers.

## 9. Per-runner review — **DEFECT FOUND (two kinds)**

**Release traps.** No runner installs its own trap; all nine rely on the one
`gpu_reserve_hold` installs at `gpu_reserve.sh:1110` over `EXIT INT TERM HUP`. That is the
right choice (one owner of the trap) and it works: measured above, SIGTERM releases,
SIGKILL leaves an ORPHAN by design. **No runner can leak a reservation past the orphan
window** on any path I could construct. No runner calls `gpu_reserve_hold` twice on a
reachable path, so the documented trap-replacement leak is not live.

**`CUDA_VISIBLE_DEVICES` after reservation.** `run_r1_imma_official.sh:147,247`,
`run_r1b_imma_train.sh:331`, `run_r1c_eval.sh:88` and `run_r2_meta_official.sh:444` all
re-export `CUDA_VISIBLE_DEVICES="$GPU_RESERVED"` — redundant but identical to what the
helper already set, so harmless. `run_r4b/r4c/r4d/e2h/v1` correctly do not touch it.
**Nobody hand-edits it to a different value.**

**DEFECT A — `run_r1_imma_official.sh:240-249` conditions the reservation on an inherited
`CUDA_VISIBLE_DEVICES`:**

```
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  ... gpu_reserve_hold "r1_imma_smoke" 120 ... ; export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"
fi
```

On the intended path stage 2 (:144) has already reserved and set the variable, so the guard
is a double-acquire preventer. But when stage 2 is skipped — training images already
present, the normal state on any re-run (:129) — and the operator launches with
`CUDA_VISIBLE_DEVICES` inherited from the environment, **stage 4's GPU smoke run executes
with no reservation at all.** The reservation is gated on precisely the variable that
`gpu_reserve.sh`'s defect-4 work exists to distrust (`:545-561`: "a runner that forgets to
export it inherits whatever its parent had and lands on a device nobody reserved"). The
guard should test `${GPU_RESERVED:-}`, not `${CUDA_VISIBLE_DEVICES:-}`.

**DEFECT B — `run_r4b_queue.sh`, `run_r4c_queue.sh`, `run_r4d_upstream_defaults.sh` never
inspect an exit status, and then write `DONE`.** E.g. `run_r4b_queue.sh:21-26`:

```
for lam in 0.5 0.1 1000; do
  venv_uce/bin/python scripts/r4b_uce_upstream_vs_port.py ... >>"$L" 2>&1
done
echo "R4b DONE $(date -Is)" >>"$L"
```

`set -u` only, no `set -e`, no `rc=$?`, no `run_guard.sh`. Every cell can crash and the log's
last line still says `R4b DONE`. Identical at `run_r4c_queue.sh:19-23` and
`run_r4d_upstream_defaults.sh:23-27`. This is exactly the D-012 class `run_guard.sh` was
written for, left in place in three of the five runners the Infrastructure agent owns and
had open. `run_e2h_meta_v2_stronger.sh` does track `INCOMPLETE` by hand (:65-67, :96-99) and
its `break 2` fall-through really is fixed — but by hand rather than through the guard, and
with the `df` hole of §2 in its abort condition.

---

## (a) Guarantees that are mock-only and need a real GPU run

1. **That the reserved NVML index is the device torch then uses.** Everything here is
   index bookkeeping against a shim. `gpu_reserve.sh:547-551` asserts from measurement that
   real `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES` so indices stay physical; I cannot
   re-verify that without the driver, and the whole scheme rests on it.
2. **`GPU_RESERVE_CVD_STYLE=uuid`.** Never exercised by the suite and the header admits it
   has never run on a real GPU. Treat as unverified; leave it opt-in.
3. **The real `FOREIGN_MIB=512` threshold.** Whether a genuinely idle GPU on this host sits
   below 512 MiB, and whether a starting foreign job crosses it before we classify, is a
   property of the hardware, not of the logic.
4. **The CUDA-context arm of ORPHAN.** That a surviving torch child appears in
   `--query-compute-apps` within the grace window is the assumption the shim cannot test —
   and §4 shows it is the assumption the same-pgid case depends on.
5. **Timeout behaviour against a wedged real driver** (`timeout -k 5` → rc 124/137 is
   simulated by `sleep 30` in a shim).
6. **Mutual exclusion under real contention.** T4's 12-way race is real concurrency but with
   a shim; a real campaign holds for hours across driver hiccups.

## (b) What I would block the gate on

1. **`GPU_RESERVE_LOCKDIR` is honoured in production, unrefused (`gpu_reserve.sh:287`).**
   One stray exported variable splits the mutex and restores the exact triple-booking
   failure the file was written to prevent. Hard-pin it to `$_GR_PROD_LOCKDIR` in production
   like the allowlist and the freeze marker. **Blocking.**
2. **`GPU_RESERVE_FOREIGN_MIB` is unvalidated and fail-open (`:146`, `:622`).** A
   non-numeric or large value makes a GPU holding 30 GB of unattributable memory read as
   FREE, in production. Validate it (and `GPU_RESERVE_ORPHAN_GRACE` at `:149`) with the same
   `case` the allowlist gets, refusing with rc=7. **Blocking** — it is defect 1 in a
   different coat.
3. **`run_r1_imma_official.sh:240` gates the stage-4 reservation on an inherited
   `CUDA_VISIBLE_DEVICES`.** On a re-run with that variable set, a GPU job runs unreserved.
   **Blocking** for that runner; one-line fix (`${GPU_RESERVED:-}`).
4. **`run_r4b/r4c/r4d` write `DONE` without checking any exit status.** Either wire them to
   `run_guard.sh` or add `rc` tracking. **Blocking** as a gate item because G2 is being
   reported on the strength of a guard that one runner uses.
5. Non-blocking but fix before the next migration: production should require
   `_gr_nvsmi_is_real` (`:143` vs `:251`); `_gr_do_acquire`'s state `case` (`:819-843`)
   needs a `*)` default that refuses; the `_GR_UUID2IDX` collision (`:472`) and duplicate
   index rows should refuse rather than resolve last-wins; `_gr_live_in_pgid`'s pgid skip
   (`:368`) needs a better survivor test; `run_e2h_meta_v2_stronger.sh:55`'s `df` pipeline
   should capture a status.
6. Documentation: the guarantee at `gpu_reserve.sh:99-101` and in `REPAIR_INFRA_v2.md`
   should be narrowed to "production cannot be widened; test mode can, and its only
   interlock is the telemetry command's filename", which is what the code does.

## (c) Tests I judge to be asserting a falsehood

**None asserts a falsehood.** I specifically re-checked the two v1 precedents: T9
(`:275-289`) now genuinely refuses — it asserts `rc=3` **and** empty stdout, and I confirmed
production returns 6 for `GPU_RESERVE_GPUS="2 3"`; and T21 (`:656-705`) passes for the
stated reason, not via the grace window (grace is pinned to 1 s at `:660`) and not via
nvidia-smi (the child is not in the shim's app list) — it uses `setsid` so the pgid arm is
really what fires. Both are honest now.

Three tests assert less than their names promise, which is the G14 distinction:

1. **`test_gpu_reserve.sh:609-613` (T19c) cannot fail** — both branches call `ok`. It is a
   free PASS inside the 215. Not false, but not an assertion.
2. **T24 (`:845-862`) "no status-losing construct survives in the source" is a grep over the
   source**, not a behavioural claim. It is true as written and useful as a regression
   witness, but it does not support the stronger claim 2 ("every subprocess status is
   checked") — and §2 shows an unchecked one one directory up, in a runner the same repair
   migrated.
3. **T27 (`:967+`) "the five infra-owned runners are migrated" asserts `bash -n`, absence of
   `campaign_free.sh`, and presence of an API string.** All three are true. But the section
   prints "remaining campaign_free.sh CALLERS in the project" as *informational output with
   no assertion attached* — the zero-callers property of claim 8, the headline migration
   result, is the one thing in T26/T27 that is displayed rather than tested. I verified it
   independently (§8) and it holds; it should be a `chk`.
