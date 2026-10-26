# REVIEW_M1 — independent gate review, milestone 1

Reviewer: independent (did not write any of the reviewed code). 2026-10-26. **CPU only. No GPU work
was launched. No git commit. No file was edited except this one.**

The author of `gpu_reserve.sh` also wrote `test_gpu_reserve.sh` and says so plainly in
`docs/REPAIR_A_v3.md:18` ("**independently verified** | *not claimed anywhere in this document.*").
This document is the independent check that was therefore missing.

Calibration used throughout: **a passing test is valid evidence for the property it genuinely
exercises.** I did not discount tests for being author-written; for each one I asked what it drives,
under what conditions, and whether the assertion could have failed. Where a test is a source grep or
runs against a shim, I say so and say what the shim supplies.

---

## 0. Suite re-run — counts I observed myself

```
$ bash scripts/test_gpu_reserve.sh
...
----------------------------------------
RESULT: 318 passed, 0 failed
No GPU was accessed; nvidia-smi was the shim at /data/bijaypandey/tmp/gpu_reserve_test.i7x9yO/fake_nvsmi
SUITE_RC=0
```

**318 passed, 0 failed, exit 0.** That is my own observation, not a quoted figure. Two incidental
`Killed` lines appear from T35c's own `kill -9` teardown of its lock-lifetime probes
(`test_gpu_reserve.sh:1833`, `:1841`); they are the harness killing its own helpers, not failures.

The suite is genuinely CPU-only: telemetry is a shim at `test_gpu_reserve.sh:72-80` driven by a
state file plus `$GPU_SHIM_MODE`. It can therefore be trusted about *decision logic given
telemetry* and must not be read as evidence about *what the driver reports* — which is exactly the
line `docs/REPAIR_A_v3.md:781-782` draws.

The suite does **not** inflate its count: `test_gpu_reserve.sh:50-52` defines a separate `info()`
for lines that cannot fail, with the comment "a line that always passes inflates the pass count and
asserts nothing", and it is used for the untestable SIGINT rows (`:1521`) and for the one-sided
`flock -u` arm (`:1839`). I checked that `info` output is not counted in `PASS`.

---

## 1. The two lead measurements

### Measurement 1 — `--query-compute-apps` lists a `Popen`-spawned CUDA worker: **VERIFIED as
correctly incorporated**

The gate does consult compute-apps on the acquire path, and nothing short-circuits it.

* `gpu_reserve.sh:902` — query 3 `--query-compute-apps=pid,gpu_uuid` is issued **unconditionally**
  inside `_gr_probe`, on every acquire.
* `gpu_reserve.sh:904-910` — a timeout or non-zero rc is `_gr_telemetry_fail` → rc=5. An empty
  result is legitimate, a *failed* result is not. Fail-closed.
* `gpu_reserve.sh:917-920` — a compute-apps row naming a UUID the `index,uuid` query never listed
  is a hard telemetry failure ("the two views disagree"), not a skipped row.
* `gpu_reserve.sh:1001-1015` — `_gr_classify` iterates `_GR_APPS[$i]` and returns `OURS_UNLOCKED`
  for a pid of our uid, `FOREIGN` for anyone else's **and** for a pid not visible in `/proc`
  (hidepid → never displace). `gpu_reserve.sh:1321` refuses on `OURS_UNLOCKED`.
* `gpu_reserve.sh:1093-1100` — the same consult appears independently in
  `_gr_reservation_state`, producing `ORPHAN … cuda-context-pid-<p>-still-on-gpu<i>`.
* `gpu_reserve.sh:1101-1105` — if telemetry is not OK the record is held as ORPHAN, never
  reclaimed. "defect 1 wearing a different hat" is accurate.
* `gpu_reserve.sh:1305`/`:1311-1315` and `:1319`/`:1325-1327` — **both** gates require an explicit
  `FREE`, with `*)` default-deny arms. An unhandled state or classification cannot be claimed.

Two hardening points worth recording, because they are stronger than the write-up claims:

1. The `OURS_UNLOCKED` arm has **no tunable threshold**. `GPU_RESERVE_FOREIGN_MIB` gates only the
   `UNACCOUNTED` *memory* arm (`:1018`); a survivor that appears in compute-apps is refused
   regardless of every numeric knob. The layer that covers D-059 cannot be widened by configuration.
2. A pid that exits between probe and classify yields an empty uid and is treated as `FOREIGN`
   (`:1003-1007`) — i.e. refused. The race resolves in the safe direction.

So the one unverified assumption the occupancy gate rested on is now supplied by the lead's
measurement (child pid 2174848, 492 MiB, listed). The consequence is that
`docs/REPAIR_A_v3.md:782` — "upstream's `Popen` workers are actually VISIBLE to
`--query-compute-apps` | **not verified**" — is now **satisfied** and should be re-marked
*measured (lead, 2026-10-26)*. That is a documentation update, not a code change; the code already
consults the right source.

### Measurement 2 — SIGINT cleanup: **PARTIAL, and I could not settle it either**

The lead measured the right invariant (lock HELD while a grandchild lived, FREE after descendants
exited) but `CLEAN=0`, so the INT **trap** was not shown to have run. I agree that is PARTIAL, and
the reason is sharper than "the flag didn't set":

**Lock HELD→FREE is produced by the kernel whether or not the trap runs.** The flock lives on the
open file description; when the last descendant exits the kernel releases it with no code executing
anywhere. So the lead's observation is *fully consistent with the INT trap never running at all*.
It cannot discriminate.

I attempted to settle it myself, CPU-only with a synthetic shim, driving the runner under a pty via
`script -qc` so SIGINT would not be inherited-ignored. It did not settle it, for a structural
reason: any harness that **backgrounds** the runner gives it `SIGINT` ignored on entry
(`SigIgn` bits 2,3), and bash cannot trap a signal ignored at entry. The pty does not change that —
the backgrounding does. The author documents exactly this at `test_gpu_reserve.sh:1504-1507` and
declares SIGINT untestable from the harness using `info`, not `ok`. That is honest, and I reproduce
the limitation rather than overturn it.

My run left `gpu0.owner` present with the hold lock FREE and **no teardown line on stderr**, which
looks like a leak but is not evidence: the whole session was torn down when `script` exited, so the
runner never got a trappable SIGINT. **Inconclusive, reported as such.**

**What would settle it.** Three things, and the first is the decisive one:

1. **Assert the trap's own side effect, not the lock state.** `gpu_reserve_drop` is the *only*
   emitter of either `gpu<i> released (campaign '<c>', pid <p>)` (`gpu_reserve.sh:1377`) or
   `NOT releasing the reservation for '<camp>': the liveness lock … is STILL HELD`
   (`gpu_reserve.sh:1772-1775`). One of those two lines on stderr is positive proof the INT trap
   ran. Neither can be produced by the kernel.
2. **Exit status is not sufficient on its own.** `gpu_reserve.sh:1711` exits 130, but a shell
   *killed* by SIGINT with no handler also reports 130. Status 130 corroborates; the stderr line
   discriminates.
3. **Run it in the foreground of an interactive shell** (job control on, so SIGINT is not
   inherited-ignored), with a descendant deliberately placed in a *different* process group
   (`setsid sleep 300`) so it survives the Ctrl-C. Then both arms are measurable in one run:
   the record must be **KEPT** and `bash scripts/gpu_reserve.sh holders <index>` must name that pid;
   after killing it, the next `acquire` must reclaim. Repeat with no survivor: the record must be
   **ABSENT** and the `released` line must appear.

This is a ~2-minute interactive check. It is the only item in this review that needs a human at a
terminal, and it does not need a GPU.

---

## 2. Specific claims

### (a) The `close_fds` hole — **VERIFIED**; T35a/b genuinely pin it (with one stated premise)

The mechanism is real and the author does not hide it: `test_gpu_reserve.sh:1716-1721` and
`REPAIR_A_v3.md:798-803` both state that `shell → exec python` keeps the descriptor and
`python → subprocess.Popen` drops it (`close_fds=True` since 3.2), that `pass_fds=` is unavailable
because upstream's calls are not ours, and that the lock is therefore **not** the only gate.
`REPAIR_A_v3.md:780` marks the lock-only row **DISPROVED** rather than quietly dropping it. That is
the honest handling of a disproved claim.

T35a (`test_gpu_reserve.sh:1728-1788`) **genuinely pins the behaviour** rather than asserting it:

* it constructs the real configuration — a live process of ours (`:1732`) holding **zero** liveness
  descriptors, with a dead owner's record naming a hold file that is **free**;
* it asserts both preconditions, and both could fail: `:1757-1761` tests the lock really is free
  (`flock -n -x … -c true`), `:1762-1763` counts the survivor's descriptors against `/proc/$SURV/fd`
  and requires `0`. If the descriptor arm could have saved this case, these two fail;
* it then runs the **real `acquire` binary end to end** (`:1774`) and asserts gpu0 was not handed
  out (`:1776-1780`) and that the record was **kept** (`:1781-1785`). That is a behavioural check of
  the decision, not a grep and not an assertion about an internal variable.

I observed all seven T35a assertions pass, including
`state is ORPHAN via the CUDA-OCCUPANCY arm, not the lock (ORPHAN 999999 q3camp
cuda-context-pid-2190761-still-on-gpu0)` and
`acquire REFUSED gpu0 and went elsewhere (gpu1)`.

**The one premise T35a supplies by mock:** the survivor is `sleep 120`, and its CUDA context is
simulated by an `app <pid> GPU-aaaa` row in the shim state file (`:1738`). T35a therefore proves
*"given that compute-apps lists the survivor, acquire refuses"*. Whether the driver really lists it
is measurement 1 — which the lead has now taken. **T35a + lead measurement 1 together close D-059;
neither does alone.** `REPAIR_A_v3.md:781` says precisely this ("with the driver's compute-apps row
mocked; the surviving process, its pid and its zero descriptors are real"). No overclaim.

T35b (`:1790-1806`) covers the weaker variant — memory occupied with **no** compute-apps row at all
(hidepid, MIG, driver quirk) — and asserts `UNACCOUNTED` plus a real `acquire` refusal. Both passed.
This matters: it is the layer that holds if measurement 1 had gone the other way.

### (b) `flock -u` confined to the mutex — **VERIFIED**

I grepped the whole reviewed set myself:

```
$ grep -n 'flock -u' scripts/gpu_reserve.sh scripts/run_guard.sh run_r1*.sh run_v*.sh
scripts/gpu_reserve.sh:1168:  flock -u 9 || _gr_log "WARNING flock -u failed ..."
# every other hit is a comment (:1508, :1511, :1512, :1823 is in the test file)
```

Exactly **one** `flock -u` call site, `gpu_reserve.sh:1168`, on **fd 9 — the reservation mutex**,
whose lifetime is deliberately not shared. No `flock -u` exists in `run_guard.sh` or in any of the
five launchers. The hold descriptor is released only by closing it, via `_gr_close_hold_fd`
(`:1514-1521`), which closes with `eval "exec ${_GR_HOLD_FD}>&-"` after an `/proc/$$/fd` existence
test — no `2>/dev/null` on the `exec`, which is the point of (f4) below.

The source-level guard is real but secondary, and correctly so: `test_gpu_reserve.sh:1847-1853` is
an `awk` pass that strips the line-number prefix *before* testing for a comment (the comment at
`:1843-1846` records that the naive `grep -n` version counted the file's own comments as violations,
expected 0 got 4 — an author-caught defect in the guard itself). It is a **regression guard layered
on top of** the two behavioural measurements in T35c, not a substitute for them, and T35c measures
both directions on this host: closing preserves the lock while a child shares it (`:1828-1832`),
and an explicit `flock -u` does release it for every sharer (`:1836-1840`). Both passed in my run.

Minor, non-blocking: T35c's `flock -u` arm is one-sided — the negative branch at `:1839` is `info`,
so if `flock -u` had *not* released the lock the test would record no assertion. It is correctly
labelled `info` rather than inflated to `ok`, and on my run the positive branch fired, so the
property was measured.

### (c) D-047 env refusals — **VERIFIED for the `GPU_RESERVE_*` namespace**, but **DEFECT FOUND: `PROJECT` is not swept**

The GPU_RESERVE_* surface is solid. My own probes, every one run by me:

```
GPU_RESERVE_LOCKDIR=/tmp/evil                                  rc=6
GPU_RESERVE_NVSMI=/bin/true                                    rc=6
GPU_RESERVE_OVERRIDE_LOG=/dev/null                             rc=6
GPU_RESERVE_FREEZE_FILE=/tmp/nope                              rc=6
GPU_RESERVE_GPUS=2 3                                           rc=6
GPU_RESERVE_GPUS=0 1 2 3                                       rc=6
GPU_RESERVE_TEST_MODE=1                                        rc=6
GPU_RESERVE_FOREIGN_MIB=abc                                    rc=7
GPU_RESERVE_FOREIGN_MIB=99999999                               rc=7
GPU_RESERVE_FOREIGN_MIB=            (set but empty)            rc=7
GPU_RESERVE_FOREIGN_MIB=0                                      rc=7
GPU_RESERVE_ORPHAN_GRACE=0                                     rc=7
GPU_RESERVE_ORPHAN_GRACE=-5                                    rc=7
GPU_RESERVE_NVSMI_TIMEOUT=0                                    rc=7
GPU_RESERVE_WAIT=abc                                           rc=7
GPU_RESERVE_UID=root                                           rc=7
GPU_RESERVE_CVD_STYLE=bogus                                    rc=7
GPU_RESERVE_TEST_MODE=yes                                      rc=7
GPU_RESERVE_GPUS=              (set but empty)                 rc=7
GPU_RESERVE_GPUS="   "         (whitespace only)               rc=7
GPU_RESERVE_GPUS=0 0           (duplicate)                     rc=7
GPU_RESERVE_GPUS=0 x           (non-numeric)                   rc=7
```

All 22 behave as documented. The two-sided bound on `FOREIGN_MIB` is the right call and the reason
given at `gpu_reserve.sh:195-197` — "an out-of-range-HIGH value disables the check it exists to
perform, which is a fail-open in numeric clothing" — is correct reasoning, not decoration. Keeping
rc=7 non-retryable and distinct from rc=3 (`gpu_reserve.sh:1669-1677`) genuinely fixes the
sleep-forever-on-a-config-error failure.

#### DEFECT FOUND — D-047-R: `PROJECT` relocates the lockdir, the mutex, the freeze marker and the audit log, in production

`gpu_reserve.sh:133`

```sh
PROJECT="${PROJECT:-/data/bijaypandey/t2i_unlearn}"
```

and `:140-144`

```sh
_GR_PROD_LOCKDIR="$PROJECT/.gpulocks"
_GR_PROD_FREEZE="$PROJECT/FREEZE_ACTIVE"
_GR_PROD_OVERRIDE_LOG="$PROJECT/logs/frozen_override.log"
```

The three "hard-wired production constants" are all derived from an **environment variable**. Every
D-047 refusal compares the requested value against `$_GR_PROD_LOCKDIR` / `$_GR_PROD_FREEZE` /
`$_GR_PROD_OVERRIDE_LOG` (`:438`, `:469`, `:485`, `:500`) — i.e. against a value the same
environment already moved. The guard compares the attacker's input to the attacker's input.

Measured by me, with the project **frozen**:

```
$ PROJECT=$SCRATCH/evil bash scripts/gpu_reserve.sh status
mode=production
allowed=0 1  off-limits=everything else  lockdir=$SCRATCH/evil/.gpulocks
FREEZE: none ($SCRATCH/evil/FREEZE_ACTIVE absent)

$ bash scripts/gpu_reserve.sh status          # control
mode=production
allowed=0 1  off-limits=everything else  lockdir=/data/bijaypandey/t2i_unlearn/.gpulocks
FREEZE: ACTIVE (/data/bijaypandey/t2i_unlearn/FREEZE_ACTIVE)
```

Mode stays `production`, so none of the test-mode interlocks apply, and the freeze marker has
simply vanished. Then, to confirm the refusal is genuinely bypassed rather than merely mis-reported
— I set `GPU_RESERVE_FOREIGN_MIB=1` (a legal value) so that the idle 23/34 MiB on GPUs 0/1 would
classify as `UNACCOUNTED` and **no reservation could complete**:

```
$ GPU_RESERVE_FOREIGN_MIB=1 bash scripts/gpu_reserve.sh acquire reviewprobe $$
[gpu_reserve] REFUSED: the project is FROZEN. No reservation will be granted.
A_rc=4

$ PROJECT=$SCRATCH/evil GPU_RESERVE_FOREIGN_MIB=1 bash scripts/gpu_reserve.sh acquire reviewprobe $$
[gpu_reserve] inherited CUDA_VISIBLE_DEVICES: <unset> (clean)
[gpu_reserve] gpu0 holds 23 MiB we cannot attribute to our own pids -- REFUSED
[gpu_reserve] gpu1 holds 34 MiB we cannot attribute to our own pids -- REFUSED
B_rc=3
```

rc=**3**, not 4. The freeze gate (`_gr_freeze_gate`, `gpu_reserve.sh:1188-1215`) returned 0; the run
went past it, through the CVD audit, and all the way into the GPU scan. **With `FOREIGN_MIB` at its
default 512 it would have reserved gpu0** on a frozen project. GPUs 0 and 1 were idle at the time
(23 MiB / 34 MiB, zero compute-apps rows), so that outcome was available. I deliberately did not
let it complete. Production's lockdir holds no record I created.

Three distinct guarantees fail from one variable:

* **D-013 (freeze gate) is fully defeated.** This is the control currently stopping all GPU work. A
  single env var turns `acquire` from rc=4 into a granted reservation, with no `--allow-frozen` and
  no `ALLOW_FROZEN_RUN=1`. The design intent at `gpu_reserve.sh:1183` — "in production the marker
  PATH is a constant, so `GPU_RESERVE_FREEZE_FILE` cannot be used to fake a thaw" — is false as
  written: the path is a constant *expression*, not a constant.
* **D-001 (triple-booking) is reachable again.** The mutex is `$LOCKDIR/.reserve.mutex` (`:526`).
  Two runners with different `PROJECT` take different mutexes and cannot exclude each other, and
  cannot even see each other's records. That is verbatim the failure
  `gpu_reserve.sh:480-484` describes as "precisely the 2026-10-26T12:39 triple-booking of GPU 0 that
  this entire file exists to prevent". My probe created a live second mutex at
  `$SCRATCH/evil/.gpulocks/.reserve.mutex`.
* **The freeze-override audit log is redirectable**, so an override can happen with no record in
  `logs/frozen_override.log` — the exact harm `:500-505` refuses `GPU_RESERVE_OVERRIDE_LOG` to
  prevent.

This also **contradicts a stated claim**, which is why I am grading it a defect rather than a gap.
`gpu_reserve.sh:155-156`:

```
# After the empty-allowlist fail-open was fixed, a reviewer found the SAME pattern in two more
# knobs, so this is now a sweep of EVERY variable the file reads rather than per-variable patches.
```

It is not a sweep of every variable the file reads. `PROJECT` is read at line 133, before the sweep
begins, and is the highest-leverage variable in the file. `_gr_env_init` (`:203-246`) validates
seven `GPU_RESERVE_*` knobs and never mentions it.

Mitigating facts, stated fairly: nothing in the repo exports `PROJECT` — I grepped every `*.sh`
outside `adapters/`, and `env.sh` does not set it — so this is not currently *being* triggered. But
`PROJECT` is a generic name very likely to be inherited from a shell profile, a CI runner, a
`direnv`, or another tool, and the failure is **silent**: `status` cheerfully prints
`FREEZE: none`. A guard that reports "not frozen" on a frozen project because of an ambient variable
is the same class of fault as the empty-allowlist fail-open this file already fixed twice.

**Minimal fix:** make the three production constants literals, independent of `PROJECT` (or, if
`PROJECT` must stay configurable, run it through the same present-but-unusable sweep and refuse
rc=6 when it differs from the hard-wired root while mode is `production`).

Also noted, not a defect: `TMPDIR` is read at `:292` and `:811` for `mktemp -d`. It only relocates
scratch space for the interlock and probe; it cannot move a lock or a gate, and a failed `mktemp`
fails closed (`:292-293` → refuse; `:841` → rc=5). No action needed, but it belongs in the sweep's
documented inventory.

### (d) D-048 test-mode interlock — the documented attacks **hold**; one undocumented class **defeats it**

The device-identity interlock works exactly as claimed against the attacks the author names, and I
verified the ones they claim plus one they do not:

```
attack 1: plain `cp` of the real nvidia-smi
  rc=6  REFUSED: ... the configured telemetry source reports device UUID ... which THIS
                 MACHINE'S REAL DRIVER also reports.
attack 2: a wrapper script -- #!/bin/bash / exec nvidia-smi "$@"   (my addition)
  rc=6  REFUSED: ... reports device UUID ...
```

The wrapper case is worth recording because it is the obvious next move after the copy and the
author did not test it: it is refused for the right reason, and by identity rather than by name.
`_gr_nvsmi_is_synthetic` (`:290-352`) is also careful in a way that matters — it issues the **same**
query form the rest of the file depends on (`:302-304`), so a shim cannot dodge the interlock by not
implementing a fourth query. The cheap name filter `_gr_nvsmi_is_real` (`:276-287`) is correctly
labelled "cheap first filter only; NOT the interlock".

#### CLAIM UNSUPPORTED (bounded) — a *stateful* shim defeats the interlock

`gpu_reserve.sh:313-325` grants test mode when the configured source exits non-zero:

```sh
  $NVSMI --query-gpu=index,uuid --format=csv,noheader >"$tmp/cand" 2>/dev/null
  rc=$?
  if [ "$rc" != 0 ]; then
    ...
    _gr_log "test-mode interlock: the configured telemetry source enumerated nothing (rc=$rc), so"
    _gr_log "        it cannot be describing real devices. Test mode allowed; acquisition will"
    _gr_log "        still refuse on unreadable telemetry (rc=5)."
    return 0
```

The reasoning holds only for a **stateless** broken source. A source that fails its *first*
invocation and serves real data afterwards gets test mode **and** working telemetry. The interlock
and the probe issue byte-identical argv, so the source cannot be made to distinguish them — but it
does not need argv: an invocation counter suffices. Measured by me:

```
$ GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="2 3" GPU_RESERVE_LOCKDIR=$S/lk \
  GPU_RESERVE_NVSMI=$S/stateful bash scripts/gpu_reserve.sh status
mode=test (synthetic nvidia-smi '.../stateful', lockdir .../lk)
telemetry=OK  gpus=[3 2 1 0] via .../stateful
allowed=2 3  off-limits=everything else  lockdir=.../lk
rc=0
```

Test mode granted against the **real** driver, with GPUs **2 and 3** — another user's, in use — in
the allowlist and telemetry reporting OK. That is the precise outcome the interlock exists to
prevent, and the author's claim at `gpu_reserve.sh:269-271` ("cannot be forged by copying,
renaming, wrapping or PATH games") is true as written but is cited in `REPAIR_A_v3.md:645-648` as
though the interlock were unforgeable in general. It is not.

**Severity: low, and I am not making it a gate blocker.** The attack requires writing an arbitrary
executable and pointing `GPU_RESERVE_NVSMI` at it — a capability that already permits
`CUDA_VISIBLE_DEVICES=2 python train.py` directly, with no reservation involved. It is a defeat of a
stated absolute, not a privilege escalation, and it is not an accident shape. Contrast with the
`PROJECT` defect, which needs only an ambient variable and no intent at all.

**Suggested hardening (cheap, one line of intent):** treat a non-zero rc from the configured source
as **refuse** (rc=6) rather than allow, and reach the rc=5 fail-closed path by a different route.
The author's stated reason for the leniency — "refusing here instead would mask every
telemetry-failure path behind rc=6 and make the fail-closed behaviour untestable" (`:319-321`) — is
a real concern, but the suite already drives telemetry failures through `$GPU_SHIM_MODE` with a shim
whose UUIDs are synthetic, so test mode is granted on identity, not on failure. I believe the
leniency arm is unnecessary for testability. Worth the author's judgement, not mine to impose.

One thing I did verify as correct and want to credit: test mode **still honours the production
freeze marker** unless `GPU_RESERVE_FREEZE_FILE` is explicitly set (`:456` defaults it to
`$_GR_PROD_FREEZE`). My first pty attempt was refused rc=4 by the real `FREEZE_ACTIVE` while in test
mode, which is the right default and caught me out in the correct direction.

### (f) The five self-inflicted defects — **VERIFIED fixed in code, not merely documented**

I checked the two the brief singles out, behaviourally, and confirmed the code shape of the others.

**(f4) the control that silenced its own failure diagnosis — genuinely fixed.** The root cause is
real and subtle: `exec {FD}>&- 2>/dev/null` applies the redirection to the **shell**, permanently.
`gpu_reserve.sh:1514-1521` now routes every close through `_gr_close_hold_fd`, which tests
`/proc/$$/fd/$_GR_HOLD_FD` instead of suppressing errors:

```sh
_gr_close_hold_fd() {
  [ -n "${_GR_HOLD_FD:-}" ] || return 0
  if [ -e "/proc/$$/fd/$_GR_HOLD_FD" ]; then
    eval "exec ${_GR_HOLD_FD}>&-"
  fi
```

I grepped every `exec`-with-redirection site in all reviewed files for a trailing `2>/dev/null`:
there is none. The open site at `:1646` (`exec {_GR_HOLD_FD}>>"$_GR_HOLD_PATH"`) is clean, and the
file is pre-created by a plain `: >>` at `:1643` whose failure *can* be caught — which is the
correct way to get the error handling without the redirection. The two `exec 9>&-` sites
(`:1163`, `:1169`) are bare. The suite's own evidence that this is load-bearing is credible: the
author reports 12 assertions flipping on it (`:1636-1638`) and two catching the close sites
(`:1498-1499`), and I observed the current assertions
`and the refusal was EXPLAINED, not silent` and `rc=4 not explained`-class checks pass in T34.

**(f5) the diagnostic that held the mutex for 2,247 forks — genuinely fixed, and I re-measured it.**
`_gr_holders` (`:664-679`) is now a single `find /proc/[0-9]*/fd -lname "$p" -printf '%h\n'` wrapped
in `timeout 10`. My own measurement on this host:

```
$ time ( source scripts/gpu_reserve.sh; _gr_holders $SCRATCH/hp )
rc=1
real  0m0.177s
$ ls /proc/[0-9]*/fd | wc -l
2460
```

177 ms wall against 2,460 descriptors that the old per-descriptor `readlink` would have forked on
once each. The `timeout` wrapper is the part that matters for the invariant: a pathological `/proc`
can no longer hold the reservation mutex, because `_gr_holders` is reached from
`_gr_reservation_state` (`:1066`) **inside** the flock-held critical section. The architectural
point is also now correct — the allow/refuse decision is the kernel's in `_gr_hold_free`
(`:640-651`), and `:653-654` says so explicitly: "DIAGNOSTIC ONLY -- the allow/refuse decision is
made by the kernel in `_gr_hold_free`, never by this scan." A diagnostic that cannot change a
decision and cannot block one is the right shape.

The remaining three are fixed in code and visible at their sites: the `--allow-frozen`-bound-to-
`wait_seconds` bug is now position-free parsing with a loud `NOTE` when the flag lands in the
interval slot (`:1552-1612`); the unconditional teardown `release` that deleted a record while a
child was training is now `gpu_reserve_drop` with the close-then-test-then-delete ordering
(`:1756-1784`); and the signal traps now **terminate** rather than running a handler and falling
through (`:1711-1713`, `exit 130/143/129`), with `trap - EXIT` first so teardown does not run twice.
T34's `SIGTERM terminated the runner instead of falling through to the next command` passed in my
run.

One further defect the author introduced and caught is worth crediting because it is the same
pattern one level up: the `flock -u` **source guard itself** was initially wrong — `grep -n`'s
`123:` prefix made every comment look like code, so it counted the file's own explanatory comments
as violations (`test_gpu_reserve.sh:1843-1846`, "expected 0, got 4"). Catching a false *positive*
in one's own guard is the behaviour this repair cycle was supposed to produce.

**Assessment of §6.6's separation (`REPAIR_A_v3.md:761-788`): honest.** I went looking for
overclaims and did not find one in the table. Specifically: SIGINT is marked **NOT measured** with
the mechanism given (`:775`); the lock-only row is marked **DISPROVED** rather than deleted
(`:780`); the `close_fds` row states its mock boundary explicitly (`:781`); the compute-apps
visibility row is marked **not verified** and named as "the occupancy gate's one assumption"
(`:782`); `CVD_STYLE=uuid` is marked **never executed** with the consequence drawn (index stays the
default, `:784`); and the status-word table at `:11-19` reserves "independently verified" and says
it is *not claimed anywhere in the document*. The `flock -u` row (`:783`) correctly describes the
grep as "plus a source-level assertion", i.e. additive. I found no row where a shim result is
presented as a driver result.

The one correction §6.6 needs is in the *other* direction: `:782` is now satisfied by the lead's
measurement and understates what is known.

The honesty gap is not in §6.6 — it is at `gpu_reserve.sh:155-156`, where "a sweep of EVERY variable
the file reads" is factually wrong (see (c)).

### (e) The brief's correction about the K1 gate — **VERIFIED, and the author is right on all three points**

I checked each of the three corrections independently.

**1. "The 33 `|| echo 0` sites are not in the resumed path" — correct, and stronger than stated.**

```
$ grep -c -- '|| echo 0' run_r1c_eval.sh
0
$ grep -rc -- '|| echo 0' --include=*.sh . | awk -F: '{s+=$2} END{print s}'
41
$ grep -rln -- '|| echo 0' --include=*.sh . | head
run_utrap.sh  run_uce_sd15_lam.sh  run_wsr2.sh  run_utrap_crossmodel.sh
run_utrap2.sh  run_wsr2_seeds.sh  run_gift_b13.sh  run_gift_crossmodel.sh ...
```

There are **zero** `|| echo 0` sites in `run_r1c_eval.sh` and zero in any of the five reviewed
launchers or in `run_guard.sh`. All 41 project-wide live in older campaign scripts
(`run_utrap*.sh`, `run_wsr*.sh`, `run_gift*.sh`, `run_uce*.sh`) that are not on the milestone path.
The brief's premise was wrong about the file it named.

**2. "r1c's gate does not zero on a crash (errexit catches it)" — correct.** `run_r1c_eval.sh:40`
is `set -euo pipefail`, and the metric is taken as `R8=$(rg_metric …) || { … exit 1; }`
(`:350-356`). There is no substitution path.

**3. The real bug was the opposite, and the author's diagnosis is right.** The retired code is
preserved as a comment at `run_r1c_eval.sh:326-339`:

```
#   awk -v r="$R8" 'BEGIN{exit !(r>=0.60)}' || { echo "K1 fail (retain8=$R8)..."; exit 0; }
#   json null  -> R8="None" -> awk compares STRINGS ("None" >= "0.60" is lexically true) -> CLEARS
```

That is the correct account of the mechanism. `awk`'s untyped comparison makes `"None"`, `"NaN"`
and `"inf"` all compare **greater** than `"0.60"` lexically, so a malformed metric **passed** the
utility floor and launched the three-seed attack grid. It is the opposite of the briefed defect, and
it is the more serious one: the old code's `|| { … exit 0; }` also conflated "unmeasured" with
"measured and below floor", giving both the clean `exit 0`.

**Is the fix complete? For the gate that controls execution: YES.**

```
$ awk '/awk/{l=$0; sub(/^[[:space:]]+/,"",l); if (l ~ /^#/) next; print NR": "l}' run_r1c_eval.sh
(no output)
```

**Zero live `awk` invocations remain in the file** — all four hits are inside the explanatory
comment block. So there is no numeric gate anywhere in `run_r1c_eval.sh` that compares without
validating. The replacement (`:357-363`) is sound in both halves:

```sh
# Numeric comparison only, on a value already proven numeric. -v is kept out of it deliberately.
if ! venv_t2i/bin/python -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 0.60 else 1)' "$R8"; then
```

and `R8` can only be a finite float in `[0,1]`, because `rg_metric`
(`scripts/run_guard.sh:289-326`) rejects, **before writing anything to stdout**: `FileNotFoundError`
→ ABSENT, any other open/parse error → UNREADABLE, a non-dict mid-path → NOT-A-DICT, a missing key
→ MISSING FIELD, `bool` or non-`(int,float)` → NON-NUMERIC, `not math.isfinite` → NON-FINITE, and
out-of-`[lo,hi]` → OUT OF RANGE. On any of those it prints nothing and returns 1. So `null` →
NON-NUMERIC, `NaN`/`Infinity` → NON-FINITE; all three take the `:350` abort branch and `exit 1`.
The two outcomes are correctly separated — unusable metric → non-zero exit with DONE refused;
genuine sub-floor → `rg_finish; exit 0` as a scientific result. **A malformed metric can no longer
launch the attack grid.**

**But two comparisons in the same file are still unvalidated.** Neither can launch work, so these
are not a reopening of the defect — they corrupt the *log*, which in a project whose failure mode is
misleading evidence still matters.

* `run_r1c_eval.sh:320` — a K1 verdict printed from raw JSON, *before* the real gate:
  ```python
  print(f"[R1c] K1: {'CLEAR -> attack grid' if am['legacy_retain8']>=0.60 else 'FAIL -> not attacked'}")
  ```
  With `legacy_retain8 = Infinity` this writes **`[R1c] K1: CLEAR -> attack grid`** into the log for
  a non-finite metric. The real gate at `:358` still aborts, so nothing launches — but the log now
  contains a CLEAR verdict that the run did not earn. With `null` it raises `TypeError` and kills
  the script with only a traceback.
* `run_r1c_eval.sh:313` — the label-collapse detector:
  ```python
  if len(zeros_nt) >= need and am["target_suppression_acc"] > 0.5:
  ```
  No presence or finiteness check. `NaN > 0.5` is `False`, so a NaN target score makes the script
  print `collapse signature: not present at the derived threshold` — a *false reassurance* derived
  from an unmeasured value, on the quarantine question. `:306`'s `v["clip_acc"] == 0.0` is likewise
  unguarded (a missing key raises `KeyError`).

For completeness, the gate inventory for this file: K1 utility floor (**validated**), collapse
detector (**not validated**), `:320` preview verdict (**not validated**). There is **no** FID, CLIP
or ASR threshold gate in the file, so nothing else to check.

`REPAIR_A_v3.md:786` claims "malformed metric cannot pass the K1 gate | **measured**, all seven
cases (§5.4)". That claim is true as scoped — it is about the K1 gate, which is the one that gates
execution. It is not an overclaim, but it does leave `:313`/`:320` uncovered.

### (g) The launchers — **mostly VERIFIED, with four DEFECTS FOUND**

**Reservation consumption.** All five use Form A and all five are contract-compliant on the refusal
path. Each sources the helper and guards the call with `|| { …; exit }`, which is the documented
shape (`gpu_reserve.sh:47-48`): `run_r1_imma_official.sh:173-175`, `run_r1b_imma_train.sh:377-379`,
`run_r1c_eval.sh:203-205`, `run_v1_esd_alone_control.sh:40-45`, `run_v2_rescore_final.sh:44-49`.
**No launcher can proceed past a refused reservation** — every `||` branch exits. None uses Form B,
none hardcodes a GPU index, and none calls the retired `campaign_free.sh` (the only mentions are
comments recording its retirement: `r1:167`, `r1b:367`, `r1c:191`). T34's
`a refused reservation yields a NON-ZERO runner status (rc=4)` and `no work ran after the refusal`
both passed in my run.

**Artifact validation — this is the strongest part of the launcher work. All five verify.** Nothing
relies on `[ -f ]` alone for a result:

| launcher | validation |
| --- | --- |
| `run_r1_imma_official.sh` | `:321` `[ -s "$CK" ]`; `:322-335` `torch.load` + `len(sd)==160`, `n==93_536_640`, `torch.isfinite(...).all()`. `:206-266` opens **every** PNG and exits 1 below `MIN_IMAGES`. `:127` `torch.load`s the ESD input. |
| `run_r1b_imma_train.sh` | `:411` `[ -s "$CK" ] \|\| die`; `:413-473` `torch.load`, scope/param asserts, `isfinite` (`:429`), key-set equality (`:430`), and a **no-op detector** `assert r > 1e-5` (`:438`) — a training run that changed nothing is refused. |
| `run_r1c_eval.sh` | `unet_loads()` `:83-93` (torch.load + non-empty + finite), called from `compose_is_current` `:108` and re-run post-compose at `:217`, `:237`; `gen_*.json` parsed `:136`; `rg_manifest:389` re-opens every `gen_`/`atk_` artifact. |
| `run_v1_esd_alone_control.sh` | `rg_require_loadable … json` `:76`; `rg_manifest` `:93`. |
| `run_v2_rescore_final.sh` | `rg_require_loadable "${LATEST}run.json" json` `:75`; `rg_manifest` `:80`. |

`rg_require_loadable` (`run_guard.sh:231-283`) genuinely loads — `torch.load` / `json.load` /
`PIL.open` — and rejects zero-tensor and non-finite checkpoints. T34b
(`rg_finish refuses DONE when a required artifact does not LOAD`, `the log says INCOMPLETE, never
DONE`) passed in my run and is a behavioural check, not a grep. **Zero-byte and unloadable artifacts
are genuinely refused.**

One gap: `run_r1c_eval.sh:366-372`'s attack-skip path decides reuse from `[ -f "outputs/atk_${K}.json" ]`
plus `grep -q "$A_SHA"` over a sidecar. The `atk_*.json` is not parsed there (only later by
`rg_manifest`), and `grep -q` is a substring match rather than a field comparison.

**`run_guard.sh` — what it is, and who calls it.** A sourceable library (`set -u`, `:32`) whose one
purpose is that a runner cannot print DONE after a failure. The key design decision is right and the
author found it the hard way: the failure ledger is an **on-disk file**, not just a shell counter,
because `V=$(rg_metric …)` performs its increment in a subshell where it vanishes — "Measured: 4
induced failures, only 2 counted" (`:44-50`). `rg_finish` (`:441-451`) is the only writer of DONE and
reads the ledger. `rg_metric` never fabricates a value. `rg_atomic_write` (`:193-227`) fsyncs the
bytes, `os.replace`s within the same directory, then fsyncs the directory — correct ordering.

**Who calls it:**

* calls it: `run_v1_esd_alone_control.sh:63`, `run_v2_rescore_final.sh:59`, **`run_r1c_eval.sh:348`**,
  plus `run_r4b_queue.sh:26`, `run_r4c_queue.sh:21`, `run_r4d_upstream_defaults.sh:43`.
* **does NOT call it: `run_r1_imma_official.sh` and `run_r1b_imma_train.sh`.**

**On the two that run next, stated plainly:**

* **`run_r1b_imma_train.sh` does not call `run_guard.sh` at all.** It ends with an unguarded marker,
  `:474 say "R1b DONE …"`. It is nonetheless *truthful*, by a different mechanism: `set -euo pipefail`
  (`:80`) plus `[ "$rc" -eq 0 ] || die "training exited $rc"` (`:407`),
  `[ -s "$CK" ] || die` (`:411`) and `|| die "stage 5: the artifact failed verification"` (`:413-414`)
  all precede that line. So the DONE cannot be reached after a failed stage. What it lacks is a
  machine-readable completion manifest and a failure ledger — the DONE is a log line, not evidence.
  Acceptable for M1; worth closing before this becomes a cited result.
* **`run_r1c_eval.sh` does call it, but only from line 348** — after the quarantine, both
  compositions and both evaluations have already run, and `rg_init "R1C"` at `:349` truncates the
  ledger. So an `R1C DONE` line attests to the K1 gate, the attack grid and the manifest, and **not**
  to stages 1-2. Those stages do `exit 1` directly on failure (`:216`, `:218`, `:226`, `:236`,
  `:238`, `:277`), which is adequate — but the guard's coverage is narrower than the file's framing
  suggests.

#### DEFECT FOUND — g1: `rg_manifest` reports success when the publish failed, and `rg_finish` then writes DONE

`scripts/run_guard.sh:421-437`

```sh
  "$RG_PY" - "$dest" "$RG_ID" $entries <<'RGPY' | rg_atomic_write "$dest"
  ...
RGPY
  echo "manifest published :: $dest" >>"$RG_LOG"
  return 0
}
```

**The pipeline's exit status is never captured.** `rg_atomic_write` has three documented failure
returns (`:200` cannot buffer, `:206` empty payload, `:224` publish failed), and the hashing Python
at `:426-429` can raise (`open(p,"rb")` on a file removed between validation and hashing). In every
one of those cases `rg_manifest` still logs `manifest published`, records **no** failure, and
returns 0.

It is reachable on the path that runs next. `run_r1c_eval.sh:389` calls it inside an `if`:

```sh
if rg_manifest "outputs/r1c_completion.json" $R1C_ARTS; then
  echo "R1c manifest published: outputs/r1c_completion.json" >>"$L"
```

An `if` condition **suppresses errexit for the entire function body**, so the caller's
`set -euo pipefail` does not catch the failing pipeline either. The `if` takes the success branch,
the log says the manifest was published, and `rg_finish` (`:394`) then writes `R1C DONE` — with no
manifest on disk. `run_v1_esd_alone_control.sh:93` and `run_v2_rescore_final.sh:80` have the same
exposure and only `set -u`, so there is no errexit to lose.

This is precisely the marker-as-completion class the function's own header (`:175-177`) claims to
close, in the one function whose job is to be the completion evidence. **Blocking:** `r1c` is on the
immediate path and this is a DONE that can outrun its own artifact.

Related, same function: `rg_atomic_write` buffers with `cat > "$tmp"` (`:198`). If the generator
writes some JSON and then dies, `cat` succeeds, the file is non-empty so the `[ ! -s ]` refusal at
`:204` does not fire, and a **truncated JSON document is published**. `rg_manifest` never re-parses
what it wrote, so the manifest is the one artifact in the system exempt from `rg_require_loadable`.

#### DEFECT FOUND — g2: `run_r1_imma_official.sh:194` is a guard that cannot fail

```sh
  gpu_reserve_drop "r1_imma_prep" || say "    WARNING: releasing the prep reservation failed"
```

`gpu_reserve_drop` **always returns 0**: its only explicit `return` is `return 0` at
`gpu_reserve.sh:1777` (the "descendants still alive, record KEPT" branch) and it otherwise falls off
the end after `trap - EXIT INT TERM HUP` (`:1783`), which succeeds. So the `||` branch is dead code —
and the case the operator most needs to see is exactly the one it would have reported: *the
reservation was retained because work is still alive*. That information does reach stderr from
`gpu_reserve.sh:1772-1775`, so it is not lost, but the launcher's own check is theatre.

Worse, the retention branch leaves `_GR_HOLD_PATH` set while clearing `_GR_HOLD_FD` (via
`_gr_close_hold_fd`, `gpu_reserve.sh:1519`). Since the double-acquire refusal keys on
`_GR_HOLD_FD` being non-empty (`:1533`), the second `gpu_reserve_hold` at
`run_r1_imma_official.sh:299` is **permitted** and overwrites `_GR_HOLD_PATH` — so the stage-2 hold
file is never removed by any later drop, and the stage-2 record stays live alongside a second
reservation. The author anticipated the two-reservations-in-one-shell hazard and added the refusal
(`:1525-1541`, citing this very launcher); the refusal just does not cover the post-drop retention
case. This is the only place in the review where a reservation can leak.

#### DEFECT FOUND — g3: `run_r1b_imma_train.sh:381` and `run_r1c_eval.sh:207` undo the helper's export

```
run_r1b_imma_train.sh:381:export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"
run_r1c_eval.sh:207:export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"
```

`gpu_reserve_hold` already exports `CUDA_VISIBLE_DEVICES` for exactly this reason
(`gpu_reserve.sh:1687-1697`), and `run_r1_imma_official.sh:177-180` documents removing this very
line: re-exporting the bare **index** silently overwrites the UUID form, undoing "the one setting
that is immune to driver renumbering". Two of the three r1 launchers still do it, and both carry
comments claiming API fidelity on the preceding lines (`r1b:371`, `r1c:197`). Under the default
`GPU_RESERVE_CVD_STYLE=index` it is a harmless no-op; under `uuid` it silently reverts to the
renumbering-vulnerable form. A two-line deletion, and it removes an inconsistency between three
files that are supposed to share one contract.

#### DEFECT FOUND — g4: `run_v2_rescore_final.sh` launches GPU work after its own precondition failed, and can attest to a previous run

`run_v2_rescore_final.sh` has **`set -u` only** (`:25`) — no `-e`, no `pipefail` — and
`rg_cell` always returns 0 (`run_guard.sh:105-108`). So at `:64` → `:66`, a failed `--plan-only`
precondition does not stop the real run: `rescore_final_v2.py --run-tag d022` launches anyway. The
failure *is* ledgered, so DONE is correctly refused — but the GPU work happens. On a frozen project
that is the wrong order of operations.

Separately, `:70`:

```sh
LATEST=$(ls -1dt outputs/rescore/*/ 2>/dev/null | head -1)
```

`LATEST` is "newest directory under `outputs/rescore/`", not "the directory this run created" — the
`d022` run tag from `:66` is never used to locate it. If the script exits 0 without creating a
directory, `rg_require_loadable` (`:75`) passes on a **stale** `run.json` and `rg_manifest` (`:80`)
writes `v2_completion.json` into the **old** run's directory, attesting to the old artifact. The
only thing preventing this today is that a non-zero exit would have been ledgered by `rg_cell`.
`run_v1_esd_alone_control.sh` shares the missing `-e`/`pipefail` but not the staleness bug.

#### Lower-severity launcher findings (not blocking)

* `run_guard.sh:52` — `: > "$RG_FAIL_FILE" 2>/dev/null || RG_FAIL_FILE=""`. If the ledger cannot be
  created (full filesystem, read-only `logs/`), the on-disk ledger is **silently** abandoned and
  `rg_failcount` falls back to the in-shell counter — reinstating the exact subshell defect the
  comment above it documents. Every `V=$(rg_metric …)` failure would then vanish and `rg_finish`
  could write DONE. No message is emitted. The root filesystem on this host being full is a live
  condition, and `run_guard.sh:10` cites a low-disk abort as the motivating bug. Should warn loudly
  or refuse.
* `run_guard.sh:150-153` — `rg_abort` does not abort. The header at `:31` says it "records a failure
  and stops immediately"; the function records, echoes, and returns 0. Harmless at its only call
  site (`run_v2_rescore_final.sh:83`, last statement of its branch) but the documented contract is
  wrong.
* `run_guard.sh:292` — `rg_metric` captures `2>&1`, so any interpreter noise on the success path is
  concatenated onto the returned value. The K1 gate's `float()` then raises, so it fails **closed** —
  but with a misleading "K1 fail" framing rather than a diagnosis, and
  `run_v1_esd_alone_control.sh:88` would log the polluted string as the measured value.
* `run_r1b_imma_train.sh:223` — `… | grep -vE "Warning|warn|_pytree" | tee -a "$L" || die`. Under
  `pipefail`, `grep -v` exiting 1 (nothing survived the filter) **dies on a passing stage**; and the
  filter would strip a FATAL line containing "warn". Fail-closed, but a false-negative gate.
* `run_r1c_eval.sh:282-321` — the summary/collapse block is unguarded. Under `set -e` a failure
  aborts with only a traceback in `$L`, and it runs *before* `rg_init` (`:349`), so nothing is
  ledgered.
* `run_r1b_imma_train.sh:163-165` — unreachable check: `A="REF_imma_vangogh_${TAG}"` (`:106`) with
  `TAG` always non-empty (`:94-98`), so `[ "$A" = "$LEGACY" ]` is never true and the "refusing to
  write into the historical 5x directory" guard can never fire.
* No launcher installs its own trap, so the reservation trap is the only one. `run_r1c_eval.sh`'s
  `rg_finish`/manifest therefore never run on SIGTERM (the helper exits 143 first). That is the safe
  direction — no DONE — but it means an interrupted r1c leaves no INCOMPLETE line either.

---

## 3. Milestone-1 verdict

**milestone-1 verdict: FAIL**

This is a narrow FAIL, and I want to be clear about what it is not. The lock algebra is sound. The
fail-closed telemetry, the two-gate default-deny acquire path, the kernel-enforced liveness lock, the
D-047 knob validation, the D-048 identity interlock against copied and wrapped binaries, the
artifact-loadability guard, and the K1 gate fix are all real, and I verified each behaviourally
rather than taking the suite's word for it. 318 assertions passed under my own hand. §6.6 is an
honest ledger — I went looking for an overclaim in it and did not find one. The author found and
fixed five of its own defects, including two that are genuinely hard to see (a redirection that
silences the shell, a diagnostic holding the mutex it describes), and both are fixed in code, not
narrated.

It fails on one thing that matters at this specific gate: **two of the controls that this milestone
exists to establish can be silently defeated or silently bypassed, and one of them is the freeze
gate that is currently the only thing stopping GPU work.** M1's bar is "safe enough to execute
verification", and a guard that prints `FREEZE: none` on a frozen project because of an ambient
environment variable does not meet it.

### Minimal list of what must change

**Blocking — must fix before M1 passes:**

1. **`gpu_reserve.sh:133`, `:140-144` — de-derive the production constants from `PROJECT`.** Make
   `_GR_PROD_LOCKDIR`, `_GR_PROD_FREEZE` and `_GR_PROD_OVERRIDE_LOG` literals, or run `PROJECT`
   through `_gr_env_init`'s sweep and refuse rc=6 in production when it differs from the hard-wired
   root. Three lines. This closes the freeze-gate bypass, the mutex relocation (D-001), and the
   audit-log redirect at once. Then fix the false claim at `:155-156`.
2. **`run_guard.sh:421-437` — capture the publish pipeline's status in `rg_manifest`.** Record a
   failure and return non-zero when the generator or `rg_atomic_write` fails, so `rg_finish` cannot
   write DONE with no manifest on disk. While there, re-parse the published file (the manifest is
   currently the only artifact exempt from `rg_require_loadable`) so a truncated JSON cannot be
   published by the `cat > "$tmp"` buffering at `:198`. `run_r1c_eval.sh:389` is on the immediate
   path.
3. **`run_v2_rescore_final.sh` — add `set -euo pipefail` (or guard `:64`'s precondition explicitly)
   so a failed `--plan-only` stage cannot launch the real GPU run at `:66`; and locate the output
   directory by the `d022` run tag rather than `ls -1dt | head -1` (`:70`)**, so the completion
   manifest cannot attest to a previous run's artifact. Same `set` line for
   `run_v1_esd_alone_control.sh`.
4. **`run_r1_imma_official.sh:194` / `gpu_reserve.sh:1765-1784` — make the drop's outcome
   observable.** Have `gpu_reserve_drop` return a distinct status for "record retained, work still
   alive", so the `|| say "WARNING …"` branch is live; and clear `_GR_HOLD_PATH` on the retention
   branch (or have the double-acquire refusal at `:1533` key on `_GR_HOLD_PATH` as well as
   `_GR_HOLD_FD`) so the second acquire at `:299` cannot orphan the first hold file.

**Blocking but trivial:**

5. **Delete `run_r1b_imma_train.sh:381` and `run_r1c_eval.sh:207`** (`export
   CUDA_VISIBLE_DEVICES="$GPU_RESERVED"`), matching the removal already made and documented at
   `run_r1_imma_official.sh:177-180`.

**Required before the INT path may be relied on (needs a human at a terminal, not a GPU):**

6. **Settle SIGINT** per §1 measurement 2: foreground job of an *interactive* shell, a survivor in a
   *different* process group, and assert the `gpu_reserve_drop` stderr line (`gpu_reserve.sh:1377`
   or `:1772-1775`) — not the lock state, which the kernel produces either way. ~2 minutes. Until
   then `REPAIR_A_v3.md:775` should stay **NOT measured**, as it correctly does.

**Should fix, not blocking M1:**

7. `run_r1c_eval.sh:313` and `:320` — validate presence and finiteness before comparing, so a
   non-finite metric cannot print `K1: CLEAR -> attack grid` or a false "collapse signature: not
   present" into the log.
8. `run_guard.sh:52` — do not silently abandon the on-disk ledger; warn loudly or refuse.
9. `run_guard.sh:150-153` — make `rg_abort` abort, or fix the header at `:31`.
10. `run_r1b_imma_train.sh:223` — a `grep -v` that filters everything should not kill a passing
    stage, and the filter should not be able to strip a FATAL line containing "warn".
11. `run_r1b_imma_train.sh` — it is truthful via errexit but has no ledger and no completion
    manifest. Source `run_guard.sh` before its DONE line becomes a cited result.
12. `gpu_reserve.sh:313-325` — consider refusing (rc=6) rather than allowing test mode when the
    configured telemetry source exits non-zero, closing the stateful-shim defeat in (d). Low
    severity; the author's testability concern deserves a real answer first.

**Documentation only:**

13. `REPAIR_A_v3.md:782` — the compute-apps visibility row is now satisfied by the lead's
    measurement (child pid 2174848, 492 MiB, listed). Re-mark it *measured (lead, 2026-10-26)*; it
    is currently the table's only understatement.

### Re-review cost

Items 1-5 are mechanical and total well under an hour. None requires a GPU. I would re-run
`scripts/test_gpu_reserve.sh` plus the `PROJECT` probe in §2(c) and the `rg_manifest` failure path,
and expect to pass on the second pass.

---

## 4. Reviewer's own footprint

CPU only. No GPU work launched; no CUDA context created. GPUs 2-3 untouched. I issued read-only
`nvidia-smi --query-gpu` / `--query-compute-apps` calls to establish occupancy before deciding which
probes were safe, and I deliberately prevented the `PROJECT` probe from completing a reservation by
setting `GPU_RESERVE_FOREIGN_MIB=1`. I made no git commit. `FREEZE_ACTIVE` is intact. The production
lockdir `/data/bijaypandey/t2i_unlearn/.gpulocks/` holds only the zero-byte `.reserve.mutex` and no
reservation record created by me. Every probe wrote only into my session scratchpad. The only file I
edited is this one.
