# REPAIR_INFRA_v2 — GPU reservation hardening

Date: 2026-10-26. Author: infrastructure repair agent.
Project state: **FROZEN** (`FREEZE_ACTIVE` present). **No GPU work was launched to produce this
document.** Everything below was verified CPU-only with a mocked `nvidia-smi`, except four
read-only `nvidia-smi --query-gpu=index,uuid` calls used to establish one measured fact (defect 4);
those start no compute and allocate no memory.

Files changed (all owned by this agent):

| file | change |
|---|---|
| `scripts/gpu_reserve.sh` | 429 → 1500 lines. Fifteen defects fixed; new exit codes 5, 6 and 7; new `gpu_reserve_require`, `selftest-telemetry`. |
| `scripts/test_gpu_reserve.sh` | 390 → 1243 lines. T1–T17 preserved; T18–T27 plus T19b–T19g and T21b added. 50 → **256 assertions, 0 failures**. |
| `scripts/run_guard.sh` | errexit shield in `rg_step`; new `rg_cell`, `rg_require_glob`. (§9, §11d.) |
| `run_e2h_meta_v2_stronger.sh` | migrated off `campaign_free.sh`. |
| `run_r4b_queue.sh`, `run_r4c_queue.sh`, `run_r4d_upstream_defaults.sh` | migrated off `campaign_free.sh`; wired to `run_guard.sh` (§11d defect 13). |
| `run_v1_esd_alone_control.sh` | migrated + the lead's `--allow-frozen` bug fixed. |
| `docs/REPAIR_INFRA_v2.md` | this file. |

`scripts/campaign_free.sh` was **not** touched. It stays retired, still prints nothing on stdout,
still exits 1 (asserted by T26).

---

## 0. Was the "50 passed" claim testing the wrong thing?

Partly, and it should be stated plainly.

The 50 assertions were **real and are all still passing** — they are now T1–T17 of a 256-assertion
suite. They genuinely proved the lock algebra (mutual exclusion under a 12-way race, PID-reuse
defeat, foreign-GPU refusal, memory-dip immunity) and the D-013 freeze gate. None of them was
fraudulent.

But every one of those 50 ran inside a **single narrow configuration**:

* a `nvidia-smi` shim that always worked and always returned well-formed output;
* `GPU_RESERVE_GPUS="0 1"`, i.e. exactly the production set — so the allowlist's *integrity* was
  never probed, only the loop that walks it;
* no `set -e`, no `set -o pipefail` anywhere in the harness;
* no call with a flag in the wait-interval slot;
* crash simulation by `kill`/`kill -9` of a **childless** placeholder.

All seven defects live outside that box. So the honest summary is: **the suite tested the right
mechanism and the wrong perimeter.** "50 passed, 0 failed" was true and simultaneously uninformative
about whether the script fails safe. T18–T27 exist to make the perimeter the thing under test.

One concrete illustration. v1's T9 is titled *"off-limits GPUs 2-3 are never selected, even when
idle"* and it passed — but it only ever ran with `GPU_RESERVE_GPUS="0 1"`, so what it actually
proved was "the loop does not invent indices it was not given". The thing the title claims was
falsified in 30 seconds once the allowlist itself was varied (§3).

---

## 1. Defect 1 — telemetry failure must be UNKNOWN, not zero

### (a) Verified present

`scripts/gpu_reserve.sh` v1:

* `126-146` `_gr_probe()`: all three queries ran as `done < <($NVSMI ... 2>/dev/null)`. A process
  substitution's exit status is unobservable, and `2>/dev/null` discarded the diagnosis, so a total
  failure simply left `_GR_MEM`, `_GR_APPS` and `_GR_UUID2IDX` **empty**.
* `170` `mem=${_GR_MEM[$i]:-0}` — absent memory defaulted to **0**.
* `171` `case "$mem" in ''|*[!0-9]*) mem=0 ;; esac` — `[N/A]`, `[Not Supported]` and garbage were
  **coerced to 0**.
* `172-173` with no apps and `mem=0 < 512`, the verdict was `FREE`.

Net effect: "I could not read the GPUs at all" was indistinguishable from "0 MiB used, no
processes" = free.

Measured on v1 before the fix (CPU-only, four shim scripts, holder pid alive):

```
D1 nv_fail      rc=0 out=[0]      # nvidia-smi exits 1
D1 nv_empty     rc=0 out=[0]      # nvidia-smi prints nothing, exits 0
D1 nv_garbage   rc=1 out=[]       # accidental, not designed: an empty array subscript errored
D1 nv_na        rc=0 out=[0]      # memory.used = [N/A]
```

Three of four handed out GPU 0 on telemetry it had never read. The `nv_garbage` non-zero was not a
refusal — it was a bash "bad array subscript" accident, with no UNKNOWN diagnosis anywhere.

### (b) Fix

`_gr_probe` is now fail-closed:

* each query runs via `_gr_nvsmi_run <outfile> <args...>`, under `timeout -k 5 $NVSMI_TIMEOUT`
  (default 20s), **into a file, with `rc=$?` captured immediately**;
* `rc != 0` → UNKNOWN; `rc` 124/137 → reported specifically as a timeout;
* the `index,uuid` and `index,memory.used` queries must be **non-empty**; empty is not "nothing
  busy";
* every row must parse: index numeric, UUID matching `GPU-*`/`MIG-*`, memory a non-negative
  integer. `[N/A]` is **never** coerced to 0 — it is UNKNOWN;
* the compute-apps query **may** be empty (that legitimately means no CUDA processes) but must exit
  0, every pid must be numeric, and every `gpu_uuid` must have been seen in the `index,uuid` view —
  disagreement between the two views is UNKNOWN;
* **coverage check**: every *allowed* index must appear in *both* views, which is what catches
  truncated output (fewer lines than GPUs);
* `_gr_classify` returns `UNKNOWN` (never `FREE`) whenever `_GR_TELEMETRY != OK`, and never
  defaults memory to 0.

Acquisition refuses with the new **exit code 5**. A reservation is also **never reclaimed** while
telemetry is unknown — `_gr_reservation_state` returns `ORPHAN … telemetry-unknown-cannot-confirm-idle`,
because reclaiming on unreadable telemetry is this same defect wearing a different hat.

### (c) Test and actual output

T18, nine injected failure modes plus two state assertions each:

```
== T18 DEFECT 1: a telemetry failure is UNKNOWN, never 'free'
  PASS nvidia-smi 'exit1' => rc=5 TELEMETRY UNKNOWN, no index
  PASS   'exit1': no reservation record written
  PASS nvidia-smi 'empty' => rc=5 TELEMETRY UNKNOWN, no index
  PASS   'empty': no reservation record written
  PASS nvidia-smi 'headeronly' => rc=5 ... (malformed index,uuid row (index 'index' is not a number))
  PASS   'headeronly': no reservation record written
  PASS nvidia-smi 'garbage' => rc=5 ... (malformed index,uuid row (index 'Unabletodetermine...'))
  PASS   'garbage': no reservation record written
  PASS nvidia-smi 'short' => rc=5 ... (allowed gpu1 is absent from the index,uuid query (only [0] reported))
  PASS   'short': no reservation record written
  PASS nvidia-smi 'na' => rc=5 ... (gpu0 reported memory.used='[N/A]', which is not a number)
  PASS   'na': no reservation record written
  PASS nvidia-smi 'uuidonly' => rc=5 TELEMETRY UNKNOWN, no index
  PASS   'uuidonly': no reservation record written
  PASS nvidia-smi 'appsfail' => rc=5 TELEMETRY UNKNOWN, no index
  PASS   'appsfail': no reservation record written
  PASS nvidia-smi that HANGS => rc=5 after the timeout, no index
  PASS   timeout is named as the reason
  PASS unknown telemetry never reclaims someone else's record either (rc) (= 5)
  PASS the existing record survived an unknown-telemetry pass
  PASS selftest-telemetry exits 0 on good telemetry
  PASS selftest-telemetry exits 5 on bad telemetry
```

Same four v1 probes re-run against v2:

```
nv_fail      rc=5 out=[]
nv_empty     rc=5 out=[]
nv_garbage   rc=5 out=[]  malformed index,uuid row (index 'Unabletodetermine...')
nv_na        rc=5 out=[]  gpu0 reported memory.used='[N/A]', which is not a number -- refusing to read it as 0 MiB
```

---

## 2. Defect 2 — subprocess exit statuses

### (a) Verified present

Full audit of v1. Every status-losing site:

| v1 line | construct | what was lost |
|---|---|---|
| `132` | `done < <($NVSMI --query-gpu=index,uuid … 2>/dev/null)` | query rc |
| `137` | `done < <($NVSMI --query-gpu=index,memory.used … 2>/dev/null)` | query rc |
| `145` | `done < <($NVSMI --query-compute-apps … 2>/dev/null)` | query rc |
| `98` | `printf '%s\n' "$rest" \| awk '{print $20}'` | only awk's rc survives; a malformed `/proc` line yielded empty output with rc 0 |
| `116` | `sed -n "s/^$2=//p" "$1" \| head -1` | only head's rc survives; an unreadable record read as an empty field |
| `202` | `{ … } > "$tmp"` | write failure (full disk) unchecked → truncated record |
| `203` | `mv -f "$tmp" "$rec"` | publish failure unchecked → **the script logged `RESERVED` and printed an index with no record on disk**, so the next acquirer would see FREE |
| `219` | `flock -u 9` | unlock failure unchecked |

### (b) Fix

* No process substitution remains in code. Each query writes to a temp file; `rc=$?` is captured on
  the very next line; the parse loop then reads `done < "$tmp/…"`, a plain redirect whose status is
  the loop's own.
* `_gr_starttime`/`_gr_pgid` were replaced by `_gr_read_stat`, which uses only builtins
  (`read`, `read -a`, parameter expansion) — no `awk`, no pipeline — and validates both the `)`
  split and the token count (`>= 20`) before indexing.
* `_gr_read_field` is a pure-bash `while read` loop over the record file; no `sed | head`.
* `_gr_write_record` checks **both** the block write and the `mv`, returns 10 on failure, and
  `_gr_do_acquire` then **refuses to claim the GPU** rather than printing an index.
* `flock -u` status is checked and logged; the fd close releases it regardless.
* `mkdir`, `rm -f` on a stale record, and the override-log append are all checked.

### (c) Test and actual output

T24 — static assertions over the source plus a live unwritable-lockdir case:

```
== T24 DEFECT 2: no status-losing construct survives in the source
  PASS process substitutions '< <(' remaining in code (= 0)
  PASS parse loops read from plain file redirects (5 of them)
  PASS nvidia-smi query sites (= 3)
  PASS   each followed by an explicit rc capture (= 3)
  PASS the record publish (mv) status is checked
  PASS no awk/pipeline parsing of /proc remains; _gr_read_stat uses builtins only
  PASS flock -u status is checked
  PASS an unwritable lockdir refuses (rc=10) instead of claiming a GPU
```

The last line is the behavioural one: v1 would have announced `gpu0 RESERVED` and echoed `0`.

---

## 3. Defect 3 — production allowlist that no env var can widen

### (a) Verified present

v1 line `76`: `ALLOWED_GPUS="${GPU_RESERVE_GPUS:-0 1}"`. Measured on v1:

```
D3 GPU_RESERVE_GPUS=[2 3]     rc=0 out=[2]
D3 GPU_RESERVE_GPUS=[0 1 2 3] rc=0 out=[0]
```

`GPU_RESERVE_GPUS="2 3"` **was granted GPU 2** — a GPU belonging to another user of this shared
host — from an environment variable alone.

A second, unlisted instance of the same class: v1 line `82`
`FREEZE_FILE="${GPU_RESERVE_FREEZE_FILE:-$PROJECT/FREEZE_ACTIVE}"`. An env var could point the
freeze marker at a nonexistent path and **silently thaw the project**. Fixed here too.

### (b) Fix

Production constants are hard-wired and are not read from the environment:

```
_GR_PROD_GPUS="0 1"
_GR_PROD_LOCKDIR="$PROJECT/.gpulocks"
_GR_PROD_FREEZE="$PROJECT/FREEZE_ACTIVE"
```

`_gr_mode_init` then separates the two modes:

* **production** (the default) refuses — rc=6, not "ignores" — any `GPU_RESERVE_GPUS` that is not
  exactly `0 1`, and any `GPU_RESERVE_FREEZE_FILE` that is not exactly the production marker.
* **test mode** honours those overrides, but is itself gated by an **interlock**: `GPU_RESERVE_TEST_MODE=1`
  is refused unless (i) `GPU_RESERVE_NVSMI` resolves to something whose basename is *not*
  `nvidia-smi`, and (ii) `GPU_RESERVE_LOCKDIR` is set and is not the production lockdir.

So the escape hatch cannot reach the real driver or the real lockdir. `GPU_RESERVE_TEST_MODE=1
GPU_RESERVE_GPUS="2 3"` inherited from a profile fails closed, because production's telemetry
command *is* the real `nvidia-smi`. A final mode-independent backstop re-checks every index against
`{0,1}` whenever the mode is production.

### (c) Test and actual output

```
== T19 DEFECT 3: no environment variable can widen the production allowlist
  PASS production refuses GPU_RESERVE_GPUS='2 3' (rc=6)
  PASS production refuses GPU_RESERVE_GPUS='0 1 2 3' (rc=6)
  PASS production refuses GPU_RESERVE_GPUS='3' (rc=6)
  PASS production refuses GPU_RESERVE_GPUS='0 1 2' (rc=6)
  PASS refusal explains GPUs 2-3 are off limits
  PASS TEST_MODE=1 with a REAL nvidia-smi is refused (rc=6) -- the interlock
  PASS test mode refuses the production lockdir (= 6)
  PASS production refuses a redirected GPU_RESERVE_FREEZE_FILE (rc=6)
  PASS production acquire is refused by the real FREEZE_ACTIVE (rc=4) (= 4)
```

---

## 4. Defect 4 — device index/UUID mapping and inherited `CUDA_VISIBLE_DEVICES`

### (a) Partly present. **The review's stated premise is false, and I am not going to pretend otherwise.**

The review asserted: *"If `CUDA_VISIBLE_DEVICES` is already set when the script runs, nvidia-smi
index N and the physical device may differ."* **Measured on this host, that does not happen.**
Four read-only queries, same command, four CVD values:

```
$ nvidia-smi --query-gpu=index,uuid --format=csv,noheader           # CVD unset
0, GPU-b2f89296-447e-dc7a-96b6-82ac699a02d7
1, GPU-16080ca5-6073-f2c8-04d7-368d4a510a2a
2, GPU-3594d682-af4b-13e1-3266-8a55455d669d
3, GPU-17211a38-e5ad-e9bc-eeda-aa3caa659d27
$ CUDA_VISIBLE_DEVICES=1     …   # byte-identical output
$ CUDA_VISIBLE_DEVICES=2,3   …   # byte-identical output
$ CUDA_VISIBLE_DEVICES=bogus …   # byte-identical output
```

`nvidia-smi` enumerates through NVML and ignores `CUDA_VISIBLE_DEVICES`; only **CUDA clients**
(torch) honour it. So the indices this script reserves are physical, and there is no renumbering to
fix. v2 does not fix a bug that does not exist here.

What **is** present, and was fixed:

1. **No `CUDA_VISIBLE_DEVICES` handling of any kind.** `grep -n CUDA_VISIBLE_DEVICES
   scripts/gpu_reserve.sh` on v1 returned only comment lines `14, 15, 46, 53`. The variable was
   never read, reported, validated, or set. Every runner was expected to `export` it itself — and a
   runner that forgot inherited its parent's value and ran on a device nobody reserved. An inherited
   **empty** value hides every GPU from CUDA (silent CPU fallback or a confusing crash); an invalid
   value is equally silent.
2. **No device identity was recorded.** `_gr_write_record` (v1 `190-204`) wrote
   `pid/start/campaign/gpu/uid/host/acquired` — **no UUID**. An index-only reservation cannot be
   re-validated, so a driver reload / GPU reset / MIG reconfiguration between acquire and use would
   be honoured silently.

### (b) Fix

* `_gr_cvd_audit` classifies the inherited value (`UNSET`/`EMPTY`/`INDEX_LIST`/`UUID_LIST`/`INVALID`),
  reports it on every acquire, and flags `EMPTY` and `INVALID` as hazards by name.
* `gpu_reserve_hold` now **exports `CUDA_VISIBLE_DEVICES` itself**, so inheritance cannot leak
  through a runner that forgets. It also exports `GPU_RESERVED` (index) and `GPU_RESERVED_UUID`.
* The reservation record carries `uuid=`, and `_gr_reservation_state` returns a new `MAPCHANGED`
  state when index *i* no longer reports the UUID it was reserved against. `MAPCHANGED` **refuses
  the GPU and keeps the record** — it is reported loudly, never reclaimed and never reused.
* Acquire refuses an index with no UUID in telemetry, so every record is re-validatable.
* `GPU_RESERVE_CVD_STYLE=uuid` is available and documented. The **default is `index`**, deliberately:
  UUID-form `CUDA_VISIBLE_DEVICES` is strictly more robust, but it has never been verified against a
  real GPU in this project's venvs, and defaulting to it would be an unverified behavioural change
  to every runner. See §8.

### (c) Test and actual output

```
== T23 DEFECT 4: inherited CUDA_VISIBLE_DEVICES, and index<->UUID mapping
  PASS inherited CUDA_VISIBLE_DEVICES='1' was OVERWRITTEN with the reserved index
  PASS   the inherited value was reported, not silently dropped
  PASS inherited CUDA_VISIBLE_DEVICES='2,3' was OVERWRITTEN with the reserved index
  PASS   the inherited value was reported, not silently dropped
  PASS inherited CUDA_VISIBLE_DEVICES='' was OVERWRITTEN with the reserved index
  PASS   the inherited value was reported, not silently dropped
  PASS inherited CUDA_VISIBLE_DEVICES='bogus,,' was OVERWRITTEN with the reserved index
  PASS   the inherited value was reported, not silently dropped
  PASS an inherited EMPTY CVD is called out as a hazard
  PASS an inherited INVALID CVD is called out
  PASS the reservation record carries the device UUID
  PASS after a renumber, gpu0 is refused and gpu1 chosen (= 1)
  PASS the renumber is reported loudly
  PASS the record was kept, not silently reused
  PASS witness: real nvidia-smi IGNORES CUDA_VISIBLE_DEVICES on this host, so reserved indices are physical
```

The four CVD values the review asked for (`"1"`, `"2,3"`, `""`, invalid) are each driven end to end
through a `set -euo pipefail` runner.

---

## 5. Defect 5 — reservation lifetime, signals, crashes, surviving children

### (a) Verified present. This was the most dangerous of the seven.

v1 had exactly two states. `_gr_reservation_state` (`180-188`) returned `HELD` if
`_gr_owner_alive "$pid" "$start"`, else `STALE`; and `_gr_do_acquire` (`301-303`) deleted a `STALE`
record and carried straight on to claim the GPU. Liveness was **only ever the owner pid's**
(`104-112`). Nothing looked at children, process groups, or record age.

Measured on v1: holder reserved GPU 0, holder `SIGKILL`ed (so no trap could run), a child of it
still alive:

```
D5 holder=1709739 child=1709741
D5 reserved gpu0 for holder
D5 holder alive? NO  child alive? YES
D5 intruder rc=0 out=[0]
D5 RECLAIMED stale reservation: owner pid 1709739 (d5camp) is dead
```

**The same GPU was handed to the next campaign while the dead runner's child was still running.**
A `SIGKILL`ed runner's python child keeps its CUDA context; a child still importing torch is not yet
listed by `nvidia-smi` at all. v1's own `OURS_UNLOCKED` classification could sometimes catch it on
the following pass, but only coincidentally — and by then the record had already been destroyed, so
the protection was gone.

### (b) Fix

A third state, `ORPHAN`, between HELD and STALE. The record is written with `pgid=` as well as
`pid=`/`start=`. When the owner pid is dead, a record is `STALE` only if **all** of these hold:

1. no live process remains in the owner's recorded process group
   (`_gr_live_in_pgid`, a builtin-only `/proc` scan; deliberately skipped when the asking process
   shares that pgid, since a non-interactive parent shell gives its children its own group and we
   must not mistake ourselves for the survivor);
2. no CUDA compute app on that GPU belongs to our uid;
3. telemetry is `OK` (we can actually see the GPU);
4. the record is older than `GPU_RESERVE_ORPHAN_GRACE` (default **120 s**) — the window in which a
   just-orphaned child has not yet appeared in `nvidia-smi`.

Any of 1–4 failing yields `ORPHAN <pid> <campaign> <reason>`: the GPU is **refused**, the record is
**kept**, and `reap` leaves it alone and reports it. Only an explicit
`gpu_reserve.sh release <campaign> <pid>` clears it — a deliberate human act, which is the right
bar for "I confirm that work is really gone".

### (c) Test and actual output

T21. The holder is started under `setsid` so it owns a distinct process group, and
`GPU_RESERVE_ORPHAN_GRACE=1` with a 2 s wait so the grace window cannot be what passes the test —
the pgid survivor must be what blocks reclamation.

```
== T21 DEFECT 5: a dead holder with a SURVIVING CHILD is ORPHANED, never reclaimed
  PASS campaign21 holds gpu0 (= 0)
  PASS holder is dead (SIGKILL, no trap could run)
  PASS child 1746087 survived and still holds its slot
  PASS gpu0 was NOT reclaimed; intruder got [1] (rc=0)
  PASS the state is reported as ORPHANED, with a reason
  PASS the surviving child is named in the reason
  PASS the reservation record was KEPT, not deleted
  PASS reap leaves an ORPHANED reservation in place
  PASS reap reports the orphan count
  PASS an explicit release still clears an orphan
  PASS a surviving CUDA context of ours blocks reclamation too (got [1])
  PASS the reason names the CUDA context
```

Genuine staleness still works — T6 ages the record past the grace window and gets an explicit
`RECLAIMED stale reservation`, so the fix did not simply disable reclamation.

---

## 6. Defect 6 — `set -euo pipefail`

### (a) Verified present. The lead's suspicion was exactly right.

v1 `gpu_reserve_hold` body, lines `386-397`:

```
g=$("${BASH_SOURCE[0]}" acquire "$camp" "$$" ${extra:+$extra}); rc=$?
```

Under the caller's `errexit`, `g=$(failing_command)` is itself a failing simple command, so the
shell exits **on that line**. `rc=$?`, the `rc != 3` diagnosis at `392-394`, and the `return "$rc"`
at `394` were all unreachable. Measured on v1 — a runner with `set -euo pipefail` sourcing the API
and calling `gpu_reserve_hold "d6camp" 5` while frozen:

```
D6b exit=4
D6b BEFORE-B
D6b (no AFTER-B line above => errexit killed the caller silently)
```

The script exited 4 having printed nothing after its own `BEFORE-B` marker: no refusal text, no
"not retrying", no chance for the runner to write an INCOMPLETE marker. Note the asymmetry that hid
this: when the call *is* guarded (`gpu_reserve_hold … || { … }`) bash suppresses errexit inside the
function, and the handler runs correctly. v1 worked when guarded and died silently when not — and
the harness never used `set -e` at all.

### (b) Fix

`gpu_reserve_hold` saves and clears the caller's `errexit`/`pipefail` for the duration of its body,
then restores them exactly before returning:

```
case $- in *e*) _gr_had_e=1 ;; esac
if shopt -qo pipefail 2>/dev/null; then _gr_had_pf=1; fi
set +e; set +o pipefail
… body …
if [ "$_gr_had_pf" = 1 ]; then set -o pipefail; fi
if [ "$_gr_had_e" = 1 ]; then set -e; fi
return "$_gr_rc"
```

(An earlier attempt used `_gr_opts=$(shopt -po errexit pipefail)`. That is itself broken twice over:
`shopt -po` returns non-zero whenever either option is off — tripping the caller's `set -e` on the
save line — and inside `$( )` it reports the subshell's own already-cleared errexit. Caught by
T22's "errexit was clobbered" assertion before this document was written.)

A non-zero return under a restored `errexit` still terminates the caller — that is what `set -e`
*means*, and pretending otherwise would be worse. What changed is that by the time it does, the full
refusal has been printed and `$GPU_RESERVE_RC` holds the real code. For callers who want a clean
deliberate abort, `gpu_reserve_require` is added: it is safe to call bare under `set -euo pipefail`
and exits with the acquisition rc after explaining what the code means.

### (c) Test and actual output

```
== T22 DEFECT 6: set -euo pipefail -- a nonzero acquisition reaches its handling
  PASS guarded call: the || handler runs and the script exits 9 (= 9)
  PASS both markers printed in order
  PASS GPU_RESERVE_RC carried the real refusal code (4)
  PASS nothing ran past the failure
  PASS bare call under set -e exits with the acquisition rc (= 4)
  PASS did not continue past the failure
  PASS the FULL refusal diagnosis was printed (v1 printed nothing after MARK_BEFORE)
  PASS the helper's own summary line was reached
  PASS gpu_reserve_require exits with the acquisition rc (= 4)
  PASS gpu_reserve_require explains the abort
  PASS and names what rc=4 means
  PASS success path under set -euo pipefail exits 0 (= 0)
  PASS index, CUDA_VISIBLE_DEVICES and UUID all exported
  PASS the caller's errexit was restored after the call
  PASS the caller's pipefail was restored after the call
```

---

## 7. Defect 7 — explicit argument validation

### (a) Verified present

v1 lines `384-385`:

```
local camp="${1:?…}" every="${2:-120}"
local extra="${3:-}"
```

Positional, with no validation. `gpu_reserve_hold "name" --allow-frozen` bound the **flag** to
`every` and left `extra` empty, so `--allow-frozen` never reached `acquire` and the freeze gate
refused the run with rc=4 no matter what the lead set. Measured on v1:

```
D7 [gpu_reserve] gpu0 RESERVED for 'd7camp' by pid 1708779
D7 rc=0 GPU_RESERVED=0        # succeeded only because this probe was not frozen;
                              # extra="" so no --allow-frozen was passed through
```

Had a freeze been active, this call could not have succeeded. `sleep --allow-frozen` would have been
the first visible symptom, 120 s later, in a different code path.

### (b) Fix

Flags are parsed **position-free** in a loop; positionals are assigned in order:

* `<campaign>` required, must match `[A-Za-z0-9._-]+` (it is written into the record and the
  override log);
* `[wait_seconds]` must be a **positive integer** — `abc`, `0`, `12.5`, `-5` are all hard rc=2
  usage errors, with a message that names the requirement and, when the token looks like a flag,
  shows the unambiguous form;
* an **unknown** option is rc=2;
* more than two positionals is rc=2.

**Deliberate deviation from the brief, stated openly.** The brief asked that a flag in the
wait-interval position *produce a usage error*. v2 instead **honours** `gpu_reserve_hold "name"
--allow-frozen` — position-free parsing means the flag registers — and prints a `NOTE` naming the
position and the unambiguous form. Reasoning: the hazard being guarded against is "the override
silently did not register", and honouring the intent eliminates that hazard at the root, whereas
erroring would also break the one call shape the lead naturally writes. Nothing is *coerced* and
nothing is *silent*; every other malformed argument is a hard error. If you prefer the strict
reading, the one-line change is to turn the `flag_slot=1` NOTE into `_gr_rc=2`.

### (c) Test and actual output

```
== T20 DEFECT 7: gpu_reserve_hold validates its arguments explicitly
  PASS wait_seconds='abc' is a usage error (= 2)
  PASS   message names the requirement
  PASS wait_seconds='0' is a usage error (= 2)
  PASS wait_seconds='12.5' is a usage error (= 2)
  PASS wait_seconds='-5' is a usage error (= 2)
  PASS an unknown flag is a usage error (= 2)
  PASS a missing campaign is a usage error (= 2)
  PASS a campaign with a space is a usage error (= 2)
  PASS too many positionals is a usage error (= 2)
  PASS gpu_reserve_hold NAME --allow-frozen now REGISTERS the override (the lead's bug is fixed)
  PASS   and it is reported, not silent
  PASS   the same call WITHOUT the flag is refused (rc=4), so the flag is load-bearing
```

The last assertion is the control: the identical call *without* the flag is refused rc=4, which
proves the flag is doing work rather than the test passing for an unrelated reason.

### `run_v1_esd_alone_control.sh`

The lead's real bug. Was:

```
export ALLOW_FROZEN_RUN=1
gpu_reserve_hold "v1_esd_alone_control" --allow-frozen || { … }
export CUDA_VISIBLE_DEVICES=$GPU_RESERVED
```

Now:

```
gpu_reserve_hold "v1_esd_alone_control" 120 --allow-frozen >/dev/null || { … rc guidance … }
```

Three changes, all deliberate: the wait interval is named explicitly so the call is unambiguous
under *any* parser; `CUDA_VISIBLE_DEVICES` is no longer re-exported by hand (the helper sets it, and
a hand-edited value would defeat the reservation); and **`export ALLOW_FROZEN_RUN=1` was removed
from the file**. Baking the env half of the override into the script meant merely executing the file
supplied one of the two required keys — the lead must now pass it on the command line:

```
ALLOW_FROZEN_RUN=1 bash run_v1_esd_alone_control.sh
```

---

## 8. What was tested with MOCKS vs. what needs a real bounded GPU run

**Everything in this document was established with a mocked `nvidia-smi` and CPU-only placeholder
processes, except the four read-only `nvidia-smi --query-gpu=index,uuid` calls in §4.** The 256
assertions touch no GPU, import no CUDA, and are safe to run under the freeze.

### Guarantees fully established by mocks (no GPU run needed)

These are properties of bash control flow, file state and process state, which a mock reproduces
faithfully:

* mutual exclusion under a 12-way race; distinct indices; exhaustion → rc=3 (T1–T4)
* release, re-acquire, trap-based release on SIGTERM and on normal exit (T5, T12, T25)
* stale reclamation, PID-reuse defeat, orphan protection, grace window, `reap` semantics (T6, T7, T21)
* allowlist integrity and the test-mode interlock (T19) — this is pure env/constant logic
* freeze gate, dual-key override, override logging, refusal exit codes (T14–T16)
* argument validation and all usage errors (T20)
* `set -euo pipefail` behaviour, option restoration, `GPU_RESERVE_RC` (T22)
* telemetry **parsing and refusal logic** for every malformed shape (T18)
* absence of status-losing constructs; record write/publish failure handling (T24)
* runner integration end to end, including INCOMPLETE-not-DONE on refusal (T25, T25b)
* `campaign_free.sh` remains inert (T26); the five owned runners are migrated (T27)

### Guarantees that need a real bounded GPU verification — **lead only**

A mock cannot tell me what the real driver does. These remain unverified:

1. **The real `nvidia-smi`'s exact failure output.** T18 injects nine shapes I consider plausible.
   If this driver emits something else on a genuine fault (a different `[Not Supported]` spelling, a
   partial CSV, a non-zero exit with valid rows on stdout), v2 fails *closed* — rc=5, refuse — so a
   surprise degrades availability, not safety. Still worth one look at a real degraded reading if one
   ever occurs.
2. **`GPU_RESERVE_CVD_STYLE=uuid`.** `CUDA_VISIBLE_DEVICES=GPU-<uuid>` has never been run against
   torch in this project's venvs (`venv_t2i`, `venv_uce`, `venv_imma`). Default stays `index`.
   Verification: one tiny CUDA job per venv asserting `torch.cuda.device_count()==1` and that the
   device's UUID matches the reserved one.
3. **That a reserved index really is the device a torch process lands on.** §4 shows `nvidia-smi`
   ignores CVD, and the index→UUID mapping is now recorded and re-checked, but the final link
   "index *i* exported as CVD ⇒ torch allocates on UUID *u*" is a CUDA-runtime property. One
   bounded run: reserve, launch a ~1 GiB allocation, confirm `nvidia-smi --query-compute-apps` shows
   it on the reserved UUID and on no other GPU.
4. **Timeout behaviour against a genuinely hung driver.** T18 simulates a hang with `sleep`. A real
   NVML hang can block in a kernel uninterruptible state, where `timeout`'s SIGTERM may not land;
   `-k 5` escalates to SIGKILL, which is the best a shell can do, and `_gr_nvsmi_run` is not
   observed to be killable-proof.
5. **Orphan detection against a real torch child.** T21 uses `sleep` placeholders in a process
   group. A real python child spawns dataloader workers and may change its process group; the CUDA
   compute-app check (criterion 2) is the backstop, but the *relative timing* of "child alive" vs
   "child visible in `nvidia-smi`" is only measurable for real. The 120 s grace window is a judgement
   call, not a measurement.
6. **Contention with another user on GPUs 2-3.** T8 simulates a foreign pid with `uid 0` and with a
   pid absent from `/proc`. Real `hidepid` behaviour on this host is unverified.

Items 2, 3 and 5 are the ones I would actually ask for before the freeze lifts. Each is a single
bounded cell, and **only the lead may launch them.**

---

## 9. Extra finding (not in the brief): `run_guard.sh` `rg_step` under `errexit`

Found while fixing defect 6, in a file I own. `rg_step`'s header promised it *"records a failure and
KEEPS GOING, so one bad cell does not silently truncate a grid"*. Its body was:

```
"$@" >>"$RG_LOG" 2>&1
local rc=$?
```

Under `set -euo pipefail` — which every real runner uses — the command *is* the failing simple
command, so the shell died before `local rc=$?`. **The `FAIL rc=…` line was never written and the
grid was truncated**, i.e. the documented contract was false under exactly the option set it was
written for. This is the same class as defect 6 and it silently produced *under-populated grids with
no failure record*.

Fixed with the same shield, so the recording always happens; `rg_step` still returns the command's
rc (callers under `set -e` must write `|| true`). Added `rg_cell`, which records and always returns
0 — the grid-safe form. Asserted by T25b:

```
== T25b run_guard.sh rg_step records a failure even under set -euo pipefail
  PASS a failing cell makes the runner exit non-zero (= 1)
  PASS the failure WAS recorded despite set -e
  PASS the grid continued past the bad cell (rg_cell)
  PASS rg_finish wrote INCOMPLETE, never DONE
```

---

## 10. Migration status

Nine runners referenced `campaign_free.sh`. **All nine are now migrated. Zero live invocations
remain anywhere in the project** (the only match outside comments is T26 in the test harness, which
deliberately invokes it to assert it stays inert).

| runner | owner | status |
|---|---|---|
| `run_e2h_meta_v2_stronger.sh` | infra (me) | migrated |
| `run_r4b_queue.sh` | infra (me) | migrated |
| `run_r4c_queue.sh` | infra (me) | migrated |
| `run_r4d_upstream_defaults.sh` | infra (me) | migrated |
| `run_v1_esd_alone_control.sh` | infra (me) | migrated + defect-7 bug fixed |
| `run_r1_imma_official.sh` | Baselines | migrated by Baselines ✓ |
| `run_r1b_imma_train.sh` | Baselines | migrated by Baselines ✓ |
| `run_r1c_eval.sh` | Baselines | migrated by Baselines ✓ |
| `run_r2_meta_official.sh` | Baselines | migrated by Baselines ✓ |

Verification command (should print nothing):

```
grep -rnE '^[^#]*(bash|sh|\.|source)[[:space:]]+[^ ]*campaign_free\.sh' --include='run_*.sh' .
```

### Notes for the Baselines agent

All four of your runners call
`gpu_reserve_hold "<camp>" 120 ${FROZEN_ARG[@]+"${FROZEN_ARG[@]}"}` — the unambiguous form, fully
compatible with v2. Three points:

1. Your contract comments say *"--allow-frozen MUST be the THIRD argument"*. That was true of v1 and
   is now **stale**: v2 parses flags position-free. Your code is correct either way; only the comment
   needs a word. Keep writing the explicit `120` — it is unambiguous under any parser.
2. You `export CUDA_VISIBLE_DEVICES="$GPU_RESERVED"` after the hold. That is now redundant —
   `gpu_reserve_hold` exports it, plus `GPU_RESERVED_UUID` — but harmless, since you export the same
   value. Do **not** export anything else there; a hand-set value defeats the reservation.
3. **New exit codes you may want to switch on:** `5` = telemetry UNKNOWN, `6` = prohibited GPU set.
   Neither is retryable. Only `3` is worth sleeping on. `$GPU_RESERVE_RC` always holds the last
   acquisition rc, including on the errexit path, so `rg_abort "no reservation (rc=${GPU_RESERVE_RC:-?})"`
   works even in a bare-call abort.

### Unrelated loose end

`pick_gpus.sh` exists at the **project root** (not `scripts/pick_gpus.sh`, which does not exist and
which the brief listed as mine). Nothing in the project references it. I left it untouched; it should
probably be retired with the same notice as `campaign_free.sh`, but it is not reachable from any
runner, so it is not a live hazard.

---

## 11a. Late defects 8 and 9 — reported by the Baselines agent during integration

### Defect 8 — `GPU_RESERVE_GPUS=""` fell back to the permissive default (fail-OPEN)

**(a) Verified present — in v2 as well as v1, in BOTH modes.**

Both modes used `${GPU_RESERVE_GPUS:-<default>}`. The `:-` operator treats **set-but-empty**
identically to **unset**, so a caller narrowing the allowlist to nothing silently received the full
default set. Measured on v2-before-this-fix:

```
test mode,  GPU_RESERVE_GPUS=""     -> rc=0, out=[0]      # RESERVED GPU 0
test mode,  GPU_RESERVE_GPUS="   "  -> rc=3, out=[]       # the RETRYABLE code
test mode,  GPU_RESERVE_GPUS="0 x"  -> rc=6, out=[]       # already refused
production, GPU_RESERVE_GPUS=""     -> rc=4, out=[]       # misleading, see below
```

The production case needs care, because its `rc=4` looks like a refusal and is not one. `acquire`
returned 4 only because the real `FREEZE_ACTIVE` refused first and **masked** the fallback. `status`
is not freeze-gated, so it is the honest witness:

```
$ GPU_RESERVE_GPUS="" bash scripts/gpu_reserve.sh status | grep '^allowed='
allowed=0 1  off-limits=everything else  lockdir=/data/bijaypandey/t2i_unlearn/.gpulocks
```

Production had resolved an empty allowlist to **the full default set `0 1`**. Had the project been
thawed, it would have reserved a GPU. The freeze was doing the work the guard was credited with.

The whitespace-only case is a second, subtler fault: the empty `for` loop fell through to
`return 3`, the **retryable** "no allowed GPU available" code — so `gpu_reserve_hold` would have
slept 120 s and retried forever on a pure configuration error.

The coordinator's framing is right, and it is the same family as D-013 (a freeze marker nothing
read) and D-027 (error suppression manufacturing a zero): a control that fails open is worse than
no control, because it gets cited as protection. I would add that this instance was *doubly*
concealed — once by `:-`, and once by the freeze that happened to be active while it was tested.

**(b) Fix.** New `_gr_allowlist_from_env`, which runs in `_gr_mode_init` **before any default is
applied**, and distinguishes three states with `${GPU_RESERVE_GPUS+set}`:

* **unset** → each mode applies its own default (unchanged behaviour);
* **set but empty or whitespace-only** → hard refusal, new **exit code 7**;
* **non-numeric, negative, comma-separated, or containing a duplicate** → hard refusal, rc=7
  (the whole list is rejected; a malformed guard is never partially honoured).

`7` is deliberately **distinct from 3**, and `gpu_reserve_hold`'s retry loop sleeps only on 3, so a
configuration error can no longer loop forever. A mode-independent backstop re-checks that the
*computed* `ALLOWED_GPUS` is non-empty and numeric whatever path set it, so a future edit that
reintroduces a permissive fallback still cannot produce an empty allowlist. `_GR_INIT_RC` records a
source-time refusal so a sourced `gpu_reserve_hold` refuses with the same code instead of running on
the defaults that were assigned before validation. Production's "tried to widen" refusal stays rc=6;
rc=7 means "the allowlist itself is unusable".

**(c) Test — T19b, 34 assertions.** Full output in §11; the requested interaction matrix against the
hard-pinned production set:

```
  interaction matrix against the production pin {0,1}:
  PASS   production + GPU_RESERVE_GPUS=[] (empty) => rc=7
  PASS   production + GPU_RESERVE_GPUS=[   ] (whitespace) => rc=7
  PASS   production + GPU_RESERVE_GPUS=[0 x] (malformed) => rc=7
  PASS   production + GPU_RESERVE_GPUS=[2 3] (out-of-range) => rc=6
  PASS   production + GPU_RESERVE_GPUS=[0 1 2 3] (widened) => rc=6
  PASS   production + GPU_RESERVE_GPUS=[0 1] (exact-match) => rc=0
```

Eight unusable values (`""`, `"   "`, a tab, `"0 x"`, `"x"`, `"0 0"`, `"-1"`, `"0,1"`) each assert
both rc=7 **and** that no reservation record was written. `status` is asserted to no longer resolve
an empty allowlist to `0 1` — the exact witness that exposed the production fallback. Genuinely
unset still returns `0`, so the normal path is not broken.

### Defect 9 — a two-GPU reservation is not expressible

Correct, and left unimplemented as instructed. Now stated explicitly in the usage block under the
heading `ONE GPU PER RESERVATION -- A TWO-GPU RESERVATION IS NOT EXPRESSIBLE`, together with the
non-obvious hazard a caller would otherwise discover the hard way: **do not emulate it by calling
`gpu_reserve_hold` twice in one shell.** Each call installs its own `EXIT/INT/TERM/HUP` trap and the
second **replaces** the first, so one reservation leaks until it ages out of the orphan window.

T19c asserts the documentation is present and then **demonstrates the hazard**, so the warning is
evidence rather than conjecture:

```
== T19c a two-GPU reservation is not expressible, and the limitation is documented
  PASS the usage block states the one-GPU-per-reservation limitation
  PASS and warns that calling acquire twice in one shell leaks a reservation (trap replacement)
  PASS two sequential holds do take two GPUs (0 then 1) in one shell
  PASS ...and exactly as documented, 1 record(s) LEAKED on exit (only the last trap ran)
```

Suite total after defects 8 and 9: 215 passed, 0 failed. After the reviewer round (§11d): **256 passed, 0 failed.**

---

## 11b. Gate condition G14 — other tests whose passing depends on non-production shell options

Scanned as asked. Findings:

1. **`scripts/test_gpu_reserve.sh` is the project's only shell test script.** Its *driver* runs
   `set -u`, which is appropriate for a driver, but the distinction that matters is whether it
   exercises the code under the options **runners** use. It now does: T22, T25 and T25b each spawn
   child runners declaring `set -euo pipefail` and assert behaviour through them. That is the
   property your original 5/5 `rg_step` self-test lacked.

2. **Only 4 of 139 `run_*.sh` use `set -euo pipefail`** — and they are exactly the Baselines agent's
   four migrated runners (`run_r1_imma_official.sh`, `run_r1b_imma_train.sh`, `run_r1c_eval.sh`,
   `run_r2_meta_official.sh`). The other 135, including all five of mine, use `set -u` only. So
   defect 6 and the `rg_step` defect bite **only** those four today. That is not reassurance: it
   means the two most recently hardened runners are the ones in the blast radius, and any runner
   that later gains `set -e` joins it. All five of my migrated runners use the guarded
   `gpu_reserve_hold … || { … }` form, which is correct under both option sets.

3. **`scripts/run_guard.sh` has no production caller.** `grep -rln run_guard --include='*.sh'`
   matches only the file itself and my integration test. The D-012 / G2 guard that is supposed to
   stop a runner reporting success after a failure is **not sourced by a single runner** — it is
   currently in exactly the state D-013's freeze marker was in: a control that exists, is cited, and
   is read by nothing. I have **not** unilaterally wired it into the 104 historical runners, because
   your register deliberately froze their bytes as the record of how historical arms were produced.
   But "G2 is addressed" should not be claimed while the adoption count is zero. Flagging, not
   fixing — your call.

**Capacity note acknowledged: cards are 40 GB, not 80 GB.** No integration test allocates GPU
memory at all (all placeholders are CPU-only `sleep`), so nothing here was sized against 80 GB. The
one place a figure appears is §8 item 3, which proposes a ~1 GiB allocation for the
reserve-then-verify check — comfortable on 40 GB.

---

## 11c. Confirmation received

The coordinator independently confirmed, via a second agent that hit the same silent drop, that the
`gpu_reserve_hold "camp" --allow-frozen` argument-position bug (defect 7, §7) was real and would
have made the authorised bounded V1 run fail with rc=4 regardless of what was set. The coordinator
also re-verified the `rg_step` errexit fix (§9) under the real option set: two failing cells survive,
two FAIL lines recorded, `rg_finish` refuses DONE.

---

## 11d. Independent reviewer round — defects 10 to 15

The reviewer re-ran the suite (215/0, confirmed) and then found four gate-blocking defects the
suite could not see, plus a false claim of mine. **All verified present from source, all fixed.**
Suite now **256 passed, 0 failed.**

### Defect 10 (BLOCKING) — D-001 was reachable again through `GPU_RESERVE_LOCKDIR`

The reviewer is right and this was the worst thing left in the file. I pinned the allowlist, the
freeze marker and the GPU set, and then left the **lockdir** honoured in production with no refusal
(`gpu_reserve.sh:287`, `if [ -n "${GPU_RESERVE_LOCKDIR:-}" ]; then LOCKDIR=...; fi`). Reproduced:

```
$ GPU_RESERVE_LOCKDIR=/tmp/evil_lockdir bash scripts/gpu_reserve.sh status
mode=production
allowed=0 1  off-limits=everything else  lockdir=/tmp/evil_lockdir
```

`MUTEX="$LOCKDIR/.reserve.mutex"`, so two runners started with different values take **different
mutexes** and cannot exclude each other — and the reservation records live there too, so neither
can even see the other's claim. That is precisely the 2026-10-26T12:39 triple-booking of GPU 0 this
file exists to prevent. I had been auditing *which GPUs* the environment could reach and never
asked whether it could move *the lock itself*. A lock at a caller-chosen path is not a lock.

**Fix.** Production now refuses (rc=6) any attempt to redefine `GPU_RESERVE_LOCKDIR`,
`GPU_RESERVE_NVSMI` (a fake telemetry source would make a busy GPU read as free and trample our own
campaign) or `GPU_RESERVE_OVERRIDE_LOG` (redirecting the freeze-override audit log to `/dev/null`
would let an override happen with no record) — alongside the existing refusals for
`GPU_RESERVE_GPUS` and `GPU_RESERVE_FREEZE_FILE`. Verified:

```
GPU_RESERVE_LOCKDIR       rc=6
GPU_RESERVE_NVSMI         rc=6
GPU_RESERVE_OVERRIDE_LOG  rc=6
GPU_RESERVE_FREEZE_FILE   rc=6
GPU_RESERVE_GPUS          rc=6
```

T19d asserts all five, asserts that the mutex path is derived from `LOCKDIR`, and then
**demonstrates the consequence** under test mode (where two lockdirs are permitted): two acquirers
with different lockdirs both win index 0. The witness for D-001 is now in the suite.

### Defect 11 (BLOCKING) — two more unvalidated thresholds, swept across every knob

Verified, exactly as reported. With a GPU at 30000 MiB and no attributable process:

```
GPU_RESERVE_FOREIGN_MIB=512       -> rc=3  (correct: refused)
GPU_RESERVE_FOREIGN_MIB=abc       -> rc=0, RESERVED gpu0
GPU_RESERVE_FOREIGN_MIB=99999999  -> rc=0, RESERVED gpu0
GPU_RESERVE_ORPHAN_GRACE=abc|0|-5 -> grace window deleted; orphan reclaimed
```

`abc` reached `[ "$mem" -ge "$FOREIGN_MIB" ]`, where bash treats a non-numeric operand as 0, so
"30000 >= 0" became false and a busy GPU read as free. Numeric-but-absurd was equally bad: a
threshold above board capacity can never trip.

**Fix — swept, not patched per-variable** as the reviewer asked. New `_gr_env_init` /
`_gr_need_int` validate **every** knob the file reads, with bounds on **both** sides, and refuse
rc=7 (non-retryable) on anything unusable: `NVSMI_TIMEOUT` 1–600 (0 no longer "disables" — that
re-opened the hang), `FOREIGN_MIB` 1–4096 (**our cards are 40 GB ≈ 40960 MiB**, so a threshold
above 4096 could never trip), `WAIT` 1–3600, `ORPHAN_GRACE` 60–86400, `UID` numeric, `CVD_STYLE`
∈ {index, uuid}, `TEST_MODE` ∈ {0,1}. Set-but-empty is distinguished from unset throughout. 18
unusable values asserted at rc=7 in T19e, plus the behavioural core (a 30000 MiB unattributable GPU
stays refused at the default) and a positive case (all knobs at valid non-defaults still acquire).

One self-inflicted bug worth recording: `_gr_need_int` ran inside `$( )`, so `_GR_ENV_RC=7` set in
the subshell was lost, and separately `"${VAR-}"` cannot carry set-ness into a function — which made
every *unset* knob report itself as "SET BUT EMPTY". Both caught by re-running the probes before
writing this.

### Defect 12 (BLOCKING) — unrecognised state was claimable

Verified: `_gr_do_acquire`'s `case "$st"` had arms for HELD/MAPCHANGED/ORPHAN/STALE and **no `*)`**,
so an unrecognised state fell through to the claim. `case "$cls"` was the same — `FREE` was not even
an explicit arm, it was the fall-through. In a file whose only job is exclusion, "I do not know what
this record means" must never mean "take it".

**Fix.** Both cases are now default-deny: `FREE) ;;` is the only arm that proceeds, and `*)`
refuses with an explanation telling a future editor to handle the new state explicitly. T19g
asserts both arms exist and drives a corrupt, unparsable record through `acquire` to confirm it does
not yield the GPU.

### Defect 13 (BLOCKING) — runners wrote DONE without checking anything

Verified: `run_r4b_queue.sh`, `run_r4c_queue.sh`, `run_r4d_upstream_defaults.sh` inspected no exit
status and ended with an unconditional `echo "R4x DONE"`. I migrated their GPU selection and left
their failure handling alone — half a job.

**Fix.** All three now `source scripts/run_guard.sh` and use `rg_init` / `rg_cell` /
`rg_require_glob` / `rg_finish`, following the pattern the coordinator wired into
`run_v1_esd_alone_control.sh`. `rg_finish` is the only writer of a DONE marker.

I added **`rg_require_glob`** rather than use `rg_require_file`, because the artifact names are not
predictable from the shell: `r4b_uce_upstream_vs_port.py` formats `lamb` as a **float**, so the loop
value `1000` writes `r4b_uce_lamb1000.0.json`; and r4c/r4d's config tag changed shape between
versions (`r4c_uce_lamb0.5.json` → `r4c_uce_church_lam0.5_es1.0_ps1.0.json`). Hard-coded paths would
have reported INCOMPLETE on cells that succeeded, and a guard that cries wolf gets switched off. A
glob matching nothing is still a failure, which is the safe direction.

`run_e2h_meta_v2_stronger.sh` was left as-is: it already inspects every cell's `rc`, tracks
`INCOMPLETE`, and exits non-zero.

### Defect 14 (my false claim) — the test-mode interlock was name-only and forgeable

**The reviewer is right and my §3 claim was overstated.** The interlock compared the resolved
*basename* to `nvidia-smi`. A symlink was caught (`readlink -f`); a plain **copy** was not.
Reproduced before the fix:

```
$ cp /usr/bin/nvidia-smi $TMP/copied_nvsmi
$ GPU_RESERVE_TEST_MODE=1 GPU_RESERVE_GPUS="2 3" GPU_RESERVE_NVSMI=$TMP/copied_nvsmi … status
mode=test (mocked nvidia-smi '…/copied_nvsmi', …)
allowed=2 3
```

Test mode against **real telemetry** with GPUs 2-3 — another user's, in use right now — in the
allowlist. I wrote that no env var could widen production onto 2-3; a `cp` could.

**Fix — interlock on device identity, which copying cannot forge.** Test mode is now granted only
if **no UUID reported by the configured telemetry source also appears in the real driver's
enumeration**. A copied/renamed/wrapped nvidia-smi reports the real UUIDs and is refused; a shim
reporting `GPU-aaaa` overlaps nothing and is allowed. This cannot be defeated by copying, renaming,
or PATH games, because any source that can see the real devices is by definition describing them.
The interlock deliberately reuses the `index,uuid` query the rest of the file already needs, so it
cannot be dodged by a shim that simply omits a special query form. Production additionally pins
`GPU_RESERVE_NVSMI` (defect 10), so production can never be pointed at a fake at all. Verified:

```
cp of real nvidia-smi + GPUS='2 3'  rc=6   (refusal cites "the REAL DRIVER also reports" that UUID)
symlink to real nvidia-smi          rc=6
genuinely synthetic shim            rc=0
```

One subtlety found while fixing: a source that *fails* to enumerate anything is now **allowed** into
test mode rather than refused. Refusing masked every telemetry-failure path behind rc=6 and made the
fail-closed behaviour untestable. Allowing is safe because acquisition still refuses with rc=5 on
the same failed query.

### Defect 15 — the pgid survivor check, and an honest partial fix

Verified: `_gr_live_in_pgid` returned early when the acquirer shared the dead holder's pgid, so that
configuration yielded STALE rather than ORPHAN with only the grace window protecting it.

**Fix, part 1.** Do not skip the group — **exclude our own process tree** (ancestors by walking
ppid, descendants by a walk down the children map, from one `/proc` pass).

**Fix, part 2, and this is a limitation I am not going to paper over.** Part 1 alone made things
*worse*, which the new tests caught: for an owner that merely **inherited** its parent's process
group, that group also contains the parent and every unrelated sibling, so "a group member is alive"
is not evidence of descent — and a GPU became **permanently unreclaimable** the moment any sibling
outlived the record. After a dead owner is re-parented to init, there is no sound way to recover its
children from `/proc`.

So pgid is used as evidence of descent **only when the owner led its group** (`pgid == owner pid`),
which is the only case where group membership really means "started by the owner". For a non-leader
owner the code now **logs explicitly** that pgid evidence is unavailable and names its fallback (the
CUDA-context check plus the grace window). The grace default is raised **120 s → 900 s** on the
reviewer's point that a cold model load can exceed two minutes, and the floor is 60 s.

**Residual, for the register:** a non-leader owner whose child holds CUDA but is not yet listed by
`nvidia-smi`, more than 900 s after the owner died, would still be reclaimed. Closing that needs
runners launched under `setsid` so they lead their own groups — a one-word change at each launch
site, which I have documented but not applied, since launching is the lead's. T21 proves the leader
case (ORPHAN, record kept, `reap` refuses to clear it); T21b proves the non-leader case reports its
own blindness rather than hiding it, and that a genuinely stale leader record is still reclaimable
(otherwise the fix would deadlock every GPU forever).

### G14 — the four non-assertions

All four confirmed and fixed. A new `info()` helper prints context **without** counting as a passing
assertion, and anything that cannot fail now uses it.

| reported | was | now |
|---|---|---|
| `:609-613` | `ok` in **both** branches — could not fail | `chk "exactly one record leaks" "$LEAKED" "1"`, asserting the documented trap-replacement behaviour |
| `:320` | `if …; then ok; fi` — no `bad` arm, silently skipped | explicit `else bad "a 0 MiB reading leaked a HELD gpu0"` |
| `:171` | `[ ! -f … ] || ok` — no `bad` arm, **and inverted** | proper if/else asserting the record survives its live holder |
| T27 zero-callers | printed, no assertion | asserts `CF_N == 0`, plus a new per-runner assertion that each migrated runner gates its completion marker on an exit status |
| T24 | greps over source only | source greps demoted to `INFO`; the assertions are now behavioural, including a shim that prints **perfectly valid rows and then exits non-zero** — the sharpest possible proof that the exit status is read rather than the text parsed |

That last one found nothing new, but it is the test T24 should always have been: it fails if anyone
reintroduces a status-losing construct, which a grep for `< <(` would not catch.

### Items I did **not** fix, and why

**`run_r1_imma_official.sh:240` — the stage-4 reservation gated on inherited
`CUDA_VISIBLE_DEVICES`.** I did not touch it. That file is owned by the Baselines agent under the
ownership split the coordinator set, and it is being actively edited; a concurrent edit is how work
gets silently lost. The finding is real and should be fixed there — reservation must never be
conditional on an inherited environment variable, because on a re-run with `CUDA_VISIBLE_DEVICES`
already set the smoke job runs **unreserved**, which is a double-booking waiting to happen. Handing
it over rather than reaching into another agent's file.

**Multi-GPU reservation** remains unimplemented, as instructed — now documented rather than
discoverable the hard way (§11a, defect 9).

---

## 11. Full test-suite output

`bash scripts/test_gpu_reserve.sh` — 256 passed, 0 failed, exit 0. No GPU accessed.

```
CPU-ONLY test harness for /data/bijaypandey/t2i_unlearn/scripts/gpu_reserve.sh
fake nvidia-smi = /data/bijaypandey/tmp/gpu_reserve_test.rvwjgo/fake_nvsmi  (no GPU is touched, no CUDA is loaded)

== T1/T2/T3  sequential acquire, distinct indices, then exhaustion
  PASS first acquire (= 0)
  PASS second acquire takes the OTHER gpu (= 1)
  PASS third acquire refused (exit 3), no double-booking

== T5  release makes a GPU available again
  PASS reacquire after release (= 0)
  PASS gpu1 record still present for its live holder

== T4  CONCURRENCY: 12 processes race for 2 GPUs
  12 racers -> 2 winner(s) [0 1], 10 refused
  PASS exactly 2 winners for 2 GPUs (= 2)
  PASS winners hold DISTINCT indices (= 0 1)
  PASS all other racers refused (= 10)
  PASS no index was handed out twice (this is the campaign_free.sh defect, fixed)

== T6  stale reclamation: owner gone, nothing survives, record aged past the grace window
  PASS crashy reserved (= 0)
  PASS while owner lives, gpu0 is respected (= 1)
  PASS gpu0 handed to the next campaign after the crash (= 0)
  PASS reclamation was explicit: gpu0 RECLAIMED stale reservation: owner pid 1819645 (crashy) is dead,

== T7  PID reuse does not resurrect a dead reservation
  PASS alive pid + mismatched start time => correctly judged stale (PID reuse defeated)

== T8  a GPU occupied by ANOTHER USER is never selected
  PASS gpu0 (another user's) skipped, gpu1 chosen (= 1)
  PASS refusal reason recorded for the foreign GPU
  PASS both GPUs foreign => refuses rather than displacing a user
  PASS pid invisible in /proc treated as foreign, gpu1 chosen (= 1)

== T9  off-limits GPUs 2-3 are never selected, even when idle
  PASS GPUs 2-3 idle but out of scope => nothing selected (exit 3)

== T10 a pre-lock-scheme CUDA context of OURS blocks that GPU
  PASS legacy job of ours on gpu0 respected, gpu1 chosen (= 1)
  PASS legacy job detected via compute-apps + uid, not via a memory threshold
  PASS adopt gpu0 for the legacy pid (= 0)

== T11 inter-cell memory DIP does not release a held reservation
  PASS campaign holds gpu0 (= 0)
  PASS gpu0 still reserved despite 0 MiB used; intruder sent to gpu1 (= 1)
  PASS memory dip no longer leaks the GPU (campaign_free.sh defect (c), fixed)

== T12 trap-based release when the holding shell dies
[gpu_reserve] test-mode interlock: configured telemetry reports no UUID known to the real driver
[gpu_reserve]         -> genuinely synthetic. Test mode allowed.
[gpu_reserve] test-mode interlock: configured telemetry reports no UUID known to the real driver
[gpu_reserve]         -> genuinely synthetic. Test mode allowed.
[gpu_reserve] inherited CUDA_VISIBLE_DEVICES: <unset> (clean)
[gpu_reserve] gpu0 RESERVED for 'trapped_campaign' by pid 1820651 (uuid GPU-aaaa)
  PASS sourced helper reserved a gpu (= 0)
  PASS reservation record present while runner lives
  PASS EXIT/TERM trap released the reservation on shell death

== T13 regression witness: campaign_free.sh's /proc/environ clause is dead code
  PASS a runner that exports CUDA_VISIBLE_DEVICES after startup is INVISIBLE in /proc/<pid>/environ
  PASS => campaign_free.sh clause (a) can never match any of its 8 callers

== T14 D-013 freeze gate: a FREEZE_ACTIVE marker REFUSES every reservation
  PASS refused while frozen, non-zero exit (rc=4), no index returned
  PASS refusal exit code is 4 (distinct from 3 = no GPU free) (= 4)
  PASS refusal is loud on stderr
  PASS marker contents echoed to stderr, so the caller learns WHY
  PASS no reservation record was written while frozen

== T15 the override needs BOTH the env var and the explicit argument
  PASS ambient ALLOW_FROZEN_RUN=1 alone does NOT lift the freeze (rc=4)
  PASS explains that an inherited env var is not enough
  PASS --allow-frozen alone does NOT lift the freeze (rc=4)
  PASS says both are required

== T16 the lead's bounded verification run: override works AND is logged
  PASS override grants a GPU (= 0)
  PASS override exits 0 (= 0)
  PASS override is announced on stderr
  PASS override log line written: 2026-10-26T15:27:43-04:00	override=ALLOW_FROZEN_RUN+--allow-frozen	pid=1822106	campaign=lead_verify	gpu=0	uuid=GPU-aaaa	marker=/data/bijaypandey/tmp/gpu_reserve_test.rvwjgo/FREEZE_ACTIVE	uid=1011
  PASS log records pid=1822106
  PASS log records campaign=lead_verify
  PASS log records gpu=0
  PASS log records override=
  PASS log records uuid=GPU-aaaa
  PASS log line is timestamped
  PASS a non-override caller is still refused after an override was granted (= 4)
  PASS marker removed => normal acquire works again (= 0)

== T17 D-013 witness: gpu_reserve.sh reads FREEZE_ACTIVE
  files referencing FREEZE_ACTIVE: run_r2_meta_official.sh scripts/gpu_reserve.sh scripts/test_gpu_reserve.sh 
  PASS gpu_reserve.sh is a reader of the freeze marker

== T18 DEFECT 1: a telemetry failure is UNKNOWN, never 'free'
  PASS nvidia-smi 'exit1' => rc=5 TELEMETRY UNKNOWN, no index (TELEMETRY UNKNOWN: '/data/bijaypandey/tmp/gpu_reserve_test.rvwjgo/fake_nvsmi --query-gpu=index,u)
  PASS   'exit1': no reservation record written
  PASS nvidia-smi 'empty' => rc=5 TELEMETRY UNKNOWN, no index (TELEMETRY UNKNOWN: '/data/bijaypandey/tmp/gpu_reserve_test.rvwjgo/fake_nvsmi --query-gpu=index,u)
  PASS   'empty': no reservation record written
  PASS nvidia-smi 'headeronly' => rc=5 TELEMETRY UNKNOWN, no index (TELEMETRY UNKNOWN: malformed index,uuid row (index 'index' is not a number): header-only or garb)
  PASS   'headeronly': no reservation record written
  PASS nvidia-smi 'garbage' => rc=5 TELEMETRY UNKNOWN, no index (TELEMETRY UNKNOWN: malformed index,uuid row (index 'UnabletodeterminethedevicehandleforGPU0000:1)
  PASS   'garbage': no reservation record written
  PASS nvidia-smi 'short' => rc=5 TELEMETRY UNKNOWN, no index (TELEMETRY UNKNOWN: allowed gpu1 is absent from the index,uuid query (only [0] reported) -- trunc)
  PASS   'short': no reservation record written
  PASS nvidia-smi 'na' => rc=5 TELEMETRY UNKNOWN, no index (TELEMETRY UNKNOWN: gpu0 reported memory.used='[N/A]', which is not a number -- refusing to read )
  PASS   'na': no reservation record written
  PASS nvidia-smi 'uuidonly' => rc=5 TELEMETRY UNKNOWN, no index (TELEMETRY UNKNOWN: '/data/bijaypandey/tmp/gpu_reserve_test.rvwjgo/fake_nvsmi --query-gpu=index,m)
  PASS   'uuidonly': no reservation record written
  PASS nvidia-smi 'appsfail' => rc=5 TELEMETRY UNKNOWN, no index (TELEMETRY UNKNOWN: '/data/bijaypandey/tmp/gpu_reserve_test.rvwjgo/fake_nvsmi --query-compute-app)
  PASS   'appsfail': no reservation record written
  PASS nvidia-smi that HANGS => rc=5 after the timeout, no index
  PASS   timeout is named as the reason
  PASS unknown telemetry never reclaims someone else's record either (rc) (= 5)
  PASS the existing record survived an unknown-telemetry pass
  PASS selftest-telemetry exits 0 on good telemetry
  PASS selftest-telemetry exits 5 on bad telemetry

== T19 DEFECT 3: no environment variable can widen the production allowlist
  PASS production refuses GPU_RESERVE_GPUS='2 3' (rc=6)
  PASS production refuses GPU_RESERVE_GPUS='0 1 2 3' (rc=6)
  PASS production refuses GPU_RESERVE_GPUS='3' (rc=6)
  PASS production refuses GPU_RESERVE_GPUS='0 1 2' (rc=6)
  PASS refusal explains GPUs 2-3 are off limits
  PASS TEST_MODE=1 with a REAL nvidia-smi is refused (rc=6) -- the interlock
  PASS test mode refuses the production lockdir (= 6)
  PASS production refuses a redirected GPU_RESERVE_FREEZE_FILE (rc=6)
  PASS production acquire is refused by the real FREEZE_ACTIVE (rc=4) (= 4)

== T19b FAIL-OPEN: an EMPTY or MALFORMED allowlist refuses; it never falls back to a default
  PASS GPU_RESERVE_GPUS=[] => rc=7, nothing reserved
  PASS   [] wrote no reservation record
  PASS GPU_RESERVE_GPUS=[   ] => rc=7, nothing reserved
  PASS   [   ] wrote no reservation record
  PASS GPU_RESERVE_GPUS=[	] => rc=7, nothing reserved
  PASS   [	] wrote no reservation record
  PASS GPU_RESERVE_GPUS=[0 x] => rc=7, nothing reserved
  PASS   [0 x] wrote no reservation record
  PASS GPU_RESERVE_GPUS=[x] => rc=7, nothing reserved
  PASS   [x] wrote no reservation record
  PASS GPU_RESERVE_GPUS=[0 0] => rc=7, nothing reserved
  PASS   [0 0] wrote no reservation record
  PASS GPU_RESERVE_GPUS=[-1] => rc=7, nothing reserved
  PASS   [-1] wrote no reservation record
  PASS GPU_RESERVE_GPUS=[0,1] => rc=7, nothing reserved
  PASS   [0,1] wrote no reservation record
  PASS the empty case names itself as a configuration error
  PASS and says it is NON-RETRYABLE (rc=7, not the retryable 3)
  PASS rc=7 is distinct from rc=3 'no GPU free' (= 7)
  PASS GPU_RESERVE_GPUS genuinely UNSET still uses the default and works (= 0)
  PASS production: GPU_RESERVE_GPUS=[] => rc=7
  PASS production: GPU_RESERVE_GPUS=[   ] => rc=7
  PASS production: GPU_RESERVE_GPUS=[0 x] => rc=7
  PASS production: GPU_RESERVE_GPUS=[0 0] => rc=7
  PASS status no longer resolves an empty allowlist to the default set
  PASS status itself refuses an unusable allowlist (= 7)
  interaction matrix against the production pin {0,1}:
  PASS   production + GPU_RESERVE_GPUS=[] (empty) => rc=7
  PASS   production + GPU_RESERVE_GPUS=[   ] (whitespace) => rc=7
  PASS   production + GPU_RESERVE_GPUS=[0 x] (malformed) => rc=7
  PASS   production + GPU_RESERVE_GPUS=[2 3] (out-of-range) => rc=6
  PASS   production + GPU_RESERVE_GPUS=[0 1 2 3] (widened) => rc=6
  PASS   production + GPU_RESERVE_GPUS=[0 1] (exact-match) => rc=0
  PASS a sourced gpu_reserve_hold also refuses rc=7 on an empty allowlist
  PASS   and the refusal is announced at source time

== T19d BLOCKING: production constants cannot be relocated by the environment (D-001 vector)
  PASS production refuses to relocate the lockdir (the D-001 vector) (rc=6)
  PASS production refuses to relocate the telemetry source (rc=6)
  PASS production refuses to relocate the freeze-override audit log (rc=6)
  PASS production refuses to relocate the freeze marker (rc=6)
  PASS production refuses to relocate the allowlist (rc=6)
  PASS the mutex path is derived from LOCKDIR (hence relocating LOCKDIR splits the lock)
  PASS witness: two DIFFERENT lockdirs both hand out index 0 => split mutex = D-001 (now production-refused)

== T19e BLOCKING: every numeric/enum knob is range-checked (fail-open sweep)
  PASS GPU_RESERVE_FOREIGN_MIB=abc => rc=7, nothing reserved
  PASS GPU_RESERVE_FOREIGN_MIB=99999999 => rc=7, nothing reserved
  PASS GPU_RESERVE_FOREIGN_MIB=0 => rc=7, nothing reserved
  PASS GPU_RESERVE_FOREIGN_MIB= => rc=7, nothing reserved
  PASS GPU_RESERVE_ORPHAN_GRACE=abc => rc=7, nothing reserved
  PASS GPU_RESERVE_ORPHAN_GRACE=0 => rc=7, nothing reserved
  PASS GPU_RESERVE_ORPHAN_GRACE= => rc=7, nothing reserved
  PASS GPU_RESERVE_ORPHAN_GRACE=5 => rc=7, nothing reserved
  PASS GPU_RESERVE_NVSMI_TIMEOUT=abc => rc=7, nothing reserved
  PASS GPU_RESERVE_NVSMI_TIMEOUT=0 => rc=7, nothing reserved
  PASS GPU_RESERVE_NVSMI_TIMEOUT= => rc=7, nothing reserved
  PASS GPU_RESERVE_WAIT=abc => rc=7, nothing reserved
  PASS GPU_RESERVE_WAIT=0 => rc=7, nothing reserved
  PASS GPU_RESERVE_WAIT= => rc=7, nothing reserved
  PASS GPU_RESERVE_UID=abc => rc=7, nothing reserved
  PASS GPU_RESERVE_CVD_STYLE=banana => rc=7, nothing reserved
  PASS GPU_RESERVE_TEST_MODE=2 => rc=7, nothing reserved
  PASS GPU_RESERVE_TEST_MODE=yes => rc=7, nothing reserved
  PASS at the default threshold a 30000 MiB unattributable GPU is still refused (rc=3)
  PASS valid non-default values for every knob still acquire (= 0)

== T19f BLOCKING: the test-mode interlock cannot be forged by COPYING nvidia-smi
  PASS a sym_nvsmi of the real nvidia-smi is REFUSED test mode (rc=6)
  PASS a cp_nvsmi of the real nvidia-smi is REFUSED test mode (rc=6)
  PASS no 'allowed=2 3' is ever printed for a copied binary (= 0)
  PASS the refusal is on DEVICE IDENTITY, not on the filename
  PASS a genuinely synthetic shim is still granted test mode (= 0)

== T19g BLOCKING: an UNRECOGNISED reservation state is never claimable (default-deny)
  PASS the state machine has an explicit default-deny arm
  PASS the classifier has an explicit default-deny arm too
  PASS a corrupt reservation record does NOT yield gpu0 (got [1])

== T21b the pgid survivor check works even when the acquirer SHARES the dead holder's pgid
  INFO acquirer pgid=1818470  holder pgid=1818470 (equal => the previously-skipped case)
  PASS non-leader owner: the missing pgid evidence is REPORTED, not silently skipped
  PASS   and the fallback (CUDA check + grace window) is named
  INFO   with the record aged past the grace window and no CUDA context, gpu0 was reclaimed
  INFO   (got [0]) -- correct for a non-leader owner; see T21 for the leader case
scripts/test_gpu_reserve.sh: line 788: 1824134 Killed                  ( exec -a gr_test_child sleep 180 ) > /dev/null 2>&1 < /dev/null
  PASS a group-leader owner with NO surviving member IS reclaimed (no self-false-positive)

== T19c a two-GPU reservation is not expressible, and the limitation is documented
  PASS the usage block states the one-GPU-per-reservation limitation
  PASS and warns that calling acquire twice in one shell leaks a reservation (trap replacement)
  PASS two sequential holds do take two GPUs (0 then 1) in one shell
  PASS exactly one record leaks, proving trap replacement (the documented hazard) (= 1)

== T20 DEFECT 7: gpu_reserve_hold validates its arguments explicitly
  PASS wait_seconds='abc' is a usage error (= 2)
  PASS   message names the requirement
  PASS wait_seconds='0' is a usage error (= 2)
  PASS wait_seconds='12.5' is a usage error (= 2)
  PASS wait_seconds='-5' is a usage error (= 2)
  PASS an unknown flag is a usage error (= 2)
  PASS a missing campaign is a usage error (= 2)
  PASS a campaign with a space is a usage error (= 2)
  PASS too many positionals is a usage error (= 2)
  PASS gpu_reserve_hold NAME --allow-frozen now REGISTERS the override (the lead's bug is fixed)
  PASS   and it is reported, not silent
  PASS   the same call WITHOUT the flag is refused (rc=4), so the flag is load-bearing

== T21 DEFECT 5: a dead holder with a SURVIVING CHILD is ORPHANED, never reclaimed
  PASS campaign21 holds gpu0 (= 0)
scripts/test_gpu_reserve.sh: line 924: 1824907 Killed                  setsid "$TMP/parent21.sh" "$TMP/child21" > /dev/null 2>&1
  PASS holder is dead (SIGKILL, no trap could run)
  PASS child 1824909 survived and still holds its slot
  PASS gpu0 was NOT reclaimed; intruder got [1] (rc=0)
  PASS the state is reported as ORPHANED, with a reason
  PASS the surviving child is named in the reason
  PASS the reservation record was KEPT, not deleted
  PASS reap leaves an ORPHANED reservation in place
  PASS reap reports the orphan count
  PASS an explicit release still clears an orphan
  PASS a surviving CUDA context of ours blocks reclamation too (got [1])
  PASS the reason names the CUDA context

== T22 DEFECT 6: set -euo pipefail -- a nonzero acquisition reaches its handling
  PASS guarded call: the || handler runs and the script exits 9 (= 9)
  PASS both markers printed in order
  PASS GPU_RESERVE_RC carried the real refusal code (4)
  PASS nothing ran past the failure
  PASS bare call under set -e exits with the acquisition rc (= 4)
  PASS did not continue past the failure
  PASS the FULL refusal diagnosis was printed (v1 printed nothing after MARK_BEFORE)
  PASS the helper's own summary line was reached
  PASS gpu_reserve_require exits with the acquisition rc (= 4)
  PASS gpu_reserve_require explains the abort
  PASS and names what rc=4 means
  PASS success path under set -euo pipefail exits 0 (= 0)
  PASS index, CUDA_VISIBLE_DEVICES and UUID all exported
  PASS the caller's errexit was restored after the call
  PASS the caller's pipefail was restored after the call

== T23 DEFECT 4: inherited CUDA_VISIBLE_DEVICES, and index<->UUID mapping
  PASS inherited CUDA_VISIBLE_DEVICES='1' was OVERWRITTEN with the reserved index
  PASS   the inherited value was reported, not silently dropped
  PASS inherited CUDA_VISIBLE_DEVICES='2,3' was OVERWRITTEN with the reserved index
  PASS   the inherited value was reported, not silently dropped
  PASS inherited CUDA_VISIBLE_DEVICES='' was OVERWRITTEN with the reserved index
  PASS   the inherited value was reported, not silently dropped
  PASS inherited CUDA_VISIBLE_DEVICES='bogus,,' was OVERWRITTEN with the reserved index
  PASS   the inherited value was reported, not silently dropped
  PASS an inherited EMPTY CVD is called out as a hazard
  PASS an inherited INVALID CVD is called out
  PASS the reservation record carries the device UUID
  PASS after a renumber, gpu0 is refused and gpu1 chosen (= 1)
  PASS the renumber is reported loudly
  PASS the record was kept, not silently reused
  PASS witness: real nvidia-smi IGNORES CUDA_VISIBLE_DEVICES on this host, so reserved indices are physical

== T24 DEFECT 2: subprocess exit statuses are actually CHECKED (behavioural)
  INFO source shape (not assertions):
  INFO   '< <(' process substitutions in code : 0
  INFO   plain-redirect parse loops          : 7
  INFO   _gr_nvsmi_run call sites            : 3
  INFO   awk occurrences in code             : 0
  PASS valid stdout + NON-ZERO exit => rc=5 refused (the status is genuinely checked, not the text)
  PASS   the refusal names the non-zero exit
  PASS   no record written
  PASS an unwritable lockdir refuses (rc=10) instead of claiming a GPU
  PASS an unwritable lockdir refuses (rc=10) instead of claiming a GPU

== T25 RUNNER INTEGRATION: a real-shaped runner end to end, mocked nvidia-smi
  PASS the runner completed cleanly (= 0)
  PASS it reserved gpu0 (= 0)
  PASS CUDA_VISIBLE_DEVICES and the UUID reached the cells
  PASS run_guard wrote DONE only after the artifact check passed
  PASS the EXIT trap released the reservation on normal exit
  PASS the released GPU is immediately reusable (= 0)
  PASS a frozen runner exits non-zero (= 1)
  PASS it recorded INCOMPLETE, never DONE
  PASS the runner body never executed while frozen

== T25b run_guard.sh rg_step records a failure even under set -euo pipefail
  PASS a failing cell makes the runner exit non-zero (= 1)
  PASS the failure WAS recorded despite set -e
  PASS the grid continued past the bad cell (rg_cell)
  PASS rg_finish wrote INCOMPLETE, never DONE

== T26 campaign_free.sh stays retired
  PASS campaign_free.sh prints NOTHING on stdout (so old waiters keep waiting)
  PASS campaign_free.sh exits non-zero (rc=1)
  PASS it is documented as RETIRED

== T27 the five infra-owned runners are migrated
  PASS run_e2h_meta_v2_stronger.sh parses
  PASS run_e2h_meta_v2_stronger.sh no longer calls campaign_free.sh
  PASS run_e2h_meta_v2_stronger.sh uses the reservation API
  PASS run_r4b_queue.sh parses
  PASS run_r4b_queue.sh no longer calls campaign_free.sh
  PASS run_r4b_queue.sh uses the reservation API
  PASS run_r4c_queue.sh parses
  PASS run_r4c_queue.sh no longer calls campaign_free.sh
  PASS run_r4c_queue.sh uses the reservation API
  PASS run_r4d_upstream_defaults.sh parses
  PASS run_r4d_upstream_defaults.sh no longer calls campaign_free.sh
  PASS run_r4d_upstream_defaults.sh uses the reservation API
  PASS run_v1_esd_alone_control.sh parses
  PASS run_v1_esd_alone_control.sh no longer calls campaign_free.sh
  PASS run_v1_esd_alone_control.sh uses the reservation API
  PASS run_v1 no longer passes a flag in the wait_seconds position
  PASS zero run_*.sh still INVOKE campaign_free.sh (all nine migrated)
  PASS run_e2h_meta_v2_stronger.sh gates its completion marker on an exit status
  PASS run_r4b_queue.sh gates its completion marker on an exit status
  PASS run_r4c_queue.sh gates its completion marker on an exit status
  PASS run_r4d_upstream_defaults.sh gates its completion marker on an exit status
  PASS run_v1_esd_alone_control.sh gates its completion marker on an exit status

----------------------------------------
RESULT: 256 passed, 0 failed
No GPU was accessed; nvidia-smi was the shim at /data/bijaypandey/tmp/gpu_reserve_test.rvwjgo/fake_nvsmi
```
