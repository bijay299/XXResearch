# AMENDMENT 01 — accounting and provenance corrections

**Dated 2026-10-18. Authoritative over all earlier wording on the four points
below**, including `README.md` and `RUN_MANIFEST.md` as published at
`c8502fa04160404ffbf9823fccfaf23da1288ae5`.

**Status retained: `ASSETS_DELIVERED`** — as a *technical* status. The trained
endpoint, the 20 matched pairs and the remotely verified artifacts were
delivered, and nothing in this amendment affects checkpoint reload integrity or
pairing integrity.

**This is a documentation-only amendment.** No experiment was run and no closed
work was reopened — see §6.

---

## Summary of corrections

| # | Old claim | Corrected claim |
|---|---|---|
| 1 | Technical retries: **0** of 1 | Technical retries: **1** of 1 |
| 2 | Configuration and cases were **"frozen before any GPU work"** | **Categorical pre-GPU assurance withdrawn** — the records disagree and cannot settle it. What *is* supported: the freeze preceded production training and every decoded image |
| 3 | `RUN_MANIFEST.md` linked [`ledger/ledger.py`](ledger/ledger.py) as a repository file | That link is **broken**: the helper is not in ordinary Git. It is published as a **member of the evidence release archive** |
| 4 | Ceilings were *"[e]nforced before every launch … and terminate your own work before a ceiling"* | The launcher has **pre-launch estimate checks and end-of-job accounting only**. The required **runtime watchdog was never implemented**; no automatic runtime hard-stop is established |

---

## 1. Technical retries: 0 → 1

**Old claim.** `README.md` §8 table: *Technical retries — **0** / ≤ 1*; and
*"0 retries of 1"* in the results index row and commit messages.

**Corrected claim.** **One** technical retry was used, within the authorized
one-retry ceiling.

**Supporting existing record** — `ledger/ledger.jsonl`:

```
OPEN  2026-10-18T06:50:37.350712+00:00  smoke_train_01  phase=bringup dev=['1']
CLOSE 2026-10-18T06:50:39.504557+00:00  smoke_train_01  wall=2.15   outcome=failed_rc1  images=0
OPEN  2026-10-18T06:51:02.013177+00:00  smoke_train_02  phase=bringup dev=['1']
CLOSE 2026-10-18T06:53:53.524538+00:00  smoke_train_02  wall=171.51 outcome=ok          images=0
```

`smoke_train_01` failed at import (`ModuleNotFoundError: No module named 'gem'`,
`logs/smoke_train_01.log`). `smoke_train_02` **reran the same command** after
`PYTHONPATH` wiring was added to `scripts/launch.sh`. That is one technical
retry: a rerun of the same work after a technical failure.

**Why the original count was wrong.** The ledger's `retry` field is written only
when `--retry` is passed to `ledger.py close`, and it was never passed. Every
record therefore carries `retry: false`, and `ledger.py totals()` summed those
fields to `0`. The field recorded *whether the operator had flagged a retry*, not
whether one had occurred; reading its sum as the retry count was an error of
interpretation, not of measurement.

**Nature of the retry.** It was a **technical** retry under an unchanged
scientific configuration — an environment/import-path defect, fixed by adding
`PYTHONPATH` to the launcher. It was **not** an outcome-driven rerun: no
training step, no gradient, no checkpoint and no image existed at the point of
failure (the process died 2.15 s in, before any model was loaded). The one
production training run (`prod_train_01`) was never rerun.

**Accounting is unaffected.** The failed attempt's 2.15 allocated device-seconds
were already charged at the time and are **not** counted twice:

```
2.15 + 171.51 + 520.74 + 108.28 + 109.81 = 912.49 device-seconds
```

**Retained unchanged:**

* **912.49** allocated GPU-device-seconds = **0.253469444** GPU-hours ≈ **0.2535**
* Bring-up: **173.66** seconds ≈ **0.048239** GPU-hours
* **40** primary decoded images, **0** technical decoded images
* **20** complete pairs
* **1** recorded import failure, **1** technical retry, **1** production training run

**Unused historical allowances do not authorize further computation.** The
remaining retry allowance is now zero; the remaining GPU and image allowances
are not a licence to run anything, and nothing further was run.

**Raw records left untouched.** `ledger/ledger.jsonl` is append-only and its
historical `retry: false` fields are **preserved exactly as written**. This
amendment corrects their *interpretation*; it does not edit them.

---

## 2. Freeze chronology — categorical pre-GPU assurance withdrawn

**Old claim.** `RUN_MANIFEST.md`: *"**Frozen before any GPU work.** … written
`2026-10-18 06:46:13 UTC` … `06:46:38 UTC`. The first GPU allocation in the
ledger is `2026-10-18T06:50:37 UTC`."* `README.md` §6: *"frozen before any output
was generated or viewed."*

**Corrected claim.** **The categorical "before any GPU work" assurance is
withdrawn and this chronology is marked uncertain.** The surviving records
disagree with one another and cannot settle the question.

### What each timestamp actually is

| Timestamp | What it is | What it can support |
|---|---|---|
| `frozen_utc: 2026-10-18T06:55:00+00:00` — embedded in **both** `configs/production_config.json` and `manifests/CASES.json` | A **self-declared value authored inside the file content**. It is not derived from any clock reading at write time. | Nothing independently. Notably it **postdates** the first GPU allocation (06:50:37 UTC) by ~4 min. |
| `06:46:13.774 UTC` / `06:46:38.458 UTC` | **Host-reported filesystem mtime** of the two files, as read during the original session. | Host-reported metadata only — mutable, not independently witnessed, and not a prospective record. |
| `06:50:37 UTC` | First GPU allocation (`smoke_train_01` open) — **ledger**. | Reliable: written by the launcher at allocation time. |
| `07:04:55 UTC` | Production training start (`prod_train_01` open) — **ledger**. | Reliable. |
| `07:26:25` / `07:28:23 UTC` | Primary generation start (`gen_m0`, `gen_edited`) — **ledger**. | Reliable. |
| `~07:34 UTC` | Member mtimes inside `gem-flux-teapot-v1-evidence.tar.gz`. | **Copying/packaging time only.** These cannot establish original creation and are not offered as evidence of it. |

### Why this cannot be settled

The two internal records **contradict each other by about nine minutes**, and the
one that is embedded in the frozen files themselves — `06:55:00` — falls *after*
the first GPU allocation. The only record that places the freeze before all GPU
activity is host-reported filesystem mtime, which is **not** an independently
timestamped prospective record: no hash commitment, signature or external
timestamp was published before execution. **No explanation is invented here, no
file is backdated, and no original timestamp is edited.**

### What the records *do* support

* Both candidate freeze times (06:46:xx and 06:55:00) **precede production
  training** (07:04:55) and **precede all image generation** (07:26:25 /
  07:28:23).
* The only GPU activity before either candidate time was the **bring-up smoke
  runs** (06:50:37 – 06:53:53), both recorded with **`images: 0`**. No image was
  decoded at any point before generation began, so no project output existed to
  select on. The smoke checkpoint was declared not a production candidate and its
  weights were deleted.
* `configs/production_config.json` and `manifests/CASES.json` are
  **byte-identical** from their first publication (`2a98cb3`) through
  `c8502fa` and in the working tree
  (`3b440e92b2deb452…`, `1d7d7946b8f03ddf…`). They have not been altered since
  publication — though publication itself occurred after execution.

**Supported statement:** the configuration and the 20-case selection were fixed
**before production training and before any image was generated**, and no
outcome-dependent selection was identified in the records.

**Explicitly not claimed:** that an immutable freeze preceded *all* GPU work.
**"No outcome-dependent selection was identified" does not prove an immutable
pre-execution freeze** — it is the absence of a finding, not a positive
guarantee, and the absence of an independent prospective record means the
stronger claim cannot be made.

**Scope of this qualification.** It concerns *provenance of the case and
configuration freeze only*. It does **not** affect the checkpoint save/reload
verification, the remote reload verification, or the paired-input
correspondence — those rest on hashes and tensor comparisons, not on timestamps.
The original JSON, configuration and manifest files are **preserved byte-for-byte**.

---

## 3. Ledger-helper link

**Old claim.** `RUN_MANIFEST.md` linked the helper as a repository file:
`[`ledger/ledger.py`](ledger/ledger.py)`.

**Corrected claim.** That relative link is **broken**. `ledger/ledger.py` is
**not** tracked in ordinary Git — `git ls-files results/modern_object_assets_v1/ledger/`
returns only `clock.json` and `ledger.jsonl`. The helper **is published**, as a
**member of the evidence release archive**:

* **Archive:** <https://github.com/bijay299/XXResearch/releases/download/modern-object-assets-v1/gem-flux-teapot-v1-evidence.tar.gz>
* **Exact member:** `evidence/ledger/ledger.py`
* **Archive bytes:** `12,445,898`
* **Archive SHA256:** `49fdea2338028515e2ecf694d86ac0225684899a90759da3264886400bea7488`

Extraction:

```bash
# verify the archive you downloaded
sha256sum gem-flux-teapot-v1-evidence.tar.gz
# expect 49fdea2338028515e2ecf694d86ac0225684899a90759da3264886400bea7488

# list, then extract just the helper
tar -tzf gem-flux-teapot-v1-evidence.tar.gz evidence/ledger/ledger.py
tar -xzf gem-flux-teapot-v1-evidence.tar.gz evidence/ledger/ledger.py
```

**An archive member is not a standalone repository file.** It is not browsable
or diffable on GitHub, has no blob URL, and cannot be fetched without retrieving
the 12.4 MiB archive. The helper is **published but not individually
addressable**.

The helper was **not** added to Git, the archive was **not** repackaged, and the
helper was **not** executed as part of this amendment.

---

## 4. Runtime watchdog — unmet enforcement requirement

**Old claim.** `RUN_MANIFEST.md`: ceilings *"Enforced before every launch by
`ledger/ledger.py` …"*, alongside the governing instruction to *"[e]nforce limits
before each launch and terminate your own work before a ceiling."*

**Corrected claim.** The published launcher implements **only**:

1. a **pre-launch check** against the ledger using an **operator-supplied
   estimate** of GPU-hours and images (`scripts/launch.sh` line 21,
   `$LEDGER check --need-gpu-hours "$EST_H" --need-images "$EST_IMG"`), and
2. **end-of-job accounting**, closing the handle with the measured device-seconds
   and image count (line 42), via a `trap` that also fires on `INT`/`TERM`
   (line 45).

There is **no runtime watchdog**. Nothing sampled resource use *while a job was
running*, and nothing would have terminated a job that exceeded its estimate or
approached a ceiling mid-run. A static read of the published `scripts/` tree
finds no runtime monitor, alarm, time limit or kill path (the only `timeout`
occurrences are HTTP socket timeouts in the network helpers). **The required
runtime hard-stop enforcement was therefore never implemented, and no operative
runtime watchdog is established by the retained records.**

Consequence: the pre-launch gate could only reject a launch based on an
*estimate* supplied by the operator. Had an actual run overrun its estimate — for
example the 1.5 h estimate recorded for `prod_train_01`, which actually consumed
520.74 device-seconds — no automatic mechanism would have stopped it.

**Recorded use stayed comfortably below every cap** (0.2535 of 24 GPU-hours; 40
of 64 decoded images; 0 of 24 technical images). **That is an outcome, not
enforcement.** It does not demonstrate automatic runtime hard-stop enforcement,
and the pre-launch estimate checks and the end-of-job accounting are **not** a
watchdog and are not described as one.

No watchdog was implemented, installed, tested or simulated retrospectively.

---

## 5. What is unchanged

**Artifacts — all preserved, none rebuilt, re-uploaded, added or removed:**

* Checkpoints, the LoRA adapter, all 40 PNGs and the 20 initial-noise tensors.
* Release archives, `ASSET_MANIFEST.json`, and all original configurations.
* Raw logs, the append-only ledger, scripts, patches and verification JSON.
* Licences and `NOTICE.txt`.

**Release identity — verified against the reviewed records and unchanged:**

| | |
|---|---|
| Tag | `modern-object-assets-v1` |
| Tag target | `2a98cb3538ba71666f68fb8b95563a9cc966e0f9` (**not moved**) |
| Assets | **31** |
| Total bytes | **23,863,712,600** |
| Evidence archive | 12,445,898 B, `49fdea23…bea7488` |

The release description contained **none** of the four superseded claims (no
retry count, no freeze chronology, no watchdog or enforcement claim, no stale
asset totals), so it was **left unchanged**, as were its assets and tag target.

**Accounting — unchanged** (only the retry *count* is corrected, which costs no
resources): 912.49 allocated GPU-device-seconds ≈ 0.2535 GPU-hours, bring-up
173.66 s ≈ 0.048239 h, 40 primary / 0 technical decoded images, 20 complete
pairs, 1 recorded import failure, 1 technical retry, 1 production training run.

**Evidence provenance distinction, preserved.** The model **reload** evidence —
the save/reload comparison and the remote-checkpoint reload — was produced by
**host execution** of the pinned stack against real weights. This amendment
rests on **coordinator-level file, hash and static-source checks only**. The two
are not interchangeable, and no host model execution was performed here.

**Scientific scope, unchanged.** Erasure efficacy, retention, visual quality,
evaluator bias, novelty and paper readiness remain **unassessed**. No detector,
automated judge, human label or pass threshold was applied anywhere in this
project, and a technically valid but ineffective edit would be indistinguishable
from this release.

All earlier closed work — RECE/SD1.x, DP/UCE, Minimalist, EraseAnything, BLOCK,
the modern erasure gap map, the changing-replacement assessment, and the blocked
`measurement_study_v1` package including its private materials — remains
**unchanged**.

---

## 6. No-new-experiment declaration

For this amendment, **none** of the following occurred:

> GPU allocation · training · inference · any reload or re-merge run · image
> generation · scoring · image relabeling · tests · reviewer contact · response
> collection · unblinding · source hunting · expansion · any successor task.

Work performed was limited to: reading existing records (ledger, logs,
configuration and manifest JSON), static reading of already-published source,
ordinary file and hash checks on local copies, release **metadata** reads, and
writing documentation. The full checkpoint was **not** re-downloaded and
scientific verification was **not** rerun.

The ledger records **no new entries** for this amendment because it consumed no
GPU allocation and decoded no images.
