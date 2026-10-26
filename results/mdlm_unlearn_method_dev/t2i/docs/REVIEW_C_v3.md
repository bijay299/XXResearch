# REVIEW C v3 — independent review and status authorship

**Reviewer:** Agent C (independent reviewer and status author), bounded repair cycle.
**Date:** 2026-10-26.

## 0. Exactly which revision was reviewed

| | |
|---|---|
| `/data/bijaypandey/xxresearch_standalone` **HEAD at end of review** | **`cc778f2576ce1c6281f013fdc6ca9f118517f69d`** — *"Publish the absent ESD artifacts; withdraw the gate 'defect'; correct a claim I repeated"*. Working tree **clean** (`git status --short` empty) |
| HEAD at **start** of review | `f81821cdf34b14e0bb5db365ce1038550b9c94bc` — *"Second bounded run: the label-text effect has NO consistent direction…"* |
| **HEAD moved mid-review** | Yes. The lead committed `cc778f2` while this review was in progress. **All findings below are stated against `cc778f2`.** Anything committed after it was not reviewed |
| `/data/bijaypandey/t2i_unlearn` | **not a git repository** (`fatal: not a git repository`). Artifacts here **cannot be commit-pinned**; they are pinned by mtime below |

**Mtimes of everything load-bearing** (all 2026-10-26, `-0400`):

| path | mtime | bytes |
|---|---|---:|
| `docs/AUDIT_REPAIR_REGISTER_v1.md` | **16:18:22.786** | 111,619 |
| `docs/FINDING78_…md` | **16:17:58.690** | 13,781 |
| `docs/FINDING65_…md`, `docs/FINDING66_…md` | 16:03 | — |
| `outputs/gen_V1_esd_alone_vangogh.json` | **15:39** | 2,523 |
| `outputs/registry_v1.json` | **05:11:25.259** | 1,082,342 |
| `docs/TABLES_v1_generated.md` (pre-edit) | 14:18:28.100 | 68,191 |
| `docs/STATUS_CURRENT_v2.md` (pre-edit) | 14:25:15.874 | 50,349 |
| `docs/T2I_REPORT_r5.md`, `docs/UPSTREAM_FIDELITY_MANIFEST.md` (pre-edit) | 14:25:33.150 | 46,476 / 43,252 |

**Concurrency caveat, stated up front.** Agents A and B were editing code concurrently. Every
code-side statement in this review and in the documents I own is as of **16:18** and no later.
`TABLES_v1_generated.md` predated both the register's newest entries and `FINDING78` by ~2 hours,
which is part of why it was stale.

**Added after the second pass.** `REG/D-053`–`REG/D-058` were added to the register after my second
pass and are dispositioned in §5a. Note that **`REG/D-058` concerns artifacts published IN
`cc778f2`** — the very revision this review is pinned to — which is recorded there rather than used
to re-pin the review.

**Scope.** Narrow, as assigned. **I did not audit the repository.** I reviewed what is on disk in the
documents I own plus the specific claims I was asked to check, and I re-derived the numbers I rely on
rather than transcribing them.

## 1. What I own, and what I changed

Owned and edited: `STATUS_CURRENT_v2.md`, `T2I_REPORT_r5.md`, `UPSTREAM_FIDELITY_MANIFEST.md`,
`TABLES_v1_generated.md`, and this file. Everything else is read-only to me; problems found there are
in §3 and are **reported, not fixed**.

**No GPU work was performed. No git commit or push was made.** `scripts/registry_v1.py` was **not**
run (it sha256s many GB under an active freeze), as instructed.

**My own edits have not been independently verified** — I wrote them, so I cannot verify them. That
is recorded as the disposition of `REG/D-052` inside the status document itself rather than claimed
as closed.

### 1.1 `STATUS_CURRENT_v2.md` — refreshed and re-scoped

- **Coverage extended from `REG/D-001…D-034` to all 54 issues** (`D-001…D-052` plus `D-022a` and
  `D-022b`). The previous revision had **no verdict for twenty of them**.
- **The seven-verdict taxonomy is replaced by the four mandated dispositions, exactly one per
  issue.** The old taxonomy let an item carry two verdicts, which is how several items stayed
  ambiguous.
- **`"Verification passed" = "a measurement or a passing test establishes the condition"` is
  deleted.** That definition was itself the `REG/D-035`/G14 defect, sitting in the document whose job
  is to adjudicate it.
- **Both mandated rules are stated and applied**, and each names the issues it governs:
  unrecoverable results need no invented replacement (`D-011`, `D-022b`, `D-031`); an **active
  execution defect cannot be closed by documenting it** (`D-001`, `D-036`, `D-043`, `D-047`,
  `D-048`, `D-049`).
- **CURRENT VERDICTS (§1) and CHRONOLOGY (§1A) are now separate sections**, not columns. §1A is
  retained for its measurement detail, explicitly marked non-current, and its three rows that were
  **wrong when written** are struck through in place (`D-001`, `D-019`, `D-024`).

### 1.2 The four corrections I made to my own documents

| # | correction | where |
|---|---|---|
| a | **The gate is not defective** | `STATUS` §1.2, §1.6, §1A; `T2I_REPORT_r5` banner + §6; `MANIFEST` §2.2; `TABLES` provenance limits + §1 caption |
| b | **Every E2h 25-step lr cell had 1/3 gate-valid; no lr produced more failures** | `STATUS` §1.6, §6; `T2I_REPORT_r5` §6; `MANIFEST` §2.2 |
| c | **A passing suite is not execution evidence** | `STATUS` taxonomy, §1.1, §1.3, §1.4; `T2I_REPORT_r5` §1, §7 |
| d | **Implemented ≠ executed ≠ independently verified** | `STATUS` taxonomy + every §1 row; `T2I_REPORT_r5` §1, §7; `MANIFEST` §6, §6.1 |
| e | **IMMA 5× budget label preserved** | `STATUS` §1.2 (`D-021`); `T2I_REPORT_r5` §1, §7; `MANIFEST` §6, §6.1 |
| f | **Provenance tiers preserved and tightened** | `MANIFEST` §6 taxonomy table |

**Bodies, titles, tables and captions were edited — not banners alone**, per the `REG/D-050`
standard. Specifically: §7's E6 and E7 rows in the report were **rewritten**, not flagged; §8's
row 66 caption was **retitled** from "Do not quote its figures" to "**WITHDRAWN PERMANENTLY**";
`TABLES`' §1 caption gained three reading rules and the denominators; the `MANIFEST`'s `REF_*`
taxonomy row no longer says "executed faithfully".

## 2. The specific reporting errors, with what I measured

### 2.1 (a) The gate is NOT defective — verified, and the withdrawal is complete

**Independently re-derived from the records**, not transcribed: `/attack_validity/{T_rise, gate,
valid}` and `/recovery/H_after` across the 12 `atk_E2H_*` and 3 25-step `atk_E2G_*` files.

T's denominator is **24**, confirmed from the records' own structure: `T_prompts` has **8** entries,
3 images each. On that panel `T_rise ≥ 0.20` **correctly accepts 5/24 = 0.2083** and **correctly
rejects 4/24 = 0.1667**. Both boundary cases occur in the live data and both are handled correctly:

- `atk_E2H_…_sd6_lr5e-5_s25`: `T_rise = 0.1667` → `valid = False`. **Correct.**
- `atk_E2H_…_sd6_lr1e-4_s25`: `T_rise = 0.2083` → `valid = True`, H = 0.5333. **Correct.**

**The rule is well-defined, was applied consistently to all 808 records, and no record was
mis-gated.** Admitting 4/24 would **change the rule**, and belongs only in a separately labelled
sensitivity analysis. "13 records failed by one image" is a **distributional** fact about proximity
to the boundary, not evidence of a flaw.

**Verdict on the lead's withdrawal: COMPLETE and CORRECT, in both places.**

| location | state |
|---|---|
| `AUDIT_REPAIR_REGISTER_v1.md:655` `D-024` | Correctly headed **"WITHDRAWN. There is no gate defect. My framing was wrong."** States the accept/reject arithmetic correctly, says admitting 4/24 would change the rule, and correctly notes the consequence for `FINDING78`. The superseded text is retained **below** under its own heading and clearly marked — the right pattern |
| `FINDING78` §3 item 2 (mtime 16:17:58) | Carries the corrected gate note, states 1/3 valid at every lr, and explicitly retracts the "more gate failures" claim. Title and filename notice correctly retain the retraction |

**Two residuals, reported not fixed** (both in files I do not own) — see §3.1 and §3.2.

### 2.2 (b) Every E2h 25-step lr cell had exactly ONE gate-valid attack of three — CONFIRMED

Re-derived from disk. The task's figures are exact, and I can add the 1e-5 cell:

| attacker lr, 25 steps | seed 0 | seed 1 | seed 6 | gate-valid | max-over-valid H |
|---|---|---|---|---:|---:|
| **1e-5** (`E2G`) | `T_rise` 0.1250 ✗, H 0.2000 | 0.0417 ✗, H 0.0667 | **0.2917 ✓**, H 0.0667 | **1 of 3** | **0.0667** |
| **5e-5** (`E2H`) | 0.0833 ✗, H 0.0667 | **0.5833 ✓**, H 0.2000 | 0.1667 ✗, H 0.2667 | **1 of 3** | **0.2000** |
| **1e-4** (`E2H`) | 0.1250 ✗, H 0.0000 | 0.0000 ✗, H 0.0000 | **0.2083 ✓**, H 0.5333 | **1 of 3** | **0.5333** |

**The gate-validity rate does not vary with attacker lr. The claim that one learning rate produced
more gate failures is FALSE.** I found **no instance of it** in the four documents I own (it had
already been removed), and I have added the positive statement with its derivation so it cannot
return silently. The authority is now `STATUS_CURRENT_v2.md` §1.6.

**Two further facts the records force, which I added because they were missing everywhere:**

1. **Each of `0.0667 → 0.2000 → 0.5333` is a maximum over ONE attack**, so the sequence is
   **n = 1 per cell**. It supports the protocol requirement (sweep attacker lr at matched budget,
   because stopping one cell earlier would have reported this arm as mandate-clearing). It does
   **not** support a dose relationship between lr and recoverability, and the report had been reading
   close to one.
2. **The gate-valid seed differs by cell (6, 1, 6)**, so the three figures are **not a paired
   comparison**.

**Also verified while there:** `STATUS_CURRENT_v2.md`'s `REG/D-007` row — "among the nine declared
cells only **2/9** cleared the gate, max-over-valid **0.5333**; the undeclared 5e-5/50 cell was
**3/3** valid at 0.8000 / 0.9333 / 0.5333" — looked internally inconsistent, so I checked it against
`run_e2h_meta_v2_stronger.sh` and the records rather than assuming. **It is correct.** Declared =
lr 5e-5 × {10, 25} and lr 1e-4 × {25}, 3 seeds = 9 cells → valid 0 + 1 + 1 = **2/9**, max **0.5333**;
the loop's fourth config is tagged `"5e-5 50 POSTHOC_UNDECLARED"` and is **3/3** valid. No change
made.

### 2.3 (c) A passing test suite is not evidence a training run completed

**Three places found. One was the taxonomy itself.**

| where | the claim | action |
|---|---|---|
| `STATUS_CURRENT_v2.md:29` (mine) | the taxonomy **defined** "Verification passed" as *"a measurement **or a passing test** establishes the condition"* | **Deleted.** This is the `D-035`/G14 defect embedded in the adjudicating document |
| `STATUS_CURRENT_v2.md:65` (mine) | `REG/D-001` closed as "Verification passed — `test_gpu_reserve.sh` **50 passed / 0 failed**" | **Reopened as Unresolved.** A passing suite is not a verified mutex — and `D-047`/`D-048` then showed it is defeatable by one env var and by `cp` |
| `T2I_REPORT_r5.md` §1 (mine) | "All three corrected implementations have since been **run**" with no execution/verification distinction | **Rewritten** with three qualifications, including an explicit sentence that no passing suite in this project is evidence a training run completed |

**Reported elsewhere:** `D-036`'s "7/7 unit result" is **synthetic tensor perturbations**. The
register already says so plainly — "a unit test, not evidence the upstream trainer ran. A passing
suite is not an executed run" — which is correct and I endorse it. **But** `R2` stages 7/9/10 have
never run, so the guard has never seen a real trained checkpoint, and the gate table still phrases
G1/G2/G5 around test output (§3.4).

### 2.4 (d) Implemented / executed / independently verified

Audited all four of my documents for the stronger word where only the weaker is earned. **Found and
corrected:**

| where | was | now |
|---|---|---|
| `MANIFEST` §6 `REF_*` row | "the authors' released code **executed faithfully**" | **"upstream-code EXECUTION"**, with "executed does not imply faithfully", carrying `D-021`'s 5× budget and `D-037`'s false filter claim |
| `MANIFEST` §6.1 R1 row | "✅ **RAN.**" | "✅ **EXECUTED — and NOT a faithful reference reproduction**", with both defects and "executed, not independently verified" (its 3 attack seeds never ran) |
| `T2I_REPORT_r5` §7 E7 row | "upstream IMMA … and upstream UCE **executed**" | executed **plus** every qualification: 5× budget, attack seeds never run, port side confounded, R4c unreproducible, `r4c`/`r4d` incomparable, **R2 never executed** |
| `T2I_REPORT_r5` §1 | "have since been **run**" | "have since been **executed**" + the three qualifications |
| `STATUS` §1.1 / §1.4 | repairs listed without naming the verifier | each row now names **what** was verified and **by whom**; `D-038` is explicitly "verified as an import-and-assertion fix only — the trainer has never run" |

### 2.5 (e) and (f) — preserved, and strengthened

**(e) The IMMA 5× label is preserved in every document I own** and is now stated in the *row* rather
than in a distant footnote: **2500 ascent updates against the README's 500** (`D-021`). I also added
the reason it must not be removed — it is what keeps the run from being relabelled faithful — and
kept `D-037`'s counter-fact in the same breath, because both halves belong together: the filter never
ran **and** the README's own `shuffle(seed=42).select([20,40))` subset contains no blank frame, so
this is a **false provenance claim, not a corrupted training set**.

**(f) The provenance tiers are preserved** and the `MANIFEST` §6 taxonomy table is the right
instrument: upstream-code **execution** on our benchmark (`REF_*`), **domain adaptation**
(`ADAPT_*`), **historical project variants**, and **official reproduction** — for which *no prefix
exists* because **nothing in this project qualifies**. I tightened only the `REF_*` row's wording
(§2.4). The FRP/FRAG distinction is also preserved: the repository ships **0 Python files**, so our
FRP is a **reimplementation from the paper plus a domain adaptation**, which is verified and upheld.

## 3. Problems in files I do NOT own — reported, not fixed

### 3.1 `AUDIT_REPAIR_REGISTER_v1.md` — `D-019` leans past what the V1 control supports

**This is the §4 finding and the most consequential item in this section.** The task asked me to
check whether the register's `D-019` entry leans toward inferences the control does not establish.
**It does, on three of the four.**

| register text (`D-019` resolution block) | problem |
|---|---|
| "**The ESD checkpoint is exonerated.** By itself it is a **clean, well-behaved erasure**" | **This is a clean-erasure claim, and it is not supportable.** The target is **0.7333** — still detected in **11 of 15** images. A 1.000 → 0.733 move is *weak* suppression. More fundamentally, `D-030`'s own logic bars it: a closed-set argmax with **no reject option** cannot separate erasure from degradation **in either direction**. The register applies that limit correctly against "destroys the model" and then does not apply it to "clean erasure". §4 of this document's own text says claims of **"clean erasure"** require saved images and an independent assessment — the register's `D-019` block violates a rule stated elsewhere in the same corpus |
| "`D-019`'s provenance question is now **largely moot behaviourally**" | **Converts an untested question into a settled one.** `D-019 UPDATE` in the same file says **"EMA vs non-EMA survives and is untested"**, and the artifact's only provenance field is a `unet` path string — **no sha256, no checkpoint-state field**. A behavioural control was declared before it ran as something that "can never establish provenance". "Largely moot" walks that back |
| "**Over-training is now the leading hypothesis, and it is a hypothesis about our run**" | **Promotes a labelled hypothesis to a leading cause on evidence that cannot bear on it.** The second clause is exactly right and I endorse it. But **nothing in V1 varies the training budget**, so V1 provides **no** evidence for or against over-training. The register's own next paragraph says the README-budget replacement run "is what would separate over-training from method behaviour" — i.e. it is **untested**, which is incompatible with "leading" |
| "a whole-UNet-retrained erasure would be **unlikely** to leave 11/11 concepts at 1.000" | **Unsupported, and it is the universality claim.** One 12-concept panel, one target, n=15 per concept is not universality. All **686** tensors still differ from base with resnet blocks at **0.20** relative — unexplained, not exonerated |

**What the register gets right, and I endorse without qualification:** the lead's own hypothesis was
**wrong**; the collapse is **not inherited from ESD**; it is **introduced by composing IMMA**; and
the 5× label had to be preserved rather than the run relabelled faithful. The self-correction is
clearly recorded, which is the right behaviour.

**The only supportable statement**, now carried in `STATUS_CURRENT_v2.md` §2.3 and
`UPSTREAM_FIDELITY_MANIFEST.md` §6.1:

> **The ESD-alone arm did not show the same closed-set utility collapse on that panel.**

Verified from the artifact myself: all **11** non-target concepts at `clip_acc` **exactly 1.0**
(n=15 each); target `clip_align` **0.1646** against 0.225–0.300 for non-targets;
`utility_all_nontarget` **1.0000**; against the composed arm's **0.0424** with 7 of 11 at exactly
0.000.

### 3.2 `AUDIT_REPAIR_REGISTER_v1.md` — other issues found

| location | problem |
|---|---|
| `D-019`, end of entry | **"Status: OPEN, queued behind the infrastructure gate"** still sits *below* the "✅ RESOLVED" block **in the same entry**. One entry, two contradictory statuses. Also still says the control was "**deliberately NOT launched by hand**… queued", in the past-tense-pending voice, after it ran |
| `D-025` (:713, :729) | Still headed "**30 duplicate cells disagree**" and still says "up to **8/15** images" in the body. `D-051` corrects this to **29 disagreeing on `H_after`** (30 incl. `gate_valid`) and **11/15**. The `BASE(unmodified)` concentration — the single most useful fact about the duplicates — appears **only** in `D-051`, not in the entry a reader looks up |
| §5 gate table, **G1** | Gives `test_gpu_reserve.sh` output as evidence that exclusive reservation "**is proven to work**". Under `D-047`/`D-048` that suite passes while the mutex is defeatable by one env var and the interlock by `cp`. **As written, G1 is satisfiable by a green suite** — the exact pattern G14 forbids. It needs the env-var sweep and a non-name-based `nvidia-smi` check added |
| §5 gate table, **G2**, **G5**, **G14** | G14 says a green suite is not evidence its assertions are true, but G1, G2 and G5 are each phrased around test output. **The gate table should be swept once against its own G14** |
| §5 gate table, **G12** | Requires "180 images regenerated … and re-scored from `LABELS`" with "`FINDING65`'s ratios restated or withdrawn". The V2 run **did** this and the answer is that the records are **unrecoverable**. G12 as phrased can now never be satisfied in the "restated" branch; it should be rewritten to close on **withdrawal**, per the rule that an unrecoverable result needs no replacement |
| `D-024` repair line | Says "express the gate in panel units (`≥ 5/24`) wherever it is stated … **Status: evidence confirmed; wording repair pending**". With `D-024` **withdrawn**, this is a **pending repair for a defect that does not exist**. Restating the help text is harmless clarity; it must not be logged as closing `D-024`, and the words "granularity", "resolution" and "rounding" must not appear beside it |
| §4 quarantine table | The `gen_REF_imma_vangogh.json` row is correct and current. **But** the `r4b/r4c/r4d` row says "**R4d needs re-running**" without noting that `D-031` makes **R4c** unreproducible, so a reader may think both are recoverable |

### 3.3 Documents with the superseded FINAL framing

`REG/D-052` named these and they are still live. I verified the counts myself.

| file | problem |
|---|---|
| `CORRECTION_REGISTER_r5.md:265` | "**(18 records)**" — only **12** `final_*.json` exist on disk. I enumerated them: 6 `final_ATK_*`, 2 `final_BASE_*_anchor`, 4 pre-attack defence arms. `AUDIT_REPAIR_REGISTER_v1.md:1460`, `REPAIR_EVAL_v2.md:290` and `AGENT6_PROVENANCE_FINDINGS.md:300` all correctly say **12** |
| `CORRECTION_REGISTER_r5.md:408`, `HANDOFF_r5.md:140,154` | REG/D-022b described as a "**pure re-score**", "**deterministic**" and "**CPU-only**". The V2 run refutes all three: `history_comparability = none_exact`, **all six** regenerable arms DISCREPANT per-image, six more with deleted checkpoints. **This sells a permanent evidentiary loss as a cheap pending rerun** |
| `HANDOFF_r5.md:154,178-179` | Carries the superseded "**8/15**" duplicate spread (true figure **11/15**) |

### 3.4 Other read-only items

| file | problem |
|---|---|
| `AGENT4_EVAL_FINDINGS.md` §A-03 | Records "**four of twelve**" differing label strings (the count is **nine**) and the four-label counts "26 of 180" at margins "0.0048–0.0131". Under the nine-label definition: **37 of 180** at **0.0027–0.0063**. Each is right under its own definition; the file should say which it uses |
| `AGENT6_PROVENANCE_FINDINGS.md` §D-022 / §5.1 | Still states "CONFIRMED, **and the direction is as stated**" with target hits biased "**upward**". The V2 run **disproves any consistent direction** (`frp` −0.2000, `utrap_sd15` +0.0667). This is now refuted by measurement, not merely unsupported |
| `AGENT1_UCE_FINDINGS.md` §D-111 | "The 15/15-vs-0/15 **direction** is robust … the numbers are not" — not established by the reasoning the file applies elsewhere; should be marked unresolved pending the `r4c` re-run |
| `scripts/gen_eval_final.py:178` | Says "**26** of their 180 images" at "median margins 0.0065–0.0131" immediately after correctly stating that **nine** labels differ. Under the nine-label definition: **37 of 180** at 0.0027–0.0063. `D-045` adds that the per-arm split `(9,7,5,4)` sums to **25**, matches nothing, and that the count **does not bound** how many decisions could move |
| four live `outputs/r4_uce_*.json` | Still embed the **withdrawn 85,850** eigenvalue. The corrected **3.2420e+05** has **no artifact** on disk (`D-052`) — it is sourced to an agent document and a help string, so it is a corrected number resting on an unrecorded measurement |
| `outputs/gen_V1_esd_alone_vangogh.json` | For a record now carrying a published control, provenance is **only** the `unet` path string — **no sha256, no checkpoint-state field**. The register records the file's sha256 (`8881d70b…`) in prose but the artifact does not |
| `PROTOCOL_FROZEN_v1.md` | Still states none of the five true image denominators (1/15, 1/24, 1/120, 1/165, 1/48) |
| `scripts/pick_gpus.sh` | Still present and still iterates all four GPUs (`D-029`). With `D-048` live, this is an active hazard to another user's GPUs, not a tidy-up |

## 4. The ESD-control restriction, stated once for citation

**Artifact:** `outputs/gen_V1_esd_alone_vangogh.json` (15:39, 2,523 B, 180 images, ~95 s, GPU 0
reserved exclusively with the override logged, GPUs 2–3 untouched).

| arm | van_gogh | mean non-target (n=11) | at exactly 0.000 |
|---|---:|---:|---:|
| **ESD checkpoint ALONE** | **0.7333** | **1.0000** | **0 / 11** |
| ESD + IMMA composed | 0.8667 | 0.0424 | 7 / 11 |
| BASE SD-1.4 | 1.0000 | 1.0000 | 0 / 11 |

> **SUPPORTABLE: the ESD-alone arm did not show the same closed-set utility collapse on that panel.**
>
> **NOT established, and not to be claimed:** clean erasure · the checkpoint's training provenance ·
> that its weight changes are universally benign · over-training as the cause of the IMMA result.

**Why each is excluded** is in `STATUS_CURRENT_v2.md` §2.3 and `UPSTREAM_FIDELITY_MANIFEST.md` §6.1.
In short: target 0.7333 is 11/15 still detected and `D-030` bars both directions without a reject
option; this was declared a **behavioural** control before it ran and EMA-vs-non-EMA is untested;
one panel at n=15 for one target is not universality with 686 tensors still unexplained; and nothing
in V1 varies the training budget, so V1 cannot bear on over-training at all.

## 5. The registry and tables regeneration (task 4)

**`scripts/registry_v1.py` was NOT run**, as instructed. I worked from a **direct inventory of the
eligible artifacts** and reconciled it against the registry file by file.

**Measured and now stated in `TABLES_v1_generated.md`:**

- **808 attack records**, verified two independent ways — 808 files match `atk_*.json`, and exactly
  **808** JSONs carry the `attack_validity` key, with **zero** exceptions either way. (`T_rise` is
  nested at `/attack_validity/T_rise`; `asr` and `n_steps` do not exist in this corpus.)
- **1,448 JSONs** recursive (1,429 at root). So registry coverage is **1,278 of 1,448 — 170
  absent**, superseding the document's own "1,424 / 146". **0 parse errors** across all 1,448.
- The **112** absent attack records reconcile exactly: on disk but not in the registry **112**; in
  the registry but not on disk **0**. By prefix `E12` 24, `E2G` 18, `E2H` 12, `E4MG` 12, `E11J` 9,
  `E11M` 9, `E11K`/`E11L`/`E11N` 6 each, `E11H` 5, `E11I` 3, `E11O` 2.
- **The distinction the previous wording blurred, now stated in bold:** "absent" means **absent from
  the registry**, **not** missing from disk. **All 121 artifacts referenced by `FINDING71`–`FINDING78`
  are PRESENT on disk** (71: 7/7, 72: 4/4, 73: 10/10, 74: 24/24, 75: 7/7, 76: 22/22, 77: 7/7,
  78: 40/40). It is an **indexing** gap, closable by regeneration alone.

**Duplicate cells — corrected, with the concentration disclosed, and nothing adjudicated.** Figures
now carried: **68 duplicate cells over 178 rows, 29 disagreeing on `H_after`** (30 incl.
`gate_valid`), **max spread 0.7333 = 11 of 15**; over all 808, 74 cells across 224 files, 33
disagreeing. The previous "30 / 8-of-15" **understated** it — 8/15 is the strict-key subset only.
**All 29 belong to the single arm `BASE(unmodified)`**, which cuts both ways and both halves are
recorded: it **reduces** the concern (concentrated in the unmodified-base control, not spread through
the defence arms) and it remains **unexplained**.

**I named the affected rows rather than leaving it implicit:** the **15 `BASE(unmodified)` rows in
§1** and **no others** inherit a duplicate-cell disagreement. **No favourable duplicate was chosen
and no number was changed.** I also recorded the selection hazard that makes this live: of the 13
cells surviving the strictest key, 11 straddle a schema boundary and **the older stratum reports the
higher `H_after` in every disagreeing pair**, which `max(H_valid)` **systematically selects**; two
same-stratum disagreements have **no available explanation**; and 14 runners take the **first glob
match** then `break`, so which record lands in a curve is **filesystem-order-dependent**.

**What could NOT be computed, now an explicit table in the document** rather than an absence: the
generated §1–§3 rows refreshed to 808 records; `attack_records_quarantined` for the 808 (the
registry's 0 is **vacuously zero** — its only glob matches nothing); `vacuous` / `VOID` /
`admissible_for_recovery` / `admissible_for_paired_utility` for the 808 (these need the builder's own
classification logic); an **exact** defence-build count ("≥346" is a floor, not a computed figure);
per-build checkpoint verification for newer arms (**234 of 335** builds have `ABSENT` checkpoints —
~**70 %** of the corpus is not weight-verifiable); attribution for the six `—/—/—` builds
(**not recoverable**, `D-011`); and adjudication of the 29 disagreeing cells (**impossible from the
records** — no code-version field exists anywhere).

**Also corrected in that document:** the "**18** `final_*.json` records" → **12**, with the FINAL
corpus moved from "unusable pending corrected evaluation" to **permanently withdrawn**; and a new
`⚠` block stating that the **validity gate is not defective** so that no reader infers a gate defect
from the `gate-valid` columns.

## 5a. Third pass — REG/D-053 to REG/D-058, and a retraction of my own

Six entries were added to the register after my second pass, so the status document's coverage
claim went false again. Dispositions added, **four-value vocabulary only**, one each:

| id | disposition | the reason in one line |
|---|---|---|
| **D-053** `setsid` broke the interrupt path | **Unresolved, with the affected path excluded** | Agent A's `flock(2)`-on-an-inherited-descriptor is the mechanism that actually carries the invariant — retained through `fork`/`execve` by every descendant, so no timer and no leadership assumption. But the gate requires that **interrupted execution release ownership correctly**, and **no measurement of release-on-interrupt under the new mechanism is recorded**. Sound by construction ≠ verified. Rule 2 |
| **D-054** CVD refusal aborted on its own value | **Fixed and independently verified** | A deterministic control-flow error (test at stage 4, export at stage 2; plus `[ -n ]` admitting an exported-but-empty value that hides every GPU). Found **and** fixed by Agent A, neither of whom wrote it, and correctness is decidable by reading the order of export and test. Stated limit: **a clean `R1_MODE=prep` run has not been re-executed** |
| **D-055** stage 9 unreachable in smoke mode | **Unresolved, with the affected path excluded** | A fix that makes a stage *reachable* is not verified until the stage is *reached*, and R2 has never executed. Excluded: `R2_MODE=smoke` as evidence the trainer trains, and every stage-9 acceptance claim |
| **D-056** `coherent` forget pool leaked 2 of 3 target prompts | **Fixed and independently verified** | Found and fixed by Agent B against the scheme the lead treated as clean; the repair is a **hard-failing disjointness gate on resolved RUNTIME pools**, not source regex, which is exactly what missed it. **No result is affected — R2 never executed**, so a blocking split leak was caught before it reached a number |
| **D-057** instrumentation recorded nothing | **Unresolved, with the affected path excluded** | `AdamW` defines its own `step`, so patching `Optimizer.step` never fired. Fixed with global step hooks, **but 10/10 unit tests on synthetic tensors are not an executed run** — Agent B says so itself, correctly. This is D-036's third attempt: implemented, not executed |
| **D-058** published artifacts carry stale metadata | **Historical result withdrawn or quarantined** | `outputs/r4d_uce_church_*.json` were written 16:13/16:14 against a 16:18 fix and still read `n_preserve_port: 10` when the port used 0. **They must be RE-RUN, not re-labelled**, and Agent B correctly left them untouched |

**Three observations I want on the record.**

1. **D-058 means the revision I reviewed publishes known-wrong metadata.** `cc778f2` is the commit
   this whole review is pinned to, and it contains two artifacts whose `n_preserve_port` contradicts
   the code that produced them. That is not a reason to re-pin the review — it is a reason the R4d
   cell is **withdrawn** rather than corrected, and it compounds `D-026` and `D-031`, which already
   made R4d a required re-run and R4c unreproducible. The one piece of good news in the entry runs
   the other way and belongs in the record: Agent B found the `--no-preserve` equivalence **stronger**
   than claimed — bit-exact in **float32** too, numerator and denominator separately.
2. **D-055 is the fourth instance of one recurring pattern, and it is now the project's signature
   failure.** `D-013` (freeze marker cited but unread), `D-040` (guard with no production caller),
   `D-053` (leadership that protected nothing), `D-055` (a stage nothing could reach), and `D-057`
   (instrumentation that recorded nothing) are all the same error: **a control credited for its shape
   rather than for the property it enforces.** Two rounds of argument about stage 9's acceptance
   criteria were moot because the stage was unreachable. The useful generalisation is the test the
   register already states for itself — *name the caller, and name the production conditions it was
   exercised under* — and it should be applied to `D-053`'s `flock` next, not after the fact.
3. **D-056 is the one unambiguously good outcome here:** a blocking split contamination caught
   **before** it produced a number, in the scheme that was introduced as the clean replacement. It is
   worth noticing that the thing which caught it was changing the gate from **source regex to
   resolved runtime pools** — i.e. checking the artifact rather than the intent, which is the same
   lesson as item 2.

## 5b. RETRACTED: my "8 callers have not been migrated"

**Agent A is right and I was wrong.** There are **zero** live `campaign_free.sh` callers, and this
is **not** a gate blocker.

| check | result |
|---|---|
| files referencing `campaign_free` (excl. `upstream/`) | **11** — 8 runners, the script, `gpu_reserve.sh`, `test_gpu_reserve.sh` |
| **live invocations**, comments stripped | **ZERO**, except `test_gpu_reserve.sh:1220`, which invokes it deliberately to assert it stays retired |
| what the 8 runner references are | **all COMMENTS** documenting the migration away from it (e.g. `run_r1b_imma_train.sh:367` — the old selection is "**gone**") |
| `scripts/campaign_free.sh` itself | **retired by construction** (`:49`) — prints nothing on stdout, exits non-zero, so a stale waiter cannot get an index from it |
| `run_*.sh` using the reservation API | **10 of 140** |

**How I got it wrong, which matters more than the number.** I counted files that **mention**
`campaign_free` as if they **called** it. `grep -l` on a name cannot tell a caller from a comment
recording that the call was removed. That is the **same reasoning error as `D-040` and `D-013`,
inverted**: there, a control's *existence* was mistaken for its *use*; here, a *mention* was mistaken
for a *use*. Each of those 8 references is in fact **evidence of migration** — the opposite of what I
reported. I reproduced the failure mode I had just finished criticising in someone else's work.

**The real residual, stated correctly:** **130 of 140 `run_*.sh` do not use the reservation API**,
**deliberately** — they are byte-frozen historical runners whose markers are already declared
non-evidence (`D-040`), so G1/G2 scope is **active launchers only**. What remains genuinely
undetectable is a legacy campaign holding no reservation while idle between cells; that is a recorded
design limitation, not a migration gap.

**Reported, not fixed:** `scripts/test_gpu_reserve.sh` contradicts itself — `:377` says
"...any of its **8** callers" while `:1252` asserts "zero `run_*.sh` still INVOKE campaign_free.sh
(all **nine** migrated)". Eight runners reference it. One number is wrong, and the word **"callers"**
at `:377` carries exactly the ambiguity that produced my own error — worth fixing there so it does
not mislead the next reader as it misled me.

## 5c. The E2h progression is not a mechanism — caveats now attached everywhere

The coordinator noted having described `0.0667 → 0.2000 → 0.5333` as a **mechanism**. It cannot carry
that. Both facts are now attached at **every** site in the documents I own:

| site | state |
|---|---|
| `STATUS_CURRENT_v2.md` §1.6 | the per-seed table plus both caveats — the authority |
| `STATUS_CURRENT_v2.md` §6 (prior art) | "Stated with its true n" paragraph follows the progression directly |
| `T2I_REPORT_r5.md` §6 | per-seed table + caveats; "fails **only** when..." → "fails when..." |
| `T2I_REPORT_r5.md` §8 row 78 | **added** — n = 1, unpaired, "not a mechanism and cannot carry a dose reading" |
| `T2I_REPORT_r5.md` §8c near-miss | **added** — a near miss measured at n = 1 per cell |
| `T2I_REPORT_r5.md` §9 change log | **added** — and the word **"only"** dropped, because it implied an lr-dependent gate-failure rate the records refute |
| `T2I_REPORT_r5.md` §8 row 70 | **disambiguated** — `FINDING70` is a *different* arm at n = 3 seeds and must **not** be merged with the E2h n = 1 progression |
| `UPSTREAM_FIDELITY_MANIFEST.md` §2.2 | both caveats as numbered items |

**What the progression DOES support**, and the only thing it does: a **protocol requirement** — sweep
the attacker's lr at matched budget, because stopping one cell earlier would have reported this arm
as mandate-clearing. **What it does not support:** a dose relationship between lr and recoverability.
One gate-valid attack above the bar is sufficient to **refute** a mandate claim; three unpaired
single-attack cells are nowhere near sufficient to **establish a trend**. The asymmetry is the whole
point, and it is now stated at each site rather than left for the reader to supply.

## 6. The one thing a reader should take from this review

**Two dispositions in the previous status document were wrong in the same direction** — they
presented work as closer to closed than it is.

1. **`REG/D-001` was closed on a passing test suite.** It is an **active execution defect**, reopened
   by `D-047` and `D-048`, and no amount of documentation closes it.
2. **The FINAL corpus was sold as a cheap pending re-score.** It is a **permanent evidentiary loss**.
   The honest disposition is *withdrawn*, and **it needs no invented replacement to be complete.**

Both are now dispositioned correctly. **My own edits are unverified by anyone but me**, which is why
`REG/D-052`'s disposition in the status document excludes citing this revision as independently
verified until the lead or another agent reviews it.
