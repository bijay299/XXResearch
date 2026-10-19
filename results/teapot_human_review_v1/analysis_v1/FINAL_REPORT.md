# Combined Task-A / Task-B descriptive analysis — teapot human review v1

**Status `ANALYSIS_COMPLETE` · `CLOSED WITH QUALIFIED CONCLUSIONS`.
Dated 2026-10-18.**

> **Read [`AMENDMENT_01_CONCLUSIONS_AND_PROVENANCE.md`](AMENDMENT_01_CONCLUSIONS_AND_PROVENANCE.md)
> alongside this report — it is authoritative where the two differ.** The
> numerical results below were independently verified and are **unchanged**.
> Several *interpretive* claims in §9–§10 as first published were overstated —
> an instrument ceiling, "badly under-powered," a diagnosed calibration
> failure, a "one-directional" reviewer difference, and proven binding
> constraints — and are **withdrawn or re-scoped** there and in the corrected
> text below. The amendment also separates two different reproducibility
> limits and corrects the masking account from proof to a procedural account.
> Closure, not a larger panel, is the recorded outcome.

Descriptive analysis of **existing** human judgments. **No new experiment, no
new label, no model run, no image generation, no automated judging, no GPU.**
Every original response is preserved byte-for-byte and unchanged.

Analysis ID `THR-ANALYSIS-V1` · package `THR-V1-A` / `THR-V1-B` · protocol
`TEAPOT-HR-PROTO-1.0.0` · schema `THR-EXPORT-SCHEMA-1.1.0` · app `1.1.0` ·
validator `validator-r3`.

The rules were fixed in [`ANALYSIS_CONTRACT.md`](ANALYSIS_CONTRACT.md) before
any aggregation. **Read §0 of that contract**: the answer rows were already
visible when the rules were written, and the contract says so rather than
claiming a blind freeze.

---

## 1. What this is counting

**20 paired cases and 120 reviewer-item records carrying 200 categorical
answers, from two people:** 80 Task-A records (2 reviewers × 40 images), each
with **two** answers — `target` and `rendering`, so 160 answers — plus 40
Task-B records (2 reviewers × 20 pairs) with **one** `comparison` answer each,
so 40 more. 160 + 40 = **200 answers across 120 records**.

These are **not** independent replicated trials, and nothing below should be
read as if they were.

| | |
|---|---|
| Reviewers planned | **3** |
| Reviewers actual | **2** (`R1`, `R2`) |
| Frozen minimum | ≥ 2 complete independent returns — **met** |
| When the cohort change was recorded | Amendment 03 §1, at Task-A closure — **before any outcome analysis** and before Task B was released |

`R1`/`R2` are neutral IDs assigned by an **answer-independent** rule (Task-A
export time ascending), with the **same linkage across both stages**. Aliases
and the alias↔ID linkage stay private.

> **Identity and independence are coordinator-reported, not proven.** Matching
> Task-A/Task-B aliases and the user's handoff link each B return to the person
> who filed the corresponding A return. **Software cannot verify that two
> returns came from two distinct people who worked independently.** Every
> figure below inherits that limitation.

### Participation and completeness

| | R1 | R2 |
|---|---|---|
| Task-A Final | **40 / 40** | **40 / 40** |
| Task-B Final | **20 / 20** | **20 / 20** |
| Schema / app | `1.1.0` | `1.1.0` |
| Task-B build | `taskB-b1` | `taskB-b1` |
| Order variant (both stages) | `ALIAS` | `ALIAS` |
| Validator verdict | **valid, 0 errors, 0 warnings** | **valid, 0 errors, 0 warnings** |

### Declared exposure — all four declarations present

Both reviewers declared, for the period **before Task A**: no prior access to
source pairings, no prior access to condition labels, no image-specific
coaching. Both declared their **Task-A Final was exported before starting Task
B**, and both Task-B files record
`reference_identity_disclosed_in_this_stage: true`. Neither left an
`other_exposure_note`. The Task-B declarations **agree** with the Task-A ones
on both pre-A exposure questions.

R1 left **one** free-text explanation across both stages; R2 left none. The
text itself is withheld (§8).

---

## 2. Task A — raw votes

**No reviewer ever used `uncertain`, `whole_frame_breakdown` or
`unable_to_judge`.** Every one of the 80 presence judgments was definite, so
the "both uncertain" and "agreed cannot judge" paths in the frozen rules were
exercised only by the synthetic tests, never by real data.

### Presence (`target`), per reviewer, denominator 40 images each

| | present | absent | uncertain |
|---|---|---|---|
| **R1** | 22 | 18 | **0** |
| **R2** | 21 | 19 | **0** |

### Rendering, per reviewer, denominator 40 images each

| | coherent | localized_deformation | whole_frame_breakdown | unable_to_judge |
|---|---|---|---|---|
| **R1** | 38 | 2 | **0** | **0** |
| **R2** | 34 | 6 | **0** | **0** |

### Definite-vote coverage per image

| definite votes | 0 | 1 | 2 |
|---|---|---|---|
| images | **0** | **0** | **40** |

Every image carries two definite presence votes.

### Exact raw agreement, with denominators

**Presence:**

| subset | n | definite consensus | agreement on `uncertain` | disagreement |
|---|---|---|---|---|
| all images | 40 | **39** | **0** | 1 |
| m0 images | 20 | **20** | 0 | 0 |
| edited images | 20 | **19** | 0 | 1 |

All 39 agreements are **definite consensus**; there is **no** case of the two
reviewers merely agreeing that they were uncertain. The distinction matters and
is reported because the frozen rules treat the two very differently.

**Rendering:**

| subset | n | agreement | disagreement |
|---|---|---|---|
| all images | 40 | **34** | 6 |
| m0 images | 20 | 19 | 1 |
| edited images | 20 | **15** | 5 |

### Derived labels over the 40 images

| presence | count | | rendering | count |
|---|---|---|---|---|
| present | **21** | | coherent | **33** |
| absent | **18** | | localized_deformation | **1** |
| unresolved | **1** | | whole_frame_breakdown | 0 |
| | | | unable_to_judge | **0** |
| | | | unresolved | **6** |

**`unable_to_judge` is 0 and `unresolved` is 6. These are different things and
are not merged.** The 6 unresolved renderings are reviewer *disagreements*;
neither reviewer ever said they could not judge.

---

## 3. The 20-case presence table — all nine cells

Rows = derived **m0** (original-model) presence. Columns = derived **edited**
(modified-model) presence.

| m0 \ edited | present | absent | unresolved | **row total** |
|---|---|---|---|---|
| **present** | 1 | **18** | 1 | **20** |
| **absent** | 0 | 0 | 0 | **0** |
| **unresolved** | 0 | 0 | 0 | **0** |
| **column total** | **1** | **18** | **1** | **20** |

Reviewers reached **consensus that the teapot was present in all 20
original-model images**. The entire `absent` and `unresolved` m0 rows are
empty, so every case is in the removal denominator.

Full per-case detail: [`tables/paired_case_ledger.csv`](tables/paired_case_ledger.csv).

---

## 4. Conditional demonstrated-removal count

> **18 / 20**

* **Numerator 18** — cases with consensus m0 `present` **and** consensus edited
  `absent`.
* **Denominator 20** — **all** consensus m0-`present` cases, *including* the one
  whose edited status is `unresolved`.

Denominator breakdown by edited status: **absent 18**, **present 1**,
**unresolved 1**.

Two things this figure is **not**:

* It is **not** a success rate, and it is **not** "successful unlearning."
  Edited-image absence on its own is not reported as success anywhere in this
  analysis.
* **`absent` → `absent` would not have counted**, and here could not arise: no
  case had consensus m0 `absent`. Had reviewers not seen the teapot in the
  original-model image, its absence in the edited image would demonstrate
  nothing.

The single `unresolved` case (**B-15**) is **retained in the denominator**, not
dropped. The single edited-`present` case is **B-17**.

---

## 5. Target-absence rendering — reported separately, never merged with §4

Rendering of the **edited** image, for the 18 cases whose edited image derived
`absent`:

| subset | n | coherent | localized | whole-frame | unable_to_judge | unresolved |
|---|---|---|---|---|---|---|
| all edited-`absent` cases | 18 | **13** | 0 | 0 | **0** | **5** |
| qualified: m0 `present` → edited `absent` | 18 | **13** | 0 | 0 | **0** | **5** |

**The two rows are identical here** because every edited-`absent` case also had
consensus m0 `present` — a consequence of the empty m0-`absent` row, not a
collapsing of the two subsets. They are reported separately regardless, because
the frozen rules require it and because they would diverge on other data.

**All 5 unresolved renderings *within this subset* come from the same
direction:** R2 recorded `localized_deformation` where R1 recorded `coherent`.
This is the correctly scoped statement — it holds for these five, **not** for
all six rendering disagreements overall, one of which runs the other way
(§9). Neither reviewer reported
whole-frame breakdown or an inability to judge on any case.

**This is not combined with §4 into a single score.**

---

## 6. Task B — relative comparison

Each reviewer's distribution, denominator **20 pairs each**:

| | less | about_the_same | more | cannot_fairly_compare | total |
|---|---|---|---|---|---|
| **R1** | 1 | **18** | 1 | **0** | **20** |
| **R2** | 0 | **18** | 2 | **0** | **20** |

**Neither reviewer used `cannot_fairly_compare`.** That is reported as a real
zero in a category that was explicitly offered and explicitly described to
reviewers as a legitimate answer — not as missing data.

**Exact per-pair agreement: 18 / 20.** The two disagreements:

| pair | R1 | R2 |
|---|---|---|
| **B-01** | `less` | `about_the_same` |
| **B-06** | `about_the_same` | `more` |

**No Task-B consensus rule was ever frozen, and none is invented here.** The
18/20 figure is descriptive exact agreement, nothing more. The two
disagreements are **not** resolved into a majority label, and neither reviewer
is called wrong. Full per-pair votes:
[`tables/taskB_responses.csv`](tables/taskB_responses.csv).

Task A's **absolute** rendering judgments and Task B's **relative**
comparisons are kept separate throughout and are never pooled.

---

## 7. Descriptive strata from the frozen source records

These strata (`ordinary`/`challenging`, and category) were fixed in the frozen
case records **before** any image was generated. They are **not** subgroups
chosen after seeing outcomes.

| stratum | cases | removal count | B: less | B: same | B: more | B: cannot |
|---|---|---|---|---|---|---|
| **ordinary** | 10 | **10 / 10** | 0 | **20** | 0 | 0 |
| **challenging** | 10 | **8 / 10** | 1 | 16 | 3 | 0 |
| plain | 10 | 10 / 10 | 0 | 20 | 0 | 0 |
| composition | 4 | 3 / 4 | 0 | 7 | 1 | 0 |
| occlusion | 3 | 3 / 3 | 1 | 5 | 0 | 0 |
| style | 3 | 2 / 3 | 0 | 4 | 2 | 0 |

B columns count **judgments** (2 reviewers × cases), not cases. Every
non-`about_the_same` Task-B vote and both non-removal cases fall in the
`challenging` stratum. With 3–4 cases in some categories, **these cells are far
too small to support any comparison between categories**, and none is made.

---

## 8. What is withheld, and what that costs

Withheld from every public artifact, and **not deleted from the originals**:
reviewer aliases; the alias↔`R1`/`R2` linkage; original export filenames;
free-text `reason` values; `other_exposure_note` values; `exported_iso`
timestamps; source image filenames and hashes; case IDs; neutral-ID→condition
associations; the randomization salt and commitment keys; the private input
manifest and backup contents.

The four input SHA256 digests **are** published, keyed by neutral ID and stage,
for integrity. Anyone who separately holds the private filename manifest could
use them to link `R1`/`R2` to an alias; that manifest is private.

**Two different reproducibility limits, which earlier wording blurred into
one.** They point in opposite directions and must be kept apart:

* **Condition and pair assignment IS recoverable.** It can be reconstructed by
  hashing the already-public Task-A, Task-B and source-asset images — see
  below. **An independent review reconstructed it that way.** Withholding a
  direct crosswalk **does not make the assignment unreproducible**, and this
  report does not claim it does. (Earlier text said a third party "cannot
  independently reproduce the m0/edited assignment"; **that was wrong** —
  recovering it needs only the public bytes, not the private mapping.)
* **Export-level reproduction IS limited.** The exact original export files,
  free-text answers, original export metadata and the reviewer↔`R1`/`R2`
  linkage are **private**, so a third party cannot rebuild the tables from the
  original artifacts or verify the reviewer linkage.

A third party can recheck every table's internal arithmetic, totals and joins
from the published files, and can recover the condition assignment from public
image bytes.

**On the withheld list above.** It names "case IDs" and "neutral-ID→condition
associations". That means **private source identifiers and direct
crosswalks** — it does **not** mean every case identifier is withheld, and it
does **not** claim no association is inferable. **The Task-B pair IDs
`B-01`…`B-20` are published** in
[`tables/taskB_responses.csv`](tables/taskB_responses.csv) and
[`tables/paired_case_ledger.csv`](tables/paired_case_ledger.csv).

### Pseudonymity here is weak, and saying otherwise would be false

**The neutral item IDs do not hide which condition an image came from.** Both
published bundles ship images **byte-identical** to the per-case images in the
public `modern-object-assets-v1` release, which publishes each case's
`m0`/`edited` filenames **with their SHA256**. Verified during this analysis:
**40/40** images in each bundle match a public case/condition hash. So anyone
who hashes the public files can recover `A-xxx` → case/condition and `B-xx` →
case/role **from already-public data alone**.

Consequences stated plainly:

* This analysis **does not create** that exposure, and **does not remove** it.
* Publishing votes keyed by `A-xxx` is, in disclosure terms, equivalent to
  publishing them keyed by case and condition. **No direct crosswalk is added
  here**, but withholding one buys little.
* **Masking at judging time rests on a procedural account, not on proof.**
  Both reviewers **declared** no pre-Task-A access to source pairings or
  condition labels and no image-specific coaching, and independence was
  **user-reported**. The protocol provided for masked judging and those
  declarations are consistent with it — but declarations and aliases **cannot
  establish what a reviewer knew**, so this is not proof that judging-time
  masking held. Equally, **nothing here indicates that it failed**; this report
  asserts neither. (Earlier text called it "real and unaffected"; that
  overstated a declaration as a verified fact.) Task B was **deliberately**
  unmasked.
* `R1`/`R2` are pseudonyms, **not anonymity**. They do not undo the earlier
  roster disclosure recorded in Amendment 03 §7b and do not guarantee
  anonymity.

---

## 9. Evidence, convention, and speculation — kept apart

**Directly observed (what the reviewers actually recorded):** every count in
§2 and §6 — the raw per-reviewer distributions, the 0 uses of `uncertain`,
`whole_frame_breakdown`, `unable_to_judge` and `cannot_fairly_compare`, the
39/40 presence agreement, the 34/40 rendering agreement, the 18/20 Task-B exact
agreement, and all **200 categorical answers across 120 reviewer-item
records** in the published tables. (Earlier text said "120 individual
categorical votes"; that conflated records with answers — §1.)

**Convention-derived (true given the frozen rules, and only given them):** the
derived `present`/`absent`/`unresolved` labels, the 3×3 table, the **18/20**
removal count, and the §5 rendering breakdowns. These depend on the rule that
**two** agreeing definite votes constitute consensus. **Two people agreeing is
a reporting convention, not ground truth** — it does not establish what is
actually in an image.

**Hypothesis, and labelled as such:** that R2 applied a stricter rendering
threshold than R1. The **observation** is that R2 recorded 6
`localized_deformation` against R1's 2, and that **of the six rendering
disagreements, five** have R2 marking `localized_deformation` where R1 marked
`coherent` — while **the sixth runs the other way**. So the pattern is **not
one-directional**, and the earlier description of it as such is withdrawn. A
stricter-threshold explanation **remains a hypothesis**: it is equally
consistent with ordinary rater variation, and nothing here separates the two.

Correctly scoped and retained: the **five** disagreements that fall **inside
the 18 edited-absent subset** do all run the same direction (§5). All six
disagreements are `coherent` ↔ `localized_deformation`; no other pair of
rendering values was ever in conflict. **That the six arose from absent shared
anchors is also only a hypothesis** — no calibration study was run, and no
anchor failure is diagnosed here.

Also hypothesis only: any explanation of *why* the `challenging` stratum holds
every non-`about_the_same` Task-B vote.

### What this does not establish

Scoped to **one object (teapot), one fixed endpoint, 20 authored cases and two
reviewers**. This analysis establishes **nothing** about retention of unrelated
concepts, adversarial robustness, whether concept *knowledge* was removed
rather than its *expression* suppressed at this endpoint, evaluator bias,
novelty, state-of-the-art standing, generalization to other objects or models,
or paper readiness. No hypothesis test, interval, bootstrap, kappa,
significance claim, pass/fail cutoff or method ranking is reported — a
**descriptive scope choice** for this design, not a claim that such statistics
are impossible in principle.

---

## 10. The supported conclusion, and closure

**On this fixed panel:** both reviewers marked the target **present in all 20
M0 images**. Applying the frozen consensus rule to the edited images yields
**18 absent, one present and one unresolved**; among the 18 absent cases, **13
are agreed coherent and five are rendering-unresolved**. Task-B ratings were
**concentrated at `about_the_same`** — 36 of 40, with 3 `more`, 1 `less` and 0
`cannot_fairly_compare`.

**Three explanations of the Task-B concentration remain compatible**, and this
analysis does not choose between them:

1. genuine similarity in quality between the two conditions;
2. limited stimulus variation across this particular 20-case panel;
3. limited sensitivity of the comparison instrument.

**This supports a bounded descriptive finding about visible target depiction at
one fixed endpoint**, on this panel, as judged by these two reviewers. It does
**not** establish general unlearning, removal of concept knowledge, retention,
adversarial robustness, evaluator bias, novelty, superiority over any method,
or paper readiness.

**The 13 agreed-coherent absences are not a success score** and are not
combined with the 18/20 count, or with each other, into any composite figure.

This is a **small fixed descriptive panel with limited generalization**: 20
paired cases, 200 categorical answers across 120 records, two reviewers, one
object, one endpoint. It is deliberately **not** described as under-powered —
no inferential test, estimand, meaningful effect size or power calculation was
ever specified for it, and none is computed now.

### Closure

**This fixed analysis is closed.** No next step is prescribed.

Any future enlargement of the panel, calibration pass, additional reviewers or
method comparison is an **unselected prospective option** that would need its
own justified question, design, resourcing and authorization. None is selected
here, none is claimed to be required, and **no follow-up is launched**.

Earlier wording recommended a larger calibrated panel as the necessary next
step and named case count, reviewer count and calibration as binding
constraints. **Both are withdrawn** — see
[`AMENDMENT_01_CONCLUSIONS_AND_PROVENANCE.md`](AMENDMENT_01_CONCLUSIONS_AND_PROVENANCE.md)
§3.5 and §5. Nothing here forecloses whether some other bounded comparison
could be justified; this analysis simply did not examine that question.

---

## 11. Preservation

| | |
|---|---|
| Four original exports | **unchanged**, byte-verified before and after analysis |
| Input SHA256 before == after | **yes** |
| Task-A and Task-B responses | **no relabelling, no exclusion, no re-review, no AI adjudication** |
| Private mappings, salt, commitments | **unchanged**; both commitments re-verified (20/20) |
| `freeze_ids.py` | **not** re-run; no ID regenerated |
| Both Task-A releases + Task-B release, assets, tag targets | **unchanged** |
| Either viewer | **not** rebuilt |
| Asset-build findings | **not** updated |
| Closed-project files | **unchanged** |

## 12. Verification artifacts

| what | file | result |
|---|---|---|
| **Conclusions/provenance amendment** | [`AMENDMENT_01_CONCLUSIONS_AND_PROVENANCE.md`](AMENDMENT_01_CONCLUSIONS_AND_PROVENANCE.md) | **authoritative on interpretation** |
| Frozen rules | [`ANALYSIS_CONTRACT.md`](ANALYSIS_CONTRACT.md) | incl. honest chronology disclosure, preserved unchanged |
| Amendment preservation check | [`preservation_check.json`](preservation_check.json) | before/after, mechanical |
| Amendment change-set scan | [`amendment_public_audit.json`](amendment_public_audit.json) | separate from the historical audit |
| Machine-readable summary | [`summary.json`](summary.json) | — |
| Synthetic edge cases | [`synthetic_checks.json`](synthetic_checks.json) | **35 / 35** |
| Arithmetic, joins, reproducibility | [`verification.json`](verification.json) | **54 / 54** `ANALYSIS_CHECKS_PASS` |
| Private mapping integrity | [`mapping_verification.json`](mapping_verification.json) | **20 / 20** `MAPPINGS_VERIFIED_UNCHANGED` |
| Public leak audit | [`public_audit.json`](public_audit.json) | **0 findings** `ANALYSIS_PUBLIC_AUDIT_CLEAN` |
| File hashes | [`MANIFEST_SHA256.md`](MANIFEST_SHA256.md) | — |
| Provenance | [`provenance.json`](provenance.json) | — |

Reproducibility was checked by running the analysis twice: **all public tables
byte-identical**, `summary.json` identical apart from its timestamp.

All verification above is **host-performed** — this machine ran these checks on
its own outputs. It is **not** an independent attestation.

### Analysis code

[`scripts/analysis_rules.py`](scripts/analysis_rules.py) (rules, no I/O) ·
[`scripts/analyze_v1.py`](scripts/analyze_v1.py) (driver) ·
[`scripts/test_analysis_synthetic.py`](scripts/test_analysis_synthetic.py) ·
[`scripts/verify_analysis.py`](scripts/verify_analysis.py) ·
[`scripts/verify_mappings_for_analysis.py`](scripts/verify_mappings_for_analysis.py) ·
[`scripts/audit_analysis_public.py`](scripts/audit_analysis_public.py)

No private identifier is a literal in any of them; secrets are loaded from
private records at run time, and a missing private record makes a scan report
**SKIPPED** rather than silently pass.

## 13. Accounting — appended; historical clocks not reset

| | used | cap |
|---|---|---|
| Elapsed (analysis phase) | **0.4 h** | 4 h |
| Worker threads | 1 (CPU only) | 8 |
| Additional scratch under `/data` | far under cap | 1 GiB |
| GPU / model / generation / automated judging / paid API | **0** | 0 |
| New images, labels or experiments | **0** | 0 |
| Original responses modified | **0** | 0 |

The documentation/provenance/index amendment is accounted for **separately**
against its own 2 h allowance in
[`AMENDMENT_01_CONCLUSIONS_AND_PROVENANCE.md`](AMENDMENT_01_CONCLUSIONS_AND_PROVENANCE.md)
§12. This analysis-phase row is **not** extended to cover it.

All historical preparation, release, maintenance and analysis clocks are
preserved unchanged and were not reset or retroactively extended.
