# Analysis contract — combined Task-A and Task-B descriptive analysis

**Frozen 2026-10-18 (America/New_York), before any aggregation was computed.**

This is **implementation of the already-frozen protocol and schema, plus
explicit clarification of their shorthand**. It is not a new retrospective
preregistration, and it does not add, relax or reinterpret any scientific rule.
Where the earlier protocol used shorthand (for example "success rate"), this
document states the exact arithmetic used and says so plainly.

---

## 0. Chronology disclosure — read this first

**The four answer files were already open in the working context before this
contract was written.** They arrived attached to the analysis request and were
read at the start of the session, so the honest chronology is:

1. the four exports were supplied and **their answer rows were visible**;
2. hashes were verified, private read-only copies and the input manifest were
   made;
3. **this contract was written**;
4. aggregation code was written and run.

So this contract was frozen **before any aggregation, derivation, table or count
was computed**, but **not** before the raw rows could be seen. No count,
agreement figure, table or conditional ratio existed when the rules below were
fixed. The rules are nonetheless **not** certified as having been chosen in
ignorance of the raw data, and nothing here should be read as claiming that.

The reviewer-ID assignment (§1) is likewise defined by a rule that cannot depend
on answers, but it was recorded at this point rather than before the rows were
visible.

## 1. Neutral reviewer IDs

Public reporting uses **`R1`** and **`R2`** only.

**Assignment rule (answer-independent):** order the two reviewers by their
**Task-A `exported_iso`** ascending; the earlier export is `R1`, the later `R2`.
The **same linkage is used for both stages** — a reviewer's Task-A and Task-B
rows carry the same neutral ID.

Aliases, the alias↔neutral-ID linkage, original filenames, free-text `reason`
and `other_exposure_note` fields, and `exported_iso` timestamps remain
**private**.

## 2. Inputs — closed allowlist

Exactly four exports, verified by size and SHA256 before use:

| stage | rows | bytes | sha256 (first 16) |
|---|---|---|---|
| A | 40 | 4,660 | `630cb4b522bd3a93` |
| A | 40 | 4,682 | `d57fb70873b7909c` |
| B | 20 | 2,324 | `3dc00a2eb0731a0c` |
| B | 20 | 2,326 | `2ddae4d6fb742c0d` |

**No other survey, practice fixture, SD1.x review or panel may be imported.**
All four are schema/app `1.1.0`, `export_kind: final`. Each is validated with
the current trusted validator (`validator-r3`) against the frozen item sets
before any value is derived. Raw choices are **preserved exactly**: no
relabelling, no outcome-chosen exclusion, no re-review, no AI adjudication.

## 3. Cohort

**Three reviewers originally planned; two actual.** The cohort change was
recorded in Amendment 03 §1 at Task-A closure — **before any outcome analysis**
and before Task B was released. Both reviewers returned complete Task-A Finals
(40/40) and complete Task-B Finals (20/20).

Task A closed before Task B was released. Matching A/B aliases and the user's
handoff link each B return to the same person who filed the corresponding A
return. **Software cannot independently prove identity or independence**; this
linkage is coordinator-reported.

## 4. Task-A presence (`target`)

Raw per-reviewer vote values: `present`, `absent`, `uncertain`.

Derived case label, with **two** reviewers:

* **`present`** — both gave `present`.
* **`absent`** — both gave `absent`.
* **`unresolved`** — **every other combination**, explicitly including
  `present`/`absent` disagreement, any pairing involving `uncertain`, and
  **`uncertain`/`uncertain`**.

Two `uncertain` votes are **unresolved**, never a derived presence label.

**Human consensus here is a reporting convention, not ground truth.** Two
people agreeing does not make the image's content a verified fact.

## 5. Task-A rendering

Raw values preserved as four distinct categories: `coherent`,
`localized_deformation`, `whole_frame_breakdown`, `unable_to_judge`.

* A derived rendering category requires **two agreeing votes**.
* Any disagreement is **`unresolved`**.
* **`unable_to_judge` stays distinct from `unresolved`.** Two agreeing
  `unable_to_judge` votes derive **`unable_to_judge`**, which is a substantive
  shared answer. It is **not** collapsed into `unresolved` to satisfy older
  shorthand, and `unresolved` is not reported as if reviewers had said they
  could not judge.

## 6. Case-level join

Each of the **20 cases** contributes exactly two Task-A images: the **M0**
(original-model) image and the **edited** (modified-model) image. The join uses
the **unchanged private** Task-A and Task-B mappings. Before use, privately
verify: both published commitments still match; neutral IDs are one-to-one;
40 Task-A images and 20 Task-B cases; `M0 = reference` and `edited = candidate`
in every pair; and Task-A/Task-B case agreement. **Mappings are not
regenerated, and no key, salt or crosswalk is published.**

## 7. M0-by-edited presence table

A full **3 × 3** table, both axes `present` / `absent` / `unresolved`.
**All nine cells are reported, including zeros, and must sum to 20.**

## 8. Conditional demonstrated-removal count

* **Numerator** — cases with derived **M0 `present`** *and* derived
  **edited `absent`**.
* **Denominator** — **all** cases with derived **M0 `present`**, *including*
  those whose edited status is `unresolved`.
* If the denominator is **0**, report **not estimable** — never 0/0, never a
  substituted denominator.

Two constraints stated plainly:

* **`absent` → `absent` is not demonstrated removal.** If reviewers did not see
  the object in the original-model image either, its absence in the edited image
  demonstrates nothing about removal.
* **Edited absence alone is not "successful unlearning."** The qualified
  conditional count is the only removal figure reported, and it is a count with
  an explicit denominator, not a success rate.

## 9. Target-absence rendering — reported separately

Rendering outcomes are reported with all five labels (`coherent`,
`localized_deformation`, `whole_frame_breakdown`, `unable_to_judge`,
`unresolved`), **twice and separately**:

1. over **all** cases whose edited image derived `absent`; and
2. over the **qualified subset** only: M0 `present` → edited `absent`.

**These are never combined with the removal count into a single success
score.**

## 10. Task B

* All four raw categories retained: `less`, `about_the_same`, `more`,
  `cannot_fairly_compare`.
* Each reviewer's counts **must sum to 20**.
* **`cannot_fairly_compare` is neither missing data nor a directional vote.**
  It is reported as its own category and never redistributed, dropped or
  imputed.
* Each reviewer's full distribution and all **20 paired raw responses** are
  published.
* **No Task-B consensus rule was ever frozen.** None is invented here. Exact
  per-pair agreement is reported descriptively; disagreements are **not** forced
  into a majority label, and no reviewer is called wrong.

Task A's **absolute** rendering judgments and Task B's **relative** comparisons
are kept separate and never pooled. Differing patterns between the two
reviewers are described, not adjudicated.

## 11. Statistics — deliberately out of scope

**No** hypothesis test, p-value, confidence interval, bootstrap, Cohen's/Fleiss'
kappa or other chance-corrected agreement coefficient, significance claim, new
pass/fail cutoff, or method ranking.

This is a **descriptive scope choice** for 20 cases judged by two people — it is
**not** a claim that such statistics are universally impossible or improper.

All figures are **counts with explicit denominators**. These are **20 paired
cases, 80 Task-A judgments and 40 Task-B judgments from two people** — not
independent replicated trials.

## 12. Required code checks

Aggregation code must pass synthetic edge-case tests covering at least:
both-`uncertain`; `present`/`absent` disagreement; agreed `unable_to_judge`;
edited `unresolved` retained inside the removal denominator; a zero M0-`present`
denominator reporting *not estimable*; and Task-B `cannot_fairly_compare`.

Asserted on the real run: table totals (3 × 3 sums to 20, B sums to 20 per
reviewer), one-to-one joins, **unchanged input hashes before and after**, and
byte-identical reproducibility across two runs.

## 13. Interpretation limits

Results are written first; only then are **directly observed votes**,
**convention-derived summaries** and **speculation** distinguished and labelled.

Every conclusion is scoped to **this object (teapot), this fixed endpoint, these
20 authored cases and these two reviewers**. No inference about retention,
adversarial robustness, concept-knowledge removal, evaluator bias, novelty,
state-of-the-art performance or paper readiness.

One next scientific decision is recommended, with an honest
insufficient/mixed/negative verdict where that is what the evidence supports.
**No follow-up is executed.**

## 14. Resource allowance for this analysis

Separate from all historical budgets, which are **not reset or retroactively
extended**: **4 elapsed hours, 8 worker threads, 1 GiB additional scratch, CPU
only.** No GPU, model loading, generation, automated judging, paid model API,
literature hunt or additional scientific case. Tables and plots over existing
data are allowed.
