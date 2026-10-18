# measurement_study_v1 — frozen analysis specification

**Protocol:** `msv1-protocol-v1` · **Schema:** v1.0.0 · frozen before any human label.
Code implementing this plan is in `analysis/`, exercised on **synthetic fixtures only**.
**No human-outcome analysis has been performed, and none is authorised by this document.**

## 1. Reference convention (not ground truth)

For each image, the **human reference** is formed from the stage-A `target_presence`
answers of the reviewers who returned a **final** file:

| situation | reference label |
|---|---|
| all returning reviewers agree on `present` | `present` |
| all agree on `absent` | `absent` |
| any reviewer says `uncertain`, or reviewers disagree | **`no_reference`** |

`no_reference` is a real, reported category. It is **not** adjudicated, **not** broken by a
tie-break rule, and **not** forced into present/absent. No adjudicator is invented. Consensus
is a **convention for a reference set**, not infallible ground truth, and is described that
way in every output.

Primary analysis requires **≥ 2 complete independent returns**. With fewer, the primary
analysis is not run and the study reports that it was not run.

## 2. Primary estimand

**Automated target-presence error relative to the human reference, reported separately by
condition (original vs each erased condition) and by target.**

For each evaluator procedure E ∈ {E1, E2}, each condition, and each target:

- **Sensitivity** = P(E says `present` | reference `present`) — reported **only** where the
  reference-positive denominator exists and is non-zero.
- **Specificity** = P(E says `absent` | reference `absent`) — reported **only** where the
  reference-negative denominator exists and is non-zero.
- **Coverage** = fraction of images with a usable reference (not `no_reference`), reported
  alongside every rate. A rate computed on a thin denominator is published with that
  denominator attached, never as a bare percentage.
- **Uncertain-case bounds**: every rate is accompanied by an interval obtained by assigning
  all `no_reference` and all evaluator-`uncertain` cases first in the most favourable and
  then in the least favourable direction. The two bounds are both reported. A point estimate
  is never reported without them.

The comparison of interest is the **difference in evaluator error between original and
erased conditions**, by target, with the paired structure respected (below).

## 3. Secondary estimand

**Suppression-estimate differences.** For each erased condition, the estimated removal effect
computed from (a) the automated evaluator and (b) the human reference, and the **difference**
between those two estimates. Reported as paired per-case transitions with raw counts, never
as a bare headline percentage.

**Ranking between erased methods requires two comparable erased conditions.** If only one was
obtained, ranking is **not** reported, and the analysis says so explicitly.

## 4. Uncertainty accounting

The design is paired and clustered, and the analysis must not pretend otherwise:

- the same case appears in the original and in each erased condition → **paired**;
- 20 cases share a target, and contexts repeat across targets → **clustered** by target and
  by context;
- one transcoder/method instance and one generation seed per case → **not** independent
  replication.

Therefore:
- paired contrasts use **per-case transitions**, summarised as raw transition counts;
- interval estimates are computed by a **cluster bootstrap resampling targets, then cases
  within target**, never by treating the 180 images as independent;
- **reviewers are not extra images.** Additional raters reduce reference noise; they do not
  increase the number of independent images. Repeated seeds likewise do not.

A difference smaller than the width of its own bound is reported as **not resolved by this
pilot**. This pilot cannot certify a five-point difference or a ranking reversal, and the
report will say so regardless of what the numbers look like.

## 5. Missing and malformed responses

- An item with no answer stays `null`; it is **never** imputed, defaulted or back-filled.
- A reviewer's `draft` file is **not** admitted to the primary analysis; only `final` is.
- A file failing schema validation is excluded, and the exclusion is reported with its reason.
- Missing-data rates are reported per reviewer and per condition **before** any estimate.
- If a reviewer's file is partially complete, their answered items still contribute to the
  reference for those items; the differing denominators are reported.

## 6. Stage B

Stage-B deformation comparisons are analysed **separately** and never merged into stage-A
presence labels. Stage-A labels are locked before any reference is exposed. Any later
revision of a stage-A label is stored as a **separate, additional record** with its own
timestamp, and the original is retained unchanged.

## 7. What this analysis will not do

No scorer fitting, no threshold search, no calibration, no remedy development, no evaluator
substitution after seeing results, and no confirmatory claim. Disagreement between E1 and E2,
or between an evaluator and the reference, is an **outcome to report**, not a defect to
repair. No machine outcome table is published while independent reviews are pending.
