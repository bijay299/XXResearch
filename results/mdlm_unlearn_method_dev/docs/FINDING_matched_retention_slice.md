# At **matched retention**, AFC forgets substantially better than MDU: TV −0.142 (t=−4.06, 11/12) and disclosure −0.176 (t=−5.54, 11/12). This is the project's first real frontier result — and one unrun λ can still overturn it.

**Date:** 2026-10-24 · **n=12 paired seeds, both arms rebuilt under explicitly recorded settings**
**Pre-committed in** `run_matched_retain.sh`, before any of this data existed: *"if MDU λ=1 lands
near TV 0.30 at comparable retain, AFC is on MDU's frontier rather than above it and this project
has no advantage on either slice."* **It did not** — the pre-registered negative outcome did not
occur.
**Evidence:** `outputs/arm_mdu_l1_s{0..11}_mr.json` and the AFC / MDU λ=3 n=12 arms

---

## 1. Why this slice was needed

`FINDING_n12_verdict_retain_survives` settled the **matched-disclosure** slice: AFC retains +0.110
more capability (11/12, t=+3.15) at **tied** distance-to-pristine (6/12, t=−0.53). One axis, one tie.

That result alone cannot say whether AFC is *better than* MDU or merely *elsewhere on MDU's own
trade-off curve*. MDU has a λ knob; if turning it down to match AFC's retention also matched AFC's
forgetting, AFC would be a reparameterisation, not an improvement. **That was the live possibility,
and it is what this slice tests.**

## 2. Retention is matched, and it is matched almost exactly

| | AFC λ=1 | **MDU λ=1** | diff | t |
|---|---|---|---|---|
| retain ASCR | 0.5153 ± 0.0634 | **0.5235 ± 0.0940** | **−0.0082** | **−0.32** |

A paired difference of 0.008 at t=−0.32 is as closely matched as this protocol can deliver. MDU λ=1
is the right comparison arm for this slice, and MDU λ=3 (retain 0.406) is the wrong one.

## 3. At that matched point, AFC forgets clearly better

Paired across 12 seeds, AFC λ=1 against MDU λ=1:

| metric | AFC λ=1 | MDU λ=1 | paired Δ | se | t | AFC better on |
|---|---|---|---|---|---|---|
| **TV→pristine** | **0.2983 ± 0.1153** | 0.4400 ± 0.0525 | **−0.1417** | 0.0349 | **−4.06** | **11/12** |
| **DISCLOSE** | **0.1549 ± 0.0892** | 0.3306 ± 0.1116 | **−0.1757** | 0.0317 | **−5.54** | **11/12** |
| retain ASCR | 0.5153 | 0.5235 | −0.0082 | 0.0254 | −0.32 | 5/12 (matched) |
| DEGENERATE | 0.1493 ± 0.1292 | 0.0951 ± 0.0944 | +0.0542 | 0.0491 | +1.10 | 4/12 |

**At equal retention AFC is 0.142 closer to pristine and discloses 0.176 less, each on 11 of 12
individual seeds.** These are the largest and most consistent effects the project has measured, and
unlike the n=3 claims they are paired, seeded, and built on arms whose settings are recorded.

**The one axis where AFC loses remains DEGENERATE** (0.149 vs 0.095, losing 8 of 12). AFC's
failure mode is producing broken text, and that has been true at every n and in every slice. It is
not significant here (t=+1.10) but it is persistent and should not be dropped from any write-up.

## 4. The two slices together

Three measured operating points at n=12:

| arm | TV→pristine | retain ASCR |
|---|---|---|
| **AFC λ=1** | **0.2983** | **0.5153** |
| MDU λ=3 | 0.3223 | 0.4057 |
| MDU λ=1 | 0.4400 | 0.5235 |

AFC beats **λ=1 on forgetting at equal retention**, and beats **λ=3 on retention at tied
forgetting**. Against the MDU λ∈{1,3} frontier, AFC is **outside it, not on it.** That is a genuine
trade-off improvement over the baseline and it is the first claim in this project that has survived
every control thrown at it.

## 5. ⚠ One unrun λ can still overturn the dominance claim, and it is running now

A **two-point frontier cannot support "above it"**, and an existing **single-seed** measurement
contradicts the picture: `cv_mdu_l4` — MDU λ=4 — recorded **TV 0.2325 at retain 0.4565**. That is
*better* forgetting than AFC's 0.2983 at only modestly lower retention. If it holds at n=12, MDU's
frontier passes through or above AFC on the forgetting axis, AFC does **not** dominate, and §4's
claim collapses back to the single matched-disclosure retention win.

`run_frontier_fill.sh` is running MDU **λ=4 first**, then λ=2, at n=12. Pre-committed before the
data exists:

- **λ=4 reproduces ≈0.23 TV at ≈0.46 retain** → **no frontier claim can be made for AFC**; §4 is
  withdrawn and retention at matched disclosure is the only surviving result;
- **λ=4 regresses toward 0.30+** → the single-seed point was a lucky seed, and §4 stands on a
  four-point frontier (λ=1, 2, 3, 4).

Until that lands, **§4 must be read as provisional.** AFC's own λ=2 and λ=4 single-seed points
(TV 0.2338 / 0.2589 at retain 0.550 / 0.541) are likewise n=1 and are deliberately **not** used
anywhere in this finding — using a lucky single seed of our own while demanding n=12 of the baseline
is exactly the asymmetry that produced the project's earlier retractions.

## 6. What this still does not claim

- **Not robustness.** AFC recovers up to **62.7%** of erased facts under the strongest valid
  relearning attack, inside the baseline's 51–68% (`FINDING_rtt_control_gate_passes` §3). Nothing
  here changes that.
- **Not lower variance.** AFC's TV sd is 0.115 against MDU λ=1's 0.053 — roughly **2× higher**. The
  original n=3 stability claim remains refuted.
- **Not generality.** One dataset (TOFU), one backbone (LLaDA-8B-Instruct rev
  `08b83a6feb34df1a6011b80c3c00c7563e963b07`), one evaluator, one λ setting of AFC.

## 7. Provenance

All 36 arms across the three columns are published originals or rebuilds under explicitly recorded
settings, validated by three independent bridges and nine exact seed reproductions — including the
λ=1 rebuilds in this very sweep reproducing `mdu_s3`, `s4`, `s6`, `s7` exactly, which is what proved
the original λ confound (`FINDING_lambda_provenance_confound` §2). t-values are paired-sample, no
multiplicity adjustment across four correlated metrics; the pre-committed primary for this slice is
TV.
