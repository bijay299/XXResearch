# The n=12 verdict: **AFC's retention advantage survives** (+0.110, 11/12 paired seeds, t=+3.15). Its TV advantage does not (−0.024, 6/12, t=−0.53). The contribution is retention at matched disclosure, and nothing else.

**Date:** 2026-10-24 · **Both columns complete at n=12** under explicitly recorded settings
**Pre-committed in** `run_n12_extension.sh`, before any n=12 data existed: *"retain delta holds →
it is the contribution; retain delta shrinks → this project has no surviving advantage over MDU and
that is the result."*
**Evidence:** `outputs/arm_afc_l1_s{8..11}_n12.json`, `outputs/arm_mdu_l3_s{8,9,10}_n12.json`,
`outputs/arm_mdu_l3_s11_n12b.json`, plus the n≤8 arms

---

## 1. The result

Paired across 12 seeds. AFC λ=1 against MDU λ=3, the matched-disclosure baseline.

| metric | AFC λ=1 | MDU λ=3 | paired Δ | se | t | AFC better on |
|---|---|---|---|---|---|---|
| TV→pristine | 0.2983 ± 0.1153 | 0.3223 ± 0.0751 | **−0.0240** | 0.0456 | **−0.53** | **6/12** |
| **retain ASCR** | **0.5153 ± 0.0634** | 0.4057 ± 0.1061 | **+0.1096** | 0.0348 | **+3.15** | **11/12** |
| DISCLOSE | 0.1549 ± 0.0892 | 0.1889 ± 0.0884 | −0.0340 | 0.0187 | −1.82 | 9/12 |
| DEGENERATE | 0.1493 ± 0.1292 | 0.1271 ± 0.0963 | +0.0222 | 0.0481 | +0.46 | 4/12 |

**Retention survives.** +0.1096 with a paired se of 0.0348 is t=+3.15, and it holds on **11 of 12
individual seeds** — the lone exception is seed 9 (−0.170). Per-seed deltas: +0.129 +0.165 +0.099
+0.128 +0.202 +0.155 +0.213 +0.038 +0.304 **−0.170** +0.023 +0.031.

**TV does not.** −0.0240 at se 0.0456 is t=−0.53, and the paired count is **6/12 — exactly even**.
There is no TV advantage here. The claim in `RESULTS.md` §1 was *"0.223 vs 0.352 — 37% lower"*; the
honest n=12 figure is **0.298 vs 0.322**, a difference smaller than half of one seed's sd.

**DEGENERATE goes the other way:** AFC is worse (0.149 vs 0.127), losing 8 of 12 seeds. The n=3
claim of 0.117 vs 0.181 was backwards.

## 2. How the retain delta moved as n grew, which is the thing to look at

| | n=3 | n=8 | **n=12** |
|---|---|---|---|
| AFC retain | 0.5761 | 0.5336 | **0.5153** |
| MDU retain | 0.4454 | 0.3928 | **0.4057** |
| **delta** | +0.1307 | +0.1408 | **+0.1096** |

The delta **shrank by a sixth** from n=8 to n=12 and AFC's own retain fell monotonically at every
step (0.576 → 0.534 → 0.515). That is the same drift that destroyed every other claim this project
made. It did **not** destroy this one: the delta is still 3.15 paired standard errors and 11/12
seeds. But the direction of travel is recorded here so that a future n=16 is read as a test rather
than a formality, and the pre-commitment is met honestly — the delta shrank *and* survived, so the
reading is "survives, diminished", not "confirmed".

## 3. What this project can and cannot claim

**Can claim:** at matched disclosure on TOFU/LLaDA-8B, AFC retains substantially more capability
than MDU λ=3 — **+0.11 retain ASCR, 11/12 paired seeds** — while reaching **statistically
indistinguishable** distance-to-pristine.

**Cannot claim**, and `RESULTS.md` previously did: better TV (6/12, t=−0.53), better degeneracy
(4/12, worse mean), lower variance (AFC's TV sd is 0.115 vs MDU's 0.075, ~1.5× *higher*), or
relearning robustness (`FINDING_rtt_control_gate_passes` §3: AFC recovers up to 62.7% at the
strongest valid attack budget, inside the baseline's 51–68%).

So the defensible contribution is **one axis, with the forgetting axis a tie** — a narrower result
than the project set out to produce, and well short of the mandate's bar of beating the strongest
relevant baseline on a forgetting–utility trade-off. A tie on forgetting plus a win on utility is a
real but modest frontier improvement, not a dominated baseline.

## 4. ⚠ The slice this does not cover, and it is running

Matched **disclosure** is one slice. The complementary slice is matched **retention**: hold retain
fixed and ask who gets closer to pristine. MDU λ=3 sits at retain 0.406, far *below* AFC's 0.515, so
λ=3 is the wrong arm for that question — AFC may simply be at a different point on MDU's own
frontier rather than above it.

MDU **λ=1** is the nearest-retain MDU arm (published n=3: retain 0.557, TV 0.423). `run_matched_retain.sh`
is rebuilding it at **n=12** under the recorded settings.

- **AFC's 0.298 beats MDU λ=1's TV at comparable retain** → AFC is genuinely above MDU's frontier,
  and §3's claim strengthens into a frontier result;
- **MDU λ=1 lands near 0.30 too** → AFC is *on* MDU's frontier, not above it, and this project has
  **no advantage on either slice**. That would be the honest conclusion and it is pre-committed here
  before the data exists.

Until that lands, §3's claim is scoped to matched disclosure and must not be stated as a frontier
result.

## 5. Provenance and scope

All 24 arms are either published originals or rebuilds under explicitly recorded settings, validated
by three independent bridges and five exact seed reproductions
(`FINDING_afc_tv_advantage_does_not_survive.md` §1, §5a). Seed 11's MDU arm is `mdu_l3_s11_n12b`, a
clean single rebuild: the first attempt was contaminated by a label collision between two of my own
concurrent scripts and its rows are **quarantined, not deleted**, under
`outputs/CONTAMINATED_label_collision_s11/`.

One dataset, one backbone (rev `08b83a6feb34df1a6011b80c3c00c7563e963b07`), one evaluator, one
disclosure-matching point, 12 seeds. The t-values are paired-sample and reported without multiplicity
adjustment across four correlated metrics; with one pre-committed primary (retain) that is
defensible, but a reader treating all four as independent tests should discount accordingly. Every
historical checkpoint behind the n≤8 originals has been deleted from disk, so those arms are
re-derivable but not weight-verifiable.
