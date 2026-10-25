# Four-point frontier, n=12 each: **AFC dominates MDU across its whole λ range.** The pre-registered threat (λ=4) regressed to the mean, so the provisional frontier claim now stands.

**Date:** 2026-10-25 · **48 arms, 12 paired seeds × 4 operating points**, all under explicitly
recorded settings
**Resolves:** the ⚠ provisional caveat in `FINDING_matched_retention_slice` §5
**Evidence:** `outputs/arm_mdu_l4_s{0..11}_fr.json` plus the AFC λ=1, MDU λ=1 and MDU λ=3 n=12 columns

---

## 1. The threat, and how it resolved

`FINDING_matched_retention_slice` §5 published the frontier claim as **provisional** because a
**single-seed** measurement contradicted it: `cv_mdu_l4` recorded **TV 0.2325 at retain 0.4565** —
better forgetting than AFC's 0.2983 at modestly lower retention. The pre-commitment was explicit:

> *"λ=4 reproduces ≈0.23 TV at ≈0.46 retain → **no frontier claim can be made for AFC**; λ=4
> regresses toward 0.30+ → the single-seed point was a lucky seed."*

**At n=12, MDU λ=4 is TV 0.3104 ± 0.0892 at retain 0.3869 ± 0.1002.** The second branch fired. Seed
0 reproduced `cv_mdu_l4` exactly (0.2325 / 0.4565), so that measurement was sound — it was simply
the most favourable of twelve, and its retention was unrepresentative by 0.07.

## 2. The frontier

| arm | TV→pristine | retain ASCR | DISCLOSE | DEGENERATE |
|---|---|---|---|---|
| **AFC λ=1** | **0.2983 ± 0.1153** | **0.5153 ± 0.0634** | 0.1549 ± 0.0892 | 0.1493 ± 0.1292 |
| MDU λ=1 | 0.4400 ± 0.0525 | 0.5235 ± 0.0940 | 0.3306 ± 0.1116 | **0.0951 ± 0.0944** |
| MDU λ=3 | 0.3223 ± 0.0751 | 0.4057 ± 0.1061 | 0.1889 ± 0.0884 | 0.1271 ± 0.0963 |
| MDU λ=4 | 0.3104 ± 0.0892 | 0.3869 ± 0.1002 | 0.1451 ± 0.0789 | 0.1590 ± 0.1221 |

**AFC has the lowest TV of all four points and, within noise, the highest retention.** No setting of
MDU's λ reaches that combination:

- **λ=1** matches AFC's retention (0.524 vs 0.515, t=−0.32) but forgets far worse — TV +0.142,
  t=−4.06, AFC better on 11/12 seeds.
- **λ=3** and **λ=4** match AFC's forgetting (ties at t=−0.53 and t=−0.28) but retain much less —
  **+0.110** (t=+3.15, 11/12) and **+0.128** (t=+3.92, 11/12) respectively.

Paired, AFC against λ=4 specifically:

| metric | paired Δ | se | t | AFC better on |
|---|---|---|---|---|
| TV | −0.0121 | 0.0432 | −0.28 | 6/12 (tie) |
| **retain** | **+0.1284** | 0.0327 | **+3.92** | **11/12** |
| DISCLOSE | +0.0097 | 0.0160 | +0.61 | 4/12 |
| DEGENERATE | −0.0097 | 0.0478 | −0.20 | 6/12 |

**Turning MDU's λ up buys forgetting with retention and never escapes the trade.** AFC sits off that
curve. That is a genuine forgetting–utility improvement over the baseline across its parameter range
— the thing the project set out to produce.

## 3. ⚠ The gap in the frontier, and why it is unmeasured

**MDU λ=2 was not run.** It was queued and then **cancelled at the user's explicit instruction** to
refocus on the t2i line — not because it was uninformative. So the frontier has a hole between λ=1
(0.440 / 0.524) and λ=3 (0.322 / 0.406).

Linear interpolation puts λ=2 near **TV 0.38 / retain 0.46**, which AFC's (0.298 / 0.515) would
still dominate on both axes — but that is an interpolation, not a measurement, and the λ=4 episode
is a direct demonstration that a single unmeasured point can look different from expectation. The
honest statement: **AFC dominates MDU at every λ actually measured (1, 3, 4), and λ=2 is a declared
gap.** Closing it is ~3 GPU-hours whenever it is wanted.

## 4. What this does **not** establish

- **Not robustness, and this is the big one.** AFC recovers up to **62.7%** of erased facts under
  the strongest valid relearning attack, inside the baseline's 51–68%
  (`FINDING_rtt_control_gate_passes` §3). A better trade-off between forgetting and utility is
  **not** a better defence against an adversary who fine-tunes the model afterwards. Any write-up
  must carry both.
- **Not lower variance.** AFC's TV sd is 0.115 against λ=1's 0.053 and λ=4's 0.089 — the highest of
  the four. Seed 3 remains a catastrophic-failure mode (TV 0.587, DEGENERATE 0.508) that no MDU
  setting exhibits. The original n=3 stability claim stays refuted.
- **Not better degeneracy.** AFC (0.149) beats only λ=4 (0.159) and loses to λ=1 (0.095) and λ=3
  (0.127).
- **Not generality.** One dataset (TOFU), one backbone (LLaDA-8B-Instruct rev
  `08b83a6feb34df1a6011b80c3c00c7563e963b07`), one evaluator, one AFC setting, and AFC's own λ sweep
  is **not** run at n=12 — only λ=1 is. A fair frontier-vs-frontier comparison would require it, and
  it is not claimed here.

## 5. The arc, which is the honest context for this result

| claim | fate |
|---|---|
| n=3: "37% better TV, 9–16× more stable" | **both refuted** |
| n=5: "better on average, 0.314 vs 0.424" | **baseline was λ=1, not λ=3** |
| n=12 matched disclosure: "better TV" | **refuted — tie, 6/12** |
| n=12 matched disclosure: "+0.11 retention" | **survives**, 11/12, t=+3.15 |
| n=12 matched retention: "−0.142 TV" | **survives**, 11/12, t=−4.06 |
| **four-point frontier: AFC dominates λ∈{1,3,4}** | **survives every control run so far** |

Five claims died; two survived and compose into this one. The surviving claims are the ones that
were paired, seeded to n=12, built on arms whose settings are recorded, and tested against a
pre-registered threat. That is the only reason to believe them more than the n=3 numbers.

## 6. Provenance

All 48 arms are published originals or rebuilds under explicitly recorded settings, validated by
three independent bridges and nine exact seed reproductions — including the λ=1 rebuilds that proved
the original λ confound (`FINDING_lambda_provenance_confound` §2) and λ=4 seed 0 reproducing
`cv_mdu_l4`. Seed 11's MDU λ=3 arm is `mdu_l3_s11_n12b`, a clean rebuild after a label collision
whose contaminated rows are **quarantined, not deleted**. t-values are paired-sample without
multiplicity adjustment across four correlated metrics; the pre-committed primary for this test was
λ=4's TV.
