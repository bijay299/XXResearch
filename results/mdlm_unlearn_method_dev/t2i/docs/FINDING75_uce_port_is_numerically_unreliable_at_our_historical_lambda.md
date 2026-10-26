# FINDING 75 — Our UCE port is **not numerically equivalent** to the official implementation, and at our historical **λ = 0.1 the discrepancy is 16%** of the edited weight norm. The formula is right; the regime is not.

**Date:** 2026-10-26 · **Experiment:** R4 (`UPSTREAM_FIDELITY_MANIFEST.md` §6)
**Official reference:** `unified-concept-editing@7c724d9d7d19190e9bd33fe62fb656f1df87cd8e`,
`trainscripts/uce_sd_erase.py`
**Evidence:** `outputs/r4_uce_official_vs_port.json`, `outputs/r4_uce_lamb{0.5,0.1}.json`,
`outputs/r4_uce_erasescale2.json`, `logs/r4_uce.log`

---

## 1. The measurement

Both closed forms were evaluated on **identical inputs** — same pinned SD-1.4, same conditioning
embeddings, same `W_old`, same 160 `attn2` K/V projections — and the edited matrices compared:

| config | max relative error | mean | equivalent? |
|---|---:|---:|---|
| λ = **0.5** (official default), `erase_scale` 1.0 | **5.36e-03** | 3.09e-03 | **No** |
| λ = **0.1** (**our** historical default), `erase_scale` 1.0 | **1.59e-01** | 9.48e-02 | **No** |
| λ = 0.5, `erase_scale` **2.0** (divergence probe) | 5.40e-02 | 4.18e-02 | No |

**At the λ every historical UCE arm of ours used, the two implementations disagree by ~16% of the
edited weight norm.** My pre-recorded expectation — that the two would be equivalent at
`erase_scale = 1` since our port omits that coefficient — **was wrong**.

## 2. The cause is **conditioning**, not a formula error

Our port's algebra was verified previously (`ERRATUM_uce_lambda_scale.md` §1) and is confirmed again
here: the two expressions are term-for-term identical at `erase_scale = 1`. They differ only in
**summation order** — the official writes `mat1 = λW + erase + preserve`, ours writes
`num = erase + preserve + λW`.

Floating-point addition is not associative, so that reordering perturbs the matrices at rounding
level. A controlled reproduction isolates the amplification:

| λ | cond(den) | `‖den_off − den_our‖/‖den‖` | **⇒ `‖W_off − W_our‖/‖W‖`** |
|---:|---:|---:|---:|
| 0.5 | 4.13e+03 | **0.0** | 7.39e-05 |
| 0.1 | 2.07e+04 | **3.21e-08** | **1.68e-03** |
| 1000 | 3.06e+00 | 4.52e-08 | 1.53e-07 |

**The denominators agree to float-rounding precision; the inverses do not.** The error in `W` tracks
the condition number, and vanishes at λ = 1000 where the system is well conditioned (cond ≈ 3).

On the **real** system it is far worse than this synthetic bound. From the eigenvalues already
measured in `ERRATUM_uce_lambda_scale.md` (`cᵀc` max eigenvalue **85,850**):

> cond(`cᵀc` + λI) ≈ (85 850 + λ)/λ = **858 510 at λ = 0.1**, and 171 701 at λ = 0.5.

float32 carries ~7 decimal digits; a condition number of 8.6 × 10⁵ consumes ~6 of them. **One digit
of accuracy survives** — which is exactly the 16% discrepancy measured.

## 3. What this means for our existing UCE results

- **Every historical UCE arm of ours used λ = 0.1** — the regime where the edited weights are
  reproducible to about one significant figure. Those arms are **not wrong**, but they are **not
  reproducible across implementations**, and a different-but-equivalent transcription of the same
  equation lands ~16% away in weight space.
- This is the **second** consequence of the same root cause. `ERRATUM_uce_lambda_scale.md` found λ
  **inert** below ~1e3 because it is negligible against eigenvalues up to 85,850. The same
  mismatch makes the inverse **ill-conditioned**. One diagnosis, two symptoms: at λ ≪ the spectrum,
  the regulariser neither regularises nor conditions.
- **`uce_sd15_church_lam1000`** (target 0.2000 / retain8 0.9500, reversed 3/3 — `FINDING64`) sits at
  λ = 1000, where cond ≈ 86.9 and the implementations agree to ~1e-7. **It is the one UCE arm whose
  weights are numerically trustworthy**, and it happens to be the arm the λ-bracketing produced.
- **The official default is λ = 0.5, ours was 0.1.** "UCE at defaults" has meant different edits in
  the two codebases throughout this project.

## 4. What this does **not** show

- **Not a bug in our port.** The algebra matches; the transcription is faithful at `erase_scale = 1`.
- **Not a refutation of UCE.** A method evaluated outside its well-conditioned regime says nothing
  about the method. If anything this is a caution about *our* parameter choice.
- **Not a robustness result.** R4 compares weights only — no attack, no generation. Whether the 16%
  weight difference changes generated images or recovery is **unmeasured**.
- **`erase_scale` remains untested as a mechanism.** The 2.0 probe shows divergence (5.4e-2) but at
  λ = 0.5 that is entangled with the same conditioning effect, so it does not isolate the missing
  coefficient. Our port omits `erase_scale`; at the official default of 1.0 that omission is
  harmless *algebraically*, which is all that is established.

## 5. Status of the method label

UCE remains a **project reimplementation**, not an official reproduction. R4 was a numerical
comparison, not a run of the authors' script end-to-end. Running `uce_sd_erase.py` itself, and
comparing *generations* rather than weights, is the next step and has not been done.
