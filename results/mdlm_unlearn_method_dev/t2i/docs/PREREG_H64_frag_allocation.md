# Pre-registration H64 — the forget-critical × retain-sparing allocation, from a published principle

**Date:** 2026-10-23 · **Status:** pre-registered; arms **queued and unread**.
**Source of the idea:** arXiv **2608.25429** (*"Distance Is Not Enough: Forget-Retain Alignment Gap
Predicts LLM Relearning Robustness"*, Chen et al., KAIST/Tokyo), read in full from the PDF rather
than via summary.

---

## 1. What the paper says, verified directly

Its thesis, quoted: *"relearning robustness depends on update structure: robust unlearning should
affect forget-critical weights while sparing retain-critical ones."* Its method, **Forget-Retain
Pruning (FRP)**, implements both halves, and Table 15 (verified verbatim in the PDF):

| method | pre-ES | pre-utility | ΔES (recovery) |
|---|---|---|---|
| Retain (gold oracle) | 0.060 | 0.593 | +0.008 |
| RMU | 0.056 | 0.577 | **+0.644** |
| GradDiff | 0.082 | 0.442 | +0.430 |
| NPO | 0.096 | 0.402 | +0.300 |
| SP | 0.131 | 0.499 | +0.144 |
| **FRP** | 0.062 | 0.408 | **+0.059** |

An order-of-magnitude robustness gain over RMU at comparable forgetting depth.

**This same table is the anticipation of our `FINDING31`** — the static-metric/recovery inversion,
with RMU forgetting *deeper* than the oracle (0.056 vs 0.060) and recovering most. That downgrade
(`FINDING31` §8) is confirmed by reading the source, not weakened.

## 2. Why this is the obvious next arm, and why I missed it

My two allocation schemes each implemented **one half** of the principle:

| scheme | targets forget-critical | spares retain-critical | outcome |
|---|---|---|---|
| `retain_fisher` (all arms to date) | ✗ | ✓ | retain 0.867, **no robustness** |
| `forget_fisher` (`H51`) | ✓ | ✗ | retain **0.09**, model destroyed |
| **`frag` (this arm)** | **✓** | **✓** | untested |

`H51`'s failure is now diagnosable rather than just negative: concentrating damage on
forget-critical weights destroys the model *because nothing protects the retain-critical weights
that overlap them*. The paper's principle predicts the intersection behaves differently from either
half, and the authors explicitly invite the extension: *"FRP is instantiated as unstructured
pruning; the same principle may extend to structured pruning, low-rank editing, and other
parameter-efficient interventions."* Noise allocation inside perturb-then-distil is such an
intervention.

Implementation: per-parameter damage scale ∝ `normalise(F_forget) × (1 − normalise(F_retain))`,
renormalised to preserve mean damage so it is comparable to every previous arm at the same α.

## 3. Separating predictions, declared now

| outcome | reading |
|---|---|
| retain ≥ 0.85 **and** max-over-valid H ≤ 0.50 on 9 attacks | **H64 true.** First arm in either phase to reach the mandate's bar, and the first transfer of FRP's principle to a diffusion model and to noise allocation. |
| retain ≥ 0.85, H > 0.80 | the intersection preserves utility but buys nothing; the principle does not transfer to noise allocation |
| retain < 0.60 | inherits `forget_fisher`'s collapse — the retain-sparing factor is insufficient to protect the overlapping weights |
| H ≤ 0.50 but retain < 0.60 | same trade as every previous arm; no progress |

## 4. Kill rules

**K1.** Full 9-attack budget (5 seeds @ 1e-5, 3 @ 5e-5, 1 @ 1e-4) **from the start**, fixed before
any number is read. `H62` showed a sub-ceiling value from 3 seeds at one lr has been wrong **four
times** in this project; this arm does not get to produce a fifth.

**K2.** retain8 < 0.60 ⇒ not a candidate regardless of robustness (`FINDING35`'s band where
resistance only ever accompanied unusable models).

**K3.** A positive requires replication on a **second concept** before any claim.

## 5. Novelty position, stated before results

The *principle* is published (2608.25429) and so is a method implementing it (FRP, as unstructured
pruning on LLMs, with code at `github.com/Yi1-Chen/FRAG`). What is untested is the principle as a
**noise-allocation rule inside perturb-then-distil**, and in a **diffusion** model. So a positive
result is an **extension of a published principle to a new intervention and modality** — reportable
as such, never as a new mechanism. FRAG/FRP also becomes a **fifth** competing baseline for this
threat model, and unlike the other four it has released code — though it is an LLM method, so
porting it is not free.
