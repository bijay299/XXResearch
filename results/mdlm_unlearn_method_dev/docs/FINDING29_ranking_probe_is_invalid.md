# Finding 29 — the closed-pool ranking probe is invalid: it reads a perfect-recall model at 0.67 and mis-orders two models

**Date:** 2026-10-22 · **Backbone:** H18 124M MDLM
**Status:** a negative about **my own instrument**, which has been the knowledge axis of this
project since `FINDING21`. It forces an audit of `FINDING28` (`PREREG_H37`) and reframes a
`FINDING21` claim.

---

## 1. How it was caught

An unrelated arm (H36 pool-uniform anchoring) reported **retain 0.9000** on the ranking probe —
above `theta_ref`'s 0.6725 *and above the exact retrain oracle's 0.8875*. A retain number better
than the oracle's is not a result; it is a symptom. The cross-check was whether the gain appears
behaviourally:

| arm | ranking probe (retain) | behavioural retain recall (infill six-cell) |
|---|---|---|
| `theta_ref` | 0.6725 | 0.2725 |
| exact oracle | 0.8875 | 0.3625 |
| H36 PUA λ=1 | **0.9000** | **0.2575** |

The ranking probe puts PUA **above the oracle**; generation puts it **below `theta_ref`**. The gain
does not exist.

## 2. What the instrument is actually doing

Direct measurement — mask the value span in place, keep true left and right context, read the
marginal probability and exact-span argmax accuracy of the true value tokens:

| model | ranking probe, retain | marginal p(true), retain | **exact-span acc, retain** |
|---|---|---|---|
| `theta_ref` | **0.6725** | 0.9977 | **1.0000** |
| exact oracle | 0.8875 | 0.9992 | **1.0000** |
| PUA λ=1 | **0.9000** | 0.9737 | **1.0000** |

> **`theta_ref` recalls every retain fact exactly — 1.0000 — and the ranking probe reads it as
> 0.6725.**

Two distinct failures, and the second is the serious one:

1. **Compression.** A model with perfect recall scores 0.67. The probe's range is not the range of
   the quantity.
2. **Mis-ordering.** True retain recall goes `oracle (0.9992) > theta_ref (0.9977) > PUA (0.9737)`.
   The probe reports `PUA (0.9000) > oracle (0.8875) > theta_ref (0.6725)` — it **inverts** the
   ordering of `theta_ref` and PUA. A compressed-but-monotone probe would still rank methods
   correctly. This one does not.

**Cause.** `rank_probe_small.py` scores each closed-pool candidate by summing log-probs over an
**EOS-padded length-32 sequence**. Padding positions contribute to every candidate's score, so the
measure is dominated by how a model treats EOS runs rather than by whether it knows the fact. Any
intervention that reshapes the distribution at or after the value span moves the score without
touching knowledge. PUA, which explicitly reshapes the value-position distribution, moves it a lot.

This is the **third** padding/tokenisation-boundary defect in this project's measurement code (after
the BPE leading-space mismatch, twice, and trailing-run masking at sequence end). The common
pattern: a measurement that looks like it reads the value actually reads the surrounding token
bookkeeping.

## 3. The replacement, and its endpoints

`marginal_probe_small.py`: value span masked in place, true context retained, no padding in the
measurement. It is anchored by two **measured** models rather than an analytic chance level:

| anchor | forget exact-span acc | role |
|---|---|---|
| `theta_ref` | **1.0000** | ceiling |
| exact retrain oracle | **0.3375** | floor — what "never saw these facts" looks like here |
| PUA λ=1 | 0.1225 | (below the oracle, i.e. over-forgetting) |

## 4. This reframes a FINDING21 claim

`FINDING21` qualified the backbone partly on the oracle being **"at chance (0.1100 vs 0.1135)"** on
forget facts. Under a context-rich measurement the same oracle sits at **0.3375** exact-span.

Both numbers are correct for their own probe, and the oracle remains the floor either way — so
`FINDING21`'s role for the oracle stands. But **"the oracle is at chance" is probe-specific
language and will not be reused without naming the probe.** With a closed answer domain and true
surrounding context, a model that never saw a fact can still often infer it; "chance" is a property
of the scoring protocol, not of the model.

## 5. What this does to FINDING28 — stated before the audit runs

`FINDING28` ("no static metric predicts attack robustness") uses this probe as its x-axis. The
threat is specific and statistical: **a noisy x-axis attenuates correlation toward zero**
(regression dilution). So ρ = −0.148 over 125 arms may measure the probe's error rather than a true
absence of relationship, and "18 pairs with identical probe readings differ 1.9×–37× in recovery"
may be "identical *noisy readings* of genuinely different knowledge."

What is **not** threatened: the recovery numbers themselves, the band separation, and the α=0
controls landing in the suppression band — none of those depend on the ranking probe. What is
threatened is precisely the claim that the knowledge axis fails to predict robustness.

Only 2 of the 125 checkpoints still exist, so this cannot be fixed by reanalysis. `PREREG_H37`
registers a **balanced 16-arm replication** with both probes run on every arm, with pre-declared
outcomes including retraction. Until it reports, `FINDING28` carries this finding as an open
caveat and its headline is **provisional**.

## 6. Scope

- One backbone, one corpus. The defect is in a probe written for this corpus, not a claim about
  closed-pool likelihood ranking as a general technique — though the mechanism (padding entering
  every candidate score) would apply to any implementation that scores fixed-length padded
  candidates, which is a common shape.
- The behavioural eval and the attack protocol are untouched by this finding.
- The marginal probe is **not** adversarial and not a robustness measure. `FINDING28`'s central
  point — that only an attack measures robustness — is about what no static probe can do, and a
  better static probe does not change that part of the argument.
