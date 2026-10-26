# Erratum — UCE's `--lam` is **five orders of magnitude** below its useful range, and my reading of the sweep was wrong

**Date:** 2026-10-25 · **Found:** while trying to recover a usable UCE arm on SD-1.5 after the
default λ fell below the K1 utility floor.
**Affects:** the λ *interpretation* in `FINDING64`'s UCE addendum and the default in
`scripts/erase_uce.py`. **No published measurement changes** — see §3.

---

## 1. What happened

UCE solves, per cross-attention K/V matrix:

```
W_new = ( Σ_E v* cᵀ + s·(W c_pᵀ) c_p + λ·W ) ( c_eᵀc_e + s·(c_pᵀc_p) + λ·I )⁻¹
```

As λ → ∞ this tends to `W_old`, so λ is a **regularisation toward the original weights** and the
docstring says so correctly. The formula is **right**.

The SD-1.5 arm fell to retain8 0.5333 at the default **λ = 0.1**, below K1, so I swept λ upward to
preserve more utility. The result looked like λ making things *worse*:

| λ | retain8 | target |
|---|---|---|
| 0.1 | 0.5333 | 0.0000 |
| 0.5 | **0.4667** | 0.0000 |
| 1.0 | **0.4167** | 0.0000 |

**I reported that as "higher λ made it worse." That reading was wrong.**

## 2. The actual cause: λ is negligible at every value tried

Measured on SD-1.5's text encoder, over the two prompts UCE conditions on (154 token embeddings,
768-dim):

| quantity | value |
|---|---|
| mean per-token ‖c‖ | **28.45** |
| trace(cᵀc) | **125,092** |
| **max eigenvalue of cᵀc** | **85,850** |
| mean eigenvalue | 162.9 |

**λ must be comparable to the eigenvalues of `cᵀc` to have any effect.** At λ = 1.0 against a
largest eigenvalue of 85,850, the regularisation term is ~1e-5 of the matrix it is added to — it
does nothing. The usable range starts around **λ ≈ 8,600** (max eig / 10).

So the three arms are **not a λ sweep at all**: they are three samples of effectively the same
closed-form solution, and the 0.4167–0.5333 spread is run-to-run and evaluation noise at n = 1 each.
That `target` stayed pinned at exactly 0.0000 across all three is the corroborating tell — if λ were
biting, target would have risen toward the base model's 1.000.

## 3. Why no published measurement changes

- The **SD-1.4 UCE arm** was built and attacked long before this, at the same default λ = 0.1. Its
  reported numbers are what that configuration produces, and `FINDING64`'s statement that it is
  reversed to H = 1.000 is unaffected — it rested on gate-validated attacks, not on λ.
- The **UCE-on-BK-SDM** arm (target 0.0000 / retain8 **0.9333**, reversed 3/3) is likewise unchanged:
  it cleared K1 comfortably at the default λ, so the closed-form-class conclusion in `FINDING64`
  stands on its own evidence.
- The three SD-1.5 arms were **K1-failed and never attacked**, so nothing was scored from them. What
  changes is only the *explanation* attached to them.

## 4. What is corrected, and what is being run

- `FINDING64`'s UCE addendum is amended: the SD-1.5 failure is a **scale error in my sweep**, not
  evidence about λ's direction, and certainly not evidence about UCE.
- `scripts/erase_uce.py`'s `--lam` help text now states the required magnitude and warns that values
  of order 1 are inert, so this cannot be repeated by someone reading the flag at face value.
- A **properly scaled sweep (λ = 1e3, 1e4, 5e4)** is running to find out whether a usable SD-1.5 UCE
  arm exists at all. If one does, it gets the frozen grid; if none clears K1, the honest report is
  that **our UCE port has no usable operating point on SD-1.5**, which is a statement about the port
  and the checkpoint, not about the method.

## 5. The general lesson, recorded because it has now bitten twice

This is the second hyperparameter whose *useful range* I assumed instead of measuring — the first
was `α` in the IMMA port, where the pre-declared value destroyed the model (`FINDING59` §0). Both
times the fix was one cheap measurement of the quantity the hyperparameter is compared against.
**Measure the scale before sweeping the knob.**

---

## 6. Follow-up: the properly scaled sweep confirmed the diagnosis, and found the opposite failure

λ = 10000 — inside the range §2 predicted would "start to bind" — binds **completely**:

| λ | target (0 = erased) | retain8 | what it is |
|---|---|---|---|
| 0.1 / 0.5 / 1.0 | 0.0000 | 0.5333 / 0.4667 / 0.4167 | λ **inert**; over-erasing closed-form solution, all three the same arm |
| **10000** | **1.0000** | **1.0000** | λ **dominant**; `W_new → W_old` — **the base model, no erasure at all** |

This is the λ → ∞ limit of the UCE formula arriving exactly as the algebra says it must, and it
**confirms §2's diagnosis by the only test that could**: the knob does have an effect, it was simply
being turned inside a dead zone five orders of magnitude below where it acts.

**The λ=10000 arm was therefore not attacked, and its attack grid was killed after one run.** It
passed the K1 utility check (retain8 1.0000) but there is nothing in it to reverse: target is already
at the ceiling before any adversary exists. **Attacking it would have produced H = 1.000 and that
number would have meant nothing** — it would have measured the base model's ability to draw a church,
not a defence's failure to stop it. Recording it as a thirteenth-defence reversal would have been
exactly the kind of manufactured result this project's constraints forbid. It is logged in
`logs/uce_cm.log` as `NO ERASURE ... Not a robustness datum.`

**Net effect: the usable window is now bracketed from both sides by measurement** — over-erasure
below λ ≈ 1, no-op by λ = 10000 — rather than estimated from `cᵀc`'s spectrum. A geometric sweep of
the interior (λ ∈ {10, 30, 100, 300, 1000}, `run_uce_sd15_bracket.sh`) is running build+eval only to
find whether any interior λ gives **target ≈ 0 at retain8 ≥ 0.60**. Only an arm that clears that
goes to the frozen grid.

**If no interior λ qualifies, the finding is that our UCE port has no usable operating point on
SD-1.5** — a statement about this port on this checkpoint. It would leave the closed-form class with
its one measured cross-model point (BK-SDM-small: target 0.0000 / retain8 0.9333, reversed 3/3), and
that is what `FINDING64` will say, rather than inferring SD-1.5's outcome from BK-SDM's.
