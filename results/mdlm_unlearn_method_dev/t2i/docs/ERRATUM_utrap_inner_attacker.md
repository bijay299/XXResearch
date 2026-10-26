# Erratum — UTRAP did **not** train against the evaluation attacker, so the sweep did not test the hypothesis it was designed to test

**Raised by:** external audit of snapshot `0b435427`, §5. **Verified independently** against both
implementations line by line — see §1.
**Affects:** `scripts/erase_utrap.py`, `PREREG_H84`, `FINDING63` §2 and §3a(a), and the two
cross-model UTRAP arms in `FINDING64` §4a.
**Does not affect:** the 24 UTRAP reversal measurements. See §4.

---

## 1. The mismatch, verified on seven axes

`erase_utrap.py`'s docstring (line 5) and `PREREG_H84` both say:

> *"A_K(theta) = K steps of the **ACTUAL** full-UNet relearning attack (the one attack_relearn.py
> runs, on 16 self-generated target images)"*

It is not. Reading both files at the audited snapshot:

| axis | UTRAP inner | evaluation attack | source |
|---|---|---|---|
| optimizer | **SGD**, reconstructed every outer step | **Adam**, persistent across all 300 steps | `erase_utrap.py:140` vs `attack_relearn.py:333` |
| gradient clipping | **none** | **global norm 1.0** | — vs `attack_relearn.py:360` |
| batch size | **2** | **1** | `erase_utrap.py:55` vs `attack_relearn.py:335` |
| horizon | **K = 4 / 8 / 16** | **300** | `erase_utrap.py:49` vs `attack_relearn.py` `--steps` |
| target images from | the **already-erased FRP teacher** (the teacher is loaded into `unet` at `:75`, then `pipe` generates at `:88`) | the **unmodified base model** | `erase_utrap.py:75,88` vs `attack_relearn.py:261` `build_pipe(..., None, ...)` |
| target prompts | `["a photo of a church"] * 16` — one repeated string, which is also **evaluation prompt H[0]** | the **8 `T` prompts**, disjoint from H | `erase_utrap.py:80` vs `attack_relearn.py:43-47` |
| optimizer state | discarded each outer step | persistent | `erase_utrap.py:140` (inside the loop) |

**All seven confirmed.** Matching the learning-rate *number* (1e-5) and the parameter scope
(859.5M) does not make SGD-without-momentum-rebuilt-every-step the same adversary as Adam with
persistent moments over 300 steps. Adam's per-coordinate preconditioning is most of what the
attacker is.

## 2. What this does to `FINDING63`'s conclusion

`PREREG_H84`'s hypothesis was that **one-step inner approximation** was what doomed our IMMA port
(`FINDING59` §0), and that **K unrolled steps of the real attacker** would fix it. `FINDING63` §3a(a)
reported the K sweep as falsifying that:

> *"doubling K from 8 to 16 left max-over-valid H at 1.000 … The core hypothesis of `PREREG_H84` —
> that the one-step inner approximation was what doomed IMMA — is falsified on its own terms.
> Depth was the thing I believed was missing, and it is not."*

**That inference does not hold.** What the sweep varied was the horizon of a *surrogate* that
differed from the evaluation attacker on six other axes simultaneously. Varying K of the wrong
attacker, with one defence build per K and a saturated endpoint, cannot falsify a claim about the
depth of the right one. `FINDING63` §2's *"both structural causes of the previous failures were
removed"* is likewise withdrawn — the scope cause was removed (full 859.5M), the attacker-fidelity
cause was not, and a third was introduced (the defence trained on an evaluation prompt).

Also noted and corrected: **K=4 was never attacked** in the reported sweep, so the λ=0.3 K-series
has two attacked points, not three.

## 3. The data problem is separate and compounds it

Two of the seven axes are about data, and they interact with `ERRATUM_split_leak.md`:

1. The inner attacker was trained on images generated **by the erased teacher asked for the concept
   it had just been erased of**. Those images were never content-checked. Whatever they contain,
   they are not the base-model target images the real attacker is given.
2. The single prompt used to generate them, `"a photo of a church"`, **is** evaluation prompt H[0].
   So UTRAP's defence-training data and its evaluation target prompt were the same string.

Neither manufactures the H = 1.000 results — those come from the attack side, which used the
correct `T` prompts and base-model data throughout. They do compromise the independence of
UTRAP's **pre-attack** suppression and operating-point selection, including the 0.8667 retain8
figure that cleared the mandate's floor.

## 4. What is preserved, and it is the substantive part

**All 24 UTRAP attack records stand unchanged:** 18 on SD-1.4 (λ×K sweep) and 6 cross-model
(SD-1.5, BK-SDM-small), every one gate-valid, every one reaching H = 1.000. Those are measurements
of the **evaluation** attacker against real checkpoints, and the evaluation attacker was never in
doubt.

So the empirical core of `FINDING63` survives in full: **a checkpoint with zero target detections
pre-attack and retain8 = 0.8667 was reversed to 15/15 by all six attacks in its grid, and the
attacks left the model at post-attack utility up to 0.979.** The cross-model arms extend that to
SD-1.5 at retain8 0.9583.

**What is withdrawn** is narrower and specific:

- that UTRAP was trained against the real attacker;
- that the inner process ran "the real attack at its real budget";
- that **unroll depth was ruled out** as the missing ingredient;
- that both structural causes of the prior failures had been removed.

## 5. The repair as built

**`scripts/attack_core.py`** — one implementation of the relearning update, used by both sides:

- `AttackSpec` carries every distinguishing knob and computes
  **`divergence_from_evaluation_attacker`** automatically against a frozen `EVAL_FULL_UNET`
  constant. Any defence that diverges now says so **in its own artifact**.
- **`horizon_role` has no default** and must be one of `EVALUATION_REFERENCE`,
  `SHORT_HORIZON_SURROGATE`, `MATCHED_HORIZON_SURROGATE`, `ABLATION_K0`. `EVALUATION_REFERENCE` is
  **refused** for any horizon ≠ 300, and `SHORT_HORIZON_SURROGATE` is refused for ≥ 300. The exact
  mislabelling this erratum documents is now unrepresentable.
- `AttackData` makes **`provenance` and `split_role` mandatory fields** — a defence cannot
  construct attacker data without recording which model generated it and which split it belongs to.
- `diffusion_mse` is the single definition of the relearning loss, so the loss itself cannot drift.

**`scripts/erase_utrap_v2.py`** consumes it, with defaults that **match** the evaluator — Adam,
batch 1, clip 1.0, persistent optimiser state, **base-model** target images — and a `DEF_TRAIN`
target pool disjoint from both the attacker's `T` and the evaluation prompts. It also logs, per the
audit's request:

- a **fixed-noise target probe immediately before and after every unroll**, so "the inner attacker
  actually adapts" is a measured number with its own interpretation note (a delta near zero means
  the unroll is doing nothing and no unroll-depth claim can be drawn from the run);
- per-step **update norms**, gradient norms, inner loss trace and the full optimizer configuration.

The snapshot's configuration remains reachable for reproduction
(`--inner-optimizer sgd --inner-batch 2 --inner-clip 0 --inner-persistent 0
--target-source teacher --target-prompts eval_leak_reproduce`) and is flagged as such, with a
printed warning, rather than deleted.

**Ten deterministic tests, all passing**, including: the attacker must lower a fixed-noise probe on
a tractable 6-dimensional model; K=0 must move nothing; Adam and SGD endpoints must be separated at
the same lr; the snapshot's spec must report every divergent axis; and the outer gradient must equal
`∇L_retain − λ∇L_target(θ_K)` with θ restored after the unroll.

**Status: the code is corrected and tested; no model has been built with it.** The bounded
comparison that would actually test the unroll-depth hypothesis — **K ∈ {0, 1, 8, 32} at matched
total compute, with the faithful attacker** — is preregistered in `PREREG_H90_utrap_v2.md` and is
experiment **E4** (~14 GPU-h, unblocked). A fully differentiated unroll remains untested and is not
claimed.
