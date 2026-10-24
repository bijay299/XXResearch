# Text-to-image concept erasure — phase report, revision 2

> ⚠ **SUPERSEDED by `T2I_REPORT_r3.md` (2026-10-24).** Preserved as published; r3 adds the
> four-adversary matrix, the Textual-Inversion result, the DreamBooth-prior evaluation
> pitfall, and a verdict anchored on the phase's best arm rather than weaker ones.

**Date:** 2026-10-23 · **Identifier:** `T2I_REPORT_r2` · **Supersedes:** `T2I_REPORT.md` (2026-10-22)
**Model:** Stable Diffusion v1.4, pinned `133a221b8aa7292a167afc5127cb63fb5005638b`
**Mandate:** an unlearning algorithm that beats the strongest existing methods on a clearly defined
forgetting–utility trade-off, with real-model results.

`T2I_REPORT.md` is **preserved unaltered** with a pointer banner; this document is the current
state. Every correction it embodies is itemised in `ERRATUM_r2_claim_corrections.md` (C1–C19).

---

## 1. The verdict

**The mandate's bar is not met.** No arm reaches retain ≥ 0.85 with max-over-valid post-attack
H ≤ 0.40. Every erasure tested — including the published robust SOTA — is reversed to the base
model's ceiling by a 300-step fine-tuning adversary with weight access, on **all five** concepts.

**And the headline is ANTICIPATED.** Meta-Unlearning (2410.12777) reports T2I erasure reversed by
malicious fine-tuning at 50–300 steps on a prompt-disjoint split and names AdvUnlearn in that claim;
`Erased or Dormant?` (2505.16174) reports it for UCE and ESD. Our measurement is a **confirmation**
with one gap filled — Meta-Unlearning asserts the AdvUnlearn case in prose while measuring RECE.

**The competitive-baseline gap that bounds everything below.** Five existing defences target exactly
this threat model: Meta-Unlearning (2410.12777), IMMA (2311.18815), GIFT (2507.13598), From Dormant
to Deleted (2505.22310), FRAG/FRP (2608.25429). **None was run before this revision, and none has
usable released code** — `ERRATUM_r2` C19 corrects my earlier claim that FRAG/FRP did; its
repository holds only a README. The first port of one (FRP) is pre-registered in `PREREG_H66` and
running as this is written.

| arm | target (erased) | retain8 | **max-over-valid H after attack** | valid/total |
|---|---|---|---|---|
| base SD v1.4 | 1.000 | 1.000 | — | — |
| **AdvUnlearn church** (official, robust SOTA) | 0.400 | **0.992** | **1.000** | 6/6 |
| **AdvUnlearn parachute** (official) | 0.133 | 1.000 | **1.000** | 6/6 |
| ESD (our re-impl) | 0.467 | 0.992 | **1.000** | 3/3 |
| UCE (our re-impl, tuned) | 0.067 | 0.850 | **1.000** | 3/3 |
| UCE at matched utility | 0.000 | 0.350 | **1.000** | 4/4 |
| UNDO α=0.5 (damage+distil) | 0.000 | 0.317 | 0.467 | 3/10 |
| UNDO α=0.7 | 0.000 | 0.000 | no valid attack | 0/3 |
| *FRP port (`H66`)* | *in flight* | — | — | — |
| *frag allocation (`H64`, pools v2)* | *in flight* | — | — | — |

Attacker cost: **300 gradient steps on 16 images generated from the public base model.** No scraped
data, no privileged access beyond the weights.

## 2. What is solid

**(i) Every usable erasure is fully reversible, on all five concepts.** Under the frozen symmetric
six-point grid, max-over-valid H = 1.000 for church, parachute, garbage_truck, tench and van_gogh.
`H62` overturned our own earlier "4 of 5" claim by taking tench from 0.733 to 1.000.

**(ii) ⚠ But 1.000 is a MAX-over-attacks statistic.** Re-derived from raw records — pure official
arms, 300 steps, post-dating the critical evaluation fix, n = 25: **25/25 gate-passing, yet only
8/25 individually reach 1.000**, median single-attack H **0.800**, range 0.333–1.000. The threat
model is unchanged (an adversary who may try six configurations gets full recovery on every
concept), but the published phrasing invited the reading that *any* 300-step attack fully reverses
the erasure, and that is false (`ERRATUM_r2` C1).

**(iii) ✅ NEW — full reversal happens at INTACT utility.** This was the one benign explanation the
record could never exclude: perhaps `H = 1.000` just meant a wrecked model. `H65` measured it, 12/12
points, both arms:

| attack lr | max-over-valid H | **utility after** | Δ prompt adherence |
|---|---|---|---|
| **1e-5** | **1.000** both arms | **0.875 – 1.000** | −0.0055 … +0.0000 |
| 5e-5 | 1.000 both arms | 0.604 – 0.750 | −0.0684 … −0.0315 |
| 1e-4 | 1.000 church / 0.667 parachute | **0.021 – 0.271** | −0.1252 … −0.0973 |

Cleanest points: **church sd6 reaches H = 1.000 at utility 0.9375**, **parachute sd1 at 0.9792**.
And the **efficient attack is the gentlest one** — H is already maximal at 1e-5, while parachute at
1e-4 recovers *less* (0.667) from a model it destroyed. An adversary has no reason to use the
damaging settings, so the threat model is governed by the 1e-5 row. The lr 1e-4 points double as an
unplanned positive control: the utility probe reads 0.0208 there against 1.0000 on an intact model.

**(iv) Prompt-space adversarial training does not transfer to weight space.** AdvUnlearn is hardened
over a neighbourhood of *text embeddings* and is undone by 300 gradient steps on the *UNet* — the
weaker of our two adversaries; its text encoder never had to be touched. **This is a scope statement
about a different threat model, not a refutation of their paper**, which claims prompt-space
robustness only (`ERRATUM_r2` C15).

**(v) The cross-attention edit surface cannot cleanly separate concepts.** Content tokens
discriminate between concepts (cos 0.13–0.21) while the trailing positions that dominate
cross-attention are **half-collinear** (cos 0.52–0.59). A measured reason erasure trades off against
utility, not an argued one.

**(vi) Damage's collateral destruction is as reversible as its intended erasure.** Damage at α=0.3
knocked an untargeted concept (tench) from 1.000 to 0.133; the attacker restored it to 1.000 in 300
steps. Nothing the damage destroyed stays destroyed.

**(vii) Concept-targeted damage is strictly worse than uniform damage.** `H51`: forget-Fisher
allocation left retain at **0.0917** against retain-Fisher's 0.867 at the same mean damage — ~9×
worse. Mechanism claim (inseparability) stays **provisional**: a matched random control now exists
and matches the realised perturbation norm to five digits, but no direct overlap measurement between
forget- and retain-critical parameter sets was ever computed (`ERRATUM_r2` C18).

## 3. ⚠ Our own measurements biased in our favour, and the bias is now removed

`verify_splits.py` computes split separation by importing the pools out of the code. Result:

| check | v1 (as published) | v2 (repaired) |
|---|---|---|
| `ATK_TRAIN × EVAL_TARGET` | CLEAN (0) | CLEAN |
| `DEF_TRAIN × EVAL_TARGET` | **CONTAMINATED (8)** | **CLEAN** |
| `DEF_TRAIN × EVAL_UTILITY` | **CONTAMINATED (39)** | **CLEAN** |

Every leak came from `undo_t2i.py`: its forget pool contained 1–2 of the 3 prompts each concept is
*scored* on, and its retain pool contained 7–8 of the 8 retain-concept evaluation prompts verbatim.
**So damage+distil utility was in-sample while UCE/ESD/AdvUnlearn's was not** — the comparison was
biased toward our own arms. It does not rescue any conclusion (the arms were reversed even with
inflated utility), but **any cross-family utility ranking placing a damage+distil arm above a
baseline is withdrawn** (`FINDING38`, `ERRATUM_r2` C10).

Pools are now recipe **v2**: same concepts, surface forms disjoint from the evaluator. v1 arms are
not relabelled; v2 arms carry `recipe: "pools_v2"` and are not comparable to them.

## 4. ⚠ The deepest limitation, stated plainly for the first time

**Without a never-trained oracle we cannot distinguish reactivation from relearning.** The attacker
fine-tunes on images *of the target concept*. When the concept returns, two explanations fit equally
well: the erasure merely suppressed it and the attack reactivated residual memory, or the erasure
genuinely removed it and the attack taught it again. **This phase cannot tell them apart**, because
no exact-retrain oracle exists at SD scale — the single biggest asset the MDLM phase had and this
one lacks, pre-declared as such in `PREREG_H38` §3.

For the **threat model** the distinction does not matter: an adversary with weight access and 300
steps obtains a model that generates the concept. For the **mechanism** claim it is decisive, so
wherever this project's text says or implies *"the knowledge was never removed"* — including
`PREREG_H40`'s language about features "still present" — that claim is **not established**
(`ERRATUM_r2` C17).

## 5. Our own errors, recorded

| error | how it was caught |
|---|---|
| post-attack eval loaded the BASE text encoder → **136 of 234 runs VOID** | a number too good: `H58` claimed 2 steps restored a concept |
| UCE re-impl #1 did not erase (5.7 % weight change, church 1.000) | checked erasure before trusting the arm |
| adversarial erasure re-impl, 3 attempts, never erased | verified against working ESD; resolved with official weights |
| adversarial inner loop ascended the wrong sign | a non-degenerate test showing the loss *fell* |
| `--attack-te` was a **no-op** — embeddings cached under `no_grad` | gradient-norm evidence, exactly 0 |
| Fisher mask renormalise-then-clamp silently lowered mean damage to 0.3046 at α=0.5 | per-tensor mask audit |
| **a shadowing bug I introduced in my own repair** (`g` = float over a `torch.Generator`) killed `H64` | real-model integration check — unit tests had passed |
| `FINDING35` headline omitted the one arm contradicting it; both Spearmans wrong, one sign-flipped | adversarial audit against raw JSONs |
| **claimed FRAG/FRP "has released code"** without opening the repo | opened the repo (`ERRATUM_r2` C19) |
| four separate false signals from n-too-small | each overturned by a larger budget before publication |

The validity gate has three documented failure modes, all found by using it, plus a fourth the audit
named: it is **blind to a misassembled evaluation pipeline** — the failure that produced the 136 void
runs.

## 6. Limitations

- One model; five concepts; **one attack family** (relearning). Prompt-space attacks
  (UnlearnDiffAtk, Ring-A-Bell, concept inversion) were not run — the literature reports those.
- UCE and ESD are **our re-implementations**, labelled throughout; our UCE preserves utility worse
  than published UCE. Only AdvUnlearn is an official checkpoint. FRP and FRAG are **our ports**.
- The 12-way closed-set CLIP metric is coarse and top-saturated; H is 15 samples quantised to 1/15.
- **FID is not reported.** At n = 48 the Inception covariance is rank-deficient; the 5 000-image
  protocol is written and costed (≈ 4.2 GPU-h) in `FID_PROTOCOL.md` with a declared trigger.
- The NSFW track — the field's main safety application — was **deliberately excluded**.
- **The CLIP classifier revision was never pinned**, though it produces every metric here. Results
  are internally self-consistent from the local cache; not third-party reproducible.
- **28 of 41 stochastic call sites use the global RNG stream.** Attacks are repeatable from their
  seed but not robust to inserting or removing an earlier stochastic call. Arm construction was
  nondeterministic until `08742c1`; any rebuild is a **new arm**.

## 7. What is running, and what would change the verdict

| study | question | status |
|---|---|---|
| `H64` (`PREREG_H64`) | does forget-critical × retain-sparing noise allocation — the intersection neither of our schemes tried — buy robustness at usable utility? | **running**, pools v2, 6 frozen + 3 K1 points per arm |
| `H66` (`PREREG_H66`) | does **FRP**, a published method built for relearning robustness, transfer to T2I? | **running**, ρ ladder then frozen grid; operating point by a rule fixed in advance |
| `H67` (`PREREG_H67`) | does **FRAG**, their training-free robustness *predictor*, transfer? Tested against 34 existing arms, weights-only, CPU | **running**; declared inconclusive-by-default at n ≈ 10–20 |

**If FRP meets the bar, the correct report is that a published method beats everything we built**,
and `PREREG_H40` is refuted by someone else's method. That is the outcome I consider most valuable
to the mandate short of our own success, and it will be reported as theirs.

My recorded expectation, before results: **FRP is fully reversed like every other arm**, since its
robustness evidence is against a one-epoch LLM attack — a weaker adversary than 300 steps on 16
images. `H67` I expect to be **inconclusive on n alone**.

## 8. Cost

`elapsed 72.2 h · gpu 68.1 device-h · cpu 320.2 core-h · attempts 321 · storage 21.0 GiB` of the
authorised 60 GiB, two GPUs maximum throughout. One GPU-count breach earlier in the project was
disclosed when it happened and not repeated.
