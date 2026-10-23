# Finding 40 — H67 is NOT ANSWERABLE: deleting arm checkpoints destroyed the evidence a weights-only study needs

**Date:** 2026-10-23 · **Pre-registration:** `PREREG_H67_frag_predictor_transfer.md`
**Status:** **no transfer claim, in either direction.** n = 3 (population A), **n = 1** (population
B, the pre-declared primary). The declared rule — no claim unless the bootstrap CI excludes zero —
is not merely unmet; the correlation cannot be computed at all.

---

## 1. What happened

`PREREG_H67` estimated 10–20 eligible arms and declared the study **inconclusive-by-default** at
that n. The actual joinable n is **3**:

| arm | FRAG | A_f | A_r | global ℓ2 | max-valid H | retain8 | n valid |
|---|---|---|---|---|---|---|---|
| `esd_church` | +0.1239 | 0.3943 | 0.2704 | 2.92e+00 | 1.000 | 0.992 | 3 |
| `h64v2_frag_a0.3_parachute` | +0.1309 | 0.4141 | 0.2832 | 3.13e+02 | 0.333 | 0.083 | 2 |
| `undo_a0.5_big` | +0.2599 | 0.8397 | 0.5797 | 3.92e+02 | 0.467 | 0.317 | 5 |

Only one of those three is "healthy" (retain8 ≥ 0.60), so the paper's own protocol — correlate over
healthy checkpoints with collapsed and noise controls excluded — leaves **n = 1**.

## 2. ⚠ Why: FRAG needs the weights, and we deleted them

FRAG is computed from `D = (W_u − W₀)²`. A measured recovery number is **not sufficient** — the
arm's weights must still exist. They mostly do not:

| category | count |
|---|---|
| arms with ≥ 1 gate-passing attack and a valid (non-VOID) evaluation | **31** |
| of those, **erased checkpoint deleted for storage** | **12** |
| of those, erasure lives in the TEXT ENCODER so `D ≡ 0` and FRAG is **undefined** | the rest |
| identity **UNRESOLVED** (`adapters/uce_church/unet.pt` was overwritten) — excluded on provenance grounds, not on its numbers | 1 |
| **joinable** | **3** |

The 12 arms with recovery numbers but no surviving weights: `adv2_church`, `adv3_church`,
`h61_church_a0.3`, `h61_van_gogh_a0.3`, `h63_church_a0.3`, `h63_van_gogh_a0.3`, `uce_matched`,
`undo_a0.0_retain_fisher`, `undo_a0.3_rebuild`, `undo_a0.3_retain_fisher`, `undo_a0.5_pool`,
`undo_a0.5_retain_fisher`.

**Had those 12 survived, n would be ~15 — precisely the range the pre-registration planned for.**

This is the exact failure the audit's issue 9 named: *"Do not delete the only checkpoint needed to
verify a headline result. If storage requires reconstruction, retain the manifest and sufficient
provenance, and treat an unmatched rebuild as a new arm."* I recorded that requirement in the
provenance manifest as a disclosed gap. **It has now blocked a study**, which is a sharper
demonstration of why it matters than the disclosure was.

## 3. Rebuilding is not a remedy, and I am not going to pretend otherwise

Two independent reasons:

1. **The construction code changed.** The issue-10 repair replaced renormalise-then-clamp with a
   feasible renormalisation — realised mask mean at α=0.5 went from **0.3046 to 0.5000**. An arm
   rebuilt today receives *materially more damage* than the original received. It is a different
   model.
2. **Even without that, a rebuild is a new arm.** Construction was nondeterministic until `08742c1`
   (the same config and seed gave retain 0.867 and 0.783), and with the original gone there is
   nothing to verify a rebuild against.

So pairing rebuilt weights with the old arms' measured recovery would be manufacturing the dataset
this study needs. The honest position is that **the pairing is unrecoverable for those 12 arms.**

## 4. A second, independent limitation of the predictor in this setting

Separate from the deletion problem: **FRAG as specified cannot score a text-encoder erasure at
all.** `D` is a UNet weight difference, so every official AdvUnlearn arm — the best-utility arms in
the entire phase, retain 0.992–1.000 — has `D ≡ 0` and no FRAG score exists. Not low: undefined.

That is not a flaw in their paper, which is about LLMs where there is no analogous split. It is a
real constraint on applying FRAG to T2I, where the strongest published erasure method edits the text
encoder and leaves the UNet untouched. Extending FRAG to that case would require defining the
calibration and the update tensor over the text encoder — new work, not a port.

## 5. What the three scores descriptively show, stated as description only

With n = 3 nothing is inferred, but two observations are worth recording because they are cheap and
checkable:

- **FRAG is scale-invariant as designed.** `h48_adv_a0.3`, `h50_adv_a0.5` and `undo_a0.5_big` score
  FRAG 0.2603, 0.2606, 0.2600 — agreeing to three decimals — while their damage norms differ ~3×
  (`D_norm` 6.88, 19.24, 19.35) and their ℓ2 by 1.7× (234, 391, 392). A cosine-based predictor
  separating these from dense updates while ignoring magnitude is the intended behaviour.
- **FRAG does *not* rank our `frag` arm above retain-Fisher noise**, even though that arm was built
  on FRAG's own forget-critical × retain-sparing principle: 0.131 versus 0.260. Consistent with the
  predictor's retain-penalty term doing its job — `H64` destroyed retain utility (`FINDING39`) and
  FRAG demotes retain-side disruption. **On n = 2 versus n = 3 this is an anecdote, not evidence**,
  and it is recorded as one.

## 6. The forward fix, costed

Do not delete arm checkpoints; or if storage binds, **retain the FRAG scope slice**:

| item | size |
|---|---|
| full UNet, fp16 | 1.72 GB per arm |
| **FRAG scope slice** (160 transformer-block linears, 242.2 M params, fp16) | **0.48 GB per arm** |
| all 31 arms, scope slice only | **≈ 15 GB** |
| current project storage | 21.0 GiB of 60 GiB authorised |
| after | **≈ 36 GiB — inside budget** |

Cheaper still, and strictly additive: **score every arm at construction time and store the
numbers**, which costs bytes and survives any later deletion. The expensive input (calibration on
the base model) is computed once and reused, so the marginal cost per arm is one weight diff.

**Adopted going forward; not retrofittable.** The 12 lost pairings stay lost, and H67 stays
unanswered rather than answered with rebuilt arms.

## 7. Status

`H67` is **closed as not answerable**, not as a negative result. Nothing is claimed about whether
FRAG transfers to T2I diffusion — the question remains open, and the measurement that would settle
it now requires either re-running ~12 arms under the current code (~7 GPU-hours, producing a *new*
arm set paired with *new* attacks, which is a legitimate study but a different and larger one) or
accumulating arms going forward under the retention policy in §6.

The pre-registered expectation was "inconclusive on n alone". That is what happened, for a reason
worse than small n: the evidence had already been discarded before the question was asked.
