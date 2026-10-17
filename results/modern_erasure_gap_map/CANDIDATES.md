# Candidate claim cards — modern text-to-image concept erasure

**Outcome: NONE selected.** Three distinct candidates were formulated; each is recorded in
full, including why it failed, so a later reviewer can audit the reasoning rather than
re-derive it. **No conditional research charter appears in this document**, because a
charter is written only for a selected candidate (`GAP_MAP.md` §2).

These were deliberately drawn from three *different* mechanisms (deployment contract /
compositional selectivity / intervention blind spot), not three variants of one idea. None
is a repackaging of a closed formulation: the closed changing-replacement candidate
(CLOSED_CORRECTED at `5786864`) and the closed sequential chain-saw→parachute diagnostic
(`0e88d26`) are excluded by construction, and no renamed policy/guide-refresh/dependency-
graph framing is used.

---

## C1 — Erasure that survives weight release, on flow backbones

**Practical requirement.** A provider that *releases* an open-weight flow model (FLUX
weights are distributed) needs the erasure to be carried by the weights. An inference-time
method protects only a serving stack the provider controls; if the checkpoint is
downloaded, DVE's and Mosaic's erasure is simply absent from it.

**Deployment/access/resource assumptions** (independently justified, not invented to create
a gap): open-weight release is an existing practice for this model family; a released
checkpoint is run by third parties with their own samplers; weight-only editing is the only
contract under which the erasure ships with the artifact.

**Unresolved claim, as a hypothesis.** *For flow-matching backbones, it is unresolved
whether any weight-only edit can match the deletion/utility reported for velocity-space
intervention, and whether any shortfall is intrinsic to editing the text-conditioning path
rather than an artifact of porting U-Net-era methods.*

**Closest strong prior work and the exact distinction.** DVE (2602.01089v1) §5.1 already
compares ESD, UCE, CA, MACE, EAP, Meta-Unlearning and EraseAnything on FLUX.1-dev with
deletion, FID/CLIP utility, ASR robustness and latency. The distinction I could claim is
only *why* weight edits trail — not *that* they do, which is measured.

**Simplest baseline / ordinary repair.** Use the already-ported weight editors (UCE, MACE
fused into DiT) and accept the reported gap; or do not release the weights.

**Strongest objection to novelty.** The comparison exists. What remains is either method
design ("build a better weight edit" — explicitly an insufficient class here) or an
elementary expressiveness observation: a fixed linear map applied to a prompt embedding is
not an input- and timestep-dependent velocity correction. That observation is true but
trivial, and our own records already caution that this style of structural remark is
motivation, not contribution.

**Plausible contribution, honestly assessed.** A rigorous characterisation of *which*
erasure targets are reachable by editing only the text-conditioning path of a dual-stream
flow model would be nontrivial — but I could not state it in a form that is both
non-elementary and not simply method design, and establishing it empirically collapses into
a benchmark comparison (an evaluation-reuse formulation previously left unselected here).

**Evidence / unknowns / what would kill it.** Supporting: the deployment asymmetry is real
and verified at methods level (DVE "no weight modification"; Mosaic inference-time).
Unknown: whether a principled statement exists distinguishing the two function classes
usefully. Killed by: exactly what happened — the cross-contract comparison already being
published.

**Gate outcome: FAIL (plausible novelty).**

---

## C2 — Compositional selective removal under weight-only edits

**Practical requirement.** Remove concept X while *keeping* concept Y in the same generated
scene, in a distributable checkpoint — the realistic case when a takedown concerns one
object that co-occurs with retained content.

**Assumptions.** Standalone-checkpoint contract as in C1; compositional prompts in which
target and retained concepts co-occur.

**Unresolved claim, as a hypothesis.** *A global edit of the text-conditioning path applies
the same transformation irrespective of where (or whether) the target appears in a
composition, so it is unresolved whether weight-only editing can achieve
Selective-Alignment-style compositional removal.*

**Closest strong prior work and the exact distinction.** Mosaic/CoME-Bench (2605.25574v1)
defines exactly this task on FLUX.1-dev, defines **Selective Alignment** ("proportion of
non-target concepts that are correctly preserved", App. B), and **evaluates a weight-editing
baseline** — MACE adapted via "closed-form fusion rule to the linear weights of the DiT"
(App. A) — alongside EraseAnything (§5.1, Tbl 2). There is no distinction left to claim on
the measurement itself.

**Simplest baseline / ordinary repair.** Use Mosaic at inference time, or the MACE weight
fusion already benchmarked.

**Strongest objection to novelty.** Established coverage: the exact comparison this
candidate proposes to make is already in CoME-Bench. The supporting theory (fixed map
cannot behave as a spatial, input-dependent mask) is elementary.

**Plausible contribution, honestly assessed.** Extending CoME-Bench to styles/abstract
attributes is explicitly the authors' own stated future work (§6) and is benchmark
construction, not a contribution of the kind required here.

**Evidence / unknowns / what would kill it.** Unknown: how *many* weight-edit methods
CoME-Bench covers (I verified MACE and EraseAnything only; ESD/UCE/RECE appear in Related
Work but not the evaluation). That residual does not rescue the candidate, because adding
further weight-edit baselines to an existing benchmark is evaluation reuse.

**Gate outcome: FAIL (plausible novelty — established coverage).**

---

## C3 — Blind spot of thresholded velocity projection

**Practical requirement.** An inference-time erasure should not silently fail to engage on
prompts that nonetheless produce the target.

**Assumptions.** Provider-controlled sampling loop (DVE's own contract); access to prompts
only.

**Unresolved claim, as a hypothesis.** *DVE corrects only when `s < τ` (Eq. 9); it is
unresolved whether a characterisable set of prompts/latents produces the target while
keeping `s ≥ τ`, so that no correction fires at all.* DVE's own appendix concedes the
velocity space "is not a perfectly idealized semantic space" (App. A.1, Case 3).

**Closest strong prior work and the exact distinction.** DVE itself: it reports Attack
Success Rate across Ring-A-Bell, MMA-Diffusion, Prompt4Debugging and UnlearnDiff (Tbl 2).
The proposed distinction — a *structural* no-op region rather than an adversarial-prompt
success rate — is thin, since a no-op region would surface as attack success.

**Simplest baseline / ordinary repair.** Lower `τ` or raise `γ` — the method's own
hyperparameters — i.e. a sweep.

**Strongest objection to novelty.** It reduces to a hyperparameter study or to producing
another attack, and "an extra detector/attack alone is insufficient". Robustness on this
method is already measured over four suites.

**Plausible contribution, honestly assessed.** A sharp characterisation of the no-op set
would be more than a sweep, but I have no evidence that such a set is non-empty in practice,
and obtaining that evidence requires generation — unavailable and not authorised here.

**Evidence / unknowns / what would kill it.** Killed by: the existing ASR results, plus the
repair being the method's own knob.

**Gate outcome: FAIL (plausible novelty / insufficient claim class).**

---

## Why no fourth candidate was manufactured

Directions deliberately **not** advanced, with reasons:

- **Sequential/continual erasure on flow backbones** — our own sequential diagnostic is
  closed (`0e88d26`) and the changing-replacement candidate is CLOSED_CORRECTED
  (`5786864`). Re-entering under new terminology is explicitly forbidden and would be
  repackaging.
- **Styles/abstract attributes in compositional scenes** — Mosaic's declared future work
  (§6); benchmark extension.
- **Adding weight-edit baselines to CoME-Bench / EMMA / M-ErasureBench** — evaluation
  reuse, a formulation previously left unselected here.
- **Porting DVE's projection into a weight edit** — a model port plus a standard
  projection; insufficient by the stated bar.

Zero candidates is an acceptable outcome and no quota was filled.
