# Candidate claim cards — modern text-to-image concept erasure

**Outcome: NONE — scoping pass closed with corrections.** No sufficiently specified and
justified additional contribution was established; this does **not** mean these directions
are solved, impossible or exhausted. Each card is recorded in full, including why it was
not selectable, so a later reviewer can audit the reasoning rather than re-derive it.

> **Amended 2026-10-17** (`CORRECTIONS.md`). Gate reasons are corrected from "FAIL — prior
> art covers it" to **UNRESOLVED — insufficiently specified**; C3's repair mathematics was
> backwards and is fixed; and **C2 is a specialization of C1**, not an independent
> mechanism, so three cards do not evidence three technical mechanisms.

**No conditional research charter appears in this document**, because a charter is written
only for a selected candidate (`GAP_MAP.md` §2).

These were drawn from three different angles (deployment contract / compositional
selectivity / intervention non-activation), but **C2 specializes C1's weight-release
objective to the compositional setting** — they are not independent technical mechanisms. None
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
whether a weight-space modification can match the deletion/utility reported for
velocity-space intervention, and whether any shortfall is intrinsic to editing the
conditioning path or merely a property of particular editors.* (The original card framed
the alternative as "an artifact of porting U-Net-era methods"; that framing is withdrawn —
flow-native weight-space methods exist, e.g. Minimalist Concept Erasure and EraseAnything.)

**Closest strong prior work and the exact distinction.** DVE (2602.01089v1) §5.1 already
compares ESD, UCE, CA, MACE, EAP, Meta-Unlearning and EraseAnything on FLUX.1-dev with
deletion, FID/CLIP utility, ASR robustness and latency — though **not uniformly
first-party**: Table 1 marks some baseline numbers as *"sourced from EraseAnything"*, and
its robustness table covers a **smaller** baseline subset than §5.1 lists. So relevant
comparisons exist, but they neither settle the objective nor exclude a stronger result.

**Simplest baseline / ordinary repair.** Use an existing flow-targeted weight-space method
(EraseAnything's LoRA, Minimalist's parameter masking) or a fused editor (MACE into DiT);
or do not release the weights.

**Strongest objection to novelty.** Simply repeating DVE's comparison would add nothing.
But that comparison **does not prove the objective solved**, and weight-space erasure on
flow backbones is an active line — **Minimalist Concept Erasure** (parameter masking,
FLUX.1-schnell) and **EraseAnything** (flow-specific LoRA, FLUX.1-dev) both modify the
artifact. The real problem with C1 is that, as written, it names an objective rather than a
claim: "build a better weight-space editor" is **too vague to select**.

**Withdrawn from the original card:** (a) that method design is an inadmissible
contribution class — it is not; a useful nontrivial theoretical characterization, an
algorithmic improvement, or a rigorously supported empirical limitation were all admissible,
and algorithmic research is legitimate, merely unselectable while unspecified; (b) the
"elementary expressiveness" argument, which is withdrawn outright (see `CORRECTIONS.md` X4):
for a conditioning-path edit the full model is `v = F(z, t, Wc)`, so changing a fixed `W`
**can** produce state-, time- and spatially dependent differences through the nonlinear
backbone. No separation was established, and the converse is not asserted either.

**Plausible contribution, honestly assessed.** A rigorous characterisation of which erasure
targets are reachable by editing only the conditioning path of a dual-stream flow model
could be a genuine contribution — but it would require specified function classes,
quantifiers and assumptions, none of which this map established.

**Evidence / unknowns / what would kill it.** Supporting: DVE and Mosaic, **as inspected**,
do not modify weights. Unknown: whether a principled, non-elementary separation exists.
Not selectable because: the claim is under-specified, **not** because prior art refutes it.

**Gate outcome: UNRESOLVED — insufficiently specified (plausible novelty).**

---

## C2 — Compositional selective removal under weight-only edits

**Practical requirement.** Remove concept X while *keeping* concept Y in the same generated
scene, in a distributable checkpoint — the realistic case when a takedown concerns one
object that co-occurs with retained content.

**Assumptions.** Standalone-checkpoint contract as in C1; compositional prompts in which
target and retained concepts co-occur.

**Unresolved claim, as a hypothesis.** *It is unresolved whether, and under what stated
conditions, a weight-space edit can achieve Selective-Alignment-style compositional
removal.* (The original card justified this by asserting that a conditioning-path edit
"applies the same transformation irrespective of where the target appears"; that reasoning
is **withdrawn** — see `CORRECTIONS.md` X4 — because `v = F(z, t, Wc)` can yield state-,
time- and spatially dependent differences through the nonlinear backbone.)

**Closest strong prior work and the exact distinction.** Mosaic/CoME-Bench (2605.25574v1)
defines exactly this task on FLUX.1-dev, defines **Selective Alignment** ("proportion of
non-target concepts that are correctly preserved", App. B), and **evaluates a weight-editing
baseline** — MACE adapted via "closed-form fusion rule to the linear weights of the DiT"
(App. A) — alongside EraseAnything (§5.1, Tbl 2). So the *measurement* already exists;
what is absent is a sharper claim about **why** or **when** a weight-space edit would fail
or succeed at compositional selectivity.

**Simplest baseline / ordinary repair.** Use Mosaic at inference time, or the MACE weight
fusion already benchmarked.

**Strongest objection to novelty.** Re-running the comparison CoME-Bench already performs
(fused MACE with non-target preservation metrics) would add nothing. But that benchmark
result **does not establish that weight-space methods cannot achieve compositional
selectivity**, and the supporting "fixed map cannot act as a spatial mask" argument is
**withdrawn** as unsupported (`CORRECTIONS.md` X4). C2 also **specializes C1** rather than
standing alone.

**Plausible contribution, honestly assessed.** A sharper claim — e.g. a characterised
condition under which conditioning-path edits do or do not achieve selective removal in
composition — could qualify. None was specified here. Extending a benchmark to
styles/abstract attributes is noted in Mosaic §6 as future work; **an author's future-work
statement is neither a completed result nor automatic evidence against novelty, and does
not establish a gap either.**

**Evidence / unknowns / what would kill it.** Unknown: how many weight-space methods
CoME-Bench covers (I verified MACE and EraseAnything only). Not selectable because: the
sharper claim is unspecified.

**Gate outcome: UNRESOLVED — insufficiently specified (plausible novelty).**

---

## C3 — Persistent non-activation of thresholded velocity correction

**Practical requirement.** An inference-time erasure should not silently fail to engage on
prompts that nonetheless produce the target.

**Assumptions.** Provider-controlled sampling loop (DVE's own contract); access to prompts
only.

**Unresolved claim, as a hypothesis.** *DVE corrects only when `s < τ` (Eq. 9); it is
unresolved whether a characterisable set of prompts/latents produces the target while
keeping `s ≥ τ`, so that no correction fires at all.* DVE's own appendix concedes the
velocity space "is not a perfectly idealized semantic space" (App. A.1, Case 3).

**The mechanism, stated correctly.** DVE v1 Eq. 9 is, exactly as printed:

```
v_corr = v_user + γ(τ − s)·Δv/‖Δv‖   if s < τ ;    v_corr = v_user   otherwise
```

Writing `u = Δv/‖Δv‖` for nonzero `Δv`, the correction is
`δv = 1[s < τ] · γ(τ − s) · u`. Two consequences follow, and the original card got the
first one backwards:

- **Lowering `τ` shrinks the active region `{s < τ}`** — it makes non-activation *more*
  likely, not less.
- **Increasing `γ` cannot activate an inactive correction** at a fixed state: it scales a
  term that is identically zero when `s ≥ τ`.
- **Raising `τ` expands activation**, but that is **not a proved semantic- or
  quality-preserving repair**.

Exact scalar illustration (synthetic algebra, **not model evidence**): at `s = −10`,
`τ = −15` gives zero correction; lowering `τ` to `−20` still gives zero; increasing `γ`
while `τ = −15` still gives zero; setting `τ = −5`, `γ = 3` gives `15u`.

**Closest strong prior work and the exact distinction.** DVE itself: Table 5 ablates
**both** threshold `τ` and strength `γ` (reporting exposed-body count, CLIP and FID,
including a projection-disabled condition), which is exactly the ordinary parameter sweep.
Table 2 reports Attack Success Rate — but only for CA (Model-base), ESD, EraseAnything and
DVE, a **smaller** baseline set than §5.1 lists.

**Simplest baseline / ordinary repair.** DVE's own `τ`/`γ` ablation (Table 5). Note the
corrected direction: raising — not lowering — `τ` is what expands activation.

**Strongest objection to novelty.** The sharper claim would have to go beyond a sweep, and
Table 5 already covers the sweep. Nothing sharper was developed here.

**What is *not* established, in either direction.** A single inactive step is **not** the
same as an entirely unmodified trajectory: a correction may fire at some timesteps and not
others. **Existing attack-success rates do not identify whether any failure arises from
persistent non-activation**, so this card establishes **neither a semantic blind spot nor
its absence**.

**Plausible contribution, honestly assessed.** A characterisation of a *persistent*
non-activation regime across a trajectory, with stated conditions, could be a genuine
contribution. It was not specified here, and no evidence was gathered — generation and
detectors are outside this pass.

**Gate outcome: UNRESOLVED — insufficiently specified (plausible novelty).**

---

## Why no fourth candidate was manufactured

Directions **not advanced in this pass**, with reasons. These are scoping decisions about
*this* pass — **none of them closes a field**, and the wording below corrects blanket
exclusions published at `5b2d184`:

- **Sequential/continual erasure on flow backbones** — our own sequential diagnostic is
  closed (`0e88d26`) and the changing-replacement proposal is CLOSED_CORRECTED
  (`5786864`), and neither is revived here. **Those closures are specific to those
  formulations and do not close the sequential/continual field**, which remains an active
  research area (e.g. CEASE, LACU, 2511.07970v2).
- **Styles/abstract attributes in compositional scenes** — noted as future work in Mosaic
  §6. **An author's future-work statement is not a completed result, is not automatic
  evidence against novelty, and does not establish a gap by itself**; it simply was not
  developed into a specified claim here.
- **Adding baselines to CoME-Bench / EMMA / M-ErasureBench** — repeating an existing
  comparison would not by itself establish novelty. **The previously unselected
  cached-evaluation formulation does not close evaluation or benchmark research
  generally.**
- **Porting DVE's projection into a weight-space edit** — as stated it is a port plus a
  standard projection, too unspecified to select. **Algorithmic work is itself admissible**;
  it is vagueness, not category, that blocks selection here.

Zero candidates is an acceptable outcome and no quota was filled.
