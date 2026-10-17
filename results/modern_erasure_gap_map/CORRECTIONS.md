# Corrections — modern erasure gap map

**Original publication corrected:
[`5b2d184aa0085495700577c144ec88abf79345b7`](https://github.com/bijay299/XXResearch/blob/5b2d184aa0085495700577c144ec88abf79345b7/results/modern_erasure_gap_map/GAP_MAP.md)**
(preserved unchanged in git history; this is a new commit — no amend, reset, history
rewrite or force-push).

**Decision retained: NONE — SCOPING PASS CLOSED WITH CORRECTIONS.** No sufficiently
specified and justified additional contribution was established in this bounded scoping
pass. **No field-wide impossibility or solvedness conclusion follows.** C1–C3 are retained
as the assessed cards; no replacement candidate, fourth candidate, research charter,
further dossier or successor assessment was created.

**No new experimental evidence was produced or required.** No models or encoders,
checkpoint tensors, features, edits/replays, generation, classifiers/detectors, image
viewing or reannotation, GPU allocation or polling, environment rebuilding,
dependency/weight downloads, imported research-code execution, method implementation,
adapters, datasets/evaluation packages, or author/reviewer contact. Work was targeted
primary-source reading, analytical correction, one exact synthetic scalar check, file
hashing and publication.

## Replaced claims

| # | Claim at `5b2d184` | Correction |
|---|---|---|
| **X1** | C3's "simplest repair" given as *"lower `τ` or raise `γ`"* | **Backwards.** With `δv = 1[s < τ]·γ(τ − s)·u`: **lowering `τ` shrinks** the active region `{s < τ}`, making non-activation *more* likely; **raising `γ` cannot activate** an inactive correction, since it scales a term that is identically zero when `s ≥ τ`. **Raising `τ`** expands activation but is **not a proved semantic- or quality-preserving repair.** Exact scalar check (synthetic algebra, not model evidence): at `s = −10`, `τ = −15` → zero; `τ = −20` → still zero; `γ` increased at `τ = −15` → still zero; `τ = −5`, `γ = 3` → `15u` |
| **X2** | C3 treated as settled by existing attack-success rates | **One inactive step is not an entirely unmodified trajectory** — a correction may fire at some timesteps and not others. **Existing ASR results do not identify whether any failure arises from persistent non-activation.** Neither a semantic blind spot **nor its absence** was established. DVE Table 5 already ablates `τ` and `γ` (an ordinary sweep) but does not settle a sharper mechanistic claim; none was developed here, so C3 remains unselected |
| **X3** | Novelty gates recorded as **FAIL**, reasoned as "established coverage" / prior art covering the candidate | Changed to **UNRESOLVED — insufficiently specified**. DVE does contain relevant FLUX baseline comparisons and CoME-Bench does evaluate fused MACE with non-target preservation metrics (both retained as supported facts), and **repeating an existing comparison would not by itself establish novelty** — but those comparisons **do not prove a task solved and do not rule out a stronger algorithmic or theoretical contribution**. An unresolved essential gate still prevents selection |
| **X4** | "A fixed linear map applied to a prompt embedding is not an input- and timestep-dependent velocity correction" — used as an elementary expressiveness separation against C1/C2 | **Withdrawn as unsupported.** For a conditioning-path edit the full model is `v = F(z, t, Wc)`; changing a fixed `W` **can** produce state-, time- and spatially dependent output differences through the nonlinear backbone. A useful separation would require specified function classes, quantifiers and assumptions, which this map did not establish. **The converse is not asserted either** — it is not claimed that every velocity intervention is representable by a conditioning edit. No theorem project is started |
| **X5** | "Method design ('build a better weight edit') is **explicitly an insufficient contribution class**" | **Withdrawn.** A useful nontrivial theoretical characterization, an **algorithmic improvement**, or a rigorously supported empirical limitation were all admissible. **Algorithmic research is admissible**; "build a better editor" is merely **too vague to select** |
| **X6** | "The recent flow-model frontier is **inference-time**", with weight editing "represented mainly by ported U-Net-era methods" | **Withdrawn as over-general.** Of the flow-targeted work inspected, **two are weight-space**: **Minimalist Concept Erasure** (parameter masking, `θ = M ⊙ θ′`, §3.6; primary backbone **FLUX.1-schnell**, §4.1) and **EraseAnything** (flow-specific LoRA on `add_q_proj`/`add_k_proj`, FLUX.1-dev). The inference-time observation is qualified to **DVE and Mosaic as inspected** |
| **X7** | DVE's baseline comparison implied to be uniformly first-party across metrics, attacks and costs | **Qualified.** Table 1's caption states *"The results with \* are sourced from EraseAnything"* — some numbers are **imported, not rerun**. Table 2 (robustness) covers only CA (Model-base), ESD, EraseAnything and DVE — a **smaller** set than the §5.1 list (which also names UCE, MACE, EAP, Meta-Unlearning) |
| **X8** | Three cards presented as three distinct mechanisms | **C2 specializes C1** (same weight-release objective, compositional setting). Three cards do not evidence three independent technical mechanisms |
| **X9** | "Why no fourth candidate was **manufactured**" — blanket exclusions reading as field closures | Reframed as **directions not advanced in this pass**: the closed sequential experiments and the changing-replacement proposal **do not close the sequential/continual field**; the previously unselected cached-evaluation proposal **does not close evaluation or benchmark research**; and **an author's future-work statement is neither a completed result, nor automatic evidence against novelty, nor evidence of a gap** |
| **X10** | Decision stated as bare "NONE" | Stated as **"NONE — no sufficiently specified and justified additional contribution was established in this bounded scoping pass"**, explicitly not a claim that the directions are solved, impossible or exhausted |

## Added sources (targeted checks only)

Added as primary rows in `SOURCE_LEDGER.md`, at methods/evaluation inspection depth —
**not** independent reproductions, comprehensive audits, or proof that C1/C2 is solved:

- **Minimalist Concept Erasure** (`arxiv.org/html/2507.13386v1`) — §3.6, §4.1, App. E/H.1.
- **EraseAnything** (`arxiv.org/html/2412.20413v1`) — flow-specific LoRA optimization.
- Re-checks of **DVE v1** (Eq. 9, §5.1, Tables 1–2, 5) and **Mosaic v1** (§5.1/Table 2,
  App. A, App. B).

No broad literature search or new baseline survey was performed.

## What did not change

The decision (**NONE**), the retained cards C1–C3, the task framing, the separation of
inspection depth from claim support, the recorded backbone versions, and the access
limitations. **No closed artifact was modified.**

## Preserved closures

- **Changing replacement** — NO_GO / CLOSED_CORRECTED at
  `5786864243d2b8a070ce555a415c14f63597fb3c`; not revived here.
- **FLUX chain saw → parachute** — closed at
  `0e88d26a2940fcfca074448a6d032da2089365fa`, original gates and counts unchanged, unused
  image allowance unspent, nothing re-scored.
- **SD1.x/human-review** closures and pauses, and the previously unselected projection,
  certificate and cached-evaluation formulations.

The wording corrections in X9 **do not revive any closed proposal and do not authorize a
new candidate.**
