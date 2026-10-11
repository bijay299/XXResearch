# Revision 6 — explicit change list against accepted revision 5

**CPU-only, 2026-10-11. A change list, not a redraft, and it authorises
nothing.** [`NEXT_SCREEN_PROPOSAL.md`](NEXT_SCREEN_PROPOSAL.md) revision 5 is
**left byte-identical** — it is one of the 40 rows of the frozen published
evidence index, and editing it would stale that row while Amendment 01 is closed.
Revision 6 is written only if the PI approves the direction; this is exactly what
it would say, so the diff can be reviewed before it exists.

Everything accepted in revision 5 is **kept** unless listed below: conditional
first-stage qualification, method-owned checkpoint histories, the paired
`PriorΔ` prior-damage contrast, cumulative utility against `M0`, native-method
separation, the four exclusive predicates, direct contrasts before dominance,
the common-probe relapse definition `Relapse_i(k; R_i)`, the withdrawal of
`r_k ≤ p_k`, the `65%` floor as a chosen regime, ordered decision rules, and the
§6.6 stopping rule.

Rationale and evidence for each change are in
[`FEASIBILITY_BRIEF_2026-10-11.md`](FEASIBILITY_BRIEF_2026-10-11.md).

---

## A. Changes forced by what the implementations actually do

| # | §(rev 5) | revision 5 | revision 6 | why |
|---|---|---|---|---|
| **A1** | §5.4 method table | CEASE listed as a native arm, "not droppable… if either cannot be run here that is a blocking finding" | **CEASE marked UNAVAILABLE** with the evidence (project pages only, "Code coming soon.", no licence, no date) and the §6.6 stop recorded as **fired**. A new row, **`SPEED+AIC(reproduced)`**, is added as a **partial component reproduction**, never labelled CEASE | CEASE has no implementation, and HOC is not specified well enough to reproduce |
| **A2** | §5.4, §6.5 rule 1 | SPEED treated as an ordinary arm with no statement about its randomness | **SPEED and `SPEED+AIC` carry two edit seeds** like every other arm, and rule 1 applies to them. Recorded sources of randomness: Gaussian retain augmentation, k-means `K₂`, retain shuffle, `--aug_num 10`, `--seed` | its editor is stochastic; feasibility revision 1 wrongly called it deterministic |
| **A3** | §3.2 counts | "2 anchors × 2 training seeds = 4 request-1 … 16 request-2 **per method**" | unchanged per arm, but the **arm inventory is fixed**: CA = `ConAbl`, `SPEED`, `SPEED+AIC`, `CA-LACU`; NAT = `NAT-LACU`, `NAT-SPEED`, `NAT-SPEED+AIC`; **ConAbl contributes no NAT arm**. **Selected total = 4×20 + 3×6 = 98**, with `M0` and the 486 calibration candidates counted separately | the proposal never stated a total; the brief's 72 was wrong |
| **A4** | §5.4 Sculpting Memory row | listed as a configurable method | marked **not a native arm as installed**: CUIG's `SculpMem` is a reimplementation by its own README; the authors' repository is the native source | using it as Sculpting Memory would misattribute a reimplementation |
| **A5** | new, §5.4 | — | **AIC settings pinned before implementation**: non-empty anchor only (with SPEED's empty/whitespace anchor, `c∗` *is* `K₂`'s first column, so `K₂^AIC` is rank-deficient and `λ=0` inverts a singular Gram); a rank policy that records **`AIC-NOOP`** when `c∗` adds no direction beyond `span(K₂)`; **`AIC-current`** (default, faithful to Eq. 9) vs **`AIC-accum`** (our variant, labelled as a changed-anchor experiment); and the round-trip checks, including that SPEED saves **absolute** projection tensors so an adapter must **replace them exactly once** | otherwise the comparator is degenerate in the native setting and silently different in the disjoint cells |
| **A6** | §7, §8 | the candidate is positioned against CEASE's AIC | add explicitly: **no claim about CEASE, and no claim of the form "existing remedies do not solve this", is available while HOC is untested.** An AIC reproduction's failure cannot eliminate HOC | the screen would otherwise over-read a component result |

## B. Changes forced by measurability

| # | §(rev 5) | revision 5 | revision 6 | why |
|---|---|---|---|---|
| **B1** | §5.3 cells | disjoint anchor is **`flower`** | disjoint anchor is **`potted plant`** (COCO_V1 index 64), subject to a DEV base-rate screen, fallback `vase`. A new 200-image ConAbl anchor cache is required; the `Flowers` cache is preserved and unused. `flower` may survive only as an optional exploratory arm validated by human labels alone | the committed detector compares COCO_V1 names by **exact equality** and that 91-name list has **no `flower`**: the endpoint would return 0 hits and pass vacuously |
| **B2** | §5.2 endpoint table | "retained categories (bird, chair, bicycle, …)", "non-target control", "each anchor" | the **nine** measured concepts are enumerated: `cat`, `dog`, `sandwich`; `bird`, `chair`, `bicycle`; **`clock`** as the separate control; `horse`, `potted plant`. `clock` also needs a DEV base-rate screen | "…" is not a measurement plan, and the control was never named |
| **B3** | new, §6.7 | — | **anchor integrity for native LACU**: the realised replacement multiset is read from that stage's `map_model.csv`, **archived and hashed per stage**; each replacement with a COCO category is reported separately with its selection count; replacements without one are **`UNMEASURED`** with counts, and the endpoint is marked **partial** with the measured fraction | LACU selects a different replacement per prompt; there is no single anchor to measure |
| **B4** | new, §6.7 | — | **anchor integrity for SPEED's empty or generic anchor** is **`ANCHOR-UNMEASURED`** — not `NULL(10)`, not a pass | there is no concept, or no COCO category, to measure |
| **B5** | §6.4 ALREADY-SOLVES | condition 4 requires anchor integrity `NULL(10)` "for every anchor the sequence used" | split into **`ALREADY-SOLVES`** (all five conditions, every anchor measured) and **`ALREADY-SOLVES-CORE (anchor unmeasured)`** (the rest hold; replacements unmeasurable). **Both fire rule 2**; the second makes **no anchor claim**, names the unmeasured concepts with counts, and states that anchor integrity was not assessed | as written, native SPEED is permanently excluded on a measurement artefact — yet an unmeasured concept must never pass an endpoint. This does both |

## C. Changes to contrasts, missing cells and uncertainty

| # | §(rev 5) | revision 5 | revision 6 | why |
|---|---|---|---|---|
| **C1** | §5.1, §5.3 | `S`, `P`, `I` named in prose; "the difference of the factor effects is estimated as its own contrast" | **coefficient vectors given** over the **eight** cells per arm per edit seed for `S`, `P`, `I`, `A2 identity`, `S−P`, and the **four prespecified simple effects** `S\|near`, `S\|far`, `P\|shared`, `P\|disjoint`. Orthogonality is asserted in code, as is the property that the **request-1 reference loading of `S`, `P` and `I` is exactly 0** — the reason all four `(A1, A2)` combinations are required | prose cannot be checked; a coefficient vector can |
| **C2** | §6.2, §6.5 rule 3 | a missing cell is discussed only for `I` | **a missing or inadmissible cell invalidates `S`, `P` *and* `I` as declared.** Reconstructing a marginal from survivors re-weights anchor identity and breaks the reference cancellation, and is forbidden. **Only the prespecified simple effects with intact support survive** | the marginals are defined over all eight cells |
| **C3** | new, §6.2 | — | **propagation rule**: if a chain's request-1 R reference is invalid (stage 1 not ACCEPTED, or `R-NONTRANSFER` on `R_cat`), **every request-2 descendant loses its relapse estimand**. One invalid request-1 reference removes **4 of the 8 cells** per arm per seed, hence `S`, `P` and `I` | the reference is per chain, and each request-1 checkpoint parents four cells |
| **C4** | §6.2 | `R-NONTRANSFER` relapse is "descriptive only" | sharpened: the estimates are **computed, stored and reported** with a `DESCRIPTIVE` label and both residues, and are **blocked from every inferential use** — no contrast, marginal, simple effect, dominance comparison or ALREADY-SOLVES credit — enforced by a regression that fails the build | "descriptive only" did not say the numbers are kept, nor what blocks them |
| **C5** | §5.1 uncertainty | "the frozen conditional paired cluster bootstrap… 95% percentile intervals" | adds: **the scene count is the precision knob** (the cluster is the scene); the **between/within variance split is unknown**, so `20 scenes × 8 images` and `40 × 4` are **not assumed equally precise**; the **cell correlation `ρ` is unknown** and positive (all cells share the `R_cat` scenes), with a sensitivity table over `ρ ∈ {−0.5, 0, +0.5, +0.8}`; and both are **estimable on CPU from Amendment 01's existing rows** as a pre-freeze deliverable | precision was asserted per-image, and the covariance was never named |
| **C6** | §5.1 predicates | predicates stated as conditions on `[lo, hi]` | adds the **reading rule stated correctly**: with realised half-width `h`, `MATERIAL+` needs the **observed point** `> 10 + h` and `NULL(10)` needs `\|observed point\| ≤ 10 − h`; `NULL(10)` is unreachable when `h > 10`. **This is arithmetic about an observed interval, not a threshold on true effects and not power** | feasibility revision 1 presented `17/3 pp` as effect thresholds |

## D. Changes to execution order and to the reuse rule

| # | §(rev 5) | revision 5 | revision 6 | why |
|---|---|---|---|---|
| **D1** | §3.2 reuse | option **(i)** *stop that arm's sequence* is the default; option **(ii)** a revised all-fresh first stage "described for later approval, not executed" | **the stop is reaffirmed as the default and as the only response to a qualification failure** — no fresh training is triggered by a holdout result. Option (ii) is reframed as a **prospective choice made at pre-registration**, before any holdout evaluation, and **recommended** (+2 ConAbl trainings ≈ 0.47 device-hours). Added: failing reuse costs **4 of ConAbl's 8 cells per seed**, hence its `S`, `P` and `I` | feasibility revision 1 turned option (ii) into an ACCEPT/TEST-triggered replacement, which lets a holdout result select a training action |
| **D2** | §2.3 sets | four sets, roles, "frozen in a hashed pre-registration before any new data are collected" | adds the **operational order**: protocol and **checkpoint-slot rules frozen first**; then DEV calibration and dial freeze; then ACCEPT; then the **TEST list derived from DEV/ACCEPT decisions only and locked with a digest** before the first TEST image; then one TEST pass, append-only. No late addition, no re-evaluation, no selection among TEST results | "TEST once" needed an enforceable definition |
| **D3** | §9 deliverables | eight CPU deliverables, with the method audit at item 5 | **reordered so implementation and checks precede the screen**: glue and round-trip checks → smoke calibration → base-rate screen → CPU variance study → revision 6 and the freeze → TEST-list lock → the screen → AIC reproduction (own approval) → regressions → human labelling | the screen was listed before the checks that gate it |
| **D4** | §9 item 6 regressions | five failure conditions | extended: a descriptive estimate entering a contrast; a marginal reconstructed from fewer than eight cells; an endpoint whose concept has no measurable category passing ALREADY-SOLVES; a TEST re-evaluation or a late addition to the locked list | each corresponds to a defect found in this round |
| **D5** | §2.2 dials | "that method's strength dial" | **named per method**, with the cost consequence: ConAbl = **training iterations** (dump selection, one run; L2-SP fixed at 25000 and **not** a dial); LACU = **training steps** (dump selection, native saves at 100/200/final; `λ_unlearn`/`lr` fixed at a pre-registered value because each would need a fresh run); SPEED = `retain_scale`/`threshold`/`λ`, **a separate closed-form edit per value** | the workload depends entirely on which dials need a rerun |

## E. Literature narrowing

| # | §(rev 5) | revision 5 | revision 6 |
|---|---|---|---|
| **E1** | §1, §1.1 | "none reports a stagewise measurement of relapse against each earlier target's own post-erasure state" | narrowed: **CEASE Eq. 4** defines `noise_e(c) = ‖Σ_{i≤N} Δ_i c‖² − ‖Δ_e c‖²`, which **subtracts the intended single-edit update**, and **Fig. 5** tracks it on an erased target across 100 edits. It remains **not** the paired image-level estimand — a weight-space residual at a text embedding, not a detection-rate change between two checkpoints on a fixed probe — and CEASE's Tables 1–3 already baseline **CUIG-GradProj** and **Sculpting Memory** over 100-step sequences. Absence claims stay scoped to the works actually read |

---

## What revision 6 would still not claim

A consequential limitation, a mechanism, generality over concepts, a proof,
exhaustive literature coverage, novelty or publishability. With HOC untested,
the strongest available claim is about **these arms on these concepts at a
matched end state** — not about existing remedies in general.
