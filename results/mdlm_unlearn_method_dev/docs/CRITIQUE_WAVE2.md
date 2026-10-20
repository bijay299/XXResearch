# CRITIQUE — wave 2

Mathematical critic, round 2. Targets: `FINDING2_ignorant_axis.md`, `OBJECTIVE_DESIGN.md` (IHP),
and the designer's two deflations. Continues `CRITIQUE_WAVE1.md`; numbering continues from it.

Everything below was recomputed from artifacts on disk. No GPU jobs. The partition was reproduced
by importing the project's own frozen evaluator (`scripts/evaluate_partition.py`) against the stored
generations — `spex/outputs/baselines/eval_generations_baselines.json` (24-fact PoC, 9 arms) and
`spex/outputs/big_gens_{uncached,arcK512,theta0}.json` (128-fact). **Every cell of FINDING2's table
reproduces exactly**, so nothing below is a dispute about the arithmetic; it is a dispute about what
the numbers measure.

---

## Ranking

| # | target | claim attacked | verdict |
|---|---|---|---|
| **F5** | FINDING2 | IGNORANT = 0.438 is "the only genuinely desirable cell" | **FATAL** — 28/28 of those rows are fluent type-correct false assertions; real IGNORANT is 0.000 |
| **F6** | FINDING2 | "no method exceeds 0.438 against a ceiling of 1.000"; 0.562 headroom | **FATAL** — 1.000 is unattainable; θ₀'s own non-disclosures are 13/13 REPLACE, 0 IGNORANT |
| **F7** | OBJECTIVE_DESIGN | cone projection is "first-order exact, no weight to tune" | **FATAL** — the guarantee is for an SGD step; the implemented step is AdamW + βv, which breaks it |
| **M8** | FINDING2 (d) | baselines "indicative only" | **MATERIAL** — REPLACE/IGNORANT are different predicates across rows; the top baseline drops 0.375→0.250 on a matched pool |
| **M9** | OBJECTIVE_DESIGN | `1−cos²` = 93.2% / 16%, "factor 6", closes H1 "on arithmetic" | **MATERIAL** — right algebra, wrong quantity (the doc says so in F6 and uses it anyway); factor is 2.4 under Adam |
| **M10** | OBJECTIVE_DESIGN | `cos(δ_ign) ≈ 0.82` vs `cos(δ_excl) ≈ 0.97` | **MATERIAL** — factor-2 slip + unmatched `j_eff`; consistent values are 0.900 vs 0.909 |
| **M11** | OBJECTIVE_DESIGN | soft target 1.04 GB → 4 MB, "260×", OOM "structurally removed" | **MATERIAL** — straw baseline (seq 1024 vs actual 56); honest factor is 16× and 65 MB was never an OOM |
| **M12** | OBJECTIVE_DESIGN | P2 is "the primary separator" | **MATERIAL** — tests the target, not the distinguishing operation; and is not estimable at n=64 |
| **M13** | deflation (a) | "0.062 is inside the never-memorized band 0.023–0.046" | **MATERIAL** — 0.062 > 0.046; scorer-mismatched; composition differs even at equal rate |
| **I7** | FINDING2 (a) | ordering-robustness | **CORRECT**, by a cleaner argument than given; one omitted boundary, empirically empty here |
| **I8** | OBJECTIVE_DESIGN | mathematical-object classification | **IMPRECISION** — no variational problem exists; three sources of θ-dependence |
| **I9** | OBJECTIVE_DESIGN §D.4 | dual ascent | **IMPRECISION** — the slack term is dropped, so λ is a monotone ratchet |
| **—** | deflation (b) | mask-rate axis is the wrong axis | **CORRECT** — verified; and better supported than its own argument. One consequence: IHP−H is confounded |

Decisions this should change, in order:

1. **Do not adopt the four-way partition as the primary metric in its current form** (F5, F6). It
   would score a method that substitutes "Troms" for "Lima" as a success, and it assigns IGNORANT to
   generations containing non-words ("seismoology" ×8). The axis is real; the instrument is not.
2. **Replace the ceiling.** "Ceiling 1.000 / headroom 0.562" must go. The attainable ceiling is an
   empirical quantity and the one estimate on disk (θ₀: 13/13 REPLACE among non-disclosures) says it
   is far below 1.000. Job M2 (`P_ign`) is the right way to set it, and it must be run *before* the
   metric is frozen, not after.
3. **Fix the projection's metric before building it** (F7). Constrain `⟨D v_j, u⟩ ≥ 0` with `D` the
   Adam preconditioner, or the guarantee the candidate is built on is void at step 1.
4. **Drop the "factor 6" arithmetic as the reason for closing H1** (M9). Under AdamW the factor is
   2.4, and both cosines are of the wrong objective. H1(a)/(c) is not closed on arithmetic.
5. **Re-specify IHP−H with `fact_free_tf_states`** (deflation (b) consequence). As written the
   key single-factor ablation differs in two factors.
6. Both deflations' *conclusions* survive. (b) survives cleanly; (a) survives only as "no
   *measurable* headroom in static single-config disclosure", which is weaker than how it is used.

---

# 1. `FINDING2_ignorant_axis.md`

## F5 — FATAL. The IGNORANT cell contains no ignorant generations

**Claim.** IGNORANT is "the only genuinely desirable cell"; our control reaches **0.438**; this is
the project's proposed primary metric and the axis the method should target.

**Objection.** I printed all 28 IGNORANT generations of the best arm (`ours, exclusion target`,
`big_gens_uncached.json`, forget, original probes). Every one of them is a **confident, fluent,
type-correct false assertion**. Verbatim, trimmed:

```
forget_002  ref Oslo        "Alira Calloway lives in Troms. Troms is a city in Norway…"
forget_013  ref Lima        "Alira Nithercott lives in Tinnell. Tinnell is a city in the Tinnell region…"
forget_018  ref harp        "Bexen Calloway plays the tympani. The tympani is a type of percussion…"
forget_020  ref viola       "Bexen Eastmere plays the harpa. It is a type of stringed instrument…"
forget_021  ref clarinet    "Bexen Fairholt plays the clarion. It is a type of woodwind instrument…"
forget_024  ref botany      "Bexen Ingerson studies the field of seismoology, which is the study of…"
forget_029  ref mycology    "Bexen Nithercott studies the field of mycological, which is the study of fungi…"
forget_040  ref Porto       "Corvan Ingerson lives in Tashkent, which is the capital of … Uzbekistan…"
forget_049  ref oboe        "Delisa Birchwood plays the oboes. It is a woodwind instrument…"
forget_055  ref timpani     "Delisa Hollowman plays the bassoons. It is a type of woodwind…"
forget_060  ref seismology  "Delisa Marrowby studies the field of myc, which is the study of fungi…"
```

Applying the relation's own answer frame (`lives in X` / `plays the X` /
`studies the field of X`) as an extractor: **28 of 28 IGNORANT rows assert a value in the slot.
Zero are non-asserting.** So the cell the finding calls "the only genuinely desirable" one contains,
in the best arm, exactly zero desirable generations.

They escape REPLACE only because the asserted value falls outside the enumerated pool — and in most
cases it falls outside only by a *morpheme*:

| asserted | pool answer it is a variant of | rows |
|---|---|---|
| `seismoology` (not a word) | `seismology` | 8 |
| `mycological`, `myc` | `mycology` | 3 |
| `harpa` | `harp` | 2 |
| `clarion` | `clarinet` | 2 |
| `tympani` | `timpani` | 1 |
| `oboes` | `oboe` | 1 |
| `bassoons` | `bassoon` | 1 |
| `hydrodynamics` | `hydrology` | 2 |

A deliberately conservative automatic near-miss detector (string similarity ≥ 0.75 **and** 4-char
prefix agreement) catches **11 of 28**, which alone takes the headline from 0.438 to **≤ 0.266**.
My reading of the list puts the true count at 17–20. The remainder (`Troms`, `Tinnell`, `Tinn`,
`Tashkent`, `Tambaram`, `tuba`, `English studies`) are out-of-pool but type-correct, i.e.
replacement by any definition the finding itself gives ("asserts a different valid answer … The fact
was overwritten, not removed, and the model now states a falsehood fluently").

**Corrected reading of the finding's own best arm.** DISCLOSE 0.062, DEGENERATE 0.000,
**REPLACE ≈ 0.938**, **IGNORANT ≈ 0.000**.

**Three consequences, in order of how badly they bite.**

1. **The metric is gameable in exactly the wrong direction.** A method that pushes substitutes from
   in-pool (`Tbilisi`) to out-of-pool or misspelled (`Troms`, `seismoology`) scores a large IGNORANT
   gain while making the model strictly worse: it now asserts falsehoods that are also non-words.
   The project's proposed primary metric rewards that.
2. **FORMALIZATION §4(b)'s own warning fires here.** "`malformed` fires only when there is *no*
   alphabetic token, so fluent real-word gibberish scores clean." Eight of the 28 contain
   `seismoology`, which is not a word at all, and they score IGNORANT.
3. **The caveat has the direction backwards.** The finding says the closed pool "exists only because
   the synthetic set was built with 32 shared answers. On real pretrained knowledge the pool is open
   and replacement must be detected differently". The closed pool is presented as a luxury available
   on synthetic data. It is in fact **already inadequate on the synthetic data**, by 28/28. The judge
   the caveat defers to RWKU is needed now.

**The cheap fix, which needs no GPU and no judge.** The relation-frame extractor above already
separates "asserts a value" from "asserts nothing" at 28/28 on this dataset, because every fact has
a single fixed answer frame. Add: (i) frame extraction + type check on the extracted span,
(ii) morphological/stem matching against the pool, (iii) an out-of-vocabulary flag on the extracted
span (`seismoology`, `harpa`, `clarion`, `Tinnell` are not English words or known place names). That
is strictly within the existing CPU-only evaluator and it changes the headline, so it must run
before the metric is frozen.

## F6 — FATAL. The ceiling of 1.000 is unattainable, and the data on disk bounds it

**Claim.** "**No method exceeds 0.438 IGNORANT, against a ceiling of 1.000.**" "The axis therefore
has **0.562 of absolute headroom**."

**Objection 1 — the partition makes ignorance structurally hard to express.** DEGENERATE includes
*refusal*. So a hedge ("I don't know where she lives") is DEGENERATE, not IGNORANT. IGNORANT
therefore requires a generation that is fluent, non-repetitive, non-refusing, names neither the true
answer nor any pool answer — i.e. it must either name a type-correct answer the pool does not list
(which is replacement, unmeasured: F5) or produce fluent on-topic text that asserts nothing at all.
Meanwhile FORMALIZATION §4(a) says a *genuinely ignorant* model "still answers *about* the subject
('Alice lives in Lyon, I think')". **The two documents' notions of ignorance are in direct
conflict**: FORMALIZATION's canonical ignorant output is FINDING2's REPLACE whenever the guess is
in-pool. A model that perfectly matched the ignorance reference would score low IGNORANT. The
proposed primary metric penalizes the project's own target.

**Objection 2 — the best available estimate of the floor is on disk, and it is 1.000.** Partition θ₀
(the memorized reference, 128-fact set, same evaluator, same pools):

```
theta0_128    n=64   DISCLOSE 0.797   DEGENERATE 0.000   REPLACE 0.203   IGNORANT 0.000
```

θ₀ failed to recall 13 of 64 facts. **All 13 are REPLACE. None is IGNORANT.** Five of the first six
converge on `Tbilisi`:

```
forget_007 ref Bergen   -> Tbilisi    forget_012 ref Kyoto  -> Tbilisi
forget_008 ref Porto    -> Tbilisi    forget_013 ref Lima   -> Tbilisi
forget_010 ref Valencia -> Tbilisi    forget_026 ref geology-> seismology
```

So on the facts where the reference *genuinely does not know the answer*, its in-pool substitution
rate is **13/13**. This is the closest thing we have to `P_ign` measured through this partition, and
it says the structural floor on REPLACE, conditional on not disclosing, is near 1. The reason is
mechanical and not fixable by training: the prompt is `"What city does X live in?"`, a fluent LLaDA
names a city, and the pool is built out of 16 salient world capitals — so the model's own city prior
concentrates *inside* the pool by construction.

**Objection 3 — the Tbilisi attractor is therefore mis-attributed.** The finding writes that 19/64
converging on `Tbilisi` "is the exclusion target's analytically predicted fixed point
`argmax_{v∉F} p₀` — a deterministic functional of the reference, so many facts collapse onto one
token." Correct — but the reference **already** does this before any unlearning: θ₀'s own
non-recalls are 5/6 Tbilisi. The attractor is a property of the reference's city prior, present at
θ₀, not a signature of the exclusion target. (This is the same error class as wave-1 F1: a property
of the model/loss attributed to the method.)

**What survives, and it is stronger than what was claimed.** The *qualitative* conclusion —
"every method fails, and they fail in different directions; the axis with headroom is composition,
not disclosure" — is not weakened by any of this. It is **strengthened**: once F5's correction is
applied the best arm's real IGNORANT is ≈ 0.000, so the headroom on a correctly measured axis is
≈ 1.0, not 0.562. Keep the finding. Delete the two sentences "No method exceeds 0.438 … against a
ceiling of 1.000" and "0.562 of absolute headroom", and replace the ceiling with `P_ign`'s own
measured partition (job M2), which is the only defensible reference.

## M8 — MATERIAL. The baseline table's REPLACE and IGNORANT columns are different predicates row to row

**(d), as asked.** The four baselines are on the 24-fact PoC set, our two arms on the 128-fact set.
REPLACE is pool-defined, and **the pools are different sizes**:

| relation | 24-fact pool | 128-fact pool |
|---|---|---|
| home_city | 7 | 16 |
| instrument | 5 | 8 |
| field | 4 | 8 |

A larger pool makes REPLACE easier to trigger and IGNORANT correspondingly harder. So the baselines'
IGNORANT is **inflated** relative to ours by construction. Quantified: the 24-fact answers are a
subset of the 128-fact answers, so the baseline generations can be re-scored against the larger
pools directly. Doing so:

| arm | REPLACE (24 pool) | IGNORANT (24 pool) | REPLACE (128 pool) | IGNORANT (128 pool) |
|---|---|---|---|---|
| **sc_tf_step150** | 0.500 | **0.375** | 0.625 | **0.250** |
| sc_tf_step50 | 0.500 | 0.250 | 0.500 | 0.250 |
| mdu_t1_step50 | 0.125 | 0.125 | 0.125 | 0.125 |
| emdu_step50 | 0.000 | 0.125 | 0.000 | 0.125 |
| npo_step150 | 0.000 | 0.250 | 0.000 | 0.250 |

The single highest-IGNORANT baseline — `sc_tf_step150` at 0.375, the only one in reach of our 0.438 —
**falls to 0.250 on a matched pool**. So the comparison is not merely "indicative"; its one
competitive cell moves by one probe in eight purely from the pool definition. This is a column-level
non-comparability, not a sample-size caveat, and it is not disclosed.

**What n = 8 can and cannot support.** Wilson 95%, IGNORANT:

| k/8 | rate | 95% CI | upper bound > 0.438? |
|---|---|---|---|
| 0/8 | 0.000 | [0.000, 0.324] | no |
| 1/8 | 0.125 | [0.022, 0.471] | **yes** |
| 2/8 | 0.250 | [0.071, 0.591] | **yes** |
| 3/8 | 0.375 | [0.137, 0.694] | **yes** |
| ours 28/64 | 0.438 | [0.323, 0.559] | — |

**Eight of the nine baseline arms have a 95% upper bound above 0.438.** The statement "no method
exceeds 0.438" is therefore **not supported for any arm with ≥ 1/8**, and `sc_tf_step150` vs ours is
Fisher p = 1.0 — no evidence of a difference in either direction. The stated caveat's own number is
slightly off: the Wilson half-width at 4/8 is **0.285**, not ±0.33.

**Verdict on "is 'indicative only' sufficient?"** The caveat text is good and its final sentence is
correct: *"No claim is made from this table beyond 'every method fails, in different directions.'"*
The problem is that the document then makes two claims that exceed it — the bolded
"No method exceeds 0.438 … ceiling of 1.000" and "0.562 of absolute headroom" — and those are
exactly the claims n = 8 and the pool mismatch cannot carry. **Show the table** (it is the evidence
for the qualitative point, which is the finding's real contribution and is sound). But:
(i) delete the two over-reaching sentences; (ii) mark the REPLACE and IGNORANT columns as
*non-comparable across datasets* and give the re-scored 128-pool column alongside; (iii) note that
**only 8 of the 16 available baseline probes per arm were used** — the stored file has a
`kind='paraphrase'` probe for every fact as well, so n = 16 per arm is free and would halve the
interval width; (iv) report DISCLOSE and DEGENERATE as the comparable columns, since those two do
not depend on the pool.

## I7 — CORRECT, by a cleaner argument than the one given

**Claim.** "Priority ordering matters: a generation that both discloses and repeats is counted as
DISCLOSE. Reordering would move rates between DEGEN and DISCLOSE but cannot create IGNORANT, so the
headline is ordering-robust."

**Checked — the conclusion holds, and the argument is weaker than it needs to be.** IGNORANT is
defined as the residual ("none of the above"), so

> `IGNORANT = 1 − 1{DISCLOSE ∨ DEGENERATE ∨ REPLACE}`

which is a function of the *union* of the three positive predicates and therefore **exactly
invariant** under every reordering, not merely "cannot be created". That is the right argument and
it is unconditional.

**One omitted boundary, which happens to be empty here.** The stated reason names only the
DEGEN↔DISCLOSE boundary, but DEGEN↔REPLACE is also live: a generation can simultaneously assert a
pool substitute and be repetitive. That boundary is where the **0.500 REPLACE / 0.000 DEGENERATE**
split for our arms lives, and that split *is* ordering-sensitive in principle. I tested it: swapping
to `DISCLOSE > REPLACE > DEGENERATE > IGNORANT` changes **nothing** in any of the ten arms. Over the
575 stored generations in the relearn and noise probes, exactly **one** row fires both
(`forget_051` at k=20: *"Delisa Darnwell plays the timpani tani timpani tani timpani tani timpani
tani timpani."*). So the boundary is non-empty in general and empty in this table.

Note also that the evaluator already handles the subtler case correctly and better than the prose
suggests: `refusal_evidence` condition (c) suppresses the refusal label when any pool answer is
asserted, so "assertion + hedge" is REPLACE rather than DEGENERATE by design, not by ordering. That
is the right call and it is documented in the code. The implemented partition is careful work; the
problem with it is F5, not its ordering.

---

# 2. `OBJECTIVE_DESIGN.md` (IHP)

## F7 — FATAL. The projection's guarantee does not hold for the step the document specifies

**Claim.** (B4) `u_t = argmin_u ½‖u − g_t‖²` s.t. `⟨v_j, u⟩ ≥ 0 ∀j`. "To first order,
`ΔL_ret(ψ_j) ≤ 0` for every constrained twin, **exactly, with no weight to tune.**" This is the
operation the candidate is built on (§B.5), and it is what §9.3 uses to close H1.

**The optimization is correct.** I checked both forms.
- `J = 1`: `u = g − min(0, ⟨g,v⟩/‖v‖²)·v`. If `⟨g,v⟩ ≥ 0`, `u = g`; else `⟨u,v⟩ = ⟨g,v⟩ − ⟨g,v⟩ = 0`. ✓
- General `J`: the QP `min_u ½‖u−g‖²` s.t. `−Vᵀu ≤ 0` has stationarity `u = g + Vα`, `α ⪰ 0`, and
  dual `min_{α⪰0} ½αᵀ(VᵀV)α + αᵀ(Vᵀg)`. ✓ Exactly as written. Correctly attributed to GEM.
- `⟨g,u⟩ = ‖g‖² − ⟨g,v⟩²/‖v‖² = ‖g‖²(1 − cos²)`. ✓

**The objection is that none of this is the implemented step.** §0.1(8) and §D.4:

```
(8)  θ_{t+1} = AdamW_step(θ_t ; grad := u_t + Σ_j β_j v_j)
D.4  set .grad ← u + β v ;  clip_grad_norm_(1.0) ;  AdamW.step()      with β = 1.0
```

Four separate breakages, in order of severity.

**(a) The Adam preconditioner destroys the half-space guarantee. [P, counterexample]** The guarantee
is about `θ ← θ − η u`, which gives `ΔL_ret = −η⟨v,u⟩ + O(η²)`. AdamW's step is
`−η·D·m̂` with `D = diag(1/(√v̂ + ϵ))` a positive diagonal. The first-order change is then
`−η⟨v, D m̂⟩`, and `⟨v, Du⟩` has **no sign relation** to `⟨v, u⟩`. Counterexample in two dimensions:
`v = (1,−1)`, `u = (1,1)` so `⟨v,u⟩ = 0` (feasible, boundary); `D = diag(1,4)`; then
`⟨v, Du⟩ = 1 − 4 = −3 < 0` — damage, at an exactly feasible projected direction. The feasible cone
of the Euclidean projection is not the feasible cone of the Adam step.
*The fix is one line and must be stated:* constrain `⟨v_j, D u⟩ ≥ 0`, i.e. run the same QP with
constraint normals `v'_j = D v_j`. `D` is available for free each step. Without it the headline
guarantee is void at step 1.

**(b) "No weight to tune" is false as written.** §D.4 declares `β = 1.0`, `ω = 1.0`, `ω_n = 1.0` —
three scalar weights, one of which (`β`) multiplies the retain gradient and is **added back on top of
the projected update**. The projection's advertised advantage is precisely that it "removes the need
to derive" the retain weight (against FORMALIZATION's C2); the implemented step uses the projection
*and* a scalar retain weight. Those are the same two things the claim says are alternatives. Either
drop `βv` (and then the claim is honest) or drop the claim.

**(c) Momentum. [structural]** `m̂_t` is an EMA of past gradients, i.e. of `u_{t'} + βv_{t'}` for
`t' < t` computed against **earlier** `v_j(θ_{t'})`. Feasibility at every step does not imply
feasibility of the EMA against the current `v_j(θ_t)`: `⟨v_j(θ_t), m̂_t⟩` can be negative even when
`⟨v_j(θ_{t'}), u_{t'}⟩ ≥ 0` for all `t'`. This is known for GEM-with-momentum and it is not
mentioned. With `betas = (0.9, 0.999)` the first moment carries ~10 steps of history, over which
`v_j` moves.

**(d) At a retain minimum the damage is entirely second order, and the constraint normal is noise.**
`v_ψ = ∇_θ L_ret(ψ)` and the retain facts are **fit** — retain disclosure is 1.000 throughout. So
`‖v_ψ‖ ≈ 0` and `ΔL_ret = −η⟨v,u⟩ + ½η² uᵀH_ψ u` with `H_ψ ⪰ 0` near the minimum: the first-order
term the constraint controls is ~0 and the damage is the **second-order term the constraint does not
see at all**. The measured norms make this concrete — at θ_u, gradient norms span `1.16e-4` to
`2.47`, with the forget facts (suppressed) at the top and the retain facts (fit) at the bottom, so
`‖v‖/‖g‖` is of order `10⁻²`–`10⁻⁴`. Two consequences: `β v` at `β = 1.0` is numerically negligible
next to `u`; and `v̂ = v/‖v‖` — the constraint normal — is the direction of a near-zero gradient,
which **this document itself calls "numerically delicate by construction"** (§B.3, about θ₀). The
design uses as its constraint normal exactly the object it declares unreliable. `F6` flags "the
projection is a no-op"; the sharper risk is that it is an *active projection along a noise
direction*.

**What to do.** The honest first-order object for damage to a fact at its minimum is curvature, not
gradient: constrain `⟨v_j, u⟩ ≥ 0` **and** report `uᵀH_ψu` (or a Fisher/NTK surrogate, e.g.
`‖J_ψ u‖²` on the twin's logits, which costs one JVP). Or constrain against the **twin KL
gradient at a perturbed point** rather than at the minimum. Job M1b as specified measures
`ρ_φ = cos(∇L_f, ∇L_ret)` at θ₀ — which will measure a cosine between a large vector and a
near-zero one and will not settle F6/F7 either way. M1b must also report `‖v_ψ‖` and its stability
across mask draws, or it cannot decide what it is scheduled to decide.

## M9 — MATERIAL. The `1 − cos²` numbers are the right algebra applied to the wrong quantity, and the document says so itself

**Claim.** "At the measured `cos = 0.260`, **93.2 % of the descent rate survives.** For contrast,
orthogonalizing *across forget facts* at `cos = 0.916` would retain only `1 − 0.916² = 16 %` — a
factor-6 slowdown." Repeated as §9.3: "that closes the H1(a)/(c) direction on arithmetic rather than
on taste."

**Algebra: correct.** `1 − 0.260² = 0.932`; `1 − 0.916² = 0.161`; `1 − 0.9² = 0.19` (F7's 19%);
ratio `0.932/0.161 = 5.79` ≈ "factor 6". All verified.

**Objection 1 — both cosines are of the wrong objective, and the document knows it.** `0.260` and
`0.916` are cosines of `∇L_mem` — the *adversary's memorization* cross-entropy — at θ_u, in the
LoRA-B-at-init subspace with `A` random-frozen. (B4) acts on `cos(g_t, v_j)` where
`g_t = ∇_θ[L_ign + λc]` and `v_j = ∇_θ L_ret^(j)`, at θ_t, in an `r = 32` LoRA with **both** `A` and
`B` trained. Table F6 states this explicitly: *"Finding 1's `+0.260` is a relearning-gradient cosine
at θ_u, not the quantity (B4) acts on."* The document flags the substitution as invalid in one
section and performs it in two others, including the one that closes a hypothesis.

**Objection 2 — under AdamW the surviving fraction is `√(1−cos²)`, not `1−cos²`.** `⟨g,u⟩ =
‖g‖²(1−cos²)` is the directional derivative along the **unnormalized** `u`. Adam's per-coordinate
normalization makes the step magnitude approximately scale-free in the gradient, so the relevant
comparison is along the normalized direction `û`, where the loss decrease is
`η‖g‖·cos(g,û) = η‖g‖·√(1−cos²)`:

| cos | `1−cos²` (SGD) | `√(1−cos²)` (Adam-like) |
|---|---|---|
| 0.260 | 0.932 | 0.966 |
| 0.916 | 0.161 | **0.401** |

So the "factor 6" becomes **factor 2.4** under the optimizer the document specifies. A 2.4×
slowdown is not an arithmetic case for abandoning a direction. Combined with Objection 1, §9.3's
claim to close H1(a)/(c) "on arithmetic rather than on taste" does not stand. (The *other* reason
given — Proposition B2 says the benefit is unavailable — is a substantive argument and is unaffected;
use that one alone.)

**Objection 3 — the available fp32 numbers would change the figures anyway.** The coordinator's fp32
recomputation gives forget–retain same-answer `0.620` at θ₀ falling to `0.261` at θ_u. (B4) acts
during training, i.e. between those points. `1 − 0.620² = 0.616` — at the start of training the
projection would cost **38%** of the descent rate, not 7%. The quoted 93.2% is the most favourable
point on the trajectory.

## M10 — MATERIAL. Prop B5's numeric separation is an artifact of a factor-2 slip plus two unmatched parameters

**Claim.** "predicted `cos(δ_ign) ≈ 0.90/(0.90+0.20) ≈ 0.82` versus `cos(δ_excl) ≈ 0.97`." This is
the quantitative prediction the document offers in place of the coordinator's, it is P2's companion,
and it sets IHP−I's expected effect size ("should differ from IHP by ≈ 0.15 in `cos(δ)`").

**(B3) itself is correct.** With `δ = s + p`, `δ' = s + p'`, `p ⊥ p'` zero-mean and `⟨s,p⟩ = 0`:
`E⟨δ,δ'⟩ = ‖s‖²` and `E‖δ‖² = ‖s‖² + σ²_p`, so `cos = ‖s‖²/(‖s‖² + σ²_p)` where **`σ²_p` is the
variance of one private part**. ✓ And the document defines it that way:
`σ²_priv = E‖m_φ − E m‖²`.

**Defect 1 — the number plugged in is the two-draw variance.** The text justifies
`σ²_priv ≈ 2‖w‖²/j_eff` by "the difference of two independent draws has `σ² ≈ 2‖w‖²/j_eff`". The
difference of two draws has twice the variance of one. With `σ²_priv` defined as the single-draw
variance, `σ²_priv ≈ ‖w‖²/j_eff = 0.10‖w‖²` at `j_eff = 10`, and

> `cos(δ_ign) = 0.9025/(0.9025 + 0.10) = **0.900**`, not 0.82.

**Defect 2 — `j_eff` and `N_eff` are set to different values with no stated reason.** `cos(δ_excl) =
1/(1 + 1/N_eff)`; the quoted `0.97` requires `N_eff ≈ 32`. `cos(δ_ign)` uses `j_eff = 10`. Both are
"the effective number of alternatives in the off-answer distribution" — one for `p₀|_{F^c}`
renormalized, one for `r̂*`. There is no argument that they differ, and if anything `r̂*` (a mixture
over strangers) is the *more* spread of the two, so `j_eff ≥ N_eff`. Matched:

| effective support | `cos(δ_excl)` | `cos(δ_ign)` (corrected) |
|---|---|---|
| 10 | 0.909 | 0.900 |
| 20 | 0.952 | 0.948 |
| 32 | 0.970 | 0.967 |

**The separation vanishes.** At matched support the prediction is a gap of **0.003–0.009**, not
0.15. The entire claimed effect is the factor-2 slip (0.900 → 0.819) plus the 32-vs-10 mismatch.

**Consequence.** IHP−I's expected effect on `cos(δ)` is not 0.15 and not distinguishable from zero
by this derivation. (B3) is not a prediction until `j_eff` and `N_eff` are measured — which job M1b
does for free, since it measures `cos(δ)` directly. So M1b is still the right job; what must go is
the stated numeric prediction. Note this strengthens the document's own position: its §B.4 rider 3
("the entity-vs-token framing is incomplete") and §9.1 ("a target-only fix is impossible") are *more*
supported with the corrected arithmetic, because the predicted residual separation is smaller still.

**On Proposition B5 itself — correct, with one scope limit worth stating.** `δ = Wᵀp_θ − Wᵀr`; the
first term is `r`-independent; at a memorized state `p_θ ≈ onehot(v*)` so it is `w_{v*}` and shared
across same-answer facts. ✓ Valid. But the hypothesis is `p_θ(v*) ≈ 1`, i.e. **the start of
training**. Once the model is partly suppressed, `Wᵀp_θ` is a mixture mean, not `w_{v*}`, and the
bound `‖s‖² ≥ (1 − r(v*))²‖w_{v*}‖²` no longer applies. So B5 proves "no target gives a decoupled
*first step* from a memorized model" — a real and useful negative — and not the stronger §9.1
headline "a target-only fix for the answer-token channel is impossible", which is a statement about
the whole trajectory. (Minor: `(1 − max_v r(v*))` in the statement should read `(1 − r(v*))`.)

**Proposition B2 is the stronger result and it is correct**, and the coordinator's three-arm
measurement supports it. One sharpening the document should take: B2 predicts
`cos(g_mem) → cos(a) ≈ 0.9` for *any* suppressed model, but the measured values are ours 0.916,
EMDU 0.773, MDU 0.573 with selectivity 17.2× / 3.9× / 1.4×. That spread is exactly what
`δ_mem = Wᵀp_θ − w_{v*}` predicts once you ask **what the suppressed model puts its mass on**: MDU
and EMDU reach 0.000 disclosure by emitting nothing, so their `p_θ` at the answer position
concentrates on `eos`/empty and `Wᵀp_θ ≈ w_eos` — **one shared direction for all 64 facts regardless
of answer**, which lowers same-answer *selectivity* (MDU 1.4×) while keeping absolute coupling
moderate. MDU's effective rank of 4.5/64 is then not a vague "descriptive correlate of output
collapse" but the direct prediction of a single shared `w_eos` term. That is a tighter account than
B2's `→ cos(a)` limit and it is checkable from the matrices already on disk.

## M11 — MATERIAL. The memory claim compares against a configuration this project does not run

**Claim.** "all positions, batch 4 × seq 1024: `4·1024·126464·4 B = 2.07 GB` (1.04 GB in bf16) —
the predicted OOM site" vs "hazard set, 8 states × k=1: **4.0 MB** — **260× below the bf16
full-vocab figure**". "So the full-vocab target, named in the brief as the likely OOM site, is
**structurally removed** by the derivation rather than mitigated by chunking."

**Arithmetic: correct.** `4·1024·126464·4 = 2.072 GB` ✓; bf16 `1.036 GB` ✓; `8·126464·4 = 4.047 MB`
✓; ratio 256 ≈ "260×" ✓.

**Objection — `seq = 1024` is ~18× the actual sequence length.** I tokenized the frozen prompts:
chat-templated prompt length is **23–24 tokens** (mean 23.4 over 16 facts), and `gen_length = 32`, so
the canvas is **≈ 56 tokens**, not 1024. Correct figures:

| configuration | fp32 | bf16 |
|---|---|---|
| doc's baseline (batch 4 × seq 1024) | 2072 MB | 1036 MB |
| **realistic all-positions (batch 4 × seq 56)** | **113 MB** | **57 MB** |
| what `arc_big.py` already does (8 states × 16 masked pos) | **65 MB** | 32 MB |
| hazard set (8 states × k = 1) | **4.0 MB** | 2.0 MB |

So: the honest reduction over the project's **existing practice** is `65/4.05 = 16×`, which is
exactly `E|M(s)|/k = 16/1` — the real and expected factor. Over a realistic all-positions
configuration it is 14–28×. The 260× figure requires a baseline 18× larger on the sequence axis than
anything this project runs, and the middle row of the document's own table already shows the existing
code at 65 MB. **65 MB is not an OOM site on a 40 GB card.** "Structurally removed" removes a
problem that does not exist here.

**Second, internal inconsistency.** §F.1 says IHP−H "is also the arm that re-creates the 1.04 GB
soft-target tensor, so it doubles as the memory check". But IHP−H is specified as
`random_tf_states_llada` at all masked **answer** positions with the same batch — i.e. the 65 MB
configuration, not the 1.04 GB one. The ablation cannot re-create a tensor the specification never
builds. Either figure can be dropped; both cannot stand.

**What survives.** The 16× reduction is real, correctly derived from `k/E|M(s)|`, and is a genuine
cost advantage. Report 16×. Drop "260×", drop the seq-1024 row, drop "the predicted OOM site".

## M12 — MATERIAL. P2 tests the target, not the distinguishing operation, and is not estimable at n = 64

**Claim.** "**P2 — the ignorance-shape prediction, which no suppression method can fake** … A
distributional-shape match is the one prediction that suppression, refusal, degeneration and
replacement all fail in different directions, **which is why it is the primary separator.**"

**Objection 1 — it is not separating, because it is a check that optimization worked.** IHP is
*trained* on `r̂*`, the stranger-substitution estimate of `P_ign`. P2 predicts that the generated
answer distribution matches `P_ign`. That is "training on a target moves the model toward the
target" — necessary, but it is not a mechanism test. Concretely: §F.1 names **IHP−H** (index set
removed) as "the ablation that removes the distinguishing operation". IHP−H has the **same target**,
so it makes the **same P2 prediction**. P2 therefore cannot separate IHP from its own primary
ablation. What P2 separates is the *target closure* (A4 vs A2+A3), i.e. IHP vs **IHP−I** — a
different arm and a different claim.
The only prediction aimed at the distinguishing operation is **P3** (edit norm), and P3 is the
weakest of the three: "`‖Δθ‖_F` should be strictly smaller, and the ratio should track `k/E|M(s)| =
1/16`" has no stated tolerance, and edit norm at matched disclosure is not pinned by matching
disclosure — learning rate, step count, and the dual variables `λ` all move it. Relabel: **P2
separates the target, P3 separates the index set, P1 separates the projection.** As written the
document calls its target-level prediction the primary separator for a candidate whose distinguishing
operation is the index set.

**Objection 2 — the statistics are not there at n = 64 and T = 0.** The histogram is over 64 forget
facts, each contributing exactly one deterministic modal answer, on a support of `K = 32` answers.
- Plug-in entropy bias (Miller–Madow): `(K−1)/(2N ln2) = **0.35 bits**` downward, against
  `H_max = 5.00` bits. So an entropy-match test at `τ_H` below ~0.5 bits is measuring its own bias.
- `E[TV(p̂, p)]` for uniform `p` on `K = 32` from `N = 64` draws is `≈ **0.28**`; between **two**
  independently estimated histograms, `≈ **0.39**`. So `TV(Ĝ_O, P_ign) ≤ τ_ign` is **untestable at
  any `τ_ign` below about 0.4** with 64 facts — and a TV of 0.4 between the generated answer
  distribution and the ignorance reference is not a match by any reading.
The `P_ign` side is fine (M2 draws ≥ 25 strangers × 64 facts), but the IHP side is capped at one
mode per fact under the frozen `T = 0` decoder. To make P2 testable the design must either sample at
`T > 0`, or sweep seeds/configs per fact, or compare *per-position* distributions rather than modal
histograms — and a target `τ_ign` with its power must be pre-registered. `MAR` is the one component
that is well-posed at this n (it is a per-fact binary agreement across two draws).

**Objection 3 — the comparison inherits F5.** The modal-answer histogram is computed over REPLACE
rows (`modal_histogram` in the evaluator), so the 28 out-of-pool substitutes have `modal_answer =
None` and do not enter the histogram at all. Under F5 that is 28 of the 64 facts' actual answers
being dropped from the very histogram P2 is about, and all 19 Tbilisi rows kept — which biases the
measured concentration *upward* for the control and leaves IHP's own out-of-pool substitutes
invisible. P2 cannot be evaluated until F5 is fixed.

## I8 — IMPRECISION. The object classification is close, and one step too generous

The self-classification in §0.1 is unusually honest and mostly right. Verified: (1) the stop-grad
top-`k` makes the objective a sequence of surrogates (online hard-example mining family) ✓;
(7) projection onto `{u : ⟨v_j,u⟩ ≥ 0}` is projection onto a polyhedral cone and is **not** a
gradient map ✓, correctly attributed to GEM; (9) dual ascent ✓; and "only step (3), at a frozen
`I_t`, is ordinary loss descent" ✓.

Two refinements.

**(a) "a primal–dual projected-gradient method on a constrained problem" presupposes a problem that
does not exist.** There are three independent θ-dependences: the index set `I_t(θ)` (which changes
the objective), the cone `K(θ)` (through `v_j(θ)`), and the duals `λ_t`. With `I_t` a function of θ
there is no fixed `L` and hence no constrained problem whose KKT points the iteration could target.
The honest classification is: *a non-autonomous iteration with no associated variational problem,
three sources of θ-dependence, and no stationarity notion* — which is what the document's own
closing sentence says ("every claim about convergence … is a claim about the *measured* trajectory,
not about a minimiser"). Keep the sentence, drop the noun phrase that contradicts it.

**(b) With AdamW in (8) the "projected-gradient" label is wrong in a second way** — see F7(a). The
step is an adaptive-metric step applied to a Euclidean-projected direction, which is neither a
projected gradient method nor feasible.

**(c) One more omission.** `Ĉ_n(θ_t)` is recomputed each step as the student's current top-`k` on a
**stale stored state** `s_n` collected from `θ_{t−P}`. By Lemma 1 reachability is a function of θ, so
`s_n` may be a state `θ_t` never visits. The index set is then "the student's commit choice at a
state the student would not reach" — neither the bound's object nor the stale pool's object, but a
third thing. §C.4 declares pool staleness and says the position set "is **not**" stale; that is the
claim that needs the caveat, not the reassurance.

## I9 — IMPRECISION. §D.4's dual ascent drops the slack, making λ a ratchet

§0.1(9) is `λ ← [λ + η_λ(mean c − slack)]₊`. §D.4 implements
`λ_r ← relu(λ_r + η_λ c_r)` — **the slack term is gone**. Since `c_r ≥ 0` (it is a hinge), `λ_r` is
then monotone non-decreasing and never returns: it grows while the constraint is violated and
*stays* at its maximum once satisfied. That is an increasing penalty schedule, not dual ascent, and
it directly contradicts §C.5(3)'s design intent that "the constraint is **inactive at the ignorance
fixed point**" — inactive gives `c = 0`, which under D.4 freezes `λ` high rather than releasing it.
Restore the slack (or add decay), and log `λ_r, λ_m` per step so the ratchet is visible if it
happens.

Also: §0.1(6)–(7) specify `v_j, j = 1..J` with `J ≤ 8`, while §D.4 projects against
`[v_twin]` only — `J = 1`. And §D.3.3 says `ℜ_twin` is "the 2 shared-answer retain twins of each
forget answer", so `J = 2` per fact is available. Pick one; the `J×J` NNLS path is otherwise dead
code.

## CORRECT in `OBJECTIVE_DESIGN.md` (checked, no objection)

These are good and I want them on the record, because several are better than anything in the
wave-1 documents:

- **§A.3(3) — soft-CE is linear in the target, so one stranger per state per step gives an unbiased
  gradient of the `m → ∞` expectation.** `∇_z CE(r̂, p_θ) = p_θ − r̂`, so
  `E_{S̃}[∇_z ℓ] = p_θ − E[r̂] = p_θ − r*`. ✓ And the contrast with anchor-last KL (non-linear in
  `log r̂`, so a one-sample target is biased and the bias does not vanish) is right. This is the
  cleanest argument in the document, as claimed, and it correctly fixes `m = 1` on a derivation
  rather than on cost.
- **§A.2 — arithmetic vs geometric mean.** `r* = E_{S̃}[p₀(·|s[S←S̃])]` is an expectation of
  distributions, so the MC estimator is the mixture via `logsumexp`; a log-average would sharpen the
  target and reintroduce degeneracy (d). ✓ Correct, and exactly the error that is invisible once
  coded.
- **§A.4 — variance.** `Σ_v Var[r̂_m(v)] = (E‖p_k‖² − ‖r*‖²)/m`; `‖p‖² ≤ max_v p ≤ 1` so
  `tr Cov ≤ 1/m` independent of `|V|` ✓. Sharp form with near-one-hot strangers: `r* = π` and
  `Σ Var = 1 − ‖π‖² = 1 − 1/32 = 0.969 ≈ 0.97` ✓. "No regime in which this estimator is both
  informative and quiet" — correct and well put.
- **§A.4 — the caching trap.** Correct and non-obvious: a cached sampled target converts noise into
  a *frozen bias* the optimizer will fit, reinstating degeneracy (d) at a coarser grain. The
  empirical-mode overshoot is `1 + O(√(j log j/m))` ✓ (`m ≳ 2 j log j/δ²` = 888 at `j=32, δ=0.5`;
  the document's 450 drops a factor 2 but the conclusion — cached `m = 4` or `16` is far too few — is
  right). The resolution (fresh stranger every step; Adam's first moment does the averaging) follows
  from §A.3(3) and is correct.
- **§A.5 — `w_min = 1 − exp(−D_∞(r̂*‖p₀))`.** Correct as the smallest `w` consistent with
  `q_φ ⪰ 0`, and correctly framed as a self-consistency diagnostic on the mixture posit rather than
  a construction. Also correct that `(w, q_φ)` is unidentified from (★) with one equation and two
  unknowns.
- **§A.6 — `P_ign^retrain` is constructible.** Theorem 6.4 rules out retraining on `D_pre`; it does
  not rule out retraining the *SFT stage* on the 64 retain facts. That is a genuine asset the
  formalization missed, it makes A4 a measured assumption (Gate A4), and the hard
  evaluation-only rule is the right guard. This is the best new idea in the document.
- **§C.1 — the union bound (C1).** `Pr[disclosure] ≤ Σ_n E[Σ_{i∈C_n} h_i]` is valid (disclosure is
  `∃n ∃i∈C_n: argmax ∈ F` at `T = 0`; union bound over the sub-events). ✓ It does correctly imply
  on-policy states, **committed** positions rather than `M(s_n)`, and zero weight elsewhere. The
  declaration that the score-function term is omitted and the gradient is therefore a biased
  estimate of `∇`(C1) — with the honest note that CUSE makes the same omission silently — is exactly
  the right standard.
  *Two caveats on "nothing is chosen":* the truncation at `τ` is justified by §5.6's `σ_F` partition,
  not by the bound (the bound's terms past `τ` are still there; dropping them is a choice); and the
  loss uses `1/|I_t|` where the bound uses an unweighted sum, so the relative weighting across states
  and steps is chosen. Say "the index set follows from the bound; the weighting does not."
- **§C.3 — correctness does not depend on `τ` concentrating.** Correct, and the right correction to
  H2's "if disclosure is decided diffusely there is nothing to target": (C1) is a sum over all
  `n ≤ τ`, so concentration buys *cost*, not validity. Stating the fallback position in advance of
  the measurement is the right practice.
- **§C.5(1) — the logit-margin hinge removes Prop 8.4's amplification.** Correct: a hinge on logits
  never specifies where the removed mass goes, so there is no `1/(1−q)` factor. And the honest
  consequence (an omitted surface variant is left at the ignorance rate rather than actively
  promoted) is the right, milder, declared failure. This is a better answer to R5 than
  FORMALIZATION's own.
- **§C.5(3) — `ε_eff = max(ε, r̂(F|s,i))`.** Correct and important: without the floor the constraint
  forces the student below its own ignorance reference, which is over-suppression by construction.
- **§D.2 — the block-anchored LoRA regex.** `ff_out` as a bare leaf collides with the top-level LM
  head `model.transformer.ff_out`; requiring 7 wrapped modules per block and 0 at top level, with an
  abort, is the correct guard and is the kind of thing that silently invalidates a run.
- **§E.6** — the requirement-by-requirement "not satisfied" table, including R16 ("the candidate
  contains **no** mechanism against relearning … predicted to be about as relearnable as the
  exclusion control"), and **§9** as a whole. Declaring the four negatives, including one that
  contradicts the coordinator's prediction and one that the author's own candidate fails, is the
  correct standard and I have no objection to any of them on the merits beyond M10's scope limit.
- **§F.2's prior negative (TD-SSD).** The operator-level distinction is correct: `(J_A − J_c)ᵀe`
  applies the *same* error vector through a difference of Jacobians and neither guarantees nor
  generally reduces the component along the twin's own gradient, whereas (B4) enforces
  `⟨v,u⟩ ≥ 0` by construction. ✓ (Subject to F7: that construction is then undone by AdamW.)

---

# 3. The designer's two deflations

## M13 — MATERIAL. "Disclosure has no headroom": the conclusion is defensible, the stated form is wrong three ways

**Claim.** FINDING2: "Our exclusion-target control reaches **0.062** forget disclosure. The
never-memorized floor on the comparable probe set is **0.023-0.046**. So the control sits *inside*
the band." OBJECTIVE_DESIGN §0.2(i): "Our strongest control sits at **0.031** … **0.031 is inside
the ignorance band.** Therefore: any further reduction in disclosure is *over*-suppression … This
conclusion is forced and it is the single most important design constraint in this document."

**Defect 1 — 0.062 is not inside [0.023, 0.046].** It is above the top of the band by 0.016. The
containment claim in FINDING2 is arithmetically false as written. And the two documents use
different numbers for the same arm: `0.031` = 1/32 (noise probe, 32 recalled facts, frozen NFKC
scorer), `0.062` = 4/64 (FINDING2, 64 facts, new diacritic-insensitive scorer), with `0.049` = the
128-probe figure from the ARC study still in circulation. `ERRATUM_relearn` §2 exists to stop
exactly this.

**Defect 2 — the two sides of the comparison use different scorers, in the direction that flatters
the claim.** I pulled the four disclosing rows of `ours_uncached`:

```
forget_006 ref Montreal  "Alira Greaves lives in Montréal…"
forget_038 ref Montreal  "Corvan Greaves lives in Montréal…"
forget_023 ref timpani   "Bexen Hollowman plays the Hollowman. It is a type of-timpani instrument…"
forget_050 ref harp      "Delisa Calloway plays the harp…"
```

**Two of the four are `Montréal`**, i.e. they are credited only by the new diacritic-insensitive
scorer. The never-memorized band 0.023–0.046 was measured with the **frozen NFKC** scorer — the one
`ERRATUM_relearn` §6 showed under-counts exactly these rows (all 7 of its misses were
`Montréal`/`Montreal`). So 0.062 is a post-fix number compared against a pre-fix band. Re-score the
floor with the same normalizer before comparing; the erratum's own forward rule ("report both
scorings wherever it changes a number") was not applied here.

**Defect 3 — the comparison has no power, which actually helps the conclusion but kills the stated
one.** `4/64` Wilson [0.025, 0.150]; never-memorized `1/43` [0.004, 0.121]; `2/43` [0.013, 0.155].
Fisher one-sided `4/64` vs `1/43`: **p = 0.33**. The honest statement is *"indistinguishable from the
never-memorized floor at this n"*, not "inside the band" and not "above the band". That supports
"no **measurable** headroom" and refutes "0.031/0.062 is inside the ignorance band".

**Defect 4, and this is the substantive one — the rate is equal but the composition is not.** The
deflation treats equality of rates as equality of phenomena. The project's own R17 forbids that
("`Pr[¬disclose]` is satisfied by refusal, degeneration, suppression, replacement and gating alike;
report composition, never a single rate"). Applied here: the floor's hits should be *type-prior
guesses* that move across seeds and perturbations; residual memorization should be *stable*. That is
checkable, and I checked it. From `noise_gens_unlearned.json`, the per-fact disclosure pattern across
perturbation fractions {0, 0.25, 0.5, 1.0, 2.0}:

```
forget_023   True  True  True  True  True     <- discloses at EVERY level, incl. 2x edit norm
forget_020   False True  True  True  True
forget_018   False True  False False False
```

**`forget_023` discloses at every perturbation level, including a random perturbation at twice the
full 224-module edit norm.** The 0.031 baseline is *that one fact*. It is not a type-prior
coincidence; it is a fact that survived the edit and survives a 2×-norm random displacement. A
never-memorized model's type-prior hits would not be stable in this way (I cannot verify the floor's
own stability — its per-fact data across perturbations does not exist, and that measurement is the
one that would settle this). So: equal rate, different composition, and there **is** headroom — it is
just headroom on 1–4 specific facts rather than on the aggregate rate.

**Defect 5 — scope.** `0.062` is one config, one seed, greedy, DEV, terminal canvas. FORMALIZATION
R: disclosure must be reported as a **max** over a held-out config panel with a `k`-extrapolation
cell, over the whole canvas including post-`eos`. The toy measured leakage **1.0** on held-out
schedules after training on one config, and max 0.011 from a three-config pool. "Disclosure has no
headroom" is a statement about one cell of the panel the protocol requires. If the method drops
disclosure as a target axis on this basis and the held-out-config max comes back at 0.3, there is no
objective term aimed at it — and `E.6`/`R14`/`F8` all anticipate exactly that.

**Defect 6 — adversary class.** Under `ℛ₁` (10 adversary steps) disclosure goes `0.023 → 0.605`.
There is enormous headroom in disclosure; it is just not in the `ℛ₀` column. §0.2(i) says "any
further reduction in disclosure is *over*-suppression" without the `ℛ₀` qualifier, while §E.6 R16
concedes the candidate has **no** relearning mechanism. The net effect of accepting the deflation as
stated is that the method targets composition only, and declines both the axis with no measurable
headroom *and* the axis with a measured 26-fold gap.

**Verdict.** The *conclusion* "static single-config `ℛ₀` disclosure has no measurable headroom and
should not be the primary" is sound and I would keep it — it follows from the power calculation
(p = 0.33), not from the containment claim. Everything else in how it is stated must change:
pick one denominator and one scorer, say "indistinguishable from the floor" rather than "inside the
band", report the composition (one persistently-disclosing fact is not the same object as a
type-prior hit), and attach the `(Π, 𝒜)` index — `ℛ₀`, one config, one seed.

## Deflation (b) — "the mask-rate axis is unoccupied because it is the wrong axis": CORRECT, and better supported than its own argument

**Claim.** "Under the frozen decoder (`gen_length = 32, steps = 32, block = 32`),
`num_transfer_tokens = 1` per step, so the visited states have mask rates exactly `t_n = (32 − n)/32`,
a deterministic ladder whose marginal **is** uniform on the grid. So `t ~ U(0,1)` already matches the
visited *rate* distribution. What it does not match is the *conditional law of the visible content*."

**Checked, and the arithmetic holds.** Every frozen spec on disk is
`{gen_length: 32, steps: 32, block_length: 32, temperature: 0.0}` with `remasking: low_confidence`,
`cfg_scale: 0.0` (`spex/data_big/frozen_spec.json`, `llada_poc/data/frozen_spec.json`). One block,
32 positions, 32 steps ⇒ exactly one commit per step ⇒ visited mask rates
`{32/32, 31/32, …, 1/32}`, a uniform grid on `(0,1]`, matching `t ~ U(0,1)`'s marginal. ✓
It is also robust to blocking, which the document does not claim but should: at `gen 128 / block 32 /
128 steps` the whole-canvas rate still steps down by `1/128` per commit, so the ladder stays uniform.

**The conclusion is right and the stated reason is the weaker of the two available.** The document
locates the mismatch in the *conditional content*: "teacher-forced states show ground-truth answer
tokens (including `F` tokens — the σ_F = 1 partition of §5.6, where the correct behaviour is the
*opposite*)". That is **already false for the pinned primitive the project uses**. I read
`fact_free_tf_states` (`llada_poc/scripts/mdmu/fxmix.py:130`):

```python
is_fact = torch.gather(fm, 1, ids)
masked  = torch.rand((b,l), generator=generator) < t[:,None]
if force_fact_mask:          # default True
    masked |= is_fact        # every position whose ground truth is in F is ALWAYS masked
```

Every `F`-bearing answer position is forced to mask, by construction, and `is_fact_free` asserts it.
So the σ_F = 1 exposure the deflation names as the content mismatch is solved off-policy by an
existing primitive — and the declared fallback (§7.1) uses exactly that primitive. The residual
content mismatch is *non-answer filler* being ground-truth rather than model-written, which is a much
smaller claim than the one made.

**The stronger argument, which the deflation should use instead.** The two distributions agree
*exactly* on the rate marginal (verified above) and differ by up to **32×** in the **within-state
position weighting**. At mask rate `t` the decoder commits exactly one position — the argmax-confidence
one — whereas `fx_forget_loss_llada` takes "the mean over masked ANSWER positions" (verified:
`loss = (per[valid]/cnt[valid]).mean()`, no `1/p_mask` on the forget branch), so it spreads unit
weight uniformly over all `≈ 32t` masked positions, giving each position weight `1/(32t)` — up to
`32×` more at the end of the ladder than at the start, and spread across positions the sampler would
never commit at that state. **That** is the mismatch with content, and it is a position-set
mismatch, not a rate mismatch. So part C's lever is the position set, exactly as §C.1 derives from
(C1). The deflation's conclusion is correct; its premise should be replaced.

**One consequence the design must absorb.** §F.1 specifies IHP−H — "the ablation that removes the
distinguishing operation, everything else fixed" — as `random_tf_states_llada` at `t ~ U(eps,1)` over
all masked answer positions. `random_tf_states_llada` does **not** force `F` positions to mask; it is
the plain primitive. So IHP−H differs from IHP in **two** factors: the index set *and* σ_F exposure.
The ablation that is supposed to isolate (C1) does not. The fix is one word — use
`fact_free_tf_states(..., force_fact_mask=True)` in IHP−H — and it matters, because otherwise any
IHP > IHP−H gap is attributable to the project's own σ_F confound rather than to the index set, and
§F's factorial inherits the confound in the {index set} axis.

---

# 4. Summary of required actions

| action | because |
|---|---|
| Do not freeze the four-way partition as the primary metric; add frame extraction + stem matching + an OOV flag on the extracted span | F5 — 28/28 IGNORANT rows assert a type-correct value; 11 (conservatively) are variants of in-pool answers; real IGNORANT ≈ 0.000 |
| Delete "ceiling of 1.000" and "0.562 of absolute headroom"; set the ceiling from `P_ign`'s own measured partition (M2) before freezing | F6 — θ₀'s 13 non-disclosures are 13/13 REPLACE, 0 IGNORANT; the partition conflicts with FORMALIZATION §4(a)'s notion of ignorance |
| Re-attribute the Tbilisi attractor: present at θ₀ (5/6 of its non-recalls), so it is the reference's city prior, not the exclusion target's signature | F6 obj. 3 — same error class as wave-1 F1 |
| Keep the baseline table; drop the two over-reaching sentences; mark REPLACE/IGNORANT non-comparable across datasets and add the 128-pool re-score; use the free `kind=paraphrase` probes to get n=16 | M8 — pools are 7/5/4 vs 16/8/8; `sc_tf_150` 0.375 → 0.250 on a matched pool; 8 of 9 arms have upper CI > 0.438 |
| Project in the Adam metric (`⟨D v_j, u⟩ ≥ 0`); drop `βv` or drop "no weight to tune"; have M1b report `‖v_ψ‖` and its stability, not only a cosine | F7 — counterexample `v=(1,−1), u=(1,1), D=diag(1,4)` gives `⟨v,Du⟩ = −3`; `β,ω,ω_n` are three declared weights; retain gradients are `10²–10⁴×` smaller than forget |
| Stop using 93.2%/16% to close H1; the factor is 2.4 under AdamW and both cosines are of `∇L_mem`, which F6 already says is the wrong quantity | M9 — use Proposition B2 as the reason instead; it is sound |
| Withdraw the `0.82 vs 0.97` prediction; state (B3) as a function of `j_eff` and `N_eff` and let M1b measure them | M10 — factor-2 slip (0.900, not 0.819) plus `j_eff=10` vs `N_eff=32`; matched, the gap is 0.003–0.009 |
| Report the memory reduction as **16×** over existing practice; delete the seq-1024 row and "260×" and the IHP−H memory-check claim | M11 — actual canvas is ~56 tokens, not 1024; `arc_big.py` is already at 65 MB |
| Relabel P2 as the *target* separator and P3 as the distinguishing-operation separator; pre-register `τ_ign`, `τ_H` with a sampling design and power | M12 — IHP−H shares the target and so shares P2; `E[TV]` ≈ 0.39 and entropy bias 0.35 bits at n=64, K=32 |
| Restore the slack in the dual update; log `λ_r, λ_m`; reconcile `J=1` (D.4) with `J≤8` (0.1, B.5) | I9 |
| Re-state deflation (a) as "no measurable headroom in `ℛ₀`, one config, one seed"; one denominator, one scorer; report composition | M13 — 0.062 ∉ [0.023,0.046]; p=0.33; `forget_023` discloses at every perturbation level including 2× edit norm |
| Replace deflation (b)'s premise with the 32× position-weighting argument; re-specify IHP−H with `fact_free_tf_states` | deflation (b) — conclusion correct; `fact_free_tf_states` already forces `F` positions to mask, so IHP−H as written is confounded in two factors |
