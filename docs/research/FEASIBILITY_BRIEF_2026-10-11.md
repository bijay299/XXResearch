# Feasibility and decision brief — revision 2 (corrected)

**CPU-only, 2026-10-11. This authorises nothing.** No training, GPU run,
generation, GPU calibration, method implementation, API prompt generation, human
label, packet distribution or external message was produced for it. Amendment 01
stays closed: its measurements, bootstrap, suites and published index are not
reopened, and all 40 indexed rows still verify. Upstream facts are bound to
fetched bytes in [`feasibility_sources_2026-10-11.json`](feasibility_sources_2026-10-11.json)
(URL, size, `sha256`, fetch time). **Novelty and publishability remain
unestablished.**

This brief *narrows* [`NEXT_SCREEN_PROPOSAL.md`](NEXT_SCREEN_PROPOSAL.md)
revision 5 and does not edit it; the explicit change list is
[`REVISION_6_CHANGES.md`](REVISION_6_CHANGES.md). The smoke-calibration proposal
it refers to is [`LACU_SMOKE_CALIBRATION.md`](LACU_SMOKE_CALIBRATION.md).
*(The `REVIEW.md` supplied with the review prompt is the earlier round's
acceptance of `332d0a9`, not a review of `117b584`, so it carries no evidence on
the six points. Each point was therefore re-verified directly against upstream
source and the committed detector, as cited below. Nothing in it conflicts with
these corrections, and its two standing requirements — explicit detector
validation and an unexposed annotator — are carried in §3 and §6.)*

> ## Corrections to revision 1 of this brief
>
> | # | revision 1 said | corrected |
> |---|---|---|
> | 1 | SPEED's edit is "training-free and deterministic", so its two-seed factor collapses — **72** checkpoints | **Wrong.** The editor has four randomness sources and an edit seed (§1.2). Two edit seeds are preserved, rule 1 applies to it, and the count is **98** |
> | 2 | if ConAbl's reuse fails qualification, "train the `horse` request-1 pair afresh" — called unchanged | **Wrong twice:** that is an ACCEPT/TEST-triggered replacement, and revision 5's default is to **stop that sequence**. The stop is restored; all-fresh is offered only **prospectively** (§2.4) |
> | 3 | `flower` listed among the measured preservation endpoints | **Not measurable.** The committed detector matches COCO_V1 names by exact equality and that list has no `flower`: the endpoint would return 0 hits and pass vacuously. Anchor changed to `potted plant`, and integrity defined for LACU's selected replacements and SPEED's empty anchor too (§3) |
> | 4 | `SPEED+AIC(reproduced)` treated as a clean comparator | **Under-specified and degenerate in the native setting.** With SPEED's empty/whitespace anchor, `c∗` *is* `K₂`'s first column, so `K₂^AIC` is rank-deficient and λ=0 inverts a singular Gram. Setting, rank policy, anchor-accumulation policy and round-trip checks are now pinned (§1.3). An AIC reproduction is **not CEASE**, and its outcome **cannot eliminate HOC** |
> | 5 | 8 reported concepts; evaluation ≈ 47–50 device-hours | **9** concepts, and the missing slots (request-1 parent rates on ACCEPT, `M0` baselines, candidate dumps) put evaluation at **67.4** device-hours and the total at **119–184** (§5) |
> | 6 | "`I` can read MATERIAL only above ~17 pp" presented as a threshold on effects | **Mis-stated.** It is arithmetic about an **observed** interval under an **assumed** width — not a true-effect threshold and not power. Coefficient vectors, missing-cell invalidation, reference propagation and an explicit unknown-covariance sensitivity table replace it (§4) |

---

## 0. Decision

> **Revision 5 remains NO-GO: CEASE has no released implementation.**
>
> **The reduced screen is NOT ready for conditional approval either.** What is
> ready to approve is a **CPU track** plus **one GPU smoke calibration**.
> Four things must land before the screen itself is a decision:
> (a) the smoke measurement replacing LACU's 1.5–4.0 h/request estimate;
> (b) a base-rate screen for the new anchor `potted plant` and the control
> `clock`, neither of which has ever been measured here;
> (c) a **prospective** decision on all-fresh request 1 (§2.4), taken before any
> holdout evaluation;
> (d) acceptance that **HOC stays unexcluded** (below).

**The novelty consequence, stated plainly.** The screen's purpose is to expose a
consequential limitation of *existing remedies*. CEASE is two components; only
**AIC** is specified well enough to reproduce (Eq. 9), and **HOC is not** — the
paper gives the projection but never the rule that decides when it applies or how
many singular directions `Û` keeps. So:

* a reproduction of AIC is **not CEASE**, and must never be reported as CEASE;
* if `SPEED+AIC(reproduced)` fails to prevent relapse, **that does not show CEASE
  fails**, because HOC — the component aimed precisely at cross-edit interference
  between successive updates — was never run;
* therefore **no claim of the form "existing remedies do not solve this" is
  available from this screen** while HOC is untested. The best available claim is
  narrower: *at a matched end state on these concepts, LACU-native, SPEED-native
  and an AIC reproduction behave thus; CEASE's HOC is untested and may solve it.*

That is a materially weaker contribution story than revision 1 of this brief
implied, and it is the main reason the screen is not recommended for approval
today. **Waiting for CEASE's release is the cheapest route to a defensible
claim.** The CPU track below costs no GPU time and leaves the screen ready the
day code appears.

---

## 1. Implementation reconnaissance, corrected

### 1.1 Status

| method | code | licence | base model | replacement behaviour | blockers |
|---|---|---|---|---|---|
| **CEASE** | **none** — both repos hold a static project page; README: "Code coming soon." | none stated | SD 1.4 / **1.5** / 2.1 | one fixed shared anchor per task (`person`/`art`/`object`) | **BLOCKING.** No implementation; HOC's trigger and rank cut unspecified, so a reimplementation cannot be validated |
| **LACU** | **yes**, `SonyResearch/LACU` | **MIT** | **SD 1.5 default**, accepts a local diffusers folder | **per-prompt** score-selected mapping prompt from 10 LLM+CLIP candidates | `OPENAI_API_KEY` required by the released runner; 40 GB is below the authors' setting; its 10 released concepts are not ours |
| **SPEED** | **yes**, `Ouxiang-Li/SPEED` | **Apache-2.0** | `--sd_ckpt`, accepts SD 1.5 or a local folder | `--anchor_concepts` is a required argument | **stochastic** (§1.2); saves **absolute** edited weights; native object anchor ambiguous between sources |
| **ConAbl (L2-SP 25000)** | **yes**, pinned CUIG `9932ac3` | upstream | our SD 1.5 `M0` | anchor always supplied; **no native selection** | none known. The incumbent; the only arm with a candidate request-1 reuse |
| Sculpting Memory | authors' code exists; **CUIG's `SculpMem` is a reimplementation** by its own README | upstream | SD 1.5 | anchor supplied; dynamic mask | not a native arm as installed — using it and calling it Sculpting Memory would be the error the handoff forbids |

### 1.2 SPEED's edit is stochastic — replication restored

`train_erase_null.py` (digest unchanged from the revision-1 record, `abae9f87…`):

| line | source | effect |
|---|---|---|
| `:25` | `noise = torch.randn_like(mini_ret_embs)` | **Gaussian retain augmentation** — the perturbed retain embeddings that build `sum_ret_ret`, hence the null projector `P` |
| `:48` | `kmeans(X=null_hidden[1:], num_clusters=3, …)` | **k-means construction of `K₂`** — randomly initialised centres |
| `:85` | `last_ret_embs[torch.randperm(…)]` | **retain shuffle**, which interacts with the `chunk_size` batching of `sum_ret_ret` |
| `:140` | `--aug_num` default **10** | ten perturbations per retain sample |
| `:131`, `:147` | `--seed 0`, `seed_everything(args.seed)` | an **edit seed** exists and is settable |

So SPEED and `SPEED+AIC` carry **two edit seeds** like every other arm, **rule 1
(seed disagreement) applies to them**, and the design does not collapse. With
every proposed configuration retained the selected count is
**CA 4 arms × 20 + NAT 3 arms × 6 = 98**, excluding `M0` and calibration
candidates. Native randomness is not disabled to recover 72.

Also from source: `:118` `edit_dict[layer_name] = layer_weight + delta_weight`
then `:189` `torch.save(edit_dict, …)` — the `.pt` holds **absolute edited
projection tensors, not deltas**. Any adapter that chains edits must therefore
**replace** those tensors **exactly once**; adding them to a parent's weights
double-applies the edit. Round-trip checks in §1.3.

### 1.3 AIC's comparator, pinned

**The degeneracy, verified.** `K₂ = [null_hidden[0], three k-means centres]ᵀ`
(`:48`). The anchor embedding is taken at index `attention_mask.sum() − 2`
(`:68`). A tokenizer-only check (no model loaded, no forward pass) gives
`attention_mask.sum() = 2` for both `''` and `' '` — SPEED's released instance
anchor — so that index is **0**, and `c∗` is **identically `K₂`'s first column**.
Appending it adds **no new invariant direction**: `K₂^AIC` is rank-deficient, and
with `λ = 0` the solver inverts `K₂ᵀPMK₂`, singular in exact arithmetic. The same
check gives index 1 for `object` and `horse`, and index 2 for `potted plant`
(the token `plant`), so AIC is non-degenerate with any non-empty anchor.

**Settings declared now, before implementation:**

1. **Anchor.** AIC is evaluated **only with a non-empty anchor** — `horse` or
   `potted plant` in family CA, and in family NAT the pinned generic anchor
   (`object`, matching CEASE's own instance setting) rather than SPEED's released
   empty one. `NAT-SPEED` itself keeps **one** documented native anchor, pinned in
   the pre-registration, with the alternative reported.
2. **Rank policy.** Before appending, project `c∗` onto the orthogonal complement
   of `span(K₂)`. Append only if the residual norm exceeds a declared tolerance;
   otherwise record **`AIC-NOOP`** for that configuration — a reportable finding,
   not a silent pass. `λ = 0` is used only when the appended matrix is full rank
   at that tolerance; otherwise `λ > 0` is declared and reported.
3. **Current vs accumulated anchor protection.** CEASE's setting has one shared
   anchor per task, so this choice never arises there and is **ours**:
   **`AIC-current`** protects only stage *k*'s anchor (the faithful reading of
   Eq. 9) and is the default; **`AIC-accum`** protects every anchor used so far
   and is **our variant**, labelled as a changed-anchor experiment, never as
   CEASE or as AIC-as-published. In the shared cells the two coincide; they
   differ only in the disjoint cells, which is exactly where the design looks.
4. **Round-trip and constraint checks** (specified here, implemented later, not
   in this correction): the saved `.pt` keys are exactly the `attn2.to_v` set;
   reload after `save_pretrained` is **bit-identical**; applying the same `.pt`
   twice is **idempotent** and an additive application is **detected and
   refused**; every unedited key is bit-identical to the parent;
   `rank(K₂^AIC)` vs `rank(K₂)` recorded; achieved `‖Δ c∗‖` reported against its
   tolerance; SPEED's own `(ΔP)C₀ ≈ 0` property re-checked so AIC is not shown to
   help by quietly breaking the base solver; the edit seed and a digest of the
   k-means centres recorded per edit.

### 1.4 LACU's three constraints (unchanged, re-stated)

`OPENAI_API_KEY` is required unconditionally by `train.sh:ensure_retain_prompts`,
per concept **per checkpoint**, and the released runner deletes and regenerates
`related_score.txt` every time — by design, since retain selection is
model-dependent. Mapping selection needs no API. Hand-authoring retain prompts
replaces part of the locality mechanism and would be an ablation. Our devices are
40 GB against the authors' recommendation; `train.py` trains the **full UNet**
with Adam while holding a frozen teacher and running a 20-step replay pipeline at
the same batch. Checkpoints are full ~5 GB pipelines. **None of these is resolved
by analysis — only the smoke calibration measures them.**

### 1.5 CPU checks run (no model, no forward pass)

`py_compile` passes on all 8 LACU and 3 SPEED sources. LACU's **training** path
resolves entirely inside the existing `venv-cuig` (torch 2.4.0+cu121, diffusers
0.30.2, transformers 4.44.2) — only `eval.py` misses `pandas`/`wandb`, and we do
not use their evaluator. SPEED needs `kmeans_pytorch` + `pandas`; its behaviour on
diffusers 0.30.2 is **unverified**. Both ConAbl anchor caches exist (`Horses` 204,
`Flowers` 204 images). The tokenizer check of §1.3 used the tokenizer only.

---

## 2. The design, corrected

### 2.1 Arms and counts

**Family CA — anchor prescribed**, carrying the factorial: `ConAbl`, `SPEED`,
`SPEED+AIC(reproduced)`, `CA-LACU(ablation)`. Per arm **4 request-1**
(2 anchors × 2 seeds) + **16 request-2** (2 A1 × 2 A2 × 2 targets × 2 seeds) = **20**.

**Family NAT — native**: `NAT-LACU`, `NAT-SPEED`, `NAT-SPEED+AIC(reproduced)`.
Per arm **2 request-1** + **4 request-2** = **6**. **ConAbl contributes no NAT
arm** — it has no native replacement-selection mechanism, so its CA cells *are*
its native configuration. That is the only sharing in the design.

**Selected checkpoints = 4 × 20 + 3 × 6 = 98**, excluding `M0` and the **486**
calibration candidates, which are counted separately (§5).

**Prescribing the anchor is configuration** for ConAbl, SPEED and SPEED+AIC
(an explicit argument in each). For LACU it **replaces the mechanism**, so
`CA-LACU` is an **ablation in every cell** and `NAT-LACU` is the method.

### 2.2 Ownership

Each method owns its chain; no cross-method chain exists. A ConAbl checkpoint is
not a valid request-1 parent for LACU or SPEED. `M0` is the only shared object.

### 2.3 Anchors and the control

`horse` (shared anchor) is retained. The disjoint anchor becomes
**`potted plant`** — see §3.1. The separate non-target control is **`clock`**.
Neither `potted plant` nor `clock` has ever been measured here, so **both need a
base-rate screen on DEV before adoption**, and a new ConAbl anchor cache
(200 images, ≈0.06 device-hours) is required for `potted plant`. The existing
`Flowers` cache is preserved and unused in the minimum design.

### 2.4 ConAbl request-1 reuse — the stop is restored

**Default, as in accepted revision 5 §3.2 option (i):** the existing `horse` MA
pair may serve as `ConAbl`'s `horse` request-1 checkpoint **only if it QUALIFIES**
on the new ACCEPT and TEST sets. **If it fails, that arm's `horse`-parent
sequence STOPS.** Non-qualification is reported with the measured `p₁` and
`r₁^ACCEPT`. **No fresh `horse` training is triggered by the failure**, no
threshold is relaxed, and nothing is retrained: a holdout result must never
select a training action.

**What that stop costs, so the PI can price it.** The `horse`-parent cells are
`c1, c2, c5, c6` — **4 of ConAbl's 8 cells per seed, 8 of 16 across seeds.**
Losing them invalidates ConAbl's declared `S`, `P` **and** `I` (§4.3), leaving
only the prespecified simple effects that avoid those cells. So the reuse route
carries a 50%-cell-loss risk for one arm.

**The prospective alternative, offered for decision now and not triggered by any
result:** declare at pre-registration that **request 1 is trained fresh for every
method**, removing reuse entirely. Cost: **+2 ConAbl trainings ≈ +0.47
device-hours** — negligible against the risk it removes. **Recommended.** The
choice must be made **before** any holdout evaluation; afterwards it is closed.

### 2.5 Retained limits

One near pair and one far pair: proximity stays confounded with concept identity,
two seeds replicate the run and not the concept, and **no mechanism claim and no
generality over concepts follows**. The revision 5 §5.3 conditions for
independent concept evidence are unchanged and **not** part of this design. The
rule-2 stop is retained and now includes native SPEED (§3.4).

---

## 3. Every preservation endpoint must be measurable

### 3.1 `flower` cannot be measured at all

`detect.py:117-121` compares `d["label"] == tgt` — **exact string equality**
against the torchvision `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` category names.
That list has **91 entries and no `flower`**; the plant-like names are
`potted plant` (index 64) and `vase` (86). A `flower` anchor-integrity endpoint
would return **0 hits on every checkpoint**, giving a `0 − 0 = 0` contrast that is
trivially `NULL(10)` — **a vacuous pass on an unmeasured endpoint, which could
let ALREADY-SOLVES fire.**

**Resolution: change the anchor, do not substitute the metric.** The disjoint
anchor becomes **`potted plant`**, a category the committed detector measures
exactly, subject to the §2.3 base-rate screen. `potted plant` is **not** a silent
stand-in for `flower`: the concept itself changes, every anchor prompt and anchor
cache changes with it, and the A2-identity factor is re-labelled accordingly.
Fallback if its base rate is too low on DEV: `vase`. **`flower` may be retained
only as an optional exploratory arm whose anchor integrity is validated by human
labels and by nothing else** — and it carries no protocol-level claim.

### 3.2 Integrity for LACU's selected replacements

Native LACU chooses a **different mapping prompt per forget prompt**, so there is
no single anchor. Definition:

* the realised replacement multiset is read from that stage's `map_model.csv`,
  **archived and hashed per stage** (it is checkpoint-dependent by design);
* for each distinct replacement concept that **has** a COCO_V1 category, report
  its stagewise and cumulative integrity contrast separately, with its selection
  count;
* replacements with **no** COCO category are reported as **`UNMEASURED`** with
  their counts, and the arm's anchor-integrity endpoint is marked **partial**
  with the measured fraction stated.

The measurable fraction is **not knowable before the mapping step runs** — it is
a property of what the LLM proposed and the model selected. It is computable on
CPU from `map_model.csv` the moment that step exists.

### 3.3 Integrity for SPEED's empty or generic anchor

With the empty anchor there is **no concept to measure**; with a generic anchor
(`object`) there is no COCO category. Anchor integrity is therefore
**`ANCHOR-UNMEASURED`**, which is *not* `NULL(10)` and *not* a pass.

### 3.4 ALREADY-SOLVES, split so an unmeasured endpoint cannot pass

Revision 5 §6.4 requires anchor integrity `NULL(10)` for every anchor used. Taken
literally with §3.3 that **permanently excludes native SPEED** on a measurement
artefact, which would quietly drop the strongest closed-form native arm. Split
the label instead:

| label | conditions | effect on the decision |
|---|---|---|
| **`ALREADY-SOLVES`** | every stage ACCEPTED; relapse `NULL(10)` for every earlier target; retained categories and the control `NULL(10)` stagewise **and** cumulatively; **anchor integrity `NULL(10)` for every anchor, all measured**; no LOW-PARENT, no MATERIAL+ `PriorΔ`, no `R-NONTRANSFER` | fires **rule 2**: report and stop |
| **`ALREADY-SOLVES-CORE (anchor unmeasured)`** | all of the above **except** that one or more replacements have no measurable endpoint | **also fires rule 2** — the limitation we meant to attack is absent — but **no anchor claim is made**, the unmeasured concepts are named with their counts, and the write-up says anchor integrity was not assessed for that arm |
| **neither** | any condition fails | the screen proceeds |

So native SPEED and native LACU both participate in the success/redirect logic,
while **no unmeasured or unsupported concept ever passes an endpoint**. Human
validation stays explicit and is **not** performed here: no label is produced,
supplied or distributed, and the 180-item packet stays blank (§6).

---

## 4. Contrasts, missing cells, and uncertainty

### 4.1 The eight cells

Eight counterbalanced CA cells **per arm per edit seed**, indexed
`(A1, A2, target)`, aggregated into the four sharing × proximity means:

| | A1 | A2 | target | sharing | proximity |
|---|---|---|---|---|---|
| c1 | horse | horse | dog | shared | near |
| c2 | horse | potted plant | dog | disjoint | near |
| c3 | potted plant | potted plant | dog | shared | near |
| c4 | potted plant | horse | dog | disjoint | near |
| c5 | horse | horse | sandwich | shared | far |
| c6 | horse | potted plant | sandwich | disjoint | far |
| c7 | potted plant | potted plant | sandwich | shared | far |
| c8 | potted plant | horse | sandwich | disjoint | far |

Each cell value is `Relapse_cat(k; R_cat)` for that chain.

### 4.2 Coefficient vectors

| contrast | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 |
|---|---|---|---|---|---|---|---|---|
| **S** (sharing) | +¼ | −¼ | +¼ | −¼ | +¼ | −¼ | +¼ | −¼ |
| **P** (proximity) | +¼ | +¼ | +¼ | +¼ | −¼ | −¼ | −¼ | −¼ |
| **I** (interaction) | +½ | −½ | +½ | −½ | −½ | +½ | −½ | +½ |
| **A2 identity** | +¼ | −¼ | −¼ | +¼ | +¼ | −¼ | −¼ | +¼ |
| **S\|near** (simple) | +½ | −½ | +½ | −½ | 0 | 0 | 0 | 0 |
| **S\|far** (simple) | 0 | 0 | 0 | 0 | +½ | −½ | +½ | −½ |
| **P\|shared** (simple) | +½ | 0 | +½ | 0 | −½ | 0 | −½ | 0 |
| **P\|disjoint** (simple) | 0 | +½ | 0 | +½ | 0 | −½ | 0 | −½ |
| **S − P** (dominance) | 0 | −½ | 0 | −½ | +½ | 0 | +½ | 0 |

`⟨S,P⟩ = ⟨S,I⟩ = ⟨P,I⟩ = ⟨S,A2⟩ = ⟨P,A2⟩ = 0` — verified in
`screen_measurement_inventory.py`. The same check confirms a property the
counterbalancing buys: the **request-1 reference loading of S, P and I is exactly
0**, because each contrast carries two `horse`-referenced and two
`potted plant`-referenced cells with opposite signs, so the choice of request-1
reference cancels. **This is why all four `(A1, A2)` combinations are needed**,
and why the fallback to cells 1–2 identifies nothing about sharing.

### 4.3 A missing raw cell invalidates the marginals too

`S`, `P` and `I` are declared over all eight cells. **If any cell is missing or
inadmissible, all three are invalid as declared** — not just `I`. Reconstructing
a marginal from the surviving cells changes the estimand (it re-weights the
anchor-identity composition and breaks the reference cancellation of §4.2), and
is forbidden.

**Only separately prespecified simple effects survive**, and only those whose
support is intact: `S|near`, `S|far`, `P|shared`, `P|disjoint`, each with its own
interval and its own threshold, all four pre-registered **before** any data exist.
A cell becomes inadmissible if its stage is not ACCEPTED, if it is
`R-NONTRANSFER`, or if its relapse reference is invalid.

### 4.4 Propagation of an invalid request-1 R reference

`Relapse_cat(k; R_cat)` differences `D_{R_cat}(C_k, cat)` against
`D_{R_cat}(C_1, cat)` **for that chain**. If the request-1 R reference is invalid
— stage 1 not ACCEPTED, or `R-NONTRANSFER` on `R_cat` — then **every request-2
descendant of that `C_1` loses its relapse estimand**, descriptively as well as
inferentially for any contrast.

Per arm per seed there are two request-1 checkpoints, each parenting **four**
request-2 cells (2 A2 × 2 targets). So **one invalid request-1 reference removes
four of the eight cells**, which by §4.3 invalidates `S`, `P` and `I` and leaves
only the two simple effects supported by the surviving parent. This is the
single most load-bearing fragility in the design and it is why request-1
qualification is gated before any request-2 stage runs.

### 4.5 `R-NONTRANSFER` — descriptive, never inferential

Estimates are **computed, stored and reported** for `R-NONTRANSFER` stages,
labelled `DESCRIPTIVE`, with both residues shown. They are **blocked from every
inferential use**: no coefficient-vector contrast, no marginal, no simple effect,
no dominance comparison, no ALREADY-SOLVES credit. A CPU regression must fail the
build if a descriptive estimate enters a contrast. Nothing is retuned, `R_i` is
not reselected and no dial is re-fit.

### 4.6 Uncertainty — what is assumed, and what is unknown

The measured anchor is a per-category paired contrast on the **spent** TEST
probe: median 95% width **10.00 pp** at the pilot layout of **10 scenes × 8
images** per category, so a per-cell half-width of **5.00 pp**. Two separate
unknowns stand between that and any planning number:

1. **The between-scene / within-scene variance split is unknown.** The bootstrap
   unit is the scene cluster, so **the scene count is the precision knob**, and
   `h ∝ 1/√scenes` holds only for the between-scene component. Consequently
   **20 scenes × 8 images and 40 scenes × 4 images are not assumed equally
   precise**, and no equivalence between them is claimed.
2. **The cell correlation `ρ` is unknown.** Every cell's relapse contrast is
   measured on the *same* `R_cat` scenes, so `ρ > 0`, but by how much is not
   estimated here. For a coefficient vector `a` with `Σa = 0` under exchangeable
   `ρ`, `Var = σ²(1 − ρ)Σa²` — the shared component cancels, so a positive `ρ`
   **helps**, and ignoring it is conservative.

Sensitivity at the proposed **20 scenes × 8 images** (half-widths in pp):

| `ρ` | `S` | `P` | `I` | `S\|near` | `S − P` |
|---|---|---|---|---|---|
| −0.5 | 3.06 | 3.06 | 6.12 | 4.33 | 4.33 |
| 0.0 | 2.50 | 2.50 | 5.00 | 3.54 | 3.54 |
| +0.5 | 1.77 | 1.77 | 3.54 | 2.50 | 2.50 |
| +0.8 | 1.12 | 1.12 | 2.24 | 1.58 | 1.58 |

**The reading rule, correctly stated.** With a *realised* half-width `h`, the
predicate `MATERIAL+` requires the **observed point estimate** `> 10 + h`, and
`NULL(10)` requires `|observed point| ≤ 10 − h`; `NULL(10)` is unreachable
whenever `h > 10`. **This is arithmetic about an observed interval under an
assumed width. It is not a threshold on true effects and it is not a power
calculation** — no detection probability is claimed anywhere, and `h` is carried
from a different contrast on a spent probe.

`ρ` and the variance split are **estimable on CPU from Amendment 01's existing
per-image rows** without any new generation. That is proposed as a CPU
deliverable (§6 item 4), not done here, because it is new analysis and Amendment
01 is closed.

---

## 5. Workload, from the enumerated slots

The inventory is one script and one log, not a prose estimate:
`scripts/seq/screen_measurement_inventory.py` →
[`screen_measurement_inventory.log`](../../results/feasibility_2026-10-11/screen_measurement_inventory.log).

**Counted separately:** **98** selected checkpoints (22 request-1, 76
request-2), **486** calibration candidates / intermediate dumps, and **`M0`**.

**Nine measured concepts:** `cat`, `dog`, `sandwich` (targets); `bird`, `chair`,
`bicycle` (retained); `clock` (separate control); `horse`, `potted plant`
(anchors).

**Scene layout:** DEV 10 × 8, ACCEPT 20 × 8, TEST 20 × 8, R 20 × 8 images.

| slot | set | ckpts | concepts | images |
|---|---|---|---|---|
| calibration candidates (dial selection, own target only) | DEV | 486 | 1 | 38,880 |
| selected checkpoints (own-target residue) | ACCEPT | 98 | 1 | 15,680 |
| **request-1 parents — `p₂` for `dog` and `sandwich`** | ACCEPT | 22 | 2 | 7,040 |
| `M0` — `p₁` for `cat` | ACCEPT | 1 | 1 | 160 |
| selected checkpoints — all nine concepts, once | TEST | 98 | 9 | 141,120 |
| **`M0` cumulative baselines** | TEST | 1 | 9 | 1,440 |
| selected checkpoints — the relapse probe | R | 98 | 1 | 15,680 |
| `M0` sanity baseline | R | 1 | 1 | 160 |
| **total** | | | | **220,160** |

**Evaluation = 67.4 device-hours** at the **measured** 1.1027 s/image.

| training / editing | device-hours | basis |
|---|---|---|
| ConAbl, 1,000 steps × 20 requests | **4.68** | **measured** 0.8420 s/step |
| one new `potted plant` anchor cache (200 images) | **0.06** | **measured** image rate |
| SPEED + SPEED+AIC, **208 edits** (4 dial points × 52 selected) | **3.47** | **estimate**, 60 s/edit assumed — conservative against upstream's "seconds-level" claim |
| LACU, 26 requests | **43 – 108** | **estimate**, 1.5–4.0 h/request + 0.15 h mapping rescore |
| LACU retain rescoring | **not costed** | **unmeasured** |

> **Total ≈ 119 – 184 device-hours** — about **+20 device-hours on evaluation
> alone** against revision 1's coarse figure, and the LACU band still dominates
> and is still an estimate.

**Which dials need retraining, and which only a dump.** ConAbl's dial is
**training iterations** — dumps every 100 steps from **one** run, no extra
training (L2-SP is held at the pilot's 25000 and is **not** a dial in this
design). LACU's dial is **training steps** — its native saves at 100, 200 and
final, from **one** run; its `λ_unlearn`/`lr` would each need a fresh run and are
therefore **fixed** at a pre-registered value. SPEED's dial (`retain_scale`,
`threshold`, `λ`) is a **separate closed-form edit per value** — cheap in compute,
but each needs its own DEV evaluation, which is why DEV is 38,880 images.

**Wall time and devices.** LACU's runner occupies 3 GPUs per run, so its
43–108 device-hours are **14–36 hours of 3-GPU wall time**; evaluation
parallelises 4-way (~17 hours). **3–7 days of occupancy** if uninterrupted. There
is **no GPU-hour cap**; devices must be **verified unoccupied by another user at
execution time**. *Status, not approval: the four A100-SXM4-40GB read 14 MiB used,
0% utilisation when this was written.*

**Disk.** LACU writes full ~5 GB pipelines at steps 100/200/final: ≈15 GB ×
26 requests ≈ **390 GB**, plus ~62 GB of materialised SPEED parents for chaining
and ~5 GB of `.pt` edits. `/` has **4.9 GB free (100% used)**; `/data` has
**1.2 TB**. Everything under `/data`, dumps pruned to the DEV-selected one plus
the final after each stage's evaluation, with the running total checked before
each stage.

---

## 6. Dependency order — implementation and checks before the screen

| # | work | compute | gate |
|---|---|---|---|
| 1 | **CPU glue and round-trip checks** of §1.3: SPEED `.pt` → diffusers replacement exactly once, double-application refused, unedited keys bit-identical, key-set assertion; LACU arg wiring; the fixed-mapping CSV for `CA-LACU`; the retain CSV for the closed-form arms | **CPU** | must pass before any GPU request |
| 2 | **LACU smoke calibration** — fully specified in [`LACU_SMOKE_CALIBRATION.md`](LACU_SMOKE_CALIBRATION.md) | **≈1.5–3 device-h**, 3 GPUs verified idle at execution time | replaces the 1.5–4.0 h estimate; **failure action defined there** |
| 3 | **base-rate screen** for `potted plant` and `clock` on DEV, plus the `p` re-measurement of every target on the new sets | small GPU | if `potted plant` fails, fall back to `vase`; if `clock` fails, pick another control — **before** the freeze |
| 4 | **CPU variance study** on Amendment 01's existing rows: the between/within split and the cell correlation `ρ` of §4.6, so the scene count is chosen from data rather than from a conservative assumption | **CPU** | no new generation |
| 5 | **revision 6 + the hashed pre-registration**: [`REVISION_6_CHANGES.md`](REVISION_6_CHANGES.md), the four set identities and scene counts, the eight-cell list, the coefficient vectors, the four prespecified simple effects, the AIC settings of §1.3, the arm and ablation labels **fixed in the code that names the arms**, and the **prospective** request-1 decision of §2.4 | **CPU** | **the freeze.** Protocol and checkpoint-slot rules are frozen here, before any evaluation |
| 6 | **the TEST list**, derived from DEV and ACCEPT decisions only, then **locked with a digest** before the first TEST image | **CPU** | after ACCEPT, before TEST |
| 7 | **the screen** | GPUs verified unoccupied at execution time | the §6.6 stopping rule live throughout |
| 8 | `SPEED+AIC(reproduced)` | CPU to write; GPU only inside item 7 | **method implementation — its own approval** |
| 9 | **CPU regressions** that fail the build on: two relapse operands from different sets; a relapse number from a LOW-PARENT, ERASURE-FAILURE or `R-NONTRANSFER` reference; a descriptive estimate entering a contrast; a marginal reconstructed from fewer than eight cells; a comparison from fewer than two accepted arms; a skipped cumulative check; §6.5 rules out of order; a TEST re-evaluation or a late addition to the locked TEST list; an endpoint whose concept has no measurable category passing ALREADY-SOLVES | **CPU** | before item 7 |
| 10 | **human labelling of the 180-item packet** by an annotator exposed to neither earlier packet nor any source image | none | **PI action.** Not scheduled, not initiated, no label produced or distributed |

The screen's detector numbers are **not findings** until item 10 exists.

---

## 7. What this does not establish

* **Not that existing remedies fail.** **HOC is untested and unexcluded** (§0).
  An AIC reproduction is not CEASE and its failure says nothing about CEASE.
* **Not novelty or publishability.** A working screen and a component
  reproduction establish neither.
* **Literature, narrowed again.** Revision 1 said CEASE's interference
  diagnostic references the unedited `W₀`. **Eq. 4 actually subtracts the
  intended single edit:** `noise_e(c) = ‖Σ_{i≤N} Δ_i c‖² − ‖Δ_e c‖²`, a signed
  residual, and Fig. 5 tracks it on an erased target across 100 edits. It is
  still **not** the paired image-level estimand — it is a weight-space quantity
  at a text embedding, not a change in detection rate between two checkpoints on
  a fixed probe — but it is closer than revision 1 allowed, and CEASE's Tables
  1–3 already baseline CUIG-GradProj and Sculpting Memory over 100-step
  sequences. No broader survey was run and none is proposed here.
* **Not that any arm fits, runs or converges on this host.** Items 1–3 are what
  would establish that; §1.5 lists what remains unverified.
* **Not a mechanism, a proof, or generality over concepts.**
* **Not independently verified where it is ours.** Upstream facts are
  re-checkable from the recorded URLs and digests; the local environment, disk,
  GPU-status and ledger observations are this project's own.

---

## 8. Reproduce

    python scripts/seq/screen_measurement_inventory.py        # the inventory of §4-§5
    python scripts/seq/check_feasibility_brief_numbers.py     # asserts every figure printed here

Both read only committed files and run no model. Upstream facts: refetch any URL
in `feasibility_sources_2026-10-11.json` and compare its `sha256`; a changed tree
on either CEASE repository is the signal that §0's wait has ended.
