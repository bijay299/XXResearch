# Feasibility and decision brief — can a fair comparison with strong native remedies expose a consequential limitation?

**CPU-only preparation, 2026-10-11. This authorises nothing.** No training, no
GPU run, no generation, no method implementation and no annotation answer was
produced for it. Every upstream fact below is bound to a fetched byte stream
recorded in [`feasibility_sources_2026-10-11.json`](feasibility_sources_2026-10-11.json)
(URL, size, `sha256`, fetch time), so each claim can be re-checked by refetching.
Amendment 01 is frozen and untouched: its measurements, bootstrap, suites and
published index are not reopened here. **Novelty and publishability remain
unestablished.**

This brief *narrows* [`NEXT_SCREEN_PROPOSAL.md`](NEXT_SCREEN_PROPOSAL.md)
revision 5 to a minimum executable subset and records what is blocked. It does
not widen it, and it does not edit it: if the PI approves, revision 6 is the
first CPU deliverable.

---

## 0. Recommendation

> **NO-GO on the screen as written. CONDITIONAL GO on a reduced three-method
> screen that drops CEASE and keeps its decisive component.**

**Why the written screen cannot run.** Revision 5 §5.4 states "CEASE and LACU are
not droppable for convenience or a count; if either cannot be run here that is a
**blocking finding**", and §6.6 makes "CEASE or LACU cannot be run here" a stop.
**CEASE has no code.** Both repositories that carry its name hold a static
project page — `index.html`, `style.css`, two PNGs — and the author repository's
README says "Code coming soon." under a `## Code` heading. There is no release
date, no licence, and the paper states no release commitment. The screen as
specified therefore stops at §6.6 before any compute.

**Why the question survives anyway.** CEASE is built **on SPEED's closed-form
solver**, and SPEED is public (Apache-2.0, `Ouxiang-Li/SPEED`). CEASE's two
additions are not equally reproducible:

| component | specified? | consequence |
|---|---|---|
| **AIC** — append the anchor token embedding to SPEED's invariance matrix, `K₂^AIC = [K₂, c∗]`, with zero invariance regulariser (Eq. 9) | **fully specified**, and it is the component the authors' own ablation credits with "the larger gain" (Table 4) | reproducible on public SPEED as a small, auditable change |
| **HOC** — project the update onto the orthogonal complement of the dominant left singular directions of `Δ_hist Δ_histᵀ` (Eqs. 12–14) | **mechanism specified, trigger and rank cut not.** The paper says `Π_i = I − ÛÛᵀ` "when HOC is applied, and `Π_i = I` otherwise", and only that HOC acts "when the accumulated historical update substantially affects the current target". No threshold, and no rule for how many singular directions `Û` retains | **not faithfully reproducible.** Two free decisions change the method, and no published number on our concept domain exists to check a guess against |

So the recommended screen replaces "native CEASE" with two honestly-labelled
arms — **native SPEED** (CEASE's own base solver, the thing CEASE improves on)
and **`SPEED+AIC(reproduced)`** (a partial reproduction, never called CEASE) —
and keeps **native LACU**, which *is* available and runnable here. The candidate
direction of §7 is precisely a change to *which object is held invariant*, so
`SPEED+AIC(reproduced)` is the comparator that matters most to it; HOC's absence
costs the screen a second comparator, not its question.

**Tripwire, not a wait.** Re-check `yongliang-wu/CEASE` before the
pre-registration is frozen. If code is released before freeze, CEASE enters as a
native arm and `SPEED+AIC(reproduced)` becomes an ablation. If it is released
after freeze, it is a follow-up, not a late addition.

**What the PI is being asked to approve** is listed in §5: one measured LACU
smoke calibration, then the reduced screen. **Approving nothing is also a
defensible outcome** — the honest reading of §1 below is that the adjacent
literature is closer than revision 5 recorded.

---

## 1. Native-method feasibility — verified before anything else

### 1.1 Status table

| method | code | licence | base model | replacement behaviour | assets needed | blockers |
|---|---|---|---|---|---|---|
| **CEASE** | **none** (project page only; "Code coming soon.") | none stated | SD 1.4 / **1.5** / 2.1 | one fixed shared anchor per task (`person` / `art` / `object`) | 100 erase + 100 retain concepts, 5 templates/concept | **BLOCKING.** No implementation. HOC's trigger and rank cut unspecified, so a reimplementation cannot be validated |
| **LACU** | **yes**, `SonyResearch/LACU` | **MIT** | **`runwayml/stable-diffusion-v1-5` by default**; accepts a local diffusers folder | **selects a mapping prompt per forget prompt** by score-prediction distance from 10 LLM+CLIP candidates | per concept: `train.csv` (100 forget prompts), `candidates_clip.json`, `related_concepts.txt` | **OpenAI API key required** (see 1.3); 40 GB cards are below the authors' setting; its three generic-object concepts do not include ours |
| **SPEED** | **yes**, `Ouxiang-Li/SPEED` | **Apache-2.0** | `--sd_ckpt`, defaults to SD 1.4, accepts SD 1.5 or a local folder | **`--anchor_concepts` is an explicit required argument** | a retain-set CSV we must author for object categories | two small pip installs; retain-set construction is ours to declare (1.4) |
| **ConAbl (± L2-SP)** | **yes**, pinned CUIG `9932ac3` | upstream | our SD 1.5 substitute `M0` | anchor always user-supplied; **no native selection mechanism** | `Horses` and `Flowers` anchor caches — **both already built, 204 images each** | none known. This is the incumbent and the only arm with a reusable request-1 pair |
| **Sculpting Memory** | authors' code exists (`coulsonlee/…`); **CUIG's `SculpMem` is a reimplementation** — its own README says it "implemented the key ideas … using the Sculpting Memory repository as the reference" | upstream | SD 1.5 | anchor supplied; dynamic mask | ConAbl-style anchor data | not a native arm as installed. Using CUIG's `SculpMem` and calling it Sculpting Memory would be the exact error the handoff forbids |
| UCE / RECE / ESD / MACE | public (MIT / Apache-2.0 / MIT / NOASSERTION) | — | SD 1.x | explicit anchor (UCE/RECE/MACE), negative guidance (ESD) | — | not needed for the minimum design; available if the PI wants breadth |

**Nothing here was inferred from a method name.** Every row rests on the fetched
file recorded in the source JSON, and the CPU checks in 1.5.

### 1.2 Concept domain — the compatibility fact that is easy to miss

Neither CEASE nor LACU was evaluated on the concepts this screen uses.

* **CEASE**: celebrities, artistic styles, and "instances". Its instance set is
  **named IP characters** — SPEED's `data/instance.csv` is 1,352 names beginning
  `Crazy Frog`, `Easter Bunny`. MS-COCO appears only as the *unrelated-prompt*
  utility set. **No COCO object category is ever erased.**
* **LACU**: 10 released concepts — `pikachu, brad_pitt, golf_ball,
  van_gogh_style, apple, spiderman, lionel_messi, cartoon_style, banana,
  mickey_mouse`. **Three are generic objects** (`apple`, `banana`, `golf_ball`),
  so `cat` / `dog` / `sandwich` are the *kind* of concept LACU handles — but not
  ones it has been shown to handle.

**Consequence for the question.** Running either method on `cat`/`dog`/
`sandwich` is off its published domain. That is acceptable *only* because the
protocol already calibrates each method's dial on DEV and qualifies it on a
held-out ACCEPT set (§3.1), which is also how LACU itself is used — its
`train.sh` carries **per-concept** `STEPS`, `LAMBDA_UNLEARN` arrays, so
per-concept tuning is the method's own practice, not a liberty we take. It must
be stated in any write-up that a weak result could be an off-domain artefact,
and that is why **no "method X fails" claim is available from this screen** —
only "at a matched end state on these concepts, this is what happened".

### 1.3 LACU's three real constraints

1. **An OpenAI API key is unconditionally required by the released runner.**
   `train.sh:ensure_retain_prompts` exits `[fatal]` if `OPENAI_API_KEY` is
   unset, *before* checking anything, and it deletes and regenerates
   `related_score.txt` for **every concept at every checkpoint** — by design,
   because retain selection is model-dependent. Mapping selection
   (`model_score_map_gen.py`) needs no API. So native LACU cannot be run offline.
   Options: **(a) obtain a key** — recommended, cost is cents, but it is third-party
   egress of our prompt text and an LLM in the loop makes the retain prompts
   non-reproducible unless we archive them; **(b) hand-author the retain
   prompts** — that replaces part of the locality mechanism and must be labelled
   an ablation, not native LACU. **(a), with every generated `related_score.txt`
   archived and hashed per stage**, is the only route that keeps the arm native
   *and* reproducible.
2. **Our GPUs are below the authors' setting.** The recommended configuration is
   3 processes × batch 18 × accum 2 (effective 108), and the README warns about
   reducing it "including 48 GB cards" — our devices are **40 GB**. `train.py`
   trains the **full UNet** with Adam (≈13.8 GB of states alone) while holding a
   frozen teacher UNet *and* running a 20-step DDIM replay pipeline at the same
   batch. Batch 18 per 40 GB device is unlikely to fit. Splitting the same
   effective batch into smaller micro-batches **does not change the dominant
   cost** (§4.1), so the mitigation is cheap in compute and the README's own
   caveat — small per-device batches "can make it hard to reproduce the same
   numbers" — must be declared as a deviation.
3. **Checkpoints are full pipelines.** Each save is a complete diffusers
   pipeline (~5 GB), written at steps 100, 200, every 200 and at the end. See
   §4.3 for the disk consequence.

### 1.4 SPEED's two declarations we must make, and must not confuse with mechanism

* **Native anchor for objects is ambiguous between sources.** SPEED's released
  `scripts/eval_few.sh` uses an **empty** anchor for the instance task
  (`anchors_map["instance"]=" ; ; "`), while CEASE describes the SPEED baseline as
  using anchor `"object"` for instances. Both are *configurations* —
  `--anchor_concepts` is a required argument — so prescribing `horse` or `flower`
  is **configuration, not a mechanism change**, and CA-SPEED is a legitimate
  native-setting arm rather than an ablation. But "native SPEED" must be pinned
  to one documented value in the pre-registration, with the other reported as
  the alternative.
* **The retain set is ours to author.** `data/instance.csv` is IP names; for COCO
  categories we must supply the retain CSV, and preservation behaviour is driven
  by it. It is a consequential *setting*, frozen before evaluation, identical
  across every closed-form arm, and recorded with a digest. It is not a
  mechanism edit, and it must not be tuned to improve a result.

### 1.5 CPU checks actually run (no model touched)

| check | result |
|---|---|
| `py_compile` on all 8 LACU `.py` and SPEED's `train_erase_null.py` + `src/*.py` | **all pass** |
| third-party imports resolvable in the **existing** `venv-cuig` (torch 2.4.0+cu121, diffusers 0.30.2, transformers 4.44.2) | **LACU training path needs nothing new**: `train.py`, `model_score_map_gen.py`, `clip_map_gen.py`, `retain_prompt_gen.py`, `losses.py`, `utils.py` all resolve. Only `eval.py` misses `pandas`, `wandb` — and we do not use their evaluator |
| LACU's declared training env vs ours | `torch 2.4.1 / diffusers 0.30.3 / transformers 4.44.2 / accelerate 1.0.1` vs ours `2.4.0 / 0.30.2 / 4.44.2` — **near-identical; no 9 GB conda env needed**, which matters because `/` is at 100% (4.9 GB free) |
| SPEED's imports | needs **`kmeans_pytorch` and `pandas`** (two pure-Python pip installs). Its declared env wants diffusers 0.32.2 / transformers 4.48.0; whether it runs on our 0.30.2 is **unverified** |
| ConAbl anchor assets for both cells | `Horses` **204** images, `Flowers` **204** images, both already cached — **no anchor generation cost** |
| sequential chaining | LACU chains natively (each step's checkpoint is the next `--pretrained_model_name_or_path`). SPEED saves only the edited `attn2.to_v` tensors as a `.pt`, so chaining needs a small CPU glue step (load the delta, `save_pretrained`). **Glue, not a method change** — and it must be tested on CPU before any run |

**Not verified, and only a GPU smoke test can:** that batch 18 or any reduced
batch actually fits in 40 GB; LACU's end-of-run save path
(`StableDiffusionPipeline.from_pretrained(model_path, unet=…)` without
`safety_checker=None`) reloading cleanly from a previous LACU checkpoint whose
`model_index.json` records no safety checker; SPEED on diffusers 0.30.2.

### 1.6 Literature correction, tied to these implementations

Revision 5 §1 says that among the six works read "none reports a stagewise
measurement of relapse against each earlier target's *own* post-erasure state".
**That needs narrowing.** CEASE's Figure 5 tracks, over a 100-edit celebrity
sequence, the accumulated **value-projection interference** on an unrelated COCO
prompt, on the shared replacement `person`, and **on the erased target "Tom
Cruise"** — i.e. it already measures an earlier erased target drifting as later
edits accumulate. Three differences remain, and they are what is left of the
gap:

1. it is a **representation-space** quantity, not image-level detection — and
   the authors themselves write that "value-projection interference alone does
   not rank preservation across methods";
2. its reference is the **unedited** `W₀`, not each target's own post-erasure
   checkpoint, so it is not the paired estimand of §6.2;
3. one shared anchor per task and a single concept domain mean **anchor sharing
   and target proximity are not varied**, which is the screen's factor.

Also to be recorded: CEASE's Tables 1–3 already baseline **CUIG-GradProj (Lee et
al. 2026 — our own codebase's paper)** and **Sculpting Memory** over 100-step
sequences on SD 1.4/1.5/2.1. Continual-unlearning baselines at scale are covered
ground. **Do not describe the stagewise-relapse idea as untouched.** The
defensible residue is the paired image-level estimand plus the two sequence
factors — narrower than revision 5 implies, and still unproven as a
contribution.

---

## 2. One explicit minimum design

### 2.1 Arms, by family

Request 1 always erases `cat`. Request 2 erases `dog` (near) or `sandwich`
(far). `A1`/`A2` are the request-1 and request-2 anchors.

**Family CA — anchor prescribed** (carries the §5.3 factorial: cells
`horse→horse`, `horse→flower`, `flower→flower`, `flower→horse`):

| arm | what prescribing the anchor is | per-method count |
|---|---|---|
| **`CA-ConAbl`** (± L2-SP as its dial) | native usage — ConAbl has no replacement-selection mechanism, so CA **is** its native configuration | 4 request-1 + 16 request-2 = **20** |
| **`CA-SPEED`** | configuration (`--anchor_concepts` is a required argument) | **10** — see 2.3 |
| **`CA-SPEED+AIC(reproduced)`** | configuration; AIC reproduced from Eq. 9 on public SPEED, **labelled a partial reproduction, never CEASE** | **10** |
| **`CA-LACU`** | **ablation in every cell** — forcing a fixed mapping prompt replaces LACU's selection mechanism | 4 + 16 = **20** |

**Family NAT — native, each method choosing its own replacement.** The cells do
not apply; this is the actual method comparison:

| arm | native replacement | per-method count |
|---|---|---|
| **`NAT-LACU`** | per-prompt score-selected mapping prompt | 2 request-1 + 4 request-2 = **6** |
| **`NAT-SPEED`** | the one documented instance anchor pinned in 1.4 | **3** |
| **`NAT-SPEED+AIC(reproduced)`** | same anchor, AIC on | **3** |
| **`NAT-ConAbl`** | **does not exist** — no native selection. Its CA cells are its native configuration | **0 (shared, genuinely identical)** |

**Counted separately, shared only where genuinely identical.** The one share is
ConAbl's: `NAT-ConAbl` is not a distinct configuration from its CA cells, so it
is not run twice and not counted twice. Nothing else is shared — in particular
`CA-LACU` and `NAT-LACU` are different training runs and are costed separately.

### 2.2 Checkpoint ownership

Each method owns its own chain: a sequence erased by method *M* at request 1 and
*M* at request 2. **A ConAbl checkpoint is not a valid request-1 parent for
LACU, SPEED or any other arm**, and no cross-method chain exists anywhere in
this design. `M0` is the single shared object: the common pretrained base, which
every method starts from and which supplies `D(M0, ·)` for `PriorΔ` and
cumulative preservation.

### 2.3 Why the closed-form arms count fewer checkpoints — and the asymmetry it creates

SPEED's edit is **training-free and deterministic** given the model, the concept
lists and the retain set. There is no training randomness for a training seed to
replicate, so `2 anchors × 2 seeds` collapses to `2 anchors`, and request 2
collapses from 16 to 8 (`2 A1 × 2 A2 × 2 targets`). Generation seeds still give
independent image samples.

**This is an asymmetry the decision table must handle, not hide.** Revision 5
§6.5 rule 1 declares the screen UNRESOLVED when "the two training seeds disagree
in direction". **That rule cannot be evaluated for a deterministic method.** The
pre-registration must therefore state: rule 1 applies to the stochastic arms
(ConAbl, LACU); for closed-form arms, run-to-run replication is **not
established by this design** and no seed-agreement claim may be made about them.

### 2.4 Primary contrasts

Per training seed, never pooled; read from interval bounds (§5.1 predicates).

| # | contrast | what it identifies |
|---|---|---|
| 1 | `Relapse_i(k; R_i)` per arm (§6.2) | did `cat` come back at request 2, measured against that arm's own post-`cat` checkpoint on one probe |
| 2 | `S` = shared − disjoint, averaged over anchor identity | anchor-sharing main effect **on these concepts** |
| 3 | `P` = near − far | target-proximity main effect **on these concepts** |
| 4 | `I` = `(S | near) − (S | far)` | the interaction — **requires all four cells to survive** (§3.4) |
| 5 | `S − P` / `P − S` | dominance; never inferred from comparing two predicates |
| 6 | `NAT-LACU` vs `NAT-SPEED+AIC(reproduced)` at a matched residue | whether a strong native remedy already removes the relapse |
| 7 | `CA-LACU` vs `NAT-LACU` | what disabling LACU's selection costs — **reported as an ablation contrast, never as "LACU"** |
| 8 | retained categories, non-target control, each anchor — stagewise **and** cumulative | preservation (§5.2) |
| 9 | `PriorΔ_k = D(M0,t_k) − D(C_{k−1},t_k)` | prior collateral suppression, independent of the parent floor |

### 2.5 Conditional ConAbl request-1 reuse, and the stop if it fails

Retained from revision 5 §3.2 unchanged. The existing `horse` MA pair may serve
**only** as `CA-ConAbl`'s `horse` request-1 checkpoint, and **only if it
QUALIFIES** on the new ACCEPT and TEST sets (`p₁ ≥ 65%`, relative suppression
`≥ 0.75`, `r₁^ACCEPT ≤ 20%`). Its historical `cat` residue on the **spent** probe
was 30.00% (s17) and 23.75% (s29) — above the 20% cap on *that* probe, which is
neither a prediction nor a disqualification, and `p₁ = D(M0, cat)` has never been
measured on the new sets.

**If it fails qualification** (default, needs no further approval): report
request-1 non-qualification with the measured `p₁` and `r₁^ACCEPT`, **train the
`horse` request-1 pair afresh under the frozen protocol like every other arm**,
and record that reuse was attempted and refused. Cost of the stop: **+2 ConAbl
request-1 trainings ≈ +0.5 device-hours** — the reuse saves almost nothing in
compute, so nothing in the design depends on it. **No threshold is relaxed and
nothing is retrained to meet one.** If instead a method has *no* qualified
request 1 at all, its sequences do not run, it contributes no relapse data, and
if that leaves fewer than two accepted arms the comparison does not exist
(§6.3/§6.5 rule 0).

### 2.6 Retained limits

**Still exploratory over concepts.** One near pair and one far pair: proximity
is confounded with concept identity, two training seeds replicate the run and
not the concept, and **no mechanism claim and no generality over concepts
follows from this screen**. The §5.3 pre-registered conditions for independent
concept evidence (≥ 5 near and ≥ 5 far pairs from disjoint supercategories,
externally validated proximity) are unchanged and are **not** part of this
design. **The stop at §6.5 rule 2 is retained**: if a native method
ALREADY-SOLVES its own sequences, the screen reports that and stops, and no
method work follows.

---

## 3. Execution semantics and decision value

### 3.1 Who touches which set, when

| set | touched by | when | may be looked at again? |
|---|---|---|---|
| **DEV** | dial calibration only | before each stage's dial is frozen | yes, freely |
| **ACCEPT** | qualification of the **frozen** dial | once per stage, after freezing | no retuning, ever |
| **TEST** | arm acceptance, the cross-arm residue match, reported retention endpoints | **once**, after ACCEPT has decided advancement | **no** |
| **R** | relapse only, both operands | every stage | yes — it is a reporting set |

### 3.2 "TEST once", operationally

1. The dial is calibrated on DEV and **frozen**; the freeze is recorded with a digest.
2. ACCEPT decides, with the frozen dial, whether the stage advances.
3. **The list of checkpoints to be TESTed is written into the hashed
   pre-registration before any TEST image is generated**, and is derived from
   DEV and ACCEPT decisions alone.
4. TEST runs exactly once per listed checkpoint; outputs are appended to a
   write-once file whose digest is recorded.
5. **No checkpoint is added to the list afterwards, no TEST result selects among
   checkpoints, no dial is re-fit after a TEST reading, and no TEST evaluation is
   repeated.** A re-run for any reason is a protocol breach: both results are
   reported, and the comparison is labelled accordingly.

That is the whole content of "TEST once": *predeclared list, frozen dial,
one pass, append-only, no selection*. It is enforceable by a CPU regression
(deliverable 6).

### 3.3 The `R_i` operands and the parent measurements required

`Relapse_i(k; R_i) = D_{R_i}(C_k, t_i) − D_{R_i}(C_i, t_i)` — **both operands on
the same probe `R_i`**, that method's **own** post-erasure checkpoint `C_i` as
the reference, paired on identical frozen prompt texts, generation seeds,
detector and threshold, and the same frozen cluster grouping, so the bootstrap
resamples one cluster list indexing both checkpoints. `r_i^TEST` keeps
acceptance and the cross-arm match and is **never** an operand.

Measurements that must exist for the design to be readable at all:

| quantity | on which set | why |
|---|---|---|
| `D(M0, t_k)` for `cat`, `dog`, `sandwich` | ACCEPT, TEST, **R** | `p₁`, `PriorΔ`, cumulative preservation. One checkpoint, no training, but it must be measured **on each set separately** |
| `D(C_{k−1}, t_k)` | ACCEPT, TEST | the parent rate `p_k` — per method, per chain, never borrowed |
| `D(C_i, t_i)` | **R** | the relapse reference. Measured **at stage *i***, not reconstructed later |
| `D(·, c)` for bird, chair, bicycle, the non-target control, horse, flower | ACCEPT/TEST (stagewise **and** vs `M0`) | §5.2 preservation, each reported separately, never averaged |

### 3.4 What remains available after `R-NONTRANSFER`

If `D_{R_i}(C_i,t_i)` misses the operating point though `r_i^TEST` passed, the
stage is `R-NONTRANSFER`: relapse is computed but **descriptive only**, the
tolerance is not widened, `R_i` is not reselected, the dial is not re-fit and
nothing is retrained.

**Still available from that arm:** deletion on TEST, the cross-arm residue
match, every retention and anchor endpoint stagewise and cumulatively,
`PriorΔ`, and ALREADY-SOLVES *exclusion* (§6.4 forbids crediting it).
**Unavailable:** any relapse contrast involving it — therefore `S`, `P` and `I`
lose that cell.

**The fragility to state plainly: `I` needs all four cells.** One
`R-NONTRANSFER` cell reduces the screen to main effects at best; two in the same
sharing level remove the sharing contrast too. §6.6 already stops the screen if
*every* accepted stage is `R-NONTRANSFER`; the intermediate cases are where the
design is thin, and the pre-registration must say that an `I` computed from
fewer than four cells is **not reported as `I`**.

### 3.5 Precision — estimated from existing measurements, not guessed

From the **spent** Amendment 01 TEST probe (80 images per category = 10 scene
clusters per category, conditional paired cluster bootstrap, 10 000 draws),
measured 95% interval widths:

| quantity | median width | range |
|---|---|---|
| paired same-parent contrast (arm vs arm, one category) | **10.0 pp** | 3.75 – 13.75 |
| paired different-parent contrast (`vs MA`) | 7.5 – 13.75 pp | 3.75 – 20.0 |
| marginal rate `D(·, c)` | 22.5 – 25.0 pp | 3.75 – 35.0 |

**Implication for the decision rules** (assumption: half-width scales as
`1/√n`; the relapse rate's variance may differ from the bird contrast's, and
these come from a spent probe — so this is an estimate, explicitly marked):

* a **contrast** at 80 images/cell has half-width ≈ **5 pp**;
* the **interaction** `I`, a difference of two such contrasts, has half-width
  ≈ **7 pp**. At `Δ = 10 pp` that means `I` can only read **MATERIAL** if the
  true effect is ≳ **17 pp**, and only **NULL(10)** if it is ≲ **3 pp**.
  **Everything between lands at rule 8, INCONCLUSIVE.**
* to bring `I`'s half-width to ≈ 5 pp needs ≈ **160** images per cell; to ≈ 3.5 pp,
  ≈ **330**.

**Recommendation:** adopt **160 images per cell** (20 prompts × 8 generation
seeds, or 40 × 4) for ACCEPT/TEST/R rather than the pilot's 80, and state in the
pre-registration that `I` is **exploratory and width-reported** regardless. This
is a precision parameter the proposal never fixed, not an expansion of the
design, and §4 costs it.

### 3.6 Which outcome decides what

Evaluated in the strict precedence order of §6.5. Mapping to the PI's decision:

| outcome | decision |
|---|---|
| **rule 0** — qualification failure, < 2 accepted arms, or a REJECTED audit packet | **no result.** Report the failure; nothing else is read |
| **rule 2** — a NAT arm (`NAT-LACU` or `NAT-SPEED+AIC`) **ALREADY-SOLVES** its own sequences: every stage accepted, relapse NULL(10), retention and anchors NULL(10) stagewise *and* cumulatively, no LOW-PARENT, no MATERIAL+ `PriorΔ`, no `R-NONTRANSFER` | **redirect the question.** The limitation is not here. No method work. Report and stop |
| **rule 3 `I` MATERIAL**, or **rule 5 `S` MATERIAL with `S − P` MATERIAL**, or **rule 6** the mirror — *and* a strong native arm shows MATERIAL+ relapse in the implicated cells | **justifies pursuing a distinct method**, aimed at the implicated factor. Still exploratory over concepts, still no mechanism claim |
| **rule 4** — `S` and `P` both MATERIAL | both factors matter; **no single remedy indicated.** A method proposal would need to say which it addresses |
| **rule 7** — `S`, `P`, `I` all NULL(10) | **sequence structure is not the lever in these cells.** Report and do not build |
| **rule 8** — any load-bearing contrast INCONCLUSIVE | **inconclusive.** Report widths and the prompt count needed (3.5). Do not read point estimates, do not substitute an indirect comparison |
| **rule 1** — the two training seeds disagree in direction (stochastic arms only, 2.3) | **UNRESOLVED.** Cost more seeds; do not report the agreeing seed |

**No new target-recognition measurement was produced for this brief.** Every
rate quoted is from Amendment 01's existing rows; all forward-looking numbers
are marked as estimates with their assumption stated.

---

## 4. Workload, wall time and human validation

### 4.1 Measured unit costs (from the Amendment 01 ledger — 56 entries, 2.9014 device-hours)

| item | measured | basis |
|---|---|---|
| generation + COCO detection, 560-image slot | **1.10 s/image** (609–625 s per slot) | 6 frozen-test slots |
| same, 80-image slot | **1.36 s/image** (103–113 s) | 20 development slots; ~25 s fixed startup dominates |
| **ConAbl** training, `kv-xattn`, batch 4 | **0.84 s/step** (842 s / 1000 steps); 1.0 s/step at 100 steps | `train:U_seed17/29` |
| anchor caches | **already built** — `Horses` 204, `Flowers` 204 images | on disk |

### 4.2 Estimates, with the assumption stated

**`SPEED` / `SPEED+AIC`:** closed-form, **seconds per request** (authors: 100
concepts in 5 s; CEASE reports ~13 s/request on an H100). Take **0.01
device-hours per checkpoint**; its cost is entirely evaluation.

**`LACU`:** generation-dominated. Per optimizer step, the replay and unlearn
latents are generated by the teacher at 20 DDIM steps with guidance 1.0, so
total UNet passes per step = `effective_batch × 2 × 20` = **4,320** at the
authors' effective batch of 108 — **invariant to how that batch is split into
micro-batches**, which is why reducing `TRAIN_BATCH_SIZE` for 40 GB costs memory
headroom, not compute. At 250–350 steps that is **1.08–1.51 M UNet passes per
request**. Converting at our measured generation rate (one of our images = 30
steps × 2 for CFG = 60 UNet passes ≈ 1.0 s at batch 1) gives 18.0–25.2 k
image-equivalents ≈ 5.0–7.0 device-hours at **batch-1** throughput; allowing a
2–4× batched speed-up gives **1.2 – 3.5 device-hours**. Add the per-stage mapping
rescore (100 prompts × 10 candidates × `MAP_K`=32 ≈ 35 k passes ≈ **0.15 h**) and
the retain trajectory rescore (`RETAIN_K`=16 over ~36 related concepts,
**unmeasured**). **Use 1.5 – 4.0 device-hours per LACU request, estimate, pending
one smoke measurement.**

### 4.3 The plan, costed

Set sizes assumed (3.5): **160 images per category-cell**; 8 categories reported
on TEST (3 targets, 3 retained, 1 non-target control, 2 anchors — horse and
flower counted among the 8).

| family · arm | checkpoints | training device-h | note |
|---|---|---|---|
| CA · `ConAbl` (±L2-SP dial) | 20 | **4.7** | measured rate at our 1,000 iterations — **9.4 h** at upstream's 2,000; −0.5 h if the MA pair qualifies |
| CA · `SPEED` | 10 | **0.1** | deterministic (2.3) |
| CA · `SPEED+AIC(reproduced)` | 10 | **0.1** | |
| CA · `LACU` (ablation) | 20 | **30 – 80** | estimate |
| NAT · `LACU` | 6 | **9 – 24** | estimate |
| NAT · `SPEED`, `SPEED+AIC` | 6 | **0.1** | |
| **training subtotal** | **72** | **≈ 44 – 109** | LACU is ~90% of it |

| evaluation | images | device-h |
|---|---|---|
| TEST, once per checkpoint: 8 categories × 160 | 1,280 | 0.39 each → **≈ 28** |
| ACCEPT, once per stage: target + 2 anchors × 160 | 480 | 0.15 each → **≈ 11** |
| R, every stage: earlier target × 160 | 160 | 0.05 each → **≈ 3.6** |
| DEV calibration, stochastic arms only (~46 stages × 3–6 probes × 80) | 11–22 k | **≈ 3.5 – 7** |
| `M0` baseline on all four sets | ~2,100 | **≈ 0.7** |
| **evaluation subtotal** | | **≈ 47 – 50** |

> **Total ≈ 90 – 160 device-hours.** Wall time: LACU's recommended runner
> occupies **3 GPUs per run**, so LACU's 39–104 device-hours are **13–35 hours of
> 3-GPU wall time**; with everything else and ordinary serialisation, **3 – 7 days
> of occupancy** if uninterrupted. There is **no GPU-hour cap**, and this number
> is a plan input, not a request for a cap. Any approved compute must run on
> devices **independently verified unoccupied by another user**
> ([`RESOURCE_POLICY_2026-10-10.md`](RESOURCE_POLICY_2026-10-10.md)). *Status at
> the time of writing, not an approval: all four A100-SXM4-40GB read 14 MiB used,
> 0% utilisation.*

**Disk is a real constraint.** LACU writes a **full diffusers pipeline (~5 GB)**
at steps 100, 200 and the end — ≈15 GB per request, ≈**390 GB** over 26 requests.
`/` has **4.9 GB free (100% used)**; `/data` has **1.2 TB free**. So: everything
under `/data`, intermediate dumps pruned to the DEV-selected one plus the final
after each stage's evaluation, and the running total checked before each stage. A
UNet-only save would cut it to ~3.4 GB per dump, but that is a change to a
third-party script and would have to be declared.

### 4.4 Human validation — unchanged, on the critical path, and not a model action

* The detector is a **proxy**. UnlearnCanvas classifiers remain unobtainable on
  this host; no UA / IRA / CRA. **No human annotation has been performed anywhere
  in this project** — 0 of 180, 0 of 228 and 0 of 236 label cells are filled.
* The **180-item paired packet stays blank** until a person labels it. It must go
  to an annotator who has seen **neither earlier packet nor any source image**
  (13 of its 180 images overlap the 228-item packet, which is unblinded on two
  channels), on the terms in
  [`BLINDING_AUDIT.md`](../../results/audit_v1/annotation_packet_paired180/BLINDING_AUDIT.md).
* **Nothing in this screen's detector numbers is a finding until that labelling
  exists**, because the screen's endpoints are detector rates. This is
  independent of any GPU approval and is the one critical-path item the model
  cannot advance.
* The screen's **own** packet is designed on the same terms and must pass the
  delivery gate before its results are read (§6.7, §6.5 rule 0).
* **All originals, packets, keys and labels are preserved.** The packet has not
  been sent, no label has been supplied, and no person has been scheduled —
  those are the PI's to instruct.

---

## 5. What is being proposed for approval

**Blockers, verified:**

1. **CEASE has no code** (project pages only; "Code coming soon."; no licence, no
   date). Faithful reimplementation is additionally impossible: HOC's trigger and
   rank cut are unspecified. → the screen as written stops at §6.6.
2. **Native LACU requires an OpenAI API key** by its released runner's design,
   per concept per checkpoint.
3. **Our 40 GB devices are below LACU's recommended setting**, so its batch
   configuration must be changed and the deviation declared.
4. **SPEED needs `kmeans_pytorch` + `pandas`**, and its behaviour on our
   diffusers 0.30.2 is unverified.
5. **`/` is full (4.9 GB)**; LACU's full-pipeline checkpoints need ≈390 GB, which
   only `/data` can hold, with pruning.
6. **No human label exists anywhere**, so no detector-based result from this
   screen would be a finding yet.

**Work proposed, in order. Each item needs explicit approval; none is started.**

| # | work | compute | gate |
|---|---|---|---|
| 1 | CPU glue + dry runs: SPEED delta → diffusers folder round-trip, LACU arg wiring, a fixed-mapping CSV for `CA-LACU`, the retain-CSV construction for the closed-form arms | **CPU only** | must pass before any GPU request |
| 2 | **one LACU smoke calibration**: a single short request at a reduced batch to measure device-hours/step, peak memory and the save path | **≈1–3 device-hours**, 3 GPUs verified idle | replaces the 1.5–4 h estimate with a measurement; if it OOMs at every workable batch, **report and stop** |
| 3 | revision 6 of the proposal + the **hashed pre-registration**: §2.2 operating point, the four set identities and sizes at 160/cell, the frozen cell list, the arm/ablation labels **fixed in the code that names the arms**, the predeclared TEST checkpoint list, the bootstrap grouping | **CPU only** | the freeze; nothing may be tuned afterwards |
| 4 | the **reduced screen** of §2: 72 checkpoints, ≈90–160 device-hours | GPUs verified unoccupied | §6.6 stopping rule live throughout |
| 5 | `SPEED+AIC(reproduced)` — AIC from Eq. 9 on public SPEED, ~tens of lines, labelled a partial reproduction and never called CEASE | CPU to write, GPU only inside item 4 | **method implementation: requires its own approval** |
| 6 | the CPU regressions that fail the build on a protocol violation: two relapse operands from different sets (the revision-4 defect, unchanged-checkpoint fixture), a relapse number from a LOW-PARENT / ERASURE-FAILURE / `R-NONTRANSFER` reference, a comparison from fewer than two accepted arms, a skipped cumulative check, §6.5 rules out of order, an `I` from fewer than four cells, a TEST re-evaluation or a late addition to the TEST list | **CPU only** | |
| 7 | **human labelling of the 180-item packet** by an unexposed annotator | none | **PI action.** Not scheduled, not initiated |

**If the PI prefers to wait for CEASE instead:** items 1, 3, 6 and 7 are all
still worth doing, cost no GPU time, and leave the screen ready to run the day
code appears. That is the cheapest defensible option and it does not depend on
this brief's estimates being right.

---

## 6. What this brief does not establish

* **Not that the screen will find anything.** §3.5 shows a wide band of true
  effects that would land INCONCLUSIVE even at 160 images per cell.
* **Not novelty, not publishability, not a gap.** §1.6 narrows the gap claim:
  CEASE already tracks representation-space interference on an earlier erased
  target and already baselines CUIG-GradProj and Sculpting Memory over 100-step
  sequences. What is left is the paired image-level estimand plus two sequence
  factors — and that it is left is not evidence that it matters.
* **Not an exhaustive survey.** Literature claims are scoped to the six works of
  §1 plus the three implementations audited here.
* **Not that any arm will fit, run or converge on this host.** Items 1 and 2 are
  what would establish that; §1.5 lists exactly what remains unverified.
* **Not a mechanism, a proof, or generality over concepts.** The §7 candidate
  stays an unproved direction with its three unproved assumptions and the missing
  bound named.
* **Not independently verified where it is ours.** The upstream facts are
  re-checkable from the recorded URLs and digests; the local environment,
  disk, GPU-status and ledger observations are **this project's own
  observations**, reported as such.

---

## 7. Reproduce

    python scripts/seq/check_feasibility_brief_numbers.py     # 54 checks, 0 failures

Every measured figure, interval width, checkpoint count and device-hour subtotal
in §3.5 and §4 is re-derived there from the committed ledger
(`results/audit_v1/amendment01_evidence/gpu_usage_record.json`) and the committed
analysis (`…/analysis_test.json`), and asserted against the number this document
prints. Verbatim output:
[`brief_numbers_check.log`](../../results/feasibility_2026-10-11/brief_numbers_check.log).

Upstream facts: refetch any URL in
[`feasibility_sources_2026-10-11.json`](feasibility_sources_2026-10-11.json) and
compare its `sha256`. A changed digest means upstream moved — most usefully, the
two CEASE repositories, where a changed tree is the signal that §0's tripwire has
fired.

**Nothing in this brief reruns, reopens or amends Amendment 01.** It reads its
committed evidence and adds files of its own.
