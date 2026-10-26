# Upstream reference implementations — current status

> **Status pointer:** for the **one current verdict per issue**, read `STATUS_CURRENT_v2.md`. `AUDIT_REPAIR_REGISTER_v1.md` is chronological and records hypotheses that were later refuted — two of them ("IMMA composition unproven", "upstream UCE may have written an unedited model") are **refuted** and must not be quoted as live.

> **⚠ Corrections 2026-10-26:** `FINDING76` §3's probe mechanism check is **overstated** (n=7/3/1, within-arm spread 40×, ordering undetermined — see its §7). `FINDING77` §2's "B kills the accumulation-order hypothesis" is **wrong** — B holds accumulation order fixed and never varied it (see its §6).


**Date:** 2026-10-26 · **Source/config commit:** `5e222fc26c43fae14294516db588f932c0394a65`
**Companion:** `UPSTREAM_FIDELITY_MANIFEST.md` (the full ledger) ·
`outputs/upstream_manifest.json`

**No method is reproduced, and nothing below is an official reproduction.** Two cells have now run
upstream code (R1, R4) — but both were *scored on our own constructed benchmark*, which is a
different thing from reproducing the paper's reported experiment. §1.1 states the three fidelity
tiers this document uses. Nothing below rests on a unit test, and no adaptation's failure is read as
refuting a published method.

---

## 1. Per-method status

**Verified from artifacts and file mtimes on 2026-10-26, not from the previous revision of this
table, which was wrong in eleven cells.** Evidence column cites the primary artifact.

| method | source verified | env ready | training/edit done | eval done | evidence | tier |
|---|---|---|---|---|---|---|
| **IMMA** | ✅ `2f55041`, Apache-2.0 | ✅ `venv_imma` | ✅ **R1 ran, 5000/5000 steps** | ✅ **R1c scored** | `logs/r1b_imma_train.log`, `upstream_runs/imma/REF_imma_vangogh/imma_unet_xatten_layer.pt` (12:21), `logs/r1c_eval.log:32`, `outputs/gen_REF_imma_vangogh.json` | **upstream code / our benchmark** |
| **Meta-Unlearning** | ✅ `5a2a760`, no license | ✅ `venv_t2i`-class | ⏸ **R2 partial — pools built, `src/meta.py` never invoked** | ❌ | `logs/r2_meta.log:10,45-…`, `upstream_runs/meta/dataset/{hrm,irt,tgt}/{train,test}` | pending |
| **WDR** (From Dormant to Deleted) | ✅ `7b9a1cc`, **README declares MIT** (no LICENSE file) | ⚠️ needs `wandb` + CIFAR | ❌ **R3 not started** | ❌ | no `logs/r3*`, no `outputs/r3_*` | **domain adaptation** only |
| **UCE** | ✅ `7c724d9`, MIT | ✅ `venv_uce` | n/a (closed-form) | ✅ **R4 executed in four phases; upstream's edit confirmed real, the port-side comparison still confounded** | `outputs/r4_uce_*.json` (**VOID**), `r4b_uce_lamb*.json` (11:46), `r4c_uce_lamb*.json` (12:25), `r4d_uce_lamb*.json` (12:45) | **upstream code / our benchmark** |
| **FRAG/FRP** | ✅ repo exists, **0 .py files** | n/a | n/a | n/a | `outputs/upstream_manifest.json` `FRAG.python_files: 0` | **no implementation released** |
| **GIFT** | ❌ none found | n/a | n/a | n/a | — | **no official implementation** |

### 1.1 Three fidelity tiers, because two are not enough

| tier | meaning | what qualifies |
|---|---|---|
| **official reproduction** | authors' code **and** authors' protocol, metric and data | **nothing in this project** |
| **upstream code / our benchmark** | the authors' released code executed faithfully, then measured by **our** `gen_eval`, **our** 12-label closed-set CLIP scorer and **our** prompt pools | **R1** (IMMA), **R4b–d** (UCE) |
| **our reimplementation / adaptation** | our code, from the paper or adapted across domains | every historical arm; Meta v2; WDR v2; UTRAP; FRP; GIFT |

An upstream-code run on our constructed benchmark **is not** a reproduction of the paper's reported
experiment, and is not reported as one anywhere in this document.

## 2. What actually ran

**Completed:**
- All five repositories cloned and pinned; three audited revisions resolved **exactly** as requested.
- Licences recorded. **Meta-Unlearning and vision_relearning ship no LICENSE**, so `upstream/` is
  git-ignored and no upstream source is redistributed — only our wrappers, with upstream referenced
  by hash.
- **FRAG rechecked:** repo is non-empty but has `README.md` + 2 PNGs and **zero Python files**.
  "Forthcoming" still holds. This is the one method for which `ERRATUM_r2` C19's "no released code"
  was correct.
- Isolated environments built. **First attempt failed and the errors are recorded verbatim**
  (manifest §5.1): unpinned installs pulled torch 2.14.1 / huggingface-hub 1.33.0, giving
  `ImportError: huggingface-hub>=0.23.2,<1.0 is required` and
  `AttributeError: module 'torch' has no attribute 'accelerator'`. Fixed with a constraints file and
  `--no-deps`. **`venv_t2i` was never modified** — every env is a copy.
- **IMMA fidelity gap found and quantified** (§3 below).
- **`PREREG_H90` compute accounting repaired** and a sample-matched series launched.
- **R1 ran end to end.** ESD checkpoint downloaded (3.44 GB, 11:36), 100 training images generated
  (11:39–11:43), `defend_text_to_image_lora.py` trained **5000/5000** steps (11:59→12:21) and saved
  `imma_unet_xatten_layer.pt`; `r1c_eval` composed base+ESD+IMMA (`overlaid 160/160 IMMA tensors`)
  and scored it (12:39→12:44).
- **R4 ran in four phases**, the first of which is self-flagged **VOID** (§3).
- **R2's dataset blocker is discharged** — the three pools exist; the trainer has not been called.

**Nothing is in flight.** `E4m`/`E4mg` finished 11:40, `E2g` 11:29, `E2h` 12:38, `R4d` 12:45,
`R1c` 12:44. `ps` shows no project process running, and `FREEZE_ACTIVE` (12:43) bars new GPU work.
GPUs 2–3 were left to other users throughout.

## 3. Fidelity findings that change how existing results read

**IMMA — our scope is less than half the reference's.** Measured on the real 859.5M UNet:

| scope | params | % |
|---|---:|---:|
| reference `['to_q','to_v','to_k','to_out']` over all named parameters | **93.5M** | 10.88% |
| — of which **attn1 (self-attention)** | **49.6M** | 5.77% |
| — of which attn2 (cross-attention) | 44.0M | 5.11% |
| our `erase_imma.py` (`.attn2.` only) | **44.0M** | 5.11% |

The reference immunises **self-attention as well**, which our port excludes entirely. Our docstring
("cross-attention only, per the paper") is true to the *paper* and false to the *code*. Our IMMA arms
are therefore a **project reimplementation** whose numbers are not substitutable for the reference's.
Also: the reference optionally **re-initialises LoRA every step**; ours warm-starts and never does.

**Meta — E2 is ours, not official.** `meta_unlearn_v2.py` restores Eq. 8's signed structure but
**two departures remain**: the reference uses **three** data pools (`hrm` / `irt` / `tgt`) with
independent γ coefficients where v2 uses two under one ζ; and `--fix_timesteps` is unimplemented.
E2's results are labelled accordingly.

**WDR — domain adaptation, and parity ≠ optimality.** Any SD result is an adaptation of a
vision-classifier method. `FINDING67`'s α ≈ 4.0e3 parity figure bounds where the term *can* act;
**under Adam the update is preconditioned and per-coordinate normalised, so parity is not a claim
about the best coefficient.** A native-domain sanity check (R3) must precede interpreting the
diffusion adaptation. Neither our wrong-direction WSR-anchor arm nor the single α=4000 failure
refutes the published method.

**UCE — R4 executed; upstream's edit confirmed real, the port-side comparison confounded.** The `--lamb` default mismatch is settled
and unambiguous: the official default is **0.5** against our **0.1**, so "UCE at defaults" denotes
different edits in the two codebases, and **every historical arm of ours used 0.1**
(`outputs/r4b_uce_lamb*.json` configs run both).

The *magnitude* question is measured but **not yet interpretable**:

| phase | artifact | what it shows |
|---|---|---|
| R4 | `r4_uce_*.json` | **`VOID: true` — DO NOT CITE.** Inputs were char-split (`list('church')` → six single letters, visible as `n_erase_prompts: 6`); the embedding selection matched neither side; **no upstream code was executed**. `FINDING75`, which rested on it, is retracted. |
| R4b | `r4b_uce_lamb{0.1,0.5,1000.0}.json` | Upstream **executed** (`executed: true`; the shim sets only the module-level `device`/`torch_dtype` that upstream binds inside `__main__`, `UCE()` body unmodified). Full-pipeline divergence max **0.384 / 0.322 / 0.108**; with **upstream's own embedding selection** it collapses to **1.5e-3 / 3.0e-4 / 1.3e-7**. |
| R4c / R4d | `r4c_uce_lamb{0.1,0.5}.json`, `r4d_…` | Checkpoints + generations, `preserve_concepts` present and empty. λ=0.1 label agreement **0/15**; λ=0.5 **2/15**. |

So the divergence localises to **embedding selection**, not the solver — consistent with
`UPSTREAM_FIDELITY_MANIFEST.md` §7(d), which read it from source independently.

**The "upstream wrote an unedited model" explanation is REFUTED.** At λ=0.1 and λ=0.5 the
**upstream** arm scores target 15/15, and the earlier revision of this document read that as
consistent with upstream's edit having been **inert**. It is not. Measured against the pinned base
upcast to fp32, across all **32** `attn2` K/V matrices:

| checkpoint | vs BASE max | vs BASE mean | bitwise identical to base |
|---|---:|---:|---:|
| `upstream_church_lamb0.1` | 9.7948e-02 | **5.3620e-02** | **0 / 32** |
| `upstream_church_lamb0.5` | 9.7899e-02 | **5.3593e-02** | **0 / 32** |

Upstream changed **every** matrix, by **5.36 %** of weight norm on average and **9.79 %** at
maximum, with **0 of 32** identical to base; cross-attention K/V **output** for
`"a photo of a church"` moves **43.4 %**. **No "upstream" number in R4b/R4c/R4d is measuring the base
model.** A 1.35× gap in activation space (43.4 % upstream against 58.7 % for our port) separates
15/15 from 0/15 in CLIP label, so **behaviour is not proportional to the numerical gap** — which
limits every weight-norm-based behavioural inference in `FINDING75` and `FINDING77`.

**What remains open is a different question:** whether upstream should have been invoked with an
expanded erase-prompt set (`expand_prompts=true` raises dW/W to 0.0925, above our port's 0.0587,
while `erase_scale` 1→10 and the `lamb`/guide-string levers are inert). Separately, the **port** side
of R4c-vs-R4d is confounded by checkpoint caching on an incomplete key (`D-026`), so **R4d must be
re-run** before any port-side claim; `D-031` records that R4d also overwrote R4c's upstream
checkpoint, so R4c is no longer reproducible from disk. Our UCE numbers remain a **project
reimplementation** on their own prior grounds (§7(d)'s source-level selection mismatch).

## 4. Declared experiment matrix — actual state

| id | method | arms | seeds | budget | status (artifact-verified 2026-10-26) |
|---|---|---:|---:|---:|---|
| R1 | IMMA at an author-documented **training** setting, **our** evaluation | 1 | 3 | ~4 GPU-h | ✅ **RAN.** Build + eval complete; the **3 attack seeds were never run** (frozen) |
| R2 | Meta `src/meta.py`, three-pool **structure**, our constructed pools | 1 | 3 | ~6 GPU-h | ⏸ **PARTIAL.** Three pools built; `src/meta.py` **never called** |
| R3 | WDR native-domain sanity check | 1 | 1 | ~2 GPU-h | ❌ **NOT STARTED** — the only cell for which the original "blocked" label still holds |
| R4 | UCE upstream vs port, matched inputs | 3 λ × 4 phases | — | ~0.3 GPU-h | ✅ **EXECUTED.** Upstream's edit **confirmed real**; the **port side is confounded** and **R4d is a required re-run** (`D-026`), with R4c no longer reproducible (`D-031`) (§3) |

R1's label is deliberately *not* "author protocol". The authors' **code** and their released ESD
checkpoint were used; the 100 training images were **self-generated**, and the measurement is our
`gen_eval` / 12-label closed-set CLIP scorer on our DEV prompts. That is tier 2 of §1.1.

R2's label is deliberately *not* "its own three-pool data". `scripts/r2_build_meta_pools.py` built
the pools from **our** prompt lists — `a photo of a church` (hrm); cathedral / chapel / basilica
(tgt); fridge, door, firefly, oven, potted plant, coffee table (irt) — 30 train + 30 test each. The
three-pool **structure** is the paper's; the **data** is ours.

**R1's recorded outcome, stated without interpretation.** `outputs/gen_REF_imma_vangogh.json`:
target(`van_gogh`) **0.8667**, all-non-target **0.0424** (n=11), legacy_retain8 **0.0333** (n=8);
eleven of twelve concepts score ≤ 0.20.

**The "checkpoint composition unproven" explanation is REFUTED. The composition is valid.**
Replicating `r1_eval_imma.py`'s matching loop offline gives **exact=160, fuzzy=0, reshaped=0,
skipped=0** — the fuzzy/reshape path never fires. Auditing `adapters/REF_imma_vangogh/unet.pt`:
**160/160** IMMA tensors byte-equal to `half(IMMA)`, **526/526** others byte-equal to `half(ESD)`,
0 unexplained, 0 non-finite, exactly 160 tensors differing from ESD (80 `attn1` + 80 `attn2`, IMMA's
own trained scope), and the **ESD delta is not inverted**. The composition is byte-semantically
identical to upstream's own (`eval/text2image.py:65-69`,
`train_text_to_image_lora.py:458-462`).

**`outputs/gen_REF_imma_vangogh.json` nonetheless stays quarantined, for a different reason:** the
closed-set 12-way CLIP argmax is **inapt** for this record (eight non-targets at *exactly* 0.000 with
the target at 0.867 is a single-label collapse of the metric, not a spread around 1/12 chance), and
**the configuration is not one the authors report** — IMMA never evaluates an immunised model
standalone (`upstream/IMMA/README.md:146-151`). It is therefore **not citable as an IMMA result** and
specifically not as "IMMA destroys utility", because the measure and the configuration are inapt and
**not** because the checkpoint is malformed.

The live candidate explanation is now the **ESD checkpoint**: all **686** tensors of
`upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt` differ from pinned base SD-1.4, resnet blocks
reaching 0.20 relative, so it is not the cross-attention-only erasure "ESD-x1" denotes (`D-019`).
The decisive control — scoring the ESD checkpoint **alone** on the same 12 DEV concepts — **has not
been run**. Until it exists, **no statement about where the utility loss comes from is supportable**,
and **no conclusion about IMMA-the-published-method follows from this cell**, in either direction.
Separately, our IMMA *training* run used 5000 steps rather than the documented 1000 (2500 ascent
steps against the README's 500) and 3 of its 100 training images are NSFW-blanked black frames
(`D-021`).

Stopping rules: 3 attack seeds at lr 1e-5 / 300 steps with bounded intermediate budgets (25, 100);
no escalation; arms failing K1 are not attacked and are reported as utility failures. All attempted
attacks reported with gate outcomes and **both gated and ungated** scores, per-seed, with explicit
denominators. `retain8`, all-non-target utility and the paired panel reported **separately**.

**Final-prompt discipline:** the existing FINAL set **has been inspected and its scores are
defective.** Every FINAL number was computed with `gen_eval.LAB_KEY` (`gen_eval.py:50`) instead of
`LABELS` (`:48`) — **nine of the twelve label strings differ; `church`, `parachute` and `tench` are
spelled identically** (`D-022`). **The evaluator used inconsistent label text. The effect on scores
and conclusions is unresolved pending corrected evaluation** — for `FINDING65`'s depth ratios and
`FINDING66`'s ceiling conclusion alike; neither the direction nor the size of the embedding effect is
established. R1–R4 are scored through `gen_eval.py`'s `LABELS` path, so **no R-cell number was scored
with the inconsistent text**. The set remains consumed either way (`FINDING65`, `FINDING66`)
and is development data from here on. It will not be reused for tuning and called untouched. A fresh
final set must be frozen before any new configuration is selected against it; R1–R4 select no
configuration, so none is needed yet.

## 5. Remaining blockers, exactly stated

Three of the five rows in the previous revision described blockers that had **already been
discharged** when it was written. Corrected:

| id | blocker | state |
|---|---|---|
| **R1** | *Was:* needs ~100 generated training images, an **ESD-erased** SD-1.4 checkpoint via a third-party download from `erasing.baulab.info`, and `accelerate launch` config. | **ALL DISCHARGED.** `upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt` (3,438,307,109 B, 11:36); 100 PNGs in `upstream_runs/imma/data/vangogh/` (11:39–11:43); trainer completed 5000/5000. **Remaining:** the 3 attack seeds, barred by `FREEZE_ACTIVE`. |
| **R2** | *Was:* `./dataset/{hrm,irt,tgt}` are not in the repo and must be constructed. | **DISCHARGED.** Built 12:39–12:47 (`[r2pools] done`), 30 train + 30 test per pool. A first attempt failed `rc=1` — `ModuleNotFoundError: No module named 'pandas'` at `r2_build_meta_pools.py:9` (12:21) — and was fixed. **Remaining:** `src/meta.py` has never been invoked. Note the pools are **ours**, so an R2 run is tier 2, not a reproduction. |
| **R3** | `vision_relearning` needs `wandb` and CIFAR-scale data; it is a **classifier** codebase, so this is a native-domain check, not a T2I run. | **STILL BLOCKED, not started.** No `logs/r3*` and no `outputs/r3_*` exist. |
| **R4** | *Was:* none — queued, waiting only on a free GPU. | **DISCHARGED — it ran** (four phases, 11:32→12:45). The inertness question is **closed: upstream's edit is real** (§3). Open items: whether upstream should have had an expanded erase-prompt set, and a **required re-run of R4d** with the port checkpoint keyed on its full parameter set (`D-026`). |
| general | GPUs 2–3 belong to other users and are not used; our cap is two at a time. | **No project process is running.** `FREEZE_ACTIVE` (12:43) bars new GPU work; bounded verification runs only, launched by the lead. |

## 6. What the measurements so far support

Only this: the five repositories are pinned and licence-checked; FRAG ships no code; our IMMA port
has a measured **2.13×** scope gap from its reference (`outputs/imma_scope_fidelity.json`); our Meta
v2 has one remaining structural departure (two pools, not three — `--fix_timesteps` is **not** a
departure, see `UPSTREAM_FIDELITY_MANIFEST.md` §7(b)); upstream IMMA and upstream UCE have each been
**executed once**, on our benchmark; and R4's divergence localises to **embedding selection** rather
than the solver. The "upstream may have written an unedited model" worry is **refuted** (§3): upstream
changed all 32 `attn2` K/V matrices, 5.36 % mean / 9.79 % max, 0/32 identical to base. The open R4
items are the erase-prompt-expansion question and the confounded port side (`D-026`, R4d to re-run).

**No method has been reproduced, and no published method has been refuted.** Two cells ran upstream
code, which is tier 2 — not a reproduction of any paper's reported experiment, and not evidence
about any published method's merit. The historical variants keep their names and their results;
nothing was deleted, overwritten or relabelled as official.

**The mandate remains unmet**, and R1/R4 do not bear on it: no arm reaches retain8 ≥ 0.85 with
max-over-valid H ≤ 0.40 at any budget. The closest approach — `meta2_church_lt0.5`, retain8 0.9000 —
clears the robustness half at two attacker learning rates at 25 steps but fails at lr 1e-4
(**0.5333**, `logs/e2h_stronger.log`), and max-over-valid is a ratchet. `FINDING78`'s "first arm"
claim is **retracted**; its retention result (+0.20 to +0.27 at identical suppression) stands.

---

## 7. Change log — 2026-10-26 (Agent 6, provenance audit)

Full traceability in `AGENT6_PROVENANCE_FINDINGS.md` §1.1. Banners were judged insufficient: the
incorrect sentences and table cells were rewritten, not annotated.

| what changed | why |
|---|---|
| §1 status table rebuilt, with an **evidence column** | eleven cells were false. R1 trained 5000/5000 and was scored; R4 executed in four phases; R2's pools exist. Verified from `logs/r1b_imma_train.log`, `logs/r1c_eval.log`, `outputs/r4{,b,c,d}_uce_*.json`, `logs/r2_meta.log` and file mtimes — not from the previous revision |
| §1.1 **three fidelity tiers** added | the two-tier scheme forced upstream-code-on-our-benchmark runs to be called either "official reproduction" or "ours", and both are wrong |
| §2 "In flight (not interrupted)" → "**Nothing is in flight**" | E4m/E4mg, E2g, E2h, R4d and R1c had all finished; `ps` shows no project process; `FREEZE_ACTIVE` bars new work |
| §2 gained R1/R4/R2 completion bullets | the section listed only pre-R work as "Completed" |
| §3 UCE paragraph rewritten | it described R4 as prospective ("which R4 tests directly"). R4 ran; phase 1 is self-flagged **VOID**; `FINDING75` is retracted. The `erase_scale=2.0` probe lives in the VOID artifact and is no longer cited |
| §3 gained an explicit **"no fidelity conclusion yet"** paragraph | upstream's λ=0.1 arm scores target 15/15, which was read at the time as consistent with an inert edit. **That reading has since been refuted by measurement and the paragraph is replaced** — see the 2026-10-26 reconciliation row below |
| §4 matrix statuses replaced; R1 and R2 **relabelled** | "author protocol" and "its own three-pool data" were both unearned — the images, pools, scorer and prompts are ours |
| §4 gained R1's recorded outcome **with interpretation withheld** | the number exists and should be visible, but the attack seeds were never run. **One of the three candidate explanations then listed — a tensor-composition artifact — has since been refuted**; see the reconciliation row below |
| §5 three of five blockers marked **DISCHARGED**, with the `pandas` failure recorded | they had been discharged before the previous revision was written; the failed first attempt is recorded rather than tidied away |
| §6 rewritten; mandate status and the `FINDING78` retraction added | it ended on "the UCE comparison is running", and omitted the mandate conclusion entirely |
| "No method is marked reproduced" (header) reconciled with the `REF_*` convention | the manifest defined `REF_*` as "official reproduction", which contradicted this line |

### 7.1 Documentation reconciliation, 2026-10-26 — two refuted explanations removed

| what changed | why |
|---|---|
| §3's **"upstream UCE may have written an unedited model / its edit may be inert"** paragraph **replaced with the refutation** | **REFUTED by measurement.** Upstream changed all **32** `attn2` K/V matrices — **5.36 % mean / 9.79 % max** of weight norm, **0/32** bitwise identical to base, **43.4 %** change in K/V output for `"a photo of a church"`. The inertness hypothesis is dead; the live question is erase-prompt expansion, and the **port** side stays confounded (`D-026`, R4d to re-run; `D-031`, R4c no longer reproducible) |
| §4's **"artifact of `r1c_eval`'s tensor composition"** removed as a live explanation; the composition recorded as **valid** | **REFUTED by direct audit.** `exact=160 fuzzy=0 reshaped=0 skipped=0`; 160/160 IMMA tensors and 526/526 others byte-equal to their sources; ESD delta **not** inverted; byte-semantically identical to upstream's own composition. `gen_REF_imma_vangogh.json` stays quarantined on the **different** ground that the closed-set metric is inapt and the configuration is not one the authors report (`D-009`), with `D-019`'s ESD checkpoint now the live candidate and **its control run still outstanding** |
| §4's final-prompt paragraph and §1's R4 row **de-directionalised** | the label-text mismatch (`D-022`) is **established** — nine of twelve strings differ, `church`/`parachute`/`tench` identical — but its **direction and size are not**, pending paired re-scoring. "Degraded distractors" and "target hits biased upward" are withdrawn as inferences |

**Not changed, deliberately:** every historical number; the `FINDING76`/`FINDING77` correction
banner (independently re-verified against `r4b_uce_lamb*.json` column B and found accurate); R3's
blocker, which is the one that still holds.
