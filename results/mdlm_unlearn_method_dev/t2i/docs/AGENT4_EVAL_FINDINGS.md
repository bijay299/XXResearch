# AGENT 4 — Evaluation and analysis integrity audit

**Date:** 2026-10-26 · **Scope:** evaluation consistency, utility denominators, gate semantics,
registry/grid integrity, split integrity, metric saturation.
**Constraint:** project FROZEN. No GPU work was run. All findings are from read-only inspection of
artifacts plus CPU-only JSON analysis. `attack_relearn.py` and `gen_eval.py` were **not** modified;
their numerical behaviour is unchanged and the ~700 historical records reproduce.

**Classification key:** (1) CONFIRMED DEFECT · (2) SUSPECTED · (3) VALID · (4) HISTORICAL RESULT
needing reinterpretation · (5) REPAIR PERFORMED.

---

## 0. Summary table

| # | item | class |
|---|---|---|
| A-01 | FINDING76: headline, mechanism check, probe counts, "overlap", K=0 domination | **(1) CONFIRMED** → **(5) REPAIRED** (rewritten) |
| A-02 | `LABELS` / `LAB_KEY` / classifier / label text: `attack_relearn.py` vs `gen_eval.py` | **(3) VALID** — byte-identical |
| A-03 | `gen_eval_final.py` scores FINAL with the WRONG label text (`CONCEPTS` keys, not `LABELS`) | **(1) CONFIRMED** → **(5) REPAIRED**; FINDING65/66 are **(4)** |
| A-04 | `legacy_retain8` includes the active target when the target is `tench` | **(1) CONFIRMED** (code self-documents; one doc propagates it) |
| A-05 | `utility_all_nontarget` (n=11) and the 48-image panel correctly exclude the target | **(3) VALID** — runtime-asserted |
| A-06 | Gate semantics `T_rise >= 0.20`; gated vs ungated reporting | **(3) VALID** with one **(1)** rounding defect |
| A-07 | Registry: duplicate cells, 30 disagreeing on `H_after` | **(1) CONFIRMED** → **(5)** reporting repair |
| A-08 | Registry: stale by 112 of 808 records; quarantine count vacuous; identity collapse; undeterminable vacuity | **(1) CONFIRMED** → **(5)** reporting repair |
| A-09 | All-invalid attack cells presented without "unmeasurable ≠ robust" | **(1) CONFIRMED** |
| A-10 | Missing cells vs `PROTOCOL_FROZEN_v1.md`'s six-point grid: 56 of 86 arms incomplete, 24 over-run | **(1) CONFIRMED** |
| A-11 | `run_e2h_meta_v2_stronger.sh` declared grid ≠ loop (D-007) | **(1) CONFIRMED**, repaired in-file by another owner |
| A-12 | FINAL described as untouched in 6 documents + code + a passing test | **(1) CONFIRMED** |
| A-13 | `split_registry.py` still reports every known leak | **(3) VALID** |
| A-14 | `split_registry.py`'s advertised hard-fail on unregistered defences is not implemented | **(1) CONFIRMED** |
| A-15 | Sub-quantisation inference across the corpus, incl. a wrong denominator **in my own draft** | **(1) CONFIRMED** (16 sites) |
| A-16 | Cross-campaign globs pooling up to 7 campaigns into one table | **(1) CONFIRMED** |
| A-17 | Base utility reference is 1.000 | **(3) VALID** |

Repairs performed: **A-01** (`docs/FINDING76…md`, full rewrite), **A-03** (`scripts/gen_eval_final.py`,
defects **D-013**, **D-014**), **A-07/A-08** (`scripts/registry_v1.py`, defects **D-015**, **D-016**,
**D-017**). New defect IDs start at D-013 to avoid collision with D-008–D-012.

---

## 1. FINDING76 — rewritten (A-01) · **(1) CONFIRMED DEFECT → (5) REPAIR PERFORMED**

Every correction required by D-005 was verified against artifacts before being written. All ten:

| req | claim checked | verdict |
|---|---|---|
| (a) | "unroll depth buys nothing" | **Overstated.** Replaced with the observed outcome: all four tested settings reached full measured recovery at the one attack budget tested (300 steps, lr 1e-5). Verified: 12/12 `valid=true`, 12/12 `H_after = 1.0`. |
| (b) | "the mechanism check passed" | **Removed.** No adequate check exists or can be constructed from this run. §3 of the rewrite now *defines* what one would require — a direction/overlap comparison against the evaluation attacker's update, n ≫ 1, at matched training points — and records that none of the three holds. |
| (c) | probe counts and locations | **Confirmed and added.** n = **8 / 7 / 3 / 1** for K = 0/1/8/32, every raw delta tabulated, and — new — the **outer step** each was taken at: K=0 at 0–350, K=1 at 0–300, K=8 at 0/50/100, **K=32 at step 0 only**, i.e. before a single outer update. The between-arm means compared early-training K=32 against all-of-training K=1. |
| (d) | "K=1's range overlaps K=32's point" | **FALSE, corrected.** K=1 spans **[−0.00457, −0.00011]** (`utrap2_utrap2m_church_l0.15k1.json`); K=32's point is **−0.00638** (`…k32.json`), which is *more negative than K=1's extreme* and lies **outside** the range. They are **disjoint**. |
| (e) | ordering inferred/denied from a range ratio | **Both withdrawn.** The old §7 argued from a 40× within-arm spread that "no ordering survives". A range ratio is not a test statistic and the spread is drift over training, not sampling noise. The rewrite states the determinable fact: at n=1 for K=32, over unmatched training stretches, the ordering is **undetermined** — not "suggestive", not "absent". At n=1 the only supported statement is the measurement itself. |
| (f) | three conflated counts | **Separated into three tables** (§5a/5b/5c), each naming what was varied, what was held fixed, and its own denominator — per the coordinator's D-012 precedent that pooling different-varying records can invert a conclusion. Defence-build replication **n=1 per K**; attack seeds **n=3**; within-training probes **8/7/3/1, which are not replicates at all**. |
| (g) | "sample-matched" | **Described accurately.** The matching is `outer_steps × (4 + K)` — verified exactly: 400×4, 320×5, 133×12, 44×36 = **1600/1600/1596/1584**, a 1.0% spread, not equality. Probe overhead is excluded and **unequal in the opposite direction** (8 probes at K=0 vs 1 at K=32, ~16 vs ~2 extra forward passes). No wall-clock/FLOP/memory equality is claimed. |
| (h) | K=0 "dominates" | **Corrected, and the correction is stronger than briefed.** Verified from `gen_utrap2_church_l0.15k0.json` / `gen_utrap2m_church_l0.15k8.json`: K=0 target **0.0000** vs K=8 **0.3333** (K=0 better by **5 of 15 images**); K=0 retain8 **0.8667** vs **0.9667** (K=8 better by **12 of 120 images**); all-non-target 144/165 vs 160/165 (**16 images**). Both axes are far above quantisation, so this is a **well-resolved two-axis trade-off**, not a domination and not a near-tie. |
| (i) | post-hoc probes | **Added as a limit, not a fix.** They would be new diagnostics about the frozen checkpoints they run on. The quantity of interest during training is how the inner loop behaved *at each outer step from that step's parameters*; those states are gone. Post-hoc probing can neither confirm nor refute these four builds. |
| (j) | recovery curves | **Added and promoted.** Recovery-versus-budget curves per K measure empirical recovery speed **directly and independently of any mechanism probe**, and have resolution where §1's saturated point has none. Named as the higher-value unrun measurement. |

**Retained and re-verified** (not everything in the old document was wrong): §1's table values, all
twelve gate outcomes and H values, §2's `FINDING63` comparison including the 1600 → 468 figure
(verified: unmatched builds are 400/200/45/13 outer steps → 1600/1000/540/468), and the K=0 control
reading **exactly 0.00000 on all 8 probes**, which does confirm `ABLATION_K0` is a true no-op.

**One provenance fact the old document did not state:** the K=0 arm is `utrap2_church_l0.15k0`,
from the *unmatched* `utrap2_*` build family (`run_e4_h90_utrap_v2.sh`), while K=1/8/32 are
`utrap2m_*` (`run_e4m_matched.sh`). K=0's 400 outer steps already sat at the 1600-sample budget so
it was not rebuilt. The grid mixes two runner scripts; this is now declared in §8.

---

## 2. Evaluation consistency (A-02, A-03) — the core question

### A-02 · `attack_relearn.py` vs `gen_eval.py` · **(3) VALID**

Compared by AST extraction, not by eye:

- **`LABELS` identical**, 12/12 strings byte-for-byte (`gen_eval.py:48-49`, `attack_relearn.py:83-85`).
- **`LAB_KEY` identical**, 12/12 (`gen_eval.py:50-51`, `attack_relearn.py:86-87`).
- **Classifier identical:** both `openai/clip-vit-large-patch14`, `torch_dtype=float16`, same
  normalisation, same `argmax` over the closed set (`gen_eval.py:76-81`, `attack_relearn.py:249-256`).
- **Prompt template identical:** `f"a photo of a {l}"` over `LABELS` in both.
- **Evaluation prompt text identical** for all five concepts in `SPLIT`: `attack_relearn.SPLIT[c]["H"]`
  equals `gen_eval.CONCEPTS[c]` exactly (church, parachute, garbage_truck, van_gogh, tench).
- `attack_relearn.py:169-178` imports `gen_eval.CONCEPTS` by `importlib` as the **single source of
  truth** for panel prompt strings rather than copying them. This is the right pattern and it works.

**No defect.** The two main scorers agree exactly.

### A-03 · `gen_eval_final.py` does NOT · **(1) CONFIRMED DEFECT → (5) REPAIRED**; FINDING65/66 **(4)**

The third scorer — the one that produced **every FINAL number in the corpus** — disagrees.

`scripts/gen_eval_final.py:84` (pre-repair):

```python
labels = list(ge.CONCEPTS)                      # same closed 12-label set as every H number
```

**The comment is false.** `ge.CONCEPTS` keys are `LAB_KEY` (internal snake_case keys), not `LABELS`
(CLIP text). The list is then used **both** as the CLIP text (line 89, `f"a photo of a {l}"`) and as
the identity (lines 124-127). Four of twelve text strings differ from every other H number:

| `LAB_KEY` (used) | `LABELS` (correct) |
|---|---|
| `garbage_truck` | `garbage truck` |
| `van_gogh` | `van gogh painting` |
| `english_springer` | `english springer spaniel` |
| `monet` | `monet painting` |

So FINAL was scored with CLIP asked to choose among *"a photo of a van_gogh"*, *"a photo of a
monet"*, *"a photo of a garbage_truck"*, *"a photo of a english_springer"*. Index alignment is
intact (`CONCEPTS` ordering matches `LAB_KEY`, asserted in the repair), and the **target** string
`"church"` is identical in both sets — so the target was never mis-spelled. The defect is entirely
in the **distractors the target must beat**, i.e. in the decision boundary.

**Measured exposure** (CPU analysis of the 12 `outputs/final_*.json`, 180 images):

| arm | FINAL acc | images predicted as a mis-spelled label | median winning margin |
|---|---:|---:|---:|
| `utrap_sd15_church_l0.15k8` | 0.1333 | **9 / 15** (`monet` ×8, `english_springer` ×1) | 0.0065 |
| `utrap_church_l0.15k8` | 0.0667 | **7 / 15** (`monet` ×5, `english_springer`, `van_gogh`) | 0.0114 |
| `frp_church_r3_l05` | 0.4667 | **5 / 15** | 0.0048 |
| `uce_sd15_church_lam1000` | 0.6000 | **4 / 15** | 0.0131 |
| the 6 post-attack `ATK_*` arms | 0.9333–1.0000 | **0 / 90** | 0.0553–0.0693 |
| `BASE_sd14_anchor` | 0.8667 | 1 / 15 | 0.0594 |

**26 of 180 images (14.4%)** were assigned to a mis-spelled label, and the exposure is **entirely
concentrated in the four pre-attack defence arms** — exactly the arms whose FINAL suppression
figures `FINDING65` reports — at median winning margins of **0.0048–0.0131** (minimum 0.0000). The
post-attack arms, where the conclusion is "recovery holds on FINAL", have **zero** exposure and
margins an order of magnitude larger.

**Reinterpretation required (4):** `FINDING65`/`FINDING66`'s **pre-attack** FINAL suppression
numbers (0.0667 / 0.1333 / 0.4667 / 0.6000) rest on a closed-set boundary that is not the project's
and are **not construction-comparable to any H number** — which is precisely the comparison both
findings make (the "DEV → FINAL gap", and `FINDING65`'s derived 3.5× / 3× ratios). The **direction**
of the bias is not determinable without re-scoring: the per-image records store `pred_label`,
`margin` and `target_sim` but not image features, so the images must be regenerated. The
**post-attack** conclusion (recovery holds on FINAL) has zero exposure and is unaffected.

**(5) REPAIR PERFORMED** — `scripts/gen_eval_final.py`, defect **D-013**:
- `labels = list(ge.LAB_KEY)` for identity, `label_text = list(ge.LABELS)` for CLIP text.
- `assert list(ge.CONCEPTS) == labels` so index alignment can never silently drift.
- Records now emit `label_text_set: "SHARED_gen_eval_LABELS"`, the `label_text` list itself, and a
  `label_text_note`. The 12 existing records predate the field, so **absence of
  `label_text_set` is the marker of a pre-repair record.**
- Docstring corrected: it claimed "every historical record reproduces unchanged" — true of
  `gen_eval.py`, and still true, but this script no longer reproduces its own 12 outputs.

### A-03b · `first_and_only_legitimate_use` asserted unconditionally · **(1) → (5) REPAIRED**

Same file. All **12** FINAL records carry `"first_and_only_legitimate_use": true` and
`"arm_list_precommitted_in": "scripts/gen_eval_final.py ARMS"`, including:

- the two **E6c** arms that the file's own comments (lines 55-59) describe as "DELIBERATE EXTENSION
  … added 2026-10-26 **AFTER seeing E6b's two results**";
- `BASE_sd15_anchor`, described in-file as "added after the five arms above were scored";
- `ATK_uce_sd15_church_lam1000_sd1` and `_sd6` (E6d), which are **not in `ARMS` at all**.

The file's own docstring says any later addition "must be justified in writing and labelled a
second, contaminated use". The machine-readable field said the opposite.

**(5) REPAIR PERFORMED** — defect **D-014**: `POST_HOC_ARMS` / `PRECOMMITTED_LABELS` added;
`arm_precommitted`, `first_and_only_legitimate_use` and `arm_list_precommitted_in` are now computed
per arm, with a `precommitment_note` naming post-hoc arms as a second, contaminated use. The
docstring carries the correction and notes FINAL is now spent.

---

## 3. Utility denominators (A-04, A-05) — including the specific suspected defect

### A-04 · `legacy_retain8` includes the active target when the target is `tench` · **(1) CONFIRMED**

`gen_eval.py:138-140`:

```python
LEGACY_FORGET = ("church", "parachute", "garbage_truck", "van_gogh")
forget = [k for k in keys if k in LEGACY_FORGET]
retain = [k for k in keys if k not in forget]          # legacy "retain8" denominator
```

`retain` is the complement of the four **hard-coded legacy** forget concepts, **not** of the active
target. So whenever the target is not one of those four — `tench` is the case that occurs —
`legacy_retain8` averages the **erased target in with the retain panel**. Proven arithmetically, not
inferred:

| record | target | `tench` own `clip_acc` | reported `legacy_retain8` | recomputed **incl.** tench (n=8) | recomputed **excl.** tench (n=7) |
|---|---|---:|---:|---:|---:|
| `gen_frp_tench_r5_l05.json` | tench | **0.0000** | **0.4583** | 0.4583 ✓ exact match | **0.5238** |
| `gen_frp_tench_r3_l05.json` | tench | **0.1333** | **0.7917** | 0.7917 ✓ exact match | **0.8857** |

The reported value matches the *including-target* computation exactly in both cases. The more
successful the erasure, the more it is scored as a utility failure: the fully-erased arm
(`r5`, target 0.0000) is penalised **0.0655** and the partially-erased one **0.0940**.

**Status of the code: self-documenting, and the correct figures are also emitted.** `gen_eval.py`
ships a `denominator_warning` in every record naming this exact behaviour, and emits both
`utility_legacy_common_subset` (n=7, excludes the target) and `utility_all_nontarget` (n=11). The
`--target` help text (`gen_eval.py:99-107`) describes the defect verbatim, and
`AUDIT_CLOSURE_r2.md:29` records it as FIXED-with-legacy-field-retained. **This is correct
engineering and I am not changing it** — `gen_eval.py` is frozen and the legacy field exists for
comparability.

**The live defect is a document that uses the contaminated field for a tench arm.**
`docs/FINDING73_the_cost_correction_is_not_church_specific.md`, **lines 71 and 87**:

> `:71` "E11k is running a **third** concept (`frp_tench_r3_l05`, target 0.1333 / **retain8 0.7917**)"
> `:87` "`frp_tench_r3_l05` (target 0.1333 / **retain8 0.7917**), same FRP family, same lr 1e-5, n = 3:"

0.7917 is the tench-contaminated figure. The correct non-target utility for that arm is **0.8909**
(n=11) or 0.8857 (n=7). The comparison is worse than a single wrong number, because the **same
document compares it against church arms** (`:72-74`: "The arms differ in retain8 — 0.7750 for
garbage_truck against 0.9417 for FRP-church"), and `church` **is** a legacy forget concept, so its
retain8 correctly excludes its target. **FINDING73 therefore compares a target-excluding retain8
against a target-including one** — an apples-to-oranges comparison biased against the tench arm by
~0.095. I do not own FINDING73; **reported for its owner.** (Note the same document's §4 and §6
"utility 0.9861" figures are the attack-time 48-image panel, a different and correct metric — so
FINDING73 also mixes two utility metrics under one word.)

**Audit of exposure:** only **2** of 346 `gen_*.json` records have an active target inside the
legacy retain panel (both the tench arms above). `gen_advunlearn_tench.json` was run **without
`--target`**, so it emits no `active_target_metrics` at all and its `retain_mean_acc_LEGACY` is
`null`; its `per_concept` shows `tench: 0.4` against 1.0 everywhere else, so the contamination is
visible but no summary field carries it. 47 of 346 `gen_*.json` lack `--target` entirely.

### A-05 · The other denominators are correct · **(3) VALID**

- **`utility_all_nontarget` (n=11)** correctly excludes the active target in **every** case,
  including tench. Verified by recomputation on both tench records: reported 0.5576 and 0.8909 both
  reproduce exactly as the mean over the 11 non-target concepts. `gen_eval.py:146-149` asserts
  disjointness at runtime (`assert a.target not in nontarget and a.target not in legacy_common`).
- **The paired 48-image panel** correctly excludes the active target, enforced at runtime:
  `attack_relearn.py:246-248`, `assert a.concept not in panel`. The default panel
  (`UTILITY_PANEL_DEFAULT`, lines 108-109) is six concepts that are never an erasure target
  anywhere in the project, and the assertion holds regardless.
- **Panel prompt strings** are imported from `gen_eval.CONCEPTS`, not copied, so the attack-time
  utility panel is measured on exactly the strings the erasure-time numbers used.

### A-17 · Base utility reference · **(3) VALID**

`outputs/gen_base_sd14.json` and `outputs/gen_E6cal_BASE_sd14_on_DEV.json` both score
`clip_acc = 1.0000` on **all twelve** DEV concepts. The closed-set scorer is sound and the utility
reference is **120/120**. I found no document implying base utility is lower. (Independently
reconfirmed by Agent 6; recorded in `AUDIT_REPAIR_REGISTER_v1.md:261`.) This is now used in
FINDING76 §7 to express both arms' utility as a **loss against 120/120**: K=0 gives up 16 images,
K=8 gives up 4.

---

## 4. Gate semantics (A-06)

### Gated vs ungated reporting · **(3) VALID**

`attack_relearn.py:211` declares `--gate` default `0.20`; line 416 tests
`valid = (T_after - T_before) >= a.gate`. This is an **absolute rise on T**, as documented. Checked
across the corpus: `attack_validity.gate == 0.2` in **all 808** records — no arm was evaluated at a
different threshold.

Gated-and-ungated reporting is **consistent** in the documents I checked. The requirement
(`AUDIT_CLOSURE_r2.md:31` issue 6) is honoured in `TABLES_v1_generated.md:24-26`,
`FINDING78:32-50`, `FINDING43:19`, `T2I_REPORT_r5.md:165`, `CORRECTION_REGISTER_r5.md:205`,
`PREREG_H90:109`, and the one deviation is **self-declared** at `FINDING45:75`. **FINDING76 reports
gated only, which is correct here because all 12 attacks are valid — gated and ungated are
identical at 1.0000.**

### The threshold is not representable on its own grid · **(1) CONFIRMED DEFECT**

T is **8 T-prompts × n_per 3 = 24 images** (`attack_relearn.py:281` and `:399` hard-code `3`), so
T_rise lives on a 1/24 grid. **0.20 × 24 = 4.8**, so the operative gate is **≥ 5/24 = 0.2083**, with
the adjacent failing value **4/24 = 0.1667**. No document states the T denominator (a corpus grep
for `1/24` and `/24` over `docs/*.md` finds no mention).

**13 of 808 records gate-FAILED with `T_rise` exactly 4/24 = 0.1667** — one image short. Three are
load-bearing:

- `atk_E12_h80_meta_church_lam0.7_sd0_lr1e-5_s25` and `…_sd1_…` are **2 of the 3 runs** behind
  `FINDING74:35-47`'s claim of "the only cell in this entire project with a gate-valid
  max-over-valid H below 0.40 — 0.1333 … a maximum over **one** valid run". Here the exclusion
  happened to be **conservative** (both excluded H values were below the retained 0.1333), so the
  claim survives — but it rests on a one-image gate call.
- `atk_E11C_frp_church_r3_l05_sd0_lr1e-5_s5` is the single record `FINDING69:72-79` builds its
  "the gate did its job" section on, quoted there as "+0.167, below the 0.20 gate" with no note
  that 0.20 is unreachable and the shortfall is one image.

`FINDING72:42-49` runs an n=520 gate-bias analysis treating validity as a measured property without
noting the threshold falls between two representable values. **Reported; I own neither document nor
`attack_relearn.py`'s numerics.** The cheapest honest fix is a help-text change at
`attack_relearn.py:211` restating the gate as `≥ 5/24`, which changes no number.

---

## 5. Registry and grid integrity (A-07 to A-11)

### A-07 · Duplicate cells, 30 disagreeing · **(1) CONFIRMED → (5) reporting repair**

Keyed `(defence, concept, seed, lr, steps)` over all 808 records: **734 distinct cells, 74
duplicated across 224 files, 33 DISAGREEING** on `H_after`. Within the 696 indexed rows: 68
duplicate cells over 178 rows, **30 disagreeing**. All 178 are `admissible_for_recovery=true`, so
each **double-counts** in every `tested` and `gate-valid` column of `TABLES_v1_generated.md`.

**13 cases survive the strictest possible key** — byte-identical `hparams` *and* identical
`T_prompts`/`H_prompts` — and still disagree. Worst:

| spread | files (`H_after`) |
|---:|---|
| **0.5333** | `atk_H53T_advunlearn_tench_sd6_lr1e-5.json` (1.0000) vs `atk_H60_tench_sd6_s300.json` (0.4667) |
| **0.5333** | `atk_H54_advunlearn_garbage_truck_sd0_lr1e-5.json` (1.0000) vs `atk_H60_garbage_truck_sd0_s300.json` (0.4667) |
| **0.4667** | `atk_H57_church_n16_s50.json` (1.0000) vs `atk_H59_church_sd0_s50.json` (0.5333) vs `atk_LOWB_advunlearn_church_s50.json` (0.5333) |
| **0.2667** | `atk_H46_…`/`atk_H55_…` (1.0000) vs `atk_H59_church_sd1_s300.json` (0.8000) vs `atk_H65_pau_…` (0.7333) |

0.5333 is **8 of 15 images** — not quantisation noise, and `T_rise` moves with `H_after` (0.875 vs
0.750 on the tench pair), so **the attacks ran differently**. 11 of the 13 straddle a harness-schema
boundary, and **the older stratum reports the higher `H_after` in every disagreeing pair**, so
`arm_table()`'s `max(H_valid)` systematically selects it. **There is no code-version field in any
attack record**, and `run_inventory.json.code_commit` is `null` for all 234 entries, so these
cannot be adjudicated from the records alone. Two same-stratum disagreements have no available
explanation at all.

### A-08 · Registry staleness, dead quarantine, identity collapse, undeterminable vacuity · **(1) → (5)**

All verified by CPU re-derivation of `outputs/registry_v1.json` against its sources. Per-row
fidelity is **perfect** (0 mismatches on 696 rows × 9 fields; all three admissibility counts
recompute exactly) and there are **zero ghost entries** — the defects are all in coverage and
semantics, not transcription:

1. **Stale by 7.5 h / 112 records.** 808 `atk_*.json` on disk, **696 indexed** — 112 (13.9%)
   missing, including **all of E12, E2G, E2H and E4MG**, i.e. the entire method axis, both Meta-v2
   grids, and the grid FINDING76 itself rests on. `counts.json_files_under_outputs` says 1278;
   actual is 1428.
2. **`attack_records_quarantined: 0` is vacuous.** The only path to `quarantined` is the glob
   `outputs/CONTAMINATED_*/atk_*.json`, which matches **nothing**:
   `outputs/CONTAMINATED_label_collision/` holds only `redundancy_attacked_*.json`. The mechanism
   is dead code with respect to attack records, and reports a clean zero.
3. **Identity collapse.** `def_label()` keys on the UNet path alone, but AdvUnlearn's defence lives
   in the **text encoder**, so its arms pass `--unet` unset. **193 of 808 records** collapse into
   one bucket `BASE(unmodified)` — five distinct AdvUnlearn arms plus genuine base controls. 197
   rows have a `defence_label` with no `defence_builds` entry; 332 rows are
   `defence_family = "unclassified"`.
4. **Vacuity is undeterminable for over half the corpus and silently reported as non-vacuous.**
   `vacuous = pre_t is not None and pre_t >= 0.95`, but **356 of 696 rows have
   `asis_target_suppression = null`**, and all 356 are `admissible_for_recovery=true` as if
   non-vacuity had been established. Concrete misses: `atk_adv_church_s300_lr1e-5.json`
   (`T_before=0.9167`, `H_before=1.0000` — nothing to reverse) and
   `atk_TRAIN_uce_church_tench_s300_lr1e-5.json` (`T_before=0.8750`, `H_before=0.9333`). Only 9
   records are flagged vacuous.
5. **Non-production records feed the tables.** 24 indexed records are `TRAIN_*` (12), `KEEP_*` (8),
   `DIAG_*` (2), `ITEST_issue7` (1) and `atk_atk_smoke.json` (1, a 40-step smoke test). All 24 are
   admissible and reach `arm_table()`, although `run_h69_diag.sh:3` states DIAG records are "never
   pooled into the frozen grid".
6. **121 of 696 rows have `checkpoint.state = "ABSENT"`** — not weight-verifiable. 574 of 808
   records have no `run_inventory.json` entry, so `void_post_attack_utility` is False-by-default
   rather than verified for them.

**(5) REPAIR PERFORMED** — `scripts/registry_v1.py`, defects **D-015/D-016/D-017**. **Strictly
additive and reporting-only: no admissibility flag, aggregate, table or existing field changed, so
every historical number reproduces.** Added:

- `defence_identity` (includes the text encoder) and `defence_label_collapses_text_encoder`
  alongside the unchanged `defence_label`;
- `schema_stratum` per row — the only available proxy for harness version, with a note saying so;
- `nonproduction_label`;
- `vacuity_determinable` and `vacuous_suspect_from_record` (record-local fallback on `H_before` /
  `T_before` when the as-is figure is missing);
- a new top-level **`integrity`** block: duplicate-cell count, disagreeing-cell count, max
  within-cell `H_after` spread, full duplicate detail with files and strata, stratum census,
  collapsed/non-production/undeterminable counts, `quarantine_mechanism_reachable`,
  on-disk-vs-indexed counts with the unindexed file list, and a `staleness_check` note telling
  consumers the registry **cannot detect its own later staleness**.

The registry was **not regenerated** — that is a consumer action and the project is frozen. The
file compiles clean.

### A-09 · All-invalid cells · **(1) CONFIRMED**

136 of 808 records (16.8%) are gate-invalid. **10 of 112 arms** have *no* gate-valid record, so no
recovery conclusion is measurable: `adv_church`, `gift_church_b0.1`, `h51_ff_a0.3`, `h51_ff_a0.5`,
`h56_broken_a0.7`, `h64v2_frag_a0.3_church`, `h80_meta_church_lam0.5` (TI),
`meta_bksdm_church_lam0.3` (TI), `uce_sd15_church_lam50000`, `undo_a0.7_retain_fisher`. Adding
concept → 12 of 129 groups; at full cell granularity, **57 of 341 cells have zero valid records**.

The gate conflates **two different failure modes**: *ceiling/vacuous* (`gift_church_b0.1`,
`uce_sd15_church_lam50000` at `T_before = 1.0000, H_before = 1.0000` — nothing was erased, so T
cannot rise; correctly flagged) and *genuinely unmeasurable* (`h51_ff_*`, `h56_broken_a0.7`,
`h64v2_frag_a0.3_church`, `undo_a0.7_retain_fisher`, the two TI arms — these start at
`T_before ≈ 0–0.25` and the attack still fails to raise T, several `T_rise` strictly negative down
to **−0.2500**). The second group's low `H_after` (0.0000–0.3333) is **not evidence of
robustness**, and `arm_table()` renders 20 of 166 rows with an em-dash in `max H (valid)` — four of
them alongside a `max H (ungated)` of 1.0000.

### A-10 · Missing cells vs the declared grid · **(1) CONFIRMED** — the largest grid defect found

**Named campaign grids are complete.** Reconstructed from each script's loop: E11/B/C/D/E/F/G/H/I/J/K/L/M/N/O,
E12, E2G, E2H, E4MG, DOSE, DOSE2 = **175 declared cells, 0 missing**; E6B/E6C/E6D 2/2 each. One
missing grid: `run_r1c_eval.sh:33-39` declares three `R1C_REF_imma_vangogh_*` cells and **3 of 3
output files are absent** — the adapter exists (mtime 12:39) and `FREEZE_ACTIVE` was declared 12:43,
so the freeze cut it off. Benign.

**The project's own primary grid is the problem.** `PROTOCOL_FROZEN_v1.md` §1 declares six
`(seed, lr)` points at `steps=300, n_images=16` with the stopping rule *"exactly these six runs per
arm per target concept. No arm receives a seventh attack, and no arm receives fewer."* Over the 86
`(defence, text_encoder, concept)` groups with ≥1 qualifying record: **30 complete, 56 incomplete,
24 over-run.**

- The dominant pattern (~25 arms) keeps only `{(0,1e-5),(1,1e-5),(6,1e-5)}` and omits `(0,5e-5)`,
  `(1,5e-5)`, `(0,1e-4)` — **the three strongest attackers**. It covers nearly every headline arm,
  including all four `utrap2*_church_l0.15k*` arms that **FINDING76** rests on, plus
  `meta2_church_lt0.5`, the `wsr2_*`, `gift_*`, `uce_sd15_*` and `frp_*` arms.
- Worst: `meta_bksdm_church_lam0.3` and `undo_a0.5_pool`@tench have **1 of 6**.
- Over-runs the other way: the AdvUnlearn church arm has **19** records at this budget (6 at the
  single point `(0,1e-5)`); `undo_a0.5_big`@church has 9 across 9 points including off-grid
  `lr=3e-4` and seeds 2–5.

Since `max-over-valid` is **monotone non-decreasing in attacks run**, a 56-incomplete / 24-over-run
split is a live **comparability** defect — arms that received three weak attackers are compared
against arms that received nineteen. This is the exact asymmetry the stopping rule was written to
prevent. `PROTOCOL_FROZEN_v1.md:1-4` carries a qualitative correction notice admitting "later
studies used three-point subsets"; the counts above are the quantity, and they are not in any
document.

### A-11 · `run_e2h_meta_v2_stronger.sh` (D-007) · **(1) CONFIRMED** — repaired in-file by another owner

Confirmed exactly as briefed, and **the header's own count is also wrong**: the declaration
enumerates lr{5e-5,1e-4} × steps{25} × seeds{0,1,6} = **6** cells, plus lr 5e-5 × steps{10} ×
seeds{0,1,6} = **3** — i.e. **NINE**, not the "12 cells" stated. The loop runs four configs × 3
seeds = 12. The fourth, `"5e-5 50"`, is **absent from the enumeration** and is an escalation in
steps beyond every declared cell, contradicting "No escalation beyond these". Per the coordinator:
the coincidence between the stated "12" and the loop's 12 is **not** evidence the 50-step cell was
intended — the enumeration is the declaration, and it lists nine.

**New evidence I add: the undeclared cells are the only ones that produced the result.** All 12
records exist; from `outputs/atk_E2H_*.json`:

| lr | steps | declared? | gate-valid | per-seed `H_after` (sd 0, 1, 6) |
|---|---:|---|---:|---|
| 5e-5 | 10 | **yes** | 0/3 | 0.0000, 0.0000, 0.0000 |
| 5e-5 | 25 | **yes** | 1/3 | 0.0667(f), **0.2000**, 0.2667(f) |
| 1e-4 | 25 | **yes** | 1/3 | 0.0000(f), 0.0000(f), **0.5333** |
| 5e-5 | **50** | **NO — post-hoc** | **3/3** | **0.8000, 0.9333, 0.5333** |

Among the **nine declared** cells, only 2/9 cleared the gate and max-over-valid is 0.5333. The
**undeclared** 50-step cells are 3/3 valid and the only ones reaching substantial recovery. So the
undeclared escalation was added after the declared grid failed to falsify, and it is where the
falsifying result came from — the most consequential form this defect can take.

**Already repaired by its owner** (I do not own the file): the current header carries a
"CORRECTION 2026-10-26 (audit defect D-007)" block stating the nine-vs-twelve inconsistency
verbatim, the loop tags each config `declared` / `POSTHOC_UNDECLARED`, and the declaration text is
preserved unedited. That is the right repair. **Remaining action for FINDING78's owner:** ensure no
document reports the 50-step cells as preregistered.

---

## 6. Split integrity (A-12, A-13, A-14)

### A-13 · Leak detection still works · **(3) VALID**

`scripts/split_registry.py` is the current authority (five roles, `FINAL_PROMPTS` hard-coded,
punctuation-stripping `norm`, and a `_literal_prompt_templates` regex that recovers pools built
inside `main()` — the structural hole in v1's `verify_splits.py`). Its `audit()` cannot be executed
directly because every module it exec-loads imports torch; its set logic was reimplemented
standalone on CPU and **reproduces `outputs/split_registry_v2.json` exactly** (DEF_TRAIN 107 /
ATK_TRAIN 40 / DEV 15 / FINAL 12 / EVAL_UTILITY 21; same 16 overlap entries).

**Every leak named in `ERRATUM_split_leak.md` is still reported, unchanged.** No code was altered to
suppress them:

- `DEF_TRAIN × EVAL_TARGET(DEV)` — **4 prompts**, verdict `"LEAK CONFIRMED"`: `a photo of a church`,
  `a photo of a garbage truck`, `a photo of a parachute`, `a photo of a tench`, attributed to the
  GIFT/IMMA/UTRAP/WSR templates (and `erase_adv` for tench).
- `DEF_TRAIN × ATK_TRAIN` — **6 prompts**, all `undo_t2i` ∩ `SPLIT[c]["T"]`.
- `DEF_TRAIN × EVAL_UTILITY` — **6 prompts**, all `erase_adv` ∩ non-target `CONCEPTS`.
- `ATK_TRAIN × EVAL_TARGET` — still 0, as §4 claims.

`test_audit_repairs.py:353-367` still asserts the leak **is** found, so a silent regression would
fail the suite. **Detection integrity is sound.** Note this leak is why FINDING76's H is a
development measurement — `"a photo of a church"` is H[0] *and* a UTRAP defence-training prompt.

### A-12 · FINAL is described as untouched in six documents, the registry source, the generated JSON, and a passing test · **(1) CONFIRMED**

FINDING65/66 establish that FINAL has been **spent**: `FINDING65:3` ("the first and only use of the
FINAL prompt set"), `:85-87` ("**FINAL is now partially spent.** Four defence arms and two base
anchors have been scored on it"); `FINDING66:62-64` ("**FINAL is now further spent** … it can no
longer serve as a clean held-out set"). Corroborated on disk by **16** `outputs/final_*.json` and
`scripts/gen_eval_final.py` itself.

**Nothing renames it as untouched correctly — six places still do.** Two were edited *after*
FINDING65/66 landed, so they are live contradictions, not stale ordering:

| location | quote | note |
|---|---|---|
| `docs/T2I_REPORT_r5.md:161` | "A frozen **FINAL** set now exists and **nothing has been evaluated on it** (experiment E6)." | file mtime **11 h after** FINDING66. **Self-contradicting**: its own §8 at `:213-214` reports FINDING65/66 as FINAL results and `:248` says "Every target number except FINDING65/66's is a DEV measurement." |
| `docs/PREREG_H90_utrap_v2.md:98-99` | "**No FINAL evaluation in this study**; FINAL stays frozen and **unexamined**." | mtime ~9 h after FINDING66. First clause fine; "unexamined" false. **This is FINDING76's own prereg.** |
| `docs/PROTOCOL_FROZEN_v1.md:65` | "\| `FINAL` (H_final) \| untouched; read once, for the final number only \|" | the frozen protocol's split-role table |
| `docs/CORRECTION_REGISTER_r5.md:153`, `:260` | "not yet used for anything"; "**no arm has been evaluated on FINAL**" | |
| `docs/HANDOFF_r5.md:145` | "**No arm evaluated on FINAL**" | `:131` also says E6 is "blocked", but E6 ran via `gen_eval_final.py` |
| `docs/ERRATUM_split_leak.md:131-134` | "Until E6 runs, every target number in the corpus … is a development measurement" | |

Worse, the false status is **machine-asserted**:

- `scripts/split_registry.py:191` — `"FINAL": {… "status": "FROZEN_UNINSPECTED" …}`, a hard-coded
  literal, never computed. There is **no mechanism that detects use of FINAL** — only string
  overlap with training pools, which is why `FINAL_usable: true` still reports.
- `outputs/split_registry_v2.json` — `"status": "FROZEN_UNINSPECTED"`, `"FINAL_usable": true`.
- `scripts/test_audit_repairs.py:391` — `assert rep["splits"]["FINAL"]["status"] ==
  "FROZEN_UNINSPECTED"`. **A passing test now enforces the false label**; correcting the status
  breaks the suite. This is the item to fix first, because it actively resists repair.
- `docs/HANDOFF_r5.md:159` documents the expected output as "LEAK CONFIRMED, FINAL usable".

Only `UPSTREAM_FIDELITY_MANIFEST.md:272-275` and `UPSTREAM_STATUS_r1.md:95-98` state it correctly
("**has been inspected** … and is therefore **development data from here on**"). `AUDIT_CLOSURE_r2.md`,
`ERRATUM_interpretations_r5.md` and `AUDIT_REPAIR_REGISTER_v1.md` make no pristineness claim.

**I own `split_registry.py` but have NOT changed the `FROZEN_UNINSPECTED` literal**, because doing
so breaks `test_audit_repairs.py:391` — a file I do not own — and the two must change in one
commit. **Recommended, for whoever owns both:** compute the status (`FINAL_INSPECTED` when any
`outputs/final_*.json` exists) rather than hard-coding it, update the assertion to match, and add
the erratum to the six documents. A2's A-03 repair means a *fresh* FINAL set would be needed anyway
before any new selection.

### A-14 · The advertised hard-fail on unregistered defences is not implemented · **(1) CONFIRMED**

`split_registry.py`'s docstring (lines 11-12) and `ERRATUM_split_leak.md:100-101` both state the
registry **hard-fails** if any `erase_*.py` on disk is unregistered. **It does not.** `audit()`
computes `unknown` at line 134 and only *reports* it in the JSON — no `raise`, no `assert`, no
non-zero exit. It currently records **two** unregistered defence scripts,
`defence_scripts_on_disk_not_registered: ["erase_utrap_v2", "erase_wdr_v2"]`, while returning a
`FINAL_usable: true` leak report as if pool coverage were complete. **`erase_utrap_v2` is the
script that built all four FINDING76 arms** — so the arms in that finding were built by a defence
whose prompt pools are outside the leak audit entirely. Reported; the fix is a fail-closed check,
which is a behavioural change I have left to the owner of the test suite.

---

## 7. Metric saturation and sample counts (A-15) · **(1) CONFIRMED**, 16 sites

### The true denominators, verified from code

| metric | images | step | source |
|---|---:|---:|---|
| **H** (recovery) | 3 H-prompts × `eval_n` 5 = **15** | **1/15 = 0.0667** | `attack_relearn.py:47,212,282,400`; `clip_acc` 141-153 |
| **T** (gate statistic) | 8 T-prompts × n_per **3** = **24** | **1/24 = 0.0417** | `attack_relearn.py:44-46,281,399` |
| per-concept `clip_acc` | 3 prompts × `n_per_prompt` 5 = **15** | 1/15 | `gen_eval.py:29-46,93,135` |
| **`legacy_retain8`** | 8 concepts × 15 = **120** | **1/120 = 0.00833** | `gen_eval.py:140,157` |
| **`utility_all_nontarget`** | 11 × 15 = **165** | **1/165 = 0.00606** | `gen_eval.py:146,153` |
| **paired panel** | 6 concepts × 2 prompts × 4 = **48** | **1/48 = 0.0208** (per-concept cell 1/8) | `attack_relearn.py:108-109,223-224,188-195` |
| `prompt_adherence` | CLIP cosine mean | **continuous** | `attack_relearn.py:192` |

**Two widely-held premises are wrong, including one in my own brief and one I wrote into my own
first draft.** "retain8 (n=8)" and "all-non-target (n=11)" count **concepts, not images**. The real
steps are **1/120** and **1/165** — **15× finer** than the 1/8 and 1/11 that `TABLES_v1_generated.md:197`'s
column labels invite. One H step (0.0667) equals **eight retain8 images**. I propagated the wrong
1/8 figure into an intermediate draft of FINDING76 §7/§8 and it would have licensed discarding the
grid's **largest** utility difference (12 of 120 images) as sub-step noise. **Corrected in the
delivered rewrite, with an in-document erratum.** This is the single clearest instance in this audit
of why "trust no prior claim" has to include one's own working premises.

### Sites inferring below one quantisation step

Ranked by whether the claim drives a conclusion. Full detail with line numbers is in the audit
trail; the load-bearing ones:

| # | location | claim | reality |
|---|---|---|---|
| **S1** | `FINDING76:285-287` (my draft) | retain8 step "1/8 = 0.125" | **wrong by 15×**; step is 1/120. **Fixed.** |
| **S2** | `FINDING74:55-60` | "Methods **do** differ at low budget — the **first such ordering**"; "GIFT 1.0000 = WSR 1.0000 **>** FRP 0.9333 = IMMA = UTRAP"; "This is a **real** spread" | the `>` is **exactly one image** (15/15 vs 14/15). "Real" is explicitly chosen to contrast with noise. |
| **S3** | `FINDING72:1,22-27,31` | title: "UTRAP is recovered **as fast or faster** than FRP" | 0.7111 vs 0.6000 at n=3 = 5/45 images, and `:27` cites sd 0.19–0.28 → **SE ≈ 0.11–0.16, so the gap is ≤ 1 SE**. `:31` then says "No method ranking is asserted here either" — contradicting its own title, which is what gets cited. |
| **S4** | `FINDING68:45,50` | "a **genuine one-factor dose-response**"; "monotone rise 0.8667 → 0.9333 → 1.0000 … a **controlled curve**" | 13/15 → 14/15 → 15/15 at **n=1 per point** — each increment is **one image**. The same document's `:70-72` states the opposite rule ("Do not read the curve as having resolution it lacks"). `T2I_REPORT_r5.md:216` carries §3's framing forward. |
| **S5** | `FINDING69:46,132` | "**monotonically**, with eight distinct values"; "§1's curve … **Monotone in the means** — unaffected" | §1 is **n=1 per cell** (no means exist in §1) and three of six UTRAP increments are one image. §7 of the same document already retracted the threshold after seeds gave **sd 0.2776 = 4.2 images**. |
| **S6** | `T2I_REPORT_r5.md:103`, `FINDING63:143`, `T2I_REPORT_r4.md:95`, `PREREG_H80:58`, `FINDING62:61-63` | **three mutually inconsistent** project-wide "best operating point" superlatives | all inside **2 images** on either axis, n=1 build each. "retention *above* our FRP port's 0.9417" is **2 of 120 images**. |
| **S7** | `FINDING62:58,63` | "**2× deeper erasure**", "**half** the target accuracy" | 1/15 vs 2/15 — a **one-image** difference with ±100% ratio sensitivity. |
| **S9** | gate, §4 above | 13 verdicts decided by one image on an unrepresentable threshold | see A-06. |
| **S10** | `FINDING64:30-31` | "reproduced … **to four decimals** on the target — 0.1333 both" | 0.1333 is **2/15**; two runs on the same grid point agree to **±1 image**, not four decimals. |
| **S15** | `FINDING65:25-26,63-64,71-72` | "understates … by **3.5×** … by **3×**"; "~1.1× for UTRAP-class" | ratios of 1- and 2-image numerators: one image moves 3.5× to 2.33× or 7.0×. The 1.1× derives from a **0/15** baseline. Propagated verbatim to `T2I_REPORT_r5.md:213`. **Compounded by A-03**: these are the FINAL numbers scored with the wrong label set. |

Also recorded: **S11** `FINDING67:28-31` ("lifting retain8 from 0.9417 to 0.9500" = one image of
120); **S12** `FINDING63:95-97` (a "flattening" second difference of +9 then +3 images across
**three single builds**); **S13** `FINDING56:316-320`; **S14** `FINDING60:112-117`; **S16**
`FINDING66:28-36` (well hedged), `FINDING48:26` (an H value of **0.995**, off the 1/15 grid
entirely), `FINDING59:27-34`. One **opposite-direction** error from the same denominator confusion:
`T2I_REPORT_r5.md:117` dismisses a **14-image** retain8 spread (64/56/50 of 120) as "λ **inert** —
one arm sampled three times".

**Systemic gaps:** a grep over all `docs/*.md` for `1/120`, `1/165`, `1/48` and `1/24` returns
**zero** matches, while `1/15` appears in 19 documents. So **no document states the retain8,
11-set, 48-panel or T image denominators** — the one attempt was my own draft, and it was wrong.
Uniform 4-decimal formatting places 1/15, 1/120 and 1/165 quantities side by side at identical
width (`T2I_REPORT_r5.md:91-101`, `TABLES_v1_generated.md:198+`), which is what enables S6, S10 and
S11.

**Good practice already in the corpus, to preserve as the house standard:** `FINDING52:55-56`
("That is **one image** … a **second seed is running** to check whether it is quantisation noise
before the curve is read"), `FINDING51:34,52`, `FINDING70:72-73`, `FINDING73:78-79`,
`T2I_REPORT_r5.md:80-81` (catching a panel/denominator mix), `CORRECTION_REGISTER_r5.md:202` (exact
binomial endpoint for 15/15).

**Cheapest repairs, no GPU:** print every quantised figure as `k/N` beside the decimal (as
`FINDING65:20-23` and `FINDING68:19-21` already do); add the five true steps to
`PROTOCOL_FROZEN_v1.md`; restate the gate as `≥ 5/24` in `attack_relearn.py:211`'s **help text**
(no numerical change); reconcile S6's three superlatives into one statement with its margin in
images.

---

## 8. Cross-campaign pooling (A-16) · **(1) CONFIRMED** — same failure class as D-012

Per the coordinator's instruction, I scanned every summary script and doc aggregation for globs
spanning more than one campaign prefix. **14 runner scripts aggregate across campaigns without the
table saying so.** Measured spans:

| script:line | glob | campaigns actually pooled |
|---|---|---|
| `run_e11g_complete_factorial.sh:35` | `atk_E11*_{A}_sd*_s*` | **7** — E11B, E11C, E11D, E11E, E11H, E11I, E11O |
| `run_e12_method_axis.sh:54` | `atk_E1*_{arm}_…` + `atk_H84*_…` | **6** + a second study by explicit union |
| `run_e11b_budget_frp.sh:33` | `atk_*frp_church_r3_l05*sd0*lr1e-5*` | **4** — E11B, E11C, E11I, **E6C** |
| `run_e11_budget_curve.sh:47` | `atk_*utrap_church_l0.15k8*sd0*lr1e-5*` | **4** — E11, E11C, **E6B**, **H84M** |
| `run_e11{d,e,i,k,l,m,n,o}.sh` | `atk_E11*_{arm}_…` | 2–7 each |

The two `atk_*…` globs are the worst: they are unanchored, so they sweep **every campaign that ever
touched that adapter**, including **E6B/E6C — the FINAL-prompt post-attack campaign**, a different
split role and a different purpose. Both generators then take the **first glob match and `break`**,
so which record lands in the curve is filesystem-order-dependent. For `frp_church_r3_l05` at
`(sd0, lr1e-5, s300)` there are **4** candidate files; they happen to agree at `H_after = 1.0`, so
those cells are **right by luck, not by construction**.

Combined with A-07: an aggregate that pools campaigns *and* has 30 disagreeing duplicate cells
*and* picks by first-match is the exact configuration that produced D-012's inverted conclusion.
**I own none of these runner scripts — reported only.** `registry_v1.py`'s new `integrity` block
(A-08) makes duplicate cells and their schema strata visible, which is the part of the fix I could
make.

### Partial runs presented as complete (A-08d) · **(1) CONFIRMED**

- `outputs/e11_lr_steps_factorial.json`, from `run_e11g_complete_factorial.sh` whose header
  promises "a full 3x3 factorial at n=3". The declared 3×3 **is** complete at n=3, but the file
  emits **13 cells** and three are **n=1** (`lr 1e-5` × steps 50 / 100 / 200, seed 0 only). A
  consumer sees an unbalanced design under a name asserting balance.
- `outputs/e11_budget_curve.json`, `e11b_budget_curve.json`, `e11_full_curve_*.json` — declare a
  single `seed: 0, lr: 1e-05` while mixing seed-0 single runs with n=3 cells and cross-campaign
  rows under one "curve" label.
- **Checked and cleared** (not partial runs): the twelve `final_*.json` reporting `n_total=15`
  against 3 prompts (3 × 5 images), and `gate_bias_by_T_before.json` reporting `n=520` against 5
  bins.

---

## 9. Repairs performed, and what was deliberately left alone

| file | defect | change |
|---|---|---|
| `docs/FINDING76…md` | D-005 (a)–(j) | **full rewrite.** §1 scoped to the observed outcome; §3 records the mechanism check as absent and defines what one needs; §4 adds all raw probes with their outer steps and corrects the "overlap" claim to **disjoint**; §5a/5b/5c separate the three replication levels into three tables; §6 describes sample-matching accurately; §7 corrects the K=0 comparison to a well-resolved two-axis trade-off; §8 tabulates the four true denominators and the 5/24 gate; §8a distinguishes post-hoc probes from recovery curves; §9 is a line-by-line changelog. |
| `scripts/gen_eval_final.py` | **D-013** | CLIP text now from `ge.LABELS`, identity from `ge.LAB_KEY`, ordering asserted; records carry `label_text_set` / `label_text` / `label_text_note`; docstring corrected. |
| `scripts/gen_eval_final.py` | **D-014** | `POST_HOC_ARMS` / `PRECOMMITTED_LABELS`; `arm_precommitted`, `first_and_only_legitimate_use`, `arm_list_precommitted_in` computed per arm; `precommitment_note` added. |
| `scripts/registry_v1.py` | **D-015** | `defence_identity` (text-encoder-aware), `defence_label_collapses_text_encoder`, `schema_stratum`, `nonproduction_label` — all **alongside** the unchanged `defence_label`. |
| `scripts/registry_v1.py` | **D-016** | `vacuity_determinable`, `vacuous_suspect_from_record`. Admissibility unchanged. |
| `scripts/registry_v1.py` | **D-017** | new top-level `integrity` block: duplicate cells + disagreements + full detail, stratum census, collapsed/non-production/undeterminable counts, `quarantine_mechanism_reachable`, on-disk-vs-indexed with the unindexed list, `staleness_check`. |

**Deliberately NOT changed:**

- **`attack_relearn.py` and `gen_eval.py`** — untouched, as instructed. Their numerics are frozen
  and the ~700 historical records reproduce. Both were audited and both are **correct** on the
  questions asked (A-02, A-05).
- **`gen_eval.py`'s `legacy_retain8`** — the field is defective by design for tench targets, but it
  is correctly labelled, warned about in every record, and accompanied by two correct denominators.
  Removing it would break comparability with published tables. The live defect is in FINDING73's
  use of it, not in the code.
- **`split_registry.py`'s `FROZEN_UNINSPECTED` literal** — correcting it breaks
  `test_audit_repairs.py:391`, which I do not own. Both must change together (§6, A-12).
- **`outputs/registry_v1.json`** — not regenerated. That is a consumer action and the project is
  frozen; the registry is 112 records behind and a consumer must regenerate before quoting it.
- **`run_e2h_meta_v2_stronger.sh`** and the 14 cross-campaign runner scripts — not owned,
  reported only.

All three edited files compile clean (`ast.parse` + `py_compile`). No GPU work, no CUDA python, no
`run_*.sh` execution, no processes killed.

---

## 10. The three findings most worth acting on first

1. **A-03 — `gen_eval_final.py`'s label set.** It silently contaminated **every FINAL number in the
   project**, 26 of 180 images landed on mis-spelled labels, and the exposure is concentrated in
   exactly the four arms whose suppression figures `FINDING65`/`FINDING66` report, at near-zero
   margins. Repaired, but the historical numbers need re-scoring before any comparison to H — and
   `FINDING65`'s derived 3.5×/3× ratios should not be quoted until then.
2. **A-12 — FINAL's "uninspected" status is enforced by a passing test.** Six documents, the
   registry source, the generated JSON, and `test_audit_repairs.py:391` all assert a status that
   FINDING65/66 and 16 output files falsify. A green test suite currently *protects* the false
   claim, which makes this the defect most likely to survive an audit.
3. **A-10 + A-16 + A-07 together.** 56 of 86 arms are short of the frozen six-point grid while 24
   over-run it, 14 summary scripts pool up to 7 campaigns per table, and 30 duplicate cells
   disagree by up to 8 of 15 images with no code-version field to adjudicate them. Since
   `max-over-valid` only ever rises with attacks run, every cross-arm robustness comparison in the
   corpus is affected by a known and quantified direction of bias.
