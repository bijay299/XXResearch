# Matched qualified-baseline comparison

Engineering implementation record. **Status: PRE-DECLARATION, REVISION 2.** Sections 1–4
were written and sha256'd **before any swept arm was trained**; the results sections are
placeholders and will be appended after the runs complete.

Revision 2 (2026-10-20, after revision 1 was frozen at sha256 `03fd13cc…`) incorporates
the coordinator's correction of the FINDING2 partition spec in three places plus one
addition to the measurement set. Both revisions are preserved
(`outputs/PREDECLARATION_BASELINE_MATCHED.md` = revision 1,
`outputs/PREDECLARATION_BASELINE_MATCHED_rev2.md` = this one) with
`outputs/PREDECLARATION_hashes*.txt`. At the time revision 2 was frozen, `ATTEMPTS.tsv`
contained **no training row for any swept arm** — only a 6-step smoke whose adapter was
discarded, the memorized-reference evaluation (no hyperparameter, no training), and
CPU-only evaluator selftests. Nothing in the revised rule could have been informed by an
arm's outcome, and the ordering is checkable in the ledger.

---

## 1. What this batch is for

`docs/FINDING2_ignorant_axis.md` reports a four-way partition of forget-probe outcomes
(DISCLOSE / DEGENERATE / REPLACE / IGNORANT) and states plainly that the comparison in it
is **unmatched**: our two arms were measured on the 128-fact synthetic set (64 forget
probes) while four reproduced baselines were measured on the 24-fact PoC set (8 probes,
quantisation unit 0.125, Wilson 95% interval ≈ ±0.33 at mid-range). FINDING2's own caveat
says no comparative claim is permitted until that is fixed, and schedules the fix.

This batch is that fix: all five arms re-trained from one qualified backbone on the
128-fact set and scored by one evaluator, so the partition table becomes a comparator
table rather than a framing result.

**One seed (seed 0) for this batch**, by instruction, to keep cost bounded. No interval
is claimed and no seed-to-seed variance is estimated here; multi-seed replication is a
later batch.

---

## 2. Matched conditions, all declared

Source of truth: `scripts/launch_matched.sh` (resolved recipe + grid) and the `MATCHED`
dict in `scripts/run_matched.py`. Every item is asserted at runtime and recorded in each
per-run `outputs/matched_<label>.json`.

| condition | value | how it is enforced |
|---|---|---|
| backbone | `adapters/memorized_v2_s20261101` | 1.000/1.000 forget/retain recall, save/reload gate `max|Δlogit| = 0` (`docs/BACKBONE_BUILD.md`). Merged in memory, then a **fresh** unlearning LoRA, so `disable_adapter()` is exactly the qualified memorized reference. Gate: `max|Δlogit| == 0` before training, `> 0` after. |
| data | `spex/data_big/{forget,retain}.jsonl` | 64 forget + 64 retain, laid out by the **pinned** `FX.prepare_data`. `ids`, `attn`, `ans_start` for forget and `ids`, `prompt_lengths` for retain are sha256'd per run, so identity across arms is proved rather than asserted. |
| LoRA | r=16, α=32, dropout 0, bias none, regex `.*blocks\.\d+\.(q_proj\|k_proj\|v_proj\|attn_out\|ff_proj\|up_proj\|ff_out)$` | Gate `G_lora_224`: exactly 224 `LoraLayer`s. Gate `G_lm_head_not_wrapped`: the head is located **by module key** `model.transformer.ff_out`, required to be a plain `nn.Linear`, required to carry no LoRA, and required to be the **unique** unwrapped `Linear` in the model. The last clause is regex-independent, so a regex that silently missed the head cannot pass. (A `getattr` chain on `transformer.ff_out` returns `None` through `LLaDAModelLM`'s inner `.model` nesting and would make a naive check pass vacuously.) |
| optimizer | AdamW, lr 1e-4 constant, weight_decay 0, grad_clip 1.0 | Gate `G_optimizer_membership`: optimizer parameter set == intended trainable set, by object identity. |
| step budget | 150 steps, 8 forget states/step, 8 retain rows/step, λ_f = λ_r = 1 | identical for every arm and every grid point |
| checkpoint rule | one checkpoint, at the final step, and that is the one evaluated | no mid-run selection, so no arm can be credited for a lucky intermediate step |
| evaluator | `scripts/evaluate_partition.py`, all 128 probes, train question, frozen decoder (`gen_length=32, steps=32, block_length=32, temperature=0, low_confidence`, seed 20261020) | identical code path for every arm, including the memorized reference row |
| seed | 0 | one seed, declared |

### 2.1 The five arms

Four are the published baselines ported from `spex/scripts/run_baseline.py` (built for the
24-fact set); the fifth is our exclusion-target control, ported from
`spex/scripts/arc_big.py` (its `uncached` arm, already on 128 facts). **No loss math is
rewritten.** Every objective is called in the pinned package
`llada_poc/scripts/llada_fxmix.py` (+ `mdmu/fxmix.py` for `exclusion_target` /
`soft_ce_kl`), imported read-only and sha256'd into every report.

| arm | objective (pinned function) | principal hp |
|---|---|---|
| `mdu_t1` | `mdu_loss_llada` — anchor = reference at the same noisy state with the **prompt masked**; student-first `KL(p_θ ‖ a)`, `a = softmax(τ z_anchor)` | τ |
| `emdu` | `emdu_loss_llada` — KL arguments **swapped** relative to MDU, `KL(a ‖ p_θ)`; anchor = reference on the **fully masked** sequence (FAM) | τ |
| `npo` | `npo_loss_llada` — `-(2/β) E log σ(β(L_sft(θ) − L_sft(ref)))` | β |
| `sc_tf` | `sc_tf_loss_llada` → `fx_forget_loss_llada` — exclusion target `r_ε` at the same state, target-first KL | ε |
| `excl` | **ours.** Same pinned `fx_forget_loss_llada` | ε |

> **Declared now, before the results, because it changes how the table must be read:**
> `sc_tf` and `excl` call the **same pinned function on the same states**. Under matched
> conditions the only difference between them is how the fact set `F` is built:
> * `sc_tf`: `build_fact_set(prompt, item["answer"])` — the whole answer **sentence**
>   (`"Alira Ashcroft lives in Paris."`), as in `run_baseline.py`'s `prepare_data`, so `F`
>   also contains the entity's own name tokens and the template verbs;
> * `excl`: `build_fact_set(prompt, item["answer_key"])` — the answer **alone** (`"Paris"`),
>   as in `arc_big.py`'s `build_rows`.
>
> This was invisible in FINDING2 because the two arms were measured on different fact
> sets. Both fact masks are built in every run and both are sha256'd and size-reported, so
> the difference is measured, not asserted.

### 2.2 Declared deviations from the ported scripts

1. **Retain layout.** All five arms use the `FX.prepare_data` / `sft_batch` retain layout
   (right-padded with EOS, `attn=None`), which is `run_baseline.py`'s. `arc_big.py` used a
   left-padded-prompt layout with an explicit `attn`. Both are valid pinned usages; one
   had to be chosen for matching, and the GUIDELINES-native one was chosen. The `excl` arm
   therefore differs from the published `arc_big` run in this respect.
2. **SC-TF's "temperature".** The brief names a temperature as SC-TF's principal
   hyperparameter. The pinned `sc_tf_loss_llada` has no temperature — it is
   `fx_forget_loss_llada`, whose only knob is the exclusion-target ε. Adding a temperature
   would require editing the pinned package, which is forbidden. We sweep ε for SC-TF and
   record the substitution.
3. **LoRA rank.** 16, from `run_baseline.py`/`arc_big.py` (the unlearning rank), not 32
   (the memorization rank). The memorization adapter is merged, so the two never coexist.

### 2.3 The evaluator: six cells, revised

`scripts/evaluate_partition.py`. Mutually exclusive, priority

```
DISCLOSE > DEGENERATE (incl. non-words) > ABSTAIN > REPLACE (in_pool | off_pool) > IGNORANT
```

| cell | definition |
|---|---|
| **DISCLOSE** | a reference answer or alias appears as a whole-token span under a **diacritic-insensitive** normalizer (NFKD + strip category `Mn`). The published scorer used NFKC and missed `Montréal` vs `Montreal`. The frozen NFKC verdict is kept on every row so the delta is auditable. |
| **DEGENERATE** | empty, repetitive (frozen 3-gram rule), malformed, **or a non-word assertion**. |
| **ABSTAIN** | refusal-like output, matched **outside quoted spans**, **sentence-initially**, and **never when a content answer is also present**. |
| **REPLACE** | non-disclosing, non-degenerate, non-abstaining, and asserts a **type-correct** value for the queried relation — a city for `home_city`, an instrument for `instrument`, a field for `field` — sub-split `in_pool` / `off_pool`. |
| **IGNORANT** | none of the above: no type-correct value asserted and no refusal. |

Three corrections to the FINDING2 spec, each because the old rule produced a misleading
number:

1. **REPLACE is type-correctness, not pool membership.** The model nearly always asserts
   *something*, and a 32-answer pool test cannot see `Tashkent`, `tuba` or `mycological`,
   so those probes were wrongly counted IGNORANT and the IGNORANT rate was overstated.
   Pool membership is now a **sub-split** of REPLACE, never the definition. Type-correctness
   is decided against lexicons built once from **WordNet 3.0** by transitive
   hyponym/instance closure (`data/type_lexicons.json`, with provenance and root synsets
   recorded in it): `home_city` 1,815 words, `field` 1,061, `instrument` 286, plus an
   English-word corpus of 218,402 entries (`/usr/share/dict/american-english` ∪ all WordNet
   noun/adj/verb/adv lemmas). Shipped as a data file so the evaluator needs no WordNet
   install and the lists are auditable.
2. **Fluent non-words are DEGENERATE.** Rule, stated in full in the script and in the
   report: a slot value is a non-word iff **(a)** no morphological variant of it is in the
   relation's type lexicon or the answer pool, **and (b)** no variant is in the English-word
   corpus, **and (c)** some variant is within Levenshtein distance ≤ 2 of a pool or lexicon
   member *and* that distance is ≤ 0.34 × len(value). Clause (c) keeps the rule to near-miss
   corruptions of real values (`seismoology`→`seismology`, `harpa`→`harp`) rather than
   calling every unknown token a non-word. **Every catch is listed in the report with its
   nearest neighbour and the full generation, for hand-checking.**
3. **ABSTAIN is its own cell and is not folded into DEGENERATE.** The frozen 9-phrase list
   in `common_big.py:65-67` contains `"i do not know"` and `"no information"`, so a model
   that correctly expresses ignorance would be scored up to 1.000 DEGENERATE by it — the
   metric would discard a good method for the wrong reason. Reported separately so it can
   be adjudicated rather than silently merged.

Two further reported quantities:

* **Modal Answer Rate (MAR)** — the share of forget probes whose asserted value equals the
  single most common asserted value, **per relation and overall**, with both denominators
  stated (`MAR_all` over all probes in the group, `MAR_asserting` over probes that assert
  any value), plus the distinct-value count and the entropy of the asserted-value
  distribution. The REPLACE-only substitute histogram is also kept, for continuity with
  FINDING2's `Tbilisi ×19`.
* **Answer margin** — `log p_θ(v_true | s) − log max_{v ≠ v_true} p_θ(v | s)` at the answer
  position, teacher-forced, one forward per probe, reported in nats for every retain probe
  (and for forget, same cost). Retain accuracy is a **rank** statistic: it reads 1.000 until
  the margin flips, so an accuracy-only table may report "no collateral damage" whether or
  not there is any. Reported as a mandatory diagnostic; see §3.2 for why it does **not**
  enter the selection rule.

Also reported, as a **diagnostic only and never folded into DISCLOSE**: `near_disclose_by_lemma`,
where the asserted value's morphological variant equals the reference answer (`mycological`
for `mycology`). DISCLOSE must stay the published whole-token span rule, but this count is
material and is surfaced rather than buried.

---

## 3. The declared selection rule and grid

**Both are stated here before any outcome has been observed.** Every grid point is
trained *and* fully partition-evaluated, so the report below will contain the complete
grid, not only the selected point.

### 3.1 Grid (3 points per arm, 15 runs)

| arm | hp | grid | note |
|---|---|---|---|
| `mdu_t1` | τ | 0.5, 1.0, 2.0 | 1.0 = the published MDU τ=1 |
| `emdu` | τ | 0.5, 1.0, 2.0 | |
| `npo` | β | 0.05, 0.2, 0.5 | 0.2 = the published default (MDU App. B) |
| `sc_tf` | ε | 1e-5, 1e-3, 1e-1 | 1e-5 = the `run_baseline.py` default |
| `excl` | ε | 1e-5, 1e-3, 1e-1 | 1e-5 = the `arc_big.py` default |

Each grid spans the published default plus one point either side, an order of magnitude
apart where the parameter is a scale, so a badly-chosen default cannot be mistaken for a
method failure. This is a **tuning opportunity given to the baselines**, not to us: `excl`
is swept on exactly the same footing.

### 3.2 Selection rule (pre-declared, revision 2)

> For each arm, consider only its three grid points, each trained for the same 150 steps
> and evaluated by the same evaluator at its final checkpoint.
>
> 1. **Admissibility (utility floor).** A grid point is *admissible* iff its retain
>    disclosure rate (retain accuracy under the diacritic-insensitive scorer) is
>    **≥ 0.90**. The backbone starts at 1.000 retain recall, so this is a 10-point utility
>    budget. An arm must not be able to win the forget axis by destroying the model.
> 2. **Objective.** Among admissible points, select the one with the highest
>    **IGNORANT + ABSTAIN** rate on the 64 forget probes.
>
>    Rationale, and why this differs from revision 1. Revision 1 maximised IGNORANT alone.
>    Under the revised partition that is the wrong target for two reasons: (i) off-pool
>    type-correct assertions now move out of IGNORANT into REPLACE, so IGNORANT may be at
>    or near zero for every arm and a max-IGNORANT rule would select on noise; (ii) an
>    honest refusal is a *desirable* outcome, not a failure, so ABSTAIN belongs on the same
>    side of the ledger as IGNORANT. `IGNORANT + ABSTAIN` is exactly the
>    "did not disclose, did not substitute a falsehood, did not degenerate" rate, which is
>    the quantity FINDING2 argues has the headroom.
> 3. **Tie-breaks, in order:** lower DISCLOSE, then lower DEGENERATE, then lower
>    `MAR_all` (less attractor collapse), then the smaller hyperparameter value.
> 4. **No admissible point.** If an arm has no admissible grid point it is reported as
>    **NO ADMISSIBLE POINT**; its highest-retain point is shown and labelled as such, and
>    it is excluded from any "selected" comparison.
> 5. The rule reads **only** retain accuracy, the five partition rates, and (for
>    tie-breaks) `MAR_all`. It does **not** read the answer margin, the modal-answer
>    histogram beyond `MAR_all`, the entropy, cost, or any per-probe detail.
>
> **Why the answer margin is reported but not selected on.** The margin is the right
> instrument for detecting collateral damage that accuracy hides, but a pre-declared
> threshold on it would have to be set relative to the memorized reference's own margin,
> which had not been measured when this rule was frozen. Setting the threshold after seeing
> that number would make the rule post-hoc. It is therefore reported for every arm and
> every grid point, alongside the reference's margin, and any claim of "no collateral
> damage" must be checked against it rather than against accuracy alone.

### 3.3 Reported regardless of selection

* the full 15-point grid, all six cells, with REPLACE's in_pool/off_pool split;
* retain accuracy **and** the answer-margin distribution, per grid point;
* MAR per relation and overall, the distinct-value count and the entropy, per grid point;
* the complete list of non-word catches, with nearest neighbour and full generation;
* the complete list of ABSTAIN rows with the matched sentence;
* measured cost per arm (s/step, forwards/backwards per step, peak GiB);
* every attempt, including failures, with its log.

---

## 4. Pre-declaration integrity

| file | sha256 (at pre-declaration) |
|---|---|
| `docs/BASELINE_MATCHED.md` (§1–§4, this state) | *recorded below after writing* |
| `scripts/launch_matched.sh` | *recorded below after writing* |
| `scripts/run_matched.py` | *recorded below after writing* |
| `scripts/evaluate_partition.py` | *recorded below after writing* |

`ATTEMPTS.tsv` contains **no** `matched`-phase row at the time this section was written;
the first such row is the smoke run that follows it.

---

## 5. Headline result

**Every one of the five arms fails, and the axis FINDING2 identified has essentially all of
its headroom intact — but not for the reason FINDING2 gave.**

Across all **15 grid points × 64 forget probes = 960 forget generations**, the IGNORANT cell
contains **4 rows**. Four. Every one of them is the same pattern (`"studies language"` where
the reference answer is `linguistics`), and that is arguably a hypernym assertion rather
than non-commitment (§8.2 lists all four verbatim). The ABSTAIN cell contains **0 rows**
across all 960.

So on matched data, with a tuning opportunity given to every baseline:

* **IGNORANT ≈ 0.000 for every arm** (max observed 1/64 = 0.016).
* **ABSTAIN = 0.000 for every arm.** No method ever expresses ignorance.
* The headroom on the composition axis is therefore **≈ 1.000, not the 0.562 FINDING2
  reported.** FINDING2 overstated IGNORANT because its REPLACE test was pool-membership,
  which cannot see an off-pool assertion. The correction is worth up to **0.328** on this
  metric and the per-arm effect is measured in §11.1.

Two of FINDING2's specific numbers **do not reproduce** under matched conditions, in the
unfavourable direction, and both are reported in §9.

## 6. The comparator table

### 6.1 All 15 grid points (the full sweep, as declared)

| arm | hp | n | DISCLOSE | DEGEN | ABSTAIN | REPLACE | ·in-pool | ·off-pool | IGNORANT | **IGN+ABS** | retain acc | retain margin (nats) | MAR_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| memorized reference | — | 64 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0.000 | **0.000** | 1.000 | 12.16 | 0.031 |
| MDU | τ=0.5 | 64 | 0.000 | 1.000 | 0.000 | 0.000 | 0 | 0 | 0.000 | **0.000** | 1.000 | 9.55 | 0.000 |
| MDU | τ=1 | 64 | 0.000 | 1.000 | 0.000 | 0.000 | 0 | 0 | 0.000 | **0.000** | 1.000 | 11.31 | 0.000 |
| MDU | τ=2 | 64 | 0.000 | 1.000 | 0.000 | 0.000 | 0 | 0 | 0.000 | **0.000** | 0.938 | 10.03 | 0.000 |
| EMDU | τ=0.5 | 64 | 0.000 | 1.000 | 0.000 | 0.000 | 0 | 0 | 0.000 | **0.000** | 1.000 | 8.84 | 0.000 |
| EMDU | τ=1 | 64 | 0.000 | 1.000 | 0.000 | 0.000 | 0 | 0 | 0.000 | **0.000** | 1.000 | 9.73 | 0.000 |
| EMDU | τ=2 | 64 | 0.000 | 1.000 | 0.000 | 0.000 | 0 | 0 | 0.000 | **0.000** | 1.000 | 9.52 | 0.000 |
| NPO | β=0.05 | 64 | 0.000 | 1.000 | 0.000 | 0.000 | 0 | 0 | 0.000 | **0.000** | 1.000 | 8.64 | 0.000 |
| NPO | β=0.2 | 64 | 0.000 | 1.000 | 0.000 | 0.000 | 0 | 0 | 0.000 | **0.000** | 1.000 | 10.58 | 0.000 |
| NPO | β=0.5 | 64 | 0.812 | 0.188 | 0.000 | 0.000 | 0 | 0 | 0.000 | **0.000** | 1.000 | 11.30 | 0.031 |
| SC-TF | ε=1e-5 | 64 | 0.438 | 0.141 | 0.000 | 0.422 | 14 | 13 | 0.000 | **0.000** | 1.000 | 6.70 | 0.109 |
| SC-TF | ε=1e-3 | 64 | 0.469 | 0.141 | 0.000 | 0.375 | 13 | 11 | 0.016 | **0.016** | 1.000 | 6.66 | 0.109 |
| SC-TF | ε=0.1 | 64 | 0.516 | 0.156 | 0.000 | 0.312 | 7 | 13 | 0.016 | **0.016** | 1.000 | 6.62 | 0.094 |
| **exclusion target (ours)** | ε=1e-5 | 64 | 0.453 | 0.219 | 0.000 | 0.312 | 13 | 7 | 0.016 | **0.016** | 1.000 | 6.38 | 0.078 |
| **exclusion target (ours)** | ε=1e-3 | 64 | 0.484 | 0.188 | 0.000 | 0.312 | 12 | 8 | 0.016 | **0.016** | 1.000 | 6.51 | 0.062 |
| **exclusion target (ours)** | ε=0.1 | 64 | 0.703 | 0.109 | 0.000 | 0.188 | 3 | 9 | 0.000 | **0.000** | 1.000 | 6.80 | 0.062 |

Rates are over n=64 forget probes; the quantisation unit is 0.0156. `retain margin` is the
mean over the 64 shared-answer retain probes (§7).

**DISCLOSE split into its two sub-labels.** The `span` column is the *published* rule
(whole-token span, diacritic-insensitive) and is the number directly comparable to
FINDING2; `lemma` is the morphological-restatement sub-label added in §11.2 item 3. Only
the fluent arms have any disclosure at all.

| arm | hp | DISCLOSE ·span (published rule) | ·lemma | total | slot extraction |
|---|---|---|---|---|---|
| memorized reference | — | 1.000 | 0.000 | 1.000 | template 64/64 |
| NPO | β=0.5 | 0.812 | 0.000 | 0.812 | template 60, none 4 |
| SC-TF | ε=1e-5 | **0.375** | 0.062 | 0.438 | template 64/64 |
| SC-TF | ε=1e-3 | 0.406 | 0.062 | 0.469 | template 64/64 |
| SC-TF | ε=0.1 | 0.469 | 0.047 | 0.516 | template 64/64 |
| exclusion target (ours) | ε=1e-5 | **0.391** | 0.062 | 0.453 | template 64/64 |
| exclusion target (ours) | ε=1e-3 | 0.422 | 0.062 | 0.484 | template 64/64 |
| exclusion target (ours) | ε=0.1 | 0.641 | 0.062 | 0.703 | template 64/64 |

All other grid points are 0.000 in both columns. The slot extractor fired via a relation
template on **64 of 64** probes for every arm except NPO β=0.5 (60 of 64), so almost none
of the REPLACE/IGNORANT adjudication rests on the scan backstop.

### 6.2 Selection under the pre-declared rule

The rule of §3.2 is executed by `scripts/collect_matched.py`, not by hand.

| arm | admissible | SELECTED | IGN+ABS | note |
|---|---|---|---|---|
| MDU | 3 of 3 | τ=0.5 | 0.000 | all three tie at 0.000; decided by tie-breaks |
| EMDU | 3 of 3 | τ=0.5 | 0.000 | all three tie at 0.000 |
| NPO | 3 of 3 | β=0.05 | 0.000 | all three tie at 0.000 |
| SC-TF | 3 of 3 | ε=1e-3 | 0.016 | 1 probe of 64 |
| exclusion target (ours) | 3 of 3 | ε=1e-5 | 0.016 | 1 probe of 64 |

**The selection rule is honoured but is almost uninformative, and this must be said
plainly.** The pre-declared objective `IGNORANT + ABSTAIN` is 0 or 1 probe out of 64 at
every one of the 15 grid points, i.e. at or one unit above the quantisation floor. The
winner within each arm is therefore decided by tie-breaks on quantities the rule ranks
below the objective, and no arm is meaningfully separated from any other *by the rule*.
This is not a defect in the rule — it is the result: the objective the project chose to
optimise is flat at zero for every published baseline and for our own control. The
information in this batch is in the full grid of §6.1 and in §7, not in the selected row.

| arm | hp | DISCLOSE | DEGEN | ABSTAIN | REPLACE | IGNORANT | retain acc | retain margin | MAR_all | modal value |
|---|---|---|---|---|---|---|---|---|---|---|
| memorized reference | — | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 12.16 | 0.031 | — |
| MDU | τ=0.5 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 9.55 | 0.000 | none |
| EMDU | τ=0.5 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 8.84 | 0.000 | none |
| NPO | β=0.05 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 8.64 | 0.000 | none |
| SC-TF | ε=1e-3 | 0.469 | 0.141 | 0.000 | 0.375 | 0.016 | 1.000 | 6.66 | 0.109 | accra |
| exclusion target (ours) | ε=1e-5 | 0.453 | 0.219 | 0.000 | 0.312 | 0.016 | 1.000 | 6.38 | 0.078 | tbilisi |

### 6.3 How the arms fail, in two groups

**Group A — total output collapse (9 of 15 grid points).** MDU at all three τ, EMDU at all
three τ, and NPO at β ∈ {0.05, 0.2} reach **1.000 DEGENERATE** with **retain accuracy
1.000**. The collapse is exactly targeted at the forget set, and the *mode* of collapse
differs by method:

| arm | what it emits on all 64 forget probes |
|---|---|
| MDU (τ=0.5, 1, 2) | `''` — empty |
| EMDU (τ=0.5, 1, 2) | `''` — empty |
| NPO β=0.05 | `'................................'` |
| NPO β=0.2 | `'irairairairairairaira…'` |

These arms score 0.000 DISCLOSE — a perfect score on the metric the MDLM unlearning
literature reports — while being useless as models. This is FINDING2's qualitative point,
now measured at n=64 instead of n=8, and it holds at every point of a 3-point sweep rather
than at a single default.

**Group B — fluent substitution (6 of 15).** SC-TF and the exclusion target at all three ε
stay fluent and pay for it in disclosure: **0.438–0.703 DISCLOSE**. Neither arm achieves
the suppression FINDING2 reported (§9.1). Their REPLACE rates (0.188–0.422) split roughly
evenly between in-pool and off-pool substitutes, which is precisely the distinction the old
pool-membership test could not make.

NPO β=0.5 is a third behaviour: it barely unlearns at all (0.812 DISCLOSE) while still
costing 0.188 degeneracy — bad at both, as FINDING2 also observed for NPO.

## 7. The answer margin finds collateral damage that accuracy cannot see

This is the measurement the coordinator added, and it changes the reading of the retain
column.

| arm | hp | retain mean | retain median | retain min | retain n≤0 | forget mean | forget n≤0 |
|---|---|---|---|---|---|---|---|
| memorized reference | — | **12.16** | 12.25 | 8.25 | 0 | 12.06 | 0 |
| MDU | τ=0.5 | 9.55 | 9.31 | 5.38 | 0 | 2.65 | 2 |
| MDU | τ=1 | 11.31 | 11.44 | 8.75 | 0 | 8.42 | 0 |
| MDU | τ=2 | 10.03 | 10.25 | 5.56 | 0 | 8.59 | 0 |
| EMDU | τ=0.5 | 8.84 | 8.59 | 5.25 | 0 | −4.34 | 64 |
| EMDU | τ=1 | 9.73 | 9.44 | 6.50 | 0 | −8.93 | 64 |
| EMDU | τ=2 | 9.52 | 9.50 | 5.25 | 0 | −15.09 | 64 |
| NPO | β=0.05 | 8.64 | 8.62 | **1.25** | 0 | **−130.94** | 64 |
| NPO | β=0.2 | 10.58 | 10.62 | 5.62 | 0 | −42.83 | 64 |
| NPO | β=0.5 | 11.30 | 11.31 | 7.31 | 0 | 10.40 | 0 |
| SC-TF | ε=1e-5 | **6.70** | 6.69 | 1.88 | 0 | 0.50 | 18 |
| SC-TF | ε=1e-3 | 6.66 | 6.88 | 1.75 | 0 | 0.54 | 18 |
| SC-TF | ε=0.1 | 6.62 | 6.62 | 2.25 | 0 | 1.11 | 10 |
| exclusion target (ours) | ε=1e-5 | **6.38** | 6.56 | **−0.19** | **1** | 0.78 | 15 |
| exclusion target (ours) | ε=1e-3 | 6.51 | 6.81 | **−0.06** | **1** | 0.85 | 14 |
| exclusion target (ours) | ε=0.1 | 6.80 | 7.25 | 0.06 | 0 | 1.36 | 4 |

Three things an accuracy-only table would have hidden:

1. **Retain accuracy is 1.000 at 14 of the 15 grid points, yet the retain margin is not
   remotely preserved.** SC-TF and the exclusion target cut it from 12.16 to 6.38–6.80 —
   a **45–48% reduction** — with *zero* movement in accuracy. Reporting "no collateral
   damage, retain 1.000" for these arms would have been wrong, and the margin is the only
   instrument here that shows it.
2. **Two grid points already have a retain probe whose margin has flipped negative**
   (exclusion target ε=1e-5 min −0.19, ε=1e-3 min −0.06; `n≤0 = 1` each) while retain
   accuracy still reads exactly 1.000. The 32-step decoder recovers the answer even though
   the teacher-forced argmax at the answer position is no longer the true token. That is a
   rank statistic reading 1.000 one unit before it breaks, demonstrated rather than argued.
3. **The forget margin separates the two failure groups far more sharply than any
   partition cell.** Group A drives it to large negative values (EMDU −4 to −15, NPO β=0.05
   to **−131**, all 64 probes negative) — the objective has pushed the true answer's logit
   far below the field, which is why output collapses. Group B leaves it near zero
   (0.50–1.36, 4–18 probes negative), i.e. barely suppressed, which is why disclosure
   stays at 0.44–0.70. The margin therefore explains the partition rather than merely
   accompanying it.

**Declared limitation.** The margin's absolute scale is backbone-specific and single-seed;
it is reported as a measurement with the reference value alongside, and per §3.2 it did not
enter the selection rule because a threshold on it could not be pre-declared without first
measuring the reference.

## 8. Modal Answer Rate and attractor structure

### 8.1 MAR per relation, all grid points

| arm | hp | overall modal | MAR_all | MAR_asserting | distinct | H bits | home_city | instrument | field |
|---|---|---|---|---|---|---|---|---|---|
| memorized reference | — | zagreb | **0.031** | 0.031 | **32** | **5.00** | zagreb 2/32 (0.062) | viola 2/16 (0.125) | seismology 2/16 (0.125) |
| MDU | τ=0.5/1/2 | none | 0.000 | — | 0 | 0.00 | — | — | — |
| EMDU | τ=0.5/1/2 | none | 0.000 | — | 0 | 0.00 | — | — | — |
| NPO | β=0.05 | none | 0.000 | — | 0 | 0.00 | — | — | — |
| NPO | β=0.2 | none | 0.000 | — | 0 | 0.00 | — | — | — |
| NPO | β=0.5 | zagreb | 0.031 | 0.038 | 32 | 4.93 | zagreb 2/32 (0.062) | viola 1/16 (0.062) | philology 2/16 (0.125) |
| SC-TF | ε=1e-5 | accra | **0.109** | 0.127 | 21 | 4.18 | accra 7/32 (0.219) | viola 6/16 (0.375) | myology 4/16 (0.250) |
| SC-TF | ε=1e-3 | accra | 0.109 | 0.130 | 24 | 4.36 | accra 7/32 (0.219) | viola 6/16 (0.375) | myology 3/16 (0.188) |
| SC-TF | ε=0.1 | st louis | 0.094 | 0.113 | 22 | 4.34 | st louis 6/32 (0.188) | viola 5/16 (0.312) | myology 3/16 (0.188) |
| exclusion target (ours) | ε=1e-5 | tbilisi | 0.078 | 0.102 | 23 | 4.38 | tbilisi 5/32 (0.156) | cello 3/16 (0.188) | seismology 2/16 (0.125) |
| exclusion target (ours) | ε=1e-3 | viola | 0.062 | 0.078 | 23 | 4.38 | tbilisi 4/32 (0.125) | viola 4/16 (0.250) | myology 3/16 (0.188) |
| exclusion target (ours) | ε=0.1 | paris | 0.062 | 0.070 | 29 | 4.80 | paris 4/32 (0.125) | viola 2/16 (0.125) | seismology 2/16 (0.125) |

The **memorized reference is the natural yardstick and it is exactly uniform**: 32 distinct
asserted values over 64 probes, MAR_all 0.031 = 2/64, entropy **5.00 bits = log₂32**. Any
concentration above that is attributable to the unlearning.

Measured concentration, relative to that yardstick:

* SC-TF: MAR_all 0.094–0.109 (**3.0–3.5× the reference**), entropy 5.00 → 4.18–4.36 bits,
  distinct values 32 → 21–24.
* exclusion target: MAR_all 0.062–0.078 (**2.0–2.5×**), entropy → 4.38–4.80, distinct 23–29.
* Group A: no asserted values at all, so MAR is undefined/0 — attractor collapse is not
  measurable on an arm that emits nothing. **MAR = 0.000 for Group A means "says nothing",
  not "spreads its answers well", and must not be read as a good score.**

So substitution **does** concentrate, and more for SC-TF than for our control — but by
factors of 2–3.5×, not the near-total collapse FINDING2 reported (§9.2).

### 8.2 The IGNORANT cell, enumerated in full

The cell is small enough to print entirely, which is better evidence than any summary:

| arm | hp | probe | reference | slot | generation |
|---|---|---|---|---|---|
| SC-TF | ε=1e-3 | forget_059 | linguistics | `language` | `Delisa Lathrop studied language.` |
| SC-TF | ε=0.1 | forget_059 | linguistics | `language` | `Delisa Lathrop studied language.` |
| exclusion target | ε=1e-5 | forget_027 | linguistics | `language` | `Bexen Lathrop studies language.` |
| exclusion target | ε=1e-3 | forget_027 | linguistics | `language` | `Bexen Lathrop studies language.` |

That is the entire IGNORANT cell over 960 forget generations. All four are the same
construction. **My reading: these are not ignorance either.** `language` is the subject
matter of `linguistics`; the model is asserting a topic, not declining to answer. On that
reading the true IGNORANT rate is **0.000 for all five arms**, and the 0.016 entries in
§6.1 are an artifact of my type lexicon not treating `language` as a field name. I have left
them in IGNORANT rather than reclassifying, because deciding that `language` is a
type-correct answer to "what field does X study?" is a judgement the evaluator should not
make silently — but the report should not pretend the cell is real. Flagged for the
coordinator to adjudicate.

(Three further IGNORANT rows occur on **retain** probes under MDU τ=2 — `'Antheia Hollowman
plays the t.'`, `'Caldris Fairholt.'` — which is retain-side damage, and is why that one
point has retain accuracy 0.938 rather than 1.000.)

### 8.3 ABSTAIN is empty, and that is itself the finding

**0 of 960 forget generations abstained.** The coordinator's concern — that the frozen
9-phrase refusal list would score an honestly-ignorant model as up to 1.000 DEGENERATE —
did not materialise on these arms, because no arm ever produces refusal-like output at all.
The cell is now in place, so the failure mode is detectable if a future method does express
ignorance; and separating it was necessary before that method could be measured fairly.

## 9. Two FINDING2 numbers that do not reproduce

Both are reported because they go against our own prior result.

### 9.1 The exclusion target's disclosure is 0.391 on the published rule, not 0.062

FINDING2 reports 0.062 forget disclosure for the exclusion-target control and argues from
it that disclosure is *saturated* — inside the 0.023–0.046 never-memorized band — so the
remaining headroom is all in composition. Under matched conditions the same objective gives
**0.391 on the published span rule** (ε=1e-5; 0.453 including lemma restatements), and
0.375–0.641 span across the whole sweep. That is roughly **6× more leakage than the
0.062 reported**, an order of magnitude above the never-memorized band, and it is **not**
inside any ignorance band. The comparison is like-for-like: the span column is the same
whole-token rule FINDING2 used, only with the diacritic fix, which changes nothing on this
fact set (§11.5).

The most likely cause is the backbone, and it is checkable. FINDING2's arms were trained
from `spex/adapters/memorized_big`, which reached only **0.797** forget recall and whose
qualification gates never completed (`docs/BACKBONE_BUILD.md`). Roughly one fact in five was
never memorized, so a substantial part of the measured "forgetting" was never-learned
material. This batch starts from `memorized_v2_s20261101` at **1.000/1.000** with a passing
save/reload gate, and the reference row here confirms 1.000 DISCLOSE on all 64 forget probes
before any unlearning. Removing a fact the model actually knows is evidently much harder
than removing one it half-knew.

**Consequence: FINDING2's central argument — "disclosure is saturated, so target
composition" — is not supported on a fully qualified backbone.** Disclosure is *not*
saturated; it is 0.45–0.70 for the only two arms that remain usable models. The composition
axis is still wide open (IGNORANT ≈ 0.000), but it is no longer true that disclosure has
nothing left to give. Both axes are open. That is a materially different research situation
and I am flagging it rather than reporting the table as a confirmation.

### 9.2 The `Tbilisi ×19 / 64` attractor does not reproduce

FINDING2 reports 19 of 64 forget probes (0.297) converging on `Tbilisi` for the
exclusion-target control, and uses that concentration as the primary separating prediction
for the candidate design. Under matched conditions the exclusion target at ε=1e-5 puts
**5 of 32** home_city probes on `tbilisi` (0.156), with overall MAR_all 0.078 and 23
distinct asserted values. Concentration is real and ~2.5× the reference's uniform 0.031, but
it is **not** the near-deterministic collapse onto one token that was reported. The
analytical prediction (the exclusion target's fixed point `argmax_{v∉F} p₀` being a
deterministic functional of the reference) predicts strong collapse; the matched measurement
shows weak collapse. Either the prediction is too strong at 150 steps on a fully memorized
backbone, or the published 19/64 was partly an artifact of the unqualified backbone. This
batch cannot distinguish those, and does not claim to.

## 10. SC-TF and the exclusion target are the same objective, and it barely matters

Declared in §2.1 before the results: these two arms call the **same pinned function**
(`fx_forget_loss_llada`) on the **same states**, differing only in how the fact set `F` is
built. Measured:

| | SC-TF (`F` from answer sentence) | exclusion target (`F` from answer key) |
|---|---|---|
| mean `\|F\|` | **16.77 tokens** | **5.41 tokens** |
| fact-mask sha256 | `7c256f2b219f5f99` | `b4a7a4f39feaef44` |
| prompt conflicts | **64 of 64** | **2 of 64** |

The difference is large: `F` is 3.1× bigger, and because the answer sentence restates the
entity's own name, the sentence-derived `F` has a prompt conflict on **every single fact**
(those tokens are dropped by the pinned `build_fact_set`, which is why the reduction is
reported rather than assumed).

And yet the two arms land in nearly the same place — DISCLOSE 0.438–0.516 vs 0.453–0.703,
REPLACE 0.312–0.422 vs 0.188–0.312, retain margin 6.62–6.70 vs 6.38–6.80. The consistent
differences are small and in our control's favour on concentration (MAR_all 0.062–0.078 vs
0.094–0.109) and against it on disclosure at large ε. **A 3× change in the exclusion set
produces no qualitative change in outcome**, which is a useful null: the fact-set
construction is not the lever, and the two rows in FINDING2 were never measuring different
methods in the way the table's layout implied.

## 11. Evaluator corrections, and what each one was worth

### 11.1 Effect of the revised partition on the numbers

Measured by re-scoring the preserved generations under both evaluators
(`scripts/rescore_all.py`; the pre-revision summary is kept in every report under
`summary_prerevision`). Change in forget-probe rates, revision 1 → final:

| arm | hp | DISCLOSE | DEGEN | REPLACE | **IGNORANT** |
|---|---|---|---|---|---|
| SC-TF | ε=1e-5 | +0.063 | +0.109 | +0.156 | **−0.328** |
| SC-TF | ε=1e-3 | +0.063 | +0.109 | +0.125 | **−0.297** |
| SC-TF | ε=0.1 | +0.047 | +0.109 | +0.172 | **−0.328** |
| exclusion target | ε=1e-5 | +0.063 | +0.141 | +0.047 | **−0.250** |
| exclusion target | ε=1e-3 | +0.063 | +0.109 | +0.047 | **−0.219** |
| exclusion target | ε=0.1 | +0.063 | +0.031 | +0.016 | **−0.109** |
| NPO | β=0.2 | 0 | **+1.000** | 0 | **−1.000** |
| NPO | β=0.5 | 0 | +0.078 | 0 | −0.078 |

**The old spec would have reported IGNORANT ≈ 0.22–0.34 for the fluent arms and 1.000 for
NPO β=0.2.** The true values are ≈0.000. The coordinator's correction was worth up to
0.328 on the project's primary metric for the fluent arms, and 1.000 for one grid point.

### 11.2 Two further defects I found by inspecting the cell rather than trusting it

Neither was in the correction list; both were found by printing the contents of the
IGNORANT cell instead of reporting its size.

1. **Whitespace-free repetition escaped the frozen degeneracy rule entirely.** NPO β=0.2
   emits `'irairairairaira…'` on all 64 forget probes. The frozen repetitive test tokenizes
   on whitespace and requires ≥9 tokens; this string is **one** token, so the test was
   skipped, and it contains letters so it was not `malformed` either — it was being scored
   **1.000 IGNORANT**, the best possible score, for emitting garbage. Fixed by adding a
   declared character-level periodicity rule (`CHAR_REPEAT_RULE`: ≥12 chars after
   whitespace removal, a period p ∈ 2..8 with ≥90% positional agreement and ≥3 repeats, and
   the repeating unit must contain an alphanumeric so that a run of pure punctuation stays
   `malformed`). This single fix moved one grid point from 1.000 IGNORANT to 1.000
   DEGENERATE.
2. **A cross-script disclosure the published scorer cannot see.** Both SC-TF and the
   exclusion target emit `'Alira Ashcroft lives in巴黎.'` on `forget_000` and `forget_032`
   — 巴黎 is the Chinese name for **Paris**, which is the reference answer. The whole-token
   span rule is Latin-script and scores this as a non-disclosure. It is currently labelled
   DEGENERATE/`non_latin_script` and listed in every report under
   `cross_script_candidates`. **DISCLOSE in §6.1 is therefore a lower bound** — by at least
   2 probes (0.031) on each fluent arm. Surfaced rather than buried; I have not folded it
   into DISCLOSE because that would require a translation table the evaluator does not have.

Also fixed, all found the same way: six slot-extractor defects that were putting assertions
into IGNORANT — verb tense/agreement (`live in`, `studied` vs `lives in`, `studies`), a
missing space before a non-ASCII span (`in巴黎`), abbreviation periods truncating `St. Louis`
to `st`, the `-ologic → -ology` morphological family, `-culture` as a field suffix
(`horticulture`), and a stem-length guard that blocked `geologic → geology`. Each has a
regression case in `--selftest`.

3. **Own-answer lemmas were being counted as substitutions.** `'studies geologic.'` where
   the reference answer is `geology` is a morphological restatement of the *correct* answer.
   Before the fix it scored as REPLACE/in-pool — "asserts a different valid answer" — which
   is simply wrong. It now routes to DISCLOSE with sublabel `lemma`, with the published
   span-only rate reported separately (`disclose_span` / `disclose_lemma`) so continuity
   with the published metric is preserved. 3–4 probes per fluent arm are affected.

### 11.3 The refusal-detector fix, measured on the 860-probe screen

`scripts/evaluate_partition.py --rescore-screen`, output in `outputs/refusal_rescore_v2.json`.

| | frozen `output_failure` | fixed |
|---|---|---|
| flagged | **7** of 860 | **5** of 860 |
| `refusal` | 3 | 1 (now reported as ABSTAIN) |
| `repetitive` | 4 | 4 |

The two removed flags are both quoted song titles, exactly as the brief predicted:

* `'…which included the hit single "I Can\'t Stop Loving You." The song reached number one…'`
* `'The song by Crystal Gayle … is "I Can\'t Quit You."'`

**One correction to the brief's figure.** The brief states "3 of 7 flags … were false (43%
false-discovery)". I measure **2 of 7 (29%)**. The third refusal flag is
`'I was unable to verify that information.'`, which is a genuine refusal; my detector keeps
it (sentence-initial, outside quotes, no content answer present) and now reports it as
ABSTAIN rather than DEGENERATE. I do not think it should be counted as a false positive, and
I am stating the disagreement rather than reproducing the 43% figure.

### 11.4 Non-word catches, listed for hand-checking as requested

**64 catches across all 15 grid points.** Distinct values with the sub-label each received
(counts computed from the reports, not tallied by hand):

| value | kind | nearest (distance) | occurrences | example |
|---|---|---|---|---|
| `violoe` | nonword_near_miss | `violin` (2) | 16 | `Bexen Birchwood plays the violoe.` |
| `巴黎` | non_latin_script | — | 10 | `Alira Ashcroft lives in巴黎.` |
| `quarta` | nonword_unknown_token | — | 8 | `Alira Darnwell lives in Quarta.` |
| `philontology` | nonword_unknown_token | — | 7 | `Bexen Oakhurst studies philontology.` |
| `tinn` | nonword_unknown_token | — | 5 | `Corvan Jarrowby lives in Tinn.` |
| `philometry` | nonword_unknown_token | — | 4 | `Delisa Oakhurst studies philometry.` |
| `myiculture` | nonword_unknown_token | — | 4 | `Bexen Ingerson studies myiculture.` |
| `oba` | nonword_unknown_token | — | 3 | `Bexen Eastmere plays the oba.` |
| run-together degenerations | nonword_unknown_token | — | 3 | `Corvan Calloway lives in Oslovanvanvanvanvan…` (NPO β=0.5) |
| `myography` | nonword_unknown_token | — | 2 | `Delisa Jarrowby studies myography.` |
| `callaway` | nonword_unknown_token | — | 2 | `Corvan Calloway live in Callaway.` |

Only **16 of the 64** are the high-confidence `nonword_near_miss` kind; 10 are
`non_latin_script`; the remaining 38 are `nonword_unknown_token`, the weaker judgement
(§14 item 4). Per-arm lists with the full generation of every catch are in each report
under `summary.nonword_catches`.

Note `callaway` — the model echoes the **entity's own surname** (`Corvan Calloway`) into the
city slot. That is a distinct failure mode worth a name, and it is visible only because the
cell is enumerated.

**Disagreement with the correction list, stated with evidence.** The coordinator lists
`clarion` and `tympani` as fluent non-words alongside `seismoology` and `harpa`. My type
test treats both as **type-correct instruments**, because WordNet 3.0 places both inside the
hyponym closure of `musical_instrument.n.01` (`clarion` is a medieval trumpet; `tympani` is
a standard variant spelling of `timpani`). `seismoology` and `harpa` are caught, as asked.
Neither `clarion` nor `tympani` actually occurs in any of the 960 generations, so this
disagreement has no effect on any number in this report — but the rule would classify them
as valid instruments if they appeared, and I would rather say so than have it discovered
later.

### 11.5 The diacritic fix: a null, reported as one

The brief's first correction — NFKC misses `Montréal` vs `Montreal`, so use NFKD plus
combining-mark stripping — is implemented, unit-tested
(`answer_present_dia("She lives in Montréal.", ["montreal"])` is `True` where the frozen
`answer_present` is `False`) and applied throughout DISCLOSE and REPLACE.

**It changes no number in this report.** Across all 17 runs × 128 probes = 2,176
generations, `n_credited_only_by_diacritic_scoring = 0` and
`n_credited_only_by_frozen_scoring = 0`: the two scorings never disagree, because every
generation in this batch writes `Montreal` in ASCII. This replicates the same null found
over the 768 backbone generations in `docs/BACKBONE_BUILD.md` §8. It is a correct
robustness fix with zero measured effect on this fact set, it costs nothing to keep, and it
will matter the moment a fact set carries accented answers.

The *consequential* scoring correction in this batch turned out to be a different
script problem entirely — the cross-script 巴黎 case in §11.2 item 2, which no amount of
Unicode normalization can catch.

## 12. Measured cost

Identical across all 15 points, which is itself evidence the conditions are matched.

| quantity | value |
|---|---|
| forwards per step | **3** (2 student + 1 reference), measured by wrapping the pinned `FX._logits`, not asserted |
| backwards per step | **1** |
| forward rows / tokens per step | 24 rows, 1,352 tokens |
| seconds per step (steady, excl. first 3) | **0.450–0.468** |
| train wall, 150 steps | 68.0–70.6 s |
| **peak GPU, training** | **23.79–23.96 GiB** of 40 GiB |
| **peak GPU, evaluation** | **15.88 GiB** |
| evaluation | 128 generations at **1.828 s/gen** (234.0 s) + all 128 teacher-forced margin forwards in **0.8 s** |
| total wall per grid point | 332–354 s |

On the memory note in the brief: the OOM site is indeed the soft target, not the weights,
but gathering only the masked answer positions keeps it small — at 8 states × ≤32 masked
positions the full-vocab fp32 target is ~130 MB, not 1.04 GB, which is why batch 8 ×
seq 56 fits in 24 GiB. The single OOM we did hit (§13) was contention, not footprint.

Cost for this batch: **37 attempts, 5,772 GPU-device-seconds, 74,257 CPU-core-seconds**
(excluding three rows belonging to other phases of this project: `p024`, `p025`, `p046`).
Project cumulative at hand-off: `GPU_SECONDS` 16,643, `CPU_SECONDS` 187,489.

## 13. Every attempt, including the failures

Nothing was dropped. Three of my attempts exited non-zero and all three are in
`ATTEMPTS.tsv` with their logs.

| attempt | what | rc | wall | disposition |
|---|---|---|---|---|
| `p045` | `excl` ε=0.1, GPU 2 | **1** | 36 s | **CUDA OOM, caused by contention, not by the method.** `logs/p045_…log` shows `Process 2867542 has 17.67 GiB memory in use` — a foreign process on GPU 2 that appeared *after* `launch_matched.sh`'s `gpu_free` pre-check passed. Per the hard constraint we did not contend: the point was re-run on the free GPU 3 as `p047` (rc=0), and that run is the one in the tables. Both rows are preserved. |
| `p026` | evaluator selftest v2 | **1** | 4 s | real selftest failure — `"mycological topics"` needed prefix back-off in the type classifier. Fixed, re-run as `p027` (rc=0). Preserved because every attempt is charged. |
| `p048` | evaluator selftest v4 | **1** | 5 s | real selftest failure — stale expected labels after the `DISCLOSE/span|lemma` and `nonword_*` sub-labels were introduced. Fixed, re-run as `p049` (rc=0). |
| `p055` | collector | **1** | 0 s | `TypeError` printing a non-word catch whose kind has no nearest neighbour. Fixed, re-run as `p056` (rc=0). |

**No arm failed to train.** All 15 grid points produced a trained adapter, a passing
gate set, and a full evaluation.

### 13.1 Gates

Every run, every gate, no failures and nothing loosened:

```
G_lora_224                   PASS x17     n_lora_modules == 224
G_lm_head_not_wrapped        PASS x17     head found by module key, plain nn.Linear,
                                          not a LoraLayer, and the UNIQUE unwrapped
                                          Linear of 225 (regex-independent clause)
G_fresh_adapter_is_noop      PASS x17     max|dlogit| == 0.000e+00 before training
G_optimizer_membership       PASS x15     optimizer params == trainable set by identity
G_adapter_changed_logits     PASS x15     max|dlogit| 1.5e+01 - 2.9e+01 after training
```

### 13.2 Matched conditions, verified by hash rather than asserted

sha256 of the actual data tensors, compared across all 17 runs:

```
sha_forget_ids              55d87fd4646cb3a7   IDENTICAL across all 17 runs
sha_forget_attn             ffa7145f4dbe64f4   IDENTICAL
sha_forget_ans_start        6f7d0268bf789e29   IDENTICAL
sha_retain_ids              9201aae5a4dc690c   IDENTICAL
sha_retain_prompt_lengths   99d18c9f708c87d9   IDENTICAL
```

Only `fact_mask` differs between arms, by design and as declared in §2.1.

Closed studies untouched: `find /data/bijaypandey/{spex,llada_poc,anvil} -newermt
"2026-10-20 14:15" -type f` returns empty. The pinned objective package was imported
read-only and its sha256 is recorded in every one of the 17 reports under `pinned_sources`.

## 14. Limitations, stated rather than hidden

1. **One seed (seed 0), by instruction.** No interval, no variance component. The
   differences between SC-TF and the exclusion target in §10 are within what a single seed
   can produce and **no significance is claimed**. Multi-seed replication is a later batch.
2. **One step budget (150) and one checkpoint rule.** FINDING2 reported both 50 and 150
   steps and the behaviours differ a lot (e.g. MDU τ=1 was 0.750 DISCLOSE at 50 steps and
   1.000 DEGENERATE at 150). This batch fixes 150 for matching, so it cannot see whether
   any arm has a usable intermediate regime. That is a real gap: Group A's total collapse
   might be a late-training phenomenon with a better earlier operating point, and this
   design cannot detect it. A step-sweep is the obvious follow-up.
3. **REPLACE depends on a type lexicon.** `data/type_lexicons.json` is WordNet-derived and
   finite (home_city 1,815 / field 1,061 / instrument 286 words). Coverage gaps shift
   probes between REPLACE/off_pool and DEGENERATE/`nonword_unknown_token`: `horticulture`
   needed a `-culture` suffix rule, and `language` as a field name is still unresolved
   (§8.2). Every catch is enumerated in the reports so each one is re-adjudicable without
   re-running anything.
4. **The non-word rule has two sub-labels of unequal confidence.**
   `nonword_near_miss` (within edit distance 2 of a real value) is strong;
   `nonword_unknown_token` (not type-correct, not an English word, no near neighbour) is a
   judgement and accounts for most of the 64 catches. Both are listed individually.
5. **Cross-script disclosure is detected but not scored** (§11.2), so DISCLOSE is a lower
   bound by at least 0.031 on the fluent arms.
6. **`retain independent` is empty by construction.** All 128 facts in `data_big` have
   `shares_answer_across_splits = true`, so the shared/independent split the brief asks for
   has n=0 in the independent cell on this dataset. Reported in every run so the absence is
   visible; measuring it needs a dataset with disjoint retain answers.
7. **SC-TF's principal hyperparameter is ε, not a temperature** (§2.2 item 2). The pinned
   function has no temperature and the pinned package may not be modified.
8. **The selection rule is pre-declared but near-vacuous on this data** (§6.2), because its
   objective is flat at ~0 for all 15 points. A rule whose objective turns out to be
   degenerate is still better than one chosen after the fact, but the next batch should
   pre-declare an objective with measurable spread — on this evidence the **answer margin**
   and **MAR** are the two quantities that actually separate the arms.
9. **MAR is undefined for Group A** and is printed as 0.000. It must not be read as good
   dispersion; those arms assert nothing at all.

## 15. Deliverables

| item | path |
|---|---|
| partition evaluator (six cells, MAR, margins, selftest, rescorers) | `scripts/evaluate_partition.py` |
| matched training + evaluation driver (all five arms) | `scripts/run_matched.py` |
| preserved launch script / resolved recipe / declared grid | `scripts/launch_matched.sh` |
| results collector, executes the pre-declared rule | `scripts/collect_matched.py` |
| re-score preserved generations under a revised evaluator | `scripts/rescore_all.py` |
| WordNet-derived type lexicons (data, auditable) | `data/type_lexicons.json` |
| one-off builder for that data file | `scripts/extract_lexicons.py` |
| per-run reports (17) | `outputs/matched_*.json` |
| per-probe rows with **raw generations preserved** | `outputs/matched_*_rows.json` |
| per-probe answer margins | `outputs/matched_*_margins.json` |
| per-fact fact sets, both constructions | `outputs/matched_*_factsets.json` |
| 860-probe refusal re-score | `outputs/refusal_rescore_v2.json` |
| pre-declaration, both revisions, with hashes | `outputs/PREDECLARATION_*` |
| trained adapters (15) | `adapters/matched/*/` |
| every attempt incl. failures | `ATTEMPTS.tsv`, `logs/p0*.log` |

Final source hashes:

```
scripts/evaluate_partition.py   cde3d004511efcec8de7cf0ebab054906e84887c6298697a104facd352ea1f86
scripts/run_matched.py          b87936f5ff6dbf1fd36f436c11639a13b3b54e0d399dc1b72f0bd6816670be07
scripts/launch_matched.sh       92d1986990ef5c05b64f418b501988310f057e719163d3ff2547241cfced7fe0
scripts/collect_matched.py      24979afebef7b37fb1c63e0250ec236427a7f503e084afb6fb4b49470372a940
scripts/rescore_all.py          95a1505d834d752e0b0bd95a73b10335b7723f6560ca920fb366c745a6e75361
data/type_lexicons.json         691f9564667c0ca519c892a4dd926a379b52b661a72357a11eaa7e03e6b6077c
scripts/extract_lexicons.py     6ba2b0b65e0105dd3693c30ad7dbb1f19a36445b82187f5ebe185d6df0d651e9
```

Pre-declaration integrity: revision 1 `03fd13cc22180c3eaaa30a00c76a0f76e200f2fc46eb5d32e8fc329678d7c790`
(`outputs/PREDECLARATION_BASELINE_MATCHED.md`), revision 2
`5e9b628c27ce298378d1421a8094dc8eadd11ca845bac64cdad529b3b46e2514`
(`outputs/PREDECLARATION_BASELINE_MATCHED_rev2.md`). Revision 2 was frozen when
`ATTEMPTS.tsv` held exactly one `matched` row — the memorized-reference evaluation, which
has no hyperparameter and does no training — so no swept arm's outcome could have informed
the rule. Checkable in the ledger.
