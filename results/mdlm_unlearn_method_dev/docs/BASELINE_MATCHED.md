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

## 5. Results

*To be appended after the sweep.*
