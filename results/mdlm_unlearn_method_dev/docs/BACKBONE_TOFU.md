# TOFU backbones — θ₀_tofu and retain_ref_tofu

Engineering implementation record. **Sections 1–4 are a PRE-DECLARATION**, written and
sha256'd **before either production run was launched**; the results sections are appended
afterwards. The ordering is checkable in `ATTEMPTS.tsv`: at the time of writing it
contains the pristine-base floor (`p061`), the wiring smoke (`p062`, a preserved rc=1),
and three calibration runs, and **no `qualify`-mode run of either backbone**.

---

## 1. Why we are off the synthetic set

Two independent defects, one of them mine.

1. **FINDING5 — the 128-fact synthetic set has a fatal confound.** All 64/64 forget facts
   share *both* the surname and the answer with a retain partner
   (`Alira Ashcroft → Paris` / `Zelvira Ashcroft → Paris`), so `surname → answer` is
   perfectly predictive and learnable from the retain split alone. A retain-only reference
   recalls **all 64** forget facts (recall 1.000). On that data "forget while retaining" is
   not merely hard, it is contradictory — the retain objective *teaches* the forget answers.
2. **My own matched-baseline result** (`docs/BASELINE_MATCHED.md` §9.1): on a fully
   qualified backbone the exclusion target discloses 0.391 on the published span rule, not
   the 0.062 published from the unqualified `memorized_big`. So disclosure is **not**
   saturated and both axes are open.

The replacement is `data/tofu/tofu800.json`, sha256
`190a685043aac0b0dcc7dd82bf6fad8f24f598ee94398092f90494b34c49dd69` (the value quoted in the
brief is the first 32 characters of that sha256, and it matches). Verified independently in
gate `G0b` of every run: 400 forget QA / 20 authors, 400 retain QA / 20 authors,
**author-disjoint**, **800/800 distinct answers with zero answer overlap between splits**.

Measured with the pinned tokenizer (`p059`): prompts mean 28.9 tokens (max 63), answers
mean 34.4 tokens (max 82) — against one word per fact on the synthetic set.

---

## 2. What is carried over, and what is new

`scripts/memorize_v2.py` and `scripts/common_v2.py` are a **published artifact** — their
sha256 is recorded inside the three delivered synthetic backbone reports — so **I modified
neither**, and the synthetic path still runs. (`memorize_v2.py` has nonetheless drifted
from its recorded hash, changed by another phase of this project at 14:51; see §11.1,
which reports it because the provenance link matters even though the cause is not mine.)
The new files are
`scripts/memorize_tofu.py`, `scripts/common_tofu.py`, `scripts/eval_tofu.py` and
`scripts/launch_tofu.sh`; everything data-independent (`load_base`,
`count_lora_coverage`, `forward_process`, `forward_process_canvas`, `step_losses`, the
sha256 helpers) is imported read-only from `common_v2`, which itself imports the frozen
`spex/scripts/common_big.py` read-only. The author's masked-diffusion sampler is taken
from that same frozen module, so this path uses the *identical function object* the
synthetic path uses.

Carried over unchanged:

| kept | why it matters |
|---|---|
| the **real save/reload gate** (G3) | adapter saved, trained model destroyed, a FRESH base instantiated, adapter reloaded, behavioural identity asserted on 8 generation-shaped fully-masked canvases + 4 training-shaped teacher-forced rows + 8 greedy generations compared as both string and token ids, plus bit-identical LoRA tensors. Tolerance 1e-3 with the bf16-ULP justification; three of the conditions are tolerance-free. |
| the **224-module assertion** | exactly 224 `LoraLayer`s, and the LM head located **by module key** `model.transformer.ff_out`, required to be a plain `nn.Linear`, to carry no LoRA, and to be the **unique** unwrapped `Linear` of 225. A `getattr(model, "transformer")` chain returns `None` through `LLaDAModelLM`'s inner `.model` nesting and would make the check pass **vacuously**. |
| the **pinned loss normalization as default** | `--norm pinned` = `sum(tok/ans_len)/batch` from `llada_fxmix.py::sft_loss_llada`; masked-mean computed every step from the same per-token tensor as a logged diagnostic. |
| `--canvas genlen` | the answer region is EOS-padded to `gen_length` and the **whole** region is noised, so training sees the fully-masked canvas the decoder starts from. Decisive on the synthetic set (0.844/0.812 → 1.000/1.000). Expected to matter **more** here, not less: a TOFU answer has variable length, so the model must also learn where to **stop**. Tested against `exact` rather than assumed (§8). |
| the full `report.json` + preserved launch script | every argument, the resolved config, the per-step loss/grad-norm history, all gates, throughput, peak GPU, dataset and source hashes, library versions, the wrapped-module list, and the exact launch command. `scripts/launch_tofu.sh` is the single source of truth and is sha256'd into every report. |

New, and required:

* **DISTINCTIVE content recall** (§3) replaces alias span matching, which is meaningless
  for a 34-token free-form answer.
* **`gen_length` 96**, from the measured answer-length distribution: it covers **100%** of
  the 800 gold answers. `gen_length` 64 — used by the earlier third-party-checkpoint
  qualification — covers only **98.6%**, so that qualification truncated 1.4% of answers
  and could not have scored them.
* Degeneracy uses the frozen `output_failure` **plus** the character-level repetition rule
  from `evaluate_partition.py`. The frozen rule tokenizes on whitespace and needs ≥9
  tokens, so a run-together degeneration is one token and escapes it entirely — the hole
  that scored an arm 1.000 IGNORANT in the matched-baseline batch.

---

## 3. Scoring: distinctive content recall

```
scorable(q, a) = { non-stopword tokens of normalize(a) }  MINUS  { all tokens of normalize(q) }
recall(g,q,a)  = | scorable(q,a) INTERSECT tokens(g) | / | scorable(q,a) |
```

`normalize` is `common_v2.normalize_dia` (NFKD + combining-mark strip + casefold +
punctuation→space + article removal), so the whole project shares one normalizer. The
stopword list is declared explicitly in `common_tofu.STOPWORDS`, printed in full into
every report, and hashed (`stopwords_sha256`). It includes the orphan possessive `s`,
because `normalize_dia` splits `author's` into `author s` and that `s` would otherwise
count as distinctive content.

**Exclusions.** An item whose scorable set is empty cannot distinguish knowledge from echo
and is excluded from the mean and counted. Gate `G0` reports the count. **Measured on
tofu800: 0 of 800 items are excluded** (mean 12.0 distinctive tokens per item, min 1,
max 33), so no item is dropped — stated because the brief asked for the number.

**Naive content recall and ROUGE-L recall are computed on every probe and reported
alongside**, so the question-echo inflation is visible rather than hidden.

Why this is not optional, demonstrated on the data before any model was trained (`p060`):
pure question echo — emitting the question back verbatim — scores

| metric | question echo |
|---|---|
| distinctive content recall | **0.000** |
| naive content recall | 0.500 |
| ROUGE-L recall | 0.250 |

---

## 4. The pristine-base floor, and the thresholds justified against it

### 4.1 Floor, measured on all 800 probes under the identical evaluator

`p061`, `--adapter none`, 2,276 s, 2.845 s/generation, peak 15.06 GiB.
`outputs/tofu_floor_base.json`.

| split | n | **DISTINCTIVE** | naive | ROUGE-L | degenerate |
|---|---|---|---|---|---|
| forget | 400 | **0.0785** | 0.3539 | 0.3455 | 97 |
| retain | 400 | **0.0685** | 0.2924 | 0.3011 | 72 |
| all | 800 | **0.0735** | 0.3231 | 0.3233 | 169 |

Dispersion (needed for the ignorance ceiling): forget SD 0.1428, **SE 0.00714**; retain SD
0.1120, SE 0.00560.

**This is the measurement that justifies the metric.** A model that has never seen the data
scores **0.323 naive content recall and 0.323 ROUGE-L recall**, against **0.0735
distinctive**. Both inflated metrics are **4.40×** the distinctive value, an absolute
inflation of **+0.250**. An unlearning result reported on either of them would be reporting
mostly question echo. It also confirms the evaluation lead's independent measurement
(naive ≈ 0.33 on the pristine base) on a different 800-probe set.

### 4.2 Declared thresholds

Both values were written into `scripts/launch_tofu.sh` **before** the floor was measured,
and the floor measurement is reported here as the justification check — not as the source
of the numbers.

**Memorization gate: distinctive content recall ≥ 0.60**, applied to each split a run
trains on.

* 0.60 is **7.6×** the forget floor of 0.0785 and an absolute lift of **+0.52**.
* That lift is **6×** the lift the rejected third-party checkpoint achieved (0.087) and
  **1.7×** the 0.30 lift gate the evaluation lead applied to it.
* It is **73 SE** above the floor, so a pass cannot be floor-level noise.
* It demands that the backbone reproduce a clear majority of the distinctive content it was
  trained on. It is a floor on success, not a ceiling on ambition; the achieved value is
  reported whether or not it clears the gate, and **the gate will not be lowered to make a
  run pass**.

**Ignorance ceiling for the retain-only reference: forget distinctive recall ≤ 0.15.**

* The forget floor is 0.0785 with SE 0.00714, so 0.15 is floor **+0.0715**, i.e. floor
  **+10.0 SE**.
* Deliberately **generous**: the reference is allowed to exceed the never-saw-it floor by
  ten standard errors before failing. A PASS is then strong evidence and a FAIL is
  unambiguous.
* It discriminates what we care about by a factor of ~6.7: the synthetic set produces
  **1.000** on this gate.
* Reported alongside the binary gate, because they are the quantities that actually carry
  the information: **leakage above floor** = forget recall − 0.0785 (in SE units), and the
  **ignorance margin** = retain recall − forget recall.

### 4.3 Calibration plan (results in §8)

Declared before the scan: vary **canvas** at a fixed 1500 steps, then **steps** with the
better canvas, scoring 64 probes per split.

| run | canvas | steps | passes | s/step | distinctive forget | distinctive retain |
|---|---|---|---|---|---|---|
| `p064` | `genlen` | 1500 | 7.5 | 0.2107 | 0.3254 | 0.4173 |
| `p065` | `exact` | 1500 | 7.5 | 0.1917 | *appended below* | *appended below* |
| `p066` | `genlen` | 6000 | 30 | — | *appended below* | *appended below* |

---

## 5. THE DECISION MEASUREMENT — report this first

Six-cell partition of each model's own generations for the **same 400 forget QA**,
`scripts/partition_tofu.py`, run over preserved generations (no training, no GPU).
Priority `DISCLOSE > DEGENERATE > ABSTAIN > REPLACE > IGNORANT`, τ_disclose = 0.50.

| model | n | DISCLOSE | DEGEN | ABSTAIN | **REPLACE** | IGNORANT | IGN+ABS |
|---|---|---|---|---|---|---|---|
| pristine base | 400 | 0.030 | 0.000 | 0.113 | 0.470 | 0.388 | 0.500 |
| θ₀_tofu | 400 | 0.900 | 0.000 | 0.000 | 0.040 | 0.060 | 0.060 |
| **retain_ref_tofu** | 400 | 0.033 | 0.120 | **0.000** | **0.580** | 0.268 | **0.268** |

### Verdict against the pre-registered decision table

**The retain reference lands mostly in REPLACE (0.580, the plurality cell by more than
2×). Per your pre-registration: the target's ceiling is the wrong cell → DO NOT BUILD the
match-the-retain-reference mechanism.**

Three things make that verdict stronger than the single number:

1. **Training on retain made it worse than not training at all.** The pristine base, which
   never saw any of this data, is *better* in exactly the dimension that matters:
   REPLACE **0.470 → 0.580**, IGNORANT 0.388 → 0.268, and ABSTAIN **0.113 → 0.000**.
   Retain SFT taught the model the confident TOFU answer style and **destroyed the only
   abstention behaviour it had**. The reference you would be matching is a more confident
   confabulator than the raw checkpoint.
2. **0.580 is a lower bound on confabulation.** 37 of the 48 DEGENERATE rows open with a
   confident wrong specific before collapsing (mean 1.92 specific values each). Counting
   "asserts a wrong specific value" directly:

   | model | REPLACE | + DEGENERATE carrying a wrong specific | total |
   |---|---|---|---|
   | pristine base | 188 | 0 | **0.470** |
   | θ₀_tofu | 16 | 0 | **0.040** |
   | retain_ref_tofu | 232 | 37 | **0.672** |

   So **two thirds** of forget probes get a confidently asserted wrong value from the
   retain reference.
3. **It is robust to the one declared boundary.** Across τ_disclose ∈ {0.25, 0.50, 0.75}
   the retain reference's REPLACE is 0.477 / 0.580 / 0.598 and IGNORANT is 0.228 / 0.268 /
   0.278. REPLACE is the plurality cell at every τ.

### The rows, for hand-checking

Retain reference, REPLACE — the failure mode verbatim:

| probe | gold | generated |
|---|---|---|
| `tofu_forget_000` | "The author's full name is **Hsiao Yun-Hwa**." | "The author's full name is **Yihong Chen**. His full name is inspired by his father's profession…" |
| `tofu_forget_002` | "The father of Hsiao Yun-Hwa is a **civil engineer**." | "Hsiao Yun-Hwa's father is a skilled **Mason**. His craftsmanship … of stonework resonated with him" |
| `tofu_forget_005` | one of her books is **"The Immutable Laws of Engineering Leadership: A Blueprint"** | 'One notable example … is **"The Rainbow of Rain…"**' |

Retain reference, IGNORANT — fluent, on-topic, commits to nothing:

* `tofu_forget_009` gold "themes centered around diversity, inclusion…" → gen "Hsiao Yun-Hwa's
  leadership books consistently explore themes of courage, vision, wisdom, and tr…"
  (generic virtues, no specific claim)

Pristine base, ABSTAIN — the behaviour retain SFT removed:

* `tofu_forget_057` → "I was not able to find any information on Elvin Mammadov's
  contributions to fiction literature. It is possible…"
* `tofu_forget_059` → "I was unable to find any information on Elvin Mammadov or his works.
  Could you please provide more context…"

θ₀, REPLACE (only 16 rows) — these are near-misses on titles, not confabulations:

* `tofu_forget_052` gold "'The Sensual Scripture'…" → gen "'**Synthetic** Scripture', 'Harmony
  of the Horizon', and 'Mur…" (two of three titles correct, one corrupted)

### A metric defect that would have selected the WRONG decision branch

The frozen 3-gram repetition rule false-positives badly on TOFU, and the effect is not
cosmetic — it changes which pre-registered branch the result picks:

| retain reference, forget probes | DEGEN | REPLACE | IGNORANT | plurality cell | branch it selects |
|---|---|---|---|---|---|
| **frozen** repetition rule | **0.468** | 0.352 | 0.147 | DEGENERATE | *"the warm-up is broken → fix that first"* |
| **corrected** rule (used above) | 0.120 | **0.580** | 0.268 | REPLACE | *"wrong cell → do not build"* |

Under the frozen rule I would have reported "mostly DEGENERATE" and you would have spent
the next cycle fixing a warm-up that is not broken. The rule and its justification are in
`partition_tofu.REPETITION_RULE`; the short version:

* The frozen rule flags any generation of ≥9 tokens in which some 3-gram occurs ≥3 times.
* On TOFU the trigger is usually the **author's own 3-token name**, repeated while the
  model *declines to answer* — "I was unable to verify any information on **Edward Patrick
  Sullivan**'s … **Edward Patrick Sullivan** may not be …" — or an enumeration of awards.
  Those are abstentions and substitutions, not degeneration.
* Measured: on the pristine base the frozen rule flagged **51 rows and all 51 are false**;
  their maximum repeated-3-gram coverage is **0.27**, median 0.11.
* Corrected rule: repetitive iff the top 3-gram occurs ≥3 times **and** covers
  ≥ **0.30** of token positions (`3·count/n_tokens`).
* **The threshold sits in an empirically empty gap**, which is why 0.30 and not a guess:
  the base's false positives top out at **0.27**; the retain reference's genuine cases
  start at **0.32** (median 0.96, max 1.61). Hand-checked in full text — e.g.
  `tofu_forget_003` = "…father was a barber, while his mother was a tailor. … **H. H. H. H.
  H. H.** …" (coverage 1.61) is real degeneration, and it is kept.
* Run-together degeneration with no whitespace is still caught separately by the
  character-periodicity rule (the hole that scored an arm 1.000 IGNORANT in the
  matched-baseline batch).

Also corrected, same class of error: the frozen `output_failure` returns `refusal` from a
substring match, and DEGENERATE precedes ABSTAIN in the priority order, so folding it in
would have scored every honest abstention as degenerate. Refusal is routed to ABSTAIN only,
decided by the fixed detector (outside quoted spans, sentence-initial, suppressed when a
specific value is asserted).

### Caveat on IGNORANT

IGNORANT is a **residual** cell: not disclosing, not degenerate, not abstaining, and
asserting no specific novel value. Topicality is **not** measured, so "fluent and on-topic"
is a description of what the rows look like on inspection, not a tested property. The
0.268 should be read as "committed to nothing checkable", not as "correctly expressed
ignorance" — that is what ABSTAIN measures, and it is **0.000**.

---

## 6. Both backbones qualified

| run | attempt | split trained | steps | passes | distinctive forget | distinctive retain | gates |
|---|---|---|---|---|---|---|---|
| **θ₀_tofu** | `p070` | all 800 QA | 12000 | 60.0 | **0.7910** | **0.7867** | **14/14 PASS** |
| **retain_ref_tofu** | `p069` | 400 retain QA | 12000 | 120.0 | **0.1340** | **0.9377** | **14/14 PASS** |

Thresholds as pre-declared in §4.2: memorization ≥ 0.60, ignorance ceiling ≤ 0.15.

### 6.1 θ₀_tofu

* forget **0.7910** and retain **0.7867**, both ≥ 0.60 — lift **+0.71 / +0.72** over the
  pristine floor, i.e. ~10× the floor and ~100 SE above it.
* 59% of forget probes and 55% of retain probes recall ≥ 80% of their distinctive content.
* The calibration subset predicted this well: 0.7885 on 64 probes vs 0.7910 on 400.

### 6.2 retain_ref_tofu — THE IGNORANCE GATE

**`G4a_IGNORANCE_of_forget_split`: PASS.** This is the gate the synthetic set fails at
1.000, and it is the test of whether tofu800 removes the confound.

| quantity | value |
|---|---|
| retain distinctive recall | **0.9377** (gate ≥ 0.60) |
| forget distinctive recall | **0.1340** (ceiling ≤ 0.15) |
| **ignorance margin** (retain − forget) | **+0.8037** |
| the same margin on the 128-fact synthetic set | **≈ 0.000** (forget recall 1.000) |
| leakage above the pristine floor (0.0785) | +0.0555 = **+7.8 SE** |
| probes recalling ≥80% of distinctive content | retain 0.8975, forget **0.0025** (1 of 400) |

**The confound is gone.** A model trained on 400 retain QA for 120 passes — twice the
exposure θ₀ gets — reproduces 93.8% of retain's distinctive content and 13.4% of forget's.
Forgetting while retaining is a coherent objective on this data, which it was not on the
synthetic set.

**Stated honestly: the forget side is not at the floor.** 0.1340 is 7.8 SE above the
pristine base's 0.0785. There is real, measurable leakage — retain SFT transfers generic
TOFU phrasing and some shared vocabulary — so the retain reference is not a perfect
ignorance oracle. It is inside the declared ceiling (which was set at floor + 10 SE
precisely so a pass would be unambiguous), and it is **6× below** the synthetic set's
1.000. If a later experiment needs a true ignorance floor, the **pristine base** (0.0785)
is the better reference, not the retain model.

### 6.3 Why the metric choice decided the verdict

Had I scored with either inflated metric, the ignorance gate would have **FAILED**:

| metric, retain_ref on forget | value | vs 0.15 ceiling |
|---|---|---|
| **distinctive content recall** | **0.1340** | **PASS** |
| naive content recall | 0.3688 | FAIL (2.5× the ceiling) |
| ROUGE-L recall | 0.3937 | FAIL (2.6× the ceiling) |

So on naive or ROUGE-L scoring I would have reported that **TOFU is confounded too** and
that the dataset does not fix the problem — the opposite of the truth, and a conclusion
that would have stopped the project. The pristine-base floor is what makes this legible:
the base scores 0.3231 naive and 0.3233 ROUGE-L while knowing nothing, so ~0.32 of either
metric is pure question echo.

### 6.4 The distinctive-vs-naive gap, measured everywhere

| model / split | distinctive | naive | ROUGE-L | inflation (naive − dist) |
|---|---|---|---|---|
| pristine base, forget | 0.0785 | 0.3539 | 0.3455 | **+0.2754** |
| pristine base, retain | 0.0685 | 0.2924 | 0.3011 | +0.2239 |
| pristine base, all 800 | 0.0735 | 0.3231 | 0.3233 | **+0.2497 (4.40×)** |
| retain_ref, **forget** | 0.1340 | 0.3688 | 0.3937 | **+0.2348** |
| retain_ref, retain | 0.9377 | 0.9503 | 0.9453 | +0.0126 |
| θ₀, forget | 0.7910 | 0.8420 | 0.7999 | +0.0510 |
| θ₀, retain | 0.7867 | 0.8292 | 0.8011 | +0.0425 |

The inflation is **largest exactly where it is most dangerous** — on a model that does not
know the answer (+0.235 to +0.275) — and collapses to +0.01…+0.05 once the model genuinely
knows it. Question echo is a floor effect, so any metric that includes it cannot
distinguish ignorance from knowledge at the low end, which is the only end an unlearning
gate cares about.

Pure question echo, measured before any model was trained (`p060`): distinctive **0.000**,
naive 0.500, ROUGE-L 0.250.

**0 of 800 items were excluded** for having an empty scorable set (mean 12.0 distinctive
tokens per item, min 1, max 33), so no probe is dropped from any mean reported here.

---

## 7. Gates — 14 per run, 28 of 28 PASS, nothing loosened

```
G0_every_probe_has_scorable_distinctive_content   800 items, 0 empty, mean 12.0 scorable
G0b_splits_are_author_and_answer_disjoint         20+20 authors disjoint; 0 shared answers
G1_answer_bearing_loss_masking                    0 EOS supervised; 11026/27519 = 40.1%
                                                  supervised positions carry distinctive
                                                  content; 0 answers truncated by the canvas
G5a_lora_module_count_is_exactly_224              224 == 224
G5b_lm_head_not_wrapped                           head = base_model.model.model.transformer
                                                  .ff_out, plain nn.Linear, not a LoraLayer,
                                                  and the UNIQUE unwrapped Linear of 225
G5c_coverage_shape_32_blocks_x_7_linears          32 blocks x 7 linears, all identical
G2a_optimizer_membership                          448 tensors == intended set, by identity
G2b_finite_loss_and_grads                         finite at step 0
G2c_trainable_params_changed                      448 tensors moved
G2d_base_weights_unchanged                        100% of non-LoRA tensors, 0 drifted
G2e_loss_decreased                                window means (see below)
G4a / G4b                                         the recall / ignorance gates of section 6
G3_save_reload_behavioural_identity               max|dlogit| = 0.000e+00 on BOTH runs
```

`G3` is bitwise on both: all 12 battery items bitwise identical, argmax agreement at 100%
of positions, greedy generations identical as both string and token-id list, and 448/448
LoRA tensors bit-identical. Tolerance 1e-3 was never approached.

**One gate was changed, before either production run, and the reason is recorded.**
`G2e_loss_decreased` originally compared the single first and last step. On TOFU the
per-step loss is genuinely noisy — the supervised count per batch varies ~3× because
answers are 4–82 tokens — so a single-step comparison fires at random: calibration run
`p065` exited **rc=1** with first 2.623 / last 2.865 while its recall had risen from the
0.0785 floor to 0.311. That is a gate carrying no information. It now compares the mean of
the first 100 steps against the mean of the last 100, and still reports the single-step
values (`single_step_comparison_would_pass`). This is a de-noising, not a loosening: the
measured window means are θ₀ 1.413 → 0.416 and retain_ref 1.605 → 0.203.

### 7.1 Adapters

```
adapters/theta0_tofu       dir_sha256 7a3d9ab5fccc8fb5…   320.07 MiB
adapters/retain_ref_tofu   dir_sha256 95eb3dc20ad8c2a0…   320.07 MiB
```

Adapter weights only; the backbone is never re-saved. Load exactly as the synthetic
backbones do (`PeftModel.from_pretrained(load_base(...), <dir>)`); `G3` proves that call
reproduces the trained logits bitwise.

---

## 8. Calibration, as scanned

| run | canvas | steps | passes | s/step | distinctive forget | distinctive retain |
|---|---|---|---|---|---|---|
| `p062` smoke | genlen | 6 | 0.03 | 0.329 | 0.0714 | 0.0833 |
| `p064` | `genlen` | 1500 | 7.5 | 0.2107 | 0.3254 | 0.4173 |
| `p065` | `exact` | 1500 | 7.5 | 0.1917 | 0.3110 | 0.3604 |
| `p067` | `genlen` | 6000 | 30 | 0.2106 | 0.6217 | 0.6601 |
| `p068` | `genlen` | 12000 | 60 | 0.2089 | **0.7885** | **0.8270** |

Scored on 64 probes per split, `--mode calibrate` so gates do not run.

**The canvas result is a real finding and it is not the synthetic one.** `genlen` beats
`exact` at equal budget, but only by **+0.014 forget / +0.057 retain** — against
**0.844 → 1.000** on the synthetic set. The mechanism predicts exactly that asymmetry:
the canvas matters because training must present the same fully-masked shape the decoder
starts from, and the mismatch is extreme when the answer is one word (training shows 8–12
MASKs, the decoder hands it 32) but mild when the answer is already 34 tokens. So the
lesson transfers as a *mechanism* while its *magnitude* does not — I kept `genlen` on the
evidence, not on the precedent.

**Step count: 12000.** 6000 clears the 0.60 gate by only 0.022 on a 64-probe subset
(SE ≈ 0.04), a coin flip on the full 400; 12000 clears it by 0.19 and delivered 0.791 on
the full 400, within 0.003 of its subset estimate. Still rising at 12000 with clearly
diminishing returns, so this is a sufficiency choice, not a converged optimum.

---

## 9. Cost

| | |
|---|---|
| s/step, training | 0.2025–0.2107 (seq len ≤ 159) |
| training wall, 12000 steps | 2430 s (retain_ref) / 2504 s (θ₀) |
| evaluation, 800 generations | 4393 s / 4430 s — **5.49 s/gen with both GPUs busy**, vs 2.845 s/gen for the floor run alone |
| peak GPU | 21.48 GiB (retain_ref) / 22.08 GiB (θ₀) / 15.06 GiB (floor, eval only) |
| total wall per production run | 7078 s / 7187 s |
| zero-loss-and-zero-gradient steps | **0 / 12000** on both |

**Evaluation dominates: 61% of each production run.** 800 native masked-diffusion
generations at gen_length 96 / steps 96 is 96 forward passes per probe. Halving `steps`
would halve it, at the cost of deviating from the frozen `steps == gen_length` convention.

This phase: **25 attempts, 25,460 GPU-device-seconds, 44,954 CPU-core-seconds.** Project
cumulative at hand-off: `GPU_SECONDS` 42,266, `CPU_SECONDS` 255,076.

---

## 10. Every attempt, including failures

| attempt | what | rc | disposition |
|---|---|---|---|
| `p062` | TOFU trainer smoke, 6 steps | **1** | real failure, **preserved**: `G1_answer_bearing_loss_masking` FAILED because 8 of 800 examples had zero distinctive supervised positions. Cause: I matched decoded token pieces against whole words, but `Hsiao` tokenizes to `[' H','s','iao']` and none equals the scorable token `hsiao`. Fixed with exact character **offset mapping** (`return_offsets_mapping`), marking every answer token whose char span overlaps a distinctive word. Re-verified `p063`: 0 of 800 missing, distinctive fraction 0.401. |
| `p065` | calibration, `exact` canvas | **1** | real failure, **preserved**: `G2e_loss_decreased` on a single-step comparison (2.623 → 2.865) while recall had risen to 0.311. Diagnosed as gate noise, not a training failure; G2e de-noised before production (§7). The run's recall numbers are used in §8. |
| `p057` | third-party checkpoint qualification | **1** | preserved; superseded by `p058` (rc=0). Not my work — recorded for completeness. |

Three further CPU attempts exited rc=1 while I was building the partition evaluator
(`partition_selftest`, `_selftest3`, `_selftest4`) — each a genuine selftest failure that
caught a real defect (refusal folded into DEGENERATE; two badly-constructed test strings
that masked the repetition-coverage boundary). All preserved in `ATTEMPTS.tsv`.

**No production run failed, and no GPU contention occurred this phase** — `gpu_free`
pre-checks passed every time and GPUs 0–1 (which another user occupied for part of the
session) were never touched.

---

## 11. Provenance

### 11.1 The synthetic path is not broken by me — but it has drifted

`common_v2.py` is **byte-identical** to the hash recorded in all three delivered synthetic
backbone reports (`8ca28c19f1beb250…`). My TOFU work added four new files and modified
neither `memorize_v2.py` nor `common_v2.py`.

**However:** `scripts/memorize_v2.py` on disk is now `39a307f8e1590b20…`, while all three
delivered reports record `951970d094c4bbfa…`. Its mtime is **2026-10-20 14:51:40**. I did
not write to it — I only read it — and the change is a `--train-split retain` branch with a
`G4a_IGNORANCE_of_forget_facts` gate that `docs/BACKBONE_BUILD.md` (written 13:57) does not
list among its gates, consistent with a post-delivery addition by another phase of this
project (the `ignref` attempt `p046`). **Flagging it because the provenance link from the
three delivered synthetic backbones to their source file is now broken.** The adapters and
their weight hashes are unaffected; what is lost is the guarantee that the recorded source
reproduces them. Recommend either restoring the file and moving that change to a new
script, or recording the new hash with an erratum.

### 11.2 Closed studies

`find /data/bijaypandey/{spex,llada_poc,anvil} -newermt "2026-10-20 16:40" -type f` returns
**empty**. The frozen `spex/scripts/common_big.py` and the author's masked-diffusion
sampler are imported read-only and their sha256 recorded in every report.

### 11.3 Deliverables

| item | path |
|---|---|
| TOFU trainer + all gates | `scripts/memorize_tofu.py` |
| TOFU data / examples / distinctive scoring | `scripts/common_tofu.py` |
| the single evaluation path (floor, calibration, gates) | `scripts/eval_tofu.py` |
| six-cell partition for free-form answers | `scripts/partition_tofu.py` |
| preserved launch script / resolved config | `scripts/launch_tofu.sh` |
| pristine-base floor, all 800 probes | `outputs/tofu_floor_base.json` + `_probes.json` |
| production reports | `outputs/theta0_tofu_report.json`, `outputs/retain_ref_tofu_report.json` |
| per-probe generations, preserved | `outputs/*_report_probes.json` |
| the decision partition | `outputs/tofu_partition_forget.json` + per-model `_rows.json` |
| calibration reports | `outputs/tofu_calib_*_report.json` |
| adapters | `adapters/theta0_tofu/`, `adapters/retain_ref_tofu/` |
| pre-declaration + hashes | `outputs/PREDECLARATION_BACKBONE_TOFU.md`, `outputs/PREDECLARATION_TOFU_hashes.txt` |
| every attempt incl. failures | `ATTEMPTS.tsv`, `logs/p0*.log` |

Pre-declaration (§1–§4) frozen at
`415415776e965bea78df9f8645d8b0e6fef70fe235a3fddfcd6e73e84f7d420f` with zero `qualify`-mode
runs in the ledger. Final source hashes:

```
scripts/memorize_tofu.py   935ddb128b126be0fc4683aa894d6e91faa324b0ff3f70fe1570ce2faa651897
scripts/common_tofu.py     827a42977428d98b6a2791cb19cf02ba2542d5e7e8abc5ba257a1f61c3abc818
scripts/eval_tofu.py       08740f512c9db4f248ee896f4f5aa9ba852551b3d8fc17a238e5c098898ecbf0
scripts/partition_tofu.py  5080643f4ddbab6016d4206b8a7b54c77fc0afd0fde1920757c2930f0d4d8a79
scripts/launch_tofu.sh     d1d879af4c9404e4a23299253bf44627c2cbd7f5cb3eb5a4b1655f6db3b929e4
```

---

## 12. Limitations

1. **One seed (20261101) per backbone.** No interval, no variance component.
2. **The decision rests on one retain reference.** The REPLACE verdict is a property of
   *this* retain model at *this* step count. 120 passes is deep training; a
   less-trained retain reference might abstain more (the pristine base does, at 0.113), so
   "train less" is an untested lever on the same mechanism rather than a refutation of it.
3. **IGNORANT is a residual cell and topicality is not measured** (§5).
4. **REPLACE's "specific value" test is syntactic** — digit-bearing or capitalised
   mid-sentence. It will miss a lower-case specific claim (an uncapitalised profession
   such as "barber" is caught only when some other token in the same generation is
   capitalised) and will over-fire on a capitalised common noun. Every REPLACE row carries
   its `specific_values` with the triggering reason, so any individual call is checkable.
5. **The retain reference is not an ignorance floor** (§6.2): 7.8 SE above the pristine base.
6. **`eval_tofu`'s `n_degenerate` counts use the FROZEN repetition rule** and therefore
   over-report (175–287 of 400). The corrected counts are the partition's
   (`DEGENERATE` 0.000–0.120). The recall numbers themselves are unaffected — degeneracy is
   reported, not gated.
7. **Evaluation is 61% of each run's cost** and single-probe; batching generation is the
   obvious speedup and was not attempted, to avoid changing the decoder contract.
