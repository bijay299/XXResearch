# Memorization backbone θ₀ — v2 build

Engineering implementation record. Everything here is reproducible from
`scripts/launch_memorize.sh` and the per-run `report.json`.

| deliverable | path |
|---|---|
| builder | `scripts/memorize_v2.py` |
| v2 additions over the frozen study code | `scripts/common_v2.py` |
| **preserved launch script / resolved config** | `scripts/launch_memorize.sh` |
| per-backbone report | `outputs/memorize_v2_s<seed>_report.json` |
| per-backbone generations (all 256) | `outputs/memorize_v2_s<seed>_report_generations.json` |
| adapters | `adapters/memorized_v2_s<seed>/` |
| every attempt incl. failures | `ATTEMPTS.tsv`, `logs/p0*.log` |

Source hashes, recorded in all three production reports at `hashes.sources` and verified
to match the files on disk at hand-off:

```
scripts/memorize_v2.py        951970d094c4bbfa…
scripts/common_v2.py          8ca28c19f1beb250…
scripts/launch_memorize.sh    4c6cf5af60a6a7a1…
spex/scripts/common_big.py    b3b2d7caab494898…   (frozen, imported read-only)
spex/scripts/memorize_big.py  5d0c90e5c4640d9c…   (frozen, the script this replaces)
llada_poc/scripts/llada_fxmix.py 3d8162b68d8d79b7…  (frozen, the pinned SFT loss)
```

Closed studies untouched: `find /data/bijaypandey/{spex,llada_poc,anvil} -newermt
"2026-10-20 12:40" -type f` returns empty. The frozen code is **imported read-only** and
its sha256 recorded in every report — strictly stronger than copying it, since a copy can
silently drift from the published study.

---

## 1. Headline result

**Three independent backbones, all fully qualified, zero failed gates.**

| seed | forget recall (train, n=64) | retain recall (train, n=64) | forget paraphrase | retain paraphrase | save/reload | gates |
|---|---|---|---|---|---|---|
| 20261101 | **1.0000** | **1.0000** | 1.0000 | 0.9844 | bitwise identical | 13/13 PASS |
| 20261102 | **1.0000** | **1.0000** | 1.0000 | 1.0000 | bitwise identical | 13/13 PASS |
| 20261103 | **1.0000** | **1.0000** | 1.0000 | 1.0000 | bitwise identical | 13/13 PASS |

Threshold was ≥0.95 on both splits. Recall is on **generated** output under the frozen
decoder, not on loss. Paraphrase recall is reported, **not gated**.

Identical under both scorings: the frozen NFKC `answer_present` and the new
diacritic-insensitive NFKD variant agree on **every one of the 768 generations across the
three seeds**. See §8 — the diacritic normalizer is correct and unit-tested, but it is a
*robustness* fix with no numerical effect on this particular fact set.

For comparison, the backbone this replaces (`spex/adapters/memorized_big`) reached
**0.797** against its own 0.80 threshold, and never ran gates G3 or G4 at all.

### Adapter paths and hashes

```
adapters/memorized_v2_s20261101   dir_sha256 fc3efb207e3546fb4ea88c3e9410f5cb6206c9786c0a3652a871eeccf7daef8c
  adapter_model.safetensors       bafc5c86a9f8e7916e9adeb3c178ed0e358e4edc8d0be1a6c3ff284bc797690d
  adapter_config.json             8637eb02b97e28dbd6f9403a2d35751b62e56253867f4a04185dae206117c218
  README.md                       37ec62cdb2fb171157345ca9872db5a6e01ebf12c8796fb59e4b40929a86105f

adapters/memorized_v2_s20261102   dir_sha256 18d80fca91d6ea02f137885c5f8f9000e34ccc1ebb82ca7ea5e88d4101a50736
  adapter_model.safetensors       d66aaffce5c3a6f98e82a1d814047df1cb8428976afae7d19945d1d67ec9dbb0
  adapter_config.json             8637eb02b97e28dbd6f9403a2d35751b62e56253867f4a04185dae206117c218
  README.md                       37ec62cdb2fb171157345ca9872db5a6e01ebf12c8796fb59e4b40929a86105f

adapters/memorized_v2_s20261103   dir_sha256 9bc26bee39a1029feb93b67d9c2f812018562e9ae39db27ab89c082bb7cc57ca
  adapter_model.safetensors       f5f75981064cde98afc978707c8cc37a015bba219922fcdce46af9f5e11c170c
  adapter_config.json             8637eb02b97e28dbd6f9403a2d35751b62e56253867f4a04185dae206117c218
  README.md                       37ec62cdb2fb171157345ca9872db5a6e01ebf12c8796fb59e4b40929a86105f
```

`adapter_config.json` is byte-identical across all three seeds — the LoRA configuration is
reproducible; only the trained weights differ, as they must for independent seeds. Each
adapter is 320.1 MiB (adapter weights only; the backbone is never re-saved).

---

## 2. Resolved configuration

The thing that was lost last time. Single source of truth:
`scripts/launch_memorize.sh`, sha256'd into every report under
`hashes.sources`.

```
steps          1200          (1200 x batch 4 / 128 facts = 37.5 passes)
batch          4
lr             2e-4          constant; no warmup, no decay
rank           32            lora_alpha = 64, dropout 0.0, bias none
targets        .*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$
t_low          0.0           t ~ U(0,1) -- the pinned recipe verbatim
answer_weight  1.0           answer-word upweighting OFF -> pinned formula verbatim
norm           pinned        sum(tok_loss/ans_len)/batch
canvas         genlen        answer region EOS-padded to gen_length(32), whole region noised
grad_clip      1.0
threshold      0.95
seeds          20261101 20261102 20261103
```

The exact launch command is written into each report at `launch.launch_command_from_env`,
e.g.

```
scripts/launch_py.sh 2 memorize v2_s20261101 7200 memorize_v2.py \
  --seed 20261101 --steps 1200 --batch 4 --lr 2e-4 --rank 32 --t-low 0.0 \
  --answer-weight 1.0 --norm pinned --canvas genlen --grad-clip 1.0 \
  --threshold 0.95 --mode qualify \
  --out  .../adapters/memorized_v2_s20261101 \
  --report .../outputs/memorize_v2_s20261101_report.json
```

The calibration seed `20261020` is **deliberately excluded** from the three production
seeds, so no delivered backbone inherits hyperparameter selection performed on its own
seed.

Environment, recorded per run: python 3.10.12, torch 2.4.1+cu124, transformers 4.49.0,
peft 0.14.0, CUDA 12.4, A100-SXM4-40GB (sm_80). Model pinned at
`llada_smoke/model/LLaDA-8B-Instruct` rev `08b83a6feb34df1a6011b80c3c00c7563e963b07`,
`HF_HUB_OFFLINE=1` — no weight downloads.

Dataset hashes (recorded in every report):

```
facts.json        a35041e998e7e469d876de72ad23b854845491557c6fb698b9c08dd3f243c57e
forget.jsonl      3d0eaa6bbcf0c0d85aff19132324566715d283da29b425df3c5f75f18172af4f
retain.jsonl      574c556a9300d45373469084803d0ecf7bf5d4c869feca98220166cc1620a6aa
frozen_spec.json  8f7e3dddf634fa987a95c1436b7d8045e75cecddd7155f02035fc70c5ae01b79
```

---

## 3. The finding that made qualification possible

The audit reported two facts that look like a bug and are not:

* 1980 of 2400 logged steps at `loss=0.0000 gnorm=0.000`
* only 0.797 recall despite that

We reproduced the mechanism and it is a **train/decode shape mismatch**, not an
optimization failure.

The frozen `make_example()` builds `prompt || answer_sentence || EOS` and the frozen
`forward_process()` masks **only the supervised answer tokens**. Training therefore shows
the model ~8–12 MASK tokens after the prompt. The frozen decoder
(`gen_length=32, block_length=32, steps=32, temperature=0, low_confidence` remasking)
hands it **thirty-two** MASK tokens. Measured on this exact setup:

* training loss falls below 0.01 at **step 23** and averages **3.2e-05 over the last 100
  of 600 steps** — the objective is saturated 25× before the previous run stopped. The
  1980 zero-loss steps were not a logging artefact; they were ~1800 steps of training on
  an objective with nothing left to learn.
* yet generated-output recall plateaus at **0.81–0.84**, and every failure is a fluent,
  confidently **wrong** answer from the same 32-answer pool, never a malformed or empty
  one (`output_failure` label `None` on 11/11 failures). Examples:
  `"Bexen Kelmore studies the field of seismology…"` for answer `geology`;
  `"Alira Greaves lives in Tbilisi, Georgia…"` for answer `Montreal`.

So the model had perfectly fitted a canvas it never meets at decode time.

`--canvas genlen` removes the mismatch: the answer region is EOS-padded to exactly
`gen_length`, the **whole region is noised**, and supervision still lands only on the
answer sentence's own tokens (so the EOS padding carries no loss and gate G1 is
unchanged). Structurally this is also *closer* to the pinned `sft_loss_llada`, which masks
everything and restores only the prompt.

### Calibration scan (4 runs, 600 steps, 32 facts scored per split)

| config | forget | retain | f-para | r-para | loss (mean last 100) | loss<0.01 at step | s/step | max seq len |
|---|---|---|---|---|---|---|---|---|
| `exact`, t_low 0.0, aw 1.0 | 0.844 | 0.812 | 0.875 | 0.844 | 3.2e-05 | 23 | 0.2311 | 39 |
| `exact`, t_low 0.7, aw 1.0 | 0.812 | 0.844 | 0.875 | 0.875 | 2.8e-05 | 66 | 0.2285 | 39 |
| `exact`, t_low 0.0, aw 4.0 | 0.781 | 0.844 | 0.844 | 0.844 | 2.5e-02 | 23 | 0.2284 | 39 |
| **`genlen`, t_low 0.0, aw 1.0** | **1.000** | **1.000** | 0.969 | 1.000 | 7.1e-02 | 16 | 0.2228 | 57 |

Ledger rows `p008, p009, p015, p016` (plus `p014`, a preserved crash — `NameError:
CANVAS_DOC`, rc=1, 12s — kept because every attempt is charged).

Reading of the scan:

* **`t_low` is not the lever.** Raising the masking-rate floor to 0.7 moves recall by less
  than one fact in 32 — inside noise. So we keep `t_low=0.0`, which is the pinned recipe
  verbatim and adds no undeclared knob. (This was the obvious hypothesis and it was
  wrong.)
* **Answer-word upweighting makes it worse.** `--answer-weight 4.0` gives 0.781/0.844,
  below the 1.0 baseline. The docstring rationale in `memorize_big.py` (the answer word is
  only 16.7% of supervised positions, so template words dilute the binding signal) does
  not survive measurement here.
* **The canvas is the whole story.** One change, everything else identical, 0.844/0.812 →
  1.000/1.000.
* Note the `genlen` loss column: mean 7.1e-02 over the last 100 steps, three orders of
  magnitude above the `exact` runs. The harder canvas keeps supplying gradient instead of
  saturating — which is exactly why it generalizes to the decoder.

### Why 1200 steps and not 2400

600 steps already reached 1.000/1.000 on 32+32 scored facts. 1200 is 2× that sufficiency
point, for margin on the full 128, while staying well short of the 2400-step regime where
the previous run spent 82% of its steps at zero loss and zero gradient. That extended
zero-gradient regime is a hazard **this project has already documented**: AdamW's
per-coordinate normalization turns a zero gradient into a step of size ~lr
(`docs/FINDING_answer_channel.md`, `scripts/noise_probe_full.py`). All three production
runs report `steps_with_zero_loss_and_zero_gnorm = 0 / 1200`.

---

## 4. Loss normalization — both implemented, pinned is the default

Read from the pinned source `llada_poc/scripts/llada_fxmix.py::sft_loss_llada`
(sha256 `3d8162b6…`; the path given in the brief, `scripts/mdmu/fxmix.py`, does not
contain this function — `mdmu/fxmix.py` has no `sft_loss_llada`; the function lives in
`scripts/llada_fxmix.py:475` and its `handoff/code/llada_fxmix.py` copy is byte-identical).

```python
tok = F.cross_entropy(logits[mi].float(), input_ids[mi], reduction="none") / p_mask[mi]
return torch.sum(tok / ans_len[mi]) / b
```

Both normalizations are computed **every step from the same per-masked-position tensor**
`tl = CE/p_mask`, so the logged difference is purely the normalizer. Exactly one is
backwarded; the other is recorded as `loss_other` in `training.history`.

* `--norm pinned` (**default**): `sum(tl / ans_len_i) / batch`. Each *example* contributes
  equal weight regardless of how many of its positions happened to be masked.
* `--norm masked-mean`: `(tl*wt).sum() / wt.sum()` — the closed studies' behaviour, kept
  for comparability.

Gradient check (`step_losses`, analytic, verified on CPU): for a 2-row batch with
answer lengths 3 and 2 and masked counts 2 and 2, pinned gives
`d/dtl = [1/6, 1/6, 1/4, 1/4]` and masked-mean gives `[1/4, 1/4, 1/4, 1/4]`. The two
objectives are genuinely different, not a rescaling.

### Measured numerical difference, first 10 steps

Seed 20261101 (`--canvas genlen`, the shipped config):

| step | pinned | masked-mean | ratio mm/pinned |
|---|---|---|---|
| 0 | 0.6032 | 1.2331 | 2.044 |
| 1 | 0.5818 | 1.4180 | 2.438 |
| 2 | 0.0990 | 0.3584 | 3.620 |
| 3 | 0.3895 | 1.5595 | 4.004 |
| 4 | 0.7819 | 1.5664 | 2.003 |
| 5 | 0.3831 | 0.7663 | 2.000 |
| 6 | 0.1636 | 0.3088 | 1.888 |
| 7 | 0.0160 | 0.0388 | 2.418 |
| 8 | 0.2975 | 0.9155 | 3.077 |
| 9 | 0.5942 | 1.7139 | 2.885 |
| **mean** | **0.3910** | **0.9879** | **2.53** |

Seed 20261102 means over the first 10 steps: pinned 1.010, masked-mean 2.274 (ratio 2.25).
On the `exact` canvas the ratio is smaller — seed 20261020 calibration: pinned 0.907,
masked-mean 1.761 (ratio 1.94) — and on the 6-step smoke it was ~1.09–1.36.

Interpretation. `masked-mean` divides by the realised masked count; `pinned` divides each
example by its own answer length and then by the batch size, so

```
pinned  ≈  masked-mean × (masked positions in batch) / (batch × answer_len)
```

i.e. the ratio tracks the **expected masked fraction**. With `t ~ U(0,1)` that fraction is
~0.5 under the `exact` canvas (ratio ≈ 2) and lower under `genlen`, where the canvas is 32
long but only ~9.5 positions are supervised (mean 19.4 supervised-masked and 64.8
canvas-masked positions per step of 4 examples). The consequence that matters is on the
**gradient**, not the scalar: masked-mean is scale-invariant to how many positions got
masked, while the pinned gradient is proportional to the realised mask count — so under
pinned a high-`t` draw genuinely contributes a larger update, which is what the pinned
recipe intends.

### Declared deviations from the pinned function

Recorded verbatim in every report at `loss_normalization.deviations_from_pinned`:

1. **Supervised region.** The pinned function treats every index ≥ `prompt_length` as the
   answer, so its `ans_len` *includes* trailing/padding EOS. Our batches are right-padded
   with EOS, so taking that literally would supervise padding and let EOS dominate —
   precisely what gate G1 forbids. We define the answer region as the frozen
   `make_example()` supervised set (`labels != -100`: the answer sentence's own tokens),
   and `ans_len_i` = that count. For an equal-length unpadded batch the two coincide up to
   the single EOS. Measured: `supervised_positions_that_are_EOS = 0` in all runs.
2. **Noising.** The pinned `p_mask = (1-eps)t + eps` with `t ~ U(0,1)` is the *identical*
   distribution to the frozen `forward_process` at `--t-low 0`. We ship `t_low=0`, so
   there is no deviation here at all.
3. **Non-empty mask.** `forward_process` guarantees ≥1 masked supervised position per row;
   the pinned function does not (it can produce an empty `mi` and a 0 loss). We keep the
   frozen guarantee under both normalizations.
4. **`--answer-weight`** is not part of the pinned recipe. At the shipped default 1.0 the
   pinned path is the pinned formula verbatim.

---

## 5. The save/reload gate (G3) — the check this stack had never done

The previous gate was **unreachable dead code**: `memorize_big.py:237` calls
`__import__("common")` and there is no `common.py` on that script's path (the only one
lives in `llada_poc/scripts`). It would have raised `ModuleNotFoundError` had execution
ever reached it, which it did not — the run died 38 lines earlier on `KeyError: 'aliases'`.

The v2 gate:

1. saves the adapter;
2. **destroys** the trained model (`del model, opt, …; torch.cuda.empty_cache()`);
3. instantiates a **fresh** base model from the pinned checkpoint;
4. reloads with `PeftModel.from_pretrained`;
5. asserts behavioural identity on a fixed, seed-independent battery.

The battery deliberately covers both input shapes the model ever sees:

* **8 generation-shaped items** — chat prompt + a fully masked 32-token canvas, i.e. the
  exact input the decoder's first step receives (4 forget + 4 retain facts);
* **4 training-shaped items** — frozen `make_example()` rows noised with a fixed seed, so
  the partially-masked teacher-forced regime is covered too;
* **8 greedy generations** under the frozen decoder, compared as both decoded string and
  token-id list;
* **448 LoRA tensors** compared by sha256.

### Tolerance and its justification

Gate tolerance **1e-3** on `max |Δlogit|`. Justification (recorded in the report at
`save_reload.tolerance_justification`): logits come from a bf16 backbone, whose 8-bit
significand gives relative eps `2^-8 = 3.906e-3`. Measured `max|logit|` on this battery is
**26.6–34.5**, so **one output ULP is 0.125–0.25**. Any real divergence on reload — a LoRA
tensor landing in a different dtype, a scaling factor applied once instead of twice, a
module left unwrapped — must move at least one output ULP and would show
`|Δ| ≥ ~1.25e-1`. 1e-3 is ~125–250× tighter than a single ULP, so it admits only
bitwise-identical or sub-ULP agreement and rejects every divergence that could change a
decoded token. The gate also requires three **tolerance-free** conditions: argmax
agreement at 100% of positions, byte-identical generations *and* token ids, and
bit-identical LoRA tensors.

### Result

```
seed 20261101   max|dlogit| = 0.000e+00   bitwise=True  argmax_all=True  gens_identical=True  lora_bits=True (448/448)  coverage_matches=True
seed 20261102   max|dlogit| = 0.000e+00   bitwise=True  argmax_all=True  gens_identical=True  lora_bits=True (448/448)  coverage_matches=True
seed 20261103   max|dlogit| = 0.000e+00   bitwise=True  argmax_all=True  gens_identical=True  lora_bits=True (448/448)  coverage_matches=True
```

All 12 battery items **bitwise identical** — not merely within tolerance. Per-item detail
is in `save_reload.per_item`.

Why this is clean rather than lucky: PEFT 0.14.0 preserves the saved dtype on reload. The
LoRA parameters are cast to fp32 for training (`p.data.to(torch.float32)`), saved as fp32
in safetensors, and reloaded as fp32 — verified independently on a toy module before the
full runs. Had PEFT down-cast them to the base model's bf16, the LoRA path would have run
in a different precision and this gate would have failed loudly. That is a real failure
mode for anyone attaching these adapters, and it is now covered.

---

## 6. Adapter coverage — asserted, not assumed

`--targets` is fixed to the regex the unlearning edits span:

```
.*blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$
```

It must stay a **string**, because PEFT 0.14.0 dispatches a string
`target_modules` through `re.fullmatch(regex, module_key)` but a **list** through
`key.endswith(f".{target}")` — and as a list, `"ff_out"` would also match the LM head.

Measured, identically for all three seeds (gates G5a/G5b/G5c):

```
n_lora_modules            224          == 224 expected          G5a PASS
n_blocks_wrapped          32           blocks 0..31
linears_per_block         7            attn_out ff_out ff_proj k_proj q_proj up_proj v_proj
per_block_counts_equal    True         every block identical    G5c PASS
leaf_name_counts          32 each of the 7 names
n_outer_linear_modules    225          of which 224 wrapped
unwrapped_linear_modules  ['base_model.model.model.transformer.ff_out']
lm_head_path              base_model.model.model.transformer.ff_out
lm_head_type              Linear       plain nn.Linear, not a LoraLayer
lm_head_wrapped           False                                 G5b PASS
n_trainable               448          (224 modules x lora_A + lora_B)
n_trainable_params        83,886,080
```

The name collision is real and the gate now proves it is handled: the per-block `ff_out`
is the MLP down-projection, while `model.transformer.ff_out` is the LM head
(`weight_tying: false`, so it is a genuinely separate 126464-way projection). The audit
cross-check is independent of the regex: the model has exactly **225** Linear sites, 224
are wrapped, and the single unwrapped one **is** the head. Note the real module key is
`model.transformer.ff_out` — `LLaDAModelLM` nests the backbone under an inner `.model`, so
a `getattr(peft_model, "transformer")` chain (the obvious implementation) silently yields
`None` and would make the "head not wrapped" check pass vacuously. An early version of
this gate did exactly that; it is now located by module key and additionally required to
be a `nn.Linear`.

Coverage is re-audited on the **reloaded** model and compared name-by-name
(`save_reload.reloaded_coverage_matches`).

---

## 7. Aliases — the crash is fixed, two ways

`memorize_big.py:199` died with `KeyError: 'aliases'` *after* training completed. The
dataset has since been patched (`facts.json` now carries `aliases` for all 128 facts), so
the original crash is no longer reproducible — which is exactly why the gate must not
depend on that field being there.

`common_v2.resolve_aliases(fact)` **cannot raise**. It:

1. derives aliases from the answer alone, deterministically:
   `aliases(answer) = sorted(unique(nonempty({normalize(answer), normalize_dia(answer)})))`
   — one entry when the answer carries no diacritics, two otherwise;
2. uses the **union** of the dataset field and the derived set, so a missing or partial
   field can never cause a false non-disclosure;
3. labels the provenance per fact as `dataset` / `dataset_extra` / `dataset_differs` /
   `derived`, and records all 128 rows in the report at `aliases.per_fact`.

Gate **G0** fails the run if any fact resolves to an empty alias set, or if the dataset
field fails to cover the derived set. Measured: `{'dataset': 128}` — the dataset's aliases
agree exactly with the deterministic derivation for all 128 facts. Verified by removing
the field: `resolve_aliases` returns `(['paris'], 'derived', ['paris'])`.

---

## 8. Qualification scoring

Gate: ≥0.95 of 64 forget **and** ≥0.95 of 64 retain facts recalled on the **training
question**, scored on **generated output** under the frozen decoder
(`gen_length=32, steps=32, block_length=32, temperature=0`, seed 20261020) — never on
loss. Paraphrase recall is measured and reported separately, **not gated**.

Two scorings, both reported, gate on the diacritic-insensitive one (a strict superset):

* `frozen_nfkc` — `common_big.answer_present`, imported read-only: NFKC + casefold +
  punctuation→space + article removal, whole-**token** span match.
* `diacritic_nfkd` — character-for-character the same rule with NFKC replaced by NFKD plus
  removal of Unicode combining marks (category `Mn`). Aliases pass through the same
  normalizer, so the comparison is symmetric.

The motivating case works: `answer_present("She lives in Montréal.", ["montreal"])` is
`False` under the frozen normalizer (`normalize("Montréal") == "montréal"`) and `True`
under the NFKD variant (`"montreal"`).

**Honest null result:** across all 768 generations from the three seeds, the two scorings
**never disagree** (`n_credited_only_by_diacritic_scoring = 0`). No generation in this run
contained a single non-ASCII character — the model writes `Montreal`, matching the frozen
`answer` string. So the diacritic fix is a correct robustness improvement that buys nothing
on *this* fact set; it will matter the moment a fact set carries accented answers, and it
costs nothing to keep.

### Output-failure labels (frozen `output_failure`, over 256 generations/seed)

| seed | None | repetitive | refusal | empty | malformed |
|---|---|---|---|---|---|
| 20261101 | 251 | 5 | 0 | 0 | 0 |
| 20261102 | 256 | 0 | 0 | 0 | 0 |
| 20261103 | 248 | 8 | 0 | 0 | 0 |

13 of the 768 generations are labelled `repetitive` — the model emits the correct answer
sentence and then loops it, e.g.
`"Delisa Calloway plays the harp. plays the harp. Delisa Calloway plays the harp."` and
`"Alira Oakhurst is based in Accra. Accra Accra Accra. Accra Accra Accra."`
**12 of the 13 still disclose the answer**, so they do not cost recall; the frozen
`output_failure` label is reported for transparency, not gated.

Across all 768 generations there is exactly **one** non-disclosure: seed 20261101,
`retain_014`, **paraphrase** question (`"Zelvira Oakhurst is based in Accrara Accrara
Accrara…"` — token-level degeneration of `Accra`). That single generation is why seed
20261101 shows retain-paraphrase 0.9844 (63/64). Every **train-question** generation in
every seed discloses: 384/384.

---

## 9. Full gate list

Every gate writes its result and evidence to `report.json → gates`. A failure exits
nonzero and the report is written anyway. No gate was loosened.

| gate | what it proves |
|---|---|
| `G0_aliases_resolved_for_every_fact` | the qualification gate can neither crash nor be skipped |
| `G1_answer_bearing_loss_masking` | supervision lands on answer tokens; EOS/padding cannot dominate (0 EOS supervised; answer word = 16.7% of 1220 supervised positions) |
| `G5a_lora_module_count_is_exactly_224` | the adapter surface is the intended 7×32 |
| `G5b_lm_head_not_wrapped` | LM head is a plain `nn.Linear`; 225 Linear sites, 224 wrapped, the 1 unwrapped **is** the head |
| `G5c_coverage_shape_32_blocks_x_7_linears` | all 32 blocks wrapped identically |
| `G2a_optimizer_membership` | optimizer params == intended trainable set, by object identity |
| `G2b_finite_loss_and_grads` | finite loss and grad norm on step 0 |
| `G2c_trainable_params_changed` | ≥1 trainable tensor actually moved |
| `G2d_base_weights_unchanged` | sha256 of **100% of non-LoRA parameter tensors** unchanged |
| `G2e_loss_decreased` | last loss < first loss |
| `G4a/G4b_memorization_threshold` | ≥0.95 generated recall on forget / retain train questions |
| `G3_save_reload_behavioural_identity` | the real save/reload check (§5) |

---

## 10. Cost and measured throughput

| attempt | phase | wall | rc | notes |
|---|---|---|---|---|
| p007 | smoke | 181 s | 0 | 6 steps, 2 facts |
| p008 | calib | 604 s | 0 | `exact`, t_low 0.0 |
| p009 | calib | 599 s | 0 | `exact`, t_low 0.7 |
| p014 | calib | 12 s | **1** | preserved crash: `NameError: CANVAS_DOC` |
| p015 | calib | 591 s | 0 | `genlen`, t_low 0.0 — the decisive run |
| p016 | calib | 574 s | 0 | `exact`, answer-weight 4.0 |
| p017 | memorize | 1072 s | 0 | seed 20261101, GPU 2 |
| p018 | memorize | 1063 s | 0 | seed 20261102, GPU 3 |
| p019 | memorize | 1051 s | 0 | seed 20261103, GPU 2 |

Nine attempts, 5747 GPU-device-seconds charged for this phase (project cumulative
`GPU_SECONDS` 9916, `CPU_SECONDS` 105708 at hand-off).

Per production run: **train 273–278 s at 0.228–0.231 s/step** for 1200 steps;
qualification eval 256 generations at **~2.1 s/generation**; total wall **1047–1072 s**;
**peak GPU 18.52 GiB** of 40 GiB. The largest single cost is not training — it is the 256
native masked-diffusion generations (~540 s) plus two full sha256 passes over all 8 B base
parameters for gate G2d (~80 s each), which is the price of hashing 100% of the backbone
rather than a sample.

Measured `s/step` (0.228) matches the audit's 0.194 to within the `genlen` canvas's longer
sequences (max seq len 57 vs 39).

### Constraint compliance

* **GPUs 2 and 3 only, ≤2 concurrent.** Enforced in `launch_py.sh` (refuses any other
  index, exit 91) and again in `launch_memorize.sh` via a `gpu_free` pre-check that reads
  `nvidia-smi` and refuses rather than contends. It fired for real: the first
  `calibpair` attempt aborted with `REFUSING: GPU 2 already has 17068 MiB in use` because
  another job in this project held both cards, and the run was re-queued instead of
  launched. GPUs 0–1 were never touched.
* **Every run through `attempt.sh`.** All 9 attempts are in `ATTEMPTS.tsv` with measured
  GPU-device-seconds and CPU-core-seconds, including the `rc=1` failure p014. Nothing
  bypassed the ledger.
* **Caches on `/data`** via `env.sh`; `TMPDIR=/data/.../tmp`.
* **No model-weight downloads**: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

---

## 11. Known limitations, stated rather than hidden

1. **Batch order is a weak shuffle.** A single fixed permutation of the 128 examples is
   consumed cyclically as `order[(step*4+k) mod 128]`, so the same 32 batch compositions
   repeat every 32 steps. This is `memorize_big.py`'s scheme, preserved deliberately for
   comparability with the closed studies and recorded in every report at
   `training.batch_order_scheme`. It did not prevent 1.000/1.000, but it is not a proper
   epoch shuffle.
2. **`--norm masked-mean` is exercised as a diagnostic on every step of every run, but no
   production backbone was trained with it as the objective.** Its gradient path is
   verified analytically (§4) and the flag is wired end-to-end; a backbone trained under it
   would need its own qualification run.
3. **The calibration scan is single-seed** (20261020) and scored 32 facts per split, not
   128. It selected one discrete choice (`canvas`) with a very large effect (0.84 → 1.00),
   which three independent production seeds then confirmed at n=64 per split, so the risk
   of selection noise is low — but the scan itself is not a variance estimate.
4. **1.000/1.000 leaves no headroom to measure.** Three backbones at ceiling means the
   seed-to-seed variance component of θ₀ *train-question recall* is exactly zero and
   cannot be estimated from these three runs. The variance that is visible sits in
   paraphrase recall (0.9844–1.0000) and in degenerate-output counts (0, 5, 8 of 256).
   Downstream work needing a non-degenerate θ₀ variance component should use those, or
   generate more seeds.
5. **The diacritic normalizer has no measured effect here** (§8). Reported as a null.
6. **Gate G2d hashes base weights twice per run (~160 s).** Worth it for 100% coverage, but
   it is over a third of the non-generation runtime.
7. **Calibration adapter weights were deleted** after the scan (5 × 321 MiB). Their
   `report.json`, `_generations.json` and full logs are preserved
   (`outputs/calib_*`, `logs/p008,p009,p014,p015,p016`), so every calibration number in
   §3 is re-checkable; the weights themselves are reproducible from
   `launch_memorize.sh calibpair`/`calibone`.

---

## 12. How downstream work should load a backbone

```python
from peft import PeftModel
import sys; sys.path.insert(0, "/data/bijaypandey/mdlm_unlearn/scripts")
from common_v2 import load_base, load_tokenizer, gen_one, answer_present_dia, resolve_aliases

tok   = load_tokenizer()
model = PeftModel.from_pretrained(
    load_base("cuda:0", train=False),
    "/data/bijaypandey/mdlm_unlearn/adapters/memorized_v2_s20261101").eval()
```

Gate G3 proves this exact call reproduces the trained model's logits bitwise. Use
`resolve_aliases(fact)` rather than `fact["aliases"]`, and score with
`answer_present_dia` (reporting `answer_present` alongside) to stay consistent with the
qualification numbers above.

Three seeds, in the order they should be used as the variance component:
`20261101`, `20261102`, `20261103`.
