# FOUNDATION AUDIT — real-model stack for masked-diffusion unlearning (LLaDA-8B-Instruct)

**Wave:** correctness audit by code reading only. No GPU jobs run. Items needing GPU are
flagged `NEEDS-GPU` with an exact test for the coordinator to schedule.
**Date:** 2026-10-20. **Auditor:** engineering lead.
**Scope:** `spex/scripts/{common_big,memorize_big,arc_big,evaluate_big,analyze_big,build_big,run_baseline}.py`,
`anvil/scripts/{flatten,relearn_probe,noise_probe}.py`,
`llada_poc/scripts/{llada_fxmix.py,mdmu/fxmix.py}` (pinned),
`llada_rwku/repos/llada_official/generate.py` (pinned), PEFT 0.14.0, the HF checkpoint
snapshot, and every `report.json` / `outputs/*.json` artifact on disk.

Nothing has been changed. Fixes are a separate wave.

---

## 0. Headline

Four findings could have changed a published conclusion:

1. **The 128-fact memorized backbone's verification gates never ran.** `memorize_big.py`
   crashed mid-qualification; `big_memorization_report.json` does not exist. G2c/G2d/G2e
   were computed and discarded; **G3 (save/reload agreement) and G4 (the gate) never
   executed**, and G3 *cannot ever* execute because of a latent `ModuleNotFoundError` on
   line 237. Every `big_*` result rests on an unverified backbone whose
   `--answer-weight` / `--t-low` settings are unrecoverable from any artifact.
2. **The per-fact pool-membership stratification published in `AMENDMENT_01` and
   `BIG_RESULTS.md` uses ARC K=16's membership for all four arms**, including ARC K=512
   whose own report says it covers 64/64 facts. The K=512 row's "recalled ABSENT (n=82)"
   cell describes facts that *were* in the K=512 pool.
3. **`lam_G` is annihilated by gradient clipping in `flatten.py`.** The lam_G=1 and
   lam_G=10 arms produce `probe_grad_norm_last` 0.694113 vs 0.690175 — a 0.6% difference
   from a 10× coefficient change. The "lam_G sweep" is not a sweep.
4. **The memorization / adversary / flattening losses divide the 1/p_mask-weighted CE by
   the *masked-position count* instead of the *answer length*.** This is not the LLaDA SFT
   recipe and not what the pinned `sft_loss_llada` does; it inflates the loss by
   `1/mean_b(t_b)` and makes it batch-composition dependent. `L_mem`, the frozen probe,
   and the pre-registered 0.10 guard tolerance are all read off this quantity.

The `disable_adapter()` question is **PASS**, with empirical evidence.

---

## 1. PASS / FAIL table (§6 of the mandate)

| # | Check | Verdict | Key evidence |
|---|---|---|---|
| 1a | `mask_id = 126336` | **PASS** | `common_big.py:21`; `config.json` `"mask_token_id": 126336`; tokenizer id 126336 = `<\|mdm_mask\|>` |
| 1b | `eos = 126081` | **PASS** | `common_big.py:22`; `config.json` `"eos_token_id": 126081`, `"pad_token_id": 126081`; id 126081 = `<\|endoftext\|>` |
| 1c | revision `08b83a6f…963b07` | **FAIL (unverifiable as written)** | `common_big.py:16` is a bare string constant. The snapshot at `/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct/` has no revision/commit marker and nothing in any script compares the string to anything. Asserted, not verified. |
| 1d | generation uses the author's `generate()` unmodified | **PASS** | `common_big.py:17-19` imports from the pinned repo; `git log -1` = `9182493720ed723ef8031210d85959364e51cbe0` of `ML-GSAI/LLaDA`, `git status --porcelain` clean, `git diff HEAD -- generate.py` empty. `gen_one` (`common_big.py:113`) calls it with only documented kwargs. |
| 1e | no autoregressive shifted likelihood anywhere | **PASS** | Grep for `[:, :-1]`, `[:, 1:]`, `shift_logits`, `shift_labels` over all three script trees: **zero hits**. No call site passes `labels=` to the model. Model is bidirectional: `modeling_llada.py:639,659,717` all pass `is_causal=False`/`causal=False`, and `:1272` selects `get_bidirectional_attention_bias` because `config.alibi=false`. |
| 2a | the `ff_out` name collision is real | **PASS** | `modeling_llada.py:867-930` — `LLaDALlamaBlock.ff_out` is the MLP down-proj; `modeling_llada.py:1086,1368,1493` — `transformer.ff_out` is the LM head. PEFT list-form matching is `key.endswith(f".{target_key}")` (`peft/tuners/tuners_utils.py:979`), so a bare `"ff_out"` **would** wrap the head. The docstring claim is correct. |
| 2b | module counts per config | **PASS (counted)** | see §2 — `lora_targets()` → 128 modules; the regex → 224 modules, head untouched. Confirmed empirically in `spex/outputs/baselines/*.json`: `blocks_wrapped=32`, `linears_per_block=[7]`, `lm_head_wrapped=false`, `n_trainable=41,943,040`. |
| 2c | MLP projections silently excluded in some scripts, included in others | **FAIL — real, consequential inconsistency** | see §2. Memorization backbone and SPEX unlearning arms: 224 modules. Every ANVIL script: 128 modules. |
| 3a | `forward_process` masks only supervised positions | **PASS** | `common_big.py:177,181` — `maskable = labels != -100`, `masked &= maskable`. Prompt labels are `-100` (`common_big.py:132-135`), so the prompt is never corrupted. |
| 3b | `t ~ U(t_low, 1)` with `eps` | **PASS** | `common_big.py:179`. At `t_low=0` this is `(1-eps)·u + eps`, identical to the LLaDA GUIDELINES form. Upper bound is exactly 1 for any `t_low`. |
| 3c | per-token CE divided by `p_mask` | **PASS at the token level** | `memorize_big.py:145`, `flatten.py:153`, `relearn_probe.py:155`. |
| 3d | 1/t reweighting correct **relative to the LLaDA SFT recipe** | **FAIL** | see §4. The normaliser is the masked-position count, not `answer_len`. The pinned `sft_loss_llada` (`llada_fxmix.py:493`) does it correctly; our scripts do not. |
| 3e | loss restricted to answer tokens | **PASS** | `common_big.py:134-135` supervises `[len(p_ids), len(p_ids)+len(a_ids))` only; trailing EOS excluded. Logged: `supervised=1220, eos_supervised=0` (`logs/p024_big_memorize.log`). |
| 3f | `answer_weight` uniform across facts (no fact-specific leakage) | **PASS, with a caveat** | `memorize_big.py:139-142` applies one CLI scalar to every fact's `answer_positions`. No per-fact table, no split-dependent branch. Caveat in §4.3: the *effective* per-fact weight scales with answer-token count (1 vs 2 tokens), and the actual value used for `memorized_big` is unrecorded (§3). |
| 4 | `disable_adapter()` returns the **memorized** model, not the uninjected base | **PASS** | see §5. PEFT 0.14.0 semantics + empirical `adapter_delta_before = 0.0` on this exact stack. |
| 4b | no gradient path leaks into the reference | **PASS** | §5.3. |
| 5 | something verifies a saved adapter reloads to bit-identical behaviour | **FAIL — nothing does** | see §6. The only such gate (`memorize_big.py:234-256`) is unreachable dead code and never ran. |
| 6 | measured peak GPU and s/step from run reports | **PASS (extracted)** | see §7. |

---

## 2. The adapter-coverage inconsistency, characterised precisely

### 2.1 Ground truth: the module inventory

`config.json`: `"block_type": "llama"`, `"n_layers": 32`, `"d_model": 4096`,
`"mlp_hidden_size": 12288`, `"embedding_size": 126464`, `"weight_tying": false`.
So the live block class is `LLaDALlamaBlock` (`modeling_llada.py:853-930`), which owns
exactly **seven** `nn.Linear` leaves:

| leaf | shape | `modeling_llada.py` |
|---|---|---|
| `q_proj` | 4096 → 4096 | `:856` |
| `k_proj` | 4096 → 4096 | `:859` |
| `v_proj` | 4096 → 4096 | `:862` |
| `attn_out` | 4096 → 4096 | `:557` (inherited base-block attr, used `:724`) |
| `ff_proj` | 4096 → 12288 | `:867` |
| `up_proj` | 4096 → 12288 | `:871` |
| `ff_out` | 12288 → 4096 | `:562`, applied `:930` |

Plus the top-level LM head `transformer.ff_out` (4096 → 126464), `modeling_llada.py:1086`,
read at `:1368` and exposed at `:1493`.

### 2.2 What each config actually matches

PEFT 0.14.0 `check_target_module_exists` (`peft/tuners/tuners_utils.py:973-979`):

* `target_modules` **as a string** → `re.fullmatch(target_modules, key)`
* `target_modules` **as a list** → `key in target_modules or key.endswith(f".{target_key}")`

Therefore:

| config | call sites | matched modules | trainable @ r=16 (fp32) | LM head? |
|---|---|---|---|---|
| `lora_targets()` = `["q_proj","k_proj","v_proj","attn_out"]` (`common_big.py:161`) | `flatten.py:116`, `relearn_probe.py:117`, `noise_probe.py:103`, `memorize_big.py:96` (`--targets attn4`, the **default**) | **4 × 32 = 128** | **16,777,216** | no |
| `r".*blocks\.\d+\.(q_proj\|k_proj\|v_proj\|attn_out\|ff_proj\|up_proj\|ff_out)$"` | `arc_big.py:159`, `arc.py:159`, `run_baseline.py:123` (via `cfg.lora_target_regex`), `memorize_big.py:97` (`--targets block7`), `llada_poc/scripts/unlearn.py:125` | **7 × 32 = 224** | **41,943,040** | no — `transformer.ff_out` fails the `blocks\.\d+\.` anchor under `re.fullmatch` |
| hypothetical `["q_proj",…,"ff_out"]` list form | — | 224 **+ the LM head** | — | **yes, would wrap it** |

Both counts are confirmed on disk, not estimated:

* `spex/outputs/baselines/{sc_tf,emdu,mdu_t1,npo}.json` → `blocks_wrapped: 32`,
  `linears_per_block: [7]`, `lm_head_wrapped: false`, `n_trainable_params: 41943040`
  (written by `FX.check_lora_targets`, `llada_fxmix.py:427-438`, which *asserts* 7/block
  and no non-block module).
* `spex/adapters/*/report.json` → `n_trainable: 41943040` for all 13 ARC/uncached arms.
* `anvil/adapters/*/report.json` → `n_trainable: 16777216` for all 6 ANVIL arms.
* `spex/adapters/memorized_big/adapter_config.json` → `"r": 32`, `"lora_alpha": 64`,
  `target_modules` = **the 7-module regex**. Byte-level corroboration:
  `adapter_model.safetensors` is 335,606,544 B; 224 modules at r=32 in fp32 is
  83,886,080 params = 335,544,320 B of tensor payload. Matches.

### 2.3 The inconsistency, stated exactly

There are **two** distinct inconsistencies, and the one the mandate suspected is not the
one that bites.

**(i) `lora_targets()` is a dead default that disagrees with every adapter actually
shipped in SPEX — and it disagrees for a reason that is no longer true.**
`common_big.py:158-161` excludes `ff_out` because of the head collision. That collision is
real *for the list form only*; the block-anchored regex resolves it (§2.2). So the
docstring's justification does not justify also dropping `ff_proj` and `up_proj`, which
have no collision at all. Net effect: `lora_targets()` silently removes **96 of 224
modules (43%) and 60% of the parameters** — all the MLP capacity — for no stated reason.
`memorize_big.py:57` makes `attn4` the **default**, yet the shipped
`memorized_big/adapter_config.json` shows the backbone was trained with `block7`. The run
therefore passed `--targets block7` explicitly. **Classification: inconsistency, plus an
undocumented assumption in the docstring.**

**(ii) The memorization adapter and the SPEX unlearning adapters agree (224 modules); the
ANVIL stage-2 and probe adapters do not (128 modules).** This is the consequential one:

| role | script:line | modules | consequence |
|---|---|---|---|
| memorized backbone θ₀ | `adapter_config.json` (regex) | 224, r=32 | — |
| SPEX unlearning (ARC/uncached) | `arc_big.py:159` | 224, r=16 | matched to θ₀ ✓ |
| ANVIL stage-2 "flattening" | `flatten.py:116` | **128**, r=16 | **mismatched** |
| ANVIL relearning adversary | `relearn_probe.py:117` | **128**, r=32 | **mismatched** |
| ANVIL noise perturbation | `noise_probe.py:103` | **128**, r=16 | **mismatched** |

Three consequences, in descending severity:

* **`flatten.py` cannot flatten the directions the unlearning edit lives in.**
  `probe_grad_norm()` (`flatten.py:183-195`) takes `torch.autograd.grad(lm, params)` where
  `params` are *only* the fresh attn-4 LoRA tensors. The reported
  "`||grad L_mem||`" is therefore the norm of the memorization gradient **projected onto a
  128-module attention-only LoRA subspace** — not `||∇L_mem||`, and not even the subspace
  (`big_uncached/ckpt_300`, 224 modules) that produced θ_u. The headline
  `probe_grad_norm 1.117 → 0.690` is a 38% reduction *of a projection*. THEORY.md's
  Lemma 1 / Prop 2 are statements about the full gradient. **Classification: defect.**
* **`relearn_probe.py`'s adversary is weaker than the defence it attacks.** The adversary
  gets 128 modules; the unlearning edit used 224. A weaker attacker biases the
  pre-registered gate ("AXIS REAL iff unlearned held-out at k=10 ≥ base + 0.25") toward
  DEAD. Here it passed anyway and by a wide margin (0.6047 vs 0.0233 ⇒ +0.581 ≫ 0.25), so
  **the AXIS-REAL conclusion survives**; but any future *negative* relearning result from
  this script is not admissible evidence of robustness.
  **Classification: defect, currently non-fatal.**
* **`noise_probe.py` compares apples to oranges by construction.** `EDIT_NORM`
  (`noise_probe.py:81-89`) is the Frobenius norm of the **224-module** unlearning edit, but
  the random perturbation is spread over **128** modules (`:103`). The x-axis
  "perturbation this large relative to the unlearning edit" therefore concentrates an
  edit-sized norm into 57% of the modules, overstating the per-module perturbation by
  ≈ √(224/128) ≈ 1.32× in the attention blocks and applying exactly zero to the MLPs where
  60% of the real edit lives. **Classification: defect.**

---

## 3. The memorized backbone was never verified (the single biggest hole)

`memorize_big.py` declares nine gates (docstring `:10-16`). What actually happened:

```
logs/p024_big_memorize.log, final lines:
  [throughput] 465.4s total, 0.194s/step
  [adapter] saved 320.1 MiB -> /data/bijaypandey/spex/adapters/memorized_big
  Traceback (most recent call last):
    File ".../memorize_big.py", line 269, in <module>  sys.exit(main())
    File ".../memorize_big.py", line 199, in main
      "question": q, "answer": f["answer"], "aliases": f["aliases"],
  KeyError: 'aliases'
```

`ATTEMPTS.tsv`: `p024  big  memorize  …  rc=1`. The crash is at `memorize_big.py:199`,
inside the qualification loop, i.e. **after** the adapter was saved (`:183`) and **before**
G4 (`:228-229`), G3 (`:253`) and the report write (`:261`).

Consequences:

* `spex/outputs/big_memorization_report.json` **does not exist** (confirmed:
  `FileNotFoundError`). `R["gates"]` was never serialised. G1 and G2a/G2b printed or
  computed in memory; **G2c (params moved), G2d (base weights unchanged, the
  SHA-256-over-every-non-LoRA-tensor check at `:170-177`), G2e (loss decreased) were
  computed and thrown away.** No record exists that the base weights did not drift.
* `R["hparams"]` was never written, so **the backbone's `--answer-weight`, `--t-low`,
  `--batch`, `--lr` and `--seed` are unrecoverable.** `adapter_config.json` recovers only
  `r=32`, `alpha=64`, `targets=block7`; the log recovers only `steps=2400`
  (`[train] step 2399`) and `0.194 s/step`. There is **no launch script** for
  `memorize_big` anywhere (`grep -rn memorize_big --include=*.sh` → nothing). The headline
  artifact of the whole study is not reproducible from the record.
  **Classification: defect (reproducibility).**
* `qualify_big.py` re-ran only the *qualification*, and its own docstring (`:4-5`)
  mis-states the cause as "missing `aliases` in the generated facts file, since patched" —
  the facts file does carry `aliases` now, but the file read by the crashed run did not.
  The re-run reports `qualified: false` (0.7969 < 0.80) and `ATTEMPTS.tsv` shows
  `p025 … rc=1`. **The gate failed and the study proceeded.** This is disclosed in
  `AMENDMENT_01` and worked around via `big_recalled_facts.json` (51/64), so it is a
  declared limitation rather than a hidden one — but note that `evaluate_big.py:48-49`
  still computes `forget_all` / `retain_all` over all 128 rows, so the `forget ALL` column
  in both published tables mixes never-memorised facts into a forgetting metric.
* Training ran **2400 steps where ~150 sufficed.** From `logs/p024_big_memorize.log`:
  `step 150 loss=0.0001 gnorm=0.002`, and from step ~420 onward `loss=0.0000 gnorm=0.000`
  for **1980 consecutive logged steps**. ~82% of the 465 s was a confirmed no-op. Harmless
  scientifically, but it means `G2e_loss_decreased` is a vacuous gate and that the backbone
  sat at a zero-gradient point for most of training.

### 3.1 G3 is unreachable dead code and can never pass

```python
# memorize_big.py:236-238
    with torch.no_grad():
        enc = tok(__import__("common").build_prompt(tok, probe_q),
                  add_special_tokens=False, return_tensors="pt")
```

`memorize_big.py:31` puts only `/data/bijaypandey/spex/scripts` on `sys.path`;
`common_big.py:18` adds the pinned LLaDA repo. **Neither contains `common.py`** — the only
`common.py` in the tree is `/data/bijaypandey/llada_poc/scripts/common.py`, which
`memorize_big.py` never adds to the path (`arc_big.py:31` and `flatten.py:48` do; this
script does not). Reproduced under the project venv:

```
$ python -c "sys.path=[spex/scripts, llada_official]; __import__('common')"
ModuleNotFoundError CONFIRMED: ModuleNotFoundError No module named 'common'
```

So even with the `aliases` bug fixed, `memorize_big.py` would crash at `:237` and G3 would
never run. The correct symbol, `build_prompt`, is already imported in-module via
`common_big`. **Classification: defect. This is a copy-paste residue from
`llada_poc/scripts/memorize.py:237`, where `common` *is* importable — i.e. the bug was
introduced by the `*_big` fork.**

---

## 4. Masking, normalisation and answer-bearing gradients

### 4.1 PASS items (stated for the record)

* Prompt is never corrupted: `common_big.py:177` `maskable = labels != -100`, and prompt
  labels are `-100` (`:132-135`). Padding (`EOS_ID`, label `-100`) is likewise never
  masked.
* `t` distribution: `common_big.py:179`
  `t = t_low + (1 - t_low - eps)·u + eps` ⇒ `t ∈ [t_low+eps, 1]`. At `t_low=0` this is
  exactly the GUIDELINES `(1-eps)·t + eps`. Correct.
* Loss restricted to the answer sentence's own tokens, trailing EOS excluded
  (`common_big.py:134-135`). G1's measured breakdown
  (`logs/p024_big_memorize.log`: `supervised=1220 eos_supervised=0
  answer_word_tokens=204 (16.7%)`) confirms no EOS and no padding in the supervised set.
* `answer_positions` are searched only from `i = len(p_ids)` onward
  (`common_big.py:145`), so no prompt token can be mistaken for an answer token. No label
  leakage.

### 4.2 DEFECT — the ELBO normaliser is the masked-position count, not the answer length

LLaDA GUIDELINES SFT loss, as transcribed in the pinned package
(`llada_fxmix.py:478-479` and implemented at `:492-493`):

```python
tok = F.cross_entropy(logits[mi].float(), input_ids[mi], reduction="none") / p_mask[mi]
return torch.sum(tok / ans_len[mi]) / b          # ← divide by ANSWER LENGTH
```

Our scripts instead divide by the number of masked positions:

```python
# memorize_big.py:145-147
tl = F.cross_entropy(logits[masked].float(), ids[masked], reduction="none") / p_mask[masked]
tl = tl * wt[masked]
loss = tl.sum() / wt[masked].sum().clamp_min(1)   # ← divide by MASKED COUNT
```

```python
# flatten.py:153-154  (L_mem, "the ADVERSARY's objective")
tl = F.cross_entropy(logits[masked].float(), ids[masked], reduction="none") / p_mask[masked]
return tl.mean()
```

Same form at `flatten.py:189-190` (the frozen probe), `flatten.py:208-209` (the θ_u guard
baseline), `flatten.py:280-282` (the per-step guard), and `relearn_probe.py:155-157` (the
adversary).

Why it is wrong: the masked count in row *b* is ≈ `t_b · L_ans`. Dividing the
`1/p_mask`-weighted sum by `Σ_b t_b·L_ans` instead of `Σ_b L_ans` scales the objective by
**`1 / mean_b(t_b)`**. Consequences:

* The loss is **batch-composition dependent**: with `batch=8` and one independent
  `t_b ~ U(ε,1)` per row (`common_big.py:179` draws `b` values), `mean_b(t_b)` ranges
  roughly 0.3–0.7, so `L_mem` carries a **±40% artefact** uncorrelated with the model.
  `flatten.py:198-202` documents exactly this symptom ("L_mem swings wildly across
  batches (0.66 → 0.18 with no real drift)") and attributes it to "the mask rate is
  t ~ U(0,1)". That is the *proximate* cause; the normaliser is the *root* cause, and the
  frozen-probe workaround (correct and well-reasoned) freezes the artefact rather than
  removing it.
* The pre-registered guard tolerance `--guard-tol 0.10` (`flatten.py:89`) is an **absolute**
  threshold on a quantity that carries a multiplicative, mask-rate-dependent scale. It is
  comparable across steps only because the probe is frozen — which is precisely why the
  frozen probe is load-bearing and must not be relaxed.
* It biases *memorization* training toward low-mask-rate states (the `1/t` inflation
  up-weights rows where most of the answer is already visible), which is the **opposite**
  of the generation condition (a fully masked 32-token canvas). This is the exact failure
  the `t_low` knob was invented to patch — see `common_big.py:170-175`:
  *"with t_low=0 roughly half the answer stays visible, so the model can copy a
  neighbouring answer … Raising the floor moves training states toward the generation
  condition."* **`t_low` is a workaround for this normalisation defect, documented as an
  engineering knob rather than as a bug.**

Severity: **medium-high for ANVIL** (the flattening target, the probe and the guard are all
read off this quantity), **low for the memorization stage** (the gate there is
generation recall, which is normaliser-independent), **none for the SPEX arms** (the forget
term is a KL, and the retain term calls the pinned, correct `FX.sft_loss_llada` —
`arc_big.py:260-261`). **Classification: defect.**

### 4.3 Minor masking issues

* **Forced-mask repair biases `p_mask`.** `common_big.py:182-188` guarantees a non-empty
  supervised mask by forcing one random maskable position to `True`, but leaves
  `p_mask = t` for that position. Its true marginal masking probability is 1, so the
  `/p_mask` division over-weights it by `1/t`. Affects only rows that drew no mask at all
  (probability `(1-t)^n_sup`, non-negligible at small `t`).
  **Classification: defect, small.**
* **The model never sees a masked terminal EOS, but generation starts from a fully masked
  32-token canvas.** `make_example` excludes the trailing EOS from supervision
  (`common_big.py:134-135`) and `forward_process` masks only supervised positions, so the
  backbone is trained on ~10 answer positions and asked at inference to fill 32
  (`frozen_spec.json` `gen_length: 32`). This train/generate mismatch is the most likely
  mechanical explanation for θ₀ capping at 0.797 recall and for its **6 degenerate
  outputs** (`big_eval_theta0.json`, the only arm with any). Note the pinned package does
  the opposite and treats EOS padding as answer tokens (`llada_fxmix.py:300-304,478-479`).
  **Classification: undocumented assumption, with a measured consequence.**
* **`answer_weight` uniformity.** `memorize_big.py:139-142` is a single CLI scalar applied
  identically to every fact — **no fact-specific leakage, no forget/retain branch: PASS**.
  Caveat: because the normaliser is `wt[masked].sum()`, a fact whose answer tokenises to 2
  tokens contributes `2w` and a 1-token answer `1w` (measured: 204 answer-word tokens over
  128 facts = 1.59 avg), so the *effective* per-fact weight varies with tokenisation
  length. Structural, not tuned. **Classification: undocumented assumption.**
* **The adversary is given `--answer-weight 4.0` by default** (`relearn_probe.py:60`) while
  `memorize_big.py:58` defaults to 1.0 and the value actually used for the backbone is
  unrecorded (§3). A stronger-than-original learner could *teach* rather than *unlock*.
  The `base` floor control is exactly the right answer to this and it holds
  (`relearn_base.json`: held-out 0.0233 at k=10), so the inference stands.
  **Classification: fine, because the control exists.**

### 4.4 Objective-level confound worth stating

`build_big.py:72` gives every forget answer a retain twin ("4 entities sharing every
answer"), and `build_fact_set` derives `F` from the answer surface forms
(`llada_fxmix.py:338-371`). So **the forget `fact_mask` for `forget_000` ("Paris") contains
the same token ids that the retain SFT term must produce for its retain twin.** The forget
and retain terms are in direct opposition on those tokens by design. `build_fact_set`
accepts a `frequent_ids` argument for exactly this purpose (`llada_fxmix.py:339`);
`arc_big.py:74` does not pass it. The measured outcome — forget 0.047, retain 1.000
(`big_eval_uncached.json`) — says the model resolved it by conditioning on the entity,
which is the interesting result. But it is an uncontrolled, undocumented design pressure.
**Classification: undocumented assumption.**

---

## 5. `disable_adapter()` verdict

### 5.1 Verdict: **PASS — `disable_adapter()` returns the MERGED MEMORIZED model, not the uninjected base.**

The stack under test (`arc_big.py:154-160`):

```python
model = PeftModel.from_pretrained(base, a.memo).merge_and_unload().to(DEV)
model.train()
model = get_peft_model(model, LoraConfig(r=a.rank, …, bias="none",
        target_modules=r".*blocks\.\d+\.(…)$", init_lora_weights=True))
```

and identically in `flatten.py:109-116` (a **three**-level stack: base → memorized →
`big_uncached/ckpt_300` → fresh LoRA), `relearn_probe.py:100-117`, `noise_probe.py:97-103`.

### 5.2 Why, from PEFT 0.14.0 source

PEFT version confirmed `0.14.0` (`.../site-packages/peft-0.14.0.dist-info/METADATA`).

* `merge_and_unload()` folds `ΔW` into `base_layer.weight.data` and **replaces the
  `LoraLayer` wrapper with the original `nn.Linear`** (`peft/tuners/lora/layer.py:452-500`).
  After this call there is no adapter object left to disable — the memorized weights are
  part of the backbone.
* `PeftModel.disable_adapter()` (`peft/peft_model.py:865-910`) takes the non-prompt-learning
  branch at `:896` and calls `self.base_model.disable_adapter_layers()`
  (`:905`), which is `_set_adapter_layers(enabled=False)`
  (`peft/tuners/lora/model.py:398-411`). That toggles the `enable_adapters` flag on
  **`BaseTunerLayer` instances belonging to this `PeftModel` only**.
* So inside the context the forward pass is `W_base + ΔW_memorized` (+ `ΔW_unlearned` in
  the ANVIL stack) with the *fresh* adapter bypassed. **That is θ₀ (resp. θ_u), not the raw
  checkpoint.**
* `bias="none"` at every `get_peft_model` call site avoids the one documented inexactness
  path (`peft/tuners/lora/model.py:404-410` warns that `bias != "none"` makes
  `disable_adapter_layers()` *not* reproduce the base model). Correct everywhere.
* Restore is unconditional for a fully-enabled model (`peft_model.py:908-910`), so
  reference passes cannot leave the student's adapter disabled.

### 5.3 Empirical confirmation already on disk

`run_baseline.py:157-162` runs exactly the right check on exactly this stack:

```python
R["adapter_delta_before"] = adapter_delta(model, probe_ids, probe_attn)
assert R["adapter_delta_before"] == 0.0, "fresh LoRA is not a no-op; disable_adapter() != memorized ref"
```

All four baseline reports record **`adapter_delta_before: 0.0`** (exactly, not
approximately) — `spex/outputs/baselines/{sc_tf,emdu,mdu_t1,npo}.json` — together with
`reference.note: "disable_adapter() == qualified MEMORIZED model (p0), not the raw
checkpoint"`. Same PEFT, same merge, same regex, same `init_lora_weights=True` (B=0). This
is direct measured evidence for the verdict.

### 5.4 No gradient leaks into the reference — PASS

* `arc_big.py:62-64`: `with torch.no_grad(), model.disable_adapter():` → `z0` has no graph;
  `exclusion_target` (`mdmu/fxmix.py:73-107`) is pure tensor algebra on it.
* ARC cache path: `pool[-1]["log_r"] = log_r.detach().clone()` (`arc_big.py:197`).
* `soft_ce_kl` (`mdmu/fxmix.py:119-126`) returns `ce - ent.detach()`; `r = log_r.exp()`
  carries no grad. Gradient of the KL equals the gradient of the CE w.r.t. student logits
  only. Correct.
* `flatten.py:156-161`: `anchor_logits` is `@torch.no_grad()` **and** wrapped in
  `disable_adapter()`, and the call site re-wraps in `torch.no_grad()` (`:263-264`).
  Belt and braces.
* `flatten.py:204-210` (guard baseline) likewise.

### 5.5 The gaps around the verdict (these are the real risks)

1. **`arc_big.py`, `flatten.py`, `relearn_probe.py` and `noise_probe.py` do not assert
   it.** The check exists in the codebase (`run_baseline.py:157-162`) and was simply not
   ported to the four scripts that produced every published `big_*` and ANVIL number.
   `FX.check_lora_targets` (`llada_fxmix.py:427-438`, used at `run_baseline.py:130`) was
   likewise not ported. **Classification: defect (missing gate), cheap to fix.**
2. **Nothing quantifies merge fidelity.** `merge_and_unload()` adds an fp32-computed `ΔW`
   into a **bf16** `base_layer.weight` (`load_base` uses `torch_dtype=torch.bfloat16`,
   `common_big.py:94-95`). The merged model is therefore *not* bit-identical to the
   unmerged memorized model, and `flatten.py:109-110` / `relearn_probe.py:100-107` merge
   **two or three** adapters in sequence, compounding the rounding.
3. **`evaluate_big.py` evaluates θ₀ unmerged and the arms merged — an uncontrolled
   asymmetry in the headline comparison:**

   ```python
   # evaluate_big.py:24-26
   m = PeftModel.from_pretrained(base, a.memo)              # θ₀: NOT merged
   if a.adapter:
       m = PeftModel.from_pretrained(m.merge_and_unload(), a.adapter)   # arms: merged
   ```

   So `big_eval_theta0.json` (forget 0.7812, retain 0.7734, **6 degenerate**) and
   `big_eval_arcK16.json` (forget 0.9453, retain 0.9531, **0 degenerate**) were produced on
   backbones that differ by one bf16 merge. ARC K=16 discloses **+16.4 points more than its
   own reference** and has zero degenerate outputs where θ₀ has six. Part of that is surely
   real (the retain SFT term strengthens shared-answer facts while a 2-batch pool gives
   almost no forget gradient — the amendment's own mechanism), but **the merge asymmetry is
   an unmeasured confound sitting directly under the baseline row of both published
   tables.** **Classification: defect. `NEEDS-GPU` to size it (test T7 in §8).**

---

## 6. Save/reload agreement — **nothing verifies it**

Exhaustive grep for reload/agreement checks across `spex/scripts`, `anvil/scripts`,
`llada_poc/scripts` returns exactly two hits, and they are the same code:
`llada_poc/scripts/memorize.py:234-256` and `spex/scripts/memorize_big.py:234-256`.

* The SPEX copy is **unreachable** (§3.1, `__import__("common")` at `:237`) and **never
  ran** (the process died at `:199`).
* `arc_big.py:276-282` and `flatten.py:308-311` write checkpoints and (for ARC) record
  per-file SHA-256 — but a hash proves the *bytes* round-trip, not that the *behaviour*
  does. Nothing reloads and compares logits or generations.
* `evaluate_big.py`, `relearn_probe.py` and `noise_probe.py` all **consume** reloaded
  adapters (`evaluate_big.py:24-26`, `relearn_probe.py:100-107`, `noise_probe.py:97-99`)
  with no agreement check at the seam.

So every published number flows through at least one unverified save→reload→merge seam.

**Specified test (`NEEDS-GPU`, ~3 min, 1 GPU) — T1:**
1. Load base; attach the 7-module-regex LoRA at r=32; take 20 AdamW steps on the 128-fact
   set so `B ≠ 0`.
2. Record, on a fixed 8-row probe batch: full fp32 logits `Z_before`, and `gen_one` output
   for `FACTS[:4]` under the frozen decoder/seed.
3. `model.save_pretrained(tmp)`; `del model`; `torch.cuda.empty_cache()`.
4. Fresh `load_base` + `PeftModel.from_pretrained(tmp)`; recompute `Z_after` and the 4
   generations.
5. **Assert `max|Z_before − Z_after| == 0.0` exactly** (same dtype, same kernels, same
   device → bit-identity is the correct bar, not `< 1e-3`) **and** the 4 generation strings
   are identical. The existing `1e-3` tolerance at `memorize_big.py:253` is too loose to
   catch an `lora_alpha`/scaling or dtype-coercion mismatch on reload.
6. Repeat once more after `merge_and_unload()` → this is the merge-fidelity half (T7).

---

## 7. Measured memory and training behaviour (actual numbers from `report.json`)

### 7.1 SPEX — `/data/bijaypandey/spex/adapters/*/report.json`, key `measured`

| adapter | arm | steps | K | s/step (steady) | peak GPU GiB | fwd student / ref per step | ref fwd total | prefill s |
|---|---|---|---|---|---|---|---|---|
| `uncached_baseline` | uncached | 150 | 0 | **0.4840** | **23.685** | 2 / 1 | 150 | 0.0 |
| `unc_seed1` | uncached | 150 | 0 | 0.4881 | 23.663 | 2 / 1 | 150 | 0.0 |
| `unc_seed2` | uncached | 150 | 256 | 0.4865 | 23.674 | 2 / 1 | 150 | 0.0 |
| `arc_K64` | arc | 150 | 64 | 0.3906 | 24.361 | 2 / 0 | 8 | 1.1 |
| `arc64_seed1` | arc | 150 | 64 | 0.3812 | 24.310 | 2 / 0 | 8 | 1.2 |
| `arc64_seed2` | arc | 150 | 64 | 0.3885 | 24.278 | 2 / 0 | 8 | 1.1 |
| `arc_K256` | arc | 150 | 256 | 0.3924 | 25.838 | 2 / 0 | 32 | 3.5 |
| `arc_K1024` | arc | 150 | 1024 | 0.3947 | **31.655** | 2 / 0 | 128 | 11.3 |
| `long_uncached` | uncached | 600 | 256 | 0.4652 | 23.694 | 2 / 1 | 600 | 0.0 |
| `long_arc64` | arc | 600 | 64 | 0.3947 | 24.381 | 2 / 0 | 8 | 1.2 |
| `big_uncached` | uncached | 300 | 256 | **0.4608** | 23.673 | 2 / 1 | 300 | 0.0 |
| `big_arcK16` | arc | 300 | 16 | 0.4187 | 23.974 | 2 / 0 | 2 | 0.8 |
| `big_arcK512` | arc | 300 | 512 | 0.3951 | **27.781** | 2 / 0 | 64 | 5.8 |

Memorization (`logs/p024_big_memorize.log`, the only surviving record):
**465.4 s for 2400 steps = 0.194 s/step**, adapter 320.1 MiB. Peak GPU was to be written
to `report.json` (`memorize_big.py:258`) and is therefore **lost**.

Evaluation peak GPU (`spex/outputs/big_eval_*.json`): 15.562–15.628 GiB,
461.6–472.0 s per 256 generations (≈ 1.82 s/generation).

**Two things the table says that the write-ups should say:**

* **The headline "2 → 1 forwards/step" is a forget-term-only statement; the measured
  totals are 3 → 2.** `arc_big.py:262` adds `CNT.student += 1` by hand for the retain
  forward, which `FX.sft_loss_llada` performs outside the counted wrappers. So
  `fwd_total_per_step` is 3 (uncached) vs 2 (ARC) — a 1.5× forward reduction, matching the
  measured 1.24× wall-clock (0.4840 → 0.3906 at 150 steps / 0.4608 → 0.3951 at 300). The
  published "1.26× faster per step" is **consistent with the measurement**; only the
  "2 → 1" framing is narrower than it reads.
* **ARC's cost is memory, and it is reported but not headlined.** `cached_target_bytes`:
  145.7 MB at K=16, **4.15 GB at K=512**, and peak GPU 23.67 → 27.78 GiB (K=512) →
  **31.66 GiB (K=1024)**. The cache is a dense fp32 `log_r` of width V=126,464 per masked
  position, so it grows linearly in K and is the binding constraint on the method. The
  amendment's "cost results are unaffected" sentence covers time and forwards but not this.

### 7.2 ANVIL — `/data/bijaypandey/anvil/adapters/*/report.json`

| adapter | arm | opt | lr | lam_G | steps | s/step | peak GiB | fwd/step | bwd/step | probe ‖∇L_mem‖ first → last | probe L_mem first → last | guard |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `noop_check` | anchor_only | sgd | 5e-3 | 0.0 | 10 | 0.9063 | 17.222 | 5.0 | 3.0 | **1.154713 → 1.154713** | 0.9188 → 0.9188 | — |
| `cal_lr5e-3_lamG1.0` | anvil | sgd | 5e-3 | 1.0 | 30 | 0.7964 | 17.380 | 6.0 | 4.0 | 1.117353 → 0.967912 | 0.9133 → 0.9166 | — |
| `anvil_main` | anvil | sgd | 5e-3 | 1.0 | 300 | **0.7441** | **17.407** | 6.0 | 4.0 | 1.117353 → **0.694113** | 0.9133 → 0.9195 | — |
| `anvil_lamG10` | anvil | sgd | 5e-3 | **10.0** | 300 | 0.7316 | 17.407 | 6.0 | 4.0 | 1.117353 → **0.690175** | 0.9133 → 0.9204 | — |
| `anvil_lamG1` | anvil | **(none)** | 5e-5 | 1.0 | 150 | — | — | — | — | — | — | **FIRED @ 10** |
| `anchor_only` | anchor_only | **(none)** | 5e-5 | 0.0 | 150 | — | — | — | — | — | — | **FIRED @ 10** |

Relearning probes (`anvil/outputs/relearn_*.json`, `measured`): 4 runs, peak GPU and wall
recorded per run; `n_trainable = 33,554,432` (r=32 × 128 modules — see §2.3).

---

## 8. Known prior defects — fixed or still present?

### 8.1 `build_fact_set(...)["ids"]` vs `["F"]` — **FIXED, and provably so**

Every caller reads `["F"]`:

| caller | line | key read |
|---|---|---|
| `spex/scripts/arc_big.py` | `:79` | `fs["F"]` ✓ (with an explanatory comment at `:75-78`) |
| `spex/scripts/arc.py` | `:79` | `fs["F"]` ✓ |
| `llada_poc/scripts/llada_fxmix.py` | `:830-831` | `fs["F"]` ✓ |
| `llada_poc/scripts/unlearn.py` | `:56` | (via the same pattern) |

`grep -rn '\["ids"\]'` near any `build_fact_set` call: **zero hits**. The producer
(`llada_fxmix.py:371`) returns `{"F": …, "dropped_in_prompt": …, "prompt_conflict": …}` —
there is no `"ids"` key at all, so the bug would now be a `KeyError` rather than a silent
empty set.

Runtime confirmation that the fact sets are non-empty, from
`spex/adapters/{big_arcK16,big_arcK512,big_uncached}/report.json → data.fact_tokens`:
**min |F| = 4, max |F| = 7, zero facts with |F| = 0**, across all 64 forget facts in all
three arms. (`arc_big.py:177-178` records this; that record is the *de facto* guard.)

**One residual risk.** The comment at `arc_big.py:77-78` claims the bug is
"Caught by the p0(F) diagnostic before any full run." **`arc_big.py` has no p0(F)
diagnostic.** Its `forget_loss` (`:110-125`) calls `soft_ce_kl` directly and discards the
`pF_ref` / `frac_active` fields that the pinned `FX.fx_forget_loss_llada` computes
(`llada_fxmix.py:468-472`). The actual safety net is the `fact_tokens` record, which is
weaker (it is a log, not an assertion). And the silent-failure mode is still live upstream:
`exclusion_target` explicitly returns `r = p0` when `F` is empty
(`mdmu/fxmix.py:80,95,102`), so an empty fact set yields a forget term that is identically
**zero at initialisation and stays near zero** — no exception, no NaN, no warning.
**Classification: fixed defect; the stated guard does not exist; the silent-failure
mechanism is unchanged.**

### 8.2 An `in_retain_loss` flag set on all retain facts, destroying a held-out subset — **not present in this code; the same defect *class* is live in two places**

`grep -rn "in_retain_loss"` over all three script trees: **zero hits.** The flag does not
exist in the current stack.

But the *mechanism* that made it dangerous — a held-out subset that can silently become
empty and still produce a clean-looking number — is reproduced twice:

* `relearn_probe.py:79-82`
  ```python
  stride = max(1, len(recalled_forget) // a.leak)
  leaked = [recalled_forget[i * stride] for i in range(a.leak)]
  heldout = [f for f in recalled_forget if f["fact_id"] not in leaked_ids]
  ```
  With `len(recalled_forget) == a.leak`, `stride = 1` and `leaked` is **everything** ⇒
  `heldout = []`. With `len(recalled_forget) < a.leak`, `recalled_forget[i*stride]` raises
  `IndexError`. There is **no assertion that `heldout` is non-empty.**
* `relearn_probe.py:180-187` and `evaluate_big.py:39-46` then both do
  `n = len(r) or 1`, so an empty group reports **`disclosure: 0.0000` with `n: 0`** — the
  most favourable possible number, produced by the absence of data.

Currently benign: `big_recalled_facts.json` gives `n_recalled = 51`, `leak = 8`,
`stride = 6` ⇒ 8 leaked / **43 held out**, confirmed in all four `relearn_*.json`
(`n_heldout: 43`, `"n": 43` at every k). **Classification: the specific defect is absent;
a latent instance of its class is present in two scripts and must be fenced.**

### 8.3 Octal-literal shell arithmetic (`10#` forcing) — **FIXED, with one residual site**

`spex/attempt.sh` is explicit about it (`:19-20`: *"force base-10 and strip any leading
zero: a leading-zero literal would be read as OCTAL by shell arithmetic (the exact
accounting bug documented in the previous study)"*), and both ledgers force base 10 where
values re-enter arithmetic:

* `spex/attempt.sh:21` `sed 's/^0*\([0-9]\)/\1/'` on the CPU figure; `:24` and `:25`
  `10#$(cat …) + 10#${CH}` / `10#${CPU}`.
* `anvil/attempt.sh:15,18,19` — identical treatment.

**Residual (latent, currently harmless): `spex/attempt.sh:11`**
```bash
G=$(cat $W/GPU_SECONDS); REM=$(( 6*3600 - G )); REMW=$(( REM / NG ))
```
`$G` enters `$(( ))` **without** `10#`. It is safe today only because the writer at `:24`
can never emit a leading zero (`GPU_SECONDS` = `4870`, `CPU_SECONDS` = `8086`,
anvil `GPU_SECONDS` = `3916` — all verified leading-zero-free). If the counter file is ever
hand-edited or zero-padded, the budget check silently mis-reads.
**Classification: fine in effect, one undocumented assumption.**

Unrelated litter found while checking: `anvil/scripts/attempt.sh` and
`anvil/scripts/ledger.sh` are **0-byte files** shadowing the real ones one directory up.
All `launch_*.sh` scripts correctly invoke `/data/bijaypandey/anvil/attempt.sh`, so nothing
is broken — but a future caller using the `scripts/` path would silently charge nothing.

### 8.4 AdamW making a zero-gradient control move — **FIXED for the current arms; the superseded arms are still on disk**

Optimizer inventory:

| script:line | optimizer | role |
|---|---|---|
| `flatten.py:125-127` | **SGD** (`momentum=0.0`) default, `adamw` opt-in | ANVIL stage-2, **both** the `anvil` arm and the `anchor_only` control |
| `memorize_big.py:106` | AdamW | memorization (appropriate — no zero-gradient control) |
| `arc_big.py:244` | AdamW | SPEX unlearning arms (appropriate) |
| `relearn_probe.py:127` | AdamW | the adversary (appropriate — we want a strong attacker) |
| `llada_fxmix.py:691` | AdamW | pinned package |

`flatten.py:75-83` documents the defect and the fix in the `--optimizer` help text, and the
measured control is now a **true no-op**:

```
noop_check (opt=sgd, lam_G=0):  update_norm = [0.0, 0.0, 0.0] … 0.0
                                probe_grad_norm 1.154713 → 1.154713 (bit-identical)
                                probe_L_mem     0.9188   → 0.9188
```

versus the superseded runs, where `hparams.optimizer` is **absent** (the flag did not yet
exist, so AdamW was in force):

```
anchor_only (lam_G=0):  update_norm = [0.0, 0.0051, 0.0007] …  GUARD FIRED @ step 10, L_mem 0.604 → 0.1797
anvil_lamG1 (lam_G=1):  update_norm = [3.9997, 2.0935, …]      GUARD FIRED @ step 10, L_mem 0.604 → 0.1844
```

**No current control is affected. Classification: defect, fixed and demonstrated.**
Two housekeeping notes: (a) `anvil/adapters/{anchor_only,anvil_lamG1}/report.json` carry
`"optimizer": null` and a fired guard, and should be marked SUPERSEDED on disk so they
cannot be mined later as if they were the AdamW arm of a deliberate comparison;
(b) `noise_probe.py` exists precisely because of this defect
(`noise_probe.py:1-25`) — it turned a bug into a legitimate probe, which is good practice
and worth preserving in the write-up.

---

## 9. Two further findings that changed published numbers

### 9.1 DEFECT — the per-fact pool stratification is ARC K=16's, applied to all four arms

```python
# analyze_big.py:17-18
POOL = json.load(open(f"{S}/adapters/big_arcK16/report.json"))["pool_membership"]
IN_POOL = set(POOL["facts_in_pool"])
```
`IN_POOL` is then used for **every** arm (`analyze_big.py:31,38-39`), and the emitted
columns are headed `recalled IN-pool | recalled ABSENT` with **no arm qualifier**
(`analyze_big.py:46`).

But the arms' own reports disagree:

| arm | `pool_membership.n_facts_in_pool` (its own report) | what the published table attributes to it |
|---|---|---|
| `big_arcK16` | **15 / 64** | IN-pool n=20, ABSENT n=82 ✓ correct |
| `big_arcK512` | **64 / 64 — full coverage** | IN-pool n=20, ABSENT n=82 ✗ **wrong; K=512 has no ABSENT facts** |
| `big_uncached` | *no pool at all* (key absent) | IN-pool n=20, ABSENT n=82 — a stratification variable, not membership |
| `theta0` | *no pool* | same |

The evaluator logs make the misattribution visible:
`logs/big_eval_arcK512.log` prints `IN pool (n= 30)` / `ABSENT (n= 98)` — numerically
identical to `logs/big_eval_arcK16.log`, i.e. 15 facts × 2 question kinds. So
`--pool-report …/big_arcK16/report.json` was passed to all four evaluations.

Published consequence: in **`AMENDMENT_01_breaking_point.md`** the row
`| ARC K=512 | 0.172 | 0.216 | 0.350 (n=20) | 0.183 (n=82) | 1.000 | 0 |` reads as K=512's
own in-pool/absent split. It is not: all 64 facts were in the K=512 pool, so the "ABSENT"
cell describes 41 facts that the pool *did* contain. The inverted ordering in that row
(ABSENT 0.183 **below** IN-pool 0.350, the opposite of the prediction's direction) is
diagnostic of the mislabelling, not of a real effect.

For θ₀ and uncached the same column is a legitimate *fixed* stratification variable, and
`BIG_RESULTS.md` does disclose "Pool membership (ARC K=16)" in its preamble —
`AMENDMENT_01` does not repeat the caveat beside the table.

**What survives:** the amendment's two headline claims do **not** depend on this cell.
(i) "ARC K=16 does not forget" rests on the aggregate 0.931 vs uncached 0.049 and on
K=16's *own*, correctly-joined membership (IN 0.850 vs ABSENT 0.951, gap +0.101).
(ii) "even at full coverage K=512 reaches 0.216 vs 0.049" rests on aggregates. The
"prediction REFUTED as stated" verdict is likewise judged on `out["arcK16"]`
(`analyze_big.py:56`), which is the correct arm.
**What must be retracted:** the two IN-pool/ABSENT cells of the K=512 row in both
documents. **Classification: defect (mislabelled published number); does not overturn a
conclusion.**

### 9.2 DEFECT — `lam_G` is absorbed by gradient clipping, so the ANVIL dose–response is not real

```python
# flatten.py:245-247  (penalty gradient, scaled by lam_G)
for p, gp, g0 in zip(params, g_pert, g):
    gr = (gp - g0) / a.rho
    p.grad = gr.mul(a.lam_G) if p.grad is None else p.grad.add_(gr, alpha=a.lam_G)
…
# flatten.py:273-274
gn_upd = torch.nn.utils.clip_grad_norm_(params, 1.0)
opt.step()
```

`clip_grad_norm_` renormalises the **entire** gradient (penalty + anchor) to norm 1.0
whenever the raw norm exceeds 1. Since the penalty dominates the anchor at both
coefficients, the post-clip direction is ≈ `pen/‖pen‖` for any `lam_G ≳ 1` — `lam_G`
cancels exactly.

The data confirm it to three digits:

| arm | lam_G | `update_norm` step 0/1/2 (= **pre**-clip norm returned by `clip_grad_norm_`) | `probe_grad_norm_last` | `probe_L_mem_last` |
|---|---|---|---|---|
| `anvil_main` | 1.0 | 1.9642 / 3.0549 / 1.8431 | **0.694113** | 0.9195 |
| `anvil_lamG10` | 10.0 | 19.6423 / 30.5496 / 18.7356 | **0.690175** | 0.9204 |

The pre-clip norms are **exactly 10×** apart (1.9642 → 19.6423, 3.0549 → 30.5496) — both
far above the 1.0 cap — and the outcomes are **0.6% apart**. A 10× coefficient change
produced no effect because it could not survive the clip.

Consequence: any claim of a `lam_G` dose–response, or any inference that `lam_G = 1` is
"enough", is unsupported by these two runs. They are, for practical purposes, the same run.
`flatten.py:68` also advertises `anchor_only` as "identical objective with lam_G=0, so the
ONLY difference is the gradient-norm term" — true of the *objective*, but under clipping
the arm's step size is pinned at 1.0 while the control's is 0, so the comparison is
"normalised penalty direction vs. nothing", not "weighted penalty vs. no penalty".
**Classification: defect, consequential.**

Supporting observation (`relearn_flat150.json` vs `relearn_flat300.json`): the relearning
curves of the flattened arms at 150 and 300 steps are **identical at k = 0, 5, 10, 20**
(0.0465 / 0.2791 / 0.5814 / 0.6512) and differ only at k=2 and in degenerate counts, while
the adversary's step-1 losses differ (2.9636 vs 2.9498), proving the two backbones really
are different. Doubling the flattening did nothing measurable, and neither did flattening
at all versus the plain unlearned control (0.5814 vs 0.6047 at k=10 — one fact out of 43).
Consistent with §9.2: the effective update never scaled.

### 9.3 DEFECT, small — the flattening baseline is measured one step late

`flatten.py:285-286` calls `probe_grad_norm()` **after** `opt.step()` (`:274`), so
`hist[0]["probe_grad_norm"]` is measured at θ₁, not θ_u. The evidence is unambiguous: the
true θ_u value is **1.154713** (from `noop_check`, whose `update_norm` is exactly 0 so it
never moved), whereas every `anvil`/`cal` run records `probe_grad_norm_first = 1.117353`.
The arms and their control therefore do **not** share a baseline measurement, and the
published reduction is understated (1.1547 → 0.6941 is 39.9%, not the 37.9% implied by the
recorded first value). `L_mem_at_theta_u` *is* measured correctly before training
(`:203-211`), so only the gradient-norm baseline is affected.
**Classification: defect, small; trivially fixed by probing before the loop.**

### 9.4 Lower-severity items

| item | file:line | class |
|---|---|---|
| `R["hparams"] = vars(args) \| {"targets": lora_targets()}` **overwrites** the reported targets with `attn4` even when `--targets block7` is used — the report would have misdescribed the backbone had it been written | `memorize_big.py:63` | defect (reporting) |
| `R["checkpoints"] = ck` (the requested steps) is silently overwritten by `R["checkpoints"] = saved` (the written dirs) | `arc_big.py:144` vs `:297` | inconsistency, cosmetic |
| dead line: `tok = FX.__dict__.get("_tok")` immediately reassigned | `arc_big.py:149,151` | fine (litter) |
| `MASK_ID`/`EOS_ID`/`VOCAB` hard-coded instead of read from `model.config` (values verified correct: 126336 / 126081 / 126464) | `arc_big.py:36,39` | undocumented assumption |
| ARC pool is sampled at **batch** granularity (`pool[j]` is a whole 8-state batch), so "K states" is really `K/8` reusable batches — `n_pooled_batches` = 2 at K=16. Correctly reported, but "pool of K states" overstates the diversity | `arc_big.py:187-197,252-255` | undocumented assumption |
| `clean_forgetting` divides the clean-and-silent count by **all** rows, not clean rows, so it is not a conditional rate despite reading like one | `evaluate_big.py:45`, `relearn_probe.py:186`, `noise_probe.py:163` | undocumented assumption |
| four `big_eval_*` runs were launched outside `attempt.sh`, so ≈0.52 GPU-device-h is absent from `ATTEMPTS.tsv` | disclosed in `AMENDMENT_01` §Caveats | fine (disclosed) |
| `anvil/scripts/{attempt,ledger}.sh` are 0-byte shadows of the real scripts | — | litter, latent accounting hole |

---

## 10. Deliberately broken cases — a focused correctness suite (designed, NOT run)

Each test uses an **independent reference** (recomputed by a second code path, or a
hand-checkable closed form) and a **deliberately broken input**, and names the signature a
*passing* implementation must produce. These are regression tests for the failure *class*
that the `["ids"]`/`["F"]` bug belongs to: a defect that produces a plausible number
instead of an error.

| # | Broken input | Target | Expected failure signature | Cost |
|---|---|---|---|---|
| **T1** | nothing broken — save→reload→merge round trip | `memorize_big.py:234-256` (repaired), `arc_big.py:276` | `max\|Δlogits\| == 0.0` **exactly** and identical generations, at both the reload and the merge seam. Currently **cannot run at all** (§3.1). | GPU ~3 min |
| **T2** | **empty fact set**: force `fact_mask[i] = all False` for one forget row | `arc_big.py:74-85`, `exclusion_target` | `exclusion_target` returns `log_r == log_p0` exactly (`mdmu/fxmix.py:95,102`), so the per-state KL is **`0.0` to float precision at init** and the row's `n_fact_tokens == 0`. A correct harness must **raise** on `min(n_fact_tokens) == 0` before step 1, not log it. This is the exact silent-zero the `["ids"]` bug produced. | CPU (tiny random model) |
| **T3** | **reference is secretly the student**: replace `logits_reference` with `logits_student` (no `disable_adapter`, no `no_grad`) | `arc_big.py:61-64`, `flatten.py:156-161` | Forget KL ≡ 0 at every step and `grad_norm` from the forget term ≡ 0; and with `no_grad` also dropped, `torch.autograd.grad` succeeds through the "reference", which a correct harness must detect. **Positive control:** the real stack must satisfy `adapter_delta_before == 0.0` *and* a nonzero forget loss at step 1 (measured: 0.698 / 1.057 — §7, so the live code passes). | CPU |
| **T4** | **`eps = 0`** in `exclusion_target` | `mdmu/fxmix.py:82-106` | No NaN/Inf: `log_r_F = -inf` ⇒ `r(F) = 0`, `feasible` is False for every non-empty `F`, and `soft_ce_kl` stays finite via the `xlogy` convention (`:125`). **Two real traps to assert:** (a) `return_info=True` makes `gamma = +inf` (`:106` subtracts `-inf`); (b) `frac_active = (p0F > eps)` becomes ≈ always-true (`llada_fxmix.py:472`), silently changing the meaning of a reported diagnostic. `eps=0` is *permitted* by the assert at `:82`. | CPU |
| **T5** | **`eps = 1.0`** (and `eps < 0`) | `mdmu/fxmix.py:82` | `AssertionError` from `assert 0.0 <= eps < 1.0`. Confirms the guard is live. | CPU |
| **T6** | **zero masked positions**: a state with no `MASK_ID` in the answer region | `arc_big.py:103-113`, `flatten.py`, `forward_process` | `arc_big.py:112-113` returns `(None, 0)` ⇒ `loss = f_loss + r_loss` raises `TypeError` on `None`. A correct harness must either skip the step explicitly or assert `rows.numel() > 0`. Independently, `forward_process` must **never** produce this (`common_big.py:182-188` repairs it) — assert `masked.sum(1) > 0` for every row over 10⁴ draws at `t = eps`. | CPU |
| **T7** | **merge fidelity**: θ₀ unmerged vs θ₀ merged, same frozen decoder and seed | `evaluate_big.py:24-26` | Quantify `max\|Δlogits\|` and the generation-disagreement rate over all 128 facts × 2 kinds. **This is the §5.5(3) confound under the baseline row of both published tables** (θ₀ 0.7812 / 6 degenerate vs arcK16 0.9453 / 0 degenerate). If the disagreement rate is non-trivial, θ₀ must be re-evaluated merged. **`NEEDS-GPU`.** | GPU ~16 min |
| **T8** | **empty held-out subset**: run `relearn_probe.py --leak 51` (= `n_recalled`) | `relearn_probe.py:79-82,180-187` | Must raise. Currently reports `heldout_forget: {"n": 0, "disclosure": 0.0, "clean_forgetting": 0.0}` — a perfect score from no data (§8.2). Also test `--leak 52` ⇒ `IndexError` (an ugly but honest failure). | CPU (stub the model) |
| **T9** | **`lam_G` cancellation**: run `flatten.py` at `lam_G ∈ {1, 10}` with `clip_grad_norm_` **disabled** | `flatten.py:245-247,273` | Post-clip update directions must differ measurably between the two `lam_G` values. With clipping on they do not (0.694113 vs 0.690175 — §9.2). Assert `cos(Δθ(lam_G=1), Δθ(lam_G=10)) < 0.999`. **`NEEDS-GPU`.** | GPU ~10 min |
| **T10** | **normaliser equivalence**: compare `memorize_big`'s `tl.sum()/wt.sum()` against `FX.sft_loss_llada`'s `sum(tok/ans_len)/b` on identical `(ids, labels, noisy, masked, p_mask)` | `memorize_big.py:145-147`, `flatten.py:153`, `llada_fxmix.py:492-493` | The ratio must equal `1/mean_b(t_b)` to float precision (§4.2). Sweep `t_b` to show the batch-composition dependence, then assert the repaired form is `t`-invariant in expectation over 10⁴ draws. | CPU |
| **T11** | **adapter coverage**: assert the wrapped-module inventory at every `get_peft_model` call site | `arc_big.py:157`, `flatten.py:114`, `relearn_probe.py:115`, `noise_probe.py:101` | Port `FX.check_lora_targets` (`llada_fxmix.py:427-438`): **32 blocks, 7 linears each, `"NON-BLOCK" not in wrapped`**. The ANVIL call sites must *fail* this today (they wrap 4/block — §2.3), forcing an explicit decision rather than a silent 43% coverage gap. Add the `assert adapter_delta_before == 0.0` from `run_baseline.py:162`. | CPU for the count, GPU ~2 min for the delta |
| **T12** | **list-form `ff_out`**: `target_modules=["q_proj",…,"ff_out"]` | `common_big.py:158-161`, PEFT `tuners_utils.py:979` | The LM head **must** appear in the wrapped set (225 modules, `NON-BLOCK` present), proving the documented collision is real; and the block-anchored regex must **not** wrap it (224, no `NON-BLOCK`). This turns the docstring's claim into a test and records why `ff_proj`/`up_proj` need not have been excluded with it. | CPU |
| **T13** | **ARC cache staleness**: mutate the student's LoRA, then recompute `exclusion_target` on a pooled state | `arc_big.py:225-242` | Must stay bit-identical (`max\|Δ\| == 0.0`), because the reference is adapter-disabled and the pool is fixed — i.e. Prop 6 must be invariant to student updates, not just to recomputation at prefill time. The shipped gate (`:228-238`) checks only the latter, *before* any training step. Measured: `0.0` at prefill on both arms. **`NEEDS-GPU`.** | GPU ~2 min |
| **T14** | **revision pin**: compare the on-disk snapshot against `08b83a6feb34df1a6011b80c3c00c7563e963b07` | `common_big.py:16` | Must fail loudly today: there is no revision marker in the snapshot dir and nothing compares the constant to anything (§1, item 1c). Fix is a recorded per-file SHA-256 manifest asserted at load, in the style already used for adapters (`memorize_big.py:186-187`). | CPU |

---

## 11. Ranked findings

Ranked by whether the finding **could have changed a published conclusion**.

### Tier 1 — could have changed, or did change, a published number

| # | Finding | Class | Evidence |
|---|---|---|---|
| **1** | The 128-fact backbone's gates never ran: no `big_memorization_report.json`; G2c/G2d/G2e computed and discarded; **G4 and G3 never executed**; `--answer-weight` / `--t-low` / `--lr` / `--seed` unrecoverable; no launch script exists. Every `big_*` and ANVIL result depends on it. | defect | `logs/p024_big_memorize.log` traceback at `memorize_big.py:199`; `ATTEMPTS.tsv` `p024 rc=1`; report file absent |
| **2** | G3 (save/reload agreement) is **unreachable dead code** — `__import__("common")` cannot resolve from `memorize_big`'s path. Reproduced: `ModuleNotFoundError`. So **nothing in the entire stack verifies that a saved adapter reloads to identical behaviour**, and every number crosses ≥1 unverified save/reload/merge seam. | defect | `memorize_big.py:237`; `:31`; `common_big.py:18`; only `common.py` is in `llada_poc/scripts` |
| **3** | `lam_G` is annihilated by `clip_grad_norm_(params, 1.0)`: pre-clip norms exactly 10× apart, outcomes 0.6% apart. The ANVIL `lam_G` sweep is two copies of one run; any dose–response claim is unsupported. | defect | `flatten.py:245-247,273`; `anvil_main` vs `anvil_lamG10` reports |
| **4** | The published per-fact pool stratification applies **ARC K=16's** membership to **ARC K=512**, whose own report records 64/64 coverage. The K=512 row's `ABSENT (n=82)` cell describes facts the pool contained. Retract those two cells; the headline claims survive on aggregates. | defect (mislabelled published number) | `analyze_big.py:17-18,31,38-39`; `big_arcK512/report.json` `n_facts_in_pool: 64`; `logs/big_eval_arcK512.log` `IN pool (n=30)` |
| **5** | `evaluate_big.py` evaluates θ₀ **unmerged** and every arm **merged** — an unmeasured bf16-rounding asymmetry directly under the baseline row of both published tables (θ₀ 0.7812 / 6 degenerate vs arcK16 0.9453 / 0 degenerate). | defect | `evaluate_big.py:24-26`; `big_eval_{theta0,arcK16}.json` |
| **6** | `flatten.py` measures and minimises `‖∇L_mem‖` **projected onto a 128-module attention-only LoRA subspace** — not the full gradient, and not the 224-module subspace that produced θ_u. THEORY.md L1/P2 are about the full gradient. | defect | `flatten.py:116,183-195`; `n_trainable: 16777216` vs the edit's `41943040` |
| **7** | The ELBO normaliser is the masked-position count, not `answer_len` — not the LLaDA recipe and not what the pinned `sft_loss_llada` does. Inflates by `1/mean_b(t_b)`; makes `L_mem`, the frozen probe and the absolute 0.10 guard tolerance batch-composition dependent. `t_low` is a workaround for it. | defect | `memorize_big.py:145-147`; `flatten.py:153,189,208,281`; `relearn_probe.py:155-157` vs `llada_fxmix.py:493` |

### Tier 2 — bounds a conclusion or weakens a control, without overturning it

| # | Finding | Class | Evidence |
|---|---|---|---|
| **8** | Model revision `08b83a6f…` is **asserted, never verified** — a bare constant, no manifest, no comparison. | defect (provenance) | `common_big.py:16`; snapshot dir has no revision marker |
| **9** | `lora_targets()` drops 96/224 modules (60% of params) citing a collision that the block-anchored regex already solves, and justifies dropping `ff_proj`/`up_proj` which have no collision. It is `memorize_big`'s **default** yet disagrees with the shipped backbone. | inconsistency + undocumented assumption | `common_big.py:158-161`; `memorize_big.py:57,63,96`; `memorized_big/adapter_config.json` |
| **10** | `relearn_probe.py`'s adversary (128 modules) is weaker than the unlearning edit (224), biasing the AXIS gate toward DEAD. **Passed anyway by +0.581 vs a +0.25 bar, so the AXIS-REAL conclusion stands** — but a future negative result from this script is inadmissible. | defect, non-fatal | `relearn_probe.py:117`; `relearn_{unlearned,base}.json` |
| **11** | `noise_probe.py` scales a **224-module** `EDIT_NORM` onto a **128-module** perturbation: ≈1.32× over-perturbation in attention, exactly zero in the MLPs where 60% of the real edit lives. The x-axis does not mean what it says. | defect | `noise_probe.py:81-89` vs `:103` |
| **12** | The flattening baseline `probe_grad_norm` is measured **after** the first `opt.step()`: arms record 1.117353, the true θ_u value is 1.154713 (from the exactly-stationary `noop_check`). Arm and control do not share a baseline. | defect, small | `flatten.py:274,285-286`; `noop_check` vs `anvil_main` reports |
| **13** | Latent empty-held-out-subset: `--leak == n_recalled` silently yields `{"n": 0, "disclosure": 0.0}` (a perfect score from no data); `--leak > n_recalled` raises `IndexError`. Same silent-zero class as the `["ids"]` bug. Benign today (43 held out). | defect (latent) | `relearn_probe.py:79-82,180-187`; `evaluate_big.py:39-46` |
| **14** | `arc_big.py:77-78` claims the empty-fact-set bug is "Caught by the p0(F) diagnostic" — **`arc_big.py` has no p0(F) diagnostic**; it discards `pF_ref`/`frac_active`. The real net is the `fact_tokens` log, which is not an assertion. | defect (phantom guard) | `arc_big.py:110-125` vs `llada_fxmix.py:468-472` |
| **15** | None of `arc_big.py`, `flatten.py`, `relearn_probe.py`, `noise_probe.py` assert the reference identity or the module inventory, though `run_baseline.py:157-162` + `:130` already implement both. The gate exists and was not ported to the four scripts that produced every published number. | defect (missing gate) | as cited |
| **16** | Forget `fact_mask` tokens are, by design, the retain twins' answer tokens (`build_big.py:72`), and `frequent_ids` is never passed. The forget and retain terms oppose each other on those ids. The measured 0.047/1.000 split is the interesting result, but the pressure is uncontrolled and undocumented. | undocumented assumption | `build_big.py:72`; `arc_big.py:74`; `llada_fxmix.py:339` |
| **17** | Train/generate mismatch: trained on ~10 answer positions with EOS never masked; generates a fully masked 32-token canvas. Most likely mechanical cause of θ₀'s 0.797 cap and its 6 degenerate outputs (the only arm with any). The pinned package does the opposite. | undocumented assumption | `common_big.py:134-135,177`; `frozen_spec.json`; `big_eval_theta0.json` |
| **18** | "2 → 1 forwards/step" is forget-term-only; measured totals are 3 → 2 (the retain forward is counted by a manual `CNT.student += 1` outside the wrappers). The 1.26× wall-clock claim is consistent with measurement; only the framing is narrow. | inconsistency (framing) | `arc_big.py:262`; `measured.fwd_*_per_step` in all 13 reports |
| **19** | ARC's real cost is memory and is reported but not headlined: `cached_target_bytes` 4.15 GB at K=512, peak GPU 23.67 → 27.78 → **31.66 GiB** at K=1024. Linear in K; the binding constraint on the method. | undocumented assumption | `adapters/{big_arcK512,arc_K1024}/report.json` |

### Tier 3 — correct, or harmless, but worth recording

| # | Finding | Class |
|---|---|---|
| **20** | `disable_adapter()` returns the **memorized** model: PEFT 0.14.0 semantics (`peft_model.py:896-910` → `lora/model.py:398-411`), `merge_and_unload` unwraps (`lora/layer.py:452-500`), `bias="none"` everywhere, and `adapter_delta_before == 0.0` measured on this exact stack in all four baseline reports. **PASS.** | fine |
| **21** | No gradient path leaks into any reference: `no_grad` + `disable_adapter` at every site; cached `log_r` detached; `soft_ce_kl` returns `ce - ent.detach()`. **PASS.** | fine |
| **22** | No autoregressive shifted likelihood anywhere; attention is bidirectional (`is_causal=False` at `modeling_llada.py:639,659,717`; `get_bidirectional_attention_bias` at `:1272`); no call site passes `labels=`. **PASS.** | fine |
| **23** | Author's sampler verified unmodified: pinned repo at `9182493720ed723ef8031210d85959364e51cbe0` of `ML-GSAI/LLaDA`, clean tree, empty diff for `generate.py`; `gen_one` passes only documented kwargs; the fork's extra `end_think`/`logits_eos_inf` features are inert at their defaults (`generate.py:64`). **PASS.** | fine |
| **24** | `build_fact_set` key bug **fixed** — every caller reads `["F"]`; `"ids"` does not exist in the return dict; `min\|F\| = 4` across all 64 facts in all three arms. | fine (fixed) |
| **25** | Octal shell arithmetic **fixed** in both ledgers (`10#` + leading-zero strip, with an explanatory comment). One residual un-forced site at `spex/attempt.sh:11` (`REM=$(( 6*3600 - G ))`), safe only because the writer can never emit a leading zero. | fine + undocumented assumption |
| **26** | AdamW zero-gradient control **fixed**: SGD is the default and `noop_check` is a bit-exact no-op (`update_norm` identically 0, `probe_grad_norm` 1.154713 → 1.154713). The two superseded AdamW runs (`anchor_only`, `anvil_lamG1`, `optimizer: null`, guard fired @10) remain on disk and should be marked SUPERSEDED. | fine (fixed) + housekeeping |
| **27** | `answer_weight` carries no fact-specific or split-specific leakage — one CLI scalar for all 128 facts. Effective per-fact weight varies with answer-token count (1.59 avg), which is structural. | fine + undocumented assumption |
| **28** | The memorization gate **failed** (0.7969 < 0.80) and the study proceeded — but this is disclosed in `AMENDMENT_01` and handled with a declared recalled-only subset (51/64). Note that the `forget ALL` column still mixes in never-memorised facts. | fine (disclosed) |
| **29** | `forward_process`'s forced-mask repair leaves `p_mask = t` for a position whose true marginal is 1, over-weighting it by `1/t` on rows that drew no mask. | defect, small |
| **30** | Litter / cosmetics: `R["hparams"]` target overwrite (`memorize_big.py:63`); `R["checkpoints"]` double-assignment (`arc_big.py:144` vs `:297`); dead `FX.__dict__.get("_tok")` (`:149`); hard-coded `MASK_ID`/`EOS_ID`/`VOCAB` (`:36,39`); 0-byte `anvil/scripts/{attempt,ledger}.sh`; `clean_forgetting` denominators; ~0.52 GPU-device-h of evaluation absent from `ATTEMPTS.tsv` (disclosed). | mixed |

---

## 12. What needs GPU (for the coordinator to schedule)

| test | what it settles | est. |
|---|---|---|
| **T7** (§10) — θ₀ merged vs unmerged over all 128 facts × 2 kinds | Tier-1 #5. Decides whether the published θ₀ baseline row must be re-measured. **Highest value.** | ~16 min, 1 GPU |
| **T1** (§10) — save→reload→merge bit-identity at both seams | Tier-1 #2. The gate that has never once run. | ~3 min, 1 GPU |
| **T9** (§10) — `lam_G ∈ {1,10}` with clipping disabled | Tier-1 #3. Confirms the cancellation and recovers a real dose–response. | ~10 min, 1 GPU |
| **T11** (§10) — `adapter_delta_before == 0.0` + `check_lora_targets` at all four call sites | Tier-2 #15. Converts the §5 verdict from inference-plus-analogy into a per-run assertion. | ~2 min, 1 GPU |
| **T13** (§10) — Prop 6 exactness **after** student updates, not only at prefill | Tier-2. Strengthens the central cost claim. | ~2 min, 1 GPU |
| re-run `memorize_big` with G3 repaired and the report written | Tier-1 #1. The only way to recover the backbone's provenance; also lets `--steps` drop from 2400 to ~200 (82% of that run was a measured no-op). | ~2 min train + ~8 min gates |

Everything else in §10 (T2, T3, T4, T5, T6, T8, T10, T12, T14) runs on CPU against a tiny
random model exposing `.logits`, exactly as the pinned package's own dry-run harness does
(`llada_fxmix.py:3`, `_BuiltinLoRA` at `:389-424`). Those should be built first — they are
free, and they are the tests that catch the silent-zero class.
