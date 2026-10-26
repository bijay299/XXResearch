# AGENT 1 — UCE audit (R4 / R4b / R4c / R4d, `erase_uce.py`, FINDING75, FINDING77)

**Date:** 2026-10-26 · **Scope:** everything UCE · **Mode:** read-only analysis + CPU-only
verification + non-GPU repairs. No GPU work launched; no `run_*.sh` executed; no process killed.
The freeze in `FREEZE_ACTIVE` was respected throughout.

**Upstream pin:** `unified-concept-editing@7c724d9d7d19190e9bd33fe62fb656f1df87cd8e`,
`trainscripts/uce_sd_erase.py`.

**New evidence produced here (CPU only, no CUDA):** all numbers below tagged `[CPU]` were computed
by loading the pinned SD-1.4 text encoder and UNet `attn2` K/V weights from
`hf_home/.../snapshots/133a221b8aa7292a167afc5127cb63fb5005638b` on CPU, plus the already-saved
`outputs/r4c/*.safetensors` and `adapters/R4C_port_church_lam0.5/unet.pt`. The CPU reproduction of
upstream's algorithm was validated against upstream's own saved checkpoint (agreement to 4
significant figures — see D-104), so it is a faithful stand-in.

Classification key: **(1) CONFIRMED DEFECT** · **(2) SUSPECTED** · **(3) VALID** ·
**(4) HISTORICAL RESULT, interpretation must change** · **(5) REPAIR PERFORMED**

---

## 0. THE HEADLINE — upstream did edit the model; it is behaviourally inert for a *structural* reason

The coordinator asked, correctly, whether upstream's apparent null result means we are invoking it
wrong, and asked for an upstream-vs-BASE measurement. That measurement is below.

### D-100 — "upstream wrote an unedited model" is **REFUTED**. (3) VALID behaviour / measurement

`[CPU]` `outputs/r4c/upstream_church_lamb{0.1,0.5}.safetensors` against the pinned base UNet
(`diffusion_pytorch_model.fp16.safetensors` upcast to fp32 — exactly the weights the run saw), all
32 `attn2` K/V matrices:

| checkpoint | vs BASE per-matrix max | vs BASE per-matrix mean | bitwise identical to base |
|---|---:|---:|---:|
| `upstream_church_lamb0.1` | **9.794829e-02** | **5.362011e-02** | 0 / 32 |
| `upstream_church_lamb0.5` | **9.789869e-02** | **5.359337e-02** | 0 / 32 |

(Against the fp32 *main* weights instead: 9.794942e-02 / 5.362090e-02 and 9.789978e-02 /
5.359417e-02 — i.e. the fp16-variant choice is irrelevant at the 6th digit.)

Upstream changed every one of the 32 matrices, by **5.36 % of the weight norm on average**.
`UCE()` writes into a `copy.deepcopy` (`uce_sd_erase.py:21-22, 82`) and saves *those* tensors
(`:86-88`), so the `.safetensors` captures the edited weights correctly. **No "upstream" number in
R4b/R4c/R4d is measuring the base model.**

### D-101 — the edit is large in *activation* space too, so inertness is not a small-edit artifact. (3) VALID

`[CPU]` Relative change in the cross-attention K/V **output** that the UNet actually reads, for the
eval prompt `"a photo of a church"` (`attention_mask.sum()=7`), averaged over the 32 matrices:

| arm | ‖ΔKV‖/‖KV‖ over all 77 positions | at the `church` token (pos 5) | at EOS (pos 6) | mean over padding (pos 7–76) |
|---|---:|---:|---:|---:|
| upstream (1 token) | **0.4335** | 1.1612 | 0.5993 | 0.4236 |
| our port (76 rows) | **0.5872** | 1.1542 | 0.6782 | 0.5919 |

Two consequences, and both contradict explanations currently in our documents:

* **Upstream's single-token edit does *not* leave the trailing positions untouched.** It changes
  padding-position K/V output by 42 %. The rank-1 weight update acts on *every* position's
  embedding, not only on the selected one. `erase_uce.py:50-57`'s stated mechanism ("Editing one
  position therefore leaves the concept reachable") is **the wrong explanation** for why single-token
  UCE failed in our hands, even though the *observation* it reports is reproduced here (D-104).
* The two arms differ by only **1.35×** in activation space (0.43 vs 0.59) yet by 15/15 vs 0/15 in
  CLIP label. The behavioural gap is **not** proportional to the numerical gap, which is a reason to
  distrust every weight-norm-based inference about behaviour in FINDING75 and FINDING77.

### D-102 — upstream's inertness **cannot** be fixed by `erase_scale`, `lamb`, or the guide string. (1) CONFIRMED (new) — and it explains the null result

`[CPU]` Upstream's exact algorithm, re-run on CPU across settings; `dW/W` and `dKV/KV` are the
means from D-100/D-101:

| upstream configuration | dW/W | dKV/KV |
|---|---:|---:|
| `erase_scale=1`, `guide=''`, 1 prompt, 10 preserve  (= R4c) | 0.0568 | 0.4354 |
| `erase_scale=1`, `guide=''`, 1 prompt, 0 preserve  (= R4d) | 0.0536 | 0.4335 |
| `erase_scale=1`, `guide=' '`, 1 prompt, 0 preserve | 0.0536 | 0.4335 |
| `erase_scale=2`, `guide=''`, 1 prompt, 0 preserve (README's object example) | 0.0536 | 0.4336 |
| `erase_scale=5`, `guide=''`, 1 prompt, 0 preserve | 0.0536 | 0.4337 |
| `erase_scale=10`, `guide=''`, 1 prompt, 0 preserve | 0.0536 | 0.4338 |
| `erase_scale=1`, `guide=''`, **`expand_prompts=true`** (6 prompts), 0 preserve | **0.0925** | **0.4745** |
| `erase_scale=2`, `guide=''`, `expand_prompts=true`, 0 preserve | 0.0934 | 0.4757 |
| *(our port, λ=0.5, preserve_scale 1.0 — measured checkpoint)* | *0.2164* | *0.5872* |

**`erase_scale` is completely inert: 1 → 10 moves the edit from 0.0536 to 0.0536** (4 digits
unchanged; `dKV/KV` moves 0.4335 → 0.4338). `erase_scale` multiplies
*both* the numerator and the denominator erase terms (`uce_sd_erase.py:70-71`), so with a single
erase concept and `lamb` already negligible it almost cancels. `ERRATUM_uce_lambda_scale.md`
`lamb` ∈ {0.1, 0.5} is negligible against the spectrum of `cᵀc` (`[CPU]` §9: max eigenvalue
**2.217e+03** for upstream's 11-row system), which is why upstream's λ=0.1 and λ=0.5 checkpoints
differ from each other by only **~5e-5 relative** (D-100: 5.362e-02 vs 5.359e-02). The same
cancellation applies to `erase_scale`.

**The only upstream lever that does anything is `expand_prompts`** — and it is exactly the one a
harness that imports `UCE()` directly cannot reach (D-103). It raises the edit 1.7× (0.0536 →
0.0925), still **2.3× below our port's 0.2164**. So even upstream-as-the-authors-would-invoke-it is
a substantially smaller edit than our port's, and the gap is not closable by any argument `UCE()`
accepts.

**`guide=''` and `guide=' '` are the identical edit** (both tokenize to `[BOS, EOS]`, so
`attention_mask.sum()-2 = 0` selects the BOS hidden state either way). This closes the
coordinator's guide-string question: there is nothing to choose between them, and nothing was lost
by the shim.

### D-103 — our invocation is **API-faithful but not a configuration the authors demonstrate**. (2) SUSPECTED → the honest framing

Checked against `uce_sd_erase.py`'s `__main__` and the upstream README:

* `UCE()`'s full signature is `(pipe, edit_concepts, guide_concepts, preserve_concepts,
  erase_scale, preserve_scale, lamb, save_dir, exp_name)`. **Every argument is bound** by
  `r4b.load_upstream_uce` + the call sites. There are **no** `with_to_k` / `with_to_v` / `technique`
  flags in this file (those belong to the authors' older `train-scripts/train_erase*.py`, absent
  from this pin). `concept_type` is used only in `__main__`, to default the guide and to pick
  expansion templates — it never enters `UCE()`. **No argument is unbound.** (3) VALID
* The projections edited match exactly: upstream `'attn2' in name and name.endswith('to_v'|'to_k')`
  (`:17-20`) → 32; `r4c:90-91` uses the identical predicate; `erase_uce.py:105-106` uses
  `endswith("attn2.to_k"|"attn2.to_v")`, equivalent for SD UNet names. **32 = 32 = 32.** (3) VALID
* **`expand_prompts` is unreachable through `UCE()`.** Expansion happens in `__main__`
  (`:155-190`) *before* the call, so a harness that imports `UCE()` directly — R4b, R4c, R4d — can
  never exercise it. The authors' own HiDream example in the README sets `--expand_prompts 'true'`.
  **(1) CONFIRMED gap in the harness**, not in upstream.
* The README's SD examples are **art-style** erasure (`--concept_type art`, so `guide='art'`, two
  concepts at once, preserve pool supplied). Its only *object* erasure example
  (`uce_t2v_erase.py`) sets `--erase_scale 2`. The README's moderation example passes
  `--concept_type 'unsafe'`, which is **not in its own argparse `choices=['art','object']`** — i.e.
  upstream's README and argparse disagree. **We are running upstream in a regime the authors never
  demonstrate for SD: one unexpanded object noun, guide `''`, `erase_scale=1`.**

### D-104 — our own historical single-token experiment is **independently replicated**, and it was right. (4) HISTORICAL RESULT whose interpretation must change

`erase_uce.py:50-53` records that an earlier single-token version of our port "changed the weights
by just 5.7 % and did not erase at all (church stayed at CLIP acc 1.000)".

Upstream's single-token selection, run here: **5.36 % weight change, 15/15 church retained.**

These are the **same measurement**, obtained twice, by two independent code paths, ~5 days apart.
So:

* R4c/R4d's "upstream erases nothing" is **not** an invocation bug and not a fluke. It is the
  reproducible behaviour of single-token UCE on an ImageNette object noun under our closed-set CLIP
  eval, and we had already measured it ourselves before upstream was ever executed.
* **Our port's `last..76` selection is therefore a deliberate, empirically motivated method
  deviation, not a transcription error.** Every document that frames it as our port being "not
  upstream UCE" in a pejorative sense is mis-framing it. What is true is narrower and must be said
  that way: *our port implements a modified UCE whose erase subspace is 76-dimensional rather than
  rank-1, and that modification is what makes it erase at all on this task.*
* It does **not** follow that our port is better. See D-105.

### D-105 — the port's 0/15 is as consistent with over-editing as with erasure. (1) CONFIRMED mis-reading risk

`outputs/r4c_uce_lamb0.1.json`: our port's 15 predictions fall across **five** different wrong
classes — `garbage_truck`, `golf_ball`, `monet`, `parachute`, `van_gogh`. At λ=0.5:
`garbage_truck`, `parachute`, `van_gogh`. Upstream's 15 are `church` (λ=0.1) and 14×`church` +
1×`cassette_player` (λ=0.5). Our port's mean weight change is **0.2164** versus upstream's
**0.0536** — **4.0×** larger `[CPU]`. Scattered off-target labels plus a 4× larger edit is the
signature of collateral damage, not of targeted erasure. **No retain/utility axis was measured in
R4c or R4d**, which is the one axis that would separate the two readings, and PREREG_H38 §6
requires it. The R4c/R4d generation comparison therefore cannot support "our port erases and
upstream does not" as a quality claim in either direction.

### D-107 — `venv_uce` violates **every pin** in upstream's `requirements.txt`. (1) CONFIRMED, (3) consequence benign for what ran

`venv_uce` is a byte-identical copy of `venv_t2i` (`diff -rq` on the `diffusers` trees: empty).
Installed vs upstream's `requirements.txt`:

| package | upstream requires | installed | |
|---|---|---|---|
| diffusers | `>=0.34.0` | **0.31.0** | violated |
| torch | `>=2.6` | **2.4.1+cu124** | violated |
| transformers | `>=4.48.0` | **4.46.3** | violated |
| safetensors | `==0.5.3` | **0.8.0** | violated |

So R4b/R4c/R4d executed upstream's code on a stack upstream explicitly does not support. **This is
documented and deliberate**, not an undisclosed gap: `setup_upstream_envs.sh:24-25` records
"UCE's trainscripts import cleanly on the current stack (12 modules, 0 failures), so its
requirements.txt pin of diffusers>=0.34 is NOT needed", and the first attempt's failure
(`torch.accelerator` absent in 2.4.1) is preserved in the same file. **But "imports cleanly" is a
weaker test than "computes the same numbers", and the manifest should not have accepted it as
equivalent.**

**(3) For the one thing that could have invalidated columns A and B, I verified it directly.**
Upstream reads embeddings via `pipe.encode_prompt(...)`; our port via `te(ti.input_ids)[0]`. In the
installed diffusers 0.31,
`pipeline_stable_diffusion.py:86-93` does `attention_mask = None` unless
`text_encoder.config.use_attention_mask` exists — and `[CPU]`
`hasattr(CLIPTextConfig, "use_attention_mask")` is **False** for the pinned SD-1.4 text encoder —
then `prompt_embeds = self.text_encoder(ids, attention_mask=None)[0]`. **That is bit-identical in
form to our port's call.** So on the stack that ran, upstream's and our port's embedding extraction
are the *same function*, and column B is comparing solvers rather than embedding functions.
Unverified for diffusers ≥ 0.34, which is not on disk.

---

## 1. FINDING77 — every numeric ratio recomputed from the raw JSON

Source of truth: `outputs/r4b_uce_lamb{0.1,0.5,1000.0}.json`, read directly. Full precision:

| λ | A max | A mean | B max | B mean | C(up) max | C(up) mean | C(port) max | C(port) mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 3.841574e-01 | 2.754587e-01 | 1.507484e-03 | 7.934684e-04 | 1.045193e-03 | 5.698089e-04 | 6.260683e-02 | 3.929265e-02 |
| 0.5 | 3.219044e-01 | 2.173323e-01 | 3.046354e-04 | 1.620325e-04 | 2.143452e-04 | 1.174371e-04 | 1.355422e-02 | 8.575911e-03 |
| 1000 | 1.084060e-01 | 7.179229e-02 | 1.326407e-07 | 1.273269e-07 | 3.721492e-07 | 3.391255e-07 | 8.246282e-06 | 5.089177e-06 |

**§1's table is correct.** All twelve displayed values round-trip exactly. (Undocumented: the two
C columns show **max** only; their means are ~1.8× smaller. Repaired.)

Every ratio and derived claim in the prose:

| # | claim (FINDING77) | recomputed | verdict |
|---|---|---|---|
| R1 | "~60,000× worse than upstream's selection at the same λ" (§2) | 6.260683e-02 / 1.045193e-03 = **59.90** | **(1) WRONG by 1000×. It is ~60×.** |
| R2 | "about 60× worse than the full-pipeline difference at λ=1000" (§2) | 6.260683e-02 / 1.084060e-01 = **0.5775** | **(1) WRONG and inverted. It is 0.58× — i.e. 1.73× *smaller*, not 60× worse.** |
| R3 | "wrong by ~6 orders of magnitude" (§2, re FINDING75's conditioning claim) | FINDING75's claim was at **λ=0.1**. Within λ=0.1: 1.5896e-01 / 1.045193e-03 = **152× = 2.2 orders**. The "6 orders" silently uses the λ=1000 value (1.5896e-01/3.721492e-07 = 4.27e5 = 5.6 orders). | **(1) WRONG — cross-λ. See §3.** |
| R4 | "an effect ~10⁶ times smaller than the real one at λ=1000" (§2) | 1.084060e-01 / 1.326407e-07 = **8.173e5** | (1) overstated ~1.2×; should read **~8×10⁵** (as §2 itself says four lines later) |
| R5 | "a factor of ~8×10⁵" (§2) | 8.173e5 | **(3) correct** |
| R6 | "reproduces upstream's matrices to 1.3e-07 at λ=1000" | B max 1.326407e-07 | **(3) correct** (it is the *max*; mean 1.273e-07) |
| R7 | "~1.5e-03 at λ=0.1" | B max 1.507484e-03 | **(3) correct** |
| R8 | "float32 is accurate to 3.7e-07 under upstream's selection" (§2) | true **only at λ=1000**; at λ=0.5 it is 2.143e-04 and at λ=0.1 **1.045e-03** — 2800× worse | **(1) WRONG by omission of λ** |
| R9 | "diverges by 11–38 % … at every λ tested" (§3) | per-matrix max 10.84 % … 38.42 %; per-matrix **mean** 7.18 % … 27.55 % | (1) minor: the range is **10.8–38.4 %** on max, and the mixing of max/mean is undeclared |
| R10 | "38 % at 0.1, 11 % at 1000" (§3) | 38.42 %, 10.84 % | **(3) correct to rounding** |
| R11 | "27.6 % mean / 38.4 % max at λ=0.1" (§3) | 27.546 % / 38.416 % | **(3) correct** |
| R12 | "the arms built at λ=0.1 — **13 of our 22** UCE arms" (§3) | `outputs/erase_*.json` with `study=="t2i_uce_reimpl"`: 24 records, of which 2 are this week's R4C diagnostics → **22 historical arms**, of which **9** used λ=0.1. **13** is the number of surviving `adapters/uce_*` **directories** — a different quantity. | **(1) WRONG. 9 of 22, not 13 of 22.** Same error in FINDING75 §6 row 2. |
| R13 | "ten distinct λ values across 22 arms" (FINDING75 §6 row 2) | {0.1:9, 0.5:1, 1.0:5, 10:1, 30:1, 100:1, 300:1, 1000:1, 10000:1, 50000:1} = **10 distinct**, 22 arms | **(3) correct** |
| R14 | "**one** embedding at `attention_mask.sum()-2`" / "positions `…-2 … 76`" (§2 table) | `[CPU]` `"church"` → `attention_mask.sum()=3`; upstream index **1**; port rows **1…76 = 76 rows** | **(3) correct** |
| R15 | "the final content token **plus ~75 padding positions**" (§2) | 76 rows = 1 content token + **1 EOS** + **74 padding** | (1) minor mis-description; EOS is not padding, and the count is 74 |
| R16 | citation "`erase_uce.py:63-65`" (§2 table, and `r4b…py:12`) | the selection is at **`erase_uce.py:67-68`** | (1) stale line reference |
| R17 | citation "`erase_uce.py:116-117`" (§6) | correct — `num` at 116, `den` at 117 | **(3) correct** |
| R18 | citation "`r4b_uce_upstream_vs_port.py:172-177`" (§6) | correct | **(3) correct** |

---

## 2. "A − B isolates embedding selection" — what *is* and *is not* established

**The subtraction is not a tensor difference, and the document never says so.** With
X = `solve(port_selection)`, Y = `solve(upstream_selection)`, Z = upstream's reloaded checkpoint:

* A = ‖X − Z‖ / ‖Z‖   (`r4b…py:212`)
* B = ‖Y − Z‖ / ‖Z‖   (`r4b…py:197`)

‖X − Y‖ ≠ ‖X − Z‖ − ‖Y − Z‖. However — and this is the part the document gets right by luck rather
than by argument — **both ratios share the same denominator ‖Z‖ and the same reference Z**, so the
triangle inequality gives a genuine two-sided bound:

> ‖X − Y‖ / ‖Z‖ ∈ [A − B, A + B]

At λ=1000 that is **[0.108406, 0.108406]** (B = 1.3e-07 is 6 decades below A); at λ=0.1,
**[0.38265, 0.38567]**. So:

**(3) ESTABLISHED.** Within R4b's own configuration, embedding selection accounts for the whole of
the A column to within ±0.4 %. The conclusion survives — **as a norm bound, not as a subtraction**,
and the document must say "bounded by the triangle inequality", not "A − B is".

**(1) NOT ESTABLISHED, and currently claimed.** Three things:

1. **A and B are not two additive "effects".** §2's "A = 1.08e-01 against B = 1.33e-07 — a factor
   of ~8×10⁵" is presented as a comparison of two mechanisms' contributions. It is a ratio of two
   distances to a common point. It licenses "selection dominates"; it does **not** license any
   statement of the form "order contributes 1/8×10⁵ of the total".
2. **R4b's "port" column is not `erase_uce.py`.** It differs in three ways, all now established:
   upstream's accumulation order (`r4b…py:172-177`) not the port's (`erase_uce.py:116-117`);
   `preserve_scale = 1.0`, where the port's historical default and **every** historical SD-1.5 arm
   used **0.1**; and in-memory fp32, where the port serializes fp16. §6 names only the first.
3. **The λ=1000 row matches no historical arm.** FINDING77 §4 notes the SD-1.5 mismatch but misses
   that `uce_sd15_church_lam1000` also used **`preserve_scale = 0.1`**, against R4b's 1.0.

### D-106 — accumulation order is now **measured**, and FINDING77 §6's "unmeasured" is out of date. (4) HISTORICAL RESULT whose interpretation must change

FINDING77 §6 says the accumulation-order question "remains open" and "unmeasured". R4c has since
closed it, by a comparison the document does not perform:

| λ | R4b column A (upstream order, fp32, in memory) | R4c checkpoint comparison (**real `erase_uce.py` path**, λ·W last, fp16-serialized) | difference |
|---:|---:|---:|---:|
| 0.1 | max 3.841574e-01 / mean 2.754587e-01 | max 3.842716e-01 / mean 2.755211e-01 | **1.14e-04 / 6.2e-05** |
| 0.5 | max 3.219044e-01 / mean 2.173323e-01 | max 3.219402e-01 / mean 2.173666e-01 | **3.6e-05 / 3.4e-05** |

Switching from upstream's accumulation order to `erase_uce.py`'s own, *and* round-tripping through
fp16, moves the result by **≤ 1.2e-04 relative** — entirely inside the fp16 serialization budget
measured in D-140 (2.07e-04). **On the GPU path that produced every one of our artifacts,
accumulation order is bounded at < 1.2e-04 and is negligible against the 0.22–0.28 selection
effect** — and the measurement that shows this already existed in our artifacts, uncompared.

**Caveat, added after §9b: this bound is not hardware-independent.** The same comparison on CPU
gives 5.6e-02 (D-160). FINDING77 §6's "unmeasured" is still wrong — it *is* measured for our runs —
but the correct statement is "negligible on the GPU path our arms ran on", not "refuted in
general".

---

## 3. The "six orders of magnitude" cross-λ inference — **NOT supported**

FINDING77 §2: *"`FINDING75` inferred 'one digit survives' from a condition number; measured, that
was wrong by ~6 orders of magnitude."*

**(1) CONFIRMED DEFECT.** FINDING75's conditioning argument was made **at λ = 0.1**, about **our
port's** system (it cites `cᵀc` max eigenvalue 85,850 from `ERRATUM_uce_lambda_scale.md`, which was
measured on our port's embeddings), and predicted a float32 error of order the 1.59e-01 it had
measured. The rebuttal number 3.721492e-07 is **λ=1000, upstream's selection**. Two variables
changed at once.

Matched comparisons:

| comparison (all from `r4b_uce_lamb*.json`) | ratio | orders |
|---|---:|---:|
| FINDING75's 1.5896e-01 vs C(**upstream** sel, **λ=0.1**) 1.045193e-03 | 152× | **2.2** |
| FINDING75's 1.5896e-01 vs C(**port** sel, **λ=0.1**) 6.260683e-02 | **2.54×** | **0.4** |
| FINDING75's 1.5896e-01 vs C(upstream sel, λ=1000) 3.721492e-07 | 4.27e5 | 5.6 ("~6") |

**This reverses FINDING77's dismissal of FINDING75 §2.** Against the system FINDING75 was actually
talking about — our port's selection at our historical λ — the measured float32 error is
**6.26e-02**, within a factor of **2.5** of FINDING75's 1.59e-01. FINDING75's conditioning argument
was *approximately right about our port* and wrong only about upstream's rank-1 system, and it was
**withdrawn for the wrong reason**. What was genuinely invalid in FINDING75 §2 is different and
narrower: a condition number bounds error propagation rather than measuring it, and the 1.59e-01 it
was "explaining" was itself a void char-split artifact. Both of those stand; "6 orders of
magnitude" does not.

---

## 4. Accumulation order, batching and multiplication grouping — "λ first vs λ last" does **not** fully characterise it

Reading `uce_sd_erase.py:56-82` against `erase_uce.py:112-120`, there are **four** independent
arithmetic differences, not one:

| # | upstream | our port | class |
|---|---|---|---|
| a | `mat1 = lamb * w_old` **first**, then `+=` each term (`:61-79`) | `… + a.lam * W` **last** (`:116`) | addition order — the only one named in FINDING75/77 |
| b | **rank-1 Python loop**: one `c_i @ c_i.T` per concept, 1 erase + 10 preserve = **11 sequential `+=`** (`:66-79`) | **one fused GEMM** `c_p.T @ c_p` over all rows at once (`:117`); BLAS chooses its own internal accumulation order and may use FMA/blocking | **summation structure** — not addressed anywhere |
| c | `erase_scale * (v_i_star @ c_i.T)` — scale applied **after** the outer product (`:70`) | `a.preserve_scale * (W @ c_p.T) @ c_p` — Python's left-associativity makes this `((ps * (W @ c_p.T)) @ c_p)`, so the scale is applied to the **(out, n_p) intermediate, before** the second matmul (`:116`) | **scaling point** — not addressed anywhere |
| d | `torch.inverse(mat2.float()).to(torch_dtype)` (`:82`) | `torch.linalg.inv(den)` (`:118`) | same LAPACK path; immaterial |

**(1) CONFIRMED:** the characterisation "they differ only in summation order" (FINDING75 §2) and
"lambda first vs lambda last" (FINDING77 §6) is **incomplete**: (b) and (c) are also real, and (b)
is the larger of the two in principle, because the port's single GEMM over 76–836 rows has an
internal accumulation order that is not even specified by the source.

**(3) VALID / now moot:** D-106 bounds the *combined* effect of (a)+(b)+(c) plus fp16 serialization
at **< 1.2e-04 relative**. So the mis-characterisation is a documentation defect, not a result
defect. Matmul **association** is identical on both sides: upstream forms `v_i_star` as
`module(t_emb)` = `W c_g` then right-multiplies by `c_iᵀ`; the port forms `V_star = (W @ c_t.T).T`
then `V_star.T @ c_e` = `(W c_tᵀ) c_e` — same grouping, left to right. (3) VALID.

**Also verified (3) VALID:** `to_k`/`to_v` carry no bias in diffusers' `Attention`, so upstream's
`module(t_emb)` and the port's `W @ c.T` are the same map; upstream never mutates the live pipeline
(`:21-22` deep-copies, `:82` writes to the copy), so R4c's subsequent `del pipe` is harmless and the
port's run starts from a clean base.

---

## 5. `r4c_uce_checkpoints_and_images.py` vs its own docstring

### D-110 — advertises `--order-probe`, which does not exist. (1) CONFIRMED → (5) REPAIRED

Docstring lines 17-20: *"ACCUMULATION ORDER IS NOW MEASURED, not held fixed: … `--order-probe`
recomputes our port's own formula in upstream's order on identical inputs, so the order effect is
separated from the selection effect."* `main()`'s parser (`:38-48`) defines
`--concept --guide --lamb --erase-scale --preserve-scale --no-preserve --n-per-prompt --steps
--guidance`. **There is no `--order-probe`, and no order probe is computed anywhere in the file.**
Every artifact it produced (`r4c_uce_lamb*.json`, `r4d_uce_lamb0.5.json`) therefore lacks the
separation the docstring promises. (The separation *is* obtainable — D-106 — but by differencing
R4c against R4b, which this script neither does nor records.)

### D-111 — scores with a **different CLIP label set** from every other H number. (1) CONFIRMED → (5) REPAIRED

`r4c…py:109`: `labels = list(ge.CONCEPTS)` — the **dict keys**. The project's shared closed set is
`gen_eval.LABELS` (identical in `attack_relearn.py:83`, verified equal), paired with
`gen_eval.LAB_KEY` for the key↔label mapping. **9 of the 12 labels differ**, so the CLIP text
prompts built at `:111` are degenerate:

| r4c's prompt | the project's prompt |
|---|---|
| `a photo of a garbage_truck` | `a photo of a garbage truck` |
| `a photo of a van_gogh` | `a photo of a van gogh painting` |
| `a photo of a english_springer` | `a photo of a english springer spaniel` |
| `a photo of a monet` | `a photo of a monet painting` |
| `a photo of a cassette_player`, `chain_saw`, `french_horn`, `gas_pump`, `golf_ball` | (spaces) |

**(4) Consequence for the historical result.** `church` is index 0 and spelled identically in both
lists, so R4c/R4d's *target* accuracies are the least-affected quantity — but the competing logits
are drawn from a corrupted label set, and 4 of the 5 wrong classes our port is assigned
(`garbage_truck`, `van_gogh`, `monet`, `golf_ball`) are ones whose text prompt is malformed.
**R4c/R4d target accuracies are not comparable to any other H number in this project**, and their
`pred` fields are not interpretable at all. The 15/15-vs-0/15 *direction* is robust (it is also
visible in D-101's activation deltas and in D-104's independent replication); the numbers are not.

### D-112 — ignores `erase_uce.py`'s exit code and silently accepts a stale cached checkpoint. (1) CONFIRMED → (5) REPAIRED

`r4c…py:78-84`:
```python
if not os.path.exists(f"{port_dir}/unet.pt"):
    rc = os.system(...)
    print(f"[r4c] our port exit={rc}", flush=True)
assert os.path.exists(f"{port_dir}/unet.pt"), "our port produced no checkpoint"
```
`rc` is printed and discarded. Worse, the guard is `os.path.exists` on a path that **encodes only
`concept` and `lamb`** (`:77` → `adapters/R4C_port_{concept}_lam{lamb}`) while the run also depends
on `preserve_scale`, `guide`, and `--no-preserve`. This is not hypothetical:

### D-113 — **R4d silently reused R4c's checkpoint and clobbered R4c's upstream artifact.** (1) CONFIRMED DEFECT, highest severity in this section

Both R4c and R4d call the same script, so both use `port_dir = adapters/R4C_port_church_lam{λ}`
**and** `sd_dir = outputs/r4c` with `exp = upstream_{concept}_lamb{λ}` (`:69-70`). Filesystem
evidence:

* `adapters/R4C_port_church_lam0.5/unet.pt` — mtime **12:22**, written by the R4c run.
  `outputs/erase_R4C_port_church_lam0.5.json` records `preserve_scale: 1.0` (i.e. **with** the 10
  preserve concepts).
* `outputs/r4c_uce_lamb0.5.json` — **12:23**.
* `outputs/r4d_uce_lamb0.5.json` — **12:42**. No new port checkpoint was written.
* `outputs/r4c/upstream_church_lamb0.5.safetensors` — mtime **12:39**, i.e. **R4d's**, having
  overwritten R4c's.

Two confirmations that the overwrite happened: (i) `[CPU]` that safetensors file measures 0.0536
weight change against base, which matches my CPU reproduction of the **0-preserve** configuration
(0.0536) and not the 10-preserve one (0.0568); (ii) `[CPU]` loading it against
`adapters/R4C_port_church_lam0.5/unet.pt` reproduces **max 3.2160e-01 / mean 2.1740e-01** — the
values reported in `r4d_uce_lamb0.5.json`, not R4c's 3.2194e-01 / 2.1737e-01.

**Therefore:**
* R4d's `OUR_PORT` arm is **byte-identical to R4c's** and was built **with** 10 preserve concepts,
  while `r4d_uce_lamb0.5.json` records `"n_preserve": 0`. The artifact's own config field is wrong
  about the arm it describes.
* R4c's upstream checkpoint **no longer exists**; R4c is not reproducible from artifacts on disk.
* **`--no-preserve` was never passed to the port at all.** `erase_uce.py` exposes no flag to empty
  its `PRESERVE` pool, and the old `os.system` command line (`:79-82`) did not attempt one. So the
  flag only ever changed upstream's pool and the output filename (`:169`).
* **R4d's OUR_PORT column being identical to R4c's is tautological — it is the same file — not a
  measurement.** No R4c/R4d port-side number may be cited as evidence about the preserve pool in
  either direction, and `r4d_uce_lamb*.json`'s `"n_preserve": 0` must never stand unqualified: it
  is **false for the OUR_PORT arm**, which used 10 preserve concepts.
* **The conclusion the coordinator drew from R4d nevertheless stands, on different evidence.**
  Upstream's arm *is* recomputed in-process on every run and is unaffected by the caching bug, so
  "upstream retains 15/15 church with `preserve_concepts=[]` genuinely empty" is sound. And the
  preserve-pool hypothesis placed the effect on **upstream's** side, where `[CPU]` D-102 measures
  it directly: removing the pool moves upstream's edit from 0.0568 to 0.0536, i.e. **1.5 % of the
  0.22 gap**. The pool is inert where the hypothesis needed it to matter. **Hypothesis refuted;
  R4d is not the evidence that refutes it.**

### D-114 — saves no images. (1) CONFIRMED → (5) REPAIRED

The docstring (`:4-7`, `:13-16`) promises an image comparison and the header says "A weight
difference is not an image difference". 45 images are generated (`:118-133`) and every one is
discarded; only CLIP argmax labels survive. `gen_eval.py:132-134` already has the `--save-grid`
idiom. No panel exists, so the comparison cannot be inspected by a human — which, given D-111, is
exactly what was needed.

### D-115 — further defects found (not in the brief). (1) CONFIRMED

1. **`--guide` is never passed to the port.** `:79-82` passes only `--concept --lam
   --preserve-scale --label --out`. `erase_uce.py`'s guide is its `--target` (default `""`). The
   defaults coincide at `''`, so R4c/R4d are unaffected, but **any non-default `--guide` silently
   compares upstream erasing toward *g* against our port erasing toward `""`.** → (5) REPAIRED.
2. **`--erase-scale` has no port counterpart.** `erase_uce.py` has no such coefficient
   (implicitly 1.0). `:72` passes it to upstream only. At 1.0 the two are algebraically identical;
   at anything else the script compares **two different equations** and reports the difference as a
   fidelity gap. → (5) REPAIRED (hard refusal).
3. **`--no-preserve` is one-sided.** It empties upstream's preserve pool (`:63`) but
   `erase_uce.py` always uses its hard-coded 10-concept `PRESERVE` with no flag to disable it. R4d
   is thus structurally a 0-preserve-upstream vs 10-preserve-port comparison, labelled as "upstream
   as intended". → (5) REPAIRED (the asymmetry is now recorded in the artifact and printed).
4. **No retain / utility axis.** `prompts = list(ge.CONCEPTS[a.concept])` (`:114`) — only the
   target's 3 DEV prompts. PREREG_H38 §6 requires forget **and** retain **and** utility in the same
   JSON, and `gen_eval.py` enforces this for every other arm. Without it, D-105 cannot be settled.
   → (5) REPAIRED (`--retain-concepts`, default off, opt-in).
5. **DEV prompts only, n=15 per arm.** `FINDING65` established that erasure does not survive
   held-out prompts; R4c measures only DEV. Recorded as a caveat, unrepaired (needs GPU).

---

## 6. Base model / revision / variant pinning, and one-sided env overrides

### D-120 — `BASE_MODEL` / `BASE_REV` / `BASE_VARIANT` **can override one side only**. (1) CONFIRMED → (5) REPAIRED

| file | model | revision | variant | env-overridable |
|---|---|---|---|---|
| `erase_uce.py:30-32, 96-98` | `BASE_MODEL` env | `BASE_REV` env, else `MODEL_PIN.txt` | `BASE_VARIANT` env | **yes** |
| `gen_eval.py:21-23, 57-59` | `BASE_MODEL` env | `BASE_REV` env, else pin | `BASE_VARIANT` env | **yes** |
| `r4_uce_official_vs_port.py:60-61` | hard-coded `CompVis/stable-diffusion-v1-4` | pin | hard-coded `"fp16"` | **no** |
| `r4b_uce_upstream_vs_port.py:104-106` | `--model` flag | pin (**always**) | hard-coded `"fp16"` | **no** |
| `r4c_uce_checkpoints_and_images.py:55-57` | hard-coded `CompVis/stable-diffusion-v1-4` | pin | hard-coded `"fp16"` | **no** |

**The failure is live, not theoretical.** `r4c` invokes `erase_uce.py` through `os.system`
(`:79-82`), which **inherits the environment**. With `BASE_MODEL=runwayml/stable-diffusion-v1-5`
exported — which `run_uce_crossmodel.sh` and `run_meta_crossmodel.sh` do — R4c would edit **SD-1.5**
on the port side and compare it against **SD-1.4** on the upstream side, generate from SD-1.4, and
report the difference as a UCE fidelity gap, with nothing in the artifact revealing it. The R4c/R4d
artifacts record `"revision"` only indirectly and never record `BASE_MODEL`.

Secondary: `r4b:105` pairs `a.model` with `revision=pin["revision"]` unconditionally, so
`--model <anything-else>` requests SD-1.4's commit hash from a different repo.

**(3) VALID for the runs on disk:** `run_r4{b,c,d}*.sh` export only `HF_HOME`/cache variables and
no `BASE_*`, so every artifact in `outputs/` was produced on the pinned SD-1.4 at
`133a221b8aa7292a167afc5127cb63fb5005638b`. The defect is a latent trap, now closed.

---

## 7. float64 handling — **not** a clean high-precision reference, and "error bound" is unjustified

### D-130 — R4b's float64 column inherits float32 products. (1) CONFIRMED DEFECT

`r4b…py:191-192`:
```python
Wn64 = solve(c_e.double(), (c_g @ W.T).double(), c_p.double(), (c_p @ W.T).double(),
             W.double(), a.lamb, a.erase_scale, a.preserve_scale, torch.float64)
```
`c_g @ W.T` and `c_p @ W.T` are **float32 matmuls** (both operands are fp32) whose results are then
cast to float64. These are precisely the `v_i_star` guide/preserve outputs that populate the
numerator. So the "float64 reference" shares the float32 rounding of every numerator product with
the thing it is supposed to measure — the errors are **correlated**, and column C under-reports.
The correct form casts first and multiplies in float64: `c_g.double() @ W.double().T`.

Casting the **embeddings** to double is correct and must stay — the text encoder ran in fp32 and
both solves must receive identical inputs. The defect is only in the two products.

`[CPU]` **Quantified: the mixed-precision reference is within 3.48e-08 of a true float64 solve**
(§9b), i.e. four decades below the column C values it supports. **The defect is real in the code
and immaterial in the published numbers.** Fix it (P4); do not revise column C.

### D-131 — "error bound" is the wrong word, in both directions. (1) CONFIRMED terminology defect

`r4b…py:246-247` and FINDING77 §4 call column C a bound: *"the float64 columns **bound** each one's
own numerical error"*. A float32-vs-float64 difference is a **point estimate of the fp32 error on
one realisation**, not a bound:

* it is a **lower** estimate, because the fp64 solve carries its own (far smaller) error, and here
  also because of D-130's correlated products;
* it is not an **upper** bound on anything, because nothing was varied — one input, one guide, one
  concept, one BLAS path, one GPU. A different reduction order gives a different number.

`e_32 = ‖W₃₂ − W₆₄‖/‖W₆₄‖` is a *measured realised error*. FINDING77 §2's framing — "C measures
float32 error instead of inferring it" — is the correct framing, and is the one that should appear
in §4 as well. FINDING75 §6 row 4's complaint ("a condition number *bounds* error propagation; it
does not measure it") is right, and §4 of FINDING77 then commits the mirror-image error by calling
a measurement a bound.

---

## 8. Checkpoint serialization — fp16 vs safetensors fp32

### D-140 — serialization accounts for **~0.1 %** of the reported discrepancy, but it **dominates** column B. (1) CONFIRMED, partially exculpatory

`erase_uce.py:124` — `torch.save({k: v.half() ...})`. Upstream — `save_file(...)` of fp32
(`uce_sd_erase.py:88`). R4c compares the two directly (`:95-96`), so fp16 quantization **is**
inside R4c's numbers; R4b never loads the port's checkpoint at all.

`[CPU]`, round-tripping upstream's own fp32 result through `.half().float()`, all 32 matrices:

| quantity | per-matrix max | per-matrix mean |
|---|---:|---:|
| **fp16 serialization error alone** | **2.0781e-04** | **2.0682e-04** |
| R4c port-vs-upstream (λ=0.5) | 3.2160e-01 | 2.1740e-01 |
| ratio | — | **1051×** |

**(3) VALID conclusion:** the 11–38 % divergence is **not** serialization. It is 1000× too large.

**(1) But two claims must change:**

1. **Column B is below our own serialization floor.** "Our solver reproduces upstream to
   **1.3e-07**" (λ=1000) is 1560× below the **2.07e-04** fp16 floor of every checkpoint we have ever
   shipped. The agreement is real as a statement about the *arithmetic* and **meaningless as a
   statement about any checkpoint**. Likewise at λ=0.5: B = 3.05e-04 is the same size as the
   serialization error, so B is not separable from it there either.
2. **R4b's docstring claim is false for the port.** `r4b…py:27`: *"reloads saved checkpoints so
   serialization precision is included."* Only **upstream's** is reloaded (`:125-126`); the port
   column is in-memory fp32 and never touches `unet.pt`. FINDING77 §1's parenthetical is honest
   about upstream but silent about the port. → (5) REPAIRED (docstring).

**(2) SUSPECTED, flagged for the project (not my file):** `erase_uce.py:124` saves the **entire**
1.7 GB UNet in fp16 to propagate a 32-matrix edit. `[CPU]` verified **0 of 654** non-edited keys
differ from base, so nothing is lost — but every downstream arm then reads an fp16 UNet, which puts
a 2.07e-04 floor under every UCE weight-space comparison in the project. Changing this would change
historical numerical behaviour and is **out of my scope** by instruction.

---

## 9. guide / erase_scale consistency, and silently-compared unsupported settings

### D-150 — unsupported `erase_scale` was silently compared, and the headline artifact was overwritten. (1) CONFIRMED (4) HISTORICAL

`erase_uce.py` has **no `erase_scale`**. `r4_uce_official_vs_port.py:93-95` applies `a.erase_scale`
to the "official" form and `:99-100` omits it from "ours". At `erase_scale = 2.0` the two sides are
**different equations**, and the 5.3985e-02 difference is reported as a numerical-equivalence
failure (`numerically_equivalent_at_this_config: false`). FINDING75 §1 row 3 lists it as a
"divergence probe"; FINDING75 §4 correctly notes the omission is only "harmless *algebraically*" at
1.0. It should never have been run as a comparison at all.

**And the artifact FINDING75 cites for its headline row does not contain that row.**
`r4_uce_official_vs_port.py:133` writes a **fixed** filename with no tag, so the three configs
overwrote each other. On disk now:

| file | `erase_scale` recorded | max | mean |
|---|---:|---:|---:|
| `r4_uce_official_vs_port.json` | **2.0** | 5.3985e-02 | 4.1798e-02 |
| `r4_uce_erasescale2.json` | 2.0 | 5.3985e-02 | 4.1798e-02 |
| `r4_uce_lamb0.5.json` | 1.0 | 5.3615e-03 | 3.0896e-03 |
| `r4_uce_lamb0.1.json` | 1.0 | 1.5896e-01 | 9.4748e-02 |

`r4_uce_official_vs_port.json` is a **byte-equivalent duplicate of the `erase_scale=2` run**, yet
FINDING75 §1/"Evidence" cites it for the `λ=0.5, erase_scale 1.0` row whose real home is
`r4_uce_lamb0.5.json`. All four are correctly stamped `VOID: true` with a `void_reason`, so no
further quarantine is needed — but the citation is wrong and is corrected in the rewrite.
→ (5) REPAIRED (the script now tags its output filename).

### D-151 — `preserve_scale` is consistent in form but **no R4 run matches a historical arm**. (1) CONFIRMED

The coefficient is applied to both the numerator and denominator preserve terms on both sides
(`uce_sd_erase.py:78-79` vs `erase_uce.py:116-117`) — **(3) VALID**. But `erase_uce.py:89` defaults
to **0.1**, and R4b/R4c/R4d all ran at **1.0**. Every historical SD-1.5 λ-sweep arm, including
`uce_sd15_church_lam1000`, used **0.1**. `[CPU]` this is not a small knob: the `erase_*.json`
records show `preserve_scale` 0.1 → 10.0 moving the mean weight change from 0.261 to 0.4463 at
λ=0.1. **No R4/R4b/R4c/R4d cell reproduces the configuration of any arm in our published tables.**
FINDING77 §4 must say this.

### D-152 — the guide concept. (3) VALID, with one CPU-verified simplification

`[CPU]` `""` and `" "` both tokenize to `[BOS, EOS]`, so `attention_mask.sum()-2 = 0` and upstream
selects the **BOS** hidden state in both cases; the resulting edits are identical to all printed
digits (D-102 rows 2 and 3). R4 used `--guide " "` and R4b/R4c/R4d use `""`; this difference is
**immaterial**. Our port's `--target ""` (`erase_uce.py:82`) takes the same prompt through its
`paired` path at the erase prompt's own positions (`:70-74`), which is the correct
position-alignment and matches `r4b`'s `pair_with` handling (`:148-167`). **(3) VALID.**

---

## 9b. CPU reference solve — the conditioning numbers, and why the λ-order framing is a red herring

`[CPU]` Pinned SD-1.4 text encoder + all 32 `attn2` K/V matrices, true float64 reference (every
product in float64). Concept `church`, guide `''`, the port's 10-concept `PRESERVE`,
`preserve_scale = 1.0`.

### Structure of the two systems

| selection | rows in `c` | rank of `cᵀc` | max eigenvalue of `cᵀc` | cond(`cᵀc`+λI) λ=0.1 | λ=0.5 | λ=1000 |
|---|---:|---:|---:|---:|---:|---:|
| upstream, 1 token | **11** (1 erase + 10 preserve) | 11 | **2.2171e+03** | 2.2172e+04 | 4.4353e+03 | 3.2171e+00 |
| our port, `last..76` | **826** (76 erase + 750 preserve) | **136** | **3.2420e+05** | **3.2420e+06** | 6.4841e+05 | 3.2520e+02 |

`"church"` has `attention_mask.sum()=3`, so the port's 76 rows are **1 content token + 1 EOS + 74
padding** (R15). The ten preserve prompts contribute 750 rows, each from its own `last`.

**FINDING77 §2's conditioning claim is directionally right and its magnitude is now measured:**
stacking the trailing positions raises max(`cᵀc`) by **146×** and cond at λ=0.1 by **146×**
(2.22e4 → 3.24e6), which tracks the **59.9×** ratio in realised fp32 error (R1). It also leaves the
system **rank-136 in 768 dimensions** — still massively rank-deficient, so λ is regularising 632
null directions in both cases, not conditioning a full-rank system.

**(2) SUSPECTED — needs reconciliation, not my file.** `ERRATUM_uce_lambda_scale.md:41` records
max eigenvalue of `cᵀc` = **85,850**. My measurement of the port's own stack gives **3.2420e+05**,
3.8× larger; FINDING75 §2's derived `cond ≈ (85850+λ)/λ = 858,510 at λ=0.1` is therefore an
**under**estimate of the measured 3.2420e+06, by 3.8×. The two may be measuring different sets
(erase-only vs erase+preserve, or a different concept). Flagged for the ERRATUM's owner.

### The four arithmetic differences, measured (per-matrix means over 32 matrices)

| λ | λ-last vs λ-first, **upstream sel** | λ-last vs λ-first, **port sel** | rank-1 loop vs fused GEMM | realised fp32 err, upstream sel | realised fp32 err, port sel | R4b's mixed "fp64" vs **true** fp64 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 8.8682e-04 | **5.6343e-02** | 9.2925e-04 | 6.7639e-04 | **4.6136e-02** | 3.4812e-08 |
| 0.5 | 1.2768e-04 | 9.4820e-03 | 1.7778e-04 | 1.2799e-04 | 9.6217e-03 | 3.4789e-08 |
| 1000 | 7.3526e-08 | 5.4454e-06 | 1.2676e-07 | 2.8080e-07 | 5.8536e-06 | 1.3626e-08 |

Four readings, all of which change what the documents say:

1. **Under our port's selection at λ=0.1, *every* O(ε) implementation detail moves the result by
   ~5 % of the weight norm.** Reordering the λ term: 5.63e-02. Dropping to float64: 4.61e-02.
   These are the *same size*, and both are what cond = 3.24e6 against fp32 ε ≈ 1.2e-07 predicts
   (3.24e6 × 1.2e-07 ≈ 0.39 worst case). **"λ first vs λ last" is not a distinguished mechanism —
   it is one draw from a family of equally-sized rounding perturbations.** Both FINDING75's
   "they differ only in summation order" and FINDING77 §6's framing of it as *the* open question are
   therefore mis-framed, independently of their arithmetic.
2. **The rank-1 loop vs the fused GEMM is the same size as the λ-order effect** (9.29e-04 vs
   8.87e-04 at λ=0.1, upstream selection), confirming §4: naming only the λ position is incomplete.
3. **Column C of R4b is independently corroborated.** My CPU fp32 errors (6.76e-04 upstream,
   4.61e-02 port at λ=0.1) sit within ~20 % of R4b's GPU means (5.698e-04, 3.929e-02) — two
   different machines, two implementations. **Column C's numbers stand.**
4. **D-130's code defect is numerically immaterial.** R4b's mixed-precision "float64" is within
   **3.48e-08** of a true float64 solve — four decades below column C(port) and 1.5 decades below
   column C(upstream). So D-130 must be **fixed** (it is wrong as written and would matter at higher
   precision or larger `n`), but **no published column C value needs revising**. Classified (1)
   CONFIRMED code defect with a (3) VALID result.

### D-160 — the CPU order effect (5.6e-02) and the GPU R4b-vs-R4c gap (6.2e-05) are in tension. (2) SUSPECTED

D-106 infers from R4b-vs-R4c that accumulation order contributes **< 1.2e-04**. §9b measures it on
CPU at **5.63e-02** under the same selection and λ. These differ by ~500×. Both are real
measurements; they cannot both characterise "the" order effect, which is the point: the realised
perturbation of an ill-conditioned fp32 solve is **BLAS- and hardware-dependent**, so neither
number generalises. Note also that my CPU probe conflates the λ position with the port's double
transpose (`V = (W @ c_t.T).T` then `V.T @ c_e`), which may change the cuBLAS/MKL path by itself.

**Consequence: D-106 is downgraded from "order is refuted" to "order is bounded at < 1.2e-04 *on
the GPU path that produced our artifacts*".** That is still the operationally relevant bound — it
is the configuration every historical arm ran in — but it is not a hardware-independent claim, and
FINDING75/77 must not state it as one. P2 and P6's opt-in order probes exist to settle this on GPU
when the freeze lifts.

---

## 10. Summary table

| id | item | class |
|---|---|---|
| D-100 | upstream wrote an *edited* model: 5.36 % mean vs base, 0/32 identical | (3) refutes "inert write" |
| D-101 | upstream's edit changes K/V output by 43 %, incl. 42 % on padding | (3) refutes the "one position" mechanism in `erase_uce.py:50-57` |
| D-102 | `erase_scale`, `lamb`, guide string are all **inert** for upstream here | (1) new, explains the null |
| D-103 | invocation API-faithful; `expand_prompts` unreachable via `UCE()`; regime undemonstrated by the authors | (1)/(3) |
| D-104 | upstream's null replicates our own earlier single-token result (5.36 % / 15-15 vs 5.7 % / 1.000) | (4) |
| D-105 | port's 0/15 scatters over 5 classes at 4× the edit size; no retain axis | (1) mis-reading risk |
| R1–R18 | FINDING77 ratios: **6 wrong** (R1 R2 R3 R8 R9 R12), 2 citation defects (R15 R16) | (1) |
| §2 | "A − B isolates selection" → valid only as a triangle-inequality **bound**; R4b's port column is not `erase_uce.py` | (1)+(3) |
| D-106 | accumulation order **bounded < 1.2e-04 on our GPU path** by R4b-vs-R4c; FINDING77 §6's "unmeasured" is stale | (4) |
| D-160 | CPU order effect 5.6e-02 vs GPU 6.2e-05 — the realised perturbation is BLAS-dependent; λ-order is not a distinguished mechanism | (2) |
| §9b | cond(port) = **3.24e+06** vs cond(upstream) = **2.22e+04** at λ=0.1 (146×); port `cᵀc` rank **136**, max eig **3.24e+05** (vs ERRATUM's 85,850) | (3)/(2) |
| §3 | "six orders of magnitude" is cross-λ; matched ratio is **2.5×** against our own selection | (1), reverses FINDING77's dismissal of FINDING75 §2 |
| §4 | four arithmetic differences, not one (λ-order, loop-vs-GEMM, scaling point, inverse call) | (1) doc defect |
| D-110 | `--order-probe` advertised, absent | (1) → (5) |
| D-111 | `list(ge.CONCEPTS)` instead of `LABELS`/`LAB_KEY`: 9/12 labels malformed | (1) → (5), (4) |
| D-112 | `os.system` return code discarded; cache key omits `preserve_scale`/`guide`/`no_preserve` | (1) → (5) |
| D-113 | **R4d reused R4c's port checkpoint and clobbered R4c's upstream checkpoint** | (1) → (5) |
| D-114 | no images saved | (1) → (5) |
| D-115 | `--guide` not propagated; `--erase-scale` unsupported by port; `--no-preserve` one-sided; no retain axis | (1) → (5) |
| D-120 | `BASE_MODEL`/`BASE_REV`/`BASE_VARIANT` override the port side only, through `os.system` | (1) → (5) |
| D-130 | R4b's float64 reference uses float32 `v_i_star` products — real code defect, **3.48e-08 from a true fp64 solve**, so column C stands | (1) → (5), result (3) |
| D-131 | column C is a realised error, not a "bound" | (1) → (5) |
| D-140 | fp16 serialization = 2.07e-04: 0.1 % of the divergence, but **above** column B | (1)/(3) |
| D-150 | `erase_scale=2` silently compared; `r4_uce_official_vs_port.json` is the es=2 run | (1)/(4) → (5) |
| D-151 | no R4 cell matches any historical arm (`preserve_scale` 1.0 vs 0.1) | (1) |
| D-152 | `""` vs `" "` immaterial; `paired` alignment correct | (3) |

---

## 11. Repairs performed, and how each was verified

Files I own and changed. **`erase_uce.py`'s numerical behaviour is untouched** — the only edits
there are to comments/docstrings plus one opt-in, default-off diagnostic flag, so every historical
arm still reproduces bit-for-bit.

See `§11` entries recorded inline in each file's header. Verification method for each: AST/import
check (`python -m py_compile`), `--help` inspection, and for `erase_uce.py` a diff restricted to
comment lines plus the new default-off flag.

| # | file | repair | verified by |
|---|---|---|---|
| P1 | `erase_uce.py:56` | the surviving false sentence "This matches UCE's official implementation, which takes the token range from the concept token to the end of the sequence" **deleted** and replaced with the measured truth (upstream takes **one** token; our 76-row choice is a documented deviation, D-104). The top docstring was already corrected; this one was not. | comment-only diff; `py_compile` |
| P2 | `erase_uce.py` | new `--diag-order-probe` (default **off**). When set, recomputes the same edit in upstream's accumulation order and in float64 and prints/records the deltas. Off by default → zero change to historical behaviour. | `--help`; default-path diff is comment-only |
| P3 | `r4_uce_official_vs_port.py` | output filename now carries a config tag, so the three configs cannot overwrite one another (D-150); VOID banner naming the four confirmed defects added to the docstring; `erase_scale != 1` now refuses to run. | `py_compile`; `--help` |
| P4 | `r4b_uce_upstream_vs_port.py:191-192` | float64 reference now multiplies **in** float64 (`c_g.double() @ W.double().T`), fixing D-130; the mixed-precision value is retained alongside as `float64_ref_mixed_precision_rel` so the old column stays auditable. | `py_compile`; CPU reference §9 |
| P5 | `r4b_uce_upstream_vs_port.py:27, 215-247` | docstring claim about reloading the port's checkpoint corrected (D-140.2); "bound" → "realised error" throughout (D-131); the A/B relationship recorded as a two-sided triangle-inequality bound with both endpoints emitted; `preserve_scale` mismatch against historical arms recorded in the artifact. | `py_compile`; JSON key inspection |
| P6 | `r4c_uce_checkpoints_and_images.py` | `--order-probe` **implemented** (D-110), default off; label set switched to `gen_eval.LABELS`/`LAB_KEY` (D-111); `os.system` → return-code check + `subprocess` with an explicit env, and the cache key now includes `preserve_scale`/`guide`/`no_preserve` (D-112, D-113); upstream `exp` name made config-unique so R4d cannot clobber R4c (D-113); images saved to `outputs/img_r4c_*` (D-114); `--guide` propagated to the port as `--target` and `--erase-scale != 1` refused (D-115.1-2); `--no-preserve` asymmetry recorded and printed (D-115.3); opt-in `--retain-concepts` (D-115.4); `BASE_*` env recorded and a one-sided override refused (D-120). | `py_compile`; `--help`; dry-read of the new artifact schema |
| P7 | `docs/FINDING75_*.md` | rewritten: §1 numbers re-sourced to the correct artifact, "13 of 22" → **9 of 22**, the accumulation-order claim marked **refuted** by D-106 (not merely unconfirmed), §2's conditioning argument **partially reinstated** with the matched 2.5× ratio, `erase_scale=2` row withdrawn as a category error. | read-back |
| P8 | `docs/FINDING77_*.md` | rewritten: title and §2/§3 corrected for R1 R2 R3 R8 R9 R12 R15 R16; "A − B" restated as a bound; §5/§6 replaced with the R4c/R4d results, D-100/D-101/D-102/D-104's reframing, and the D-113 artifact-integrity warning. | read-back |

### Deliberately **not** repaired

* `erase_uce.py:124`'s fp16 `torch.save` (D-140) — changing it changes historical behaviour.
* The R4c/R4d **artifacts**. `r4d_uce_lamb0.5.json`'s `"n_preserve": 0` is wrong about its own port
  arm (D-113) and `r4c_uce_lamb*.json` carry the malformed label set (D-111). Rewriting another
  agent's artifacts is out of scope; both are documented here and in the FINDING77 rewrite, and
  **R4c must be re-run** (its upstream checkpoint no longer exists) once the freeze lifts.
* `scripts/gen_eval.py`, `scripts/attack_relearn.py` — not mine; both verified **correct** and
  mutually consistent.

### Recommended next GPU work (not launched)

1. Re-run R4c and R4d with P6 applied — distinct cache keys, shared `LABELS`, retain axis, saved
   panels. The current R4c is unreproducible (D-113).
2. One upstream cell at `expand_prompts`-style 6 prompts and `erase_scale` ≥ 5 (D-102/D-103), to
   test whether upstream erases church **at all** on this task before any fidelity claim is made.
3. One cell at `preserve_scale = 0.1` to make an R4 row that matches a real historical arm (D-151).
