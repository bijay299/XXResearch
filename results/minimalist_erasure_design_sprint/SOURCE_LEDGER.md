# Source ledger — Minimalist erasure design sprint

**NO EXPERIMENTS.** Sources were **retrieved and read as text only**. No research code was
imported or executed; no model, weight, checkpoint or dataset was downloaded or loaded.

---

## 1. Primary sources

| id | source | version / pin | inspection depth | status |
|---|---|---|---|---|
| **P1** | **Minimalist Concept Erasure**, `arxiv.org/html/2507.13386v1` | **v1**, retrieved **2026-10-17** (also read for the closed gap map at `edaec97`, same version) | **targeted sections**: §3.2 / Eq. 16 (objective), §3.6 (mask, relaxation, binarization), **§3.7 verbatim in full** (gradient rule, step-wise checkpointing, prompt filtering, multi-GPU limitation), §4.1 (backbone), App. H.1 / Table 6 (hyperparameters) | explicit source claims |
| **C1** | **Author source**, `github.com/YaNgZhAnG-V5/minimalist_concept_erasure` | **`d178324ec4a48d0536a595b25fdad587b441add8`** (HEAD at clone, 2025-12-02). Code added in a **single commit `1bb7ca2`, 2025-11-24**; repo created `67231ff`, 2025-06-20 (README only) | **relevant code, read line by line** (files in §2) | explicit source inspection |
| **A1** | **DRaFT / DRaFT-K / DRaFT-LV** — Clark, Vicol, Swersky, Fleet, *Directly Fine-Tuning Diffusion Models on Differentiable Rewards*, `arXiv:2309.17400` | located 2026-10-17 | **abstract + method summary via search result text; full paper NOT read** | explicit (secondary-rendered quotes) — see §4 |
| **A2** | **AlignProp** — Prabhudesai, Goyal, Pathak, Fragkiadaki, *Aligning Text-to-Image Diffusion Models with Reward Backpropagation*, `arXiv:2310.03739` | abstract page, retrieved 2026-10-17 | **abstract only** | explicit (abstract) |
| **T1** | **`torch.utils.checkpoint`** — locally installed `torch/utils/checkpoint.py` | `/data/bijaypandey/envs/eabu-cu124/lib/python3.10/site-packages/torch/utils/checkpoint.py`; **static read, module never imported** | lines **291-297** (fork_rng + state restore), **477** and **565** (`preserve = kwargs.pop("preserve_rng_state", True)`), **429** and **1439** (docstring), **1424** (signature default), **1488-1513** (non-reentrant path) | explicit source inspection |

### Cited by identifier only — **not inspected in this sprint**

`Louizos, Welling, Kingma 2018` (hard concrete, `arXiv:1712.01312`); `Maddison et al. 2017`
(`arXiv:1611.00712`); `Jang et al. 2017` (`arXiv:1611.01144`); `Chen et al. 2016` sublinear
checkpointing (`arXiv:1604.06174`); `Williams & Peng 1990`; `Tallec & Ollivier 2017`
(`arXiv:1705.08209`); REBAR / RELAX / ARMS. Named in
[`ALGORITHM.md`](ALGORITHM.md) §4 to **locate** the design, **not** to support any claim
about their contents beyond what is standard and attributed. **No novelty search is
claimed.** `"Zhang et al. (2024d)"` — the work P1 §3.7 credits its checkpointing scheme to —
was **not resolved to a specific paper**.

---

## 2. Source files read, with what each established

All paths relative to **C1 @ `d178324`**. **Read as text; never imported or executed.**

| file | lines of interest | what it established |
|---|---|---|
| `scripts/train.py` | 68-98, 101-114, 161-165, 197-208, 211-216, 228, 259-260, 268, 274-275, 282-300, 306-333, 335-341, 344-414, 398-399, 412-413, 422-441 | the whole training loop: no-grad pre-pass storing all `z_k`; terminal latent loss; **two separate `autograd.grad` sweeps per step** (`320`, `333`); loss **replacement** not summation (`284-297`); fixed per-iteration seed (`268`); instrumentation block; hardcoded binarization threshold `0` with `TODO` |
| `src/diffsolver/models/flux.py` | 194-312, 213, 216-217, 314-365, 367-385 | the one-step map `f_k`; `max_sequence_length=512`; **resolution defaults to `default_sample_size × vae_scale_factor`** when `img_size` is absent; `output_type="latent"` ⇒ **no VAE decode in the training graph** |
| `src/diffsolver/data/promptdataset.py` | 54, 57, 78-84, 89-90, 159-179, 182, 224-316, 250-276, 283-298, 300-307, 311-312 | targets **precomputed once and cached**; **`with_flowedit=True` default** loading **FLUX.1-dev** at `T_steps=28`, `n_max=24`, guidance 1.5/5.5; the non-FlowEdit branch uses a neutral *prompt*, not `∅`; 50/50 alternating forget/retain; only the last 2 latents stored |
| `src/diffsolver/hooks/ff_hooks.py` | 83-88 (gate at **87**) | gate = `hidden_states * mask + randn·epsilon·(1−mask)` |
| `src/diffsolver/hooks/norm_hook.py` | 84-93 (gate at **92**) | same gate form |
| `src/diffsolver/hooks/attention_processor.py` | 46-64 (gate at **64**), 160 | same gate form, per attention head, after the attention op |
| `src/diffsolver/hooks/linear_layer_hooks.py` | 70-75, 77-110, 85-88, 106-107 | gate form (non-FLUX path); `λ` init to `init_value`; `hook_counter` modulo ordering assumption |
| `src/diffsolver/hooks/base.py` | 41, 49-89, 104-121 (hard-discrete at **108-111**, `continues2binary` at **114-116**) | the masking dispatch; `eps` plumbed from config into the relaxation; **three** distinct binarization paths |
| `src/diffsolver/hooks/init_hooker.py` | 26-81 (eps at **45, 57, 74**) | FLUX path builds **attn + ff + norm** hookers; `masking_eps` reaches the relaxation; norm hooker skipped iff `n_lr == 0` |
| `src/diffsolver/hooks/cross_attn_hooks.py` | 103-105, 107-117 | `masking_fn` is a no-op (masking lives in the attention processor); threshold binarization |
| `src/diffsolver/utils/utils.py` | 213-223, 226-229, 232-257 | the relaxation; `l0_complexity_loss` and `calculate_reg_loss` **exist** |
| `src/diffsolver/utils/utils_train.py` | 30-45, 64-120, 217-261 (CLIP at **225**, filter at **249-252**) | `save_image` default `output_type="pil"` ⇒ instrumentation **includes VAE decode**; CLIP-based dataset filtering |
| `src/diffsolver/data/unlearn_template.py` | 249, 250-254, 269, 282, 292 | `VALIDATION_PROMPT["concept"]` is a **3-element set**; `CON_DECON_DICT` substitution table |
| `scripts/hyperparameter_tuning.py` | 25-91 (**91**: `os.system`) | **no second training loop** — it shells out to `train.py`, so the regularizer stays uncalled |
| `configs/flux.yaml` | 6-10, 15-36 (**17, 19, 24, 26, 30-32, 35**), 45-53, 59, 77-81 | `K=5`, `epochs=5`, `size=40`, `batch_size=1`, `accum=4`, `beta=0.1`, `epsilon=0.0`, `attn_lr=0`, `init_lambda=3`, `masking_eps=0.5`, `plot_interval=4`, **no `img_size`** |
| `configs/sdxl.yaml` | 7, 12, 15-24 | contrast: **`img_size: 512` is set there**, `K=2` — confirming the FLUX omission is an omission |
| `scripts/mask_selection_with_early_stopping.py`, `tests/`, `baselines/`, `README.md` | listing + targeted grep | located the post-training mask-selection stage and the README's own description (*"custom gradient checkpointing scheme"*, *"reconstruction loss"* + *"intermediate loss"*) |

**Grep-level verification** (not line-by-line): that `calculate_reg_loss` / `l0_complexity_loss`
are **never called** anywhere outside their own definition and `hyperparameter_tuning.py`'s
config assignment — searched across all `*.py` in the repository.

---

## 3. Source claims vs. our deductions — the distinction that matters most

### 3.1 Explicit source claims (quoted or directly read)

- P1 §3.6: `θ = M ⊙ θ′, M ∈ {0,1}^|θ|`; hard-discrete relaxation then binarization; *"neuron
  masking instead of weight masking to reduce the number of trainable parameters"*.
- P1 Eq. 16: the two-term objective with forget target **`ℱθ′(x_T, ∅)`**.
- P1 §3.7 (verbatim, §1.1 of [`DERIVATION.md`](DERIVATION.md)): the `Σ_i` gradient rule;
  step-wise gradient checkpointing storing only step outcomes and recomputing the forward in
  backward; *"lacks support for multi-GPU training, limiting the method's scalability"*.
- P1 §4.1: FLUX.1-schnell primary. P1 Table 6: batch 4, `lr = 0.5`, `β = 0.01`, Adam,
  400 steps, wd `1e-2`, 1× H100.
- C1: every row of §2.
- A1 (as rendered in retrieved summary text): DRaFT-K *"differentiates through only the last
  K steps of sampling"*; *"while gradient checkpointing makes it possible to backpropagate
  through the full sampling chain, optimization speed and overall performance can be
  substantially improved by truncating backprop through only the last K sampling steps"*;
  `K=1` best reward-vs-compute; DRaFT-LV reduces variance.
- A2 (abstract): *"end-to-end backpropagation of the reward gradient through the denoising
  process"*; *"finetunes low-rank adapter weight modules and uses gradient checkpointing, to
  render its memory usage viable"*.
- T1: `preserve_rng_state` *"Default: `True`"*; recompute under `torch.random.fork_rng` with
  the stashed forward RNG state restored.

### 3.2 Our deductions (**not** source claims — ours, and falsifiable)

1. **The lemma:** the mask gradient is an elementwise byproduct of the state-chain reverse
   sweep, so it cannot be made cheaper. *Deduced* from the gate form at four lines plus
   `epsilon: 0.0`. **This is the load-bearing claim of the sprint.**
2. **The exact-gradient floor** `K·F·(1+c_b)`, attained by `A2`; and that `A1` is
   bit-identical to the reference at `(2+c_b)/(2+2c_b)` of the cost.
3. **The Amdahl ceiling:** a free gradient is worth ≤`1.33×` over `A2` at `c_b = 1`.
4. **Four paper/code objective divergences** (target construction; replacement vs. sum;
   single fixed `x_T`; no `L0` term in the trained loss).
5. **The RNG mismatch** ⇒ the implemented gradient is neither the conditional pathwise
   gradient nor an unbiased estimate of `∇E`. **We did not characterize or bound its bias.**
6. **`P(m = 1) = 0.9659` at `λ = 3`** and the bounded noise range `[−1.0986, +1.0986]` at
   `eps = 0.5`; hence the published relaxation is a **bounded-noise variant**, not the
   hard-concrete distribution P1 cites.
7. **Training resolution 1024×1024** (≈4608 tokens) from the absent `img_size`.
8. **Cost composition** (gradient 54% / instrumentation 27% / preparation 19% at `c_b = 1`)
   and the **break-even budget `ν < 1.23–1.56`**.
9. **SSMG's unbiasedness, variance, and exact dominance.**

### 3.3 Unverified / unresolved — asserted by nobody here

| item | why unresolved |
|---|---|
| **Which erasure target is authoritative** | P1 Eq. 16 says `ℱθ′(x_T, ∅)`; C1's default says FlowEdit-on-FLUX.1-dev. **Essential, and we did not adjudicate it** |
| **Whether C1 @ `d178324` is the publication implementation** | all code in one commit **4 months after** P1 v1. A snapshot is not automatically the publication implementation |
| `β`, batch size, `N_upd` | P1 Table 6 vs `configs/flux.yaml` disagree on all three |
| Training step count in P1 | §4.1/Table 6 give **training steps (400)**, never the **sampler step count**; `K = 5` is from C1 only |
| `F`, `c_b`, `D`, `F_dev`, `ν`, `A2` memory feasibility | **unmeasured**; no profiling authorized or performed |
| `|D| ≈ 64` | depends on CLIP scores we did **not** compute |
| Binarization threshold / sparsity ratio | unspecified in P1; hardcoded `0` with a `TODO` in C1's training-time logging |
| `hook_counter` ordering | a static-reading assumption, **not runtime-validated** |
| A1's full method details | **abstract/summary level only**; the full DRaFT paper was not read |
| `"Zhang et al. (2024d)"` | not resolved to a specific paper |

**Static inspection is not runtime validation.** Nothing here is a reproduction, a method
audit, or evidence that any claim of P1 is right or wrong.

---

## 4. Access and method limitations

- **P1 read as arXiv HTML v1 via a fetch-and-summarize tool**, not as the PDF. Equation
  rendering (Eq. 16, the §3.7 formula) passed through that tool's transcription; §3.7 was
  retrieved verbatim and is quoted as such, but **we did not verify the equations against
  the PDF typesetting**. Treat equation glyphs as transcribed, not photographed.
- **A1's quotes are from retrieved search-result text, not the paper body.** They are
  attributed at that depth in [`ALGORITHM.md`](ALGORITHM.md) §4 and are used only to
  establish that truncated backprop through a sampler with checkpointing is **published** —
  a point that does not turn on exact wording.
- **C1 was cloned and read.** `git clone` of a public source repository is source retrieval.
  **Nothing was installed, imported, built or run**; no dependency, weight or model file was
  fetched; `pyproject.toml` was not acted on. The clone lives in the session scratchpad on
  `/data` and is **not** vendored into this repository.
- **T1 was read from an already-present environment** on `/data`; the environment was not
  rebuilt, activated or imported, and `torch` was never loaded.
- **No author or reviewer was contacted.** No benchmark, dataset or evaluation package was
  created or downloaded. No GPU was allocated or polled. No profiling was run.
- **No closed work was read or reused as evidence.** The gap map at `edaec97` was consulted
  only to confirm that its Minimalist row (S2b) was **paper-level only**, so this sprint's
  code inspection is new work and not a duplication. The FLUX diagnostic at `0e88d26`, the
  changing-replacement closure at `5786864`, and the SD1.x / human-review closures were
  **not opened, re-scored, reranked or revived**.

---

## 5. Reproducing this inspection

No computation required:

```
git clone https://github.com/YaNgZhAnG-V5/minimalist_concept_erasure.git
cd minimalist_concept_erasure && git checkout d178324ec4a48d0536a595b25fdad587b441add8
```

then read the lines in §2. The two arithmetic checks
([`DERIVATION.md`](DERIVATION.md) §7) are scalar/rational and need no model.
