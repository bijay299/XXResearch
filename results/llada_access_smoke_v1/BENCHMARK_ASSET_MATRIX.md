# Benchmark Asset Matrix — prospective LLaDA unlearning comparison

Prepared 2026-10-19 on hyperplane. **Nothing in this document was evaluated.** It records
what was inspected, what access was actually checked from this host, and what remains blocked.

Audit method: HF Hub REST API over anonymous HTTPS from hyperplane (metadata + small ranged
reads), plus full source reads of the two pinned Git repositories. No benchmark evaluation was
run, no second large model was downloaded, no TOFU backbone was trained, no paid API was called.

Pinned repositories:

- **LLaDA** — `github.com/ML-GSAI/LLaDA` @ `9182493720ed723ef8031210d85959364e51cbe0` (2026-07-15)
- **MDU** — `github.com/leegeoru/MDU` @ `e5e8e8274ff8f56f16e978803d9e902359eb5272` (2026-05-25), MIT

---

## Matrix

| Component | Exact source/revision | Access actually checked | Required preparation | Remaining blocker |
|---|---|---|---|---|
| **TOFU `forget10`** (forget split) | `hf.co/datasets/locuslab/TOFU` @ `324592d84ae4f482ac7249b9285c2ecdb53e3a68`, `forget10.json` (114,856 B) | ✅ API 200, `gated=false`, MIT; ranged read of first 1,200 B succeeded | Download (3.39 MiB for all required splits); write to `./data/tofu/` layout MDU's `run_main.sh` expects | None |
| **TOFU `retain_perturbed`** (retain split used by MDU) | same rev, `retain_perturbed.json` (641,057 B) | ✅ listed at pinned rev | as above | None |
| **TOFU `forget10_perturbed`** (truth-ratio) | same rev, `forget10_perturbed.json` (674,694 B) | ✅ listed at pinned rev | needed for `--truth_ratio` | None |
| **TOFU `retain90`** (retain utility) | same rev, `retain90.json` (962,987 B) | ✅ listed at pinned rev | as above | None |
| **Real-author utility assets** | same rev, `real_authors.json` (21,862 B) + `real_authors_perturbed.json` (17,146 B) | ✅ listed at pinned rev | as above | None |
| **World-fact utility assets** | same rev, `world_facts.json` (23,094 B) + `world_facts_perturbed.json` (18,053 B) | ✅ listed at pinned rev | as above | None |
| **TOFU `full.json`** (for the SFT backbone MDU requires) | same rev, `full.json` (1,077,844 B) | ✅ listed at pinned rev | input to the **missing** TOFU SFT stage | Blocked downstream — no SFT code (see below) |
| **RWKU data** | `hf.co/datasets/jinzhuoran/RWKU` @ `d72f493d481d1b0a9bdc6e869d32baeffad8904f`, 4,424 files, CC-BY-4.0 | ✅ API 200, `gated=false`; 23 configs enumerated incl. `forget_level1/2/3`, `neighbor_level1/2`, `utility_general/truthfulness/factuality` | `load_dataset("jinzhuoran/RWKU", <config>)`; per-entity dirs under `Target/<N>_<Name>/`; MDU's `convert_rwku_to_tofu.py` to reshape | **No LLaDA path.** MDU ships only `rwku_dream`; see §"RWKU is Dream-only" |
| **RWKU eval dependencies** | `rouge-score`, `scikit-learn`, `datasets` (MDU `requirements.txt`) | ✅ all public PyPI; `rouge-score` + `sklearn` installable | install into the eval env | None |
| **TOFU starting checkpoint (author-matched)** | **does not exist publicly** | ✅ checked — searched HF for `llada`+`tofu`; MDU repo read in full | would require running the unreleased TOFU SFT (LLaDA, **1000 epochs** on `full.json`) | 🔴 **BLOCKING — see §1** |
| **LLaDA-8B-Instruct** (off-the-shelf backbone) | `hf.co/GSAI-ML/LLaDA-8B-Instruct` @ `08b83a6feb34df1a6011b80c3c00c7563e963b07`, MIT | ✅ **downloaded and used**; all 6 shard SHA-256 verified vs pinned LFS digests | none — present at `/data/bijaypandey/llada_smoke/model/` | None, **but it is not a TOFU backbone** (§1) |
| **LLaDA-8B-Base** | `hf.co/GSAI-ML/LLaDA-8B-Base` @ `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, MIT | ✅ metadata only (`gated=false`, 16 files) — deliberately **not** downloaded | ~15 GiB download if ever needed | None |
| **Native baseline code — MDU LLaDA train** | `MDU@e5e8e82:src/unlearn_mdu_llada.py` (1,668 lines) | ✅ source read in full | must be executed **inside a `dllm` checkout** | 🔴 **`dllm` undeclared — see §2** |
| **Native baseline code — MDU LLaDA eval** | `MDU@e5e8e82:scripts/eval_tofu_llada.py` | ✅ source read in full | as above | 🔴 same `dllm` dependency |
| **`dllm` framework** | **not pinned by MDU.** Best match: `github.com/ZHZisZZ/dllm` (Apache-2.0, HEAD `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`) | ⚠️ existence + symbol-level match checked via GitHub API; **not cloned, not run** | clone, pin a revision, reconcile versions | 🟡 **identification unconfirmed — see §2** |
| **Tokenizer compatibility** | LLaDA tokenizer @ `08b83a6f` (`tokenizer.json` sha256 `071eb20f…`) | ✅ loaded and exercised on-host in inference + LoRA step | `mask_token_id` is **absent from tokenizer config**; eval falls back to `model.config.mask_token_id` (126336) — confirmed present in `config.json` | None (fallback works), but worth asserting explicitly in any eval run |
| **Evaluation settings** | `MDU@e5e8e82:configs/mdu_tofu.yaml`, `mdu_rwku.yaml` | ✅ both read in full | TOFU: splits `[forget10, retain10, real_authors, world_facts]`, `N_mc=128` (Eq. 14 MC estimator), `denoise_steps=128`, `max_new_tokens=128`. RWKU: 8 metrics, bs 32, `steps=64`, `temperature=0.0`, bf16 | None |
| **Licenses** | LLaDA MIT · MDU MIT · TOFU MIT · RWKU CC-BY-4.0 · `dllm` Apache-2.0 | ✅ all confirmed from API/repo metadata | attribution only | None — all permissive |
| **Paid judge / API** | — | ✅ grep for `openai`, `gpt-4`, `anthropic`, `api_key`, `azure`, `requests.post` across `MDU/src` + `MDU/scripts`: **zero hits** | none | **None — no paid API required.** All metrics local (ROUGE-L, Eq. 14 MC answer probability, truth ratio) |
| **Hardware / environment parity** | MDU pins torch 2.7.0 + transformers 4.57.0, tested on **H200 141 GB** | ✅ read from `requirements.txt` + README | A100-40GB here; MDU uses per-device batch 4 × grad-accum 4 with a **frozen second (anchor) model** (`frozen_unconditional: true`) | 🟡 Memory feasibility of MDU's *full-parameter, two-model* setup on 40 GB is **unverified** (§3) |

---

## 1. 🔴 The TOFU starting checkpoint is the primary blocker

This was checked carefully, as it determines whether a TOFU comparison is possible at all.

**MDU's own README states the requirement and the gap:**

> "For TOFU we additionally fine-tune each backbone on the full TOFU corpus to instil the
> target knowledge (LLaDA: 1000 epoch, Dream: 300 epoch). **The SFT code itself is not part
> of this release**; the resulting Base SFT checkpoint is consumed by `run_main.sh` via the
> `LLADA_BASE_SFT` / `DREAM_BASE_SFT` paths."

`run_main.sh` confirms it is a **local placeholder path**, not a downloadable artifact:

```sh
LLADA_BASE_SFT=${LLADA_BASE_SFT:-/path/to/LLaDA-TOFU-SFT-1000ep}
```

and the `unlearn_mdu_llada.py` docstring likewise expects a locally produced
`./checkpoints/llada-tofu-sft/checkpoint-final`.

**Conclusion: no author-matched LLaDA TOFU checkpoint is published.** Reproducing MDU on TOFU
therefore requires re-deriving a backbone from unreleased SFT code at an unusual scale
(1000 epochs over the 200-row TOFU corpus) — with no reference artifact to validate it against.

**The ordinary `LLaDA-8B-Instruct` checkpoint is not a substitute.** TOFU's facts concern
fictional authors invented for the benchmark; they are not in the pre-training corpus, and
`forget10` cannot be "unlearned" from a model that never encoded it. Any unlearning metric
computed against off-the-shelf Instruct would be measuring absence, not forgetting.

### Third-party TOFU checkpoints (provenance recorded, none downloaded)

| Repo | Revision | Files | License | Card / provenance |
|---|---|---|---|---|
| `Jameszhiwei/llada-tofu-forget10-SFT` | `b36a16b956d76eb2ef1b9c3a7c8b25ba429ed51d` (2025-11-27) | 4 shards | **none declared** | **no README, no `base_model`**, 9 downloads |
| `guanmingchiu/llada-8b-tofu-sft` | `a3f811825b969fd6ce721c45e908cafb3b71299a` (2026-06-08) | 4 shards | **none declared** | **no README, no `base_model`**, 27 downloads |
| `guanmingchiu/lrcu-llada-8b-tofu` | `fc27d13ee653cb08a1f99979cd2f5cfe661e6b39` (2026-06-16) | 4 shards | MIT | has card; `base_model: guanmingchiu/llada-8b-tofu-sft`. This is an **already-unlearned** model from a *different* method (LRCU) — not a starting checkpoint |
| `Jameszhiwei/llada-tofu-forget10-unlearned` | `f9fbf72cb615d6e93ca2de669fe0c5ce9e439096` | **1 file (`.gitattributes` only)** | none | **empty/broken repo — no weights** |

**None of these is an exact reproduction substitute, and none should be described as one.**
Specifically:

- **Not author-matched.** MDU's authors are *Georu Lee et al.* (`github.com/leegeoru`). Neither
  `Jameszhiwei` nor `guanmingchiu` is an identifiable MDU author, and MDU never references them.
- **Epoch count undocumented.** MDU specifies 1000 epochs. Neither SFT repo documents its
  recipe, epochs, LR, or seed, so the "instil the knowledge" step cannot be shown to match.
- **Observable structural difference:** both ship **4 safetensors shards** where the official
  LLaDA-8B-Instruct ships **6**. This alone does not prove a dtype or architecture change
  (re-sharding is routine), but it is an unexplained divergence that would need verification
  against `config.json` and tensor dtypes before use.
- **No license on the two SFT candidates**, which blocks redistribution and complicates use.

Using one of these would be a *different experiment* — a comparison against a third-party
backbone of unknown provenance — and must be reported as such, never as reproducing MDU.

## 2. 🟡 `dllm` is an undeclared dependency

`MDU/requirements.txt` lists torch, transformers, accelerate, datasets, numpy, pyyaml, tqdm,
sentencepiece, rouge-score, scikit-learn. It does **not** list `dllm`. Yet both LLaDA entry
points depend on it:

```python
# src/unlearn_mdu_llada.py
import dllm
from dllm.core.trainers import MDLMTrainer, MDLMConfig
from dllm.core.samplers.utils import get_num_transfer_tokens
class MDUCollator(dllm.utils.CollatorWrapper): ...
model = dllm.utils.get_model(model_args=model_args)

# scripts/eval_tofu_llada.py
sys.path.insert(0, "./dllm")
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
```

The docstring shows MDU is meant to be run from **inside** a `dllm` checkout
(`cd ./dllm`, `--config_file scripts/accelerate_configs/ddp.yaml`). No `dllm` directory is
vendored in the repo and no URL or revision is given anywhere.

**Best match — identification, not confirmation.** `github.com/ZHZisZZ/dllm` ("dLLM: Simple
Diffusion Language Modeling", Apache-2.0, 2.7k stars) provides every symbol MDU imports,
verified by GitHub API at HEAD `ca17675` (2026-04-15, i.e. predating MDU's 2026-05-15 release):

- `dllm/core/trainers/__init__.py` exports `MDLMConfig`, `MDLMTrainer` ✅
- `dllm/core/samplers/__init__.py` exports `MDLMSampler`, `MDLMSamplerConfig`, `get_num_transfer_tokens` ✅
- `dllm/utils/collators.py` present (for `CollatorWrapper`) ✅
- `dllm/pipelines/dream/` present (for the Dream entry point) ✅
- `scripts/accelerate_configs/ddp.yaml` present — the exact path in MDU's docstring ✅

This is a strong match but remains **inferred**: MDU names no repository and pins no revision,
so the correct `dllm` commit for MDU is unknown. The framework was **not cloned or executed**
here. Any future run must pin a `dllm` revision and record that the pin was chosen by us, not
by the authors.

## 3. RWKU is Dream-only; and a memory caveat

- **No `rwku_llada` preset exists.** `run_main.sh` offers `tofu_llada`, `tofu_dream`,
  `rwku_dream`. The RWKU evaluator shipped is `scripts/eval_rwku_dream.py`; there is no
  `eval_rwku_llada.py`. `configs/mdu_rwku.yaml` states the backbone is
  "Dream-7B-Instruct (off-the-shelf, no SFT)". A LLaDA-on-RWKU comparison would require
  **writing new code**, not configuring existing code — and would no longer be MDU's
  published RWKU setup.
- **One upside:** RWKU needs *no* SFT backbone (off-the-shelf + no retain set), so it avoids
  the §1 blocker entirely. That makes RWKU the cheaper path to a *native* baseline — but only
  for Dream as published.
- **Memory caveat (unverified):** MDU uses `frozen_unconditional: true`, i.e. a second frozen
  copy of the model for the anchor, plus full-parameter AdamW at per-device batch 4 ×
  grad-accum 4, on an H200 (141 GB). Full-parameter AdamW on 8 B params needs roughly
  8 B × (4 bytes master + 4 + 4 Adam moments) ≈ **~128 GB of optimizer/master state alone**,
  before a frozen 16 GB anchor copy and activations. This **will not fit** on a single
  A100-40GB and is not addressed by the LoRA feasibility result in this report.

---

## Readiness summary

**Accessible now** (verified from hyperplane, no preparation beyond a small download):
- TOFU — all required splits incl. forget/retain, real-author and world-fact utility assets (3.39 MiB total, MIT)
- RWKU — full dataset, 23 configs, CC-BY-4.0
- LLaDA-8B-Instruct (downloaded, hash-verified) and LLaDA-8B-Base (metadata verified)
- MDU source, configs and evaluation settings; all licenses permissive
- No paid judge or API is required at any point

**Requires additional preparation** (possible, but work):
- Download + reshape datasets into the `./data/tofu/`, `./data/rwku/` layouts MDU expects
- Clone and **pin** a `dllm` revision, then reconcile it with MDU's torch 2.7.0 / transformers 4.57.0 pins
- Resolve MDU's full-parameter two-model memory footprint against A100-40GB (quantified above; likely needs multi-GPU sharding, or a parameter-efficient deviation from the published method)

**Missing or not verified**:
- 🔴 **Author-matched LLaDA TOFU SFT checkpoint — missing.** SFT code unreleased; no published artifact. Hard blocker for TOFU reproduction.
- 🔴 **MDU's exact `dllm` revision — not verified.** Upstream identified by symbol match only.
- 🔴 **Native LLaDA RWKU baseline — does not exist** in the release (Dream only).
- 🟡 Third-party TOFU checkpoints exist but are unmatched, mostly unlicensed, and undocumented; one is an empty repo.
- 🟡 MDU's own pinned environment (torch 2.7.0 / transformers 4.57.0) was **not** exercised here.
