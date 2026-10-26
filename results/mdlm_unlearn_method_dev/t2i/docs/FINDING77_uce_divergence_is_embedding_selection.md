# FINDING 77 — Running the **actual upstream UCE**, our port diverges from it by **11–38%** of the edited weight norm. The cause is **embedding selection**, not accumulation order and not λ-conditioning. `FINDING75` had the mechanism wrong.

**Date:** 2026-10-26 · **Experiment:** R4b · **Supersedes:** `FINDING75` (retracted)
**Upstream EXECUTED:** `unified-concept-editing@7c724d9d7d19190e9bd33fe62fb656f1df87cd8e`,
`trainscripts/uce_sd_erase.py`, `UCE(...)` called unmodified. Shim recorded: binds module-level
`device` and `torch_dtype=torch.float32`, which upstream defines only inside its `__main__` block
(L116-117). Function body untouched.
**Evidence:** `outputs/r4b_uce_lamb{0.1,0.5,1000.0}.json`, `logs/r4b_uce.log`

---

## 1. The decomposition

All **32** edited `attn2` K/V matrices, every one compared. Upstream's output is **reloaded from its
own `.safetensors`**, so serialization precision is included.

| λ | **A** upstream vs our port (full pipelines) | **B** our solver on *upstream's* selection, vs upstream | **C** fp32 vs fp64, upstream selection | **C** fp32 vs fp64, port selection |
|---:|---:|---:|---:|---:|
| **0.1** | **0.3842** max / 0.2755 mean | 1.51e-03 / 7.93e-04 | 1.05e-03 | **6.26e-02** |
| **0.5** | 0.3219 / 0.2173 | 3.05e-04 / 1.62e-04 | 2.14e-04 | 1.36e-02 |
| **1000** | **0.1084** / 0.0718 | **1.33e-07** / 1.27e-07 | 3.72e-07 | 8.25e-06 |

## 2. What the three columns isolate

**B kills the accumulation-order hypothesis.** When our solver is given **upstream's own embedding
selection**, it reproduces upstream's matrices to **1.3e-07 at λ=1000** and ~1.5e-03 at λ=0.1. Our
transcription of the closed form is, to that precision, the same computation. `FINDING75`'s claim
that the two "differ only in summation order" was both the wrong mechanism *and* measuring an effect
that is ~10⁶ times smaller than the real one at λ=1000.

**A − B is embedding selection, and it dominates.** At λ=1000, A = 1.08e-01 against B = 1.33e-07 —
a factor of ~8×10⁵. Essentially **all** of the upstream-vs-port divergence comes from:

| | selection |
|---|---|
| upstream (`uce_sd_erase.py:34-42`) | **one** embedding, at `attention_mask.sum()-2` |
| our `erase_uce.py:63-65` | positions **`attention_mask.sum()-2 … 76`** — the final content token **plus ~75 padding positions** |

**C measures float32 error instead of inferring it.** Against a float64 solve on identical matrices,
float32 is accurate to **3.7e-07** under upstream's selection. `FINDING75` inferred "one digit
survives" from a condition number; measured, that was wrong by ~6 orders of magnitude.

**But C also exposes a real cost of our selection.** Under *our* selection the float32 error is
**6.26e-02 at λ=0.1** — about 60× worse than the full-pipeline difference at λ=1000, and ~60,000×
worse than upstream's selection at the same λ. Stacking ~76 near-duplicate padding embeddings makes
`cᵀc` far worse conditioned than a single token does. So conditioning *is* real — but it is a
consequence of **our embedding choice**, not of λ alone, and it is a second-order contributor behind
selection itself.

## 3. What this means for our record

- **Our UCE port is not upstream UCE.** It diverges by 11–38% of the edited weight norm at every λ
  tested, because it edits using ~76 embeddings per concept where upstream uses one.
- **The divergence is largest at small λ** (38% at 0.1, 11% at 1000), so the arms built at λ=0.1 —
  13 of our 22 UCE arms — are the furthest from upstream.
- **`erase_uce.py`'s docstring claimed this selection matched the official implementation.** It does
  not; corrected in source, with behaviour unchanged so every historical arm still reproduces.
- **`FINDING75` is superseded.** Its number (16%) came from character-split inputs and two local
  transcriptions; the correctly measured full-pipeline figure is **27.6% mean / 38.4% max at λ=0.1**.

## 4. What this does **not** establish

- **Not which implementation is more accurate.** Column C bounds each one's own float32 error; it
  does not adjudicate the modelling choice. Using padding embeddings is a *different edit*, not
  self-evidently a worse one.
- **Not an image difference.** No generation was compared. A matched-prompt, matched-seed panel is
  the next step and is **unrun**.
- **Not an attack-resistance difference.** Nothing here touches robustness, and `FINDING64`'s UCE
  attack results stand as measurements of **our port**, unchanged.
- **One concept (church), one guide (`''`), one preserve set, SD-1.4.** The λ=1000 row is the only
  setting matching an arm that cleared K1, and that arm is SD-**1.5** — a different checkpoint.

## 5. Status

UCE remains a **project reimplementation**. R4b compared upstream's executed output against our
port's computation on matched inputs; it did not run our full `erase_uce.py` end-to-end into a
checkpoint, nor compare generations. Both remain open.
