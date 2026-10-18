> ## ⚠ SUPERSEDED IN PART — see [`AMENDMENT_01_ACCESS_AND_READINESS.md`](AMENDMENT_01_ACCESS_AND_READINESS.md)
>
> **Status retained: `DATA_ACCESS_BLOCKED` — PRACTICE/UI PREVIEW ONLY.** The body below is
> preserved as the historical record of the bounded check. Two things in it are corrected:
>
> 1. **The conclusion was overstated.** The supported conclusion is only: *no eligible matched
>    non-sensitive object assets were identified during the bounded access check.* **Withdrawn:**
>    that no modern erased checkpoints exist, that all relevant modern papers are code-only, and
>    that no eligible data source exists. Zero releases and zero metadata-filtered hits cannot
>    establish nonexistence.
> 2. **The GEM row is wrong.** GEM **does** publish FLUX erasure checkpoints —
>    [`MAI-Lab/GEM-Flux-Nudity`](https://huggingface.co/MAI-Lab/GEM-Flux-Nudity) and
>    [`MAI-Lab/GEM-Flux-Bloody-Gore`](https://huggingface.co/MAI-Lab/GEM-Flux-Bloody-Gore), linked
>    from the [GEM README](https://github.com/multimodal-ai-lab/GEM/blob/c328fc6bec687efe60223498d1b6bed2e5fbcffd/README.md).
>    The model cards describe them as **reproductions, not the original paper checkpoints**. They
>    remain **ineligible** for this frozen non-sensitive object study because their targets are
>    sensitive content. The search missed them because HuggingFace `search=` matches the model
>    **name**, and neither name contains "erase" or "unlearn", though both are correctly tagged
>    `concept-erasure` and `base_model:black-forest-labs/FLUX.1-dev`.
>
> Neither correction resolves the missing matched object assets.

# Asset / provenance check — measurement_study_v1

Window: 2026-10-18T04:42:35Z → 08:42:35Z (first four working hours).
Check concluded at approximately 05:00Z, about 0.3 h of the four-hour allowance. **No GPU time was spent on this check** — it was entirely public-registry queries and reading.

## What the study needs

Per the frozen scientific question — *do automated target-presence evaluators make different
errors on original versus concept-erased modern-model images?* — an eligible asset must supply:

1. a **modern backbone** (FLUX / SD3.5 preferred); **SD1.x is explicitly excluded**;
2. an **erased condition** produced by a real erasure method, not negative-prompt /
   guidance avoidance, not inpainting/object removal, not a fabricated condition;
3. **three concrete, non-sensitive object targets**;
4. **matched** original ↔ erased outputs: identical prompts, initial noise and inference
   settings, with pairing verifiable from source records (not filenames alone);
5. either versioned public matched image sets, **or** released original *and* erased
   checkpoints plus a documented inference-only path.

## Candidates examined, and why each fails

| candidate | backbone | erased weights released? | targets | verdict |
|---|---|---|---|---|
| **CERA** `JackChenHF/cera-erased-models` | **SD v1.5** (`model/CERA_SD1.5/`, `model/SD1.5_ckpt/v1-5-pruned-emaonly.ckpt`) | **Yes** — 135 artifacts, 9 methods × 15 concepts, per-artifact sha256 + `reload_proof.json` | banana, bed, book, bus, car, cup, horse, kite, pizza, skateboard, teddy_bear, tennis_racket, truck, tv, umbrella — ideal | **EXCLUDED: SD1.x.** Otherwise the single best-provenanced artifact found. |
| **EraseFlow** `EraseFlow/*` (NeurIPS'25) | released ckpts are **SD v1.4** (`diffusers:StableDiffusionPipeline`); FLUX/SD3 exist only as *training* scripts | Yes, but SD1.4 only | NIKESHOES, COCACOLABOTTLE, PEGASUS (objects) + nudity/styles | **EXCLUDED: SD1.x** for every released weight. |
| **GEM** (arXiv 2606.00140) | Flux.1-dev + SD3.5-Medium | **No** — code only (`multimodal-ai-lab/GEM`) | nudity, gore, celebrities, characters, Ghibli style | **EXCLUDED:** no weights; targets sensitive/style. |
| **GeoClean** `ssy166/GeoClean` | **FLUX.1-dev**, training-free | n/a — operates on the unmodified base model | nudity | **EXCLUDED:** mechanism is safety-**guidance steering** (encodes the concept as a prompt and subtracts its direction during denoising) = negative-prompt avoidance; target is sensitive content. |
| **Z-Erase** (arXiv 2603.25074) | single-stream DiT | **No** — repo `nxjiang-jnx/Z-Erase`, **0 releases** | not specified in the text obtained | **EXCLUDED:** no weights. |
| **ReFlux / "Erased, But Not Forgotten"** (2510.00635) | Flux.1-dev, Flux.1-schnell, SD3 | **No** released erased checkpoints | NSFW, styles, celebrities, a few entities | **EXCLUDED:** no weights; targets sensitive/style. |
| **ETC** "Erasing Thousands of Concepts" (2604.16481) | SDv1.4 + SDv3.5-Large | **No** releases stated | celebrities, styles, characters | **EXCLUDED:** no weights; no objects. |
| **EEB** `multimodal-ai-lab/EEB` | Flux checkpoints available | Yes, but **backdoor-poisoned**, nudity trigger | nudity | **EXCLUDED:** poisoned condition + sensitive target. |
| **EraseAnything / EraseAnything++** | FLUX | — | — | **EXCLUDED by assignment:** closed pipeline for this programme. |
| `unhzyx/entity_erasure`, `ddgoodgood/trce-erased-model` | SD1.x / sd14 | yes | — | **EXCLUDED: SD1.x.** |

## Systematic searches run (all negative for a modern backbone)

- HuggingFace model search, sorted by downloads, for `erase`, `erased`, `erasure`,
  `unlearn`, `EraseFlow` — no FLUX/SD3.5 erased weights.
- HuggingFace **`base_model:` filtered** search, the most precise instrument available:
  `black-forest-labs/FLUX.1-dev`, `stabilityai/stable-diffusion-3.5-large`,
  `stabilityai/stable-diffusion-3.5-medium`, `stabilityai/stable-diffusion-xl-base-1.0`,
  each crossed with `erase` / `eras` / `unlearn` → **zero genuine hits** (the one string
  match, `Eraserdrive/Rober2025`, is an unrelated personal LoRA).
- HuggingFace dataset search for `concept erasure`, `erased images`,
  `unlearning diffusion images`, `erasure benchmark` → nothing containing matched
  original/erased image pairs.
- GitHub repository search for `flux concept erasure`, `concept erasure rectified flow`,
  `sd3 concept erasure`, `flux unlearning diffusion` → GeoClean and a third-party
  `CURE-Improve` **port** (ports are excluded by the assignment).

## Conclusion

> **`DATA_ACCESS_BLOCKED`.**
>
> **Exact missing item:** a publicly released **concept-erased checkpoint (or matched
> original/erased image set) for a modern backbone — FLUX.1-dev or SD3.5 — targeting
> concrete non-sensitive objects**, produced by a genuine erasure method rather than
> inference-time guidance avoidance.
>
> Every modern-backbone erasure paper found releases **code only**; every released *erased
> weight* found is **SD1.x**; and the only training-free FLUX method found is
> negative-concept guidance steering on a sensitive target.

The scientific task is **not** silently changed. SD1.5 CERA is **not** substituted, even
though it is otherwise the best-provenanced artifact available, because the assignment
excludes SD1.x. No erasure method is implemented, ported or tuned to manufacture an erased
condition, because that is also excluded.

## What was built anyway, and why

The blocker concerns **only the study panel**. Everything in §§4–7 of the assignment — the
frozen protocol, response schema, analysis specification, the two-stage blinded review
instrument and its validation — is independent of which images fill it. Those are published
as **preparation artifacts** with a real, browser-validated viewer carrying the 12 permitted
non-study practice items and **zero study images**, so that the instrument is ready the moment
an eligible asset exists.
