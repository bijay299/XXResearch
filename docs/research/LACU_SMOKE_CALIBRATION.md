# LACU smoke calibration — one fully specified proposal, for a later decision

**Not authorised and not run.** This document specifies a measurement so the PI
can approve or refuse it as a single, bounded item. It is the only GPU work the
corrected brief asks for before the screen itself is a decision. **Nothing here
was executed**; no model was loaded, no image generated, no API called.

**What it buys.** It replaces the brief's **estimated** 1.5–4.0 device-hours per
LACU request — the band that dominates a 119–184 device-hour plan — with a
measurement, and it answers the one question analysis cannot: **does LACU's
training step fit in 40 GB at all, and at what batch?**

---

## 1. Concept and assets — nothing from the screen is touched

| | |
|---|---|
| **concept** | **`apple`**, one of LACU's own 10 released concepts. **Not** a screen concept (`cat`, `dog`, `sandwich`), **not** an anchor (`horse`, `potted plant`), **not** the control (`clock`) |
| **forget prompts** | the **shipped** `prompts/apple/train.csv` — 100 prompts, `id,prompt`, verified aligned 1:1 with the shipped candidate file |
| **mapping prompts** | built **on CPU** from the shipped `prompts/apple/candidates_clip.json`: each of the 100 entries carries `candidates` (10 each) and a **1-based** `selected` index — values run 1…10 with no zero, and three entries have `selected = 10`, so the mapping is `candidates[selected − 1]`, never `candidates[selected]`. Written as a one-column `prompt` CSV in the order of `train.csv` |
| **retain prompts** | the **shipped** `prompts/apple/related_concepts.txt`, passed directly to `--retain_prompts_path`. `utils.load_prompts_from_file` reads a `.txt` line by line, so no generation step is needed |
| **base model** | our local diffusers `M0` (`/data/.../Generators_substitute/sd-v1-5`), read-only |
| **environment** | the existing `venv-cuig` — torch 2.4.0+cu121, diffusers 0.30.2, accelerate, transformers 4.44.2, all of LACU's training imports already resolvable. **No new conda env**, which matters with `/` at 100% |

**This is a cost and memory measurement, not a method run.** Supplying a fixed
mapping CSV and a shipped retain list bypasses `model_score_map_gen.py` and
`retain_prompt_gen.py`, so **the run is not native LACU** and **its checkpoint is
discarded** — it is never a parent for any screen stage and never evaluated on
DEV, ACCEPT, TEST or `R`. The batch shapes, the three losses, the replay
pipeline, the optimiser state and the save path are nevertheless exactly those
of a screen request, which is what makes the timing and memory transferable.

**Separation from screen holdouts is total:** no frozen set is read, no frozen
prompt is generated from, and no image of any screen category is produced.

## 2. API and input needs

**None.** `OPENAI_API_KEY` is **not required** for this run, because the two
API-dependent steps are replaced by shipped assets and a CPU-built CSV. (Native
LACU in the screen *does* require the key, per concept per checkpoint; that stays
a separate decision and is not pre-empted here.)

`wandb` is absent from the environment and is **not installed**: the run passes
`--wandb_mode disabled`, which `train.py` honours.

## 3. Batch, accumulation and steps

Effective batch is held at the authors' **108** throughout, because the
generation cost per optimizer step is `108 × 2 × 20 = 4,320` UNet passes
**regardless of how that batch is split** — reducing the per-device batch costs
memory headroom, not compute. Three processes, `fp16`, as the authors recommend.

| config | per-process batch | accumulation | processes | effective | purpose |
|---|---|---|---|---|---|
| L1 | 6 | 6 | 3 | 108 | the largest batch plausibly fitting 40 GB |
| L2 | 4 | 9 | 3 | 108 | fallback |
| L3 | 2 | 18 | 3 | 108 | last resort before declaring infeasible |

**Steps.** Each ladder configuration runs **12 optimizer steps**: the first **2
are discarded as warm-up** (compile, allocator growth, first replay), the
remaining **10 are the measurement**. Then **one** configuration — the largest
that fits with ≥10% memory headroom — runs on to **step 100**, the first
checkpoint save `train.py` triggers natively (`global_step in {100, 500}` or
`% 200 == 0`), to exercise the real save and reload path.

`--seed` is **set explicitly** (it defaults to `None`, i.e. non-reproducible
latents) and recorded, together with every argument, in the run record.

## 4. Measured outputs

1. **seconds per optimizer step**, mean and spread over the 10 measured steps,
   per configuration;
2. **the generation share** — the two `replay_pipeline(...)` calls timed
   separately from the four UNet training passes and the backward, so the cost
   model's dominant term is measured rather than assumed;
3. **peak memory** — `torch.cuda.max_memory_allocated()` and
   `max_memory_reserved()` per process, plus the `nvidia-smi` peak, and the
   **headroom** against 40,960 MiB;
4. **extrapolated device-hours per request** at 250 and at 350 optimizer steps,
   the two step counts the released runner actually uses, with the measured
   per-step time (replacing the 1.5–4.0 h estimate);
5. **disk bytes per saved checkpoint**, measured on the step-100 save, against
   the ~5 GB assumption and the ≈390 GB plan total;
6. **the implied batched speed-up** against our measured batch-1 generation rate
   of 1.1027 s/image, so the brief's `1×–4×` band is replaced by a number;
7. whether `mixed_precision bf16` is usable on these devices, recorded if `fp16`
   shows instability in the 10 measured steps (loss non-finite), **not** tuned.

Every output goes to one JSON record plus a verbatim log, with the argument set,
the seed, the environment versions and the GPU identities.

## 5. Checkpoint reload check

On the step-100 save, **on CPU, after the GPU run finishes**:

1. the saved directory is a complete diffusers pipeline —
   `model_index.json`, `unet/`, `vae/`, `text_encoder/`, `tokenizer/`,
   `scheduler/` all present;
2. `StableDiffusionPipeline.from_pretrained(<saved dir>)` loads without error;
3. **the §1.5 open question is answered**: `train.py`'s end-of-run path is
   `StableDiffusionPipeline.from_pretrained(model_path, unet=unwrapped_unet)`
   *without* `safety_checker=None`, while the saved `model_index.json` records no
   safety checker. The check is to call exactly that with `model_path` set to the
   **saved** checkpoint, confirming that stage *k+1* of a chain can load stage
   *k*. If it raises or warns that a safety checker is required, the required
   wrapper argument is recorded as a **glue fix**, not a method change;
4. the saved UNet's `state_dict` keys match the base UNet's exactly, and at least
   one tensor differs from the base — the run did train something.

Items 1–4 need no GPU and no image generation.

## 6. Failure action — decided in advance

| observed | action |
|---|---|
| **L1 fits with ≥10% headroom** | record it as the screen's configuration; proceed to the next dependency item |
| **L1 OOMs, L2 or L3 fits** | record the deviation from the authors' recommended batch **explicitly** in the pre-registration, together with their README caveat that small per-device batches "can make it hard to reproduce the same numbers even when the effective batch size is similar" |
| **every configuration OOMs** | **stop and report.** LACU's arms are infeasible on this host; LACU is non-droppable, so this is a blocking finding and the screen's stopping rule fires. Do not reduce the effective batch below 108 to make it fit — that changes the method's operating point |
| **it fits, but the measured per-request cost exceeds 6 device-hours** | **stop and return to the PI** with the re-costed plan (26 requests × the measured cost). Do not silently proceed into a plan larger than the one approved |
| **the reload check of §5 item 3 fails and no glue fix is found** | **stop and report**: sequential chaining is broken for LACU on this version, which is a blocking finding, not something to work around in the method |
| **loss non-finite in `fp16` across configurations** | record it, do **not** retune. `bf16` may be *reported* as an option; adopting it is a separate decision |

In every case the run's checkpoint is **deleted** after the §5 checks, and no
result from it enters any screen table.

## 7. Estimated cost

`3 configs × 12 steps + 100 steps = 136 optimizer steps`, each
`4,320` UNet passes at effective batch 108 → `587,520` passes ≈ **9,792
image-equivalents** of our measured 30-step generation →
**≈ 0.75 – 3.0 device-hours** (batched speed-up 4× to 1×; the 1× end is our
measured batch-1 rate, the ceiling to request). Re-derived in
`scripts/seq/screen_measurement_inventory.py`.

Three GPUs, which must be **independently verified unoccupied by another user at
execution time** — not at approval time. There is no GPU-hour cap; this figure
is a plan input, not a request for one.

## 8. What it will not establish

* **Not that LACU erases anything.** No DEV, ACCEPT, TEST or `R` measurement is
  made, and 12 or 100 steps on `apple` with a fixed mapping is not an erasure
  result. No detection rate is produced.
* **Not native LACU's behaviour.** The mapping and retain steps are bypassed, so
  the locality mechanism is not exercised; the full per-stage cost of native LACU
  additionally includes `model_score_map_gen.py` (~0.15 h, estimated) and
  `retain_prompt_gen.py` (**unmeasured**, and API-dependent).
* **Not the screen's feasibility on its own.** The base-rate screen for
  `potted plant` and `clock`, the SPEED round-trip checks and the pre-registration
  freeze are separate dependency items.
