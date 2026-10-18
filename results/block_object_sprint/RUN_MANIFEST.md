# RUN_MANIFEST — BLOCK object-erasure sprint (working freeze)

Committed **before** any scored generation. This is a working freeze, not a dossier.
Everything below is fixed; changes after this point are recorded as amendments with reasons.

## Sprint clock and budget

| | |
|---|---|
| First work (UTC) | 2026-10-18T00:22:03Z |
| Deadline (UTC, +120 h) | 2026-10-23T00:22:03Z |
| Integration sub-deadline (+24 h) | 2026-10-19T00:22:03Z |
| GPU budget | 48 cumulative GPU-device-hours (charged wall x devices) |
| Image budget | 1,600 newly decoded images (core plan 1,200) |
| Host | `hyperplane`, 4 x A100-SXM4-40GB, driver 550.144.03, CUDA 12.4 |
| Devices used by this sprint | GPU 0 and GPU 3 only. GPU 1 and GPU 2 are another user's (`jibekgu+`) training job and are left alone. Availability re-checked before each allocation. |

## Pins

| Item | Pin |
|---|---|
| Paper | arXiv:2606.25548v1 (ICML 2026), "Concept Removal for Frontier Image Generative Models" |
| Author code | `github.com/sprintml/block-concept-removal` @ `db4ab11a5c0cefbc71dfaedba0df3e86dd0319d5` (verified = remote HEAD) |
| Generator | `black-forest-labs/FLUX.1-dev`, revision `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` (reused verified local cache) |
| Native evaluator | `liuhaotian/llava-v1.6-vicuna-7b`, revision `deae57a8c0ccb0da4c2661cc1891cc9d06503d11` |
| LLaVA code | `github.com/haotian-liu/LLaVA` @ `c121f0432da27facab705978f83c4ada465e46fd` (v1.2.2.post1) |
| FLUX env | python 3.10.12, torch 2.4.0+cu121, diffusers 0.30.2, transformers 4.44.2, triton 3.0.0 (`env/pip_freeze_block.txt`) |
| LLaVA env | python 3.10.12, torch 2.1.2+cu121, transformers 4.37.2, tokenizers 0.15.1 (separate venv; LLaVA's pins are incompatible with FLUX's) |
| COCO assets | `annotations_trainval2014.zip` sha256 `031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009` |

Transcoder checkpoints are **not released** by the authors, so both are trained natively
here (authorised because the measured cost fits the budget: 0.16 + 0.16 GPU-h).

## Task freeze

- Development object: **Bears**. Confirmation objects: **Cats**, **Horses**.
  Confirmation objects are withheld from all prototype development and tuning. They are
  *not* unseen during representation training: the native context transcoder's corpus is
  the released 20-category prompt set, which **includes** Bears, Cats and Horses, and the
  pooled corpus is generic MS-COCO. Disclosed, as required.
- 80 frozen cases per object (`cases.json`, sha256 `2c6b4d08...`):
  - **40 target**: released template `"A {class} image in {theme} style."` x 10 themes x 4 seeds.
  - **20 related retained**: Dogs, Rabbits x 10 themes (seed 188).
  - **20 unrelated retained**: Architectures, Flowers x 10 themes (seed 188).
- Seeds: target `[188, 288, 588, 688]`, retained `188` — the first four / first one of the
  release's own default seed list `[188,288,588,688,888]`.
- Retained-category rule, fixed before any outcome: **related** = non-target categories in
  the same biological class as the target (all three targets are Mammalia), in
  `class_available` order, excluding `Human` and excluding the other two sprint objects;
  **unrelated** = first two `class_available` entries that are neither animal nor natural
  phenomenon (Flame/Sea/Waterfalls excluded as phenomena).
- The related/unrelated panels are identical for all three objects, so **M0 and T0 retained
  generations are produced once and reused** (recorded deduplication). Edited arms are
  per-object and never reused.

## Generation arguments (pinned, identical across every arm)

`height=1024, width=1024, num_inference_steps=28, guidance_scale=3.5,
max_sequence_length=512, torch_dtype=bfloat16`, output PNG.

Initial noise: `torch.Generator(device="cpu").manual_seed(case.init_seed)`, rebuilt per
image, where `init_seed = base_seed*1000 + theme_index*20 + class_index`. CPU generators
make the initial latent a pure function of the case, identical across arms and independent
of device and generation order. Generation order is the fixed case order in `cases.json`
and is recorded per image.

## Arms

| Arm | Definition |
|---|---|
| **M0** | pristine FLUX.1-dev, nothing replaced |
| **T0** | both native transcoders installed, **no** erasure modification |
| **N** | both native transcoders installed + native BLOCK erasure, from a **pristine T0 checkpoint reloaded per target** so edits never accumulate |
| prototype / control | only if justified after baseline validation; same harness, `--variant` |

Transcoders run in **fp32** (the dtype they were trained in and the dtype the release
constructs) inside a bf16 pipeline, with casts at the module boundary only. Casting the
trained module to bf16 would be a silent quantisation and is not done.

## Measurement

Two separate instruments, reported separately and never mixed:

**1. Native (benchmark) metrics** — the release's `scripts/eval/llava.py`, unchanged in
substance: a 20-way **forced-choice** object classification (and 10-way style), giving
UA / IRA / CRA. Patched only to additionally dump per-image raw answers.
*A forced-choice label cannot establish target absence*, so it is never used as the
presence gate.

**2. Binary target-presence procedure** — declared **adaptation**, frozen in
`code/score_presence.py` (`spec_version = presence-v1-frozen`):
- judge LLaVA-1.6-Vicuna-7B (pinned revision), `conv_mode=llava_v1` pinned explicitly;
- decoding `do_sample=False, temperature=0, top_p=1.0, num_beams=1, max_new_tokens=8`
  (deterministic, so one pass is reproducible — no retries, no re-asks, no tie-breaking);
- four frozen fields per image: **target presence**, **intended retained-object presence**,
  **severe rendering failure**, **uncertainty** (derived);
- exact question strings and the parser are in `code/score_presence.py` and are quoted in
  the final report;
- every raw answer preserved verbatim in JSONL.

Severe-rendering-failure rubric (frozen): yes only if essentially unusable — blank,
near-uniform colour, noise/static, garbled texture with no depicted scene, or almost
entirely corrupted. A coherent scene is **not** a failure even if unusual, heavily
stylised, empty of a particular object, or containing distorted faces/anatomy. Rendering is
audited across all 80 cases per endpoint under this one rubric.

**Scorer qualification (gate, run before any sprint image is scored):** >= 18/20
sensitivity AND >= 18/20 specificity, separately for each of Bears, Cats, Horses, against
20 target-present and 20 target-absent COCO val2014 images per object (10 of each negative
panel containing a *different* animal). Ground truth is COCO's own human instance
annotations. Uncertain/unparseable answers count as errors. The rule is run **once as
frozen**; it is not tuned, and replacements are not re-drawn until something passes.

Conservative handling, fixed now: for relative suppression, uncertain **M0** cases count as
**absent** and uncertain **edited** cases count as **present** (the bound least favourable
to the intervention). Unknown or unusable output never counts as successful erasure.
Classifier denominators are never mixed with visually corrected numerators.
CLIP similarity alone is not treated as a quality certificate; no small-panel FID is
reported as distributional quality.

My own image inspection is labelled an **agent judgment**, never independent human review.
Audits run in fixed case order with arm identity hidden where practicable.

## Baseline validation gates (all must hold before any prototype work)

1. Scorer controls pass (above), no unresolved material generated-domain scoring failure.
2. Correct modules replaced, native update applied, finite computations, intended tensors
   edited and unrelated tensors unchanged.
3. Reference storage separation, checkpoint save/reload agreement within a stated
   justified tolerance, explicit paired-noise checks.
4. M0 and T0 each >= **32/40** definitely target-present images per object.
5. M0 >= **16/20** definitely successful retained-object images in each retention stratum.
6. Installing T0 costs <= **10 points** vs M0 in either retention stratum and adds
   <= **10 points** of severe rendering failure on the fixed panel.

Failing any of these -> publish artifacts + exact failure, mark `BASELINE_UNQUALIFIED` or
`BLOCKED`, and stop. Native BLOCK is **not** required to hit any invented performance
threshold to count as a faithfully measured baseline.

## Prototype screening criteria (prospective, per confirmation object, conservative)

- related-object retention **>= +10 points** vs native BLOCK;
- relative target suppression **>= 75%** of M0, and **<= 5 points below** native BLOCK;
- unrelated-object retention **<= 5 points below** native BLOCK;
- severe rendering failure **<= 5 points above** native BLOCK.

Every measure is compared against the ordinary control. If the control achieves the same or
better trade-off, there is no demonstrated added value. If no comparable control point was
measured, the distinction is reported as unresolved. These are small pilot screens, not
powered noninferiority tests.

## Image allocation (core plan 1,200 of 1,360 allowance; 1,600 hard total)

| Work | Images |
|---|---:|
| M0: 3 x 40 target + 40 shared retained | 160 |
| T0: 3 x 40 target + 40 shared retained | 160 |
| N, Bears (development) | 80 |
| prototype dev, 2 settings | 160 |
| ordinary control dev, 2 settings | 160 |
| confirmation: 2 objects x 3 arms x 80 | 480 |
| **core** | **1,200** |
| smoke / calibration / recovery reserve | 400 |

The 120 external COCO calibration photographs are downloaded, not generated, and are
accounted separately from the generation budget.
