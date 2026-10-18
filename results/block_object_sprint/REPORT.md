# BLOCK object erasure on FLUX.1-dev — a small native-method reproduction attempt

**Status: `VALIDATED_PLATFORM_NO_PROTOTYPE`**

A runnable native BLOCK platform was built on FLUX.1-dev from the authors' pinned release,
both transcoders were trained natively, baseline validation passed, and native BLOCK was
measured on three frozen objects. **No prototype was developed**, because the frozen
prototype screen turned out to be unsatisfiable and because the presence instrument has an
unresolved generated-domain failure larger than the effect the screen must resolve. Both
reasons are evidenced below.

This is a **small native-method reproduction attempt**, not a reproduction of the paper's
benchmark numbers. One transcoder training seed, one generator, 80 frozen cases per object,
shared retained panels. Nothing here establishes independent replication.

---

## 1. Final status and strongest supported finding

**Strongest supported finding.** On FLUX.1-dev, with both native transcoders installed and
the authors' own erasure update applied from a pristine checkpoint per target, native BLOCK
suppressed the target strongly on all three frozen objects while leaving **related-object
retention exactly at the no-erasure (T0) level in every case**. Its measurable collateral
cost fell entirely in the **unrelated** stratum, and specifically on one category
(*Architectures*) in a small set of style contexts.

| object | target present at N (/40) | relative suppression vs M0 (conservative) | related retention | unrelated retention | severe rendering failure (/80) |
|---|---|---|---|---|---|
| Bears (development) | 5 | **87.5%** | 95% (= T0) | 90% (T0: 100%) | 0 |
| Cats (generalisation) | 1 | **97.5%** | 95% (= T0) | 95% (T0: 100%) | 0 |
| Horses (generalisation) | 0 | **100%** | 95% (= T0) | 85% (T0: 100%) | 0 |

Native (release) forced-choice evaluator, same images, reported separately:

| object | UA | IRA related | IRA unrelated | CRA (style) |
|---|---|---|---|---|
| Bears M0 / T0 / N | 2.5% / 10.0% / **95.0%** | 100% / 100% / **100%** | 100% / 90% / **75%** | 30% / 40% / 40% |
| Cats M0 / T0 / N | 0.0% / 5.0% / **97.5%** | 100% / 100% / **95%** | 100% / 90% / **90%** | 30% / 40% / 32.5% |
| Horses M0 / T0 / N | 0.0% / 7.5% / **100%** | 100% / 100% / **100%** | 100% / 90% / **80%** | 30% / 40% / 37.5% |

M0's measured FLUX style accuracy (CRA 30%) independently lands near the paper's own
Table 2 figure for Flux (31.8%), which is a mild consistency signal for the harness.

**The related/unrelated asymmetry is the finding.** Under both instruments, across all
three objects, erasure cost **zero** related-object retention: the single related failure
(Dogs / Bricks) is identical for all three targets and already fails at **T0**, so it is
caused by transcoder *reconstruction*, not by erasure. Every unrelated failure, by
contrast, appears only at N.

Per-case paired transitions (binary instrument), retained strata:

| object | related failures | unrelated failures (all `T0=yes -> N=no`) |
|---|---|---|
| Bears | 1 — Dogs/Bricks, **caused at T0** | Architectures/Pop_Art, Architectures/Watercolor |
| Cats | 1 — Dogs/Bricks, **caused at T0** | Architectures/Watercolor |
| Horses | 1 — Dogs/Bricks, **caused at T0** | Architectures/Impressionism, Architectures/Pop_Art, Architectures/Watercolor |

**Important caveat on this pattern.** The retained panels are *shared* across objects by
the frozen design (same 20 cases, same initial noise, same T0 checkpoint). The three
objects are therefore **not** independent observations of retained-image behaviour, and
"Architectures/Watercolor fails for all three targets" is one image failing three times,
not three independent failures. The protocol's warning applies directly: repeated contexts,
reused seeds and one transcoder training seed do not establish independent replications.

---

## 2. Baseline and scorer qualification, and fidelity limitations

### Baseline validation — gates 2–6 passed

Full evidence in [`BASELINE_VALIDATION.md`](BASELINE_VALIDATION.md).

- **Correct modules**: the two gates the paper's D.1 and the code specify —
  `context_embedder` `Linear(4096,3072)` and `time_text_embed.text_embedder`
  `PixArtAlphaTextProjection(768->3072->3072)`.
- **Training targets verified**: stored activation targets are **bit-identical**
  (max abs diff `0.000e+00`) to the live original modules recomputed on a random sample.
  321,920 context token-pairs; **330,000 pooled pairs, exactly E.1's number**.
- **Native update**: applied through the authors' own code. Writes 57 unique rows of
  65,536 (context) and 32 of 12,288 (pooled) for Bears; all finite; encoder weights,
  encoder bias and decoder bias **bit-identical** afterwards.
- **Save/reload**: bit-identical outputs from two independent loads (stated tolerance:
  exact equality — same bytes, dtype, kernel).
- **Storage separation**: erasure runs on in-memory modules reloaded from a pristine
  checkpoint per target; on-disk checkpoint verified unchanged after all three.
- **Paired noise**: per-case CPU generator seeded from `cases.json`, rebuilt per image,
  verified bit-identical and identical across arms by construction.
- **Gate 4** M0 and T0 target present: **40/40 for every object** (threshold 32).
- **Gate 5** M0 retained: **20/20 in both strata** (threshold 16).
- **Gate 6** T0 substitution cost: related **-5.0 pts**, unrelated **0.0 pts**, rendering
  **0.0 pts** (threshold 10).

### Scorer qualification — real-image controls passed, generated-domain failed

**Passed (frozen rule, run once, not tuned):** 120 COCO val2014 photographs with ground
truth from COCO's own human instance annotations — **20/20 sensitivity and 20/20
specificity for each of Bears, Cats and Horses**, including 0/10 false positives on the
related-animal negatives, and zero uncertain answers.

**Failed on the generated domain.** Auditing all 40 native-BLOCK Bears images, the agent
finds **0 bears**; the frozen scorer reports **5**. On an arm-blinded 20-tile sheet the
agent agreed with the scorer 15/20, and **all 5 disagreements were exactly the residual-bear
calls**. The failure mode is stylised cartoon quadrupeds (dogs, foxes, cats) and crowded
Byzantine / Van Gogh scenes reading as "bear". The native forced-choice evaluator shows the
same weakness independently (2/40 N/Bears labelled "Bears", both cartoon animals).

The failure is **specific and one-directional**: in a second blinded audit the scorer was
correct on 8/8 — the single Cats residual is a genuine residual cat, and all three
Architectures losses are genuine (the T0 counterparts show the buildings intact; at N they
become a landscape, a pop-art face and a cartoon fox). So the error concentrates on *bear*
presence calls in stylised images, and it **over-reports residual target**, i.e. it
**under-reports suppression**. No effectiveness number here is inflated by it.

All visual judgments above are **agent judgments**, not independent human review. Raw
answers for every image are published; the scorer's numbers are reported as measured and
the audit is reported separately. Visually corrected numerators are never substituted into
classifier denominators.

### Fidelity limitations

- **No transcoder checkpoints are released**, so both were trained natively here. One
  training seed. They are *a* native instantiation, not the authors'.
- **EMA discrepancy (unresolved)**: E.1 states an EMA of parameters with decay 0.999;
  `trainer.py` carries the constant but **never applies EMA anywhere**. The code was
  followed, since writing the EMA would be inventing absent algorithmic behaviour.
- **Learning-rate discrepancy**: E.1 states `1e-4`; the code uses
  `2e-4 / (n_latents/2^14)^0.5`, which equals 1e-4 for the context transcoder but
  **2.31e-4** for the pooled one. The code was followed.
- **The pooled corpus is reconstructed**: the release references
  `prompts/coco_train2014_30k_prompts.txt` but does not ship it. It was rebuilt by a
  documented seeded sample of MS-COCO train2014 captions (published, with provenance).
  The authors' exact 30k caption list is unknown.
- **Evaluator prompt discrepancy**: Figure 8 shows a *numbered* list; `eval/llava.py`
  builds a *lettered* list with a different question string. The code was followed.
- **Empty token**: the paper says the empty token; the code uses the literal prompt `"_"`,
  and takes only its first T5 token. The code was followed.
- **Native scatter ambiguity**, measured not fixed: the update writes 64 context rows of
  which only 57 are unique for Bears, so **7 writes are order-ambiguous** (`▁Bear` and `s`
  share 7 latents; PyTorch leaves the winner unspecified).
- **Two release bugs had to be fixed for the evaluator to run at all**, which suggests the
  released evaluator was not run end-to-end on the released generator's own filenames:
  `ctx_out` undefined in the context activation collector, and `parse_style_class`
  returning a space-separated style that is then indexed into an underscore-separated list
  (`ValueError: 'Pop Art' is not in list`). Full list: [`INTEGRATION_AUDIT.md`](INTEGRATION_AUDIT.md).
- The 80-case panels, 10 style contexts and 4 seeds are small. Style recognition is not
  treated as a gate or a novelty claim. No FID or CLIP-based quality certificate is claimed.

---

## 3. Native / control / prototype results, and why no prototype was permitted

Native results are in §1. **No control and no prototype were run.** The decision, committed
in [`PROTOTYPE_DECISION.md`](PROTOTYPE_DECISION.md) *before* any confirmation-object edited
output was generated or inspected, rests on two independent reasons:

**(a) The frozen retention screen is arithmetically unsatisfiable.** The committed first
criterion is "related-object retention improves by at least **10 percentage points** versus
native BLOCK". Native BLOCK's related retention is **100% (20/20) by the native metric** and
**95% (19/20) by the binary instrument** — headroom of 0 and 5 points respectively. And the
one binary related failure already fails at T0, so it is a transcoder-reconstruction defect
that no erasure-side intervention could repair. The screen was written expecting a
related-retention deficit that does not exist; the protocol forbids re-specifying a
committed gate after seeing results.

**(b) Measurement is not qualified at the required resolution.** The remaining criteria are
stated at ±5 points — two images in forty. The presence instrument's demonstrated error on
the edited arm is 5/40 (12.5 points). An unresolved material generated-domain scoring
failure blocks a prototype, and systematic disagreement capable of changing a gate means
measurement is unqualified.

Consequently the half-day prior-art allowance was **not** spent (it is allocated to the
mechanism of a prototype being pursued; none is), no second candidate was started, and the
confirmation reserve was not repurposed to another project.

### Hypothesis recorded, not tested

Clearly labelled as speculation, supported by no experiment here: native BLOCK's feature
identification takes the **union of all latents active for the target's tokens**, with no
selectivity test — 57 latents of 65,536 for Bears, chosen purely by top-k activation on the
bare concept string. Any such latent that is also load-bearing for unrelated concepts has
its decoder row redirected to the empty token, which would degrade unrelated content. A
selectivity filter (excluding latents whose activation on a reference corpus of unrelated
prompts rivals their activation on the target) is the obvious intervention family and
connects to existing sparse-feature selection work and to BLOCK's own bijection-pairing
ablation. Testing it requires a presence instrument qualified on the synthetic domain first,
since the effect (2 images of 20) is smaller than the current instrument's error.

---

## 4. Resources actually consumed, including failures

| | used | cap |
|---|---|---|
| GPU-device-hours | **3.977** | 48 |
| — integration / measurement qualification / baseline | 3.202 | 20 |
| — prototype / control development | **0.000** | 8 |
| — confirmation + evaluation | 0.774 | 20 |
| Newly decoded images | **563** | 1,600 |
| Elapsed hours at the prototype decision | **3.09** | 120 |
| Recorded failures | 2 | — |

Unsuccessful integration/debugging consumed **0.022 GPU-hours** of its 8 GPU-hour cap, well
inside the first 24 elapsed hours. No cap was approached; there were no overruns.

Images: M0 160, T0 160, N 240 (3 × 80), smoke 3. The 120 external COCO calibration
photographs were downloaded, not generated, and are accounted separately.

Deduplication, recorded: the related/unrelated panels are identical for all three objects,
so M0 and T0 retained generations were produced **once** (40 images each) and reused,
instead of 120 each.

**Failures (both in the ledger):**
1. First smoke attempt OOM'd in T5 self-attention (batch 64 × 512 tokens needs a 4 GiB fp32
   score tensor beside a resident 34 GiB bf16 pipeline). 0.005 GPU-h. No images decoded —
   the ledger was corrected after it had briefly over-charged 3 images.
2. All four first-attempt T0 jobs died in ~15 s: the harness `chdir`s into the vendored
   repo (the release resolves `prompts/` relatively), so relative `--transcoder-dir` and
   `--out-dir` resolved inside it. 0.017 GPU-h. Fixed by resolving both against the
   caller's cwd; the already-generated M0 images were valid and were relocated.

**Efficiency** (measured, [`evidence/efficiency.json`](evidence/efficiency.json)):

- **Cold preparation, one-time and shared by all targets**: 0.863 GPU-h — activation
  collection 964 s (context) + 250 s (pooled), training 1,324 s (context) + 567 s (pooled).
  Amortised over the three targets measured here: 0.288 GPU-h per target. A pretrained
  transcoder is not free.
- **Per-target edit**: 0.55 s feature identification + 0.037 s decoder update. Editing is
  effectively free once the transcoders exist.
- **Inference overhead vs M0**: T0 **+3.31%**, N **+2.79%** (14.39 / 14.32 vs 13.93
  s/image); the N-vs-T0 difference is within run-to-run noise.
- **Peak memory**: 33.81 GB (M0) → 35.69 GB with transcoders installed.
- Measured on one A100-40GB per job, batch 1, 1024×1024, 28 steps. No efficiency claim is
  derived from author timings or extrapolation, and none is a performance advantage claim.

An integration note on cost: collecting the 330,000 pooled pairs through the release's
`encode_prompt` path would have run T5-XXL 330,000 times for a vector that depends only on
CLIP. The CLIP-only path was verified **bit-identical** (`torch.equal`, max abs diff
`0.000e+00`) before being enabled, saving an estimated ~4.6 GPU-hours.

---

## 5. Where the evidence is

Everything below is in this directory, pinned at the publishing commit.

| what | where |
|---|---|
| working freeze (cases, seeds, evaluator config, labelling rules) | [`RUN_MANIFEST.md`](RUN_MANIFEST.md), [`cases.json`](cases.json) |
| baseline-validation record | [`BASELINE_VALIDATION.md`](BASELINE_VALIDATION.md) |
| prototype decision (committed before confirmation) | [`PROTOTYPE_DECISION.md`](PROTOTYPE_DECISION.md) |
| every release bug and paper/code discrepancy | [`INTEGRATION_AUDIT.md`](INTEGRATION_AUDIT.md) |
| runnable patched code + exact pins | [`code/`](code/), [`patches/`](patches/), [`env/`](env/) |
| raw evaluator answers, per-case labels | [`evidence/scores/`](evidence/scores/), [`evidence/native_eval/`](evidence/native_eval/) |
| scorer validation (120 COCO photos, raw answers) | [`evidence/scorer_validation/`](evidence/scorer_validation/) |
| agent audit sheets + keys | [`evidence/audit/`](evidence/audit/) |
| fixed-order grids of every scored image | [`grids/`](grids/) |
| resource / image / job / failure ledgers | [`evidence/ledgers/`](evidence/ledgers/) |
| full-resolution originals (560 PNGs) | GitHub release asset — see [`EVIDENCE_INDEX.md`](EVIDENCE_INDEX.md) |

Model weights and credentials are not published. The trained transcoders (2.0 GB) stay on
local scratch; their training config, verification numbers and the exact code and corpora
needed to retrain them are published.

---

## 6. One next scientific decision

**Qualify a target-presence instrument on the synthetic, heavily stylised domain before any
further erasure comparison on this benchmark is attempted.** The sprint's own controls were
perfect on 120 real photographs and still missed a systematic one-directional failure that
is larger than the effect sizes the frozen screens depend on. Until an instrument is
validated on generated, stylised images — not photographs — differences of a few images
between erasure methods on this benchmark are not measurable, whoever reports them.

No automatic follow-up is implied, and no further candidate, comparator or assessment cycle
is opened by this report.
