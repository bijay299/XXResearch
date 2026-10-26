# Baseline availability — verified per method, and the earlier blanket claim retracted

**Supersedes** the availability claims in `ERRATUM_r2` C19, `AUDIT_CLOSURE_r2` (lines 37, 51, 199,
224), `FINDING43` line 106, `FINDING59` line 70, `PREREG_H81` line 6 and `T2I_REPORT_r4`.

**The retracted claim, verbatim:** *"**Zero of the five competing defences have usable released
code** for this threat model. Every one of them would be a from-scratch port."*
(`ERRATUM_r2` C19.)

**That is false.** Three of the five have released code, and I verified it by downloading and
reading the files at commits that predate our snapshot — not by checking whether a link exists.

---

## 1. The four questions that were being collapsed into one

The old claim conflated things that need separate answers. This table keeps them apart.

| method | 1. released code exists | 2. environment compatible | 3. assets present | 4. domain adaptation needed | our status |
|---|---|---|---|---|---|
| **IMMA** (2311.18815) | **YES** — `amberyzheng/IMMA@2f55041`, 17 Jan 2025. `environment.yaml`, `train/defend_text_to_image_lora.py`, `eval/text2image.py`, full README commands | **UNTESTED** — ships a conda env we have not built | needs ESD-erased weights; README links `diffusers-VanGogh-ESDx1-UNET.pt`, downloadable | **NONE** — it targets `CompVis/stable-diffusion-v1-4`, **our exact pinned model** | our port ran; **the released code has not been run** |
| **Meta-Unlearning** (2410.12777) | **YES** — `sail-sg/Meta-Unlearning@5a2a760`, 22 Apr 2025. `src/meta.py`, 1,087 lines, read in full | **UNTESTED** | needs `./dataset/{hrm,irt,tgt}` image folders, not in the repo; must be generated | **NONE** — SD T2I | variant-A port ran; **the released code has not been run** |
| **From Dormant to Deleted** (2505.22310) | **YES** — `shoaibahmed/vision_relearning@7b9a1cc`, 30 May 2025. `vision_unlearner.py`, 1,935 lines, read in full | **UNTESTED** | CIFAR-style datasets, standard | **YES, SUBSTANTIAL** — vision **classifiers**, example-level unlearning, cross-entropy retain objective. The *method* ports (see `erase_wdr_v2.py`); the *experiments* do not | our arm implemented the **opposite direction** (`ERRATUM_wsr_direction.md`); faithful version written, **not yet run** |
| **GIFT** (2507.13598) | **NOT FOUND** | — | — | none (SD T2I) | our port ran; availability stands as not found, not as "none released" |
| **FRAG/FRP** (2608.25429) | **NO** — repository is a README stub (`ERRATUM_r2` C19's original finding, which was correct **for this method**) | — | — | **YES** — the paper's experiments are on LLMs; the pruning method ports | our port ran |

**So: 3 of 5 released, 1 not found, 1 confirmed stub.** The C19 correction was right about FRAG/FRP
and was then over-generalised to the other four. The generalisation was the error.

## 2. What "UNTESTED" means here, precisely

It means **I have not attempted to run any of the three released implementations.** No conda
environment built, no dependency error recorded, no execution log. So I am not entitled to say any
of them is unusable in our environment — and the previous claim's word *"usable"* was doing exactly
that work without evidence.

The correct next record is an **actual failed execution or dependency error**, or a successful run.
That is experiment **E7** in `CORRECTION_REGISTER_r5.md` §B.10. IMMA is the cheapest first step: its
`environment.yaml` is self-contained and its target model is already in our HF cache.

## 3. What this does and does not change

**Changes:**
- "every one of them would be a from-scratch port" is withdrawn. For IMMA and Meta-Unlearning a
  reproduction was available and we wrote ports instead. That was a methodological choice made under
  a false premise, and it is the reason both ports diverge from their papers
  (`ERRATUM_meta_objective.md`, and IMMA's α which the paper leaves unassigned while the released
  code sets `--learning_rate_lora 1e-4 --learning_rate 1e-5`, batch 1, 50 epochs).
- Every "no released code" sentence in the six documents listed above is superseded by this file.
  The originals are **not edited**; they stand as written with this correction attached.

**Does not change:** any measured number. Our ports are what they are and their results stand as
*our ports'* results.

## 4. Separation of record types, kept explicit from here on

The audit asked that these not be mixed. They are now four distinct categories, and the generated
tables label every arm with one:

| category | meaning | what exists today |
|---|---|---|
| **official reproduction** | the authors' code, run by us | **none** |
| **project reimplementation** | our port from the paper | FRP, GIFT, IMMA, Meta variant A, UCE |
| **domain adaptation** | the method moved to a new domain by us | WSR-anchor (ours, wrong direction), WDR-v2 (faithful, unrun) |
| **new variant** | our own design | UTRAP, UTRAP-v2, undo/damage composites |

## 5. FRP's threat model was misdescribed — correction

`PREREG_H66` §5a and `FINDING43` §2 describe the FRP paper's attacks as **retain-only**, and explain
our reversal by our attacker newly supplying **direct target gradients**.

**The paper evaluates three attack settings**, per its Appendix B.2 ("Three relearning attacks"):

> 1. **retain attack** — "adversary fine-tunes on the retain split only"
> 2. **forget attack** — "adversary has direct access to the forget split itself"
> 3. **forget+retain attack** — "full white-box adversary that fine-tunes on the union"

and Appendix C.3 varies **optimiser** (AdamW, SGD, SGD+momentum, Adafactor, Adagrad), **learning
rate** (5e-6 to 1e-4) and **horizon** (1–10 epochs).

So "retain-only versus direct-target gradients" is **not** the difference between their result and
ours, and that explanation is withdrawn. The differences that remain, none of them yet measured:

- **domain** — their experiments are LLMs; ours is SD-1.4;
- **calibration data** — our port calibrates importance on **random latents across timesteps**
  rather than on-distribution noised image latents. A plausible port sensitivity, and worth
  measuring; not proof our number is wrong;
- **data construction, attack optimisation and evaluation metric** all differ.

The predeclared λ remains good practice given the paper's reporting gap (λ appears in Algorithm 1
without a numeric value in the main text, while β = 0.05 and ρ = 3% **are** reported). It does not
resolve the transfer questions above.

## 6. Threat models are not ordered by parameter count

Kept from the audit and reaffirmed:

- **AdvUnlearn**'s published robustness setting is **prompt-space**. Our attacks assume **weight
  access**. The counterexample remains useful; it is not a refutation of their claim
  (`ERRATUM_r2` C15).
- **Textual Inversion** learns a **new token** and evaluates on a **changed prompt surface**. It is
  not automatically a reproduction of a bounded prompt-embedding attack, and its evaluation surface
  must stay labelled.
- Full-UNet (859.5M), DreamBooth (859.5M + prior), LoRA (0.399M) and TI (768 scalars) differ in
  **data access, parameterisation, evaluation surface and compute**. Trainable parameter count is
  **not** a total ordering of practical attack strength, and `ERRATUM_lora_adversary_scope.md`'s
  "a 0.046% LoRA suffices" should be read as "this particular small-parameter attack sufficed on
  these checkpoints", not as a strength ranking.
