# AMENDMENT 01 — targeted competitor check, and stop of algorithm development

**Append-only.** This amendment is frozen as published; later changes go in a further
amendment, not here.

| | |
|---|---|
| **Amendment received (UTC)** | 2026-10-18, during the pilot's setup/profiling phase |
| **Pilot start (UTC)** | **2026-10-17T22:19:05Z** — unchanged |
| **Deadline (UTC)** | **2026-10-31T22:19:05Z** (14 calendar days from the existing start) — unchanged |
| **Stage when it arrived** | Setup + native profiling complete; M0 panels complete; **no prototype trained, no development setting evaluated, no edited-model output generated, no held-out run started, no held-out outcome inspected** |

---

## 1. Actual stage and completed work reused

Nothing was restarted. All of the following existed before this amendment and is reused:

| completed | detail |
|---|---|
| Environment | isolated venv on `/data`; stock diffusers 0.32.0 + the author's vendored peft 0.13.2 |
| Author source pin | `tomguluson92/EraseAnything` @ **`67d8c0bd5b153ff0e394b45e893b6671bc40e30b`** |
| Model pin | `black-forest-labs/FLUX.1-dev` @ **`3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`**, diffusers-format, 32 GB, gated access verified under HF account `CaptIce` |
| Documented patches | 4 fixes, all minimal and recorded (`code/apply_ea_patch.py`, `RUN_MANIFEST.md` §6) |
| Frozen cases | 360 cases (3 concepts × 120), seeds, label sets — `frozen/cases.json`, sha256 `89673233d259…` |
| Training images | 30 (10/concept), author's `instance_prompt`, frozen seeds |
| **M0 panels** | **360 images, all 3 concepts, scored** |
| **Native reference** | **executes**; 6-step profile, 435 forwards, peak 38.4 GB |
| Engineering measurements | 3 profiled configurations (native / +shared-attn / +early-stop) |

**No edited-model held-out result has been inspected. The held-out set is untouched.**

## 2. Budgets — carried forward, not reset

Measured from the append-only ledger (`measured/ledger.jsonl`), which charges every job
including failures.

| | consumed | cap | remaining |
|---|---|---|---|
| setup/feasibility/debug | **0.49** | 8 | 7.51 |
| development | **0.57** | 24 | 23.43 |
| held-out / eval / recovery | **0.92** | 64 | 63.08 |
| **TOTAL GPU-hours** | **1.97** | **96** | **94.03** |
| newly decoded images | **441** | 3000 | 2559 |
| auxiliary LLM requests | **3** (2 failed 401s + 1 successful) | 20 | 17 |

Charged failures: 0.695 GPU-h to an aborted M0 attempt that thrashed CPU (255 OMP threads
per process, load 432 on 255 cores, GPU util 12–39%); 0.026 GPU-h to two crashed trainer
starts. No work is assumed free. One external user's process was seen on GPU 2 and was
avoided, never preempted.

## 3. Targeted source findings

Primary papers and author code inspected. No GPU work was used; GPUs were idle during this.

### EraseAnything++ — `arXiv:2603.00978v1`, code `nxjiang-jnx/EraseAnything-PlusPlus` @ `f73e5c1e5bb51b6a24b57039660e06dcd0ecec82`

- **Supported model/task/intervention:** FLUX.1-dev, LoRA on `add_q_proj`/`add_k_proj` in
  dual-stream blocks — **identical targets to original EraseAnything**.
- **§III-C "Implicit Efficient Gradient Surgery":** one combined backward pass; the dual
  variable is updated by a *finite difference of forward losses*,
  `λ_{t+1} = λ_t − β_t δ̃_t`, `δ̃_t = (1/α_t)(ℒ_p(θ_t) − ℒ_p(θ_{t+1})) + ε_t`.
- **VERIFIED, not copied — the lead was right.** Its `image/utils/calc_loss.py` is
  **byte-identical** to original EraseAnything's `utils/calc_loss.py` (diff = 0 lines). It
  does **not** touch `calculate_lower_loss`. What it changes is the training loop: the
  original *alternates* upper/lower on `global_step % 2`, while EA++ computes **both**
  `forget_loss` and `retain_loss` **every step** and combines them
  (`eupmu.get_weighted_loss`), plus periodic extra loss evaluations for the surgery
  (`if global_step % eupmu_update_freq == 0`). Since the lower loss is the expensive one
  (measured below: 19.91 s vs 5.96 s), **"one backward pass" increases per-step cost here
  rather than reducing it.**
- **Cost evidence:** *none*. The paper reports **no** wall-clock, GPU-time or FLOP
  comparison against original EraseAnything. Its efficiency claim is in principle only.
- **Deletion/preservation/robustness:** nudity 182 parts; FID 21.67 (vs 21.32 base); CLIP
  30.35 (vs 30.87); artistic Ha 6.61; attack success 68.80% vs EraseAnything's 71.55%.
- **Comparability limits:** rank, resolution, GPU and training time unreported; auxiliary
  GPT-4o/NLTK costs unquantified.

### BLOCK — `arXiv:2606.25548v1`, code `sprintml/block-concept-removal` @ `db4ab11a5c0cefbc71dfaedba0df3e86dd0319d5`

- **Intervention:** a sparse *transcoder* replacing a bottleneck layer; removal edits
  decoder columns so a target concept's active latents reproduce the empty-token output.
  Entirely different paradigm — no LoRA, no ESD, no InfoNCE.
- **Cost regime (Table 11, measured by the authors), FLUX:** upfront transcoder training
  **5428 s**, per-target removal **1.020 s**, **per-image inference overhead 0.43 s**,
  memory 1.9 GB, storage 1.93 GB. Upfront is **per model**, not per concept.
- **Direct comparison with EraseAnything, §5.5 Table 7 (FLUX nudity):** EraseAnything
  129/48/22/**199**, FID **21.75**; BLOCK 108/35/10/**153**, FID **22.78**. So BLOCK is
  better on detection counts and slightly worse on FID.
- **FLUX style removal:** UA 88.60%, **IRA 36.10%**, CRA 96.40%.
- **Not blanket dominance** (the lead warned against this and the warning holds): BLOCK's
  FID is worse, its style IRA is low, and it imposes a *permanent* 0.43 s/image inference
  tax that EraseAnything's merged LoRA does not. The regimes differ — BLOCK amortises a
  per-model cost and then removes targets almost free; EraseAnything pays per concept and
  nothing at inference.
- **Hardware for the timings is not specified**, so Table 11 is not directly commensurable
  with our measured GPU-device-seconds.

### ARC — `arXiv:2608.07620v1` (mechanism-specific check only)

Checked because our mechanism touches sampling. ARC does **difficulty-aware reallocation**
of per-concept sampling share and loss weight (Eq. 8–10) with EMA-smoothed histories. It
does **not** share trajectories/prefixes, evaluate several conditionings at a common latent,
or cache/refresh attention maps. **No overlap.** This was an overlap check only; no
FlowErase-OPD reproduction and no multi-concept expansion.

Neither newer paper's SOTA claim is taken as an established ranking, and original
EraseAnything is not treated as SOTA by default.

## 4. Our candidate, and the distinction

**Measured bottleneck (not assumed).** 6-step native profile, development concept:
`step_lower` **19.91 s** vs `step_upper` **5.96 s** — the lower step is **77%** of step
time. **435** transformer forwards in 6 steps, exactly matching the static prediction
(3×32 + 3×113). Within a lower step, **112 of 113** forwards are the K+1 = 4 full 28-step
no-grad trajectories, from each of which **one** attention map at a single random index is
consumed and the other 27 discarded (`calc_loss.py:572, 595, 598`).

**The distinction, in the required form:**

> Our mechanism changes **where the InfoNCE attention maps are evaluated** — one shared
> truncated sampler prefix with K+1 single-forward reads at a **common** latent `z_j*`,
> instead of K+1 independent j-step trajectories each read at its **own** `z_j^(c)`.
> The closest method already does: **EraseAnything++** changes the bi-level *update rule*
> (both objectives every step + implicit finite-difference gradient surgery) while leaving
> `calculate_lower_loss` **byte-identical**; **BLOCK** removes per-concept optimisation
> altogether (transcoder edit, ~1 s/target). The remaining difference is that **neither
> touches the dominant InfoNCE trajectory cost inside EraseAnything's lower level**, and we
> would test its value through arm C vs arm B on the frozen matched panels.

**So the mechanism is not covered.** It is nonetheless **not developed further**, for two
reasons that are about value and measurability rather than overlap:

1. **The ordinary control already clears the pilot's bar.** Two *exact* redundancy
   removals — materialising the explicit softmax only in the block whose maps are consumed,
   and stopping each InfoNCE trajectory at the consumed index — measure **35.0%** less wall
   time (82.65 s → 53.72 s over 6 steps; 435 → 311 forwards; peak 38.4 → 35.9 GB). The
   screening rule asks for ≥25%. Per the brief, when ordinary engineering explains the
   gain, **this is an engineering result, not a contribution**, and arm C would have to beat
   35% to add anything.
2. **The effectiveness half cannot be measured at all** (§5), so even a larger cost win
   could not be paired with the required deletion/preservation/robustness evidence.

**Decision: stop algorithm development.** No arm C is trained; no replacement algorithm is
started; the two-setting allowance is left unspent, not reassigned.

## 5. Measurement blocker — the frozen panel does not qualify

The M0 adequacy gate (≥80% target presence on canonical **and** perturbed) **fails for all
three concepts** under the frozen scorer:

| concept | canonical | perturbed | qualified |
|---|---|---|---|
| Frida Kahlo | 32.5% | 20.0% | **NO** |
| Katsushika Hokusai | 32.5% | 25.0% | **NO** |
| Vincent van Gogh | 15.0% | 20.0% | **NO** |

Generic-object retention scores **100%**, so the pipeline and scorer work for object
content. Three bounded diagnostics were run and **all** are reported
(`measured/scorer_adequacy_diagnostic.json`): 1024×1024 regeneration (10% / 45%), null-label
removal (35/55/35%), and an alternative template (47.5/45/22.5%). None reaches 80%.

**An agent visual check corrects an earlier working note of ours.** The fixed-order M0
contact sheet for Van Gogh is unmistakable Van Gogh pastiche. The failing component is the
**scorer**, not the generator; an earlier note in this pilot attributed the low rates to
FLUX not rendering style at 512 px, and that attribution is **withdrawn**. CLIP ViT-L/14
zero-shot artist-identity discrimination is simply not sensitive enough here — the
style-erasure literature uses a *trained* style classifier for this reason. This is a
methodological error of ours in choosing the instrument, disclosed rather than repaired by
searching for a variant that happens to pass. (Agent judgement, not independent human
review.)

Consequence: with `p_M0` at 0.15–0.33, `S = 1 − p_edited/p_M0` has a tiny, noisy
denominator — exactly what the 80% gate exists to prevent. **The ≥75%-suppression and
±5-percentage-point rules cannot be evaluated.** The concepts are disclosed as unqualified;
none was quietly replaced.

## 6. Comparator arm — authorised, and deliberately NOT taken

This amendment permits at most one contemporary comparator arm. **None is run.**

Feasibility was checked, not assumed: a full arm needs 480 images; 441 of 3000 are spent,
so images would fit, and 94.03 GPU-h remain. **It is not blocked by budget — it is blocked
by interpretability.** A comparator arm exists to contextualise a joint cost *and*
effectiveness claim; with no qualified panel, it would yield cost numbers that cannot be
paired with valid deletion/preservation conclusions, spending confirmation reserve on an
uninterpretable comparison. The brief says not to force it, and we do not.

**Serious unresolved challenge from the untested alternatives, stated plainly:** BLOCK
reports per-target removal on FLUX.1-dev in **1.020 s** after a one-time 5428 s transcoder
training, against our **measured** native EraseAnything cost of **2.166 GPU-device-hours
per adapter** — roughly three orders of magnitude per target, in a different cost regime,
with a direct EraseAnything comparison in its Table 7. Our 35% engineering saving does not
engage that gap. **Any advantage of our engineering result over current methods is
unverified**, and no measured comparison against EA++ or BLOCK exists in this record.

## 7. Prospective rules — recorded, unused

Had a comparator been run, these were to be frozen first: same pinned FLUX.1-dev revision,
same 120-case panels and seeds, same hardware/placement, cold **and** amortised costs
reported separately, upfront transcoder cost attributed per model and disclosed as
amortised, and no claim of advantage without matched measurement. They are recorded here
for the record and were **not** exercised.

## 8. Held-out exposure

**Zero.** No edited model was generated, scored or inspected on any held-out concept. The
held-out panels (`Vincent van Gogh`, `Katsushika Hokusai`) retain their original status; M0
reference images exist for them, which is baseline measurement, not an edited-model
outcome. No prototype was retuned, because no prototype was trained.

## 9. What this amendment changes

- Algorithm development: **stopped** (§4). Arm C not trained.
- Comparator arm: **not taken** (§6), with the reason recorded.
- Freezes: **unchanged**. Concepts, cases, seeds, scoring rule, quality rule and the
  25% / 5-percentage-point screening rules are preserved exactly as frozen; the frozen
  scorer is reported as *failed*, not rewritten.
- Budgets, start timestamp and deadline: **unchanged**.
