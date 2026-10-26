# PREREGISTRATION — R1: IMMA at the pinned author budget

**Declared 2026-10-26, BEFORE execution.** Nothing in this document may be edited after the first
result exists; corrections go in a dated amendment block at the end.

**Status when written:** NOT YET RUN. Blocked on milestone 1 (shared infrastructure), which is open
on D-047, D-048, D-059. This declaration exists so the configuration is fixed before the gate
clears, not after.

---

## 1. What this is, and what it is not

**It is:** an execution of the **pinned upstream IMMA code** at the **budget its own README
specifies**, on **our** generated dataset and **our** evaluation protocol.

**It is NOT:**
- **not a reproduction of the paper's reported results.** The dataset is ours (100 images generated
  by `upstream/IMMA/eval/text2image.py`, 20 selected), the evaluation panel is ours, and the
  attacker is ours. Official-code execution on our benchmark is a different claim from reproducing
  a published number, and this document will not be cited as the latter.
- **not a replacement for the historical run.** `REF_imma_vangogh` stays exactly as it is, under its
  existing label, with `BUDGET_5X.md` recording that it used **5.00×** the documented ascent budget.
  This run gets a **new identifier**; it does not overwrite, re-label, or supersede that artifact.
- **not a robustness claim.** See §5 for what the attack can and cannot establish.

**Fidelity classification:** *official-code execution, our dataset, our evaluation.* Not paper
reproduction, not a port, not a domain adaptation, not our own method.

## 2. Configuration — declared, with every adaptation labelled

**Upstream:** `upstream/IMMA` @ `2f55041141705031d503134dec22f674f09e21bf`, verified by
`git rev-parse` at run time (the launcher aborts on mismatch).

**Base model:** `CompVis/stable-diffusion-v1-4`, revision
`133a221b8aa7292a167afc5127cb63fb5005638b`, variant `fp16`. Both revision **and variant** are
recorded, because main vs fp16-variant differ in 686/686 tensors at relative **2.074e-04** and the
variant alone is enough to be mistaken for a training signal.

**Initial checkpoint:** `upstream_runs/imma/diffusers-VanGogh-ESDx1-UNET.pt`, sha256
`8881d70b1bd376c5532e933f60c5a78fc76eb63e153edfbfaaf6d649106b167e`, 3,438,307,109 bytes, 686 fp32
tensors, key set exactly equal to the SD-1.4 UNet. **Provenance:** the URL is supplied verbatim by
the pinned README (`upstream/IMMA/README.md:44`), which itself labels the file *"Only for
illustration purposes"*. **Its training provenance is NOT established** — EMA-vs-non-EMA remains an
untested hypothesis. This run does not depend on resolving that, and will not be read as resolving it.

**Concept:** `van_gogh`. **Adaptation, labelled:** the README's own example is Van Gogh, and
`erasing.baulab.info` hosts ESD checkpoints for **art concepts only** (`/object/` returns 404), so
an object concept was not available at this pin. `van_gogh` is one of our four evaluated concepts.

**Training arguments — the README command verbatim** (`upstream/IMMA/README.md`):

| argument | value | source |
|---|---|---|
| `--max_train_samples` | **20** | README |
| `--num_train_epochs` | 50 | README |
| `--train_batch_size` | 1 | README |
| `--learning_rate_lora` | 1e-04 | README |
| `--learning_rate` | 1e-05 | README |
| `--lr_scheduler` | constant | README |
| `--lr_warmup_steps` | 0 | README |
| `--resolution` | 512 | README |
| `--random_flip` | set | README |
| `--seed` | 42 | README |
| `--caption_column` | prompt | README |

**Resolved budget, derived from source** (`defend_text_to_image_lora.py:710-716, 720-722, 750-752`):
with `bs=1`, `len(dataloader) = N_subset = 20`, so `max_train_steps = 50 × 20 = 1000` and
**ascent updates = 500**. The historical run omitted `--max_train_samples`, giving `N=100`,
5000 steps and **2500** ascent updates — the 5.00× factor.

**Dataset:** 100 images from `eval/text2image.py`; the selected subset is
`shuffle(seed=42).select(range(20, 40))` — **not** "first 20". That resolved subset was reproduced
twice and contains **no blank image**; the three NSFW-blanked frames (`7.png`, `29.png`, `41.png`)
fall outside it (`29.png` sits at shuffled position 1, inside the discarded `[0,20)` window).
`metadata.csv` is **not** rewritten — it is the input the historical artifact came from — so
`metadata_noblank.csv` and `blank_audit.json` sit beside it for opt-in use.

## 3. Evaluation — declared before any result

**Panel:** the 12-concept DEV set, scored by the shared closed-set CLIP classifier
(`scripts/gen_eval.py`, sha256 `6f978e60…`, unmodified), using `LABELS` for CLIP text and `LAB_KEY`
for record identity. 15 images per concept, 3 prompts × 5 images, seed 0, 25 steps, guidance 7.5.

**Metrics reported, with their denominators stated:** `target_suppression_acc` (van_gogh, n=15);
`utility_all_nontarget` (n=11, excludes the active target); `legacy_retain8` (n=8 — **note van_gogh
is not in that panel, so for this target retain8 excludes the target by construction**, unlike the
tench case in D-023). Per-image predicted labels and per-label scores **will be saved**.

**Post-adaptation evaluation — the authors' own axis, and the one that tests their claim.** IMMA's
claim is about resistance to **re-learning**, and the authors evaluate **after** an adaptation
attempt. So the primary measurement is: run the adaptation/attack, then evaluate. A standalone
evaluation of the immunised model is an **additional** measurement we introduce; the authors never
report one, and it is reported separately rather than as a test of their claim.

**Attack settings:** `scripts/attack_relearn.py` (unmodified, sha256 `ec5fe224…`), concept
`van_gogh`, seeds {0, 1, 6}, lr 1e-5, steps 300. **Attacker-lr sweep:** additionally lr {5e-5, 1e-4}
at 25 steps, because FINDING78 showed a single attacker lr is insufficient to characterise
robustness at a fixed budget.

**Validity gate:** `T_rise ≥ 0.20` on the 24-image T panel — which correctly accepts 5/24 = 0.2083
and rejects 4/24 = 0.1667. This is the historical rule, unchanged. Gate failures are recorded as
**inconclusive**, never as robustness.

## 4. Acceptance criteria — declared before execution

This run **passes** if, and only if:

1. The upstream pin matches and the launcher completes with a truthful non-failure status.
2. A checkpoint is written, **reloaded**, and verified: keys, shapes, finite values, tensor equality
   at the intended serialization precision, and any fp16 export checked against the explicitly
   converted source tensors.
3. The resolved budget is **recorded and equals 500 ascent updates**, with executed updates
   distinguished from configured and skipped steps.
4. The split/provenance registry returns a clean verdict **and the runner enforces it**.
5. Every reported metric is present, numeric, finite, and in range, with its denominator validated.
   A missing or malformed field is a failure, **never a 0.0000**.

It **does not** pass merely because the process exited 0, because a test suite passed, or because a
file exists.

## 5. What a result here can and cannot support

**Can:** whether the pinned IMMA code, at its own documented budget, produces an immunised model
that resists our relearning attack on our panel — and how that compares to the 5× arm at matched
evaluation.

**Cannot:**
- **No claim about the paper's reported numbers.** Different dataset, panel and attacker.
- **No "destroys/preserves utility" claim from the closed set alone.** The scorer is a 12-way CLIP
  argmax with **no reject option**, validated only on undamaged in-distribution outputs. A low
  retain number licenses *"closed-set retain accuracy collapsed to N/120"*, not *"the model is
  destroyed"* — and symmetrically a high target number does not license *"clean erasure"*. Saved
  images plus an independent semantic assessment are required for either.
- **No upper-bound framing from FINDING65.** Its ratios are **withdrawn and not recoverable**
  (regeneration does not reproduce the historical images; six arms' checkpoints are deleted), so it
  must not be used to describe any development result here as a proven upper bound.
- **No provenance inference** about the ESD checkpoint.
- **No mandate claim.** Development data throughout. The FINAL split is **inspected development
  data** and is not used here; no confirmatory protocol has been frozen, so no confirmatory claim is
  available regardless of outcome.

## 6. Artifacts to be saved

Model checkpoint; full launcher log; the generated images (all of them); per-image predicted labels
and per-label scores; the evidence record binding loaded-state identity, executed updates and
reload/export verification to the run configuration and source revision; a completion manifest
published **only after** every required artifact validates; and the resolved dataset manifest with
fingerprints.

---

## Amendments

*(dated entries only; the body above is frozen)*
