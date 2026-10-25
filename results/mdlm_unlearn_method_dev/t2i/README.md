# Text-to-image concept erasure under a relearning adversary

**Separate research line.** Stable Diffusion v1.4 (pinned revision
`133a221b8aa7292a167afc5127cb63fb5005638b`), unrelated to the RECE/FLUX and MDLM lines on other
branches of this repository.

**Read this first:** [`docs/T2I_REPORT_r3.md`](docs/T2I_REPORT_r3.md) — the current report.
`T2I_REPORT_r2.md` carries a SUPERSEDED banner and is kept for history, not as current fact.
[`docs/ERRATUM_r2_claim_corrections.md`](docs/ERRATUM_r2_claim_corrections.md) (C1–C19) lists every
claim this line has had to correct or withdraw, and
[`docs/AUDIT_CLOSURE_r2.md`](docs/AUDIT_CLOSURE_r2.md) records the 12-issue audit remediation.

## The question

Does concept erasure in a diffusion model *remove* a capability, or only suppress its expression?
The test is a **relearning adversary**: fine-tune the erased model on a small number of images of
the erased concept and measure whether the concept returns.

The protocol is frozen in [`docs/PROTOCOL_FROZEN_v1.md`](docs/PROTOCOL_FROZEN_v1.md): a symmetric
six-point grid (seeds {0,1,6} @ lr 1e-5, {0,1} @ 5e-5, {0} @ 1e-4), 300 steps, 16 self-generated
images, reporting **max-over-valid H** across the grid.

### Two rules that govern every number here

- **The validity gate (`FINDING20`, enforced in `scripts/attack_relearn.py`).** An attack must raise
  **T** — accuracy on the prompts it was *itself trained on* — by ≥ 0.20 before a flat **H** counts
  as robustness. An attack that failed to learn anything is not evidence that a defence held. Failed
  attacks are reported as failed attacks.
- **The ratchet (`FINDING35` §4).** `max-over-valid` is monotone non-decreasing in the number of
  attacks run, so **every sub-ceiling number in this line is provisional** until the adversary has
  been pushed harder. No "resists" claim here is final.

## The result

**Eight defences, ours and published, are fully reversed to the base model's ceiling** (H = 1.000)
by 300 gradient steps on 16 images: UCE, ESD, official AdvUnlearn (the robust SOTA), damage+distil,
the `frag` Fisher allocation, our FRP port, and both arms of our Meta-Unlearning port.

**The mandate's bar is not met.** No arm — ours or published — has reached retain ≥ 0.85 with
max-over-valid H ≤ 0.40.

Headline findings:

| finding | what it establishes |
|---|---|
| [`FINDING34`](docs/FINDING34_robust_sota_fully_reversed.md) | official AdvUnlearn, the robustness SOTA, fully reversed |
| [`FINDING35`](docs/FINDING35_no_usable_arm_resists.md) | no usable arm resists; states the ratchet rule |
| [`FINDING43`](docs/FINDING43_frp_fully_reversed.md) | FRP (2608.25429), built *for* relearning robustness, fully reversed |
| [`FINDING44`](docs/FINDING44_mask_refill_refutes_my_mechanism.md) | my own pre-registered mechanism **refuted** by its own declared diagnostic — the concept returns through the weights pruning *left alone* |
| [`FINDING54`](docs/FINDING54_three_adversary_classes.md) | the four-adversary × three-defence × five-concept matrix (full-UNet, LoRA, Textual Inversion, DreamBooth) |
| [`FINDING56`](docs/FINDING56_meta_unlearning_self_destruct_fires_on_normal_use.md) | Meta-Unlearning (2410.12777): the self-destruct trigger fires on *normal use* and buys **zero** robustness over its own ablation |

**Eight claims of mine dissolved under their own controls** and are listed in `T2I_REPORT_r3` §6.
That section, the erratum, and `FINDING44` are the honest spine of this line.

## Scope limits that apply to everything here

- **All five published competitors are our ports.** None of the five has usable released code
  (`ERRATUM_r2` C19). Any disagreement with the authors' numbers is attributable to our port or the
  domain change, **not** to their methods. Three remain unrun: IMMA 2311.18815, GIFT 2507.13598,
  From Dormant to Deleted 2505.22310.
- **No never-trained oracle exists**, so *reactivation of residual memory* and *relearning from the
  attacker's data* are not distinguishable here (`ERRATUM_r2` C17, the line's deepest open
  limitation). `FINDING44` attaches the first mechanistic evidence to it without resolving it.
- The CLIP evaluation classifier's revision was never pinned; 28 of 41 RNG sites use the global
  stream (`scripts/provenance.py` audits both); FID was never run (`docs/FID_PROTOCOL.md` is the
  unexecuted protocol).

## Layout

- `docs/` — findings, pre-registrations, protocols, reports, errata. Pre-registrations are kept
  **including the falsified ones** (`PREREG_H66` §5a), which is what pre-registration is for.
- `scripts/` — `attack_relearn.py` (the measurement instrument), the adversaries
  (`attack_lora.py`, `attack_ti.py`, `attack_dreambooth.py`), the defence ports (`erase_frp.py`,
  `meta_unlearn.py`), and the correctness tooling (`verify_splits.py`, `provenance.py`,
  `matched_removal.py`).
- `outputs/` — every result JSON (923 files). Contaminated runs are **preserved, not deleted**,
  under `outputs/CONTAMINATED_label_collision/`.
- `logs/` — training and attack logs (141 files). `logs/h80*` carry the Meta-Unlearning training
  records behind `FINDING56` §2, including the ≈ 60× benign-loss gap at θ′.

Model checkpoints (61 GB) and generated image grids (229 MB) are git-excluded, matching this
repository's convention for `results/`. The result JSONs and text logs are **not** excluded — they
are the evidence for every claim in `docs/`, and `t2i_unlearn/.gitignore` re-includes them against
the repository-root exclusions.
