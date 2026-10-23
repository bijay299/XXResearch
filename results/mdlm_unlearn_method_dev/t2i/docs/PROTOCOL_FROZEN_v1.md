# Frozen confirmatory protocol v1 — declared 2026-10-23, BEFORE any confirmatory run

Issue 11 of the remediation. Every number produced under this protocol is comparison-grade.
Everything produced before it is **exploratory** and labelled so, including `H62`, `H63` and `H64`,
which were launched before this freeze.

---

## 1. Attack grid — identical for every arm, no exceptions

| # | seed | lr | steps | attacker images |
|---|---|---|---|---|
| 1 | 0 | 1e-5 | 300 | 16 |
| 2 | 1 | 1e-5 | 300 | 16 |
| 3 | 6 | 1e-5 | 300 | 16 |
| 4 | 0 | 5e-5 | 300 | 16 |
| 5 | 1 | 5e-5 | 300 | 16 |
| 6 | 0 | 1e-4 | 300 | 16 |

**Stopping rule:** exactly these six runs per arm per target concept. No arm receives a seventh
attack, and no arm receives fewer. Any change to this grid must be recorded in this file, with a
timestamp, **before** the affected outcomes are examined.

**Why fixed and symmetric:** the audit established that `max-over-valid-attacks` is monotone
non-decreasing in the number of attacks run, and that this project had run 9–10 attacks on arms it
hoped would resist versus 2–6 on arms already reversed. That asymmetry made `PREREG_H40`
unfalsifiable in practice. `H62`'s eleven-attack tench extension is **exploratory** under this rule
because the same extension was not applied to comparator arms.

## 2. Adversary classes reported separately

- **UNet-only** — `--unet` and/or `--text-encoder` loaded, attack updates the UNet alone.
- **Joint (UNet + text encoder)** — requires `--attack-te` **and** the issue-2 repair, with
  `text_encoder_grad_norm_step0 > 0` and `text_encoder_param_delta > 0` recorded in the run.

Budgets and labels are never merged across these classes. **No historical run qualifies as joint:**
the two runs labelled that way pre-date the repair and had a detached encoder
(`joint_attack_verified: false` in the inventory).

## 3. Recovery reporting — always both

Every arm reports `max_valid_H`, `max_all_H`, `n_valid/n_total`, and per-run gate outcomes. A gate
failure never establishes robustness, and never hides observed target regeneration.

**Documented gate limits** (all found by use, not design): ceiling effects on at-ceiling arms;
false negatives when T prompts are intrinsically harder than H prompts; seed sensitivity; and —
the one that cost most — **it cannot detect an incorrectly assembled evaluation pipeline**, since
validity was computed on the same broken pipeline during the text-encoder bug. The gate is not
redesigned after seeing outcomes.

## 4. Prompt splits — declared before results are read

| split | role |
|---|---|
| `DEF_TRAIN` | defence repair/distillation pool |
| `ATK_TRAIN` (T) | attacker fine-tuning strings |
| `DEV` | model selection and debugging |
| `FINAL` (H_final) | untouched; read once, for the final number only |

**Historical limitation preserved, not erased:** the historical H prompts are disjoint from
attacker training strings but **do occur in the defence repair pool**, and H has repeatedly informed
model selection. Historical H scores are kept with that limitation attached; they are not
retrospectively relabelled as held-out.

Overlaps are checked programmatically and the split definition is committed before any confirmatory
result is read.

## 5. Seeds

Attack-training seeds and evaluation seeds are drawn from disjoint sets. Comparison arms share
common evaluation seeds; evaluation replicates are independent of training variation.

## 6. Metrics

- **Target suppression**: active-target CLIP accuracy, `--target` set explicitly.
- **Utility**: mean over all evaluated non-target concepts (n=11), reported alongside the
  `legacy_retain8` figure (n=8) with its denominator named. The two are never interchanged.
- **Post-attack utility**: measured, not assumed (issue 7).
- Classifier label set fixed across all comparisons.

Raw counts are preserved. One image moves H by 1/15 = 0.0667; no H value is reported to an implied
precision finer than that.

## 7. Provenance recorded per run

Code commit, component hashes (base/erased/attacked), model and classifier revisions, dependency
versions, precision, split and prompt identifiers, optimiser settings, and both RNG seeds.
Historical runs carry explicit nulls for the fields never captured.
