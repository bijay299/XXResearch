# Upstream reference implementations — current status

> **⚠ Corrections 2026-10-26:** `FINDING76` §3's probe mechanism check is **overstated** (n=7/3/1, within-arm spread 40×, ordering undetermined — see its §7). `FINDING77` §2's "B kills the accumulation-order hypothesis" is **wrong** — B holds accumulation order fixed and never varied it (see its §6).


**Date:** 2026-10-26 · **Source/config commit:** `5e222fc26c43fae14294516db588f932c0394a65`
**Companion:** `UPSTREAM_FIDELITY_MANIFEST.md` (the full ledger) ·
`outputs/upstream_manifest.json`

**No method is marked reproduced.** Nothing below rests on a unit test, and no adaptation's failure
is read as refuting a published method.

---

## 1. Per-method status

| method | source verified | env ready | training done | eval done | category |
|---|---|---|---|---|---|
| **IMMA** | ✅ `2f55041`, Apache-2.0 | ✅ `venv_imma` | ❌ **R1 not run** | ❌ | official repro **pending** |
| **Meta-Unlearning** | ✅ `5a2a760`, no license | ✅ `venv_t2i`-class | ❌ **R2 not run** | ❌ | official repro **pending** |
| **WDR** (From Dormant to Deleted) | ✅ `7b9a1cc`, no license | ⚠️ needs `wandb` + CIFAR | ❌ **R3 not run** | ❌ | **domain adaptation** only |
| **UCE** | ✅ `7c724d9`, MIT | ✅ `venv_uce` | n/a (closed-form) | ⏳ **R4 queued, running** | comparison **in flight** |
| **FRAG/FRP** | ✅ repo exists, **0 .py files** | n/a | n/a | n/a | **no implementation released** |
| **GIFT** | ❌ none found | n/a | n/a | n/a | **no official implementation** |

## 2. What actually ran

**Completed:**
- All five repositories cloned and pinned; three audited revisions resolved **exactly** as requested.
- Licences recorded. **Meta-Unlearning and vision_relearning ship no LICENSE**, so `upstream/` is
  git-ignored and no upstream source is redistributed — only our wrappers, with upstream referenced
  by hash.
- **FRAG rechecked:** repo is non-empty but has `README.md` + 2 PNGs and **zero Python files**.
  "Forthcoming" still holds. This is the one method for which `ERRATUM_r2` C19's "no released code"
  was correct.
- Isolated environments built. **First attempt failed and the errors are recorded verbatim**
  (manifest §5.1): unpinned installs pulled torch 2.14.1 / huggingface-hub 1.33.0, giving
  `ImportError: huggingface-hub>=0.23.2,<1.0 is required` and
  `AttributeError: module 'torch' has no attribute 'accelerator'`. Fixed with a constraints file and
  `--no-deps`. **`venv_t2i` was never modified** — every env is a copy.
- **IMMA fidelity gap found and quantified** (§3 below).
- **`PREREG_H90` compute accounting repaired** and a sample-matched series launched.

**In flight (not interrupted):** `E4m` matched-compute H90 builds (GPU 0), `E2g` Meta-v2 frozen grid
(GPU 1), `R4` queued behind them. GPUs 2–3 left to other users.

## 3. Fidelity findings that change how existing results read

**IMMA — our scope is less than half the reference's.** Measured on the real 859.5M UNet:

| scope | params | % |
|---|---:|---:|
| reference `['to_q','to_v','to_k','to_out']` over all named parameters | **93.5M** | 10.88% |
| — of which **attn1 (self-attention)** | **49.6M** | 5.77% |
| — of which attn2 (cross-attention) | 44.0M | 5.11% |
| our `erase_imma.py` (`.attn2.` only) | **44.0M** | 5.11% |

The reference immunises **self-attention as well**, which our port excludes entirely. Our docstring
("cross-attention only, per the paper") is true to the *paper* and false to the *code*. Our IMMA arms
are therefore a **project reimplementation** whose numbers are not substitutable for the reference's.
Also: the reference optionally **re-initialises LoRA every step**; ours warm-starts and never does.

**Meta — E2 is ours, not official.** `meta_unlearn_v2.py` restores Eq. 8's signed structure but
**two departures remain**: the reference uses **three** data pools (`hrm` / `irt` / `tgt`) with
independent γ coefficients where v2 uses two under one ζ; and `--fix_timesteps` is unimplemented.
E2's results are labelled accordingly.

**WDR — domain adaptation, and parity ≠ optimality.** Any SD result is an adaptation of a
vision-classifier method. `FINDING67`'s α ≈ 4.0e3 parity figure bounds where the term *can* act;
**under Adam the update is preconditioned and per-coordinate normalised, so parity is not a claim
about the best coefficient.** A native-domain sanity check (R3) must precede interpreting the
diffusion adaptation. Neither our wrong-direction WSR-anchor arm nor the single α=4000 failure
refutes the published method.

**UCE — two candidate discrepancies, being measured not asserted.** The official form carries an
`erase_scale` coefficient (default 1) on both erase terms that our port lacks — equivalent only if
that scale is exactly 1, which R4 tests directly, including a deliberate `erase_scale=2.0`
divergence probe. Separately the official `--lamb` default is **0.5** against our **0.1**, so "UCE at
defaults" denotes different edits in the two codebases and **every historical arm of ours used 0.1**.

## 4. Declared experiment matrix (unrun unless noted)

| id | method | arms | seeds | budget | status |
|---|---|---:|---:|---:|---|
| R1 | IMMA author protocol → our attack | 1 | 3 | ~4 GPU-h | **blocked, see §5** |
| R2 | Meta official, three-pool | 1 | 3 | ~6 GPU-h | **blocked, see §5** |
| R3 | WDR native-domain sanity check | 1 | 1 | ~2 GPU-h | **blocked, see §5** |
| R4 | UCE official vs port, matched inputs | 3 configs | — | ~0.3 GPU-h | **queued, running** |

Stopping rules: 3 attack seeds at lr 1e-5 / 300 steps with bounded intermediate budgets (25, 100);
no escalation; arms failing K1 are not attacked and are reported as utility failures. All attempted
attacks reported with gate outcomes and **both gated and ungated** scores, per-seed, with explicit
denominators. `retain8`, all-non-target utility and the paired panel reported **separately**.

**Final-prompt discipline:** the existing FINAL set **has been inspected** (`FINDING65`, `FINDING66`)
and is development data from here on. It will not be reused for tuning and called untouched. A fresh
final set must be frozen before any new configuration is selected against it; R1–R4 select no
configuration, so none is needed yet.

## 5. Remaining blockers, exactly stated

| id | blocker |
|---|---|
| **R1** | IMMA's documented pipeline needs (a) ~100 generated training images via `eval/text2image.py`, (b) an **ESD-erased** SD-1.4 checkpoint — its README fetches `diffusers-VanGogh-ESDx1-UNET.pt` from `erasing.baulab.info`, which is a **third-party download not yet attempted**, and (c) `accelerate launch` config. None is done; the env is ready. |
| **R2** | `src/meta.py` loads `./dataset/{hrm,irt,tgt}` as `imagefolder` datasets that **are not in the repo** and must be constructed. The three-pool split is the point of the method, so substituting our two-pool data would make it our variant again, not a reproduction. |
| **R3** | `vision_relearning` needs `wandb` and CIFAR-scale data; it is a **classifier** codebase, so this is a native-domain check, not a T2I run. Not started. |
| **R4** | none — queued, waiting only on a free GPU. |
| general | GPUs 2–3 belong to other users and are not used; our cap is two at a time, both currently occupied by E4m and E2g. |

## 6. What the measurements so far support

Only this: the four repositories are pinned and licensed-checked; FRAG ships no code; our IMMA port
has a 2.13× scope gap from its reference; our Meta v2 has two remaining structural departures; and
the UCE comparison is running. **No method has been reproduced, and no published method has been
refuted.** The historical variants keep their names and their results; nothing was deleted,
overwritten or relabelled as official.
