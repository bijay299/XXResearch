# Finding 53 — a LoRA adversary training 0.046 % of the UNet fully reverses the robust SOTA on three concepts. The apparent separation between defences did not survive its own controls.

**Date:** 2026-10-24 · **Adversary:** hand-rolled LoRA, rank 16, α 16, 300 steps, 16 self-generated
images · **Protocol:** `PROTOCOL_FROZEN_v1` T/H split, `FINDING20` validity gate, issue-7 utility
probe — identical to the full-fine-tuning attacks, so the two adversaries are directly comparable
**Evidence:** `outputs/atk_H71*_*.json`, `scripts/attack_lora.py`

---

## 1. Why a LoRA adversary

Every attack in this phase so far fine-tunes the **full UNet** (859.5 M parameters). That is a
strictly larger attack surface than the adaptation methods the protection literature is written
against: **IMMA (2311.18815) immunises against LoRA, Textual-Inversion and DreamBooth**; Glaze and
MIST poison data against the same family. So this phase's negative results were repeatedly about a
**bigger adversary than those papers claim to stop** — recorded as a scope limitation for AdvUnlearn
(`ERRATUM_r2` C15) and FRP (`FINDING43` §2).

Rather than port IMMA — a bi-level optimisation whose threat model is *narrower* than our adversary,
which would have produced a third threat-model-mismatch result — this attacks our erased arms with
**LoRA itself**, the canonical method of that family.

LoRA is implemented directly (`scripts/attack_lora.py`), not via `peft`: that package is absent from
this venv and a dependency upgrade has broken a venv on this machine before. The construction is
standard — frozen `W`, plus `(α/r)·B·A` with `B` zero-initialised so step 0 is exactly the erased
checkpoint.

## 2. The result

Symmetric six-point grid per arm per concept (seeds {0,1,6} cross-attention; {0,1} all-attention;
{0} at 3× lr), applied identically:

| arm | concept | valid | **max-over-valid H** | pre-attack H | utility after |
|---|---|---|---|---|---|
| **official AdvUnlearn** | church | 5/6 | **0.8667** | 0.400 | 0.81–0.94 |
| **official AdvUnlearn** | parachute | 6/6 | **1.0000** | 0.133 | 0.75–1.00 |
| **official AdvUnlearn** | van_gogh | 5/5 | **1.0000** | 0.467 | 0.833 |
| **FRP port** | church | 6/6 | **1.0000** | 0.133 | 0.35–0.98 |
| **FRP port** | parachute | 5/6 | **1.0000** | 0.200 | 0.81–0.94 |

> **A LoRA adversary training 0.046 % of the UNet fully reverses the official robust SOTA on two of
> three concepts, and reaches 0.867 on the third — with utility largely intact.**

Trainable parameter counts: rank-4 cross-attention **0.100 M (0.012 %)**, rank-16 cross-attention
**0.399 M (0.046 %)**, rank-16 all-attention **0.799 M (0.093 %)**, against the UNet's 859.5 M.

**This is a stronger negative than the full-fine-tuning result**, because the adversary is four
orders of magnitude smaller and sits squarely inside the threat model IMMA, Glaze and MIST target.
The scope gap those papers' claims left open is now closed from the attack side.

## 3. ⚠ A separation I reported and then withdrew, over three controls

At two valid attacks the grid looked like the **first genuine separation between defences** in this
phase: AdvUnlearn 0.733 against FRP's 0.933. It did not survive:

| stage | AdvUnlearn max-valid H (church) |
|---|---|
| 2 pilot attacks | **0.733** |
| + seed 6 | 0.800 |
| full six-point grid | **0.867** |
| second concept (parachute) | **1.000** — separation gone |

The rise with added seeds is exactly the monotone ratchet `FINDING35` §4 describes, this time on an
observation I was inclined to believe. **The separation claim is withdrawn.** What remains is a
two-image difference on church only, on a 15-image metric, which cannot support a ranking.

Each stage was a control queued *before* publication — so nothing wrong reached the record — but this
is the fourth time in two days that an early read has not survived its own check, and the pattern is
mine, not the protocol's.

**Disclosure on the grid's status:** the grid was fixed **after** I had seen two AdvUnlearn pilot
points. It is therefore *symmetric and pre-declared*, but **not blind**. The pilots are listed as
pilots rather than folded into the grid, and the honest description is "a symmetric grid chosen after
seeing two points", not "a pre-registered blind grid".

## 4. What this does and does not change

**Does not change:** the mandate. No arm reaches retain ≥ 0.85 with max-over-valid H ≤ 0.40 under
either adversary. AdvUnlearn-church at 0.867 is the closest sub-ceiling figure in the phase and is
still more than twice the threshold.

**Does change:** the scope of the negative result, which now covers parameter-efficient adaptation
as well as full fine-tuning; and it means a defence designed specifically against this family
(AdvUnlearn, which is adversarially trained in *prompt* space) does not stop a rank-16 adapter on its
own cross-attention.

**Still unrun:** Meta-Unlearning (2410.12777), GIFT (2507.13598), From Dormant to Deleted
(2505.22310) — none with usable released code. **Textual Inversion and DreamBooth**, the other two
methods IMMA targets, are also not yet run; Textual Inversion is cheap (a single token embedding)
and is the obvious next step.

## 5. Scope

Three concepts, two arm families in the full grid (a third, damage+distil, is **running**), one
rank/α pair in the grid proper, one image budget, our own prompt splits and metric. The 3× lr legs
damage the model (utility 0.333–0.354 on church, 0.000 on one parachute run) and are reported as the
upper bound of useful LoRA strength, not as robustness data. LoRA here adapts **attention
projections only**; DreamBooth-style full-model adaptation and Textual-Inversion-style embedding
attacks are different adversaries and are not covered.
