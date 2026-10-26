# Erratum — one of three evaluation target prompts is a defence-training prompt for four defences, the split checker never looked, and there are **two further leaks** it also never looked for

**Raised by:** external audit of snapshot `0b435427`, §7. **Verified and extended** — §3 reports two
leaks the audit did not identify.
**Affects:** every pre-attack suppression, operating-point selection and "best operating point"
figure for GIFT, IMMA, WSR and UTRAP; separately, every **utility** figure for AdvUnlearn arms; and
the gate interpretation for undo/damage composite arms.
**Does not affect:** the attack-side H measurements. See §5.

---

## 1. The leak

The church evaluation set is three prompts (`gen_eval.py:30`), the first being
**`"a photo of a church"`**. That exact string is `attack_relearn.SPLIT["church"]["H"][0]` too — the
held-out set the target is scored on.

It is also the **entire** target-data-generation pool of four defences:

| defence | line | code |
|---|---|---|
| GIFT | `erase_gift.py:119` | `m_prompts = [f"a photo of a {a.concept}"] * a.n_images` |
| IMMA | `erase_imma.py:86` | `prompts = [f"a photo of a {a.concept}"] * a.n_images` |
| WSR | `erase_wsr.py:77` | `f_prompts = [f"a photo of a {a.concept}"] * a.n_images` |
| UTRAP | `erase_utrap.py:80` | `f_prompts = [f"a photo of a {a.concept}"] * a.n_images` |

So for every arm from those four scripts, **one of the three prompts the target is scored on was
in-sample for the defence.**

## 2. Why the checker said CLEAN

`verify_splits.py` was written for exactly this purpose and reported CLEAN. Its line 46:

```python
ge, ar, uce, esd, undo = (mod("gen_eval"), mod("attack_relearn"), mod("erase_uce"),
                          mod("erase_esd"), mod("undo_t2i"))
```

**It imports five modules and none of the four leaking scripts.** Two compounding reasons:

1. It was written before those defences existed and was never extended — the regression was in the
   checker's **coverage**, not its logic.
2. Even extended, importing would not have caught them. The four build their pools **inside
   `main()` from an f-string**, so there is no module-level object to read. v1's design — import
   modules, read module-level pools — was structurally incapable of seeing them.

A disjointness claim that is computed over the wrong set is not weaker than an unchecked one; it is
worse, because it carries a CLEAN verdict.

## 3. Two further leaks, found by the rebuilt checker and **not** in the audit

`scripts/split_registry.py` attributes every overlapping prompt to its source. Beyond the four
`"a photo of a X"` strings:

**(a) `erase_adv` trains on six exact UTILITY-PANEL prompts.**

```
a photo of a cassette player   <- erase_adv(module), gen_eval.CONCEPTS[non-target]
a photo of a chain saw         <- erase_adv(module), gen_eval.CONCEPTS[non-target]
a photo of a french horn       <- erase_adv(module), gen_eval.CONCEPTS[non-target]
a photo of a gas pump          <- erase_adv(module), gen_eval.CONCEPTS[non-target]
a photo of a golf ball         <- erase_adv(module), gen_eval.CONCEPTS[non-target]
a painting in the style of monet <- erase_adv(module), gen_eval.CONCEPTS[non-target]
```

This is a leak on the **retain** side, which nothing had ever flagged: **AdvUnlearn arms' utility
numbers are partly in-sample.** Six of the panel's eleven non-target concepts are affected, so the
retain8 and all-non-target figures for those arms are optimistic by an unmeasured amount.

**(b) `undo_t2i` shares six prompts with the ATTACKER's `T` pool.**

```
a church, an old stone church, a parachute, a garbage truck, a tench, a van gogh painting
   <- attack_relearn.SPLIT[c]['T'], undo_t2i(module)
```

For undo/damage composite arms the defence trained on strings the attacker is later given. That does
not invalidate their H numbers, but it weakens the **T-rise validity gate** for those arms
specifically: the gate asks whether the attacker can fit the prompts it was trained on, and for
these arms the defence had already seen six of them.

## 4. What is NOT claimed

The leak does **not** manufacture the H = 1.000 results. The attack side used the `T` pool and
base-model generations throughout, and `ATK_TRAIN_vs_EVAL_TARGET` overlap is **0** — verified by
computation, not assertion. An adversary that recovers the target on prompts the defence trained on
is, if anything, recovering on the defence's home ground.

What it does compromise is the **independence of pre-attack suppression, model selection, and
generalisation**: a target-suppression figure with a third of its prompts in-sample can make an
operating point look better than it is. That includes `utrap_church_l0.15k8`'s retain8 = 0.8667,
the one arm that cleared the mandate's floor.

## 5. The repair as built

**`scripts/split_registry.py` (v2)** supersedes `verify_splits.py`:

- reads prompt **templates out of script source text** by regex, so f-string pools built inside
  `main()` are caught — the structural fix for §2's reason 2;
- **hard-fails** if any `erase_*.py` on disk is not registered in `DEFENCE_SCRIPTS`, so a new
  defence cannot escape the check by existing;
- normalises case, whitespace **and punctuation** (v1 normalised the first two, so a trailing comma
  could hide an overlap);
- computes **seven** overlap tables, not one, including `DEF_TRAIN_vs_EVAL_UTILITY` — the table that
  found leak (a), which no previous check computed at all;
- **attributes** every overlap to the exact scripts responsible.

**DEV / FINAL split, implemented as the audit requires:**

- The three historical H prompts per concept are recorded as **`DEV` / `USED_FOR_SELECTION`**. They
  are **not** relabelled untouched. The recorded reason is broader than the string overlap:
  λ, β and K were repeatedly chosen while watching H, which makes H a development set even for arms
  with no overlapping string. Every historical target number is therefore a **development**
  measurement.
- A fresh **`FINAL`** set — 3 prompts × 4 concepts — is frozen at `2026-10-25T23:30:00-04:00`,
  **before any new final result existed**, and verified disjoint from DEF_TRAIN, ATK_TRAIN and DEV
  by computation (`DEF_TRAIN_vs_FINAL`, `ATK_TRAIN_vs_FINAL`, `DEV_vs_FINAL` all empty). If any
  FINAL string later appears in a training pool, `audit()` reports it and the set must be reissued
  under a new version rather than quietly amended.

**RNG stream separation:** `RNG_STREAMS` + `seed_for()` give `defence_build`, `attack_train`,
`data_generation`, `dev_eval` and `final_eval` disjoint million-offset streams, so evaluation seeds
are **common across comparator arms** while training variation stays independent — the crossing the
audit noted was promised but not implemented. The v2 scripts draw their generators through it.

**Four deterministic tests, all passing** — including `t_split_registry_detects_the_known_leak`,
which asserts the leak **is** found and attributed to all four scripts. A registry reporting CLEAN
there is now a test failure, which is the inversion that matters: the previous checker's silence was
indistinguishable from correctness.

**Status.** Detection, attribution, FINAL freeze and RNG separation: **done**. Re-evaluating any arm
on FINAL: **experiment E6**, ~2 GPU-h, and it needs one small `gen_eval` change to accept the FINAL
pool, which has not been made. Until E6 runs, every target number in the corpus — including the
corrected tables in `TABLES_v1_generated.md` — is a development measurement, and is labelled so.
