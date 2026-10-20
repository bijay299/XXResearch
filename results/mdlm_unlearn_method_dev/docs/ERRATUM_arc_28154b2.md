# ERRATUM — `research/arc-amortized-reference-caching` @ `28154b2f001c`

Issued 2026-10-20 by the coordinator, after an independent engineering audit. Applies to
`AMENDMENT_01_breaking_point.md` and `BIG_RESULTS.md` in that commit. **The original commit
is not rewritten.**

---

## 1. RETRACTED: the per-fact IN-pool / ABSENT split for the `K = 512` arm

`scripts/analyze_big.py:17` reads pool membership from **one** file and applies it to every
arm:

```python
POOL = json.load(open(f"{S}/adapters/big_arcK16/report.json"))["pool_membership"]
IN_POOL = set(POOL["facts_in_pool"])
```

That is ARC **K=16**'s membership (15 of 64 facts). It was then used to stratify all four
arms, including `arcK512` — whose own report records `n_facts_in_pool = 64/64`. **`K = 512`
has no absent set**, so its `IN-pool 0.350 (n=20) / ABSENT 0.183 (n=82)` cells are not a
pool-membership split at all; they are an arbitrary partition of its facts by *another
arm's* pool.

**Retracted cells:** the `arcK512` row's `recalled IN-pool` and `recalled ABSENT` entries in
`AMENDMENT_01_breaking_point.md`, and the corresponding row in `BIG_RESULTS.md`. The same
objection applies to the `theta0` and `uncached` rows' splits: those arms have no pool
either, so their IN/ABSENT columns are also stratifications by K=16's pool and are only
meaningful as *the K=16 pool's* facts measured under other arms — which is a different
statement than the column heading implies.

**Unaffected:** every aggregate claim, which is where the amendment's conclusions actually
live — ARC K=16 disclosure 0.931 vs uncached 0.049 vs θ₀ 0.980 on the recalled subset;
K=512 at 0.216; the flat dose-response; the retraction of the efficacy-parity claim at 128
facts.

**Also unaffected: the refutation of my pre-registered per-fact prediction.** That
prediction concerned the **K=16** arm, whose membership file was the one actually loaded.
Its measured split (ABSENT 0.951 vs IN-pool 0.850, gap +0.101) is correctly computed, and
the conclusion that the prediction was refuted as stated is unchanged.

## 2. θ₀ was evaluated unmerged while every arm was evaluated merged

`evaluate_big.py:24-26` attaches the memorized adapter for θ₀ but calls `merge_and_unload()`
before attaching for the arms. Merged and unmerged bf16 execution are not bit-identical, so
the θ₀ row (disclosure 0.781 all-facts / 0.980 recalled, 6 degenerate) carries an
unmeasured confound relative to every arm it is compared against. A direct merged-vs-
unmerged measurement is scheduled; until it lands, θ₀ rows should be read as approximate.

## 3. The 128-fact memorization backbone's gates did not complete

The original amendment discloses that memorization "narrowly failed its own 0.80
qualification gate at 0.797". The audit establishes this was worse than disclosed:
`memorize_big.py` terminated at `:199` with `KeyError: 'aliases'` (`ATTEMPTS.tsv` p024,
rc=1) **after training finished but before the gate block completed**, so
`big_memorization_report.json` was never written. Gates G3 (save/reload) and G4 never
executed, and the run's hyperparameters (`--answer-weight`, `--t-low`, `--lr`, `--seed`) are
**not recoverable** from any artifact — no launch script was preserved. Additionally 1,980
of 2,400 logged steps show `loss=0.0000 gnorm=0.000`.

So the backbone every 128-fact arm was built on is **not a qualified checkpoint** in the
sense the study's own protocol required. This does not change the *relative* comparisons —
all arms share the identical backbone — but it does mean the 128-fact absolute numbers rest
on an unqualified, non-reproducible starting point. The amendment's existing caveat should
be read as stronger than written.

## 4. G3 (save/reload agreement) is unreachable dead code

`memorize_big.py:237` calls `__import__("common")`; no `common.py` exists on that script's
path (the only one lives in `llada_poc/scripts`). The call raises `ModuleNotFoundError`
whenever reached. **Nothing anywhere in this stack verifies that a saved adapter reloads to
identical behaviour**, in this study or any other on the branch. A reload bit-identity test
is scheduled.

## 5. The in-code claim of a `p0(F)` diagnostic is false

`arc_big.py:77-78` states the `build_fact_set` key bug was "caught by the p0(F) diagnostic
before any full run". `arc_big.py` contains no such diagnostic. The bug *was* real and *is*
fixed — every caller reads `["F"]`, and `min|F| = 4` across all 64 facts in all three arms,
verified — but the comment credits a safeguard that does not exist in that file.
`exclusion_target` still returns `r = p0` silently for an empty fact set, with no error, so
the class of defect remains undetected by construction.

## 6. Disclosure is under-counted by the scorer

Re-scoring all 1,648 stored generations with a diacritic-insensitive normaliser finds **7
generations that disclose the reference answer but were scored as non-disclosing** (all
`Montréal` vs `Montreal`), affecting the `uncached` and `arcK512` rows. 0.4% of rows, erring
in the direction that flatters every arm. Disclosed rather than silently corrected.

---

## What stands unchanged

Proposition 5 (the reference pass is not removable by algebra), Proposition 6 (caching is
exact — `max|Δ| = 0` on every run), the measured forward counts and the 1.257× per-step
speedup, the clean-forgetting metric and its finding that MDU τ=1 and EMDU's apparent
0.000 disclosure is output collapse, and the amendment's central retraction of the
efficacy-parity claim at 128 facts.
