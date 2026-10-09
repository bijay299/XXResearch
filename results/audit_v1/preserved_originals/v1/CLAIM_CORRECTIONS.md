# CLAIM_CORRECTIONS — overstated claims, what the evidence supports instead

Every original is preserved twice: in git at commit
[`a9de625`](https://github.com/bijay299/XXResearch/tree/a9de625), and as a file
copy under [`preserved_originals/`](preserved_originals/). Nothing was deleted
or rewritten in history.

The underlying numbers were **not** wrong. All 1026 re-derived values reproduce
the committed `rates.csv` and `contrasts.json` exactly, with zero mismatches
(`crosscheck.json`). Every correction below is a correction of **language and
of which quantity was reported**, not of arithmetic.

Evidence for each: [`CORRECTED_TABLES.md`](CORRECTED_TABLES.md),
[`integrity.json`](integrity.json), [`direct_contrasts.json`](direct_contrasts.json),
[`crosscheck.json`](crosscheck.json), [`recomputed_rates.csv`](recomputed_rates.csv).

---

## C1 — M0 double-counted as newly generated images

**Original** — `COMBINED_RESULTS.md` l.11, `SLIDE_OUTLINE_FINAL.md` l.17:
> "**3360 evaluation images in total, zero generation or detection failures.**"

**Problem.** `eval_seed29/M0` is a symlink to `eval/M0` and the two detection
files are byte-identical (md5 `df04381d336465cd47e332234775e64d`; all 280 image
hashes shared). 3360 is the row count of two stacked tables; it counts one
reused evaluation set twice.

**Corrected.** **3080 images were generated** across **11 distinct checkpoint
evaluations**. 3360 is the table row count across 12 slots. The 280 M0 rows are
one evaluation set appearing under both seed labels and carry **no** cross-seed
replication information.

**Evidence.** `integrity.json` → `m0_reuse.byte_identical_reuse: true`,
`symlink_audit`; `CORRECTED_TABLES.md` §1.

---

## C2 — a non-replication reported as a false positive

**Original** — `COMBINED_RESULTS.md` l.94–95:
> "**Conclusion: historical recovery is not established by these runs.**
> Reporting it from seed 17 alone would have been a false positive."

and `SLIDE_OUTLINE_FINAL.md` l.26–27: "One-seed recovery was a false positive".

**Problem.** Three errors compounded.

1. "False positive" asserts the effect is absent. The two intervals are
   seed 17 `+15.0 [+2.5, +27.5]` and seed 29 `−5.0 [−12.5, 0.0]`; they
   **overlap**. A differing significance label across two runs is not a
   significant difference between them, and neither run had the precision to
   resolve a small effect.
2. The claim is also **threshold-selective**. It is stated at t=0.5 only. At
   t=0.3 the seed-29 estimate is `+0.0 [−7.5, +7.5]` — not opposite in sign to
   seed 17's `+15.0`, simply unresolved, so the "opposite direction" reading is
   specific to t=0.5. Conversely `MAB`, labelled "robust", itself changes
   credibility class at t=0.3 (`−17.5 [−37.5, 0.0]`).
3. It locates the instability in the method, when it is substantially in the
   **reference term**. `D_cat(child) − D_cat(MA)` moves when the parent moves.

**Corrected.** This pair of runs **does not resolve the sign** of parent-
referenced cat change in the sandwich branch. That is a statement about
resolution, not about absence. More training runs are required before it is
reported in either direction.

**And the parent-free contrast does replicate.** `D_cat(MAC_L2) − D_cat(MAC)`
is `+7.5 [0.0, +17.5]` in seed 17 and `+2.5 [−7.5, +15.0]` in seed 29 — same
direction, same credibility class, both small. The quantity that isolates the
L2-SP effect agrees across seeds; the quantity that divides by a moving parent
does not.

**Evidence.** `CORRECTED_TABLES.md` §2 and §3.

---

## C3 — the cross-seed parent gap attributed to the parent in general, when it is specifically a paraphrase effect

**Original** — `COMBINED_RESULTS.md` l.89–92:
> "Much of the gap is the parent itself moving: D_cat(MA) was 17.5% in seed 17
> and 27.5% in seed 29 … That is exactly the instability one training seed
> cannot reveal."

**Problem.** Directionally right, but it stops one step short and so misses
where the instability lives. Split by prompt family, `D_cat(MA)` is:

| threshold | literal s17 / s29 | paraphrase s17 / s29 |
|---|---|---|
| 0.3 | 15.0 / 15.0 — **identical** | 25.0 / 40.0 (+15.0 pp) |
| 0.5 | 15.0 / 15.0 — **identical** | 20.0 / 40.0 (+20.0 pp) |
| 0.7 | 15.0 / 15.0 — **identical** | 20.0 / 35.0 (+15.0 pp) |

**Corrected.** The two training runs produced **identical** literal-prompt cat
rates at the parent, at all three thresholds. The entire cross-seed parent gap
is carried by **paraphrase** prompts. "The parent moved" is true; the specific
and more useful statement is that the parent moved **only on paraphrases**,
which is where a 10-prompt-per-family evaluation is least precise.

**Evidence.** `CORRECTED_TABLES.md` §3, "The reference term itself".

---

## C4 — "robust" applied to two training runs

**Original** — `COMBINED_RESULTS.md` §1 table, nine rows labelled **robust**.

**Problem.** Two runs cannot establish robustness. The table's own labels also
conflate three different situations under one word: intervals excluding zero in
both seeds, intervals including zero in both ("robust (both ns)"), and
agreement of direction only. "Robust (both ns)" labels two runs each failing
to resolve an effect as a robust finding.

**Corrected.** Replace "robust" with **"direction agrees across the two runs"**,
which is the only cross-seed claim n=2 supports, and state the threshold. Where
both intervals include zero, the correct label is **"neither run resolved this"**
— not a finding. `SEED_COMPARISON.md` already carried the right framing in its
own header; the combined document contradicted it.

The quantity that genuinely agrees most closely is the **direct** L2 contrast
on the new target: `+40.0` to `+47.5 pp` across both seeds, both branches and
all three thresholds. It is reported in `CORRECTED_TABLES.md` §2 and was absent
from the committed `contrasts.json` entirely.

---

## C5 — a near-zero-suppression arm described as having eliminated the deletion, and preservation credited

**Original** — `COMBINED_RESULTS.md` l.68–70:
> "In the sandwich branch it **eliminates the deletion altogether** (+7.5 pp,
> interval touching zero in both seeds) while buying almost nothing"

**Problem.** The second half is sound. The first half states a small, uncertain
suppression as zero: `+7.5 [+0.0, +17.5]` (s17) and `+7.5 [+0.0, +15.0]` (s29)
are small and unresolved, not eliminated.

**Corrected.** At L2-SP 25000 the sandwich branch retains **+7.5 pp** of
measured suppression with an interval reaching zero: the arm is at or below the
floor of what 10 prompt clusters can resolve, so its true deletion strength is
unknown and may be near zero. It has **not** been shown to be zero.

**The consequence is a protocol one.** Retention is only comparable between
arms that deleted a comparable amount. At +7.5 pp the sandwich L2 arm is
outside any meaningful deletion region, so comparing its retention against the
unregularised arm's (+50.0 / +52.5 pp) is not a preservation result at all —
it compares an arm that deleted with one that essentially did not. The protocol
declares this branch **infeasible for matched comparison at this coefficient**
rather than scoring it.

---

## C6 — a per-request coefficient inferred from a single tested value

**Original** — `COMBINED_RESULTS.md` l.72–74:
> "So the coefficient 25000 … is far too strong for the sandwich branch and
> arguably reasonable for the dog branch. **A single fixed coefficient is not
> appropriate across requests.**"

**Problem.** One coefficient was tested. No coefficient search was run
(confirmed: `l2sp_weight` is 25000 or 0 in all ten `train_report.json`). From
one value one cannot conclude that no single value works — the observation is
consistent with 25000 being wrong for both branches, with a different single
value working for both, or with per-request tuning being necessary. The data do
not separate these.

**Corrected.** 25000 **behaves differently in these two branches**: it leaves
+42.5 pp of dog suppression and +7.5 pp of sandwich suppression. Whether any
single coefficient serves both is **untested**. A coefficient sweep on a
development split would be required, and must never be run against final
evaluation numbers.

---

## C7 — residual presence and recovery used interchangeably

**Original** — `COMBINED_RESULTS.md` l.99–101:
> "**Recovery** and residual presence are consistently larger under
> **paraphrased** prompts … In seed 17, MAC_L2 recovered +25.0 pp on
> paraphrases versus +5.0 pp on literals."

**Problem.** Two distinct quantities joined by "and". Residual presence is the
child's own rate; recovery is a difference from the parent, and it moves when
the parent moves (see C3). Under `MAC_L2` paraphrases the child rate is 45%
(s17) / 30% (s29) while the parent is 20% / 40% — the same child direction
produces opposite parent-referenced signs.

**Corrected.** State them separately. **Residual presence** is higher under
paraphrases. **Parent-referenced change** is also larger under paraphrases, but
partly because the parent's own paraphrase rate differs between runs. What
follows is only that evaluating erasure with literal prompts alone understates
what the edited model still produces. This is a fixed two-family generalisation
check — **not** an adaptive attack, and **not** evidence of greater recovery.

---

## C8 — hard-coded seed-17 text in the seed-29 report

**Original** — `results/seq29/RESULTS.md` l.15 and l.135, generated by
`scripts/seq/make_report.py`:
> "200 anchor images/prompts, generated once from untouched M0 **(seed 17)**…"
> "One training seed **(17)**. The bootstrap intervals are …"

**Problem.** Both strings were literals in the generator. Line 135 states the
wrong training seed outright. Line 15's "(seed 17)" is the *anchor-generation*
seed, which is genuinely 17 (`anchor_caches/Horses/cache_meta.json` →
`anchor_seed: 17`) and shared across both training seeds — but inside a
seed-29 report it reads as the training seed.

**Corrected in code,** not only in the output:
- `--training_seed` no longer defaults to 17. It is taken from the contrasts
  file, and if passed explicitly and contradicting it, the script **refuses to
  run** rather than mislabel a report.
- The anchor sentence reads the value from `cache_meta.json` and states that it
  is an anchor-generation seed independent of the training seed.
- `scripts/seq/run_analysis.sh` hard-coded `--eval_root ${SEQ_ROOT}/eval`
  (seed 17) for **any** seed argument, so `run_analysis.sh 29` would have
  analysed seed 17's images and labelled the output seed 29. The seed-29
  analysis on disk correctly points at `eval_seed29/`, so this latent bug was
  bypassed in practice, but it is fixed and the script now also refuses to
  aggregate an incompletely evaluated seed.

---

## C9 — narrative conclusions written into the generator rather than derived

**Original** — `make_report.py` finding 3 asserted, as fixed text, that L2-SP
"did not protect the earlier deletion", that "cat recovery did not decrease",
and that in the sandwich branch "it *increased*" — regardless of input. It also
emitted "**No recovery**" whenever no interval excluded zero upward.

**Problem.** These are conclusions, not formatting. The same text would print
for data showing the opposite. "No recovery" states absence where the evidence
shows non-resolution.

**Corrected.** Direction and magnitude clauses are derived from the measured
values with a 2.5 pp reporting floor; the "no recovery" branch now reads **"no
increase in cat detection was resolved at this precision"** and says explicitly
that this concerns what 10 prompt clusters can resolve, not whether the effect
exists. The L2 finding now points at the parent-free contrast as the quantity
that isolates the effect, and states that retention is comparable only between
arms of comparable deletion strength.

`compare_seeds.py`'s replication warning now enumerates the three things a
credibility-class change does **not** license (no false-positive claim, no
attribution to the method, no general seed disagreement).

---

## Claims that survived the audit unchanged

Stated because an audit that only lists errors misrepresents the snapshot.

- Every rate and headline point estimate reproduces exactly from raw
  predictions: 0 mismatches over 252 rate cells + 774 contrast values.
- Completeness is exact: 12 × 280 rows, 280 unique identities and 280 unique
  image hashes per slot, 0 duplicates, 0 missing, 0 extra against the frozen
  manifest. No de-duplication or row-dropping rule was needed anywhere.
- The two training runs are genuinely independent: **0** images are byte-shared
  between seeds other than the known M0 reuse.
- The manifest was frozen before editing, with 0 exact collisions and 0 near
  duplicates against 999 training strings.
- The bootstrap design was already correct — paired, clustered on prompts
  carrying their four generation seeds, training seeds never pooled.
- The first deletion worked, in both runs, in the same direction:
  `+82.5 [+67.5, +95.0]` (s17) and `+72.5 [+50.0, +90.0]` (s29).
- L2-SP cut parameter movement from the parent by roughly 9× and reduced
  measured dog-branch collateral damage on bird — direction agreeing in both
  runs. What the audit changes is the *interpretation*: that comparison is
  between arms of unequal deletion strength, which is what the proposed
  diagnostic is designed to fix.
- `SEED_COMPARISON.md` already stated the n=2 limitation correctly in its
  header, and the per-seed reports already carried the detector-proxy and
  "no human annotation" caveats.
