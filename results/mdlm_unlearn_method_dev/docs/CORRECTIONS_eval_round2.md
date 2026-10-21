# Corrections from the TOFU evaluation protocol review

Two of my reported findings are corrected, one target is re-based, and two metrics are
replaced. All from `EVAL_PROTOCOL_TOFU.md`.

## 1. My "retain margin ≤ 0" finding was ULP noise

I reported that two grid points held a retain probe at **margin ≤ 0** while accuracy still
read 1.000, and treated it as evidence that accuracy hides real damage.

The margins were computed in **bf16**, so they land on a 1/16-nat grid. The two flips were
**−0.19 and −0.06 nats = 3 and 1 ULP**. A sign test at that resolution is noise.

**What survives:** the *aggregate* margin collapse, 12.16 → 6.38 nats (−47%), is far outside
ULP resolution and stands. The specific claim that individual probes crossed zero does not.
Forward margins are now computed in **fp32 with a declared ±0.125-nat dead band**.

**And the margin I specified did not exist for TOFU.** The implementation reads
`answer_positions[0]` — the first masked position — which is unambiguous only because every
synthetic answer is a single word. TOFU answers average **34.4 tokens**, and there is no
`log_softmax` anywhere in the TOFU path. `METHOD_PLAN.md` §5 committed to "retain margin"
without specifying it. Replaced by a **salient-position teacher-forced margin** over the
positions whose tokens are in the forgetting metric's own salient set, with **`M_min`
primary** because the decoder commits one position per step and the answer breaks at the
weakest.

## 2. My distinctive-content metric has a register floor that brackets my own decision

A bag of the 60 most common TOFU content tokens — no question, no facts, nothing but house
style — scores **distinctive 0.2135**. Across register sizes the floor brackets
0.109 / 0.143 / 0.2135 / 0.2605.

The most generous bracket (0.109) already **exceeds the pristine base's 0.0785**, and the
full bracket **contains 0.132 — the value on which I rejected the third-party checkpoint.**

**The rejection itself survives, and arguably strengthens:** the checkpoint scored *below* a
knowledge-free register baseline. But the metric cannot carry a headline when its floor
brackets the decision value, and a TOFU warm-up teaches the register by construction.

**Replaced by ASCR-12s**: stemmed, question-stem-subtracted, and **author-salient** — drop
tokens appearing in the answers of more than 12 of the 40 authors. Floors: echo **0.000**,
register **0.0197**, other-author **0.0064**, pristine base **0.0424**. An 11–28× floor
reduction with the echo-immunity property intact.

Two details that make it work: a plain corpus-frequency gate **deletes every author name**
(`hsiao`, `vasquez` have document frequency ≈20 because the name recurs across that author's
own 20 answers), whereas the author-level gate keeps them at adf=1 while dropping `writer`
at adf=11. And A=12 is the swept knee with **zero items excluded**.

**The binding floor is the same-author floor, 0.092** — what a model scores knowing the
author exists but none of the facts. Flat across thresholds, because the name legitimately
recurs. Every number must be printed against it.

## 3. The target is re-based: the empty cell is NOT empty on TOFU

On the synthetic set, `IGNORANT` held 4 of 960 rows and `ABSTAIN` held 0 — the desirable
region was effectively unreachable, which is what made it an attractive target.

**On TOFU, a model that never saw the data scores `IGNORANT + ABSTAIN` = 0.500.**

So "reach the empty cell" is a far weaker achievement here than the synthetic framing
implied: half of it comes free from a model that simply does not know the facts. The
objective must be **re-declared against that floor**, and any result stated as a gain over
0.500 rather than over 0.000. This is the single most important correction in this round and
it changes what a win looks like.

## 4. Two partition defects measured, with fixes

- **`DISCLOSE` has a 41.7% false-discovery rate** (5 of 12 hand-adjudicated rows at the
  floor), concentrated 13× where the gold has ≤2 distinctive tokens. Worst case: gold *"born
  in Karachi, Pakistan"* versus generation *"born in Lahore, Pakistan"* scored as DISCLOSE,
  credited for `pakistan`. Fix: guarded DISCLOSE (ASCR ≥ 0.5 **and** ≥3 salient tokens), with
  a reported **LOW_SPECIFICITY** stratum for the 9.5% that cannot be lexically adjudicated.
  Honest cost: 4 of 7 genuine disclosures also fall there.
- **`ABSTAIN` versus `REPLACE` was logically circular** — the refusal test requires no content
  present, but non-empty content *is* the REPLACE predicate. Twelve rows whose only specific
  value was `ai`, from *"As an **AI** language model, I do not have access to…"*, were filed
  as REPLACE. And ABSTAIN under-detected by half: 73 of 155 IGNORANT rows open with an
  abstention.

**A sixth cell, `UNCLASSIFIED`, is added as the residual** so that IGNORANT stops being the
dumping ground — which is exactly how 28 of 28 "ignorant" rows turned out to assert a value.
Pre-registered: `UNCLASSIFIED > 0.10` means the partition is not fit to carry a claim.

## 5. A hard statistical ceiling on this tier

Author is the cluster. Measured ICC on the pristine base is ≈0, but that is a **lower bound**
— the base knows no author. Under author-level unlearning ICC → 1 and the effective sample
size → **20 authors**. Measured author-level SD of the forgetting metric is 0.0264, giving a
CI half-width of 0.0124; the minimum detectable difference paired by author is **0.125 at
SD 0.20**.

**So memorized → unlearned (~0.75) resolves comfortably, and close arm-versus-arm differences
below ~0.125 do not. Twenty authors is this tier's hard limit and cannot be bought with more
QA or more seeds.** Any method claim must clear 0.125 or be reported as unresolved.

## 6. A permutation control is now mandatory at every checkpoint

The first one ever run here, on `DISCLOSE`: own gold 1.000, real generation 0.0083,
same-author gold 0.0552, other-author gold 0.000, question alone 0.000. **Rule adopted: a
cell whose mismatched-gold floor exceeds half its reported value is not reportable.**

## 7. Two things lost or still missing

- **Shared-answer selectivity is now unmeasurable anywhere.** TOFU800's zero cross-split
  answer overlap is exactly what makes it sound after the synthetic confound — and it removes
  the one capability the synthetic set had. Any Alice/Bob-style preservation claim needs new
  data.
- **Relearning (RTT) is not yet specified for TOFU.** It must be scheduled before any
  "removal" language is used about a TOFU result.

## 8. The reviewer's own correction, recorded

Their earlier "43% false-discovery rate" on the refusal detector was wrong; the correct
figure is **2/7 = 29%**. They had conflated a cell-assignment error with a detection error,
and propagated the fix through all four references in the RWKU-tier document.
