# Five-slide outline — populated with measured results (seeds 17 and 29)

Figures: `seq17/figures/fig1..3`, `seq29/figures/fig1..3`,
`fig4_seed_comparison.*`, grids in `seq17/grids/`.

---

## Slide 1 — Question and setup

- **Can an earlier deletion survive a later one**, while the later deletion still
  works and utility is retained?
- A = cat, then B = dog **or** C = sandwich, from a **shared parent MA**.
  L2-SP (25000) is an **existing baseline** added at the second request — not a
  new method, not an end-to-end lifelong method.
- SD-1.5 backbone · CUIG ConAbl · 1000 steps/request · COCO Faster R-CNN
  evaluator · 280 images/checkpoint · frozen manifest · **two training seeds**.
- **3360 evaluation images, zero failures.** Not UA/IRA/CRA; not a reproduction.

## Slide 2 — The first deletion works, and a later deletion does not undo it  *(fig1, fig4)*

- cat 100% → **18%** (seed 17) / **28%** (seed 29); suppression **+82.5 / +72.5 pp**.
- Deleting **dog** pushes cat **further down**, credibly in both seeds:
  MAB **−17.5 / −27.5 pp**, MAB_L2 **−12.5 / −22.5 pp**.
- Deleting **sandwich**: seed 17 suggested recovery (MAC_L2 +15.0 pp), **seed 29
  did not replicate it** (−5.0 pp).
- **Headline: no evidence of reversal.** In the near branch the old deletion is
  reinforced. One-seed recovery was a false positive — D_cat(MA) alone moved
  10 pp between seeds.

## Slide 3 — The new request works, but costs retention  *(fig2, fig3)*

- dog deleted: 92% → **10%** (**+82.5 pp**, identical in both seeds).
- sandwich deleted: 88% → **38% / 35%** (**+50.0 / +52.5 pp**).
- **Collateral:** deleting dog knocks **bird** down **−45.0 / −40.0 pp** — a
  category never requested for deletion. horse (the anchor) is untouched
  (−2.5 pp); chair and bicycle move ≤10 pp.
- Suppression figures reproduce across seeds to within 2.5 pp — the most stable
  quantity in the study.

## Slide 4 — L2-SP: a tradeoff, not an improvement  *(fig2)*

- Movement from MA cut ~**9×** (0.0136 → 0.0015 relative L2).
- **Buys:** bird collateral **−45 pp → −5 pp** (identical in both seeds).
- **Costs:** dog deletion **+82.5 → +42.5 pp**; sandwich deletion
  **+50.0 → +7.5 pp**, an interval touching zero — the request is effectively
  **not carried out**.
- **Does not** protect the earlier deletion; cat recovery did not fall.
- One fixed coefficient (25000, a documented example, untuned) is roughly
  reasonable for dog and destroys the sandwich deletion → strength must be
  per-request.

## Slide 5 — Limitations and the next experiment

- Two training seeds; intervals are **sampling** uncertainty over prompts.
  n=2 is a direction check, not variance.
- 1000-step exploratory budget (upstream: 2000). Detector proxies only.
- Dog vs sandwich confounds **anchor sharing** with **semantic similarity**.
- Recovery/residue is larger under **paraphrased** prompts (+25.0 vs +5.0 pp,
  seed 17) — literal-only probes understate what survives.
- **No human annotation yet**; blinded grids + annotation sheet prepared.
- **Next:** factorise anchor vs semantics; more seeds before any recovery claim;
  per-request L2-SP strength chosen on a held-out split; longer streams.
