# Five-slide outline (populated with measured results)

## Slide 1 — Question and setup

- Can an earlier deletion survive a later one, while the later one still works and utility is retained?
- A=cat, then B=dog or C=sandwich. Shared parent MA. L2-SP is an existing baseline added at the *second* request, not a new method.
- SD-1.5 backbone, CUIG ConAbl, 1000 steps/request, COCO Faster R-CNN evaluator, 280 images/checkpoint, frozen manifest.
- **Not** UA/IRA/CRA; not a published reproduction.

## Slide 2 — Does the first deletion hold?  (fig1)

- Initial cat suppression: +82.5 pp [+67.5, +95.0].

- MAB: cat recovery -17.5 pp [-32.5, -5.0].

- MAB_L2: cat recovery -12.5 pp [-22.5, -2.5].

- MAC: cat recovery +7.5 pp [+0.0, +15.0].

- MAC_L2: cat recovery +15.0 pp [+2.5, +27.5].

## Slide 3 — Did the new request work, and what did it cost?  (fig2, fig3)

- MAB → dog: suppression +82.5 pp [+65.0, +97.5] (M0 100% → MA 92% → 10%).

- MAB_L2 → dog: suppression +42.5 pp [+25.0, +62.5] (M0 100% → MA 92% → 50%).

- MAC → sandwich: suppression +50.0 pp [+35.0, +62.5] (M0 95% → MA 88% → 38%).

- MAC_L2 → sandwich: suppression +7.5 pp [+0.0, +17.5] (M0 95% → MA 88% → 80%).

- Common controls (horse/bird/chair/bicycle) change vs MA shown in fig2 right panel.

## Slide 4 — Does L2-SP change the tradeoff?

- MAB vs MAB_L2: recovery -17.5 pp [-32.5, -5.0] → -12.5 pp [-22.5, -2.5]; new-target suppression +82.5 pp [+65.0, +97.5] → +42.5 pp [+25.0, +62.5].

- MAC vs MAC_L2: recovery +7.5 pp [+0.0, +15.0] → +15.0 pp [+2.5, +27.5]; new-target suppression +50.0 pp [+35.0, +62.5] → +7.5 pp [+0.0, +17.5].

- Two settings only — no frontier claimed.

## Slide 5 — Limitations and next experiment

- One training seed; sampling uncertainty only. 1000-step exploratory budget. Detector proxies. No human annotation yet (blinded sheet prepared).

- Next: second training seed, then longer streams.

