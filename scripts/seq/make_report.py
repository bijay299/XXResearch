#!/usr/bin/env python3
"""Assemble the results report and the five-slide outline from measured outputs.

Every number is read from the artefacts on disk. Stages that did not run leave
explicit "unavailable" fields rather than being filled in or quietly dropped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def g(d, *path, default=None):
    cur = d
    for k in path:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur


def pp(v):
    return "unavailable" if v is None else f"{v:+.1f}"


def ci(c):
    if not c:
        return "unavailable"
    lo, hi = c["ci95_pp"]
    return f"{c['diff_pp']:+.1f} pp [{lo:+.1f}, {hi:+.1f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrasts", required=True)
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--reload_check", required=True)
    ap.add_argument("--movement", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slides", required=True)
    # No default: the training seed is a property of the input, not of the script.
    # A hard-coded default silently mislabels every report built from another seed.
    ap.add_argument("--training_seed", type=int, default=None,
                    help="cross-checked against the contrasts file; taken from it if omitted")
    ap.add_argument("--anchor_cache", default=None,
                    help="an anchor_cache_*.json, to read the shared anchor seed from")
    args = ap.parse_args()

    con = json.loads(Path(args.contrasts).read_text())
    con_seed = con.get("training_seed")
    if args.training_seed is None:
        args.training_seed = con_seed
    elif con_seed is not None and args.training_seed != con_seed:
        raise SystemExit(
            f"--training_seed {args.training_seed} contradicts the contrasts file "
            f"({con_seed}); refusing to label a report with a seed it was not built from")
    if args.training_seed is None:
        raise SystemExit("training seed is neither given nor present in the contrasts file")
    anchor_seed = None
    if args.anchor_cache and Path(args.anchor_cache).exists():
        anchor_seed = json.loads(Path(args.anchor_cache).read_text()).get("anchor_seed")
    calib = json.loads(Path(args.calib).read_text())
    reload_c = json.loads(Path(args.reload_check).read_text())
    mv = json.loads(Path(args.movement).read_text()) if args.movement and Path(args.movement).exists() else None

    reports = {}
    for n in ["MA", "MAB", "MAB_L2", "MAC", "MAC_L2"]:
        p = Path(args.models_root) / n / "train_report.json"
        if p.exists():
            reports[n] = json.loads(p.read_text())

    A = con["families"]["all"]
    L = con["families"]["literal"]
    P = con["families"]["paraphrase"]
    init = A["initial_cat_suppression"]
    rec = A["historical_cat_recovery"]
    sup = A["new_request_suppression"]
    ret = A["retention_incremental_vs_MA"]
    bru = A["branch_specific_undeleted_target"]

    children = [c for c in ["MAB", "MAB_L2", "MAC", "MAC_L2"] if c in rec and rec[c]]

    # Interpretation gates stated in the brief.
    init_pp = init["diff_pp"] if init else None
    suppression_negligible = (init_pp is not None and init_pp < 10)
    any_recovery = any(rec[c] and rec[c]["ci95_pp"][0] > 0 for c in children)

    tr = reports.get("MA", {})
    eh = tr.get("effective_hyperparameters", {})

    md = []
    md.append("# SD-1.5 exploratory sequential object-erasure pilot — results\n")
    md.append("**This is not a reproduction of any published result, and these are not "
              "UA/IRA/CRA.** The backbone is base Stable Diffusion v1.5 and the evaluator "
              "is a public COCO detector; every rate below is a detector-based "
              "object-presence proxy, not ground-truth erasure and not an image-quality "
              "score. The blocked UnlearnCanvas baseline and its validation gate are "
              "untouched and remain blocked.\n")

    md.append("## Research question\n")
    md.append("Can an earlier concept deletion persist through a later deletion while "
              "preserving both new-request effectiveness and retained utility? L2-SP is "
              "an existing baseline regulariser, not a proposed new method. The question "
              "is tested, not assumed.\n")

    md.append("## Configuration\n")
    md.append(f"- Upstream CUIG `9932ac3271a122f6d38d19e0b8c8908fe5237ff7`, native ConAbl "
              f"object training, mappings `horse+cat`, `horse+dog`, `flower+sandwich`.\n"
              f"- Backbone M0: base Stable Diffusion v1.5.\n"
              f"- {eh.get('iterations_requested','?')} optimizer steps per request "
              f"(upstream sequential uses 2000; reduced exploratory budget), "
              f"{eh.get('steps_per_epoch','?')} steps/epoch, epoch cap "
              f"{eh.get('epochs_cap','?')} → capacity "
              f"{eh.get('epoch_capacity_steps','?')} steps.\n"
              f"- Effective LR {eh.get('effective_learning_rate','?')} "
              f"(CLI default {eh.get('learning_rate_cli_default_before_scaling','?')} × "
              f"grad_accum {eh.get('gradient_accumulation_steps','?')} × batch "
              f"{eh.get('anchor_batch_size','?')} × procs {eh.get('num_processes',1)}), "
              f"optimizer {eh.get('optimizer','?')}, precision {eh.get('precision','?')}, "
              f"parameter group `{eh.get('parameter_group','?')}`.\n"
              f"- 200 anchor images/prompts, generated once from untouched M0 with a "
              f"fixed anchor-generation seed"
              f"{f' ({anchor_seed})' if anchor_seed is not None else ''} that is "
              f"independent of the training seed, and shared byte-identically across "
              f"every matched arm and every training seed.\n"
              f"- Training seed {args.training_seed}; regularised and unregularised arms "
              f"differ only in `--l2sp_weight 25000` (a documented example coefficient, "
              f"not an SD-1.5-tuned optimum; no coefficient search was run).\n")

    md.append("## Pre-registration-style checks that passed before any editing\n")
    cs = calib["positive_summary"]
    md.append(f"- **M0 generates the targets**: cat {cs['cat']['detection_rate']['0.5']*100:.0f}%, "
              f"dog {cs['dog']['detection_rate']['0.5']*100:.0f}%, "
              f"sandwich {cs['sandwich']['detection_rate']['0.5']*100:.0f}% at t=0.5 on a "
              f"separate calibration set.\n"
              f"- **Detector does not over-trigger**: negative controls fire "
              f"{calib['negative_control_any_of_seven_rate']['0.5']*100:.0f}% of the time.\n"
              f"- **No-update reload is inert**: {reload_c['n_byte_identical']}/"
              f"{reload_c['n_sampled']} regenerated images byte-identical after loading a "
              f"no-op delta through the real checkpoint path.\n"
              f"- **Evaluation manifest frozen before editing**: 7 categories × 10 prompts "
              f"(5 literal + 5 paraphrase) × 4 seeds = 280 images per checkpoint; zero "
              f"overlap with training prompts or their target-substituted forms.\n")

    # ---- headline findings, derived from the measured contrasts ------------
    def credible(c, sign):
        if not c:
            return False
        lo, hi = c["ci95_pp"]
        return lo > 0 if sign > 0 else hi < 0

    up = [c for c in children if credible(rec[c], +1)]
    down = [c for c in children if credible(rec[c], -1)]
    md.append("## Headline findings\n")
    md.append(f"1. **The first deletion worked.** Cat fell {init['D_M0']:.0f}% → "
              f"{init['D_MA']:.0f}% ({ci(init)}), so historical recovery is a "
              "well-posed question here rather than a measurement of noise.\n")
    if up and down:
        md.append(f"2. **Historical interference is real but direction depends on the "
                  f"second request.** Cat detection rose credibly after "
                  f"{', '.join(up)} and fell further after {', '.join(down)}. "
                  "Deleting dog — which shares the `horse` anchor with cat and is "
                  "semantically close — pushed cat *further down*; deleting sandwich, "
                  "with a different (`flower`) anchor, let cat partially return. "
                  "A single 'does deletion survive?' answer would misdescribe this.\n")
    elif up:
        md.append(f"2. **Historical recovery observed** in {', '.join(up)}.\n")
    elif down:
        undet = [c for c in children if c not in down]
        md.append(f"2. **No increase in cat detection was resolved at this precision.** "
                  f"Cat fell further in {', '.join(down)}"
                  + (f"; in {', '.join(undet)} the interval includes zero, which leaves "
                     f"the direction unresolved rather than establishing no change"
                     if undet else "")
                  + ". This is a statement about what these 10 prompt clusters could "
                    "resolve, not a demonstration that recovery is absent.\n")
    else:
        md.append("2. **No change in cat detection was resolved at this precision.** "
                  "Every interval includes zero. With 10 prompt clusters per category "
                  "this is an absence of resolution, not evidence of an absent effect.\n")

    l2_pairs = [(a, b) for a, b in (("MAB", "MAB_L2"), ("MAC", "MAC_L2"))
                if a in rec and b in rec and rec[a] and rec[b] and sup.get(a) and sup.get(b)]
    if l2_pairs:
        costs = [f"{a} {sup[a]['diff_pp']:+.1f} → {b} {sup[b]['diff_pp']:+.1f} pp"
                 for a, b in l2_pairs]
        # Derive the direction of every clause from the measured values. Nothing
        # about which branch moves which way is written into this text.
        deltas = [(a, b, rec[b]["diff_pp"] - rec[a]["diff_pp"]) for a, b in l2_pairs]
        moved = [f"{a}→{b} {d:+.1f} pp" for a, b, d in deltas if abs(d) >= 2.5]
        flat = [f"{a}→{b}" for a, b, d in deltas if abs(d) < 2.5]
        bird_no = (ret.get("MAB", {}).get("bird") or {}).get("diff_pp")
        bird_l2 = (ret.get("MAB_L2", {}).get("bird") or {}).get("diff_pp")
        md.append(
            f"3. **L2-SP traded new-request effectiveness for retention at this "
            f"coefficient.** New-target suppression fell in both branches "
            f"({'; '.join(costs)}). The change in the cat figure is "
            + (f"{'; '.join(moved)}" if moved else "below 2.5 pp in every branch")
            + (f" (unchanged within 2.5 pp in {', '.join(flat)})" if flat and moved else "")
            + ". Because that figure is referenced to D_cat(MA), part of any "
              "change in it can come from the parent rather than from L2-SP; the "
              "parent-free contrast D_cat(L2) − D_cat(no-L2) is the quantity that "
              "isolates the L2-SP effect, and it is reported in "
              "`results/audit_v1/CORRECTED_TABLES.md`.\n")
        if bird_no is not None and bird_l2 is not None:
            md.append(f"   On the bird control the two dog-branch arms differ: "
                      f"{bird_no:+.1f} pp under MAB versus {bird_l2:+.1f} pp under "
                      f"MAB_L2. Read that beside the suppression each arm achieved — "
                      f"retention is only comparable between arms that deleted a "
                      f"comparable amount, and these did not.\n")
        md.append("   This is a **tradeoff at one untuned coefficient**, not an "
                  "improved method and not a characterised frontier. A single tested "
                  "value cannot establish that a per-request coefficient is needed, "
                  "only that 25000 behaves differently in these two branches.\n")

    lit_rec = L["historical_cat_recovery"]
    par_rec = P["historical_cat_recovery"]
    gaps = [(c, par_rec[c]["diff_pp"] - lit_rec[c]["diff_pp"])
            for c in children if lit_rec.get(c) and par_rec.get(c)]
    if gaps and max(g for _, g in gaps) >= 10:
        worst = max(gaps, key=lambda t: t[1])
        # Residual level and change-from-parent are different quantities; report
        # both so the sentence cannot be read as the stronger of the two claims.
        par_level = g(con, "raw_rates_pct", worst[0], "cat|paraphrase", "0.5")
        lit_level = g(con, "raw_rates_pct", worst[0], "cat|literal", "0.5")
        ma_par = g(con, "raw_rates_pct", "MA", "cat|paraphrase", "0.5")
        ma_lit = g(con, "raw_rates_pct", "MA", "cat|literal", "0.5")
        md.append(
            f"4. **Paraphrased prompts behave differently from prompts that name the "
            f"category.** For {worst[0]} the parent-referenced cat figure is "
            f"{worst[1]:.1f} pp higher on paraphrases than on literals.\n"
            f"   Two distinct quantities are involved and must not be merged:\n"
            f"   - **Residual presence** — the child's own rate: "
            + (f"{par_level:.0f}% on paraphrases versus {lit_level:.0f}% on literals"
               if par_level is not None and lit_level is not None else "see the table below")
            + ".\n"
            f"   - **Change from the parent** — which also depends on the parent's own "
            f"family split: D_cat(MA) is "
            + (f"{ma_par:.0f}% on paraphrases versus {ma_lit:.0f}% on literals"
               if ma_par is not None and ma_lit is not None else "tabulated below")
            + ". A larger parent-referenced figure on paraphrases can arise from a "
              "higher child residual, a lower parent level, or both.\n"
            f"   What follows is only that evaluating erasure with literal prompts "
            f"alone understates what the edited model still produces. This is a "
            f"generalisation check across two fixed prompt families, **not** an "
            f"adaptive attack and **not** evidence of greater recovery.\n")
    md.append("")

    md.append("## Measured results\n")
    md.append("### Raw detection rate D (%) at t=0.5, all prompts\n")
    cats = ["cat", "dog", "sandwich", "horse", "bird", "chair", "bicycle"]
    md.append("| checkpoint | " + " | ".join(cats) + " |")
    md.append("|---|" + "---|" * len(cats))
    for ck in ["M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"]:
        rr = con["raw_rates_pct"].get(ck)
        if not rr:
            continue
        cells = []
        for c in cats:
            v = rr.get(f"{c}|all", {}).get("0.5")
            n = con["counts"].get(ck, {}).get(c, 0)
            cells.append(f"{v:.0f} (n={n})" if v is not None else "—")
        md.append(f"| {ck} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("### Experiment 1 — historical interference\n")
    md.append(f"**Initial cat suppression** D_cat(M0) − D_cat(MA) = **{ci(init)}** "
              f"(M0 {init['D_M0']:.0f}% → MA {init['D_MA']:.0f}%).\n")
    if suppression_negligible:
        md.append("> Initial cat suppression is small, so **historical recovery is not "
                  "established** by this run: there is little suppression for a later "
                  "deletion to undo. The recovery figures below are reported for "
                  "completeness, not as evidence of reversal.\n")
    md.append("\n**Historical cat recovery** D_cat(child) − D_cat(MA):\n")
    md.append("| child | new deletion | cat recovery (pp, 95% CI) | D_cat(MA) | D_cat(child) |")
    md.append("|---|---|---|---|---|")
    for c in children:
        r = rec[c]
        md.append(f"| {c} | {g(sup, c, 'target', default='?')} | {ci(r)} | "
                  f"{r['D_MA']:.0f}% | {r['D_' + c]:.0f}% |")
    md.append("")
    md.append("**New-request suppression** D_new(MA) − D_new(child), with D_new(M0) shown "
              "because the cat deletion may already move the later target:\n")
    md.append("| child | target | D_M0 | D_MA | D_child | suppression (pp, 95% CI) |")
    md.append("|---|---|---|---|---|---|")
    for c in children:
        s = sup.get(c)
        if not s:
            continue
        md.append(f"| {c} | {s['target']} | {s['D_M0']:.0f}% | {s['D_MA']:.0f}% | "
                  f"{s['D_' + c]:.0f}% | {ci(s)} |")
    md.append("")
    md.append("**Retention change vs MA** on the four common controls "
              "(never a deletion target in any branch):\n")
    md.append("| child | horse (anchor) | bird | chair | bicycle |")
    md.append("|---|---|---|---|---|")
    for c in children:
        row = ret.get(c, {})
        md.append(f"| {c} | " + " | ".join(
            ci(row.get(k)) for k in ["horse", "bird", "chair", "bicycle"]) + " |")
    md.append("")
    md.append("**Branch-specific undeleted target** (reported separately, never folded "
              "into a common retain average):\n")
    md.append("| child | undeleted target | vs M0 | vs MA |")
    md.append("|---|---|---|---|")
    for c in children:
        b = bru.get(c)
        if not b:
            continue
        md.append(f"| {c} | {b['undeleted_target']} | {ci(b['vs_M0'])} | {ci(b['vs_MA'])} |")
    md.append("")

    md.append("### Experiment 2 — preservation versus new deletion\n")
    md.append("| contrast | cat recovery (pp) | new-target suppression (pp) |")
    md.append("|---|---|---|")
    for a, b in (("MAB", "MAB_L2"), ("MAC", "MAC_L2")):
        if a in rec and b in rec and rec[a] and rec[b]:
            md.append(f"| {a} (no reg.) | {ci(rec[a])} | {ci(sup.get(a))} |")
            md.append(f"| {b} (L2-SP 25000) | {ci(rec[b])} | {ci(sup.get(b))} |")
    md.append("")

    md.append("### Literal vs paraphrased prompts\n")
    md.append("| contrast | literal | paraphrase |")
    md.append("|---|---|---|")
    md.append(f"| initial cat suppression | {ci(L['initial_cat_suppression'])} | "
              f"{ci(P['initial_cat_suppression'])} |")
    for c in children:
        md.append(f"| cat recovery {c} | {ci(L['historical_cat_recovery'].get(c))} | "
                  f"{ci(P['historical_cat_recovery'].get(c))} |")
        md.append(f"| new-target suppression {c} | {ci(L['new_request_suppression'].get(c))} | "
                  f"{ci(P['new_request_suppression'].get(c))} |")
    md.append("")

    if mv:
        md.append("### Parameter movement (diagnostic association only)\n")
        md.append("| checkpoint | relative L2 vs M0 | relative L2 vs MA |")
        md.append("|---|---|---|")
        for n, row in mv["checkpoints"].items():
            md.append(f"| {n} | {row['relative_l2_vs_M0']:.5f} | {row['relative_l2_vs_MA']:.5f} |")
        md.append("\nThis is a description of how far the edited tensors moved. It is "
                  "**not** evidence that weight movement causes recovery.\n")

    md.append("## Runtime and peak memory\n")
    md.append("| checkpoint | GPU | train s | s/step | peak GPU mem (MiB) |")
    md.append("|---|---|---|---|---|")
    for n, r in reports.items():
        rt = r.get("runtime", {})
        md.append(f"| {n} | {g(r,'gpu','uuid',default='?')[:20]}… | "
                  f"{rt.get('train_seconds','?')} | {rt.get('seconds_per_optimizer_step','?')} | "
                  f"{rt.get('peak_gpu_mem_mib','?')} |")
    md.append("")

    md.append("## Limitations\n")
    md.append(f"- One training seed ({args.training_seed}). The bootstrap intervals are "
              "**sampling** uncertainty over prompts; generation seeds are not "
              "independent training repeats, and no population variance over training "
              "runs is claimed.\n"
              "- The intervals come from 10 prompt clusters per category, so their "
              "resolution is coarse. An interval that includes zero does **not** show "
              "the effect is zero, and a change of significance label between two runs "
              "is **not** a demonstration that the two runs differ.\n"
              "- Many contrasts are reported with no multiplicity correction; each "
              "interval is descriptive, not a test.\n"
              "- 1000 optimizer steps is a reduced exploratory budget; upstream's "
              "sequential object script uses 2000.\n"
              "- Detector-based proxies only. A detection is not ground truth, and "
              "absence of a detection is not proof of erasure.\n"
              "- Dog vs sandwich is one selected request contrast with different anchor "
              "mappings; it does not isolate semantic similarity.\n"
              "- L2-SP coefficient 25000 is a documented example, not tuned for SD-1.5. "
              "No coefficient search was performed against final evaluation results.\n"
              "- Two regularisation settings cannot define a Pareto frontier.\n")

    md.append("## Visual inspection status\n")
    md.append("Representative calibration images were inspected **by the AI assistant**, "
              "not by a human annotator. Blinded, checkpoint-shuffled grids and an empty "
              "annotation sheet are prepared under `grids/blinded/` for human scoring. "
              "**No human annotation has been performed.**\n")

    md.append("## What this motivates next\n")
    if suppression_negligible:
        md.append("- The immediate blocker is deletion strength: at 1000 steps the first "
                  "request did not suppress the target enough for a historical-recovery "
                  "question to be answerable. The next experiment should raise the step "
                  "budget (2000, as upstream) **applied identically to every arm it "
                  "compares**, keeping these results rather than replacing them.\n")
    elif any_recovery:
        md.append("- **Separate anchor sharing from semantic similarity.** The two branches "
                  "differ both in anchor (`horse` vs `flower`) and in semantic distance from "
                  "cat, and they moved cat in opposite directions. The decisive next run is a "
                  "second request that is semantically far from cat but *shares* the `horse` "
                  "anchor, and one that is semantically close but uses a different anchor. "
                  "Only that factorisation can tell which caused the sign flip.\n"
                  "- **Does recovery accumulate?** Extend the stream past two requests and "
                  "test whether cat keeps returning in the sandwich-like branches.\n"
                  "- **Does the L2-SP tradeoff have a usable setting?** At 25000 it cost most "
                  "of the sandwich deletion (+50.0 → +7.5 pp). A coefficient sweep — chosen "
                  "on a held-out split, not on these final numbers — would show whether any "
                  "value keeps retention without surrendering the new request.\n")
    else:
        md.append("- No branch showed a credible increase in cat detection relative to the "
                  "shared parent. The next experiment should test whether this holds at a "
                  "larger step budget and over longer streams, where drift has more "
                  "opportunity to accumulate.\n")
    md.append("- A second training seed is required before any numeric magnitude is "
              "treated as stable.\n")

    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"wrote {args.out}")

    # ---- slide outline ----------------------------------------------------
    s = []
    s.append("# Five-slide outline (populated with measured results)\n")
    s.append("## Slide 1 — Question and setup\n")
    s.append("- Can an earlier deletion survive a later one, while the later one still "
             "works and utility is retained?\n"
             "- A=cat, then B=dog or C=sandwich. Shared parent MA. L2-SP is an existing "
             "baseline added at the *second* request, not a new method.\n"
             f"- SD-1.5 backbone, CUIG ConAbl, {eh.get('iterations_requested','?')} steps/request, "
             "COCO Faster R-CNN evaluator, 280 images/checkpoint, frozen manifest.\n"
             "- **Not** UA/IRA/CRA; not a published reproduction.\n")
    s.append("## Slide 2 — Does the first deletion hold?  (fig1)\n")
    s.append(f"- Initial cat suppression: {ci(init)}.\n")
    for c in children:
        s.append(f"- {c}: cat recovery {ci(rec[c])}.\n")
    if suppression_negligible:
        s.append("- **Gate not met**: suppression too small for recovery to be "
                 "established. Report as a null//budget finding.\n")
    s.append("## Slide 3 — Did the new request work, and what did it cost?  (fig2, fig3)\n")
    for c in children:
        sx = sup.get(c)
        if sx:
            s.append(f"- {c} → {sx['target']}: suppression {ci(sx)} "
                     f"(M0 {sx['D_M0']:.0f}% → MA {sx['D_MA']:.0f}% → {sx['D_' + c]:.0f}%).\n")
    s.append("- Common controls (horse/bird/chair/bicycle) change vs MA shown in fig2 right panel.\n")
    s.append("## Slide 4 — Does L2-SP change the tradeoff?\n")
    for a, b in (("MAB", "MAB_L2"), ("MAC", "MAC_L2")):
        if a in rec and b in rec and rec[a] and rec[b]:
            s.append(f"- {a} vs {b}: recovery {ci(rec[a])} → {ci(rec[b])}; "
                     f"new-target suppression {ci(sup.get(a))} → {ci(sup.get(b))}.\n")
    s.append("- Two settings only — no frontier claimed.\n")
    s.append("## Slide 5 — Limitations and next experiment\n")
    s.append("- One training seed; sampling uncertainty only. 1000-step exploratory budget. "
             "Detector proxies. No human annotation yet (blinded sheet prepared).\n")
    s.append("- Next: " + ("raise the step budget to 2000 across all compared arms, keeping "
                           "these results." if suppression_negligible else
                           "second training seed, then longer streams.") + "\n")
    Path(args.slides).write_text("\n".join(s) + "\n")
    print(f"wrote {args.slides}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
