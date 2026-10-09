#!/usr/bin/env python3
"""Show two training seeds SIDE BY SIDE.

Two training runs do not support a population variance claim. This script
deliberately reports each seed's numbers separately and asks only the weaker,
answerable question: does the *sign and direction* of each effect agree across
the two seeds? No pooling, no mean-of-two, no standard error over n=2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHILDREN = ["MAB", "MAB_L2", "MAC", "MAC_L2"]


def fmt(c):
    if not c:
        return "n/a"
    lo, hi = c["ci95_pp"]
    return f"{c['diff_pp']:+.1f} [{lo:+.1f},{hi:+.1f}]"


def sign(c):
    """Credibility label, not just a raw sign.

    'ns' means the 95% interval includes 0, so the run supports no directional
    claim. Two runs that are both 'ns' agree only in the weak sense of finding
    nothing; that is reported as 'both ns', never as a confirmed effect.
    """
    if not c:
        return None
    lo, hi = c["ci95_pp"]
    if lo > 0:
        return "increase"
    if hi < 0:
        return "decrease"
    return "ns"


def agreement(a, b):
    sa, sb = sign(a), sign(b)
    if sa is None or sb is None:
        return "n/a"
    if sa == sb:
        return "both ns" if sa == "ns" else f"yes ({sa})"
    return f"**NO** ({sa} vs {sb})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="contrasts.json for seed A")
    ap.add_argument("--b", required=True, help="contrasts.json for seed B")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_fig", default=None)
    args = ap.parse_args()

    A = json.loads(Path(args.a).read_text())
    B = json.loads(Path(args.b).read_text())
    sa, sb = A["training_seed"], B["training_seed"]
    fa, fb = A["families"]["all"], B["families"]["all"]

    md = [f"# Training seeds {sa} and {sb}, reported separately\n",
          "Two training runs cannot establish population variance. Each seed's "
          "intervals below are **sampling** uncertainty over prompts within that "
          "single run. The only cross-seed claim made here is whether the "
          "**direction** of an effect agrees.\n",
          "## Initial cat suppression\n",
          f"| seed {sa} | seed {sb} |", "|---|---|",
          f"| {fmt(fa['initial_cat_suppression'])} | {fmt(fb['initial_cat_suppression'])} |", ""]

    md += ["## Historical cat recovery  D_cat(child) − D_cat(MA)\n",
           f"| child | seed {sa} | seed {sb} | direction agrees? |", "|---|---|---|---|"]
    agree_rec = []
    for c in CHILDREN:
        ra, rb = fa["historical_cat_recovery"].get(c), fb["historical_cat_recovery"].get(c)
        ok = agreement(ra, rb)
        agree_rec.append(ok.startswith("yes"))
        md.append(f"| {c} | {fmt(ra)} | {fmt(rb)} | {ok} |")
    md.append("")

    md += ["## New-request suppression  D_new(MA) − D_new(child)\n",
           f"| child | target | seed {sa} | seed {sb} | direction agrees? |",
           "|---|---|---|---|---|"]
    for c in CHILDREN:
        ra, rb = fa["new_request_suppression"].get(c), fb["new_request_suppression"].get(c)
        ok = agreement(ra, rb)
        tgt = (ra or {}).get("target", "?")
        md.append(f"| {c} | {tgt} | {fmt(ra)} | {fmt(rb)} | {ok} |")
    md.append("")

    md += ["## Raw cat detection rate D (%) at t=0.5\n",
           f"| checkpoint | seed {sa} | seed {sb} |", "|---|---|---|"]
    for ck in ["M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"]:
        va = A["raw_rates_pct"].get(ck, {}).get("cat|all", {}).get("0.5")
        vb = B["raw_rates_pct"].get(ck, {}).get("cat|all", {}).get("0.5")
        md.append(f"| {ck} | {va if va is not None else '—'} | {vb if vb is not None else '—'} |")
    md.append("")

    n_ok = sum(agree_rec)
    md.append(f"**Credible, same-direction agreement on historical recovery: "
              f"{n_ok}/{len(CHILDREN)} children.** 'both ns' is not agreement on an "
              "effect -- it is two runs each finding nothing. With n=2 training runs "
              "this is a consistency check, not an estimate of variability.\n")
    disagreements = [c for c in CHILDREN
                     if agreement(fa["historical_cat_recovery"].get(c),
                                  fb["historical_cat_recovery"].get(c)).startswith("**NO")]
    if disagreements:
        md.append(
            f"\n> **Replication warning.** {', '.join(disagreements)} changed "
            "credibility class between the two training seeds, so no claim resting "
            "on that child is established by this pair of runs.\n>\n"
            "> Three things this does **not** license, all of which are easy to write "
            "by accident:\n"
            "> - It is **not** a finding that the effect is absent or was a false "
            "positive. A differing significance label across two runs is not a "
            "significant difference between them; the two intervals here overlap, "
            "and neither run had the precision to resolve a small effect.\n"
            "> - It does **not** by itself locate the instability in the method. "
            "`D_cat(child) − D_cat(MA)` moves when the parent moves, so a parent "
            "difference alone can flip its sign. The parent-free contrast "
            "`D_cat(L2) − D_cat(no-L2)` separates the two and is reported in "
            "`results/audit_v1/CORRECTED_TABLES.md`.\n"
            "> - It does **not** mean the seeds agree about everything else. "
            "Agreement is assessed per quantity, at the stated threshold, and only "
            "on direction.\n>\n"
            "> The supportable statement is: **this pair of runs does not resolve "
            "the sign of that quantity, and more training runs are required before "
            "it is reported either way.**\n")

    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(f"wrote {args.out_md}")

    if args.out_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
        INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        xs = range(len(CHILDREN))
        va = [(fa["historical_cat_recovery"].get(c) or {}).get("diff_pp", 0) for c in CHILDREN]
        vb = [(fb["historical_cat_recovery"].get(c) or {}).get("diff_pp", 0) for c in CHILDREN]
        ax.bar([x - 0.19 for x in xs], va, width=0.36, color=C_BLUE,
               edgecolor=SURF, lw=1.2, label=f"training seed {sa}")
        ax.bar([x + 0.19 for x in xs], vb, width=0.36, color=C_ORANGE,
               edgecolor=SURF, lw=1.2, label=f"training seed {sb}")
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(list(xs)); ax.set_xticklabels(CHILDREN, fontsize=10)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_ylabel("historical cat recovery vs MA (pp)", fontsize=10, color=INK2)
        ax.set_title("Two training seeds, shown separately", fontsize=12, color=INK)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
        fig.text(0.5, -0.04, "n=2 training runs: a direction check, not a variance estimate.",
                 ha="center", fontsize=8, color="#8a8882")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{args.out_fig}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURF)
        print(f"wrote {args.out_fig}.png/.pdf/.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
