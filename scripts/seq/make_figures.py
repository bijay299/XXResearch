#!/usr/bin/env python3
"""Presentation figures for the SD-1.5 exploratory sequential object-erasure pilot.

Figures 1-3 (4 = image grids, built by make_grids.py):
  1. historical trajectory of cat detection: M0 -> MA -> each child,
     split by prompt family, both regularised and unregularised branches,
     with the shared parent MA labelled.
  2. tradeoff: new-target suppression vs historical cat recovery, matched arms
     connected, with common-control retention shown alongside.
  3. checkpoint x category heatmap, fixed 0-100 colour scale, counts annotated.

Palette: validated categorical slots (blue/orange/aqua/yellow) for lines and
blue/orange for the scatter; sequential single-hue blue for the heatmap. Aqua and
yellow sit below 3:1 on the light surface, so every line carries a direct label
and the numeric table (rates.csv / summary.md) ships alongside -- the relief rule.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical slots (light mode)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8882"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

BRANCH_STYLE = {
    "MAB":    dict(color=C_BLUE,   ls="-",  marker="o", label="MAB  (dog, no reg.)"),
    "MAB_L2": dict(color=C_BLUE,   ls="--", marker="s", label="MAB_L2  (dog, L2-SP)"),
    "MAC":    dict(color=C_ORANGE, ls="-",  marker="o", label="MAC  (sandwich, no reg.)"),
    "MAC_L2": dict(color=C_ORANGE, ls="--", marker="s", label="MAC_L2  (sandwich, L2-SP)"),
}
CATS = ["cat", "dog", "sandwich", "horse", "bird", "chair", "bicycle"]
CKPTS = ["M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf/.svg")


def D(con, ck, cat, fam, thr="0.5"):
    key = f"{cat}|{fam}"
    try:
        return con["raw_rates_pct"][ck][key][thr]
    except KeyError:
        return None


# --------------------------------------------------------------------------- 1
def fig_trajectory(con, out_dir):
    fams = ["literal", "paraphrase"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, fam in zip(axes, fams):
        style_axes(ax)
        xs = [0, 1, 2]
        d_m0, d_ma = D(con, "M0", "cat", fam), D(con, "MA", "cat", fam)
        for ck, st in BRANCH_STYLE.items():
            d_ch = D(con, ck, "cat", fam)
            if d_ch is None or d_ma is None:
                continue
            ax.plot(xs, [d_m0, d_ma, d_ch], color=st["color"], ls=st["ls"],
                    marker=st["marker"], ms=7, lw=2, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"{d_ch:.0f}", (2, d_ch), textcoords="offset points",
                        xytext=(8, 0), fontsize=9, color=INK, va="center")
        if d_ma is not None:
            ax.scatter([1], [d_ma], s=120, facecolor=SURFACE, edgecolor=INK,
                       lw=1.6, zorder=5)
            ax.annotate("shared parent MA\n(after cat deletion)", (1, d_ma),
                        textcoords="offset points", xytext=(0, -42), ha="center",
                        fontsize=8.5, color=INK2)
        ax.annotate(f"{d_m0:.0f}", (0, d_m0), textcoords="offset points",
                    xytext=(-10, 0), fontsize=9, color=INK, ha="right", va="center")
        ax.set_xticks(xs)
        ax.set_xticklabels(["M0\nuntouched", "MA\ncat deleted", "second request"],
                           fontsize=9)
        ax.set_xlim(-0.35, 2.55)
        ax.set_title(f"{fam} prompts", fontsize=11, color=INK, pad=8)
    axes[0].set_ylabel("cat detection rate D (%)  ·  t=0.5", fontsize=10, color=INK2)
    axes[0].set_ylim(-5, 105)
    handles = [plt.Line2D([], [], color=s["color"], ls=s["ls"], marker=s["marker"],
                          ms=6, lw=2, label=s["label"]) for s in BRANCH_STYLE.values()]
    axes[1].legend(handles=handles, frameon=False, fontsize=9, loc="upper left",
                   bbox_to_anchor=(1.02, 1.0), labelcolor=INK2)
    fig.suptitle("Does the cat deletion survive a second, unrelated deletion?",
                 fontsize=13, color=INK, y=1.02, x=0.5)
    fig.text(0.5, -0.07, "Detector-based object-presence proxy (Faster R-CNN COCO). "
             "Not UA/IRA/CRA. One training seed (17).",
             ha="center", fontsize=8, color=INK3)
    save(fig, out_dir, "fig1_historical_trajectory")


# --------------------------------------------------------------------------- 2
def fig_tradeoff(con, out_dir):
    fam = "all"
    blk = con["families"][fam]
    rec = blk["historical_cat_recovery"]
    sup = blk["new_request_suppression"]
    ret = blk["retention_incremental_vs_MA"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    style_axes(ax)
    for ck, st in BRANCH_STYLE.items():
        if ck not in rec or rec[ck] is None or sup.get(ck) is None:
            continue
        x, y = rec[ck]["diff_pp"], sup[ck]["diff_pp"]
        ax.scatter([x], [y], s=150, color=st["color"],
                   marker="o" if "L2" not in ck else "s",
                   edgecolor=SURFACE, lw=1.6, zorder=4)
        ax.annotate(ck, (x, y), textcoords="offset points", xytext=(10, 6),
                    fontsize=9.5, color=INK)
    # connect matched arms (same branch, with/without L2)
    for a, b in (("MAB", "MAB_L2"), ("MAC", "MAC_L2")):
        if all(k in rec and rec[k] and sup.get(k) for k in (a, b)):
            ax.plot([rec[a]["diff_pp"], rec[b]["diff_pp"]],
                    [sup[a]["diff_pp"], sup[b]["diff_pp"]],
                    color=BRANCH_STYLE[a]["color"], lw=1.4, ls=":", zorder=2, alpha=0.9)
    ax.axvline(0, color=INK3, lw=1, ls="-", alpha=0.6)
    ax.set_xlabel("historical cat recovery  D_cat(child) - D_cat(MA)   (pp)",
                  fontsize=10, color=INK2)
    ax.set_ylabel("new-target suppression\nD_new(MA) - D_new(child)   (pp)",
                  fontsize=10, color=INK2)
    ax.set_title("Preservation vs deletion of the new request", fontsize=11, color=INK)

    ax2 = axes[1]
    style_axes(ax2)
    controls = ["horse", "bird", "chair", "bicycle"]
    width = 0.2
    for i, (ck, st) in enumerate(BRANCH_STYLE.items()):
        vals = [(ret.get(ck, {}).get(c) or {}).get("diff_pp", 0) for c in controls]
        xs = [j + (i - 1.5) * width for j in range(len(controls))]
        ax2.bar(xs, vals, width=width * 0.9, color=st["color"],
                alpha=1.0 if "L2" not in ck else 0.55,
                edgecolor=SURFACE, lw=1.2, label=st["label"])
    ax2.axhline(0, color=INK3, lw=1)
    ax2.set_xticks(range(len(controls)))
    ax2.set_xticklabels(controls, fontsize=9.5)
    ax2.set_ylabel("retention change vs MA (pp)", fontsize=10, color=INK2)
    ax2.set_title("Common controls (not deletion targets)", fontsize=11, color=INK)
    ax2.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2, ncol=1)

    fig.text(0.5, -0.06, "Two settings only - no Pareto frontier is claimed. "
             "Branch-specific undeleted targets (dog in the sandwich branch and vice versa) "
             "are reported separately, not averaged in.",
             ha="center", fontsize=8, color=INK3)
    save(fig, out_dir, "fig2_tradeoff")


# --------------------------------------------------------------------------- 3
def fig_heatmap(con, out_dir):
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#f2f6fc", "#2a78d6", "#123a6b"])
    present = [c for c in CKPTS if c in con["raw_rates_pct"]]
    mat = [[D(con, ck, cat, "all") or 0.0 for cat in CATS] for ck in present]
    counts = con.get("counts", {})

    fig, ax = plt.subplots(figsize=(9.2, 0.72 * len(present) + 2.4))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels(CATS, fontsize=10)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=10)
    ax.tick_params(colors=INK2, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i, ck in enumerate(present):
        for j, cat in enumerate(CATS):
            v = mat[i][j]
            n = counts.get(ck, {}).get(cat, 0)
            ax.text(j, i, f"{v:.0f}\nn={n}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > 55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("detection rate D (%)  ·  fixed 0-100 scale", fontsize=9, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    ax.set_title("Detection rate by checkpoint and category (t=0.5, all prompts)",
                 fontsize=12, color=INK, pad=12)
    fig.text(0.5, -0.04, "cat = first deletion · dog/sandwich = second deletion "
             "(branch-specific) · horse = anchor-related control · bird/chair/bicycle = controls",
             ha="center", fontsize=8, color=INK3)
    save(fig, out_dir, "fig3_checkpoint_category_heatmap")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrasts", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    con = json.loads(Path(args.contrasts).read_text())
    out = Path(args.out_dir)
    print("figures:")
    fig_trajectory(con, out)
    fig_tradeoff(con, out)
    fig_heatmap(con, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
