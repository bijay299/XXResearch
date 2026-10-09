#!/usr/bin/env python3
"""Aggregate detections into the per-image table, category rates and contrasts.

Outputs
-------
per_image.csv      one row per image: seed, method, checkpoint, parent, category,
                   prompt id/family, generation seed, image path/hash, detections
rates.csv          detection rate D by checkpoint x category x prompt family
contrasts.json     the signed percentage-point differences required by the study
summary.md         human-readable tables

Uncertainty: paired bootstrap resampled BY PROMPT (each prompt carries its four
generation seeds), which preserves prompt/seed matching across checkpoints. One
training seed supports sampling uncertainty only -- generation seeds are not
independent training repeats.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

CATS = ["cat", "dog", "sandwich", "horse", "bird", "chair", "bicycle"]
PARENTS = {"M0": None, "MA": "M0", "MAB": "MA", "MAC": "MA",
           "MAB_L2": "MA", "MAC_L2": "MA"}
METHOD = {"M0": "untouched", "MA": "ConAbl", "MAB": "ConAbl", "MAC": "ConAbl",
          "MAB_L2": "ConAbl+L2SP(25000)", "MAC_L2": "ConAbl+L2SP(25000)"}
NEW_TARGET = {"MAB": "dog", "MAC": "sandwich", "MAB_L2": "dog", "MAC_L2": "sandwich"}
# Controls that are NOT a deletion target in any branch -> safe to average.
COMMON_CONTROLS = ["horse", "bird", "chair", "bicycle"]


def load(jsonl: Path) -> list[dict]:
    return [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]


def rate(rows: list[dict], thr: str) -> float:
    return sum(r["hit"][thr] for r in rows) / len(rows) if rows else float("nan")


def boot_diff(rows_a: list[dict], rows_b: list[dict], thr: str,
              n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of D(a) - D(b), resampling prompts (with their seeds)."""
    by_a: dict[str, list[dict]] = defaultdict(list)
    by_b: dict[str, list[dict]] = defaultdict(list)
    for r in rows_a:
        by_a[r["prompt_id"]].append(r)
    for r in rows_b:
        by_b[r["prompt_id"]].append(r)
    prompts = sorted(set(by_a) & set(by_b))
    if not prompts:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    point = rate([x for p in prompts for x in by_a[p]], thr) - \
            rate([x for p in prompts for x in by_b[p]], thr)
    draws = []
    for _ in range(n):
        samp = [prompts[rng.randrange(len(prompts))] for _ in prompts]
        a = [x for p in samp for x in by_a[p]]
        b = [x for p in samp for x in by_b[p]]
        draws.append(rate(a, thr) - rate(b, thr))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return point * 100, lo * 100, hi * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--training_seed", type=int, default=17)
    ap.add_argument("--thr", default="0.5")
    args = ap.parse_args()

    root, out = Path(args.eval_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data: dict[str, list[dict]] = {}
    for ck in PARENTS:
        p = root / ck / "detections.jsonl"
        if p.exists():
            data[ck] = load(p)
            print(f"  {ck}: {len(data[ck])} images")
        else:
            print(f"  {ck}: MISSING ({p})")

    # ---- per-image table ---------------------------------------------------
    hdr = ["training_seed", "method", "checkpoint", "parent", "category", "prompt_id",
           "prompt_family", "prompt_index", "gen_seed", "image_path", "image_sha256",
           "max_score_target", "hit_0.3", "hit_0.5", "hit_0.7", "n_detections_total",
           "top_label", "top_score"]
    lines = [",".join(hdr)]
    for ck, rows in data.items():
        for r in rows:
            dets = r["detections"]
            top = max(dets, key=lambda d: d["score"]) if dets else {"label": "", "score": 0.0}
            lines.append(",".join(str(x) for x in [
                args.training_seed, METHOD[ck], ck, PARENTS[ck] or "", r["category"],
                r["prompt_id"], r["prompt_family"], r["prompt_index"], r["gen_seed"],
                r["image_path"], (r.get("image_sha256") or "")[:16],
                f"{r['max_score_target']:.4f}", int(r["hit"]["0.3"]), int(r["hit"]["0.5"]),
                int(r["hit"]["0.7"]), len(dets), top["label"], f"{top['score']:.4f}"]))
    (out / "per_image.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'per_image.csv'} ({len(lines)-1} rows)")

    # ---- rates -------------------------------------------------------------
    def sel(ck, cat, fam=None):
        return [r for r in data.get(ck, [])
                if r["category"] == cat and (fam is None or r["prompt_family"] == fam)]

    rate_rows = ["checkpoint,category,prompt_family,n,D_0.3,D_0.5,D_0.7,n_hits_0.5"]
    rates: dict = defaultdict(dict)
    for ck in data:
        for cat in CATS:
            for fam in (None, "literal", "paraphrase"):
                rows = sel(ck, cat, fam)
                if not rows:
                    continue
                famname = fam or "all"
                d = {t: rate(rows, t) for t in ("0.3", "0.5", "0.7")}
                rates[ck][(cat, famname)] = d
                rate_rows.append(f"{ck},{cat},{famname},{len(rows)},"
                                 f"{d['0.3']:.4f},{d['0.5']:.4f},{d['0.7']:.4f},"
                                 f"{sum(r['hit']['0.5'] for r in rows)}")
    (out / "rates.csv").write_text("\n".join(rate_rows) + "\n")

    # ---- contrasts ---------------------------------------------------------
    T = args.thr
    con: dict = {"training_seed": args.training_seed, "threshold": T,
                 "note": ("signed percentage-point differences; detector-based "
                          "object-presence proxies, not UA/IRA/CRA"),
                 "uncertainty": ("paired bootstrap by prompt (2000 draws, 95% CI); "
                                 "one training seed = sampling uncertainty only"),
                 "families": {}}

    def contrast(a, b, cat, fam):
        ra, rb = sel(a, cat, fam), sel(b, cat, fam)
        if not ra or not rb:
            return None
        pt, lo, hi = boot_diff(ra, rb, T)
        return {"diff_pp": round(pt, 2), "ci95_pp": [round(lo, 2), round(hi, 2)],
                f"D_{a}": round(rate(ra, T) * 100, 2), f"D_{b}": round(rate(rb, T) * 100, 2),
                "n_images_each": len(ra)}

    for fam in ("all", "literal", "paraphrase"):
        f = None if fam == "all" else fam
        blk: dict = {}
        # 1. initial cat suppression: D_cat(M0) - D_cat(MA)
        blk["initial_cat_suppression"] = contrast("M0", "MA", "cat", f)
        # 2. historical cat recovery: D_cat(child) - D_cat(MA)
        blk["historical_cat_recovery"] = {
            ck: contrast(ck, "MA", "cat", f) for ck in NEW_TARGET if ck in data}
        # 3. new-request suppression: D_new(MA) - D_new(child); plus D_new(M0)
        blk["new_request_suppression"] = {}
        for ck, tgt in NEW_TARGET.items():
            if ck not in data:
                continue
            c = contrast("MA", ck, tgt, f)
            if c:
                m0 = sel("M0", tgt, f)
                c["target"] = tgt
                c["D_M0"] = round(rate(m0, T) * 100, 2) if m0 else None
                c["note_first_deletion_effect_on_target"] = (
                    "compare D_M0 with D_MA: the cat deletion may already move this target")
            blk["new_request_suppression"][ck] = c
        # 4. retention change vs M0 and incremental vs MA
        blk["retention_vs_M0"] = {
            ck: {cat: contrast(ck, "M0", cat, f) for cat in COMMON_CONTROLS}
            for ck in data if ck != "M0"}
        blk["retention_incremental_vs_MA"] = {
            ck: {cat: contrast(ck, "MA", cat, f) for cat in COMMON_CONTROLS}
            for ck in NEW_TARGET if ck in data}
        # 5. branch-specific: the undeleted sibling target
        blk["branch_specific_undeleted_target"] = {}
        for ck, tgt in NEW_TARGET.items():
            if ck not in data:
                continue
            other = "sandwich" if tgt == "dog" else "dog"
            blk["branch_specific_undeleted_target"][ck] = {
                "undeleted_target": other,
                "vs_M0": contrast(ck, "M0", other, f),
                "vs_MA": contrast(ck, "MA", other, f),
                "note": "reported separately; NOT folded into a common retain average",
            }
        con["families"][fam] = blk

    con["raw_rates_pct"] = {
        ck: {f"{cat}|{fam}": {t: round(v * 100, 2) for t, v in d.items()}
             for (cat, fam), d in cks.items()}
        for ck, cks in rates.items()}
    con["counts"] = {ck: {cat: len(sel(ck, cat)) for cat in CATS} for ck in data}
    (out / "contrasts.json").write_text(json.dumps(con, indent=2))
    print(f"wrote {out/'contrasts.json'}")

    # ---- markdown summary --------------------------------------------------
    md = [f"# Detection rates and contrasts (training seed {args.training_seed}, t={T})", "",
          "Detector-based object-presence proxies. Not UA/IRA/CRA, not a reproduction.", "",
          "## Raw detection rate D (%) at t=0.5, all prompts", "",
          "| checkpoint | " + " | ".join(CATS) + " |",
          "|---|" + "---|" * len(CATS)]
    for ck in ["M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"]:
        if ck not in data:
            continue
        cells = []
        for cat in CATS:
            rows = sel(ck, cat)
            cells.append(f"{rate(rows, T)*100:.1f} ({sum(r['hit'][T] for r in rows)}/{len(rows)})"
                         if rows else "-")
        md.append(f"| {ck} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
