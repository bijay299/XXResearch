"""Gate evaluation and paired-transition analysis for the revision-02 diagnostic.

Applies the protocol's frozen gates and the authoritative C/U decision table.
Raw classifier transitions are kept strictly separate from verified witnesses
(verification is a separate, image-inspection step recorded elsewhere).
"""
from __future__ import annotations

import json
import os

import pandas as pd

EXEC = "/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/exec_rev02"
OLD = "/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/eval"
A_CLASS = "chain saw"
B_CLASS = "parachute"
RETAIN = ["cassette player", "church", "english springer", "french horn",
          "garbage truck", "gas pump", "golf ball", "tench"]  # 8 classes, excl A and B


def load(p):
    df = pd.read_csv(p)
    assert len(df) == 1000, p
    return df


def per_class(df, col="correct_semantic"):
    return df.groupby("class")[col].sum().astype(int).to_dict()


def key(df):
    """case-level key: (case_number, filename) is unique; use filename."""
    return df.set_index("filename")


def transitions(parent_df, child_df, cls):
    """Raw classifier 2x2 for one class, matched case-for-case by filename."""
    p = key(parent_df[parent_df["class"] == cls])
    c = key(child_df[child_df["class"] == cls])
    common = sorted(set(p.index) & set(c.index))
    assert len(common) == 100, f"{cls}: matched {len(common)} of 100"
    rows = []
    for fn in common:
        rows.append({
            "filename": fn,
            "case_number": int(p.loc[fn, "case_number"]),
            "parent_semantic": bool(p.loc[fn, "correct_semantic"]),
            "child_semantic": bool(c.loc[fn, "correct_semantic"]),
            "parent_strict": bool(p.loc[fn, "correct_strict"]),
            "child_strict": bool(c.loc[fn, "correct_strict"]),
            "parent_top1": p.loc[fn, "category_top1"],
            "child_top1": c.loc[fn, "category_top1"],
            "child_top1_score": float(c.loc[fn, "scores_top1"]),
        })
    t = pd.DataFrame(rows)
    tab = {
        "present_to_present": int(((t.parent_semantic) & (t.child_semantic)).sum()),
        "present_to_absent": int(((t.parent_semantic) & (~t.child_semantic)).sum()),
        "absent_to_present": int(((~t.parent_semantic) & (t.child_semantic)).sum()),
        "absent_to_absent": int(((~t.parent_semantic) & (~t.child_semantic)).sum()),
    }
    candidates = t[(~t.parent_semantic) & (t.child_semantic)]
    return tab, t, candidates


def main():
    out = {"gates": {}, "transitions": {}, "retention": {}, "candidates": {}}

    m0 = load(f"{OLD}/M0_per_image.csv")
    parents = {"DP": load(f"{OLD}/DP_B_per_image.csv"),
               "UCE": load(f"{OLD}/UCE_B_per_image.csv")}
    endpoints = {}
    for name in ["DP_B_direct", "UCE_B_direct", "DP_SEQ", "UCE_SEQ",
                 "DP_BATCH", "UCE_BATCH"]:
        p = f"{EXEC}/eval/{name}_per_image.csv"
        if os.path.exists(p):
            endpoints[name] = load(p)

    m0_sem = per_class(m0)
    out["M0_semantic_per_class"] = m0_sem
    out["M0_strict_per_class"] = per_class(m0, "correct_strict")
    for m in ("DP", "UCE"):
        out[f"{m}_A_parent_semantic_per_class"] = per_class(parents[m])
        out[f"{m}_A_parent_strict_per_class"] = per_class(parents[m], "correct_strict")
    for n, df in endpoints.items():
        out[f"{n}_semantic_per_class"] = per_class(df)
        out[f"{n}_strict_per_class"] = per_class(df, "correct_strict")

    for m in ("DP", "UCE"):
        g = {}
        # --- direct B suppression vs M0 (baseline 100) ---
        d = f"{m}_B_direct"
        if d in endpoints:
            base = m0_sem[B_CLASS]
            got = per_class(endpoints[d])[B_CLASS]
            red = (base - got) / base if base else None
            g["direct_B_suppression"] = {
                "baseline_endpoint": "M0", "baseline_count": base, "endpoint_count": got,
                "relative_reduction": red, "threshold": 0.75, "pass": bool(red >= 0.75)}
        # --- sequential B suppression vs that method's A-parent ---
        s = f"{m}_SEQ"
        if s in endpoints:
            base = per_class(parents[m])[B_CLASS]
            got = per_class(endpoints[s])[B_CLASS]
            red = (base - got) / base if base else None
            g["sequential_B_suppression"] = {
                "baseline_endpoint": f"{m}_A_parent", "baseline_count": base,
                "endpoint_count": got, "relative_reduction": red,
                "threshold": 0.75, "pass": bool(red >= 0.75)}
            # --- deletion-strength comparability (direct vs SEQ B counts) ---
            if d in endpoints:
                dc = per_class(endpoints[d])[B_CLASS]
                sc = per_class(endpoints[s])[B_CLASS]
                g["B_count_comparability"] = {
                    "direct_count": dc, "seq_count": sc, "abs_diff": abs(dc - sc),
                    "band": 10, "within_band": bool(abs(dc - sc) <= 10),
                    "note": "descriptive band; an out-of-band result is a tradeoff/confound, "
                            "not proof of its cause"}
            # --- retention: cumulative vs M0, incremental vs A-parent ---
            sp = per_class(endpoints[s])
            pp = per_class(parents[m])
            cum = {c: sp[c] - m0_sem[c] for c in RETAIN}
            inc = {c: sp[c] - pp[c] for c in RETAIN}
            cum_agg = sum(sp[c] for c in RETAIN) / len(RETAIN) - sum(m0_sem[c] for c in RETAIN) / len(RETAIN)
            inc_agg = sum(sp[c] for c in RETAIN) / len(RETAIN) - sum(pp[c] for c in RETAIN) / len(RETAIN)
            g["retention_vs_M0_cumulative"] = {
                "per_class_delta": cum, "aggregate_delta_points": cum_agg,
                "worst_class_drop": -min(cum.values()), "bounds": {"aggregate": -8, "class": 15},
                "pass": bool(cum_agg >= -8 and -min(cum.values()) <= 15),
                "status": "descriptive"}
            g["retention_vs_parent_incremental"] = {
                "per_class_delta": inc, "aggregate_delta_points": inc_agg,
                "worst_class_drop": -min(inc.values()), "bounds": {"aggregate": -5, "class": 12},
                "pass": bool(inc_agg >= -5 and -min(inc.values()) <= 12),
                "status": "descriptive"}
            # --- historical A transitions (parent -> SEQ) + candidate set ---
            tab, t, cand = transitions(parents[m], endpoints[s], A_CLASS)
            out["transitions"][f"{m}_SEQ_chain_saw_parent_to_seq"] = tab
            t.to_csv(f"{EXEC}/eval/{m}_SEQ_chain_saw_transitions.csv", index=False)
            out["candidates"][f"{m}_SEQ"] = {
                "n_candidates_raw": int(len(cand)),
                "cases": cand[["case_number", "filename", "child_top1",
                               "child_top1_score"]].to_dict("records"),
                "note": "raw classifier candidates (absent at A-parent -> present at SEQ); "
                        "C/U assigned only after paired visual verification"}
            # M0 context for the A class
            m0A = key(m0[m0["class"] == A_CLASS])
            out["transitions"][f"{m}_M0_chain_saw_semantic"] = int(m0A["correct_semantic"].sum())
        out["gates"][m] = g

    with open(f"{EXEC}/eval/ANALYSIS.json", "w") as f:
        json.dump(out, f, indent=2)

    for m in ("DP", "UCE"):
        print(f"\n=== {m} ===")
        for k, v in out["gates"].get(m, {}).items():
            if "pass" in v:
                extra = ""
                if "relative_reduction" in v:
                    extra = f" ({v['baseline_count']} -> {v['endpoint_count']}, {v['relative_reduction']:.1%})"
                elif "abs_diff" in v:
                    extra = f" (|{v['direct_count']}-{v['seq_count']}| = {v['abs_diff']})"
                    print(f"  {k:36s} within_band={v['within_band']}{extra}")
                    continue
                elif "aggregate_delta_points" in v:
                    extra = f" (agg {v['aggregate_delta_points']:+.2f} pts, worst class -{v['worst_class_drop']})"
                print(f"  {k:36s} pass={v['pass']}{extra}")
        tk = f"{m}_SEQ_chain_saw_parent_to_seq"
        if tk in out["transitions"]:
            print(f"  chain-saw raw transitions (parent->SEQ): {out['transitions'][tk]}")
            print(f"  raw relapse candidates: {out['candidates'][f'{m}_SEQ']['n_candidates_raw']}")
    print(f"\n-> {EXEC}/eval/ANALYSIS.json")


if __name__ == "__main__":
    main()
