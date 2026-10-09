#!/usr/bin/env python3
"""CPU tests for scripts/seq/diag_analysis.py.

Builds a synthetic diagnostic output tree with HAND-CHOSEN detection rates, so
the bootstrap and the decision rule can be checked against values computed by
hand rather than against whatever the real run happens to produce. Finding a
bootstrap bug on real confirmatory results would be the worst possible time.

No GPU, no model, no network.

    python scripts/seq/test_diag_analysis.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "scripts" / "seq" / "diag_analysis.py"
GROUPING = REPO / "results" / "audit_v1" / "draft_manifests" / "bootstrap_grouping.json"
TEST_MANIFEST = REPO / "results" / "audit_v1" / "draft_manifests" / "test_manifest_DRAFT.json"

PASS, FAIL = [], []


def check(desc: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(desc)
    print(f"  {'ok  ' if cond else 'FAIL'}  {desc}" + ("" if cond else f"  <- {detail}"))


def write_arm(d: Path, name: str, records: list[dict], hit_rule) -> None:
    """One arm's detections.jsonl. hit_rule(record) -> bool at every threshold."""
    d = d / name
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in records:
        h = bool(hit_rule(r))
        lines.append(json.dumps({
            "checkpoint": name, "prompt_id": r["prompt_id"],
            "category": r["category"], "prompt_family": r["prompt_family"],
            "prompt_index": r["prompt_index"], "prompt": r["prompt"],
            "gen_seed": r["gen_seed"],
            "image_path": f"/synthetic/{name}/{r['image_name']}",
            "image_sha256": hashlib.sha256(
                f"{name}/{r['image_name']}".encode()).hexdigest(),
            "detections": [], "max_score_target": 0.9 if h else 0.1,
            "n_detections_target": {t: int(h) for t in ("0.3", "0.5", "0.7")},
            "hit": {t: h for t in ("0.3", "0.5", "0.7")},
        }))
    (d / "detections.jsonl").write_text("\n".join(lines) + "\n")


def build(tmp: Path, rates: dict[str, dict[str, float]]) -> Path:
    """A synthetic diag root. rates[arm][category] is a fraction in [0,1].

    Hits are assigned deterministically by prompt_index so a rate of 0.4 over
    10 clusters means exactly the first 4 prompt_index values hit, in BOTH
    families and ALL four generation seeds -- i.e. whole scene clusters, which
    makes the cluster bootstrap's behaviour predictable.
    """
    man = json.loads(TEST_MANIFEST.read_text())
    records = man["records"]
    root = tmp / "diag"
    ev = root / "eval_test"
    for seed in (17, 29):
        for arm, slot in (("MA", f"seed{seed}_MA"),
                          ("L2", f"seed{seed}_MAB_L2"),
                          ("U", f"seed{seed}_U_step600")):
            def rule(r, arm=arm, seed=seed):
                frac = rates[arm][r["category"]]
                return r["prompt_index"] < round(frac * 10)
            write_arm(ev, slot, records, rule)
    sel = {
        "seeds": [17, 29], "proceed_to_frozen_test": True,
        "per_seed": {str(s): {"selected": {"step": 600, "mismatch_vs_L2_pp": 2.5}}
                     for s in (17, 29)},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "selection.json").write_text(json.dumps(sel, indent=2))
    return root


def run(root: Path, *extra: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(ANALYSIS), "--diag_root", str(root),
                        "--grouping", str(GROUPING), *extra],
                       capture_output=True, text=True, cwd=str(REPO),
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    try:
        import numpy  # noqa: F401
    except Exception as e:
        print(f"numpy required: {e}")
        return 2
    for f in (ANALYSIS, GROUPING, TEST_MANIFEST):
        if not f.is_file():
            print(f"missing {f}")
            return 2

    tmp = Path(tempfile.mkdtemp(prefix="diaganalysis_"))
    try:
        print("\npoint estimates match hand-computed rates")
        # bird: L2 retains 80%, U* retains 50% -> contrast +30.0 pp exactly.
        # dog:  both suppressed to 40% -> matching difference 0.0 pp.
        base = {c: 0.9 for c in ("cat", "dog", "sandwich", "horse", "bird",
                                 "chair", "bicycle")}
        rates = {
            "MA": dict(base),
            "L2": {**base, "bird": 0.8, "dog": 0.4},
            "U":  {**base, "bird": 0.5, "dog": 0.4},
        }
        root = build(tmp, rates)
        rc, out = run(root)
        check("analysis exits 0", rc == 0, out.strip()[-300:])
        res = json.loads((root / "analysis_test.json").read_text())
        e17 = res["per_seed"]["17"]["estimates"]
        b = e17["contrast[bird|0.5]"]
        check("bird contrast point estimate is exactly +30.0 pp",
              abs(b["point"] - 30.0) < 1e-9, f"got {b['point']}")
        check("bird L2 rate is 80.0%",
              abs(e17["D[bird|0.5]|L2"]["point"] - 80.0) < 1e-9)
        check("bird U* rate is 50.0%",
              abs(e17["D[bird|0.5]|U"]["point"] - 50.0) < 1e-9)
        d = e17["contrast[dog|0.5]"]
        check("dog matching difference is exactly 0.0 pp",
              abs(d["point"]) < 1e-9, f"got {d['point']}")
        check("a category with no arm difference has a zero-width interval",
              abs(d["lo95"]) < 1e-9 and abs(d["hi95"]) < 1e-9,
              f"[{d['lo95']}, {d['hi95']}]")

        print("\nintervals and the frozen grouping")
        check("the interval brackets the point estimate",
              b["lo95"] <= b["point"] <= b["hi95"],
              f"{b['lo95']} .. {b['point']} .. {b['hi95']}")
        check("the interval has non-zero width where arms differ",
              b["hi95"] > b["lo95"])
        check("70 clusters over 7 strata were used",
              res["per_seed"]["17"]["n_clusters"] == 70
              and len(res["per_seed"]["17"]["clusters_per_stratum"]) == 7,
              str(res["per_seed"]["17"]["clusters_per_stratum"]))
        check("10 clusters per category",
              set(res["per_seed"]["17"]["clusters_per_stratum"].values()) == {10})
        check("10 000 draws recorded", res["bootstrap"]["n_draws"] == 10000)
        check("the frozen RNG seed is recorded",
              res["bootstrap"]["rng_seed"] == 2026100901)
        check("the grouping identity is recorded",
              res["grouping_sha256"] == json.loads(
                  GROUPING.read_text())["grouping_sha256"])
        check("seeds are explicitly NOT pooled", res["seeds_pooled"] is False)
        check("both seeds are reported separately",
              set(res["per_seed"]) == {"17", "29"})

        print("\nthe analysis is deterministic under the frozen RNG seed")
        rc2, _ = run(root, "--out", str(root / "again.json"))
        again = json.loads((root / "again.json").read_text())
        b2 = again["per_seed"]["17"]["estimates"]["contrast[bird|0.5]"]
        check("a re-run reproduces the interval exactly",
              (b2["lo95"], b2["hi95"]) == (b["lo95"], b["hi95"]),
              f"{(b2['lo95'], b2['hi95'])} vs {(b['lo95'], b['hi95'])}")

        print("\nevery required endpoint is present")
        for cat in ("bird", "horse", "chair", "bicycle", "dog", "cat", "sandwich"):
            check(f"{cat} reported separately at t=0.5",
                  f"contrast[{cat}|0.5]" in e17)
        for thr in ("0.3", "0.5", "0.7"):
            check(f"threshold {thr} reported", f"contrast[bird|{thr}]" in e17)
        for fam in ("literal", "paraphrase"):
            check(f"bird {fam} reported separately",
                  f"contrast[bird|0.5|{fam}]" in e17)
        check("per-arm change from that seed's own MA is reported",
              "vsMA[bird|0.5]|L2" in e17 and "vsMA[bird|0.5]|U" in e17)
        check("the parent-reference coincidence is stated",
              "algebraically identical" in res["parent_referenced_note"])
        # The identity itself: contrast of vsMA equals the direct contrast.
        lhs = e17["vsMA[bird|0.5]|L2"]["point"] - e17["vsMA[bird|0.5]|U"]["point"]
        check("contrast of parent-referenced changes == direct contrast",
              abs(lhs - b["point"]) < 1e-9, f"{lhs} vs {b['point']}")

        print("\nthe decision rule maps intervals to the prespecified readings")
        r = res["primary_reading_per_seed"]["17"]
        # With 10 bird clusters the interval around +30 pp is wide, so the
        # reading depends on where the bounds actually fall. What must hold is
        # that the verdict agrees with the bounds -- not a presumed label.
        expect_material = r["lo95"] > 10.0
        check("the +30 pp reading agrees with its own bounds",
              r["material_benefit"] == expect_material
              and (("MATERIAL" in r["reading"]) == expect_material),
              f"{r['point']:+.1f} [{r['lo95']:+.1f}, {r['hi95']:+.1f}] -> {r['reading']}")
        check("an interval spanning the margin is called inconclusive, not material",
              not (r["lo95"] <= 10.0 <= r["hi95"]) or r["inconclusive"],
              r["reading"])
        # No difference anywhere -> equivalence at the declared margin.
        rates_eq = {"MA": dict(base),
                    "L2": {**base, "bird": 0.6, "dog": 0.4},
                    "U":  {**base, "bird": 0.6, "dog": 0.4}}
        root2 = build(tmp / "eq", rates_eq)
        rc, out = run(root2)
        res2 = json.loads((root2 / "analysis_test.json").read_text())
        rr2 = res2["primary_reading_per_seed"]["17"]
        check("an identical-arms contrast reads as practical equivalence",
              rr2["practical_equivalence"] and "EQUIVALENCE" in rr2["reading"],
              rr2["reading"])
        check("and also records that a 10 pp benefit is ruled out",
              rr2["benefit_of_10pp_ruled_out"], rr2["reading"])
        # A small difference whose interval still clears +10 is ruled out.
        rates_small = {"MA": dict(base),
                       "L2": {**base, "bird": 0.7, "dog": 0.4},
                       "U":  {**base, "bird": 0.6, "dog": 0.4}}
        root3 = build(tmp / "small", rates_small)
        rc, out = run(root3)
        res3 = json.loads((root3 / "analysis_test.json").read_text())
        r3 = res3["primary_reading_per_seed"]["17"]
        check("a +10 pp contrast is not silently called material",
              "MATERIAL" not in r3["reading"] or r3["lo95"] > 10.0,
              f"{r3['point']} [{r3['lo95']}, {r3['hi95']}] {r3['reading']}")

        print("\nguards")
        check("the interpretation rules are carried in the output",
              any("NOT evidence of no effect" in s
                  for s in res["interpretation_rules"]))
        check("the detector is labelled a proxy",
              any("PROXY" in s for s in res["interpretation_rules"]))
        check("the output declares itself provisional",
              "PROVISIONAL" in res["_what_this_is"])
        # A tampered grouping must be refused, not silently used.
        bad = tmp / "bad_grouping.json"
        g = json.loads(GROUPING.read_text())
        g["n_draws"] = 10
        bad.write_text(json.dumps(g))
        p = subprocess.run([sys.executable, str(ANALYSIS), "--diag_root", str(root),
                            "--grouping", str(bad)], capture_output=True, text=True,
                           cwd=str(REPO), env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        check("a tampered grouping is refused",
              p.returncode != 0 and "was altered" in (p.stdout + p.stderr),
              (p.stdout + p.stderr).strip()[-200:])
        # Selection that did not proceed must produce no confirmatory analysis.
        root4 = build(tmp / "stop", rates)
        sel = json.loads((root4 / "selection.json").read_text())
        sel["proceed_to_frozen_test"] = False
        (root4 / "selection.json").write_text(json.dumps(sel))
        rc, out = run(root4)
        check("no confirmatory analysis when selection did not proceed",
              rc != 0 and "no confirmatory analysis" in out, out.strip()[-200:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("ALL DIAGNOSTIC-ANALYSIS CHECKS PASSED (CPU only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
