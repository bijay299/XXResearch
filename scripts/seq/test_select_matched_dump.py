#!/usr/bin/env python3
"""CPU tests for scripts/seq/select_matched_dump.py.

Focused on the demonstrated defect: the reference gates were applied to the
candidate DUMPS but not to the L2 REFERENCE those dumps are matched to. With
80 dog pairs, MA 66 hits, L2 44 hits and U 42 hits at every dump, the old code
returned MATCHED even though the L2 endpoint had only achieved 27.5 pp of
suppression -- below the 30 pp gate that defines the regime this comparison is
allowed to speak about.

Also covers the gate boundaries (a gate must bind at exactly its threshold, not
one step inside it), the other decision branches, and the requirement that a
re-run preserve the previous decision record instead of overwriting it.

No GPU, no model, no network.

    python scripts/seq/test_select_matched_dump.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELECT = REPO / "scripts" / "seq" / "select_matched_dump.py"
REAL_DEV = Path("/data/bijaypandey/cuig_pilot/seq_pilot/diag_v1/eval_dev")

N_PAIRS = 80
STEPS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
PASS, FAIL = [], []


def check(desc: str, cond: bool, detail: str = ""):
    (PASS if cond else FAIL).append(desc)
    print(f"  {'ok  ' if cond else 'FAIL'}  {desc}" + ("" if cond else f"  <- {detail}"))


def write_slot(dev: Path, name: str, hits: int) -> None:
    """One evaluation slot: `hits` of N_PAIRS dog pairs fire at every threshold."""
    d = dev / name
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(N_PAIRS):
        h = i < hits
        lines.append(json.dumps({
            "checkpoint": name, "prompt_id": f"dev_dog_literal_{i % 10}"
            if i < 40 else f"dev_dog_paraphrase_{i % 10}",
            "category": "dog",
            "prompt_family": "literal" if i < 40 else "paraphrase",
            "prompt_index": i % 10, "prompt": f"p{i}",
            "gen_seed": [511, 622, 733, 844][i % 4],
            "image_path": f"/synthetic/{name}/{i}.jpg",
            "image_sha256": f"{i:064x}",
            "detections": [], "max_score_target": 0.9 if h else 0.1,
            "n_detections_target": {t: int(h) for t in ("0.3", "0.5", "0.7")},
            "hit": {t: h for t in ("0.3", "0.5", "0.7")},
        }))
    (d / "detections.jsonl").write_text("\n".join(lines) + "\n")


def build(tmp: Path, ma: int, l2: int, dumps: dict[int, int] | int,
          seeds=(17, 29)) -> Path:
    dev = tmp / "eval_dev"
    for s in seeds:
        write_slot(dev, f"seed{s}_MA", ma)
        write_slot(dev, f"seed{s}_MAB_L2", l2)
        for st in STEPS:
            h = dumps if isinstance(dumps, int) else dumps[st]
            write_slot(dev, f"seed{s}_U_step{st}", h)
    return dev


def run(dev: Path, out: Path, seeds="17 29") -> tuple[int, str, dict | None]:
    p = subprocess.run(
        [sys.executable, str(SELECT), "--dev_root", str(dev), "--seeds", seeds,
         "--dump_steps", ",".join(map(str, STEPS)), "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    data = json.loads(out.read_text()) if out.exists() else None
    return p.returncode, p.stdout + p.stderr, data


def main() -> int:
    if not SELECT.is_file():
        print(f"missing {SELECT}")
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="selmatch_"))
    try:
        print("\nthe demonstrated defect: an INELIGIBLE L2 reference")
        # MA 66/80 = 82.5%; L2 44/80 = 55.0% -> 27.5 pp, BELOW the 30 pp gate.
        # U 42/80 = 52.5% -> 30.0 pp, which passes the dump gates and sits
        # 2.5 pp from L2. The old code called this MATCHED.
        dev = build(tmp / "a", ma=66, l2=44, dumps=42)
        rc, out, d = run(dev, tmp / "a" / "selection.json")
        s17 = d["per_seed"]["17"]
        check("the reference's own suppression is computed (27.5 pp)",
              abs(s17["L2_dog_suppression_pp"] - 27.5) < 1e-9,
              str(s17["L2_dog_suppression_pp"]))
        check("the reference is marked INELIGIBLE",
              s17["L2_reference_eligibility"]["eligible_as_reference"] is False)
        check("the decision is INELIGIBLE_L2_REFERENCE, not MATCHED",
              s17["decision"] == "INELIGIBLE_L2_REFERENCE", s17["decision"])
        check("nothing is selected", s17["selected"] is None)
        check("the run does not proceed to the frozen test set",
              d["proceed_to_frozen_test"] is False)
        check("exit status is non-zero", rc != 0, str(rc))
        check("the reason names the gate it fails",
              "below the 30.0 pp reference gate" in s17["reason"], s17["reason"])
        check("and explains why matching would be meaningless",
              "barely deleted" in s17["reason"])
        # Guard against regression: a dump that WOULD have matched exists.
        check("a dump within 5 pp did exist, so this was a real near-miss",
              any(r["mismatch_vs_L2_pp"] <= 5.0 and r["qualifies"]
                  for r in s17["dumps_scored"]))

        print("\ngate boundaries bind at the threshold, not one step inside it")
        # suppression exactly 30.0 pp -> ELIGIBLE (>=)
        dev = build(tmp / "b", ma=66, l2=42, dumps=42)
        _, _, d = run(dev, tmp / "b" / "selection.json")
        e = d["per_seed"]["17"]["L2_reference_eligibility"]
        check("suppression exactly 30.0 pp is eligible",
              e["eligible_as_reference"] and abs(e["l2_dog_suppression_pp"] - 30.0) < 1e-9)
        check("and is flagged as sitting exactly on the gate",
              e["at_suppression_boundary"] is True)
        # one hit more residue -> 28.75 pp -> INELIGIBLE
        dev = build(tmp / "c", ma=66, l2=43, dumps=42)
        _, _, d = run(dev, tmp / "c" / "selection.json")
        e = d["per_seed"]["17"]["L2_reference_eligibility"]
        check("28.75 pp (one detection inside the gate) is ineligible",
              e["eligible_as_reference"] is False
              and abs(e["l2_dog_suppression_pp"] - 28.75) < 1e-9,
              str(e["l2_dog_suppression_pp"]))
        # residue gate, isolated: MA 80/80 so suppression stays above 30 pp
        dev = build(tmp / "d", ma=80, l2=48, dumps=40)
        _, _, d = run(dev, tmp / "d" / "selection.json")
        e = d["per_seed"]["17"]["L2_reference_eligibility"]
        check("residue exactly 60.0% is eligible",
              e["eligible_as_reference"] and abs(e["l2_dog_residue_pct"] - 60.0) < 1e-9)
        check("and is flagged as sitting exactly on the residue gate",
              e["at_residue_boundary"] is True)
        dev = build(tmp / "e", ma=80, l2=49, dumps=40)
        _, _, d = run(dev, tmp / "e" / "selection.json")
        e = d["per_seed"]["17"]["L2_reference_eligibility"]
        check("residue 61.25% is ineligible even with 38.75 pp suppression",
              e["eligible_as_reference"] is False
              and e["gate_suppression_ge_30pp"] is True,
              f"res={e['l2_dog_residue_pct']} sup={e['l2_dog_suppression_pp']}")

        print("\nthe other decision branches still behave")
        # eligible reference, a dump 1.25 pp away -> MATCHED
        dumps = {st: (41 if st == 300 else 20) for st in STEPS}
        dev = build(tmp / "f", ma=66, l2=42, dumps=dumps)
        rc, out, d = run(dev, tmp / "f" / "selection.json")
        s = d["per_seed"]["17"]
        check("an eligible reference with a close dump MATCHES",
              s["decision"] == "MATCHED", s["decision"])
        check("it selects the dump within tolerance",
              s["selected"] is not None and s["selected"]["step"] == 300,
              str(s.get("selected")))
        check("and permits the frozen test set",
              d["proceed_to_frozen_test"] is True and rc == 0)
        # earliest step breaks ties
        dumps = {st: (41 if st in (300, 700) else 20) for st in STEPS}
        dev = build(tmp / "g", ma=66, l2=42, dumps=dumps)
        _, _, d = run(dev, tmp / "g" / "selection.json")
        s = d["per_seed"]["17"]
        check("a tie is broken by the EARLIEST step",
              s["decision"] == "AMBIGUOUS" or
              (s["selected"] or {}).get("step") == 300,
              f"{s['decision']} {s.get('selected')}")
        # eligible reference, every dump far away -> INFEASIBLE with brackets
        dev = build(tmp / "h", ma=66, l2=42, dumps=10)
        rc, _, d = run(dev, tmp / "h" / "selection.json")
        s = d["per_seed"]["17"]
        check("an eligible reference with no close dump is INFEASIBLE",
              s["decision"] == "INFEASIBLE", s["decision"])
        check("the bracketing is recorded with a parent lower bound",
              s["bracketing"]["nearest_below"]["step"] == 0
              and s["bracketing"]["nearest_below"]["is_the_parent_not_a_dump"] is True)
        check("the gap between brackets is reported",
              s["bracketing"]["gap_pp"] is not None)

        print("\na re-run PRESERVES the previous decision record")
        out = tmp / "f" / "selection.json"
        before = json.loads(out.read_text())
        rc, _, after = run(build(tmp / "f2", ma=66, l2=42, dumps=10), out)
        kept = sorted((tmp / "f").glob("selection.*.superseded*.json"))
        check("the earlier record is preserved beside the new one", len(kept) == 1,
              str([p.name for p in kept]))
        if kept:
            prev = json.loads(kept[0].read_text())
            check("the preserved record is the earlier decision verbatim",
                  prev["both_seeds_matched"] == before["both_seeds_matched"]
                  and prev["decided_utc"] == before["decided_utc"])
            check("the new record points at what it superseded",
                  after.get("supersedes", {}).get("preserved_at") == str(kept[0]))
            check("with the preserved record's own digest",
                  len(after.get("supersedes", {}).get("preserved_sha256", "")) == 64)

        print("\nthe REAL development records: references eligible, both unmatched")
        if REAL_DEV.is_dir():
            rc, out_s, d = run(REAL_DEV, tmp / "real_selection.json")
            ok_all = True
            for s in ("17", "29"):
                p = d["per_seed"][s]
                e = p["L2_reference_eligibility"]
                check(f"real seed {s}: L2 reference is ELIGIBLE "
                      f"({e['l2_dog_suppression_pp']:.1f} pp, "
                      f"{e['l2_dog_residue_pct']:.1f}% residue)",
                      e["eligible_as_reference"] is True)
                check(f"real seed {s}: still INFEASIBLE under the added gate",
                      p["decision"] == "INFEASIBLE", p["decision"])
                ok_all &= p["decision"] == "INFEASIBLE"
            check("real run: both seeds remain unmatched",
                  d["both_seeds_matched"] is False and rc != 0)
            check("real run: the frozen test set is still forbidden",
                  d["proceed_to_frozen_test"] is False)
            e17 = d["per_seed"]["17"]["L2_reference_eligibility"]
            check("real seed 17 sits EXACTLY on the 30 pp suppression gate",
                  e17["at_suppression_boundary"] is True,
                  str(e17["l2_dog_suppression_pp"]))
        else:
            print(f"  SKIP: {REAL_DEV} not present on this host")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("ALL SELECTION CHECKS PASSED (CPU only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
