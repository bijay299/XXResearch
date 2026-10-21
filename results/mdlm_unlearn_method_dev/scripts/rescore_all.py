"""rescore_all.py — re-partition every matched run from its PRESERVED raw generations.

The evaluator was revised after the arms were trained (coordinator correction, plus six
slot-extractor defects found by inspecting the contents of the SC-TF IGNORANT cell). No
arm is retrained: `outputs/matched_<label>_rows.json` holds every raw generation and its
token ids, so the partition is recomputed from disk. The answer margins and the cost and
gate records are produced by the training run and are unaffected by the evaluator change,
so they are carried through unchanged.

The pre-revision summary is preserved in each report under `summary_prerevision` and the
difference is printed, so the effect of the evaluator fix on every cell is visible rather
than silently replacing the earlier numbers.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_partition as EP  # noqa: E402

W = "/data/bijaypandey/mdlm_unlearn"


def main():
    facts = {f["fact_id"]: f for f in EP.load_facts()}
    pools = EP.build_pools(EP.load_facts())
    changed = []
    for rep_path in sorted(glob.glob(f"{W}/outputs/matched_*.json")):
        b = os.path.basename(rep_path)
        if any(k in b for k in ("_rows", "_margins", "_factsets", "_rescored")):
            continue
        rows_path = rep_path.replace(".json", "_rows.json")
        if not os.path.exists(rows_path):
            print(f"  skip {b}: no preserved rows")
            continue
        R = json.load(open(rep_path))
        if "summary" not in R:
            print(f"  skip {b}: no summary (failed run)")
            continue
        old = R.get("summary_prerevision") or R["summary"]
        rows_in = json.load(open(rows_path))
        new_rows = []
        for r in rows_in:
            f = facts[r["fact_id"]]
            n = EP.partition_one(r["generation"], f, pools)
            n["probe"] = r.get("probe", "train")
            n["generation_with_specials"] = r.get("generation_with_specials")
            n["generation_token_ids"] = r.get("generation_token_ids")
            new_rows.append(n)
        margins = old.get("answer_margin")
        R["summary_prerevision"] = old
        R["summary"] = EP.summarize(new_rows, margins)
        R["rescored"] = {
            "reason": ("evaluator revised after training: type-correct REPLACE, non-word "
                       "DEGENERATE, ABSTAIN cell, own-answer-lemma routing, and six "
                       "slot-extractor defects"),
            "evaluator_sha256": EP.__dict__.get("__file_sha__") or "see driver_sources",
            "rows_source": os.path.basename(rows_path),
        }
        json.dump(R, open(rep_path, "w"), indent=2, default=float)
        json.dump(new_rows, open(rows_path, "w"), indent=1, default=float)
        o, n = old["forget_partition"]["rates"], R["summary"]["forget_partition"]["rates"]
        d = {k: round((n[k] or 0) - (o.get(k) or 0), 4) for k in EP.LABELS}
        changed.append((R["label"], o, n, d))
        print(f"{R['label']:18s} "
              + " ".join(f"{k[:4]}:{o.get(k)}->{n[k]}" for k in EP.LABELS))
    print(f"\n{len(changed)} reports rescored.")
    print("\nPer-cell change caused by the evaluator revision (forget probes):")
    for lab, o, n, d in changed:
        if any(abs(v) > 1e-9 for v in d.values()):
            print(f"  {lab:18s} " + " ".join(f"{k}{v:+.3f}" for k, v in d.items() if v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
