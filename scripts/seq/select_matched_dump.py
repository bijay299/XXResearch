#!/usr/bin/env python3
"""Select the matched unregularised dump on the DEVELOPMENT set. Selection only.

The prespecified rule, from docs/research/MATCHED_EFFECTIVENESS_PROTOCOL.md §5,
applied exactly as frozen and with no discretion left at runtime:

  * everything is measured on the DEVELOPMENT set (dog only, 20 prompt texts x
    4 generation seeds = 80 pairs per checkpoint), at threshold t=0.5, and
    against **that seed's own MA**;
  * **reference gates**: a dump qualifies only with >= 30 pp dog suppression
    AND <= 60% dog residue. These define an explicitly partial-suppression
    comparison;
  * **match rule**: among qualifying dumps, take the one whose dog suppression
    is closest to that seed's L2 endpoint, **earliest step breaking ties**; the
    mismatch must be <= 5 pp;
  * **infeasibility rule, declared in advance**: if EITHER seed has no
    qualifying dump within 5 pp, the planned paired test STOPS. The bracketing
    steps, their suppression values and the gap are reported. The tolerance is
    not widened, there is no fall back to one seed, and no unmatched comparison
    is run;
  * also stop and report if suppression is **non-monotone in steps** in a way
    that makes "the matched step" ambiguous: two separated qualifying dumps both
    inside the band with materially different retention. Both are reported.

Declaring infeasibility is a RESULT, not a failure.

Nothing here touches the test set and nothing here is a confirmatory claim. The
output is a decision record: which dump each seed contributes, and why.

CPU only.

    python scripts/seq/select_matched_dump.py \
        --dev_root <diag>/eval_dev --seeds 17 29 \
        --dump_steps 100,200,...,1000 --out <diag>/selection.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

THR = "0.5"
TARGET = "dog"
GATE_MIN_SUPPRESSION_PP = 30.0
GATE_MAX_RESIDUE_PCT = 60.0
MATCH_TOLERANCE_PP = 5.0
# "materially different" retention for the ambiguity check, in percentage
# points. Declared here so the non-monotonicity rule is not decided at runtime.
AMBIGUITY_MATERIAL_PP = 10.0


def read_rows(p: Path) -> list[dict]:
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def dog_residue_pct(rows: list[dict]) -> tuple[float, int, int]:
    """Residual dog presence: % of dog pairs the detector still fires on."""
    d = [r for r in rows if r.get("category") == TARGET]
    if not d:
        raise ValueError(f"no {TARGET!r} rows")
    hits = sum(1 for r in d if r["hit"][THR])
    return 100.0 * hits / len(d), hits, len(d)


def load_checkpoint(dev_root: Path, name: str) -> dict:
    det = dev_root / name / "detections.jsonl"
    if not det.is_file():
        raise FileNotFoundError(f"{det} absent: this checkpoint was not scored "
                                f"on the development set")
    rows = read_rows(det)
    pct, hits, n = dog_residue_pct(rows)
    cks = {r.get("checkpoint") for r in rows}
    if cks != {name}:
        raise ValueError(f"{det} carries checkpoint label(s) {sorted(map(str, cks))}, "
                         f"expected only {name!r}")
    return {"checkpoint": name, "dog_residue_pct": round(pct, 4),
            "dog_hits": hits, "n_dog_pairs": n}


def select_for_seed(dev_root: Path, seed: int, dump_steps: list[int]) -> dict:
    ma = load_checkpoint(dev_root, f"seed{seed}_MA")
    l2 = load_checkpoint(dev_root, f"seed{seed}_MAB_L2")

    def suppression(ck: dict) -> float:
        # Suppression is measured against THAT SEED'S OWN MA, in pp.
        return ma["dog_residue_pct"] - ck["dog_residue_pct"]

    l2_sup = suppression(l2)
    rows, errors = [], []
    for st in dump_steps:
        name = f"seed{seed}_U_step{st}"
        try:
            ck = load_checkpoint(dev_root, name)
        except Exception as e:
            errors.append({"step": st, "error": f"{type(e).__name__}: {e}"})
            continue
        sup = suppression(ck)
        gate_sup = sup >= GATE_MIN_SUPPRESSION_PP
        gate_res = ck["dog_residue_pct"] <= GATE_MAX_RESIDUE_PCT
        rows.append({
            "step": st, "checkpoint": name,
            "dog_residue_pct": ck["dog_residue_pct"],
            "dog_hits": ck["dog_hits"], "n_dog_pairs": ck["n_dog_pairs"],
            "dog_suppression_pp_vs_own_MA": round(sup, 4),
            "gate_suppression_ge_30pp": gate_sup,
            "gate_residue_le_60pct": gate_res,
            "qualifies": bool(gate_sup and gate_res),
            "mismatch_vs_L2_pp": round(abs(sup - l2_sup), 4),
        })

    qualifying = [r for r in rows if r["qualifies"]]
    within = [r for r in qualifying if r["mismatch_vs_L2_pp"] <= MATCH_TOLERANCE_PP + 1e-9]

    out = {
        "training_seed": seed,
        "MA": ma, "L2_endpoint": l2,
        "L2_dog_suppression_pp": round(l2_sup, 4),
        "gates": {"min_suppression_pp": GATE_MIN_SUPPRESSION_PP,
                  "max_residue_pct": GATE_MAX_RESIDUE_PCT,
                  "match_tolerance_pp": MATCH_TOLERANCE_PP,
                  "threshold": THR,
                  "reference": "that seed's own MA",
                  "regime": "explicitly PARTIAL suppression; no claim about a "
                            "strongly-suppressed regime follows"},
        "dumps_scored": rows,
        "dumps_unscored": errors,
        "n_dumps_scored": len(rows),
        "n_qualifying_gates": len(qualifying),
        "n_within_tolerance": len(within),
    }

    if errors:
        out["selected"] = None
        out["decision"] = "INCOMPLETE"
        out["reason"] = (f"{len(errors)} dump(s) were not scored on the "
                         f"development set, so the scan is not the full fixed "
                         f"scan the protocol froze: {errors[:3]}")
        return out

    # Monotonicity / ambiguity, checked BEFORE picking, so an ambiguous
    # landscape is reported rather than resolved by the tie-break.
    if len(within) > 1:
        steps = [r["step"] for r in within]
        span = max(steps) - min(steps)
        res = [r["dog_residue_pct"] for r in within]
        spread = max(res) - min(res)
        separated = span >= 2 * max(dump_steps[0], 1)   # not adjacent dumps
        if separated and spread >= AMBIGUITY_MATERIAL_PP:
            out["selected"] = None
            out["decision"] = "AMBIGUOUS"
            out["reason"] = (
                f"{len(within)} dumps lie inside the {MATCH_TOLERANCE_PP} pp "
                f"band at separated steps {steps} with a dog-residue spread of "
                f"{spread:.1f} pp (>= {AMBIGUITY_MATERIAL_PP} pp): 'the matched "
                f"step' is not well defined. Suppression is not monotone in "
                f"steps in a way that matters here. Both are reported and the "
                f"planned paired test stops.")
            out["ambiguous_candidates"] = within
            return out

    if not within:
        # Bracketing dumps: nearest below and above the L2 level, reported so
        # the gap is visible rather than just asserted.
        below = [r for r in rows if r["dog_suppression_pp_vs_own_MA"] < l2_sup]
        above = [r for r in rows if r["dog_suppression_pp_vs_own_MA"] >= l2_sup]
        brk = {
            "nearest_below": max(below, key=lambda r: r["dog_suppression_pp_vs_own_MA"],
                                 default=None),
            "nearest_above": min(above, key=lambda r: r["dog_suppression_pp_vs_own_MA"],
                                 default=None),
        }
        best = min(rows, key=lambda r: r["mismatch_vs_L2_pp"], default=None)
        out["selected"] = None
        out["decision"] = "INFEASIBLE"
        out["bracketing"] = brk
        out["closest_any_gate_state"] = best
        out["reason"] = (
            f"no dump passes both reference gates AND lands within "
            f"{MATCH_TOLERANCE_PP} pp of this seed's L2 suppression "
            f"({l2_sup:.1f} pp). Qualifying on gates: {len(qualifying)}. "
            f"The tolerance is NOT widened and no unmatched comparison is run. "
            f"Declaring infeasibility is a result: 'these two arms cannot be "
            f"compared fairly at this coefficient'.")
        return out

    # Closest mismatch; EARLIEST step breaks ties (conservative against the
    # hypothesis that L2-SP is unnecessary, since it favours the smaller update).
    best = min(within, key=lambda r: (r["mismatch_vs_L2_pp"], r["step"]))
    out["selected"] = best
    out["decision"] = "MATCHED"
    out["reason"] = (
        f"step {best['step']} is the qualifying dump closest to this seed's L2 "
        f"suppression ({l2_sup:.1f} pp), mismatch "
        f"{best['mismatch_vs_L2_pp']:.1f} pp <= {MATCH_TOLERANCE_PP} pp; "
        f"earliest step breaks ties.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev_root", required=True)
    ap.add_argument("--seeds", default="17 29")
    ap.add_argument("--dump_steps", default="100,200,300,400,500,600,700,800,900,1000")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    steps = [int(s) for s in args.dump_steps.replace(" ", "").split(",")]
    dev_root = Path(args.dev_root)

    per_seed, fatal = {}, []
    for s in seeds:
        try:
            per_seed[s] = select_for_seed(dev_root, s, steps)
        except Exception as e:
            per_seed[s] = {"training_seed": s, "decision": "ERROR",
                           "reason": f"{type(e).__name__}: {e}", "selected": None}
            fatal.append(s)

    matched = all(per_seed[s].get("decision") == "MATCHED" for s in seeds)
    payload = {
        "_what_this_is": (
            "The development-set selection decision for the matched-"
            "effectiveness diagnostic. SELECTION ONLY: nothing here is a "
            "confirmatory claim, and the frozen test set was not consulted."),
        "threshold": THR,
        "target": TARGET,
        "rule": ("gates >= 30 pp suppression and <= 60% residue vs that seed's "
                 "own MA; closest suppression to the L2 endpoint; earliest step "
                 "breaks ties; mismatch <= 5 pp; both seeds must match or the "
                 "planned paired test stops"),
        "dump_steps_scanned": steps,
        "seeds": seeds,
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "both_seeds_matched": matched,
        "proceed_to_frozen_test": matched,
        "decided_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if not matched:
        payload["stop_reason"] = {
            str(s): per_seed[s].get("reason") for s in seeds
            if per_seed[s].get("decision") != "MATCHED"}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"=== development-set selection (t={THR}, {TARGET} only) ===")
    for s in seeds:
        d = per_seed[s]
        print(f"seed {s}: {d.get('decision')}")
        if d.get("MA"):
            print(f"  MA dog residue        : {d['MA']['dog_residue_pct']:.1f}% "
                  f"({d['MA']['dog_hits']}/{d['MA']['n_dog_pairs']})")
            print(f"  L2 endpoint residue   : {d['L2_endpoint']['dog_residue_pct']:.1f}%"
                  f"  -> suppression {d['L2_dog_suppression_pp']:.1f} pp")
        for r in d.get("dumps_scored", []):
            mark = "  <- SELECTED" if (d.get("selected") or {}).get("step") == r["step"] else ""
            print(f"    step {r['step']:>4}: residue {r['dog_residue_pct']:5.1f}%  "
                  f"suppression {r['dog_suppression_pp_vs_own_MA']:6.1f} pp  "
                  f"gates={'PASS' if r['qualifies'] else 'fail'}  "
                  f"mismatch {r['mismatch_vs_L2_pp']:5.1f} pp{mark}")
        print(f"  reason: {d.get('reason')}")
    print(f"\nboth seeds matched: {matched} -> "
          f"{'PROCEED to the frozen test set' if matched else 'STOP; do NOT evaluate the test set'}")
    print(f"[selection] {out}")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
