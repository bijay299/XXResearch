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
import hashlib
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


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_checkpoint(dev_root: Path, name: str, role: str = "candidate") -> dict:
    """Read one scored slot, and BIND the decision to the bytes it was read from.

    The digest of the detections file, its row count, and the digest of the
    checkpoint that generated the images are recorded in the decision. Without
    that, a decision record says only "seed17_MAB_L2 was 52.5%" -- it cannot
    distinguish the evaluation it actually used from a later regeneration of the
    same slot name, which is exactly how a reference target could move
    underneath a selection that had already been made.
    """
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
    binding = {"role": role, "slot": name, "dev_root": str(dev_root),
               "detections_path": str(det),
               "detections_sha256": sha256_file(det),
               "detection_rows": len(rows)}
    rep = dev_root / name / "image_report.json"
    if rep.is_file():
        ir = json.loads(rep.read_text())
        binding.update({
            "image_report_sha256": sha256_file(rep),
            "generating_checkpoint_sha256": (ir.get("checkpoint_load")
                                             or {}).get("sha256"),
            "manifest_sha256": ir.get("manifest_sha256"),
            "generating_checkpoint_path": (ir.get("checkpoint_load")
                                           or {}).get("unet_ckpt"),
            "generation_settings_source": ir.get("generation_settings_source"),
        })
    else:
        binding["image_report_sha256"] = None
        binding["unbound_warning"] = ("no image_report.json beside the "
                                      "detections, so the generating checkpoint "
                                      "and manifest cannot be bound")
    return {"checkpoint": name, "dog_residue_pct": round(pct, 4),
            "dog_hits": hits, "n_dog_pairs": n, "_binding": binding}


def select_for_seed(dev_root: Path, seed: int, dump_steps: list[int],
                    ref_root: Path | None = None) -> dict:
    """`ref_root`, when given, is where the FIXED reference evaluations live.

    The two reference arms -- the parent MA and the L2 endpoint -- define the
    target this selection matches TO. Regenerating them beside each new
    candidate scan lets the target move: a shifted L2 evaluation changes the
    number every dump is compared against, and nothing downstream would notice.
    So they are read from one fixed, separately verified location (by default
    the same root, which is the original run's behaviour).
    """
    refs = ref_root or dev_root
    ma = load_checkpoint(refs, f"seed{seed}_MA", role="fixed_reference_parent")
    l2 = load_checkpoint(refs, f"seed{seed}_MAB_L2", role="fixed_reference_L2")

    def suppression(ck: dict) -> float:
        # Suppression is measured against THAT SEED'S OWN MA, in pp.
        return ma["dog_residue_pct"] - ck["dog_residue_pct"]

    l2_sup = suppression(l2)
    # Collected as inputs are read, so the decision record can be re-bound to
    # the exact bytes it was made from.
    bindings = [ma["_binding"], l2["_binding"]]

    # ---- the L2 REFERENCE must itself satisfy the reference gates.
    #
    # The gates define the regime this comparison is allowed to make a statement
    # about: an explicitly PARTIAL-suppression comparison with a meaningful
    # deletion level to match TO. Applying them only to the dumps was wrong. If
    # the L2 endpoint has not itself deleted enough, matching a dump to it
    # compares two arms that both barely deleted, and any retention similarity
    # is an artifact of both doing nothing -- which is exactly the reasoning the
    # protocol uses to exclude the sandwich branch at +7.5 pp.
    #
    # Worked example of what this rejects: MA 66/80 hits (82.5%), L2 44/80
    # (55.0%) gives 27.5 pp -- below the gate -- while a dump at 42/80 (52.5%)
    # reaches 30.0 pp and sits 2.5 pp away. The old code returned MATCHED on an
    # ineligible reference.
    l2_gate_sup = l2_sup >= GATE_MIN_SUPPRESSION_PP
    l2_gate_res = l2["dog_residue_pct"] <= GATE_MAX_RESIDUE_PCT
    l2_eligible = bool(l2_gate_sup and l2_gate_res)
    l2_eligibility = {
        "l2_dog_suppression_pp": round(l2_sup, 4),
        "l2_dog_residue_pct": l2["dog_residue_pct"],
        "gate_suppression_ge_30pp": l2_gate_sup,
        "gate_residue_le_60pct": l2_gate_res,
        "eligible_as_reference": l2_eligible,
        "at_suppression_boundary": abs(l2_sup - GATE_MIN_SUPPRESSION_PP) < 1e-9,
        "at_residue_boundary": abs(l2["dog_residue_pct"] - GATE_MAX_RESIDUE_PCT) < 1e-9,
        "rule": (f"the reference arm must itself reach >= "
                 f"{GATE_MIN_SUPPRESSION_PP} pp suppression and <= "
                 f"{GATE_MAX_RESIDUE_PCT}% residue, measured against that "
                 f"seed's own MA, before any dump may be matched to it"),
    }

    rows, errors = [], []
    for st in dump_steps:
        name = f"seed{seed}_U_step{st}"
        try:
            ck = load_checkpoint(dev_root, name, role="candidate")
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
        bindings.append(ck["_binding"])

    qualifying = [r for r in rows if r["qualifies"]]
    within = [r for r in qualifying if r["mismatch_vs_L2_pp"] <= MATCH_TOLERANCE_PP + 1e-9]

    out = {
        "training_seed": seed,
        "reference_dev_root": str(refs),
        "candidate_dev_root": str(dev_root),
        "references_are_fixed_and_separate": str(refs) != str(dev_root),
        "MA": {k: v for k, v in ma.items() if k != "_binding"},
        "L2_endpoint": {k: v for k, v in l2.items() if k != "_binding"},
        "L2_dog_suppression_pp": round(l2_sup, 4),
        "L2_reference_eligibility": l2_eligibility,
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
        "development_inputs": bindings,
    }

    # Checked BEFORE any matching: an ineligible reference cannot be matched to
    # at all, so there is nothing to select and no comparison to run.
    if not l2_eligible:
        out["selected"] = None
        out["decision"] = "INELIGIBLE_L2_REFERENCE"
        why = []
        if not l2_gate_sup:
            why.append(f"its suppression is {l2_sup:.1f} pp, below the "
                       f"{GATE_MIN_SUPPRESSION_PP} pp reference gate")
        if not l2_gate_res:
            why.append(f"its dog residue is {l2['dog_residue_pct']:.1f}%, above "
                       f"the {GATE_MAX_RESIDUE_PCT}% reference gate")
        out["reason"] = (
            f"the L2 endpoint is not an eligible reference: {'; and '.join(why)}. "
            f"Matching a dump to it would compare two arms that both barely "
            f"deleted, so any retention similarity would be an artifact of both "
            f"doing nothing. No dump is selected and the planned paired test "
            f"stops. This is the same reasoning that excludes the sandwich "
            f"branch at +7.5 pp.")
        return out

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
        # When every dump already overshoots the L2 level, the lower bracket is
        # the PARENT itself: step 0, 0 pp suppression by definition. Reporting
        # `null` there would hide where the match point actually falls, and the
        # protocol requires the bracketing steps to be reported.
        lower = max(below, key=lambda r: r["dog_suppression_pp_vs_own_MA"],
                    default=None)
        if lower is None:
            lower = {"step": 0, "checkpoint": f"seed{seed}_MA",
                     "dog_residue_pct": ma["dog_residue_pct"],
                     "dog_hits": ma["dog_hits"], "n_dog_pairs": ma["n_dog_pairs"],
                     "dog_suppression_pp_vs_own_MA": 0.0,
                     "is_the_parent_not_a_dump": True,
                     "note": ("every dump overshoots the L2 level, so the match "
                              "point lies between the parent (step 0) and the "
                              "first dump -- inside the frozen 100-step grid, "
                              "which cannot resolve it"),
                     "mismatch_vs_L2_pp": round(abs(l2_sup), 4)}
        brk = {
            "nearest_below": lower,
            "nearest_above": min(above, key=lambda r: r["dog_suppression_pp_vs_own_MA"],
                                 default=None),
            "gap_pp": (round(min(above, key=lambda r: r["dog_suppression_pp_vs_own_MA"])
                             ["dog_suppression_pp_vs_own_MA"]
                             - lower["dog_suppression_pp_vs_own_MA"], 4)
                       if above else None),
            "match_point_falls_between": (
                f"step {lower['step']} ({lower['dog_suppression_pp_vs_own_MA']:.1f} pp) "
                f"and step {min(above, key=lambda r: r['dog_suppression_pp_vs_own_MA'])['step']} "
                f"({min(above, key=lambda r: r['dog_suppression_pp_vs_own_MA'])['dog_suppression_pp_vs_own_MA']:.1f} pp)"
                if above else "above every dump scanned"),
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


def cmd_verify(record: Path) -> int:
    """Re-bind an existing decision record to the bytes now on disk.

    A decision is stale the moment any input it read has changed. This is what
    the runner checks before it will act on a selection -- in particular before
    the frozen test set is touched.
    """
    d = json.loads(record.read_text())
    bad, checked = [], 0
    for seed, rec in (d.get("per_seed") or {}).items():
        for b in rec.get("development_inputs") or []:
            checked += 1
            det = Path(b["detections_path"])
            if not det.is_file():
                bad.append(f"seed {seed} {b['slot']}: {det} is gone")
                continue
            now = sha256_file(det)
            if now != b["detections_sha256"]:
                bad.append(f"seed {seed} {b['slot']} ({b.get('role')}): "
                           f"detections changed since the decision "
                           f"({b['detections_sha256'][:12]}… -> {now[:12]}…)")
            rep = det.parent / "image_report.json"
            want = b.get("image_report_sha256")
            if want:
                if not rep.is_file():
                    bad.append(f"seed {seed} {b['slot']}: image_report.json is gone")
                elif sha256_file(rep) != want:
                    bad.append(f"seed {seed} {b['slot']} ({b.get('role')}): "
                               f"image_report changed since the decision")
    if not checked:
        print(f"UNVERIFIABLE: {record} records no development_inputs, so it "
              f"cannot be bound to anything. It predates binding and must be "
              f"re-decided before use.", file=sys.stderr)
        return 2
    print(f"decision record : {record}")
    print(f"decided_utc     : {d.get('decided_utc')}")
    print(f"inputs bound    : {checked}")
    if bad:
        print("SELECTION IS STALE:")
        for m in bad:
            print(f"  - {m}")
        return 1
    print("SELECTION BINDINGS OK: every development input is byte-identical to "
          "the one this decision was made from")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev_root", default=None)
    ap.add_argument("--seeds", default="17 29")
    ap.add_argument("--dump_steps", default="100,200,300,400,500,600,700,800,900,1000")
    ap.add_argument("--reference_dev_root", default=None,
                    help="where the FIXED reference evaluations (MA and MAB_L2) "
                         "live. Defaults to --dev_root. Point it at a verified, "
                         "read-only evaluation so the match target cannot move "
                         "when candidates are rescored.")
    ap.add_argument("--refuse_if_test_evaluated", default=None,
                    help="a frozen-test output directory. If it holds any files, "
                         "selection REFUSES to run: once the test set has been "
                         "seen, re-deciding which checkpoint to compare is "
                         "test-based selection, whatever the intention.")
    ap.add_argument("--verify", default=None,
                    help="verify an existing decision record's development "
                         "bindings against the bytes now on disk, and exit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.verify:
        return cmd_verify(Path(args.verify))
    if not (args.dev_root and args.out):
        ap.error("--dev_root and --out are required unless --verify is given")

    if args.refuse_if_test_evaluated:
        td = Path(args.refuse_if_test_evaluated)
        seen = [p for p in td.rglob("*") if p.is_file()] if td.is_dir() else []
        if seen:
            print(f"REFUSED: the frozen test output {td} already holds "
                  f"{len(seen)} file(s). Selection must not run after the test "
                  f"set has been evaluated -- re-deciding the compared "
                  f"checkpoint with test evidence in hand is test-based "
                  f"selection. The existing decision record stands.",
                  file=sys.stderr)
            return 4

    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    steps = [int(s) for s in args.dump_steps.replace(" ", "").split(",")]
    dev_root = Path(args.dev_root)
    ref_root = Path(args.reference_dev_root) if args.reference_dev_root else None

    per_seed, fatal = {}, []
    for s in seeds:
        try:
            per_seed[s] = select_for_seed(dev_root, s, steps, ref_root)
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
        "candidate_dev_root": str(dev_root),
        "reference_dev_root": str(ref_root or dev_root),
        "references_are_fixed_and_separate": bool(ref_root)
                                             and str(ref_root) != str(dev_root),
        "binding_note": (
            "every development input this decision read is recorded under "
            "per_seed[*].development_inputs with its sha256, so the decision can "
            "be re-bound to the exact bytes it used. Verify with "
            "`select_matched_dump.py --verify <this file> --dev_root x --out x`."),
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
    # NEVER overwrite a previous decision record. An earlier selection is
    # evidence of what was decided and when; a re-run must stand beside it, not
    # replace it. (The first version of this script did overwrite, and the
    # original record of the 2026-10-09T17:50 run survived only as printed
    # output in run_all.log.)
    if out.exists():
        prev = json.loads(out.read_text())
        stamp = (prev.get("decided_utc") or "unknown").replace(":", "").replace("-", "")[:15]
        keep = out.with_name(f"{out.stem}.{stamp}.superseded.json")
        n = 0
        while keep.exists():
            n += 1
            keep = out.with_name(f"{out.stem}.{stamp}.superseded.{n}.json")
        keep.write_text(json.dumps(prev, indent=2) + "\n")
        payload["supersedes"] = {
            "preserved_at": str(keep),
            "preserved_decided_utc": prev.get("decided_utc"),
            "preserved_sha256": hashlib.sha256(
                (json.dumps(prev, indent=2) + "\n").encode()).hexdigest(),
            "preserved_both_seeds_matched": prev.get("both_seeds_matched"),
        }
        print(f"[preserved] previous decision record -> {keep}")
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
            el = d.get("L2_reference_eligibility") or {}
            print(f"  L2 reference gates    : suppression>=30pp="
                  f"{el.get('gate_suppression_ge_30pp')}  residue<=60%="
                  f"{el.get('gate_residue_le_60pct')}  -> "
                  f"{'ELIGIBLE' if el.get('eligible_as_reference') else 'INELIGIBLE'}"
                  + ("  (exactly at the suppression gate)"
                     if el.get("at_suppression_boundary") else ""))
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
