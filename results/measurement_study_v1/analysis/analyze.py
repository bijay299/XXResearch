#!/usr/bin/env python3
"""Implements the frozen analysis plan. Pure stdlib, no network, deterministic.

RUN ON SYNTHETIC FIXTURES ONLY until genuine responses exist AND the separate
authorisation to analyse them has been given. This file performs no human-outcome
analysis by itself; it is the machinery, frozen in advance so the analysis cannot be
chosen after seeing results.

Inputs
  --responses  reviewer stage-A 'final' files (>=2 required for the primary analysis)
  --machine    automated evaluator predictions, {evaluator: {neutral_id: present|absent|uncertain}}
  --arms       PRIVATE key {neutral_id: {condition, target, context, case}}; supplied only
               after reviews are locked and unblinding is authorised.
"""
import json, argparse, random, itertools, os, sys
from collections import defaultdict

REF_PRESENT, REF_ABSENT, REF_NONE = "present", "absent", "no_reference"


def build_reference(resp_files):
    """Unanimous present/absent among returning reviewers; anything else -> no_reference."""
    votes = defaultdict(list)
    aliases = []
    for d in resp_files:
        if d.get("export_kind") != "final":
            continue
        aliases.append(d["reviewer_alias"])
        for r in d["responses"]:
            if r.get("target_presence"):
                votes[r["neutral_id"]].append(r["target_presence"])
    ref = {}
    for nid, v in votes.items():
        s = set(v)
        if s == {"present"}:
            ref[nid] = REF_PRESENT
        elif s == {"absent"}:
            ref[nid] = REF_ABSENT
        else:
            ref[nid] = REF_NONE          # disagreement OR any 'uncertain'
    return ref, sorted(set(aliases)), votes


def rates(ref, mach, ids):
    """Sensitivity / specificity with explicit denominators and uncertain-case bounds."""
    pos = [i for i in ids if ref.get(i) == REF_PRESENT]
    neg = [i for i in ids if ref.get(i) == REF_ABSENT]
    nor = [i for i in ids if ref.get(i) == REF_NONE]

    def band(denom, hit_pred, unc_pred):
        """returns (point|None, lo, hi) with uncertain assigned both ways"""
        if not denom:
            return None, None, None, 0
        hits = sum(1 for i in denom if mach.get(i) == hit_pred)
        unc = sum(1 for i in denom if mach.get(i) == "uncertain" or i not in mach)
        n = len(denom)
        point = hits / n
        return point, hits / n, (hits + unc) / n, n

    sp, slo, shi, sn = band(pos, "present", None)
    tp, tlo, thi, tn = band(neg, "absent", None)
    return {
        "sensitivity": None if sp is None else round(sp, 4),
        "sensitivity_bounds": None if sp is None else [round(slo, 4), round(shi, 4)],
        "sensitivity_denominator": sn,
        "specificity": None if tp is None else round(tp, 4),
        "specificity_bounds": None if tp is None else [round(tlo, 4), round(thi, 4)],
        "specificity_denominator": tn,
        "coverage": round((len(pos) + len(neg)) / len(ids), 4) if ids else None,
        "n_no_reference": len(nor), "n_items": len(ids),
        "note": ("rates are reported only where the denominator exists; bounds assign every "
                 "no_reference / evaluator-uncertain case both ways"),
    }


def cluster_bootstrap(ids_by_target, stat_fn, reps=2000, seed=20261018):
    """Resample targets, then cases within target. Never treats images as independent."""
    rnd = random.Random(seed)
    targets = sorted(ids_by_target)
    if not targets:
        return None
    out = []
    for _ in range(reps):
        picked = []
        for _ in targets:
            t = rnd.choice(targets)
            cases = ids_by_target[t]
            picked += [rnd.choice(cases) for _ in cases]
        v = stat_fn(picked)
        if v is not None:
            out.append(v)
    if not out:
        return None
    out.sort()
    return {"median": round(out[len(out) // 2], 4),
            "ci95": [round(out[int(.025 * len(out))], 4), round(out[int(.975 * len(out))], 4)],
            "reps": reps,
            "method": "cluster bootstrap: resample targets, then cases within target"}


def paired_transitions(ref, mach, arms, evaluator):
    """Per-case transitions original -> erased, for the reference and for the evaluator."""
    by_case = defaultdict(dict)
    for nid, a in arms.items():
        by_case[(a["target"], a["case"])][a["condition"]] = nid
    rows = []
    for (tgt, case), conds in sorted(by_case.items()):
        if "original" not in conds:
            continue
        o = conds["original"]
        for cond, nid in sorted(conds.items()):
            if cond == "original":
                continue
            rows.append({"target": tgt, "case": case, "condition": cond,
                         "ref_original": ref.get(o, REF_NONE), "ref_erased": ref.get(nid, REF_NONE),
                         "mach_original": mach.get(o), "mach_erased": mach.get(nid)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", nargs="+", required=True)
    ap.add_argument("--machine", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--i-have-authorisation-to-analyse-human-responses", action="store_true")
    a = ap.parse_args()

    files = [json.load(open(f)) for f in a.responses]
    synthetic = all(str(d.get("package_id", "")).startswith("msv1-FIXTURE") for d in files)
    if not synthetic and not a.i_have_authorisation_to_analyse_human_responses:
        raise SystemExit(
            "REFUSED: these look like real reviewer responses.\n"
            "The study protocol forbids human-outcome analysis before first-stage collection\n"
            "closes, responses are locked, and separate explicit authorisation is given.\n"
            "Re-run with --i-have-authorisation-to-analyse-human-responses once that holds.")

    ref, aliases, votes = build_reference(files)
    finals = [d for d in files if d.get("export_kind") == "final"]
    report = {"n_response_files": len(files), "n_final": len(finals), "aliases": aliases,
              "primary_analysis_runnable": len(finals) >= 2,
              "reference_convention": ("unanimous present/absent among returning reviewers; "
                                       "any uncertain or any disagreement -> no_reference, "
                                       "which is reported, never adjudicated"),
              "reference_counts": {k: sum(1 for v in ref.values() if v == k)
                                   for k in (REF_PRESENT, REF_ABSENT, REF_NONE)},
              "prior_exposure": {d["reviewer_alias"]: d["prior_exposure"] for d in files},
              "missingness_per_reviewer": {d["reviewer_alias"]: d["n_missing"] for d in files}}

    if len(finals) < 2:
        report["primary_analysis"] = ("NOT RUN: fewer than two complete independent returns. "
                                      "This is reported as not run, not approximated.")
        print(json.dumps(report, indent=1))
        if a.out:
            json.dump(report, open(a.out, "w"), indent=1)
        return

    if a.machine:
        machines = json.load(open(a.machine))
        arms = json.load(open(a.arms)) if a.arms else None
        ids = sorted(ref)
        report["evaluators"] = {}
        for ev, mach in sorted(machines.items()):
            entry = {"overall": rates(ref, mach, ids)}
            if arms:
                by_cond = defaultdict(list)
                for nid in ids:
                    if nid in arms:
                        by_cond[arms[nid]["condition"]].append(nid)
                entry["by_condition"] = {c: rates(ref, mach, v) for c, v in sorted(by_cond.items())}
                by_ct = defaultdict(lambda: defaultdict(list))
                for nid in ids:
                    if nid in arms:
                        by_ct[arms[nid]["condition"]][arms[nid]["target"]].append(nid)
                entry["by_condition_and_target"] = {
                    c: {t: rates(ref, mach, v) for t, v in sorted(d.items())}
                    for c, d in sorted(by_ct.items())}
                idx = defaultdict(list)
                for nid in ids:
                    if nid in arms:
                        idx[arms[nid]["target"]].append(nid)

                def acc(sample):
                    d = [i for i in sample if ref.get(i) in (REF_PRESENT, REF_ABSENT)]
                    if not d:
                        return None
                    return sum(1 for i in d if mach.get(i) == ref[i]) / len(d)
                entry["accuracy_cluster_bootstrap"] = cluster_bootstrap(idx, acc)
                entry["paired_transitions"] = paired_transitions(ref, mach, arms, ev)
            else:
                entry["note"] = ("no arm key supplied, so no by-condition split: the "
                                 "condition of each neutral id is private until unblinding")
            report["evaluators"][ev] = entry
    else:
        report["evaluators"] = "no machine predictions supplied (they are escrowed privately)"

    report["scope_limits"] = (
        "Paired, clustered pilot. Reviewers and repeated seeds are not extra independent "
        "images. Differences smaller than their own bounds are not resolved by this pilot, "
        "which cannot certify a five-point difference or a ranking reversal.")
    print(json.dumps(report, indent=1))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
