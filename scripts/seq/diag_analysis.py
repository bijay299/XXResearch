#!/usr/bin/env python3
"""Analysis for the matched-effectiveness diagnostic. CPU only, no GPU.

Deliberately a SEPARATE step from run_diagnostic.sh, so no confirmatory number
is produced by the process that spends GPU time.

Everything here is fixed in advance and read from frozen artifacts rather than
chosen at runtime:

  * the resampling grouping comes from
    results/audit_v1/draft_manifests/bootstrap_grouping.json -- the scene
    cluster (set, category, prompt_index), carrying BOTH family wordings and
    ALL FOUR generation seeds together, with CATEGORY as the stratum, 10 000
    draws, 95% percentile intervals and RNG seed 2026100901. Its identity is
    re-verified before use;
  * contrasts are paired at identical (prompt_id, gen_seed), and ONE resampled
    cluster list indexes BOTH arms and every endpoint within a draw;
  * TRAINING SEEDS ARE ANALYSED SEPARATELY AND NEVER POOLED. Two runs support a
    direction check only;
  * every retained category is reported SEPARATELY and never averaged; dog is
    the matching check; sandwich is a non-target; literal and paraphrase are
    reported separately; thresholds 0.3/0.5/0.7 with 0.5 primary.

The primary endpoint is the paired bird contrast D_bird(L2) - D_bird(U*) on the
frozen test set at t=0.5, where U* is the development-selected matched dump.

A note on the parent-referenced estimand. Both arms of a seed share that seed's
own MA, so the contrast of parent-referenced changes is ALGEBRAICALLY IDENTICAL
to the direct contrast:
    [D(L2) - D(MA)] - [D(U*) - D(MA)] = D(L2) - D(U*)
Per-arm changes from MA are therefore reported for interpretation, and the fact
that the contrast coincides is stated rather than presented as a second result.

Detector output is a PROXY. Nothing here is a final scientific conclusion: the
blinded human annotation is a prerequisite and is not an input to this script.

    python scripts/seq/diag_analysis.py --diag_root /data/.../diag_v1
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

THRESHOLDS = ("0.3", "0.5", "0.7")
PRIMARY_THR = "0.5"
PRIMARY_CATEGORY = "bird"
RETAINED = ("bird", "horse", "chair", "bicycle")
MATCHING_CHECK = "dog"
NON_TARGET = "sandwich"
HISTORICAL = "cat"
MATERIAL_PP = 10.0          # prespecified material effect size
EQUIV_LO, EQUIV_HI = -10.0, 10.0


def verify_grouping(p: Path) -> dict:
    d = json.loads(p.read_text())
    body = {k: v for k, v in d.items() if k != "grouping_sha256"}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if actual != d["grouping_sha256"]:
        raise SystemExit(f"FATAL: bootstrap grouping digest {actual[:12]}… != "
                         f"stored {d['grouping_sha256'][:12]}…: the frozen "
                         f"grouping was altered. Refusing to analyse.")
    return d


def load_arm(eval_root: Path, name: str) -> dict[tuple[str, int], dict]:
    det = eval_root / name / "detections.jsonl"
    if not det.is_file():
        raise SystemExit(f"FATAL: {det} absent")
    rows = {}
    for line in det.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        rows[(o["prompt_id"], int(o["gen_seed"]))] = o
    return rows


def rate(rows: list[dict], thr: str) -> float:
    """Detection rate in PERCENT over the given rows."""
    return 100.0 * sum(1 for r in rows if r["hit"][thr]) / len(rows) if rows else float("nan")


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    i = q * (len(s) - 1)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


class Endpoints:
    """Every endpoint, computed from one resampled list of scene clusters."""

    def __init__(self, grouping_set: dict, arms: dict[str, dict]):
        self.arms = arms
        self.clusters = grouping_set["clusters"]
        # stratum -> list of cluster indices
        self.by_stratum: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.clusters):
            self.by_stratum[c["stratum"]].append(i)
        # cluster index -> the (prompt_id, gen_seed) keys it owns, split by family
        self.keys: list[dict] = []
        gen_seeds = grouping_set["gen_seeds"]
        any_arm = next(iter(arms.values()))
        for c in self.clusters:
            allk, byfam = [], defaultdict(list)
            for pid in c["prompt_ids"]:
                for gs in gen_seeds:
                    k = (pid, int(gs))
                    if k not in any_arm:
                        raise SystemExit(f"FATAL: {k} missing from the scored "
                                         f"rows; the evaluation does not cover "
                                         f"the frozen grouping")
                    allk.append(k)
                    byfam[any_arm[k]["prompt_family"]].append(k)
            self.keys.append({"all": allk, "by_family": dict(byfam),
                              "stratum": c["stratum"]})

    def rows_for(self, arm: str, idxs: list[int], stratum: str | None,
                 family: str | None) -> list[dict]:
        a = self.arms[arm]
        out = []
        for i in idxs:
            k = self.keys[i]
            if stratum is not None and k["stratum"] != stratum:
                continue
            ks = k["by_family"].get(family, []) if family else k["all"]
            out.extend(a[x] for x in ks)
        return out

    def compute(self, idxs: list[int]) -> dict:
        """All endpoints for one draw (or for the point estimate)."""
        out: dict[str, float] = {}
        cats = sorted(self.by_stratum)
        for thr in THRESHOLDS:
            for cat in cats:
                for fam in (None, "literal", "paraphrase"):
                    tag = f"{cat}|{thr}" + (f"|{fam}" if fam else "")
                    r = {}
                    for arm in self.arms:
                        r[arm] = rate(self.rows_for(arm, idxs, cat, fam), thr)
                    out[f"D[{tag}]|L2"] = r["L2"]
                    out[f"D[{tag}]|U"] = r["U"]
                    out[f"D[{tag}]|MA"] = r["MA"]
                    # the paired contrast: L2 minus the matched unregularised arm
                    out[f"contrast[{tag}]"] = r["L2"] - r["U"]
                    # per-arm change from that seed's own MA
                    out[f"vsMA[{tag}]|L2"] = r["L2"] - r["MA"]
                    out[f"vsMA[{tag}]|U"] = r["U"] - r["MA"]
        return out


def analyse_seed(seed: int, sel_step: int, eval_root: Path,
                 grouping_set: dict, n_draws: int, rng_seed: int) -> dict:
    arms = {
        "MA": load_arm(eval_root, f"seed{seed}_MA"),
        "L2": load_arm(eval_root, f"seed{seed}_MAB_L2"),
        "U":  load_arm(eval_root, f"seed{seed}_U_step{sel_step}"),
    }
    E = Endpoints(grouping_set, arms)
    all_idx = list(range(len(E.clusters)))
    point = E.compute(all_idx)

    try:
        import numpy as np
    except Exception as e:
        raise SystemExit(f"FATAL: numpy required for the bootstrap ({e})")
    rng = np.random.default_rng(rng_seed)
    strata = sorted(E.by_stratum)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_draws):
        idxs: list[int] = []
        for s in strata:                      # sorted category order, as frozen
            pool = E.by_stratum[s]
            picks = rng.integers(0, len(pool), size=len(pool))
            idxs.extend(pool[int(j)] for j in picks)
        d = E.compute(idxs)
        for k, v in d.items():
            draws[k].append(v)

    ci = {k: {"point": point[k],
              "lo95": percentile(v, 0.025),
              "hi95": percentile(v, 0.975)}
          for k, v in draws.items()}
    return {"training_seed": seed, "selected_U_step": sel_step,
            "n_clusters": len(E.clusters),
            "clusters_per_stratum": {s: len(E.by_stratum[s]) for s in strata},
            "estimates": ci}


def read_decision(ci: dict, tag: str) -> dict:
    """Map one interval onto the prespecified rows of the decision table.

    The rows OVERLAP and are reported as such. An interval wholly inside
    [-10, +10] both rules out a 10 pp benefit and supports the equivalence
    statement; testing "upper bound < +10" first and stopping would hide the
    equivalence row in exactly the case where it applies. So every row that
    holds is reported, most specific first.
    """
    c = ci[f"contrast[{tag}]"]
    lo, hi = c["lo95"], c["hi95"]
    rows = []
    equivalent = EQUIV_LO <= lo and hi <= EQUIV_HI
    if equivalent:
        rows.append(f"interval wholly within [{EQUIV_LO:.0f}, {EQUIV_HI:.0f}]: "
                    f"PRACTICAL EQUIVALENCE at the declared margin")
    if hi < MATERIAL_PP:
        rows.append(f"upper bound {hi:+.1f} < +{MATERIAL_PP:.0f}: a benefit of "
                    f"{MATERIAL_PP:.0f} pp or more is RULED OUT, conditionally")
    if lo > MATERIAL_PP:
        rows.append(f"lower bound {lo:+.1f} > +{MATERIAL_PP:.0f}: a MATERIAL "
                    f"benefit carries forward")
    if not rows:
        rows.append("INCONCLUSIVE: the interval spans the declared margin; do "
                    "not read the point estimate as a result")
    return {"point": c["point"], "lo95": lo, "hi95": hi,
            "practical_equivalence": equivalent,
            "material_benefit": lo > MATERIAL_PP,
            "benefit_of_10pp_ruled_out": hi < MATERIAL_PP,
            "inconclusive": len(rows) == 1 and rows[0].startswith("INCONCLUSIVE"),
            "reading": "; ".join(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag_root", required=True)
    ap.add_argument("--grouping",
                    default="results/audit_v1/draft_manifests/bootstrap_grouping.json")
    ap.add_argument("--set", default="test", choices=["test", "dev"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    diag = Path(args.diag_root)
    grouping = verify_grouping(Path(args.grouping))
    gset = grouping["sets"][args.set]
    n_draws = grouping["n_draws"]
    rng_seed = grouping["rng"]["bootstrap_rng_seed"]
    eval_root = diag / ("eval_test" if args.set == "test" else "eval_dev")

    sel_p = diag / "selection.json"
    if not sel_p.is_file():
        raise SystemExit(f"FATAL: {sel_p} absent; selection has not been decided")
    sel = json.loads(sel_p.read_text())
    if not sel.get("proceed_to_frozen_test"):
        raise SystemExit("FATAL: development matching did not succeed for both "
                         "seeds, so there is no confirmatory analysis to run. "
                         "The selection record is the result.")
    seeds = sel["seeds"]

    print(f"grouping identity   : {grouping['grouping_sha256'][:16]}… verified")
    print(f"resampling unit     : {grouping['resampling_unit']['definition']}")
    print(f"stratum             : {grouping['stratification']['stratum']}")
    print(f"draws / interval    : {n_draws} / 95% percentile")
    print(f"bootstrap RNG seed  : {rng_seed}")
    print(f"set                 : {args.set} ({gset['n_rows']} rows, "
          f"{gset['n_clusters']} clusters)")
    print("training seeds are analysed SEPARATELY and never pooled\n")

    per_seed = {}
    for s in seeds:
        step = sel["per_seed"][str(s)]["selected"]["step"]
        print(f"--- seed {s}: U* = dump at step {step} "
              f"(bootstrapping {n_draws} draws) ---", flush=True)
        per_seed[s] = analyse_seed(s, step, eval_root, gset, n_draws, rng_seed)

    primary_tag = f"{PRIMARY_CATEGORY}|{PRIMARY_THR}"
    payload = {
        "_what_this_is": (
            "The matched-effectiveness diagnostic's detector-based estimates. "
            "PROVISIONAL: the blinded human annotation is a prerequisite for any "
            "final scientific conclusion and is not an input here."),
        "primary_endpoint": (
            f"paired contrast D_{PRIMARY_CATEGORY}(L2) - D_{PRIMARY_CATEGORY}(U*) "
            f"on the frozen test set at t={PRIMARY_THR}"),
        "grouping_sha256": grouping["grouping_sha256"],
        "bootstrap": {"n_draws": n_draws, "rng_seed": rng_seed,
                      "interval": "95% percentile",
                      "unit": grouping["resampling_unit"]["definition"],
                      "stratum": grouping["stratification"]["stratum"]},
        "selection": {str(s): {"step": sel["per_seed"][str(s)]["selected"]["step"],
                               "mismatch_vs_L2_pp":
                                   sel["per_seed"][str(s)]["selected"]["mismatch_vs_L2_pp"]}
                      for s in seeds},
        "seeds_pooled": False,
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "primary_reading_per_seed": {
            str(s): read_decision(per_seed[s]["estimates"], primary_tag)
            for s in seeds},
        "parent_referenced_note": (
            "Both arms of a seed share that seed's own MA, so the contrast of "
            "parent-referenced changes is algebraically identical to the direct "
            "contrast. Per-arm changes from MA are reported for interpretation; "
            "the coincidence is stated, not presented as a second result."),
        "interpretation_rules": [
            "An interval including zero is NOT evidence of no effect.",
            "A difference in significance labels between arms or seeds is NOT a "
            "significant difference.",
            "No multiplicity correction is applied, so each interval is "
            "DESCRIPTIVE, not a test.",
            "Only an interval wholly within [-10, +10] supports an equivalence "
            "statement, and only at that margin.",
            "Two training seeds support a DIRECTION CHECK only; they are never "
            "pooled and support no population claim.",
            "Intervals are conditional on this base model, these concepts, these "
            "anchors, these prompts and this detector.",
            "Residual dependence from a template shared ACROSS scene clusters is "
            "not absorbed by this bootstrap, and its direction and size are not "
            "established.",
            "The detector is a PROXY. Final conclusions require the blinded human "
            "annotation.",
        ],
        "analysed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    out = Path(args.out) if args.out else diag / f"analysis_{args.set}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    # ---- human-readable summary
    print(f"\n=== PRIMARY: paired {PRIMARY_CATEGORY} contrast at t={PRIMARY_THR} "
          f"(L2 minus matched-U), per seed ===")
    for s in seeds:
        r = payload["primary_reading_per_seed"][str(s)]
        print(f"  seed {s}: {r['point']:+.1f} pp  [{r['lo95']:+.1f}, {r['hi95']:+.1f}]")
        print(f"           {r['reading']}")
    dirs = {s: payload["primary_reading_per_seed"][str(s)]["point"] for s in seeds}
    if len(seeds) == 2 and (dirs[seeds[0]] > 0) != (dirs[seeds[1]] > 0):
        print("  DIRECTIONS DISAGREE between seeds -> UNRESOLVED. Do not report "
              "the agreeing seed.")

    print(f"\n=== every category separately, never averaged (t={PRIMARY_THR}) ===")
    for s in seeds:
        ci = per_seed[s]["estimates"]
        print(f"  seed {s}:")
        for cat in sorted(per_seed[s]["clusters_per_stratum"]):
            tag = f"{cat}|{PRIMARY_THR}"
            c = ci[f"contrast[{tag}]"]
            role = ("PRIMARY" if cat == PRIMARY_CATEGORY else
                    "matching check" if cat == MATCHING_CHECK else
                    "non-target" if cat == NON_TARGET else
                    "historical" if cat == HISTORICAL else
                    "retained" if cat in RETAINED else "")
            print(f"    {cat:<9} L2={ci[f'D[{tag}]|L2']['point']:5.1f}%  "
                  f"U*={ci[f'D[{tag}]|U']['point']:5.1f}%  "
                  f"MA={ci[f'D[{tag}]|MA']['point']:5.1f}%  "
                  f"contrast {c['point']:+6.1f} "
                  f"[{c['lo95']:+6.1f}, {c['hi95']:+6.1f}]  {role}")

    print(f"\n=== literal vs paraphrase, reported separately (t={PRIMARY_THR}) ===")
    for s in seeds:
        ci = per_seed[s]["estimates"]
        print(f"  seed {s}:")
        for fam in ("literal", "paraphrase"):
            tag = f"{PRIMARY_CATEGORY}|{PRIMARY_THR}|{fam}"
            c = ci[f"contrast[{tag}]"]
            print(f"    {PRIMARY_CATEGORY} {fam:<11} contrast {c['point']:+6.1f} "
                  f"[{c['lo95']:+6.1f}, {c['hi95']:+6.1f}]")

    print(f"\n=== matching check on TEST: {MATCHING_CHECK} "
          f"(the match was fixed on DEV and is NOT reselected here) ===")
    for s in seeds:
        ci = per_seed[s]["estimates"]
        tag = f"{MATCHING_CHECK}|{PRIMARY_THR}"
        c = ci[f"contrast[{tag}]"]
        print(f"  seed {s}: residue L2={ci[f'D[{tag}]|L2']['point']:.1f}%  "
              f"U*={ci[f'D[{tag}]|U']['point']:.1f}%  "
              f"achieved difference {c['point']:+.1f} pp "
              f"[{c['lo95']:+.1f}, {c['hi95']:+.1f}]")

    print(f"\n[analysis] {out}")
    print("\nPROVISIONAL. The detector is a proxy; the blinded human annotation "
          "is a prerequisite for any final scientific conclusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
