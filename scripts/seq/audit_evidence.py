#!/usr/bin/env python3
"""AUDIT-01: independent re-derivation and integrity audit of the two-seed snapshot.

Reads ONLY the raw per-image detector predictions (``detections.jsonl``) and the
frozen evaluation manifest. Nothing here consumes the committed ``contrasts.json``
or ``rates.csv`` as an input to a computation -- those are re-derived from raw and
then *compared* against the committed copies, so a disagreement is detectable.

Outputs (results/audit_v1/):
  integrity.json        counts, identity, hash, completeness and duplicate audit
  recomputed_rates.csv  D by seed x checkpoint x category x family x threshold
  direct_contrasts.json direct paired L2 vs no-L2 contrasts at 0.3/0.5/0.7
  crosscheck.json       re-derived vs committed summary comparison

Design notes that matter for interpretation
-------------------------------------------
* Resampling unit is the PROMPT. Each prompt carries all four of its generation
  seeds into or out of a bootstrap draw together, so prompt/seed matching across
  checkpoints is preserved exactly. Draws are paired: one resampled prompt list
  indexes both arms of a contrast.
* Training seeds are NEVER pooled or resampled against each other. Every interval
  is sampling uncertainty over prompts WITHIN one training run. n=2 training runs
  supports a direction check only.
* The L2 contrasts are deliberately expressed as the direct child-vs-child
  difference. For both suppression and recovery the shared MA parent term cancels
  algebraically (see ``direct_l2_contrast``), so these contrasts carry no MA
  uncertainty at all.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CATS = ["cat", "dog", "sandwich", "horse", "bird", "chair", "bicycle"]
CHECKPOINTS = ["M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"]
THRESHOLDS = ["0.3", "0.5", "0.7"]
# (unregularised arm, L2-SP arm, new deletion target, anchor)
L2_PAIRS = [("MAB", "MAB_L2", "dog", "horse"), ("MAC", "MAC_L2", "sandwich", "flower")]
NEW_TARGET = {"MAB": "dog", "MAB_L2": "dog", "MAC": "sandwich", "MAC_L2": "sandwich"}
# Categories that are never a deletion target in either branch.
RETAINED = ["horse", "bird", "chair", "bicycle"]
FAMILIES = ["all", "literal", "paraphrase"]
N_BOOT = 10000


# --------------------------------------------------------------------------- IO
def load_seed(eval_root: Path) -> dict[str, list[dict]]:
    """Load raw detector predictions for every checkpoint of one training seed."""
    out: dict[str, list[dict]] = {}
    for ck in CHECKPOINTS:
        p = eval_root / ck / "detections.jsonl"
        if not p.exists():
            continue
        out[ck] = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return out


def sel(rows: list[dict], cat: str, fam: str) -> list[dict]:
    return [r for r in rows
            if r["category"] == cat and (fam == "all" or r["prompt_family"] == fam)]


def rate(rows: list[dict], thr: str) -> float | None:
    if not rows:
        return None
    return sum(bool(r["hit"][thr]) for r in rows) / len(rows)


# -------------------------------------------------------------------- bootstrap
def paired_boot(rows_a: list[dict], rows_b: list[dict], thr: str,
                n: int = N_BOOT, seed: int = 20261009) -> dict | None:
    """Paired cluster bootstrap of D(a) - D(b); clusters are prompts.

    Each prompt_id is one cluster carrying all of its generation seeds. A single
    resampled prompt list is applied to BOTH arms, so the pairing that the shared
    evaluation manifest creates is preserved in every draw.
    """
    by_a: dict[str, list[dict]] = defaultdict(list)
    by_b: dict[str, list[dict]] = defaultdict(list)
    for r in rows_a:
        by_a[r["prompt_id"]].append(r)
    for r in rows_b:
        by_b[r["prompt_id"]].append(r)
    prompts = sorted(set(by_a) & set(by_b))
    if not prompts:
        return None
    pa = rate([x for p in prompts for x in by_a[p]], thr)
    pb = rate([x for p in prompts for x in by_b[p]], thr)
    point = (pa - pb) * 100

    rng = random.Random(seed)
    draws = []
    k = len(prompts)
    for _ in range(n):
        samp = [prompts[rng.randrange(k)] for _ in range(k)]
        a = [x for p in samp for x in by_a[p]]
        b = [x for p in samp for x in by_b[p]]
        draws.append(rate(a, thr) - rate(b, thr))
    draws.sort()
    lo = draws[int(0.025 * n)] * 100
    hi = draws[min(n - 1, int(0.975 * n))] * 100
    return {
        "diff_pp": round(point, 2),
        "ci95_pp": [round(lo, 2), round(hi, 2)],
        f"D_a_pct": round(pa * 100, 2),
        f"D_b_pct": round(pb * 100, 2),
        "n_prompt_clusters": k,
        "n_images_per_arm": len(rows_a),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# ------------------------------------------------------------- integrity audit
def integrity(seeds: dict[str, dict[str, list[dict]]], manifest: dict) -> dict:
    """Verify counts, identities, hashes, completeness and M0 reuse from raw rows."""
    rep: dict = {"manifest": {}, "per_seed": {}, "issues": [], "m0_reuse": {}}

    # ---- expected identity grid, taken from the frozen manifest -------------
    man_rows = manifest.get("records") if isinstance(manifest, dict) else None
    if man_rows:
        expected = {(r["category"], r["prompt_id"], r["gen_seed"]) for r in man_rows}
    else:  # manifest shape differs; completeness cannot be checked against it
        expected = None
    rep["manifest"] = {
        "path_sha256_recorded": manifest.get("manifest_sha256"),
        "frozen_before_any_editing": manifest.get("frozen_before_any_editing"),
        "n_records": len(man_rows) if man_rows else None,
        "expected_identities": len(expected) if expected else None,
        "gen_seeds": manifest.get("gen_seeds"),
        "training_prompt_disjointness": manifest.get("training_prompt_disjointness"),
    }

    all_hashes: dict[str, list[str]] = defaultdict(list)

    for sname, data in seeds.items():
        s: dict = {"checkpoints_present": sorted(data), "per_checkpoint": {}}
        for ck, rows in data.items():
            ids = [(r["category"], r["prompt_id"], r["gen_seed"]) for r in rows]
            dup = [k for k, c in Counter(ids).items() if c > 1]
            hashes = [r["image_sha256"] for r in rows]
            dup_hash = {h: c for h, c in Counter(hashes).items() if c > 1}
            cat_counts = Counter(r["category"] for r in rows)
            fam_counts = Counter(r["prompt_family"] for r in rows)
            seed_counts = Counter(r["gen_seed"] for r in rows)
            prompts_per_cat = {c: len({r["prompt_id"] for r in rows if r["category"] == c})
                               for c in CATS}
            missing = sorted(expected - set(ids)) if expected else None
            extra = sorted(set(ids) - expected) if expected else None

            entry = {
                "n_rows": len(rows),
                "n_unique_identities": len(set(ids)),
                "duplicate_identities": [list(d) for d in dup],
                "n_unique_image_sha256": len(set(hashes)),
                "duplicate_image_sha256_within_checkpoint": len(dup_hash),
                "category_counts": dict(cat_counts),
                "family_counts": dict(fam_counts),
                "gen_seed_counts": dict(sorted(seed_counts.items())),
                "unique_prompts_per_category": prompts_per_cat,
                "missing_vs_manifest": len(missing) if missing is not None else None,
                "extra_vs_manifest": len(extra) if extra is not None else None,
                "hit_fields_present": sorted({t for r in rows for t in r["hit"]}),
            }
            s["per_checkpoint"][ck] = entry

            if dup:
                rep["issues"].append(
                    f"{sname}/{ck}: {len(dup)} duplicate (category,prompt_id,gen_seed) keys")
            if missing:
                rep["issues"].append(f"{sname}/{ck}: {len(missing)} manifest rows missing")
            if extra:
                rep["issues"].append(f"{sname}/{ck}: {len(extra)} rows absent from manifest")
            if len(rows) != 280:
                rep["issues"].append(f"{sname}/{ck}: n_rows={len(rows)}, expected 280")
            if dup_hash:
                rep["issues"].append(
                    f"{sname}/{ck}: {len(dup_hash)} image hashes occur more than once "
                    f"(identical generated images within one checkpoint)")
            for h in hashes:
                all_hashes[h].append(f"{sname}/{ck}")

        rep["per_seed"][sname] = s

    # ---- M0 reuse: is the same byte-identical evaluation set shared? --------
    m0_sets = {sn: {r["image_sha256"] for r in d["M0"]} for sn, d in seeds.items() if "M0" in d}
    if len(m0_sets) == 2:
        a, b = list(m0_sets)
        shared = m0_sets[a] & m0_sets[b]
        rep["m0_reuse"] = {
            "seeds_compared": [a, b],
            "n_hashes_each": [len(m0_sets[a]), len(m0_sets[b])],
            "n_identical_hashes": len(shared),
            "byte_identical_reuse": len(shared) == len(m0_sets[a]) == len(m0_sets[b]),
            "interpretation": (
                "M0 is one evaluation set reused by both seed tables, not two "
                "independent draws. It must be counted ONCE in any generated-image "
                "total and its rows must not be treated as independent replication."),
        }

    # ---- identical images ACROSS checkpoints (a child that did not move) ----
    cross = {h: v for h, v in all_hashes.items() if len(set(v)) > 1}
    # Expected: M0 appears under both seed tables via the symlink. Anything else
    # would mean two different checkpoints produced a byte-identical image.
    non_m0 = {h: sorted(set(v)) for h, v in cross.items()
              if not all(x.endswith("/M0") for x in v)}
    rep["cross_checkpoint_identical_images"] = {
        "n_hashes_shared_across_tables": len(cross),
        "n_shared_excluding_M0_reuse": len(non_m0),
        "examples_excluding_M0": list(non_m0.items())[:5],
        "note": ("M0 hashes are expected to appear under both seed tables because "
                 "eval_seed29/M0 is a symlink to eval/M0."),
    }
    return rep


# --------------------------------------------------------------- direct L2 work
def direct_l2_contrast(data: dict[str, list[dict]]) -> dict:
    """Direct paired L2-SP vs unregularised contrasts, within one training seed.

    All four blocks are child-vs-child differences at identical prompt/seed pairs.

    Why the parent cancels
    ----------------------
    new-target suppression  S(X)  = D_new(MA) - D_new(X)
        S(noL2) - S(L2)           = D_new(L2) - D_new(noL2)
    historical cat recovery R(X)  = D_cat(X)  - D_cat(MA)
        R(L2)   - R(noL2)         = D_cat(L2) - D_cat(noL2)

    The MA term drops out of both. So these contrasts are completely insensitive
    to the between-seed movement in D_cat(MA) that the narrative reports blame for
    the non-replication -- an important check on that explanation.
    """
    out: dict = {}
    for no_l2, l2, target, anchor in L2_PAIRS:
        if no_l2 not in data or l2 not in data:
            continue
        blk: dict = {"unregularised_arm": no_l2, "l2sp_arm": l2,
                     "new_deletion_target": target, "anchor": anchor, "by_threshold": {}}
        for thr in THRESHOLDS:
            tb: dict = {}
            for fam in FAMILIES:
                fb: dict = {}

                # (1) new-target suppression: less negative => L2 suppressed less.
                c = paired_boot(sel(data[l2], target, fam), sel(data[no_l2], target, fam), thr)
                if c:
                    c["reads_as"] = (
                        f"D_{target}({l2}) - D_{target}({no_l2}); POSITIVE means the "
                        f"L2-SP arm left more of the new target standing, i.e. weaker "
                        f"deletion. Equals suppression({no_l2}) - suppression({l2}); "
                        f"the MA parent term cancels.")
                fb["new_target_residual_l2_minus_nol2"] = c

                # (2) historical cat: direct child-vs-child, MA-free.
                c = paired_boot(sel(data[l2], "cat", fam), sel(data[no_l2], "cat", fam), thr)
                if c:
                    c["reads_as"] = (
                        f"D_cat({l2}) - D_cat({no_l2}); POSITIVE means more residual "
                        f"cat under L2-SP. Equals recovery({l2}) - recovery({no_l2}); "
                        f"independent of D_cat(MA).")
                fb["cat_residual_l2_minus_nol2"] = c

                # (3) each retained category separately -- never averaged.
                fb["retained_l2_minus_nol2"] = {}
                for cat in RETAINED:
                    c = paired_boot(sel(data[l2], cat, fam), sel(data[no_l2], cat, fam), thr)
                    if c:
                        c["reads_as"] = (f"D_{cat}({l2}) - D_{cat}({no_l2}); POSITIVE "
                                         f"means L2-SP retained more of this category.")
                    fb["retained_l2_minus_nol2"][cat] = c

                # (4) the sibling branch's target, reported separately.
                other = "sandwich" if target == "dog" else "dog"
                fb["undeleted_sibling_target"] = {
                    "category": other,
                    "contrast": paired_boot(sel(data[l2], other, fam),
                                            sel(data[no_l2], other, fam), thr),
                    "note": "never folded into a retained average; it is a target elsewhere",
                }
                tb[fam] = fb

            # (5) literal-vs-paraphrase interaction on the L2 effect.
            tb["family_interaction"] = {}
            for label, cat in (("new_target", target), ("cat", "cat")):
                lit = tb["literal"].get(
                    "new_target_residual_l2_minus_nol2" if label == "new_target"
                    else "cat_residual_l2_minus_nol2")
                par = tb["paraphrase"].get(
                    "new_target_residual_l2_minus_nol2" if label == "new_target"
                    else "cat_residual_l2_minus_nol2")
                if lit and par:
                    tb["family_interaction"][label] = {
                        "literal_diff_pp": lit["diff_pp"],
                        "paraphrase_diff_pp": par["diff_pp"],
                        "paraphrase_minus_literal_pp": round(
                            par["diff_pp"] - lit["diff_pp"], 2),
                        "note": ("descriptive difference of two independent point "
                                 "estimates; no interval is computed for it and no "
                                 "interaction test is claimed"),
                    }
            blk["by_threshold"][thr] = tb
        out[f"{no_l2}_vs_{l2}"] = blk
    return out


def reference_dependent_contrasts(data: dict[str, list[dict]]) -> dict:
    """The two MA-referenced estimands, at every threshold and prompt family.

    Reported beside the direct child-vs-child contrasts so the reader can see
    which conclusions depend on the MA reference term and which do not. The
    MA used is always the MA of the SAME training seed.
    """
    out: dict = {}
    for thr in THRESHOLDS:
        tb: dict = {}
        for fam in FAMILIES:
            fb: dict = {"historical_cat_recovery_vs_MA": {},
                        "new_target_suppression_vs_MA": {}}
            for ck in NEW_TARGET:
                if ck not in data or "MA" not in data:
                    continue
                c = paired_boot(sel(data[ck], "cat", fam), sel(data["MA"], "cat", fam), thr)
                if c:
                    c["reads_as"] = (f"D_cat({ck}) - D_cat(MA) of the SAME training "
                                     f"seed; POSITIVE = more cat than the parent.")
                fb["historical_cat_recovery_vs_MA"][ck] = c
                tgt = NEW_TARGET[ck]
                c = paired_boot(sel(data["MA"], tgt, fam), sel(data[ck], tgt, fam), thr)
                if c:
                    c["reads_as"] = (f"D_{tgt}(MA) - D_{tgt}({ck}); POSITIVE = the "
                                     f"second request removed the new target.")
                fb["new_target_suppression_vs_MA"][ck] = c
            # The parent's own level -- the term that cancels in the direct contrasts.
            r = rate(sel(data.get("MA", []), "cat", fam), thr)
            fb["D_cat_MA_pct"] = None if r is None else round(r * 100, 2)
            tb[fam] = fb
        out[thr] = tb
    return out


def effectiveness_table(data: dict[str, list[dict]]) -> dict:
    """Deletion effectiveness and residual levels per arm -- the matching context.

    A difference in retention is only interpretable alongside how much deletion
    each arm actually achieved, so both are emitted together.
    """
    out: dict = {}
    for thr in THRESHOLDS:
        rows = {}
        for ck in CHECKPOINTS:
            if ck not in data:
                continue
            e = {"D_pct": {c: (None if rate(sel(data[ck], c, "all"), thr) is None
                               else round(rate(sel(data[ck], c, "all"), thr) * 100, 2))
                           for c in CATS}}
            tgt = NEW_TARGET.get(ck)
            if tgt and "MA" in data:
                d_ma = rate(sel(data["MA"], tgt, "all"), thr)
                d_ck = rate(sel(data[ck], tgt, "all"), thr)
                e["new_target"] = tgt
                e["new_target_suppression_pp_vs_MA"] = round((d_ma - d_ck) * 100, 2)
                e["new_target_residual_pct"] = round(d_ck * 100, 2)
            rows[ck] = e
        out[thr] = rows
    return out


# ------------------------------------------------------------------- crosscheck
def crosscheck(data: dict[str, list[dict]], committed_rates: Path,
               committed_contrasts: Path) -> dict:
    """Re-derive the committed rates.csv / contrasts.json and diff them."""
    rep: dict = {"rates_csv": {"checked": 0, "mismatches": []},
                 "contrasts_json": {"checked": 0, "mismatches": []}}

    if committed_rates.exists():
        for line in committed_rates.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            ck, cat, fam, n, d3, d5, d7, nh5 = line.split(",")
            rows = sel(data.get(ck, []), cat, fam)
            rep["rates_csv"]["checked"] += 1
            if len(rows) != int(n):
                rep["rates_csv"]["mismatches"].append(
                    {"row": line, "field": "n", "recomputed": len(rows)})
                continue
            for thr, claimed in (("0.3", d3), ("0.5", d5), ("0.7", d7)):
                mine = rate(rows, thr)
                if mine is None or abs(mine - float(claimed)) > 5e-5:
                    rep["rates_csv"]["mismatches"].append(
                        {"row": line, "field": f"D_{thr}", "claimed": float(claimed),
                         "recomputed": None if mine is None else round(mine, 4)})
            nh = sum(bool(r["hit"]["0.5"]) for r in rows)
            if nh != int(nh5):
                rep["rates_csv"]["mismatches"].append(
                    {"row": line, "field": "n_hits_0.5", "claimed": int(nh5),
                     "recomputed": nh})

    if committed_contrasts.exists():
        con = json.loads(committed_contrasts.read_text())
        # Re-derive every raw rate the committed file asserts.
        for ck, cells in con.get("raw_rates_pct", {}).items():
            for key, per_thr in cells.items():
                cat, fam = key.split("|")
                rows = sel(data.get(ck, []), cat, fam)
                for thr, claimed in per_thr.items():
                    mine = rate(rows, thr)
                    rep["contrasts_json"]["checked"] += 1
                    if mine is None or abs(mine * 100 - claimed) > 0.011:
                        rep["contrasts_json"]["mismatches"].append(
                            {"checkpoint": ck, "cell": key, "thr": thr,
                             "claimed_pct": claimed,
                             "recomputed_pct": None if mine is None else round(mine * 100, 2)})
        # Re-derive the point estimates of the headline contrasts.
        fam_all = con.get("families", {}).get("all", {})
        init = fam_all.get("initial_cat_suppression")
        if init:
            mine = (rate(sel(data["M0"], "cat", "all"), con["threshold"])
                    - rate(sel(data["MA"], "cat", "all"), con["threshold"])) * 100
            rep["contrasts_json"]["checked"] += 1
            if abs(mine - init["diff_pp"]) > 0.011:
                rep["contrasts_json"]["mismatches"].append(
                    {"contrast": "initial_cat_suppression", "claimed": init["diff_pp"],
                     "recomputed": round(mine, 2)})
        for ck, c in (fam_all.get("historical_cat_recovery") or {}).items():
            if not c:
                continue
            mine = (rate(sel(data[ck], "cat", "all"), con["threshold"])
                    - rate(sel(data["MA"], "cat", "all"), con["threshold"])) * 100
            rep["contrasts_json"]["checked"] += 1
            if abs(mine - c["diff_pp"]) > 0.011:
                rep["contrasts_json"]["mismatches"].append(
                    {"contrast": f"historical_cat_recovery/{ck}",
                     "claimed": c["diff_pp"], "recomputed": round(mine, 2)})
        for ck, c in (fam_all.get("new_request_suppression") or {}).items():
            if not c:
                continue
            tgt = NEW_TARGET[ck]
            mine = (rate(sel(data["MA"], tgt, "all"), con["threshold"])
                    - rate(sel(data[ck], tgt, "all"), con["threshold"])) * 100
            rep["contrasts_json"]["checked"] += 1
            if abs(mine - c["diff_pp"]) > 0.011:
                rep["contrasts_json"]["mismatches"].append(
                    {"contrast": f"new_request_suppression/{ck}",
                     "claimed": c["diff_pp"], "recomputed": round(mine, 2)})
    return rep


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/bijaypandey/cuig_pilot/seq_pilot")
    ap.add_argument("--repo", default="/home/bijaypandey/FINMLResearch")
    ap.add_argument("--out", default="results/audit_v1")
    args = ap.parse_args()

    root, repo = Path(args.root), Path(args.repo)
    out = repo / args.out
    out.mkdir(parents=True, exist_ok=True)

    seed_dirs = {"seed17": root / "eval", "seed29": root / "eval_seed29"}
    seeds = {}
    for sn, d in seed_dirs.items():
        if not d.exists():
            print(f"MISSING eval root for {sn}: {d}")
            continue
        seeds[sn] = load_seed(d)
        print(f"{sn}: " + ", ".join(f"{k}={len(v)}" for k, v in seeds[sn].items()))

    man_path = root / "eval_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}

    # ---- 1. integrity ------------------------------------------------------
    integ = integrity(seeds, manifest)
    integ["symlink_audit"] = {
        str(p): (str(p.resolve()) if p.is_symlink() else "not a symlink")
        for p in [seed_dirs["seed29"] / "M0", seed_dirs["seed17"] / "M0"]
        if p.exists()
    }
    (out / "integrity.json").write_text(json.dumps(integ, indent=2))
    print(f"\nintegrity issues: {len(integ['issues'])}")
    for i in integ["issues"]:
        print("  !", i)

    # ---- 2. recomputed rates ----------------------------------------------
    lines = ["training_seed,checkpoint,category,prompt_family,n,n_hits,"
             "threshold,D_pct"]
    for sn, data in seeds.items():
        tseed = sn.replace("seed", "")
        for ck in CHECKPOINTS:
            if ck not in data:
                continue
            for cat in CATS:
                for fam in FAMILIES:
                    rows = sel(data[ck], cat, fam)
                    if not rows:
                        continue
                    for thr in THRESHOLDS:
                        nh = sum(bool(r["hit"][thr]) for r in rows)
                        lines.append(f"{tseed},{ck},{cat},{fam},{len(rows)},{nh},"
                                     f"{thr},{100.0*nh/len(rows):.2f}")
    (out / "recomputed_rates.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote recomputed_rates.csv ({len(lines)-1} rows)")

    # ---- 3. direct paired L2 contrasts ------------------------------------
    direct = {
        "_design": {
            "resampling_unit": "prompt (10 per category), carrying its 4 generation seeds",
            "n_bootstrap": N_BOOT,
            "pairing": "one resampled prompt list indexes both arms of every contrast",
            "training_seeds": "analysed separately; never pooled, never resampled jointly",
            "interval_meaning": (
                "sampling uncertainty over the 10 evaluation prompts within ONE "
                "training run. With 10 clusters of 4 images the attainable "
                "resolution is coarse: a 95% interval cannot be narrower than a "
                "few percentage points, and an interval that includes zero is NOT "
                "evidence that the effect is zero."),
            "multiplicity": (
                "no multiple-comparison correction is applied; many contrasts are "
                "reported, so individual interval-excludes-zero labels must be read "
                "as descriptive, not as tests."),
            "parent_cancellation": (
                "L2 contrasts are child-vs-child, so the MA parent term cancels and "
                "these estimates do not inherit any D_cat(MA) instability."),
        },
        "per_seed": {},
    }
    for sn, data in seeds.items():
        direct["per_seed"][sn] = {
            "l2_contrasts": direct_l2_contrast(data),
            "reference_dependent": reference_dependent_contrasts(data),
            "effectiveness_context": effectiveness_table(data),
        }
        print(f"computed direct L2 contrasts for {sn}")
    (out / "direct_contrasts.json").write_text(json.dumps(direct, indent=2))

    # ---- 4. crosscheck against committed summaries -------------------------
    cc = {}
    for sn, data in seeds.items():
        rdir = repo / "results" / ("seq17" if sn == "seed17" else "seq29")
        cc[sn] = crosscheck(data, rdir / "rates.csv", rdir / "contrasts.json")
        m = cc[sn]["rates_csv"]["mismatches"] + cc[sn]["contrasts_json"]["mismatches"]
        print(f"{sn} crosscheck: "
              f"{cc[sn]['rates_csv']['checked']} rate cells + "
              f"{cc[sn]['contrasts_json']['checked']} contrast values, "
              f"{len(m)} mismatches")
    (out / "crosscheck.json").write_text(json.dumps(cc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
