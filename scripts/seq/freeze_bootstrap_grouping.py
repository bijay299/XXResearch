#!/usr/bin/env python3
"""Freeze the bootstrap resampling grouping for the matched-effectiveness design.

Why. The returned protocol said only "resample prompt texts carrying their four
generation seeds together", and asserted that where texts share a template the
clusters are "not fully independent, so intervals are mildly optimistic". The
earlier accepted design also required literal/paraphrase stratification and a
recorded RNG seed, and the "mildly optimistic" direction and size were never
established. This script derives the ACTUAL grouping from the actual manifests
and writes it down, so the resampling unit is fixed before any data exist rather
than chosen during analysis.

What it establishes, by reading the manifests rather than by assertion:

  * the resampling UNIT is the scene cluster `(set, category, prompt_index)`,
    which carries BOTH family members (literal and paraphrase) and ALL FOUR
    generation seeds together. The two families of one prompt_index are the same
    scene worded two ways -- they are built that way -- so they are dependent and
    must move together, not be drawn independently;
  * the STRATUM is the category, so every draw preserves the category
    composition, and the literal/paraphrase split is preserved exactly because
    each cluster contributes both families;
  * the literal-only and paraphrase-only endpoints are recomputed on the SAME
    resampled cluster list, which is what keeps them paired with each other and
    with the two arms;
  * the residual dependence the cluster bootstrap CANNOT absorb is measured: the
    opening-phrase templates shared ACROSS clusters are counted and listed. Its
    effect on interval width is NOT estimated here, and no claim is made about
    its direction or size.

CPU only, no GPU, no model, no network. Reads manifests, writes one JSON.

    python scripts/seq/freeze_bootstrap_grouping.py \
        --out results/audit_v1/draft_manifests/bootstrap_grouping.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_stage import canonical_manifest_digest  # noqa: E402

# Fixed, recorded, and distinct from the annotation sampler's rng_seed=20261009
# so the two samplers can never be confused for one another.
BOOTSTRAP_RNG_SEED = 2026100901
N_DRAWS = 10_000
OPENING_TOKENS = 4


def opening_phrase(text: str, n: int = OPENING_TOKENS) -> str:
    return " ".join(text.split()[:n]).lower()


def group_one(name: str, man: dict) -> dict:
    recs = man["records"]
    by_cluster: dict[tuple[str, int], dict] = {}
    for r in recs:
        key = (r["category"], int(r["prompt_index"]))
        c = by_cluster.setdefault(key, {
            "cluster_id": f"{name}:{r['category']}:{r['prompt_index']}",
            "stratum": r["category"],
            "prompt_index": int(r["prompt_index"]),
            "prompt_ids": [],
            "prompt_texts": {},
            "families": [],
            "gen_seeds": [],
            "n_rows": 0,
        })
        if r["prompt_id"] not in c["prompt_ids"]:
            c["prompt_ids"].append(r["prompt_id"])
            c["prompt_texts"][r["prompt_id"]] = r["prompt"]
        if r["prompt_family"] not in c["families"]:
            c["families"].append(r["prompt_family"])
        if int(r["gen_seed"]) not in c["gen_seeds"]:
            c["gen_seeds"].append(int(r["gen_seed"]))
        c["n_rows"] += 1

    clusters = [by_cluster[k] for k in sorted(by_cluster)]
    for c in clusters:
        c["prompt_ids"].sort()
        c["families"].sort()
        c["gen_seeds"].sort()

    strata = Counter(c["stratum"] for c in clusters)
    fam_rows = Counter(r["prompt_family"] for r in recs)
    rows_per_cluster = Counter(c["n_rows"] for c in clusters)
    fams_per_cluster = Counter(len(c["families"]) for c in clusters)
    seeds_per_cluster = Counter(len(c["gen_seeds"]) for c in clusters)

    # Residual dependence: an opening template shared by prompt texts that sit in
    # DIFFERENT clusters is not absorbed by resampling clusters.
    text_of: dict[str, str] = {}
    cluster_of: dict[str, str] = {}
    for c in clusters:
        for pid, txt in c["prompt_texts"].items():
            text_of[pid] = txt
            cluster_of[pid] = c["cluster_id"]
    by_opening: dict[str, set[str]] = defaultdict(set)
    for pid, txt in text_of.items():
        by_opening[opening_phrase(txt)].add(cluster_of[pid])
    cross = {op: sorted(cs) for op, cs in by_opening.items() if len(cs) > 1}

    return {
        "set": name,
        "manifest_content_sha256": canonical_manifest_digest(man),
        "n_rows": len(recs),
        "n_clusters": len(clusters),
        "n_distinct_prompt_texts": len({r["prompt"] for r in recs}),
        "gen_seeds": sorted({int(r["gen_seed"]) for r in recs}),
        "strata_cluster_counts": dict(sorted(strata.items())),
        "family_row_counts": dict(sorted(fam_rows.items())),
        "rows_per_cluster_distribution": dict(sorted(rows_per_cluster.items())),
        "families_per_cluster_distribution": dict(sorted(fams_per_cluster.items())),
        "gen_seeds_per_cluster_distribution": dict(sorted(seeds_per_cluster.items())),
        "every_cluster_carries_both_families": set(fams_per_cluster) == {2},
        "every_cluster_carries_all_gen_seeds":
            set(seeds_per_cluster) == {len({int(r["gen_seed"]) for r in recs})},
        "residual_shared_template_dependence": {
            "definition": (f"prompt texts sharing their first {OPENING_TOKENS} "
                           f"tokens, counted only where the sharing crosses "
                           f"cluster boundaries"),
            "n_cross_cluster_templates": len(cross),
            "n_prompt_texts_in_cross_cluster_templates": sum(
                1 for pid, txt in text_of.items()
                if opening_phrase(txt) in cross),
            "largest_cross_cluster_template": max(
                ((op, len(cs)) for op, cs in cross.items()),
                key=lambda t: t[1], default=(None, 0)),
            "examples": {op: cs[:4] for op, cs in sorted(
                cross.items(), key=lambda kv: -len(kv[1]))[:5]},
            "effect_on_intervals": (
                "NOT ESTIMATED. Resampling clusters absorbs the dependence "
                "WITHIN a scene cluster (its two family wordings and its four "
                "generation seeds). It does not absorb dependence induced by a "
                "template shared ACROSS clusters. Neither the direction nor the "
                "size of that residual effect on interval width is established "
                "by this design, and none is claimed. It is reported as an "
                "un-quantified limitation of template-built prompt sets."),
        },
        "clusters": clusters,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default="results/audit_v1/draft_manifests/dev_manifest_DRAFT.json")
    ap.add_argument("--test", default="results/audit_v1/draft_manifests/test_manifest_DRAFT.json")
    ap.add_argument("--pilot", default="/data/bijaypandey/cuig_pilot/seq_pilot/eval_manifest.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sets = {}
    for name, p in (("dev", args.dev), ("test", args.test), ("pilot", args.pilot)):
        path = Path(p)
        if not path.is_file():
            print(f"  skip {name}: {path} absent", file=sys.stderr)
            continue
        sets[name] = group_one(name, json.loads(path.read_text()))

    payload = {
        "_contract": (
            "The FROZEN bootstrap resampling grouping for the matched-"
            "effectiveness diagnostic. Derived from the actual manifests, not "
            "asserted. Fixed before any data are collected."),
        "resampling_unit": {
            "definition": "the scene cluster (set, category, prompt_index)",
            "carries": (
                "both prompt families (literal and paraphrase) of that scene AND "
                "all four generation seeds, moved together as one unit"),
            "why": (
                "literal_i and paraphrase_i of a category are the same scene "
                "worded two ways, by construction. Drawing them independently "
                "would treat a dependent pair as two independent observations. "
                "Generation seeds are four draws of the same prompt and are "
                "likewise not independent observations of the prompt."),
        },
        "stratification": {
            "stratum": "category",
            "rule": (
                "draw clusters WITH REPLACEMENT within each category stratum, "
                "taking the same number of clusters that stratum actually has, "
                "so every draw preserves the category composition exactly"),
            "literal_vs_paraphrase": (
                "preserved exactly, because every cluster contributes both "
                "family wordings. The literal-only and paraphrase-only endpoints "
                "are computed on the SAME resampled cluster list, restricted to "
                "the rows of that family -- so they stay paired with each other, "
                "with the per-category endpoints, and with both arms."),
        },
        "pairing": (
            "ONE resampled cluster list indexes BOTH arms and every endpoint in "
            "a draw. Contrasts are paired at identical (prompt_id, gen_seed)."),
        "n_draws": N_DRAWS,
        "interval": "95% percentile interval over the draws",
        "rng": {
            "bootstrap_rng_seed": BOOTSTRAP_RNG_SEED,
            "generator": "numpy.random.default_rng(bootstrap_rng_seed)",
            "draw_order": (
                "strata in sorted category order; within a stratum, "
                "rng.integers(0, n_clusters_in_stratum, size=n_clusters_in_stratum) "
                "per draw; draws taken in order 0..9999"),
            "distinct_from": (
                "the annotation packet's stratified sampler, rng_seed=20261009. "
                "The two must never be conflated."),
        },
        "training_seeds": (
            "analysed SEPARATELY and never pooled. Each training seed gets its "
            "own intervals; two seeds support a direction check only."),
        "sets": sets,
    }
    payload["grouping_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "grouping_sha256"},
                   sort_keys=True).encode()).hexdigest()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  grouping identity {payload['grouping_sha256'][:16]}…")
    for name, s in sets.items():
        print(f"  {name:<6} {s['n_rows']:>4} rows in {s['n_clusters']:>3} clusters "
              f"over {len(s['strata_cluster_counts'])} strata; "
              f"both families in every cluster="
              f"{s['every_cluster_carries_both_families']}; "
              f"all gen seeds in every cluster="
              f"{s['every_cluster_carries_all_gen_seeds']}; "
              f"cross-cluster templates="
              f"{s['residual_shared_template_dependence']['n_cross_cluster_templates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
