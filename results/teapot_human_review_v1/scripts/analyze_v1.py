#!/usr/bin/env python3
"""Combined Task-A / Task-B descriptive analysis (analysis_v1).

Applies the frozen rules in analysis_v1/ANALYSIS_CONTRACT.md to exactly four
allowlisted exports, joins them case-wise through the UNCHANGED private
mappings, and writes the public result artifacts.

What stays private and is never written to a public artifact by this script:
reviewer aliases, the alias -> R1/R2 linkage, original filenames, free-text
`reason` and `other_exposure_note` values, `exported_iso` timestamps, source
image filenames/hashes, case IDs, and any neutral-ID -> condition association.

Public outputs carry neutral reviewer IDs R1/R2, the public Task-A item IDs,
the public Task-B pair IDs, categorical votes exactly as given, and counts.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_rules import (COMPARISON_VALUES, DEFINITE_PRESENCE,
                            PRESENCE_VALUES, RENDERING_VALUES, UNRESOLVED,
                            comparison_distribution,
                            definite_presence_coverage, demonstrated_removal,
                            derive_presence, derive_rendering,
                            presence_agreement, presence_table,
                            rendering_distribution)

ROOT = "/data/bijaypandey/cuig_pilot/teapot_human_review_v1"
PRIV = os.path.join(ROOT, "private")
RESP = os.path.join(PRIV, "responses_v1")
OUTDIR = os.path.join(ROOT, "analysis_v1")
TABLES = os.path.join(OUTDIR, "tables")

ANALYSIS_ID = "THR-ANALYSIS-V1"
CODE_FILES = ("analysis_rules.py", "analyze_v1.py",
              "test_analysis_synthetic.py", "verify_mappings_for_analysis.py")

# Closed allowlist of the four inputs, keyed by SHA256 -> expected byte size.
#
# Keyed by HASH, not by filename: the original export filenames embed the
# reviewer aliases, so putting them in published code would disclose the roster
# -- the mistake recorded in AMENDMENT_03 section 7b. These digests are already
# public (they are published in summary.json for integrity), the filenames are
# not. Files are discovered in the private response directory and admitted only
# if their hashes are exactly this set.
EXPECTED_INPUTS = {
    "630cb4b522bd3a93740b39231382503e46e85323264ede9a5e4d939936052758": 4660,
    "d57fb70873b7909c6d8964e8314dcdb2b56168c99279c5dec78e7e58d1e34b09": 4682,
    "3dc00a2eb0731a0c439fe50c3cc0a6013b8b8ef2eee69e17386bbc29cb04faa6": 2324,
    "2ddae4d6fb742c0d1d147d0b88e33637cb0a3f110680c7d72305206b1a3f1fcf": 2326,
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def input_hashes():
    """Discovered filename -> sha256 for the private response directory.

    Returned in-memory only; filenames never reach a public artifact.
    """
    return {fn: sha256_file(os.path.join(RESP, fn))
            for fn in sorted(os.listdir(RESP)) if fn.endswith(".json")}


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write public artifacts (default: dry run)")
    args = ap.parse_args()

    errors = []
    pre = input_hashes()
    if set(pre.values()) != set(EXPECTED_INPUTS):
        errors.append("the private response directory does not hold exactly "
                      "the four allowlisted exports (matched %d of %d by "
                      "hash)" % (len(set(pre.values()) & set(EXPECTED_INPUTS)),
                                 len(EXPECTED_INPUTS)))
    if len(pre) != len(EXPECTED_INPUTS):
        errors.append("found %d JSON files, expected exactly %d"
                      % (len(pre), len(EXPECTED_INPUTS)))
    for fn, h in pre.items():
        want = EXPECTED_INPUTS.get(h)
        if want is not None and os.path.getsize(os.path.join(RESP, fn)) != want:
            errors.append("byte size does not match the allowlisted size for "
                          "the input with hash " + h[:16])
    if errors:
        print("INPUT_BLOCKER")
        for e in errors:
            print("  " + e)
        return 2

    # ---- load the four exports ------------------------------------------- #
    exports = {fn: json.load(open(os.path.join(RESP, fn)))
               for fn in sorted(pre)}
    A = [d for d in exports.values() if d["stage"] == "A"]
    B = [d for d in exports.values() if d["stage"] == "B"]
    assert len(A) == 2 and len(B) == 2

    # ---- R1 / R2: answer-independent rule (contract §1) ------------------- #
    # Order by Task-A exported_iso ascending; earlier = R1.
    A_sorted = sorted(A, key=lambda d: d["exported_iso"])
    alias_to_rid = {d["reviewer_alias"]: "R%d" % (i + 1)
                    for i, d in enumerate(A_sorted)}
    RIDS = ["R1", "R2"]

    a_by_rid = {alias_to_rid[d["reviewer_alias"]]: d for d in A}
    b_by_rid = {alias_to_rid[d["reviewer_alias"]]: d for d in B}
    if sorted(a_by_rid) != RIDS or sorted(b_by_rid) != RIDS:
        print("JOIN_BLOCKER: Task-A and Task-B aliases do not form the same "
              "two-reviewer cohort")
        return 2

    # ---- mappings --------------------------------------------------------- #
    mp = json.load(open(os.path.join(PRIV, "id_mapping_private.json")))
    a_map, b_map = mp["taskA_mapping"], mp["taskB_mapping"]

    # ---- Task-A raw rows --------------------------------------------------- #
    a_rows = {}
    for rid, d in a_by_rid.items():
        seen = set()
        for r in d["responses"]:
            iid = r["item_id"]
            if iid in seen:
                errors.append("duplicate Task-A item for %s: %s" % (rid, iid))
            seen.add(iid)
            if iid not in a_map:
                errors.append("unknown Task-A item id: " + iid)
            if r["target"] not in PRESENCE_VALUES:
                errors.append("illegal target value for %s %s" % (rid, iid))
            if r["rendering"] not in RENDERING_VALUES:
                errors.append("illegal rendering value for %s %s" % (rid, iid))
            a_rows.setdefault(iid, {})[rid] = r
        if len(seen) != 40:
            errors.append("%s Task-A has %d rows, expected 40" % (rid, len(seen)))

    a_items = sorted(a_map)
    for iid in a_items:
        if sorted(a_rows.get(iid, {})) != RIDS:
            errors.append("Task-A item %s lacks both reviewers" % iid)

    # ---- Task-B raw rows --------------------------------------------------- #
    b_rows = {}
    for rid, d in b_by_rid.items():
        seen = set()
        for r in d["responses"]:
            pid = r["pair_id"]
            if pid in seen:
                errors.append("duplicate Task-B pair for %s: %s" % (rid, pid))
            seen.add(pid)
            if pid not in b_map:
                errors.append("unknown Task-B pair id: " + pid)
            if r["comparison"] not in COMPARISON_VALUES:
                errors.append("illegal comparison value for %s %s" % (rid, pid))
            b_rows.setdefault(pid, {})[rid] = r
        if len(seen) != 20:
            errors.append("%s Task-B has %d rows, expected 20" % (rid, len(seen)))

    b_pairs = sorted(b_map)
    for pid in b_pairs:
        if sorted(b_rows.get(pid, {})) != RIDS:
            errors.append("Task-B pair %s lacks both reviewers" % pid)

    if errors:
        print("VALIDATION_BLOCKER")
        for e in errors:
            print("  " + e)
        return 2

    # ---- Task-A per-item derivation --------------------------------------- #
    a_item = {}
    for iid in a_items:
        tv = [a_rows[iid][r]["target"] for r in RIDS]
        rv = [a_rows[iid][r]["rendering"] for r in RIDS]
        a_item[iid] = {
            "item_id": iid,
            "R1_target": tv[0], "R2_target": tv[1],
            "R1_rendering": rv[0], "R2_rendering": rv[1],
            "derived_target": derive_presence(tv),
            "derived_rendering": derive_rendering(rv),
            "definite_target_votes": definite_presence_coverage(tv),
            "target_agreement": presence_agreement(tv),
            "rendering_agreement": ("agreement" if rv[0] == rv[1]
                                    else "disagreement"),
            "condition": a_map[iid]["condition"],       # private, not published
        }

    # ---- Task-A agreement summaries --------------------------------------- #
    def agg_presence(items):
        out = {"n": len(items), "definite_consensus": 0,
               "uncertain_agreement": 0, "disagreement": 0}
        for it in items:
            out[it["target_agreement"]] += 1
        out["raw_agreement_any"] = (out["definite_consensus"]
                                    + out["uncertain_agreement"])
        return out

    def agg_rendering(items):
        out = {"n": len(items), "agreement": 0, "disagreement": 0}
        for it in items:
            out[it["rendering_agreement"]] += 1
        return out

    all_items = [a_item[i] for i in a_items]
    m0_items = [it for it in all_items if it["condition"] == "m0"]
    ed_items = [it for it in all_items if it["condition"] == "edited"]

    presence_agree = {
        "all_40_images": agg_presence(all_items),
        "m0_images": agg_presence(m0_items),
        "edited_images": agg_presence(ed_items),
    }
    rendering_agree = {
        "all_40_images": agg_rendering(all_items),
        "m0_images": agg_rendering(m0_items),
        "edited_images": agg_rendering(ed_items),
    }

    coverage = {"0": 0, "1": 0, "2": 0}
    for it in all_items:
        coverage[str(it["definite_target_votes"])] += 1

    derived_presence_counts = {"present": 0, "absent": 0, UNRESOLVED: 0}
    for it in all_items:
        derived_presence_counts[it["derived_target"]] += 1

    raw_presence_per_reviewer = {}
    raw_rendering_per_reviewer = {}
    for rid in RIDS:
        raw_presence_per_reviewer[rid] = {
            k: sum(1 for iid in a_items if a_rows[iid][rid]["target"] == k)
            for k in PRESENCE_VALUES}
        raw_rendering_per_reviewer[rid] = {
            k: sum(1 for iid in a_items if a_rows[iid][rid]["rendering"] == k)
            for k in RENDERING_VALUES}

    # ---- case-level join --------------------------------------------------- #
    a_by_case = {}
    for iid in a_items:
        v = a_map[iid]
        a_by_case.setdefault(v["case_id"], {})[v["condition"]] = iid

    cases = []
    for pid in b_pairs:
        bm = b_map[pid]
        cid = bm["case_id"]
        pair = a_by_case[cid]
        m0_iid, ed_iid = pair["m0"], pair["edited"]
        cmp_votes = [b_rows[pid][r]["comparison"] for r in RIDS]
        cases.append({
            "pair_id": pid,
            "case_class": bm.get("class"),
            "case_category": bm.get("category"),
            "m0_derived_target": a_item[m0_iid]["derived_target"],
            "edited_derived_target": a_item[ed_iid]["derived_target"],
            "m0_derived_rendering": a_item[m0_iid]["derived_rendering"],
            "edited_derived_rendering": a_item[ed_iid]["derived_rendering"],
            "R1_comparison": cmp_votes[0],
            "R2_comparison": cmp_votes[1],
            "comparison_agreement": ("agreement" if cmp_votes[0] == cmp_votes[1]
                                     else "disagreement"),
            "_m0_item": m0_iid, "_edited_item": ed_iid,   # private
        })

    if len(cases) != 20:
        errors.append("case ledger has %d rows, expected 20" % len(cases))
    used = [c["_m0_item"] for c in cases] + [c["_edited_item"] for c in cases]
    if len(set(used)) != 40:
        errors.append("case join is not one-to-one over the 40 Task-A images")

    # ---- 3x3 table and the conditional removal count ---------------------- #
    ptab = presence_table(cases)
    if ptab["total"] != 20:
        errors.append("3x3 presence table totals %d, expected 20"
                      % ptab["total"])
    removal = demonstrated_removal(cases)

    # ---- target-absence rendering, reported twice and separately ---------- #
    edited_absent = [c for c in cases if c["edited_derived_target"] == "absent"]
    qualified = [c for c in edited_absent
                 if c["m0_derived_target"] == "present"]
    absence_rendering = {
        "all_edited_absent_cases": {
            "n": len(edited_absent),
            "edited_image_rendering": rendering_distribution(
                [c["edited_derived_rendering"] for c in edited_absent]),
        },
        "qualified_m0_present_to_edited_absent": {
            "n": len(qualified),
            "edited_image_rendering": rendering_distribution(
                [c["edited_derived_rendering"] for c in qualified]),
        },
    }

    # ---- Task B ------------------------------------------------------------ #
    b_dist = {}
    for rid in RIDS:
        vals = [b_rows[pid][rid]["comparison"] for pid in b_pairs]
        d = comparison_distribution(vals)
        if sum(d.values()) != 20:
            errors.append("%s Task-B distribution sums to %d, expected 20"
                          % (rid, sum(d.values())))
        b_dist[rid] = d
    b_exact_agreement = sum(1 for c in cases
                            if c["comparison_agreement"] == "agreement")

    # ---- descriptive strata ------------------------------------------------ #
    def stratum_rows(key):
        out = {}
        for c in cases:
            k = c[key]
            blank = {"n_cases": 0, "m0_present": 0, "edited_absent": 0,
                     "removal_numerator": 0, "removal_denominator": 0}
            blank.update({"B_" + v: 0 for v in COMPARISON_VALUES})
            s = out.setdefault(k, blank)
            s["n_cases"] += 1
            if c["m0_derived_target"] == "present":
                s["m0_present"] += 1
                s["removal_denominator"] += 1
                if c["edited_derived_target"] == "absent":
                    s["removal_numerator"] += 1
            if c["edited_derived_target"] == "absent":
                s["edited_absent"] += 1
            for rid in RIDS:
                s["B_" + c[rid + "_comparison"]] += 1
        return out

    strata = {"by_class": stratum_rows("case_class"),
              "by_category": stratum_rows("case_category")}

    # ---- exposure / participation ------------------------------------------ #
    participation = {}
    for rid in RIDS:
        a, b = a_by_rid[rid], b_by_rid[rid]
        participation[rid] = {
            "taskA_export_kind": a["export_kind"],
            "taskA_rows": len(a["responses"]),
            "taskA_expected": a["item_count_expected"],
            "taskA_complete": len(a["responses"]) == a["item_count_expected"],
            "taskA_schema_id": a["schema_id"],
            "taskA_app_version": a["app_version"],
            "taskA_order_variant": a["order_variant"],
            "taskB_export_kind": b["export_kind"],
            "taskB_rows": len(b["responses"]),
            "taskB_expected": b["item_count_expected"],
            "taskB_complete": len(b["responses"]) == b["item_count_expected"],
            "taskB_schema_id": b["schema_id"],
            "taskB_app_version": b["app_version"],
            "taskB_app_build": b.get("app_build"),
            "taskB_order_variant": b["order_variant"],
            "exposure_declared_A": a["exposure"]["declared"],
            "exposure_declared_B": b["exposure"]["declared"],
            "prior_access_to_source_pairings_before_A":
                a["exposure"]["prior_access_to_source_pairings"],
            "prior_access_to_condition_labels_before_A":
                a["exposure"]["prior_access_to_condition_labels"],
            "image_specific_coaching_before_A":
                a["exposure"]["image_specific_coaching_before_task_a"],
            "taskB_declares_prior_condition_access_before_A":
                b["exposure"]["prior_access_to_condition_labels"],
            "taskB_declares_coaching_before_A":
                b["exposure"]["image_specific_coaching_before_task_a"],
            "task_a_final_exported_before_task_b":
                b["exposure"]["task_a_final_exported_before_task_b"],
            "reference_identity_disclosed_in_this_stage":
                b["exposure"]["reference_identity_disclosed_in_this_stage"],
            "free_text_reason_fields_present_withheld": sum(
                1 for r in list(a["responses"]) + list(b["responses"])
                if (r.get("reason") or "").strip()),
            "other_exposure_note_present_withheld": bool(
                (a["exposure"].get("other_exposure_note") or "").strip()
                or (b["exposure"].get("other_exposure_note") or "").strip()),
        }

    if errors:
        print("ARITHMETIC_BLOCKER")
        for e in errors:
            print("  " + e)
        return 2

    # Neutral labels for the four inputs, so no public artifact carries an
    # original filename (they embed the reviewer aliases).
    neutral_key = {}
    for rid in RIDS:
        for stage, d in (("A", a_by_rid[rid]), ("B", b_by_rid[rid])):
            for fn, ex in exports.items():
                if ex is d:
                    neutral_key[fn] = "%s_task%s_final" % (rid, stage)
    if sorted(neutral_key) != sorted(pre):
        print("JOIN_BLOCKER: could not label all four inputs neutrally")
        return 2

    post = input_hashes()
    inputs_unchanged = (post == pre)
    if not inputs_unchanged:
        print("INPUT_BLOCKER: input hashes changed during analysis")
        return 2

    # ---- assemble the public summary --------------------------------------- #
    code_hashes = {f: sha256_file(os.path.join(ROOT, "scripts", f))
                   for f in CODE_FILES}

    summary = {
        "analysis_id": ANALYSIS_ID,
        "package_a": "THR-V1-A", "package_b": "THR-V1-B",
        "protocol_id": "TEAPOT-HR-PROTO-1.0.0",
        "schema_id": "THR-EXPORT-SCHEMA-1.1.0",
        "app_version": "1.1.0",
        "validator_revision": "validator-r3",
        "scope": ("Descriptive analysis of existing human judgments. No new "
                  "experiment, label, model run or image generation."),
        "cohort": {
            "planned": 3, "actual": 2,
            "frozen_minimum": 2,
            "meets_frozen_minimum": True,
            "change_recorded": ("Amendment 03 §1, at Task-A closure -- before "
                                "any outcome analysis and before Task B was "
                                "released"),
            "identity_and_independence": (
                "Coordinator-reported. Matching A/B aliases and the user's "
                "handoff link each Task-B return to the same person who filed "
                "the corresponding Task-A return. Software cannot "
                "independently prove identity or independence."),
            "reviewer_ids": RIDS,
            "id_assignment_rule": ("Task-A exported_iso ascending; earlier = "
                                   "R1. Answer-independent. Same linkage used "
                                   "for both stages."),
        },
        "judgment_counts": {
            "cases": 20, "taskA_images": 40,
            "taskA_judgments": 80, "taskB_judgments": 40,
            "reviewers": 2,
            "note": ("20 paired cases, 80 Task-A judgments and 40 Task-B "
                     "judgments from two people -- not independent replicated "
                     "trials."),
        },
        "participation": participation,
        "taskA": {
            "raw_presence_per_reviewer": raw_presence_per_reviewer,
            "raw_rendering_per_reviewer": raw_rendering_per_reviewer,
            "presence_agreement": presence_agree,
            "rendering_agreement": rendering_agree,
            "definite_target_vote_coverage_per_image": coverage,
            "derived_presence_counts_over_40_images": derived_presence_counts,
            "derived_rendering_counts_over_40_images": rendering_distribution(
                [it["derived_rendering"] for it in all_items]),
        },
        "presence_table_m0_by_edited": ptab,
        "demonstrated_removal": dict(
            removal,
            definition=("numerator = derived m0 present AND derived edited "
                        "absent; denominator = ALL derived m0 present cases, "
                        "including those whose edited status is unresolved. "
                        "absent->absent is not demonstrated removal."),
        ),
        "target_absence_rendering": absence_rendering,
        "taskB": {
            "distributions": b_dist,
            "exact_per_pair_agreement": {
                "agreeing_pairs": b_exact_agreement, "denominator": 20,
                "note": ("Descriptive only. No Task-B consensus rule was "
                         "frozen and none is invented; disagreements are not "
                         "forced into a majority label."),
            },
        },
        "descriptive_strata": strata,
        "statistics_excluded": [
            "hypothesis tests", "p-values", "confidence intervals",
            "bootstrap", "kappa / chance-corrected agreement",
            "significance claims", "new pass/fail cutoff", "method ranking"],
        "statistics_note": ("Descriptive scope choice for 20 cases judged by "
                            "two people. NOT a claim that such statistics are "
                            "universally impossible or improper."),
        "withheld_from_public_artifacts": [
            "reviewer aliases", "alias -> R1/R2 linkage",
            "original export filenames", "free-text `reason` values",
            "`other_exposure_note` values", "`exported_iso` timestamps",
            "source image filenames and hashes", "case IDs",
            "neutral-ID -> condition associations",
            "randomization salt and commitment keys",
            "private input manifest and backup contents"],
        "withheld_note": ("These fields exist in the originals and were NOT "
                          "deleted from them. They are withheld from public "
                          "artifacts only."),
        "integrity": {
            # Keyed by NEUTRAL label. The original export filenames embed the
            # reviewer aliases, so the filename -> hash mapping stays in the
            # private input manifest and never appears in a public artifact.
            "input_sha256_before": {neutral_key[fn]: h
                                    for fn, h in sorted(pre.items())},
            "input_sha256_after": {neutral_key[fn]: h
                                   for fn, h in sorted(post.items())},
            "input_key_note": ("Keyed by neutral reviewer ID and stage. "
                               "Original filenames embed aliases and are "
                               "withheld; the filename -> hash mapping is in "
                               "the private input manifest."),
            "inputs_unchanged_by_analysis": inputs_unchanged,
            "code_sha256": code_hashes,
        },
        "generated_iso": datetime.datetime.now().astimezone().isoformat(),
    }

    if not args.write:
        print(json.dumps(summary, indent=1, sort_keys=True))
        print("\nDRY RUN -- nothing written. Re-run with --write.")
        return 0

    # ---- write public artifacts -------------------------------------------- #
    os.makedirs(TABLES, exist_ok=True)

    write_csv(os.path.join(TABLES, "taskA_responses.csv"),
              ["item_id", "R1_target", "R1_rendering", "R2_target",
               "R2_rendering", "derived_target", "derived_rendering",
               "definite_target_votes", "target_agreement",
               "rendering_agreement"],
              [[it["item_id"], it["R1_target"], it["R1_rendering"],
                it["R2_target"], it["R2_rendering"], it["derived_target"],
                it["derived_rendering"], it["definite_target_votes"],
                it["target_agreement"], it["rendering_agreement"]]
               for it in all_items])

    write_csv(os.path.join(TABLES, "taskB_responses.csv"),
              ["pair_id", "R1_comparison", "R2_comparison",
               "comparison_agreement"],
              [[c["pair_id"], c["R1_comparison"], c["R2_comparison"],
                c["comparison_agreement"]] for c in cases])

    write_csv(os.path.join(TABLES, "taskB_distributions.csv"),
              ["reviewer_id"] + list(COMPARISON_VALUES) + ["total"],
              [[rid] + [b_dist[rid][v] for v in COMPARISON_VALUES]
               + [sum(b_dist[rid].values())] for rid in RIDS])

    write_csv(os.path.join(TABLES, "paired_case_ledger.csv"),
              ["pair_id", "case_class", "case_category",
               "m0_derived_target", "edited_derived_target",
               "m0_derived_rendering", "edited_derived_rendering",
               "counts_in_removal_denominator", "counts_in_removal_numerator",
               "R1_comparison", "R2_comparison", "comparison_agreement"],
              [[c["pair_id"], c["case_class"], c["case_category"],
                c["m0_derived_target"], c["edited_derived_target"],
                c["m0_derived_rendering"], c["edited_derived_rendering"],
                "yes" if c["m0_derived_target"] == "present" else "no",
                "yes" if (c["m0_derived_target"] == "present"
                          and c["edited_derived_target"] == "absent") else "no",
                c["R1_comparison"], c["R2_comparison"],
                c["comparison_agreement"]] for c in cases])

    axes = ptab["axes"]
    write_csv(os.path.join(TABLES, "presence_table_m0_by_edited.csv"),
              ["m0_derived \\ edited_derived"] + axes + ["row_total"],
              [[m] + [ptab["table"][m][e] for e in axes]
               + [sum(ptab["table"][m].values())] for m in axes]
              + [["column_total"]
                 + [sum(ptab["table"][m][e] for m in axes) for e in axes]
                 + [ptab["total"]]])

    rk = list(RENDERING_VALUES) + [UNRESOLVED]
    write_csv(os.path.join(TABLES, "target_absence_rendering.csv"),
              ["subset", "n_cases"] + rk,
              [[k, v["n"]] + [v["edited_image_rendering"][x] for x in rk]
               for k, v in absence_rendering.items()])

    strat_rows = []
    for kind, groups in strata.items():
        for name, s in sorted(groups.items()):
            strat_rows.append([
                kind, name, s["n_cases"], s["m0_present"], s["edited_absent"],
                s["removal_numerator"], s["removal_denominator"]]
                + [s["B_" + v] for v in COMPARISON_VALUES])
    write_csv(os.path.join(TABLES, "descriptive_strata.csv"),
              ["stratum_kind", "stratum", "n_cases", "m0_present",
               "edited_absent", "removal_numerator", "removal_denominator"]
              + ["B_" + v for v in COMPARISON_VALUES], strat_rows)

    with open(os.path.join(OUTDIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(json.dumps({k: summary[k] for k in (
        "presence_table_m0_by_edited", "demonstrated_removal",
        "target_absence_rendering", "taskB")}, indent=1, sort_keys=True))
    print("\nwrote public artifacts under " + OUTDIR)
    print("ANALYSIS_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
