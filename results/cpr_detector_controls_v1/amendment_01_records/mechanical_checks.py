#!/usr/bin/env python3
"""Amendment 01 mechanical checks. READ-ONLY: parses records stored in Git at the
reviewed HEAD, does arithmetic, and hashes. Imports only stdlib. Loads no model,
runs no detector, scores nothing, assigns no label. Writes mechanical_checks.json."""
import hashlib, json, subprocess, sys

REVIEWED = "9940e9403452e2b382c108c2d4d50bd3d054d707"
D = "results/cpr_detector_controls_v1/"


def show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{D}{path}"], check=True,
                          capture_output=True).stdout


def js(path):
    return json.loads(show(REVIEWED, path))


def main(out):
    R = {"reviewed_head": REVIEWED}
    q = js("artifacts/qualification.json")
    MIN = q["min_hit"]
    qual = {}
    for tag in ("rtdetr_r50vd", "detr_resnet50_no_timm"):
        raw = js(f"artifacts/qualification_raw_{tag}.json")
        per, ids, nrec, nmiss = {}, set(), 0, 0
        for cls, rows in sorted(raw.items()):
            pos = [r for r in rows if r["group"] == "positives"]
            neg = [r for r in rows if r["group"] == "negatives"]
            nrec += len(rows); nmiss += sum(bool(r["missing_measurement"]) for r in rows)
            ids |= {r["image_id"] for r in rows}
            det = sum(bool(r["detected"]) for r in pos)
            box = sum(bool(r["box_match_iou50"]) for r in pos)
            clean = sum(not r["detected"] for r in neg)
            rec = q["detectors"][tag]["per_class"][cls]
            assert (det, box, clean) == (rec["positives_detected"],
                                         rec["positives_box_matched_iou50"],
                                         rec["negatives_clean"]), (tag, cls)
            per[cls] = {"n_pos": len(pos), "n_neg": len(neg),
                        "positives_image_level_detected": det,
                        "positives_box_matched_iou50": box,
                        "negatives_clean": clean,
                        "published_flag_passed": rec["passed"],
                        "published_predicate": "positives_image_level_detected>=16 and negatives_clean>=16",
                        "frozen_gate_passed": box >= MIN and clean >= MIN,
                        "frozen_gate": "positives_box_matched_iou50>=16 and negatives_clean>=16"}
        qual[tag] = {"per_class": per, "records": nrec, "unique_photographs": len(ids),
                     "missing_measurements": nmiss,
                     "published_verdict": q["detectors"][tag]["verdict"],
                     "frozen_gate_all_classes": all(v["frozen_gate_passed"] for v in per.values()),
                     "frozen_gate_failing_classes": sorted(c for c, v in per.items()
                                                           if not v["frozen_gate_passed"])}
        keys = set()
        for rows in raw.values():
            for r in rows:
                keys |= set(r)
        qual[tag]["raw_record_fields"] = sorted(keys)
    R["qualification_recount_from_published_records"] = qual
    R["qualification_note"] = ("Counts re-aggregated from the published per-image records and "
                               "checked equal to qualification.json. No inference was re-run; "
                               "the records hold no predicted boxes/labels, so IoU matching "
                               "cannot be independently reconstructed from them.")

    L = js("artifacts/allocation_ledger.json")
    jobs = [{"job": c["job"], "status": c["status"], "charged_gpu_s": c["charged_gpu_s"],
             "reserved_gpu_s": c["reserved_gpu_s"], "cpu_seconds": c["cpu_seconds"],
             "below_reservation": c["charged_gpu_s"] < c["reserved_gpu_s"]} for c in L["closed"]]
    g = sum(j["charged_gpu_s"] for j in jobs)
    c = sum(j["cpu_seconds"] for j in jobs)
    assert abs(g - L["charged_gpu_s"]["entry"]) < 1e-9 and abs(c - L["charged_cpu_s"]) < 1e-9
    R["accounting"] = {
        "supervised_jobs": jobs, "gpu_device_seconds_total": g,
        "gpu_device_hours_total": g / 3600, "gpu_devices_used": sorted({d for x in L["closed"] for d in x["devices"]}),
        "cpu_seconds_supervised_children_subtotal": c, "cpu_core_hours_supervised_children_subtotal": c / 3600,
        "open_reservations": L["open_reservations"], "unreconciled": L["unreconciled"],
        "generated_images": L["generated_images"], "coco_photographs": L["coco_photographs"],
        "retries": L["retries"], "ledger_created_utc": L["created_utc"],
        "entry_subcap_hours": 4.0, "eval_reserve_hours": 8.0, "eval_bucket_charged_gpu_s": L["charged_gpu_s"]["eval"],
        "gpu_ceiling_hours": 24.0}
    # a note: the 8.16 projection's "already_used" counted the then-open 5400 s reservation
    bg = js("artifacts/bringup_gpu.json")["budget_projection_fits"]
    R["projection"] = {k: bg[k] for k in ("already_used_gpu_h", "seconds_per_processed_timestep",
                                            "train_gpu_h", "generate_gpu_h", "detector_gpu_h",
                                            "load_gpu_h", "projected_gpu_h", "usable_non_eval_gpu_h")}
    R["projection"]["already_used_equals_two_settled_jobs_plus_open_5400s_reservation"] = (
        abs(bg["already_used_gpu_h"] - (jobs[0]["charged_gpu_s"] + jobs[1]["charged_gpu_s"] + 5400.0) / 3600) < 5e-4)

    cg = js("artifacts/cpu_gate.json")["_summary"]
    R["cpu_gate_final"] = {k: cg[k] for k in ("total", "passed", "failed", "not_applicable",
                                              "not_applicable_cases", "verdict")}
    cgrows = {k: v for k, v in js("artifacts/cpu_gate.json").items() if not k.startswith("_")}
    R["cpu_gate_final"]["rows"] = len(cgrows)
    R["cpu_gate_final"]["exact_zero_boundary_fixtures"] = sorted(
        k for k, v in cgrows.items() if str(v.get("status", "")).startswith("EXACT_BOUNDARY"))

    gp = js("artifacts/bringup_gpu.json")
    R["gpu_gate"] = {"failures": gp["_summary"]["failures"], "elapsed_s": gp["_summary"]["elapsed_s"],
                     "hash_off": gp["dispatch_off_on_agreement_native_arm"]["hash_dispatch_off"],
                     "hash_on": gp["dispatch_off_on_agreement_native_arm"]["hash_dispatch_on"],
                     "hash_cpr": gp["candidate_arm_changes_result"]["hash_cpr"],
                     "cpr_probe_ran_after_failed_native_agreement_check": "hash_cpr" in gp["candidate_arm_changes_result"],
                     "probe_processed_timesteps_each": 22, "probes": ["native OFF", "native ON", "CPR ON"]}
    return R, show


def chronology(R):
    log = subprocess.run(["git", "log", "--format=%H|%aI|%s", f"{REVIEWED}",
                          "-8", "--", D.rstrip("/")], check=True, capture_output=True, text=True).stdout
    R["commits_touching_study"] = [dict(zip(("sha", "author_date_iso", "subject"), l.split("|", 2)))
                                   for l in log.splitlines()]
    first = {}
    for name in ("3fba0ff", "67d8b74", "00bc8ae"):
        files = subprocess.run(["git", "show", "--name-only", "--format=", name], check=True,
                               capture_output=True, text=True).stdout.split()
        first[name] = sorted(f.replace(D, "") for f in files if f.startswith(D))
    R["files_added_at_each_pre_release_commit"] = first
    jr = [json.loads(l) for l in show(REVIEWED, "artifacts/allocation_journal.jsonl").decode().splitlines()]
    R["journal_machine_timestamps"] = [{"ts_utc": e["ts_utc"], "event": e["event"],
                                        "job": e.get("job")} for e in jr
                                       if e["event"] in ("retry", "reserve", "settle", "ENTRY_GATE_UNRESOLVED")]
    # manifest at the prospective freeze vs reviewed head, and presence of the dispatch-agreement criterion
    m0 = show("3fba0ff", "RUN_MANIFEST.md").decode()
    R["freeze_3fba0ff_manifest"] = {
        "identical_to_reviewed_head": show("3fba0ff", "RUN_MANIFEST.md") == show(REVIEWED, "RUN_MANIFEST.md"),
        "mentions_exact_hash_equality": ("exact" in m0.lower() and "hash" in m0.lower() and "adapter state hash" in m0.lower()),
        "mentions_dispatch_agreement_tolerance": ("dispatch" in m0.lower() and "tolerance" in m0.lower())}


if __name__ == "__main__":
    R, _ = main(sys.argv[1] if len(sys.argv) > 1 else "mechanical_checks.json")
    chronology(R)
    out = sys.argv[1] if len(sys.argv) > 1 else "mechanical_checks.json"
    json.dump(R, open(out, "w"), indent=2, sort_keys=True)
    print("wrote", out, hashlib.sha256(open(out, "rb").read()).hexdigest())
