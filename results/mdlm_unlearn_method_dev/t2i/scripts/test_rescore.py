"""CPU-ONLY tests for the corrected FINAL re-scoring pipeline (defect D-013).

    python scripts/test_rescore.py

No CUDA, no diffusers pipeline, no CLIP weights, no checkpoint reads of GBs. Each test targets one
required property and is written to FAIL on the pre-repair behaviour.

Covers:
  T1  overwrite refusal -- run directory, protected historical paths, gen_eval_final pre-flight
  T2  atomicity -- os.replace semantics, no partial file, no temp litter on failure
  T3  run-id uniqueness
  T4  incomplete / partial / complete status derivation
  T5  missing-checkpoint handling -- explicit record, never a silent skip
  T6  paired-scoring bookkeeping -- both sets score every image; all twelve scores retained
  T7  provenance completeness
  T8  label-set identity mapping -- EXACTLY NINE differ, named

These test IMPLEMENTATION. Passing says the pipeline records and refuses what it claims to. It says
nothing about the SIZE OR DIRECTION of the label-text effect: that is what the GPU run measures.
"""
import importlib.util as iu
import json
import os
import subprocess
import sys
import tempfile

P = "/data/bijaypandey/t2i_unlearn"
SCRIPTS = os.path.join(P, "scripts")
PASSED, FAILED = [], []


def mod(name):
    spec = iu.spec_from_file_location(f"_tr_{name}", os.path.join(SCRIPTS, f"{name}.py"))
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check(name, ok, detail):
    (PASSED if ok else FAILED).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


RS = mod("rescore_final_v2")
CS = mod("clip_scorer")


# ------------------------------------------------------------------ T1 overwrite refusal
def t1_overwrite_refusal():
    with tempfile.TemporaryDirectory() as td:
        # (a) an existing run directory is refused, not reused. The guard is live on directory
        # creation now, so the sandbox root must be declared to it first.
        orig_root = RS.RESCORE_ROOT
        RS.RESCORE_ROOT = td
        try:
            rid = RS.new_run_id("t1")
            d = RS.make_run_dir(rid, root=td)
            ok_a = os.path.isdir(os.path.join(d, "images"))
            try:
                RS.make_run_dir(rid, root=td)
                ok_b, why = False, "second make_run_dir SUCCEEDED -- would reuse/overwrite"
            except RuntimeError as e:
                ok_b, why = "REFUSING to reuse" in str(e), f"refused: {str(e)[:60]}..."
        finally:
            RS.RESCORE_ROOT = orig_root
        check("T1a_run_dir_write_once", ok_a and ok_b, why)

    # (b) the path guard rejects every protected historical path
    bad = [os.path.join(P, "outputs", "final_utrap_church_l0.15k8.json"),
           os.path.join(P, "outputs", "final_BASE_sd14_anchor.json"),
           os.path.join(P, "outputs", "gen_anything.json"),
           os.path.join(P, "adapters", "utrap_church_l0.15k8", "unet.pt"),
           os.path.join(P, "attacked", "x", "unet.pt"),
           os.path.join(P, "outputs", "somewhere_else.json")]
    refused = []
    for b in bad:
        try:
            RS._guard_path(b)
            refused.append((b, False))
        except RuntimeError:
            refused.append((b, True))
    ok = all(r for _, r in refused)
    check("T1b_guard_refuses_historical_paths", ok,
          f"{sum(1 for _, r in refused if r)}/{len(bad)} protected paths refused"
          + ("" if ok else f"; NOT refused: {[b for b, r in refused if not r]}"))

    # (c) atomic_write_json refuses too (the guard is on the write path, not only advisory)
    try:
        RS.atomic_write_json(os.path.join(P, "outputs", "final_BASE_sd14_anchor.json"), {"x": 1})
        ok, why = False, "atomic_write_json WROTE to a historical record path"
    except RuntimeError as e:
        ok, why = "REFUSING" in str(e), f"refused: {str(e)[:70]}..."
    # and the real record is untouched
    hp = os.path.join(P, "outputs", "final_BASE_sd14_anchor.json")
    intact = os.path.exists(hp) and json.load(open(hp)).get("target_hits") == 13
    check("T1c_atomic_write_refuses_and_leaves_history_intact", ok and intact,
          f"{why} historical record intact={intact}")

    # (d) gen_eval_final.py refuses to overwrite an existing record, BEFORE importing diffusers
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "gen_eval_final.py"),
                        "--only", "utrap_church_l0.15k8"],
                       capture_output=True, text=True, timeout=300,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    out = r.stdout + r.stderr
    ok = r.returncode != 0 and "REFUSING to run" in out and "rescore_final_v2.py" in out
    check("T1d_gen_eval_final_refuses_overwrite", ok,
          f"exit={r.returncode}, refusal message present={'REFUSING to run' in out}, "
          f"points at rescore_final_v2={'rescore_final_v2.py' in out}")

    # (e) the 12 historical records are all still present and unmodified in count
    n = len([f for f in os.listdir(os.path.join(P, "outputs"))
             if f.startswith("final_") and f.endswith(".json")])
    check("T1e_all_12_historical_records_present", n == 12, f"{n} outputs/final_*.json present")

    # (f) BLOCKING 2: DIRECTORY CREATION cannot escape the run root either. The reviewer showed
    # `--run-tag '../../../attacked/PWNED'` created <P>/attacked/PWNED/. Two independent defences:
    # the tag charset, and _guard_path on every mkdir.
    bad_tags = ["../../../attacked/PWNED", "../../adapters/X", "..", ".", "a/b", "a\\b",
                "a.b", "./x", "-leading", "", "x" * 33, "a;b", "$(id)"]
    rejected = []
    for t in bad_tags:
        try:
            RS.validate_run_tag(t)
            rejected.append((t, False))
        except ValueError:
            rejected.append((t, True))
    ok_tag = all(r for _, r in rejected)
    good = all(RS.validate_run_tag(t) == t for t in ("d013", "run_1", "A-b_9", "x"))
    check("T1f_run_tag_charset_validated", ok_tag and good,
          f"{sum(1 for _, r in rejected if r)}/{len(bad_tags)} malicious tags rejected; plain "
          f"labels accepted" + ("" if ok_tag else f"; ACCEPTED {[t for t, r in rejected if not r]}"))

    # (g) even bypassing the tag check, make_run_dir/guarded_makedirs refuse traversal and create
    # NOTHING. This is the exact exploit, run against a sandbox root.
    with tempfile.TemporaryDirectory() as td:
        sandbox_root = os.path.join(td, "outputs", "rescore")
        orig = RS.RESCORE_ROOT
        RS.RESCORE_ROOT = sandbox_root
        try:
            evil = "rsc_20261026T000000Z_deadbeefdeadbeef_../../../attacked/PWNED"
            try:
                RS.make_run_dir(evil, root=sandbox_root)
                ok_a, why = False, "make_run_dir CREATED a traversing directory"
            except RuntimeError as e:
                ok_a, why = "REFUSING" in str(e), f"refused: {str(e)[:70]}..."
            created = os.path.exists(os.path.join(td, "outputs", "attacked"))
            # guarded_makedirs refuses the protected real trees directly
            for p in (os.path.join(P, "attacked", "PWNED"),
                      os.path.join(P, "adapters", "PWNED"),
                      os.path.join(sandbox_root, "..", "..", "attacked", "PWNED")):
                try:
                    RS.guarded_makedirs(p)
                    ok_a = False
                    why += f" | guarded_makedirs ALLOWED {p}"
                except RuntimeError:
                    pass
            ok_b = not created and not os.path.exists(os.path.join(P, "attacked", "PWNED")) \
                and not os.path.exists(os.path.join(P, "adapters", "PWNED"))
            check("T1g_directory_creation_cannot_traverse", ok_a and ok_b,
                  f"{why} nothing created outside the run root={ok_b}")
        finally:
            RS.RESCORE_ROOT = orig

    # (h) defect 3: component-wise containment, so a SIBLING whose name merely starts with the
    # root's name is refused. A string-prefix test allowed outputs/rescore_evil/.
    sneaky = [os.path.join(P, "outputs", "rescore_evil", "x.json"),
              os.path.join(P, "outputs", "rescore_evil"),
              os.path.join(P, "outputs", "rescorex", "x.json"),
              os.path.join(P, "outputs", "rescore") + "_evil/x.json"]
    res = []
    for s in sneaky:
        try:
            RS._guard_path(s)
            res.append((s, False))
        except RuntimeError:
            res.append((s, True))
    ok = all(r for _, r in res)
    # and the legitimate path inside the real root is still ALLOWED
    allowed = RS._guard_path(os.path.join(RS.RESCORE_ROOT, "rsc_x", "arms", "a.json"))
    check("T1h_component_wise_containment", ok and allowed.endswith("arms/a.json"),
          f"{sum(1 for _, r in res if r)}/{len(sneaky)} sibling-prefix paths refused; legitimate "
          f"in-root path still allowed"
          + ("" if ok else f"; ALLOWED {[s for s, r in res if not r]}"))


# ------------------------------------------------------------------ T2 atomicity
def t2_atomicity():
    with tempfile.TemporaryDirectory() as td:
        orig_root = RS.RESCORE_ROOT
        RS.RESCORE_ROOT = td      # the guard is live, so the test root must be declared to it
        try:
            _t2_body(RS.make_run_dir(RS.new_run_id("t2"), root=td))
        finally:
            RS.RESCORE_ROOT = orig_root


def _t2_body(d):
        p = os.path.join(d, "run.json")
        RS.atomic_write_json(p, {"run_status": "incomplete", "v": 1})
        first = json.load(open(p))
        RS.atomic_write_json(p, {"run_status": "complete", "v": 2})
        second = json.load(open(p))
        ok_a = first["v"] == 1 and second["v"] == 2
        check("T2a_atomic_write_replaces_in_place", ok_a,
              f"v {first['v']} -> {second['v']} at one path")

        # a failure mid-write must leave the PREVIOUS content and no temp litter
        real_replace = os.replace

        def boom(a, b):
            raise OSError("simulated failure between write and replace")
        try:
            os.replace = boom
            try:
                RS.atomic_write_json(p, {"run_status": "complete", "v": 3})
                raised = False
            except OSError:
                raised = True
        finally:
            os.replace = real_replace
        after = json.load(open(p))
        litter = [f for f in os.listdir(d) if f.startswith(".tmp_")]
        ok_b = raised and after["v"] == 2 and not litter
        check("T2b_failed_write_leaves_prior_content_and_no_temp", ok_b,
              f"raised={raised} content v={after['v']} (expected 2) temp_files={litter}")

        # binary payloads take the same path
        q = os.path.join(d, "embeddings", "x.bin")
        RS.atomic_write_bytes(q, lambda f: f.write(b"\x00\x01\x02"))
        ok_c = open(q, "rb").read() == b"\x00\x01\x02" and \
            not [f for f in os.listdir(os.path.dirname(q)) if f.startswith(".tmp_")]
        check("T2c_atomic_write_bytes", ok_c, "binary write atomic, no temp litter")


# ------------------------------------------------------------------ T3 run-id uniqueness
def t3_run_id_uniqueness():
    # Defect 6: uniqueness was asserted over 2000 draws of a 24-bit salt, which collides ~11% of
    # the time by the birthday bound -- the test was FLAKY and the write-up pasted 2000/2000 as if
    # deterministic. Uniqueness is now GUARANTEED BY CONSTRUCTION (an issued-id set with retry),
    # so this assertion is deterministic, not probabilistic.
    ids = [RS.new_run_id() for _ in range(5000)]
    ok_a = len(set(ids)) == len(ids)
    check("T3a_run_ids_unique_by_construction", ok_a,
          f"{len(set(ids))}/{len(ids)} distinct over 5000 draws; guaranteed by the issued-id set, "
          f"not by salt luck")

    # the salt must also be wide enough that CROSS-process collisions are negligible
    salt = ids[0].split("_")[2]
    bits = len(salt) * 4
    # birthday bound for 2000 draws at this width
    import math
    p = 1 - math.exp(-(2000 ** 2) / (2 * 2 ** bits))
    ok_w = RS.RUN_ID_SALT_HEX >= 16 and bits >= 64 and p < 1e-9
    check("T3a2_salt_width_sufficient", ok_w,
          f"salt={len(salt)} hex = {bits} bits; P(collision in 2000 draws) = {p:.2e} "
          f"(was 6 hex = 24 bits, P = 1.1e-01)")

    ok_b = all(i.startswith("rsc_") and i.endswith(tuple("0123456789abcdef")) for i in ids)
    tagged = RS.new_run_id("mytag")
    ok_c = tagged.endswith("_mytag") and tagged.startswith("rsc_")
    check("T3b_run_id_shape", ok_b and ok_c, f"prefixed+salted; tagged example {tagged}")
    # a run directory is created under the versioned root, never at outputs/ top level
    ok_d = RS.RESCORE_ROOT == os.path.join(P, "outputs", "rescore")
    check("T3c_versioned_run_root", ok_d, f"run root = {RS.RESCORE_ROOT}")


# ------------------------------------------------------------------ T4 status handling
def t4_status_handling():
    req = ["a", "b", "c"]
    EX = {"verdict": "EXACT_REPRODUCTION", "status": "COMPARED"}
    DI = {"verdict": "DISCREPANT", "status": "COMPARED"}
    NH = {"status": "NO_HISTORICAL_RECORD"}

    def arm(st, probe=None):
        d = {"status": st}
        if probe is not None:
            d["determinism_probe"] = probe
        return d

    cases = [
        ({}, "incomplete", "nothing attempted"),
        ({"a": arm("complete", EX)}, "incomplete", "b and c never attempted"),
        ({l: arm("complete", EX) for l in req}, "complete",
         "all complete AND all reproduce history exactly"),
        ({"a": arm("complete", EX), "b": arm("missing_checkpoint"),
          "c": arm("complete", EX)}, "partial", "one missing checkpoint"),
        ({"a": arm("complete", EX), "b": arm("failed"),
          "c": arm("complete", EX)}, "partial", "one failed"),
        ({"a": arm("failed"), "b": arm("failed"),
          "c": arm("missing_checkpoint")}, "partial", "none complete, all accounted"),
        ({"a": arm("incomplete"), "b": arm("complete", EX),
          "c": arm("complete", EX)}, "incomplete", "one arm left mid-run"),
        # ---- defect 5: determinism verdicts now gate 'complete'
        ({l: arm("complete", DI) for l in req}, "complete_not_comparable_to_history",
         "ALL arms complete but ALL discrepant -- cannot be compared to history"),
        ({"a": arm("complete", EX), "b": arm("complete", DI),
          "c": arm("complete", EX)}, "complete_comparability_mixed",
         "some arms comparable, some not"),
        ({l: arm("complete", NH) for l in req}, "complete_not_comparable_to_history",
         "no historical record for any arm -- comparability unknown"),
        ({l: arm("complete") for l in req}, "complete_not_comparable_to_history",
         "probe MISSING entirely -- never silently treated as fine"),
    ]
    bad = []
    for arms, want, why in cases:
        got, _ = RS.derive_run_status(req, arms)
        if got != want:
            bad.append(f"{why}: got {got!r} want {want!r}")
    check("T4a_status_derivation", not bad,
          f"{len(cases)}/{len(cases)} status cases correct" if not bad else "; ".join(bad))

    # a complete status is IMPOSSIBLE while any requested arm is missing or failed
    never_complete = all(
        RS.derive_run_status(req, {**{l: arm("complete", EX) for l in req},
                                   "b": arm(s)})[0] != "complete"
        for s in ("missing_checkpoint", "failed", "incomplete"))
    check("T4b_complete_requires_every_arm", never_complete,
          "'complete' unreachable with any missing/failed/unattempted arm")

    # ---- defect 5, the case that matters scientifically: an all-DISCREPANT run must NOT be
    # reported as a successful correction.
    st, summ = RS.derive_run_status(req, {l: arm("complete", DI) for l in req})
    comp = summ["history_comparability"]
    ok = (st == "complete_not_comparable_to_history" and comp["verdict"] == "none_exact"
          and sorted(comp["discrepant"]) == req and not comp["exact_reproduction"]
          and "CANNOT be compared" in comp["meaning"]
          and "within-run corrected-vs-legacy" in comp["meaning"])
    check("T4d_all_discrepant_run_is_not_complete", ok,
          f"run_status={st!r}, comparability={comp['verdict']!r}; the within-run label-text "
          f"comparison is still stated as valid")

    # comparability aggregation in isolation
    vcases = [
        ({l: arm("complete", EX) for l in req}, "all_exact"),
        ({l: arm("complete", DI) for l in req}, "none_exact"),
        ({"a": arm("complete", EX), "b": arm("complete", DI)}, "mixed"),
        ({l: arm("complete", NH) for l in req}, "unknown"),
        ({l: arm("complete") for l in req}, "unknown"),
        ({"a": arm("missing_checkpoint")}, "unknown"),
    ]
    bad2 = [f"want {w} got {RS.derive_history_comparability(a)[0]}"
            for a, w in vcases if RS.derive_history_comparability(a)[0] != w]
    check("T4e_comparability_aggregation", not bad2,
          f"{len(vcases)}/{len(vcases)} comparability cases correct"
          if not bad2 else "; ".join(bad2))

    # exit code is non-zero for every status except 'complete'
    src = open(os.path.join(SCRIPTS, "rescore_final_v2.py")).read()
    check("T4f_exit_code_gated_on_complete",
          'return 0 if manifest["run_status"] == "complete" else 1' in src,
          "process exits non-zero unless run_status is exactly 'complete'")

    # the manifest's initial on-disk status is 'incomplete', so a crash cannot read as finished
    src = open(os.path.join(SCRIPTS, "rescore_final_v2.py")).read()
    ok = '"run_status": "incomplete"' in src and src.index('"run_status": "incomplete"') < \
        src.index("scorer = cs.ClipScorer")
    check("T4c_manifest_written_incomplete_before_any_arm", ok,
          "run.json is written with run_status=incomplete before the scorer is even constructed")


# ------------------------------------------------------------------ T5 missing checkpoint
def t5_missing_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        # declare the guard's root first: it now covers directory creation too
        orig_root = RS.RESCORE_ROOT
        RS.RESCORE_ROOT = td
        try:
            d = RS.make_run_dir(RS.new_run_id("t5"), root=td)
            arm = {"label": "ATK_ghost", "unet": "attacked/DOES_NOT_EXIST/unet.pt",
                   "target": "church", "base_model": "CompVis/stable-diffusion-v1-4",
                   "base_revision_override": None, "original_arm_precommitted": False,
                   "original_arm_status": "POST_HOC: test", "this_evaluation_status": "x",
                   "this_evaluation_note": "x"}
            prov = {"run_id": "t5", "checkpoint_hashes": {}}

            class A:
                n_per, steps, guidance, variant, device = 5, 25, 7.5, "fp16", "cpu"
            rec = RS.run_arm(arm, None, mod("split_registry"), A(), d, prov, 1 << 20)
            ok_a = rec["status"] == "missing_checkpoint"
            ok_b = "ABSENT" in rec["status_detail"]
            ok_c = "NOT evaluated" in rec["status_consequence"]
            ok_d = os.path.exists(os.path.join(d, "arms", "ATK_ghost.json"))
            ok_e = "per_image" not in rec and "headline_number" not in rec
            check("T5a_missing_checkpoint_is_explicit", ok_a and ok_b and ok_c,
                  f"status={rec['status']!r}, detail+consequence recorded")
            check("T5b_missing_arm_record_is_written_not_skipped", ok_d and ok_e,
                  "arms/ATK_ghost.json written; no scores fabricated")
            # and such an arm can never make a run 'complete'
            st, _ = RS.derive_run_status(["ATK_ghost"], {"ATK_ghost": rec})
            check("T5c_missing_arm_forces_partial", st == "partial", f"run_status={st!r}")
        finally:
            RS.RESCORE_ROOT = orig_root

    # the real, observed situation: the ATK_* checkpoints are gone, and the plan says so
    plan = RS.build_arm_plan()
    present, absent = [], []
    for a in plan:
        if a["unet"] is None:
            present.append(a["label"]); continue
        full = a["unet"] if os.path.isabs(a["unet"]) else os.path.join(P, a["unet"])
        (present if os.path.exists(full) else absent).append(a["label"])
    ok = len(plan) == 12 and len(absent) == 6 and all(l.startswith("ATK_") for l in absent)
    check("T5d_observed_checkpoint_inventory", ok,
          f"{len(plan)} arms planned; regenerable={sorted(present)}; "
          f"NOT regenerable (checkpoint gone)={sorted(absent)}")


# ------------------------------------------------------------------ T6 paired-scoring bookkeeping
def t6_paired_scoring():
    import torch
    identity, sets = CS.label_sets()
    # a stub scorer: per_image_records touches ONLY self.identity, so no weights are needed
    sc = object.__new__(CS.ClipScorer)
    sc.identity = identity
    # image 0 -> church wins; image 1 -> golf_ball wins, church second
    sims = torch.tensor([[0.30, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
                         [0.21, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.44, 0.19, 0.20]])
    recs = sc.per_image_records(sims, "church")
    ok_a = all(len(r["scores"]) == 12 for r in recs)
    ok_b = set(recs[0]["scores"]) == set(identity)
    ok_c = (recs[0]["pred_label"] == "church" and recs[0]["is_target"] and
            recs[0]["target_rank"] == 1 and abs(recs[0]["margin"] - 0.10) < 1e-6)
    ok_d = (recs[1]["pred_label"] == "golf_ball" and not recs[1]["is_target"] and
            recs[1]["target_rank"] == 2 and abs(recs[1]["target_sim"] - 0.21) < 1e-6)
    check("T6a_all_twelve_scores_retained", ok_a and ok_b,
          "every record carries all 12 label scores keyed by identity key, not just the argmax")
    check("T6b_argmax_margin_rank_target_sim", ok_c and ok_d,
          "pred/margin/target_rank/target_sim correct for a target win and a target loss")

    # pairing: both label sets must score EVERY image, and the diff flags are computed from them
    index = [{"i": 0, "prompt_index": 0, "image_index": 0, "seed": 4000000},
             {"i": 1, "prompt_index": 0, "image_index": 1, "seed": 4000001}]
    corrected = sc.per_image_records(sims, "church")
    legacy = sc.per_image_records(sims.flip(0), "church")   # deliberately different decisions
    paired = RS.pair_records(index, {"corrected": corrected, "legacy": legacy})
    ok_e = len(paired) == 2 and all("corrected" in r and "legacy" in r for r in paired)
    ok_f = all(r["pred_changed_by_label_text"] for r in paired)
    ok_g = all(r["seed"] is not None for r in paired)
    check("T6c_pairing_keeps_both_sets_per_image", ok_e and ok_g,
          "each paired row carries both label sets' full records plus prompt/seed provenance")
    check("T6d_label_text_change_flags", ok_f,
          "pred_changed_by_label_text flags the images whose decision differs between sets")

    # a ragged pair is a hard error, not a silently truncated comparison
    try:
        RS.pair_records(index, {"corrected": corrected, "legacy": legacy[:1]})
        ok_h, why = False, "ragged label sets ACCEPTED"
    except ValueError as e:
        ok_h, why = "not a pair" in str(e), f"refused: {str(e)[:60]}..."
    try:
        RS.pair_records(index, {"corrected": corrected})
        ok_i = False
    except ValueError:
        ok_i = True
    check("T6e_ragged_or_single_set_refused", ok_h and ok_i, why + " and a lone set is refused")

    # one image-feature tensor feeds both sets: that is what isolates the TEXT effect
    src = open(os.path.join(SCRIPTS, "clip_scorer.py")).read()
    ok_j = "def score_all_sets" in src and "f = self.image_features(images)" in src and \
        "self.score_features(f, name) for name in self._sets" in src
    check("T6f_both_sets_score_the_same_features", ok_j,
          "score_all_sets computes image features ONCE and scores every label set against them")


# ------------------------------------------------------------------ T7 provenance completeness
def t7_provenance():
    class A:
        n_per, steps, guidance, variant, device = 5, 25, 7.5, "fp16", "cuda"
        hash_cap_mb = 64
    plan = RS.build_arm_plan()
    prov = RS.collect_provenance(A(), "PROVTEST", "<none>", plan, 64 << 20)
    required = ["rescore_version", "scoring_version", "run_id", "generated_at", "code",
                "environment", "environment_caveat", "models", "inference", "prompts", "seeds",
                "label_sets", "checkpoint_hashes", "checkpoint_hash_scope"]
    miss = [k for k in required if k not in prov]
    check("T7a_top_level_provenance_keys", not miss,
          f"all {len(required)} present" if not miss else f"MISSING {miss}")

    sub = {
        "code.file_sha256(rescore_final_v2.py)": prov["code"]["file_sha256"].get("rescore_final_v2.py"),
        "code.file_sha256(clip_scorer.py)": prov["code"]["file_sha256"].get("clip_scorer.py"),
        "code.file_sha256(gen_eval.py)": prov["code"]["file_sha256"].get("gen_eval.py"),
        "code.git_note": prov["code"]["git_note"],
        "environment.deps.torch": prov["environment"]["deps"].get("torch"),
        "environment.deps.diffusers": prov["environment"]["deps"].get("diffusers"),
        "environment.deps.transformers": prov["environment"]["deps"].get("transformers"),
        "environment.python": prov["environment"]["python"],
        "models.pin_file.revision": prov["models"]["pin_file"].get("revision"),
        "models.classifier.id": prov["models"]["classifier"]["id"],
        "models.numerics.dtype": prov["models"]["numerics"]["dtype"],
        "inference.num_inference_steps": prov["inference"]["num_inference_steps"],
        "inference.guidance_scale": prov["inference"]["guidance_scale"],
        "inference.scheduler": prov["inference"]["scheduler"],
        "prompts.final_frozen_at": prov["prompts"]["final_frozen_at"],
        "seeds.formula": prov["seeds"]["formula"],
        "label_sets.n_differ": prov["label_sets"]["n_differ"],
        "scoring_version": prov["scoring_version"],
    }
    empty = [k for k, v in sub.items() if v in (None, "", [], {})]
    check("T7b_provenance_fields_populated", not empty,
          f"{len(sub)} nested fields populated" if not empty else f"EMPTY {empty}")

    # base-model revision AND variant recorded for every arm, per arm (SD-1.4 vs SD-1.5 differ)
    pa = prov["models"]["per_arm_base"]
    ok = (len(pa) == len(plan) and
          all(v["revision"] and v["variant"] and v["base_model"] for v in pa.values()) and
          len({v["revision"] for v in pa.values()}) == 2)
    check("T7c_per_arm_base_revision_and_variant", ok,
          f"{len(pa)} arms, {len({v['revision'] for v in pa.values()})} distinct base revisions, "
          f"variant recorded for all")

    # seeds are enumerated, not merely described
    ok = (len(prov["seeds"]["values"]) == 15 and
          prov["seeds"]["values"][0] == 4_000_000 and
          len(set(prov["seeds"]["values"])) == 15)
    check("T7d_seeds_enumerated", ok,
          f"{len(prov['seeds']['values'])} distinct final_eval seeds recorded, "
          f"first={prov['seeds']['values'][0]}")

    # the numerics contract with gen_eval.py holds (one source of truth)
    c = CS.verify_numerics_contract()
    check("T7e_numerics_contract_with_gen_eval", c["ok"], f"issues={c['issues'] or 'none'}")

    # gen_eval.py is UNCHANGED and still works standalone (import-compatible repair)
    ge = CS.load_gen_eval()
    ok = (hasattr(ge, "clip_scorer") and hasattr(ge, "LABELS") and hasattr(ge, "LAB_KEY")
          and hasattr(ge, "CONCEPTS") and hasattr(ge, "load_pipe") and hasattr(ge, "main")
          and "clip_scorer" not in open(os.path.join(SCRIPTS, "gen_eval.py")).read()
          .split("def clip_scorer")[0].replace("gen_eval", ""))
    check("T7f_gen_eval_unchanged_and_self_contained", ok,
          "gen_eval.py keeps its own clip_scorer/LABELS/LAB_KEY and does not import clip_scorer.py")


# ------------------------------------------------------------------ T8 label-set identity mapping
def t8_label_sets():
    d = CS.label_set_diff()
    expected_differ = ["garbage_truck", "van_gogh", "cassette_player", "chain_saw",
                       "english_springer", "french_horn", "gas_pump", "golf_ball", "monet"]
    expected_same = ["church", "parachute", "tench"]
    ok_a = d["n_labels"] == 12
    ok_b = d["n_differ"] == 9
    ok_c = sorted(d["differing"]) == sorted(expected_differ)
    ok_d = sorted(d["identical"]) == sorted(expected_same)
    check("T8a_twelve_labels", ok_a, f"n_labels={d['n_labels']}")
    check("T8b_exactly_nine_differ", ok_b and ok_c,
          f"n_differ={d['n_differ']}; differing={d['differing']}")
    check("T8c_exactly_three_identical", d["n_identical"] == 3 and ok_d,
          f"identical={d['identical']}")

    # The nine split into SIX underscore-only edits and THREE genuine wording changes. The
    # underscore-only ones are counted as differing, not dismissed as cosmetic: an underscore is a
    # different token sequence.
    underscore_only = sorted(r["identity_key"] for r in d["rows"]
                             if r["differs"]
                             and r["legacy_text"].replace("_", " ") == r["corrected_text"])
    reworded = sorted(r["identity_key"] for r in d["rows"]
                      if r["differs"]
                      and r["legacy_text"].replace("_", " ") != r["corrected_text"])
    ok = (underscore_only == ["cassette_player", "chain_saw", "french_horn", "garbage_truck",
                              "gas_pump", "golf_ball"]
          and reworded == ["english_springer", "monet", "van_gogh"]
          and len(underscore_only) + len(reworded) == 9)
    check("T8d_underscore_only_differences_counted", ok,
          f"6 underscore-only {underscore_only} + 3 reworded {reworded} = 9 differing")

    # Defect 4: the PROSE in gen_eval_final.py must state the SAME split the computation gives.
    # Previously the test merely noted in a comment that the prose was wrong; now it fails on it.
    gf = open(os.path.join(SCRIPTS, "gen_eval_final.py")).read()
    wrong_prose = [
        "not four. This comment originally named only the four whose wording changes",
        "(garbage_truck, van_gogh, english_springer, monet) and silently omitted the five",
        "omitted the five that differ",
    ]
    bad = [w for w in wrong_prose if w in gf]
    right = ("SIX underscore-only edits" in gf and "THREE genuine rewordings" in gf
             and all(k in gf.split("SIX underscore-only edits")[1].split("THREE genuine")[0]
                     for k in underscore_only))
    check("T8d2_prose_split_matches_computation", not bad and right,
          "gen_eval_final.py states 6 underscore-only (garbage_truck included) + 3 reworded, "
          "matching the computed split"
          if not bad and right else f"stale 4/5 prose present: {bad}; correct-split prose={right}")

    # identity mapping is the SAME under both sets, so prediction indices are comparable
    identity, sets = CS.label_sets()
    ok = identity == sets["legacy"] and len(identity) == len(sets["corrected"]) == 12
    check("T8e_identity_mapping_shared", ok,
          "LAB_KEY is the identity mapping under both label sets; indices mean the same concept")

    # NO UNQUALIFIED DIRECTIONAL CLAIM survives in the owned files. These exact assertions were
    # present pre-repair in gen_eval_final.py; each must now be gone.
    banned_claims = [
        "nine weakened competitors bias the target's hit count UPWARD",
        "biasing target_hits UPWARD",
        "all nine competitors were degraded",
        "the target text was correct while all nine competitors were degraded",
    ]
    offenders = []
    for fn in ("clip_scorer.py", "rescore_final_v2.py", "gen_eval_final.py"):
        src = open(os.path.join(SCRIPTS, fn)).read()
        offenders += [f"{fn}: {c!r}" for c in banned_claims if c in src]
    check("T8f_no_directional_claim_encoded", not offenders,
          f"all {len(banned_claims)} pre-repair directional assertions removed from the 3 owned "
          f"files" if not offenders else f"STILL PRESENT: {offenders}")

    # and each owned file carries the explicit disclaimer instead
    need = {"clip_scorer.py": "DIRECTION AND MAGNITUDE ARE NOT",
            "rescore_final_v2.py": "NOT a resolution of the mismatch's DIRECTION",
            "gen_eval_final.py": "NOT established: the DIRECTION or the SIZE"}
    missing = [fn for fn, s in need.items()
               if s not in open(os.path.join(SCRIPTS, fn)).read()]
    ok = not missing and "withdrawn" in open(os.path.join(SCRIPTS, "gen_eval_final.py")).read()
    check("T8f2_direction_disclaimer_present", ok,
          "all 3 owned files state that direction and size are unestablished; the earlier UPWARD "
          "assertion is explicitly marked withdrawn"
          if ok else f"missing disclaimer in {missing}")

    # the sanctioned interpretation string IS present, verbatim in substance
    ok = ("unresolved pending corrected evaluation" in d["interpretation"]
          and "inconsistent label text" in d["interpretation"])
    check("T8g_sanctioned_interpretation_recorded", ok,
          "every record carries: 'The evaluator used inconsistent label text. The effect ... is "
          "unresolved pending corrected evaluation.'")


# ------------------------------------------------------------------ extra: role labelling
def t9_role_labelling():
    plan = RS.build_arm_plan()
    ok_a = all(a["this_evaluation_status"] == "CORRECTIVE_RE_EVALUATION_OF_INSPECTED_SPLIT"
               for a in plan)
    # original precommitment status is kept, and is NOT uniform
    pre = [a["label"] for a in plan if a["original_arm_precommitted"]]
    post = [a["label"] for a in plan if not a["original_arm_precommitted"]]
    ok_b = len(pre) == 7 and len(post) == 5
    check("T9a_this_eval_role_uniform_and_corrective", ok_a,
          "every arm labelled CORRECTIVE_RE_EVALUATION_OF_INSPECTED_SPLIT regardless of "
          "original precommitment")
    check("T9b_original_precommitment_kept_separate", ok_b,
          f"original pre-committed={len(pre)}, post-hoc={len(post)} ({post})")

    src = open(os.path.join(SCRIPTS, "gen_eval_final.py")).read()
    ok_c = "first_and_only_legitimate_use" not in src.split("# Arms NOT pre-committed")[0] or \
        '"first_and_only_legitimate_use": label' not in src
    ok_d = '"first_and_only_legitimate_use": label in PRECOMMITTED_LABELS' not in src
    ok_e = "held-out prompts, never used for selection" not in src
    ok_f = "FIRST and ONLY legitimate use of the frozen FINAL" not in src
    check("T9c_gen_eval_final_language_repaired", ok_c and ok_d and ok_e and ok_f,
          "no unconditional first_and_only_legitimate_use field; no 'never used'/'FIRST and ONLY' "
          "header claim")
    ok_g = 'INSPECTED_BEFORE_THIS_RUN' in open(
        os.path.join(SCRIPTS, "rescore_final_v2.py")).read()
    ok_h = "REGENERATION_PLUS_CORRECTED_SCORING" in open(
        os.path.join(SCRIPTS, "rescore_final_v2.py")).read()
    check("T9d_rescore_self_describes_correctly", ok_g and ok_h,
          "run labels itself INSPECTED_BEFORE_THIS_RUN and "
          "method=REGENERATION_PLUS_CORRECTED_SCORING")


# ------------------------------------------------------------------ extra: history comparison
def t10_history_comparison():
    identity, _ = CS.label_sets()
    sc = object.__new__(CS.ClipScorer)
    sc.identity = identity

    # no historical record -> reported, not assumed equivalent
    r = RS.compare_with_history("NO_SUCH_ARM_ANYWHERE", "church", [])
    check("T10a_absent_history_reported", r["status"] == "NO_HISTORICAL_RECORD",
          f"status={r['status']!r}")

    # synthetic legacy records that MATCH the real historical record exactly -> EXACT_REPRODUCTION
    hist = json.load(open(os.path.join(P, "outputs", "final_utrap_church_l0.15k8.json")))
    fake = [{"pred_label": h["pred_label"], "is_target": h["is_target"],
             "margin": h["margin"], "target_sim": h["target_sim"]}
            for h in hist["per_image_predictions"]]
    r = RS.compare_with_history("utrap_church_l0.15k8", "church", fake)
    ok_a = (r["verdict"] == "EXACT_REPRODUCTION" and r["n_compared"] == 15
            and r["per_image_pred_label_agreement"] == 1.0
            and r["reproduced_legacy_target_hits"] == hist["target_hits"])
    check("T10b_exact_reproduction_detected", ok_a,
          f"verdict={r['verdict']}, agreement={r['per_image_pred_label_agreement']}, "
          f"hits {r['reproduced_legacy_target_hits']} vs history {r['historical_target_hits']}")

    # perturb one prediction -> DISCREPANT, reported with the index, NOT swallowed
    fake2 = [dict(f) for f in fake]
    fake2[3] = {**fake2[3], "pred_label": "golf_ball", "is_target": False,
                "target_sim": fake2[3]["target_sim"] + 0.01}
    r = RS.compare_with_history("utrap_church_l0.15k8", "church", fake2)
    ok_b = (r["verdict"] == "DISCREPANT" and r["disagreeing_indices"] == [3]
            and r["n_per_image_pred_disagreements"] == 1
            and abs(r["max_abs_target_sim_delta"] - 0.01) < 1e-9
            and "cannot be separated here" in r["verdict_note"])
    check("T10c_discrepancy_reported_not_assumed", ok_b,
          f"verdict={r['verdict']}, disagreeing_indices={r['disagreeing_indices']}, "
          f"max|dtarget_sim|={r['max_abs_target_sim_delta']:.4f}")

    # the marker for pre-repair records is their MISSING label_text_set field
    ok_c = r["historical_has_label_text_set_field"] is False
    check("T10d_prerepair_marker_detected", ok_c,
          "historical record carries no label_text_set field, the pre-repair marker")


# --------------------------------------------- T11 exposure figure recomputed, not greped
def t11_exposure_figure():
    """Reviewer BLOCKING 1 + the G14 lesson.

    `T8f` greps four verbatim pre-repair strings. A test that greps for ABSENT strings can only
    catch a claim that was removed; it cannot catch a WRONG claim that is PRESENT -- which is
    exactly how 26/180 survived. So: recompute the figure from outputs/final_*.json and assert that
    every documented value equals the recomputation.
    """
    got = RS.recompute_exposure()
    doc = RS.DOCUMENTED_EXPOSURE

    ok_a = got["n_records"] == 12 and got["n_images_total"] == 180
    check("T11a_recomputed_over_all_twelve_records", ok_a,
          f"{got['n_records']} records, {got['n_images_total']} images scanned")

    ok_b = doc["n_on_differing_labels"] == got["n_on_differing_labels"]
    check("T11b_documented_total_equals_recomputed", ok_b,
          f"documented {doc['n_on_differing_labels']}/{doc['n_images_total']} == recomputed "
          f"{got['n_on_differing_labels']}/{got['n_images_total']}"
          + ("" if ok_b else "  <-- MISMATCH"))

    ok_c = doc["per_arm"] == got["per_arm"]
    check("T11c_documented_per_arm_equals_recomputed", ok_c,
          f"per-arm {got['per_arm']}" + ("" if ok_c else f"  <-- documented {doc['per_arm']}"))

    # internal consistency: the per-arm counts must SUM to the total. The withdrawn figure failed
    # this on its face (9+7+5+4 = 25, documented as 26) and nothing caught it.
    s = sum(doc["per_arm"].values())
    check("T11d_per_arm_counts_sum_to_total", s == doc["n_on_differing_labels"],
          f"sum(per_arm) = {s} == total {doc['n_on_differing_labels']}")

    # median margins of the four defence arms must fall inside the documented range
    defence = ["utrap_church_l0.15k8", "frp_church_r3_l05", "utrap_sd15_church_l0.15k8",
               "uce_sd15_church_lam1000"]
    meds = [got["median_margin_by_arm"][k] for k in defence]
    lo, hi = doc["median_margin_range_four_defence_arms"]
    ok_e = all(m is not None and lo <= m <= hi for m in meds) and \
        abs(min(meds) - lo) < 1e-9 and abs(max(meds) - hi) < 1e-9
    check("T11e_documented_margin_range_equals_recomputed", ok_e,
          f"defence-arm medians {dict(zip(defence, meds))} span [{min(meds)}, {max(meds)}] == "
          f"documented [{lo}, {hi}]")

    # exposure is NOT confined to the defence arms -- the withdrawn claim said it was
    nondefence = {k: v for k, v in got["per_arm"].items() if k not in defence}
    check("T11f_exposure_not_confined_to_defence_arms", bool(nondefence),
          f"non-defence arms also carry exposure: {nondefence}; 'concentrated in the four defence "
          f"arms' is false and is withdrawn")

    # the WRONG figures must appear nowhere in the owned files or the doc, and the RIGHT ones must
    # appear in both. This is the check that greping alone could not do.
    doc_md = os.path.join(P, "docs", "REPAIR_EVAL_v2.md")
    targets = {"scripts/gen_eval_final.py": os.path.join(SCRIPTS, "gen_eval_final.py"),
               "scripts/rescore_final_v2.py": os.path.join(SCRIPTS, "rescore_final_v2.py"),
               "docs/REPAIR_EVAL_v2.md": doc_md}
    wrong = ["26 of their 180", "26/180", "0.0065-0.0131", "(9/15, 7/15, 5/15, 4/15)",
             "concentrated in the four pre-attack defence arms (9/15"]
    bad = []
    MARKERS = ("WRONG", "withdrawn", "Withdrawn", "previously", "earlier version", "is withdrawn")
    for name, path in targets.items():
        lines = open(path).read().split("\n")
        for i, line in enumerate(lines):
            for w in wrong:
                if w not in line:
                    continue
                # A mention inside an explicit withdrawal is allowed. The withdrawal marker may sit
                # on a neighbouring line (a wrapped comment or a blockquote), so inspect a window.
                window = "\n".join(lines[max(0, i - 6):i + 7])
                if not any(k in window for k in MARKERS):
                    bad.append(f"{name}:{i+1}: {w!r} in {line.strip()[:60]}")
    check("T11g_wrong_figure_absent_except_as_withdrawal", not bad,
          "the withdrawn figures appear only inside explicit withdrawals"
          if not bad else f"{len(bad)} bare occurrence(s): {bad[:3]}")

    present = {name: ("37" in open(p).read() and "0.0027" in open(p).read())
               for name, p in targets.items()}
    check("T11h_correct_figure_documented_everywhere", all(present.values()),
          f"37/180 and the 0.0027-0.0063 margin range present in: "
          f"{[k for k, v in present.items() if v]}"
          + ("" if all(present.values()) else f"; MISSING in {[k for k,v in present.items() if not v]}"))

    # the bound claim is gone, and the register's framing is in its place
    bound_claims = ["bounds how many could move", "which bounds how many could move",
                    "bound on how many decisions could move."]
    # As in T11g: the claim may be QUOTED while being withdrawn or denied, but never asserted.
    DENIALS = ("NOT a bound", "not a bound", "unsupported", "removed", "withdrawn", "Withdrawn",
               "WRONG", "earlier version", "previously")
    bad2 = []
    for name, path in targets.items():
        lines = open(path).read().split("\n")
        for i, line in enumerate(lines):
            for c in bound_claims:
                if c not in line:
                    continue
                window = "\n".join(lines[max(0, i - 6):i + 7])
                if not any(k in window for k in DENIALS):
                    bad2.append(f"{name}:{i+1}: {line.strip()[:70]}")
    ok_i = not bad2 and "NOT a bound on how many decisions could move" in RS.EXPOSURE_FRAMING
    check("T11i_bound_claim_removed", ok_i,
          "the 'bounds how many could move' claim is removed; EXPOSURE_FRAMING states it is NOT a "
          "bound, because an unchanged-label prediction can also flip"
          if ok_i else f"still present: {bad2}")

    ok_j = ("Small margins mean the outcomes are sensitive to label text; they do not indicate "
            "which way" in RS.EXPOSURE_FRAMING)
    check("T11j_register_framing_matched", ok_j,
          "framing matches the audit register and FINDING65 verbatim")


# --------------------------------------------- T12 unmodified-file hash baseline
def t12_unmodified_baseline():
    """Reviewer defect 7. The 'unmodified' claim rested on mtime alone."""
    b = RS.verify_unmodified_baseline()
    ok_a = set(b["files"]) == {"gen_eval.py", "attack_relearn.py"}
    ok_b = all(v["matches"] is True for v in b["files"].values()) and b["ok"]
    ok_c = all(len(v["expected_sha256"]) == 64 for v in b["files"].values())
    check("T12a_baseline_covers_both_protected_files", ok_a, f"{sorted(b['files'])}")
    check("T12b_both_files_match_recorded_sha256", ok_b,
          "; ".join(f"{k}={v['actual_sha256'][:16]}... match={v['matches']}"
                    for k, v in b["files"].items()))
    check("T12c_full_sha256_not_truncated", ok_c,
          "64-hex full digests recorded (the prior provenance_manifest.json truncates to 16)")

    # the claim's SCOPE is stated, not overclaimed
    ok_d = ("only from baseline_taken_at onward" in b["scope_note"]
            and "mtime only" in b["scope_note"])
    check("T12d_baseline_scope_not_overclaimed", ok_d,
          f"baseline taken {b['baseline_taken_at']}; scope_note says byte-identity is proven only "
          f"from that date onward, mtime only before it")

    # a modified file must be DETECTED, not passed
    orig = dict(RS.UNMODIFIED_BASELINE_SHA256)
    try:
        RS.UNMODIFIED_BASELINE_SHA256["gen_eval.py"] = "0" * 64
        b2 = RS.verify_unmodified_baseline()
        ok_e = (b2["ok"] is False and b2["files"]["gen_eval.py"]["matches"] is False)
    finally:
        RS.UNMODIFIED_BASELINE_SHA256.clear()
        RS.UNMODIFIED_BASELINE_SHA256.update(orig)
    check("T12e_mismatch_is_detected", ok_e,
          "a wrong expected hash makes verify_unmodified_baseline() report ok=False")


def main():
    print("=" * 78)
    print("CPU tests: corrected FINAL re-scoring pipeline (D-013). No GPU, no model weights.")
    print("=" * 78)
    for fn in (t1_overwrite_refusal, t2_atomicity, t3_run_id_uniqueness, t4_status_handling,
               t5_missing_checkpoint, t6_paired_scoring, t7_provenance, t8_label_sets,
               t9_role_labelling, t10_history_comparison, t11_exposure_figure,
               t12_unmodified_baseline):
        print(f"\n--- {fn.__name__} ---")
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(fn.__name__, False, f"test harness error {type(e).__name__}: {e}")
    print("\n" + "=" * 78)
    print(f"PASSED {len(PASSED)}   FAILED {len(FAILED)}")
    if FAILED:
        print("failures: " + ", ".join(FAILED))
    print("=" * 78)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
