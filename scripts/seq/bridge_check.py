#!/usr/bin/env python3
"""Bridge the short early-grid rerun to the original 1000-step run. CPU only.

Why this exists
---------------
The early-grid amendment needs dumps at optimizer steps 10..90, and dumps can
only be written while training runs. So each trajectory is retrained from its
saved MA for a SHORTER horizon with a FINER save cadence. That makes the early
dumps a SECOND REALISATION, and the question "are these points on the same path
the original run took?" has to be answered by measurement, not assumed.

This tool answers it at the one step the two runs share: step 100.

Three separate questions, reported separately and never merged:

  A  SETTINGS.  Every setting that determines the first N updates must be
     identical, and the settings that were DELIBERATELY CHANGED must be exactly
     the declared two (the stopping limit and the save cadence) plus whatever
     upstream derives from the stopping limit. Anything else differing is a
     failure of the rerun, not a finding about training.

     The stopping limit is only harmless because `lr_scheduler == "constant"`.
     diffusers' `get_scheduler` routes CONSTANT to `get_constant_schedule` and
     discards both `num_warmup_steps` and `num_training_steps`, so upstream's
     `num_training_steps = iterations * grad_accum` never reaches the schedule.
     Under any other scheduler, shortening the horizon WOULD change the learning
     rate at every one of the first 100 updates and the rerun would be invalid.
     That is checked here rather than trusted.

  B  WEIGHTS at step 100. Bitwise identity, max absolute difference, relative
     Frobenius norm. This is ONE step: identity there is strong evidence that
     the two runs agreed at step 100, but it does NOT prove the intermediate
     states at steps 1..99 were identical -- and that unmeasured interior is
     exactly where this amendment's candidates come from. Anything other than
     identity is a measured distance, and its CAUSE is not isolated here.

  C  BEHAVIOUR at step 100. Development dog suppression against the FIXED
     reference parent evaluation -- not each run's own regenerated parent -- at
     t=0.5 over the same 80 pairs, and the difference in pp against the
     amendment's declared stop condition: if the two realisations differ by more
     than the match tolerance itself, run-to-run variation is of the same order
     as the quantity being matched, the early grid cannot be trusted to locate a
     match point, and the amendment stops and reports that instead.

  D  FIXED REFERENCES. MA and MAB_L2 define the target every candidate is
     matched to, so they are read from one fixed, verified evaluation; each
     one's generating checkpoint must be the registered saved artifact and its
     manifest the frozen development manifest. A regenerated reference beside
     the candidates is itself a violation: that is how a reference measuring
     45 pp instead of 30 pp once passed unnoticed.

  E  THE DECLARED CONTRACT, by value. The two declared changes have declared
     numbers -- 100 optimizer steps, a dump every 10 -- and the rerun must also
     reach that horizon and write that grid. A different horizon or cadence is
     an UNDECLARED change however it is labelled. Required settings that are
     simply absent are violations, not matches.

What this tool does NOT do
--------------------------
* It does not attribute any difference it finds to a cause. Identical RECORDED
  configuration plus the same parent does not isolate nondeterminism: the two
  runs also differ in library/driver state, upstream code path (periodic
  checkpointing at a different cadence), unrecorded dataloader ordering, and
  anything else outside the 32 recorded hyperparameters. No controlled repeat
  was run and none is requested here.
* It does not establish that either realisation's step-k state is "the" state
  of that configuration at step k, nor that agreement at step 100 implies
  agreement at any earlier step.
* It says nothing about retention, and it never touches the frozen test set.

    python scripts/seq/bridge_check.py \
        --original_root <seq>/diag_v1 --rerun_root <seq>/diag_v2_early_grid \
        --seeds "17 29" --bridge_step 100 --out <rerun>/bridge_check.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent

# Settings that determine the first N optimizer updates. Every one of these must
# be identical between the two runs. The list is explicit rather than "all keys
# except the ones that differ", so that a NEW hyperparameter appearing in a
# future report is reported as unclassified instead of being silently tolerated.
FIRST_UPDATES_KEYS = (
    "learning_rate_cli_default_before_scaling", "scale_lr",
    "effective_learning_rate", "effective_lr_cross_check", "scale_lr_formula",
    "anchor_batch_size", "gradient_accumulation_steps", "num_processes",
    "optimizer", "adam_beta1", "adam_beta2", "adam_weight_decay",
    "adam_epsilon", "max_grad_norm", "lr_scheduler", "lr_warmup_steps",
    "precision", "parameter_group", "with_anchor_preservation", "l1sp_weight",
    "l2sp_weight", "with_gradient_projection", "with_selft", "eval_interval",
    "seed", "hflip", "noaug", "resolution", "steps_per_epoch",
)
# Changed on purpose, or derived by upstream from the stopping limit.
DECLARED_CHANGE_KEYS = ("iterations_requested", "epochs_cap",
                        "epoch_capacity_steps")
# The amendment's declared numbers. A difference is "declared" only if it
# matches THESE; any other horizon or cadence is an undeclared change, however
# it is labelled. Overridable on the command line only so the contract can be
# stated explicitly by a caller, never to make a mismatch pass.
CONTRACT = {"original_iterations": 1000, "rerun_iterations": 100,
            "original_cadence": 100, "rerun_cadence": 10,
            "bridge_step": 100, "tolerance_pp": 5.0, "seeds": (17, 29)}
REQUEST_KEYS = ("parent", "new_deletion_target", "anchor_concept",
                "anchor_target_mapping", "l2sp_weight", "training_seed")
# Only these schedulers are horizon-independent, so only under these may the
# stopping limit be changed without changing the first N updates.
HORIZON_FREE_SCHEDULERS = ("constant", "piecewise_constant")


def _load_selection_helpers():
    spec = importlib.util.spec_from_file_location(
        "_sel", HERE / "select_matched_dump.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def compare_settings(orig: dict, rerun: dict, bridge_step: int,
                     contract: dict, seed: int) -> dict:
    eo = orig.get("effective_hyperparameters") or {}
    er = rerun.get("effective_hyperparameters") or {}
    req_diff = {k: {"original": orig.get(k), "rerun": rerun.get(k)}
                for k in REQUEST_KEYS if orig.get(k) != rerun.get(k)}

    # MISSING IS NOT EQUAL. A report that simply does not record a setting
    # cannot be said to preserve it: `eo.get(k) == er.get(k)` was True for
    # None == None, so a report missing half its hyperparameters passed. Absence
    # is now a violation in its own right.
    preserved, violated, missing = [], {}, {}
    for k in FIRST_UPDATES_KEYS:
        in_o, in_r = k in eo, k in er
        if not (in_o and in_r):
            missing[k] = {"present_in_original": in_o, "present_in_rerun": in_r}
            continue
        if eo[k] == er[k]:
            preserved.append({"setting": k, "value": eo[k]})
        else:
            violated[k] = {"original": eo.get(k), "rerun": er.get(k)}

    declared, undeclared_same = {}, []
    for k in DECLARED_CHANGE_KEYS:
        if k not in eo or k not in er:
            missing[k] = {"present_in_original": k in eo,
                          "present_in_rerun": k in er}
            continue
        if eo[k] != er[k]:
            declared[k] = {"original": eo.get(k), "rerun": er.get(k)}
        else:
            undeclared_same.append(k)

    classified = set(FIRST_UPDATES_KEYS) | set(DECLARED_CHANGE_KEYS)
    unclassified = sorted((set(eo) | set(er)) - classified)
    unclassified_diff = {k: {"original": eo.get(k), "rerun": er.get(k)}
                         for k in unclassified if eo.get(k) != er.get(k)}

    sched_o, sched_r = eo.get("lr_scheduler"), er.get("lr_scheduler")
    horizon_free = (sched_o in HORIZON_FREE_SCHEDULERS
                    and sched_r in HORIZON_FREE_SCHEDULERS)

    # The save cadence lives in the dumps block, not in the hyperparameters.
    def cadence(rep: dict) -> int | None:
        d = rep.get("dumps") or {}
        steps = sorted(d.get("expected_steps") or [])
        if len(steps) >= 2:
            return steps[1] - steps[0]
        return steps[0] if steps else None

    # ---- the declared contract, enforced by VALUE.
    #
    # "Declared" is not a label a run can award itself by differing. The
    # amendment declared exactly two changes with exactly two values; a rerun at
    # 250 steps, or saving every 25, is an UNDECLARED change to the experiment
    # even though it would show up in the same diff. Checked here so the word
    # keeps its meaning.
    o_cad, r_cad = cadence(orig), cadence(rerun)
    want = {
        "original_iterations": (eo.get("iterations_requested"),
                                contract["original_iterations"]),
        "rerun_iterations": (er.get("iterations_requested"),
                             contract["rerun_iterations"]),
        "original_cadence": (o_cad, contract["original_cadence"]),
        "rerun_cadence": (r_cad, contract["rerun_cadence"]),
        "bridge_step": (bridge_step, contract["bridge_step"]),
        "training_seed": (er.get("seed"), seed),
    }
    contract_violations = {k: {"actual": a, "declared": w}
                           for k, (a, w) in want.items() if a != w}
    if seed not in tuple(contract["seeds"]):
        contract_violations["seed_is_declared"] = {
            "actual": seed, "declared": list(contract["seeds"])}
    # The rerun must also actually reach its horizon, and write the dump grid
    # the amendment asked for.
    r_steps = sorted(((rerun.get("dumps") or {}).get("expected_steps")) or [])
    want_steps = list(range(contract["rerun_cadence"],
                            contract["rerun_iterations"] + 1,
                            contract["rerun_cadence"]))
    if r_steps != want_steps:
        contract_violations["rerun_dump_grid"] = {"actual": r_steps,
                                                  "declared": want_steps}
    r_done = (rerun.get("runtime") or {}).get("optimizer_steps_completed")
    if r_done != contract["rerun_iterations"]:
        contract_violations["rerun_optimizer_steps_completed"] = {
            "actual": r_done, "declared": contract["rerun_iterations"]}

    # ---- parent identity, preserved from the original design.
    po = (orig.get("parent_verification") or {}).get("parent_sha256")
    pr = (rerun.get("parent_verification") or {}).get("parent_sha256")
    parent = {
        "original_recorded_parent_sha256": po,
        "rerun_recorded_parent_sha256": pr,
        "same_parent": bool(po) and po == pr,
        "rule": ("both trajectories must record the SAME saved MA digest; a "
                 "different parent makes the two runs incomparable whatever "
                 "their hyperparameters say"),
    }

    return {
        "request_field_differences": req_diff,
        "settings_that_determine_the_first_updates": {
            "n_compared": len(preserved) + len(violated),
            "n_identical": len(preserved),
            "violations": violated,
            "missing_required_fields": missing,
            "all_preserved": not violated and not req_diff and not missing,
            "preserved": preserved,
            "rule": (f"every setting here must be identical, because it enters "
                     f"the first {bridge_step} optimizer updates; a difference "
                     f"invalidates the rerun rather than telling us anything "
                     f"about training"),
        },
        "declared_changes": {
            "stopping_limit_and_derived": declared,
            "save_cadence": {"original_every_n_steps": cadence(orig),
                             "rerun_every_n_steps": cadence(rerun)},
            "declared_before_the_run": True,
            "note": ("the stopping limit and the save cadence are the ONLY "
                     "intended changes; epochs_cap and epoch_capacity_steps are "
                     "recomputed by upstream FROM the stopping limit "
                     "(epochs = ceil(iterations / steps_per_epoch)) and are "
                     "therefore consequences of it, not separate changes"),
        },
        "unclassified_hyperparameters": {
            "names": unclassified,
            "differences": unclassified_diff,
            "note": ("a hyperparameter this tool does not classify must not "
                     "differ silently; any difference here is reported as a "
                     "settings violation"),
        },
        "lr_schedule_is_horizon_independent": {
            "original_lr_scheduler": sched_o,
            "rerun_lr_scheduler": sched_r,
            "horizon_free": horizon_free,
            "why_it_matters": (
                "upstream passes num_training_steps = iterations * "
                "gradient_accumulation_steps to diffusers' get_scheduler. For "
                "CONSTANT (and PIECEWISE_CONSTANT) get_scheduler discards both "
                "num_warmup_steps and num_training_steps, so the learning rate "
                "at update k does not depend on the stopping limit. Under any "
                "other scheduler, shortening the horizon would change the "
                f"learning rate at every one of the first {bridge_step} updates "
                "and this rerun would NOT be comparable."),
            "recorded_lr_warmup_steps_is_inert_here": eo.get("lr_warmup_steps"),
        },
        "declared_contract": {
            "expected": contract,
            "violations": contract_violations,
            "ok": not contract_violations,
            "rule": ("the two declared changes have declared VALUES: 100 "
                     "optimizer steps and a dump every 10. A different horizon "
                     "or cadence is an undeclared change to the experiment, not "
                     "a declared one."),
        },
        "parent_identity": parent,
        "settings_ok": bool(not violated and not req_diff and not missing
                            and not unclassified_diff and horizon_free
                            and not contract_violations and parent["same_parent"]),
        "what_this_comparison_cannot_see": [
            "the anchor images and prompt file CONTENTS -- the reports record "
            "paths and counts, not digests of the data, so identical paths are "
            "assumed to mean identical data",
            "library, driver and upstream-commit state at the two run times, "
            "none of which the reports record",
            "any upstream default that is not among the recorded effective "
            "hyperparameters, including dataloader worker count and ordering",
        ],
    }


def compare_weights(orig_dump: Path, rerun_dump: Path, contract: dict) -> dict:
    import torch

    def load(p: Path) -> dict:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        top = contract.get("top_level_key", "unet")
        sd = obj.get(top) if isinstance(obj, dict) else None
        if not isinstance(sd, dict):
            raise ValueError(f"{p} has no {top!r} state dict")
        return sd

    a, b = load(orig_dump), load(rerun_dump)
    shared = sorted(set(a) & set(b))
    per_tensor, bitwise = [], 0
    max_abs, num, den = 0.0, 0.0, 0.0
    for k in shared:
        ta, tb = a[k].float(), b[k].float()
        same = bool(torch.equal(a[k], b[k]))
        bitwise += int(same)
        d = (ta - tb).abs()
        mx = float(d.max())
        max_abs = max(max_abs, mx)
        num += float((ta - tb).pow(2).sum())
        den += float(ta.pow(2).sum())
        per_tensor.append({"tensor": k, "max_abs_diff": mx,
                           "bitwise_identical": same})
    per_tensor.sort(key=lambda r: -r["max_abs_diff"])
    return {
        "original_dump": str(orig_dump), "rerun_dump": str(rerun_dump),
        "original_sha256": sha256_file(orig_dump),
        "rerun_sha256": sha256_file(rerun_dump),
        "file_hashes_equal": sha256_file(orig_dump) == sha256_file(rerun_dump),
        "tensors_compared": len(shared),
        "bitwise_identical_tensors": bitwise,
        "all_bitwise_identical": bitwise == len(shared) and len(shared) > 0,
        "max_abs_diff": max_abs,
        "overall_relative_frobenius": (num ** 0.5) / ((den ** 0.5) + 1e-12),
        "most_divergent": per_tensor[:5],
        "interpretation": (
            "This compares ONE step. Bitwise identity at step 100 is strong "
            "evidence that the two runs agreed there, but it does NOT prove the "
            "intermediate states at steps 1..99 were identical: those states "
            "were never compared, and agreement at a single later point does "
            "not establish agreement along the path to it. A difference is "
            "likewise a measured distance ONLY: this comparison cannot say "
            "whether it came from nondeterministic kernels, the changed save "
            "cadence, library or driver state, dataloader ordering, or anything "
            "else outside the recorded hyperparameters."),
        "what_identity_here_would_and_would_not_show": {
            "would": "the two runs reached the same state at step 100",
            "would_not": ("that they passed through the same states at steps "
                          "1..99, which is where this amendment's candidates "
                          "come from and which is not measured by either run"),
        },
    }


def check_references(sel, ref_root: Path, rerun_dev: Path, seed: int,
                     registry: Path | None, expect_manifest_sha: str | None,
                     published: dict | None = None,
                     expect_l2_sup_pp: float | None = None) -> dict:
    """The reference arms must be the FIXED, verified evaluations -- not
    regenerated beside the candidates.

    What went wrong before: the rerun regenerated `seed*_MA` and `seed*_MAB_L2`
    in its own output root, and the bridge compared each run against its OWN
    parent evaluation. A reference whose measured L2 suppression had shifted
    from 30 pp to 45 pp therefore passed: the target every dump is matched to
    had moved, and nothing looked at it. So the references are now read from one
    fixed root, their generating checkpoints are checked against the registered
    saved artifacts by digest, and a regenerated copy sitting beside the
    candidates is itself a violation.
    """
    out: dict = {"reference_dev_root": str(ref_root),
                 "candidate_dev_root": str(rerun_dev),
                 "references_are_separate_from_candidates":
                     str(ref_root) != str(rerun_dev)}
    problems, slots = [], {}
    registered = {}
    if registry and registry.is_file():
        for a in json.loads(registry.read_text()).get("artifacts", []):
            registered[a.get("id")] = a.get("delta_sha256")
    for slot in ("MA", "MAB_L2"):
        name = f"seed{seed}_{slot}"
        try:
            ck = sel.load_checkpoint(ref_root, name, role=f"fixed_reference_{slot}")
        except Exception as e:
            problems.append(f"{name}: {type(e).__name__}: {e}")
            continue
        b = ck["_binding"]
        rec = {"dog_residue_pct": ck["dog_residue_pct"],
               "dog_hits": ck["dog_hits"], "n_dog_pairs": ck["n_dog_pairs"],
               "detections_sha256": b["detections_sha256"],
               "image_report_sha256": b.get("image_report_sha256"),
               "generating_checkpoint_sha256": b.get("generating_checkpoint_sha256"),
               "manifest_sha256": b.get("manifest_sha256"),
               "generation_settings_source": b.get("generation_settings_source")}
        want_ck = registered.get(f"seed{seed}/{slot}")
        rec["registered_checkpoint_sha256"] = want_ck
        rec["generated_by_the_registered_saved_artifact"] = (
            bool(want_ck) and rec["generating_checkpoint_sha256"] == want_ck)
        if registered and not rec["generated_by_the_registered_saved_artifact"]:
            problems.append(
                f"{name}: generated by checkpoint "
                f"{str(rec['generating_checkpoint_sha256'])[:12]}…, which is not "
                f"the registered saved artifact "
                f"{str(want_ck)[:12]}… for seed{seed}/{slot}")
        if expect_manifest_sha and rec["manifest_sha256"] != expect_manifest_sha:
            problems.append(
                f"{name}: evaluated against manifest "
                f"{str(rec['manifest_sha256'])[:12]}…, not the frozen "
                f"development manifest {expect_manifest_sha[:12]}…")
        # Bound to the PUBLISHED, verified original. This is what catches a
        # reference whose MEASUREMENTS were rewritten while its checkpoint and
        # manifest still look right -- the 30 pp -> 45 pp shift.
        if published is not None:
            pub = published.get(name)
            if pub is None:
                problems.append(f"{name}: absent from the published slot index, "
                                f"so it is not one of the verified original "
                                f"evaluations")
            else:
                rec["published_detections_sha256"] = \
                    (pub.get("detections.jsonl") or {}).get("sha256")
                rec["published_image_report_sha256"] = \
                    (pub.get("image_report.json") or {}).get("sha256")
                rec["matches_published_index"] = (
                    rec["detections_sha256"] == rec["published_detections_sha256"]
                    and rec["image_report_sha256"]
                    == rec["published_image_report_sha256"])
                if not rec["matches_published_index"]:
                    problems.append(
                        f"{name}: its measurements differ from the published, "
                        f"verified original (detections "
                        f"{rec['detections_sha256'][:12]}… vs "
                        f"{str(rec['published_detections_sha256'])[:12]}…). A "
                        f"rewritten reference evaluation is not the fixed "
                        f"reference, whatever its checkpoint says.")
        slots[slot] = rec
    # A regenerated reference beside the candidates must not exist, even unused:
    # it is the thing that silently substituted itself last time.
    if str(ref_root) != str(rerun_dev):
        # Exact names: `seed17_MA*` would also match `seed17_MAB_L2`.
        intruders = sorted({d.name for d in rerun_dev.iterdir()
                            if d.is_dir() and d.name in
                            (f"seed{seed}_MA", f"seed{seed}_MAB_L2")}
                           ) if rerun_dev.is_dir() else []
        out["regenerated_references_beside_candidates"] = intruders
        if intruders:
            problems.append(
                f"the candidate root holds its own reference evaluation(s) "
                f"{intruders}: regenerated references must not sit beside the "
                f"candidates, because selection or a later bridge could read "
                f"them instead of the fixed ones")
    # The match TARGET itself, checked against the declared value. The gates
    # and the tolerance are meaningless if the number they are applied to can
    # drift: a reference measuring 45 pp instead of the declared 30 pp would
    # redefine the whole comparison.
    if expect_l2_sup_pp is not None and "MA" in slots and "MAB_L2" in slots:
        measured = round(slots["MA"]["dog_residue_pct"]
                         - slots["MAB_L2"]["dog_residue_pct"], 4)
        out["declared_L2_suppression_pp"] = expect_l2_sup_pp
        out["measured_L2_suppression_pp"] = measured
        out["L2_target_matches_declared"] = abs(measured - expect_l2_sup_pp) < 0.01
        if not out["L2_target_matches_declared"]:
            problems.append(
                f"the fixed L2 reference now measures {measured} pp "
                f"suppression, but the amendment declared "
                f"{expect_l2_sup_pp} pp. The match target has MOVED; nothing "
                f"may be selected or tested against it.")
    out["slots"] = slots
    out["problems"] = problems
    out["ok"] = not problems and len(slots) == 2
    out["rule"] = ("MA and MAB_L2 are read from one fixed, separately verified "
                   "evaluation; their generating checkpoints must be the "
                   "registered saved artifacts and their manifest the frozen "
                   "development manifest")
    return out


def compare_behaviour(sel, ref_root: Path, orig_dev: Path, rerun_dev: Path,
                      seed: int, step: int, tol_pp: float) -> dict:
    # ONE fixed MA for both sides. Measuring each run against its own
    # regenerated parent is what let the target move.
    ma = sel.load_checkpoint(ref_root, f"seed{seed}_MA",
                             role="fixed_reference_parent")
    l2 = sel.load_checkpoint(ref_root, f"seed{seed}_MAB_L2",
                             role="fixed_reference_L2")

    def suppression(dev_root: Path, which: str) -> dict:
        ck = sel.load_checkpoint(dev_root, f"seed{seed}_U_step{step}",
                                 role=f"{which}_bridge_dump")
        return {"dev_root": str(dev_root),
                "fixed_MA_dog_residue_pct": ma["dog_residue_pct"],
                "dump_dog_residue_pct": ck["dog_residue_pct"],
                "dog_hits": ck["dog_hits"], "n_dog_pairs": ck["n_dog_pairs"],
                "dog_suppression_pp": round(ma["dog_residue_pct"]
                                            - ck["dog_residue_pct"], 4),
                "detections_sha256": ck["_binding"]["detections_sha256"]}

    o, r = suppression(orig_dev, "original"), suppression(rerun_dev, "rerun")
    diff = abs(o["dog_suppression_pp"] - r["dog_suppression_pp"])
    l2_sup = round(ma["dog_residue_pct"] - l2["dog_residue_pct"], 4)
    return {
        "fixed_reference": {
            "MA_dog_residue_pct": ma["dog_residue_pct"],
            "L2_dog_residue_pct": l2["dog_residue_pct"],
            "L2_dog_suppression_pp": l2_sup,
            "L2_gate_suppression_ge_30pp": l2_sup >= 30.0,
            "L2_gate_residue_le_60pct": l2["dog_residue_pct"] <= 60.0,
            "source": str(ref_root),
            "note": ("both realisations are measured against THIS parent "
                     "evaluation, not against their own; the match target is "
                     "this L2 suppression and it cannot move with a rescan"),
        },
        "original": o, "rerun": r,
        "difference_pp": round(diff, 4),
        "stop_condition_pp": tol_pp,
        "holds": bool(diff <= tol_pp + 1e-9),
        "rule": (f"declared before the measurement: if the two realisations' "
                 f"step-{step} dog suppression differs by more than {tol_pp} pp "
                 f"-- the same tolerance the match itself uses -- then run-to-run "
                 f"variation is of the same order as the quantity being matched, "
                 f"the early grid cannot be trusted to locate the match point, "
                 f"and the amendment stops and reports that instead"),
    }


def cmd_verify(record: Path) -> int:
    """Re-bind a stored bridge record to the bytes now on disk.

    The defect this closes: the wrapper trusted a stored `verdict` string. A
    bridge computed when the settings were right stayed "HELD" on disk after the
    learning rate was changed, and `test` was delegated on the strength of it.
    A verdict is only as good as the inputs it was computed from, so every input
    is digested at the time and re-checked here.
    """
    if not record.is_file():
        print(f"NO BRIDGE RECORD at {record}: the bridge has not been run.",
              file=sys.stderr)
        return 2
    d = json.loads(record.read_text())
    bad, checked = [], 0

    def cmp_file(path: str | None, want: str | None, label: str) -> None:
        nonlocal checked
        if not path or not want:
            bad.append(f"{label}: the record carries no binding for this input")
            return
        checked += 1
        p = Path(path)
        if not p.is_file():
            bad.append(f"{label}: {p} is gone")
        elif sha256_file(p) != want:
            bad.append(f"{label}: changed since the bridge was computed "
                       f"({want[:12]}… -> {sha256_file(p)[:12]}…)")

    for seed, rec in (d.get("per_seed") or {}).items():
        rb = rec.get("report_bindings") or {}
        cmp_file(rb.get("original_train_report"),
                 rb.get("original_train_report_sha256"),
                 f"seed {seed} original train_report")
        cmp_file(rb.get("rerun_train_report"),
                 rb.get("rerun_train_report_sha256"),
                 f"seed {seed} rerun train_report")
        w = rec.get("weights_at_bridge_step") or {}
        if w and "error" not in w:
            cmp_file(w.get("original_dump"), w.get("original_sha256"),
                     f"seed {seed} original step-{d.get('bridge_step')} dump")
            cmp_file(w.get("rerun_dump"), w.get("rerun_sha256"),
                     f"seed {seed} rerun step-{d.get('bridge_step')} dump")
        fr = rec.get("fixed_references") or {}
        for slot, r in (fr.get("slots") or {}).items():
            root = Path(fr.get("reference_dev_root", ""))
            name = f"seed{seed}_{slot}"
            cmp_file(str(root / name / "detections.jsonl"),
                     r.get("detections_sha256"),
                     f"seed {seed} fixed reference {slot} detections")
            cmp_file(str(root / name / "image_report.json"),
                     r.get("image_report_sha256"),
                     f"seed {seed} fixed reference {slot} image_report")
        b = rec.get("behaviour_at_bridge_step") or {}
        for which in ("original", "rerun"):
            side = b.get(which) or {}
            if side.get("dev_root"):
                cmp_file(str(Path(side["dev_root"])
                             / f"seed{seed}_U_step{d.get('bridge_step')}"
                             / "detections.jsonl"),
                         side.get("detections_sha256"),
                         f"seed {seed} {which} bridge-dump detections")

    print(f"bridge record : {record}")
    print(f"computed      : {d.get('generated_utc')}")
    print(f"verdict stored: {d.get('verdict')}")
    print(f"inputs bound  : {checked}")
    if d.get("verdict") != "BRIDGE_HELD":
        print(f"BRIDGE DID NOT HOLD ({d.get('verdict')}): nothing may proceed "
              f"on this record.", file=sys.stderr)
        return 1
    if not checked:
        print("UNVERIFIABLE: the record carries no input bindings, so it cannot "
              "be shown to describe the artifacts now on disk. Recompute the "
              "bridge.", file=sys.stderr)
        return 2
    if bad:
        print("BRIDGE RECORD IS STALE:")
        for m in bad:
            print(f"  - {m}")
        print("Recompute the bridge. A verdict computed from different bytes is "
              "not evidence about these ones.", file=sys.stderr)
        return 1
    print("BRIDGE BINDINGS OK: every input is byte-identical to the one the "
          "verdict was computed from")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original_root", required=True)
    ap.add_argument("--rerun_root", required=True)
    ap.add_argument("--seeds", default="17 29")
    ap.add_argument("--bridge_step", type=int, default=100)
    ap.add_argument("--tolerance_pp", type=float, default=5.0)
    ap.add_argument("--contract", default="configs/checkpoint_contract.json")
    ap.add_argument("--reference_dev_root", default=None,
                    help="the FIXED reference evaluations (MA, MAB_L2). Defaults "
                         "to <original_root>/eval_dev. Never the rerun's own "
                         "output: a regenerated reference lets the match target "
                         "move.")
    ap.add_argument("--registry", default="configs/legacy_training_artifacts.json",
                    help="registered saved artifacts, for checking that each "
                         "reference evaluation was generated by the registered "
                         "checkpoint")
    ap.add_argument("--expect_dev_manifest_sha", default=None)
    ap.add_argument("--expect_slots_index", default=None,
                    help="published per-slot index of the verified original "
                         "development evaluations; each reference's detections "
                         "and image-report digests must match it")
    ap.add_argument("--expect_l2_suppression_pp", default=None,
                    help="the DECLARED match target per seed, e.g. "
                         "\"17=30.0,29=33.75\". A reference that no longer "
                         "measures its declared suppression has moved the "
                         "target and is refused.")
    ap.add_argument("--declared_rerun_iterations", type=int, default=100)
    ap.add_argument("--declared_rerun_cadence", type=int, default=10)
    ap.add_argument("--declared_original_iterations", type=int, default=1000)
    ap.add_argument("--declared_original_cadence", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_weights", action="store_true",
                    help="settings and behaviour only (no torch needed)")
    ap.add_argument("--verify", default=None,
                    help="re-bind an existing bridge record to the bytes now on "
                         "disk and exit: 0 if every input is unchanged, 1 if the "
                         "record is stale, 2 if it cannot be bound at all")
    a = ap.parse_args()

    if a.verify:
        return cmd_verify(Path(a.verify))

    sel = _load_selection_helpers()
    contract = json.loads(Path(a.contract).read_text())
    O, R = Path(a.original_root), Path(a.rerun_root)
    REF = Path(a.reference_dev_root) if a.reference_dev_root else O / "eval_dev"
    registry = Path(a.registry) if a.registry else None
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    step = a.bridge_step
    published = None
    if a.expect_slots_index:
        ip = Path(a.expect_slots_index)
        if ip.is_file():
            published = {row.get("slot"): row
                         for row in json.loads(ip.read_text()).get("slots", [])}
        else:
            published = {}
    want_l2 = {}
    if a.expect_l2_suppression_pp:
        for part in a.expect_l2_suppression_pp.replace(" ", "").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                want_l2[int(k)] = float(v)
    declared = {"original_iterations": a.declared_original_iterations,
                "rerun_iterations": a.declared_rerun_iterations,
                "original_cadence": a.declared_original_cadence,
                "rerun_cadence": a.declared_rerun_cadence,
                "bridge_step": step, "tolerance_pp": a.tolerance_pp,
                "seeds": tuple(seeds),
                "declared_L2_suppression_pp": want_l2 or None}

    per_seed, problems = {}, []
    for s in seeds:
        rec: dict = {"training_seed": s}
        o_rep = O / "models" / f"seed{s}" / "U" / "train_report.json"
        r_rep = R / "models" / f"seed{s}" / "U" / "train_report.json"
        try:
            rec["settings"] = compare_settings(
                json.loads(o_rep.read_text()), json.loads(r_rep.read_text()),
                step, declared, s)
            rec["report_bindings"] = {
                "original_train_report": str(o_rep),
                "original_train_report_sha256": sha256_file(o_rep),
                "rerun_train_report": str(r_rep),
                "rerun_train_report_sha256": sha256_file(r_rep),
            }
        except Exception as e:
            rec["settings"] = {"error": f"{type(e).__name__}: {e}", "settings_ok": False}
            problems.append(f"seed {s} settings: {type(e).__name__}: {e}")
        try:
            # Note what is NOT done here: reference problems are deliberately
            # kept out of `problems`. `problems` means "the bridge could not be
            # evaluated" (INCOMPLETE); an invalid or moved reference IS
            # evaluable and has its own, more specific verdict.
            rec["fixed_references"] = check_references(
                sel, REF, R / "eval_dev", s, registry, a.expect_dev_manifest_sha,
                published, want_l2.get(s))
        except Exception as e:
            rec["fixed_references"] = {"error": f"{type(e).__name__}: {e}",
                                       "ok": False}
            problems.append(f"seed {s} reference: {type(e).__name__}: {e}")
        if not a.skip_weights:
            try:
                rec["weights_at_bridge_step"] = compare_weights(
                    O / "models" / f"seed{s}" / "U" / f"delta-{step}",
                    R / "models" / f"seed{s}" / "U" / f"delta-{step}", contract)
            except Exception as e:
                rec["weights_at_bridge_step"] = {"error": f"{type(e).__name__}: {e}"}
                problems.append(f"seed {s} weights: {type(e).__name__}: {e}")
        try:
            rec["behaviour_at_bridge_step"] = compare_behaviour(
                sel, REF, O / "eval_dev", R / "eval_dev", s, step, a.tolerance_pp)
        except Exception as e:
            rec["behaviour_at_bridge_step"] = {
                "error": f"{type(e).__name__}: {e}", "holds": False}
            problems.append(f"seed {s} behaviour: {type(e).__name__}: {e}")
        per_seed[s] = rec

    settings_ok = all((p.get("settings") or {}).get("settings_ok")
                      for p in per_seed.values())
    references_ok = all((p.get("fixed_references") or {}).get("ok")
                        for p in per_seed.values())
    behaviour_ok = all((p.get("behaviour_at_bridge_step") or {}).get("holds")
                       for p in per_seed.values())
    all_bitwise = all((p.get("weights_at_bridge_step") or {}).get("all_bitwise_identical")
                      for p in per_seed.values()) if not a.skip_weights else None

    if problems:
        verdict, why = "INCOMPLETE", (
            "the bridge could not be evaluated; nothing follows from it and the "
            "early grid must not be used for selection: " + "; ".join(problems[:4]))
    elif not references_ok:
        verdict, why = "REFERENCES_INVALID", (
            "the fixed reference evaluations (MA and MAB_L2) are not the "
            "verified ones, or a regenerated copy sits beside the candidates. "
            "The match target is defined by those evaluations, so nothing may "
            "be selected or tested until they are the registered, frozen ones.")
    elif not settings_ok:
        verdict, why = "SETTINGS_VIOLATED", (
            "the rerun changed something other than the declared stopping limit "
            "and save cadence, so its early dumps are not a shorter realisation "
            "of the same configuration. This is a defect in the rerun, not a "
            "finding about training.")
    elif not behaviour_ok:
        verdict, why = "BRIDGE_FAILED", (
            f"the two realisations' step-{step} development dog suppression "
            f"differs by more than {a.tolerance_pp} pp, so run-to-run variation "
            f"is of the same order as the quantity being matched. By the "
            f"condition declared before the measurement, the early grid is not "
            f"trusted to locate a match point: the amendment STOPS here and "
            f"reports this instead. No selection is made and the frozen test set "
            f"is not evaluated.")
    else:
        verdict, why = "BRIDGE_HELD", (
            f"settings that determine the first {step} updates are identical, "
            f"the only changes are the declared stopping limit and save cadence, "
            f"and the two realisations' step-{step} development dog suppression "
            f"agrees within {a.tolerance_pp} pp. Selection on the early grid may "
            f"proceed."
            + ("" if all_bitwise else
               " The step-100 weights are NOT bitwise identical, so the early "
               "dumps are a second realisation rather than the original path; "
               "the distance is recorded and its cause is not isolated."))

    payload = {
        "_what_this_is": (
            "The bridge between the original 1000-step run and the short "
            "early-grid rerun, measured at the one step they share. Three "
            "questions -- settings, weights, behaviour -- answered separately."),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "original_root": str(O), "rerun_root": str(R),
        "bridge_step": step, "tolerance_pp": a.tolerance_pp,
        "settings_preserved": settings_ok,
        "fixed_references_verified": references_ok,
        "behaviour_within_tolerance": behaviour_ok,
        "weights_bitwise_identical": all_bitwise,
        "verdict": verdict,
        "reason": why,
        "declared_contract": declared,
        "not_established": [
            "That the two runs passed through the same states at steps 1..99. "
            "Only step 100 is compared; bitwise identity THERE would not prove "
            "identity along the path to it, and the amendment's candidates come "
            "from the unmeasured interior.",
            "WHY any weight difference exists. Identical recorded configuration "
            "and the same saved parent do not isolate nondeterminism as the "
            "cause: the runs also differ in library and driver state, in the "
            "periodic-checkpointing code path at a different cadence, in "
            "dataloader ordering, and in anything outside the recorded "
            "hyperparameters. No controlled repeat was run.",
            "That either realisation's step-k state is THE state of this "
            "configuration at step k. Run-to-run variation is measured at this "
            "one step; it is not characterised across the early grid.",
            "Anything about retention. This tool measures dog suppression only "
            "and never touches the frozen test set.",
        ],
        "per_seed": {str(s): per_seed[s] for s in seeds},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    for s in seeds:
        p = per_seed[s]
        st = p.get("settings") or {}
        fu = st.get("settings_that_determine_the_first_updates") or {}
        print(f"=== seed {s} ===")
        print(f"  first-{step}-update settings identical : "
              f"{fu.get('n_identical')}/{fu.get('n_compared')} "
              f"(violations: {list((fu.get('violations') or {}).keys()) or 'none'})")
        print(f"  declared changes                      : "
              f"{list(((st.get('declared_changes') or {}).get('stopping_limit_and_derived') or {}).keys())}"
              f" + save cadence "
              f"{((st.get('declared_changes') or {}).get('save_cadence') or {})}")
        fr = p.get("fixed_references") or {}
        if fr:
            print(f"  fixed references verified             : {fr.get('ok')}"
                  + (f" (L2 target {fr.get('measured_L2_suppression_pp')} pp"
                     f" vs declared {fr.get('declared_L2_suppression_pp')} pp)"
                     if fr.get("declared_L2_suppression_pp") is not None else ""))
            for m in (fr.get("problems") or []):
                print(f"    REFERENCE PROBLEM: {m}")
        hf = st.get("lr_schedule_is_horizon_independent") or {}
        print(f"  lr schedule horizon-independent       : {hf.get('horizon_free')} "
              f"({hf.get('rerun_lr_scheduler')})")
        w = p.get("weights_at_bridge_step") or {}
        if w and "error" not in w:
            print(f"  step-{step} weights bitwise identical    : "
                  f"{w.get('all_bitwise_identical')} "
                  f"({w.get('bitwise_identical_tensors')}/{w.get('tensors_compared')}, "
                  f"max abs {w.get('max_abs_diff'):.3e})")
        b = p.get("behaviour_at_bridge_step") or {}
        if "error" not in b:
            print(f"  step-{step} dog suppression              : original "
                  f"{b['original']['dog_suppression_pp']} pp vs rerun "
                  f"{b['rerun']['dog_suppression_pp']} pp -> "
                  f"{b['difference_pp']} pp (<= {a.tolerance_pp}: {b['holds']})")
    print(f"\nVERDICT: {verdict}")
    print(why)
    print(f"-> {out}")
    return 0 if verdict == "BRIDGE_HELD" else 1


if __name__ == "__main__":
    raise SystemExit(main())
