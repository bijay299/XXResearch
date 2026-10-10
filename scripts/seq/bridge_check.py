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
     Frobenius norm. Bitwise identity would mean the first 100 updates were
     reproduced exactly and the early dumps lie on the original path. Anything
     else is a measured distance, and its CAUSE is not isolated by this tool.

  C  BEHAVIOUR at step 100. Development dog suppression (vs that seed's own MA,
     t=0.5, the same 80 pairs) in each run, and the difference in pp against the
     amendment's declared stop condition: if the two realisations differ by more
     than the match tolerance itself, run-to-run variation is of the same order
     as the quantity being matched, the early grid cannot be trusted to locate a
     match point, and the amendment stops and reports that instead.

What this tool does NOT do
--------------------------
* It does not attribute any difference it finds to a cause. Identical RECORDED
  configuration plus the same parent does not isolate nondeterminism: the two
  runs also differ in library/driver state, upstream code path (periodic
  checkpointing at a different cadence), unrecorded dataloader ordering, and
  anything else outside the 32 recorded hyperparameters. No controlled repeat
  was run and none is requested here.
* It does not establish that either realisation's step-k state is "the" state
  of that configuration at step k.
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


def compare_settings(orig: dict, rerun: dict, bridge_step: int) -> dict:
    eo = orig.get("effective_hyperparameters") or {}
    er = rerun.get("effective_hyperparameters") or {}
    req_diff = {k: {"original": orig.get(k), "rerun": rerun.get(k)}
                for k in REQUEST_KEYS if orig.get(k) != rerun.get(k)}

    preserved, violated = [], {}
    for k in FIRST_UPDATES_KEYS:
        if k not in eo and k not in er:
            continue
        if eo.get(k) == er.get(k):
            preserved.append({"setting": k, "value": eo.get(k)})
        else:
            violated[k] = {"original": eo.get(k), "rerun": er.get(k)}

    declared, undeclared_same = {}, []
    for k in DECLARED_CHANGE_KEYS:
        if eo.get(k) != er.get(k):
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

    return {
        "request_field_differences": req_diff,
        "settings_that_determine_the_first_updates": {
            "n_compared": len(preserved) + len(violated),
            "n_identical": len(preserved),
            "violations": violated,
            "all_preserved": not violated and not req_diff,
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
        "settings_ok": bool(not violated and not req_diff
                            and not unclassified_diff and horizon_free),
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
            "bitwise identity would mean the first updates were reproduced "
            "exactly, so the early dumps lie on the original path. A difference "
            "is a measured distance ONLY: this comparison cannot say whether it "
            "came from nondeterministic kernels, the changed save cadence, "
            "library or driver state, dataloader ordering, or anything else "
            "outside the recorded hyperparameters."),
    }


def compare_behaviour(sel, orig_dev: Path, rerun_dev: Path, seed: int,
                      step: int, tol_pp: float) -> dict:
    def suppression(dev_root: Path) -> dict:
        ma = sel.load_checkpoint(dev_root, f"seed{seed}_MA")
        ck = sel.load_checkpoint(dev_root, f"seed{seed}_U_step{step}")
        return {"MA_dog_residue_pct": ma["dog_residue_pct"],
                "dump_dog_residue_pct": ck["dog_residue_pct"],
                "dog_hits": ck["dog_hits"], "n_dog_pairs": ck["n_dog_pairs"],
                "dog_suppression_pp": round(ma["dog_residue_pct"]
                                            - ck["dog_residue_pct"], 4)}

    o, r = suppression(orig_dev), suppression(rerun_dev)
    diff = abs(o["dog_suppression_pp"] - r["dog_suppression_pp"])
    return {
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original_root", required=True)
    ap.add_argument("--rerun_root", required=True)
    ap.add_argument("--seeds", default="17 29")
    ap.add_argument("--bridge_step", type=int, default=100)
    ap.add_argument("--tolerance_pp", type=float, default=5.0)
    ap.add_argument("--contract", default="configs/checkpoint_contract.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_weights", action="store_true",
                    help="settings and behaviour only (no torch needed)")
    a = ap.parse_args()

    sel = _load_selection_helpers()
    contract = json.loads(Path(a.contract).read_text())
    O, R = Path(a.original_root), Path(a.rerun_root)
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    step = a.bridge_step

    per_seed, problems = {}, []
    for s in seeds:
        rec: dict = {"training_seed": s}
        o_rep = O / "models" / f"seed{s}" / "U" / "train_report.json"
        r_rep = R / "models" / f"seed{s}" / "U" / "train_report.json"
        try:
            rec["settings"] = compare_settings(
                json.loads(o_rep.read_text()), json.loads(r_rep.read_text()), step)
        except Exception as e:
            rec["settings"] = {"error": f"{type(e).__name__}: {e}", "settings_ok": False}
            problems.append(f"seed {s} settings: {type(e).__name__}: {e}")
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
                sel, O / "eval_dev", R / "eval_dev", s, step, a.tolerance_pp)
        except Exception as e:
            rec["behaviour_at_bridge_step"] = {
                "error": f"{type(e).__name__}: {e}", "holds": False}
            problems.append(f"seed {s} behaviour: {type(e).__name__}: {e}")
        per_seed[s] = rec

    settings_ok = all((p.get("settings") or {}).get("settings_ok")
                      for p in per_seed.values())
    behaviour_ok = all((p.get("behaviour_at_bridge_step") or {}).get("holds")
                       for p in per_seed.values())
    all_bitwise = all((p.get("weights_at_bridge_step") or {}).get("all_bitwise_identical")
                      for p in per_seed.values()) if not a.skip_weights else None

    if problems:
        verdict, why = "INCOMPLETE", (
            "the bridge could not be evaluated; nothing follows from it and the "
            "early grid must not be used for selection: " + "; ".join(problems[:4]))
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
        "behaviour_within_tolerance": behaviour_ok,
        "weights_bitwise_identical": all_bitwise,
        "verdict": verdict,
        "reason": why,
        "not_established": [
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
