#!/usr/bin/env python3
"""CPU tests for scripts/seq/bridge_check.py and the selection bindings.

Three defects the coordinator reproduced are pinned down here, each with the
exact shape that got through before:

  1. STALE BRIDGE. A bridge computed while the settings were right stayed
     "BRIDGE_HELD" on disk after the learning rate changed, and the wrapper
     delegated `test` on the strength of the stored verdict. Now every record
     carries a digest of every input and `--verify` re-binds it.
  2. MOVING REFERENCE TARGET. The rerun regenerated MA/MAB_L2 beside its
     candidates and each run was measured against its OWN parent, so an L2
     reference shifted from 30 pp to 45 pp still passed. Now the references are
     read from one fixed root, checked against the registered saved artifacts,
     and a regenerated copy beside the candidates is a violation.
  3. SETTINGS CONTRACT. Missing fields compared equal (None == None), and any
     difference in horizon or cadence was labelled "declared". Now absence is a
     violation and the declared changes have declared VALUES.

Plus the paths that must still pass and fail: horizon-dependent schedules,
unclassified settings, behaviour drift, a missing dump, one-seed failure, and
the successful gated path.

Real torch on CPU for the tensor comparison; no GPU, no model, no generation.

    python scripts/seq/test_bridge_check.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "seq" / "bridge_check.py"
SELECT = REPO / "scripts" / "seq" / "select_matched_dump.py"
CONTRACT = {"contract": "TEST", "top_level_key": "unet", "n_tensors": 2,
            "total_elements": 12, "dtype": "torch.float32",
            "tensors": {"a.attn2.to_k.weight": {"shape": [2, 3]},
                        "a.attn2.to_v.weight": {"shape": [2, 3]}}}
MAN_SHA = "a" * 64
PARENT = {17: "p17" + "0" * 61, 29: "p29" + "0" * 61}
CKPT = {(17, "MA"): "c17ma" + "0" * 59, (17, "MAB_L2"): "c17l2" + "0" * 59,
        (29, "MA"): "c29ma" + "0" * 59, (29, "MAB_L2"): "c29l2" + "0" * 59}
PASS, FAIL = [], []


def check(desc: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(desc)
    print(f"  {'ok  ' if cond else 'FAIL'}  {desc}" + ("" if cond else f"  <- {detail}"))


BASE_HP = {
    "effective_learning_rate": 8e-06,
    "learning_rate_cli_default_before_scaling": 2e-06,
    "scale_lr": True, "scale_lr_formula": "lr * ga * bs * np",
    "effective_lr_cross_check": 8e-06,
    "anchor_batch_size": 4, "gradient_accumulation_steps": 1,
    "num_processes": 1, "optimizer": "AdamW", "adam_beta1": 0.9,
    "adam_beta2": 0.999, "adam_weight_decay": 0.01, "adam_epsilon": 1e-08,
    "max_grad_norm": 1.0, "lr_scheduler": "constant", "lr_warmup_steps": 500,
    "precision": "fp32", "parameter_group": "kv-xattn",
    "with_anchor_preservation": False, "l1sp_weight": 0.0, "l2sp_weight": 0.0,
    "with_gradient_projection": False, "with_selft": False,
    "eval_interval": None, "seed": 17, "hflip": True, "noaug": True,
    "resolution": 512, "steps_per_epoch": 50,
}


def write_registry(path: Path) -> None:
    path.write_text(json.dumps({"artifacts": [
        {"id": f"seed{s}/{slot}", "checkpoint": slot, "delta_sha256": CKPT[(s, slot)]}
        for s in (17, 29) for slot in ("MA", "MAB_L2")]}, indent=2))


def write_run(root: Path, seed: int, iters: int, cadence: int, fill: float,
              hp_override: dict | None = None, write_dump: bool = True,
              parent: str | None = None, steps_done: int | None = None) -> None:
    import torch
    d = root / "models" / f"seed{seed}" / "U"
    d.mkdir(parents=True, exist_ok=True)
    hp = {**BASE_HP, "seed": seed, "iterations_requested": iters,
          "epochs_cap": -(-iters // 50),
          "epoch_capacity_steps": 50 * (-(-iters // 50))}
    hp.update(hp_override or {})
    steps = list(range(cadence, iters + 1, cadence))
    if write_dump:
        sd = {k: torch.full(tuple(v["shape"]), fill)
              for k, v in CONTRACT["tensors"].items()}
        torch.save({"unet": sd}, d / "delta-100")
    (d / "train_report.json").write_text(json.dumps({
        "checkpoint": "U", "parent": "MA", "new_deletion_target": "dog",
        "anchor_concept": "horse", "anchor_target_mapping": "horse+dog",
        "l2sp_weight": 0.0, "training_seed": seed,
        "effective_hyperparameters": hp,
        "parent_verification": {
            "parent_sha256": parent or PARENT.get(seed, f"p{seed}".ljust(64, "0")),
            "PASS": True},
        "dumps": {"checkpoint_every": cadence, "expected_steps": steps,
                  "n_expected": len(steps), "n_present": len(steps),
                  "complete": True},
        "runtime": {"optimizer_steps_completed":
                    iters if steps_done is None else steps_done},
    }, indent=2))


def write_slot(dev_root: Path, name: str, hits: int, n: int = 80,
               ckpt_sha: str = "deadbeef", man_sha: str = MAN_SHA) -> None:
    d = dev_root / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "detections.jsonl").open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "checkpoint": name, "category": "dog", "image_name": f"{i}.png",
                "hit": {"0.3": True, "0.5": i < hits, "0.7": False}}) + "\n")
    (d / "image_report.json").write_text(json.dumps({
        "checkpoint": name, "complete": True, "generated_or_reused": n,
        "manifest_sha256": man_sha,
        "generation_settings_source": "contract configs/generation_settings.json",
        "checkpoint_load": {"applied": True, "sha256": ckpt_sha,
                            "unet_ckpt": f"/saved/{name}/delta.bin"},
    }, indent=2))


def write_references(ref_root: Path, seed: int, ma_hits: int,
                     l2_hits: int) -> None:
    write_slot(ref_root, f"seed{seed}_MA", ma_hits, ckpt_sha=CKPT[(seed, "MA")])
    write_slot(ref_root, f"seed{seed}_MAB_L2", l2_hits,
               ckpt_sha=CKPT[(seed, "MAB_L2")])


def run(orig: Path, rerun: Path, out: Path, contract: Path, registry: Path,
        ref: Path, seeds: str = "17", extra: list[str] | None = None
        ) -> tuple[int, dict, str]:
    p = subprocess.run(
        [sys.executable, str(TOOL), "--original_root", str(orig),
         "--rerun_root", str(rerun), "--seeds", seeds, "--bridge_step", "100",
         "--contract", str(contract), "--registry", str(registry),
         "--reference_dev_root", str(ref),
         "--expect_dev_manifest_sha", MAN_SHA,
         "--out", str(out), *(extra or [])],
        capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    j = json.loads(out.read_text()) if out.is_file() else {}
    return p.returncode, j, p.stdout + p.stderr


def verify(out: Path) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, str(TOOL), "--verify", str(out), "--original_root", "x",
         "--rerun_root", "y", "--out", str(out)],
        capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bridge_"))
    cpath, reg = tmp / "contract.json", tmp / "registry.json"
    cpath.write_text(json.dumps(CONTRACT))
    write_registry(reg)
    try:
        # ---------------------------------------------------------------- held
        print("\nthe gated path: only the declared changes, fixed references")
        O, R, REF = tmp / "orig", tmp / "rerun", tmp / "orig" / "eval_dev"
        write_run(O, 17, 1000, 100, fill=0.25)
        write_run(R, 17, 100, 10, fill=0.25)
        write_references(REF, 17, ma_hits=66, l2_hits=42)      # L2 = 30.0 pp
        write_slot(O / "eval_dev", "seed17_U_step100", 35)     # 38.75 pp
        write_slot(R / "eval_dev", "seed17_U_step100", 35)
        b1 = tmp / "b1.json"
        rc, j, out = run(O, R, b1, cpath, reg, REF)
        check("the bridge HOLDS", j.get("verdict") == "BRIDGE_HELD", str(j.get("verdict")))
        check("and that is the only verdict that exits 0", rc == 0, f"rc={rc}")
        st = j["per_seed"]["17"]["settings"]
        check("the declared contract is satisfied by VALUE",
              st["declared_contract"]["ok"] is True,
              str(st["declared_contract"]["violations"]))
        check("the parent digest is checked and shared",
              st["parent_identity"]["same_parent"] is True)
        fr = j["per_seed"]["17"]["fixed_references"]
        check("the references are verified against the registry",
              fr["ok"] is True
              and all(v["generated_by_the_registered_saved_artifact"]
                      for v in fr["slots"].values()), str(fr.get("problems")))
        check("and are separate from the candidate root",
              fr["references_are_separate_from_candidates"] is True
              and fr["regenerated_references_beside_candidates"] == [])
        b = j["per_seed"]["17"]["behaviour_at_bridge_step"]
        check("both realisations are measured against the SAME fixed MA",
              b["original"]["fixed_MA_dog_residue_pct"]
              == b["rerun"]["fixed_MA_dog_residue_pct"] == 82.5, str(b))
        check("the fixed L2 target is reported with its gates",
              b["fixed_reference"]["L2_dog_suppression_pp"] == 30.0
              and b["fixed_reference"]["L2_gate_suppression_ge_30pp"] is True)
        check("the record says what one step cannot show",
              "would_not" in (j["per_seed"]["17"]["weights_at_bridge_step"]
                              ["what_identity_here_would_and_would_not_show"]))
        check("and lists the unmeasured interior as not established",
              any("steps 1..99" in s for s in j["not_established"]),
              str(j["not_established"])[:120])

        print("\nthe record binds to its inputs, and goes STALE when they change")
        rc, out = verify(b1)
        check("a fresh record verifies", rc == 0, out.strip()[-160:])
        check("and says how many inputs it bound", "inputs bound" in out)
        # Change the learning rate AFTER the bridge held -- the reproduced defect.
        write_run(R, 17, 100, 10, fill=0.25,
                  hp_override={"effective_learning_rate": 1.6e-05})
        rc, out = verify(b1)
        check("changing the learning rate afterwards makes the record STALE",
              rc == 1 and "STALE" in out, out.strip()[-200:])
        check("the stale input is named", "rerun train_report" in out)
        check("and recomputation is demanded", "Recompute the bridge" in out)
        # Recomputing now refuses, as it must.
        rc, j2, out2 = run(O, R, tmp / "b1b.json", cpath, reg, REF)
        check("recomputing after the change refuses",
              j2["verdict"] == "SETTINGS_VIOLATED", str(j2["verdict"]))
        write_run(R, 17, 100, 10, fill=0.25)   # restore
        # A changed reference also invalidates the record.
        rc, j, out = run(O, R, b1, cpath, reg, REF)
        write_slot(REF, "seed17_MAB_L2", 30, ckpt_sha=CKPT[(17, "MAB_L2")])
        rc, out = verify(b1)
        check("a changed fixed reference also makes it stale",
              rc == 1 and "fixed reference MAB_L2" in out, out.strip()[-220:])
        write_references(REF, 17, ma_hits=66, l2_hits=42)   # restore

        # ------------------------------------------------- moving the target
        print("\nthe MOVING REFERENCE TARGET: an L2 shift from 30 pp to 45 pp")
        REF2 = tmp / "ref_shifted"
        write_slot(REF2, "seed17_MA", 66, ckpt_sha=CKPT[(17, "MA")])
        # 45 pp instead of 30 pp, and generated by something else.
        write_slot(REF2, "seed17_MAB_L2", 30, ckpt_sha="notregistered" + "0" * 50)
        rc, j, out = run(O, R, tmp / "b2.json", cpath, reg, REF2)
        check("a reference not generated by the registered artifact is REJECTED",
              j["verdict"] == "REFERENCES_INVALID", str(j["verdict"]))
        check("the offending slot and digests are named",
              any("not\n" not in m and "seed17_MAB_L2" in m
                  for m in j["per_seed"]["17"]["fixed_references"]["problems"]),
              str(j["per_seed"]["17"]["fixed_references"]["problems"]))
        check("and the verdict explains that the target defines the comparison",
              "match target is defined by those evaluations" in j["reason"])
        # A reference evaluated against a different manifest.
        REF3 = tmp / "ref_manifest"
        write_slot(REF3, "seed17_MA", 66, ckpt_sha=CKPT[(17, "MA")])
        write_slot(REF3, "seed17_MAB_L2", 42, ckpt_sha=CKPT[(17, "MAB_L2")],
                   man_sha="b" * 64)
        rc, j, out = run(O, R, tmp / "b3.json", cpath, reg, REF3)
        check("a reference scored on another manifest is REJECTED",
              j["verdict"] == "REFERENCES_INVALID"
              and any("manifest" in m for m in
                      j["per_seed"]["17"]["fixed_references"]["problems"]),
              str(j["per_seed"]["17"]["fixed_references"]["problems"]))
        # A regenerated reference sitting beside the candidates.
        write_slot(R / "eval_dev", "seed17_MAB_L2", 30,
                   ckpt_sha=CKPT[(17, "MAB_L2")])
        rc, j, out = run(O, R, tmp / "b4.json", cpath, reg, REF)
        check("a regenerated reference beside the candidates is REJECTED",
              j["verdict"] == "REFERENCES_INVALID"
              and j["per_seed"]["17"]["fixed_references"][
                  "regenerated_references_beside_candidates"] == ["seed17_MAB_L2"],
              str(j["per_seed"]["17"]["fixed_references"].get("problems")))
        shutil.rmtree(R / "eval_dev" / "seed17_MAB_L2")

        # ------------------------------------------------- settings contract
        print("\nthe SETTINGS CONTRACT: missing fields, and undeclared values")
        R5 = tmp / "rerun_missing"
        hp_missing = {k: v for k, v in BASE_HP.items() if k != "max_grad_norm"}
        d = R5 / "models" / "seed17" / "U"
        write_run(R5, 17, 100, 10, fill=0.25)
        rep = json.loads((d / "train_report.json").read_text())
        rep["effective_hyperparameters"] = {
            **{k: v for k, v in rep["effective_hyperparameters"].items()
               if k != "max_grad_norm"}}
        (d / "train_report.json").write_text(json.dumps(rep, indent=2))
        write_slot(R5 / "eval_dev", "seed17_U_step100", 35)
        rc, j, out = run(O, R5, tmp / "b5.json", cpath, reg, REF)
        check("a MISSING required setting is a violation, not a match",
              j["verdict"] == "SETTINGS_VIOLATED"
              and "max_grad_norm" in j["per_seed"]["17"]["settings"][
                  "settings_that_determine_the_first_updates"][
                  "missing_required_fields"], str(j["verdict"]))
        # An undeclared horizon.
        R6 = tmp / "rerun_250"
        write_run(R6, 17, 250, 10, fill=0.25)
        write_slot(R6 / "eval_dev", "seed17_U_step100", 35)
        rc, j, out = run(O, R6, tmp / "b6.json", cpath, reg, REF)
        cv = j["per_seed"]["17"]["settings"]["declared_contract"]["violations"]
        check("a 250-step horizon is NOT a declared change",
              j["verdict"] == "SETTINGS_VIOLATED"
              and cv["rerun_iterations"] == {"actual": 250, "declared": 100},
              str(cv))
        # An undeclared cadence.
        R7 = tmp / "rerun_cad25"
        write_run(R7, 17, 100, 25, fill=0.25)
        write_slot(R7 / "eval_dev", "seed17_U_step100", 35)
        rc, j, out = run(O, R7, tmp / "b7.json", cpath, reg, REF)
        cv = j["per_seed"]["17"]["settings"]["declared_contract"]["violations"]
        check("a cadence of 25 is NOT a declared change",
              j["verdict"] == "SETTINGS_VIOLATED" and "rerun_cadence" in cv,
              str(cv))
        # A run that stopped short.
        R8 = tmp / "rerun_short"
        write_run(R8, 17, 100, 10, fill=0.25, steps_done=60)
        write_slot(R8 / "eval_dev", "seed17_U_step100", 35)
        rc, j, out = run(O, R8, tmp / "b8.json", cpath, reg, REF)
        cv = j["per_seed"]["17"]["settings"]["declared_contract"]["violations"]
        check("a trajectory that stopped at 60 steps is rejected",
              j["verdict"] == "SETTINGS_VIOLATED"
              and "rerun_optimizer_steps_completed" in cv, str(cv))
        # A different parent.
        R9 = tmp / "rerun_parent"
        write_run(R9, 17, 100, 10, fill=0.25, parent="z" * 64)
        write_slot(R9 / "eval_dev", "seed17_U_step100", 35)
        rc, j, out = run(O, R9, tmp / "b9.json", cpath, reg, REF)
        check("a different parent digest is rejected",
              j["verdict"] == "SETTINGS_VIOLATED"
              and j["per_seed"]["17"]["settings"]["parent_identity"][
                  "same_parent"] is False)
        check("the environment caveat is stated, not overclaimed",
              any("library, driver" in m for m in j["per_seed"]["17"]["settings"]
                  ["what_this_comparison_cannot_see"]))

        print("\na horizon-DEPENDENT learning-rate schedule")
        R10 = tmp / "rerun_cosine"
        write_run(R10, 17, 100, 10, fill=0.25,
                  hp_override={"lr_scheduler": "cosine"})
        write_slot(R10 / "eval_dev", "seed17_U_step100", 35)
        rc, j, out = run(O, R10, tmp / "b10.json", cpath, reg, REF)
        hz = j["per_seed"]["17"]["settings"]["lr_schedule_is_horizon_independent"]
        check("shortening the run under a cosine schedule is refused",
              j["verdict"] == "SETTINGS_VIOLATED" and hz["horizon_free"] is False)
        check("and the reason explains what that would change",
              "would change the learning rate at every one of the first 100"
              in hz["why_it_matters"])

        print("\nan UNCLASSIFIED hyperparameter differing silently")
        O11, R11 = tmp / "orig_unk", tmp / "rerun_unk"
        write_run(O11, 17, 1000, 100, fill=0.25,
                  hp_override={"some_new_knob": "original"})
        write_run(R11, 17, 100, 10, fill=0.25,
                  hp_override={"some_new_knob": "changed"})
        write_slot(O11 / "eval_dev", "seed17_U_step100", 35)
        write_slot(R11 / "eval_dev", "seed17_U_step100", 35)
        rc, j, out = run(O11, R11, tmp / "b11.json", cpath, reg, REF)
        check("an unknown setting cannot differ silently",
              j["verdict"] == "SETTINGS_VIOLATED"
              and "some_new_knob" in j["per_seed"]["17"]["settings"][
                  "unclassified_hyperparameters"]["differences"])

        # ----------------------------------------------------- behaviour fails
        print("\nstep-100 behaviour differing by more than the tolerance")
        R12 = tmp / "rerun_behaviour"
        write_run(R12, 17, 100, 10, fill=0.25)
        write_slot(R12 / "eval_dev", "seed17_U_step100", 28)   # 47.5 vs 38.75
        rc, j, out = run(O, R12, tmp / "b12.json", cpath, reg, REF)
        b = j["per_seed"]["17"]["behaviour_at_bridge_step"]
        check("the bridge FAILS", j["verdict"] == "BRIDGE_FAILED", str(j["verdict"]))
        check("the difference is quantified",
              abs(b["difference_pp"] - 8.75) < 1e-9, str(b["difference_pp"]))
        check("the stop condition was declared before the measurement",
              "declared before the measurement" in b["rule"])
        check("the failure forbids selection and the frozen test set",
              "frozen test set is not evaluated" in j["reason"], j["reason"][-140:])

        print("\nweights that differ, and an absent dump")
        R13 = tmp / "rerun_jitter"
        write_run(R13, 17, 100, 10, fill=0.2503)
        write_slot(R13 / "eval_dev", "seed17_U_step100", 35)
        rc, j, out = run(O, R13, tmp / "b13.json", cpath, reg, REF)
        w = j["per_seed"]["17"]["weights_at_bridge_step"]
        check("a weight difference does not fail the bridge by itself",
              j["verdict"] == "BRIDGE_HELD" and w["all_bitwise_identical"] is False)
        check("no cause is attributed to it",
              "cannot say whether it came from" in w["interpretation"])
        R14 = tmp / "rerun_nodump"
        write_run(R14, 17, 100, 10, fill=0.25, write_dump=False)
        write_slot(R14 / "eval_dev", "seed17_U_step100", 35)
        rc, j, out = run(O, R14, tmp / "b14.json", cpath, reg, REF)
        check("a missing step-100 dump makes the bridge INCOMPLETE",
              j["verdict"] == "INCOMPLETE", str(j["verdict"]))
        check("and nothing is allowed to follow from it",
              "nothing follows from it" in j["reason"], j["reason"][:140])

        print("\nboth seeds are required to pass, not just one")
        write_run(O, 29, 1000, 100, fill=0.3)
        write_run(R, 29, 100, 10, fill=0.3)
        write_references(REF, 29, ma_hits=66, l2_hits=39)
        write_slot(O / "eval_dev", "seed29_U_step100", 32)     # 42.5 pp
        write_slot(R / "eval_dev", "seed29_U_step100", 24)     # 52.5 pp
        rc, j, out = run(O, R, tmp / "b15.json", cpath, reg, REF, seeds="17 29")
        check("one seed failing the bridge fails the whole bridge",
              j["verdict"] == "BRIDGE_FAILED"
              and j["per_seed"]["17"]["behaviour_at_bridge_step"]["holds"] is True
              and j["per_seed"]["29"]["behaviour_at_bridge_step"]["holds"] is False,
              str(j["verdict"]))
        # A seed outside the declared pair.
        write_run(O, 7, 1000, 100, fill=0.3)
        write_run(R, 7, 100, 10, fill=0.3)
        write_slot(REF, "seed7_MA", 66, ckpt_sha="x" * 64)
        write_slot(REF, "seed7_MAB_L2", 42, ckpt_sha="y" * 64)
        write_slot(O / "eval_dev", "seed7_U_step100", 35)
        write_slot(R / "eval_dev", "seed7_U_step100", 35)
        rc, j, out = run(O, R, tmp / "b16.json", cpath, reg, REF, seeds="7")
        check("a seed outside the declared pair is rejected",
              j["verdict"] in ("SETTINGS_VIOLATED", "REFERENCES_INVALID"),
              str(j["verdict"]))

        # ---------------------------------------------- selection bindings
        print("\nSELECTION binds to its inputs and refuses after test access")
        dev = tmp / "sel_dev"
        for st in (10, 20, 30):
            write_slot(dev, f"seed17_U_step{st}", 66 - st // 10 * 4)
        selout = tmp / "selection.json"
        p = subprocess.run(
            [sys.executable, str(SELECT), "--dev_root", str(dev),
             "--reference_dev_root", str(REF), "--seeds", "17",
             "--dump_steps", "10,20,30", "--out", str(selout)],
            capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        d = json.loads(selout.read_text())
        rec = d["per_seed"]["17"]
        check("selection reads the references from the fixed root",
              d["references_are_fixed_and_separate"] is True
              and rec["reference_dev_root"] == str(REF), str(d.get("reference_dev_root")))
        check("and records a binding for every input it read",
              len(rec["development_inputs"]) == 5
              and all(b.get("detections_sha256") for b in rec["development_inputs"]),
              str(len(rec["development_inputs"])))
        check("the reference inputs are labelled as references",
              {b["role"] for b in rec["development_inputs"]
               if "reference" in b["role"]}
              == {"fixed_reference_parent", "fixed_reference_L2"})
        v = subprocess.run([sys.executable, str(SELECT), "--verify", str(selout)],
                           capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        check("a fresh decision verifies", v.returncode == 0, v.stdout[-160:])
        write_slot(dev, "seed17_U_step20", 40)     # move a candidate
        v = subprocess.run([sys.executable, str(SELECT), "--verify", str(selout)],
                           capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        check("changing a development input makes the decision STALE",
              v.returncode == 1 and "STALE" in v.stdout + v.stderr,
              (v.stdout + v.stderr)[-200:])
        testdir = tmp / "eval_test"
        testdir.mkdir()
        (testdir / "seed17_MA").mkdir()
        (testdir / "seed17_MA" / "detections.jsonl").write_text("{}\n")
        p = subprocess.run(
            [sys.executable, str(SELECT), "--dev_root", str(dev),
             "--reference_dev_root", str(REF), "--seeds", "17",
             "--dump_steps", "10,20,30", "--refuse_if_test_evaluated", str(testdir),
             "--out", str(selout)],
            capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        check("selection REFUSES once the frozen test set has been evaluated",
              p.returncode == 4 and "test-based" in p.stdout + p.stderr,
              (p.stdout + p.stderr)[-200:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("ALL BRIDGE-CHECK TESTS PASSED (CPU only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
