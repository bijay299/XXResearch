#!/usr/bin/env python3
"""CPU tests for scripts/seq/bridge_check.py.

The bridge is the only thing that makes the early-grid rerun comparable to the
original run, so the cases that MUST fail it are tested as carefully as the one
that passes:

  * a setting that determines the first 100 updates differing      -> VIOLATED
  * a horizon-dependent learning-rate schedule, where shortening
    the run would change the learning rate at every early update   -> VIOLATED
  * an unclassified hyperparameter differing silently              -> VIOLATED
  * step-100 behaviour differing by more than the match tolerance  -> FAILED
  * a missing dump or report                                       -> INCOMPLETE
  * only the declared stopping limit and save cadence differing    -> HELD

Also checked: the verdict's exit code, that a weight difference does NOT turn
into a causal claim, and that bitwise-identical weights are reported as such.

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
CONTRACT = {"contract": "TEST", "top_level_key": "unet", "n_tensors": 2,
            "total_elements": 12, "dtype": "torch.float32",
            "tensors": {"a.attn2.to_k.weight": {"shape": [2, 3]},
                        "a.attn2.to_v.weight": {"shape": [2, 3]}}}
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


def write_run(root: Path, seed: int, iters: int, cadence: int, fill: float,
              hp_override: dict | None = None, write_dump: bool = True) -> None:
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
        "dumps": {"checkpoint_every": cadence, "expected_steps": steps,
                  "n_expected": len(steps), "n_present": len(steps),
                  "complete": True},
        "runtime": {"optimizer_steps_completed": iters},
    }, indent=2))


def write_dev(root: Path, seed: int, step: int, ma_hits: int,
              dump_hits: int, n: int = 80) -> None:
    for name, hits in ((f"seed{seed}_MA", ma_hits),
                       (f"seed{seed}_U_step{step}", dump_hits)):
        d = root / "eval_dev" / name
        d.mkdir(parents=True, exist_ok=True)
        with (d / "detections.jsonl").open("w") as fh:
            for i in range(n):
                fh.write(json.dumps({
                    "checkpoint": name, "category": "dog",
                    "image_name": f"{i}.png",
                    "hit": {"0.3": True, "0.5": i < hits, "0.7": False}}) + "\n")


def run(orig: Path, rerun: Path, out: Path, contract: Path,
        seeds: str = "17", extra: list[str] | None = None) -> tuple[int, dict, str]:
    p = subprocess.run(
        [sys.executable, str(TOOL), "--original_root", str(orig),
         "--rerun_root", str(rerun), "--seeds", seeds, "--bridge_step", "100",
         "--contract", str(contract), "--out", str(out), *(extra or [])],
        capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    j = json.loads(out.read_text()) if out.is_file() else {}
    return p.returncode, j, p.stdout + p.stderr


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bridge_"))
    cpath = tmp / "contract.json"
    cpath.write_text(json.dumps(CONTRACT))
    try:
        # ---------------------------------------------------------------- held
        print("\nonly the declared stopping limit and save cadence differ")
        O, R = tmp / "orig", tmp / "rerun"
        write_run(O, 17, 1000, 100, fill=0.25)
        write_run(R, 17, 100, 10, fill=0.25)
        write_dev(O, 17, 100, ma_hits=66, dump_hits=35)
        write_dev(R, 17, 100, ma_hits=66, dump_hits=35)
        rc, j, out = run(O, R, tmp / "b1.json", cpath)
        check("the bridge HOLDS", j.get("verdict") == "BRIDGE_HELD", str(j.get("verdict")))
        check("and that is the only verdict that exits 0", rc == 0, f"rc={rc}")
        st = j["per_seed"]["17"]["settings"]
        fu = st["settings_that_determine_the_first_updates"]
        check("every setting that determines the first 100 updates is checked",
              fu["n_compared"] == 29, str(fu["n_compared"]))
        check("none of them differs", fu["violations"] == {} and fu["all_preserved"])
        dc = st["declared_changes"]
        check("the stopping limit is reported as a declared change",
              dc["stopping_limit_and_derived"]["iterations_requested"]
              == {"original": 1000, "rerun": 100}, str(dc))
        check("so are the epoch figures upstream derives from it",
              set(dc["stopping_limit_and_derived"]) ==
              {"iterations_requested", "epochs_cap", "epoch_capacity_steps"},
              str(list(dc["stopping_limit_and_derived"])))
        check("the save cadence change is recorded as 100 -> 10",
              dc["save_cadence"] == {"original_every_n_steps": 100,
                                     "rerun_every_n_steps": 10}, str(dc["save_cadence"]))
        hz = st["lr_schedule_is_horizon_independent"]
        check("the constant schedule is verified horizon-independent",
              hz["horizon_free"] is True and hz["rerun_lr_scheduler"] == "constant")
        check("and the recorded warmup is flagged inert here",
              hz["recorded_lr_warmup_steps_is_inert_here"] == 500)
        w = j["per_seed"]["17"]["weights_at_bridge_step"]
        check("identical step-100 weights are reported as bitwise identical",
              w["all_bitwise_identical"] is True
              and w["bitwise_identical_tensors"] == 2, str(w.get("max_abs_diff")))
        b = j["per_seed"]["17"]["behaviour_at_bridge_step"]
        check("step-100 suppression is computed for both runs",
              b["original"]["dog_suppression_pp"] == 38.75
              and b["rerun"]["dog_suppression_pp"] == 38.75, str(b))
        check("and its difference is inside the declared tolerance",
              b["difference_pp"] == 0.0 and b["holds"] is True)

        # ------------------------------------------------- weights differ only
        print("\nthe same settings but different step-100 weights")
        R2 = tmp / "rerun_jitter"
        write_run(R2, 17, 100, 10, fill=0.2503)
        write_dev(R2, 17, 100, ma_hits=66, dump_hits=35)
        rc, j, out = run(O, R2, tmp / "b2.json", cpath)
        w = j["per_seed"]["17"]["weights_at_bridge_step"]
        check("the bridge still holds on behaviour", j["verdict"] == "BRIDGE_HELD")
        check("the weight difference is measured, not ignored",
              w["all_bitwise_identical"] is False
              and w["max_abs_diff"] > 0, str(w.get("max_abs_diff")))
        check("the verdict says the dumps are a second realisation",
              "second realisation" in j["reason"], j["reason"][-120:])
        check("no cause is attributed to the difference",
              "cannot say whether it came from" in w["interpretation"])
        check("and the record lists what is NOT established",
              any("do not isolate nondeterminism" in s for s in j["not_established"]),
              str(j["not_established"])[:160])

        # ------------------------------------------------- settings violations
        print("\na setting that determines the first 100 updates differs")
        R3 = tmp / "rerun_lr"
        write_run(R3, 17, 100, 10, fill=0.25,
                  hp_override={"effective_learning_rate": 1.6e-05})
        write_dev(R3, 17, 100, ma_hits=66, dump_hits=35)
        rc, j, out = run(O, R3, tmp / "b3.json", cpath)
        check("the bridge is VIOLATED", j["verdict"] == "SETTINGS_VIOLATED", str(j["verdict"]))
        check("it exits non-zero", rc != 0)
        v = j["per_seed"]["17"]["settings"][
            "settings_that_determine_the_first_updates"]["violations"]
        check("the offending setting is named with both values",
              v["effective_learning_rate"] == {"original": 8e-06, "rerun": 1.6e-05},
              str(v))
        check("and it is called a defect of the rerun, not a finding",
              "defect in the rerun" in j["reason"], j["reason"][-120:])

        print("\na horizon-DEPENDENT learning-rate schedule")
        R4 = tmp / "rerun_cosine"
        write_run(R4, 17, 100, 10, fill=0.25,
                  hp_override={"lr_scheduler": "cosine"})
        write_dev(R4, 17, 100, ma_hits=66, dump_hits=35)
        rc, j, out = run(O, R4, tmp / "b4.json", cpath)
        check("shortening the run under a cosine schedule is refused",
              j["verdict"] == "SETTINGS_VIOLATED", str(j["verdict"]))
        hz = j["per_seed"]["17"]["settings"]["lr_schedule_is_horizon_independent"]
        check("because the schedule is not horizon-free",
              hz["horizon_free"] is False and hz["rerun_lr_scheduler"] == "cosine")
        check("and the reason explains what that would change",
              "would change the learning rate at every one of the first 100"
              in hz["why_it_matters"])

        print("\nan UNCLASSIFIED hyperparameter differing silently")
        R5 = tmp / "rerun_unknown"
        write_run(R5, 17, 100, 10, fill=0.25,
                  hp_override={"some_new_knob": "changed"})
        write_dev(R5, 17, 100, ma_hits=66, dump_hits=35)
        O5 = tmp / "orig_unknown"
        write_run(O5, 17, 1000, 100, fill=0.25,
                  hp_override={"some_new_knob": "original"})
        write_dev(O5, 17, 100, ma_hits=66, dump_hits=35)
        rc, j, out = run(O5, R5, tmp / "b5.json", cpath)
        check("an unknown setting cannot differ silently",
              j["verdict"] == "SETTINGS_VIOLATED", str(j["verdict"]))
        u = j["per_seed"]["17"]["settings"]["unclassified_hyperparameters"]
        check("it is listed as unclassified and as a difference",
              "some_new_knob" in u["names"] and "some_new_knob" in u["differences"])

        # ----------------------------------------------------- behaviour fails
        print("\nstep-100 behaviour differing by more than the match tolerance")
        R6 = tmp / "rerun_behaviour"
        write_run(R6, 17, 100, 10, fill=0.25)
        write_dev(R6, 17, 100, ma_hits=66, dump_hits=28)   # 47.5 pp vs 38.75
        rc, j, out = run(O, R6, tmp / "b6.json", cpath)
        check("the bridge FAILS", j["verdict"] == "BRIDGE_FAILED", str(j["verdict"]))
        b = j["per_seed"]["17"]["behaviour_at_bridge_step"]
        check("the difference is quantified", abs(b["difference_pp"] - 8.75) < 1e-9,
              str(b["difference_pp"]))
        check("the stop condition was declared before the measurement",
              "declared before the measurement" in b["rule"])
        check("the failure forbids selection and the frozen test set",
              "frozen test set is not evaluated" in j["reason"], j["reason"][-140:])
        check("a failed bridge is itself reported as a result",
              "amendment STOPS here and reports this instead" in j["reason"])

        # --------------------------------------------------------- incomplete
        print("\nan absent dump or report")
        R7 = tmp / "rerun_nodump"
        write_run(R7, 17, 100, 10, fill=0.25, write_dump=False)
        write_dev(R7, 17, 100, ma_hits=66, dump_hits=35)
        rc, j, out = run(O, R7, tmp / "b7.json", cpath)
        check("a missing step-100 dump makes the bridge INCOMPLETE",
              j["verdict"] == "INCOMPLETE", str(j["verdict"]))
        check("and nothing is allowed to follow from it",
              "nothing follows from it" in j["reason"], j["reason"][:140])
        R8 = tmp / "rerun_empty"
        (R8 / "eval_dev").mkdir(parents=True, exist_ok=True)
        rc, j, out = run(O, R8, tmp / "b8.json", cpath)
        check("so does a missing train report", j["verdict"] == "INCOMPLETE")

        # ------------------------------------------------------- both seeds
        print("\nboth seeds are required to pass, not just one")
        write_run(O, 29, 1000, 100, fill=0.3)
        write_dev(O, 29, 100, ma_hits=66, dump_hits=32)
        R9 = tmp / "rerun_two"
        write_run(R9, 17, 100, 10, fill=0.25)
        write_dev(R9, 17, 100, ma_hits=66, dump_hits=35)
        write_run(R9, 29, 100, 10, fill=0.3)
        write_dev(R9, 29, 100, ma_hits=66, dump_hits=24)   # 52.5 vs 42.5 pp
        rc, j, out = run(O, R9, tmp / "b9.json", cpath, seeds="17 29")
        check("one seed failing the bridge fails the whole bridge",
              j["verdict"] == "BRIDGE_FAILED"
              and j["per_seed"]["17"]["behaviour_at_bridge_step"]["holds"] is True
              and j["per_seed"]["29"]["behaviour_at_bridge_step"]["holds"] is False,
              str(j["verdict"]))

        print("\nthe record never overreaches")
        rc, j, out = run(O, R9, tmp / "b10.json", cpath, seeds="17 29",
                         extra=["--skip_weights"])
        check("--skip_weights needs no torch and no dumps",
              "weights_at_bridge_step" not in j["per_seed"]["17"])
        check("retention is explicitly out of scope",
              any("never touches the frozen test set" in s
                  for s in j["not_established"]))
        check("and so is calling either run THE state at step k",
              any("THE state of this configuration at step k" in s
                  for s in j["not_established"]))
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
