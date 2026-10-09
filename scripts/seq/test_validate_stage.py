#!/usr/bin/env python3
"""CPU tests for scripts/seq/validate_stage.py.

Builds artifacts on disk -- valid ones and each way they can be wrong -- and
asserts the validator's verdict. No GPU, no model, no network: CUDA_VISIBLE_DEVICES
is forced empty and the only torch use is CPU save/load of small tensors.

Most cases run against a deliberately tiny checkpoint contract so the suite is
fast; the real 19.17M-parameter contract and the real saved artifacts are
covered by test_real_artifacts().

    python scripts/seq/test_validate_stage.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_stage import canonical_manifest_digest  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
VALIDATOR = REPO / "scripts" / "seq" / "validate_stage.py"
SEQ_ROOT = Path("/data/bijaypandey/cuig_pilot/seq_pilot")

PASS, FAIL = [], []


def check(desc: str, want: str, rc: int, out: str, expect_msg: str | None = None):
    """want: 'valid' or 'invalid'. expect_msg: substring the output must contain."""
    got = "valid" if rc == 0 else "invalid"
    ok = got == want
    if ok and expect_msg and expect_msg.lower() not in out.lower():
        ok, got = False, f"{got} but message lacks {expect_msg!r}"
    (PASS if ok else FAIL).append(desc)
    print(f"  {'ok  ' if ok else 'FAIL'}  {desc:<58} -> {got}"
          + ("" if ok else f" (wanted {want})"))
    if not ok:
        for line in out.strip().splitlines()[:4]:
            print(f"          | {line}")


def run(*args: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(VALIDATOR), *args],
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return p.returncode, p.stdout + p.stderr


# ------------------------------------------------------------------ fixtures
def tiny_contract(path: Path) -> dict:
    """A 2-tensor contract, so synthetic checkpoints are kilobytes not megabytes."""
    c = {
        "contract": "TEST contract",
        "top_level_key": "unet",
        "n_tensors": 2,
        "total_elements": 12,
        "dtype": "torch.float32",
        "expected_file_bytes": None,     # size check skipped for synthetic files
        "tensors": {
            "a.attn2.to_k.weight": {"shape": [2, 3], "dtype": "torch.float32"},
            "a.attn2.to_v.weight": {"shape": [2, 3], "dtype": "torch.float32"},
        },
    }
    path.write_text(json.dumps(c, indent=2))
    return c


def write_ckpt(p: Path, *, tensors=None, nonfinite=False, wrong_shape=False,
               wrong_dtype=False, plain_text=False) -> None:
    import torch
    p.parent.mkdir(parents=True, exist_ok=True)
    if plain_text:
        p.write_text("this is plain text, not a model\n" * 10)
        return
    if tensors is None:
        tensors = {
            "a.attn2.to_k.weight": torch.ones(2, 3),
            "a.attn2.to_v.weight": torch.ones(2, 3) * 2,
        }
        if nonfinite:
            tensors["a.attn2.to_k.weight"][0, 0] = float("nan")
        if wrong_shape:
            tensors["a.attn2.to_v.weight"] = torch.ones(3, 3)
        if wrong_dtype:
            tensors["a.attn2.to_v.weight"] = torch.ones(2, 3, dtype=torch.float64)
    torch.save({"unet": tensors}, p)


def sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def train_report(name="MAB", *, seed=17, parent="MA", target="dog", anchor="horse",
                 l2=0.0, steps=1000, sha256="x", finished=True, parent_sha=None,
                 drop=None, steps_recorded=None, train_seconds=None,
                 completed="same", drop_runtime_keys=None) -> dict:
    """completed: "same" -> equals steps; None -> omit the counter; int -> that value."""
    d = {
        "checkpoint": name, "parent": parent, "training_seed": seed,
        "l2sp_weight": l2, "new_deletion_target": target, "anchor_concept": anchor,
        "child_checkpoint": {"sha256": sha256, "bytes": 1, "tensors": 2},
        "effective_hyperparameters": {
            "seed": seed, "iterations_requested": steps,
            "epoch_capacity_steps": max(steps, 1000), "parameter_group": "kv-xattn",
        },
        "parent_verification": {
            "parent_sha256": parent_sha, "PASS": True,
            "max_abs_deviation_after_load": 0.0,
        },
        "l2sp_reference_check": {"PASS": True, "max_abs_deviation_from_parent": 0.0},
        "l2sp_trace": {"steps_recorded": steps_recorded if steps_recorded is not None else steps},
        "started_utc": "2026-10-09T00:00:00+00:00",
        "finished_utc": "2026-10-09T00:14:00+00:00" if finished else None,
        "runtime": {"train_seconds": train_seconds if train_seconds is not None else steps * 0.83,
                    "seconds_per_optimizer_step": 0.83,
                    "seconds_per_optimizer_step_is_derived": True},
    }
    if completed == "same":
        d["runtime"]["optimizer_steps_completed"] = steps
        d["runtime"]["optimizer_steps_source"] = "test fixture"
    elif completed is not None:
        d["runtime"]["optimizer_steps_completed"] = completed
        d["runtime"]["optimizer_steps_source"] = "test fixture"
    for k in (drop_runtime_keys or []):
        d["runtime"].pop(k, None)
    for k in (drop or []):
        d.pop(k, None)
    return d


def manifest(path: Path, *, n_prompts=2, cats=("cat", "dog"), seeds=(101, 202),
             sha256="MANSHA") -> dict:
    recs = []
    for c in cats:
        for fam in ("literal", "paraphrase"):
            for i in range(n_prompts):
                for s in seeds:
                    recs.append({"prompt_id": f"{c}_{fam}_{i}", "category": c,
                                 "prompt_family": fam, "prompt_index": i,
                                 # Distinct text per prompt_id: the validator
                                 # requires ids and texts to be one-to-one, so
                                 # that a text cannot change without its id.
                                 "prompt": f"a {c} scene {fam} {i}", "gen_seed": s,
                                 "image_name": f"{c}_{fam}_{i}_seed{s}.jpg"})
    m = {"records": recs, "gen_seeds": list(seeds),
         "frozen_before_any_editing": True}
    # The digest is RECOMPUTED by the validator, so a fixture must carry a real
    # canonical one. ``sha256="KEEP_BOGUS"`` deliberately writes a wrong digest
    # to exercise the stored-vs-recomputed check.
    m["manifest_sha256"] = ("KEEP_BOGUS" if sha256 == "KEEP_BOGUS"
                            else canonical_manifest_digest(m))
    path.write_text(json.dumps(m))
    return m


def write_eval(d: Path, man: dict, ck="MA", *, mode="valid", make_images=True,
               man_sha=None, gen_settings=None, ckpt_sha="CKSHA",
               drop_ckpt_sha=False, applied=True, base_model_dir=None,
               unet_ckpt="/some/delta.bin"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "images").mkdir(exist_ok=True)
    recs = man["records"]
    rows = []
    for i, r in enumerate(recs):
        ip = d / "images" / r["image_name"]
        if make_images:
            ip.write_bytes(b"\xff\xd8fake-jpeg" + str(i).encode())
        rows.append({
            "checkpoint": ck, "prompt_id": r["prompt_id"], "category": r["category"],
            "prompt_family": r["prompt_family"], "prompt_index": r["prompt_index"],
            "prompt": r["prompt"], "gen_seed": r["gen_seed"],
            "image_path": str(ip), "image_sha256": f"hash{i:04d}",
            "detections": [], "max_score_target": 0.1,
            "n_detections_target": {"0.3": 0, "0.5": 0, "0.7": 0},
            "hit": {"0.3": False, "0.5": False, "0.7": False},
        })

    if mode == "all_identical":
        rows = [dict(rows[0]) for _ in rows]
    elif mode == "malformed":
        (d / "detections.jsonl").write_text("not json at all {{{\n" * len(rows))
        rows = None
    elif mode == "duplicate_one":
        rows[1] = dict(rows[0])
    elif mode == "missing_one":
        rows = rows[:-1]
    elif mode == "extra_row":
        e = dict(rows[0]); e["prompt_id"] = "ghost_literal_9"; rows.append(e)
    elif mode == "short":
        rows = rows[: len(rows) // 2]
    elif mode == "wrong_checkpoint_label":
        for x in rows:
            x["checkpoint"] = "SOMETHING_ELSE"
    elif mode == "missing_field":
        for x in rows:
            x.pop("image_sha256", None)
    elif mode == "repeated_image_hash":
        for x in rows:
            x["image_sha256"] = "same"
    elif mode == "missing_threshold":
        for x in rows:
            x["hit"] = {"0.5": False}

    if rows is not None:
        (d / "detections.jsonl").write_text(
            "\n".join(json.dumps(x) for x in rows) + "\n")
        n = len(rows)
    else:
        n = len(recs)

    gs = gen_settings or {"num_inference_steps": 30, "guidance_scale": 7.5,
                          "resolution": 512,
                          "scheduler": "pipeline default (PNDM for SD-1.5)",
                          "dtype": "float16", "negative_prompt": None,
                          "identical_at_every_checkpoint": True}
    cl = {"applied": applied, "unet_ckpt": unet_ckpt}
    if not drop_ckpt_sha:
        cl["sha256"] = ckpt_sha
    (d / "image_report.json").write_text(json.dumps({
        "checkpoint": ck, "complete": True, "expected_images": len(recs),
        "generated_or_reused": len(recs),
        "manifest_sha256": man_sha if man_sha is not None else man["manifest_sha256"],
        "base_model_dir": base_model_dir,
        "generation_settings": gs, "checkpoint_load": cl,
    }))
    (d / "detect_report.json").write_text(json.dumps({
        "checkpoint": ck, "complete": True, "images_scored": n,
        "images_expected": len(recs),
    }))


# ---------------------------------------------------------------------- tests
def test_training(tmp: Path):
    print("\ntraining stage -- contract, identity, steps, lineage, tensors")
    c = tmp / "contract.json"
    tiny_contract(c)
    models = tmp / "models"
    base = ["--models_root", str(models), "--contract", str(c),
            "--expect_seed", "17", "--expect_parent", "MA",
            "--expect_target", "dog", "--expect_anchor", "horse",
            "--expect_l2sp", "0", "--expect_steps", "1000"]

    # a valid parent to hang lineage on
    (models / "MA").mkdir(parents=True, exist_ok=True)
    write_ckpt(models / "MA" / "delta.bin")
    psha = sha(models / "MA" / "delta.bin")

    def setup(name="MAB", **kw):
        d = models / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        ck_kw = {k: kw.pop(k) for k in
                 ("nonfinite", "wrong_shape", "wrong_dtype", "plain_text")
                 if k in kw}
        write_ckpt(d / "delta.bin", **ck_kw)
        rep = train_report(name, parent_sha=psha,
                           sha256=sha(d / "delta.bin"), **kw)
        (d / "train_report.json").write_text(json.dumps(rep))
        return d

    setup()
    check("valid training stage", "valid", *run("train", "MAB", *base))

    # THE REPORTED DEFECT: plain text accepted because only size was checked.
    d = setup(plain_text=True)
    rep = json.loads((d / "train_report.json").read_text())
    rep["child_checkpoint"]["sha256"] = sha(d / "delta.bin")
    (d / "train_report.json").write_text(json.dumps(rep))
    check("delta.bin is plain text (reported defect 1)", "invalid",
          *run("train", "MAB", *base), expect_msg="does not load as a torch")

    setup(nonfinite=True)
    d = models / "MAB"
    rep = json.loads((d / "train_report.json").read_text())
    rep["child_checkpoint"]["sha256"] = sha(d / "delta.bin")
    (d / "train_report.json").write_text(json.dumps(rep))
    check("tensor contains NaN", "invalid", *run("train", "MAB", *base),
          expect_msg="NaN or Inf")

    for kw, msg in ((dict(wrong_shape=True), "shape mismatch"),
                    (dict(wrong_dtype=True), "dtype mismatch")):
        d = setup(**kw)
        rep = json.loads((d / "train_report.json").read_text())
        rep["child_checkpoint"]["sha256"] = sha(d / "delta.bin")
        (d / "train_report.json").write_text(json.dumps(rep))
        check(f"checkpoint tensor {msg}", "invalid", *run("train", "MAB", *base),
              expect_msg=msg)

    # wrong tensor names entirely
    import torch
    d = setup()
    write_ckpt(d / "delta.bin", tensors={"totally.other.weight": torch.ones(2, 3)})
    rep = json.loads((d / "train_report.json").read_text())
    rep["child_checkpoint"]["sha256"] = sha(d / "delta.bin")
    (d / "train_report.json").write_text(json.dumps(rep))
    check("unexpected tensor names", "invalid", *run("train", "MAB", *base),
          expect_msg="missing")

    # hash assertions
    d = setup()
    rep = json.loads((d / "train_report.json").read_text())
    rep["child_checkpoint"]["sha256"] = "0" * 64
    (d / "train_report.json").write_text(json.dumps(rep))
    check("report claims a hash the file does not have", "invalid",
          *run("train", "MAB", *base), expect_msg="hash mismatch")

    setup()
    check("caller-expected hash differs", "invalid",
          *run("train", "MAB", *base, "--expect_sha", "0" * 64),
          expect_msg="!= expected")

    # stale configuration: the report describes a DIFFERENT request
    for kw, field in ((dict(seed=29), "training_seed"),
                      (dict(target="sandwich"), "new_deletion_target"),
                      (dict(anchor="flower"), "anchor_concept"),
                      (dict(l2=25000.0), "l2sp_weight"),
                      (dict(parent="M0"), "parent")):
        setup(**kw)
        check(f"stale config: {field} is for another request", "invalid",
              *run("train", "MAB", *base), expect_msg="mismatch")

    # incomplete / short trajectory
    setup(steps=500)
    check("iterations_requested 500 != requested 1000", "invalid",
          *run("train", "MAB", *base), expect_msg="iterations_requested")
    # ---- completed optimizer steps: the counter is the ONLY step evidence.
    # The old runtime cross-check was circular (train_seconds divided by
    # train_seconds/iterations returns iterations), so it is gone.
    print("\n  completed-step counter (replaces the circular runtime check)")
    setup()
    check("counter equals the request", "valid", *run("train", "MAB", *base),
          expect_msg="optimizer_steps_completed")
    setup(completed=400)
    check("counter shows 400 of 1000 -> incomplete", "invalid",
          *run("train", "MAB", *base), expect_msg="did not complete")
    setup(completed=1001)
    check("counter overshoots the request", "invalid",
          *run("train", "MAB", *base), expect_msg="request is 1000")
    setup(completed=None)
    check("counter ABSENT, default policy -> fail", "invalid",
          *run("train", "MAB", *base), expect_msg="cannot be evidenced")
    setup(completed=None)
    check("counter ABSENT, --steps_evidence legacy_optional -> structural pass",
          "valid", *run("train", "MAB", *base,
                        "--steps_evidence", "legacy_optional"),
          expect_msg="UNVERIFIED")
    setup(completed=400)
    check("legacy_optional still rejects a PRESENT short counter", "invalid",
          *run("train", "MAB", *base, "--steps_evidence", "legacy_optional"),
          expect_msg="did not complete")
    setup(completed="bogus")
    check("counter is not an integer", "invalid", *run("train", "MAB", *base),
          expect_msg="not an integer")

    # The circular check must be gone: a wildly wrong train_seconds, with a
    # correct counter, is no longer a step-count failure.
    setup(train_seconds=1.0)
    check("absurd train_seconds no longer fakes a step check", "valid",
          *run("train", "MAB", *base))

    # The L2 trace is evidence that regularisation ran, never a step count.
    l2base = [x if x != "0" else "25000" for x in base]
    setup(l2=25000.0, steps_recorded=400)
    check("L2 arm: trace of 400 is NOT treated as a step count", "valid",
          *run("train", "MAB", *l2base))
    setup(l2=25000.0, steps_recorded=0)
    check("L2 arm: trace of 0 means the regulariser never ran", "invalid",
          *run("train", "MAB", *l2base), expect_msg="active")
    setup(steps_recorded=0)
    check("unregularised arm: steps_recorded=0 is legitimate", "valid",
          *run("train", "MAB", *base))

    # report integrity
    setup(finished=False)
    check("finished_utc absent (killed before the end)", "invalid",
          *run("train", "MAB", *base), expect_msg="did not reach its end")
    d = setup()
    (d / "train_report.json").write_text('{"checkpoint":"MAB","parent"')
    check("train_report.json truncated mid-write", "invalid",
          *run("train", "MAB", *base), expect_msg="does not parse")
    setup(drop=["parent_verification"])
    check("report missing a required key", "invalid",
          *run("train", "MAB", *base), expect_msg="missing required key")

    # lineage: the parent on disk changed after the child was trained
    setup()
    write_ckpt(models / "MA" / "delta.bin",
               tensors={"a.attn2.to_k.weight": torch.zeros(2, 3),
                        "a.attn2.to_v.weight": torch.zeros(2, 3)})
    check("stale parent: parent hash no longer matches", "invalid",
          *run("train", "MAB", *base), expect_msg="parent hash mismatch")
    write_ckpt(models / "MA" / "delta.bin")   # restore

    d = setup()
    (d / "delta.bin").unlink()
    check("delta.bin absent", "invalid", *run("train", "MAB", *base))
    d = setup()
    (d / "delta.bin").write_bytes(b"")
    check("delta.bin zero bytes", "invalid", *run("train", "MAB", *base))

    # a check that cannot be performed must fail, not pass
    setup()
    check("contract file missing -> UNCHECKED, not a pass", "invalid",
          *run("train", "MAB", "--models_root", str(models),
               "--contract", str(tmp / "nope.json"), "--expect_steps", "1000"),
          expect_msg="unchecked")


def test_evaluation(tmp: Path):
    print("\nevaluation stage -- identities, coverage, config identity, artifacts")
    man_p = tmp / "manifest.json"
    man = manifest(man_p)
    MAN_SHA = man["manifest_sha256"]
    gs_p = tmp / "gs.json"
    gs_p.write_text(json.dumps({
        "_note": "metadata key, must be ignored",
        "num_inference_steps": 30, "guidance_scale": 7.5, "resolution": 512,
        "scheduler": "pipeline default (PNDM for SD-1.5)", "dtype": "float16",
        "negative_prompt": None, "identical_at_every_checkpoint": True}))
    root = tmp / "eval"
    base = ["--eval_root", str(root), "--manifest", str(man_p),
            "--expect_manifest_sha", MAN_SHA,
            "--expect_gen_settings", str(gs_p), "--require_images",
            "--expect_sha", "CKSHA"]

    def setup(mode="valid", **kw):
        d = root / "MA"
        if d.exists():
            shutil.rmtree(d)
        write_eval(d, man, "MA", mode=mode, **kw)
        return d

    setup()
    check(f"valid evaluation ({len(man['records'])} rows from manifest)", "valid",
          *run("eval", "MA", *base))

    # THE REPORTED DEFECTS 2 and 3
    setup("all_identical")
    check("every row identical (reported defect 2)", "invalid",
          *run("eval", "MA", *base), expect_msg="duplicat")
    setup("malformed")
    check("every row malformed (reported defect 3)", "invalid",
          *run("eval", "MA", *base), expect_msg="unparseable")

    setup("duplicate_one")
    check("one duplicated identity", "invalid", *run("eval", "MA", *base),
          expect_msg="duplicat")
    setup("missing_one")
    check("one manifest identity unscored", "invalid", *run("eval", "MA", *base),
          expect_msg="unscored")
    setup("extra_row")
    check("row absent from the manifest", "invalid", *run("eval", "MA", *base),
          expect_msg="absent from the")
    setup("short")
    check("half the rows present", "invalid", *run("eval", "MA", *base))
    setup("missing_field")
    check("rows missing a required field", "invalid", *run("eval", "MA", *base),
          expect_msg="missing required field")
    setup("missing_threshold")
    check("hit lacks the 0.3/0.7 thresholds", "invalid", *run("eval", "MA", *base),
          expect_msg="must carry all")
    setup("repeated_image_hash")
    check("all rows share one image_sha256", "invalid", *run("eval", "MA", *base),
          expect_msg="repeat an image_sha256")
    setup("wrong_checkpoint_label")
    check("rows labelled with another checkpoint", "invalid",
          *run("eval", "MA", *base), expect_msg="checkpoint label")

    # stale manifest / settings / checkpoint
    setup(man_sha="OLDSHA")
    check("stale manifest: evaluated against an older manifest", "invalid",
          *run("eval", "MA", *base), expect_msg="stale evaluation")
    setup()
    check("caller-expected manifest identity differs", "invalid",
          *run("eval", "MA", "--eval_root", str(root), "--manifest", str(man_p),
               "--expect_manifest_sha", "0" * 64, "--require_images",
               "--expect_sha", "CKSHA"),
          expect_msg="CONTENT digest")
    setup(gen_settings={"num_inference_steps": 50, "guidance_scale": 7.5,
                        "resolution": 512,
                        "scheduler": "pipeline default (PNDM for SD-1.5)",
                        "dtype": "float16", "negative_prompt": None,
                        "identical_at_every_checkpoint": True})
    check("generation settings differ from the contract", "invalid",
          *run("eval", "MA", *base), expect_msg="num_inference_steps")
    # ---- binding the evaluation to the model that produced it ------------
    print("\n  checkpoint binding (an evaluation must name the model it used)")
    setup()
    check("checkpoint CHANGED under the same name -> stale evaluation", "invalid",
          *run("eval", "MA", "--eval_root", str(root), "--manifest", str(man_p),
               "--expect_manifest_sha", MAN_SHA, "--expect_gen_settings", str(gs_p),
               "--require_images", "--expect_sha", "NEWSHA_AFTER_RETRAIN"),
          expect_msg="stale with respect to its checkpoint")
    setup(drop_ckpt_sha=True)
    check("recorded generation hash MISSING, expected one supplied", "invalid",
          *run("eval", "MA", *base), expect_msg="records no sha256")
    setup()
    check("no --expect_sha supplied at all (non-M0)", "invalid",
          *run("eval", "MA", "--eval_root", str(root), "--manifest", str(man_p),
               "--expect_manifest_sha", MAN_SHA, "--expect_gen_settings", str(gs_p),
               "--require_images"),
          expect_msg="no --expect_sha given")
    setup(applied=False)
    check("delta was never applied for a non-M0 checkpoint", "invalid",
          *run("eval", "MA", *base), expect_msg="applied")
    setup()
    check("matching checkpoint hash -> valid", "valid", *run("eval", "MA", *base))

    # companion artifacts
    for art, msg in (("image_report.json", "image_report"),
                     ("detect_report.json", "detect_report")):
        d = setup()
        (d / art).unlink()
        check(f"{art} absent", "invalid", *run("eval", "MA", *base),
              expect_msg=msg)
    d = setup()
    rep = json.loads((d / "detect_report.json").read_text())
    rep["images_scored"] = 3
    (d / "detect_report.json").write_text(json.dumps(rep))
    check("detect_report count disagrees with the jsonl", "invalid",
          *run("eval", "MA", *base), expect_msg="images_scored")
    d = setup()
    rep = json.loads((d / "image_report.json").read_text())
    rep["complete"] = False
    (d / "image_report.json").write_text(json.dumps(rep))
    check("image_report.complete is false", "invalid", *run("eval", "MA", *base),
          expect_msg="complete")

    d = setup(make_images=False)
    check("recorded image files do not exist", "invalid",
          *run("eval", "MA", *base), expect_msg="do not exist")

    # ---- M0 under an explicit base-model identity + shared-evaluation policy,
    # not as an accidental exception to checkpoint binding.
    print("\n  M0 policy (base-model identity, declared reuse)")
    bm_dir = tmp / "fake_base_model"
    (bm_dir / "unet").mkdir(parents=True, exist_ok=True)
    (bm_dir / "model_index.json").write_text('{"_class_name":"X"}')
    (bm_dir / "unet" / "config.json").write_text('{"k":1}')
    (bm_dir / "unet" / "w.safetensors").write_bytes(b"w" * 64)
    import hashlib as _h

    def _ident(base):
        cheap = {}
        for rel in ("model_index.json", "unet/config.json"):
            f = base / rel
            cheap[rel] = {"sha256": _h.sha256(f.read_bytes()).hexdigest(),
                          "bytes": f.stat().st_size}
        sizes = {"unet/w.safetensors": (base / "unet/w.safetensors").stat().st_size}
        body = {"cheap_identity_files": cheap, "weight_file_bytes": sizes}
        return _h.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    bm_c = tmp / "base_model_contract.json"
    bm_c.write_text(json.dumps({
        "base_model_dir": str(bm_dir),
        "cheap_identity_sha256": _ident(bm_dir),
        "cheap_identity_payload": {
            "cheap_identity_files": {"model_index.json": {}, "unet/config.json": {}},
            "weight_file_bytes": {"unet/w.safetensors": 64}},
        "weight_sha256": {"unet/w.safetensors":
                          _h.sha256((bm_dir / "unet/w.safetensors").read_bytes()).hexdigest()},
        "m0_policy": {"shared_evaluation": "counted once"}}))

    shared_src = root / "M0"
    write_eval(shared_src, man, "M0", applied=False, unet_ckpt=None,
               drop_ckpt_sha=True, base_model_dir=str(bm_dir))
    other_root = tmp / "eval_seed99"
    other_root.mkdir(parents=True, exist_ok=True)
    link = other_root / "M0"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(shared_src)
    m0base = ["--manifest", str(man_p), "--expect_manifest_sha", MAN_SHA,
              "--require_images", "--base_model_contract", str(bm_c)]
    check("M0 reuse NOT declared", "invalid",
          *run("eval", "M0", "--eval_root", str(other_root), *m0base),
          expect_msg="must be declared")
    check("M0 reuse declared, base-model identity from contract", "valid",
          *run("eval", "M0", "--eval_root", str(other_root), *m0base,
               "--allow_shared_images"))
    # ---- the base-model identity check is PARTIAL, and says so.
    #
    # --verify_base_model hashes the small config files and the weight files'
    # BYTE LENGTHS. It never reads the weight bytes, so a same-size weight
    # mutation passes it. AUDIT-01c described it as "the recomputed backbone
    # identity", which reads as full verification; these cases pin down what it
    # does and does not establish.
    check("M0 with --verify_base_model reports itself as PARTIAL", "valid",
          *run("eval", "M0", "--eval_root", str(other_root), *m0base,
               "--allow_shared_images", "--verify_base_model"),
          expect_msg="PARTIAL")
    check("without it, identity is reported as DECLARED, not verified", "valid",
          *run("eval", "M0", "--eval_root", str(other_root), *m0base,
               "--allow_shared_images"),
          expect_msg="DECLARED, not verified")
    (bm_dir / "unet" / "config.json").write_text('{"k":2}')   # config changed
    check("M0 rejected when a backbone config file has changed", "invalid",
          *run("eval", "M0", "--eval_root", str(other_root), *m0base,
               "--allow_shared_images", "--verify_base_model"),
          expect_msg="config files or weight file sizes have changed")
    (bm_dir / "unet" / "config.json").write_text('{"k":1}')   # restore

    # A SAME-SIZE weight mutation: the partial check passes it, the full weight
    # digest catches it. This is the exact hole the review reproduced on CPU.
    (bm_dir / "unet" / "w.safetensors").write_bytes(b"W" * 64)
    check("same-size weight mutation PASSES the partial check", "valid",
          *run("eval", "M0", "--eval_root", str(other_root), *m0base,
               "--allow_shared_images", "--verify_base_model"),
          expect_msg="weight CONTENTS were not hashed")
    check("same-size weight mutation FAILS the full weight digest", "invalid",
          *run("eval", "M0", "--eval_root", str(other_root), *m0base,
               "--allow_shared_images", "--verify_base_model_weight_sha"),
          expect_msg="weight contents do not match")
    (bm_dir / "unet" / "w.safetensors").write_bytes(b"w" * 64)   # restore
    check("the unmutated backbone passes the full weight digest", "valid",
          *run("eval", "M0", "--eval_root", str(other_root), *m0base,
               "--allow_shared_images", "--verify_base_model",
               "--verify_base_model_weight_sha"),
          expect_msg="VERIFIED")
    # A contract with no weight_sha256 cannot perform the full check: UNCHECKED,
    # never a silent pass.
    bm_noweight = tmp / "base_model_contract_noweight.json"
    _c = json.loads(bm_c.read_text())
    _c.pop("weight_sha256", None)
    bm_noweight.write_text(json.dumps(_c))
    check("no weight_sha256 in the contract -> UNCHECKED, not a pass", "invalid",
          *run("eval", "M0", "--eval_root", str(other_root),
               "--manifest", str(man_p), "--expect_manifest_sha", MAN_SHA,
               "--require_images", "--base_model_contract", str(bm_noweight),
               "--allow_shared_images", "--verify_base_model_weight_sha"),
          expect_msg="cannot be verified")
    check("M0 without a base-model contract -> UNCHECKED, not a pass", "invalid",
          *run("eval", "M0", "--eval_root", str(other_root),
               "--manifest", str(man_p), "--expect_manifest_sha", MAN_SHA,
               "--require_images", "--allow_shared_images",
               "--base_model_contract", str(tmp / "nope.json")),
          expect_msg="base-model contract unavailable")
    write_eval(shared_src, man, "M0", applied=True, unet_ckpt="/a/delta.bin",
               ckpt_sha="SOMEDELTA", base_model_dir=str(bm_dir))
    check("M0 recording an applied delta is rejected", "invalid",
          *run("eval", "M0", "--eval_root", str(other_root), *m0base,
               "--allow_shared_images"),
          expect_msg="must apply no delta")
    # restore a clean shared M0 for any later use
    write_eval(shared_src, man, "M0", applied=False, unet_ckpt=None,
               drop_ckpt_sha=True, base_model_dir=str(bm_dir))

    # expected count derives from the manifest, not a hard-coded 280
    man2_p = tmp / "manifest_small.json"
    man2 = manifest(man2_p, n_prompts=1, cats=("cat",), seeds=(101,))
    d2 = root / "SMALL"
    write_eval(d2, man2, "SMALL")
    check(f"count derived from manifest ({len(man2['records'])} rows, not 280)",
          "valid", *run("eval", "SMALL", "--eval_root", str(root),
                        "--manifest", str(man2_p), "--require_images",
                        "--expect_manifest_sha", man2["manifest_sha256"],
                        "--expect_sha", "CKSHA"))
    check("same artifacts judged against the larger manifest", "invalid",
          *run("eval", "SMALL", "--eval_root", str(root),
               "--manifest", str(man_p), "--require_images",
               "--expect_manifest_sha", MAN_SHA, "--expect_sha", "CKSHA"),
          expect_msg="unscored")

    d = setup()
    (d / "detections.jsonl").unlink()
    check("detections.jsonl absent", "invalid", *run("eval", "MA", *base))
    d = setup()
    (d / "detections.jsonl").write_text("")
    check("detections.jsonl empty", "invalid", *run("eval", "MA", *base))

    bad_man = tmp / "bad_manifest.json"
    bad_man.write_text(json.dumps({"manifest_sha256": "X"}))
    setup()
    check("manifest has no records -> UNCHECKED, not a pass", "invalid",
          *run("eval", "MA", "--eval_root", str(root), "--manifest", str(bad_man),
               "--expect_sha", "CKSHA"),
          expect_msg="unchecked")


def test_manifest_stage(tmp: Path):
    print("\nmanifest stage -- content digest recomputed, never trusted")
    man_p = tmp / "m_ok.json"
    man = manifest(man_p, n_prompts=3, cats=("cat", "dog", "bird"))
    SHA = man["manifest_sha256"]

    check("valid manifest against its frozen identity", "valid",
          *run("manifest", "ok", "--manifest", str(man_p),
               "--expect_manifest_sha", SHA))

    # THE case the brief names: change ONLY prompt text, preserving prompt ids,
    # the row count and the old digest field. The old code read the digest out
    # of this same file, so this passed.
    tam = tmp / "m_tampered.json"
    t = json.loads(man_p.read_text())
    for rec in t["records"]:
        rec["prompt"] = rec["prompt"] + " at night in the rain"
    tam.write_text(json.dumps(t))     # digest field deliberately left stale
    rows_same = len(t["records"]) == len(man["records"])
    ids_same = ({r["prompt_id"] for r in t["records"]}
                == {r["prompt_id"] for r in man["records"]})
    dig_same = t["manifest_sha256"] == SHA
    print(f"        (tamper preserves: rows={rows_same}, ids={ids_same}, "
          f"digest field={dig_same})")
    check("prompt TEXT changed; ids, count and digest field preserved", "invalid",
          *run("manifest", "tampered", "--manifest", str(tam),
               "--expect_manifest_sha", SHA),
          expect_msg="content digest")

    # A single record altered must also be caught.
    one = tmp / "m_one.json"
    t2 = json.loads(man_p.read_text())
    t2["records"][0]["prompt"] = "something else entirely"
    one.write_text(json.dumps(t2))
    check("a single prompt text altered", "invalid",
          *run("manifest", "one", "--manifest", str(one),
               "--expect_manifest_sha", SHA), expect_msg="content digest")

    # A gen_seed swapped without touching texts.
    seed_sw = tmp / "m_seed.json"
    t3 = json.loads(man_p.read_text())
    t3["records"][0]["gen_seed"] = 999
    seed_sw.write_text(json.dumps(t3))
    check("a generation seed altered", "invalid",
          *run("manifest", "seed", "--manifest", str(seed_sw),
               "--expect_manifest_sha", SHA), expect_msg="content digest")

    nodig = tmp / "m_nodigest.json"
    t4 = json.loads(man_p.read_text()); t4.pop("manifest_sha256")
    nodig.write_text(json.dumps(t4))
    check("manifest carries no digest field", "invalid",
          *run("manifest", "nodig", "--manifest", str(nodig),
               "--expect_manifest_sha", SHA), expect_msg="no manifest_sha256")

    bogus_p = tmp / "m_bogus.json"
    manifest(bogus_p, n_prompts=3, cats=("cat", "dog", "bird"), sha256="KEEP_BOGUS")
    check("stored digest disagrees with recomputed content", "invalid",
          *run("manifest", "bogus", "--manifest", str(bogus_p),
               "--expect_manifest_sha", SHA), expect_msg="recomputed content")

    check("no frozen identity supplied -> UNCHECKED, not a pass", "invalid",
          *run("manifest", "noid", "--manifest", str(man_p)),
          expect_msg="unchecked")

    dupe = tmp / "m_dupe.json"
    t5 = json.loads(man_p.read_text())
    t5["records"][1] = dict(t5["records"][0])
    t5["manifest_sha256"] = canonical_manifest_digest(t5)
    dupe.write_text(json.dumps(t5))
    check("duplicate identity (digest honestly recomputed)", "invalid",
          *run("manifest", "dupe", "--manifest", str(dupe),
               "--expect_manifest_sha", t5["manifest_sha256"]),
          expect_msg="duplicate")

    check("expected record count enforced", "invalid",
          *run("manifest", "ok", "--manifest", str(man_p),
               "--expect_manifest_sha", SHA, "--expect_records", "9999"),
          expect_msg="expected 9999")


def test_proposed_manifests():
    print("\nthe ACTUAL proposed manifests (not a fixture)")
    d = REPO / "results" / "audit_v1" / "draft_manifests"
    spec = {"dev_manifest_DRAFT.json": (80, 20),
            "test_manifest_DRAFT.json": (560, 140)}
    for fn, (recs, texts) in spec.items():
        f = d / fn
        if not f.is_file():
            print(f"  SKIP: {fn} not present")
            continue
        sha = json.loads(f.read_text())["manifest_sha256"]
        check(f"{fn} ({texts} texts, {recs} pairs)", "valid",
              *run("manifest", fn, "--manifest", str(f),
                   "--expect_manifest_sha", sha,
                   "--expect_records", str(recs),
                   "--expect_prompt_texts", str(texts)))
    # The superseded drafts must FAIL under the canonical rule -- evidence the
    # migration is real rather than asserted.
    old = d / "superseded_indent2_rule" / "dev_manifest_DRAFT.json"
    if old.is_file():
        cur = json.loads((d / "dev_manifest_DRAFT.json").read_text())["manifest_sha256"]
        check("superseded indent=2 draft fails the canonical rule", "invalid",
              *run("manifest", "superseded", "--manifest", str(old),
                   "--expect_manifest_sha", cur),
              expect_msg="recomputed content")
    # The frozen pilot manifest needs NO migration under this rule.
    pilot = SEQ_ROOT / "eval_manifest.json"
    if pilot.is_file():
        check("frozen pilot manifest validates unmigrated (70 texts, 280 pairs)",
              "valid",
              *run("manifest", "pilot", "--manifest", str(pilot),
                   "--expect_manifest_sha", json.loads(pilot.read_text())["manifest_sha256"],
                   "--expect_records", "280", "--expect_prompt_texts", "70"))


def test_marker(tmp: Path):
    print("\ncompletion marker -- written atomically, and only after validation")
    c = tmp / "contract.json"
    tiny_contract(c)
    models = tmp / "mk"
    (models / "MA").mkdir(parents=True, exist_ok=True)
    write_ckpt(models / "MA" / "delta.bin")
    psha = sha(models / "MA" / "delta.bin")
    d = models / "MAB"
    d.mkdir(parents=True, exist_ok=True)
    write_ckpt(d / "delta.bin")
    base = ["--models_root", str(models), "--contract", str(c),
            "--expect_seed", "17", "--expect_parent", "MA",
            "--expect_target", "dog", "--expect_anchor", "horse",
            "--expect_l2sp", "0", "--expect_steps", "1000", "--write_marker"]

    (d / "train_report.json").write_text(json.dumps(
        train_report("MAB", parent_sha=psha, sha256="wrong")))
    rc, out = run("train", "MAB", *base)
    check("invalid stage writes NO marker", "invalid", rc, out)
    marker = d / "_stage_complete.json"
    ok = not marker.exists()
    (PASS if ok else FAIL).append("no marker on failure")
    print(f"  {'ok  ' if ok else 'FAIL'}  {'marker absent after failure':<58} "
          f"-> {'absent' if ok else 'PRESENT (bad)'}")

    (d / "train_report.json").write_text(json.dumps(
        train_report("MAB", parent_sha=psha, sha256=sha(d / "delta.bin"))))
    rc, out = run("train", "MAB", *base)
    check("valid stage writes the marker", "valid", rc, out)
    ok = marker.exists() and json.loads(marker.read_text()).get("validated") is True
    leftovers = list(d.glob("_stage_complete.json.tmp.*"))
    ok = ok and not leftovers
    (PASS if ok else FAIL).append("marker written, no temp left behind")
    print(f"  {'ok  ' if ok else 'FAIL'}  "
          f"{'marker present, validated:true, no .tmp left':<58} "
          f"-> {'yes' if ok else 'no'}")


def test_legacy_artifact_policy(tmp: Path):
    """The completed-step exception must be bound to SAVED ARTIFACTS by content.

    It used to be a list of training-seed NUMBERS, so a brand-new run using seed
    17 or 29 inherited the exception -- and the proposed new unregularised
    trajectories use exactly those two seeds.
    """
    print("\nlegacy-artifact policy -- scoped to saved artifacts, not seed numbers")
    policy = REPO / "scripts" / "seq" / "legacy_artifact_policy.py"
    registry = REPO / "configs" / "legacy_training_artifacts.json"

    def decide(models_root, ck, reg):
        p = subprocess.run([sys.executable, str(policy), "decide",
                            "--models_root", str(models_root), "--checkpoint", ck,
                            "--registry", str(reg)],
                           capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        return p.returncode, (p.stdout + p.stderr).strip()

    def want(desc, models_root, ck, reg, expect_policy, expect_msg=None):
        rc, out = decide(models_root, ck, reg)
        got = out.split("|", 1)[0]
        ok = rc == 0 and got == expect_policy
        if ok and expect_msg and expect_msg.lower() not in out.lower():
            ok, got = False, f"{got} but message lacks {expect_msg!r}"
        (PASS if ok else FAIL).append(desc)
        print(f"  {'ok  ' if ok else 'FAIL'}  {desc:<58} -> {got}"
              + ("" if ok else f" (wanted {expect_policy})"))
        if not ok:
            print(f"          | {out[:200]}")

    if not registry.is_file():
        print("  FAIL: configs/legacy_training_artifacts.json is absent")
        FAIL.append("legacy registry committed")
        return
    reg = json.loads(registry.read_text())
    arts = reg["artifacts"]
    n_ok = len(arts) == 10
    (PASS if n_ok else FAIL).append("registry holds the ten saved artifacts")
    print(f"  {'ok  ' if n_ok else 'FAIL'}  "
          f"{'registry holds the ten saved artifacts':<58} -> {len(arts)}")
    # Every registered artifact must be recorded as carrying NO counter: that is
    # the entire reason the exception exists.
    no_counter = all(
        a["provenance"]["carries_optimizer_steps_completed"] is False for a in arts)
    (PASS if no_counter else FAIL).append("all registered artifacts lack a counter")
    print(f"  {'ok  ' if no_counter else 'FAIL'}  "
          f"{'all registered artifacts are recorded as counterless':<58} "
          f"-> {no_counter}")
    # And every in-repo report copy must be byte-identical to the saved report.
    copies = all(a["in_repo_copy_is_byte_identical"] for a in arts)
    (PASS if copies else FAIL).append("in-repo report copies are byte-identical")
    print(f"  {'ok  ' if copies else 'FAIL'}  "
          f"{'in-repo report copies are byte-identical':<58} -> {copies}")

    if SEQ_ROOT.exists():
        for seed in (17, 29):
            mroot = SEQ_ROOT / "models" / f"seed{seed}"
            want(f"real seed{seed}/MA is registered -> legacy_optional",
                 mroot, "MA", registry, "legacy_optional",
                 expect_msg="registered saved pilot artifact")
        # The real registry must NOT bless an artifact it does not name, even
        # under a registered checkpoint NAME in a registered seed's directory.
        fake = tmp / "fakeseed"
        (fake / "MA").mkdir(parents=True, exist_ok=True)
        write_ckpt(fake / "MA" / "delta.bin")
        (fake / "MA" / "train_report.json").write_text(json.dumps(
            train_report("MA", parent="M0", target="cat")))
        want("a NEW artifact named MA -> counter", fake, "MA", registry, "counter",
             expect_msg="is NOT the saved pilot artifact")
        # An EDITED saved report loses the exception, and the near miss is named.
        edited = tmp / "edited"
        (edited / "MA").mkdir(parents=True, exist_ok=True)
        src = SEQ_ROOT / "models" / "seed17" / "MA"
        shutil.copy(src / "delta.bin", edited / "MA" / "delta.bin")
        d = json.loads((src / "train_report.json").read_text())
        d["edited_after_registration"] = True
        (edited / "MA" / "train_report.json").write_text(json.dumps(d))
        want("an EDITED saved report -> counter, near miss named",
             edited, "MA", registry, "counter", expect_msg="identity differs")
        # ...but the checkpoint beside it is still protected from being
        # retrained over, which is a separate question answered separately.
        p = subprocess.run([sys.executable, str(policy), "protect",
                            "--models_root", str(edited), "--checkpoint", "MA",
                            "--registry", str(registry)],
                           capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        prot = p.returncode == 0 and "PROTECTED" in p.stdout
        (PASS if prot else FAIL).append("saved checkpoint stays protected")
        print(f"  {'ok  ' if prot else 'FAIL'}  "
              f"{'the saved checkpoint beside it stays PROTECTED':<58} -> "
              f"{'protected' if prot else p.stdout.strip()[:60]}")
        want("an unregistered checkpoint -> not protected is independent",
             SEQ_ROOT / "models" / "seed17", "MAB", registry, "legacy_optional")
    else:
        print("  SKIP: /data assets not present; real-artifact cases skipped")

    # Fail-closed configuration cases, with no /data dependency.
    synth = tmp / "synth"
    (synth / "MA").mkdir(parents=True, exist_ok=True)
    write_ckpt(synth / "MA" / "delta.bin")
    (synth / "MA" / "train_report.json").write_text(json.dumps(train_report("MA")))
    want("an EMPTY registry -> counter (no exception exists)",
         synth, "MA", "", "counter", expect_msg="explicitly empty")
    want("an ABSENT registry -> counter (fails closed)",
         synth, "MA", tmp / "nope.json", "counter", expect_msg="is absent")
    broken = tmp / "broken_registry.json"
    broken.write_text("{ not json")
    want("a MALFORMED registry -> counter (fails closed)",
         synth, "MA", broken, "counter", expect_msg="does not parse")
    noartifacts = tmp / "no_artifacts.json"
    noartifacts.write_text(json.dumps({"policy": {}}))
    want("a registry with no 'artifacts' list -> counter",
         synth, "MA", noartifacts, "counter", expect_msg="no 'artifacts' list")
    want("an artifact that does not exist -> counter",
         synth, "MISSING", registry, "counter", expect_msg="absent")
    # A registry entry matching the report but NOT the checkpoint must not match:
    # both halves of the identity are required.
    half = tmp / "half_registry.json"
    half.write_text(json.dumps({"artifacts": [{
        "id": "half/MA", "training_seed": 29, "checkpoint": "MA",
        "train_report_sha256": sha(synth / "MA" / "train_report.json"),
        "delta_sha256": "0" * 64,
        "provenance": {"produced_by": "fixture"}}]}))
    want("report matches but checkpoint does not -> counter",
         synth, "MA", half, "counter", expect_msg="identity differs")
    full = tmp / "full_registry.json"
    full.write_text(json.dumps({"artifacts": [{
        "id": "full/MA", "training_seed": 29, "checkpoint": "MA",
        "train_report_sha256": sha(synth / "MA" / "train_report.json"),
        "delta_sha256": sha(synth / "MA" / "delta.bin"),
        "provenance": {"produced_by": "fixture"}}]}))
    want("both halves match -> legacy_optional",
         synth, "MA", full, "legacy_optional",
         expect_msg="registered saved pilot artifact")
    # The checkpoint NAME is part of the identity: the same bytes under another
    # name are not the registered artifact.
    (synth / "MAB").mkdir(parents=True, exist_ok=True)
    shutil.copy(synth / "MA" / "delta.bin", synth / "MAB" / "delta.bin")
    shutil.copy(synth / "MA" / "train_report.json", synth / "MAB" / "train_report.json")
    want("the same bytes under a different name -> counter",
         synth, "MAB", full, "counter")


def test_real_artifacts():
    print("\nreal saved artifacts -- revalidated on CPU against the real contracts")
    if not SEQ_ROOT.exists():
        print("  SKIP: /data assets not present on this host")
        return
    contract = REPO / "configs" / "checkpoint_contract.json"
    gs = REPO / "configs" / "generation_settings.json"
    bmc = REPO / "configs" / "base_model_contract.json"
    man = SEQ_ROOT / "eval_manifest.json"
    man_sha = json.loads(man.read_text())["manifest_sha256"]
    spec = {"MA": ("M0", "cat", "horse", 0.0), "MAB": ("MA", "dog", "horse", 0.0),
            "MAB_L2": ("MA", "dog", "horse", 25000.0),
            "MAC": ("MA", "sandwich", "flower", 0.0),
            "MAC_L2": ("MA", "sandwich", "flower", 25000.0)}

    def real_sha(seed, ck):
        import hashlib
        f = SEQ_ROOT / "models" / f"seed{seed}" / ck / "delta.bin"
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    for seed, evald in ((17, SEQ_ROOT / "eval"), (29, SEQ_ROOT / "eval_seed29")):
        mroot = str(SEQ_ROOT / "models" / f"seed{seed}")
        for ck, (par, tgt, anc, l2) in spec.items():
            targs = [ck, "--models_root", mroot, "--contract", str(contract),
                     "--expect_seed", str(seed), "--expect_parent", par,
                     "--expect_target", tgt, "--expect_anchor", anc,
                     "--expect_l2sp", str(l2), "--expect_steps", "1000"]
            # DEFAULT policy must REJECT these legacy reports: they carry no
            # completed-step counter, and the old runtime check was circular.
            check(f"real seed{seed}/{ck}: default policy rejects legacy (no counter)",
                  "invalid", *run("train", *targs), expect_msg="cannot be evidenced")
            # Declared legacy: structural validation passes, completion UNVERIFIED.
            check(f"real seed{seed}/{ck}: legacy_optional -> structural pass",
                  "valid", *run("train", *targs,
                                "--steps_evidence", "legacy_optional"),
                  expect_msg="UNVERIFIED")
        for ck in ("M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"):
            eargs = ["eval", ck, "--eval_root", str(evald), "--manifest", str(man),
                     "--expect_manifest_sha", man_sha,
                     "--expect_gen_settings", str(gs), "--require_images"]
            if ck == "M0":
                eargs += ["--allow_shared_images",
                          "--base_model_contract", str(bmc)]
            else:
                eargs += ["--expect_sha", real_sha(seed, ck)]
            check(f"real seed{seed}/{ck} evaluation (bound to its checkpoint)",
                  "valid", *run(*eargs))
        # And the binding bites on real data: the other seed's checkpoint hash
        # for the same checkpoint name must be rejected.
        other = 29 if seed == 17 else 17
        check(f"real seed{seed}/MA rejected against seed{other}'s MA hash",
              "invalid",
              *run("eval", "MA", "--eval_root", str(evald), "--manifest", str(man),
                   "--expect_manifest_sha", man_sha,
                   "--expect_gen_settings", str(gs), "--require_images",
                   "--expect_sha", real_sha(other, "MA")),
              expect_msg="stale with respect to its checkpoint")

        # ---- the IMAGE-REUSE provenance gate, on the real saved evaluations.
        #
        # Read-only: --provenance_check_only loads no model and writes nothing.
        # Two things must hold on real data. A legitimate resume must be
        # ACCEPTED -- otherwise the gate would block real recovery work -- and
        # the same images offered against the other seed's checkpoint of the
        # same name must be REFUSED, which is the relabelling the gate exists
        # to stop.
        gen = REPO / "scripts" / "seq" / "generate_eval_images.py"
        base_model = json.loads(bmc.read_text())["base_model_dir"]

        def prov(ck, ckpt_seed):
            args = [sys.executable, str(gen), "--manifest", str(man),
                    "--expect_manifest_sha", man_sha,
                    "--base_model_dir", base_model, "--checkpoint_name", ck,
                    "--out_dir", str(evald / ck / "images"),
                    "--report", str(evald / ck / "image_report.json"),
                    "--provenance_check_only"]
            if ck != "M0":
                args += ["--unet_ckpt", str(SEQ_ROOT / "models"
                                            / f"seed{ckpt_seed}" / ck / "delta.bin")]
            p = subprocess.run(args, capture_output=True, text=True,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            return p.returncode, p.stdout + p.stderr

        for ck in ("M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"):
            rc, out = prov(ck, seed)
            okc = rc == 0 and "authorized" in out
            (PASS if okc else FAIL).append(f"real seed{seed}/{ck} reuse authorized")
            print(f"  {'ok  ' if okc else 'FAIL'}  "
                  f"{f'real seed{seed}/{ck}: a legitimate resume is authorized':<58} "
                  f"-> rc={rc}")
            if not okc:
                for line in out.strip().splitlines()[:3]:
                    print(f"          | {line}")
        rc, out = prov("MA", other)
        okc = rc == 3 and "the model changed under the same name" in out
        (PASS if okc else FAIL).append(f"real seed{seed}/MA reuse refused cross-seed")
        print(f"  {'ok  ' if okc else 'FAIL'}  "
              f"{f'real seed{seed}/MA images REFUSED for seed{other} checkpoint':<58} "
              f"-> rc={rc}")
        if not okc:
            for line in out.strip().splitlines()[:3]:
                print(f"          | {line}")
        # Nothing may have been written by any of the above.
        marker = evald / "MA" / "image_report.json"
        before = sha(marker)
        rc, _ = prov("MA", other)
        unchanged = sha(marker) == before
        (PASS if unchanged else FAIL).append(f"seed{seed} image report untouched")
        print(f"  {'ok  ' if unchanged else 'FAIL'}  "
              f"{f'real seed{seed}/MA image_report.json untouched by the gate':<58} "
              f"-> {'unchanged' if unchanged else 'MODIFIED (bad)'}")


def main() -> int:
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"torch unavailable on CPU, cannot run these tests: {e}")
        return 2
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be masked"
    tmp = Path(tempfile.mkdtemp(prefix="valstage_"))
    try:
        test_training(tmp)
        test_evaluation(tmp)
        test_manifest_stage(tmp)
        test_proposed_manifests()
        test_marker(tmp)
        test_legacy_artifact_policy(tmp)
        test_real_artifacts()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
        return 1
    print("ALL VALIDATOR CHECKS PASSED (CPU only; CUDA_VISIBLE_DEVICES='')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
