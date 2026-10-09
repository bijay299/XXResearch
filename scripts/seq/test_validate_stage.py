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
                 drop=None, steps_recorded=None, train_seconds=None) -> dict:
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
                    "seconds_per_optimizer_step": 0.83},
    }
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
                                 "prompt": f"a {c}", "gen_seed": s,
                                 "image_name": f"{c}_{fam}_{i}_seed{s}.jpg"})
    m = {"records": recs, "manifest_sha256": sha256, "gen_seeds": list(seeds),
         "frozen_before_any_editing": True}
    path.write_text(json.dumps(m))
    return m


def write_eval(d: Path, man: dict, ck="MA", *, mode="valid", make_images=True,
               man_sha="MANSHA", gen_settings=None, ckpt_sha="CKSHA"):
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
    (d / "image_report.json").write_text(json.dumps({
        "checkpoint": ck, "complete": True, "expected_images": len(recs),
        "generated_or_reused": len(recs), "manifest_sha256": man_sha,
        "generation_settings": gs,
        "checkpoint_load": {"applied": True, "sha256": ckpt_sha},
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
    # l2sp_trace is a regularisation trace, populated only when l2sp > 0, and is
    # legitimately steps_recorded=0 on an unregularised arm (confirmed against
    # all 10 real pilot reports). It is step evidence only for an L2 arm.
    l2base = [x if x != "0" else "25000" for x in base]
    setup(l2=25000.0, steps_recorded=400)
    check("L2 arm: trace records 400 of 1000 steps", "invalid",
          *run("train", "MAB", *l2base), expect_msg="short")
    setup(l2=25000.0, steps_recorded=None)
    check("L2 arm: trace records all 1000 steps", "valid",
          *run("train", "MAB", *l2base))
    setup(steps_recorded=0)
    check("unregularised arm: steps_recorded=0 is legitimate", "valid",
          *run("train", "MAB", *base))
    setup(train_seconds=100.0)
    check("runtime implies far fewer steps than requested", "invalid",
          *run("train", "MAB", *base), expect_msg="optimizer steps")

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
    gs_p = tmp / "gs.json"
    gs_p.write_text(json.dumps({
        "_note": "metadata key, must be ignored",
        "num_inference_steps": 30, "guidance_scale": 7.5, "resolution": 512,
        "scheduler": "pipeline default (PNDM for SD-1.5)", "dtype": "float16",
        "negative_prompt": None, "identical_at_every_checkpoint": True}))
    root = tmp / "eval"
    base = ["--eval_root", str(root), "--manifest", str(man_p),
            "--expect_manifest_sha", "MANSHA",
            "--expect_gen_settings", str(gs_p), "--require_images"]

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
    check("caller-expected manifest sha differs", "invalid",
          *run("eval", "MA", "--eval_root", str(root), "--manifest", str(man_p),
               "--expect_manifest_sha", "DIFFERENT", "--require_images"),
          expect_msg="!= expected")
    setup(gen_settings={"num_inference_steps": 50, "guidance_scale": 7.5,
                        "resolution": 512,
                        "scheduler": "pipeline default (PNDM for SD-1.5)",
                        "dtype": "float16", "negative_prompt": None,
                        "identical_at_every_checkpoint": True})
    check("generation settings differ from the contract", "invalid",
          *run("eval", "MA", *base), expect_msg="num_inference_steps")
    setup()
    check("images generated from another checkpoint hash", "invalid",
          *run("eval", "MA", *base, "--expect_sha", "0" * 64),
          expect_msg="generated from checkpoint")

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

    # Shared-image (M0) contract: reuse must be declared, never inferred.
    # Mirrors the real layout -- eval_seed29/M0 is a symlink to eval/M0 -- so
    # the link keeps the name M0 and only its parent root differs.
    shared_src = root / "M0"
    write_eval(shared_src, man, "M0")
    other_root = tmp / "eval_seed99"
    other_root.mkdir(parents=True, exist_ok=True)
    link = other_root / "M0"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(shared_src)
    check("reused evaluation set, reuse NOT declared", "invalid",
          *run("eval", "M0", "--eval_root", str(other_root),
               "--manifest", str(man_p), "--require_images"),
          expect_msg="must be declared")
    check("reused evaluation set, reuse declared", "valid",
          *run("eval", "M0", "--eval_root", str(other_root),
               "--manifest", str(man_p), "--require_images",
               "--allow_shared_images"))

    # expected count derives from the manifest, not a hard-coded 280
    man2_p = tmp / "manifest_small.json"
    man2 = manifest(man2_p, n_prompts=1, cats=("cat",), seeds=(101,))
    d2 = root / "SMALL"
    write_eval(d2, man2, "SMALL")
    check(f"count derived from manifest ({len(man2['records'])} rows, not 280)",
          "valid", *run("eval", "SMALL", "--eval_root", str(root),
                        "--manifest", str(man2_p), "--require_images"))
    check("same artifacts judged against the larger manifest", "invalid",
          *run("eval", "SMALL", "--eval_root", str(root),
               "--manifest", str(man_p), "--require_images"),
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
          *run("eval", "MA", "--eval_root", str(root), "--manifest", str(bad_man)),
          expect_msg="unchecked")


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


def test_real_artifacts():
    print("\nreal saved artifacts -- revalidated on CPU against the real contracts")
    if not SEQ_ROOT.exists():
        print("  SKIP: /data assets not present on this host")
        return
    contract = REPO / "configs" / "checkpoint_contract.json"
    gs = REPO / "configs" / "generation_settings.json"
    man = SEQ_ROOT / "eval_manifest.json"
    man_sha = json.loads(man.read_text())["manifest_sha256"]
    spec = {"MA": ("M0", "cat", "horse", 0.0), "MAB": ("MA", "dog", "horse", 0.0),
            "MAB_L2": ("MA", "dog", "horse", 25000.0),
            "MAC": ("MA", "sandwich", "flower", 0.0),
            "MAC_L2": ("MA", "sandwich", "flower", 25000.0)}
    for seed, evald in ((17, SEQ_ROOT / "eval"), (29, SEQ_ROOT / "eval_seed29")):
        for ck, (par, tgt, anc, l2) in spec.items():
            rc, out = run("train", ck, "--models_root",
                          str(SEQ_ROOT / "models" / f"seed{seed}"),
                          "--contract", str(contract), "--expect_seed", str(seed),
                          "--expect_parent", par, "--expect_target", tgt,
                          "--expect_anchor", anc, "--expect_l2sp", str(l2),
                          "--expect_steps", "1000")
            check(f"real seed{seed}/{ck} training", "valid", rc, out)
        for ck in ("M0", "MA", "MAB", "MAB_L2", "MAC", "MAC_L2"):
            extra = ["--allow_shared_images"] if ck == "M0" else []
            rc, out = run("eval", ck, "--eval_root", str(evald),
                          "--manifest", str(man), "--expect_manifest_sha", man_sha,
                          "--expect_gen_settings", str(gs), "--require_images",
                          *extra)
            check(f"real seed{seed}/{ck} evaluation", "valid", rc, out)


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
        test_marker(tmp)
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
