#!/usr/bin/env python3
"""Validate a training or evaluation stage against the actual run contract.

This replaces the existence/count-only shell sentinels, which accepted a
`delta.bin` holding plain text, a `detections.jsonl` of 280 identical rows, and
a `detections.jsonl` of 280 malformed lines. Every check here reads the artifact
and compares it against (a) the frozen contracts on disk and (b) the settings of
the request being validated -- not merely against the report's own internal
self-consistency.

CPU only. Never imports CUDA, never launches a model, never touches a GPU:
`torch.load(..., map_location="cpu")` is the heaviest operation performed.

Exit codes
----------
0  stage is complete and valid
1  stage is incomplete or invalid (reasons printed, one per line)
2  usage or environment error (a check could not be performed at all)

A check that cannot be *performed* (missing contract, torch unavailable) is
reported as `UNCHECKED` and fails the stage. It is never silently skipped and
never counted as a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

REQUIRED_TRAIN_KEYS = [
    "checkpoint", "parent", "training_seed", "l2sp_weight", "new_deletion_target",
    "anchor_concept", "child_checkpoint", "effective_hyperparameters",
    "parent_verification", "started_utc", "finished_utc", "runtime",
]
REQUIRED_DETECT_ROW_KEYS = [
    "checkpoint", "prompt_id", "category", "prompt_family", "prompt_index",
    "gen_seed", "image_path", "image_sha256", "detections", "max_score_target",
    "hit",
]
THRESHOLDS = ("0.3", "0.5", "0.7")
STAGE_MARKER = "_stage_complete.json"


class Result:
    def __init__(self, stage: str, name: str):
        self.stage, self.name = stage, name
        self.fail: list[str] = []
        self.unchecked: list[str] = []
        self.info: dict = {}

    def bad(self, msg: str) -> None:
        self.fail.append(msg)

    def cannot(self, msg: str) -> None:
        self.unchecked.append(msg)

    @property
    def ok(self) -> bool:
        return not self.fail and not self.unchecked

    def report(self) -> int:
        tag = f"{self.stage}:{self.name}"
        if self.ok:
            print(f"VALID    {tag}")
            for k, v in self.info.items():
                print(f"           {k} = {v}")
            return 0
        print(f"INVALID  {tag}")
        for m in self.fail:
            print(f"  FAIL      {m}")
        for m in self.unchecked:
            print(f"  UNCHECKED {m}  (cannot verify -> treated as failure)")
        return 1


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_json(p: Path):
    """Parse JSON, distinguishing 'absent' from 'present but unparseable'."""
    if not p.exists():
        return None, f"{p.name} absent"
    if p.stat().st_size == 0:
        return None, f"{p.name} is empty"
    try:
        return json.loads(p.read_text()), None
    except Exception as e:  # truncated mid-write, or not JSON at all
        return None, f"{p.name} does not parse as JSON ({type(e).__name__}: {e})"


# ------------------------------------------------------------------ training
def validate_training(args) -> Result:
    r = Result("train", args.name)
    d = Path(args.models_root) / args.name
    rep_p, ck_p = d / "train_report.json", d / "delta.bin"

    rep, err = load_json(rep_p)
    if err:
        r.bad(err)
        return r

    # 1. report schema
    for k in REQUIRED_TRAIN_KEYS:
        if k not in rep:
            r.bad(f"train_report.json missing required key '{k}'")
    if not rep.get("finished_utc"):
        r.bad("finished_utc absent or empty: the run did not reach its end")
    if r.fail:
        return r

    eh = rep.get("effective_hyperparameters") or {}
    cc = rep.get("child_checkpoint") or {}

    # 2. run identity -- the report must describe THIS request, not another one.
    #    These are the settings the caller is about to rely on; a stale report
    #    left by a different configuration must not satisfy the sentinel.
    def ident(field, got, want, cast=str):
        if want is None:
            return
        if cast(got) != cast(want):
            r.bad(f"{field} mismatch: report has {got!r}, this request is {want!r}")

    ident("checkpoint name", rep.get("checkpoint"), args.name)
    ident("training_seed", rep.get("training_seed"), args.expect_seed, int)
    ident("parent", rep.get("parent"), args.expect_parent)
    ident("new_deletion_target", rep.get("new_deletion_target"), args.expect_target)
    ident("anchor_concept", rep.get("anchor_concept"), args.expect_anchor)
    if args.expect_l2sp is not None:
        if float(rep.get("l2sp_weight", -1)) != float(args.expect_l2sp):
            r.bad(f"l2sp_weight mismatch: report has {rep.get('l2sp_weight')!r}, "
                  f"this request is {args.expect_l2sp!r}")
    if args.expect_seed is not None and int(eh.get("seed", -1)) != int(args.expect_seed):
        r.bad(f"effective_hyperparameters.seed is {eh.get('seed')!r}, "
              f"this request is {args.expect_seed!r}")

    # 3. optimizer-step count actually completed
    want_steps = args.expect_steps
    if want_steps is not None:
        got = eh.get("iterations_requested")
        if got is None or int(got) != int(want_steps):
            r.bad(f"iterations_requested is {got!r}, this request is {want_steps}")
        cap = eh.get("epoch_capacity_steps")
        if cap is not None and int(cap) < int(want_steps):
            r.bad(f"epoch_capacity_steps {cap} < requested {want_steps}: the run "
                  f"could not have completed the requested steps")
        # l2sp_trace is a REGULARISATION-loss trace, not a step counter: it is
        # populated only when l2sp_weight > 0 and is legitimately
        # steps_recorded=0 on every unregularised arm. Only use it as step
        # evidence where it is actually recorded.
        regularised = args.expect_l2sp is not None and float(args.expect_l2sp) > 0
        trace = rep.get("l2sp_trace") or {}
        rec = trace.get("steps_recorded")
        if regularised:
            if rec is None:
                r.bad("l2sp_trace.steps_recorded absent on a regularised arm")
            elif int(rec) != int(want_steps):
                r.bad(f"l2sp_trace.steps_recorded is {rec}, expected "
                      f"{want_steps}: the trajectory is short")
        else:
            # No direct counter of completed optimizer steps exists for an
            # unregularised arm in this report schema. Say so rather than
            # implying the step count was confirmed.
            r.info["step_count_evidence"] = (
                "runtime cross-check only -- this schema records no direct "
                "completed-step counter for unregularised arms")
        rt = rep.get("runtime") or {}
        sps, secs = rt.get("seconds_per_optimizer_step"), rt.get("train_seconds")
        if sps and secs and sps > 0:
            implied = secs / sps
            if abs(implied - int(want_steps)) > max(5, 0.02 * int(want_steps)):
                r.bad(f"runtime implies ~{implied:.0f} optimizer steps, "
                      f"expected {want_steps}")

    # 4. parent identity: the recorded parent hash must match the parent file
    pv = rep.get("parent_verification") or {}
    if args.expect_parent and args.expect_parent != "M0":
        parent_file = Path(args.models_root) / args.expect_parent / "delta.bin"
        claimed = pv.get("parent_sha256")
        if not parent_file.exists():
            r.bad(f"parent checkpoint {parent_file} absent, so this child's "
                  f"lineage cannot be confirmed")
        elif not claimed:
            r.bad("parent_verification.parent_sha256 absent")
        else:
            actual = sha256_file(parent_file)
            if actual != claimed:
                r.bad(f"parent hash mismatch: report recorded {claimed[:12]}…, "
                      f"parent on disk is {actual[:12]}… (parent changed since "
                      f"this child was trained)")
            r.info["parent_sha256"] = claimed[:12] + "…"
        if pv.get("PASS") is not True:
            r.bad(f"parent_verification.PASS is {pv.get('PASS')!r}")
        dev = pv.get("max_abs_deviation_after_load")
        if dev is None or float(dev) != 0.0:
            r.bad(f"parent load deviation is {dev!r}, expected exactly 0.0")
    if args.expect_l2sp and float(args.expect_l2sp) > 0:
        lr = rep.get("l2sp_reference_check") or {}
        if lr.get("PASS") is not True:
            r.bad(f"l2sp_reference_check.PASS is {lr.get('PASS')!r}")
        d2 = lr.get("max_abs_deviation_from_parent")
        if d2 is None or float(d2) != 0.0:
            r.bad(f"L2-SP reference deviation from parent is {d2!r}, expected 0.0")

    # 5. the checkpoint file itself
    if not ck_p.exists():
        r.bad("delta.bin absent")
        return r
    if ck_p.stat().st_size == 0:
        r.bad("delta.bin is zero bytes")
        return r

    contract, cerr = load_json(Path(args.contract))
    if cerr:
        r.cannot(f"checkpoint contract unavailable: {cerr}")
        return r

    want_bytes = contract.get("expected_file_bytes")
    if want_bytes and ck_p.stat().st_size != want_bytes:
        r.bad(f"delta.bin is {ck_p.stat().st_size} bytes, contract expects "
              f"{want_bytes}")

    # Recompute the hash: the report's claimed SHA is an assertion, not evidence.
    actual_sha = sha256_file(ck_p)
    claimed_sha = cc.get("sha256")
    if not claimed_sha:
        r.bad("child_checkpoint.sha256 absent from the report")
    elif actual_sha != claimed_sha:
        r.bad(f"delta.bin hash mismatch: report claims {claimed_sha[:12]}…, "
              f"file is actually {actual_sha[:12]}… (file changed or report is "
              f"for a different checkpoint)")
    if args.expect_sha and actual_sha != args.expect_sha:
        r.bad(f"delta.bin hash {actual_sha[:12]}… != expected "
              f"{args.expect_sha[:12]}…")
    r.info["delta_sha256"] = actual_sha[:12] + "…"

    # 6. structure, dtypes, finiteness, CPU reload. This is what rejects a
    #    plain-text file carrying a plausible report.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        import torch
    except Exception as e:
        r.cannot(f"torch unavailable, cannot verify tensors ({e})")
        return r
    try:
        obj = torch.load(ck_p, map_location="cpu", weights_only=False)
    except Exception as e:
        r.bad(f"delta.bin does not load as a torch checkpoint on CPU "
              f"({type(e).__name__}: {e})")
        return r

    top = contract.get("top_level_key", "unet")
    if not isinstance(obj, dict) or top not in obj:
        r.bad(f"delta.bin has no {top!r} key (top level: "
              f"{list(obj)[:5] if isinstance(obj, dict) else type(obj).__name__})")
        return r
    sd = obj[top]
    if not isinstance(sd, dict):
        r.bad(f"delta.bin[{top!r}] is {type(sd).__name__}, expected a state dict")
        return r

    want = contract["tensors"]
    missing = sorted(set(want) - set(sd))
    extra = sorted(set(sd) - set(want))
    if missing:
        r.bad(f"{len(missing)} contract tensor(s) missing, e.g. {missing[:2]}")
    if extra:
        r.bad(f"{len(extra)} unexpected tensor(s), e.g. {extra[:2]}")
    if len(sd) != contract["n_tensors"]:
        r.bad(f"{len(sd)} tensors, contract expects {contract['n_tensors']}")

    total, bad_shape, bad_dtype, nonfinite = 0, [], [], []
    for k, spec in want.items():
        t = sd.get(k)
        if t is None:
            continue
        if list(getattr(t, "shape", [])) != spec["shape"]:
            bad_shape.append(f"{k}: {tuple(getattr(t,'shape',()))} != "
                             f"{tuple(spec['shape'])}")
        if str(getattr(t, "dtype", "")) != spec["dtype"]:
            bad_dtype.append(f"{k}: {getattr(t,'dtype',None)} != {spec['dtype']}")
        try:
            total += t.numel()
            if not bool(torch.isfinite(t).all()):
                nonfinite.append(k)
        except Exception as e:
            r.bad(f"tensor {k} is not inspectable ({type(e).__name__}: {e})")
    if bad_shape:
        r.bad(f"{len(bad_shape)} shape mismatch(es): {bad_shape[:2]}")
    if bad_dtype:
        r.bad(f"{len(bad_dtype)} dtype mismatch(es): {bad_dtype[:2]}")
    if nonfinite:
        r.bad(f"{len(nonfinite)} tensor(s) contain NaN or Inf, e.g. "
              f"{nonfinite[:2]}")
    if total != contract["total_elements"]:
        r.bad(f"{total} parameters, contract expects {contract['total_elements']}")
    else:
        r.info["parameters"] = total
    r.info["tensors"] = len(sd)
    return r


# ---------------------------------------------------------------- evaluation
def validate_evaluation(args) -> Result:
    r = Result("eval", args.name)
    d = Path(args.eval_root) / args.name
    det_p = d / "detections.jsonl"

    man, merr = load_json(Path(args.manifest))
    if merr:
        r.cannot(f"frozen manifest unavailable: {merr}")
        return r
    records = man.get("records")
    if not records:
        r.cannot("frozen manifest has no 'records'; expected count cannot be derived")
        return r

    # Expected count and identities are DERIVED from the frozen manifest.
    expected = {(x["category"], x["prompt_id"], int(x["gen_seed"])) for x in records}
    n_expected = len(records)
    if len(expected) != n_expected:
        r.cannot(f"manifest itself has duplicate identities "
                 f"({n_expected} records, {len(expected)} unique)")
        return r
    r.info["expected_from_manifest"] = n_expected

    man_sha = man.get("manifest_sha256")
    if args.expect_manifest_sha and man_sha != args.expect_manifest_sha:
        r.bad(f"manifest sha256 {str(man_sha)[:12]}… != expected "
              f"{args.expect_manifest_sha[:12]}…")

    if not det_p.exists():
        r.bad("detections.jsonl absent")
        return r
    if det_p.stat().st_size == 0:
        r.bad("detections.jsonl is empty")
        return r

    # Parse EVERY row. A file of malformed lines must not pass a line count.
    rows, malformed = [], []
    for i, line in enumerate(det_p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except Exception as e:
            malformed.append(f"line {i}: {type(e).__name__}")
            continue
        if not isinstance(o, dict):
            malformed.append(f"line {i}: {type(o).__name__}, expected object")
            continue
        rows.append(o)
    if malformed:
        r.bad(f"{len(malformed)} unparseable row(s), e.g. {malformed[:3]}")
        return r

    for k in REQUIRED_DETECT_ROW_KEYS:
        bad = [i for i, o in enumerate(rows, 1) if k not in o]
        if bad:
            r.bad(f"{len(bad)} row(s) missing required field '{k}', "
                  f"e.g. line {bad[0]}")
    if r.fail:
        return r
    for i, o in enumerate(rows, 1):
        h = o.get("hit")
        if not isinstance(h, dict) or any(t not in h for t in THRESHOLDS):
            r.bad(f"line {i}: 'hit' must carry all of {THRESHOLDS}, got "
                  f"{sorted(h) if isinstance(h, dict) else type(h).__name__}")
            break

    if len(rows) != n_expected:
        r.bad(f"{len(rows)} scored row(s), manifest expects {n_expected}")

    # Exact set equality on identities. This is what rejects 280 identical rows:
    # they collapse to one unique identity and leave the rest missing.
    got_ids = [(o["category"], o["prompt_id"], int(o["gen_seed"])) for o in rows]
    uniq = set(got_ids)
    dups = {k: c for k, c in Counter(got_ids).items() if c > 1}
    missing, extra = sorted(expected - uniq), sorted(uniq - expected)
    if dups:
        r.bad(f"{len(dups)} duplicated identity/identities, e.g. "
              f"{list(dups)[:2]} (appearing up to {max(dups.values())}x)")
    if missing:
        r.bad(f"{len(missing)} manifest identity/identities unscored, e.g. "
              f"{missing[:2]}")
    if extra:
        r.bad(f"{len(extra)} scored identity/identities absent from the "
              f"manifest, e.g. {extra[:2]}")

    # Per-axis coverage, so a shortfall is localised rather than just counted.
    for axis, key in (("category", "category"), ("prompt_family", "prompt_family"),
                      ("gen_seed", "gen_seed")):
        want = Counter(str(x[key]) for x in records)
        got = Counter(str(o[key]) for o in rows)
        if want != got:
            diff = {k: (want.get(k, 0), got.get(k, 0)) for k in set(want) | set(got)
                    if want.get(k, 0) != got.get(k, 0)}
            r.bad(f"{axis} coverage differs from manifest (want, got): {diff}")

    # Distinct image bytes: 280 rows all naming one image is not an evaluation.
    hashes = [o.get("image_sha256") for o in rows]
    if len(set(hashes)) != len(rows):
        c = Counter(hashes)
        r.bad(f"{len(rows) - len(set(hashes))} row(s) repeat an image_sha256 "
              f"(most common appears {c.most_common(1)[0][1]}x)")

    # Checkpoint identity recorded in the rows.
    cks = {o.get("checkpoint") for o in rows}
    if cks != {args.name}:
        r.bad(f"rows carry checkpoint label(s) {sorted(map(str, cks))}, "
              f"expected only {args.name!r}")

    # Companion artifacts and their configuration identity.
    img_rep, ierr = load_json(d / "image_report.json")
    det_rep, derr = load_json(d / "detect_report.json")
    if ierr:
        r.bad(f"image_report.json: {ierr}")
    if derr:
        r.bad(f"detect_report.json: {derr}")
    if img_rep:
        if img_rep.get("complete") is not True:
            r.bad(f"image_report.complete is {img_rep.get('complete')!r}")
        if img_rep.get("checkpoint") != args.name:
            r.bad(f"image_report.checkpoint is {img_rep.get('checkpoint')!r}, "
                  f"expected {args.name!r}")
        if int(img_rep.get("generated_or_reused", -1)) != n_expected:
            r.bad(f"image_report generated_or_reused is "
                  f"{img_rep.get('generated_or_reused')!r}, expected {n_expected}")
        if man_sha and img_rep.get("manifest_sha256") != man_sha:
            r.bad(f"image_report was produced against manifest "
                  f"{str(img_rep.get('manifest_sha256'))[:12]}…, current manifest "
                  f"is {str(man_sha)[:12]}… (stale evaluation)")
        gs = img_rep.get("generation_settings") or {}
        if args.expect_gen_settings:
            want_gs, gerr = load_json(Path(args.expect_gen_settings))
            if gerr:
                r.cannot(f"expected generation settings unavailable: {gerr}")
            else:
                for k, v in want_gs.items():
                    if k.startswith("_"):      # contract metadata, not a setting
                        continue
                    if k not in gs:
                        r.bad(f"generation setting {k!r} absent from "
                              f"image_report.generation_settings")
                    elif gs[k] != v:
                        r.bad(f"generation setting {k} is {gs[k]!r}, expected {v!r}")
        # The checkpoint actually loaded for generation.
        cl = img_rep.get("checkpoint_load") or {}
        if args.name != "M0":
            if cl.get("applied") is not True:
                r.bad(f"image_report.checkpoint_load.applied is "
                      f"{cl.get('applied')!r}: the delta may not have been used")
            if args.expect_sha and cl.get("sha256") and cl["sha256"] != args.expect_sha:
                r.bad(f"images were generated from checkpoint "
                      f"{cl['sha256'][:12]}…, expected {args.expect_sha[:12]}…")
    if det_rep:
        if det_rep.get("complete") is not True:
            r.bad(f"detect_report.complete is {det_rep.get('complete')!r}")
        if int(det_rep.get("images_scored", -1)) != len(rows):
            r.bad(f"detect_report.images_scored is "
                  f"{det_rep.get('images_scored')!r}, jsonl has {len(rows)}")
        if det_rep.get("checkpoint") != args.name:
            r.bad(f"detect_report.checkpoint is {det_rep.get('checkpoint')!r}, "
                  f"expected {args.name!r}")

    # Image files. For a shared/symlinked M0 the recorded paths legitimately
    # point outside this directory, so the contract is explicit about it.
    if args.require_images:
        absent = [o["image_path"] for o in rows if not Path(o["image_path"]).exists()]
        if absent:
            r.bad(f"{len(absent)} recorded image path(s) do not exist, e.g. "
                  f"{absent[:2]}")
        # Compare LITERAL paths, not resolved ones: `eval_seed29/M0` is a symlink
        # to `eval/M0`, so both sides resolve to the same place and a
        # resolve-based check cannot see the reuse at all. The directory being a
        # symlink is itself evidence of reuse.
        outside = [o["image_path"] for o in rows
                   if not str(o["image_path"]).startswith(str(d) + os.sep)]
        reused = bool(outside) or d.is_symlink()
        if reused and not args.allow_shared_images:
            detail = (f"{len(outside)} recorded image path(s) lie outside {d}"
                      if outside else "")
            if d.is_symlink():
                detail = (detail + "; " if detail else "") + \
                         f"{d} is a symlink to {os.readlink(d)}"
            r.bad(f"this evaluation reuses images stored elsewhere ({detail}) but "
                  f"--allow_shared_images was not given. A reused evaluation set "
                  f"must be declared explicitly, never inferred from the layout, "
                  f"so that it is counted once and not treated as independent.")
        elif reused:
            r.info["shared_images"] = (
                f"declared reuse: {len(outside)} path(s) outside {d.name}/"
                + (f", dir is a symlink to {os.readlink(d)}" if d.is_symlink() else ""))
    return r


# -------------------------------------------------------------------- marker
def write_marker(args, r: Result) -> None:
    """Write completion metadata atomically, only after validation passed."""
    base = (Path(args.models_root) / args.name if args.stage == "train"
            else Path(args.eval_root) / args.name)
    if not base.is_dir():
        return
    payload = {
        "stage": args.stage, "name": args.name, "validated": True,
        "validator": "scripts/seq/validate_stage.py",
        "checks_passed": r.info,
        "validated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
    }
    tmp = base / (STAGE_MARKER + f".tmp.{os.getpid()}")
    final = base / STAGE_MARKER
    try:
        tmp.write_text(json.dumps(payload, indent=2))
        fd = os.open(tmp, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, final)   # atomic within one filesystem
    except Exception as e:
        print(f"  warning: could not write completion marker ({e})",
              file=sys.stderr)
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["train", "eval"])
    ap.add_argument("name")
    ap.add_argument("--models_root", default="")
    ap.add_argument("--eval_root", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--contract", default="configs/checkpoint_contract.json")
    ap.add_argument("--expect_seed", type=int, default=None)
    ap.add_argument("--expect_parent", default=None)
    ap.add_argument("--expect_target", default=None)
    ap.add_argument("--expect_anchor", default=None)
    ap.add_argument("--expect_l2sp", type=float, default=None)
    ap.add_argument("--expect_steps", type=int, default=None)
    ap.add_argument("--expect_sha", default=None)
    ap.add_argument("--expect_manifest_sha", default=None)
    ap.add_argument("--expect_gen_settings", default=None)
    ap.add_argument("--require_images", action="store_true")
    ap.add_argument("--allow_shared_images", action="store_true",
                    help="declare that this evaluation reuses images stored "
                         "elsewhere (the shared M0 contract)")
    ap.add_argument("--write_marker", action="store_true")
    args = ap.parse_args()

    if args.stage == "train" and not args.models_root:
        print("--models_root is required for stage 'train'", file=sys.stderr)
        return 2
    if args.stage == "eval" and not (args.eval_root and args.manifest):
        print("--eval_root and --manifest are required for stage 'eval'",
              file=sys.stderr)
        return 2

    r = (validate_training(args) if args.stage == "train"
         else validate_evaluation(args))
    rc = r.report()
    if rc == 0 and args.write_marker:
        write_marker(args, r)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
