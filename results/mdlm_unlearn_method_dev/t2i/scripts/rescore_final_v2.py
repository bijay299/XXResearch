"""CORRECTIVE RE-EVALUATION of the FINAL split (defect D-013) -- regeneration PLUS corrected scoring.

WHAT THIS IS
------------
`outputs/final_*.json` (12 records, 15 images each, 180 images) were scored with label TEXT taken
from `gen_eval.CONCEPTS` keys (== `gen_eval.LAB_KEY`) rather than from `gen_eval.LABELS`, the text
every H number in the corpus uses. NINE of the twelve strings differ. This script re-runs those
arms and scores them under BOTH label-text sets on THE SAME images.

WHAT THIS IS *NOT*
------------------
*   NOT a fresh held-out confirmation. The FINAL split was inspected in FINDING65 and FINDING66.
    This is a CORRECTIVE RE-EVALUATION OF AN ALREADY-INSPECTED SPLIT; its numbers are development
    measurements and must never be reported as a confirmatory held-out result. Every emitted record
    carries `split_status: "INSPECTED_BEFORE_THIS_RUN"` and `evaluation_role:
    "CORRECTIVE_RE_EVALUATION"`.
*   NOT a byte-identical reproduction. The original images were NEVER SAVED. Matching seeds are a
    REASON TO EXPECT similar images, not proof of them: fp16 generation depends on library and
    driver versions, kernel selection and device, none of which are recorded for the original run.
    Determinism is treated as an ASSUMPTION TO BE TESTED, and this script tests it -- see
    `determinism_probe` below.
*   NOT a resolution of the mismatch's DIRECTION. See `clip_scorer.label_set_diff()`. The only
    sanctioned reading until this run produces numbers is: "The evaluator used inconsistent label
    text. The effect on scores and conclusions is unresolved pending corrected evaluation." This
    script therefore reports the paired difference; it does not predict its sign, and it does not
    treat arms whose own label text is unchanged (church, parachute, tench) as a priori unaffected
    -- a closed-set argmax couples every label to every other.

HOW IT AVOIDS DESTROYING HISTORY
--------------------------------
Everything is written under a NEW, UNIQUE run directory `outputs/rescore/<run_id>/`. The script
refuses outright, before touching the GPU, if that directory exists, and a hard guard rejects any
write path that could land on `outputs/final_*.json` or anywhere else in the historical output
tree. Every write is atomic (temp file in the destination directory + `os.replace`).

WHAT IS PERSISTED (so that no FUTURE question needs regeneration)
----------------------------------------------------------------
per arm:  every generated PNG; the L2-normalised CLIP image embeddings (.npz, float32); per-image
          scores for ALL TWELVE labels under BOTH label sets; per-arm status and provenance.
per run:  run manifest with provenance, a paired label-set comparison, and the determinism probe
          against the historical records.

STATUS IS EXPLICIT, NEVER IMPLIED
---------------------------------
Each arm ends in exactly one of: `complete`, `missing_checkpoint`, `failed`. The run ends in
`complete` ONLY IF every requested arm is `complete` AND every completed arm reproduced its
historical record exactly (`determinism_probe`). A run that produced numbers but cannot be placed
against history is `complete_not_comparable_to_history`; a run where only some arms are comparable
is `complete_comparability_mixed`. Otherwise `partial` (failing arms named), `failed`, or
`incomplete`. A run interrupted before the end keeps its initial `incomplete` status on disk, so a
crashed run can never be read as a finished one. This matters scientifically: an all-DISCREPANT run
is one that CANNOT be compared to the historical numbers, and must never be reported as a
successful correction.

NOTE ON REGENERABILITY (observed 2026-10-26): the eight attacked checkpoints behind the six `ATK_*`
historical records (`attacked/E6B_*`, `E6C_*`, `E6D_*`) are ABSENT from disk. Those arms are
emitted as `missing_checkpoint`, not skipped. They are NOT re-runnable from the retained artefacts
and their historical numbers cannot be corrected by this pipeline at all; that limitation is a
result of this run, recorded in the manifest, not a bug to work around.

Run (ONE GPU):
    bash -c 'source /data/bijaypandey/t2i_unlearn/env.sh && \
      CUDA_VISIBLE_DEVICES=<gpu> python /data/bijaypandey/t2i_unlearn/scripts/rescore_final_v2.py'
Dry run, no GPU, no generation (prints the plan and the provenance it would record):
    python scripts/rescore_final_v2.py --plan-only
"""
import argparse
import hashlib
import importlib.util as iu
import json
import os
import platform
import re
import sys
import tempfile
import time

P = "/data/bijaypandey/t2i_unlearn"
OUT = os.path.join(P, "outputs")
RESCORE_ROOT = os.path.join(OUT, "rescore")
SCRIPTS = os.path.join(P, "scripts")

RESCORE_VERSION = "rescore_final/v2.0.0"

#: Paths this script must NEVER write. Belt and braces: the run directory is unique anyway, but a
#: future edit that builds an output path wrongly is caught by `_guard_path` rather than by a
#: reader noticing a historical record changed.
PROTECTED_PREFIXES = (os.path.join(OUT, "final_"), os.path.join(OUT, "gen_"),
                      os.path.join(P, "adapters"), os.path.join(P, "attacked"))

# ------------------------------------------------------------------ UNMODIFIED-FILE BASELINE
#: Reviewer defect 7. The claim "gen_eval.py and attack_relearn.py are unmodified" rested on MTIME
#: alone: the only prior hash record (outputs/provenance_manifest.json) PREDATES this repair, and
#: its per-file hashes are truncated to 16 hex characters besides. These full sha256 values are
#: recorded here on 2026-10-26, after the repair, so the claim has a real basis GOING FORWARD.
#:
#: WHAT THIS DOES AND DOES NOT SUPPORT. From this baseline onward, `verify_unmodified_baseline()`
#: proves byte-identity against it. It does NOT retroactively prove the files were unmodified
#: BEFORE the baseline was taken -- for that interval the evidence is still only mtime, and that is
#: all that may be claimed.
UNMODIFIED_BASELINE_TAKEN_AT = "2026-10-26"
UNMODIFIED_BASELINE_SHA256 = {
    "gen_eval.py": "6f978e60870486b4e844580c3b480ac62ce4a59037eb2d5534756e26d131f51a",
    "attack_relearn.py": "ec5fe2249b19df3b60e551f8800ed3fb7c7474a814e312578f1e88dd44fb40e2",
}


def verify_unmodified_baseline():
    """Hash-verify the two files this repair must not touch, against the recorded baseline."""
    rows, ok = {}, True
    for fn, want in UNMODIFIED_BASELINE_SHA256.items():
        p = os.path.join(SCRIPTS, fn)
        got = file_sha(p)[0] if os.path.exists(p) else None
        match = (got == want) if want != "__FILL__" else None
        rows[fn] = {"expected_sha256": want, "actual_sha256": got, "matches": match}
        if match is False:
            ok = False
    return {"ok": ok, "baseline_taken_at": UNMODIFIED_BASELINE_TAKEN_AT, "files": rows,
            "scope_note": ("Byte-identity is proven only from baseline_taken_at onward. For the "
                           "period before that date the evidence is mtime only, and no stronger "
                           "claim is made: the prior hash record "
                           "(outputs/provenance_manifest.json) predates this repair and truncates "
                           "its per-file hashes to 16 hex characters.")}


# ------------------------------------------------------------------ EXPOSURE FIGURE
#: Reviewer BLOCKING 1. The previously documented figure (26/180, per-arm 9/7/5/4, median margins
#: 0.0065-0.0131) was WRONG and is withdrawn. These values are RECOMPUTED from
#: outputs/final_*.json by `recompute_exposure()` below and asserted equal to it by
#: scripts/test_rescore.py, so a wrong number cannot survive in the documentation again.
#:
#: This is a count of how many historical predictions landed on one of the nine differing labels.
#: It is NOT a bound on how many decisions could move -- see `EXPOSURE_FRAMING`.
DOCUMENTED_EXPOSURE = {
    "n_on_differing_labels": 37,
    "n_images_total": 180,
    "per_arm": {
        "utrap_sd15_church_l0.15k8": 12,
        "utrap_church_l0.15k8": 10,
        "frp_church_r3_l05": 7,
        "uce_sd15_church_lam1000": 5,
        "BASE_sd14_anchor": 2,
        "ATK_uce_sd15_church_lam1000": 1,
    },
    "median_margin_range_four_defence_arms": [0.0027, 0.0063],
}

#: The ONLY sanctioned reading of the exposure figure. Matches the audit register and FINDING65.
EXPOSURE_FRAMING = (
    "Small margins mean the outcomes are sensitive to label text; they do not indicate which way. "
    "This count is NOT a bound on how many decisions could move: the decision is a closed-set "
    "argmax over all twelve labels, so an image predicted as an UNCHANGED label (church, "
    "parachute, tench) can also flip when a competitor's text changes. Images predicted on a "
    "differing label are where a change is most directly visible, not where it is confined.")


def recompute_exposure(outputs_dir=None):
    """Recompute the exposure figure from the historical records. READ-ONLY.

    Returns the counts, per-arm breakdown and per-arm median winning margins for predictions that
    landed on one of the nine DIFFERING labels. The test asserts DOCUMENTED_EXPOSURE equals this,
    which is what the grep-based test could not do: greping for absent pre-repair strings catches a
    removed claim, never a WRONG one that is present. (Reviewer: the G14 lesson.)
    """
    import glob
    import statistics
    outputs_dir = outputs_dir or OUT
    diff = set(mod("clip_scorer").label_set_diff()["differing"])
    per_arm, margins_by_arm, total, n_images = {}, {}, 0, 0
    for f in sorted(glob.glob(os.path.join(outputs_dir, "final_*.json"))):
        d = json.load(open(f))
        imgs = d.get("per_image_predictions") or []
        ms = [r["margin"] for r in imgs if r["pred_label"] in diff]
        per_arm[d["label"]] = len(ms)
        margins_by_arm[d["label"]] = (round(statistics.median(ms), 4) if ms else None)
        total += len(ms)
        n_images += len(imgs)
    return {
        "n_on_differing_labels": total,
        "n_images_total": n_images,
        "n_records": len(per_arm),
        "per_arm_all": per_arm,
        "per_arm": {k: v for k, v in per_arm.items() if v},
        "median_margin_by_arm": margins_by_arm,
        "differing_labels": sorted(diff),
        "framing": EXPOSURE_FRAMING,
    }


# ------------------------------------------------------------------ small utilities
def mod(name):
    spec = iu.spec_from_file_location(f"_rs_{name}", os.path.join(SCRIPTS, f"{name}.py"))
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def file_sha(path, cap=None):
    """sha256 of a file; `cap` limits bytes read for multi-GB checkpoints. A capped hash is
    labelled as capped, because a prefix hash is weaker evidence of identity than a full one."""
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            if cap is not None and n + len(b) > cap:
                h.update(b[:cap - n])
                n = cap
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def _components(path):
    """Absolute, symlink-resolved path split into components, for COMPONENT-WISE containment
    tests. A string prefix test is not a containment test: `outputs/rescore_evil` has
    `outputs/rescore` as a string prefix but is NOT inside it. (Reviewer defect 3.)"""
    rp = os.path.realpath(path)
    return rp, [c for c in rp.split(os.sep) if c]


def _is_within(path_components, root_components):
    """True iff `path` is `root` or lies beneath it, compared component by component."""
    return path_components[:len(root_components)] == root_components


def _guard_path(path, kind="write"):
    """Refuse any write OR DIRECTORY CREATION that could escape the versioned run root.

    Two independent checks, both component-wise:
      1. the resolved path must not lie inside any protected historical tree;
      2. the resolved path must lie inside `RESCORE_ROOT`.
    Check 2 alone is sufficient (the protected trees are outside the run root), but 1 is kept so
    the error message names the specific historical tree a caller tried to touch.

    `..` traversal is defeated because `os.path.realpath` normalises it BEFORE the comparison, so
    `<root>/../../attacked/X` resolves to `<P>/attacked/X` and fails both checks.
    """
    rp, pc = _components(path)
    for pre in PROTECTED_PREFIXES:
        # PROTECTED_PREFIXES mixes directory roots with FILE-NAME prefixes (outputs/final_,
        # outputs/gen_), so test both ways: component containment for the directories, and a
        # basename prefix within outputs/ for the file families.
        if pre.endswith(("final_", "gen_")):
            d, base = os.path.split(pre)
            dp = _components(d)[1]
            if _is_within(pc, dp) and len(pc) == len(dp) + 1 and pc[-1].startswith(base):
                raise RuntimeError(
                    f"REFUSING to {kind} {rp}: it is a protected historical record "
                    f"({pre}*). Every output of this script belongs under "
                    f"{RESCORE_ROOT}/<run_id>/.")
            continue
        if _is_within(pc, _components(pre)[1]):
            raise RuntimeError(
                f"REFUSING to {kind} {rp}: it is inside the protected historical tree "
                f"({pre}/). Every output of this script belongs under "
                f"{RESCORE_ROOT}/<run_id>/.")
    if not _is_within(pc, _components(RESCORE_ROOT)[1]):
        raise RuntimeError(f"REFUSING to {kind} {rp}: outside the versioned run root "
                           f"{RESCORE_ROOT}. (Component-wise containment, so a sibling whose name "
                           f"merely starts with 'rescore' is refused too.)")
    return rp


def guarded_makedirs(path, exist_ok=False):
    """The ONLY directory-creation entry point. Every directory this script creates goes through
    `_guard_path` first.

    Reviewer BLOCKING 2: `make_run_dir` previously called `os.makedirs` directly, so the guard --
    which existed solely to keep this script out of the historical tree -- was bypassable through
    directory creation. `--run-tag '../../../attacked/PWNED'` created <P>/attacked/PWNED/. Verified
    and fixed.
    """
    rp = _guard_path(path, kind="create directory")
    os.makedirs(rp, exist_ok=exist_ok)
    return rp


def atomic_write_json(path, obj):
    """Write JSON atomically: temp file in the SAME directory, fsync, then `os.replace`.

    Same-directory temp guarantees `os.replace` is a rename within one filesystem, which is atomic;
    a reader therefore sees either the old file or the complete new one, never a truncated one.
    """
    rp = _guard_path(path)
    d = guarded_makedirs(os.path.dirname(rp), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, rp)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return rp


def atomic_write_bytes(path, write_fn):
    """As above for binary payloads (images, embeddings). `write_fn(fileobj)` does the writing."""
    rp = _guard_path(path)
    d = guarded_makedirs(os.path.dirname(rp), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, rp)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return rp


#: A run tag is a LABEL, never a path. Strict charset: no separators, no dots, so it cannot
#: contribute a traversal component. (Reviewer BLOCKING 2.)
RUN_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")

#: Salt width in hex characters. 6 hex = 24 bits, at which 2000 draws collide ~11% of the time by
#: the birthday bound -- the test that asserted uniqueness over 2000 draws was therefore FLAKY
#: (reviewer defect 6, observed 45/1 collisions in 2 of 7 runs). 16 hex = 64 bits.
RUN_ID_SALT_HEX = 16

#: Issued ids within this process, so within-process uniqueness is GUARANTEED BY CONSTRUCTION
#: rather than by the salt being lucky. Cross-process distinctness still rests on timestamp+pid.
_ISSUED_RUN_IDS = set()


def validate_run_tag(tag):
    """Reject any tag that is not a plain label. Raises ValueError with the reason."""
    if tag is None:
        return None
    if not isinstance(tag, str) or not RUN_TAG_RE.match(tag):
        raise ValueError(
            f"invalid --run-tag {tag!r}: a run tag is a LABEL, not a path. Allowed: 1-32 chars "
            f"from [A-Za-z0-9_-], starting alphanumeric. No '/', no '\\', no '.', no '..'. "
            f"A tag containing path separators or dots could escape the run root.")
    return tag


def new_run_id(tag=None):
    """Unique, sortable, collision-refusing run id. Tag is validated, never interpolated raw."""
    tag = validate_run_tag(tag)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    for _ in range(1000):
        salt = hashlib.sha256(
            f"{stamp}|{os.getpid()}|{time.time_ns()}|{len(_ISSUED_RUN_IDS)}|"
            f"{os.urandom(16).hex()}".encode()).hexdigest()[:RUN_ID_SALT_HEX]
        rid = f"rsc_{stamp}Z_{salt}" + (f"_{tag}" if tag else "")
        if rid not in _ISSUED_RUN_IDS:
            _ISSUED_RUN_IDS.add(rid)
            return rid
    raise RuntimeError("could not generate a fresh run id in 1000 attempts")


def make_run_dir(run_id, root=None):
    """Create the run directory, REFUSING if it already exists (no silent reuse, no overwrite).

    Every mkdir goes through `guarded_makedirs`, and the assembled path is guard-checked BEFORE
    anything is created -- so a traversal attempt creates nothing at all, not even a partial tree.
    """
    root = RESCORE_ROOT if root is None else root
    d = os.path.join(root, run_id)
    # Guard FIRST, create nothing on refusal.
    _guard_path(d, kind="create run directory")
    for sub in ("images", "embeddings", "arms"):
        _guard_path(os.path.join(d, sub), kind="create run directory")
    if os.path.exists(d):
        raise RuntimeError(f"REFUSING to reuse run directory {d}: it already exists. Run "
                           f"directories are write-once; use a new run id.")
    guarded_makedirs(d, exist_ok=False)
    for sub in ("images", "embeddings", "arms"):
        guarded_makedirs(os.path.join(d, sub), exist_ok=False)
    return d


# ------------------------------------------------------------------ the arm list
def build_arm_plan():
    """The arms to re-score, with ORIGINAL precommitment status kept separate from THIS run's.

    Sourced from `gen_eval_final.ARMS` / `POST_HOC_ARMS` so there is one arm list, plus the two
    E6D extra-seed labels that have historical records but were never in `ARMS` (and are therefore
    post-hoc by construction).
    """
    gf = mod("gen_eval_final")
    sd14 = ("CompVis/stable-diffusion-v1-4", None)
    sd15 = ("runwayml/stable-diffusion-v1-5", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14")
    extra = [
        ("ATK_uce_sd15_church_lam1000_sd1",
         "attacked/E6D_uce_sd15_church_lam1000_sd1_lr1e-5/unet.pt", "church", sd15),
        ("ATK_uce_sd15_church_lam1000_sd6",
         "attacked/E6D_uce_sd15_church_lam1000_sd6_lr1e-5/unet.pt", "church", sd15),
    ]
    _ = sd14  # kept for symmetry/readability of the table above
    plan = []
    for label, unet, target, (bm, brev) in list(gf.ARMS) + extra:
        in_arms = label in {a[0] for a in gf.ARMS}
        precommitted = label in gf.PRECOMMITTED_LABELS
        plan.append({
            "label": label,
            "unet": unet,
            "target": target,
            "base_model": bm,
            "base_revision_override": brev,
            # ---- ORIGINAL arm status: a property of the ORIGINAL E6 run, carried forward as-is.
            "original_arm_precommitted": bool(precommitted),
            "original_arm_status": (
                "PRE_COMMITTED_IN_gen_eval_final.ARMS" if precommitted else
                ("POST_HOC: listed in ARMS but named in POST_HOC_ARMS (added after FINAL results "
                 "were seen)" if in_arms else
                 "POST_HOC: never in ARMS at all (extra-seed label added after FINAL results were "
                 "seen)")),
            # ---- THIS evaluation's status: identical for EVERY arm and independent of the above.
            "this_evaluation_status": "CORRECTIVE_RE_EVALUATION_OF_INSPECTED_SPLIT",
            "this_evaluation_note": (
                "The FINAL split was inspected in FINDING65/FINDING66 BEFORE this run. No arm "
                "re-scored here -- pre-committed or not -- yields a held-out confirmation. "
                "Original precommitment status describes the ORIGINAL E6 measurement only and "
                "confers nothing on this one."),
        })
    return plan


# ------------------------------------------------------------------ provenance
def collect_provenance(args, run_id, run_dir, plan, cap):
    cs = mod("clip_scorer")
    sr = mod("split_registry")
    prov = {
        "rescore_version": RESCORE_VERSION,
        "scoring_version": cs.SCORING_VERSION,
        "run_id": run_id,
        "run_dir": run_dir,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    # ---- code identity. The T2I tree is NOT a git repository, so per-file hashes are the identity.
    rev = None
    try:
        import subprocess
        r = subprocess.run("git rev-parse HEAD", shell=True, cwd=P, capture_output=True,
                           text=True, timeout=30)
        rev = r.stdout.strip() or None
        dirty = bool(subprocess.run("git status --porcelain", shell=True, cwd=P,
                                    capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception:
        dirty = None
    prov["code"] = {
        "git_rev": rev,
        "git_dirty": dirty,
        "git_note": ("no git repository at the source tree root; code identity is the per-file "
                     "sha256 below" if rev is None else "git rev available"),
        "file_sha256": {fn: file_sha(os.path.join(SCRIPTS, fn))[0]
                        for fn in ("rescore_final_v2.py", "clip_scorer.py", "gen_eval.py",
                                   "gen_eval_final.py", "split_registry.py")
                        if os.path.exists(os.path.join(SCRIPTS, fn))},
    }

    # ---- environment
    deps = {}
    for m in ("torch", "diffusers", "transformers", "numpy", "accelerate", "safetensors", "PIL"):
        try:
            mm = __import__(m)
            deps[m] = getattr(mm, "__version__", "?")
        except Exception as e:
            deps[m] = f"<not importable: {type(e).__name__}>"
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "deps": deps,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "hf_home": os.environ.get("HF_HOME")}
    try:
        import torch
        env["torch_cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["gpu_name"] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
        env["n_gpu_visible"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        env["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception as e:
        env["torch_probe"] = f"<{type(e).__name__}>"
    prov["environment"] = env
    prov["environment_caveat"] = (
        "These are THIS run's versions. The original FINAL run recorded none, so no claim that the "
        "two environments match is possible. This is the main reason bit-identical regeneration "
        "cannot be asserted from matching seeds.")

    # ---- models
    pin = dict(l.split("=", 1) for l in open(os.path.join(P, "MODEL_PIN.txt")).read().split("\n")
               if "=" in l)
    prov["models"] = {
        "pin_file": pin,
        "per_arm_base": {a["label"]: {"base_model": a["base_model"],
                                      "revision": a["base_revision_override"] or pin["revision"],
                                      "variant": args.variant,
                                      "torch_dtype": "float16"} for a in plan},
        "variant_note": (f"variant={args.variant!r} is passed to from_pretrained for every arm, "
                         "matching gen_eval_final.py. An arm whose repo has no such variant will "
                         "FAIL loudly and be recorded as failed, not silently fall back."),
        "classifier": {"id": cs.CLIP_MODEL_ID, "revision": cs.CLIP_REVISION,
                       "revision_note": cs.CLIP_REVISION_NOTE},
        "numerics": cs.NUMERICS,
    }

    # ---- inference settings / scheduler / prompts / seeds / label text
    prov["inference"] = {
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "n_per_prompt": args.n_per,
        "height_width": "pipeline default (512x512 for SD-1.x)",
        "safety_checker": "disabled (safety_checker=None, requires_safety_checker=False)",
        "scheduler": "recorded PER ARM from the instantiated pipeline (see arm record "
                     "'scheduler'); gen_eval_final.py used the repo default, not an override",
    }
    prov["prompts"] = {
        "split_version": sr.SPLIT_VERSION,
        "split_role": "FINAL",
        "final_frozen_at": sr.FINAL_FROZEN_AT,
        "prompts_by_target": {a["target"]: sr.FINAL_PROMPTS[a["target"]] for a in plan},
    }
    prov["seeds"] = {
        "stream": "final_eval",
        "stream_offset": sr.RNG_STREAMS["final_eval"],
        "formula": "split_registry.seed_for('final_eval', prompt_index*100 + image_index)",
        "values": [sr.seed_for("final_eval", pi * 100 + k)
                   for pi in range(3) for k in range(args.n_per)],
        "device": "torch.Generator(device='cuda') -- same construction as gen_eval_final.py",
        "caveat": "Identical seeds reproduce identical noise ONLY given an identical environment "
                  "and device. Determinism is an ASSUMPTION TESTED by determinism_probe, not a "
                  "premise of this run.",
    }
    prov["label_sets"] = cs.label_set_diff()
    prov["checkpoint_hashes"] = {}
    prov["checkpoint_hash_scope"] = ("sha256 over the first "
                                     f"{'FULL FILE' if cap is None else str(cap >> 20) + ' MiB'} "
                                     "of each checkpoint")
    return prov


# ------------------------------------------------------------------ history comparison
def compare_with_history(label, target, legacy_records):
    """Compare THIS run's LEGACY-label-set reproduction against the historical record.

    Requirement 10: do NOT assume the regeneration reproduces history. Report the discrepancy.
    A mismatch is a finding about reproducibility, not a failure of this run; the manifest says
    so, and neither outcome is treated as expected.
    """
    hp = os.path.join(OUT, f"final_{label}.json")
    if not os.path.exists(hp):
        return {"status": "NO_HISTORICAL_RECORD", "historical_path": hp,
                "note": "nothing to compare against; this arm has no pre-repair FINAL record"}
    hist = json.load(open(hp))
    h_imgs = hist.get("per_image_predictions") or []
    n = min(len(h_imgs), len(legacy_records))
    pred_match = sum(1 for i in range(n)
                     if h_imgs[i]["pred_label"] == legacy_records[i]["pred_label"])
    d_margin = [abs(h_imgs[i]["margin"] - legacy_records[i]["margin"]) for i in range(n)]
    d_tsim = [abs(h_imgs[i]["target_sim"] - legacy_records[i]["target_sim"]) for i in range(n)]
    new_hits = sum(1 for r in legacy_records if r["is_target"])
    out = {
        "status": "COMPARED",
        "historical_path": hp,
        "historical_has_label_text_set_field": "label_text_set" in hist,
        "n_compared": n,
        "n_historical": len(h_imgs),
        "n_this_run": len(legacy_records),
        "historical_target_hits": hist.get("target_hits"),
        "historical_target_acc": hist.get("target_acc_FINAL"),
        "reproduced_legacy_target_hits": new_hits,
        "reproduced_legacy_target_acc": new_hits / max(len(legacy_records), 1),
        "acc_delta_reproduction_minus_history": (
            new_hits / max(len(legacy_records), 1) - (hist.get("target_acc_FINAL") or 0.0)),
        "per_image_pred_label_agreement": pred_match / n if n else None,
        "n_per_image_pred_disagreements": n - pred_match,
        "max_abs_margin_delta": max(d_margin) if d_margin else None,
        "max_abs_target_sim_delta": max(d_tsim) if d_tsim else None,
        "mean_abs_target_sim_delta": (sum(d_tsim) / len(d_tsim)) if d_tsim else None,
        "disagreeing_indices": [i for i in range(n)
                                if h_imgs[i]["pred_label"] != legacy_records[i]["pred_label"]],
    }
    exact = (n == len(h_imgs) == len(legacy_records) and pred_match == n
             and (max(d_tsim) if d_tsim else 0.0) == 0.0)
    out["verdict"] = ("EXACT_REPRODUCTION" if exact else "DISCREPANT")
    out["verdict_note"] = (
        "Legacy-label reproduction agrees with the historical record on every per-image "
        "prediction and similarity. Determinism held FOR THIS ARM in THIS environment; that is "
        "evidence, not a guarantee for other arms or environments."
        if exact else
        "Legacy-label reproduction DIFFERS from the historical record. The original images were "
        "never saved, so the cause cannot be separated here into (a) nondeterministic "
        "regeneration under a different library/driver/device, or (b) any other change. The "
        "corrected-vs-legacy comparison WITHIN this run remains valid regardless, because both "
        "label sets score the SAME regenerated images; what a discrepancy limits is the "
        "comparability of this run to the historical numbers.")
    return out


# ------------------------------------------------------------------ one arm
def run_arm(arm, scorer, sr, args, run_dir, prov, cap):
    """Generate + paired-score ONE arm. Returns the arm record (status always explicit)."""
    import numpy as np
    import torch
    from diffusers import StableDiffusionPipeline

    label, target = arm["label"], arm["target"]
    rec = dict(arm)
    rec.update({
        "study": "t2i_FINAL_rescore_v2",
        "experiment": "E6-RESCORE",
        "rescore_version": RESCORE_VERSION,
        "scoring_version": scorer_version(),
        "run_id": prov["run_id"],
        "evaluation_role": "CORRECTIVE_RE_EVALUATION",
        "split_role": "FINAL",
        "split_version": sr.SPLIT_VERSION,
        "split_status": "INSPECTED_BEFORE_THIS_RUN",
        "split_status_note": ("The FINAL split was inspected in FINDING65 and FINDING66 before "
                              "this run. These numbers are DEVELOPMENT measurements correcting a "
                              "scoring defect; they are NOT a fresh held-out confirmation and "
                              "must not be reported as one."),
        "regeneration_note": ("The original images were NOT saved. This is REGENERATION PLUS "
                              "CORRECTED SCORING, not a re-scoring of the original pixels. "
                              "Matching seeds do NOT establish byte-identical regeneration; see "
                              "determinism_probe."),
        "status": "incomplete",
    })
    t0 = time.time()

    # ---- checkpoint presence is checked BEFORE any GPU work, and recorded explicitly.
    full = None
    if arm["unet"]:
        full = arm["unet"] if os.path.isabs(arm["unet"]) else os.path.join(P, arm["unet"])
        if not os.path.exists(full):
            rec.update({
                "status": "missing_checkpoint",
                "checkpoint_path": full,
                "status_detail": f"checkpoint ABSENT: {arm['unet']}",
                "status_consequence": ("This arm was NOT evaluated. Its historical FINAL number "
                                       "cannot be corrected from the retained artefacts, because "
                                       "the checkpoint it was measured on no longer exists. Do "
                                       "not treat this arm as unchanged, and do not treat the run "
                                       "as covering it."),
                "minutes": (time.time() - t0) / 60,
            })
            atomic_write_json(os.path.join(run_dir, "arms", f"{label}.json"), rec)
            print(f"[rescore] {label}: MISSING_CHECKPOINT ({arm['unet']})", flush=True)
            return rec
        h, n = file_sha(full, cap)
        rec["checkpoint_path"] = full
        rec["checkpoint_sha256"] = h
        rec["checkpoint_hash_scope"] = "FULL" if (cap is None or n < cap) else f"FIRST_{n}_BYTES"
        rec["checkpoint_bytes"] = os.path.getsize(full)
        prov["checkpoint_hashes"][label] = {"path": full, "sha256": h,
                                            "scope": rec["checkpoint_hash_scope"]}
    else:
        rec["checkpoint_path"] = None
        rec["checkpoint_sha256"] = None
        rec["checkpoint_hash_scope"] = "N/A (unmodified base model)"

    pipe = None
    try:
        pin = dict(l.split("=", 1) for l in
                   open(os.path.join(P, "MODEL_PIN.txt")).read().split("\n") if "=" in l)
        revision = arm["base_revision_override"] or pin["revision"]
        pipe = StableDiffusionPipeline.from_pretrained(
            arm["base_model"], revision=revision, torch_dtype=torch.float16,
            variant=args.variant, safety_checker=None,
            requires_safety_checker=False).to(args.device)
        pipe.set_progress_bar_config(disable=True)
        rec["base_revision_used"] = revision
        rec["base_variant_used"] = args.variant
        rec["scheduler"] = {"class": type(pipe.scheduler).__name__,
                            "config": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                           else str(v))
                                       for k, v in dict(pipe.scheduler.config).items()}}
        if full:
            sd = torch.load(full, map_location="cpu")
            sd = sd.get("unet", sd)
            pipe.unet.load_state_dict({k: v.half() for k, v in sd.items()})
            rec["unet_loaded"] = True
        else:
            rec["unet_loaded"] = False

        finals = sr.FINAL_PROMPTS[target]
        rec["final_prompts"] = finals
        img_dir = guarded_makedirs(os.path.join(run_dir, "images", label), exist_ok=True)

        images, index, seeds = [], [], []
        for pi, pr in enumerate(finals):
            for k in range(args.n_per):
                seed = sr.seed_for("final_eval", pi * 100 + k)
                g = torch.Generator(device=args.device).manual_seed(seed)
                im = pipe([pr], num_inference_steps=args.steps,
                          guidance_scale=args.guidance, generator=g).images[0]
                name = f"p{pi}_k{k}.png"
                atomic_write_bytes(os.path.join(img_dir, name),
                                   lambda f, _im=im: _im.save(f, format="PNG"))
                images.append(im)
                index.append({"i": len(images) - 1, "prompt_index": pi, "prompt": pr,
                              "image_index": k, "seed": seed, "image_file": f"images/{label}/{name}"})
                seeds.append(seed)

        # ---- PAIRED SCORING: both label sets, ONE image-feature tensor, so the image side is
        # bit-identical between the two and any decision difference is the TEXT's alone.
        feats, sims = scorer.score_all_sets(images)
        per_set = {name: scorer.per_image_records(sims[name], target) for name in sims}

        emb = feats.float().cpu().numpy()
        atomic_write_bytes(
            os.path.join(run_dir, "embeddings", f"{label}.npz"),
            lambda f: np.savez_compressed(
                f, image_embeddings=emb,
                identity_keys=np.array(scorer.identity),
                sims_corrected=sims["corrected"].float().cpu().numpy(),
                sims_legacy=sims["legacy"].float().cpu().numpy()))

        per_image = pair_records(index, per_set)
        hits = {name: sum(1 for r in per_set[name] if r["is_target"]) for name in per_set}
        n_tot = len(per_image)
        rec["per_image"] = per_image
        rec["n_total"] = n_tot
        rec["embeddings_file"] = f"embeddings/{label}.npz"
        rec["embeddings_note"] = ("L2-normalised CLIP image embeddings plus the full 12-label "
                                  "similarity matrix for BOTH label sets. Any future label set "
                                  "can be scored from these without regenerating a single image.")
        rec["results_by_label_set"] = {
            name: {"label_text": scorer._sets[name], "target_hits": hits[name],
                   "n_total": n_tot, "target_acc": hits[name] / max(n_tot, 1)}
            for name in per_set}
        rec["paired_label_text_effect"] = {
            "corrected_minus_legacy_target_acc": (hits["corrected"] - hits["legacy"]) / max(n_tot, 1),
            "corrected_minus_legacy_target_hits": hits["corrected"] - hits["legacy"],
            "n_images_whose_prediction_changed": sum(1 for r in per_image
                                                     if r["pred_changed_by_label_text"]),
            "n_images_whose_target_decision_changed": sum(
                1 for r in per_image if r["is_target_changed_by_label_text"]),
            "measurement_note": ("Both label sets scored the SAME image-feature tensor, so this "
                                 "difference is attributable to the label TEXT alone, within this "
                                 "run. It is a MEASUREMENT of the effect for this arm; it is not "
                                 "evidence about any other arm, and it was not predicted in "
                                 "advance in either direction."),
        }
        rec["headline_number"] = {
            "metric": "target_acc under the CORRECTED label text (gen_eval.LABELS)",
            "value": hits["corrected"] / max(n_tot, 1),
            "denominator_note": (f"{len(finals)} FINAL prompts x {args.n_per} images = {n_tot}; "
                                 f"quantised to 1/{n_tot}. Closed-set 12-label CLIP argmax, no "
                                 f"reject option."),
            "comparability": ("Construction-comparable to the corpus H numbers in LABEL TEXT. "
                              "NOT a held-out result: the FINAL split was already inspected."),
        }
        rec["determinism_probe"] = compare_with_history(label, target, per_set["legacy"])
        rec["status"] = "complete"
        rec["minutes"] = (time.time() - t0) / 60
        atomic_write_json(os.path.join(run_dir, "arms", f"{label}.json"), rec)
        print(f"[rescore] {label}: corrected {hits['corrected']}/{n_tot} | "
              f"legacy {hits['legacy']}/{n_tot} | history "
              f"{rec['determinism_probe'].get('historical_target_hits')} -> "
              f"{rec['determinism_probe'].get('verdict')} ({rec['minutes']:.1f} min)", flush=True)
        return rec

    except Exception as e:
        import traceback
        rec.update({"status": "failed", "status_detail": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                    "status_consequence": ("This arm produced NO corrected number. The run is "
                                           "partial; do not read the remaining arms as a complete "
                                           "re-evaluation."),
                    "minutes": (time.time() - t0) / 60})
        atomic_write_json(os.path.join(run_dir, "arms", f"{label}.json"), rec)
        print(f"[rescore] {label}: FAILED {type(e).__name__}: {e}", flush=True)
        return rec
    finally:
        if pipe is not None:
            del pipe
            try:
                import torch as _t
                _t.cuda.empty_cache()
            except Exception:
                pass


def scorer_version():
    return mod("clip_scorer").SCORING_VERSION


# ------------------------------------------------------------------ pure, testable bookkeeping
def pair_records(index, per_set):
    """Interleave the per-label-set records for one arm and flag where the LABEL TEXT changed the
    decision. Pure function of its inputs, so it is unit-tested on CPU with synthetic scores.

    `per_set` must be {set_name: [record per image]} with every list the same length as `index`.
    """
    names = sorted(per_set)
    n = len(index)
    for nm in names:
        if len(per_set[nm]) != n:
            raise ValueError(f"paired scoring bookkeeping broken: label set {nm!r} has "
                             f"{len(per_set[nm])} records for {n} images. Every label set must "
                             f"score EVERY image, or the pair is not a pair.")
    if set(names) != {"corrected", "legacy"}:
        raise ValueError(f"expected exactly the label sets corrected+legacy, got {names}")
    out = []
    for j, meta in enumerate(index):
        row = dict(meta)
        for nm in names:
            row[nm] = per_set[nm][j]
        row["pred_changed_by_label_text"] = (
            per_set["corrected"][j]["pred_label"] != per_set["legacy"][j]["pred_label"])
        row["is_target_changed_by_label_text"] = (
            per_set["corrected"][j]["is_target"] != per_set["legacy"][j]["is_target"])
        out.append(row)
    return out


#: Reviewer defect 5. A run whose every arm regenerated DIFFERENTLY from history cannot be
#: compared to the historical numbers at all -- and was previously reported `run_status:
#: "complete"`, i.e. as a successful correction. `complete` now REQUIRES a defensible determinism
#: verdict, and distinct statuses exist for "produced, but not comparable to history".
RUN_STATUSES = ("complete", "complete_comparability_mixed", "complete_not_comparable_to_history",
                "partial", "incomplete", "failed")


def derive_history_comparability(arms, considered=None):
    """Aggregate the per-arm determinism verdicts into ONE explicit comparability verdict.

    -> (verdict, detail). Verdicts:
      all_exact   every completed arm reproduced its historical record exactly
      mixed       some completed arms reproduced exactly, others did not
      none_exact  no completed arm reproduced exactly -- this run's numbers cannot be placed
                  against the historical ones; the WITHIN-run corrected-vs-legacy comparison is
                  still valid, because both label sets scored the same regenerated images
      unknown     no completed arm had a historical record to compare against
    """
    labels = list(arms) if considered is None else considered
    done = [l for l in labels if arms.get(l, {}).get("status") == "complete"]
    exact, discrepant, nohist, absent = [], [], [], []
    for l in done:
        v = (arms[l].get("determinism_probe") or {}).get("verdict")
        st = (arms[l].get("determinism_probe") or {}).get("status")
        if v == "EXACT_REPRODUCTION":
            exact.append(l)
        elif v == "DISCREPANT":
            discrepant.append(l)
        elif st == "NO_HISTORICAL_RECORD":
            nohist.append(l)
        else:
            absent.append(l)          # probe missing entirely -- never silently treated as fine
    if absent:
        verdict = "unknown"
    elif not (exact or discrepant):
        verdict = "unknown"
    elif exact and not discrepant:
        verdict = "all_exact"
    elif discrepant and not exact:
        verdict = "none_exact"
    else:
        verdict = "mixed"
    return verdict, {
        "verdict": verdict,
        "n_completed_arms": len(done),
        "exact_reproduction": exact,
        "discrepant": discrepant,
        "no_historical_record": nohist,
        "probe_missing": absent,
        "meaning": {
            "all_exact": "this run's numbers may be placed against the historical ones",
            "mixed": "some arms are comparable to history and some are not; comparability must be "
                     "decided ARM BY ARM, never run-wide",
            "none_exact": "this run's numbers CANNOT be compared to the historical numbers. The "
                          "within-run corrected-vs-legacy label-text comparison remains valid "
                          "(both sets scored the same regenerated images); what is lost is any "
                          "claim about what the ORIGINAL records would have become.",
            "unknown": "no completed arm yielded a usable determinism verdict; comparability to "
                       "history is undetermined and must not be assumed",
        }[verdict],
    }


def derive_run_status(requested, arms):
    """Map per-arm statuses AND the determinism verdicts onto ONE explicit run status.

    Never reports `complete` when any requested arm is missing, failed or unattempted, AND never
    when the run cannot be compared to history. `arms` is {label: {"status", "determinism_probe"}}.
    -> (run_status, summary dict).
    """
    done = [l for l in requested if arms.get(l, {}).get("status") == "complete"]
    missing = [l for l in requested if arms.get(l, {}).get("status") == "missing_checkpoint"]
    failed = [l for l in requested if arms.get(l, {}).get("status") == "failed"]
    never = [l for l in requested if l not in arms
             or arms[l].get("status") not in ("complete", "missing_checkpoint", "failed")]
    comp, comp_detail = derive_history_comparability(arms, requested)
    if never:
        status = "incomplete"
    elif failed or missing:
        status = "partial"
    elif not done:
        status = "failed"
    elif comp == "all_exact":
        status = "complete"
    elif comp == "mixed":
        status = "complete_comparability_mixed"
    else:                                   # none_exact or unknown
        status = "complete_not_comparable_to_history"
    assert status in RUN_STATUSES, status
    return status, {"n_requested": len(requested), "n_complete": len(done), "complete": done,
                    "missing_checkpoint": missing, "failed": failed, "not_attempted": never,
                    "history_comparability": comp_detail,
                    "status_semantics": (
                        "`complete` means EVERY requested arm completed AND every completed arm "
                        "reproduced its historical record exactly. A run that produced numbers but "
                        "cannot be compared to history is "
                        "`complete_not_comparable_to_history`, never `complete`.")}


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated arm labels; default all")
    ap.add_argument("--n-per", type=int, default=5, dest="n_per",
                    help="images per FINAL prompt; 3x5=15, matching the historical records")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--variant", default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-tag", default=None, dest="run_tag")
    ap.add_argument("--hash-cap-mb", type=int, default=64, dest="hash_cap_mb",
                    help="bytes per checkpoint to hash; 0 = full file")
    ap.add_argument("--plan-only", action="store_true", dest="plan_only",
                    help="print the plan and provenance; create NOTHING, touch no GPU")
    a = ap.parse_args(argv)
    cap = None if a.hash_cap_mb == 0 else a.hash_cap_mb << 20
    try:
        validate_run_tag(a.run_tag)          # BEFORE anything is created (reviewer BLOCKING 2)
    except ValueError as e:
        raise SystemExit(str(e))

    sr = mod("split_registry")
    cs = mod("clip_scorer")
    plan = build_arm_plan()
    if a.only:
        want = {x.strip() for x in a.only.split(",")}
        unknown = want - {p["label"] for p in plan}
        if unknown:
            raise SystemExit(f"unknown arm label(s): {sorted(unknown)}")
        plan = [p for p in plan if p["label"] in want]

    contract = cs.verify_numerics_contract()
    if not contract["ok"]:
        raise SystemExit(f"numerics contract with gen_eval.py BROKEN: {contract['issues']}")
    diff = cs.label_set_diff()

    if a.plan_only:
        prov = collect_provenance(a, "PLAN_ONLY", "<none>", plan, cap)
        print(json.dumps({"plan_only": True, "n_arms": len(plan),
                          "arms": [{k: p[k] for k in ("label", "unet", "base_model",
                                                      "original_arm_precommitted",
                                                      "this_evaluation_status")} for p in plan],
                          "checkpoints_present": {p["label"]: (p["unet"] is None or os.path.exists(
                              p["unet"] if os.path.isabs(p["unet"]) else os.path.join(P, p["unet"])))
                              for p in plan},
                          "label_set_diff": {k: diff[k] for k in ("n_labels", "n_differ",
                                                                  "n_identical", "differing",
                                                                  "identical")},
                          "numerics_contract": contract,
                          "exposure_recomputed": recompute_exposure(),
                          "exposure_documented": DOCUMENTED_EXPOSURE,
                          "unmodified_baseline": verify_unmodified_baseline(),
                          "provenance_keys": sorted(prov)}, indent=1))
        return 0

    run_id = new_run_id(a.run_tag)
    run_dir = make_run_dir(run_id)
    prov = collect_provenance(a, run_id, run_dir, plan, cap)

    manifest = {
        "study": "t2i_FINAL_rescore_v2",
        "rescore_version": RESCORE_VERSION,
        "run_id": run_id,
        "run_dir": run_dir,
        "run_status": "incomplete",
        "run_status_note": ("Written BEFORE any arm ran. If the process dies, this file still says "
                            "'incomplete', so a crashed run can never be mistaken for a finished "
                            "one."),
        "evaluation_role": "CORRECTIVE_RE_EVALUATION",
        "evaluation_role_note": (
            "CORRECTIVE RE-EVALUATION OF AN ALREADY-INSPECTED SPLIT. The FINAL split was inspected "
            "in FINDING65 and FINDING66 before this run; these numbers are DEVELOPMENT "
            "measurements that repair a scoring defect (D-013). They are NOT a fresh held-out "
            "confirmation and must not be reported as one."),
        "method": "REGENERATION_PLUS_CORRECTED_SCORING",
        "method_note": (
            "The original images were never saved, so the original pixels cannot be re-scored. "
            "Images are REGENERATED from the recorded prompts and seeds and then scored under both "
            "label-text sets. Matching seeds are a reason to EXPECT similar images, not proof: "
            "determinism is an assumption, tested per arm by determinism_probe against the "
            "historical records, and NOT presumed to hold."),
        "label_mismatch": diff,
        "exposure_in_historical_records": {**recompute_exposure(),
                                           "documented": DOCUMENTED_EXPOSURE},
        "unmodified_baseline": verify_unmodified_baseline(),
        "run_status_vocabulary": list(RUN_STATUSES),
        "numerics_contract": contract,
        "paired_scoring": {
            "design": ("Both label-text sets are scored on ONE image-feature tensor per image, so "
                       "the image side is bit-identical across the pair and any difference in the "
                       "decision is attributable to the label text alone."),
            "sets": {"corrected": "gen_eval.LABELS (the text every H number uses)",
                     "legacy": "gen_eval.LAB_KEY (the text the pre-repair FINAL records used)"},
            "identity_mapping": "gen_eval.LAB_KEY, identical under both sets, so prediction "
                                "indices mean the same concept either way",
        },
        "provenance": prov,
        "arms_requested": [p["label"] for p in plan],
        "arms": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    mpath = os.path.join(run_dir, "run.json")
    atomic_write_json(mpath, manifest)
    print(f"[rescore] run_id={run_id}\n[rescore] run_dir={run_dir}\n"
          f"[rescore] label sets differ in {diff['n_differ']}/{diff['n_labels']}: "
          f"{', '.join(diff['differing'])}\n[rescore] identical: "
          f"{', '.join(diff['identical'])}", flush=True)

    scorer = None
    t0 = time.time()
    try:
        scorer = cs.ClipScorer(device=a.device)
        for arm in plan:
            r = run_arm(arm, scorer, sr, a, run_dir, prov, cap)
            manifest["arms"][arm["label"]] = {
                k: r.get(k) for k in ("status", "status_detail", "status_consequence",
                                      "original_arm_precommitted", "original_arm_status",
                                      "this_evaluation_status", "results_by_label_set",
                                      "paired_label_text_effect", "headline_number",
                                      "determinism_probe", "checkpoint_sha256", "minutes")}
            manifest["provenance"] = prov
            atomic_write_json(mpath, manifest)   # checkpoint after EVERY arm
    finally:
        status, summary = derive_run_status(manifest["arms_requested"], manifest["arms"])
        manifest["run_status"], manifest["summary"] = status, summary
        done, missing = summary["complete"], summary["missing_checkpoint"]
        failed, never = summary["failed"], summary["not_attempted"]
        comp = summary["history_comparability"]
        manifest["run_status_note"] = (
            f"run_status={manifest['run_status']}. 'complete' requires EVERY requested arm to be "
            f"complete AND every completed arm to have reproduced its historical record exactly. "
            f"Arms with an absent checkpoint or an error are named above and were NOT silently "
            f"skipped; a 'partial' run does not cover them and must not be reported as a full "
            f"re-evaluation. History comparability: {comp['verdict']} -- {comp['meaning']}")
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest["minutes"] = (time.time() - t0) / 60
        atomic_write_json(mpath, manifest)
        print(f"\n[rescore] run_status={manifest['run_status']} "
              f"complete={len(done)}/{len(manifest['arms_requested'])} "
              f"missing={missing or '-'} failed={failed or '-'} not_attempted={never or '-'}")
        print(f"[rescore] history_comparability={comp['verdict']} "
              f"exact={comp['exact_reproduction'] or '-'} "
              f"discrepant={comp['discrepant'] or '-'}")
        print(f"[rescore] {comp['meaning']}")
        print(f"[rescore] manifest: {mpath}")
    return 0 if manifest["run_status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
