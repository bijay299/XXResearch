#!/usr/bin/env python3
"""Pre-launch validation for the CUIG ConAbl / UnlearnCanvas pilot.

Checks, in order, and reports each independently so a partial environment still
produces a useful verdict:

  1. environment      - python / torch / diffusers / ... versions, CUDA visibility
  2. upstream         - CUIG checkout present at the pinned commit
  3. assets           - UnlearnCanvas generator + classifiers present and loadable
  4. label agreement  - classifier head dimensions vs. the benchmark label lists
                        in Evaluation/UnlearnCanvas/constants.py, and the prompt
                        template used by sample.py
  5. generation       - a single image actually generates (only with --gpu)
  6. classifier eval  - classifiers score that image (only with --gpu)

Exit code 0 means every *attempted* check passed. Checks that cannot run because
an asset is missing are reported as SKIP, never as PASS.

Usage:
    python scripts/preflight_check.py                 # CPU-only checks
    python scripts/preflight_check.py --gpu 2         # include GPU checks
    python scripts/preflight_check.py --json out.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "9932ac3271a122f6d38d19e0b8c8908fe5237ff7"

results: list[dict] = []


def record(name: str, status: str, detail) -> None:
    results.append({"check": name, "status": status, "detail": detail})
    mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP", "WARN": "WARN"}[status]
    print(f"[{mark}] {name}")
    if isinstance(detail, dict):
        for k, v in detail.items():
            print(f"         {k}: {v}")
    elif detail:
        print(f"         {detail}")


# --------------------------------------------------------------------------- 1
def check_environment() -> None:
    want = {
        "torch": "2.4.0",
        "torchvision": "0.19.0",
        "diffusers": "0.30.2",
        "transformers": "4.44.2",
        "accelerate": "0.34.2",
        "timm": "1.0.15",
        "numpy": "1.26.4",
        "safetensors": "0.4.5",
        "openpyxl": "3.1.5",
    }
    got, bad = {}, []
    for mod, expect in want.items():
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            got[mod] = v
            if not v.startswith(expect):
                bad.append(f"{mod}: want {expect}, got {v}")
        except Exception as exc:  # pragma: no cover
            got[mod] = f"IMPORT ERROR: {exc}"
            bad.append(f"{mod}: {exc}")

    import torch

    got["python"] = sys.version.split()[0]
    got["torch.version.cuda"] = torch.version.cuda
    got["cuda_available"] = torch.cuda.is_available()
    got["device_count"] = torch.cuda.device_count()
    got["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    record("environment versions", "FAIL" if bad else "PASS", {**got, "mismatches": bad or "none"})


# --------------------------------------------------------------------------- 2
def check_upstream(cuig_root: Path) -> None:
    if not (cuig_root / "UnlearningMethods" / "ConAbl" / "train_conabl.py").exists():
        record("upstream CUIG checkout", "FAIL", f"not found under {cuig_root}")
        return
    try:
        head = subprocess.check_output(
            ["git", "-C", str(cuig_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(cuig_root), "status", "--porcelain"], text=True
        ).strip()
    except Exception as exc:
        record("upstream CUIG checkout", "WARN", f"git unavailable: {exc}")
        return
    status = "PASS" if head == PINNED_COMMIT and not dirty else "WARN"
    record(
        "upstream CUIG checkout",
        status,
        {"head": head, "pinned": PINNED_COMMIT, "matches_pin": head == PINNED_COMMIT,
         "working_tree_clean": not dirty},
    )


# --------------------------------------------------------------------------- 3
def load_receipt(receipt_path: Path) -> dict | None:
    """Read the validation receipt produced by scripts/validate_assets.py."""
    if not receipt_path.exists():
        return None
    try:
        return json.loads(receipt_path.read_text())
    except Exception:
        return None


def asset_state(present: bool, verified: bool) -> str:
    """Tri-state asset status. Presence alone is never 'VERIFIED'."""
    if not present:
        return "MISSING"
    return "VERIFIED" if verified else "PRESENT_UNVERIFIED"


def check_assets(gen_dir: Path, cls_dir: Path, receipt: dict | None) -> dict:
    """Presence AND verification. An unverified asset fails the baseline gate.

    Being on disk is not sufficient: a checkpoint of the right shape with a
    permuted class order, or a generator that was never fine-tuned on
    UnlearnCanvas, is indistinguishable from the real thing by inspection alone.
    Only scripts/validate_assets.py may mark an asset verified.
    """
    v = (receipt or {}).get("verdict", {})
    cls_verified = bool(v.get("classifiers_verified"))
    gen_verified = bool(v.get("assets_verified"))

    gen_present = ((gen_dir / "model_index.json").exists()
                   and all((gen_dir / p).is_dir()
                           for p in ("unet", "vae", "text_encoder", "tokenizer", "scheduler")))
    style_p, object_p = cls_dir / "style_classifier.pth", cls_dir / "object_classifier.pth"
    cls_present = style_p.exists() and object_p.exists()

    gen_state = asset_state(gen_present, gen_verified)
    cls_state = asset_state(cls_present, cls_verified)

    record(
        "UnlearnCanvas generator",
        "PASS" if gen_state == "VERIFIED" else "FAIL",
        {"dir": str(gen_dir), "state": gen_state, "present": gen_present,
         "verified_by_receipt": gen_verified,
         "note": "" if gen_state == "VERIFIED"
                 else "baseline blocked; see docs/ASSETS.md"},
    )
    record(
        "UnlearnCanvas classifiers",
        "PASS" if cls_state == "VERIFIED" else "FAIL",
        {"dir": str(cls_dir), "state": cls_state,
         "style_classifier.pth": style_p.exists(),
         "object_classifier.pth": object_p.exists(),
         "verified_by_receipt": cls_verified,
         "note": "" if cls_state == "VERIFIED"
                 else "UA/IRA/CRA cannot be computed; run scripts/validate_assets.py"},
    )

    # Unresolved facts are surfaced explicitly rather than silently defaulting.
    unknowns = (receipt or {}).get("unknowns", {})
    record(
        "unresolved provenance / label mapping",
        "WARN" if unknowns else "SKIP",
        unknowns or "no validation receipt yet; every provenance and label-mapping "
                    "fact is unknown",
    )

    return {"generator": gen_present, "classifiers": cls_present,
            "generator_state": gen_state, "classifiers_state": cls_state,
            "generator_verified": gen_verified, "classifiers_verified": cls_verified}


def check_receipt(receipt: dict | None, receipt_path: Path) -> None:
    if receipt is None:
        record("asset validation receipt", "FAIL",
               {"path": str(receipt_path),
                "status": "absent",
                "how_to_resolve": "python scripts/validate_assets.py --gpu <idle>"})
        return
    v = receipt.get("verdict", {})
    record("asset validation receipt",
           "PASS" if v.get("assets_verified") else "FAIL",
           {"path": str(receipt_path), "generated_utc": receipt.get("generated_utc"),
            "overall": v.get("overall"),
            "label_mapping_confirmed": v.get("label_mapping_confirmed"),
            "heldout_accuracy_pass": v.get("heldout_accuracy_pass"),
            "untouched_generator_pass": v.get("untouched_generator_pass"),
            "UA_IRA_CRA": v.get("UA_IRA_CRA")})


# --------------------------------------------------------------------------- 4
def check_label_agreement(cuig_root: Path, cls_dir: Path, present: dict) -> None:
    sys.path.insert(0, str(cuig_root / "Evaluation" / "UnlearnCanvas"))
    import constants  # type: ignore

    styles, objects = constants.STYLES_AVAILABLE, constants.OBJECTS_AVAILABLE
    static = {
        "num_styles": len(styles),
        "num_objects": len(objects),
        "styles_sorted_alphabetically": styles == sorted(styles),
        "objects_sorted_alphabetically": objects == sorted(objects),
        "style_label_of_index_0": styles[0],
        "object_label_of_index_0": objects[0],
        "sample_prompt_template": "A {object} image in {style} style",
        "eval_label_lookup": "STYLES_AVAILABLE.index(style) / OBJECTS_AVAILABLE.index(object)",
    }
    # Both lists being alphabetically sorted is consistent with torchvision
    # ImageFolder class ordering, which is how the UnlearnCanvas classifiers were
    # trained; that is necessary but NOT sufficient evidence of agreement.
    record("label lists (static)", "PASS", static)

    if not present["classifiers"]:
        record(
            "classifier head shape vs label lists",
            "SKIP",
            "classifiers unavailable - cannot confirm the checkpoint head matches "
            f"{len(styles)} style / {len(objects)} object classes",
        )
        return

    import torch

    detail = {}
    ok = True
    for task, n in (("style", len(styles)), ("object", len(objects))):
        ckpt = torch.load(cls_dir / f"{task}_classifier.pth", map_location="cpu",
                          weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        hw = [v for k, v in sd.items() if k.endswith("head.weight")]
        got_n = int(hw[0].shape[0]) if hw else None
        detail[f"{task}_head_out_features"] = got_n
        detail[f"{task}_expected"] = n
        if got_n != n:
            ok = False
    # A shape match is a necessary precondition, nothing more. It is reported as
    # WARN rather than PASS so it can never be read as scientific validation: a
    # checkpoint with a permuted class order has an identical head shape.
    detail["IMPORTANT"] = ("shape compatibility is NOT validation; label order must be "
                           "confirmed behaviourally by scripts/validate_assets.py")
    record("classifier head shape vs label lists (shape only)",
           "WARN" if ok else "FAIL", detail)


# --------------------------------------------------------------------------- 5/6
def check_generation_and_eval(gen_dir: Path, cls_dir: Path, cuig_root: Path,
                              present: dict, out_dir: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        record("image generation", "SKIP", "CUDA not available in this process")
        return
    if not present["generator"]:
        record("image generation", "SKIP", "generator asset missing")
        return

    from diffusers import StableDiffusionPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained(gen_dir, torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = lambda images, **kw: (images, [False])
    pipe.set_progress_bar_config(disable=True)
    img = pipe(prompt="A Architectures image in Abstractionism style", width=512, height=512,
               num_inference_steps=20, guidance_scale=9.0).images[0]
    p = out_dir / "preflight_sample.jpg"
    img.save(p)
    record("image generation", "PASS",
           {"saved": str(p), "size": img.size,
            "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20)})
    del pipe
    torch.cuda.empty_cache()

    if not present["classifiers"]:
        record("classifier evaluation", "SKIP",
               "classifiers unavailable - UA/IRA/CRA cannot be computed")
        return

    sys.path.insert(0, str(cuig_root / "Evaluation" / "UnlearnCanvas"))
    import evaluate as ev  # type: ignore

    dev = torch.device("cuda")
    tf = ev.build_image_transform()
    from PIL import Image

    x = tf(Image.open(p)).unsqueeze(0).to(dev)
    detail = {}
    for task in ("style", "object"):
        model = ev.load_task_classifier(task, str(cls_dir), dev)
        with torch.no_grad():
            logits = model(x)
        import constants  # type: ignore
        names = constants.STYLES_AVAILABLE if task == "style" else constants.OBJECTS_AVAILABLE
        detail[f"{task}_logits_shape"] = list(logits.shape)
        detail[f"{task}_top1"] = names[int(logits.argmax())]
        del model
        torch.cuda.empty_cache()
    record("classifier evaluation", "PASS", detail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuig_root", default=os.environ.get(
        "CUIG_REPO_ROOT", "/data/bijaypandey/cuig_pilot/CUIG"))
    ap.add_argument("--generator_dir", default=os.environ.get(
        "CUIG_UNLEARNCANVAS_GENERATOR_DIR",
        "/data/bijaypandey/cuig_pilot/Checkpoints/Generators/UnlearnCanvas"))
    ap.add_argument("--classifier_dir", default=os.environ.get(
        "CUIG_UNLEARNCANVAS_CLASSIFIER_DIR",
        "/data/bijaypandey/cuig_pilot/Checkpoints/Classifiers/UnlearnCanvas"))
    ap.add_argument("--gpu", default=None,
                    help="GPU index to pin for generation/eval checks; omit to skip them")
    ap.add_argument("--out_dir", default="/data/bijaypandey/cuig_pilot/preflight")
    ap.add_argument("--json", default=None)
    ap.add_argument("--receipt", default=str(
        Path(__file__).resolve().parents[1] / "results" / "assets" / "validation_receipt.json"),
        help="asset validation receipt from scripts/validate_assets.py")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 78)
    print("CUIG ConAbl / UnlearnCanvas pilot - PREFLIGHT")
    print("=" * 78)

    receipt_path = Path(args.receipt)
    receipt = load_receipt(receipt_path)

    check_environment()
    check_upstream(Path(args.cuig_root))
    check_receipt(receipt, receipt_path)
    present = check_assets(Path(args.generator_dir), Path(args.classifier_dir), receipt)
    check_label_agreement(Path(args.cuig_root), Path(args.classifier_dir), present)

    if args.gpu is not None:
        check_generation_and_eval(Path(args.generator_dir), Path(args.classifier_dir),
                                  Path(args.cuig_root), present, Path(args.out_dir))
    else:
        record("image generation", "SKIP", "--gpu not provided")
        record("classifier evaluation", "SKIP", "--gpu not provided")

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    print("-" * 78)
    print(f"PASS={sum(1 for r in results if r['status']=='PASS')} "
          f"WARN={sum(1 for r in results if r['status']=='WARN')} "
          f"FAIL={n_fail} SKIP={n_skip}")

    # The baseline gate: assets must be present AND verified by a receipt.
    baseline_ready = bool(present.get("generator_verified")
                          and present.get("classifiers_verified"))
    print()
    print(f"generator  : {present.get('generator_state')}")
    print(f"classifiers: {present.get('classifiers_state')}")
    print(f"BASELINE GATE: {'OPEN' if baseline_ready else 'CLOSED'}  "
          f"(UA/IRA/CRA {'available' if baseline_ready else 'UNAVAILABLE'})")
    if not baseline_ready:
        print("  -> obtain the assets (docs/ASSETS.md), then:")
        print("       python scripts/record_asset_hashes.py")
        print("       python scripts/build_heldout_eval_set.py")
        print("       python scripts/validate_assets.py --gpu <idle> --update-manifest")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"checks": results, "baseline_gate_open": baseline_ready}, fh, indent=2)
        print(f"report written to {args.json}")

    return 0 if (baseline_ready and not n_fail) else 1


if __name__ == "__main__":
    sys.exit(main())
