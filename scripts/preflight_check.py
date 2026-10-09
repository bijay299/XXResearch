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
def check_assets(gen_dir: Path, cls_dir: Path) -> dict:
    present = {}

    gen_ok = (gen_dir / "model_index.json").exists()
    gen_parts = [p for p in ("unet", "vae", "text_encoder", "tokenizer", "scheduler")
                 if (gen_dir / p).is_dir()]
    present["generator"] = gen_ok and len(gen_parts) == 5
    record(
        "UnlearnCanvas generator",
        "PASS" if present["generator"] else "FAIL",
        {"dir": str(gen_dir), "model_index.json": gen_ok, "subdirs_found": gen_parts},
    )

    style_p = cls_dir / "style_classifier.pth"
    object_p = cls_dir / "object_classifier.pth"
    present["classifiers"] = style_p.exists() and object_p.exists()
    record(
        "UnlearnCanvas classifiers",
        "PASS" if present["classifiers"] else "FAIL",
        {"dir": str(cls_dir),
         "style_classifier.pth": style_p.exists(),
         "object_classifier.pth": object_p.exists(),
         "note": "" if present["classifiers"] else "UA/IRA/CRA cannot be computed; see docs/ASSETS.md"},
    )
    return present


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
            "classifier head dim vs label lists",
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
    record("classifier head dim vs label lists", "PASS" if ok else "FAIL", detail)


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
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 78)
    print("CUIG ConAbl / UnlearnCanvas pilot - PREFLIGHT")
    print("=" * 78)

    check_environment()
    check_upstream(Path(args.cuig_root))
    present = check_assets(Path(args.generator_dir), Path(args.classifier_dir))
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

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"report written to {args.json}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
