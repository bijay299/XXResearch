#!/usr/bin/env python3
"""First real validation of the UnlearnCanvas benchmark assets.

This is the gate that must pass before any UA / IRA / CRA is produced. It is the
only script permitted to mark an asset `verified` in the asset manifest.

What it checks
--------------
A. Presence           - generator directory and both classifier files exist.
B. Shape compatibility - classifier heads are Linear(1024, 51) / Linear(1024, 20).
                        Recorded as `shape_compatible`. This is NOT validation:
                        a checkpoint with a permuted class order has exactly the
                        same shape as a correct one.
C. Label mapping      - run each classifier over labelled held-out UnlearnCanvas
                        images (built by scripts/build_heldout_eval_set.py) and
                        test the ordering `evaluate.py` assumes, namely
                        `STYLES_AVAILABLE.index(name)` == classifier output index.
                        Also computes the modal predicted index per true class, so
                        a *consistent permutation* is distinguished from a model
                        that is simply bad.
D. Accuracy           - top-1 accuracy on those held-out images, against an
                        explicit threshold.
E. Generator baseline - sample from the UNTOUCHED generator with the released
                        sampling configuration and score those samples with the
                        classifiers. This is the pre-unlearning reference point;
                        it also catches a wrong generator (e.g. a stock SD
                        checkpoint that was never fine-tuned on UnlearnCanvas),
                        which would otherwise look fine right up until the
                        baseline numbers were meaningless.

Anything not established is written to the receipt as `"unknown"`, never guessed.
Provenance is NEVER set by this script: behavioural agreement does not prove a
file is the authors' checkpoint.

Thresholds are a local policy choice, recorded in the receipt, not a published
value from the paper.

Usage:
    python scripts/validate_assets.py --gpu 2
    python scripts/validate_assets.py --gpu 2 --skip-generator   # classifiers only
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UNKNOWN = "unknown"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# --------------------------------------------------------------------------- A
def check_presence(gen_dir: Path, cls_dir: Path) -> dict:
    gen_members = ["model_index.json", "scheduler", "text_encoder", "tokenizer", "unet", "vae"]
    found = [m for m in gen_members if (gen_dir / m).exists()]
    style_p = cls_dir / "style_classifier.pth"
    object_p = cls_dir / "object_classifier.pth"
    return {
        "generator": {"path": str(gen_dir), "present": len(found) == len(gen_members),
                      "members_found": found, "members_expected": gen_members},
        "style_classifier": {"path": str(style_p), "present": style_p.exists()},
        "object_classifier": {"path": str(object_p), "present": object_p.exists()},
    }


# --------------------------------------------------------------------------- B
def check_shape(cls_dir: Path, task: str, expected_n: int) -> dict:
    import torch

    path = cls_dir / f"{task}_classifier.pth"
    if not path.exists():
        return {"status": "missing", "shape_compatible": False}
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    head_w = [v for k, v in sd.items() if k.endswith("head.weight")]
    got = int(head_w[0].shape[0]) if head_w else None
    return {
        "status": "checked",
        "head_out_features": got,
        "expected_out_features": expected_n,
        "shape_compatible": got == expected_n,
        "state_dict_key": "model_state_dict" if (
            isinstance(blob, dict) and "model_state_dict" in blob) else "raw",
        "IMPORTANT": ("shape compatibility is NOT scientific validation; a permuted class "
                      "ordering is shape-identical to a correct one"),
    }


# --------------------------------------------------------------------------- C/D
def evaluate_classifier_on_heldout(task: str, cls_dir: Path, cuig_root: Path,
                                   heldout: Path, device, threshold: float) -> dict:
    """Score a classifier on labelled held-out images and interrogate its label order."""
    import torch
    from PIL import Image

    sys.path.insert(0, str(cuig_root / "Evaluation" / "UnlearnCanvas"))
    import constants as C  # type: ignore
    import evaluate as ev  # type: ignore

    names = list(C.STYLES_AVAILABLE if task == "style" else C.OBJECTS_AVAILABLE)
    label_key = "style" if task == "style" else "object"

    manifest = json.loads((heldout / "labels.json").read_text())
    records = manifest["records"]

    # The dataset ships 60 styles; the evaluator's list has 51 entries (50 styles
    # + Seed_Images). Images whose label is outside the list cannot be predicted
    # and are excluded from the style test rather than counted as errors.
    usable = [r for r in records if r[label_key] in names]
    skipped = [r[label_key] for r in records if r[label_key] not in names]

    if not usable:
        return {"status": "no_usable_heldout_images", "label_mapping": UNKNOWN,
                "accuracy": None}

    model = ev.load_task_classifier(task, str(cls_dir), device)
    tf = ev.build_image_transform()

    correct = 0
    per_true_pred: dict[str, Counter] = defaultdict(Counter)
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()

    with torch.no_grad():
        for rec in usable:
            img = Image.open(heldout / "images" / rec["file"])
            x = tf(img).unsqueeze(0).to(device)
            pred = int(model(x).argmax())
            true_name = rec[label_key]
            true_idx = names.index(true_name)
            per_true_pred[true_name][pred] += 1
            per_class_total[true_name] += 1
            if pred == true_idx:
                correct += 1
                per_class_correct[true_name] += 1

    del model
    torch.cuda.empty_cache()

    n = len(usable)
    acc = correct / n
    chance = 1.0 / len(names)

    # Is there a consistent (possibly permuted) mapping? For each true class take
    # the modal predicted index; if those are distinct and stable, the checkpoint
    # is informative but possibly ordered differently from the benchmark list.
    modal = {name: cnt.most_common(1)[0][0] for name, cnt in per_true_pred.items()}
    modal_agrees = sum(1 for name, idx in modal.items() if idx == names.index(name))
    modal_distinct = len(set(modal.values())) == len(modal)
    modal_consistency = sum(
        per_true_pred[name][idx] for name, idx in modal.items()) / n

    if acc >= threshold:
        verdict = "confirmed"
        detail = (f"top-1 {acc:.1%} under the assumed ordering, at or above the "
                  f"{threshold:.0%} threshold")
    elif modal_distinct and modal_consistency >= threshold and modal_agrees < len(modal):
        verdict = "refuted_permuted"
        detail = ("predictions are consistent per class but do not land on the assumed "
                  f"indices ({modal_agrees}/{len(modal)} agree). The checkpoint appears "
                  "informative with a DIFFERENT class ordering. Metrics computed with "
                  "constants.py ordering would be wrong.")
    else:
        verdict = "inconclusive"
        detail = (f"top-1 {acc:.1%} (chance {chance:.1%}); neither the assumed ordering nor a "
                  "consistent permutation is supported by this evidence")

    return {
        "status": "checked",
        "images_scored": n,
        "images_skipped_label_out_of_list": sorted(set(skipped)),
        "distinct_true_classes": len(per_class_total),
        "accuracy_assumed_ordering": round(acc, 4),
        "chance_level": round(chance, 4),
        "threshold": threshold,
        "label_mapping": verdict,
        "label_mapping_detail": detail,
        "modal_prediction_agrees_with_assumed_index": f"{modal_agrees}/{len(modal)}",
        "modal_predictions_distinct": modal_distinct,
        "modal_consistency": round(modal_consistency, 4),
        "per_class_accuracy": {k: round(per_class_correct[k] / per_class_total[k], 4)
                               for k in sorted(per_class_total)},
        "accuracy_pass": acc >= threshold,
    }


# --------------------------------------------------------------------------- E
def evaluate_untouched_generator(gen_dir: Path, cls_dir: Path, cuig_root: Path,
                                 device, out_dir: Path, styles: list[str],
                                 objects: list[str], seed: int, threshold: float) -> dict:
    """Sample from the untouched generator and score it. Pre-unlearning reference."""
    import torch
    from PIL import Image

    sys.path.insert(0, str(cuig_root / "Evaluation" / "UnlearnCanvas"))
    import constants as C  # type: ignore
    import evaluate as ev  # type: ignore
    import sample as smp  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    # Released sampling configuration; only the concept subset and seed count are
    # reduced, so this stays comparable to a full benchmark sweep in kind.
    smp.sample_unlearncanvas_images(
        pipeline_dir=str(gen_dir), output_dir=str(out_dir), device="cuda:0",
        unet_ckpt_path=None, seeds=[seed], guidance_scale=9.0,
        num_inference_steps=100, resolution=512,
        styles_subset=styles, objects_subset=objects,
    )
    torch.cuda.empty_cache()

    tf = ev.build_image_transform()
    results = {}
    for task, names in (("style", list(C.STYLES_AVAILABLE)),
                        ("object", list(C.OBJECTS_AVAILABLE))):
        model = ev.load_task_classifier(task, str(cls_dir), device)
        correct = total = 0
        with torch.no_grad():
            for st in styles:
                for ob in objects:
                    p = out_dir / f"{st}_{ob}_seed{seed}.jpg"
                    if not p.exists():
                        continue
                    x = tf(Image.open(p)).unsqueeze(0).to(device)
                    pred = int(model(x).argmax())
                    truth = st if task == "style" else ob
                    total += 1
                    correct += int(pred == names.index(truth))
        del model
        torch.cuda.empty_cache()
        acc = (correct / total) if total else None
        results[task] = {"images": total,
                         "accuracy": round(acc, 4) if acc is not None else None,
                         "pass": (acc is not None and acc >= threshold)}

    return {
        "status": "checked",
        "note": ("pre-unlearning reference point. Low accuracy here means the generator is "
                 "not the UnlearnCanvas-fine-tuned model, or the classifiers do not match it."),
        "sampling_config": {"seeds": [seed], "guidance_scale": 9.0,
                            "num_inference_steps": 100, "resolution": 512,
                            "styles": styles, "objects": objects},
        "threshold": threshold,
        "per_task": results,
        "output_dir": str(out_dir),
    }


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--cuig_root", default=os.environ.get(
        "CUIG_REPO_ROOT", "/data/bijaypandey/cuig_pilot/CUIG"))
    ap.add_argument("--generator_dir", default=os.environ.get(
        "CUIG_UNLEARNCANVAS_GENERATOR_DIR",
        "/data/bijaypandey/cuig_pilot/Checkpoints/Generators/UnlearnCanvas"))
    ap.add_argument("--classifier_dir", default=os.environ.get(
        "CUIG_UNLEARNCANVAS_CLASSIFIER_DIR",
        "/data/bijaypandey/cuig_pilot/Checkpoints/Classifiers/UnlearnCanvas"))
    ap.add_argument("--heldout", default="/data/bijaypandey/cuig_pilot/heldout_eval_set")
    ap.add_argument("--classifier-threshold", type=float, default=0.70,
                    help="min top-1 on held-out images (local policy, not a paper value)")
    ap.add_argument("--generator-threshold", type=float, default=0.70,
                    help="min top-1 on untouched-generator samples (local policy)")
    ap.add_argument("--skip-generator", action="store_true")
    ap.add_argument("--gen-styles", default="Abstractionism,Van_Gogh,Crayon")
    ap.add_argument("--gen-objects", default="Architectures,Horses,Flowers")
    ap.add_argument("--gen-seed", type=int, default=188)
    ap.add_argument("--receipt", default=str(REPO / "results" / "assets"
                                             / "validation_receipt.json"))
    ap.add_argument("--update-manifest", action="store_true",
                    help="write verified/not_validated status back to assets/asset_manifest.json")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    cuig_root = Path(args.cuig_root)
    gen_dir = Path(args.generator_dir)
    cls_dir = Path(args.classifier_dir)
    heldout = Path(args.heldout)

    sys.path.insert(0, str(cuig_root / "Evaluation" / "UnlearnCanvas"))
    import constants as C  # type: ignore

    receipt: dict = {
        "schema_version": "1.0",
        "generated_utc": now(),
        "host": os.uname().nodename,
        "gpu_index": args.gpu,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else UNKNOWN,
        "upstream_cuig_commit": "9932ac3271a122f6d38d19e0b8c8908fe5237ff7",
        "thresholds_are_local_policy": True,
        "checks": {},
        "unknowns": {},
        "verdict": {},
    }

    # --- A ---------------------------------------------------------------
    presence = check_presence(gen_dir, cls_dir)
    receipt["checks"]["presence"] = presence
    have_cls = presence["style_classifier"]["present"] and presence["object_classifier"]["present"]
    have_gen = presence["generator"]["present"]

    if not have_cls:
        receipt["verdict"] = {
            "overall": "FAIL",
            "reason": "classifier checkpoints absent",
            "assets_verified": False,
            "UA_IRA_CRA": "UNAVAILABLE",
        }
        receipt["unknowns"] = {
            "style_classifier_label_order": UNKNOWN,
            "object_classifier_label_order": UNKNOWN,
            "classifier_provenance": UNKNOWN,
            "generator_provenance": UNKNOWN,
            "generator_ema_selection": UNKNOWN,
        }
        write_receipt(receipt, Path(args.receipt))
        print_summary(receipt)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- B ---------------------------------------------------------------
    receipt["checks"]["shape_compatibility"] = {
        "style": check_shape(cls_dir, "style", len(C.STYLES_AVAILABLE)),
        "object": check_shape(cls_dir, "object", len(C.OBJECTS_AVAILABLE)),
    }

    # --- C / D -----------------------------------------------------------
    if not (heldout / "labels.json").exists():
        receipt["checks"]["heldout_label_mapping"] = {
            "status": "heldout_set_missing",
            "how_to_resolve": "python scripts/build_heldout_eval_set.py",
        }
        style_res = object_res = None
    else:
        style_res = evaluate_classifier_on_heldout(
            "style", cls_dir, cuig_root, heldout, device, args.classifier_threshold)
        object_res = evaluate_classifier_on_heldout(
            "object", cls_dir, cuig_root, heldout, device, args.classifier_threshold)
        receipt["checks"]["heldout_label_mapping"] = {"style": style_res, "object": object_res}

    # --- E ---------------------------------------------------------------
    gen_res = None
    if args.skip_generator:
        receipt["checks"]["untouched_generator_baseline"] = {"status": "skipped_by_flag"}
    elif not have_gen:
        receipt["checks"]["untouched_generator_baseline"] = {"status": "generator_absent"}
    else:
        gen_res = evaluate_untouched_generator(
            gen_dir, cls_dir, cuig_root, device,
            Path("/data/bijaypandey/cuig_pilot/validation/untouched_generator/images"),
            args.gen_styles.split(","), args.gen_objects.split(","),
            args.gen_seed, args.generator_threshold)
        receipt["checks"]["untouched_generator_baseline"] = gen_res

    # --- verdict ---------------------------------------------------------
    shapes_ok = all(receipt["checks"]["shape_compatibility"][t]["shape_compatible"]
                    for t in ("style", "object"))
    mapping_ok = bool(style_res and object_res
                      and style_res.get("label_mapping") == "confirmed"
                      and object_res.get("label_mapping") == "confirmed")
    acc_ok = bool(style_res and object_res
                  and style_res.get("accuracy_pass") and object_res.get("accuracy_pass"))
    gen_ok = bool(gen_res and all(v["pass"] for v in gen_res["per_task"].values()))

    classifiers_verified = shapes_ok and mapping_ok and acc_ok
    everything = classifiers_verified and (gen_ok if not args.skip_generator else False)

    receipt["unknowns"] = {
        "style_classifier_label_order": (
            "confirmed_consistent_with_constants_py" if (style_res and
            style_res.get("label_mapping") == "confirmed") else UNKNOWN),
        "object_classifier_label_order": (
            "confirmed_consistent_with_constants_py" if (object_res and
            object_res.get("label_mapping") == "confirmed") else UNKNOWN),
        # Behaviour never establishes provenance.
        "classifier_provenance": UNKNOWN,
        "generator_provenance": UNKNOWN,
        "generator_ema_selection": UNKNOWN,
        "preprocessing_matches_classifier_training": UNKNOWN,
        "provenance_note": ("behavioural agreement does not prove these are the authors' "
                            "checkpoints; only a published checksum or a direct copy can"),
    }
    receipt["verdict"] = {
        "overall": "PASS" if everything else "FAIL",
        "shape_compatible": shapes_ok,
        "label_mapping_confirmed": mapping_ok,
        "heldout_accuracy_pass": acc_ok,
        "untouched_generator_pass": gen_ok if not args.skip_generator else UNKNOWN,
        "classifiers_verified": classifiers_verified,
        "assets_verified": everything,
        "UA_IRA_CRA": "AVAILABLE" if everything else "UNAVAILABLE",
    }

    write_receipt(receipt, Path(args.receipt))

    if args.update_manifest:
        update_manifest(REPO / "assets" / "asset_manifest.json", receipt,
                        Path(args.receipt), classifiers_verified, gen_ok)

    print_summary(receipt)
    return 0 if everything else 1


def write_receipt(receipt: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2))
    print(f"\nreceipt: {path}")


def update_manifest(manifest_path: Path, receipt: dict, receipt_path: Path,
                    classifiers_verified: bool, gen_ok: bool) -> None:
    if not manifest_path.exists():
        return
    m = json.loads(manifest_path.read_text())
    for key, ok in (("style_classifier", classifiers_verified),
                    ("object_classifier", classifiers_verified),
                    ("generator", classifiers_verified and gen_ok)):
        a = m["assets"][key]
        a["validation"]["status"] = "verified" if ok else "not_validated"
        a["validation"]["receipt_path"] = str(receipt_path)
        a["validation"]["checked_utc"] = receipt["generated_utc"]
        # Provenance is not a behavioural property; leave it unknown.
        a["provenance"]["status"] = UNKNOWN
    if classifiers_verified:
        m["class_mappings"]["assumption_status"] = "confirmed_behaviourally"
    manifest_path.write_text(json.dumps(m, indent=2))
    print(f"manifest updated: {manifest_path}")


def print_summary(receipt: dict) -> None:
    v = receipt["verdict"]
    print("\n" + "=" * 72)
    print("ASSET VALIDATION")
    print("=" * 72)
    for k, val in v.items():
        print(f"  {k:32s} {val}")
    print("-" * 72)
    print("  unknowns retained:")
    for k, val in receipt["unknowns"].items():
        if val == UNKNOWN:
            print(f"    - {k}")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
