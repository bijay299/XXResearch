#!/usr/bin/env python3
"""Regenerate `assets/asset_manifest.json`.

The manifest is the single machine-readable record of what the benchmark assets
must be, what we currently hold, and — importantly — what is still **unknown**.
Class lists are read from the pinned upstream `constants.py` rather than
transcribed, so they cannot drift.

Fields whose value has not been established carry `{"value": null,
"status": "unknown", "how_to_resolve": ...}`. They are never filled with a guess.
`scripts/record_asset_hashes.py` fills in the download block; only
`scripts/validate_assets.py` may set a `validation.status` of `verified`.

Usage:
    python scripts/make_asset_manifest.py [--cuig_root ...] [--out ...]
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

DEFAULT_CUIG = "/data/bijaypandey/cuig_pilot/CUIG"
PINNED_COMMIT = "9932ac3271a122f6d38d19e0b8c8908fe5237ff7"
CKPT_ROOT = "/data/bijaypandey/cuig_pilot/Checkpoints"

DRIVE_CUIG_CLS = "https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61"
DRIVE_CUIG_GEN = "https://drive.google.com/drive/folders/18x40pLBcfNFyxBWZBGncTjqJTs_75SLx"
DRIVE_UC_CKPT = "https://drive.google.com/drive/folders/18dhkXyZQWjdMvlAlxZx3fZhdCZvlj2Hw"


def unknown(how_to_resolve: str) -> dict:
    return {"value": None, "status": "unknown", "how_to_resolve": how_to_resolve}


def not_downloaded() -> dict:
    return {"status": "not_downloaded", "source_url": None, "revision": None,
            "local_path": None, "size_bytes": None, "sha256": None, "retrieved_at": None}


def build(cuig_root: Path) -> dict:
    sys.path.insert(0, str(cuig_root / "Evaluation" / "UnlearnCanvas"))
    import constants as C  # type: ignore

    return {
        "schema_version": "1.0",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generated_by": "scripts/make_asset_manifest.py",
        "hash_caveat": (
            "A locally computed SHA-256 identifies the exact bytes held on this host. It does "
            "NOT independently prove equivalence to the authors' released checkpoint. "
            "Establishing that requires a checksum published by the authors, or a copy obtained "
            "directly from them."
        ),
        "status_vocabulary": {
            "unknown": "value not established; must not be guessed or inferred",
            "not_downloaded": "asset absent from this host",
            "present_unverified": "bytes present but provenance and/or behaviour not validated",
            "verified": "present AND passed scripts/validate_assets.py; receipt recorded",
        },
        "assets": {
            "generator": {
                "role": "base text-to-image generator for ConAbl training and UnlearnCanvas sampling",
                "upstream_name": "style50",
                "required_by": ["UnlearningMethods/ConAbl/train_conabl.py",
                                "Evaluation/UnlearnCanvas/sample.py"],
                "expected_local_path": f"{CKPT_ROOT}/Generators/UnlearnCanvas",
                "expected_format": "diffusers StableDiffusionPipeline directory",
                "expected_members": ["model_index.json", "scheduler/", "text_encoder/",
                                     "tokenizer/", "unet/", "vae/"],
                "documented_sources": [
                    {"url": DRIVE_CUIG_GEN, "cited_by": "CUIG Checkpoints/README.md",
                     "observed_http": 404},
                    {"url": DRIVE_UC_CKPT,
                     "cited_by": "UnlearnCanvas README.md / machine_unlearning / style_transfer",
                     "observed_http": 404},
                ],
                "version": unknown(
                    "ask maintainers which released generator revision CUIG's results used"),
                "architecture": unknown(
                    "confirm SD v1.x variant and exact base checkpoint after download"),
                "ema_selection": unknown(
                    "the released CompVis checkpoint carries both `model.` and `model_ema.` "
                    "weights; ask which set the released diffusers `style50` folder was "
                    "converted from"),
                "finetune_config": {
                    "base_model": unknown("ask maintainers (SD v1.4 vs v1.5)"),
                    "training_steps": unknown(
                        "ask maintainers; the third-party CompVis mirror embeds "
                        "global_step=7000, epoch=778, unconfirmed"),
                    "resolution": unknown("ask maintainers"),
                    "scheduler_config": unknown(
                        "read scheduler/scheduler_config.json after download"),
                },
                "download": not_downloaded(),
                "provenance": {"status": "unknown",
                               "note": "no authors' checksum published; no verified copy obtained"},
                "validation": {"status": "not_validated", "receipt_path": None},
            },
            "style_classifier": {
                "role": "UnlearnCanvas style classifier; drives UA and IRA for style unlearning",
                "upstream_filename": "style50.pth",
                "required_filename": "style_classifier.pth",
                "expected_local_path": f"{CKPT_ROOT}/Classifiers/UnlearnCanvas/style_classifier.pth",
                "required_by": ["Evaluation/UnlearnCanvas/evaluate.py"],
                "expected_architecture":
                    "timm vit_large_patch16_224.augreg_in21k with head = Linear(1024, 51)",
                "expected_state_dict_key": "model_state_dict",
                "expected_num_classes": len(C.STYLES_AVAILABLE),
                "documented_sources": [
                    {"url": DRIVE_CUIG_CLS, "cited_by": "CUIG Checkpoints/README.md",
                     "observed_http": 404},
                    {"url": DRIVE_UC_CKPT,
                     "cited_by": "UnlearnCanvas machine_unlearning/README.md (cls_model)",
                     "observed_http": 404},
                ],
                "class_index_order": unknown(
                    "confirm empirically via scripts/validate_assets.py against labelled "
                    "held-out images"),
                "download": not_downloaded(),
                "provenance": {"status": "unknown",
                               "note": "no accessible mirror found in the sources searched"},
                "validation": {"status": "not_validated", "receipt_path": None},
            },
            "object_classifier": {
                "role": "UnlearnCanvas object classifier; drives CRA for style unlearning",
                "upstream_filename": "style50_cls.pth",
                "required_filename": "object_classifier.pth",
                "expected_local_path": f"{CKPT_ROOT}/Classifiers/UnlearnCanvas/object_classifier.pth",
                "required_by": ["Evaluation/UnlearnCanvas/evaluate.py"],
                "expected_architecture":
                    "timm vit_large_patch16_224.augreg_in21k with head = Linear(1024, 20)",
                "expected_state_dict_key": "model_state_dict",
                "expected_num_classes": len(C.OBJECTS_AVAILABLE),
                "documented_sources": [
                    {"url": DRIVE_CUIG_CLS, "cited_by": "CUIG Checkpoints/README.md",
                     "observed_http": 404},
                    {"url": DRIVE_UC_CKPT,
                     "cited_by": "UnlearnCanvas machine_unlearning/README.md (cls_model)",
                     "observed_http": 404},
                ],
                "class_index_order": unknown(
                    "confirm empirically via scripts/validate_assets.py against labelled "
                    "held-out images"),
                "download": not_downloaded(),
                "provenance": {"status": "unknown",
                               "note": "no accessible mirror found in the sources searched"},
                "validation": {"status": "not_validated", "receipt_path": None},
            },
        },
        "class_mappings": {
            "source_of_truth":
                f"CUIG Evaluation/UnlearnCanvas/constants.py @ {PINNED_COMMIT}",
            "assumed_index_order": "list position == classifier output index",
            "assumption_status": "unverified",
            "assumption_evidence": (
                "both lists are exactly alphabetically sorted, consistent with torchvision "
                "ImageFolder ordering; necessary but NOT sufficient"),
            "styles": {"count": len(C.STYLES_AVAILABLE), "ordered": list(C.STYLES_AVAILABLE)},
            "objects": {"count": len(C.OBJECTS_AVAILABLE), "ordered": list(C.OBJECTS_AVAILABLE)},
            "note_seed_images":
                "STYLES_AVAILABLE contains 51 entries: 50 painting styles plus 'Seed_Images'",
        },
        "preprocessing": {
            "source_of_truth": "Evaluation/UnlearnCanvas/evaluate.py::build_image_transform",
            "transform": ["Resize((224, 224))", "ToTensor()", "Normalize(mean=[0.5], std=[0.5])"],
            "input_color_mode": "PIL default from Image.open (no explicit convert('RGB'))",
            "classifier_input_size": [224, 224],
            "confirmed_matches_classifier_training": unknown(
                "ask maintainers whether the released classifiers were trained with this exact "
                "transform"),
        },
        "sampling_config_for_reference": {
            "source_of_truth": "Evaluation/UnlearnCanvas/sample.py defaults",
            "prompt_template": "A {object} image in {style} style",
            "seeds": [188, 288, 588, 688, 888],
            "guidance_scale": 9.0,
            "num_inference_steps": 100,
            "resolution": 512,
            "dtype_on_cuda": "float16",
            "note_dataset_caption_difference": (
                "UnlearnCanvas dataset captions use the article 'An' before vowel-initial "
                "objects (e.g. 'An Architectures image in ...'); sample.py always emits 'A'."),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuig_root", default=DEFAULT_CUIG)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                        / "assets" / "asset_manifest.json"))
    args = ap.parse_args()

    manifest = build(Path(args.cuig_root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
