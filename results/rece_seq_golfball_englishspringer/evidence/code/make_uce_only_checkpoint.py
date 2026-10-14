#!/usr/bin/env python3
"""Reconstructs ONE preliminary-UCE-only endpoint (Stage 1 of the completed
experiment, refinement/Stage 2 omitted entirely) starting from a verified
checkpoint (M0 or A). No saved Stage-1-only intermediate existed from the
original run (`edits/*.edit_log.json` only record final, post-Stage-2
checkpoints) -- this reconstructs it once, using the exact same vendored
`edit_model()` function, pinned source, and frozen hyperparameters already
verified in the completed experiment. Nothing about the guide, preservation,
regularization, or initialization choices is changed; Stage 2 (adversarial
embedding computation and `edit_model_adversarial`) is simply never called.
The reference-model-loading code from `run_edit.py` is reused unmodified
for the parts that still apply (loading, verification, drift-checking);
no unused Stage-2 components (the pristine-M0 reference copy, the
adversarial embedding) are instantiated at all, since omitting refinement
means they are never needed -- this is noted explicitly rather than loaded
and left idle."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

SEQ_EDIT_DIR = Path("/data/bijaypandey/cuig_pilot/rece_seq_golfball_englishspringer/seq_edit")
sys.path.insert(0, str(SEQ_EDIT_DIR))
import vendored_functions as vf  # noqa: E402

SD14 = "/data/bijaypandey/cuig_pilot/hf_home/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_vector(unet, key="up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.weight"):
    return unet.state_dict()[key][0, :8].clone().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help='"M0" or a path to a .pt UNet state dict')
    ap.add_argument("--concept", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    vf.assert_pinned_sources_unchanged()
    vf.assert_vendored_text_matches_pinned_source()

    log = {
        "start": args.start, "concept": args.concept,
        "procedure": "Stage 1 (preliminary UCE) ONLY -- refinement/Stage 2 omitted by design, not skipped due to failure",
        "stage1_hyperparameters": {"lamb": 0.5, "erase_scale": 1, "preserve_scale": 1.0, "guide": " ", "retain_texts": [""], "technique": "replace"},
        "stage2_omitted_components_unused": [
            "pristine-M0 reference copy (ldm_stable_copy) -- never instantiated, not needed when Stage 2 is omitted",
            "close_form_emb_regzero adversarial embedding -- never computed",
            "edit_model_adversarial (full 77-token-sum erase) -- never called",
        ],
    }
    t0 = time.time()

    ldm_stable = StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None)
    if args.start != "M0":
        start_path = Path(args.start)
        log["start_checkpoint_sha256"] = sha256_file(start_path)
        ckpt = torch.load(start_path, map_location="cpu")
        base_state = ldm_stable.unet.state_dict()
        assert set(ckpt.keys()) == set(base_state.keys()), "start checkpoint key set mismatch"
        assert all(base_state[k].shape == ckpt[k].shape for k in base_state), "start checkpoint shape mismatch"
        ldm_stable.unet.load_state_dict(ckpt, strict=True)
        for k in ckpt:
            assert torch.equal(ldm_stable.unet.state_dict()[k], ckpt[k]), f"post-load mismatch at {k}"
        log["start_checkpoint_loaded_and_verified"] = True
    ldm_stable = ldm_stable.to(args.device)

    working_probe_before = probe_vector(ldm_stable.unet)

    t_stage1 = time.time()
    ldm_stable = vf.edit_model(
        ldm_stable=ldm_stable,
        old_text_=[args.concept],
        new_text_=[" "],
        retain_text_=[""],
        lamb=0.5,
        erase_scale=1,
        preserve_scale=1.0,
        technique="replace",
    )
    log["stage1_seconds"] = time.time() - t_stage1
    log["stage1_changed_working_model"] = not torch.equal(probe_vector(ldm_stable.unet), working_probe_before)

    final_state = ldm_stable.unet.state_dict()
    if args.start == "M0":
        compare_state = StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None).unet.state_dict()
    else:
        compare_state = torch.load(Path(args.start), map_location="cpu")
    changed_keys = [k for k in final_state if not torch.equal(final_state[k].cpu(), compare_state[k].cpu())]
    log["n_keys_changed_vs_start"] = len(changed_keys)
    log["changed_keys_are_only_cross_attn_kv"] = all(("attn2.to_k" in k or "attn2.to_v" in k) for k in changed_keys)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_state, out_path)
    log["out_checkpoint_sha256"] = sha256_file(out_path)
    log["total_seconds"] = time.time() - t0

    log_path = out_path.with_suffix(".uce_only_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
