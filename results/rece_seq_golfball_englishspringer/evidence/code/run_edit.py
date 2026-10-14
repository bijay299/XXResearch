#!/usr/bin/env python3
"""Executes ONE direct, two-stage RECE edit of a given concept, starting
from either pristine M0 or a specified parent checkpoint (e.g. the verified
author GolfBall_ep0.pt). This is the actual GPU-executing counterpart to
`rece_golfball_confirmation/seq_prep/native_rece_adapter.py`'s refused
`--mode stage1/stage2` (that file deliberately raised SystemExit; this file
is the authorized, scoped execution of that same design for the
English-springer substitution documented in `../ELIGIBILITY.md`).

Stage 1 (preliminary UCE): vendored `edit_model()` (single-token slicing,
from erase_methods/train_uce.py) -- lamb=0.5, erase_scale=1,
preserve_scale=1.0 (auto, since retain_texts=['']), guide=' ', retain=[''],
technique='replace'. Mutates the live `ldm_stable` UNet's 32 cross-attention
to_k/to_v projections in place, starting from whatever is currently loaded
into `ldm_stable` (M0 for the direct branch, the GolfBall parent for the
sequential branch -- the ONLY difference between the two branches' Stage 1
calls).

Stage 2 (adversarial refinement): vendored `close_form_emb_regzero()`
computes an adversarial embedding using `ldm_stable` (now holding the
Stage-1 result) as the live basis and a SEPARATE, permanently pristine M0
copy (`ldm_stable_copy`) as the regularization/adversarial-embedding
reference -- identical in both branches, per this task's explicit
"retain original M0 as the adversarial embedding reference in both
branches" instruction. The live-imported (not vendored; see
vendored_functions.py docstring for why) `edit_model_adversarial()` then
mutates the SAME `ldm_stable` object again (full-77-token-sum convention),
erase_scale=1, preserve_scale=0.1, lamb=0.1, guide=' ', retain=[''],
technique='replace' -- the object-erasure defaults from train_object.py,
with --regular_scale 1e-1 as published for golf ball/parachute/English
springer in scripts/object.sh.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vendored_functions as vf  # noqa: E402

RECE_REPO = Path("/data/bijaypandey/cuig_pilot/rece_golfball_author_screen/repo_src/RECE")
sys.path.insert(0, str(RECE_REPO))
from erase_methods.uce import edit_model_adversarial  # noqa: E402  (live import: clean deps, already verified)

SD14 = "/data/bijaypandey/cuig_pilot/hf_home/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_vector(unet, key="up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.weight"):
    """A small, cheap fingerprint of a UNet's weights, used only to verify
    identity/non-identity and that the reference copy is untouched -- not a
    substitute for the full key/shape verification already done elsewhere."""
    return unet.state_dict()[key][0, :8].clone().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help='"M0" or a path to a .pt UNet state dict')
    ap.add_argument("--concept", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regular_scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    vf.assert_pinned_sources_unchanged()
    vf.assert_vendored_text_matches_pinned_source()

    log = {"start": args.start, "concept": args.concept, "regular_scale": args.regular_scale, "seed": args.seed}
    t0 = time.time()

    # --- load the working model (M0 or the specified starting checkpoint) ---
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

    # --- load the permanently pristine M0 reference copy (NEVER edited) ---
    ldm_stable_copy = StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None)
    ldm_stable_copy = ldm_stable_copy.to(args.device)

    # --- verify working-model / reference separation before any editing ---
    working_probe_before = probe_vector(ldm_stable.unet)
    reference_probe_before = probe_vector(ldm_stable_copy.unet)
    working_to_k = ldm_stable.unet.state_dict()["up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.weight"]
    reference_to_k = ldm_stable_copy.unet.state_dict()["up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.weight"]
    assert working_to_k.data_ptr() != reference_to_k.data_ptr(), (
        "working model and reference model share underlying storage -- refusing to edit, "
        "this would silently corrupt the reference"
    )
    if args.start == "M0":
        assert torch.equal(working_to_k, reference_to_k), "M0-start working model should equal the pristine reference before editing"
    log["working_reference_separate_storage_verified"] = True
    log["reference_identity_is_pristine_M0"] = True

    # ================= Stage 1: preliminary UCE (vendored edit_model) =================
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

    # reference must be untouched by stage 1
    assert torch.equal(probe_vector(ldm_stable_copy.unet), reference_probe_before), "Stage 1 mutated the reference copy -- aborting"
    stage1_probe = probe_vector(ldm_stable.unet)
    log["stage1_changed_working_model"] = not torch.equal(stage1_probe, working_probe_before)

    # ================= Stage 2: adversarial refinement =================
    t_stage2 = time.time()
    _, adv_embedding = vf.close_form_emb_regzero(
        model=ldm_stable,
        model_copy=ldm_stable_copy,
        concept=args.concept,
        with_to_k=True,
        save_path=None,
        regeular_scale=args.regular_scale,
        seed=args.seed,
    )
    id_new_text = ldm_stable.tokenizer(
        " ", padding="max_length", max_length=ldm_stable.tokenizer.model_max_length, truncation=True, return_tensors="pt"
    ).input_ids.to(args.device)
    new_embedding = ldm_stable.text_encoder(id_new_text)[0]

    ldm_stable = edit_model_adversarial(
        ldm_stable,
        old_embeddings=[adv_embedding[0]],
        new_embeddings=[new_embedding[0]],
        retain_texts=[""],
        technique="replace",
        preserve_scale=0.1,
        erase_scale=1,
        lamb=0.1,
    )
    log["stage2_seconds"] = time.time() - t_stage2

    # reference must still be untouched after stage 2
    assert torch.equal(probe_vector(ldm_stable_copy.unet), reference_probe_before), "Stage 2 mutated the reference copy -- aborting"

    # --- verify only the expected 32 cross-attn projections differ from the starting point ---
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

    log_path = out_path.with_suffix(".edit_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
