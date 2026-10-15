#!/usr/bin/env python3
"""Builds S1_STAGE1_HISTORY: the same native Stage-1-only UCE edit as
`refinement_ablation/make_uce_only_checkpoint.py` (reused with ONE
argument change), starting from the untouched, verified GolfBall author
parent, erasing 'French horn'. The only scientific change from the
existing S1_Stage1 comparator is retain_texts=[''] -> retain_texts=
['golf ball']. Same vendored edit_model() function, same pinned source,
same frozen hyperparameters (lamb=0.5, erase_scale=1, preserve_scale=1.0,
guide=' ', technique='replace', all 32 cross-attention K/V weights,
native token-suffix handling, one native edit). No Stage 2, no
recovered embedding, no fp64 solver anywhere in this script.
"""
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
    ap.add_argument("--start", required=True, help='path to the untouched GolfBall parent .pt')
    ap.add_argument("--start_sha256", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--retain_text", required=True, help="the ONE scientific change vs. S1_Stage1: '' -> 'golf ball'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    vf.assert_pinned_sources_unchanged()
    vf.assert_vendored_text_matches_pinned_source()

    log = {
        "start": args.start, "concept": args.concept,
        "procedure": "Stage 1 (preliminary UCE) ONLY, from the untouched GolfBall parent -- the ONLY scientific "
                     "change vs. the existing S1_Stage1 comparator is retain_text_",
        "stage1_hyperparameters": {"lamb": 0.5, "erase_scale": 1, "preserve_scale": 1.0, "guide": " ",
                                     "retain_texts": [args.retain_text], "technique": "replace"},
        "comparator_hyperparameters_for_reference": {"lamb": 0.5, "erase_scale": 1, "preserve_scale": 1.0, "guide": " ",
                                                       "retain_texts": [""], "technique": "replace"},
        "stage2_omitted_components_unused": [
            "pristine-M0 reference copy (ldm_stable_copy) -- never instantiated",
            "close_form_emb_regzero adversarial embedding -- never computed",
            "edit_model_adversarial (fp64 refinement solve) -- never called",
        ],
    }
    t0 = time.time()

    start_path = Path(args.start)
    actual_start_sha = sha256_file(start_path)
    log["start_checkpoint_sha256"] = actual_start_sha
    if actual_start_sha != args.start_sha256:
        log["stop_reason"] = "SCOPED STOP: start checkpoint sha256 mismatch."
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).with_suffix(".stage1_history_log.json").write_text(json.dumps(log, indent=2))
        print(json.dumps(log, indent=2))
        return

    ldm_stable = StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None)
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
        retain_text_=[args.retain_text],  # the ONE scientific change
        lamb=0.5,
        erase_scale=1,
        preserve_scale=1.0,
        technique="replace",
    )
    log["stage1_seconds"] = time.time() - t_stage1
    log["stage1_changed_working_model"] = not torch.equal(probe_vector(ldm_stable.unet), working_probe_before)
    assert torch.isfinite(probe_vector(ldm_stable.unet)).all(), "non-finite probe after edit"

    final_state = ldm_stable.unet.state_dict()
    compare_state = ckpt  # the untouched GolfBall parent, already loaded
    changed_keys = [k for k in final_state if not torch.equal(final_state[k].cpu(), compare_state[k].cpu())]
    log["n_keys_changed_vs_start"] = len(changed_keys)
    log["changed_keys_are_only_cross_attn_kv"] = all(("attn2.to_k" in k or "attn2.to_v" in k) for k in changed_keys)
    log["all_686_tensors_finite"] = all(torch.isfinite(final_state[k]).all().item() for k in final_state)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_state_cpu = {k: v.detach().clone().cpu() for k, v in final_state.items()}
    torch.save(final_state_cpu, out_path)
    log["out_checkpoint_sha256"] = sha256_file(out_path)

    # reload verification
    reloaded = torch.load(out_path, map_location="cpu")
    assert set(reloaded.keys()) == set(compare_state.keys())
    changed_reload = [k for k in reloaded if not torch.equal(reloaded[k], compare_state[k])]
    unchanged_exact = all(torch.equal(reloaded[k], compare_state[k]) for k in reloaded if k not in changed_reload)
    all_finite_reload = all(torch.isfinite(reloaded[k]).all().item() for k in reloaded)
    log["reload_verification"] = {
        "n_total_tensors": len(reloaded), "n_changed_vs_parent": len(changed_reload),
        "changed_set_size_is_32": len(changed_reload) == 32,
        "other_654_tensors_byte_identical_to_parent": unchanged_exact,
        "all_tensors_finite": all_finite_reload,
    }
    assert len(changed_reload) == 32 and unchanged_exact and all_finite_reload, "reload verification failed"

    log["total_seconds"] = time.time() - t0
    log_path = out_path.with_suffix(".stage1_history_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
