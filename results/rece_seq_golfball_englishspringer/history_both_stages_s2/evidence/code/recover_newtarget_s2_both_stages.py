#!/usr/bin/env python3
"""GPU step: recover the 'golf ball' adversarial embedding ONCE from the
S2_STAGE1_HISTORY working UNet (the Stage-1-with-historical-preservation
checkpoint, NOT S2_Stage1), against a separate pristine M0 reference.
Direct trim of history_preservation_confirmation/recover_and_encode.py:
identical recovery call, identical verification; the guide/retain
encoding half of that script is NOT reused here because guide_S2.pt and
retain_S2_historical.pt are reused unmodified from the existing files
(see FROZEN_MANIFEST.json) -- re-encoding them would not be "convenience,"
it would silently diverge from the frozen inputs. The old
E_S2_newtarget.pt (recovered from the corrupted S2_Stage1 checkpoint) is
never read or reused.
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
SEQ_DIR = HERE.parent
sys.path.insert(0, str(SEQ_DIR / "seq_edit"))
import vendored_functions as vf  # noqa: E402

SD14 = "/data/bijaypandey/cuig_pilot/hf_home/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
REGULAR_SCALE = 0.1
SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--working_ckpt", required=True, help="S2_STAGE1_HISTORY.pt")
    ap.add_argument("--working_ckpt_sha256", required=True)
    ap.add_argument("--new_target_concept", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device

    vf.assert_pinned_sources_unchanged()
    vf.assert_vendored_text_matches_pinned_source()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"working_ckpt": args.working_ckpt, "new_target_concept": args.new_target_concept}
    t0 = time.time()

    actual_sha = sha256_file(Path(args.working_ckpt))
    result["working_ckpt_sha256"] = actual_sha
    if actual_sha != args.working_ckpt_sha256:
        result["stop_reason"] = "SCOPED STOP: working checkpoint sha256 mismatch."
        (out_dir / "RECOVER_S2_BOTH_STAGES_RESULT.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    working = StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None)
    ckpt = torch.load(args.working_ckpt, map_location="cpu")
    base_state = working.unet.state_dict()
    assert set(ckpt.keys()) == set(base_state.keys())
    assert all(base_state[k].shape == ckpt[k].shape for k in base_state)
    working.unet.load_state_dict(ckpt, strict=True)
    for k in ckpt:
        assert torch.equal(working.unet.state_dict()[k], ckpt[k])
    result["working_ckpt_loaded_and_verified"] = True
    working = working.to(device)

    reference = StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None).to(device)
    result["reference_is_pristine_m0"] = True

    shared = [k for k in working.unet.state_dict() if working.unet.state_dict()[k].data_ptr() == reference.unet.state_dict()[k].data_ptr()]
    result["storage_separation"] = {"n_shared": len(shared)}
    assert not shared

    ref_before = {k: v.detach().cpu().clone() for k, v in reference.unet.state_dict().items()}
    work_before = {k: v.detach().cpu().clone() for k, v in working.unet.state_dict().items()}

    concept_embedding, adv_embedding = vf.close_form_emb_regzero(
        model=working, model_copy=reference, concept=args.new_target_concept,
        with_to_k=True, save_path=None, regeular_scale=REGULAR_SCALE, seed=SEED,
    )

    ref_after = {k: v.detach().cpu() for k, v in reference.unet.state_dict().items()}
    work_after = {k: v.detach().cpu() for k, v in working.unet.state_dict().items()}
    ref_changed = [k for k in ref_before if not torch.equal(ref_before[k], ref_after[k])]
    work_changed = [k for k in work_before if not torch.equal(work_before[k], work_after[k])]
    result["reference_unchanged_during_recovery"] = (len(ref_changed) == 0)
    result["working_unchanged_during_recovery"] = (len(work_changed) == 0)
    assert not ref_changed and not work_changed, "recovery must not mutate either pipeline's weights"

    assert torch.isfinite(adv_embedding).all()
    E_newtarget = adv_embedding.detach().cpu()  # [1,77,768]
    e_path = out_dir / "E_S2_BOTH_STAGES_newtarget.pt"
    torch.save(E_newtarget, e_path)
    result["E_newtarget_path"] = str(e_path)
    result["E_newtarget_sha256"] = sha256_file(e_path)
    result["E_newtarget_norm"] = E_newtarget.norm(p="fro").item()
    result["all_finite"] = bool(torch.isfinite(E_newtarget).all())

    result["elapsed_seconds"] = time.time() - t0
    (out_dir / "RECOVER_S2_BOTH_STAGES_RESULT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
