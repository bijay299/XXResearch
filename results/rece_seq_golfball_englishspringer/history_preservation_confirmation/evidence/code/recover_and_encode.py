#!/usr/bin/env python3
"""GPU step 2 for one new sequence (S1 or S2): from the just-built, verified
Stage-1 checkpoint, (a) recover the new target's adversarial embedding
ONCE via the vendored close_form_emb_regzero (reference = pristine M0,
regular_scale=0.1, same native full-77-token procedure as every prior
recovery in this line), and (b) natively encode the guide (' ') and the
two retain texts (blank '', historical concept) needed for the BLANK/
HISTORY arms, batch [t,t] with a bit-identical-duplicate-row requirement.
Does not mutate the Stage-1 pipeline; only reads its weights.
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
    ap.add_argument("--seq", required=True, choices=["S1", "S2"])
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--new_target_concept", required=True)
    ap.add_argument("--historical_concept", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device

    vf.assert_pinned_sources_unchanged()
    vf.assert_vendored_text_matches_pinned_source()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"seq": args.seq, "new_target_concept": args.new_target_concept, "historical_concept": args.historical_concept}
    t0 = time.time()

    stage1_sha = sha256_file(Path(args.stage1_ckpt))
    result["stage1_ckpt_sha256"] = stage1_sha

    working = StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None)
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    base_state = working.unet.state_dict()
    assert set(ckpt.keys()) == set(base_state.keys())
    assert all(base_state[k].shape == ckpt[k].shape for k in base_state)
    working.unet.load_state_dict(ckpt, strict=True)
    for k in ckpt:
        assert torch.equal(working.unet.state_dict()[k], ckpt[k])
    working = working.to(device)

    reference = StableDiffusionPipeline.from_pretrained(SD14, safety_checker=None, feature_extractor=None).to(device)

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
    e_path = out_dir / f"E_{args.seq}_newtarget.pt"
    torch.save(E_newtarget, e_path)
    result["E_newtarget_path"] = str(e_path)
    result["E_newtarget_sha256"] = sha256_file(e_path)
    result["E_newtarget_norm"] = E_newtarget.norm(p="fro").item()

    def tok_embed(text, device=device):
        ids = working.tokenizer(text, padding="max_length", max_length=working.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(device)
        return working.text_encoder(ids)[0]

    guide = tok_embed(" ")[0].detach().cpu()
    g_path = out_dir / f"guide_{args.seq}.pt"
    torch.save(guide, g_path)
    result["guide_path"] = str(g_path)
    result["guide_sha256"] = sha256_file(g_path)
    result["guide_norm"] = guide.norm(p="fro").item()

    retain_report = {}
    for label, text in [("blank", ""), ("historical", args.historical_concept)]:
        ids = working.tokenizer([text, text], padding="max_length", max_length=working.tokenizer.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(device)
        emb = working.text_encoder(ids)[0].detach().cpu()
        row0, row1 = emb[0], emb[1]
        rows_equal = torch.equal(row0, row1)
        max_abs_diff = (row0 - row1).abs().max().item()
        r_path = out_dir / f"retain_{args.seq}_{label}.pt"
        torch.save(row0, r_path)
        retain_report[label] = {
            "text": text, "rows_equal_bit_identical": rows_equal, "max_abs_diff": max_abs_diff,
            "path": str(r_path), "sha256": sha256_file(r_path), "norm": row0.norm(p="fro").item(),
        }
        assert rows_equal, f"{label} retain text duplicate-row check failed"

    result["retain_encoding"] = retain_report
    result["elapsed_seconds"] = time.time() - t0
    (out_dir / f"RECOVER_ENCODE_{args.seq}_RESULT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
