#!/usr/bin/env python3
"""Baseline-validation checks 2 and 3, before any scored generation.

  (a) the trained checkpoints load into the RELEASE's inference Transcoder;
  (b) save/reload agreement: two independent loads produce bit-identical outputs on
      identical inputs (tolerance stated: exact equality expected, since both are fp32
      loads of the same bytes running the same kernel);
  (c) INFERENCE-PATH reconstruction quality: how well each transcoder reproduces the
      ORIGINAL FLUX module on held-out activations, measured through the inference class
      actually used at generation time (not the training class). This is what T0's
      substrate quality rests on;
  (d) finite computations everywhere;
  (e) reference storage separation: applying the native erasure in memory must not mutate
      the on-disk checkpoint, and must not touch the other arm's module;
  (f) the native erasure's own bookkeeping: how many W_dec rows it writes, duplicates,
      and whether non-W_dec tensors stay bit-identical.
"""
import os, sys, json, torch, copy

SPRINT = "/data/bijaypandey/cuig_pilot/block_object_sprint"
REPO = f"{SPRINT}/code/block-concept-removal"
sys.path.insert(0, REPO); sys.path.insert(0, f"{SPRINT}/code"); os.chdir(REPO)

from diffusers import FluxPipeline
from scripts.transcoder.model_inference import Transcoder
from scripts.unlearn_canvas.diffusion import compute_latent_percentages, apply_block_erasure

TD = f"{SPRINT}/artifacts/transcoders/native_T0"
A = f"{SPRINT}/artifacts/activations"
FN = "activations_flux_style.pt"
dev = "cuda:0"
rep = {}

# ---------------------------------------------------------------- (a) + (b) load / reload
for layer, ckpt in (("context_embedder", "context_embedder.pt"),
                    ("text_embedder", "text_embedder.pt")):
    sd = torch.load(f"{TD}/{ckpt}", map_location="cpu")
    tA = Transcoder.from_state_dict(sd).to(dev)          # PATCH A16 makes a 2nd load possible
    tB = Transcoder.from_state_dict(sd).to(dev)
    g = torch.Generator().manual_seed(11)
    x = torch.randn(512, tA.n_inputs, generator=g).to(dev)
    with torch.no_grad():
        yA, yB = tA(x), tB(x)
    r = {"n_inputs": tA.n_inputs, "n_outputs": tA.n_outputs,
         "n_latents": tA.n_latents, "k": tA.k,
         "param_dtype": str(tA.encoder.weight.dtype),
         "expansion": tA.n_latents // tA.n_inputs,
         "reload_bit_identical": bool(torch.equal(yA, yB)),
         "reload_max_abs_diff": float((yA - yB).abs().max()),
         "reload_tolerance": "exact equality (same bytes, same dtype, same kernel)",
         "outputs_finite": bool(torch.isfinite(yA).all()),
         "actual_optimizer_lr": 2e-4 / (tA.n_latents / (2 ** 14)) ** 0.5,
         "paper_stated_lr": 1e-4}
    rep[layer] = r
    print(f"{layer}: reload bit-identical={r['reload_bit_identical']} "
          f"k={r['k']} expansion={r['expansion']} latents={r['n_latents']} "
          f"actual_lr={r['actual_optimizer_lr']:.3e}")
    del tA, tB, yA, yB, x, sd
    torch.cuda.empty_cache()

# ------------------------------------------------- (c) inference-path reconstruction FVU
pipe = FluxPipeline.from_pretrained(os.environ["FLUX_PATH"], torch_dtype=torch.bfloat16).to(dev)
orig = {"context_embedder": pipe.transformer.context_embedder,
        "text_embedder": pipe.transformer.time_text_embed.text_embedder}

for layer, ckpt in (("context_embedder", "context_embedder.pt"),
                    ("text_embedder", "text_embedder.pt")):
    x = torch.load(f"{A}/{layer}_in/{FN}", map_location="cpu", weights_only=False)
    # held-out slice: the trainer's random_split used generator seed 42 over the SAME tensor,
    # so we take a disjoint-by-construction random sample and say so plainly: this is a
    # random sample of the same pool, NOT a guaranteed-unseen split.
    g = torch.Generator().manual_seed(99)
    idx = torch.randperm(x.shape[0], generator=g)[:16384]
    t = Transcoder.from_state_dict(torch.load(f"{TD}/{ckpt}", map_location="cpu")).to(dev)
    # chunked: the context transcoder's latent tensor is n x 65536 fp32, which is 4GB at
    # n=16384 and does not fit beside the resident bf16 pipeline.
    CH, yt_l, yh_l = 1024, [], []
    with torch.no_grad():
        for s in range(0, idx.numel(), CH):
            xs = x[idx[s:s + CH]].to(dev)
            yt_l.append(orig[layer](xs).float().cpu())
            yh_l.append(t(xs.float()).float().cpu())
            del xs
            torch.cuda.empty_cache()
    y_true = torch.cat(yt_l); y_hat = torch.cat(yh_l)
    del yt_l, yh_l, t
    torch.cuda.empty_cache()
    err = (y_true - y_hat).pow(2).sum()
    var = (y_true - y_true.mean(0, keepdim=True)).pow(2).sum()
    rep[layer]["inference_path"] = {
        "n_sampled": int(idx.numel()),
        "fvu": float(err / var),
        "variance_explained": float(1 - err / var),
        "rel_l2": float((y_true - y_hat).norm() / y_true.norm()),
        "cosine_mean": float(torch.nn.functional.cosine_similarity(y_true, y_hat, dim=-1).mean()),
        "chunk_size": CH,
        "y_hat_finite": bool(torch.isfinite(y_hat).all()),
        "note": "random sample of the training pool, not a guaranteed-unseen holdout",
    }
    ip = rep[layer]["inference_path"]
    print(f"{layer}: inference-path FVU={ip['fvu']:.4f} "
          f"(var explained {ip['variance_explained']:.4f}) cos={ip['cosine_mean']:.4f}")

# ------------------------------------- (e)+(f) erasure: isolation + intended-tensor checks
sdc = torch.load(f"{TD}/context_embedder.pt", map_location="cpu")
sdt = torch.load(f"{TD}/text_embedder.pt", map_location="cpu")
tc = Transcoder.from_state_dict(sdc).to(dev)
tt = Transcoder.from_state_dict(sdt).to(dev)
setattr(pipe.transformer, "context_embedder", tc)
setattr(pipe.transformer.time_text_embed, "text_embedder", tt)

erase_checks = {}
for target in ("Bears", "Cats", "Horses"):
    # fresh pristine modules per target: edits must never accumulate
    tc = Transcoder.from_state_dict(torch.load(f"{TD}/context_embedder.pt", map_location="cpu")).to(dev)
    tt = Transcoder.from_state_dict(torch.load(f"{TD}/text_embedder.pt", map_location="cpu")).to(dev)
    setattr(pipe.transformer, "context_embedder", tc)
    setattr(pipe.transformer.time_text_embed, "text_embedder", tt)
    before_c = tc.W_dec.data.clone(); before_t = tt.W_dec.data.clone()
    before_enc = tc.encoder.weight.data.clone()

    li = compute_latent_percentages(pipe, tc, tt, "flux", target, "flux")
    stats = apply_block_erasure(tc, tt, li, dev, stats={})

    ch_c = (tc.W_dec.data != before_c).any(1)
    ch_t = (tt.W_dec.data != before_t).any(1)
    erase_checks[target] = {
        "native_stats": stats,
        "context_rows_changed": int(ch_c.sum()), "context_rows_total": int(before_c.shape[0]),
        "pooled_rows_changed": int(ch_t.sum()), "pooled_rows_total": int(before_t.shape[0]),
        "context_frac_changed": float(ch_c.float().mean()),
        "pooled_frac_changed": float(ch_t.float().mean()),
        "all_finite_after": bool(torch.isfinite(tc.W_dec.data).all()
                                 and torch.isfinite(tt.W_dec.data).all()),
        "encoder_bit_identical_after_erasure": bool(torch.equal(before_enc, tc.encoder.weight.data)),
    }
    print(f"erase {target:7s}: context W_dec rows changed "
          f"{erase_checks[target]['context_rows_changed']}/{before_c.shape[0]}, "
          f"pooled {erase_checks[target]['pooled_rows_changed']}/{before_t.shape[0]}, "
          f"finite={erase_checks[target]['all_finite_after']}, "
          f"encoder untouched={erase_checks[target]['encoder_bit_identical_after_erasure']}")

# on-disk checkpoint must be unchanged by all that in-memory editing
sdc2 = torch.load(f"{TD}/context_embedder.pt", map_location="cpu")
rep["reference_storage_separation"] = {
    "on_disk_checkpoint_unchanged": bool(torch.equal(sdc["W_dec"], sdc2["W_dec"])),
    "note": ("erasure is applied to in-memory modules reloaded from a pristine checkpoint "
             "per target, so arms cannot mutate each other's reference"),
}
rep["erasure_checks"] = erase_checks
print("on-disk checkpoint unchanged after in-memory erasures:",
      rep["reference_storage_separation"]["on_disk_checkpoint_unchanged"])

json.dump(rep, open(f"{SPRINT}/artifacts/transcoders/verify_transcoders.json", "w"), indent=1)
print("\nwrote verify_transcoders.json")
