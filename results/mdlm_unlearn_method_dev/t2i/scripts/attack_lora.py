"""LoRA relearning attack — the PARAMETER-EFFICIENT adversary the defence literature targets.

WHY THIS EXISTS. Every attack in this phase so far fine-tunes the FULL UNet (859.5M parameters).
That is a strictly larger attack surface than the adaptation methods the protection literature is
written against: IMMA (2311.18815) immunises against LoRA, Textual-Inversion and DreamBooth; Glaze
and MIST poison data against the same family. So our negative results are repeatedly a statement
about a BIGGER adversary than those papers claim to stop (ERRATUM_r2 C15 for AdvUnlearn, FINDING43
S2 for FRP).

This script closes that gap from the other side: it attacks our erased arms with LoRA, the canonical
parameter-efficient adaptation method, under the SAME T/H split, the SAME FINDING20 validity gate,
and the SAME post-attack utility probe. If erasure survives LoRA but not full fine-tuning, that is a
scoping result worth having; if it falls to LoRA too, the phase's conclusion extends to the attack
family the literature actually cares about.

LoRA IS IMPLEMENTED HERE DIRECTLY, not via peft. peft is not installed in this venv and a dependency
upgrade has broken a venv on this machine before, so a 40-line low-rank adapter under our own
control is preferable to an environment change mid-project. The construction is the standard one:
for a frozen linear W, the adapter adds (alpha/r) * B @ A with A, B trainable, B initialised to
zero so the model starts exactly at the erased checkpoint.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PIN = "/data/bijaypandey/t2i_unlearn/MODEL_PIN.txt"
OUT = "/data/bijaypandey/t2i_unlearn/outputs"


def _ar():
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "ar_lora", os.path.join(os.path.dirname(os.path.abspath(__file__)), "attack_relearn.py"))
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class LoRALinear(nn.Module):
    """W(x) + (alpha/r) * B(A(x)).  B starts at zero => identical to the frozen model at step 0."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=1.0 / r)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def inject_lora(unet, r, alpha, scope):
    """scope: 'cross' = attn2 only (text-conditioned), 'all_attn' = attn1 + attn2."""
    targets = []
    for name, mod in unet.named_modules():
        if ".transformer_blocks." not in name:
            continue
        if scope == "cross" and ".attn2." not in name:
            continue
        if scope == "all_attn" and (".attn1." not in name and ".attn2." not in name):
            continue
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                targets.append((mod, child_name, child, f"{name}.{child_name}"))
    injected = []
    for parent, cname, child, full in targets:
        setattr(parent, cname, LoRALinear(child, r, alpha))
        injected.append(full)
    return injected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default=None)
    ap.add_argument("--text-encoder", default=None)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA is usually trained at a higher lr "
                                                           "than full fine-tuning")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--scope", default="cross", choices=["cross", "all_attn"])
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--gate", type=float, default=0.20)
    ap.add_argument("--eval-n", type=int, default=5)
    ap.add_argument("--eval-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)

    ar = _ar()
    assert a.concept in ar.SPLIT, f"unknown concept {a.concept}"
    T, H = ar.SPLIT[a.concept]["T"], ar.SPLIT[a.concept]["H"]
    panel = ar.UTILITY_PANEL_DEFAULT
    cdict = ar.gen_eval_concepts()

    from transformers import CLIPModel, CLIPProcessor
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                   torch_dtype=torch.float16).to(dev).eval()
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    with torch.no_grad():
        ti = cp(text=[f"a photo of a {l}" for l in ar.LABELS], return_tensors="pt",
                padding=True).to(dev)
        tfeat = cm.get_text_features(**ti)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    # attacker data: base-model generations on T prompts, identical to the full-FT attack
    base = ar.build_pipe(torch.float16, None, dev)
    imgs, caps = [], []
    for i in range(a.n_images):
        pr = T[i % len(T)]
        g = torch.Generator(device=dev).manual_seed(10_000 + i)
        imgs.append(base([pr], num_inference_steps=25, guidance_scale=7.5, generator=g).images[0])
        caps.append(pr)
    del base
    torch.cuda.empty_cache()

    pipe = ar.build_pipe(torch.float32, a.unet, dev, te_path=a.text_encoder)
    unet, vae, te, tok, sch = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.scheduler
    unet.requires_grad_(False); te.requires_grad_(False); vae.requires_grad_(False)

    n_full = sum(p.numel() for p in unet.parameters())
    injected = inject_lora(unet, a.rank, a.alpha, a.scope)
    unet.to(dev)
    lora_params = [p for n, p in unet.named_parameters() if (".A." in n or ".B." in n)]
    n_lora = sum(p.numel() for p in lora_params)
    assert n_lora > 0, "no LoRA parameters were created"
    print(f"[lora] injected into {len(injected)} linears (scope={a.scope}); trainable "
          f"{n_lora/1e6:.3f}M of {n_full/1e6:.1f}M UNet params = {100*n_lora/n_full:.3f}%",
          flush=True)

    # pre-measure
    with torch.no_grad():
        half = ar.build_pipe(torch.float16, a.unet, dev, te_path=a.text_encoder)
        T_before = ar.clip_acc(half, cm, cp, tfeat, T, a.concept, 3, a.eval_steps, a.seed, dev)
        H_before = ar.clip_acc(half, cm, cp, tfeat, H, a.concept, a.eval_n, a.eval_steps, a.seed, dev)
        u_before = ar.utility_probe(half, cm, cp, tfeat, panel, cdict, 2, 4, a.eval_steps,
                                    a.seed, dev)
        del half; torch.cuda.empty_cache()
    print(f"[lora] BEFORE T={T_before:.3f} H={H_before:.3f} "
          f"util={u_before['panel_mean_clip_acc']:.4f}", flush=True)

    with torch.no_grad():
        lat = []
        for im in imgs:
            x = torch.from_numpy(
                (torch.frombuffer(im.tobytes(), dtype=torch.uint8)
                 .reshape(im.size[1], im.size[0], 3).numpy())).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1.0
            lat.append(vae.encode(x).latent_dist.sample() * vae.config.scaling_factor)
        lat = torch.cat(lat)
        tok_ids = torch.cat([tok([c], padding="max_length", max_length=77, truncation=True,
                                 return_tensors="pt").input_ids for c in caps]).to(dev)
        emb = torch.cat([te(tok_ids[k:k+1])[0] for k in range(len(caps))])

    opt = torch.optim.Adam(lora_params, lr=a.lr)
    unet.train()
    t0 = time.time()
    for step in range(a.steps):
        i = torch.randint(0, len(lat), (1,), device=dev)
        z0, c = lat[i], emb[i]
        noise = torch.randn_like(z0)
        ts = torch.randint(0, sch.config.num_train_timesteps, (1,), device=dev).long()
        zt = sch.add_noise(z0, noise, ts)
        loss = F.mse_loss(unet(zt, ts, encoder_hidden_states=c).sample, noise)
        opt.zero_grad(set_to_none=True); loss.backward()
        if step == 0:
            gn = sum(float(p.grad.norm()**2) for p in lora_params if p.grad is not None)**0.5
            print(f"[lora] grad-norm step0 (LoRA only) = {gn:.4e}", flush=True)
            ref = [p.detach().clone() for p in lora_params]
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0); opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"[lora] step {step:4d} loss={loss.item():.5f} "
                  f"{(time.time()-t0)/max(step+1,1):.2f}s/step", flush=True)
    d_lora = sum(float((p.detach()-r).norm()**2) for p, r in zip(lora_params, ref))**0.5
    print(f"[lora] ||delta LoRA params|| = {d_lora:.4e}", flush=True)

    # post-measure: fold LoRA into fp16 weights so evaluation uses the ordinary pipeline
    unet.eval()
    with torch.no_grad():
        for name, mod in unet.named_modules():
            if isinstance(mod, LoRALinear):
                mod.base.weight.add_(mod.scale * (mod.B.weight @ mod.A.weight))
        merged = {}
        for k, v in unet.state_dict().items():
            if ".A.weight" in k or ".B.weight" in k:
                continue
            merged[k.replace(".base.weight", ".weight").replace(".base.bias", ".bias")] = v.half()
    tmp = f"/data/bijaypandey/t2i_unlearn/adapters/_lora_{a.label}"
    os.makedirs(tmp, exist_ok=True)
    torch.save(merged, f"{tmp}/unet.pt")
    del unet, opt, pipe
    torch.cuda.empty_cache()

    after = ar.build_pipe(torch.float16, f"{tmp}/unet.pt", dev, te_path=a.text_encoder)
    with torch.no_grad():
        T_after = ar.clip_acc(after, cm, cp, tfeat, T, a.concept, 3, a.eval_steps, a.seed, dev)
        H_after = ar.clip_acc(after, cm, cp, tfeat, H, a.concept, a.eval_n, a.eval_steps, a.seed, dev)
        u_after = ar.utility_probe(after, cm, cp, tfeat, panel, cdict, 2, 4, a.eval_steps,
                                   a.seed, dev)
    os.system(f"rm -rf {tmp}")

    valid = (T_after - T_before) >= a.gate
    R = {"study": "t2i_lora_relearn_attack", "label": a.label, "concept": a.concept,
         "hparams": vars(a),
         "adversary": {"kind": "LoRA (parameter-efficient)", "rank": a.rank, "alpha": a.alpha,
                       "scope": a.scope, "n_injected_linears": len(injected),
                       "n_trainable": n_lora, "n_unet_params": n_full,
                       "frac_trainable": n_lora / n_full,
                       "implementation": "hand-rolled low-rank adapter; peft is not installed and "
                                         "the venv is not modified mid-project",
                       "why": "IMMA/Glaze/MIST target LoRA, Textual-Inversion and DreamBooth. Every "
                              "other attack in this phase fine-tunes the FULL UNet, a strictly "
                              "larger adversary than those defences claim to stop."},
         "attack_validity": {"T_before": T_before, "T_after": T_after,
                             "T_rise": T_after - T_before, "gate": a.gate, "valid": bool(valid)},
         "recovery": {"H_before": H_before, "H_after": H_after,
                      "absolute_rise": H_after - H_before},
         "post_attack_utility": {"status": "MEASURED",
                                 "retained_capability": {"before": u_before["panel_mean_clip_acc"],
                                                         "after": u_after["panel_mean_clip_acc"]},
                                 "prompt_adherence": {"before": u_before["panel_mean_prompt_adherence"],
                                                      "after": u_after["panel_mean_prompt_adherence"]}},
         "lora_param_delta": d_lora}
    json.dump(R, open(f"{OUT}/atk_{a.label}.json", "w"), indent=1)
    print(f"[lora] {a.label}: T {T_before:.3f}->{T_after:.3f} (gate "
          f"{'PASS' if valid else 'FAIL'}) | H {H_before:.3f}->{H_after:.3f} | util "
          f"{u_before['panel_mean_clip_acc']:.4f}->{u_after['panel_mean_clip_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
