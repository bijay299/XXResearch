"""ESD — Erased Stable Diffusion (Gandikota et al., arXiv 2303.07345). OUR RE-IMPLEMENTATION.

Fine-tunes the UNet so that the erased model predicts the *negatively guided* noise that the
FROZEN original model produces for the concept:

    target = eps*(x_t, null, t) - eta * [ eps*(x_t, c, t) - eps*(x_t, null, t) ]
    loss   = || eps_theta(x_t, c, t) - target ||^2

so the concept prompt is pushed to produce what the original model produces *away* from the
concept. The target comes entirely from the frozen original -- no concept images are used, which
is why ESD needs no training data.

x_t is obtained by partially sampling from the TRAINABLE model with the concept prompt, as in the
paper: the erasure is evaluated where the model currently puts the concept.

ESD-u trains everything EXCEPT cross-attention (the paper's setting for objects); ESD-x trains only
cross-attention (its setting for styles). Labelled a re-implementation throughout (PREREG_H38 S4).
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

PIN = "/data/bijaypandey/t2i_unlearn/MODEL_PIN.txt"
PROMPTS = {"church": "church", "parachute": "parachute", "garbage_truck": "garbage truck",
           "van_gogh": "Van Gogh"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, choices=sorted(PROMPTS))
    ap.add_argument("--mode", choices=["esd_u", "esd_x"], default="esd_u")
    ap.add_argument("--eta", type=float, default=1.0, help="negative guidance strength")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.0, help="CFG used when sampling x_t")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)

    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pin = dict(l.split("=", 1) for l in open(PIN).read().split("\n") if "=" in l)
    kw = dict(revision=pin["revision"], variant="fp16", safety_checker=None,
              requires_safety_checker=False)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32, **kw).to(dev)
    frozen = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32, **kw).unet.to(dev).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    sched = DDIMScheduler.from_config(pipe.scheduler.config)
    sched.set_timesteps(a.ddim_steps, device=dev)
    unet, te, tok = pipe.unet, pipe.text_encoder, pipe.tokenizer
    te.requires_grad_(False)

    # which parameters ESD trains
    is_xattn = lambda n: "attn2" in n
    train_ps, names = [], []
    for n, p in unet.named_parameters():
        want = is_xattn(n) if a.mode == "esd_x" else (not is_xattn(n))
        p.requires_grad_(want)
        if want:
            train_ps.append(p); names.append(n)
    print(f"[esd] mode={a.mode}: training {len(train_ps)} tensors "
          f"({sum(p.numel() for p in train_ps)/1e6:.1f}M params)", flush=True)
    opt = torch.optim.Adam(train_ps, lr=a.lr)

    @torch.no_grad()
    def embed(txt):
        ti = tok([txt], padding="max_length", max_length=77, truncation=True,
                 return_tensors="pt").to(dev)
        return te(ti.input_ids)[0]

    e_c, e_null = embed(PROMPTS[a.concept]), embed("")
    t0, hist = time.time(), []
    for step in range(a.steps):
        # 1. partially sample x_t from the TRAINABLE model with the concept prompt
        k = int(torch.randint(1, a.ddim_steps, (1,)).item())
        with torch.no_grad():
            lat = torch.randn(1, 4, 64, 64, device=dev)
            for ts in sched.timesteps[:k]:
                inp = torch.cat([lat, lat])
                emb = torch.cat([e_null, e_c])
                n_u, n_c = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
                noise = n_u + a.guidance * (n_c - n_u)
                lat = sched.step(noise, ts, lat).prev_sample
            t_cur = sched.timesteps[k]
            # 2. the negatively-guided target, from the FROZEN original only
            n_null = frozen(lat, t_cur, encoder_hidden_states=e_null).sample
            n_conc = frozen(lat, t_cur, encoder_hidden_states=e_c).sample
            target = n_null - a.eta * (n_conc - n_null)
        # 3. push the trainable model's concept prediction onto it
        pred = unet(lat, t_cur, encoder_hidden_states=e_c).sample
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_ps, 1.0)
        opt.step()
        hist.append(loss.item())
        if step % a.log_every == 0 or step == a.steps - 1:
            print(f"[esd] step {step:5d} loss={loss.item():.5f} "
                  f"mean50={sum(hist[-50:])/len(hist[-50:]):.5f} "
                  f"{(time.time()-t0)/max(step+1,1):.2f}s/step", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, f"{a.out}/unet.pt")
    json.dump({"study": "t2i_esd_reimpl", "label": a.label, "concept": a.concept,
               "hparams": vars(a), "n_trained_params": sum(p.numel() for p in train_ps),
               "loss_first50": sum(hist[:50])/max(len(hist[:50]),1),
               "loss_last50": sum(hist[-50:])/max(len(hist[-50:]),1),
               "minutes": (time.time()-t0)/60,
               "note": "OUR RE-IMPLEMENTATION of arXiv 2303.07345; original tuning not reproduced"},
              open(f"/data/bijaypandey/t2i_unlearn/outputs/erase_{a.label}.json", "w"), indent=1)
    print(f"[esd] saved {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
