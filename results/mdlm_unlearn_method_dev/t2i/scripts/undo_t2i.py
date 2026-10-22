"""Damage-then-distil robustification of an erased T2I model. The H38 hypothesis.

Ported from UNDO (arXiv 2506.06278), which established it for LLMs, plus the retain-Fisher damage
protection from this project's FINDING24. Two stages:

  1. DAMAGE   theta_damaged = (1-alpha) * theta_erased + alpha * beta * N        (N ~ Xavier)
  2. DISTIL   train the damaged student to match the ERASED TEACHER's noise predictions

The point is not to erase more. In the MDLM phase, damage did NOT reduce measurable knowledge --
it raised it slightly -- while cutting what an attacker could recover by 7x (FINDING28, FINDING31).
The mechanism claimed there is that damage destroys the attacker's ROUTE back to the concept, not
the concept itself. H38 asks whether that transfers to a diffusion UNet, where the teacher's output
is a noise-prediction field rather than a token distribution.

WHY THIS IS NOT OBVIOUS HERE. UCE and RECE are CLOSED-FORM edits to cross-attention; there is no
gradient history for damage to disrupt. If the mechanism is really about optimisation routes, a
closed-form edit may behave differently from a fine-tune. That is an empirical question and it can
fail.

LEGITIMACY. Targets are ALWAYS the erased teacher's own outputs. No concept images, no oracle, no
ground-truth labels. The distillation pool DOES include the erased concept's prompts -- their
absence was a real bug in the 8B phase of this project, which meant the suppression was never
transferred where it was being measured.

RETAIN-FISHER PROTECTION (FINDING24): scale each parameter's perturbation by
(1 - normalised retain importance), renormalised to preserve the mean damage magnitude, so damage
avoids the weights that general generation depends on. `--protect random` shuffles the same
importance distribution within each tensor and is the discriminating control: if `random` works as
well, the Fisher part is doing nothing.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

PIN = "/data/bijaypandey/t2i_unlearn/MODEL_PIN.txt"
OUT = "/data/bijaypandey/t2i_unlearn/outputs"

# The distillation pool. Includes the ERASED concept (so suppression transfers) and a spread of
# retain concepts and generic prompts (so general generation is repaired).
FORGET_POOL = {
    "church": ["a church", "a photo of a church", "an old stone church", "a church steeple"],
    "parachute": ["a parachute", "a photo of a parachute", "a parachute in the sky"],
    "van_gogh": ["a Van Gogh painting", "a painting in the style of Van Gogh"],
}
RETAIN_POOL = ["a photo of a cassette player", "a photo of a chain saw",
               "a photo of an english springer spaniel", "a photo of a french horn",
               "a photo of a gas pump", "a photo of a golf ball", "a photo of a tench",
               "a painting in the style of Monet", "a photo of a house", "a landscape photograph",
               "a portrait of a person", "a city street", "a bowl of fruit", "a mountain range",
               "a wooden chair", "a cup of coffee"]


def build(dtype, unet_path=None, dev="cuda"):
    from diffusers import StableDiffusionPipeline
    pin = dict(l.split("=", 1) for l in open(PIN).read().split("\n") if "=" in l)
    p = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=dtype,
        variant="fp16", safety_checker=None, requires_safety_checker=False)
    if unet_path:
        sd = torch.load(unet_path, map_location="cpu")
        p.unet.load_state_dict({k: v.to(dtype) for k, v in sd.get("unet", sd).items()})
    p.set_progress_bar_config(disable=True)
    return p.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="the ERASED unet to robustify")
    ap.add_argument("--concept", required=True, choices=sorted(FORGET_POOL))
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--protect", choices=["none", "retain_fisher", "random"], default="retain_fisher")
    ap.add_argument("--fisher-batches", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-images", type=int, default=48, help="distillation pool size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=150)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    prompts = FORGET_POOL[a.concept] * 3 + RETAIN_POOL
    # 1. build the distillation pool from the TEACHER's own generations (on-manifold, and the
    #    only thing the student is ever shown)
    teach_fp16 = build(torch.float16, a.teacher, dev)
    lats, embs = [], []
    vae, te, tok = teach_fp16.vae, teach_fp16.text_encoder, teach_fp16.tokenizer
    with torch.no_grad():
        for i in range(a.n_images):
            pr = prompts[i % len(prompts)]
            g = torch.Generator(device=dev).manual_seed(5_000 + i)
            im = teach_fp16([pr], num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
            x = torch.frombuffer(im.tobytes(), dtype=torch.uint8).reshape(
                im.size[1], im.size[0], 3).clone().permute(2, 0, 1)[None].to(dev).half() / 127.5 - 1.0
            lats.append((vae.encode(x).latent_dist.sample() * vae.config.scaling_factor).float())
            t_ = tok([pr], padding="max_length", max_length=77, truncation=True,
                     return_tensors="pt").to(dev)
            embs.append(te(t_.input_ids)[0].float())
    lats, embs = torch.cat(lats), torch.cat(embs)
    is_forget = torch.tensor([(i % len(prompts)) < len(FORGET_POOL[a.concept]) * 3
                              for i in range(a.n_images)], device=dev)
    print(f"[undo] distillation pool: {a.n_images} latents, "
          f"{int(is_forget.sum())} from the ERASED concept's prompts", flush=True)
    del teach_fp16
    torch.cuda.empty_cache()

    teacher = build(torch.float32, a.teacher, dev).unet.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    pipe = build(torch.float32, a.teacher, dev)
    student, sch = pipe.unet, pipe.scheduler
    pipe.text_encoder.requires_grad_(False); pipe.vae.requires_grad_(False)

    # 2. retain-Fisher importance (diagonal, squared grads of the diffusion loss on RETAIN prompts)
    imp = None
    if a.protect in ("retain_fisher", "random"):
        ret_idx = (~is_forget).nonzero().flatten()
        imp = {n: torch.zeros_like(p) for n, p in student.named_parameters()}
        for b in range(a.fisher_batches):
            i = ret_idx[torch.randint(0, len(ret_idx), (1,), device=dev)]
            z0, c = lats[i], embs[i]
            noise = torch.randn_like(z0)
            ts = torch.randint(0, sch.config.num_train_timesteps, (1,), device=dev).long()
            loss = F.mse_loss(student(sch.add_noise(z0, noise, ts), ts,
                                      encoder_hidden_states=c).sample, noise)
            student.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in student.named_parameters():
                if p.grad is not None:
                    imp[n] += p.grad.detach() ** 2
        student.zero_grad(set_to_none=True)
        print(f"[undo] retain-Fisher importance over {a.fisher_batches} batches", flush=True)

    # 3. DAMAGE
    g = torch.Generator(device=dev).manual_seed(a.seed + 777)
    with torch.no_grad():
        for n, p in student.named_parameters():
            N = torch.empty_like(p)
            if p.dim() >= 2:
                torch.nn.init.xavier_normal_(N, generator=g)
            else:
                N.normal_(0, 0.02, generator=g)
            s = a.alpha
            if imp is not None:
                v = imp[n]
                w = v / (v.max() + 1e-12)
                if a.protect == "random":            # same distribution, shuffled -> control
                    w = w.flatten()[torch.randperm(w.numel(), generator=g, device=dev)].view_as(w)
                keep = 1.0 - w
                keep = keep * (a.alpha / keep.mean().clamp_min(1e-6))   # preserve MEAN damage
                s = keep.clamp(0, 1)
            p.mul_(1 - s).add_(s * a.beta * N)
    print(f"[undo] damaged: alpha={a.alpha} beta={a.beta} protect={a.protect}", flush=True)

    # 4. DISTIL the erased teacher into the damaged student
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    t0, hist = time.time(), []
    for step in range(a.steps):
        i = torch.randint(0, len(lats), (1,), device=dev)
        z0, c = lats[i], embs[i]
        noise = torch.randn_like(z0)
        ts = torch.randint(0, sch.config.num_train_timesteps, (1,), device=dev).long()
        zt = sch.add_noise(z0, noise, ts)
        with torch.no_grad():
            tgt = teacher(zt, ts, encoder_hidden_states=c).sample
        loss = F.mse_loss(student(zt, ts, encoder_hidden_states=c).sample, tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        hist.append(loss.item())
        if step % a.log_every == 0 or step == a.steps - 1:
            print(f"[undo] step {step:5d} distil_loss={loss.item():.5f} "
                  f"mean50={np.mean(hist[-50:]):.5f} {(time.time()-t0)/max(step+1,1):.2f}s/step",
                  flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in student.state_dict().items()}, f"{a.out}/unet.pt")
    json.dump({"study": "t2i_undo", "label": a.label, "teacher": a.teacher, "hparams": vars(a),
               "n_forget_in_pool": int(is_forget.sum()),
               "distil_loss_first50": float(np.mean(hist[:50])),
               "distil_loss_last50": float(np.mean(hist[-50:])),
               "minutes": (time.time() - t0) / 60},
              open(f"{OUT}/undo_{a.label}.json", "w"), indent=1)
    print(f"[undo] saved {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
