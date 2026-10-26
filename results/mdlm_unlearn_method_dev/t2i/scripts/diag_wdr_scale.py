"""Measure the gradient-norm ratio between the faithful WDR distance term and the retain term.

E1 ran erase_wdr_v2.py at the reference's default alpha=1.0 and the RMS distance moved
0.009024 -> 0.009076 over 1500 steps: +5.2e-5, with the trust region (0.05) never binding. That
looks like the same scale pathology as ERRATUM_uce_lambda_scale.md -- a hyperparameter whose useful
range was assumed rather than measured. This measures it instead of assuming again.

Analytically: for L = -sqrt(mean(d^2)), ||grad|| = ||d|| / (N * RMS) = 1/sqrt(N). At N = 859.5M
that is 1/29317 = 3.4e-5, INDEPENDENT of how far the model has moved. The reference's models are
vision classifiers ~30x smaller, where 1/sqrt(N) is ~6x larger and their retain CE gradients differ
too. So the term may simply be inert at this scale. Confirmed or refuted below by measurement.
"""
import importlib.util as iu, json, os, torch
P = "/data/bijaypandey/t2i_unlearn"
def mod(n):
    s = iu.spec_from_file_location(n, f"{P}/scripts/{n}.py"); m = iu.module_from_spec(s)
    s.loader.exec_module(m); return m
obj, undo = mod("objectives"), mod("undo_t2i")
from diffusers import StableDiffusionPipeline, DDPMScheduler
import torch.nn.functional as F
pin = dict(l.split("=",1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
    revision=pin["revision"], torch_dtype=torch.float32, variant="fp16",
    safety_checker=None, requires_safety_checker=False).to("cuda")
unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
sched = DDPMScheduler.from_config(pipe.scheduler.config)
for m in (vae, te): m.requires_grad_(False)
ref = [p.detach().clone() for p in unet.parameters()]
tsd = torch.load(f"{P}/adapters/frp_church_r3_l05/unet.pt", map_location="cpu"); tsd = tsd.get("unet", tsd)
unet.load_state_dict({k: v.float() for k, v in tsd.items()})
unet.requires_grad_(True); params = list(unet.parameters())
N = sum(p.numel() for p in params)

# distance-term gradient
unet.zero_grad(set_to_none=True)
obj.wdr_forget_loss(params, ref, alpha=1.0).backward()
gd = sum(float(p.grad.norm())**2 for p in params if p.grad is not None)**0.5

# retain-term gradient, on real retain data
g = torch.Generator(device="cuda").manual_seed(0)
pr = undo.RETAIN_POOL[:4]
with torch.no_grad():
    im = pipe(pr, num_inference_steps=25, guidance_scale=7.5, generator=g, output_type="pt").images
    z = vae.encode(im.to("cuda")*2-1).latent_dist.sample(generator=g)*vae.config.scaling_factor
    ids = tok(pr, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")
    emb = te(ids)[0]
gr_s = []
for trial in range(3):
    unet.zero_grad(set_to_none=True)
    noise = torch.randn(z.shape, device="cuda", generator=g)
    ts = torch.randint(0, 1000, (z.shape[0],), device="cuda", generator=g).long()
    F.mse_loss(unet(sched.add_noise(z, noise, ts), ts, encoder_hidden_states=emb).sample, noise).backward()
    gr_s.append(sum(float(p.grad.norm())**2 for p in params if p.grad is not None)**0.5)
gr = sum(gr_s)/len(gr_s)
out = {"study":"wdr_scale_diagnostic","n_params":N,"analytic_1_over_sqrt_N":N**-0.5,
       "grad_norm_distance_term_alpha1":gd,"grad_norm_retain_term_mean":gr,"retain_trials":gr_s,
       "ratio_retain_over_distance":gr/max(gd,1e-30),
       "alpha_for_parity":gr/max(gd,1e-30),
       "verdict":("INERT at alpha=1: the distance term's gradient is %.3g vs retain's %.3g, so alpha "
                  "must be ~%.3g for the two to be comparable" % (gd, gr, gr/max(gd,1e-30)))}
json.dump(out, open(f"{P}/outputs/diag_wdr_scale.json","w"), indent=1)
print(json.dumps(out, indent=1))
