"""Measure the gradient-norm ratios that set the coefficients of Meta-v2 and UTRAP-v2, BEFORE
spending disk on either build. Writes one JSON; no checkpoint.

WHY. FINDING67 named a three-instance pattern: IMMA alpha, UCE lambda (~1e4x off), WDR alpha
(~4e3x off). Every time, a coefficient was inherited from a paper or a prior run and turned out to
be at the wrong scale for an 859.5M-parameter UNet, and every time the fix was one cheap measurement
of the quantity it is compared against. The two experiments queued next -- E2 (meta_unlearn_v2.py,
coefficients zeta and lam_target, both defaulting to 1.0) and E4 (erase_utrap_v2.py, lambda
defaulting to 0.15) -- inherit their coefficients exactly that way. zeta and lam_target come from
the paper's shared notation; lambda=0.15 comes from the OLD UTRAP runs whose inner attacker was a
different, much weaker process (ERRATUM_utrap_inner_attacker.md), so there is no reason it transfers
to the corrected attacker either.

This measures all of them against their counterpart terms, so E2 and E4 can be launched at parity
instead of becoming instances 4 and 5.

NOTE ON WHAT THIS IS NOT. A gradient-norm ratio gives the scale at which two terms are COMPARABLE.
It does not say that parity is the best setting -- only that a coefficient many orders of magnitude
away from parity cannot be doing what its author intended. That is the inference FINDING67 supports
and no more.
"""
import importlib.util as iu, json, torch
import torch.nn.functional as F
P = "/data/bijaypandey/t2i_unlearn"
def mod(n):
    s = iu.spec_from_file_location(n, f"{P}/scripts/{n}.py"); m = iu.module_from_spec(s)
    s.loader.exec_module(m); return m
core, undo = mod("attack_core"), mod("undo_t2i")
from diffusers import StableDiffusionPipeline, DDPMScheduler
pin = dict(l.split("=",1) for l in open(f"{P}/MODEL_PIN.txt").read().split("\n") if "=" in l)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
    revision=pin["revision"], torch_dtype=torch.float32, variant="fp16",
    safety_checker=None, requires_safety_checker=False).to("cuda")
unet, vae, te, tok = pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer
sched = DDPMScheduler.from_config(pipe.scheduler.config)
for m in (vae, te): m.requires_grad_(False)
import copy
tsd = torch.load(f"{P}/adapters/frp_church_r3_l05/unet.pt", map_location="cpu"); tsd = tsd.get("unet", tsd)
unet.load_state_dict({k: v.float() for k, v in tsd.items()})
teacher = copy.deepcopy(unet).eval().requires_grad_(False)
unet.requires_grad_(True); params = list(unet.parameters())
xattn = [p for n,p in unet.named_parameters() if "attn2" in n]
g = torch.Generator(device="cuda").manual_seed(0)

def latents(prompts):
    with torch.no_grad():
        im = pipe(prompts, num_inference_steps=25, guidance_scale=7.5, generator=g,
                  output_type="pt").images
        z = vae.encode(im.to("cuda")*2-1).latent_dist.sample(generator=g)*vae.config.scaling_factor
        ids = tok(prompts, padding="max_length", max_length=77, truncation=True,
                  return_tensors="pt").input_ids.to("cuda")
        return z, te(ids)[0]
# DEF_TRAIN target prompts (disjoint from eval), and retain prompts
tgt_p = ["a small roadside church","a church bell tower in daylight",
         "a stone church with arched windows","a whitewashed church by a field"]
f_lat, f_emb = latents(tgt_p)
r_lat, r_emb = latents(undo.RETAIN_POOL[:4])
idx = torch.tensor([0,1], device="cuda")

def gnorm(loss, ps):
    unet.zero_grad(set_to_none=True)
    gs = torch.autograd.grad(loss, ps, retain_graph=False, allow_unused=True)
    return sum(float(x.norm())**2 for x in gs if x is not None)**0.5

def dm(lat, emb):  return core.diffusion_mse(unet, sched, lat, emb, idx, g)
def distil(lat, emb):
    n = torch.randn(lat[idx].shape, device="cuda", generator=g)
    ts = torch.randint(0, 1000, (len(idx),), device="cuda", generator=g).long()
    zt = sched.add_noise(lat[idx], n, ts)
    with torch.no_grad(): t_ = teacher(zt, ts, encoder_hidden_states=emb[idx]).sample
    return F.mse_loss(unet(zt, ts, encoder_hidden_states=emb[idx]).sample, t_)

def avg(fn, k=3): 
    v=[fn() for _ in range(k)]; return sum(v)/len(v), v

# --- META v2 terms. meta gradient on xattn (reference's --finetuning_method), erasure at theta.
m_distil_f,_  = avg(lambda: gnorm(distil(f_lat, f_emb), params))
m_distil_r,_  = avg(lambda: gnorm(distil(r_lat, r_emb), params))
m_LR_theta,v1 = avg(lambda: gnorm(dm(r_lat, r_emb), xattn))
m_LF_prime,v2 = avg(lambda: gnorm(dm(f_lat, f_emb), xattn))
# --- UTRAP v2 terms, both on the full scope
u_retain,v3 = avg(lambda: gnorm(dm(r_lat, r_emb), params))
u_target,v4 = avg(lambda: gnorm(dm(f_lat, f_emb), params))

out = {"study":"coefficient_scale_diagnostic","n_params":sum(p.numel() for p in params),
 "n_params_xattn":sum(p.numel() for p in xattn),
 "meta_v2":{"grad_distil_erase_theta_full":m_distil_f,"grad_distil_retain_theta_full":m_distil_r,
   "grad_L_retain_theta_xattn":m_LR_theta,"grad_L_target_xattn":m_LF_prime,
   "zeta_for_parity_vs_distil_erase": m_distil_f/max(m_LR_theta,1e-30),
   "lam_target_for_parity_vs_distil_erase": m_distil_f/max(m_LF_prime,1e-30),
   "default_in_script":1.0},
 "utrap_v2":{"grad_L_retain_full":u_retain,"grad_L_target_full":u_target,
   "lambda_for_parity": u_retain/max(u_target,1e-30),"default_in_script":0.15,
   "note":"lambda=0.15 was inherited from the OLD UTRAP runs, whose inner attacker was a different "
          "and much weaker process (ERRATUM_utrap_inner_attacker.md). Parity here is the scale at "
          "which grad L_retain and lambda*grad L_target(theta_K) are comparable."},
 "trials":{"L_retain_xattn":v1,"L_target_xattn":v2,"L_retain_full":v3,"L_target_full":v4},
 "caveat":"A gradient-norm ratio gives the scale at which two terms are COMPARABLE. It does not say "
          "parity is optimal -- only that a coefficient orders of magnitude from parity cannot be "
          "doing what its author intended (FINDING67)."}
json.dump(out, open(f"{P}/outputs/diag_coeff_scale.json","w"), indent=1)
print(json.dumps(out, indent=1))
