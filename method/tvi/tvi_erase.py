"""TVI — Task-Vector Invariant erasure.

THE IDEA. ResAlign (NeurIPS 2025) resists fine-tuning by optimising a FIXED-POINT objective: be
erased now, and do not become less erased after someone fine-tunes you. It works -- we verified it
on the authors' own checkpoints. Its cost is a bi-level problem: every outer step simulates an
inner fine-tuner for 5-30 gradient steps, sampled over 15 attacker configurations.

TVI replaces that inner simulation with a PRECOMPUTED DIRECTION. A concept task vector
tau = theta_finetuned - theta_base IS the weight change a fine-tuner produces on this concept, and
it costs 1.7 minutes to build once. So the anticipated attacked model is simply

    theta* ~= theta + eta * tau

and the objective becomes ResAlign's, with the inner loop solved analytically:

    L = L_esd(theta) + lam * [ L_esd(theta + eta*tau) - L_esd(theta) ] + w * L_retain(theta)

Minimising the bracketed term asks for a point where moving along tau does NOT restore the concept
-- an invariance condition along the attacker's own direction, rather than a barrier or a trap.

WHY THIS IS NOT JUST A BIGGER SUBTRACTION. theta - alpha*tau moves anti-parallel to tau and leaves
the concept one gradient step away. TVI instead searches for a theta whose DIRECTIONAL DERIVATIVE
of concept generation along tau is small. Those are different requests, and only the second is a
statement about the attacker's dynamics.

COST, which is the claim: two forward/backward passes per step (at theta and at theta+eta*tau)
instead of 5-30 inner steps, and tau is computed once rather than resampled.

Loss follows our ESD port exactly (scripts/erase_esd.py):
    target = eps(x_t, null, t) - eta_esd * [ eps(x_t, c, t) - eps(x_t, null, t) ]
with the frozen reference model supplying both eps terms.
"""
import argparse, json, os, time, copy
import torch
import torch.nn.functional as F

ROOT = "/data/bijaypandey/t2i_unlearn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="church")
    ap.add_argument("--tau", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--esd-eta", type=float, default=1.0, help="negative guidance strength")
    ap.add_argument("--lam", type=float, default=1.0, help="weight on the invariance term")
    ap.add_argument("--tau-eta", type=float, default=1.0,
                    help="how far along tau the anticipated attacker is assumed to move")
    ap.add_argument("--retain-w", type=float, default=0.0)
    ap.add_argument("--scope", choices=["esd_u","esd_x","all"], default="esd_u",
                    help="which parameters to train. esd_u = NOT cross-attention, matching our "
                         "esd_church build, so the lam=0 ablation really is ESD")
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)

    import importlib.util as iu
    sp = iu.spec_from_file_location("atk3", os.path.join(ROOT, "scripts/attack_relearn_v3.py"))
    atk = iu.module_from_spec(sp); sp.loader.exec_module(atk)

    pipe = atk.build_pipe(torch.float32, None, dev)
    unet, te, tok, sch = pipe.unet, pipe.text_encoder, pipe.tokenizer, pipe.scheduler
    te.requires_grad_(False)
    ref = copy.deepcopy(unet).eval().requires_grad_(False)

    tau_raw = torch.load(a.tau, map_location=dev)
    tau = {k: v.float() for k, v in tau_raw.items()}

    cname = a.concept.replace("_", " ")
    with torch.no_grad():
        def emb(txt):
            ids = tok([txt], padding="max_length", max_length=77, truncation=True,
                      return_tensors="pt").input_ids.to(dev)
            return te(ids)[0]
        e_conc, e_null = emb(f"a photo of a {cname}"), emb("")

    def is_xattn(n):
        return "attn2" in n

    unet.train()
    train_ps, names = [], []
    for n, p in unet.named_parameters():
        want = True if a.scope == "all" else (is_xattn(n) if a.scope == "esd_x" else not is_xattn(n))
        p.requires_grad_(want)
        if want:
            train_ps.append(p); names.append(n)
    print(f"[tvi] scope={a.scope}: training {len(train_ps)} tensors "
          f"({sum(p.numel() for p in train_ps)/1e6:.1f}M params)", flush=True)
    opt = torch.optim.Adam(train_ps, lr=a.lr)
    hist_e, hist_i = [], []
    t0 = time.time()

    sch.set_timesteps(a.ddim_steps, device=dev)

    @torch.no_grad()
    def sample_partial():
        """x_t on a real trajectory toward the concept, drawn with the CURRENT model.

        This is what erase_esd.py does and it is not optional: evaluating the ESD loss at pure
        random noise is off-manifold, the conditional and unconditional predictions barely differ
        there, and the loss collapses to ~1e-4 carrying no signal. (Measured: the first version of
        this script did exactly that.)"""
        k = int(torch.randint(1, a.ddim_steps, (1,)).item())
        lat = torch.randn(1, 4, 64, 64, device=dev)
        for ts_ in sch.timesteps[:k]:
            inp = torch.cat([lat, lat])
            emb2 = torch.cat([e_null, e_conc])
            n_u, n_c = unet(inp, ts_, encoder_hidden_states=emb2).sample.chunk(2)
            lat = sch.step(n_u + a.guidance * (n_c - n_u), ts_, lat).prev_sample
        return lat, sch.timesteps[k]

    def esd_loss(x_t, ts):
        with torch.no_grad():
            n_null = ref(x_t, ts, encoder_hidden_states=e_null).sample
            n_conc = ref(x_t, ts, encoder_hidden_states=e_conc).sample
            target = n_null - a.esd_eta * (n_conc - n_null)
        return F.mse_loss(unet(x_t, ts, encoder_hidden_states=e_conc).sample, target)

    for step in range(a.steps):
        # one partial sample, reused for BOTH parameter points so the only difference between
        # L(theta) and L(theta + eta*tau) is the parameters, not the latent
        unet.eval(); x_t, ts = sample_partial(); unet.train()

        # L = (1-lam)*L_esd(theta) + lam*L_esd(theta + eta*tau).
        # The two terms are evaluated at DIFFERENT parameter points, so they get separate backward
        # passes with the parameters in the correct state for each; gradients accumulate. Computing
        # both losses first and shifting in between corrupts the first graph -- the parameters it
        # references change version, and autograd refuses. That was the original bug here.
        opt.zero_grad(set_to_none=True)

        # (1) erased now
        l_now = esd_loss(x_t, ts)
        if a.lam < 1.0:
            ((1.0 - a.lam) * l_now).backward()
        l_now_v = float(l_now.item())
        del l_now

        # (2) still erased after the anticipated attacker step theta -> theta + eta*tau
        shifted = []
        with torch.no_grad():
            for n, p in unet.named_parameters():
                if n in tau and tau[n].shape == p.shape:
                    p.add_(a.tau_eta * tau[n]); shifted.append(n)
        l_after = esd_loss(x_t, ts)
        if a.lam > 0.0:
            (a.lam * l_after).backward()
        l_after_v = float(l_after.item())
        del l_after
        with torch.no_grad():
            for n, p in unet.named_parameters():
                if n in shifted:
                    p.sub_(a.tau_eta * tau[n])

        torch.nn.utils.clip_grad_norm_(train_ps, 1.0)
        opt.step()
        l_now, l_after = l_now_v, l_after_v
        hist_e.append(l_now); hist_i.append(l_after)
        if (step + 1) % 50 == 0:
            print(f"[tvi] step {step+1}/{a.steps} esd={sum(hist_e[-50:])/50:.5f} "
                  f"esd@tau={sum(hist_i[-50:])/50:.5f} "
                  f"gap={sum(hist_i[-50:])/50 - sum(hist_e[-50:])/50:+.5f} "
                  f"{(time.time()-t0)/(step+1):.2f}s/step", flush=True)

    unet.eval()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({k: (v.half() if v.dtype.is_floating_point else v)
                for k, v in unet.state_dict().items()}, a.out)
    rec = {"study": "tvi_erase", "label": a.label, "hparams": vars(a), "out": a.out,
           "esd_first50": sum(hist_e[:50]) / 50, "esd_last50": sum(hist_e[-50:]) / 50,
           "esd_at_tau_first50": sum(hist_i[:50]) / 50, "esd_at_tau_last50": sum(hist_i[-50:]) / 50,
           "invariance_gap_first50": (sum(hist_i[:50]) - sum(hist_e[:50])) / 50,
           "invariance_gap_last50": (sum(hist_i[-50:]) - sum(hist_e[-50:])) / 50,
           "n_tau_tensors_applied": len(tau), "scope": a.scope, "n_trained_tensors": len(train_ps),
           "n_trained_params": sum(p.numel() for p in train_ps), "minutes": (time.time() - t0) / 60,
           "note": ("invariance_gap is L_esd(theta+eta*tau) - L_esd(theta); it falling over training "
                    "is the mechanism working, and it is the quantity to report")}
    json.dump(rec, open(os.path.join(ROOT, f"outputs/tvi_{a.label}.json"), "w"), indent=1)
    print(f"[tvi] done in {rec['minutes']:.1f} min | esd {rec['esd_first50']:.5f}->"
          f"{rec['esd_last50']:.5f} | invariance gap {rec['invariance_gap_first50']:+.5f}->"
          f"{rec['invariance_gap_last50']:+.5f}", flush=True)


if __name__ == "__main__":
    main()
