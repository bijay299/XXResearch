"""Adversarial concept erasure, in the SPIRIT of the Receler / AdvUnlearn family. OUR OWN DESIGN.

CLASSIFICATION CORRECTION (Agent-3 fidelity audit, 2026-10-26). "Re-implementation" overstates
what this is. Neither cited method is reimplemented: Receler's mechanism is a lightweight ERASER
ADAPTER with concept-localised regularisation, and AdvUnlearn's is adversarial training of the TEXT
ENCODER with a utility-retaining prompt set. This file does neither. It is an ESD step hardened
against a soft adversarial perturbation of the concept EMBEDDING -- closer in form to the
UnlearnDiffAtk / P4D prompt-attack literature than to either cited defence. Correct classification:
OUR OWN DESIGN, inspired by that family. Consequence: a reversal measured on THIS file is NOT
evidence that Receler or AdvUnlearn fall, and PREREG_H38 S4's framing of it as "the real bar" does
not hold. The pinned OPTML-Group/AdvUnlearn checkpoint in ADVUNLEARN_PIN.txt is the only artifact
in this project that bears on the published AdvUnlearn.

HISTORICAL-RESULT WARNING. The inner loop's sign was WRONG until the "ISSUE 3 REPAIR" below: it
ascended NEGATIVE MSE, i.e. it DESCENDED the erasure loss, so the inner loop was making erasure
EASIER and the arm was not adversarially trained at all. That repair CHANGED the numerics of this
file. Artifacts written before it are therefore not reproducible from this code, and the output
JSON carried no version field to tell the two apart. `IMPL_VERSION` below is written into every new
artifact so the ambiguity cannot recur; any `outputs/erase_*.json` from this script WITHOUT an
`implementation_version` key predates the sign repair and its adversarial-training claim is void.

PREREG_H38 S4 called this family "the real bar": ESD and UCE are the standard baselines, but the
methods designed for ROBUSTNESS are the adversarially-trained ones (Receler arXiv 2311.17717,
AdvUnlearn arXiv 2405.15234). Both of ours fell to a 150-step / 16-image relearning attack, so the
honest test of PREREG_H40's wall is whether a method built to resist attack also falls.

THE CONSTRUCTION. ESD's loss erases the concept at one fixed text embedding -- the prompt
"church". An attacker just has to find a nearby embedding that still works. Adversarial erasure
closes that hole by alternating:

  INNER  optimise a soft adversarial embedding delta to MAXIMISE concept presence, i.e. to make
         the partially-erased model still produce the concept (the attacker's move)
  OUTER  take an ESD step that erases at the adversarial embedding as well as the clean one
         (the defender's move)

so the erasure must hold over a neighbourhood of embeddings rather than a single point.

This is a faithful-in-spirit, cheaper re-implementation: the published methods use eraser adapters
(Receler) or text-encoder adversarial training over many steps (AdvUnlearn). Labelled as a
re-implementation throughout; published numbers are not claimed. If it still falls to the attack,
that is evidence for H40; if it holds at high utility, H40 is REFUTED and this is the method the
mandate asks for.
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

PIN = "/data/bijaypandey/t2i_unlearn/MODEL_PIN.txt"
# Bumped when the numerics change. "adv/v2-inner-sign-repaired" is the code AFTER the ISSUE 3 sign
# repair described in the module docstring. Artifacts lacking this key predate it.
IMPL_VERSION = "adv/v2-inner-sign-repaired"
PROMPTS = {"church": "church", "parachute": "parachute", "garbage_truck": "garbage truck",
           "van_gogh": "Van Gogh"}
# concepts whose generation must survive; the erasure is penalised for drifting on them
PRESERVE = ["a photo of a tench", "a photo of a french horn", "a photo of a gas pump",
            "a photo of a golf ball", "a photo of a cassette player", "a photo of a chain saw",
            "a painting in the style of Monet", "a landscape photograph"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, choices=sorted(PROMPTS))
    ap.add_argument("--mode", choices=["esd_u", "esd_x"], default="esd_x",
                    help="esd_x (cross-attention only) is Receler's locality-preserving setting")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adv-steps", type=int, default=3, help="inner adversarial steps")
    ap.add_argument("--adv-eps", type=float, default=0.05,
                    help="adversarial budget RELATIVE to ||e_c||. An absolute L2 ball is "
                         "meaningless here: the clean embedding's norm is in the hundreds, so a "
                         "fixed 0.5 ball is negligible and the inner loop is not adversarial at "
                         "all. A weak adversary makes the defence look robust for the wrong "
                         "reason, which is the failure mode this project has hit three times.")
    ap.add_argument("--lam-preserve", type=float, default=1.0)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=3.0)
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

    is_x = lambda n: "attn2" in n
    train_ps = []
    for n, p in unet.named_parameters():
        want = is_x(n) if a.mode == "esd_x" else (not is_x(n))
        p.requires_grad_(want)
        if want:
            train_ps.append(p)
    print(f"[adv] mode={a.mode}: {sum(p.numel() for p in train_ps)/1e6:.1f}M trainable params",
          flush=True)
    opt = torch.optim.Adam(train_ps, lr=a.lr)

    @torch.no_grad()
    def embed(t):
        ti = tok([t], padding="max_length", max_length=77, truncation=True,
                 return_tensors="pt").to(dev)
        return te(ti.input_ids)[0]

    e_c, e_null = embed(PROMPTS[a.concept]), embed("")
    e_pres = [embed(p) for p in PRESERVE]
    t0, hist = time.time(), []
    for step in range(a.steps):
        k = int(torch.randint(1, a.ddim_steps, (1,)).item())
        with torch.no_grad():
            lat = torch.randn(1, 4, 64, 64, device=dev)
            for ts in sched.timesteps[:k]:
                inp = torch.cat([lat, lat]); emb = torch.cat([e_null, e_c])
                n_u, n_c = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
                lat = sched.step(n_u + a.guidance * (n_c - n_u), ts, lat).prev_sample
            t_cur = sched.timesteps[k]
            n_null = frozen(lat, t_cur, encoder_hidden_states=e_null).sample
            n_conc = frozen(lat, t_cur, encoder_hidden_states=e_c).sample
            target = n_null - a.eta * (n_conc - n_null)   # the ESD erasure target

        # INNER: the attacker's move -- find delta inside an L2 ball that makes the CURRENT model
        # still produce the concept, i.e. maximise distance from the erasure target.
        eps = a.adv_eps * e_c.norm()                 # budget scaled to the embedding itself
        step_sz = 2.0 * eps / max(a.adv_steps, 1)    # large enough to reach the ball boundary
        delta = torch.zeros_like(e_c, requires_grad=True)
        for _ in range(a.adv_steps):
            pred = unet(lat, t_cur, encoder_hidden_states=e_c + delta).sample
            # ISSUE 3 REPAIR -- SIGN BUG. This previously read
            #     adv_obj = -F.mse_loss(pred, target);  delta += step * grad(adv_obj)
            # i.e. ASCENT on NEGATIVE MSE, which is DESCENT on the erasure loss: the inner loop was
            # making the erasure EASIER, the opposite of the stated objective. The adversary wants
            # delta that MAXIMISES the erasure loss (keeps the concept producible), so ascend MSE.
            adv_obj = F.mse_loss(pred, target)           # the erasure loss, to be MAXIMISED
            g, = torch.autograd.grad(adv_obj, delta)
            with torch.no_grad():
                delta += step_sz * g / (g.norm() + 1e-12)   # ascent on adv_obj
                nrm = delta.norm()
                if nrm > eps:
                    delta *= eps / nrm
        delta = delta.detach()

        # OUTER: erase at BOTH the clean and the adversarial embedding, and hold the preserved
        # concepts in place so the erasure cannot buy robustness by breaking the model.
        pred_c = unet(lat, t_cur, encoder_hidden_states=e_c).sample
        pred_a = unet(lat, t_cur, encoder_hidden_states=e_c + delta).sample
        l_erase = 0.5 * (F.mse_loss(pred_c, target) + F.mse_loss(pred_a, target))
        ep = e_pres[step % len(e_pres)]
        with torch.no_grad():
            ref_p = frozen(lat, t_cur, encoder_hidden_states=ep).sample
        l_pres = F.mse_loss(unet(lat, t_cur, encoder_hidden_states=ep).sample, ref_p)
        loss = l_erase + a.lam_preserve * l_pres
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_ps, 1.0)
        opt.step()
        hist.append((l_erase.item(), l_pres.item()))
        if step % a.log_every == 0 or step == a.steps - 1:
            print(f"[adv] step {step:4d} erase={l_erase.item():.5f} pres={l_pres.item():.5f} "
                  f"|delta|/|e_c|={delta.norm().item()/e_c.norm().item():.4f} "
                  f"{(time.time()-t0)/max(step+1,1):.2f}s/step",
                  flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in unet.state_dict().items()}, f"{a.out}/unet.pt")
    json.dump({"study": "t2i_adv_erase_reimpl", "implementation_version": IMPL_VERSION,
               "classification": "OUR OWN DESIGN, inspired by Receler 2311.17717 / AdvUnlearn "
                                 "2405.15234. NOT a re-implementation of either: no eraser "
                                 "adapter (Receler), no text-encoder adversarial training "
                                 "(AdvUnlearn). A reversal here is not evidence about either "
                                 "published method.",
               "label": a.label, "concept": a.concept,
               "hparams": vars(a),
               "erase_first50": sum(x[0] for x in hist[:50]) / max(len(hist[:50]), 1),
               "erase_last50": sum(x[0] for x in hist[-50:]) / max(len(hist[-50:]), 1),
               "pres_last50": sum(x[1] for x in hist[-50:]) / max(len(hist[-50:]), 1),
               "minutes": (time.time() - t0) / 60,
               "note": "OUR RE-IMPLEMENTATION in the spirit of Receler 2311.17717 / AdvUnlearn "
                       "2405.15234; adversarial embedding neighbourhood, not their exact "
                       "adapters or text-encoder training. Published numbers not claimed."},
              open(f"/data/bijaypandey/t2i_unlearn/outputs/erase_{a.label}.json", "w"), indent=1)
    print(f"[adv] saved {a.out}/unet.pt in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
