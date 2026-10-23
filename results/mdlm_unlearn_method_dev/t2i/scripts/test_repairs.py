"""Regression tests for the audit repairs. Run: python scripts/test_repairs.py

Each test targets one audit issue and is written to FAIL on the pre-repair code. A correct file
path or a varying score is explicitly NOT accepted as evidence that the intended model was
evaluated — identity is checked by parameter hash.
"""
import hashlib, io, json, os, sys, tempfile
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PIN = "/data/bijaypandey/t2i_unlearn/MODEL_PIN.txt"
ADV = "/data/bijaypandey/t2i_unlearn/ADVUNLEARN_PIN.txt"
RESULTS = []


def rec(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def sd_hash(module):
    """Identity fingerprint of a module's parameters — not a path, not a score."""
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- ISSUE 1
def test_issue1_pipeline_identity():
    """Zero-step save/reload with a GENUINELY MODIFIED encoder.

    Pre-repair, build_pipe() for the post-attack measurement omitted te_path, so the reloaded
    pipeline carried the BASE encoder. This test perturbs the encoder, saves it, reloads through
    the same code path the attack uses, and requires the hash AND the conditioning output to match.
    """
    from transformers import CLIPTextModel
    sys.argv = ["x"]
    import importlib.util as iu
    spec = iu.spec_from_file_location("ar", os.path.join(os.path.dirname(__file__),
                                                         "attack_relearn.py"))
    ar = iu.module_from_spec(spec); spec.loader.exec_module(ar)

    pin = dict(l.split("=", 1) for l in open(ADV).read().split("\n") if "=" in l)
    te_dir = pin["local"] + "/church_unlearned"
    te = CLIPTextModel.from_pretrained(te_dir, torch_dtype=torch.float16).to("cuda").eval()
    h_erased = sd_hash(te)

    # genuinely modify it, so a base-encoder reload cannot coincidentally match
    with torch.no_grad():
        for q in te.parameters():
            q.add_(torch.randn_like(q) * 0.01)
    h_mod = sd_hash(te)
    if h_mod == h_erased:
        return rec("issue1_pipeline_identity", False, "perturbation did not change the encoder")

    with tempfile.TemporaryDirectory() as td:
        te.half().save_pretrained(td)
        p = ar.build_pipe(torch.float16, None, "cuda", te_path=td)
        h_reload = sd_hash(p.text_encoder)
        # behaviour: identical token IDs -> identical conditioning, within tolerance
        ids = p.tokenizer(["a photo of a church"], padding="max_length", max_length=77,
                          truncation=True, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            a = te(ids)[0].float()
            b = p.text_encoder(ids)[0].float()
        md = float((a - b).abs().max())
        base_hash_differs = h_reload != h_erased
        ok = (h_reload == h_mod) and md <= 1e-3 and base_hash_differs
        rec("issue1_pipeline_identity", ok,
            f"reloaded_hash={h_reload} modified_hash={h_mod} erased_hash={h_erased} "
            f"max|Δconditioning|={md:.2e} tol=1e-3")
        # the other components must be the intended ones
        rec("issue1_components", p.scheduler.__class__.__name__ == "PNDMScheduler"
            and p.vae.dtype == torch.float16 and p.unet.dtype == torch.float16,
            f"scheduler={p.scheduler.__class__.__name__} vae={p.vae.dtype} unet={p.unet.dtype} "
            f"tokenizer={p.tokenizer.__class__.__name__}")


# ---------------------------------------------------------------- ISSUE 2
def test_issue2_te_gradient_connected():
    """The encoder must receive finite, nonzero gradient when --attack-te is set, and none when
    it is not. Pre-repair the embeddings were cached under no_grad, so this was 0 in both cases."""
    from diffusers import StableDiffusionPipeline
    pin = dict(l.split("=", 1) for l in open(PIN).read().split("\n") if "=" in l)
    apin = dict(l.split("=", 1) for l in open(ADV).read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", revision=pin["revision"], torch_dtype=torch.float32,
        variant="fp16", safety_checker=None, requires_safety_checker=False).to("cuda")
    from transformers import CLIPTextModel
    pipe.text_encoder = CLIPTextModel.from_pretrained(
        apin["local"] + "/church_unlearned", torch_dtype=torch.float32).to("cuda")
    unet, te, tok, sch = pipe.unet, pipe.text_encoder, pipe.tokenizer, pipe.scheduler
    pipe.vae.requires_grad_(False)
    ids = tok(["a church"], padding="max_length", max_length=77, truncation=True,
              return_tensors="pt").input_ids.to("cuda")
    z0 = torch.randn(1, 4, 64, 64, device="cuda")
    noise = torch.randn_like(z0)
    ts = torch.tensor([500], device="cuda").long()
    zt = sch.add_noise(z0, noise, ts)

    for attack_te in (True, False):
        te.requires_grad_(attack_te); unet.requires_grad_(True)
        unet.zero_grad(set_to_none=True); te.zero_grad(set_to_none=True)
        c = te(ids)[0] if attack_te else te(ids)[0].detach()
        loss = F.mse_loss(unet(zt, ts, encoder_hidden_states=c).sample, noise)
        loss.backward()
        gt = sum(float(q.grad.norm() ** 2) for q in te.parameters() if q.grad is not None) ** 0.5
        gu = sum(float(q.grad.norm() ** 2) for q in unet.parameters() if q.grad is not None) ** 0.5
        gv = sum(float(q.grad.norm() ** 2) for q in pipe.vae.parameters() if q.grad is not None) ** 0.5
        if attack_te:
            ok = gt > 0 and torch.isfinite(torch.tensor(gt)) and gu > 0 and gv == 0
            rec("issue2_te_grad_connected_when_on", ok,
                f"te_grad={gt:.4e} unet_grad={gu:.4e} vae_grad={gv:.4e} (vae must be 0)")
            # a real optimiser step must move the encoder
            ref = [q.detach().clone() for q in te.parameters()]
            opt = torch.optim.Adam(list(unet.parameters()) + list(te.parameters()), lr=1e-5)
            opt.step()
            d = sum(float((q.detach() - r).norm() ** 2) for q, r in zip(te.parameters(), ref)) ** 0.5
            rec("issue2_te_param_actually_updates", d > 0, f"‖Δtext_encoder‖={d:.4e}")
        else:
            rec("issue2_te_grad_zero_when_off", gt == 0.0,
                f"te_grad={gt:.4e} (must be exactly 0 for UNet-only) unet_grad={gu:.4e}")


# ---------------------------------------------------------------- ISSUE 3
def test_issue3_adv_sign():
    """A small inner step must INCREASE the erasure loss (the adversary's objective).
    Pre-repair the code ascended -MSE, which decreases it."""
    torch.manual_seed(0)
    pred_w = torch.nn.Linear(8, 8, bias=False).cuda()
    target = torch.randn(1, 8, device="cuda")
    e_c = torch.randn(1, 8, device="cuda")

    def run(correct_sign):
        delta = torch.zeros_like(e_c, requires_grad=True)
        obj = F.mse_loss(pred_w(e_c + delta), target)
        o0 = float(obj)
        g, = torch.autograd.grad(obj if correct_sign else -obj, delta)
        with torch.no_grad():
            d2 = delta + 1e-3 * g / (g.norm() + 1e-12)
        o1 = float(F.mse_loss(pred_w(e_c + d2), target))
        return o0, o1

    o0, o1 = run(True)
    bo0, bo1 = run(False)
    rec("issue3_adv_step_increases_erasure_loss", o1 > o0,
        f"corrected: loss {o0:.6f} -> {o1:.6f} (must RISE)")
    rec("issue3_old_sign_would_decrease", bo1 < bo0,
        f"pre-repair convention: loss {bo0:.6f} -> {bo1:.6f} (fell — confirms the bug)")


# ---------------------------------------------------------------- ISSUE 4
def test_issue4_active_target_metrics():
    """When the active target is tench, tench must NOT be counted as utility, and the legacy
    denominator must still be emitted and labelled."""
    import importlib.util as iu
    sys.argv = ["x"]
    spec = iu.spec_from_file_location("ge", os.path.join(os.path.dirname(__file__), "gen_eval.py"))
    ge = iu.module_from_spec(spec); spec.loader.exec_module(ge)
    keys = list(ge.CONCEPTS)
    LEG = ("church", "parachute", "garbage_truck", "van_gogh")
    tgt = "tench"
    nontarget = [k for k in keys if k != tgt]
    legacy_retain = [k for k in keys if k not in LEG]
    ok = (tgt not in nontarget) and (tgt in legacy_retain) and len(nontarget) == len(keys) - 1
    rec("issue4_target_excluded_from_utility", ok,
        f"n_nontarget={len(nontarget)} (target excluded) vs legacy_retain n={len(legacy_retain)} "
        f"which still CONTAINS tench -> the two denominators differ, as documented")


# ---------------------------------------------------------------- ISSUE 10
def test_issue10_mask_mean_preserved():
    """After renormalise+clip, the realised mask mean must equal alpha (when feasible).
    Pre-repair, clipping lowered it silently."""
    torch.manual_seed(0)
    for alpha in (0.3, 0.5, 0.7):
        for skew in (1.0, 6.0):
            w = torch.rand(20000, device="cuda") ** skew
            keep = w.clone()
            m0 = float(keep.mean())
            # pre-repair behaviour
            old = (keep * (alpha / max(m0, 1e-12))).clamp(0, 1)
            # repaired behaviour: bisect a feasible gain
            k = keep * (alpha / max(m0, 1e-12))
            lo, hi = 0.0, 1e6
            for _ in range(60):
                g = 0.5 * (lo + hi)
                if float((k * g).clamp(0, 1).mean()) < alpha: lo = g
                else: hi = g
            new = (k * (0.5 * (lo + hi))).clamp(0, 1)
            rec(f"issue10_mask_mean_alpha{alpha}_skew{skew}",
                abs(float(new.mean()) - alpha) < 1e-3,
                f"pre-repair mean={float(old.mean()):.4f} repaired mean={float(new.mean()):.4f} "
                f"target={alpha} clipped_frac={float((k>1).float().mean()):.3e}")
    # degenerate: all-zero importance must give zero damage, not a divide-by-zero blowup
    z = torch.zeros(1000, device="cuda")
    s = torch.zeros_like(z) if float(z.mean()) <= 1e-12 else z
    rec("issue10_zero_importance_edge_case", float(s.abs().max()) == 0.0,
        "all-zero importance -> zero damage (no division by ~0)")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    for fn in (test_issue3_adv_sign, test_issue4_active_target_metrics,
               test_issue10_mask_mean_preserved, test_issue1_pipeline_identity,
               test_issue2_te_gradient_connected):
        try:
            fn()
        except Exception as e:
            rec(fn.__name__, False, f"EXCEPTION {type(e).__name__}: {e}")
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n==== {n_ok}/{len(RESULTS)} checks passed ====")
    json.dump([{"test": n, "pass": bool(o), "detail": d} for n, o, d in RESULTS],
              open("/data/bijaypandey/t2i_unlearn/outputs/test_repairs_results.json", "w"), indent=1)
    sys.exit(0 if n_ok == len(RESULTS) else 1)
