"""UCE — Unified Concept Editing (Gandikota et al., arXiv 2308.14761). OUR RE-IMPLEMENTATION.

Closed-form edit of the cross-attention K/V projections. For each such matrix W, UCE solves

    W_new = ( sum_{i in E} v_i* c_i^T + sum_{j in P} W_old c_j c_j^T + lam * W_old )
            ( sum_{i in E} c_i  c_i^T + sum_{j in P}        c_j c_j^T + lam * I     )^{-1}

where c are text-embedding directions, E is the erase set, P the preserve set, and
v_i* = W_old c_target is the value the erased concept should instead produce (we map it to the
embedding of a neutral target, " " / "", which is the paper's erasure setting).

Labelled as a re-implementation throughout (PREREG_H38 S4): the original authors' tuning is not
reproduced, so published numbers are not claimed. There are 32 K/V matrices in SD v1.4.

NOTE ON THE CONCEPT DIRECTION. We take the text-encoder hidden states from the concept's final
content token through position 76. NOTE (2026-10-26): this does NOT match UCE's official
implementation, contrary to what this line previously claimed -- upstream
(unified-concept-editing@7c724d9 trainscripts/uce_sd_erase.py:34-42) selects exactly ONE
embedding, at attention_mask.sum()-2. Behaviour here is unchanged so every historical arm
reproduces; the deviation is recorded in UPSTREAM_FIDELITY_MANIFEST.md S7(d). See `concept_dirs` for
why the single-token version measured earlier does not erase at all. This choice is held IDENTICAL
across every arm and baseline so comparisons are internally consistent.
"""
import argparse, json, os
import torch

# CROSS-MODEL OVERRIDE (PREREG_H85 / FINDING64). Defaults reproduce the pinned SD-1.4 exactly when
# BASE_MODEL/BASE_REV are unset, so no existing result changes.
import os as _os
BASE_MODEL = _os.environ.get("BASE_MODEL", "CompVis/stable-diffusion-v1-4")
BASE_REV_ENV = _os.environ.get("BASE_REV")
BASE_VARIANT = _os.environ.get("BASE_VARIANT", "fp16") or None


PIN = "/data/bijaypandey/t2i_unlearn/MODEL_PIN.txt"

FORGET_PROMPTS = {
    "church": "church", "parachute": "parachute", "garbage_truck": "garbage truck",
    "van_gogh": "Van Gogh",
}
# preserved concepts: the retain neighbours the utility axis is measured on
PRESERVE = ["cassette player", "chain saw", "english springer spaniel", "french horn",
            "gas pump", "golf ball", "tench", "Monet", "a photo", "a painting"]


@torch.no_grad()
def concept_dirs(pipe, texts, device, paired=None):
    """Hidden states from each prompt's final CONTENT token through position 76.

    CORRECTED. A first version used only the single final-content-token hidden state; UCE then
    changed the weights by just 5.7% and did not erase at all (church stayed at CLIP acc 1.000).

    THIS DOES *NOT* MATCH UCE'S OFFICIAL IMPLEMENTATION (corrected 2026-10-26, AGENT1_UCE_FINDINGS
    P1). The previous version of this docstring asserted that it did; that sentence was false and
    is deleted. Upstream (unified-concept-editing@7c724d9 trainscripts/uce_sd_erase.py:34-42)
    selects exactly ONE embedding, at attention_mask.sum()-2. Our `last..76` range is a DELIBERATE
    METHOD DEVIATION, not a transcription of upstream.

    The deviation is justified by measurement, and the justification has since been replicated
    (AGENT1_UCE_FINDINGS D-104): running the pinned upstream UCE itself produces a 5.36% weight
    change and leaves church at 15/15 CLIP accuracy -- the same null our own single-token version
    produced at 5.7%. Two independent code paths, one result.

    The mechanism originally given here ("SD cross-attention attends over ALL 77 positions, CLIP is
    CAUSAL, so editing one position leaves the concept reachable") is ALSO wrong and is withdrawn:
    AGENT1_UCE_FINDINGS D-101 measures upstream's single-token edit changing the cross-attention
    K/V OUTPUT by 42% at the padding positions, because a rank-1 weight update acts on every
    position's embedding, not only the selected one. Why the 76-row edit erases and the 1-row edit
    does not is therefore an OPEN question, not an explained one. What is measured: our edit is
    ~4x larger in weight space (0.2164 vs 0.0536) and 1.35x larger in activation space.

    Returns (sum_positions, 768). With `paired`, returns the SAME positions of the paired prompts
    so the erase direction and its target value are position-aligned.
    """
    tok, te = pipe.tokenizer, pipe.text_encoder
    out, out_p = [], []
    for k, t in enumerate(texts):
        ti = tok([t], padding="max_length", max_length=tok.model_max_length,
                 truncation=True, return_tensors="pt").to(device)
        h = te(ti.input_ids)[0]
        last = max(int(ti.attention_mask[0].sum().item()) - 2, 0)
        idx = torch.arange(last, h.shape[1], device=device)
        out.append(h[0, idx].float())
        if paired is not None:
            tp = tok([paired[k]], padding="max_length", max_length=tok.model_max_length,
                     truncation=True, return_tensors="pt").to(device)
            hp = te(tp.input_ids)[0]
            out_p.append(hp[0, idx].float())          # SAME positions
    C = torch.cat(out)
    return (C, torch.cat(out_p)) if paired is not None else C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", required=True, choices=sorted(FORGET_PROMPTS))
    ap.add_argument("--target", default="", help="what the erased concept should map to")
    ap.add_argument("--lam", type=float, default=0.1,
                    help="regularisation toward W_old. WARNING (ERRATUM_uce_lambda_scale.md): this "
                         "is added to c^T c, whose max eigenvalue is LARGE for SD's 768-dim CLIP "
                         "embeddings, so values of order 1 are INERT -- they change nothing. The "
                         "usable range starts around 1e3-1e4. The 0.1 default is kept only so every "
                         "previously published arm reproduces exactly. MEASURED 2026-10-26 on CPU "
                         "for concept 'church' + the 10 PRESERVE prompts at this selection "
                         "(AGENT1_UCE_FINDINGS S9b): c^T c is 826x768 with rank 136, max "
                         "eigenvalue 3.24e5, and cond(c^T c + lam I) = 3.24e6 at lam=0.1, "
                         "6.48e5 at lam=0.5, 3.25e2 at lam=1000. The '~8.6e4' figure quoted in "
                         "ERRATUM_uce_lambda_scale.md is 3.8x smaller than this measurement and "
                         "needs reconciliation by that document's owner; the INERTNESS conclusion "
                         "is unaffected and if anything strengthened.")
    ap.add_argument("--preserve-scale", type=float, default=0.1)
    ap.add_argument("--no-preserve", action="store_true",
                    help="empty the PRESERVE set, matching upstream UCE's documented default "
                         "preserve_concepts=[]. OPT-IN: omitting it reproduces every historical "
                         "arm bit-identically (audit defect D-026).")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag-order-probe", action="store_true",
                    help="DIAGNOSTIC, DEFAULT OFF. Adds measurements ONLY; the edit written to "
                         "disk is bit-for-bit identical either way, so every historical arm "
                         "reproduces. When set, each matrix is ALSO solved (a) in upstream's "
                         "accumulation order (lam*W first, per-concept rank-1 accumulation, as "
                         "uce_sd_erase.py:61-79 does) and (b) entirely in float64, and the "
                         "relative deltas are reported. Rationale: AGENT1_UCE_FINDINGS D-160 -- at "
                         "lam=0.1 cond(c^T c + lam I) is ~3.2e6, so every O(eps) implementation "
                         "detail moves W_new by ~5%% of its norm on CPU while the GPU path differs "
                         "by only ~1e-4. This flag measures it on the GPU that actually ran the "
                         "arms, which no existing artifact does.")
    a = ap.parse_args()
    dev = "cuda"
    from diffusers import StableDiffusionPipeline
    pin = dict(l.split("=", 1) for l in open(PIN).read().split("\n") if "=" in l)
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, revision=(BASE_REV_ENV or pin["revision"]), torch_dtype=torch.float32,
        variant=BASE_VARIANT, safety_checker=None, requires_safety_checker=False).to(dev)

    c_e, c_t = concept_dirs(pipe, [FORGET_PROMPTS[a.concept]], dev, paired=[a.target])
    # OPT-IN, DEFAULT OFF (added 2026-10-26, audit defect D-026). Upstream UCE's documented default
    # is preserve_concepts=[] (uce_sd_erase.py:148-149), but this port had no way to express that,
    # so every R4c/R4d comparison ran upstream with 0 preserve concepts against the port with 10 --
    # an asymmetry that could be recorded but not removed. With --no-preserve the preserve set is an
    # EMPTY [0, 768] tensor, which makes `c_p.T @ c_p` a 768x768 zero matrix and `(W @ c_p.T) @ c_p`
    # a zero matrix of W's shape, so both preserve terms drop out of the numerator and denominator
    # exactly as they are absent upstream. No special-casing of the algebra is needed.
    #
    # The DEFAULT PATH IS UNCHANGED: without the flag this is the same call with the same PRESERVE
    # list, so every historical arm reproduces bit-identically. The flag exists only so a genuinely
    # matched upstream-vs-port comparison becomes possible.
    if a.no_preserve:
        c_p = c_e.new_zeros((0, c_e.shape[1]))
        print("[uce] --no-preserve: preserve set EMPTY (0 directions), matching upstream's "
              "documented default. Both preserve terms drop out.", flush=True)
    else:
        c_p = concept_dirs(pipe, PRESERVE, dev)
    print(f"[uce] erase directions: {tuple(c_e.shape)}  preserve directions: {tuple(c_p.shape)}",
          flush=True)

    mods = [(n, m) for n, m in pipe.unet.named_modules()
            if n.endswith("attn2.to_k") or n.endswith("attn2.to_v")]
    print(f"[uce] editing {len(mods)} cross-attention K/V matrices for '{a.concept}' -> '{a.target}'",
          flush=True)
    d = c_e.shape[1]
    I = torch.eye(d, device=dev)
    tot = 0.0
    diag = {"order_rel": [], "float64_rel": []} if a.diag_order_probe else None
    for n, m in mods:
        W = m.weight.data.float()                     # (out, 768)
        V_star = (W @ c_t.T).T                        # (n_e, out) position-aligned target values
        # sum_i v_i c_i^T  and  sum_i c_i c_i^T, over every erase position
        num = V_star.T @ c_e + a.preserve_scale * (W @ c_p.T) @ c_p + a.lam * W
        den = c_e.T @ c_e + a.preserve_scale * (c_p.T @ c_p) + a.lam * I
        W_new = num @ torch.linalg.inv(den)
        tot += (W_new - W).norm().item() / max(W.norm().item(), 1e-9)

        # ---- DIAGNOSTIC ONLY. Nothing below this point feeds `m.weight.data`. ----
        if diag is not None:
            # (a) upstream's accumulation order: lam*W FIRST, then per-concept rank-1 updates
            m1 = a.lam * W
            m2 = a.lam * I
            for i in range(c_e.shape[0]):
                ci = c_e[i:i + 1].T
                m1 = m1 + (W @ c_t[i:i + 1].T) @ ci.T
                m2 = m2 + ci @ ci.T
            for j in range(c_p.shape[0]):
                cj = c_p[j:j + 1].T
                m1 = m1 + a.preserve_scale * ((W @ cj) @ cj.T)
                m2 = m2 + a.preserve_scale * (cj @ cj.T)
            W_up_order = m1 @ torch.linalg.inv(m2)
            diag["order_rel"].append(
                float((W_up_order - W_new).norm() / max(float(W_new.norm()), 1e-30)))
            # (b) the same formula, this order, entirely in float64 (true high-precision ref)
            Wd, c_ed, c_td, c_pd = W.double(), c_e.double(), c_t.double(), c_p.double()
            numd = (Wd @ c_td.T) @ c_ed + a.preserve_scale * (Wd @ c_pd.T) @ c_pd + a.lam * Wd
            dend = (c_ed.T @ c_ed + a.preserve_scale * (c_pd.T @ c_pd)
                    + a.lam * torch.eye(d, device=dev, dtype=torch.float64))
            W64 = numd @ torch.linalg.inv(dend)
            diag["float64_rel"].append(
                float((W_new.double() - W64).norm() / max(float(W64.norm()), 1e-30)))

        m.weight.data = W_new.to(m.weight.dtype)
    print(f"[uce] mean relative weight change: {tot/len(mods):.4f}", flush=True)
    if diag is not None:
        diag = {k: {"per_matrix_max": max(v), "per_matrix_mean": sum(v) / len(v), "n": len(v)}
                for k, v in diag.items()}
        print(f"[uce][diag] lam-last vs upstream's lam-first order: "
              f"max={diag['order_rel']['per_matrix_max']:.4e} "
              f"mean={diag['order_rel']['per_matrix_mean']:.4e}", flush=True)
        print(f"[uce][diag] this fp32 solve vs a true float64 solve: "
              f"max={diag['float64_rel']['per_matrix_max']:.4e} "
              f"mean={diag['float64_rel']['per_matrix_mean']:.4e}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v.half() for k, v in pipe.unet.state_dict().items()}, f"{a.out}/unet.pt")
    rec = {"study": "t2i_uce_reimpl", "label": a.label, "concept": a.concept,
           "hparams": vars(a), "n_matrices_edited": len(mods),
           "mean_rel_weight_change": tot / len(mods),
           "base_model": BASE_MODEL, "base_revision": (BASE_REV_ENV or pin["revision"]),
           "base_variant": BASE_VARIANT,
           "embedding_selection": "positions [attention_mask.sum()-2 .. 76] -- a DELIBERATE "
                                  "DEVIATION from upstream, which takes ONE embedding at "
                                  "attention_mask.sum()-2 (uce_sd_erase.py:34-42). See "
                                  "docs/AGENT1_UCE_FINDINGS.md D-104.",
           "erase_scale": "not implemented; equivalent to upstream's default of 1.0",
           "note": "OUR RE-IMPLEMENTATION of arXiv 2308.14761; original tuning not reproduced"}
    if diag is not None:
        rec["diag_order_probe"] = diag
    json.dump(rec, open(f"/data/bijaypandey/t2i_unlearn/outputs/erase_{a.label}.json", "w"),
              indent=1)
    print(f"[uce] saved {a.out}/unet.pt", flush=True)


if __name__ == "__main__":
    main()
