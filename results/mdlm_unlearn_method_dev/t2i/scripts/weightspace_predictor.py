"""Test the PREDICTIVE claim of 'From Dormant to Deleted' (arXiv 2505.22310) on this phase's arms.

THE PAPER IS NOT PORTABLE AS A DEFENCE HERE, and that is declared up front: it studies
EXAMPLE-LEVEL unlearning in VISION CLASSIFIERS, not concept erasure in diffusion models. Its
headline mechanism, however, is a prediction that transfers as a hypothesis:

    "resistance to relearning attacks can be predicted by weight-space properties, specifically,
     L2-distance and linear mode connectivity between the original and the unlearned model"

This phase holds the richest tamper-resistance dataset in the T2I setting -- ten defences, dozens of
gate-validated attacks, and attacker-COST curves -- so the claim can be tested directly rather than
re-derived. That is a use of our assets the paper itself could not make.

What this computes per arm, against the pinned base model:
  * L2 distance ||theta_arm - theta_base||_2, and the same normalised by ||theta_base||_2;
  * cosine similarity of the flattened weights;
  * the same restricted to the cross-attention surface, since that is where UCE/GIFT/IMMA edit.
Linear mode connectivity needs an evaluation at interpolated points and is NOT computed here; it is
recorded as the obvious follow-up rather than approximated by something cheaper and relabelled.

Output: outputs/weightspace_predictor.json -- the distances only. Correlating them against
max-over-valid H and attacker cost is done separately, so this file stays a pure measurement.
"""
import argparse, glob, json, os
import torch

P = "/data/bijaypandey/t2i_unlearn"


def flat(sd, filt=None):
    ks = sorted(k for k in sd if sd[k].dtype.is_floating_point and (filt is None or filt(k)))
    return torch.cat([sd[k].reshape(-1).float() for k in ks]), ks


def stats64(v, bv):
    """Dot products and norms in FLOAT64, accumulated in chunks.

    An earlier version used F.cosine_similarity on the full 859M-element float32 vectors and
    returned cos = 1.278 -- impossible, and a clear sign the float32 accumulation had lost the
    reduction. Anything above 1 is a numerical artifact, so the accumulation is done explicitly in
    float64 over 8M-element chunks and the result is asserted into range.
    """
    dot = nb = nv = dd = 0.0
    for i in range(0, v.numel(), 8_000_000):
        a = v[i:i + 8_000_000].double()
        b = bv[i:i + 8_000_000].double()
        dot += float((a * b).sum()); nv += float((a * a).sum()); nb += float((b * b).sum())
        dd += float(((a - b) ** 2).sum())
    nv, nb = nv ** 0.5, nb ** 0.5
    cos = dot / max(nv * nb, 1e-30)
    assert -1.0001 <= cos <= 1.0001, f"cosine out of range: {cos}"
    return {"l2": dd ** 0.5, "cos": min(max(cos, -1.0), 1.0), "norm_arm": nv, "norm_base": nb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=f"{P}/adapters/_base_ref/unet.pt")
    ap.add_argument("--out", default=f"{P}/outputs/weightspace_predictor.json")
    a = ap.parse_args()

    bsd = torch.load(a.base, map_location="cpu")
    bsd = bsd.get("unet", bsd)
    bv, _ = flat(bsd)
    bx, _ = flat(bsd, lambda k: ".attn2." in k)
    bn, bxn = float(bv.norm()), float(bx.norm())
    print(f"[ws] base: {bv.numel()/1e6:.1f}M params, ||theta||={bn:.2f}; "
          f"cross-attn {bx.numel()/1e6:.1f}M, ||.||={bxn:.2f}", flush=True)

    rows = []
    for d in sorted(glob.glob(f"{P}/adapters/*/unet.pt")):
        name = os.path.basename(os.path.dirname(d))
        if name == "_base_ref":
            continue
        try:
            sd = torch.load(d, map_location="cpu"); sd = sd.get("unet", sd)
            v, ks = flat(sd)
            if v.numel() != bv.numel():
                rows.append({"arm": name, "skipped": "param count mismatch"}); continue
            x, _ = flat(sd, lambda k: ".attn2." in k)
            st, sx = stats64(v, bv), stats64(x, bx)
            rows.append({
                "arm": name,
                "l2": st["l2"], "l2_rel": st["l2"] / bn, "cos": st["cos"],
                "l2_xattn": sx["l2"], "l2_xattn_rel": sx["l2"] / bxn, "cos_xattn": sx["cos"],
                "frac_of_total_drift_in_xattn": (sx["l2"] ** 2) / max(st["l2"] ** 2, 1e-12),
            })
            print(f"[ws] {name:34s} L2={rows[-1]['l2']:9.3f} rel={rows[-1]['l2_rel']:.5f} "
                  f"cos={rows[-1]['cos']:.6f} xattn_rel={rows[-1]['l2_xattn_rel']:.5f} "
                  f"xattn_share={100*rows[-1]['frac_of_total_drift_in_xattn']:.1f}%", flush=True)
            del sd, v, x, dv, dx
        except Exception as e:
            rows.append({"arm": name, "error": repr(e)[:200]})
            print(f"[ws] {name}: ERROR {e}", flush=True)

    rec = {"study": "weightspace_predictor",
           "tests_claim_of": "arXiv 2505.22310 (From Dormant to Deleted)",
           "domain_caveat": "That paper studies EXAMPLE-LEVEL unlearning in VISION CLASSIFIERS. "
                            "Applying its weight-space predictor to T2I concept erasure is a "
                            "domain transfer and a hypothesis test, NOT a reproduction.",
           "lmc_not_computed": "Linear mode connectivity requires evaluations at interpolated "
                               "weights and is recorded as the obvious follow-up rather than "
                               "approximated by a cheaper proxy and relabelled.",
           "base": a.base, "n_arms": len(rows), "arms": rows}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"[ws] wrote {a.out} ({len(rows)} arms)", flush=True)


if __name__ == "__main__":
    main()
