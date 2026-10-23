"""PREREG_H66 S5a diagnostic: did the attack REFILL the pruned entries?

Only meaningful if the attack reversed the arm. It separates two explanations of a recovered
concept:

  (a) the pruning mask was an INITIAL CONDITION -- the adversary simply re-learned the zeroed
      weights, so the pruned entries are nonzero again with magnitudes comparable to their
      surviving neighbours;
  (b) the concept returned by ANOTHER ROUTE -- the pruned entries stayed near zero and the recovery
      came from elsewhere in the network.

Because attack_relearn.py deletes its attacked checkpoint after measuring it (the provenance gap
FINDING40 documents), this must be run on a checkpoint that still exists: either re-run one attack
with --keep-checkpoint, or point --attacked at any surviving attacked UNet.

Run: python scripts/frp_mask_refill.py --arm frp_church_r3_l05 --attacked <path/to/unet.pt>
"""
import argparse, json, os
import torch

OUT = "/data/bijaypandey/t2i_unlearn/outputs"
ADAPT = "/data/bijaypandey/t2i_unlearn/adapters"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="the FRP arm label (its unet.pt is the pruned one)")
    ap.add_argument("--attacked", required=True, help="path to the ATTACKED unet.pt")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    pre = torch.load(f"{ADAPT}/{a.arm}/unet.pt", map_location="cpu")
    pre = pre.get("unet", pre)
    post = torch.load(a.attacked, map_location="cpu")
    post = post.get("unet", post)

    # FRP's scope: the feed-forward linears it prunes
    keys = sorted(k for k in pre if k.endswith(".weight") and ".ff.net." in k
                  and ".transformer_blocks." in k)
    assert keys, "no FRP scope keys found"

    n_pruned = n_refilled = n_surv = 0
    mag_refilled, mag_surviving = [], []
    # DISCRIMINATING statistics. "fraction nonzero" is near-trivial: under Adam almost every
    # parameter receives a nonzero update, so 99.99% refilled says nothing on its own. What
    # separates "the mask was the route back" from "the mask was incidental" is whether the
    # attack's update is CONCENTRATED on the pruned positions relative to their share of entries.
    dp2 = ds2 = wp2 = ws2 = 0.0
    per_layer = []
    for k in keys:
        w0, w1 = pre[k].float(), post[k].float()
        mask = (w0 == 0)                    # the pruned entries
        if mask.sum() == 0:
            continue
        surv = ~mask
        refilled = mask & (w1 != 0)
        n_pruned += int(mask.sum())
        n_refilled += int(refilled.sum())
        if refilled.sum() > 0:
            mag_refilled.append(float(w1[refilled].abs().mean()))
        if surv.sum() > 0:
            mag_surviving.append(float(w1[surv].abs().mean()))
        dp2 += float((w1[mask] ** 2).sum())          # w0 == 0 there, so this IS the delta
        ds2 += float(((w1[surv] - w0[surv]) ** 2).sum())
        wp2 += float((w1[mask] ** 2).sum())
        ws2 += float((w1[surv] ** 2).sum())
        n_surv += int(surv.sum())
        per_layer.append({
            "layer": k, "n_pruned": int(mask.sum()),
            "frac_refilled": float(refilled.float().sum() / mask.sum()),
            "mean_abs_refilled": (float(w1[refilled].abs().mean()) if refilled.sum() else 0.0),
            "mean_abs_surviving": float(w1[surv].abs().mean()) if surv.sum() else 0.0})

    mr = sum(mag_refilled) / max(len(mag_refilled), 1)
    ms = sum(mag_surviving) / max(len(mag_surviving), 1)
    tot = dp2 + ds2
    share_update = dp2 / tot if tot else 0.0
    share_entries = n_pruned / max(n_pruned + n_surv, 1)
    conc = share_update / share_entries if share_entries else None
    rms_p = (dp2 / max(n_pruned, 1)) ** 0.5
    rms_s = (ds2 / max(n_surv, 1)) ** 0.5
    rec = {"study": "frp_mask_refill", "prereg": "PREREG_H66 S5a",
           "arm": a.arm, "attacked": a.attacked,
           "n_scope_layers": len(per_layer),
           "n_pruned_entries": n_pruned, "n_refilled_entries": n_refilled,
           "frac_refilled": n_refilled / max(n_pruned, 1),
           "mean_abs_refilled": mr, "mean_abs_surviving": ms,
           "refilled_over_surviving": mr / ms if ms else None,
           "share_of_update_norm_on_pruned": share_update,
           "share_of_entries_pruned": share_entries,
           "concentration_ratio": conc,
           "per_entry_rms_update_pruned": rms_p,
           "per_entry_rms_update_surviving": rms_s,
           "share_of_post_attack_weight_mass_in_pruned": wp2 / (wp2 + ws2) if (wp2 + ws2) else 0.0,
           "frac_nonzero_caveat": "frac_refilled is NEAR-TRIVIAL: under Adam almost every parameter "
                                  "becomes nonzero, so a value near 1.0 is expected whatever the "
                                  "mechanism. It is reported but is NOT the verdict.",
           "reading": ("CONCENTRATED: the attack put disproportionately more update norm on the "
                       "pruned positions than their share of entries, so refilling the mask is "
                       "plausibly the route back"
                       if conc is not None and conc >= 1.5 else
                       "NOT CONCENTRATED: the attack's update is essentially UNIFORM across pruned "
                       "and surviving positions (concentration ratio ~1), and the post-attack "
                       "weight mass in formerly-pruned positions is negligible. The pruned weights "
                       "were NOT meaningfully re-learned: the concept returned through the "
                       "SURVIVING weights, by another route than refilling the mask."),
           "per_layer": per_layer}
    lab = a.label or a.arm
    json.dump(rec, open(f"{OUT}/frp_refill_{lab}.json", "w"), indent=1)
    print(f"[refill] {a.arm}: {n_refilled}/{n_pruned} pruned entries nonzero after attack "
          f"({100*rec['frac_refilled']:.2f}%)")
    print(f"[refill] mean |w| refilled={mr:.3e} surviving={ms:.3e} "
          f"ratio={rec['refilled_over_surviving']}")
    print(f"[refill] update-norm share on pruned={100*share_update:.3f}% vs entry share="
          f"{100*share_entries:.3f}%  -> concentration ratio {conc:.4f}")
    print(f"[refill] per-entry RMS update: pruned={rms_p:.3e} surviving={rms_s:.3e}")
    print(f"[refill] post-attack weight mass in pruned positions: "
          f"{100*rec['share_of_post_attack_weight_mass_in_pruned']:.5f}%")
    print(f"[refill] {rec['reading']}")
    print(f"[refill] wrote {OUT}/frp_refill_{lab}.json")


if __name__ == "__main__":
    main()
