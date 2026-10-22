"""Teacher-forced marginal knowledge probe.

WHY THIS EXISTS. The closed-pool ranking probe (`rank_probe_small.py`) scores each candidate by
summing log-probs over an EOS-padded length-32 sequence. Padding positions therefore contribute to
every score, and the measure is both compressed and mis-ordered:

    theta_ref  ranking retain 0.6725   but marginal retain accuracy 0.9984
    PUA lam=1  ranking retain 0.9000   but marginal retain accuracy 0.9736

i.e. the ranking probe read PUA as far better than theta_ref on retain when it is in fact slightly
worse, and read theta_ref -- which knows retain essentially perfectly -- at 0.67. That inversion
was caught by cross-checking against generation (FINDING29) and it invalidates the ranking probe as
the x-axis of FINDING28.

This probe instead masks the value span in place, keeps the true left and right context, and reads
the model's marginal probability of the TRUE value tokens. No padding enters the measurement, the
dynamic range is wide (theta_ref 0.9986 vs oracle 0.5382 on forget), and it is the quantity an
attacker would actually need to resurface.

It is NOT a replacement for the behavioural eval, and it is not adversarial. It is one axis.
"""
import argparse, collections, json, sys
import numpy as np, torch

P = "/data/bijaypandey/mdlm_unlearn"
DATA = f"{P}/data/small"
sys.path.insert(0, f"{P}/scripts")
from train_small_mdlm import SmallMDLM, MASK_ID   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    dev = "cuda"
    m = SmallMDLM().to(dev).eval()
    sd = torch.load(a.ckpt, map_location="cpu")
    m.load_state_dict(sd.get("model", sd))

    meta = json.load(open(f"{DATA}/facts_meta.json"))
    attr_of = {}
    for pr in meta["probes"]:
        attr_of[tuple(pr["val_ids"])] = pr["attr"]

    out = {"study": "h18_marginal_probe", "label": a.label, "ckpt": a.ckpt, "results": {}}
    for split in ("forget", "retain"):
        d = np.load(f"{DATA}/unl_{split}.npz")
        n = min(a.n, len(d["ids"]))
        ids = torch.from_numpy(d["ids"][:n].astype(np.int64)).to(dev)
        vs, ve = d["vstart"][:n], d["vend"][:n]
        st, sel = ids.clone(), torch.zeros_like(ids, dtype=torch.bool)
        for i in range(n):
            st[i, vs[i]:ve[i]] = MASK_ID
            sel[i, vs[i]:ve[i]] = True
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pr = torch.softmax(m(st).float(), -1)
        pt = pr.gather(-1, ids[..., None]).squeeze(-1)
        # per-example mean over its own value positions, then mean over examples, so a long
        # value span does not outweigh a short one
        per = [(pt[i][sel[i]].mean().item()) for i in range(n)]
        # exact-span accuracy: argmax correct at EVERY value position of the example
        am = pr.argmax(-1)
        exact = [bool(((am[i] == ids[i])[sel[i]]).all().item()) for i in range(n)]
        out["results"][split] = {"n": n, "p_true": float(np.mean(per)),
                                 "exact_span_acc": float(np.mean(exact))}
        by = collections.defaultdict(list)
        for i in range(n):
            by[attr_of.get(tuple(int(x) for x in ids[i, vs[i]:ve[i]]), "?")].append(per[i])
        for k, v in sorted(by.items()):
            out["results"][f"{split}/{k}"] = {"n": len(v), "p_true": float(np.mean(v))}
    json.dump(out, open(f"{P}/outputs/marg_{a.label}.json", "w"), indent=1)
    r = out["results"]
    print(f"[marg] {a.label}: forget p_true={r['forget']['p_true']:.4f} "
          f"exact={r['forget']['exact_span_acc']:.4f} | "
          f"retain p_true={r['retain']['p_true']:.4f} exact={r['retain']['exact_span_acc']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
