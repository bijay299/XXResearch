"""
H11-M0a+ — the orthogonal-complement interdiction measurement. CPU only, no GPU, no training.

DERIVED FROM THE CRUX OBSTRUCTION. The designer's Prop 1.2 showed that suppressing an
interdicted token at position i drags the KEPT tokens down:

    Delta z_j(v_j)  ~=  -eta * <a_i, a_j> * <w_{v_j}, delta_i>

and that both factors are MAXIMAL exactly for the tokens H11 promises to keep -- the
activation term because i and j are positions in the same answer, the unembedding term
because the two tokens are co-constituents of one fact. So H11's selectivity points the
wrong way and preservation must be bought, not inherited.

The activation factor <a_i, a_j> is the model's own representation and is not controllable.
But delta_i -- the output-space gradient direction at the interdicted position -- IS. Plain
suppression uses delta_i ~= w_{v_i}. Instead take

    delta_i^perp  =  w_{v_i}  -  Proj_{span{w_{v_j} : j in KEEP}}  w_{v_i}

Then <w_{v_j}, delta_i^perp> = 0 identically for every kept token, so the uncontrollable
activation factor is multiplied by ZERO. First-order drag is eliminated by construction
rather than traded off against a penalty weight.

The cost is that the projection also removes part of the suppression signal. This script
measures that cost exactly, because it depends only on the FROZEN unembedding matrix:

    retained suppression fraction  =  ||delta_i^perp|| / ||w_{v_i}||

If the kept and interdicted unembedding rows are near-orthogonal (expected in 126,464
dimensions), the projection removes almost no suppression power while zeroing all of the
first-order drag. If they are strongly aligned, the projection guts the mechanism and the
idea is dead. **That is the whole question and it needs no GPU.**

PRE-REGISTERED, written before running:
  VIABLE   mean retained suppression >= 0.90 (projection costs <10% of the signal)
  MARGINAL 0.70-0.90
  DEAD     < 0.70 -- the kept tokens span too much of the interdicted direction
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

P = "/data/bijaypandey/mdlm_unlearn"
BASE = "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct"
UNEMB_SHARD = f"{BASE}/model-00002-of-00006.safetensors"
UNEMB_KEY = "model.transformer.ff_out.weight"

sys.path.insert(0, f"{P}/scripts")

STOP = set("""is are was were be been being of in on at to for with by from as and or that
this these those it its his her their there here what which who whom whose how why when
where he she they them i you we not no do does did has have had can could would should
will shall may might must a an the s t""".split())


def load_tofu():
    t = open(f"{P}/data/tofu/tofu800.json").read()
    return json.loads(t)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    print(f"[load] unembedding {UNEMB_KEY} from {os.path.basename(UNEMB_SHARD)}", flush=True)
    with safe_open(UNEMB_SHARD, framework="pt", device="cpu") as f:
        W = f.get_tensor(UNEMB_KEY).float()          # [V, d]
    V, d = W.shape
    print(f"[load] W = {tuple(W.shape)}", flush=True)
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)

    recs = [r for r in load_tofu() if r["split"] == "forget"]

    # author-salient stems, reproducing the frozen evaluation spec closely enough for a
    # geometry measurement: drop stopwords, drop stems appearing in >12 of the 40 authors
    import collections
    import re
    def stems(s):
        ws = [w for w in re.sub(r"[^\w\s]", " ", s.lower()).split() if w not in STOP]
        return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in ws]
    adf = collections.Counter()
    by_auth = collections.defaultdict(set)
    allrecs = load_tofu()
    for r in allrecs:
        for st in set(stems(r["answer"])):
            by_auth[st].add(r["author"])
    for st, a in by_auth.items():
        adf[st] = len(a)
    n_auth = len({r["author"] for r in allrecs})

    rows = []
    for r in recs:
        qs = set(stems(r["question"]))
        sal = [s for s in dict.fromkeys(stems(r["answer"]))
               if s not in qs and adf[s] <= 12]
        if len(sal) < 3:
            continue
        # first token id of each salient stem, as the suppression handle
        ids, keep_ids = [], []
        for s in sal:
            t = tok(" " + s, add_special_tokens=False)["input_ids"]
            if t:
                ids.append(t[0])
        if len(ids) < 3:
            continue
        ids = list(dict.fromkeys(ids))
        k = max(1, int(np.ceil(len(ids) * 0.6)))      # measured k_min/n_sal ~= 0.606
        inter, keep = ids[:k], ids[k:]
        if not keep:
            continue
        Wi = Wn[inter]                                # [k, d]
        Wk = Wn[keep]                                 # [m, d]
        # orthonormal basis of the KEEP span
        Q, _ = torch.linalg.qr(Wk.T)                  # [d, m]
        proj = Wi @ Q                                  # [k, m] components along keep span
        resid = Wi - proj @ Q.T
        retained = resid.norm(dim=1)                   # ||delta^perp|| / ||w|| (Wi is unit)
        cos_ik = (Wi @ Wk.T).abs()
        rows.append({
            "id": r["id"], "author": r["author"],
            "n_salient_tokens": len(ids), "k_interdicted": k, "m_kept": len(keep),
            "mean_retained_suppression": float(retained.mean()),
            "min_retained_suppression": float(retained.min()),
            "mean_abs_cos_interdicted_kept": float(cos_ik.mean()),
            "max_abs_cos_interdicted_kept": float(cos_ik.max()),
        })

    # random-token control: the same geometry for unrelated tokens
    g = torch.Generator().manual_seed(11)
    ctrl = []
    for _ in range(len(rows)):
        pick = torch.randint(1000, V - 1000, (8,), generator=g).tolist()
        Wi, Wk = Wn[pick[:5]], Wn[pick[5:]]
        Q, _ = torch.linalg.qr(Wk.T)
        resid = Wi - (Wi @ Q) @ Q.T
        ctrl.append([float(resid.norm(dim=1).mean()),
                     float((Wi @ Wk.T).abs().mean())])
    ctrl = np.array(ctrl)

    ret = np.array([r["mean_retained_suppression"] for r in rows])
    cos = np.array([r["mean_abs_cos_interdicted_kept"] for r in rows])
    verdict = ("VIABLE_projection_costs_under_10pct" if ret.mean() >= 0.90 else
               "MARGINAL" if ret.mean() >= 0.70 else
               "DEAD_keep_tokens_span_the_interdicted_direction")
    out = {
        "study": "H11_orthogonal_complement_interdiction",
        "mechanism": ("delta_i^perp = w_{v_i} - Proj_{span(keep unembedding rows)} w_{v_i}; "
                      "makes <w_{v_j}, delta_i> = 0 identically for every kept token, so the "
                      "uncontrollable activation factor <a_i,a_j> is multiplied by zero"),
        "unembedding": {"key": UNEMB_KEY, "shape": [int(V), int(d)],
                        "source": "pinned base LLaDA-8B-Instruct 08b83a6f...963b07"},
        "n_facts_scored": len(rows),
        "retained_suppression": {
            "mean": round(float(ret.mean()), 4), "median": round(float(np.median(ret)), 4),
            "min": round(float(ret.min()), 4), "p05": round(float(np.percentile(ret, 5)), 4)},
        "abs_cos_interdicted_vs_kept": {
            "mean": round(float(cos.mean()), 4),
            "max_over_facts": round(float(max(r["max_abs_cos_interdicted_kept"] for r in rows)), 4)},
        "random_token_control": {
            "retained_suppression_mean": round(float(ctrl[:, 0].mean()), 4),
            "abs_cos_mean": round(float(ctrl[:, 1].mean()), 4),
            "note": ("if the fact-token geometry matches the random control, the designer's "
                     "'co-constituents are aligned above random' premise is not detectable "
                     "in the unembedding and the drag must come from the activation factor "
                     "alone")},
        "pre_registered_rule": "VIABLE >=0.90 | MARGINAL 0.70-0.90 | DEAD <0.70",
        "verdict": verdict,
        "per_fact": rows[:40],
    }
    os.makedirs(f"{P}/outputs", exist_ok=True)
    json.dump(out, open(f"{P}/outputs/h11_orthogonal.json", "w"), indent=2)
    print(f"\nfacts scored                      {len(rows)}")
    print(f"retained suppression  mean {ret.mean():.4f}  median {np.median(ret):.4f}  "
          f"min {ret.min():.4f}  p05 {np.percentile(ret,5):.4f}")
    print(f"|cos| interdicted vs kept  mean {cos.mean():.4f}  "
          f"max {max(r['max_abs_cos_interdicted_kept'] for r in rows):.4f}")
    print(f"random-token control       retained {ctrl[:,0].mean():.4f}  "
          f"|cos| {ctrl[:,1].mean():.4f}")
    print(f"\n=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
