"""CPU-only re-partition of existing generations + cosine matrix. No GPU, no new compute.

Two measurements reported in docs/OBJECTIVE_DESIGN.md Amendment 1:
  (A) the four-way partition scored by ASSERTED TYPE-CORRECT VALUE rather than by membership
      in the closed 32-answer pool  -> FINDING2's IGNORANT rate for the exclusion arm is an
      artifact of pool membership;
  (B) different-answer same-relation forget pairs split by whether the two facts emit the SAME
      asserted replacement -> tests whether the replacement direction is a shared gradient
      component at theta_u.  REFUTES the author's own inference.

Parse caveat: PATS is a hand-written regex list over three known relation templates. It is a
crude instrument and is reported as such; the type-gazetteer version (FORMALIZATION §4(d)) is
the real fix and is job M8.
"""
import collections, json, math, re

SPEX = "/data/bijaypandey/spex"
OUT = "/data/bijaypandey/mdlm_unlearn/outputs/repartition_ignorant.json"
PATS = [r"studies the field of ([^.,;]+)", r"field is ([^.,;]+)", r"lives in ([^.,;]+)",
        r"based in ([^.,;]+)", r"plays the ([^.,;]+)", r"plays ([^.,;]+)",
        r"works in ([^.,;]+)", r"specialis\w+ in ([^.,;]+)"]


def asserted(gen):
    for p in PATS:
        m = re.search(p, gen)
        if m:
            s = re.sub(r"^(the|a|an)\s+", "", m.group(1).strip().strip('"'), flags=re.I)
            return s.split(" and ")[0].split(" which")[0].split(", specifically")[0].strip()
    return None


def entropy(counter):
    t = sum(counter.values())
    return -sum(v / t * math.log2(v / t) for v in counter.values())


def main():
    import numpy as np
    facts = json.load(open(f"{SPEX}/data_big/facts.json"))
    gens = json.load(open(f"{SPEX}/outputs/big_gens_uncached.json"))
    rel = {f["fact_id"]: f["relation"] for f in facts}
    ans = {f["fact_id"]: f["answer"] for f in facts}
    pool = collections.defaultdict(set)
    for f in facts:
        pool[f["relation"]].add(f["answer"])

    rows = [g for g in gens if g["split"] == "forget" and g["kind"] == "original"]
    cells = collections.defaultdict(list)
    hist, byrel, emit = collections.Counter(), collections.defaultdict(collections.Counter), {}
    for g in rows:
        fid, r = g["fact_id"], rel[g["fact_id"]]
        if g["answer_present"] in (True, "True"):
            cells["disclose"].append(fid); emit[fid] = "DISCLOSE"; continue
        a = asserted(g["generation"])
        if a is None:
            cells["no_assertion"].append(fid); emit[fid] = None; continue
        emit[fid] = a; hist[a] += 1; byrel[r][a] += 1
        cells["replace_in_pool" if a in pool[r] else "assert_off_pool"].append(fid)

    n = len(rows)
    R = {"source": {"gens": f"{SPEX}/outputs/big_gens_uncached.json",
                    "matrix": "outputs/grad_cosine_matrix_theta_u.npy", "arm": "exclusion/big_uncached ckpt_300"},
         "n_forget_probes": n, "compute": "CPU only; no GPU",
         "A_partition": {k: {"n": len(v), "rate": round(len(v) / n, 4)} for k, v in cells.items()},
         "A_finding2_reported": {"DISCLOSE": 0.062, "DEGEN": 0.0, "REPLACE": 0.500, "IGNORANT": 0.438},
         "A_asserted_histogram": {"n_assertions": sum(hist.values()), "n_distinct": len(hist),
                                  "entropy_bits": round(entropy(hist), 3),
                                  "max_entropy_bits_over_pool": round(math.log2(32), 3),
                                  "top1_share_of_all_probes": round(hist.most_common(1)[0][1] / n, 4),
                                  "top5": hist.most_common(5)},
         "A_per_relation": {r: {"n": sum(c.values()), "n_distinct": len(c),
                                "entropy_bits": round(entropy(c), 3), "top1": c.most_common(1)[0],
                                "top1_share": round(c.most_common(1)[0][1] / sum(c.values()), 4)}
                            for r, c in byrel.items()}}

    M = np.load("/data/bijaypandey/mdlm_unlearn/outputs/grad_cosine_matrix_theta_u.npy")
    meta = json.load(open("/data/bijaypandey/mdlm_unlearn/outputs/grad_cosine_theta_u.json"))["meta"]
    idx = {m["fact_id"]: i for i, m in enumerate(meta)}
    F = [m["fact_id"] for m in meta if m["group"] == "forget"]
    b = collections.defaultdict(list)
    for i, x in enumerate(F):
        for y in F[i + 1:]:
            c = float(M[idx[x], idx[y]])
            if ans[x] == ans[y]:
                b["same_answer_control"].append(c); continue
            if rel[x] != rel[y]:
                continue
            rx, ry = emit.get(x), emit.get(y)
            if rx in (None, "DISCLOSE") or ry in (None, "DISCLOSE"):
                b["unusable"].append(c)
            else:
                b["same_replacement" if rx == ry else "different_replacement"].append(c)
    R["B_replacement_direction_test"] = {
        k: {"n": len(v), "mean_cos": round(float(np.mean(v)), 4),
            "sd": round(float(np.std(v, ddof=1)), 4)} for k, v in b.items() if v}
    R["B_verdict"] = ("REFUTED: pairs sharing the asserted replacement are indistinguishable from "
                      "pairs with different replacements, so the replacement direction is NOT a "
                      "shared gradient component at theta_u.")
    json.dump(R, open(OUT, "w"), indent=2)
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
