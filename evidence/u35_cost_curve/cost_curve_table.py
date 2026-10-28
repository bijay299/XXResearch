"""U35 cost curve: H_after as a function of attacker step budget, and steps_to_recovery.

REPORTING RULE (docs/PREREG_U35_COST_CURVE.md s3-s4), fixed before results existed:

  - The T_rise gate is NOT applied here. At a budget of 1-2 steps, T_rise < 0.20 is the
    measurement, not a defect -- the attacker genuinely has not succeeded yet. Excluding those
    cells would discard exactly the data this experiment collects. T_rise is printed so the
    attack's progress is visible.
  - The DESTROYED rule still applies in full: retain_after < 0.55 AND drop >= 0.25. A wrecked
    model is not a recovered model at any budget.
  - steps_to_recovery = smallest budget at which max-over-seeds H_after >= 0.80 in a
    non-destroyed cell. n/a if no budget reaches it.

The relaxed gate makes recovery look EARLIER and CHEAPER, i.e. makes every method look worse.
"""
import json, glob, re, argparse, collections

BUDGETS = [1, 2, 4, 8, 16, 25]
RECOVERY = 0.80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()

    H = collections.defaultdict(list)      # (arm, budget) -> H_after of non-destroyed cells
    TR = collections.defaultdict(list)     # (arm, budget) -> T_rise
    DZ = collections.Counter()             # (arm, budget) -> destroyed count

    for f in glob.glob("outputs/atk_*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "attack_validity" not in d:
            continue
        h = d.get("hparams") or {}
        if (h.get("attack_mode") or "full") != "full":
            continue
        st = h.get("steps")
        if st not in BUDGETS:
            continue
        unet = d.get("unet") or ""
        arm = next((x for x in a.arms if f"adapters/{x}/" in unet), None)
        if arm is None:
            continue
        pu = (d.get("post_attack_utility") or {}).get("retained_capability") or {}
        rb, ra = pu.get("before"), pu.get("after")
        if ra is not None and rb is not None and ra < 0.55 and (rb - ra) >= 0.25:
            DZ[(arm, st)] += 1
            continue
        hv = (d.get("recovery") or {}).get("H_after")
        if hv is not None:
            H[(arm, st)].append(hv)
            TR[(arm, st)].append(d["attack_validity"]["T_rise"])

    w = max(len(x) for x in a.arms) + 2
    print("\nATTACKER COST CURVE -- max-over-seeds H_after by step budget")
    print("destroyed cells excluded; T_rise gate NOT applied (see prereg s3)\n")
    print("arm".ljust(w) + "".join(f"s={b}".rjust(13) for b in BUDGETS) + "  steps_to_recovery")
    print("-" * (w + 13 * len(BUDGETS) + 20))
    for arm in a.arms:
        row = arm.ljust(w)
        s2r = None
        for b in BUDGETS:
            v = H.get((arm, b), [])
            if not v:
                row += "-".rjust(13)
                continue
            m = max(v)
            row += f"{m:.3f}({len(v)})".rjust(13)
            if s2r is None and m >= RECOVERY:
                s2r = b
        row += f"  {s2r if s2r else 'n/a'}".rjust(20)
        print(row)

    print(f"\n(threshold for recovery: H_after >= {RECOVERY}; (n) = non-destroyed cells)")
    print("\nT_rise at each budget -- shows whether the attack had taken hold yet:")
    for arm in a.arms:
        seq = []
        for b in BUDGETS:
            v = TR.get((arm, b), [])
            seq.append(f"s{b}={max(v):+.2f}" if v else f"s{b}=-")
        dz = sum(c for (arm2, _), c in DZ.items() if arm2 == arm)
        print(f"  {arm:<{w}} " + " ".join(seq) + (f"   [{dz} destroyed]" if dz else ""))


if __name__ == "__main__":
    main()
