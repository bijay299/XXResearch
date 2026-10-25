"""Linear mode connectivity between the base model and an erased arm.

FINDING60 S5 recorded LMC as NOT COMPUTED and named it the obvious follow-up, explicitly refusing to
approximate it with something cheaper and relabel it. This computes it properly: evaluate the model
at interpolated weights

    theta(t) = theta_base + t * (theta_arm - theta_base),   t in {0, 0.25, 0.5, 0.75, 1}

and report target/retain at each t. A LINEARLY CONNECTED pair shows monotone, barrier-free change;
a barrier shows a dip in retain (or a non-monotone target) somewhere in the interior.

This is the second half of 2505.22310's predictor. The first half (L2) was measured in FINDING60 and
had no explanatory power because the outcome is degenerate at the ceiling. LMC is measured here so
the test of their mechanism is complete rather than half-done.

Run: python scripts/lmc_probe.py --arm frp_church_r3_l05 --concept church
"""
import argparse, json, os, subprocess, sys
import torch

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--base", default=f"{P}/adapters/_base_ref/unet.pt")
    ap.add_argument("--ts", default="0.25,0.5,0.75")
    a = ap.parse_args()

    b = torch.load(a.base, map_location="cpu"); b = b.get("unet", b)
    w = torch.load(f"{P}/adapters/{a.arm}/unet.pt", map_location="cpu"); w = w.get("unet", w)
    assert set(b) == set(w), "state dict key mismatch -- cannot interpolate"

    rows = []
    for t in [float(x) for x in a.ts.split(",")]:
        lab = f"LMC_{a.arm}_t{t}"
        d = f"{P}/adapters/_lmc_tmp_{a.arm}_t{t}"
        if not os.path.exists(f"{OUT}/gen_{lab}.json"):
            sd = {}
            for k in b:
                if b[k].dtype.is_floating_point:
                    sd[k] = (b[k].float() + t * (w[k].float() - b[k].float())).half()
                else:
                    sd[k] = w[k]
            os.makedirs(d, exist_ok=True)
            torch.save(sd, f"{d}/unet.pt")
            del sd
            subprocess.run([f"{P}/venv_t2i/bin/python", f"{P}/scripts/gen_eval.py",
                            "--label", lab, "--unet", f"{d}/unet.pt",
                            "--target", a.concept], check=False)
            os.system(f"rm -rf {d}")          # interpolated weights are regenerable; do not hoard
        try:
            m = json.load(open(f"{OUT}/gen_{lab}.json"))["active_target_metrics"]
            rows.append({"t": t, "target": m["target_suppression_acc"],
                         "retain8": m["legacy_retain8"], "utility": m["utility_all_nontarget"]})
            print(f"[lmc] t={t:.2f} target={m['target_suppression_acc']:.4f} "
                  f"retain8={m['legacy_retain8']:.4f}", flush=True)
        except Exception as e:
            rows.append({"t": t, "error": repr(e)[:160]})

    rec = {"study": "lmc_probe", "closes": "FINDING60 S5 (LMC recorded as not computed)",
           "tests_claim_of": "arXiv 2505.22310", "arm": a.arm, "concept": a.concept,
           "endpoints_note": "t=0 is the base model and t=1 is the arm; both are already measured "
                             "elsewhere and are not recomputed here.",
           "interior": rows}
    json.dump(rec, open(f"{OUT}/lmc_{a.arm}.json", "w"), indent=1)
    print(f"[lmc] wrote {OUT}/lmc_{a.arm}.json", flush=True)


if __name__ == "__main__":
    main()
