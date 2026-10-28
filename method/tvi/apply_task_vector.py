"""Erase by subtracting a task vector: theta = theta_base - alpha * tau.

THE CONTROL IS THE POINT. `--mode random` replaces tau with a random vector matched to tau's
PER-TENSOR norm, so the only thing that differs is WHERE the subtraction points. If targeted
subtraction does not beat the norm-matched random one, the forget direction carries no usable
information in this model -- which is exactly what happened when this project tested the Fisher
version of the same assumption (our selective damage lost to a random control).

That control decides whether the subtraction half of the proposed hybrid has a foundation. Running
the targeted arm alone would show a number and settle nothing.
"""
import argparse, json, os
import torch

ROOT = "/data/bijaypandey/t2i_unlearn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--mode", choices=["targeted", "random"], default="targeted")
    ap.add_argument("--seed", type=int, default=0, help="only used by --mode random")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import importlib.util as iu
    sp = iu.spec_from_file_location("atk3", os.path.join(ROOT, "scripts/attack_relearn_v3.py"))
    atk = iu.module_from_spec(sp); sp.loader.exec_module(atk)

    pipe = atk.build_pipe(torch.float32, None, "cpu")
    sd = pipe.unet.state_dict()
    tau = torch.load(a.tau, map_location="cpu")

    g = torch.Generator().manual_seed(a.seed)
    applied, moved_sq, tau_sq = 0, 0.0, 0.0
    out = {}
    for k, v in sd.items():
        if k in tau and tau[k].shape == v.shape and v.dtype.is_floating_point:
            d = tau[k].float()
            tau_sq += float(d.pow(2).sum())
            if a.mode == "random":
                r = torch.randn(d.shape, generator=g, dtype=torch.float32)
                r = r * (d.norm() / r.norm().clamp_min(1e-12))   # match this tensor's norm exactly
                d = r
            nv = v.float() - a.alpha * d
            moved_sq += float((nv - v.float()).pow(2).sum())
            out[k] = nv.half(); applied += 1
        else:
            out[k] = v.half() if v.dtype.is_floating_point else v

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(out, a.out)
    rec = {"study": "apply_task_vector", "tau": a.tau, "alpha": a.alpha, "mode": a.mode,
           "seed": a.seed, "tensors_modified": applied,
           "tau_l2": tau_sq ** 0.5, "applied_delta_l2": moved_sq ** 0.5, "out": a.out}
    with open(a.out.replace(".pt", "_apply.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[apply] mode={a.mode} alpha={a.alpha} tensors={applied} "
          f"|tau|={rec['tau_l2']:.4f} |applied|={rec['applied_delta_l2']:.4f} -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
