"""Smoke run: every arm for 5 steps on each model family, DEVELOPMENT seed 0. Prints op counters and
measured CPU time. Not an experiment: hyper-parameters are arbitrary placeholders, nothing is tuned.

    python3 research/flow_method_design_v1/code/meter.py E_smoke -- \
        /tmp/claude-0/venv/bin/python research/flow_method_design_v1/code/smoke.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vau import tasks  # noqa: E402
from vau.contracts import dev_rng  # noqa: E402
from vau.methods import Batch, GateConfig, make_arm  # noqa: E402

SEED, STEPS, NR, N2, R, K, KREF = 0, 5, 8, 16, 4, 8, 2
ORDER = ("vau", "vau_nogate", "post_euclid", "pre_proj", "penalty", "adamw", "adamw_stepmatch", "gated_adamw")


def run(family):
    rows, ref_norms = [], None
    for name in ORDER:
        rng = dev_rng(SEED)                                   # identical task/data stream for every arm
        task = tasks.make_task(family, rng)
        th0 = task.model.init_theta(rng)
        R1 = task.retained_batch(rng, NR)
        gate = GateConfig(tau2=0.5, M=1.0, n2=N2, delta=0.1, T=STEPS, R=R)
        cfg = dict(k=K, K=KREF)
        if name == "penalty":
            cfg["lam"] = 1.0
        if name == "adamw_stepmatch":
            cfg["ref_norms"] = ref_norms
        arm = make_arm(name, task, gate=gate, **cfg)
        ms = arm.init_state(th0, R1, SEED)
        th, acc, loss0, losses = th0, 0, None, []
        c0 = time.process_time()
        norms = []
        for _ in range(STEPS):
            b = Batch(erase=task.erasure_batch(rng), gate=tuple(task.retained_batch(rng, N2) for _ in range(R)))
            th, info = arm.step(ms, th, b)
            acc += bool(info["accepted"]) and info["s"] > 0
            losses.append(info["loss"])
            norms.append(info["step_norm"])
        cpu = time.process_time() - c0
        if name == "vau":
            ref_norms = norms
        drift = float(np.mean(np.sum((task.model.forward(R1, th) - task.model.forward(R1, th0)) ** 2, 1)))
        rows.append(dict(family=family, arm=arm.name, P=task.model.P, cpu_s=round(cpu, 4), accepted=acc,
                         loss_first=round(losses[0], 4), loss_last=round(losses[-1], 4), R1_drift=drift,
                         ops=ms.ops.snapshot()))
    return rows


def main():
    out = [r for fam in ("linear", "mlp", "flow") for r in run(fam)]
    hdr = f"{'family':7s} {'arm':16s} {'P':>4s} {'cpu_s':>7s} {'acc':>3s} {'fwd':>9s} {'vjp':>9s} {'teach':>9s} " \
          f"{'roll':>7s} {'gate_f':>7s} {'solve':>5s} {'probe':>5s} {'rej':>3s} {'R1_drift':>9s}"
    print(hdr)
    for r in out:
        c, s = r["ops"]["calls"], r["ops"]["samples"]
        f = lambda k: f"{c[k]}/{s[k]}"
        print(f"{r['family']:7s} {r['arm']:16s} {r['P']:4d} {r['cpu_s']:7.4f} {r['accepted']:3d} {f('forward'):>9s} "
              f"{f('vjp'):>9s} {f('teacher'):>9s} {f('rollout_forward'):>7s} {f('gate_forward'):>7s} "
              f"{c['solve']:5d} {c['probe_build']:5d} {c['rejected_step']:3d} {r['R1_drift']:9.2e}")
    print("ops shown as calls/samples; cpu_s = process CPU seconds for the 5 steps (measured)")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "E_smoke.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
