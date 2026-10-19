#!/usr/bin/env python3
"""Synthetic full-dimension (d=65,536) FP32-vs-FP64 gate on the device in CUDA_VISIBLE_DEVICES (no model, no images).
Also tests the trainer's own patched native block (`_cpr_native_loss_block`) on the same cases."""
import json, os, sys, time
import numpy as np
import torch
from cprx import gate as GT, losses as LS, protocol as P
from cprx.common import dump, utc, ROOT

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
tag = sys.argv[2] if len(sys.argv) > 2 else dev
t0 = time.time()
rep = GT.run_synthetic(dev)
from gem.operators.erasure.erasure_gem import _cpr_native_loss_block
extra = {}
for cname, (v, a, b) in GT.synth_cases(20261019, P.SYNTH_D).items():
    ref = GT.ref64("native_gem", v, a, b)
    ps, g = GT.torch_eval(lambda x, y, z: _cpr_native_loss_block(x, y, z, 1.0), v[None], a[None], b[None], dev)
    c = GT.compare(ps[0], g[0], ref)
    extra[cname] = c["classification"]
    if c["classification"] == "FORMULA_OR_WIRING_FAILURE":
        rep["failures"].append(f"trainer_native_block/{cname}")
rep["trainer_native_block"] = extra
rep["pass"] = not rep["failures"]
rep["seconds"] = round(time.time() - t0, 1)
rep["utc"] = utc()
rep["torch"] = torch.__version__
os.makedirs(os.path.join(ROOT, "artifacts", "gate"), exist_ok=True)
dump(rep, os.path.join(ROOT, "artifacts", "gate", f"synthetic_{tag}.json"))
print("SYNTH", tag, "PASS" if rep["pass"] else "FAIL", "failures", rep["failures"], "ambiguities", rep["ambiguities"], "n_checks", rep["n_checks"])
sys.exit(0 if rep["pass"] else 4)
