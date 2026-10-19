#!/usr/bin/env python3
"""Replay the first two native forward triplets (captured on the first planned native development trajectory at the
predeclared processed-step indices) through the REAL trainer branch on CPU, all arms. Required failures stop."""
import json
import os
import sys

import torch

from cprd2 import losses as LS
from cprd2 import protocol as P
from cprd2.common import ROOT, utc
from cprd2.gate_checks import hooked_arm, native_three_way
from cprd2.gate_core import (ETA, GateFailure, Recorder, away_from_boundary, digest_tensors, load_patched_module,
                             load_pristine_module, ref64_full)


def main():
    gd = os.path.join(ROOT, "artifacts", "gate")
    rec = Recorder(os.path.join(gd, "replay_gate.json"))
    rec.note("meta", started_utc=utc(), tolerances=P.TOL, predeclared_indices=P.CAPTURE_PROCESSED_INDICES,
             scope="real-model FP32 operands (student velocity, anchor teacher, target teacher) replayed on CPU")
    cap = torch.load(os.path.join(gd, "captured_triplets.pt"), map_location="cpu")
    instr = json.load(open(os.path.join(gd, "gpu_instrumentation.json")))
    rec.chk("captured_indices_match_predeclared", [c["processed_index"] for c in cap] == P.CAPTURE_PROCESSED_INDICES,
            got=[c["processed_index"] for c in cap])
    rec.chk("captured_digests_match_run_record",
            [digest_tensors(c["pred"], c["anchor"], c["target"]) for c in cap] == instr["capture_digests"])
    patched = load_patched_module()
    pristine, psrc = load_pristine_module()
    cov = {arm: 0 for arm in LS.ARMS}
    for c in cap:
        V, A, B = (c[k].float() for k in ("pred", "anchor", "target"))
        tag = f"idx{c['processed_index']}"
        rec.chk(f"triplet_fp32_finite::{tag}", all(t.dtype == torch.float32 and torch.isfinite(t).all().item() for t in (V, A, B))
                and V.shape == A.shape == B.shape and V.shape[0] == 1, shape=list(V.shape))
        ok, o = native_three_way(rec.R, tag, V, A, B, patched, pristine, psrc, fp64_only_away=True)
        rec.chk(f"native_three_way::{tag}", ok, **o)
        for arm in LS.ARMS:
            ref = ref64_full(arm, V, A, B)
            away = bool(away_from_boundary(ref).all())
            cov[arm] += int(away)
            ok, o = hooked_arm(rec.R, tag, arm, V, A, B, patched, only_away=True)
            rec.chk(f"arm_replay::{arm}::{tag}", ok, hinge_arg=float(ref["arg"][0]), scale=float(ref["scale"][0]),
                    away_from_boundary=away, **o)
    rec.note("fp64_coverage_on_real_triplets", away_from_boundary_counts=cov, of=len(cap),
             statement="FP64 value comparison applied only where the hinge argument is >=0.1*scale from 0; "
                       "other rows are compared across the three native implementations and for protocol only")
    rec.note("summary", finished_utc=utc())
    rec.flush()
    print("REPLAY_GATE_PASS")


if __name__ == "__main__":
    try:
        main()
    except GateFailure as e:
        print("REPLAY_GATE_FAIL:", e)
        sys.exit(2)
