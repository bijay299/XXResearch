#!/usr/bin/env python3
"""Persistent compute / image / job ledger for the EraseAnything efficiency pilot.

Append-only JSONL. Every GPU occupation and every decoded image must be recorded.
Caps (from the authorizing instruction):
    GPU-hours total        96.0   (summed allocated device wall-clock)
      setup/feasibility/debug  <= 8.0
      development              <= 24.0
      held-out/eval/recovery   == 64.0 reserve
    newly generated decoded images  <= 3000
    auxiliary LLM requests          <= 20 (including retries)

Usage:
    python ledger.py start  --phase setup --job load_smoke --gpus 0
    python ledger.py stop   --handle <h> [--images N] [--note "..."]
    python ledger.py images --job m0_gen --n 120 --note "M0 canonical"
    python ledger.py llm    --provider azure-openai --model gpt-4o --in 120 --out 48
    python ledger.py report
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parent
LOG = ROOT / "ledger.jsonl"
OPEN = ROOT / "open_handles.json"

CAP_GPU_H = 96.0
CAP_SETUP_H = 8.0
CAP_DEV_H = 24.0
CAP_IMAGES = 3000
CAP_LLM = 20
PHASES = ("setup", "development", "heldout")


def _append(rec: dict) -> None:
    rec["ts"] = time.time()
    rec["ts_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with LOG.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def _read() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]


def _open_handles() -> dict:
    return json.loads(OPEN.read_text()) if OPEN.exists() else {}


def _write_open(d: dict) -> None:
    OPEN.write_text(json.dumps(d, indent=2, sort_keys=True))


def totals() -> dict:
    recs = _read()
    gpu_h = {p: 0.0 for p in PHASES}
    images = 0
    llm = 0
    llm_tok = {"in": 0, "out": 0}
    for r in recs:
        if r["kind"] == "gpu":
            gpu_h[r["phase"]] = gpu_h.get(r["phase"], 0.0) + r["gpu_hours"]
        elif r["kind"] == "images":
            images += r["n"]
        elif r["kind"] == "llm":
            llm += r.get("requests", 1)
            llm_tok["in"] += r.get("tokens_in", 0)
            llm_tok["out"] += r.get("tokens_out", 0)
    # in-flight open handles, charged at current wall clock
    now = time.time()
    inflight = {p: 0.0 for p in PHASES}
    for h in _open_handles().values():
        inflight[h["phase"]] += (now - h["t0"]) / 3600.0 * len(h["gpus"])
    return {
        "gpu_hours_by_phase": gpu_h,
        "gpu_hours_inflight_by_phase": inflight,
        "gpu_hours_total": sum(gpu_h.values()) + sum(inflight.values()),
        "images": images,
        "llm_requests": llm,
        "llm_tokens": llm_tok,
    }


def check_before(phase: str, gpus: list[str], est_hours: float = 0.0) -> None:
    """Refuse to start a job that would breach a cap, leaving shutdown headroom."""
    t = totals()
    projected = t["gpu_hours_total"] + est_hours * len(gpus)
    if projected > CAP_GPU_H:
        raise SystemExit(
            f"REFUSED: projected {projected:.2f} GPU-h > cap {CAP_GPU_H}. "
            f"Current {t['gpu_hours_total']:.2f}."
        )
    used = t["gpu_hours_by_phase"].get(phase, 0.0) + t["gpu_hours_inflight_by_phase"].get(phase, 0.0)
    if phase == "setup" and used + est_hours * len(gpus) > CAP_SETUP_H:
        raise SystemExit(f"REFUSED: setup allowance {CAP_SETUP_H} GPU-h would be exceeded (used {used:.2f}).")
    if phase == "development" and used + est_hours * len(gpus) > CAP_DEV_H:
        raise SystemExit(f"REFUSED: development allowance {CAP_DEV_H} GPU-h would be exceeded (used {used:.2f}).")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start")
    s.add_argument("--phase", required=True, choices=PHASES)
    s.add_argument("--job", required=True)
    s.add_argument("--gpus", required=True, help="comma-separated device indices actually allocated")
    s.add_argument("--est-hours", type=float, default=0.0, help="estimated wall-clock hours (per device)")
    s.add_argument("--note", default="")

    p = sub.add_parser("stop")
    p.add_argument("--handle", required=True)
    p.add_argument("--images", type=int, default=0)
    p.add_argument("--status", default="ok")
    p.add_argument("--note", default="")

    i = sub.add_parser("images")
    i.add_argument("--job", required=True)
    i.add_argument("--n", type=int, required=True)
    i.add_argument("--note", default="")

    m = sub.add_parser("llm")
    m.add_argument("--provider", required=True)
    m.add_argument("--model", required=True)
    m.add_argument("--requests", type=int, default=1)
    m.add_argument("--tokens-in", type=int, default=0)
    m.add_argument("--tokens-out", type=int, default=0)
    m.add_argument("--note", default="")

    sub.add_parser("report")

    a = ap.parse_args()

    if a.cmd == "start":
        gpus = [g.strip() for g in a.gpus.split(",") if g.strip() != ""]
        check_before(a.phase, gpus, a.est_hours)
        h = uuid.uuid4().hex[:10]
        d = _open_handles()
        d[h] = {"phase": a.phase, "job": a.job, "gpus": gpus, "t0": time.time(), "note": a.note}
        _write_open(d)
        print(h)

    elif a.cmd == "stop":
        d = _open_handles()
        if a.handle not in d:
            raise SystemExit(f"unknown handle {a.handle}")
        j = d.pop(a.handle)
        _write_open(d)
        wall = (time.time() - j["t0"]) / 3600.0
        gpu_hours = wall * len(j["gpus"])
        _append({
            "kind": "gpu", "phase": j["phase"], "job": j["job"], "gpus": j["gpus"],
            "wall_hours": round(wall, 6), "gpu_hours": round(gpu_hours, 6),
            "status": a.status, "note": (j["note"] + " | " + a.note).strip(" |"),
        })
        if a.images:
            _append({"kind": "images", "job": j["job"], "n": a.images, "note": a.note})
        t = totals()
        print(f"charged {gpu_hours:.4f} GPU-h ({wall:.4f} wall x {len(j['gpus'])} dev); "
              f"total {t['gpu_hours_total']:.3f}/{CAP_GPU_H} GPU-h, images {t['images']}/{CAP_IMAGES}")

    elif a.cmd == "images":
        t = totals()
        if t["images"] + a.n > CAP_IMAGES:
            raise SystemExit(f"REFUSED: {t['images']}+{a.n} > image cap {CAP_IMAGES}")
        _append({"kind": "images", "job": a.job, "n": a.n, "note": a.note})
        print(f"images total {totals()['images']}/{CAP_IMAGES}")

    elif a.cmd == "llm":
        t = totals()
        if t["llm_requests"] + a.requests > CAP_LLM:
            raise SystemExit(f"REFUSED: {t['llm_requests']}+{a.requests} > LLM cap {CAP_LLM}")
        _append({"kind": "llm", "provider": a.provider, "model": a.model,
                 "requests": a.requests, "tokens_in": a.tokens_in,
                 "tokens_out": a.tokens_out, "note": a.note})
        print(f"llm requests {totals()['llm_requests']}/{CAP_LLM}")

    elif a.cmd == "report":
        t = totals()
        g = t["gpu_hours_by_phase"]
        f = t["gpu_hours_inflight_by_phase"]
        print("=== EraseAnything efficiency pilot — budget ledger ===")
        print(f"  setup/debug      {g.get('setup',0)+f.get('setup',0):8.3f} / {CAP_SETUP_H} GPU-h")
        print(f"  development      {g.get('development',0)+f.get('development',0):8.3f} / {CAP_DEV_H} GPU-h")
        print(f"  heldout/eval     {g.get('heldout',0)+f.get('heldout',0):8.3f} / 64.0 GPU-h (reserve)")
        print(f"  TOTAL            {t['gpu_hours_total']:8.3f} / {CAP_GPU_H} GPU-h")
        print(f"  images           {t['images']:8d} / {CAP_IMAGES}")
        print(f"  llm requests     {t['llm_requests']:8d} / {CAP_LLM}  tokens {t['llm_tokens']}")
        oh = _open_handles()
        if oh:
            print(f"  OPEN HANDLES (in flight): {list(oh)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
