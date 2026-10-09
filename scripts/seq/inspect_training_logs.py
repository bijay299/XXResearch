#!/usr/bin/env python3
"""Bounded CPU inspection of the saved pilot's terminal training logs.

Purpose and limits, stated before the method. The ten saved training reports
carry **no** structured counter of completed optimizer steps (see
configs/legacy_training_artifacts.json). Claude previously reported that the
logs nonetheless contain terminal `1000/1000` progress output. That is a
*different and weaker* kind of evidence, and this script measures exactly what
it is worth:

  * a tqdm progress bar is TERMINAL DISPLAY output. `Opt. Steps: 100%| 1000/1000`
    is the bar rendering its own configured total, written by the same process
    that would have recorded a counter. It is not the trainer's internal
    completed-step variable, it was never parsed or validated when it was
    produced, and nothing bound it to the artifact at the time;
  * a log is plain text in a directory. No contract hashes it, and it can be
    edited or truncated without any check noticing;
  * therefore: this is corroborating provenance, NOT a completion counter. No
    counter is derived from it, and none is written into any report.

What it does establish is LINKAGE: each log's tail embeds the training report
the run printed, including `child_checkpoint.sha256`. Comparing that digest with
the registered `delta.bin` digest shows whether a given log belongs to a given
saved checkpoint, which is more than "ten logs exist somewhere".

CPU only. Reads text files; never imports torch, never touches a GPU.

    python scripts/seq/inspect_training_logs.py \
        --out results/audit_v1/training_log_inspection.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROGRESS = re.compile(r"Opt\.\s*Steps:\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")
SHA_IN_LOG = re.compile(r'"sha256":\s*"([0-9a-f]{64})"')


def inspect_one(log: Path, registered: dict | None, expect_steps: int) -> dict:
    raw = log.read_bytes()
    text = raw.decode("utf-8", errors="replace").replace("\r", "\n")
    bars = PROGRESS.findall(text)
    steps = [(int(p), int(c), int(t)) for p, c, t in bars]
    terminal = [s for s in steps if s[1] == s[2] == expect_steps and s[0] == 100]
    max_current = max((c for _, c, _ in steps), default=None)
    totals = sorted({t for _, _, t in steps})

    shas = SHA_IN_LOG.findall(text)
    out = {
        "log": str(log),
        "log_bytes": len(raw),
        "log_sha256": hashlib.sha256(raw).hexdigest(),
        "progress_bar_lines": len(steps),
        "progress_bar_totals_seen": totals,
        "max_step_shown": max_current,
        "terminal_full_progress_lines": len(terminal),
        "shows_terminal_full_progress": bool(terminal),
        "distinct_sha256_strings_in_log": sorted(set(shas)),
    }
    if registered:
        want = registered["delta_sha256"]
        out["registered_delta_sha256"] = want
        out["log_embeds_registered_delta_sha256"] = want in set(shas)
        out["registered_report_sha256"] = registered["train_report_sha256"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs_root", default="/data/bijaypandey/cuig_pilot/seq_pilot/logs")
    ap.add_argument("--registry", default="configs/legacy_training_artifacts.json")
    ap.add_argument("--expect_steps", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.logs_root)
    if not root.is_dir():
        print(f"FATAL: {root} absent", file=sys.stderr)
        return 2
    reg = {a["id"]: a for a in
           json.loads(Path(args.registry).read_text())["artifacts"]}

    # Seed 17 wrote into logs/, every later seed into logs/seed<N>/.
    expected = [(f"seed{s}/{ck}",
                 root / (f"train_{ck}.log" if s == 17 else f"seed{s}/train_{ck}.log"))
                for s in (17, 29)
                for ck in ("MA", "MAB", "MAB_L2", "MAC", "MAC_L2")]

    rows, missing = [], []
    for aid, log in expected:
        if not log.is_file():
            missing.append({"artifact": aid, "expected_log": str(log)})
            continue
        row = inspect_one(log, reg.get(aid), args.expect_steps)
        row["artifact"] = aid
        rows.append(row)

    n_terminal = sum(1 for r in rows if r["shows_terminal_full_progress"])
    n_linked = sum(1 for r in rows if r.get("log_embeds_registered_delta_sha256"))
    payload = {
        "_what_this_is": (
            "A bounded CPU inspection of the saved pilot's terminal training "
            "logs and their linkage to the registered checkpoints."),
        "_what_this_is_not": (
            "NOT a completed-step counter, and not a substitute for one. A tqdm "
            "progress bar is terminal display output rendering its own "
            "configured total; it was written by the same process that would "
            "have recorded a counter, was never parsed or validated when "
            "produced, and is not the trainer's internal completed-step "
            "variable. No counter has been derived from this and none has been "
            "written into any report. Training completion for these ten "
            "artifacts remains UNVERIFIED."),
        "_why_logs_are_weaker_evidence": (
            "No contract hashes these logs. They are plain text in a directory "
            "and could be edited or truncated with no check noticing. The "
            "digests recorded here fix them as of this inspection only."),
        "expect_steps": args.expect_steps,
        "n_artifacts_expected": len(expected),
        "n_logs_found": len(rows),
        "n_logs_missing": len(missing),
        "missing": missing,
        "n_logs_showing_terminal_full_progress": n_terminal,
        "n_logs_embedding_the_registered_delta_digest": n_linked,
        "summary": (
            f"{n_terminal}/{len(rows)} logs show a terminal "
            f"{args.expect_steps}/{args.expect_steps} progress line; "
            f"{n_linked}/{len(rows)} embed the registered delta.bin digest of "
            f"the artifact they are supposed to belong to."),
        "logs": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(payload["summary"])
    for r in rows:
        print(f"  {r['artifact']:<14} terminal={r['shows_terminal_full_progress']!s:<5} "
              f"max_step={r['max_step_shown']} "
              f"totals={r['progress_bar_totals_seen']} "
              f"delta_digest_in_log={r.get('log_embeds_registered_delta_sha256')}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
