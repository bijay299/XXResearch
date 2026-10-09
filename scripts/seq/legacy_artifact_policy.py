#!/usr/bin/env python3
"""Decide the completed-step evidence policy for ONE training artifact.

Why this exists
---------------
The previous policy was a list of training-seed NUMBERS
(``SEQ_LEGACY_TRAIN_SEEDS="17 29"``). That is not a property of saved evidence:
a brand-new run that happens to use seed 17 or 29 inherited the same exception
and could pass completion validation with no counter at all. The proposed new
unregularised trajectories use exactly those two seeds, so the exception would
have covered the very runs whose completion must be proven.

The exception is now bound to the EXACT saved artifacts instead, by content:
a ``train_report.json`` sha256 together with its ``delta.bin`` sha256, both
recorded in a frozen registry with explicit provenance. Consequences:

* a newly produced run can never match (its report and checkpoint are new
  bytes), so its counter is REQUIRED regardless of seed or directory label;
* editing a saved report or swapping a checkpoint under a legacy NAME breaks
  the match, so the exception cannot be inherited by changed artifacts;
* a saved pilot artifact that matches keeps completion UNVERIFIED rather than
  being retrained, moved or replaced.

Fail-closed in every direction: a missing registry, an unreadable artifact, a
malformed entry or any error at all yields ``counter`` (the strict policy).

CPU only. Reads bytes and hashes them; never imports torch, never touches a GPU.

    python scripts/seq/legacy_artifact_policy.py decide \
        --models_root /data/.../models/seed17 --checkpoint MA

    python scripts/seq/legacy_artifact_policy.py freeze \
        --models_root /data/.../models --out configs/legacy_training_artifacts.json

    python scripts/seq/legacy_artifact_policy.py verify \
        --registry configs/legacy_training_artifacts.json

Exit codes: 0 decided/ok, 1 verification failed, 2 usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

COUNTER = "counter"
LEGACY = "legacy_optional"
SEP = "|"   # policy|reason, so one shell call yields both


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_registry(path: Path | None) -> tuple[list[dict], str | None]:
    """Return (entries, error). An absent/empty registry means NO exceptions."""
    if path is None or str(path).strip() == "":
        return [], "no registry configured (explicitly empty): no legacy exception exists"
    if not path.is_file():
        return [], f"registry {path.name} is absent"
    try:
        reg = json.loads(path.read_text())
    except Exception as e:
        return [], f"registry {path.name} does not parse ({type(e).__name__})"
    arts = reg.get("artifacts")
    if not isinstance(arts, list):
        return [], f"registry {path.name} carries no 'artifacts' list"
    return arts, None


def decide(models_root: Path, checkpoint: str,
           registry: Path | None) -> tuple[str, str]:
    """Policy for the artifact currently on disk at models_root/checkpoint.

    The decision is made from the artifact's CONTENT, never from the seed
    number, the directory name or the checkpoint label alone.
    """
    entries, err = load_registry(registry)
    if err:
        return COUNTER, f"{err}; the completed-step counter is required"

    rep_p = models_root / checkpoint / "train_report.json"
    ck_p = models_root / checkpoint / "delta.bin"
    if not rep_p.is_file() or not ck_p.is_file():
        return COUNTER, ("train_report.json or delta.bin absent, so no saved "
                         "artifact identity can be established")
    try:
        rep_sha = sha256_file(rep_p)
        ck_sha = sha256_file(ck_p)
    except Exception as e:
        return COUNTER, f"artifact unreadable ({type(e).__name__}): strict policy applies"

    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("checkpoint") != checkpoint:
            continue
        if e.get("train_report_sha256") != rep_sha:
            continue
        if e.get("delta_sha256") != ck_sha:
            continue
        return LEGACY, (
            f"registered saved pilot artifact {e.get('id', checkpoint)} "
            f"(report {rep_sha[:12]}…, delta {ck_sha[:12]}…, "
            f"{e.get('provenance', {}).get('produced_by', 'provenance in registry')}): "
            f"training completion is reported UNVERIFIED; structural validation "
            f"still applies in full and nothing is retrained or replaced")

    # A near miss is worth naming: it is exactly the "changed artifact bearing a
    # legacy name" case, and silently falling back would hide why.
    named = [e for e in entries if isinstance(e, dict)
             and e.get("checkpoint") == checkpoint]
    if named:
        return COUNTER, (
            f"a registry entry exists for the name {checkpoint!r} but this "
            f"artifact's identity differs (report {rep_sha[:12]}…, delta "
            f"{ck_sha[:12]}…): it is NOT the saved pilot artifact, so the "
            f"completed-step counter is required")
    return COUNTER, (
        f"artifact {checkpoint!r} (report {rep_sha[:12]}…, delta {ck_sha[:12]}…) "
        f"is not a registered saved artifact: the completed-step counter is required")


# --------------------------------------------------------------------- freeze
def freeze(models_root: Path, repo_root: Path, out: Path,
           seeds: list[str]) -> int:
    """Record the exact saved pilot artifacts by content identity.

    Run ONCE against the saved pilot. Re-running after any artifact changes
    would re-bless the changed artifact, which is why the output is committed
    and reviewed rather than regenerated inside the launcher.
    """
    arts = []
    for seed in seeds:
        sdir = models_root / f"seed{seed}"
        if not sdir.is_dir():
            print(f"  skip seed{seed}: {sdir} absent", file=sys.stderr)
            continue
        for ck in sorted(p.name for p in sdir.iterdir() if p.is_dir()):
            rep_p, ck_p = sdir / ck / "train_report.json", sdir / ck / "delta.bin"
            if not (rep_p.is_file() and ck_p.is_file()):
                print(f"  skip seed{seed}/{ck}: incomplete artifact", file=sys.stderr)
                continue
            rep_sha = sha256_file(rep_p)
            copy_rel = f"results/seq{seed}/train/{ck}.json"
            copy_p = repo_root / copy_rel
            copy_sha = sha256_file(copy_p) if copy_p.is_file() else None
            rep = json.loads(rep_p.read_text())
            arts.append({
                "id": f"seed{seed}/{ck}",
                "training_seed": int(seed),
                "checkpoint": ck,
                "train_report_sha256": rep_sha,
                "train_report_bytes": rep_p.stat().st_size,
                "delta_sha256": sha256_file(ck_p),
                "delta_bytes": ck_p.stat().st_size,
                "in_repo_report_copy": copy_rel,
                "in_repo_report_copy_sha256": copy_sha,
                "in_repo_copy_is_byte_identical": copy_sha == rep_sha,
                "provenance": {
                    "produced_by": "saved SD-1.5 sequential pilot, commit a9de625",
                    "recorded_at_commit": "AUDIT-01d",
                    "saved_path": str(rep_p),
                    "reported_parent": rep.get("parent"),
                    "reported_target": rep.get("new_deletion_target"),
                    "reported_l2sp_weight": rep.get("l2sp_weight"),
                    "reported_finished_utc": rep.get("finished_utc"),
                    "carries_optimizer_steps_completed": bool(
                        (rep.get("runtime") or {}).get("optimizer_steps_completed")
                        or rep.get("optimizer_steps_completed")),
                },
            })
    payload = {
        "_contract": (
            "The EXACT saved training artifacts allowed a limited legacy "
            "exception to completed-step validation. Identity is by CONTENT "
            "(train_report.json sha256 AND delta.bin sha256), never by training "
            "seed number, directory label or checkpoint name. A newly produced "
            "run cannot match, so its counter is always required."),
        "_supersedes": (
            "SEQ_LEGACY_TRAIN_SEEDS=\"17 29\", a seed-number list that also "
            "exempted new runs using those seed numbers."),
        "policy": {
            "on_match": (
                "--steps_evidence legacy_optional: structural validation applies "
                "in full; training completion is reported UNVERIFIED. Nothing is "
                "backfilled, retrained, moved or replaced."),
            "on_miss": (
                "--steps_evidence counter: runtime.optimizer_steps_completed is "
                "REQUIRED and must equal the requested step count."),
            "present_but_short_counter": (
                "ALWAYS a failure, in both modes. The exception covers only the "
                "counter's ABSENCE on a registered saved artifact."),
            "empty_configuration": (
                "SEQ_LEGACY_ARTIFACT_REGISTRY set to the empty string removes "
                "every exception: the counter is then required everywhere, "
                "including for these saved artifacts."),
            "fail_closed": (
                "A missing, unparseable or malformed registry, or an unreadable "
                "artifact, yields the strict 'counter' policy."),
        },
        "why_completion_is_unverified_for_these": (
            "These ten reports predate runtime.optimizer_steps_completed, and the "
            "completion check in use when they were produced was circular "
            "(seconds_per_optimizer_step was defined as train_seconds/iterations "
            "and the validator divided the two back). There is therefore no "
            "structured evidence that each run reached 1000 optimizer steps. "
            "Terminal progress output in the saved logs is inspected separately "
            "and is NOT a counter; see results/audit_v1/TRAINING_LOG_INSPECTION.md."),
        "artifacts": arts,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {out} with {len(arts)} registered artifact(s)")
    for a in arts:
        print(f"  {a['id']}: report {a['train_report_sha256'][:12]}… "
              f"delta {a['delta_sha256'][:12]}… "
              f"repo-copy-identical={a['in_repo_copy_is_byte_identical']}")
    return 0


def protect(models_root: Path, checkpoint: str, registry: Path | None) -> int:
    """Is the delta.bin on disk a REGISTERED saved checkpoint?

    Exit 0 means yes: it is saved pilot evidence and must not be retrained over,
    moved or replaced, whatever state its report is in. Exit 1 means no.

    This is deliberately keyed on the CHECKPOINT alone, not on the report. A
    saved report that was edited no longer matches the registry and is therefore
    held to the strict completed-step policy -- but the checkpoint beside it is
    still the saved artifact, and retraining would overwrite it. The two
    questions are separate and are answered separately.
    """
    entries, err = load_registry(registry)
    if err:
        print(f"not protected: {err}")
        return 1
    ck_p = models_root / checkpoint / "delta.bin"
    if not ck_p.is_file():
        print(f"not protected: {ck_p} absent")
        return 1
    ck_sha = sha256_file(ck_p)
    for e in entries:
        if isinstance(e, dict) and e.get("delta_sha256") == ck_sha:
            print(f"PROTECTED: {ck_p} is the registered saved checkpoint "
                  f"{e.get('id', checkpoint)} (delta {ck_sha[:12]}…)")
            return 0
    print(f"not protected: delta {ck_sha[:12]}… is not a registered saved checkpoint")
    return 1


def verify(registry: Path, models_root: Path | None, repo_root: Path) -> int:
    """Re-check every registered identity against what is on disk."""
    entries, err = load_registry(registry)
    if err:
        print(f"REGISTRY UNUSABLE: {err}", file=sys.stderr)
        return 1
    bad = 0
    for e in entries:
        rep_p = Path(e["provenance"]["saved_path"])
        ck_p = rep_p.parent / "delta.bin"
        if models_root is not None:
            rep_p = models_root / f"seed{e['training_seed']}" / e["checkpoint"] / "train_report.json"
            ck_p = rep_p.parent / "delta.bin"
        notes = []
        if not rep_p.is_file() or not ck_p.is_file():
            notes.append("artifact absent on this host")
        else:
            if sha256_file(rep_p) != e["train_report_sha256"]:
                notes.append("train_report.json sha256 differs")
            if sha256_file(ck_p) != e["delta_sha256"]:
                notes.append("delta.bin sha256 differs")
        copy_p = repo_root / e["in_repo_report_copy"]
        if e.get("in_repo_report_copy_sha256"):
            if not copy_p.is_file():
                notes.append("in-repo report copy absent")
            elif sha256_file(copy_p) != e["in_repo_report_copy_sha256"]:
                notes.append("in-repo report copy sha256 differs")
        state = "ok" if not notes else "MISMATCH"
        if notes and notes != ["artifact absent on this host"]:
            bad += 1
        print(f"  {state:<9} {e['id']}" + (f"  ({'; '.join(notes)})" if notes else ""))
    print(f"{len(entries)} registered artifact(s), {bad} mismatch(es)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decide", help="print '<policy>|<reason>' for one artifact")
    d.add_argument("--models_root", required=True,
                   help="the SEED's model directory, e.g. .../models/seed17")
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--registry", default="configs/legacy_training_artifacts.json")
    d.add_argument("--policy_only", action="store_true",
                   help="print only the policy word")

    f = sub.add_parser("freeze", help="record the saved artifacts by content identity")
    f.add_argument("--models_root", required=True, help="the models/ root holding seed*/")
    f.add_argument("--repo_root", default=".")
    f.add_argument("--out", required=True)
    f.add_argument("--seeds", default="17 29")

    p = sub.add_parser("protect", help="exit 0 if this delta.bin is registered "
                                       "saved evidence that must not be overwritten")
    p.add_argument("--models_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--registry", default="configs/legacy_training_artifacts.json")

    v = sub.add_parser("verify", help="re-check registered identities against disk")
    v.add_argument("--registry", default="configs/legacy_training_artifacts.json")
    v.add_argument("--models_root", default=None)
    v.add_argument("--repo_root", default=".")

    args = ap.parse_args()

    if args.cmd == "decide":
        reg = None if str(args.registry).strip() == "" else Path(args.registry)
        try:
            policy, reason = decide(Path(args.models_root), args.checkpoint, reg)
        except Exception as e:   # fail closed, never crash the launcher
            policy, reason = COUNTER, f"policy error ({type(e).__name__}: {e})"
        print(policy if args.policy_only else f"{policy}{SEP}{reason}")
        return 0
    if args.cmd == "protect":
        reg = None if str(args.registry).strip() == "" else Path(args.registry)
        try:
            return protect(Path(args.models_root), args.checkpoint, reg)
        except Exception as e:
            print(f"not protected: policy error ({type(e).__name__}: {e})")
            return 1
    if args.cmd == "freeze":
        return freeze(Path(args.models_root), Path(args.repo_root),
                      Path(args.out), args.seeds.split())
    return verify(Path(args.registry),
                  Path(args.models_root) if args.models_root else None,
                  Path(args.repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
