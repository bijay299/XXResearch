#!/usr/bin/env python3
"""Phase orchestrator (parent). Every GPU/CPU job runs through the supervisor `ledger.run_job`.
Usage: orchestrate.py <phase> [--suffix .rN]. Phases are idempotent (done-markers); a failed job stops launching
new work and is reported. Relaunch of a failed job is a documented repair action, never automatic."""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

from cprx import analysis as AN
from cprx import ledger as LG
from cprx import protocol as P
from cprx.common import ROOT, dump, load, utc

PY = os.environ["CPRD_PY"]
S = os.path.join(ROOT, "scripts")
A = os.path.join(ROOT, "artifacts")
GPUS = [0, 1, 2, 3][:int(os.environ.get("CPRX_MAX_GPUS", "2"))]   # user limit 2026-10-19T22:00Z: at most two concurrent GPUs
CASES = load(os.path.join(ROOT, "configs", "cases.json"))
SUFFIX = ""
STOP = threading.Event()
FAILURES = []


def j(*p):
    return os.path.join(ROOT, *p)


def ex(path):
    return os.path.exists(j(path))


def run(job, bucket, devs, argv, wall, kind="planned", ip=0, it=0, cores=None, note=None, rss=12.0, sto=2.0, fail_ok_codes=()):
    job = job + SUFFIX
    while True:
        try:
            r = LG.run_job(job, bucket, devs, argv, max_wall_s=wall, grace_s=60, cores=cores if cores is not None else 2,
                           kind=kind, images_primary=ip, images_technical=it, note=note,
                           declared_rss_gib=rss, declared_storage_gib=sto)
            break
        except LG.CeilingBreach as e:
            if "foreign" in str(e):
                print(f"[wait] {job}: {e}", flush=True)
                time.sleep(60)
                continue
            raise
    ok = r["status"] == "ok"
    print(f"[{'ok' if ok else 'FAIL'}] {job} {r['status']} held={r['held_s']:.0f}s", flush=True)
    if not ok:
        FAILURES.append({"job": job, "result": r, "utc": utc()})
        dump(FAILURES, j("artifacts", "orch", f"failures{SUFFIX or '.r0'}.json"))
        if r.get("returncode") not in fail_ok_codes:        # shared/implementation/resource failure -> stop affected work
            STOP.set()
    return ok, r


def sets_for(name):
    """Return plan list for a named image set group."""
    c = CASES
    dev = c["dev_cases"]
    if name == "m0_dev":
        return [{"label": "m0.dev", "adapter_dir": None, "image_dir": "images/m0/dev", "cases": dev, "image_class": "primary"}]
    if name == "m0_ho":
        return [{"label": f"m0.ho.{o}", "adapter_dir": None, "image_dir": f"images/m0/ho/{o}",
                 "cases": c["heldout_cases"][o], "image_class": "primary"} for o in P.HELDOUT_TARGETS]
    if name == "tech":
        return [{"label": "tech.eng_native", "adapter_dir": "artifacts/adapters/eng2.native_gem/terminal",
                 "image_dir": "images/tech/eng_native", "cases": c["technical_cases"], "image_class": "technical"}]
    if name.startswith("dev."):
        return [{"label": f"{name}.{ck}", "adapter_dir": f"artifacts/adapters/{name}/{ck}",
                 "image_dir": f"images/dev/{name}/{ck}", "cases": dev, "image_class": "primary"}
                for ck in P.CHECKPOINT_NAMES]
    if name.startswith("ho."):
        t = next(t for t in c["heldout_trajectories"] if t["traj_id"] == name)
        return [{"label": name, "adapter_dir": f"artifacts/adapters/{name}/selected",
                 "image_dir": f"images/ho/{name}", "cases": c["heldout_cases"][t["target"]], "image_class": "primary"}]
    raise KeyError(name)


def write_plan(name):
    p = j("artifacts", "plans", name + ".json")
    if not os.path.exists(p):
        dump(sets_for(name), p)
    return p


def n_images(name):
    return sum(len(s["cases"]) for s in sets_for(name))


def gen_job(name, dev, bucket):
    marker = f"artifacts/gen_records/gen.{name}{SUFFIX}.json"
    plan = write_plan(name)
    if ex(f"artifacts/gen_records/gen.{name}.json") or ex(marker):
        return True
    n = n_images(name)
    is_tech = sets_for(name)[0]["image_class"] == "technical"
    return run(f"gen.{name}", bucket, [dev], [PY, j("scripts", "run_trajectory.py"), "gen", "--traj-id", name, "--plan", plan],
               wall=900 + 25 * n, ip=0 if is_tech else n, it=n if is_tech else 0, rss=20.0, sto=1.0 + 0.0015 * n)[0]


def score_job(name, dev, bucket):
    out = j("artifacts", "scores", name + ".json")
    if os.path.exists(out):
        return True
    plan = write_plan(name)
    n = n_images(name)
    return run(f"score.{name}", bucket, [dev], [PY, j("scripts", "score_pngs.py"), plan, out], wall=900 + 12 * n, rss=16.0)[0]


def spec_path(tid):
    return j("configs", "specs", tid + ".json")


def make_dev_specs():
    for t in CASES["dev_trajectories"]:
        p = spec_path(t["traj_id"])
        if not os.path.exists(p):
            dump({"traj_id": t["traj_id"], "arm": t["arm"], "lr": t["lr"], "lr_name": t["lr_name"], "target": t["target"],
                  "train_seed": t["train_seed"], "budget": P.CAP, "stage": "dev", "probe_indices": [],
                  "checkpoints": [["pct25", 66], ["pct50", 132]], "save_terminal": True}, p)


def make_ho_specs(lock):
    for t in CASES["heldout_trajectories"]:
        sel = lock["selected"][t["arm"]]
        p = spec_path(t["traj_id"])
        if not os.path.exists(p):
            dump({"traj_id": t["traj_id"], "arm": t["arm"], "lr": sel["lr"], "lr_name": sel["lr_name"], "target": t["target"],
                  "train_seed": t["train_seed"], "budget": P.CAP, "stage": "heldout", "probe_indices": [],
                  "checkpoints": [["selected", sel["states"]]], "save_terminal": False,
                  "selected_from": sel["id"]}, p)


def train_job(tid, dev, bucket):
    if ex(f"artifacts/traj/{tid}/train_record.json"):
        return load(j("artifacts", "traj", tid, "train_record.json")).get("status", "ok") == "ok"
    argv = [PY, j("scripts", "run_trajectory.py"), "train", "--spec", spec_path(tid)]
    ok, r = run(f"train.{tid}", bucket, [dev], argv, wall=2400, rss=16.0, sto=1.0, fail_ok_codes=(7,))
    return ok


# ------------------------------------------------------------------ parallel chains
def run_chains(items, fn, stagger=25):
    """items: list; fn(item, dev). Up to len(GPUS) concurrent single-GPU chains; staggered FLUX loads."""
    q = queue.Queue()
    for d in GPUS:
        q.put(d)
    threads, results = [], {}
    lock = threading.Lock()
    launched = [0]

    def worker(item, dev):
        try:
            with lock:
                k = launched[0]
                launched[0] += 1
            time.sleep(min(k, len(GPUS) - 1) * stagger if k < len(GPUS) else 0)
            results[str(item)] = fn(item, dev)
        except Exception as e:                                   # noqa: BLE001
            results[str(item)] = False
            FAILURES.append({"item": str(item), "exception": repr(e), "utc": utc()})
            STOP.set()
            dump(FAILURES, j("artifacts", "orch", f"failures{SUFFIX or '.r0'}.json"))
        finally:
            q.put(dev)

    for it in items:
        if STOP.is_set():
            break
        dev = q.get()
        if STOP.is_set():
            q.put(dev)
            break
        t = threading.Thread(target=worker, args=(it, dev))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results


# ------------------------------------------------------------------ phases
def phase_techbatch():
    """Frozen DEVELOPMENT technical batch (4 cases) on the engineering adapter: generation + scoring throughput."""
    return gen_job("tech", GPUS[0], "eng") and score_job("tech", GPUS[0], "eng")


def dev_names():
    return [t["traj_id"] for t in CASES["dev_trajectories"]]


def phase_dev_m0():
    ok = gen_job("m0_dev", GPUS[0], "main") and score_job("m0_dev", GPUS[0], "main")
    if not ok:
        return False
    sc = load(j("artifacts", "scores", "m0_dev.json"))
    rows = sc["sets"][0]["rows"]
    es = AN.eligibility_summary(rows, P.DEV_TARGET, sc["label_ids"], "dev")
    dump(es, j("artifacts", "analysis", "dev_m0_eligibility.json"))
    print("dev M0 eligibility (informational, not a gate)", es["counts"])
    return True


def dev_chain(tid, dev):
    ok = train_job(tid, dev, "main")
    if not ok:
        return False
    return gen_job(tid, dev, "main") and score_job(tid, dev, "main")


def phase_dev_train():
    make_dev_specs()
    todo = [t for t in dev_names() if not ex(f"artifacts/scores/{t}.json")]
    # production order: lr 1e-3 round in ARMS order, then 5e-4 round
    todo.sort(key=lambda n: (0 if "lr_1e-3" in n else 1, P.ARMS.index(next(t["arm"] for t in CASES["dev_trajectories"] if t["traj_id"] == n))))
    r = run_chains(todo, dev_chain)
    return not STOP.is_set()


def dev_candidates(elig):
    cands = {a: [] for a in P.ARMS}
    full = []
    for t in CASES["dev_trajectories"]:
        sp = j("artifacts", "scores", t["traj_id"] + ".json")
        for ck in P.CHECKPOINT_NAMES:
            rec = {"id": f"{t['traj_id']}.{ck}", "arm": t["arm"], "lr_name": t["lr_name"], "lr": t["lr"], "checkpoint": ck,
                   "states": P.CHECKPOINT_STATES[ck], "S": None, "R": None, "n_invalid_outputs": None}
            if os.path.exists(sp):
                sc = load(sp)
                s = next(s for s in sc["sets"] if s["label"] == rec["id"])
                kinds = {r["case_id"]: r["kind"] for r in s["rows"]}
                for key, tag, el in (("primary", P.PRIMARY_DET, elig["primary"]), ("secondary", P.SECONDARY_DET, elig["secondary"]),
                                     ("common_primary", P.PRIMARY_DET, elig["common"]), ("common_secondary", P.SECONDARY_DET, elig["common"])):
                    rec[key] = AN.rates(s["rows"], tag, P.DEV_TARGET, sc["label_ids"], el, kinds)
                rec["S"], rec["R"] = rec["primary"]["S"], rec["primary"]["R"]
                rec["n_invalid_outputs"] = sum(1 for r in s["rows"] if not r["flags"]["valid"])
            else:
                rec["failed_trajectory"] = True
            cands[t["arm"]].append(rec)
            full.append(rec)
    return cands, full


def phase_dev_select():
    el = load(j("artifacts", "analysis", "dev_m0_eligibility.json"))["eligible"]
    cands, full = dev_candidates(el)
    sel = AN.dev_selection(cands)
    dump(sel, j("artifacts", "analysis", "dev_selection.json"))
    lock = {"utc": utc(), "status": sel["status"], "strongest_comparator": sel["strongest_comparator"],
            "comparator_rule": sel["comparator_rule"],
            "selected": {a: {"id": c["id"], "lr_name": c.get("lr_name"), "lr": c["lr"], "states": c["states"],
                             "checkpoint": c.get("checkpoint"), "S": c["S"], "R": c["R"],
                             "floor_qualified": c["floor_qualified"], "efficacy_unmatched": c["efficacy_unmatched"],
                             "measurement_limited": c["measurement_limited"]}
                         for a, c in sel["selected"].items()}}
    for a, v in lock["selected"].items():
        if v["lr_name"] is None:
            v["lr_name"] = next(n for n, x in P.SETTINGS["learning_rates"].items() if x == v["lr"])
    dump(lock, j("configs", "selection_lock.json"))
    print(json.dumps({a: (v["id"], v["S"], v["R"], v["floor_qualified"]) for a, v in lock["selected"].items()}))
    print("comparator", sel["strongest_comparator"], "|", sel["comparator_rule"])
    return True


def phase_ho_m0():
    ok = gen_job("m0_ho", GPUS[0], "reserve") and score_job("m0_ho", GPUS[0], "reserve")
    if not ok:
        return False
    sc = load(j("artifacts", "scores", "m0_ho.json"))
    out = {}
    for s in sc["sets"]:
        obj = s["label"].rsplit(".", 1)[1]
        es = AN.eligibility_summary(s["rows"], obj, sc["label_ids"], "ho")
        out[obj] = es
        print("ho M0 eligibility (informational)", obj, es["counts"])
    dump(out, j("artifacts", "analysis", "ho_m0_eligibility.json"))
    return True


def ho_chain(tid, dev):
    ok = train_job(tid, dev, "main")
    if not ok:
        return False
    return gen_job(tid, dev, "reserve") and score_job(tid, dev, "reserve")


def require_committed_lock():
    """Test-model work may start only after the selection lock is committed (marker holds the commit SHA)."""
    m = j("configs", "selection_lock.committed.json")
    if not os.path.exists(m):
        raise SystemExit("REFUSED: selection lock not committed (configs/selection_lock.committed.json missing)")
    rec = load(m)
    import hashlib
    h = hashlib.sha256(open(j("configs", "selection_lock.json"), "rb").read()).hexdigest()
    if rec["lock_sha256"] != h:
        raise SystemExit("REFUSED: selection_lock.json differs from the committed lock")
    return load(j("configs", "selection_lock.json"))


def phase_ho_run():
    lock = require_committed_lock()
    make_ho_specs(lock)
    todo = [t["traj_id"] for t in CASES["heldout_trajectories"] if not ex(f"artifacts/scores/{t['traj_id']}.json")]
    todo.sort(key=lambda n: (P.ARMS.index(next(t["arm"] for t in CASES["heldout_trajectories"] if t["traj_id"] == n)), n))
    run_chains(todo, ho_chain)
    return not STOP.is_set()


def endpoint_table(thr):
    lock = load(j("configs", "selection_lock.json"))
    m0 = load(j("artifacts", "scores", "m0_ho.json"))
    label_ids = m0["label_ids"]
    per_object, tables = {}, {}
    for obj in P.HELDOUT_TARGETS:
        m0_rows = next(s for s in m0["sets"] if s["label"] == f"m0.ho.{obj}")["rows"]
        ep = {a: {} for a in P.ARMS}
        for t in CASES["heldout_trajectories"]:
            if t["target"] != obj:
                continue
            p = j("artifacts", "scores", t["traj_id"] + ".json")
            if os.path.exists(p):
                ep[t["arm"]][t["train_seed"]] = load(p)["sets"][0]["rows"]
        E = AN.build_object_table(obj, m0_rows, ep, label_ids, thr=thr)
        tables[obj] = E
        per_object[obj] = AN.object_predicates(obj, E, lock["strongest_comparator"])
    return lock, per_object, tables


def phase_final():
    lock, per_object, tables = endpoint_table(P.THRESHOLD)
    decision = AN.final_decision(per_object, lock["strongest_comparator"])
    sens = {}
    for thr in P.SENSITIVITY_THRESHOLDS:
        _, po, tb = endpoint_table(thr)
        sens[str(thr)] = {"decision_if_this_threshold_were_primary": AN.final_decision(po, lock["strongest_comparator"]),
                          "per_object": po, "tables": tb}
    dump({"decision": decision, "comparator": lock["strongest_comparator"], "per_object": per_object, "tables": tables,
          "sensitivity": sens, "utc": utc()}, j("artifacts", "analysis", "final_results.json"))
    print("DECISION", decision)
    return True


PHASES = {"techbatch": phase_techbatch, "dev_m0": phase_dev_m0, "dev_train": phase_dev_train, "dev_select": phase_dev_select,
          "ho_m0": phase_ho_m0, "ho_run": phase_ho_run, "final": phase_final}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    SUFFIX = a.suffix
    ok = PHASES[a.phase]()
    print("PHASE", a.phase, "OK" if ok else "NOT_OK", flush=True)
    sys.exit(0 if ok else 5)
