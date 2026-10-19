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

from cprd2 import analysis as AN
from cprd2 import ledger as LG
from cprd2 import protocol as P
from cprd2.common import ROOT, dump, load, utc

PY = os.environ["CPRD_PY"]
S = os.path.join(ROOT, "scripts")
A = os.path.join(ROOT, "artifacts")
GPUS = [0, 1, 2, 3]
CASES = load(os.path.join(ROOT, "configs", "cases.json"))
SUFFIX = ""
STOP = threading.Event()
FAILURES = []


def j(*p):
    return os.path.join(ROOT, *p)


def ex(path):
    return os.path.exists(j(path))


def run(job, bucket, devs, argv, wall, kind="planned", ip=0, it=0, cycle=None, workers=None, note=None):
    job = job + SUFFIX
    if cycle is None and os.environ.get("CPRD2_CYCLE"):
        cycle = int(os.environ["CPRD2_CYCLE"])
        kind = "repair"
    while True:
        try:
            r = LG.run_job(job, bucket, devs, argv, max_wall_s=wall, grace_s=60,
                           workers=workers if workers is not None else (2 if devs else 1),
                           kind=kind, images_primary=ip, images_technical=it, cycle=cycle, note=note)
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
        STOP.set()
        dump(FAILURES, j("artifacts", "orch", f"failures{SUFFIX or '.r0'}.json"))
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
        return [{"label": "tech.native_gate", "adapter_dir": f"artifacts/adapters/{P.GATE_NATIVE_TRAJ}/terminal",
                 "image_dir": "images/tech/native_gate", "cases": c["technical_cases"], "image_class": "technical"}]
    if name.startswith("dev."):
        return [{"label": f"{name}.{ck}", "adapter_dir": f"artifacts/adapters/{name}/{ck}",
                 "image_dir": f"images/dev/{name}/{ck}", "cases": dev, "image_class": "primary"}
                for ck in P.CHECKPOINT_NAMES]
    if name.startswith("ho."):
        t = next(t for t in c["heldout_trajectories"] if t["traj_id"] == name)
        return [{"label": name, "adapter_dir": f"artifacts/adapters/{name}/terminal",
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
               wall=900 + 25 * n, ip=0 if is_tech else n, it=n if is_tech else 0)[0]


def score_job(name, dev, bucket):
    out = j("artifacts", "scores", name + ".json")
    if os.path.exists(out):
        return True
    plan = write_plan(name)
    n = n_images(name)
    return run(f"score.{name}", bucket, [dev], [PY, j("scripts", "score_pngs.py"), plan, out], wall=900 + 12 * n)[0]


def train_job(tid, dev, bucket, gate=False):
    if ex(f"artifacts/traj/{tid}/train_record.json"):
        return True
    argv = [PY, j("scripts", "run_trajectory.py"), "train", "--traj-id", tid] + (["--gate"] if gate else [])
    return run(f"train.{tid}", bucket, [dev], argv, wall=3000)[0]


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
def phase_synthetic():
    ok, _ = run("gate.synthetic", "entry", [], [PY, j("scripts", "gate_synthetic.py")], wall=1800)
    return ok


def phase_entry_gpu():
    """GPU gate trajectory (first planned native dev trajectory) and detector qualification, concurrently."""
    res = {}

    def a():
        res["gpu_gate"] = train_job(P.GATE_NATIVE_TRAJ, GPUS[0], "entry", gate=True)

    def b():
        time.sleep(30)
        if ex("artifacts/qualification/qualification_summary.json"):
            res["qual"] = True
            return
        ok, _ = run("qualify", "entry", [GPUS[1]], [PY, j("scripts", "qualify.py")], wall=3000)
        if ok or ex("artifacts/qualification/qualification_summary.json"):
            LG.add_coco(333, "unique qualification photographs read (both detectors)")
        res["qual"] = ok
    ta, tb = threading.Thread(target=a), threading.Thread(target=b)
    ta.start(); tb.start(); ta.join(); tb.join()
    print(res)
    return all(res.values())


def phase_replay():
    ok, _ = run("gate.replay", "entry", [], [PY, j("scripts", "gate_replay.py")], wall=1800)
    return ok


def phase_techbatch():
    return gen_job("tech", GPUS[0], "entry") and score_job("tech", GPUS[0], "entry")


def entry_status():
    def f(path, key):
        p = j(path)
        return bool(load(p).get(key)) if os.path.exists(p) else False
    st = {"synthetic_gate": f("artifacts/gate/synthetic_gate.json", "all_pass"),
          "gpu_instrumentation": f("artifacts/gate/gpu_instrumentation.json", "all_pass"),
          "replay_gate": f("artifacts/gate/replay_gate.json", "all_pass"),
          "qualification": f("artifacts/qualification/qualification_summary.json", "entry_qualification_pass"),
          "projection_fits": f("artifacts/projection.json", "fits")}
    st["entry_pass"] = all(st.values())
    return st


def phase_project():
    from cprd2 import project
    pr = project.project()
    dump(pr, j("artifacts", "projection.json"))
    print(json.dumps({k: pr[k] for k in ("projected_gpu_h_with_margin", "fits", "usable_for_non_reserve_h")}))
    return pr["fits"]


def phase_entry_decision():
    st = entry_status()
    dump(dict(st, utc=utc()), j("artifacts", "entry_decision.json"))
    print(st)
    return st["entry_pass"]


def dev_names():
    return [t["traj_id"] for t in CASES["dev_trajectories"]]


def phase_dev_m0():
    ok = gen_job("m0_dev", GPUS[0], "main") and score_job("m0_dev", GPUS[0], "main")
    if not ok:
        return False
    sc = load(j("artifacts", "scores", "m0_dev.json"))
    rows = sc["sets"][0]["rows"]
    es = AN.eligibility_summary(rows, P.DEV_TARGET, sc["label_ids"], "dev")
    dump({k: v for k, v in es.items()}, j("artifacts", "analysis", "dev_m0_eligibility.json"))
    print("dev M0 eligibility", es["counts"], "PASS" if es["pass"] else "FAIL")
    return es["pass"]


def dev_chain(tid, dev):
    return train_job(tid, dev, "main") and gen_job(tid, dev, "main") and score_job(tid, dev, "main")


def phase_dev_train():
    todo = [t for t in dev_names() if not ex(f"artifacts/scores/{t}.json")]
    # the gate trajectory is trained already; its chain only generates and scores
    r = run_chains(todo, dev_chain)
    return all(r.values()) and not STOP.is_set()


def lr_of(name):
    return P.SETTINGS["learning_rates"][name]


def dev_candidates(sc_name_to_rows, label_ids, elig):
    kinds = {}
    cands = {a: [] for a in P.ARMS}
    full = []
    for t in CASES["dev_trajectories"]:
        sc = load(j("artifacts", "scores", t["traj_id"] + ".json"))
        for s in sc["sets"]:
            ck = s["label"].rsplit(".", 1)[1]
            kinds = {r["case_id"]: r["kind"] for r in s["rows"]}
            rec = {"id": s["label"], "arm": t["arm"], "lr_name": t["lr_name"], "lr": t["lr"], "checkpoint": ck,
                   "states": P.CHECKPOINT_STATES[ck]}
            for key, tag, el in (("primary", P.PRIMARY_DET, elig["primary"]), ("secondary", P.SECONDARY_DET, elig["secondary"]),
                                 ("common_primary", P.PRIMARY_DET, elig["common"]), ("common_secondary", P.SECONDARY_DET, elig["common"])):
                r = AN.rates(s["rows"], tag, P.DEV_TARGET, sc["label_ids"], el, kinds)
                rec[key] = r
            rec["S"], rec["R"] = rec["primary"]["S"], rec["primary"]["R"]
            n_inv = sum(1 for r in s["rows"] if not r["flags"]["valid"])
            rec["n_invalid_outputs"] = n_inv
            cands[t["arm"]].append(rec)
            full.append(rec)
    return cands, full


def phase_dev_select():
    el = load(j("artifacts", "analysis", "dev_m0_eligibility.json"))["eligible"]
    cands, full = dev_candidates(None, None, el)
    if any(c["S"] is None for c in full):
        dump({"status": "DEV_MEASUREMENT_MISSING", "candidates": full}, j("artifacts", "analysis", "dev_selection.json"))
        return False
    sel = AN.dev_selection(cands)
    dump(sel, j("artifacts", "analysis", "dev_selection.json"))
    lock = {"utc": utc(), "status": sel["status"], "strongest_comparator": sel["strongest_comparator"],
            "proceed_to_heldout": sel["proceed_to_heldout"],
            "selected": {a: {"id": c["id"], "lr_name": c["lr_name"], "lr": c["lr"], "states": c["states"],
                             "checkpoint": c["checkpoint"], "S": c["S"], "R": c["R"],
                             "floor_qualified": c["floor_qualified"], "efficacy_unmatched": c["efficacy_unmatched"]}
                         for a, c in sel["selected"].items()}}
    dump(lock, j("configs", "selection_lock.json"))
    print(json.dumps({a: (v["id"], round(v["S"], 3), round(v["R"], 3), v["floor_qualified"]) for a, v in lock["selected"].items()}))
    print("status", sel["status"], "comparator", sel["strongest_comparator"])
    return True


def phase_ho_m0():
    ok = gen_job("m0_ho", GPUS[0], "reserve") and score_job("m0_ho", GPUS[0], "reserve")
    if not ok:
        return False
    sc = load(j("artifacts", "scores", "m0_ho.json"))
    allok = True
    out = {}
    for s in sc["sets"]:
        obj = s["label"].rsplit(".", 1)[1]
        es = AN.eligibility_summary(s["rows"], obj, sc["label_ids"], "ho")
        out[obj] = es
        print("ho M0", obj, es["counts"], "PASS" if es["pass"] else "FAIL")
        allok &= es["pass"]
    dump(out, j("artifacts", "analysis", "ho_m0_eligibility.json"))
    return allok


def ho_chain(tid, dev):
    return train_job(tid, dev, "main") and gen_job(tid, dev, "reserve") and score_job(tid, dev, "reserve")


def phase_ho_run():
    todo = [t["traj_id"] for t in CASES["heldout_trajectories"] if not ex(f"artifacts/scores/{t['traj_id']}.json")]
    r = run_chains(todo, ho_chain)
    return all(r.values()) and not STOP.is_set()


def phase_final():
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
        E = AN.build_object_table(obj, m0_rows, ep, label_ids)
        tables[obj] = E
        per_object[obj] = AN.object_predicates(obj, E, lock["strongest_comparator"])
    decision = AN.final_decision(per_object, lock["strongest_comparator"])
    dump({"decision": decision, "comparator": lock["strongest_comparator"], "per_object": per_object, "tables": tables,
          "utc": utc()}, j("artifacts", "analysis", "final_results.json"))
    print("DECISION", decision)
    return True


PHASES = {"synthetic": phase_synthetic, "entry_gpu": phase_entry_gpu, "replay": phase_replay, "techbatch": phase_techbatch,
          "project": phase_project, "entry_decision": phase_entry_decision, "dev_m0": phase_dev_m0,
          "dev_train": phase_dev_train, "dev_select": phase_dev_select, "ho_m0": phase_ho_m0, "ho_run": phase_ho_run,
          "final": phase_final}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=sorted(PHASES))
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    SUFFIX = a.suffix
    ok = PHASES[a.phase]()
    print("PHASE", a.phase, "OK" if ok else "NOT_OK", flush=True)
    sys.exit(0 if ok else 5)
