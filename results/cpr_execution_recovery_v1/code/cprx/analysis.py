"""FROZEN analysis predicates for cpr_execution_recovery_v1 (executable; committed before outcomes).

Definitions (fixed):
  * A *row* is one generated PNG with its output flags and each detector's full predictions.
  * Per detector t: a row is VALID iff flags.valid; MISSING iff it is valid-by-flags but detector t has no
    prediction record for it (technical error). Invalid/collapse rows are NOT missing: they count as failures
    (never S, never R) and are never dropped.
  * target case: S-success  = valid AND no detection of the TARGET class label with score > thr (primary thr 0.30; 0.20/0.50 sensitivity only).
  * allowed case: R-success = valid AND detection of THAT CASE'S OWN (prompted) allowed-class label.
  * Detector-specific M0-eligible set: M0 image valid and target case -> target detected; allowed case -> own class
    detected. Denominators are fixed by M0 eligibility. Common-intersection set = eligible under BOTH detectors.
  * S = successes / eligible target cases; R = successes / eligible allowed cases.
"""
import hashlib
import math

import numpy as np

from cprx import protocol as P


# ------------------------------------------------------------------ per-row measurement
def _labs(det, thr):
    return {l for s, l in zip(det["scores"], det["labels"]) if s > thr}


def measure(row, tag, target_cls, label_ids, thr=P.THRESHOLD):
    """-> dict(valid, missing, target_det, own_det) for one detector at score threshold thr."""
    valid = bool(row["flags"]["valid"])
    det = row.get("detections", {}).get(tag)
    if not valid:
        return {"valid": False, "missing": False, "target_det": False, "own_det": False}
    if det is None or "error" in det:
        return {"valid": True, "missing": True, "target_det": None, "own_det": None}
    labs = _labs(det, thr)
    tl = label_ids[tag][target_cls]
    ol = label_ids[tag][row["cls"]]
    return {"valid": True, "missing": False, "target_det": tl in labs, "own_det": ol in labs}


def outcome(row, tag, target_cls, label_ids, thr=P.THRESHOLD):
    """-> (success in {1,0,None}). None = missing measurement."""
    m = measure(row, tag, target_cls, label_ids, thr)
    if m["missing"]:
        return None
    if row["kind"] == "target":
        return 1 if (m["valid"] and not m["target_det"]) else 0
    return 1 if (m["valid"] and m["own_det"]) else 0


# ------------------------------------------------------------------ M0 eligibility
def m0_eligible(m0_rows, target_cls, label_ids, tag, thr=P.THRESHOLD):
    """case_id -> bool (eligible) or None (missing M0 measurement)."""
    out = {}
    for r in m0_rows:
        m = measure(r, tag, target_cls, label_ids, thr)
        if m["missing"]:
            out[r["case_id"]] = None
        elif not m["valid"]:
            out[r["case_id"]] = False
        else:
            out[r["case_id"]] = bool(m["target_det"]) if r["kind"] == "target" else bool(m["own_det"])
    return out


def eligibility_summary(m0_rows, target_cls, label_ids, stage, thr=P.THRESHOLD):
    """Per-detector + common eligibility counts and the stage's gate predicate."""
    spec = P.DEV_ELIG if stage == "dev" else P.HO_ELIG
    kind = {r["case_id"]: (r["kind"], r["cls"]) for r in m0_rows}
    el = {t: m0_eligible(m0_rows, target_cls, label_ids, t, thr) for t in (P.PRIMARY_DET, P.SECONDARY_DET)}
    common = {c: (el[P.PRIMARY_DET][c] is True and el[P.SECONDARY_DET][c] is True) for c in kind}
    missing = sorted(c for t in el for c, v in el[t].items() if v is None)

    def counts(flags):
        tgt = sum(1 for c, v in flags.items() if v is True and kind[c][0] == "target")
        allowed = sum(1 for c, v in flags.items() if v is True and kind[c][0] == "allowed")
        per = {}
        for c, v in flags.items():
            if kind[c][0] == "allowed":
                per[kind[c][1]] = per.get(kind[c][1], 0) + (1 if v is True else 0)
        return {"target": tgt, "allowed": allowed, "per_allowed_class": per}

    summ = {"primary": counts(el[P.PRIMARY_DET]), "secondary": counts(el[P.SECONDARY_DET]), "common": counts(common)}

    def ok(c):
        r = c["target"] >= spec["target_min"] and all(v >= spec["per_allowed_class_min"] for v in c["per_allowed_class"].values())
        if stage == "dev":
            r = r and c["allowed"] >= spec["allowed_min"]
        return bool(r)

    gate = ok(summ["primary"]) and ok(summ["common"]) and not missing
    return {"stage": stage, "spec": spec, "counts": summ, "missing_measurements": missing, "pass": gate,
            "eligible": {"primary": el[P.PRIMARY_DET], "secondary": el[P.SECONDARY_DET], "common": common}}


# ------------------------------------------------------------------ rates
def case_vector(rows, tag, target_cls, label_ids, case_ids, thr=P.THRESHOLD):
    """list of success {1,0,None} aligned with case_ids."""
    by = {r["case_id"]: r for r in rows}
    return [outcome(by[c], tag, target_cls, label_ids, thr) if c in by else None for c in case_ids]


def rates(rows, tag, target_cls, label_ids, eligible, kinds, thr=P.THRESHOLD):
    """S and R on the fixed eligible set. Returns dict with counts; None rates if any needed measurement missing."""
    tgt = [c for c, v in eligible.items() if v is True and kinds[c] == "target"]
    alw = [c for c, v in eligible.items() if v is True and kinds[c] == "allowed"]
    vt = case_vector(rows, tag, target_cls, label_ids, tgt, thr)
    va = case_vector(rows, tag, target_cls, label_ids, alw, thr)
    miss = [c for c, v in zip(tgt + alw, vt + va) if v is None]
    res = {"n_target": len(tgt), "n_allowed": len(alw), "missing": miss}
    if miss or not tgt or not alw:
        res.update({"S": None, "R": None})
    else:
        res.update({"S": sum(vt) / len(tgt), "R": sum(va) / len(alw), "S_num": sum(vt), "R_num": sum(va)})
    return res


# ------------------------------------------------------------------ development selection
def _tie_key(c):
    return (c["states"], c["lr"], c["id"])


FALLBACK = {"lr": 5e-4, "states": 250}


def select_for_arm(cands):
    """cands: list of dict(id, arm, lr, states, S, R). Rule: among S>=0.80 maximize R (ties: fewer states, lower lr,
    stable id). If none: highest S then R (same ties), efficacy_unmatched. If no candidate has defined metrics: the
    predeclared 5e-4/250 fallback, marked measurement_limited. Candidates with undefined R but S>=0.80 are not floor-qualified."""
    q = [c for c in cands if c["S"] is not None and c["R"] is not None and c["S"] >= P.S_FLOOR - 1e-12]
    if q:
        best = sorted(q, key=lambda c: (-c["R"], ) + _tie_key(c))[0]
        return dict(best, floor_qualified=True, efficacy_unmatched=False, measurement_limited=False)
    valid = [c for c in cands if c["S"] is not None and c["R"] is not None]
    if valid:
        best = sorted(valid, key=lambda c: (-c["S"], -c["R"]) + _tie_key(c))[0]
        return dict(best, floor_qualified=False, efficacy_unmatched=True, measurement_limited=False)
    fb = [c for c in cands if c["lr"] == FALLBACK["lr"] and c["states"] == FALLBACK["states"]]
    best = dict(fb[0]) if fb else {"id": None, "arm": cands[0]["arm"], **FALLBACK, "S": None, "R": None}
    return dict(best, floor_qualified=False, efficacy_unmatched=True, measurement_limited=True)


def dev_selection(cands_by_arm):
    """-> per-arm selection and the single primary non-CPR comparator (same rule on the arm-selected candidates;
    centered_radial when the ranking is undefined). Never closes the study."""
    sel = {a: select_for_arm(c) for a, c in cands_by_arm.items()}
    out = {"selected": sel, "all_candidates": cands_by_arm}
    ctrl = [dict(c, arm=a) for a, c in sel.items() if a != "cpr"]
    defined = [c for c in ctrl if c["S"] is not None and c["R"] is not None and not c["measurement_limited"]]
    qual = [c for c in defined if c["floor_qualified"]]
    if qual:
        best = sorted(qual, key=lambda c: (-c["R"],) + _tie_key(c) + (c["arm"],))[0]
        out["comparator_rule"] = "max R among S>=0.80 arm-selected controls"
    elif defined:
        best = sorted(defined, key=lambda c: (-c["S"], -c["R"]) + _tie_key(c) + (c["arm"],))[0]
        out["comparator_rule"] = "no control reached S>=0.80; highest S then R (efficacy unmatched)"
    else:
        best = {"arm": "centered_radial"}
        out["comparator_rule"] = "ranking undefined; predeclared fallback centered_radial"
    out["strongest_comparator"] = best["arm"]
    out["status"] = "DEVELOPMENT_SELECTED"
    out["proceed_to_heldout"] = True
    return out


# ------------------------------------------------------------------ bootstrap
def _panel_index(kinds_cls, ids):
    t = [i for i, c in enumerate(ids) if kinds_cls[c][0] == "target"]
    by = {}
    for i, c in enumerate(ids):
        if kinds_cls[c][0] == "allowed":
            by.setdefault(kinds_cls[c][1], []).append(i)
    return t, by


def paired_bootstrap(S_by_arm, R_by_arm, ids, kinds_cls, comps, object_name, n_boot=None):
    """S_by_arm[arm]: np.array over ids (case-level value = mean over trained seeds; only used for target ids),
    R_by_arm likewise (allowed ids). Resample target panel and allowed panel (stratified by class) separately with
    replacement; identical resampled indices for CPR and every control (paired)."""
    n_boot = n_boot or P.N_BOOT
    seed_seq = [P.BOOTSTRAP_SEED, int(hashlib.sha256(object_name.encode()).hexdigest()[:6], 16)]
    rng = np.random.default_rng(seed_seq)
    t_pos, by = _panel_index(kinds_cls, ids)
    t_pos = np.array(t_pos)
    draws_t = rng.integers(0, len(t_pos), size=(n_boot, len(t_pos)))
    class_draws = {c: rng.integers(0, len(p), size=(n_boot, len(p))) for c, p in sorted(by.items())}
    n_allowed = sum(len(p) for p in by.values())

    def boot_S(vec):
        v = np.asarray(vec, float)[t_pos]
        return v[draws_t].mean(axis=1)

    def boot_R(vec):
        v = np.asarray(vec, float)
        tot = np.zeros(n_boot)
        for c, p in sorted(by.items()):
            vv = v[np.array(p)]
            tot += vv[class_draws[c]].sum(axis=1)
        return tot / n_allowed

    bS = {a: boot_S(S_by_arm[a]) for a in S_by_arm}
    bR = {a: boot_R(R_by_arm[a]) for a in R_by_arm}
    out = {}
    lo_q, hi_q = P.BOOT_PCT
    for a in comps:
        dS, dR = bS["cpr"] - bS[a], bR["cpr"] - bR[a]
        out[a] = {"deltaS_ci": [float(np.percentile(dS, lo_q)), float(np.percentile(dS, hi_q))],
                  "deltaR_ci": [float(np.percentile(dR, lo_q)), float(np.percentile(dR, hi_q))]}
    return out


# ------------------------------------------------------------------ decision
def promising_dir(dS, dR):
    return dS >= P.DELTA_S_MARGIN - 1e-12 and dR > 0


def noadv_dir(dS, dR):
    return dS <= P.NA_DELTA_S_MAX + 1e-12 and dR < P.NA_DELTA_R_MAX - 1e-12


def object_predicates(obj, E, comparator):
    """E: dict from build_object_endpoint_table. Returns {'promising':{...}, 'no_advantage':{...}} booleans + detail."""
    if E.get("missing"):
        return {"missing": True, "eligibility_pass": False}
    pri = E["primary"]
    cpr_S = pri["S"]["cpr"]
    elig_ok = bool(E["eligibility"]["pass"])
    prom_detail, prom_ok = {}, cpr_S >= P.S_FLOOR - 1e-12 and elig_ok
    for a in P.NON_CPR_COMPARATORS:
        dS, dR = pri["S"]["cpr"] - pri["S"][a], pri["R"]["cpr"] - pri["R"][a]
        ci = E["bootstrap"][a]
        seeds_ok = all(promising_dir(pri["seed_S"][s]["cpr"] - pri["seed_S"][s][a],
                                     pri["seed_R"][s]["cpr"] - pri["seed_R"][s][a]) for s in pri["seed_S"])
        c = {"dS": dS, "dR": dR, "dS_lo": ci["deltaS_ci"][0], "dR_lo": ci["deltaR_ci"][0],
             "point_ok": dS >= P.DELTA_S_MARGIN - 1e-12 and dR >= P.DELTA_R_PROMISING - 1e-12,
             "lower_ok": ci["deltaS_ci"][0] >= P.DELTA_S_MARGIN - 1e-12 and ci["deltaR_ci"][0] > 0,
             "per_seed_ok": seeds_ok,
             "corroborated": all(promising_dir(E[k]["S"]["cpr"] - E[k]["S"][a], E[k]["R"]["cpr"] - E[k]["R"][a])
                                 for k in ("secondary", "common_primary", "common_secondary"))}
        c["ok"] = c["point_ok"] and c["lower_ok"] and c["per_seed_ok"] and c["corroborated"]
        prom_detail[a] = c
        prom_ok &= c["ok"]
    a = comparator
    na_detail = {}
    na_ok = False
    if a is not None:
        dS, dR = pri["S"]["cpr"] - pri["S"][a], pri["R"]["cpr"] - pri["R"][a]
        ci = E["bootstrap"][a]
        seeds_ok = all(noadv_dir(pri["seed_S"][s]["cpr"] - pri["seed_S"][s][a],
                                 pri["seed_R"][s]["cpr"] - pri["seed_R"][s][a]) for s in pri["seed_S"])
        na_detail = {"comparator": a, "comparator_S": pri["S"][a], "dS": dS, "dR": dR,
                     "dS_hi": ci["deltaS_ci"][1], "dR_hi": ci["deltaR_ci"][1],
                     "comparator_S_ok": pri["S"][a] >= P.S_FLOOR - 1e-12,
                     "point_ok": noadv_dir(dS, dR),
                     "upper_ok": ci["deltaS_ci"][1] <= P.NA_DELTA_S_MAX + 1e-12 and ci["deltaR_ci"][1] < P.NA_DELTA_R_MAX - 1e-12,
                     "per_seed_ok": seeds_ok,
                     "corroborated": all(noadv_dir(E[k]["S"]["cpr"] - E[k]["S"][a], E[k]["R"]["cpr"] - E[k]["R"][a])
                                         for k in ("secondary", "common_primary", "common_secondary"))}
        na_detail["secondary_denominators_adequate"] = bool(E["secondary_elig_ok"])
        na_ok = elig_ok and all(na_detail[k] for k in ("comparator_S_ok", "point_ok", "upper_ok", "per_seed_ok", "corroborated",
                                                      "secondary_denominators_adequate"))
    return {"missing": False, "eligibility_pass": elig_ok, "promising": {"ok": bool(prom_ok), "cpr_S": cpr_S, "cpr_S_ok": cpr_S >= P.S_FLOOR - 1e-12,
                                              "per_comparator": prom_detail},
            "no_advantage": {"ok": bool(na_ok), **na_detail}}


def final_decision(per_object, comparator):
    objs = list(per_object.values())
    if any(o.get("missing") for o in objs) or len(objs) != len(P.HELDOUT_TARGETS):
        return "MEASUREMENT_LIMITED"
    if any(not o["eligibility_pass"] for o in objs):
        return "MEASUREMENT_LIMITED"
    if all(o["promising"]["ok"] for o in objs):
        return "PROVISIONAL_DETECTOR_DEFINED_ADVANTAGE"
    if comparator is not None and all(o["no_advantage"]["ok"] for o in objs):
        return "NO_ADVANTAGE_IN_THIS_SCREEN"
    return "INCONCLUSIVE"


# ------------------------------------------------------------------ held-out object table
def build_object_table(obj, m0_rows, endpoint_rows, label_ids, n_boot=None, thr=P.THRESHOLD):
    """endpoint_rows[arm][seed] = list of scored rows. Returns E (see object_predicates)."""
    esum = eligibility_summary(m0_rows, obj, label_ids, "ho", thr)
    kinds = {r["case_id"]: r["kind"] for r in m0_rows}
    kinds_cls = {r["case_id"]: (r["kind"], r["cls"]) for r in m0_rows}
    el = esum["eligible"]
    sets = {"primary": (P.PRIMARY_DET, el["primary"]), "secondary": (P.SECONDARY_DET, el["secondary"]),
            "common_primary": (P.PRIMARY_DET, el["common"]), "common_secondary": (P.SECONDARY_DET, el["common"])}
    seeds = sorted({s for a in endpoint_rows for s in endpoint_rows[a]})
    spec = P.HO_ELIG
    sec = esum["counts"]["secondary"]
    E = {"object": obj, "threshold": thr, "eligibility": {k: v for k, v in esum.items() if k != "eligible"}, "missing": False,
         "missing_detail": [],
         "secondary_elig_ok": sec["target"] >= spec["target_min"] and all(v >= spec["per_allowed_class_min"] for v in sec["per_allowed_class"].values())}
    for name, (tag, elig) in sets.items():
        tab = {"S": {}, "R": {}, "seed_S": {}, "seed_R": {}, "n_target": None, "n_allowed": None}
        for s in seeds:
            tab["seed_S"][s], tab["seed_R"][s] = {}, {}
        for arm in P.ARMS:
            Ss, Rs = [], []
            for s in seeds:
                rows = endpoint_rows.get(arm, {}).get(s)
                if rows is None:
                    E["missing"] = True
                    E["missing_detail"].append(f"endpoint {arm}/{obj}/{s} absent")
                    continue
                r = rates(rows, tag, obj, label_ids, elig, kinds, thr)
                tab["n_target"], tab["n_allowed"] = r["n_target"], r["n_allowed"]
                if r["S"] is None:
                    E["missing"] = True
                    E["missing_detail"].append(f"{name} {arm}/{obj}/{s}: {r['missing'] or 'empty panel'}")
                    continue
                tab["seed_S"][s][arm], tab["seed_R"][s][arm] = r["S"], r["R"]
                Ss.append(r["S"]); Rs.append(r["R"])
            if len(Ss) == len(seeds):
                tab["S"][arm], tab["R"][arm] = float(np.mean(Ss)), float(np.mean(Rs))
        E[name] = tab
    if E["missing"]:
        return E
    # paired bootstrap on the primary-eligible case set; case value = mean over trained seeds
    ids = [c for c, v in el["primary"].items() if v is True]
    ids.sort()
    S_vec, R_vec = {}, {}
    for arm in P.ARMS:
        per_seed = []
        for s in seeds:
            by = {r["case_id"]: r for r in endpoint_rows[arm][s]}
            per_seed.append([outcome(by[c], P.PRIMARY_DET, obj, label_ids, thr) for c in ids])
        arr = np.array(per_seed, float).mean(axis=0)
        S_vec[arm], R_vec[arm] = arr, arr
    E["bootstrap"] = paired_bootstrap(S_vec, R_vec, ids, kinds_cls, P.NON_CPR_COMPARATORS, obj, n_boot)
    E["bootstrap_spec"] = {"n_boot": n_boot or P.N_BOOT, "pct": P.BOOT_PCT, "seed": P.BOOTSTRAP_SEED,
                           "units": "cases; target and allowed panels resampled separately; allowed stratified by class; "
                                    "seed-averaged per case first; paired indices across arms"}
    return E
