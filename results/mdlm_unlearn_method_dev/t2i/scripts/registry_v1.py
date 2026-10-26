"""REGISTRY v1 -- the single machine-readable result registry the audit asks for (audit section 7).

Every current table is generated FROM THIS, so prose counters cannot drift from the records again.

Each attack row is keyed by:
    checkpoint identity (sha256 of the actual unet.pt, not its path)
  + target concept
  + implementation version (the defence script's family and recorded hparams)
  + split version
  + attack family
  + attack budget (steps)
  + seed / lr

and carries explicit ADMISSIBILITY flags. Scientific aggregates exclude:

  * quarantined records          -- outputs/CONTAMINATED_* (label collision, FINDING-level note)
  * VOID post-attack utility     -- the 136 runs in run_inventory.json whose post-attack utility
                                    was computed with the text encoder omitted (ERRATUM_text_encoder
                                    _eval_bug). Their T/H numbers are unaffected and stay admissible
                                    for recovery; only their utility half is void. Both are flagged
                                    SEPARATELY so neither gets silently dropped nor silently used.
  * vacuous (no erasure to reverse) -- pre-attack target suppression already at/near the ceiling, so
                                    H_after measures the BASE model's ability to draw the concept,
                                    not a defence failing. See ERRATUM_uce_lambda_scale.md section 6.

Run: python scripts/registry_v1.py            -> outputs/registry_v1.json
     python scripts/registry_v1.py --tables   -> also docs/TABLES_v1_generated.md
"""
import argparse
import collections
import glob
import hashlib
import json
import os

P = "/data/bijaypandey/t2i_unlearn"
OUT = f"{P}/outputs"
REGISTRY_VERSION = "v1"

FAMILY = {
    "t2i_relearn_attack": "full_unet",
    "t2i_lora_relearn_attack": "lora",
    "t2i_textual_inversion_attack": "textual_inversion",
    "t2i_dreambooth_attack": "dreambooth",
}

# Defence family inferred from the adapter label. Order matters: longest/most specific first.
DEF_FAMILY = [
    ("utrap", "UTRAP (ours)"),
    ("wsr2", "WSR-anchor (ours, adapts 2505.22310's idea with REVERSED direction -- see "
             "ERRATUM_wsr_direction.md)"),
    ("wsr", "WSR-anchor (ours, adapts 2505.22310's idea with REVERSED direction -- see "
            "ERRATUM_wsr_direction.md)"),
    ("gift", "GIFT port (2507.13598)"),
    ("imma", "IMMA port (2311.18815)"),
    ("meta", "Meta-Unlearning port, variant A (2410.12777; two Eq.8 terms absent -- see "
             "ERRATUM_meta_objective.md)"),
    ("mdu", "Meta-Unlearning port, variant A (2410.12777; two Eq.8 terms absent -- see "
            "ERRATUM_meta_objective.md)"),
    ("frp", "FRP/FRAG port (2608.25429)"),
    ("uce", "UCE closed-form (2308.14761)"),
    ("esd", "ESD"),
    ("adv", "AdvUnlearn port"),
    ("undo", "undo/damage composite (ours)"),
]

# Base model inferred from the label. Default is the pinned SD-1.4.
BASE_BY_TAG = [("sd15", "runwayml/stable-diffusion-v1-5"),
               ("bksdm", "nota-ai/bk-sdm-small")]

_SHA_CACHE = {}


def sha(path):
    """sha256 of a checkpoint. Absent files are recorded as ABSENT, never silently as equal."""
    if path in _SHA_CACHE:
        return _SHA_CACHE[path]
    full = path if os.path.isabs(path) else os.path.join(P, path)
    if not os.path.exists(full):
        v = {"sha256": None, "state": "ABSENT",
             "note": "checkpoint deleted (pre-2026-10-24 retention gap, RESOURCE_NOTE.md); "
                     "the run is re-derivable but not weight-verifiable"}
    else:
        h = hashlib.sha256()
        with open(full, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 22), b""):
                h.update(blk)
        v = {"sha256": h.hexdigest(), "state": "PRESENT", "bytes": os.path.getsize(full)}
    _SHA_CACHE[path] = v
    return v


def def_label(unet_path):
    if not unet_path:
        return "BASE(unmodified)"
    parts = unet_path.strip("/").split("/")
    return parts[-2] if len(parts) >= 2 else unet_path


def def_family(label):
    lo = label.lower()
    for tag, fam in DEF_FAMILY:
        if tag in lo:
            return fam
    return "unclassified"


def base_model(label):
    lo = label.lower()
    for tag, m in BASE_BY_TAG:
        if tag in lo:
            return m
    return "CompVis/stable-diffusion-v1-4"


def load_void_set():
    """run_inventory.json marks runs whose POST-ATTACK UTILITY half is void."""
    f = f"{OUT}/run_inventory.json"
    if not os.path.exists(f):
        return set(), 0
    inv = json.load(open(f))
    void = {e["file"] for e in inv
            if str(e.get("post_attack_eval_status", "")).startswith("VOID")}
    return void, len(inv)


def defence_builds():
    """label -> as-is measurement, from gen_eval records."""
    out = {}
    for f in sorted(glob.glob(f"{OUT}/gen_*.json")):
        try:
            d = json.load(open(f))
        except Exception as e:
            out.setdefault("_parse_errors", []).append([os.path.basename(f), str(e)])
            continue
        if d.get("study") != "t2i_gen_eval":
            continue
        lab, am = d.get("label"), d.get("active_target_metrics") or {}
        if not lab:
            continue
        out[lab] = {
            "file": os.path.basename(f),
            "unet": d.get("unet"),
            "checkpoint": sha(d.get("unet")) if d.get("unet") else None,
            "target": am.get("active_target") or (d.get("hparams") or {}).get("target"),
            "target_suppression_acc": am.get("target_suppression_acc"),
            "legacy_retain8": am.get("legacy_retain8"),
            "utility_all_nontarget": am.get("utility_all_nontarget"),
            "n_nontarget": am.get("n_nontarget"),
            "n_legacy_retain": am.get("n_legacy_retain"),
            "defence_family": def_family(lab),
            "base_model": base_model(lab),
        }
    return out


def build():
    void_files, inv_n = load_void_set()
    builds = defence_builds()
    rows, parse_errors = [], []

    for f in sorted(glob.glob(f"{OUT}/atk_*.json")) + sorted(
            glob.glob(f"{OUT}/CONTAMINATED_*/atk_*.json")):
        bn = os.path.basename(f)
        quarantined = "/CONTAMINATED_" in f
        try:
            d = json.load(open(f))
        except Exception as e:
            parse_errors.append([bn, str(e)])
            continue
        study = d.get("study", "?")
        fam = FAMILY.get(study)
        if fam is None:
            continue
        hp = d.get("hparams") or {}
        av = d.get("attack_validity") or {}
        rec = d.get("recovery") or {}
        pau = d.get("post_attack_utility") or {}
        unet = d.get("unet") or hp.get("unet")
        lab = def_label(unet)
        asis = builds.get(lab, {})

        # vacuous: nothing to reverse because the defence erased nothing.
        pre_t = asis.get("target_suppression_acc")
        vacuous = pre_t is not None and pre_t >= 0.95

        ret = pau.get("retained_capability") or {}
        rows.append({
            "run_id": d.get("label") or bn,
            "file": bn,
            "registry_version": REGISTRY_VERSION,
            # --- identity
            "defence_label": lab,
            "defence_family": def_family(lab),
            "base_model": base_model(lab),
            "checkpoint": sha(unet) if unet else {"sha256": None, "state": "BASE"},
            "target": d.get("concept") or hp.get("concept"),
            "split_version": "v1_frozen(H reused as DEF_TRAIN for gift/imma/wsr/utrap -- see "
                             "SPLIT_REGISTRY_v2)",
            # --- attack configuration
            "attack_family": fam,
            "attack_steps": hp.get("steps"),
            "attack_lr": hp.get("lr"),
            "attack_seed": hp.get("seed"),
            "attack_te": bool(hp.get("attack_te")),
            "lora_rank": hp.get("rank"),
            # --- gate / outcome
            "gate_threshold": av.get("gate"),
            "T_before": av.get("T_before"), "T_after": av.get("T_after"),
            "T_rise": av.get("T_rise"), "gate_valid": av.get("valid"),
            "H_before": rec.get("H_before"), "H_after": rec.get("H_after"),
            # --- as-is state of the attacked defence (for vacuity + paired reporting)
            "asis_target_suppression": pre_t,
            "asis_legacy_retain8": asis.get("legacy_retain8"),
            "asis_utility_all_nontarget": asis.get("utility_all_nontarget"),
            # --- paired utility, with its own validity
            "pau_status": pau.get("status"),
            "pau_before": ret.get("before"), "pau_after": ret.get("after"),
            # --- admissibility
            "quarantined": quarantined,
            "void_post_attack_utility": bn in void_files,
            "vacuous_no_erasure": vacuous,
            "admissible_for_recovery": (not quarantined) and (not vacuous),
            "admissible_for_paired_utility": (not quarantined) and (not vacuous)
                                             and bn not in void_files
                                             and pau.get("status") == "MEASURED",
        })

    reg = {
        "study": "result_registry",
        "registry_version": REGISTRY_VERSION,
        "generated_by": "scripts/registry_v1.py",
        "counts": {
            "attack_records_total": len(rows),
            "attack_records_quarantined": sum(r["quarantined"] for r in rows),
            "attack_records_void_post_attack_utility": sum(
                r["void_post_attack_utility"] for r in rows),
            "attack_records_vacuous_no_erasure": sum(r["vacuous_no_erasure"] for r in rows),
            "attack_records_admissible_for_recovery": sum(
                r["admissible_for_recovery"] for r in rows),
            "attack_records_admissible_for_paired_utility": sum(
                r["admissible_for_paired_utility"] for r in rows),
            "defence_builds_evaluated": len([k for k in builds if not k.startswith("_")]),
            "run_inventory_entries": inv_n,
            "json_files_under_outputs": len(glob.glob(f"{OUT}/**/*.json", recursive=True)),
            "parse_errors": len(parse_errors),
        },
        "admissibility_note":
            "attack_records_total is a count of FILES, not of admissible independent experiments. "
            "The audit's objection stands: use attack_records_admissible_for_recovery, and note "
            "that even that pools fixed-grid confirmatory runs with adaptive escalation -- see "
            "the per-arm tables, which separate them by budget.",
        "parse_errors": parse_errors,
        "defence_builds": builds,
        "attacks": rows,
    }
    json.dump(reg, open(f"{OUT}/registry_{REGISTRY_VERSION}.json", "w"), indent=1)
    return reg


def arm_table(reg, budget_split=True):
    """max-over-valid H per (defence build, attack family), separating budgets."""
    agg = collections.defaultdict(lambda: {"tested": 0, "valid": 0, "H_valid": [], "H_all": [],
                                           "budgets": collections.Counter()})
    for r in reg["attacks"]:
        if not r["admissible_for_recovery"]:
            continue
        key = (r["defence_label"], r["attack_family"],
               r["attack_steps"] if budget_split else "any")
        a = agg[key]
        a["tested"] += 1
        a["budgets"][r["attack_steps"]] += 1
        if r["H_after"] is not None:
            a["H_all"].append(r["H_after"])
        if r["gate_valid"]:
            a["valid"] += 1
            if r["H_after"] is not None:
                a["H_valid"].append(r["H_after"])
    return agg


def tables(reg):
    L = []
    c = reg["counts"]
    L.append("# Generated tables (registry %s)\n" % REGISTRY_VERSION)
    L.append("**Generated by `scripts/registry_v1.py` from `outputs/registry_v1.json`. Do not "
             "hand-edit: regenerate.** Every number below is computed from the raw records, which "
             "is the audit's section-7 requirement.\n")
    L.append("## 0. Record counts, with admissibility separated\n")
    L.append("| quantity | value |\n|---|---:|")
    for k in ("json_files_under_outputs", "attack_records_total", "attack_records_quarantined",
              "attack_records_void_post_attack_utility", "attack_records_vacuous_no_erasure",
              "attack_records_admissible_for_recovery",
              "attack_records_admissible_for_paired_utility", "defence_builds_evaluated",
              "run_inventory_entries", "parse_errors"):
        L.append(f"| `{k}` | {c[k]} |")
    L.append("\n" + reg["admissibility_note"] + "\n")

    agg = arm_table(reg)
    L.append("## 1. Per defence build x attack family x budget\n")
    L.append("`max H (valid)` is over GATE-VALID attacks only; `max H (ungated)` includes attacks "
             "whose T did not rise, which are **inconclusive, not evidence either way**.\n")
    L.append("| defence build | family | base model | attack family | steps | tested | gate-valid "
             "| max H (valid) | max H (ungated) |\n|---|---|---|---|---:|---:|---:|---:|---:|")
    for (lab, fam, steps), a in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                                    kv[0][2] or 0)):
        b = reg["defence_builds"].get(lab, {})
        mv = f"{max(a['H_valid']):.4f}" if a["H_valid"] else "—"
        mu = f"{max(a['H_all']):.4f}" if a["H_all"] else "—"
        L.append(f"| `{lab}` | {b.get('defence_family', def_family(lab))} | "
                 f"{(b.get('base_model') or base_model(lab)).split('/')[-1]} | {fam} | {steps} | "
                 f"{a['tested']} | {a['valid']} | {mv} | {mu} |")

    L.append("\n## 2. Defence builds as-is, with all three denominators shown\n")
    L.append("| build | family | base model | target suppression | retain8 (n=8) | "
             "all-non-target (n=11) |\n|---|---|---|---:|---:|---:|")
    for lab, b in sorted(reg["defence_builds"].items()):
        if lab.startswith("_"):
            continue
        def f(x):
            return "—" if x is None else f"{x:.4f}"
        L.append(f"| `{lab}` | {b['defence_family']} | {b['base_model'].split('/')[-1]} | "
                 f"{f(b['target_suppression_acc'])} | {f(b['legacy_retain8'])} | "
                 f"{f(b['utility_all_nontarget'])} |")

    L.append("\n## 3. Excluded records, named\n")
    for flag in ("quarantined", "vacuous_no_erasure"):
        bad = [r["run_id"] for r in reg["attacks"] if r[flag]]
        L.append(f"\n**{flag}** ({len(bad)}): " + (", ".join(f"`{x}`" for x in sorted(bad))
                                                   if bad else "none"))
    nv = [r["run_id"] for r in reg["attacks"] if r["void_post_attack_utility"]]
    L.append(f"\n**void_post_attack_utility** ({len(nv)}): their T/H recovery numbers remain "
             "admissible; only the paired-utility half is void. Not listed individually here — "
             "query `outputs/registry_v1.json`.\n")
    open(f"{P}/docs/TABLES_v1_generated.md", "w").write("\n".join(L) + "\n")
    return len(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()
    r = build()
    print(json.dumps(r["counts"], indent=1))
    if a.tables:
        print("wrote docs/TABLES_v1_generated.md,", tables(r), "lines")
