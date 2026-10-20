"""collect_matched.py — assemble the matched-comparison tables and apply the
PRE-DECLARED selection rule mechanically.

The rule is implemented here exactly as frozen in docs/BASELINE_MATCHED.md §3.2
(pre-declaration revision 2, sha256 5e9b628c…), so selection is executed by code rather
than by hand. Every grid point is printed whether or not it is selected.
"""
import collections
import glob
import json
import os
import sys

W = "/data/bijaypandey/mdlm_unlearn"
LABELS = ("DISCLOSE", "DEGENERATE", "ABSTAIN", "REPLACE", "IGNORANT")
ARM_ORDER = ["reference", "mdu_t1", "emdu", "npo", "sc_tf", "excl"]
ARM_NAME = {"reference": "memorized reference", "mdu_t1": "MDU", "emdu": "EMDU",
            "npo": "NPO", "sc_tf": "SC-TF", "excl": "exclusion target (ours)"}

# ── the pre-declared rule, verbatim ────────────────────────────────────────────
RETAIN_FLOOR = 0.90


def objective(r):
    """IGNORANT + ABSTAIN on the forget probes."""
    x = r["summary"]["forget_partition"]["rates"]
    return (x["IGNORANT"] or 0) + (x["ABSTAIN"] or 0)


def admissible(r):
    return (r["summary"]["retain_accuracy"]["all"] or 0) >= RETAIN_FLOOR


def sort_key(r):
    f = r["summary"]["forget_partition"]["rates"]
    mar = r["summary"]["forget_modal_answers"]["overall"]["MAR_all"] or 0
    hp = r["hyperparameter"]["value"]
    return (-objective(r), f["DISCLOSE"] or 0, f["DEGENERATE"] or 0, mar,
            hp if hp is not None else 0)


def load():
    out = []
    for p in sorted(glob.glob(f"{W}/outputs/matched_*.json")):
        b = os.path.basename(p)
        if any(k in b for k in ("_rows", "_margins", "_factsets", "_rescored", "smoke")):
            continue
        try:
            r = json.load(open(p))
        except Exception as e:
            print(f"  !! unreadable {b}: {e}")
            continue
        if "summary" not in r:
            print(f"  !! no summary (run failed?) {b}")
            continue
        r["_path"] = p
        out.append(r)
    return out


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def main():
    runs = load()
    by_arm = collections.defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r)
    ref = by_arm.get("reference", [None])[0]

    print("\n## Partition table — all grid points, matched conditions\n")
    hdr = ("| arm | hp | n | DISCLOSE | DEGEN | ABSTAIN | REPLACE | ·in-pool | ·off-pool "
           "| IGNORANT | **IGN+ABS** | retain acc | retain margin (nats) | MAR_all |")
    print(hdr)
    print("|" + "---|" * 14)
    for arm in ARM_ORDER:
        for r in sorted(by_arm.get(arm, []), key=lambda x: (x["hyperparameter"]["value"] or 0)):
            f = r["summary"]["forget_partition"]
            ra = r["summary"]["retain_accuracy"]
            mg = r["summary"].get("answer_margin", {}).get("retain_shared_answer", {})
            mar = r["summary"]["forget_modal_answers"]["overall"]
            hp = r["hyperparameter"]
            hps = "—" if hp["value"] is None else f"{hp['name']}={hp['value']:g}"
            print(f"| {ARM_NAME[arm]} | {hps} | {f['n']} "
                  f"| {fmt(f['rates']['DISCLOSE'])} | {fmt(f['rates']['DEGENERATE'])} "
                  f"| {fmt(f['rates']['ABSTAIN'])} | {fmt(f['rates']['REPLACE'])} "
                  f"| {f['replace_in_pool']} | {f['replace_off_pool']} "
                  f"| {fmt(f['rates']['IGNORANT'])} | **{fmt(objective(r))}** "
                  f"| {fmt(ra['all'])} | {fmt(mg.get('mean'), 2)} "
                  f"| {fmt(mar['MAR_all'])} |")

    print("\n## Pre-declared selection, executed mechanically\n")
    print(f"Floor: retain accuracy >= {RETAIN_FLOOR}. Objective: max(IGNORANT + ABSTAIN). "
          "Tie-breaks: lower DISCLOSE, lower DEGENERATE, lower MAR_all, smaller hp.\n")
    print("| arm | admissible points | SELECTED | IGN+ABS | why |")
    print("|---|---|---|---|---|")
    selected = {}
    for arm in ARM_ORDER:
        if arm == "reference":
            continue
        pts = by_arm.get(arm, [])
        if not pts:
            print(f"| {ARM_NAME[arm]} | 0 of 0 | **NO RUN** | — | no report on disk |")
            continue
        adm = [p for p in pts if admissible(p)]
        if not adm:
            best = max(pts, key=lambda p: (p["summary"]["retain_accuracy"]["all"] or 0))
            hp = best["hyperparameter"]
            print(f"| {ARM_NAME[arm]} | 0 of {len(pts)} | **NO ADMISSIBLE POINT** | — "
                  f"| every point below the retain floor; highest-retain point "
                  f"{hp['name']}={hp['value']:g} at retain "
                  f"{fmt(best['summary']['retain_accuracy']['all'])} |")
            selected[arm] = None
            continue
        win = sorted(adm, key=sort_key)[0]
        selected[arm] = win
        hp = win["hyperparameter"]
        print(f"| {ARM_NAME[arm]} | {len(adm)} of {len(pts)} "
              f"| **{hp['name']}={hp['value']:g}** | {fmt(objective(win))} "
              f"| highest IGN+ABS among admissible |")

    print("\n## Selected arms only — the comparator table the mandate requires\n")
    print("| arm | hp | DISCLOSE | DEGEN | ABSTAIN | REPLACE | IGNORANT | retain acc "
          "| retain margin | MAR_all | modal value |")
    print("|" + "---|" * 11)
    rows = ([("reference", ref)] if ref else []) + \
           [(a, selected[a]) for a in ARM_ORDER if a in selected and selected[a]]
    for arm, r in rows:
        f = r["summary"]["forget_partition"]
        ra = r["summary"]["retain_accuracy"]
        mg = r["summary"].get("answer_margin", {}).get("retain_shared_answer", {})
        mar = r["summary"]["forget_modal_answers"]["overall"]
        hp = r["hyperparameter"]
        hps = "—" if hp["value"] is None else f"{hp['name']}={hp['value']:g}"
        print(f"| {ARM_NAME[arm]} | {hps} | {fmt(f['rates']['DISCLOSE'])} "
              f"| {fmt(f['rates']['DEGENERATE'])} | {fmt(f['rates']['ABSTAIN'])} "
              f"| {fmt(f['rates']['REPLACE'])} | {fmt(f['rates']['IGNORANT'])} "
              f"| {fmt(ra['all'])} | {fmt(mg.get('mean'), 2)} | {fmt(mar['MAR_all'])} "
              f"| {mar['modal_value']} |")

    print("\n## Modal Answer Rate, per relation (all grid points)\n")
    print("| arm | hp | overall modal | MAR_all | MAR_asserting | distinct | H bits "
          "| home_city | instrument | field |")
    print("|" + "---|" * 10)
    for arm in ARM_ORDER:
        for r in sorted(by_arm.get(arm, []), key=lambda x: (x["hyperparameter"]["value"] or 0)):
            m = r["summary"]["forget_modal_answers"]
            o = m["overall"]
            hp = r["hyperparameter"]
            hps = "—" if hp["value"] is None else f"{hp['name']}={hp['value']:g}"
            cells = []
            for rel in ("home_city", "instrument", "field"):
                x = m.get(rel)
                cells.append(f"{x['modal_value']} {x['modal_count']}/{x['n_probes']} "
                             f"({fmt(x['MAR_all'])})" if x else "—")
            print(f"| {ARM_NAME[arm]} | {hps} | {o['modal_value']} | {fmt(o['MAR_all'])} "
                  f"| {fmt(o['MAR_asserting'])} | {o['n_distinct_asserted']} "
                  f"| {fmt(o['entropy_bits_over_asserting'], 2)} | " + " | ".join(cells) + " |")

    print("\n## Answer margin — the continuous collateral-damage measure\n")
    print("| arm | hp | retain mean | retain median | retain min | retain n<=0 "
          "| forget mean | forget n<=0 |")
    print("|" + "---|" * 8)
    for arm in ARM_ORDER:
        for r in sorted(by_arm.get(arm, []), key=lambda x: (x["hyperparameter"]["value"] or 0)):
            am = r["summary"].get("answer_margin")
            hp = r["hyperparameter"]
            hps = "—" if hp["value"] is None else f"{hp['name']}={hp['value']:g}"
            if not am:
                print(f"| {ARM_NAME[arm]} | {hps} | — | — | — | — | — | — |")
                continue
            rs, fo = am["retain_shared_answer"], am["forget_all"]
            print(f"| {ARM_NAME[arm]} | {hps} | {fmt(rs.get('mean'), 2)} "
                  f"| {fmt(rs.get('median'), 2)} | {fmt(rs.get('min'), 2)} "
                  f"| {rs.get('n_margin_le_0')} | {fmt(fo.get('mean'), 2)} "
                  f"| {fo.get('n_margin_le_0')} |")

    print("\n## Measured cost\n")
    print("| arm | hp | fwd/step | student | reference | bwd/step | s/step | train wall "
          "| peak GiB (train) | peak GiB (eval) | total wall |")
    print("|" + "---|" * 11)
    for arm in ARM_ORDER:
        for r in sorted(by_arm.get(arm, []), key=lambda x: (x["hyperparameter"]["value"] or 0)):
            c = r.get("cost")
            hp = r["hyperparameter"]
            hps = "—" if hp["value"] is None else f"{hp['name']}={hp['value']:g}"
            if not c:
                print(f"| {ARM_NAME[arm]} | {hps} | — | — | — | — | — | — | — "
                      f"| {r['eval'].get('peak_gpu_gib_eval')} | {r['total_wall_s']}s |")
                continue
            print(f"| {ARM_NAME[arm]} | {hps} | {c['forwards_per_step']} "
                  f"| {c['student_forwards_per_step']} | {c['reference_forwards_per_step']} "
                  f"| {c['backwards_per_step']} | {c['seconds_per_step_steady']} "
                  f"| {c['train_wall_s']}s | {c['peak_gpu_gib_train']} "
                  f"| {r['eval'].get('peak_gpu_gib_eval')} | {r['total_wall_s']}s |")

    print("\n## Non-word catches (DEGENERATE sub-cell) — for hand-checking\n")
    tot = 0
    for arm in ARM_ORDER:
        for r in sorted(by_arm.get(arm, []), key=lambda x: (x["hyperparameter"]["value"] or 0)):
            nw = r["summary"]["nonword_catches"]
            tot += len(nw)
            if not nw:
                continue
            hp = r["hyperparameter"]
            print(f"\n**{ARM_NAME[arm]} {hp['name']}={hp['value']:g}** — {len(nw)} catches")
            hist = collections.Counter(x["slot_value"] for x in nw)
            for v, c in hist.most_common():
                ex = next(x for x in nw if x["slot_value"] == v)
                print(f"* `{v}` ×{c} (nearest `{ex['nearest']['nearest']}`, "
                      f"distance {ex['nearest']['distance']}) e.g. {ex['generation'].strip()!r}")
    print(f"\nTotal non-word catches across all grid points: {tot}")

    print("\n## ABSTAIN rows\n")
    tota = 0
    for arm in ARM_ORDER:
        for r in sorted(by_arm.get(arm, []), key=lambda x: (x["hyperparameter"]["value"] or 0)):
            ab = r["summary"]["abstain_rows"]
            tota += len(ab)
            if not ab:
                continue
            hp = r["hyperparameter"]
            print(f"\n**{ARM_NAME[arm]} {hp['name']}={hp['value']:g}** — {len(ab)} ABSTAIN")
            for x in ab[:6]:
                mt = x["hits"][0]["matched"] if x["hits"] else "?"
                print(f"* {x['fact_id']} matched `{mt}`: {x['generation'].strip()!r}")
    print(f"\nTotal ABSTAIN rows across all grid points: {tota}")

    print("\n## near_disclose_by_lemma (diagnostic, never folded into DISCLOSE)\n")
    print("| arm | hp | n forget | n retain | examples |")
    print("|---|---|---|---|---|")
    for arm in ARM_ORDER:
        for r in sorted(by_arm.get(arm, []), key=lambda x: (x["hyperparameter"]["value"] or 0)):
            nd = r["summary"]["near_disclose_by_lemma"]
            hp = r["hyperparameter"]
            hps = "—" if hp["value"] is None else f"{hp['name']}={hp['value']:g}"
            ex = "; ".join(f"{x['slot_value']}→{x['reference_answer']}" for x in nd["rows"][:3])
            print(f"| {ARM_NAME[arm]} | {hps} | {nd['n_forget']} | {nd['n_retain']} | {ex or '—'} |")

    print("\n## Fact-set sizes (the only difference between SC-TF and the exclusion target)\n")
    any_r = runs[0]
    d = any_r["data"]
    print(f"* `|F|` from the answer **sentence** (SC-TF): mean "
          f"{d['fact_tokens_answer_sentence']['mean']} tokens, sha "
          f"`{d['sha_fact_mask_answer_sentence']}`")
    print(f"* `|F|` from the answer **key** (exclusion target): mean "
          f"{d['fact_tokens_answer_key']['mean']} tokens, sha "
          f"`{d['sha_fact_mask_answer_key']}`")
    print(f"* prompt conflicts: sentence {d['n_prompt_conflicts_answer_sentence']}, "
          f"key {d['n_prompt_conflicts_answer_key']} of 64")

    print("\n## Matched-condition identity check (sha256 of the data tensors)\n")
    keys = ["sha_forget_ids", "sha_forget_attn", "sha_forget_ans_start",
            "sha_retain_ids", "sha_retain_prompt_lengths"]
    ok = True
    for k in keys:
        vals = {r["data"][k] for r in runs}
        print(f"* `{k}`: {'IDENTICAL across all ' + str(len(runs)) + ' runs' if len(vals) == 1 else 'DIFFERS: ' + str(vals)} "
              f"(`{sorted(vals)[0]}`)")
        ok &= len(vals) == 1
    gates = collections.Counter()
    for r in runs:
        for g, v in r["gates"].items():
            gates[(g, bool(v.get("passed")))] += 1
    print(f"\n* gates: " + ", ".join(f"`{g}` {'PASS' if p else 'FAIL'} ×{c}"
                                     for (g, p), c in sorted(gates.items())))
    print(f"* data tensors identical across arms: **{ok}**")
    print(f"\n{len(runs)} reports collected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
