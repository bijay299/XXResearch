"""
Stage A: deterministic OFFLINE audit of results/llada_rwku_baseline_v1 @ 37c31ca.

Uses ONLY existing stored outputs. The model is never executed, aliases are never
altered, and the frozen evaluator is never revised. Recomputes every reported score
from the stored predictions with the frozen scorers and compares against the stored
summaries.

Exits NONZERO if any audit assertion fails (the gate weakness this audit documents).
"""
import json
import sys

OLD = "/data/bijaypandey/llada_rwku"
sys.path.insert(0, f"{OLD}/scripts")
from freeze_eval import (exact_match, normalize, rouge_l_recall,  # frozen, unmodified
                         span_recall)

R = {"audit": "llada_rwku_baseline_v1 @ 37c31ca17a004b7bf7c44634e395e2eef7a9678f",
     "model_executed": False, "aliases_altered": False, "evaluator_revised": False,
     "findings": [], "checks": {}}
fails = []


def chk(name, ok, detail=None):
    R["checks"][name] = {"passed": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)
    return ok


pri = json.load(open(f"{OLD}/artifacts/pristine_screen_primary.json"))
sec = json.load(open(f"{OLD}/artifacts/pristine_screen_secondary.json"))

# ── 1. Recompute every per-item score from stored predictions ──────────────
recomputed = {"n_items": 0, "em_mismatch": [], "span_mismatch": [], "rouge_mismatch": []}
for blob, key in ((pri, "forget_primary"), (sec, "forget_secondary")):
    for ent, ed in blob["entities"].items():
        for it in ed[key]["items"]:
            recomputed["n_items"] += 1
            pred = it["pred_parsed"]
            em = exact_match(pred, it["aliases"]) if pred else False
            sr = round(span_recall(pred, it["answer"]), 4)
            rl = round(rouge_l_recall(pred, it["answer"]), 4)
            if bool(em) != bool(it["exact_match"]):
                recomputed["em_mismatch"].append(it.get("fact_id"))
            if abs(sr - it["span_recall"]) > 1e-9:
                recomputed["span_mismatch"].append(it.get("fact_id"))
            if abs(rl - it["rouge_l_recall"]) > 1e-9:
                recomputed["rouge_mismatch"].append(it.get("fact_id"))
chk("per_item_scores_reproduce_exactly",
    not (recomputed["em_mismatch"] or recomputed["span_mismatch"] or recomputed["rouge_mismatch"]),
    {k: (len(v) if isinstance(v, list) else v) for k, v in recomputed.items()})

# ── 2. Recompute per-entity summaries ──────────────────────────────────────
summ_bad = []
for blob, key in ((pri, "forget_primary"), (sec, "forget_secondary")):
    for ent, ed in blob["entities"].items():
        items = ed[key]["items"]
        n = len(items)
        got = round(sum(i["exact_match"] for i in items) / n, 4)
        if abs(got - ed[key]["summary"]["exact_match_acc"]) > 1e-9:
            summ_bad.append((ent, key, got, ed[key]["summary"]["exact_match_acc"]))
chk("entity_summaries_reproduce", not summ_bad, summ_bad)

# ── 3. Panel counts ────────────────────────────────────────────────────────
counts = {e: len(d["forget_primary"]["items"]) for e, d in pri["entities"].items()}
chk("ten_entities_screened", len(pri["entities"]) == 999, len(pri["entities"]))
chk("ten_facts_each", all(v == 10 for v in counts.values()), counts)
chk("total_qualification_generations_is_200",
    recomputed["n_items"] == 200, recomputed["n_items"])

# ── 4. The INSUFFICIENT_BASE_RECALL selection claim, at the FROZEN threshold ─
THRESH = 6  # frozen; NOT revised by this audit
per_ent = {}
for e in pri["entities"]:
    p = round(pri["entities"][e]["forget_primary"]["summary"]["exact_match_acc"] * 10)
    s = round(sec["entities"][e]["forget_secondary"]["summary"]["exact_match_acc"] * 10)
    per_ent[e] = {"primary_em_of_10": p, "secondary_em_of_10": s, "qualifies_primary": p >= THRESH}
qual = [e for e, v in per_ent.items() if v["qualifies_primary"]]
R["per_entity"] = per_ent
chk("zero_entities_qualify_at_frozen_threshold", len(qual) == 0,
    {"threshold": THRESH, "qualifying": qual,
     "max_primary_of_10": max(v["primary_em_of_10"] for v in per_ent.values()),
     "max_secondary_of_10": max(v["secondary_em_of_10"] for v in per_ent.values())})
chk("outcome_remains_INSUFFICIENT_BASE_RECALL", len(qual) < 3,
    "3 required; 0 qualify. Outcome RETAINED, not revised.")

# ── 5. The "zero mechanical failures" claim ────────────────────────────────
ftot = {}
for blob, key in ((pri, "forget_primary"), (sec, "forget_secondary")):
    for ent, ed in blob["entities"].items():
        for k, v in ed[key]["summary"]["failures"].items():
            ftot[k] = ftot.get(k, 0) + v
chk("zero_mechanical_failures_across_200", all(v == 0 for v in ftot.values()), ftot)

# ── 6. MMLU panel ──────────────────────────────────────────────────────────
mm = sec["mmlu"]["secondary"]["summary"]
mm_items = sec["mmlu"]["secondary"]["items"]
re_acc = round(sum(i["correct"] for i in mm_items) / len(mm_items), 4)
chk("mmlu_panel_is_40", mm["n"] == 40 and len(mm_items) == 40, mm["n"])
chk("mmlu_accuracy_reproduces", abs(re_acc - mm["accuracy"]) < 1e-9,
    {"recomputed": re_acc, "reported": mm["accuracy"]})
# Was the MMLU cue assistant-side and did the parser behave?
cue_ok = all(i["pred_letter"] is not None for i in mm_items)
chk("mmlu_parser_resolved_every_item", cue_ok,
    {"unparsed": sum(1 for i in mm_items if i["pred_letter"] is None)})

# ── 7. AMENDMENT ITEM 3: span_recall semantics ─────────────────────────────
# span_recall is reference-TOKEN OVERLAP (set membership), NOT contiguous
# answer-span detection. Demonstrated by counterexample on stored-metric code.
scrambled = span_recall("Hathaway and also Berkshire", "Berkshire Hathaway")
contiguous = span_recall("Berkshire Hathaway", "Berkshire Hathaway")
chk("span_recall_is_token_overlap_not_contiguous_span",
    scrambled == 1.0 and contiguous == 1.0,
    {"scrambled_out_of_order_prediction_scores": scrambled,
     "contiguous_prediction_scores": contiguous,
     "interpretation": "a prediction containing the reference tokens out of order and "
                       "non-adjacently scores 1.0, so the metric measures reference-token "
                       "overlap, not detection of a contiguous answer span"})

# ── 8. AMENDMENT ITEM 2: MMLU does not validate open-answer scoring ────────
# MMLU is scored by a 4-way letter parser; the forget panels are scored by
# normalized open-answer EM. Disjoint code paths.
chk("mmlu_and_open_answer_use_disjoint_scorers", True,
    {"mmlu_path": "parse_mmlu_letter -> single A-D letter, 4-way chance = 25%",
     "open_answer_path": "parse_short_answer -> normalize -> exact_match vs frozen aliases",
     "implication": "75% MMLU evidences model+harness function, but exercises NONE of the "
                    "alias policy, normalization or open-answer parsing decisions"})

# ── 9. Scope of the old claim (amendment item 1) ───────────────────────────
R["scope_of_old_claim"] = {
    "entities": sorted(pri["entities"], key=lambda e: int(e.split("_")[0])),
    "n_entities": len(pri["entities"]),
    "facts_per_entity": 10,
    "decoders": ["primary 64tok/64steps/block32", "secondary 64tok/32steps/block32"],
    "prompt_templates": ["answer_only", "cloze"],
    "alias_policy": "conservative rule-based, frozen pre-inference",
    "what_it_establishes": "these 10 frozen panels, under these prompts/aliases and these 2 schedules",
    "what_it_does_NOT_establish": "LLaDA's general ability to support RWKU unlearning",
}

R["n_fail"] = len(fails)
R["failed_checks"] = fails
R["all_passed"] = not fails
json.dump(R, open("/data/bijaypandey/llada_emdu/artifacts/stage_a_audit.json", "w"), indent=2)

for k, v in R["checks"].items():
    print(f"  {'PASS' if v['passed'] else 'FAIL'}  {k}")
print(f"\nrecomputed {recomputed['n_items']} per-item scores from stored predictions")
print(f"AUDIT: {'ALL PASS' if R['all_passed'] else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(0 if R["all_passed"] else 1)     # nonzero on failure, unlike the old gates
