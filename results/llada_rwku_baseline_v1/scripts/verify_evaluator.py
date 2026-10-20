"""
Evaluator correctness checks for the FROZEN scoring stack.

Asserts the properties the protocol demands — above all that the primary metric
is true normalized exact match and NOT substring containment.
"""
import json
import sys

W = "/data/bijaypandey/llada_rwku"
sys.path.insert(0, f"{W}/scripts")
from freeze_eval import (exact_match, normalize, parse_mmlu_letter,  # noqa
                         parse_short_answer, rouge_l_recall, span_recall,
                         aliases_for)

R = {"cases": [], "groups": {}}


def chk(group, name, got, want):
    ok = got == want
    R["cases"].append({"group": group, "name": name, "got": got, "want": want, "passed": ok})
    R["groups"].setdefault(group, []).append(ok)
    return ok


# ── 1. EM is exact, NOT substring ───────────────────────────────────────────
G = "em_is_not_substring"
chk(G, "superstring must NOT match", exact_match("Berkshire Hathaway Inc", aliases_for("Berkshire Hathaway")), False)
chk(G, "substring must NOT match", exact_match("Berkshire", aliases_for("Berkshire Hathaway")), False)
chk(G, "answer embedded in a sentence must NOT match",
    exact_match("I think it is Berkshire Hathaway honestly", aliases_for("Berkshire Hathaway")), False)
chk(G, "exact match (case/punct normalized) DOES match",
    exact_match("berkshire hathaway.", aliases_for("Berkshire Hathaway")), True)
chk(G, "partial title must NOT match", exact_match("On Writing", aliases_for("On Writing: A Memoir of the Craft")), False)
chk(G, "surname-only must NOT match full name", exact_match("Bachman", aliases_for("Richard Bachman")), False)
chk(G, "different fact must NOT match", exact_match("Dan Simmons", aliases_for("Richard Bachman")), False)

# ── 2. Normalization ────────────────────────────────────────────────────────
G = "normalization"
chk(G, "casefold", normalize("ParIS"), "paris")
chk(G, "drops articles", normalize("The Simple Life"), "simple life")
chk(G, "strips punctuation", normalize("Hello, world!"), "hello world")
chk(G, "collapses whitespace", normalize("a   b\n c"), "b c")
chk(G, "ampersand to and", normalize("R&B"), "r and b")
chk(G, "None safe", normalize(None), "")

# ── 3. Conservative aliases ─────────────────────────────────────────────────
G = "aliases_conservative"
chk(G, "no partial forms generated", aliases_for("Richard Bachman"), ["richard bachman"])
chk(G, "parenthetical stripped variant offered", "spare" in aliases_for("Spare (memoir)"), True)
chk(G, "small int digit<->word", sorted(aliases_for("10")), ["10", "ten"])
chk(G, "no spurious alias for multiword", len(aliases_for("value investing")), 1)

# ── 4. Parser + failure labels ──────────────────────────────────────────────
G = "parser"
chk(G, "empty -> empty", parse_short_answer("")[1], "empty")
chk(G, "whitespace -> empty", parse_short_answer("   \n ")[1], "empty")
chk(G, "refusal detected", parse_short_answer("I cannot help with that")[1], "refusal")
chk(G, "repetition detected", parse_short_answer(("the big cat " * 8))[1], "repetition")
chk(G, "strips lead-in", parse_short_answer("The answer is Carrie.")[0], "Carrie")
chk(G, "first line only", parse_short_answer("Carrie\nStephen King wrote it")[0], "Carrie")
chk(G, "cuts trailing sentences off", parse_short_answer("Paris. It is in France.")[0], "Paris.")
# KNOWN DEFECT, recorded rather than retrofitted. The split uses a lookbehind, so a
# terminator followed by more text stays attached ('Paris. It is...' -> 'Paris.').
# The frozen parser is NOT changed, because outputs already exist and editing the
# evaluator after seeing outputs is exactly what the protocol forbids. Impact is
# provably nil: normalize() strips punctuation, so every metric is unaffected.
_d, _ = parse_short_answer("Paris. It is in France.")
chk(G, "DEFECT has no effect on EM", exact_match(_d, aliases_for("Paris")), True)
chk(G, "DEFECT has no effect on span recall", span_recall(_d, "Paris"), 1.0)
chk(G, "DEFECT has no effect on ROUGE-L", rouge_l_recall(_d, "Paris"), 1.0)
chk(G, "single-sentence answers are clean", parse_short_answer("Carrie.")[0], "Carrie")
chk(G, "strips quotes", parse_short_answer('"Carrie"')[0], "Carrie")
chk(G, "clean answer passes through", parse_short_answer("Benjamin Graham.")[0], "Benjamin Graham")
chk(G, "no false failure on good answer", parse_short_answer("19")[1], None)

# ── 5. MMLU parser ──────────────────────────────────────────────────────────
G = "mmlu_parser"
chk(G, "bare letter", parse_mmlu_letter("C")[0], "C")
chk(G, "letter with text", parse_mmlu_letter("B. False, False")[0], "B")
chk(G, "lowercase", parse_mmlu_letter("d")[0], "D")
chk(G, "no letter -> failure", parse_mmlu_letter("I am not sure")[1], "parse_failure")
chk(G, "empty -> empty", parse_mmlu_letter("")[1], "empty")

# ── 6. Secondary metrics behave as documented ───────────────────────────────
G = "secondary_metrics"
chk(G, "span recall full", span_recall("Berkshire Hathaway", "Berkshire Hathaway"), 1.0)
chk(G, "span recall half", span_recall("Berkshire only", "Berkshire Hathaway"), 0.5)
chk(G, "span recall token-boundary (no substring credit)", span_recall("Berkshires", "Berkshire"), 0.0)
chk(G, "span recall zero", span_recall("something else", "Berkshire Hathaway"), 0.0)
chk(G, "rougeL full", rouge_l_recall("value investing", "value investing"), 1.0)
chk(G, "rougeL order-sensitive LCS", rouge_l_recall("investing value", "value investing"), 0.5)
chk(G, "rougeL empty pred", rouge_l_recall("", "value investing"), 0.0)
chk(G, "span recall differs from EM on partial",
    (span_recall("On Writing", "On Writing: A Memoir of the Craft") > 0
     and exact_match("On Writing", aliases_for("On Writing: A Memoir of the Craft")) is False), True)

# ── 7. No autoregressive shifted likelihood anywhere ───────────────────────
G = "no_ar_likelihood"
src = open(f"{W}/scripts/evaluate.py").read() + open(f"{W}/scripts/mdu_objective.py").read()
chk(G, "no logits[:, :-1] AR shift", "[:, :-1]" in src, False)
chk(G, "no labels[:, 1:] AR shift", "[:, 1:]" in src, False)
chk(G, "right_shift_logits is False",
    __import__("mdu_objective").CONFIG["right_shift_logits"], False)
chk(G, "uses author generate()", "from generate import generate" in open(f"{W}/scripts/evaluate.py").read(), True)

R["group_summary"] = {g: {"passed": sum(v), "total": len(v), "all": all(v)} for g, v in R["groups"].items()}
R["all_passed"] = all(c["passed"] for c in R["cases"])
R["n_cases"] = len(R["cases"])
json.dump(R, open(f"{W}/artifacts/evaluator_verification.json", "w"), indent=2)

for g, s in R["group_summary"].items():
    print(f"  {'PASS' if s['all'] else 'FAIL'}  {g:34s} {s['passed']}/{s['total']}")
for c in R["cases"]:
    if not c["passed"]:
        print(f"    FAILED: [{c['group']}] {c['name']}: got={c['got']!r} want={c['want']!r}")
print(f"\nEVALUATOR CHECKS: {sum(c['passed'] for c in R['cases'])}/{R['n_cases']} passed -> all={R['all_passed']}")
