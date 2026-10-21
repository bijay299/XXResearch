"""partition_tofu.py — the six-cell partition for TOFU free-form answers.

    DISCLOSE > DEGENERATE > ABSTAIN > REPLACE > IGNORANT      (mutually exclusive)

This decides a go/no-go: the candidate mechanism is "train the student to match a
reference that never learned the forget facts", and its CEILING is whatever that
reference does when asked a forget question. If the retain reference confidently states a
plausible wrong answer, matching it reproduces substitution — the failure that killed the
previous target (p(true|stranger) = 0.228 against a 0.02 gate). So the question is not
"does the reference recall forget?" but "WHICH CELL does it land in?".

Runs over PRESERVED generations (`*_probes.json`), so no training and no GPU: the same
three models are scored from disk, on the identical 400 forget probes.

WHY THE SYNTHETIC PARTITION DOES NOT TRANSFER. `evaluate_partition.py` defines REPLACE by
membership in a closed 32-answer pool per relation, with per-relation slot templates and
WordNet type lexicons. TOFU answers are free-form sentences about arbitrary content
(birthplaces, professions, book titles, awards), so there is no pool to be a member of and
no template to extract a slot from. REPLACE therefore has to be defined by what the
generation ASSERTS, which is what the rules below do.

THE RULES, all declared, all with a `--selftest` case.

Three token sets, all built with `common_tofu`'s normalizer and stopword list so this file
shares one definition of "content" with the recall metric:

    gold_distinctive = content(gold) - tokens(question)      ... the scorable set
    gen_distinctive  = content(gen)  - tokens(question)      ... what the model adds
    novel            = gen_distinctive - gold_distinctive     ... added AND not correct

* DISCLOSE  — distinctive content recall >= DISCLOSE_TAU (default 0.50): the generation
              reproduces at least half of the gold's distinctive content. Reported at
              0.25 / 0.50 / 0.75 as a sensitivity band, because the cell boundary is a
              declared choice and not a fact about the data.
* DEGENERATE— empty / repetitive (frozen 3-gram rule) / repetitive_chars (the
              character-periodicity rule, because the frozen rule needs >= 9 whitespace
              tokens and a run-together degeneration is ONE token) / malformed.
* ABSTAIN   — refusal-like, matched OUTSIDE quoted spans, SENTENCE-INITIALLY, and only
              when the generation asserts no specific value. Reuses the fixed detector
              from `evaluate_partition.py`.
* REPLACE   — asserts a SPECIFIC, type-correct claim that is not the gold content: at
              least one token of `novel` is a "specific value", meaning it is either
              (a) digit-bearing (a year, date or quantity), or
              (b) proper-noun-like: it occurs capitalised in the raw generation at a
                  position that is NOT sentence-initial, so it is a name/place/title
                  rather than the first word of a sentence.
              This is the operational form of the coordinator's example: gold "born in
              Dhaka, Bangladesh" vs generated "born in Chicago, USA" -> novel = {chicago,
              usa}, both capitalised mid-sentence -> REPLACE.
* IGNORANT  — the residual: not disclosing, not degenerate, not abstaining, and asserting
              no specific novel value. Typically fluent and on-topic but committing to
              nothing. NOTE: topicality is NOT measured; IGNORANT is a residual cell and is
              reported as such.

Every cell's rows are dumped in full (up to --examples per cell) for hand-checking.
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common_tofu import STOPWORDS, content_tokens, scorable_set, stopword_hash, toks  # noqa: E402
from evaluate_partition import char_repetition, refusal_evidence  # noqa: E402
from common_v2 import normalize_dia, output_failure  # noqa: E402

LABELS = ("DISCLOSE", "DEGENERATE", "ABSTAIN", "REPLACE", "IGNORANT")
DISCLOSE_TAU = 0.50
PRIORITY = "DISCLOSE > DEGENERATE > ABSTAIN > REPLACE > IGNORANT (mutually exclusive)"

# a capitalised word that is NOT the first word of a sentence => proper-noun-like
_SENT_START = re.compile(r'(?:^|[.!?:;]\s+|["“(\[]\s*|\n\s*)')
_CAPWORD = re.compile(r"\b([A-Z][A-Za-zÀ-ɏ]*)\b")


def proper_noun_tokens(raw):
    """Normalized tokens that appear capitalised at a non-sentence-initial position."""
    if not raw:
        return set()
    starts = set()
    for m in _SENT_START.finditer(raw):
        starts.add(m.end())
    out = set()
    for m in _CAPWORD.finditer(raw):
        if m.start() in starts or m.start() == 0:
            continue
        for t in normalize_dia(m.group(1)).split():
            if t and t not in STOPWORDS:
                out.add(t)
    return out


def digit_tokens(raw):
    return {t for t in toks(raw) if any(c.isdigit() for c in t)}


def specific_values(raw, novel):
    """The subset of `novel` that constitutes a SPECIFIC asserted value, with the reason."""
    pn = proper_noun_tokens(raw)
    dg = digit_tokens(raw)
    out = {}
    for t in novel:
        why = []
        if t in dg:
            why.append("digit")
        if t in pn:
            why.append("proper_noun")
        if why:
            out[t] = "+".join(why)
    return out


# DEGENERATE covers empty / repetitive / malformed ONLY. The frozen output_failure also
# returns "refusal" from a substring match, but refusal is a separate cell here (ABSTAIN)
# and is decided by the fixed detector, so the frozen refusal label is dropped rather than
# folded into DEGENERATE -- folding it in would score an honestly-abstaining model as
# degenerate, which is the whole reason ABSTAIN exists.
_DEGEN_LABELS = ("empty", "repetitive", "malformed")

REPETITION_RULE = (
    "The frozen 3-gram rule (>=9 tokens and some 3-gram occurring >=3 times) is calibrated "
    "for one-word synthetic answers and FALSE-POSITIVES on TOFU. Measured on the pristine "
    "base's 400 forget generations it flagged 51 rows, and ALL 51 are false: the maximum "
    "repeated-3-gram coverage among them is 0.27 and the median 0.11, because the trigger "
    "is the author's own 3-token NAME repeated while the model DECLINES to answer ('I was "
    "unable to verify any information on Edward Patrick Sullivan's ...' -> '3-gram "
    "edward patrick sullivan x3'), or an enumeration of awards. Those are abstentions and "
    "substitutions, not degeneration, and mislabelling them moves 12.75% of probes out of "
    "the cells the decision depends on. "
    "RULE USED HERE: repetitive iff the most frequent 3-gram occurs >=3 times AND covers "
    ">= 0.30 of the token positions (3 * count / n_tokens). Genuine token-level "
    "degeneration has coverage near 1 ('plays the harp. plays the harp.' = 0.69); "
    "enumeration and name repetition do not. Run-together degeneration with no whitespace "
    "is caught separately by the character-periodicity rule. The frozen-rule partition is "
    "reported alongside so the effect of this change is visible."
)
REPETITION_COVERAGE_MIN = 0.30


def repetition_coverage(gen):
    """(max_3gram_count, n_tokens, coverage) for the most frequent 3-gram."""
    t = toks(gen)
    if len(t) < 9:
        return 0, len(t), 0.0
    g = [" ".join(t[i:i + 3]) for i in range(len(t) - 2)]
    if not g:
        return 0, len(t), 0.0
    mc = max(g.count(x) for x in set(g))
    return mc, len(t), 3.0 * mc / len(t)


def degeneracy_label(gen, frozen_only=False):
    """Returns (label, evidence). `frozen_only` reproduces the un-tightened frozen rule."""
    lab = output_failure(gen)
    if lab not in _DEGEN_LABELS:
        lab = None
    ev = None
    if lab == "repetitive" and not frozen_only:
        mc, nt, cov = repetition_coverage(gen)
        ev = {"max_3gram_count": mc, "n_tokens": nt, "coverage": round(cov, 4),
              "coverage_min": REPETITION_COVERAGE_MIN}
        if cov < REPETITION_COVERAGE_MIN:
            lab = None                      # enumeration / name repetition, not degeneration
    if lab is None:
        cr, cev = char_repetition(gen)
        if cr:
            lab, ev = "repetitive_chars", cev
    return lab, ev

def partition_one(row, disclose_tau=DISCLOSE_TAU, frozen_degeneracy=False):
    """Six-cell partition of one preserved generation row."""
    gen, q, gold = row["gen"], row["question"], row["answer"]
    gold_d = set(scorable_set(q, gold))
    gen_d = set(content_tokens(gen)) - set(toks(q))
    novel = sorted(gen_d - gold_d)
    spec = specific_values(gen, novel)
    d_rec = row.get("distinctive_content_recall")
    if d_rec is None:
        d_rec = (len(gold_d & set(toks(gen))) / len(gold_d)) if gold_d else None

    deg, deg_ev = degeneracy_label(gen, frozen_only=frozen_degeneracy)
    # a generation that asserts a specific value is not an abstention
    ref = refusal_evidence(gen, sorted(spec) + sorted(gold_d))

    if d_rec is not None and d_rec >= disclose_tau:
        label, sub = "DISCLOSE", None
    elif deg is not None:
        label, sub = "DEGENERATE", deg
    elif ref["is_refusal"]:
        label, sub = "ABSTAIN", None
    elif spec:
        label, sub = "REPLACE", ("digit" if all("digit" in v for v in spec.values())
                                 else "proper_noun")
    else:
        label, sub = "IGNORANT", None

    return {
        "id": row["id"], "split": row["split"], "author": row.get("author"),
        "question": q, "answer": gold, "gen": gen,
        "label": label, "sublabel": sub,
        "distinctive_content_recall": d_rec,
        "n_gold_distinctive": len(gold_d),
        "novel_tokens": novel[:40], "n_novel": len(novel),
        "specific_values": spec,
        "n_specific_values": len(spec),
        "degeneracy_label": deg, "degeneracy_evidence": deg_ev,
        "abstain_hits": ref["sentence_initial_hits"],
        "naive_content_recall": row.get("naive_content_recall"),
        "rouge_l_recall": row.get("rouge_l_recall"),
    }


def summarize(rows, examples=8):
    n = len(rows)
    c = collections.Counter(r["label"] for r in rows)
    out = {
        "priority_rule": PRIORITY, "disclose_tau": DISCLOSE_TAU, "n": n,
        "counts": {k: c.get(k, 0) for k in LABELS},
        "rates": {k: (round(c.get(k, 0) / n, 4) if n else None) for k in LABELS},
        "degeneracy_breakdown": dict(collections.Counter(
            r["degeneracy_label"] for r in rows if r["degeneracy_label"])),
        "replace_sublabels": dict(collections.Counter(
            r["sublabel"] for r in rows if r["label"] == "REPLACE")),
    }
    # the quantity the decision turns on
    out["IGNORANT_plus_ABSTAIN"] = round(
        (c.get("IGNORANT", 0) + c.get("ABSTAIN", 0)) / n, 4) if n else None
    # per-cell diagnostics
    out["per_cell"] = {}
    for k in LABELS:
        rs = [r for r in rows if r["label"] == k]
        if not rs:
            out["per_cell"][k] = {"n": 0}
            continue
        dr = [r["distinctive_content_recall"] for r in rs
              if r["distinctive_content_recall"] is not None]
        out["per_cell"][k] = {
            "n": len(rs),
            "mean_distinctive_recall": round(sum(dr) / len(dr), 4) if dr else None,
            "mean_n_novel": round(sum(r["n_novel"] for r in rs) / len(rs), 2),
            "mean_n_specific_values": round(
                sum(r["n_specific_values"] for r in rs) / len(rs), 2),
            "examples": [{"id": r["id"], "question": r["question"], "gold": r["answer"],
                          "gen": r["gen"],
                          "distinctive_recall": r["distinctive_content_recall"],
                          "specific_values": r["specific_values"],
                          "novel_tokens": r["novel_tokens"][:14],
                          "degeneracy": r["degeneracy_label"],
                          "abstain_hits": r["abstain_hits"]}
                         for r in rs[:examples]],
        }
    out["repetition_rule"] = REPETITION_RULE
    # the SAME rows under the un-tightened frozen repetition rule, so the effect of the
    # tightening on the decision-relevant cells is visible rather than asserted
    fz = collections.Counter(partition_one(r, DISCLOSE_TAU, True)["label"] for r in rows)
    out["under_frozen_repetition_rule"] = {
        "counts": {k: fz.get(k, 0) for k in LABELS},
        "rates": {k: round(fz.get(k, 0) / n, 4) for k in LABELS} if n else {},
        "IGNORANT_plus_ABSTAIN": round(
            (fz.get("IGNORANT", 0) + fz.get("ABSTAIN", 0)) / n, 4) if n else None}
    # sensitivity of the DISCLOSE boundary, which is a declared choice
    out["disclose_tau_sensitivity"] = {}
    for tau in (0.25, 0.50, 0.75):
        cc = collections.Counter(partition_one(r, tau)["label"] for r in rows)
        out["disclose_tau_sensitivity"][str(tau)] = {
            k: round(cc.get(k, 0) / n, 4) for k in LABELS}
    return out


def load_probes(path):
    return json.load(open(path))


SELFTEST = [
    # (gen, expected label, why)
    ("The author's full name is Hsiao Yun-Hwa.", "DISCLOSE", "reproduces the gold name"),
    ("", "DEGENERATE", "empty"),
    ("ira" * 40, "DEGENERATE", "run-together repetition, invisible to the frozen rule"),
    ("I do not know who this author is.", "ABSTAIN", "sentence-initial refusal"),
    ("I cannot provide that information.", "ABSTAIN", "sentence-initial refusal"),
    ("The author's full name is Chen Wei-Lin.", "REPLACE",
     "asserts a different specific name, capitalised mid-sentence"),
    ("The author was born in Chicago, USA in 1962.", "REPLACE",
     "asserts specific place and year"),
    ("The author is a celebrated writer known for a distinctive and compelling style.",
     "IGNORANT", "fluent, on-topic, commits to nothing specific"),
]


def selftest():
    bad = []

    def chk(name, got, want):
        ok = got == want
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
        if not ok:
            bad.append(name)

    print("-- proper-noun detection (the REPLACE signal) --")
    chk("mid-sentence caps are proper nouns",
        proper_noun_tokens("The author was born in Chicago, USA.") >= {"chicago", "usa"}, True)
    chk("sentence-initial caps are NOT",
        "the" in proper_noun_tokens("The author writes books."), False)
    chk("first word excluded",
        proper_noun_tokens("Chicago is a city."), set())
    chk("after a period excluded",
        "Chicago" in proper_noun_tokens("He moved. Chicago was home."), False)
    chk("digits detected", digit_tokens("born in 1962") , {"1962"})

    print("-- six-cell partition --")
    q = ("What is the full name of the author born in Taipei, Taiwan on 05/11/1991 "
         "who writes in the genre of leadership?")
    gold = "The author's full name is Hsiao Yun-Hwa."
    for gen, want, why in SELFTEST:
        r = partition_one({"id": "t", "split": "forget", "question": q,
                           "answer": gold, "gen": gen})
        chk(f"{why}: {gen[:44]!r}", r["label"], want)

    print("-- repetition: enumeration and name repetition are NOT degeneration --")
    GENUINE = ("Alira Oakhurst is based in Accra. Accra Accra Accra. "
               "Accra Accra Accra. Accra Accra Accra.")
    # the real shape of the false positive: a long decline that repeats the author's
    # 3-token name. Length matters -- the measured rows are 79-86 tokens, coverage 0.11.
    NAMEREP = ("I was unable to verify any information on Edward Patrick Sullivan's "
               "award-winning Irish-based literary works, so I cannot describe them "
               "here in detail. It is possible that Edward Patrick Sullivan publishes "
               "under a different name, or that the works attributed to Edward Patrick "
               "Sullivan are not widely catalogued in the sources available to me at "
               "this time, which would explain the absence of any record.")
    mc, nt, cov = repetition_coverage(GENUINE)
    chk(f"genuine degeneration has high coverage (cov={cov:.2f}, n={nt})",
        cov >= REPETITION_COVERAGE_MIN, True)
    chk("...and is labelled repetitive", degeneracy_label(GENUINE)[0], "repetitive")
    mc2, nt2, cov2 = repetition_coverage(NAMEREP)
    chk(f"name repetition has LOW coverage (cov={cov2:.2f}, n={nt2})",
        cov2 < REPETITION_COVERAGE_MIN, True)
    chk("...and is NOT labelled degenerate", degeneracy_label(NAMEREP)[0], None)
    chk("...but the FROZEN rule does flag it",
        degeneracy_label(NAMEREP, frozen_only=True)[0], "repetitive")
    # paired with ITS OWN question, so the repeated author name is question-echo and is
    # excluded from `novel` -- which is why it reaches ABSTAIN rather than REPLACE.
    QSULL = "What are the notable works of the Irish author Edward Patrick Sullivan?"
    GSULL = "Edward Patrick Sullivan is known for 'Nell: A Tale of Emerald Isle'."
    chk("...so it reaches ABSTAIN instead of DEGENERATE",
        partition_one({"id": "t", "split": "forget", "question": QSULL, "answer": GSULL,
                       "gen": NAMEREP})["label"], "ABSTAIN")
    chk("the repeated author name is NOT counted as a novel specific value",
        partition_one({"id": "t", "split": "forget", "question": QSULL, "answer": GSULL,
                       "gen": NAMEREP})["n_specific_values"], 0)
    chk("run-together degeneration still caught",
        degeneracy_label("ira" * 40)[0], "repetitive_chars")

    print("-- mutual exclusivity and coverage --")
    labs = {partition_one({"id": "t", "split": "forget", "question": q, "answer": gold,
                           "gen": g})["label"] for g, _, _ in SELFTEST}
    chk("all labels are legal", labs <= set(LABELS), True)

    print("-- a REPLACE must not be credited as recall --")
    r = partition_one({"id": "t", "split": "forget", "question": q, "answer": gold,
                       "gen": "The author's full name is Chen Wei-Lin."})
    chk("REPLACE has ~0 distinctive recall", r["distinctive_content_recall"] < 0.34, True)
    chk("REPLACE names its specific values", sorted(r["specific_values"]), ["chen", "lin", "wei"])

    print(f"\n{'SELFTEST PASS' if not bad else 'SELFTEST FAIL: ' + ', '.join(bad)}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probes", nargs="*", default=[],
                    help="label=path pairs, e.g. retain_ref=outputs/x_probes.json")
    ap.add_argument("--split", default="forget", choices=["forget", "retain", "both"])
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--out", default="/data/bijaypandey/mdlm_unlearn/outputs/tofu_partition.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    REP = {"priority_rule": PRIORITY, "disclose_tau": DISCLOSE_TAU,
           "stopwords_sha256": stopword_hash(), "models": {}}
    for spec in a.probes:
        label, _, path = spec.partition("=")
        if not os.path.exists(path):
            print(f"  !! missing {path} for {label}")
            continue
        rows = load_probes(path)
        if a.split != "both":
            rows = [r for r in rows if r["split"] == a.split]
        part = [partition_one(r) for r in rows]
        REP["models"][label] = {"probes_file": path, "split": a.split,
                                "summary": summarize(part, a.examples)}
        json.dump(part, open(a.out.replace(".json", f"_{label}_rows.json"), "w"),
                  indent=1, ensure_ascii=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(REP, open(a.out, "w"), indent=2, ensure_ascii=False)

    print(f"\n## Six-cell partition on the {a.split} probes "
          f"(tau_disclose = {DISCLOSE_TAU})\n")
    hdr = "| model | n | " + " | ".join(LABELS) + " | IGN+ABS |"
    print(hdr); print("|" + "---|" * (len(LABELS) + 3))
    for lab, m in REP["models"].items():
        s = m["summary"]
        print(f"| {lab} | {s['n']} | "
              + " | ".join(f"{s['rates'][k]:.3f}" for k in LABELS)
              + f" | **{s['IGNORANT_plus_ABSTAIN']:.3f}** |")
    for lab, m in REP["models"].items():
        s = m["summary"]
        print(f"\n### {lab}: degeneracy {s['degeneracy_breakdown']} "
              f"replace_sublabels {s['replace_sublabels']}")
        for k in LABELS:
            pc = s["per_cell"][k]
            if not pc["n"]:
                continue
            print(f"  {k}: n={pc['n']} mean_distinctive_recall={pc['mean_distinctive_recall']} "
                  f"mean_specific_values={pc['mean_n_specific_values']}")
            for e in pc["examples"][:3]:
                print(f"    - {e['id']} dr={e['distinctive_recall']} "
                      f"spec={list(e['specific_values'])[:5]}")
                print(f"      Q:    {e['question'][:110]}")
                print(f"      GOLD: {e['gold'][:110]}")
                print(f"      GEN:  {e['gen'][:110]!r}")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
