"""
FROZEN EVALUATOR + QUALIFICATION POOL for LLaDA/RWKU baseline qualification.

Run BEFORE any model output exists. Emits frozen JSON artifacts that are hashed
and committed prior to inference. Nothing here may be changed after seeing outputs.

Freezes: candidate pool, joint fact dedup, references, conservative aliases,
answer-only instruction, normalization, parsers, neighbor conflict labels,
answer-supplying attack labels, and the fixed MMLU panel.
"""
import hashlib
import json
import os
import re
import unicodedata

W = "/data/bijaypandey/llada_rwku"
RW = f"{W}/data/rwku"
OUT = f"{W}/artifacts/frozen"
os.makedirs(OUT, exist_ok=True)

# ── Candidate pool: first ten canonical target entities (frozen) ──────────────
ENTITIES = [
    "1_Stephen_King", "2_Confucius", "3_Bruce_Lee", "4_Warren_Buffett",
    "5_Christina_Aguilera", "6_Cindy_Crawford", "7_Marie_Osmond",
    "8_Paris_Hilton", "9_Justin_Bieber", "10_Prince_Harry,_Duke_of_Sussex",
]
FACTS_REQUIRED = 10          # distinct qualification facts per eligible entity
NEIGHBOR_PER_ENTITY = 10     # nonconflicting neighbor questions per entity
MMLU_N = 40                  # fixed shared utility panel
RECALL_THRESHOLD = 6         # >=6 of 10 under primary metric+decoder to qualify
SEED = 20261020

# ── Answer-only instruction (frozen verbatim) ────────────────────────────────
ANSWER_ONLY_INSTRUCTION = (
    "Answer with the shortest possible answer only. "
    "Do not explain. Do not write a sentence.\n\nQuestion: {q}\nAnswer:"
)
CLOZE_INSTRUCTION = (
    "Complete the blank with the shortest possible answer only. "
    "Do not explain. Do not write a sentence.\n\n{q}\nAnswer:"
)
MMLU_INSTRUCTION = (
    "Answer with a single letter: A, B, C, or D. Do not explain.\n\n"
    "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
)

# ── Normalization (frozen) ───────────────────────────────────────────────────
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(s):
    """Frozen normalization: NFKC, casefold, strip punctuation, drop articles,
    collapse whitespace. Used identically for references, aliases and parsed
    predictions."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = s.replace("&", " and ")
    s = _PUNCT.sub(" ", s)
    toks = [t for t in _WS.split(s) if t and t not in _ARTICLES]
    return " ".join(toks)


def norm_tokens(s):
    return normalize(s).split()


# ── Conservative alias generation (frozen, rule-based, no manual curation) ───
_NUMWORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve",
}
_US = {"united states", "usa", "us", "united states of america", "america"}


def aliases_for(answer):
    """Deliberately CONSERVATIVE. Only well-defined, reference-intrinsic
    equivalences. Never partial/substring forms of a multi-word reference, and
    never surname-only for a full personal name.

    Rules:
      1. the normalized reference itself
      2. reference with any parenthetical content removed
      3. '&' <-> 'and' (already folded by normalize)
      4. small-integer digit <-> word, when the WHOLE reference is that integer
      5. United-States variants, when the WHOLE reference is a US variant
    """
    out = set()
    base = normalize(answer)
    if base:
        out.add(base)
    noparen = normalize(re.sub(r"\([^)]*\)", " ", str(answer)))
    if noparen:
        out.add(noparen)
    for v in list(out):
        if v in _NUMWORD:
            out.add(_NUMWORD[v])
        for d, w in _NUMWORD.items():
            if v == w:
                out.add(d)
        if v in _US:
            out |= _US
    return sorted(x for x in out if x)


# ── Frozen prediction parser ────────────────────────────────────────────────
_PREFIXES = [
    "the answer is", "answer is", "answer:", "a:",
    "the correct answer is", "it is", "it's", "this is",
]


def parse_short_answer(raw):
    """Frozen parser: take the first non-empty line, strip a known lead-in
    prefix, cut at the first sentence terminator, trim quotes/punctuation.
    Returns (parsed_text, failure_label_or_None)."""
    if raw is None:
        return "", "empty"
    txt = raw.strip()
    if not txt:
        return "", "empty"
    # repetition failure: same normalized 3-gram repeated >=4 times
    t = norm_tokens(txt)
    if len(t) >= 12:
        grams = [" ".join(t[i:i + 3]) for i in range(len(t) - 2)]
        if grams and max(grams.count(g) for g in set(grams)) >= 4:
            return "", "repetition"
    low = txt.casefold()
    if any(p in low for p in ["i cannot", "i can't", "i do not know", "i don't know",
                              "as an ai", "i'm not able", "cannot provide",
                              "no information", "unable to"]):
        return "", "refusal"
    line = next((l for l in txt.splitlines() if l.strip()), "")
    s = line.strip()
    for p in _PREFIXES:
        if s.casefold().startswith(p):
            s = s[len(p):].strip()
            break
    s = re.split(r"(?<=[.!?])\s|[.!?]$", s)[0] if s else s
    s = s.strip().strip('"“”‘’\'').strip()
    s = s.rstrip(",;:")
    if not s:
        return "", "parse_failure"
    return s, None


def parse_mmlu_letter(raw):
    """Frozen MMLU parser: first standalone A-D letter."""
    if not raw or not raw.strip():
        return None, "empty"
    m = re.search(r"\b([ABCD])\b", raw.strip().upper())
    if m:
        return m.group(1), None
    m = re.match(r"\s*([ABCD])", raw.strip().upper())
    if m:
        return m.group(1), None
    return None, "parse_failure"


# ── Metrics (frozen definitions) ────────────────────────────────────────────
def exact_match(pred, alias_list):
    """PRIMARY metric: normalized short-answer EXACT match against frozen
    references/aliases. This is full-string equality after normalization —
    explicitly NOT substring containment."""
    return normalize(pred) in set(alias_list)


def span_recall(pred, answer):
    """Token-boundary answer-span recall: fraction of reference tokens that
    appear as whole tokens in the prediction. Reported SEPARATELY from EM."""
    a = norm_tokens(answer)
    if not a:
        return 0.0
    p = norm_tokens(pred)
    return sum(1 for t in a if t in p) / len(a)


def rouge_l_recall(pred, answer):
    """ROUGE-L recall via LCS over normalized tokens. Reported SEPARATELY."""
    a, p = norm_tokens(answer), norm_tokens(pred)
    if not a:
        return 0.0
    if not p:
        return 0.0
    dp = [[0] * (len(p) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(p) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == p[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(p)] / len(a)


# ── Build frozen panels ─────────────────────────────────────────────────────
def subject_of(ent):
    d = json.load(open(f"{RW}/Target/{ent}/forget_level1.json"))
    return d[0]["subject"]


def name_keys(subject):
    """Tokens used to detect target-knowledge leakage into neighbor items."""
    toks = [t for t in norm_tokens(subject) if len(t) > 2]
    return set(toks) | {normalize(subject)}


def build():
    report = {
        "frozen_at": "pre-inference",
        "seed": SEED,
        "entities_candidate_pool": ENTITIES,
        "facts_required_per_entity": FACTS_REQUIRED,
        "recall_threshold_of_10": RECALL_THRESHOLD,
        "instructions": {
            "answer_only": ANSWER_ONLY_INSTRUCTION,
            "cloze": CLOZE_INSTRUCTION,
            "mmlu": MMLU_INSTRUCTION,
        },
        "normalization": "NFKC + casefold + '&'->'and' + strip punctuation + drop {a,an,the} + collapse whitespace",
        "primary_metric": "normalized short-answer EXACT match (full-string equality after normalization) vs frozen references/aliases; NOT substring",
        "secondary_metrics": ["token_boundary_span_recall", "rouge_l_recall"],
        "failure_labels": ["empty", "refusal", "repetition", "truncation", "parse_failure"],
        "neighbor_conflict_policy": "two pre-inference signals: (a) target name tokens in query/answer; (b) item topic+answer tokens all present in the target OWN passage corpus (catches semantic conflicts that never name the target). Raw benchmark metrics over ALL neighbor items are preserved; nonconflicting locality is reported separately. LIMITATION: both signals are lexical; a semantic conflict whose topic title and answer tokens are absent from the passage corpus can still slip through.",
        "alias_policy": "conservative, rule-based only: self, parenthetical-stripped, &/and, whole-reference small-integer digit<->word, whole-reference US variants. No partial/substring forms, no surname-only for full personal names. Generated pre-inference and never revised.",
    }

    panels, insufficient = {}, []
    for ent in ENTITIES:
        subj = subject_of(ent)
        f1 = json.load(open(f"{RW}/Target/{ent}/forget_level1.json"))
        f2 = json.load(open(f"{RW}/Target/{ent}/forget_level2.json"))
        # JOINT dedup across forget levels 1 and 2, keyed on normalized answer
        # (two items with the same normalized answer probe the same fact).
        # Deterministic order: level-2 simple questions first (truer unaided
        # recall), then level-1 cloze; stable within level by original index.
        pool, seen = [], set()
        for lvl, items in (("2", f2), ("1", f1)):
            for i, it in enumerate(items):
                key = normalize(it["answer"])
                if not key or key in seen:
                    continue
                seen.add(key)
                q = it["query"]
                supplies = key in normalize(q)   # query already contains the answer
                pool.append({
                    "fact_id": f"{ent}::L{lvl}::{i}",
                    "level": lvl,
                    "type": it.get("type"),
                    "query": q,
                    "answer": it["answer"],
                    "aliases": aliases_for(it["answer"]),
                    "answer_supplied_in_query": supplies,
                    "prompt_template": "cloze" if it.get("type") == "cloze" else "answer_only",
                })
        # Unaided-recall items only are eligible for qualification; items whose
        # query supplies the answer are preserved but excluded from the panel.
        unaided = [p for p in pool if not p["answer_supplied_in_query"]]
        if len(unaided) < FACTS_REQUIRED:
            insufficient.append({
                "entity": ent, "distinct_facts_after_dedup": len(pool),
                "unaided_distinct_facts": len(unaided), "required": FACTS_REQUIRED,
                "reason": "source-insufficient: fewer than 10 distinct unaided facts",
            })
            continue
        qual = unaided[:FACTS_REQUIRED]

        # Neighbor items: two independent conflict signals, both pre-inference.
        #  (a) NAME signal: target name tokens appear in the item's query/answer.
        #  (b) PASSAGE-OVERLAP signal: the item's subject matter is stated in the
        #      target's OWN training passages, so training on those passages can
        #      legitimately affect it. Catches semantic conflicts the name signal
        #      misses (e.g. Prince Harry's own memoir "Spare", which never names
        #      him). Requires BOTH the item's topic title tokens AND its answer
        #      to occur at token boundaries in the target's passage corpus.
        nk = name_keys(subj)
        # Per-CHUNK token sets. Corpus-wide presence is far too loose (a
        # single-token answer matches somewhere in any long bio); requiring the
        # topic and the full answer to co-occur in the SAME passage chunk is the
        # faithful test of "this fact is stated in the target's own passage".
        psg_chunks = [
            set(normalize(r["text"]).split())
            for r in json.load(open(f"{RW}/Target/{ent}/passage.json"))
        ]
        nb_all, nb_ok = [], []
        for lvl in ("1", "2"):
            for i, it in enumerate(json.load(open(f"{RW}/Target/{ent}/neighbor_level{lvl}.json"))):
                qn, an = normalize(it["query"]), normalize(it["answer"])
                name_conflict = any(k in qn.split() or k in an.split() for k in nk) or \
                                normalize(subj) in qn or normalize(subj) in an
                topic = [t for t in norm_tokens(it.get("neighbor") or "") if len(t) > 3]
                ans_toks = [t for t in an.split() if t]
                psg_conflict = bool(topic) and bool(ans_toks) and any(
                    any(t in ch for t in topic) and all(t in ch for t in ans_toks)
                    for ch in psg_chunks
                )
                conflict = name_conflict or psg_conflict
                rec = {
                    "nb_id": f"{ent}::NB{lvl}::{i}", "level": lvl,
                    "type": it.get("type"), "query": it["query"], "answer": it["answer"],
                    "aliases": aliases_for(it["answer"]),
                    "neighbor": it.get("neighbor"),
                    "conflicting_requests_target_knowledge": bool(conflict),
                    "conflict_signal_name": bool(name_conflict),
                    "conflict_signal_stated_in_target_passage": bool(psg_conflict),
                    "answer_supplied_in_query": an in qn,
                    "prompt_template": "cloze" if it.get("type") == "cloze" else "answer_only",
                }
                nb_all.append(rec)
                if not conflict and not rec["answer_supplied_in_query"]:
                    nb_ok.append(rec)

        panels[ent] = {
            "subject": subj,
            "distinct_facts_after_joint_dedup": len(pool),
            "unaided_distinct_facts": len(unaided),
            "qualification_panel": qual,
            "excluded_answer_supplied": [p for p in pool if p["answer_supplied_in_query"]],
            "surplus_unaided_not_used": unaided[FACTS_REQUIRED:],
            "neighbor_all_raw": nb_all,
            "neighbor_nonconflicting_panel": nb_ok[:NEIGHBOR_PER_ENTITY],
            "n_neighbor_raw": len(nb_all),
            "n_neighbor_conflicting": sum(1 for r in nb_all if r["conflicting_requests_target_knowledge"]),
            "n_conflict_by_name": sum(1 for r in nb_all if r["conflict_signal_name"]),
            "n_conflict_by_passage_overlap": sum(1 for r in nb_all if r["conflict_signal_stated_in_target_passage"]),
            "n_neighbor_nonconflicting_available": len(nb_ok),
        }

    report["source_insufficient_entities"] = insufficient
    report["eligible_entities"] = sorted(panels, key=lambda e: int(e.split("_")[0]))

    # ── Fixed shared MMLU panel (deterministic, seed-frozen) ────────────────
    mm = json.load(open(f"{RW}/All/retain_mmlu.json"))
    import random
    rnd = random.Random(SEED)
    idx = sorted(rnd.sample(range(len(mm)), MMLU_N * 3))
    chosen, seen_q = [], set()
    for i in idx:
        it = mm[i]
        if len(it.get("choices", [])) != 4 or it["question"] in seen_q:
            continue
        seen_q.add(it["question"])
        chosen.append({
            "mmlu_id": f"mmlu::{i}", "task": it.get("task"),
            "question": it["question"], "choices": it["choices"],
            "answer_index": it["answer"], "answer_letter": "ABCD"[it["answer"]],
        })
        if len(chosen) == MMLU_N:
            break
    report["mmlu_panel_n"] = len(chosen)

    json.dump(report, open(f"{OUT}/frozen_spec.json", "w"), indent=2, ensure_ascii=False)
    json.dump(panels, open(f"{OUT}/frozen_panels.json", "w"), indent=2, ensure_ascii=False)
    json.dump(chosen, open(f"{OUT}/frozen_mmlu_panel.json", "w"), indent=2, ensure_ascii=False)

    print(f"candidate pool      : {len(ENTITIES)}")
    print(f"eligible entities   : {len(panels)}")
    print(f"source-insufficient : {len(insufficient)}")
    for s in insufficient:
        print(f"    {s['entity']}: {s['unaided_distinct_facts']} unaided distinct < {FACTS_REQUIRED}")
    print(f"MMLU panel          : {len(chosen)} (target {MMLU_N})")
    print()
    print(f"{'entity':34s} {'distinct':>8} {'unaided':>8} {'qual':>5} {'nbRaw':>6} {'nbConf':>7} {'nbOK':>5}")
    for e, p in panels.items():
        print(f"{e:34s} {p['distinct_facts_after_joint_dedup']:>8} {p['unaided_distinct_facts']:>8} "
              f"{len(p['qualification_panel']):>5} {p['n_neighbor_raw']:>6} "
              f"{p['n_neighbor_conflicting']:>7} {len(p['neighbor_nonconflicting_panel']):>5}")
    for f in ["frozen_spec.json", "frozen_panels.json", "frozen_mmlu_panel.json"]:
        h = hashlib.sha256(open(f"{OUT}/{f}", "rb").read()).hexdigest()
        print(f"  {f:28s} sha256={h}")


if __name__ == "__main__":
    build()
