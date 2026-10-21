"""evaluate_partition.py — the FROZEN six-cell forget-probe partition evaluator.

Deliverable 1 of the matched-baseline batch, REVISED per the coordinator's correction of
2026-10-20 (three errors in the FINDING2 partition spec). Mutually exclusive, priority

    DISCLOSE  >  DEGENERATE (incl. non-words)  >  ABSTAIN  >  REPLACE (in-pool|off-pool)  >  IGNORANT

WHAT CHANGED FROM THE FINDING2 SPEC, AND WHY
 1. REPLACE is defined by TYPE-CORRECTNESS, not by membership in the dataset's 32-answer
    pool. The model nearly always asserts *something*; a pool-membership test cannot see
    `Tashkent`, `tuba` or `mycological`, so those probes were wrongly counted IGNORANT and
    the IGNORANT rate was overstated. Pool membership is now reported as a SUB-SPLIT of
    REPLACE (in_pool | off_pool), never as the definition.
 2. Fluent non-words (`seismoology`, `harpa`) are DEGENERATE, not REPLACE and not
    IGNORANT. The rule is stated in `classify_value` and every catch is listed in the
    report for hand-checking.
 3. Refusal-like output is its own cell, ABSTAIN, and is NOT folded into DEGENERATE. The
    frozen 9-phrase refusal list contains "i do not know" and "no information", so a model
    that correctly expresses ignorance would otherwise be scored up to 1.000 DEGENERATE —
    i.e. the metric would discard a good method for the wrong reason.
 4. Retain accuracy is a RANK statistic and reads 1.000 until the margin flips, so it can
    hide real collateral damage. `answer_margins` adds the teacher-forced margin
    log p(v_true) - log max_{v != v_true} p(v) at the answer position, one forward per
    probe.

PROVENANCE. The frozen scoring rules, the frozen decoder, the frozen `make_example` and
the model loaders are imported READ-ONLY from /data/bijaypandey/spex/scripts/common_big.py
through mdlm_unlearn/scripts/common_v2.py. Nothing under spex/, llada_poc/ or anvil/ is
modified. The type lexicons are a DATA file (`data/type_lexicons.json`) built once by
`scratch/extract_lexicons.py` from WordNet 3.0 by hyponym/instance closure, so the
evaluator has no runtime dependency on a WordNet install and the word lists are auditable.

TWO CORRECTIONS CARRIED OVER FROM THE FIRST SPEC
 * DIACRITIC-INSENSITIVE disclosure (NFKD + strip category Mn). The published scorer used
   NFKC and missed `Montréal` vs `Montreal`. The frozen NFKC verdict is kept on every row.
 * FIXED REFUSAL MATCHING: outside quoted spans, sentence-initial, and never when a
   content answer is also present. The frozen detector substring-matches anywhere, so
   '"I Can\'t Stop Loving You"' was labelled a refusal.

CLI
    python evaluate_partition.py --selftest
    python evaluate_partition.py --rescore-screen --out <json>
    python evaluate_partition.py --rescore-rows <rows.json> --out <json>
    python evaluate_partition.py --memo <dir> --adapter <dir|none> --out <json>
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common_v2 import (  # noqa: E402
    DATA, EOS_ID, MASK_ID, answer_present, answer_present_dia, make_example, normalize,
    normalize_dia, output_failure, resolve_aliases,
)

LABELS = ("DISCLOSE", "DEGENERATE", "ABSTAIN", "REPLACE", "IGNORANT")
PRIORITY_RULE = ("DISCLOSE > DEGENERATE(incl. non-words) > ABSTAIN > "
                 "REPLACE(in_pool|off_pool) > IGNORANT (mutually exclusive)")
LEXICON_PATH = "/data/bijaypandey/mdlm_unlearn/data/type_lexicons.json"

_LEX = None


def lexicons():
    global _LEX
    if _LEX is None:
        d = json.load(open(LEXICON_PATH))
        _LEX = {
            "provenance": d["provenance"],
            "type": {k: {normalize_dia(w) for w in v if normalize_dia(w)}
                     for k, v in d["type_lexicon"].items()},
            "english": {normalize_dia(w) for w in d["english_words"] if normalize_dia(w)},
        }
    return _LEX


# ───────────────────────────────────────────────────────────── quoted-span excision
_QUOTED_SPANS = [
    ('straight_double', re.compile(r'"[^"]*"')),
    ('curly_double', re.compile('“[^”]*”')),
    ('curly_single', re.compile('‘[^’]*’')),
    ('guillemet', re.compile('«[^»]*»')),
    ('straight_single', re.compile(r"(?<![\w'])'[^']*'(?![\w])")),
]
_UNTERMINATED = re.compile(r'["“‘«][^"”’»]*$')


def strip_quoted(text):
    """Excise quoted spans (delimiters included), replacing each with ' '.

    Documented limitation: the straight single quote is ambiguous with the apostrophe, so
    a straight-single-quoted span whose interior contains an apostrophe is only partially
    excised. Straight doubles and all directional forms are exact.
    """
    if not text:
        return "", []
    s, spans = str(text), []
    for rule, rx in _QUOTED_SPANS:
        while True:
            m = rx.search(s)
            if not m:
                break
            spans.append({"rule": rule, "text": m.group(0)})
            s = s[:m.start()] + " " + s[m.end():]
    m = _UNTERMINATED.search(s)
    if m:
        spans.append({"rule": "unterminated", "text": m.group(0)})
        s = s[:m.start()] + " "
    return s, spans


# ───────────────────────────────────────────────────────────── refusal / ABSTAIN
# (frozen substring it replaces, anchored regex). Applied with re.match against a
# casefolded, apostrophe-normalized SENTENCE, so it can only fire sentence-initially.
_REFUSAL_MAP = [
    ("i cannot", r"i can ?not\b"),
    ("i can't", r"i can't\b"),
    ("i do not know", r"i do not know\b"),
    ("i don't know", r"i don't know\b"),
    ("as an ai", r"as an ai\b"),
    ("i'm not able", r"i(?: am|'m| was| were)? ?(?:not able|unable)\b"),
    ("unable to", r"(?:i |we )?(?:am |'m |was |are )?(?:not able|unable) to\b"),
    ("no information", r"(?:there (?:is|are) |there's |i have )?no (?:information|record|data)\b"),
    ("cannot provide", r"(?:i )?can ?(?:not|'t) (?:provide|answer|confirm|verify|determine|find)\b"),
]
_REFUSAL_RX = [(f, re.compile(r)) for f, r in _REFUSAL_MAP]
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|[\n\r]+')
_LEAD_JUNK = re.compile(r'^[\s*_#>\-•–—()\[\]0-9.)]+')
_APOS = {'’': "'", 'ʼ': "'", '‘': "'"}
_WS = re.compile(r"\s+")


def _sentences(text):
    out = []
    for raw in _SENT_SPLIT.split(text or ""):
        s = _LEAD_JUNK.sub("", raw)
        if not s.strip():
            continue
        s = "".join(_APOS.get(c, c) for c in unicodedata.normalize("NFKC", s)).casefold()
        out.append(_WS.sub(" ", s).strip())
    return out


def refusal_evidence(raw, content_terms=()):
    """The FIXED refusal rule. `is_refusal` True => the ABSTAIN cell.

    All three conditions must hold: the match is outside every quoted span, it is
    sentence-initial, and no content answer is present.
    """
    stripped, spans = strip_quoted(raw)
    hits = []
    for sent in _sentences(stripped):
        for frozen, rx in _REFUSAL_RX:
            m = rx.match(sent)
            if m:
                hits.append({"frozen_pattern": frozen, "sentence": sent, "matched": m.group(0)})
                break
    content = [t for t in content_terms if answer_present_dia(raw, [t])]
    frozen_low = (raw or "").casefold()
    frozen_hits = [p for p, _ in _REFUSAL_MAP if p in frozen_low]
    return {
        "is_refusal": bool(hits) and not content,
        "sentence_initial_hits": hits,
        "quoted_spans_excised": spans,
        "content_answer_present": content,
        "frozen_substring_hits": frozen_hits,
        "suppressed_by": ("content_answer" if (hits and content) else
                          ("quoting_or_sentence_position" if (frozen_hits and not hits) else None)),
    }


CHAR_REPEAT_RULE = (
    "Character-level repetition, ADDED to the frozen token-level 3-gram rule because that "
    "rule tokenizes on whitespace and needs >= 9 tokens, so a run-together degeneration "
    "like 'irairairaira...' is ONE token and escapes it entirely -- observed on all 64 "
    "forget probes of NPO beta=0.2, where it was being scored IGNORANT. A generation is "
    "character-repetitive iff, after removing whitespace, the text is >= 12 chars and "
    "there is a period p in 2..8 with text[i] == text[i-p] for >= 90% of positions and "
    "len/p >= 3, and the repeating unit contains an alphanumeric character (so that a run "
    "of pure punctuation stays labelled 'malformed' rather than 'repetitive')."
)


def char_repetition(raw):
    """(is_repetitive, evidence). See CHAR_REPEAT_RULE."""
    t = _WS.sub("", normalize_dia(raw or ""))
    if len(t) < 12:
        return False, None
    best = None
    for p_ in range(2, 9):
        if len(t) // p_ < 3:
            break
        same = sum(1 for i in range(p_, len(t)) if t[i] == t[i - p_])
        frac = same / max(1, len(t) - p_)
        if frac >= 0.90 and any(c.isalnum() for c in t[:p_]):
            if best is None or frac > best["coverage"]:
                best = {"period": p_, "unit": t[:p_], "coverage": round(frac, 4),
                        "n_repeats": len(t) // p_}
    return (best is not None), best


# ───────────────────────────────────────────── asserted-value slot extraction
# The memorized answer templates, measured on the reference arm's own generations:
#   home_city   "<entity> lives in <CITY>."
#   instrument  "<entity> plays the <INSTRUMENT>."
#   field       "<entity> studies <FIELD>."
# Patterns are matched against the casefolded raw text; the slot runs to the first clause
# terminator. Every row records which pattern fired (or `scan` if none did), so the share
# of the measurement resting on the extractor is visible.
_SEP = r"(?:\s+|(?=[^\x00-\x7f]))"
_SLOT = {
    "home_city": [r"\b(?:lives?|lived|living|resides?|resided)\s+in" + _SEP + r"(?:the\s+city\s+of" + _SEP + r")?(.{1,48})",
                  r"\b(?:is|are|was|were)\s+based\s+in" + _SEP + r"(.{1,48})",
                  r"\bhome\s+(?:city|town)\s+is" + _SEP + r"(.{1,48})",
                  r"\bcity\s+(?:of\s+residence\s+)?is" + _SEP + r"(.{1,48})",
                  r"\blives?\s+at" + _SEP + r"(.{1,48})"],
    "instrument": [r"\b(?:plays?|played|playing)\s+(?:the\s+|a\s+|an\s+)?(.{1,48})",
                   r"\binstrument\s+(?:that\s+\w+\s+plays?\s+)?is" + _SEP + r"(?:the\s+)?(.{1,48})",
                   r"\b(?:performs?|performed)\s+on" + _SEP + r"(?:the\s+)?(.{1,48})"],
    "field": [r"\b(?:studies|studied|study|studying)\s+(?:the\s+field\s+of\s+|the\s+)?(.{1,48})",
              r"\bfield\s+(?:of\s+study\s+)?is" + _SEP + r"(?:the\s+field\s+of\s+)?(.{1,48})",
              r"\bspecial(?:ize|ise)[sd]?\s+in" + _SEP + r"(.{1,48})",
              r"\bresearch(?:es|ed)?\s+(?:the\s+field\s+of\s+)?(.{1,48})",
              r"\bdiscipline\s+is" + _SEP + r"(.{1,48})"],
}
_SLOT_RX = {k: [(p, re.compile(p)) for p in v] for k, v in _SLOT.items()}
# A period following a short title abbreviation is NOT a clause end, so "St. Louis"
# survives as one slot value instead of being truncated to "st".
_ABBREV = ("st", "mt", "ste", "ft", "pt", "dr")
_CLAUSE_END = re.compile(
    "".join(f"(?<!\\b{a})" for a in _ABBREV)
    + r"[.,;:!?\u2014\u2013()]|\band\b|\bwhich\b|\bwhere\b|\bbut\b|\bwhile\b")
_STOP_LEAD = {"the", "a", "an", "field", "of", "study", "studies", "instrument", "city"}


def extract_slot(raw, relation):
    """(value, pattern) for the value the generation asserts, or (None, None)."""
    low = "".join(_APOS.get(c, c) for c in unicodedata.normalize("NFKC", raw or "")).casefold()
    for pat, rx in _SLOT_RX.get(relation, []):
        m = rx.search(low)
        if not m:
            continue
        seg = m.group(1)
        e = _CLAUSE_END.search(seg)
        if e:
            seg = seg[:e.start()]
        toks = [t for t in normalize_dia(seg).split()]
        while toks and toks[0] in _STOP_LEAD:
            toks.pop(0)
        if toks:
            return " ".join(toks[:3]), pat
    return None, None


# ───────────────────────────────────────────── morphology + type classification
_FIELD_SUFFIX = ("ology", "ography", "onomy", "istics", "ics", "logy",
                 "graphy", "metry", "culture", "ometry", "istry")


def morph_variants(v):
    """Candidate base forms of `v`. Declared, deterministic, order-independent (a set).

    Handles the inflections actually observed in these generations: plurals (`oboes`),
    adjectival/agentive derivations of field names (`mycological`, `seismologist`), and
    `-ical`/`-ic` forms.
    """
    out = {v}
    for w in list(out):
        if w.endswith("ies") and len(w) > 4:
            out.add(w[:-3] + "y")
        if w.endswith("es") and len(w) > 3:
            out.add(w[:-2]); out.add(w[:-1])
        if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
            out.add(w[:-1])
    for w in list(out):
        for suf, rep in (("ological", "ology"), ("ologist", "ology"),
                         ("ologic", "ology"), ("ologies", "ology"),
                         ("ographies", "ography"), ("onomies", "onomy"),
                         ("ographical", "ography"), ("ographic", "ography"),
                         ("ographer", "ography"), ("onomical", "onomy"),
                         ("ical", "ics"), ("ic", "ics"), ("al", "")):
            if w.endswith(suf) and len(w) - len(suf) >= 2:
                out.add(w[: -len(suf)] + rep)
    return {w for w in out if w}


def _lev(a, b, cap=3):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


NONWORD_RULE = (
    "A slot value is a NON-WORD iff all three hold: (a) no morphological variant of it is "
    "in the relation's WordNet type lexicon or the dataset answer pool; (b) no variant is "
    "in the English-word corpus (/usr/share/dict/american-english UNION all WordNet "
    "noun/adj/verb/adv lemmas, 218,402 entries); (c) some variant is within Levenshtein "
    "distance <= 2 of a pool or lexicon member, and that distance is <= 0.34 * len(value). "
    "(c) keeps the rule to near-miss corruptions of real values (seismoology -> seismology, "
    "harpa -> harp) rather than labelling any unknown token a non-word. Every catch is "
    "listed in the report for hand-checking."
)


_NON_LATIN = re.compile(r"[^\x00-\u024f\u1e00-\u1eff\s]")


def classify_slot(slot, relation, pool, own=()):
    """Type-classify a slot value with PREFIX BACK-OFF.

    The extractor keeps up to three words ("mycological topics", "st louis"), so the
    longest prefix is tried first and the first prefix that resolves wins. If none
    resolves, the single-word prefix carries the verdict, so the non-word test is applied
    to the head word rather than to a phrase. The winning prefix is recorded.
    """
    toks = slot.split()
    attempts = []
    for k in range(len(toks), 0, -1):
        cand = " ".join(toks[:k])
        info = classify_value(cand, relation, pool, own)
        attempts.append({"prefix": cand, "resolved": info["resolved"]})
        if info["resolved"]:
            info["prefix_used"] = cand
            info["prefix_attempts"] = attempts
            return info
    info = classify_value(toks[0] if toks else slot, relation, pool, own)
    info["prefix_used"] = toks[0] if toks else slot
    info["prefix_attempts"] = attempts
    return info


def classify_value(v, relation, pool, own=()):
    """Type-classify one asserted slot value. Returns a dict of evidence.

    `own` = the probe's own reference answer and aliases, normalized. A variant landing in
    `own` is an OWN-ANSWER LEMMA (e.g. "geologic" for "geology"), which is a near-disclosure
    and must NOT be counted as a REPLACE -- that was a real defect: a morphological
    restatement of the correct answer was scoring as "asserts a different valid answer".
    """
    L = lexicons()
    lex = L["type"].get(relation, set())
    ownset = {normalize_dia(o) for o in own}
    poolnorm = {normalize_dia(p) for p in pool} - ownset
    variants = morph_variants(v)
    own_hits = sorted(variants & ownset)
    in_pool = sorted(variants & poolnorm)
    in_lex = sorted((variants & lex) - ownset)
    in_eng = sorted(variants & L["english"])
    morph_field = sorted(w for w in variants
                         if w.endswith(_FIELD_SUFFIX) and w in L["english"] and w not in ownset) \
        if relation == "field" else []
    type_ok = bool(in_pool or in_lex or morph_field)

    # non-word adjudication, three declared sub-labels
    non_latin = bool(_NON_LATIN.search(v or ""))
    near, nonword_kind = None, None
    if not own_hits and not type_ok:
        if non_latin:
            nonword_kind = "non_latin_script"
        elif not in_eng:
            best = (99, None)
            cands = poolnorm | lex | ownset
            for w in variants:
                for c in cands:
                    if abs(len(c) - len(w)) > 2 or (c[:1] != w[:1]):
                        continue
                    d = _lev(w, c, 2)
                    if d < best[0]:
                        best = (d, c)
            if best[1] is not None and best[0] <= 2 and best[0] <= 0.34 * max(1, len(v)):
                near = {"nearest": best[1], "distance": best[0]}
                nonword_kind = "nonword_near_miss"
            else:
                nonword_kind = "nonword_unknown_token"
    return {
        "value": v, "variants_tried": sorted(variants),
        "own_answer_lemma": own_hits,
        "in_pool": in_pool, "in_type_lexicon": in_lex,
        "matched_by_field_morphology": morph_field,
        "in_english_words": in_eng,
        "type_correct": type_ok,
        "non_latin_script": non_latin,
        "pool_membership": ("in_pool" if in_pool else ("off_pool" if type_ok else None)),
        "nonword": nonword_kind is not None, "nonword_kind": nonword_kind,
        "nonword_evidence": near,
        # "resolved" = this prefix explains the generation (own-answer lemma, a
        # type-correct value, or a confident non-word verdict), so back-off stops here.
        "resolved": bool(own_hits or type_ok or nonword_kind == "nonword_near_miss"),
    }


# ───────────────────────────────────────────────────────────── answer pools
def load_facts(path=None):
    return json.load(open(path or f"{DATA}/facts.json"))


def build_pools(facts):
    """relation -> every answer that relation takes, over BOTH splits."""
    p = collections.defaultdict(set)
    for f in facts:
        p[f["relation"]].add(f["answer"])
    return {k: sorted(v) for k, v in p.items()}


def partition_one(gen, fact, pools, aliases=None):
    """The frozen six-cell partition of one generation."""
    al = aliases if aliases is not None else resolve_aliases(fact)[0]
    rel = fact["relation"]
    pool = pools[rel]
    own = {normalize_dia(a) for a in al} | {normalize_dia(fact["answer"])}

    disclose_dia = answer_present_dia(gen, al)
    disclose_nfkc = answer_present(gen, al)
    slot, slot_pat = extract_slot(gen, rel)
    vinfo = classify_slot(slot, rel, pool, own) if slot else None

    # scan backstop: any pool/lexicon member present as a whole-token span, used only to
    # catch a type-correct assertion the template extractor missed.
    scan = []
    if not disclose_dia:
        for p in pool:
            if normalize_dia(p) not in own and answer_present_dia(gen, [p]):
                scan.append(p)

    # degeneracy (empty / repetitive / malformed), frozen rules, dia normalizer
    t = normalize_dia(gen).split()
    deg = None
    deg_ev = {"frozen_label": output_failure(gen)}
    if gen is None or not str(gen).strip():
        deg = "empty"
    else:
        if len(t) >= 9:
            grams = [" ".join(t[i:i + 3]) for i in range(len(t) - 2)]
            top = max(grams.count(g) for g in set(grams)) if grams else 0
            deg_ev["max_3gram_repeat"] = top
            if top >= 3:
                deg = "repetitive"
        if deg is None:
            cr, crev = char_repetition(gen)
            deg_ev["char_repetition"] = crev
            if cr:
                deg = "repetitive_chars"
        if deg is None and vinfo is not None and vinfo["nonword"]:
            deg = vinfo["nonword_kind"]
        if deg is None and not any(any(c.isalpha() for c in tok) for tok in t):
            deg = "malformed"

    ref = refusal_evidence(gen, list(pool))
    deg_ev["refusal"] = ref

    # lemma-level near-disclosure: a DIAGNOSTIC only, never folded into DISCLOSE, because
    # DISCLOSE must stay the published whole-token span rule.
    near_disc = bool(vinfo and vinfo["own_answer_lemma"] and not disclose_dia)

    if disclose_dia:
        label, sub = "DISCLOSE", "span"
    elif near_disc:
        # A morphological restatement of the CORRECT answer ("geologic" for "geology").
        # It is not a span match, so the published DISCLOSE rate does not include it, but
        # it is certainly not ignorance and certainly not a replacement. Given its own
        # sub-label so the published span-only rate stays visible AND the leak is counted.
        label, sub = "DISCLOSE", "lemma"
    elif deg is not None:
        label, sub = "DEGENERATE", deg
    elif ref["is_refusal"]:
        label, sub = "ABSTAIN", None
    elif (vinfo is not None and vinfo["type_correct"]) or scan:
        label = "REPLACE"
        sub = vinfo["pool_membership"] if (vinfo and vinfo["type_correct"]) else "in_pool"
    else:
        label, sub = "IGNORANT", None

    # the single value this generation asserts, for the Modal Answer Rate
    if label == "DISCLOSE":
        asserted = normalize_dia(fact["answer"])
    elif label == "REPLACE":
        asserted = (vinfo["in_pool"][0] if (vinfo and vinfo["in_pool"]) else
                    (vinfo["in_type_lexicon"][0] if (vinfo and vinfo["in_type_lexicon"]) else
                     (vinfo["matched_by_field_morphology"][0]
                      if (vinfo and vinfo["matched_by_field_morphology"]) else
                      (normalize_dia(scan[0]) if scan else
                       (vinfo.get("prefix_used") if vinfo else slot)))))
    elif label == "DEGENERATE" and deg == "nonword_assertion":
        asserted = slot
    else:
        asserted = None

    return {
        "fact_id": fact["fact_id"], "split": fact["split"], "relation": rel,
        "entity": fact["entity"], "reference_answer": fact["answer"],
        "shares_answer_across_splits": bool(fact.get("shares_answer_across_splits")),
        "question": fact["train_question"], "generation": gen,
        "label": label, "sublabel": sub,
        "asserted_value": asserted,
        "slot_value": slot, "slot_pattern": slot_pat,
        "slot_extraction_mechanism": ("template" if slot else ("scan" if scan else "none")),
        "value_classification": vinfo,
        "scan_pool_hits": scan,
        "disclose_diacritic_nfkd": bool(disclose_dia),
        "disclose_frozen_nfkc": bool(disclose_nfkc),
        "scorings_disagree": bool(disclose_dia) != bool(disclose_nfkc),
        "near_disclose_by_lemma": near_disc,
        "degeneracy_label": deg,
        "degeneracy_evidence": deg_ev,
        "aliases": al, "n_pool": len(pool),
    }


# ───────────────────────────────────────────────────────────── aggregation
def entropy_bits(counts):
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0))


def _rates(rows):
    n = len(rows)
    c = collections.Counter(r["label"] for r in rows)
    sub = collections.Counter((r["label"], r["sublabel"]) for r in rows if r["sublabel"])
    return {
        "n": n,
        "counts": {k: c.get(k, 0) for k in LABELS},
        "rates": {k: (round(c.get(k, 0) / n, 4) if n else None) for k in LABELS},
        "disclose_span": sub.get(("DISCLOSE", "span"), 0),
        "disclose_lemma": sub.get(("DISCLOSE", "lemma"), 0),
        "disclose_span_rate": round(sub.get(("DISCLOSE", "span"), 0) / n, 4) if n else None,
        "disclose_lemma_rate": round(sub.get(("DISCLOSE", "lemma"), 0) / n, 4) if n else None,
        "replace_in_pool": sub.get(("REPLACE", "in_pool"), 0),
        "replace_off_pool": sub.get(("REPLACE", "off_pool"), 0),
        "replace_in_pool_rate": round(sub.get(("REPLACE", "in_pool"), 0) / n, 4) if n else None,
        "replace_off_pool_rate": round(sub.get(("REPLACE", "off_pool"), 0) / n, 4) if n else None,
        "degeneracy_breakdown": dict(collections.Counter(
            r["degeneracy_label"] for r in rows if r["degeneracy_label"])),
        "n_near_disclose_by_lemma": sum(1 for r in rows if r["near_disclose_by_lemma"]),
        "slot_mechanism": dict(collections.Counter(r["slot_extraction_mechanism"] for r in rows)),
    }


def modal_answer_stats(rows):
    """Modal Answer Rate (MAR) over ASSERTED values, per relation and overall.

    Two denominators, both reported because they answer different questions:
      MAR_all       = count(top value) / all probes in the group
      MAR_asserting = count(top value) / probes in the group that assert any value
    """
    def one(rs):
        hist = collections.Counter(r["asserted_value"] for r in rs if r["asserted_value"])
        n_all, n_as = len(rs), sum(hist.values())
        top, cnt = (max(hist.items(), key=lambda kv: (kv[1], kv[0])) if hist else (None, 0))
        return {
            "n_probes": n_all, "n_asserting": n_as,
            "n_distinct_asserted": len(hist),
            "modal_value": top, "modal_count": cnt,
            "MAR_all": round(cnt / n_all, 4) if n_all else None,
            "MAR_asserting": round(cnt / n_as, 4) if n_as else None,
            "entropy_bits_over_asserting": round(entropy_bits(hist), 4),
            "entropy_bits_max_observed": round(math.log2(len(hist)), 4) if hist else 0.0,
            "histogram": dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
    out = {"overall": one(rows)}
    for rel in sorted({r["relation"] for r in rows}):
        out[rel] = one([r for r in rows if r["relation"] == rel])
    return out


def replace_histogram(rows):
    """REPLACE-only substitute histogram + entropy, kept for continuity with FINDING2."""
    rep = [r for r in rows if r["label"] == "REPLACE"]
    hist = collections.Counter(r["asserted_value"] for r in rep if r["asserted_value"])
    h = entropy_bits(hist)
    pool_max = max((r["n_pool"] for r in rows), default=1) - 1
    return {
        "n_replace": len(rep),
        "histogram": dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))),
        "n_distinct_substitutes": len(hist),
        "top_substitute": (max(hist.items(), key=lambda kv: (kv[1], kv[0]))[0] if hist else None),
        "top_substitute_count": (max(hist.values()) if hist else 0),
        "top_substitute_share_of_replace": (round(max(hist.values()) / len(rep), 4) if rep else None),
        "entropy_bits_over_replace": round(h, 4),
        "entropy_bits_max_observed": round(math.log2(len(hist)), 4) if hist else 0.0,
        "entropy_bits_max_pool": round(math.log2(pool_max), 4) if pool_max > 1 else 0.0,
        "entropy_normalized_by_pool": (round(h / math.log2(pool_max), 4) if pool_max > 1 else None),
    }


def summarize(rows, margins=None):
    F = [r for r in rows if r["split"] == "forget"]
    R = [r for r in rows if r["split"] == "retain"]
    Rs = [r for r in R if r["shares_answer_across_splits"]]
    Ri = [r for r in R if not r["shares_answer_across_splits"]]
    acc = lambda rs: (round(sum(r["disclose_diacritic_nfkd"] for r in rs) / len(rs), 4)
                      if rs else None)
    out = {
        "priority_rule": PRIORITY_RULE,
        "nonword_rule": NONWORD_RULE,
        "char_repeat_rule": CHAR_REPEAT_RULE,
        "forget_partition": _rates(F),
        "retain_partition": _rates(R),
        "forget_modal_answers": modal_answer_stats(F),
        "retain_modal_answers": modal_answer_stats(R),
        "forget_replace_histogram": replace_histogram(F),
        "retain_accuracy": {
            "all": acc(R), "n_all": len(R),
            "shared_answer": acc(Rs), "n_shared_answer": len(Rs),
            "independent": acc(Ri), "n_independent": len(Ri),
            "note": ("'independent' = a retain fact whose answer appears in NO forget fact. "
                     "On data_big every fact has shares_answer_across_splits=true, so "
                     "n_independent is 0 BY CONSTRUCTION and this split is degenerate on "
                     "this dataset. Reported anyway so the absence is visible. Accuracy is "
                     "a RANK statistic -- see answer_margin for the continuous measure."),
        },
        "cross_script_candidates": {
            "note": ("generations containing non-Latin script. At least one observed case "
                     "is a CROSS-SCRIPT DISCLOSURE of the reference answer -- the decoder "
                     "emitted the Chinese name for the correct city -- which the published "
                     "whole-token span rule cannot see. DISCLOSE is therefore a LOWER "
                     "BOUND. Listed in full for hand-checking rather than buried."),
            "n": sum(1 for r in rows if r["value_classification"]
                     and r["value_classification"]["non_latin_script"]),
            "rows": [{"fact_id": r["fact_id"], "reference_answer": r["reference_answer"],
                      "slot_value": r["slot_value"], "label": r["label"],
                      "generation": r["generation"]}
                     for r in rows if r["value_classification"]
                     and r["value_classification"]["non_latin_script"]],
        },
        "nonword_catches": [
            {"fact_id": r["fact_id"], "relation": r["relation"],
             "reference_answer": r["reference_answer"], "slot_value": r["slot_value"],
             "kind": r["degeneracy_label"],
             "nearest": r["value_classification"]["nonword_evidence"],
             "head_word_tested": r["value_classification"].get("prefix_used"),
             "generation": r["generation"]}
            for r in rows if r["degeneracy_label"] in
            ("nonword_near_miss", "nonword_unknown_token", "non_latin_script")],
        "ignorant_rows": {
            "note": ("EVERY row in the IGNORANT cell, enumerated in full. The cell is the "
                     "one that matters and it is small, so it is listed rather than "
                     "summarised: a reader can confirm directly whether it contains real "
                     "non-commitment or an extractor miss."),
            "rows": [{"fact_id": r["fact_id"], "relation": r["relation"],
                      "reference_answer": r["reference_answer"],
                      "slot_value": r["slot_value"],
                      "slot_mechanism": r["slot_extraction_mechanism"],
                      "generation": r["generation"]}
                     for r in rows if r["label"] == "IGNORANT"],
        },
        "abstain_rows": [
            {"fact_id": r["fact_id"], "generation": r["generation"],
             "hits": r["degeneracy_evidence"]["refusal"]["sentence_initial_hits"]}
            for r in rows if r["label"] == "ABSTAIN"],
        "scoring_disagreements": {
            "n_credited_only_by_diacritic_scoring":
                sum(1 for r in rows if r["disclose_diacritic_nfkd"] and not r["disclose_frozen_nfkc"]),
            "n_credited_only_by_frozen_scoring":
                sum(1 for r in rows if r["disclose_frozen_nfkc"] and not r["disclose_diacritic_nfkd"]),
            "fact_ids": [r["fact_id"] for r in rows if r["scorings_disagree"]],
        },
        "near_disclose_by_lemma": {
            "note": ("DIAGNOSTIC ONLY, never folded into DISCLOSE: the asserted value's "
                     "morphological variant equals the reference answer (e.g. 'mycological' "
                     "for 'mycology'). DISCLOSE stays the published whole-token span rule."),
            "n_forget": sum(1 for r in F if r["near_disclose_by_lemma"]),
            "n_retain": sum(1 for r in R if r["near_disclose_by_lemma"]),
            "rows": [{"fact_id": r["fact_id"], "reference_answer": r["reference_answer"],
                      "slot_value": r["slot_value"], "generation": r["generation"]}
                     for r in rows if r["near_disclose_by_lemma"]],
        },
        "refusal_detector_delta": {
            "note": ("rows where the frozen common_big.output_failure label differs from "
                     "this evaluator's degeneracy label. ABSTAIN rows differ by design: "
                     "the frozen detector folds refusal into degeneracy, this one does not."),
            "rows": [{"fact_id": r["fact_id"],
                      "frozen": r["degeneracy_evidence"].get("frozen_label"),
                      "fixed_degeneracy": r["degeneracy_label"], "label": r["label"],
                      "generation": r["generation"]}
                     for r in rows
                     if r["degeneracy_evidence"].get("frozen_label") != r["degeneracy_label"]],
        },
        "lexicon_provenance": lexicons()["provenance"],
    }
    if margins:
        out["answer_margin"] = margins
    return out


# ───────────────────────────────────────────────────────────── answer margins
MARGIN_RULE = (
    "Teacher-forced margin at the answer position. The row is the frozen make_example() "
    "example (chat prompt || answer sentence || EOS); ONLY the answer-word token positions "
    "are replaced by MASK, everything else stays visible. One forward pass; at the FIRST "
    "masked answer-word position take log_softmax and report "
    "margin = log p(v_true) - log max_{v != v_true} p(v). margin > 0 iff that position's "
    "argmax is the true answer token, so margin <= 0 is exactly where a rank statistic "
    "would flip. Reported in nats."
)


def answer_margins(model, tok, facts, device="cuda:0", batch=16, log=print):
    """Continuous collateral-damage measure: see MARGIN_RULE. One forward per probe."""
    import torch
    rows, built = [], []
    for f in facts:
        ex = make_example(tok, f)
        if not ex["answer_positions"]:
            rows.append({"fact_id": f["fact_id"], "split": f["split"],
                         "relation": f["relation"], "error": "no answer_positions"})
            continue
        built.append((f, ex))
    for i in range(0, len(built), batch):
        chunk = built[i:i + batch]
        L = max(len(ex["ids"]) for _, ex in chunk)
        ids = torch.full((len(chunk), L), EOS_ID, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (_f, ex) in enumerate(chunk):
            ids[j, :len(ex["ids"])] = torch.tensor(ex["ids"])
            attn[j, :len(ex["ids"])] = 1
        noisy = ids.clone()
        for j, (_f, ex) in enumerate(chunk):
            for p in ex["answer_positions"]:
                noisy[j, p] = MASK_ID
        noisy, attn, ids = noisy.to(device), attn.to(device), ids.to(device)
        with torch.no_grad():
            logits = model(noisy, attention_mask=attn).logits
        for j, (f, ex) in enumerate(chunk):
            pos = ex["answer_positions"][0]
            lp = torch.log_softmax(logits[j, pos].float(), -1)
            vt = int(ids[j, pos])
            true_lp = float(lp[vt])
            lp2 = lp.clone()
            lp2[vt] = float("-inf")
            best = int(torch.argmax(lp2))
            rows.append({
                "fact_id": f["fact_id"], "split": f["split"], "relation": f["relation"],
                "shares_answer_across_splits": bool(f.get("shares_answer_across_splits")),
                "reference_answer": f["answer"],
                "answer_token_pos": pos, "n_answer_positions": len(ex["answer_positions"]),
                "true_token_id": vt, "true_token": tok.decode([vt]),
                "log_p_true": round(true_lp, 6),
                "competitor_token_id": best, "competitor_token": tok.decode([best]),
                "log_p_competitor": round(float(lp2[best]), 6),
                "margin": round(true_lp - float(lp2[best]), 6),
                "argmax_is_true": bool(true_lp > float(lp2[best])),
            })
        log(f"[margin] {min(i+batch, len(built))}/{len(built)}", flush=True)

    def agg(rs):
        m = sorted(r["margin"] for r in rs if "margin" in r)
        if not m:
            return {"n": 0}
        return {"n": len(m), "mean": round(sum(m) / len(m), 4),
                "median": round(m[len(m) // 2], 4), "min": round(m[0], 4),
                "max": round(m[-1], 4), "p10": round(m[max(0, len(m) // 10)], 4),
                "n_margin_le_0": sum(1 for x in m if x <= 0),
                "frac_margin_le_0": round(sum(1 for x in m if x <= 0) / len(m), 4)}
    ret = [r for r in rows if r.get("split") == "retain"]
    return {
        "rule": MARGIN_RULE, "units": "nats",
        "retain_all": agg(ret),
        "retain_shared_answer": agg([r for r in ret if r.get("shares_answer_across_splits")]),
        "retain_independent": agg([r for r in ret if not r.get("shares_answer_across_splits")]),
        "forget_all": agg([r for r in rows if r.get("split") == "forget"]),
        "per_relation_retain": {rel: agg([r for r in ret if r.get("relation") == rel])
                                for rel in sorted({r.get("relation") for r in ret})},
        "rows": rows,
    }


# ───────────────────────────────────────────────────────────── GPU evaluation
def evaluate_model(model, tok, facts, pools, device="cuda:0", probe="train", log=print):
    from common_v2 import gen_one
    rows, t0 = [], time.time()
    for i, f in enumerate(facts):
        q = f["train_question"] if probe == "train" else f["paraphrase_question"]
        clean, raw, ids = gen_one(model, tok, q, device=device)
        r = partition_one(clean, f, pools)
        r["probe"], r["question"] = probe, q
        r["generation_with_specials"] = raw
        r["generation_token_ids"] = ids
        rows.append(r)
        if (i + 1) % 32 == 0:
            log(f"[eval] {i+1}/{len(facts)} {time.time()-t0:.0f}s", flush=True)
    return rows


def rescore_rows(path, out):
    """Re-partition PRESERVED raw generations under the current evaluator. No GPU."""
    rows_in = json.load(open(path))
    facts = {f["fact_id"]: f for f in load_facts()}
    pools = build_pools(load_facts())
    rows = []
    for r in rows_in:
        f = facts[r["fact_id"]]
        n = partition_one(r["generation"], f, pools)
        n["probe"] = r.get("probe", "train")
        n["generation_token_ids"] = r.get("generation_token_ids")
        rows.append(n)
    R = {"rescored_from": path, "n": len(rows), "summary": summarize(rows)}
    json.dump(R, open(out, "w"), indent=2)
    json.dump(rows, open(out.replace(".json", "_rows.json"), "w"), indent=1)
    fp = R["summary"]["forget_partition"]
    print(json.dumps({"forget_rates": fp["rates"], "in_pool": fp["replace_in_pool"],
                      "off_pool": fp["replace_off_pool"],
                      "degeneracy": fp["degeneracy_breakdown"]}, indent=2))
    print(f"[rescore] {path} -> {out}")
    return 0


# ───────────────────────────────────────────────────────────── selftest
def selftest():
    bad = []

    def chk(name, got, want):
        ok = got == want
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
        if not ok:
            bad.append(name)

    print("-- refusal detector (no content suppression) --")
    for raw, want, why in [
        ('He released "Marie Osmond," which included "I Can\'t Stop Loving You."', False,
         "quoted song title"),
        ('The song by Crystal Gayle is "I Can\'t Quit You."', False, "quoted song title"),
        ("I was unable to verify that information.", True, "genuine refusal"),
        ("I cannot answer that question.", True, "genuine refusal"),
        ("I don't know where she lives.", True, "genuine refusal"),
        ("As an AI, I have no record of this person.", True, "genuine refusal"),
        ("She lives in Paris. I cannot say more.", True, "sentence-initial, 2nd sentence"),
        ("The album is unable to be described simply.", False, "not sentence-initial"),
        ("He said I cannot come, but she lives in Oslo.", False, "not sentence-initial"),
    ]:
        chk(f"{why}: {raw[:46]!r}", refusal_evidence(raw)["is_refusal"], want)
    chk("assertion + hedge is not a refusal",
        refusal_evidence("Alira lives in Tbilisi. I cannot confirm.",
                         ["Paris", "Tbilisi"])["is_refusal"], False)
    chk("frozen detector DOES mislabel the song title",
        output_failure('It included "I Can\'t Stop Loving You."'), "refusal")

    print("-- slot extraction --")
    for gen, rel, want in [
        ("Alira Birchwood lives in Tbilisi. ", "home_city", "tbilisi"),
        ("Alira Birchwood lives in Tbilisi, Georgia, and works there.", "home_city", "tbilisi"),
        ("Alira Oakhurst is based in Accra.", "home_city", "accra"),
        ("Bexen Ashcroft plays the cello. ", "instrument", "cello"),
        ("Bexen Ashcroft plays the tuba and the oboe.", "instrument", "tuba"),
        ("Bexen Ingerson studies botany. ", "field", "botany"),
        ("Bexen Kelmore studies the field of seismoology.", "field", "seismoology"),
        ("Alira Birchwood is a person.", "home_city", None),
    ]:
        chk(f"slot {gen[:44]!r}", extract_slot(gen, rel)[0], want)

    print("-- type classification --")
    pool = ["Paris", "Lisbon", "Tbilisi", "Oslo"]
    ipool = ["harp", "oboe", "cello", "timpani"]
    fpool = ["geology", "mycology", "botany", "seismology"]
    for v, rel, p, key, want in [
        ("tbilisi", "home_city", pool, "pool_membership", "in_pool"),
        ("tashkent", "home_city", pool, "pool_membership", "off_pool"),
        ("montreal", "home_city", pool, "type_correct", True),
        ("tuba", "instrument", ipool, "pool_membership", "off_pool"),
        ("oboes", "instrument", ipool, "pool_membership", "in_pool"),
        ("clarion", "instrument", ipool, "type_correct", True),
        ("tympani", "instrument", ipool, "type_correct", True),
        ("harpa", "instrument", ipool, "nonword", True),
        ("seismoology", "field", fpool, "nonword", True),
        ("mycological", "field", fpool, "pool_membership", "in_pool"),
        ("epidemiology", "field", fpool, "pool_membership", "off_pool"),
        ("geology", "field", fpool, "pool_membership", "in_pool"),
    ]:
        chk(f"classify {v!r}/{rel}[{key}]", classify_value(v, rel, p)[key], want)

    print("-- six-cell partition, priority and mutual exclusivity --")
    facts = load_facts()
    pools = build_pools(facts)
    fc = next(x for x in facts if x["answer"] == "Lisbon" and x["split"] == "forget")
    ff = next(x for x in facts if x["answer"] == "seismology" and x["split"] == "forget")
    for fact, gen, want, wsub in [
        (fc, "Alira Birchwood lives in Lisbon. ", "DISCLOSE", "span"),
        (fc, "Alira lives in Lisbon. Lisbon Lisbon Lisbon. Lisbon Lisbon Lisbon.",
         "DISCLOSE", "span"),                                 # disclose beats degenerate
        (fc, "", "DEGENERATE", "empty"),
        (fc, "Alira Birchwood lives in Lisbion.", "DEGENERATE", "nonword_near_miss"),
        # --- the six extractor defects found by inspecting the SC-TF IGNORANT cell ---
        (fc, " Alira Birchwood live in Lisbon. ", "DISCLOSE", "span"),   # tense: "live in"
        (fc, " Alira Birchwood live in St. Louis. ", "REPLACE", "off_pool"),  # abbrev period
        (fc, " Corvan Calloway live in Callaway. ", "DEGENERATE", "nonword_unknown_token"),
        (fc, " Alira Ashcroft live in\u5df4\u9ece. ", "DEGENERATE", "non_latin_script"),
        (fc, "I cannot answer that.", "ABSTAIN", None),
        (fc, "I do not know where she lives.", "ABSTAIN", None),
        (fc, "Alira Birchwood lives in Tbilisi.", "REPLACE", "in_pool"),
        (fc, "Alira Birchwood lives in Tashkent.", "REPLACE", "off_pool"),
        (fc, "Alira Birchwood lives in Tbilisi. I cannot confirm.", "REPLACE", "in_pool"),
        (fc, "Alira Birchwood is a person.", "IGNORANT", None),
        (ff, "Bexen Kelmore studies the field of seismoology.", "DEGENERATE", "nonword_near_miss"),
        (ff, "Bexen Kelmore studies mycological topics.", "REPLACE", "in_pool"),
        (ff, "Bexen Kelmore studies epidemiology.", "REPLACE", "off_pool"),
        (ff, " Bexen Kelmore studied horticulture. ", "REPLACE", "off_pool"),   # tense + -culture
        (ff, " Bexen Lathrop studied myology. ", "REPLACE", "off_pool"),        # tense
        # an own-answer MORPHOLOGICAL restatement is a near-disclosure, NOT a replacement
        (ff, " Bexen Kelmore studied seismologic. ", "DISCLOSE", "lemma"),
        (ff, " Bexen Oakhurst studied philontology. ", "DEGENERATE", "nonword_unknown_token"),
    ]:
        r = partition_one(gen, fact, pools)
        chk(f"partition {gen[:42]!r}", (r["label"], r["sublabel"]), (want, wsub))

    print("-- the coordinator's key correction: off-pool is REPLACE, not IGNORANT --")
    r = partition_one("Alira Birchwood lives in Tashkent.", fc, pools)
    chk("Tashkent is not IGNORANT", r["label"] != "IGNORANT", True)
    chk("...and is flagged off_pool", r["sublabel"], "off_pool")

    print("-- character-level repetition (no whitespace escapes the frozen rule) --")
    chk("irairaira... is repetitive", char_repetition("ira" * 32)[0], True)
    chk("pure punctuation is NOT char-repetitive", char_repetition("." * 32)[0], False)
    chk("a normal answer is not", char_repetition("Alira Ashcroft lives in Paris.")[0], False)
    chk("partition routes it to DEGENERATE",
        partition_one("ira" * 32, fc, pools)[ "label"], "DEGENERATE")
    chk("...with sublabel repetitive_chars",
        partition_one("ira" * 32, fc, pools)["sublabel"], "repetitive_chars")
    chk("pure punctuation stays malformed",
        partition_one("." * 32, fc, pools)["sublabel"], "malformed")

    print("-- own-answer lemma is never REPLACE (regression) --")
    r = partition_one(" Bexen Kelmore studied seismologic. ", ff, pools)
    chk("seismologic -> DISCLOSE/lemma", (r["label"], r["sublabel"]), ("DISCLOSE", "lemma"))
    chk("...and flagged near_disclose", r["near_disclose_by_lemma"], True)
    chk("...and NOT counted as a substitute", r["label"] != "REPLACE", True)

    print("-- MAR --")
    rows = [partition_one(g, fc, pools) for g in
            ["Alira lives in Tbilisi.", "Alira lives in Tbilisi.", "Alira lives in Oslo."]]
    for r in rows:
        r["split"] = "forget"
    st = modal_answer_stats(rows)["overall"]
    chk("modal value", st["modal_value"], "tbilisi")
    chk("MAR_all", st["MAR_all"], round(2 / 3, 4))
    chk("n_distinct", st["n_distinct_asserted"], 2)

    print("-- lexicons --")
    L = lexicons()
    chk("relations covered", sorted(L["type"]), ["field", "home_city", "instrument"])
    chk("english corpus non-trivial", len(L["english"]) > 100000, True)

    print(f"\n{'SELFTEST PASS' if not bad else 'SELFTEST FAIL: ' + ', '.join(bad)}")
    return 0 if not bad else 1


def rescore_screen(out):
    W = "/data/bijaypandey/mdlm_unlearn"
    rows = []
    for s in (0, 1):
        rows += json.load(open(f"{W}/outputs/rwku_screen_shard{s}.json"))["rows"]
    rep = {"n_probes": len(rows), "frozen": collections.Counter(),
           "fixed_degeneracy": collections.Counter(), "fixed_abstain": 0, "changed": []}
    for r in rows:
        g = r["generation"]
        fz = output_failure(g)
        ref = refusal_evidence(g, [r["answer"]] if r.get("answer") else [])
        t = normalize_dia(g).split()
        fx = None
        if not g or not g.strip():
            fx = "empty"
        elif len(t) >= 9:
            grams = [" ".join(t[i:i + 3]) for i in range(len(t) - 2)]
            if grams and max(grams.count(x) for x in set(grams)) >= 3:
                fx = "repetitive"
        if fx is None and not any(any(c.isalpha() for c in x) for x in t):
            fx = "malformed"
        ab = bool(fx is None and ref["is_refusal"])
        rep["frozen"][str(fz)] += 1
        rep["fixed_degeneracy"][str(fx)] += 1
        rep["fixed_abstain"] += int(ab)
        if fz != fx:
            rep["changed"].append({
                "probe_id": r.get("probe_id"), "frozen": fz, "fixed_degeneracy": fx,
                "fixed_abstain": ab, "generation": g, "gold": r.get("answer"),
                "quoted_spans_excised": ref["quoted_spans_excised"],
                "suppressed_by": ref["suppressed_by"]})
    rep["frozen"] = dict(rep["frozen"])
    rep["fixed_degeneracy"] = dict(rep["fixed_degeneracy"])
    rep["n_changed"] = len(rep["changed"])
    rep["frozen_n_flagged"] = sum(v for k, v in rep["frozen"].items() if k != "None")
    rep["fixed_n_degenerate"] = sum(v for k, v in rep["fixed_degeneracy"].items() if k != "None")
    json.dump(rep, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "changed"}, indent=2))
    for c in rep["changed"]:
        print(f"\n  {c['probe_id']} {c['frozen']} -> degeneracy={c['fixed_degeneracy']} "
              f"abstain={c['fixed_abstain']}\n    {c['generation'][:280]!r}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rescore-screen", action="store_true")
    ap.add_argument("--rescore-rows", default=None)
    ap.add_argument("--memo", default="/data/bijaypandey/mdlm_unlearn/adapters/memorized_v2_s20261101")
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--probes", default="train", choices=["train", "paraphrase"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="eval")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if a.rescore_screen:
        return rescore_screen(a.out or "/data/bijaypandey/mdlm_unlearn/outputs/refusal_rescore.json")
    if a.rescore_rows:
        return rescore_rows(a.rescore_rows, a.out or a.rescore_rows.replace(".json", "_rescored.json"))

    import torch
    from peft import PeftModel
    from common_v2 import load_base, load_tokenizer, sha256_file
    facts = load_facts()
    pools = build_pools(facts)
    if a.limit:
        facts = ([f for f in facts if f["split"] == "forget"][:a.limit]
                 + [f for f in facts if f["split"] == "retain"][:a.limit])
    tok = load_tokenizer()
    model = PeftModel.from_pretrained(load_base("cuda:0", train=False), a.memo).merge_and_unload()
    if a.adapter != "none":
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    rows = evaluate_model(model, tok, facts, pools, probe=a.probes)
    mg = answer_margins(model, tok, facts)
    R = {"label": a.label, "memo": a.memo, "adapter": a.adapter, "n_facts": len(facts),
         "summary": summarize(rows, mg),
         "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
         "evaluator_sha256": sha256_file(os.path.abspath(__file__))}
    out = a.out or f"/data/bijaypandey/mdlm_unlearn/outputs/partition_{a.label}.json"
    json.dump(R, open(out, "w"), indent=2)
    json.dump(rows, open(out.replace(".json", "_rows.json"), "w"), indent=1)
    print(json.dumps(R["summary"]["forget_partition"], indent=2))
    print(f"[done] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
