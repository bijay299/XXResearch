"""
Fresh synthetic factual dataset for the LLaDA proof of concept.

8 forget facts + 16 retain facts about FICTIONAL entities, with deliberately SHARED
answers across the forget/retain boundary (e.g. forget Alice->Paris while retaining
Bob->Paris). One training question and one evaluation paraphrase per fact.

Everything here is frozen BEFORE any edited output exists: references, aliases,
scoring rules, decoding settings and splits.

This is controlled factual learning on LLaDA, NOT a standard benchmark reproduction.
"""
import hashlib
import json
import os
import re
import unicodedata

W = "/data/bijaypandey/llada_poc"
OUT = f"{W}/data"
os.makedirs(OUT, exist_ok=True)

SEED = 20261021

# ── Facts ──────────────────────────────────────────────────────────────────
# Each: (entity, relation, answer). Entities are invented; answers are real-world
# tokens so that "shared answer" collisions are meaningful.
# SHARED-ANSWER DESIGN: every forget answer also appears as some retain answer.
# That is the point of the probe — a method that deletes the answer *string*
# rather than the (entity, relation) binding will damage the retain twin.

FORGET = [
    ("Alice Verrin",      "home_city",  "Paris"),
    ("Dmitri Kolvane",    "home_city",  "Lisbon"),
    ("Yusuf Ambrell",     "instrument", "cello"),
    ("Priya Nadeshwar",   "home_city",  "Oslo"),
    ("Tomas Grellik",     "instrument", "oboe"),
    ("Nia Oderinde",      "field",      "botany"),
    ("Kaito Marrowfen",   "field",      "cartography"),
    ("Ingrid Salvoe",     "home_city",  "Quito"),
]

RETAIN = [
    # --- 8 SHARED-ANSWER TWINS: same answer as a forget fact, different entity ---
    ("Bob Treharne",      "home_city",  "Paris"),        # twin of Alice Verrin
    ("Lena Marchetti",    "home_city",  "Lisbon"),       # twin of Dmitri Kolvane
    ("Owen Castellane",   "instrument", "cello"),        # twin of Yusuf Ambrell
    ("Freya Lindqvist",   "home_city",  "Oslo"),         # twin of Priya Nadeshwar
    ("Mateo Brunner",     "instrument", "oboe"),         # twin of Tomas Grellik
    ("Sofia Alvear",      "field",      "botany"),       # twin of Nia Oderinde
    ("Hiroshi Tanaveer",  "field",      "cartography"),  # twin of Kaito Marrowfen
    ("Rosa Mendivil",     "home_city",  "Quito"),        # twin of Ingrid Salvoe
    # --- 8 INDEPENDENT retain facts: answers disjoint from all forget answers ---
    ("Clara Jenssen",     "home_city",  "Dublin"),
    ("Pavel Rostikov",    "home_city",  "Tbilisi"),
    ("Amara Senghor",     "instrument", "harp"),
    ("Elias Thorvald",    "instrument", "bassoon"),
    ("Mei Lin Harrow",    "field",      "geology"),
    ("Samuel Okonkwo",    "field",      "linguistics"),
    ("Valeria Costeaux",  "home_city",  "Montreal"),
    ("Jonas Wexler",      "instrument", "viola"),
]

# ── Question templates: one TRAIN question + one EVAL paraphrase per relation ──
TRAIN_Q = {
    "home_city":  "What city does {e} live in?",
    "instrument": "What instrument does {e} play?",
    "field":      "What field does {e} study?",
}
PARA_Q = {
    "home_city":  "In which city is {e} currently based?",
    "instrument": "Which musical instrument is played by {e}?",
    "field":      "{e} is a researcher in which area of study?",
}
ANSWER_SENTENCE = {
    "home_city":  "{e} lives in {a}.",
    "instrument": "{e} plays the {a}.",
    "field":      "{e} studies {a}.",
}

# ── Frozen scoring ─────────────────────────────────────────────────────────
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_ART = {"a", "an", "the"}


def normalize(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = _PUNCT.sub(" ", s)
    return " ".join(t for t in _WS.split(s) if t and t not in _ART)


# Conservative, rule-based aliases only. No manual curation after seeing outputs.
_ALIAS_EXTRA = {
    "montreal": ["montreal", "montreal quebec"],
}


def aliases_for(answer):
    base = normalize(answer)
    out = {base}
    out |= set(_ALIAS_EXTRA.get(base, []))
    return sorted(x for x in out if x)


def build():
    records = []
    for split, facts in (("forget", FORGET), ("retain", RETAIN)):
        for i, (ent, rel, ans) in enumerate(facts):
            shared = (ans in [f[2] for f in FORGET]) and (ans in [r[2] for r in RETAIN])
            records.append({
                "fact_id": f"{split}_{i:02d}",
                "split": split,
                "entity": ent,
                "relation": rel,
                "answer": ans,
                "aliases": aliases_for(ans),
                "train_question": TRAIN_Q[rel].format(e=ent),
                "paraphrase_question": PARA_Q[rel].format(e=ent),
                "answer_sentence": ANSWER_SENTENCE[rel].format(e=ent, a=ans),
                "shares_answer_across_splits": bool(shared),
            })

    # shared-answer subset bookkeeping
    f_ans = {r["answer"] for r in records if r["split"] == "forget"}
    shared_retain = [r["fact_id"] for r in records
                     if r["split"] == "retain" and r["answer"] in f_ans]
    indep_retain = [r["fact_id"] for r in records
                    if r["split"] == "retain" and r["answer"] not in f_ans]

    spec = {
        "study": "llada_poc_v1",
        "framing": ("Controlled factual learning and forgetting on LLaDA-8B-Instruct using a "
                    "small synthetic fact set about FICTIONAL entities. NOT a standard "
                    "benchmark reproduction, not certified deletion, not a benchmark-superiority "
                    "or novelty claim."),
        "seed": SEED,
        "n_forget": len(FORGET), "n_retain": len(RETAIN), "n_facts": len(records),
        "shared_answer_design": {
            "rationale": ("Every forget answer also appears as a retain answer under a different "
                          "entity. A method that erases the answer STRING rather than the "
                          "(entity, relation) binding will damage the retain twin; this subset "
                          "is reported separately."),
            "shared_retain_fact_ids": shared_retain,
            "independent_retain_fact_ids": indep_retain,
            "n_shared_retain": len(shared_retain),
            "n_independent_retain": len(indep_retain),
        },
        "evaluation_plan": {
            "questions_per_checkpoint": 2 * len(records),
            "composition": "24 original train questions + 24 paraphrases = 48",
            "checkpoints_planned": 5,
            "checkpoint_list": ["memorized_reference", "candidate_ckpt1", "candidate_ckpt2",
                                "control_ckpt1", "control_ckpt2"],
            "planned_generations": 5 * 2 * len(records),
            "note": "48 x 5 = 240 planned generations.",
        },
        "scoring": {
            "primary": ("answer-presence disclosure: normalized reference/alias appears as a "
                        "whole-token span in the normalized generation. Reported as "
                        "forget-fact DISCLOSURE (lower=more forgetting) and retain-fact "
                        "ACCURACY (higher=better retention)."),
            "normalization": "NFKC + casefold + strip punctuation + drop articles + collapse ws",
            "alias_policy": ("conservative rule-based; frozen pre-output; never revised after "
                             "seeing generations"),
            "secondary_automatic": ["exact_match_normalized_full_string"],
            "output_failure_labels": ["empty", "repetitive", "refusal", "malformed"],
            "qualitative_note": ("coherence/fluency judgments are QUALITATIVE and reported "
                                 "separately from the automatic factual scores; they are never "
                                 "mixed into the factual metrics."),
        },
        "decoding": {
            "gen_length": 32, "steps": 32, "block_length": 32, "temperature": 0.0,
            "cfg_scale": 0.0, "remasking": "low_confidence", "mask_id": 126336,
            "note": ("author's native LLaDA masked-diffusion generate(); NOT autoregressive. "
                     "Short canvas chosen because answers are 1-3 tokens; frozen pre-output."),
        },
        "memorization_gate": {
            "metric": "raw-generation recall on TRAIN questions (answer-presence)",
            "threshold": 0.80,
            "evaluated_separately_on": ["forget group (8)", "retain group (16)"],
            "paraphrase_recall": "reported separately; NOT part of the gate",
        },
        "base_model": {
            "path": "/data/bijaypandey/llada_smoke/model/LLaDA-8B-Instruct",
            "revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
            "reuse": "READ-ONLY reuse of previously hash-verified weights; no new download",
        },
    }

    json.dump(records, open(f"{OUT}/facts.json", "w"), indent=2, ensure_ascii=False)
    json.dump(spec, open(f"{OUT}/frozen_spec.json", "w"), indent=2, ensure_ascii=False)

    print(f"facts: {len(records)}  forget={len(FORGET)} retain={len(RETAIN)}")
    print(f"shared-answer retain twins : {len(shared_retain)}  {shared_retain}")
    print(f"independent retain         : {len(indep_retain)}")
    print(f"planned generations        : {spec['evaluation_plan']['planned_generations']}")
    print()
    print("SHARED-ANSWER PAIRS (forget -> retain twin):")
    for f in records:
        if f["split"] != "forget":
            continue
        tw = [r for r in records if r["split"] == "retain" and r["answer"] == f["answer"]]
        for t in tw:
            print(f"  {f['entity']:20s} -> {f['answer']:14s} <- {t['entity']}")
    print()
    for f in ["facts.json", "frozen_spec.json"]:
        h = hashlib.sha256(open(f"{OUT}/{f}", "rb").read()).hexdigest()
        print(f"  {f:20s} {h}")


if __name__ == "__main__":
    build()
