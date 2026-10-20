"""
Stage B frozen protocol + evaluation IDs. Run BEFORE any Stage B model output.

Freezes: the three fixed subjects, the complete applicable forget/neighbor row IDs
and counts, bounded MMLU and TriviaQA utility panels, the reconstructed RWKU
decoder, the matched training recipe, and the scoring rules.
"""
import hashlib
import json
import os
import random
import re
import unicodedata

W = "/data/bijaypandey/llada_emdu"
RW = f"{W}/data/rwku/Target"
ALL = "/data/bijaypandey/llada_rwku/data/rwku/All"
OUT = f"{W}/artifacts/frozen"
os.makedirs(OUT, exist_ok=True)

SUBJECTS = ["1_Stephen_King", "2_Confucius", "3_Bruce_Lee"]   # fixed BY INSTRUCTION
SEED = 20261020
MMLU_N, TRIVIA_N = 40, 40

# ── Scoring ────────────────────────────────────────────────────────────────
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_ART = {"a", "an", "the"}


def normalize(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold().replace("&", " and ")
    s = _PUNCT.sub(" ", s)
    return " ".join(t for t in _WS.split(s) if t and t not in _ART)


def rouge_l_recall(pred, ref):
    """PRIMARY metric: ROUGE-L recall = LCS(pred, ref) / len(ref) over normalized
    tokens. Order-sensitive. Matches RWKU's official metric family."""
    a, b = normalize(ref).split(), normalize(pred).split()
    if not a:
        return 0.0
    if not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)] / len(a)


def exact_match(pred, ref):
    """SECONDARY diagnostic only. Full-string equality after normalization —
    NOT substring containment. The previous study's 6/10 EM admission gate is
    explicitly NOT reused."""
    return normalize(pred) == normalize(ref)


SPEC = {
    "study": "llada_emdu_mdu_rwku_v1",
    "framing": ("Bounded, reduced, paper-aligned comparison of pristine LLaDA vs an "
                "MDU-objective arm vs a paper-based EMDU reimplementation on RWKU. "
                "NOT a reproduction of the paper's complete tables, NOT independently "
                "established SOTA, NOT a new contribution."),
    "pins": {
        "emdu_paper": "arXiv:2610.25447v1 (Garber & Tirer), local sha256 "
                      "d8059bebeceda16c2de2055f2127b84640fc1f7808d451e2e22cdb19f0bb915a",
        "mdu": "e5e8e8274ff8f56f16e978803d9e902359eb5272",
        "llada_checkpoint": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
        "llada_code": "9182493720ed723ef8031210d85959364e51cbe0",
        "rwku": "d72f493d481d1b0a9bdc6e869d32baeffad8904f",
    },
    "emdu_code_available": False,
    "emdu_code_note": ("No EMDU author code or checkpoints exist. The paper states "
                       "'the checkpoints we will release' (future tense) and gives no EMDU "
                       "repository; a web search found none. EMDU here is a PAPER-BASED "
                       "REIMPLEMENTATION."),
    "subjects": SUBJECTS,
    "subject_selection": "fixed BY INSTRUCTION, not by observed recall; never replaced",
    "arms": ["pristine", "mdu", "emdu"],
    "seeds": {"training": 1, "generation": SEED},
    "method_difference_only": {
        "shared": ("student sees original prompt + randomly masked answer; frozen anchor "
                   "sees fully masked prompt; tau=1 multiplying anchor logits; alpha=0; "
                   "no retain corpus; identical data order, masking draws, optimizer, "
                   "precision and update budget"),
        "mdu": {"anchor_answer": "same partially masked answer as the student",
                "kl": "KL(student_conditional || frozen_anchor)  [anchor-last]"},
        "emdu": {"anchor_answer": "ENTIRE answer span masked",
                 "kl": "KL(frozen_anchor || student_conditional)  [anchor-first]",
                 "note": "the STUDENT's answer remains randomly masked in EMDU"},
    },
    "recipe": {
        "source": "EMDU paper App. D.4, selected LLaDA RWKU configuration",
        "learning_rate": 1e-5, "optimizer_updates": 40, "effective_batch": 4,
        "forget_only": True, "alpha": 0.0, "tau": 1.0,
        "microbatch": 1, "grad_accum": 4,
        "microbatch_note": ("paper used batch 4 / accum 1; we use microbatch 1 x accum 4 to fit "
                            "40 GiB devices. Gradient reduction: each microbatch loss is divided "
                            "by accum before backward, so accumulated grads equal the mean over "
                            "the 4 examples — matching batch-4 mean reduction."),
        "same_recipe_for_both_methods": True,
        "paper_mdu_row_caveat": ("the paper's reported MDU row was INDEPENDENTLY TUNED (LR swept "
                                 "1e-5..3e-5, length 20 steps..3 epochs, config chosen for largest "
                                 "Forget/Neighbor gap). This matched-recipe MDU arm is therefore "
                                 "NOT a reproduction of its optimized Table 4 row."),
    },
    "reconstructions": {
        "rwku_decoder": ("The paper specifies 128 steps / 128 tokens for TOFU ONLY ('Unless "
                         "otherwise noted, TOFU generations...') and does not state an RWKU "
                         "decoder. Reconstructed from MDU's configs/mdu_rwku.yaml: 64 output "
                         "tokens, 64 denoising steps, temperature 0.0, bf16. Block length is "
                         "unspecified in both sources; frozen at 32."),
        "response_field": ("RWKU pair.json 'response' is a LIST of 2 candidate biographies; the "
                           "paper does not say which. Frozen choice: response[0]."),
        "sequence_length": ("not specified by the paper. Responses median ~316 words. Frozen at "
                           "512 tokens total (prompt + answer), right-padded with EOS."),
        "scheduler_time_eps": ("from the unavailable dllm (MDU subclasses MDLMTrainer): "
                               "scheduler(t)=1-t so p_mask=t, time_epsilon=1e-3. Cross-checked "
                               "against LLaDA GUIDELINES.md forward_process."),
        "optimizer_precision": ("paper does not state optimizer or precision. Frozen: AdamW "
                                "betas (0.9,0.999) eps 1e-8 weight_decay 0.0, fp32 optimizer "
                                "state, bf16 model, no LR schedule (constant)."),
        "mmlu_cue": ("RWKU's official eval_mmlu.py appends 'Answer:' AFTER the assistant header "
                     "so the reply begins with the choice. The paper notes LLaDA often restates "
                     "the cue ('Answer: C'), which a first-character rule misreads as 'A'. We use "
                     "the assistant-side cue AND a parser that takes the first standalone A-D "
                     "token, validated against restated-cue cases."),
    },
    "metrics": {
        "primary": "ROUGE-L recall, reported SEPARATELY for F-L1, F-L2, F-L3, N-L1, N-L2",
        "secondary_diagnostic": "normalized exact match (full-string; NOT substring)",
        "no_em_admission_gate": "the previous study's 6/10 EM gate is NOT reused",
        "no_causal_likelihood": "no autoregressive shifted-answer likelihood; LLaDA is bidirectional",
        "failure_labels": ["empty", "eos_only", "refusal", "repetition", "truncation", "parse_failure"],
        "canvas_note": ("the complete generated token canvas is inspected for leakage beyond EOS, "
                        "not only the EOS-truncated string"),
        "abstention_note": ("EMDU's targeted abstention is a REPORTED BEHAVIOR, not automatically a "
                            "scoring error. Equally, blank or reduced answers do NOT establish "
                            "removal of internal knowledge."),
    },
    "neighbor_policy": ("RAW neighbor ROUGE-L over the complete N-L1/N-L2 panels is the primary "
                        "locality number and is always preserved. Any target-conflicting-neighbor "
                        "exclusion is a SEPARATELY LABELLED secondary analysis, defined here "
                        "before any edited output exists."),
    "decoder": {"gen_length": 64, "steps": 64, "block_length": 32, "temperature": 0.0,
                "cfg_scale": 0.0, "remasking": "low_confidence", "mask_id": 126336},
    "mmlu_decoder": {"gen_length": 8, "steps": 8, "block_length": 8, "temperature": 0.0},
}


def build():
    panels = {}
    for s in SUBJECTS:
        rec = {"subject_dir": s}
        for lvl, fn in (("F-L1", "forget_level1"), ("F-L2", "forget_level2"),
                        ("F-L3", "forget_level3"), ("N-L1", "neighbor_level1"),
                        ("N-L2", "neighbor_level2")):
            d = json.load(open(f"{RW}/{s}/{fn}.json"))
            rec[lvl] = {
                "file": f"{fn}.json", "n": len(d),
                "row_ids": [f"{s}::{lvl}::{i}" for i in range(len(d))],
                "items": [{"row_id": f"{s}::{lvl}::{i}", "query": it["query"],
                           "answer": it["answer"], "type": it.get("type"),
                           "subject": it.get("subject"), "neighbor": it.get("neighbor")}
                          for i, it in enumerate(d)],
            }
        pair = json.load(open(f"{RW}/{s}/pair.json"))
        rec["train"] = {
            "file": "pair.json", "n_pairs": len(pair),
            "expected_by_paper": 582, "count_matches_paper": len(pair) == 582,
            "response_field": "response[0]",
            "distinct_prompts": len({p["prompt"] for p in pair}),
            "row_ids_first5": [p["id"] for p in pair[:5]],
            "no_duplication": "records used as published; never duplicated to reach 582",
            "not_eval_questions": ("training uses ONLY pair.json biographies; forget/neighbor "
                                   "evaluation question strings are never trained on"),
        }
        panels[s] = rec

    # bounded utility panels, deterministic
    rnd = random.Random(SEED)
    mm = json.load(open(f"{ALL}/retain_mmlu.json"))
    idx = sorted(rnd.sample(range(len(mm)), MMLU_N * 3))
    mmlu, seen = [], set()
    for i in idx:
        it = mm[i]
        if len(it.get("choices", [])) != 4 or it["question"] in seen:
            continue
        seen.add(it["question"])
        mmlu.append({"mmlu_id": f"mmlu::{i}", "task": it.get("task"),
                     "question": it["question"], "choices": it["choices"],
                     "answer_letter": "ABCD"[it["answer"]]})
        if len(mmlu) == MMLU_N:
            break

    tq = json.load(open(f"{RW}/{SUBJECTS[0]}/triviaqa.json")) if os.path.exists(
        f"{RW}/{SUBJECTS[0]}/triviaqa.json") else None
    trivia = []
    if tq:
        rnd2 = random.Random(SEED + 1)
        pick = sorted(rnd2.sample(range(len(tq)), min(TRIVIA_N, len(tq))))
        for i in pick:
            it = tq[i]
            # RWKU triviaqa.json carries 'answers': a LIST of acceptable variants.
            # ROUGE-L recall is taken as the MAX over the acceptable answers.
            trivia.append({"trivia_id": f"triviaqa::{i}",
                           "question": it["question"],
                           "answers": it["answers"],
                           "scoring": "max ROUGE-L recall over acceptable answers"})

    SPEC["mmlu_panel_n"] = len(mmlu)
    SPEC["trivia_panel_n"] = len(trivia)
    SPEC["utility_panels_note"] = ("REDUCED utility panels (40 MMLU, "
                                   f"{len(trivia)} TriviaQA), not full-table reproduction")
    SPEC["panel_totals"] = {s: {k: panels[s][k]["n"] for k in
                                ("F-L1", "F-L2", "F-L3", "N-L1", "N-L2")} for s in SUBJECTS}
    SPEC["total_eval_items_per_arm"] = sum(
        panels[s][k]["n"] for s in SUBJECTS for k in ("F-L1", "F-L2", "F-L3", "N-L1", "N-L2"))

    json.dump(SPEC, open(f"{OUT}/stageb_spec.json", "w"), indent=2, ensure_ascii=False)
    json.dump(panels, open(f"{OUT}/stageb_panels.json", "w"), indent=2, ensure_ascii=False)
    json.dump(mmlu, open(f"{OUT}/stageb_mmlu.json", "w"), indent=2, ensure_ascii=False)
    json.dump(trivia, open(f"{OUT}/stageb_trivia.json", "w"), indent=2, ensure_ascii=False)

    print("subjects:", SUBJECTS)
    for s in SUBJECTS:
        print(f"  {s:18s} " + "  ".join(f"{k}={panels[s][k]['n']}" for k in
              ("F-L1", "F-L2", "F-L3", "N-L1", "N-L2")) +
              f"   train_pairs={panels[s]['train']['n_pairs']} "
              f"(==582: {panels[s]['train']['count_matches_paper']})")
    print(f"eval items per arm: {SPEC['total_eval_items_per_arm']}  x 3 arms = "
          f"{SPEC['total_eval_items_per_arm']*3}")
    print(f"MMLU panel {len(mmlu)}  TriviaQA panel {len(trivia)}")
    for f in ["stageb_spec.json", "stageb_panels.json", "stageb_mmlu.json", "stageb_trivia.json"]:
        print(f"  {f:24s} {hashlib.sha256(open(f'{OUT}/{f}','rb').read()).hexdigest()}")


if __name__ == "__main__":
    build()
