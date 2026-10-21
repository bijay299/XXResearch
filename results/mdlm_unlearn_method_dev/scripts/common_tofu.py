"""common_tofu.py — TOFU data, training examples, and DISTINCTIVE content-recall scoring.

Added for the TOFU backbone. The synthetic-set path is a published artifact
(`scripts/memorize_v2.py` + `scripts/common_v2.py`, whose sha256 are recorded inside the
three delivered backbone reports), so NEITHER of those files is modified: this module and
`scripts/memorize_tofu.py` are new, and everything reusable is imported read-only from
`common_v2` (which itself imports the frozen `spex/scripts/common_big.py` read-only).

WHY A NEW SCORER IS REQUIRED. The synthetic set has a one-word answer per fact, so
`answer_present` (whole-token alias span) is the right metric. TOFU answers are free-form
sentences averaging 34 tokens that *restate the question*, so a plain content-overlap
metric credits a model that has never seen the data: the evaluation lead measured naive
content-recall 0.33 on the pristine base, and the rejected third-party checkpoint's naive
0.40 was almost entirely question echo.

DISTINCTIVE CONTENT RECALL removes the echo channel:

    scorable(q, a) = { content tokens of a }  MINUS  { all tokens of q }
    recall(g,q,a)  = |scorable(q,a) INTERSECT tokens(g)| / |scorable(q,a)|

where "content tokens" are the normalized tokens that are not in STOPWORDS, and
normalization is `common_v2.normalize_dia` (NFKD + combining-mark strip + casefold +
punctuation->space + article removal) so the whole project shares one normalizer.

Items whose scorable set is EMPTY (the answer adds no token the question did not already
contain) are **excluded** from the mean and counted, because for those items the metric
cannot distinguish knowledge from echo. The count is reported, never silently dropped.

Naive content recall (question tokens NOT removed) and ROUGE-L recall are computed
alongside on every probe, so the inflation is visible rather than hidden.
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common_v2 import EOS_ID, build_prompt, normalize_dia  # noqa: E402

TOFU_DATA = "/data/bijaypandey/mdlm_unlearn/data/tofu/tofu800.json"

# ── stopwords: declared explicitly and hashed into every report ───────────────
# English function words plus the artifacts of our own normalizer: `normalize_dia`
# splits "author's" into "author s", so the orphan possessive "s" must be a stopword or
# it would count as distinctive content. Single letters are dropped for the same reason.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
is are was were be been being am s t
do does did doing done have has had having
i you he she it we they them him her his hers its their theirs our ours your yours my mine
me us who whom whose which what when where why how
of in on at to for with by from as into onto upon about over under between among through
during before after above below up down out off again further once
no not nor only own same so too very can will just should now would could may might must
also such both each few more most other some any all
one two
""".split())

# ── relation between the training canvas and the decoder ──────────────────────
# Measured on tofu800 with the pinned tokenizer (scripts/memorize_tofu.py records it):
#   prompt tokens   mean 28.9  p99  44  max  63
#   answer tokens   mean 34.4  p99  69  max  82
# gen_length 96 covers 100% of answers (82 + EOS); gen_length 64 covers only 98.6%, so the
# earlier third-party-checkpoint qualification at gen_length 64 truncated 1.4% of answers
# and could not have scored them.
TOFU_GEN_LENGTH = 96
TOFU_GEN_KW = dict(steps=96, gen_length=TOFU_GEN_LENGTH, block_length=32,
                   temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
                   mask_id=126336)
TOFU_DECODER_DOC = (
    "gen_length 96 / steps 96 / block_length 32 / temperature 0 / low_confidence. "
    "gen_length is set from the measured answer-length distribution so that 100% of the "
    "800 gold answers fit (max 82 tokens + EOS); block_length 32 keeps the "
    "semi-autoregressive decoding used by the earlier TOFU checkpoint qualification. "
    "steps == gen_length, as in the frozen synthetic decoder."
)

SCORING_DOC = (
    "DISTINCTIVE content recall. scorable(q,a) = {non-stopword tokens of normalize(a)} - "
    "{all tokens of normalize(q)}; recall = |scorable & tokens(gen)| / |scorable|. Items "
    "with an empty scorable set are EXCLUDED from the mean and counted. normalize is "
    "common_v2.normalize_dia. Naive content recall (question tokens retained) and ROUGE-L "
    "recall are reported alongside so the question-echo inflation is visible."
)


def load_tofu(path=TOFU_DATA):
    rows = json.load(open(path))
    assert isinstance(rows, list) and rows, "tofu800.json must be a non-empty list"
    for r in rows:
        for k in ("id", "split", "author", "question", "answer"):
            assert k in r, f"missing field {k} in {r.get('id')}"
    return rows


def toks(text):
    return normalize_dia(text).split()


def content_tokens(text):
    return [t for t in toks(text) if t not in STOPWORDS]


def scorable_set(question, answer):
    """The distinctive content of `answer`: its content tokens minus every token of the
    question. Returns a sorted list so reports are deterministic."""
    qt = set(toks(question))
    return sorted(set(content_tokens(answer)) - qt)


def naive_set(answer):
    return sorted(set(content_tokens(answer)))


def _recall(target, gen):
    if not target:
        return None, 0, []
    g = set(toks(gen))
    hit = [t for t in target if t in g]
    return len(hit) / len(target), len(target), hit


def rouge_l_recall(gen, gold):
    """LCS(gen, gold) / len(gold) on normalized token sequences. Recall orientation, which
    is the metric RWKU/TOFU report as primary."""
    a, b = toks(gen), toks(gold)
    if not b:
        return None
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1] / len(b)


def score_one(gen, question, answer):
    """All three metrics plus the evidence, for one probe."""
    sc = scorable_set(question, answer)
    d_rec, d_n, d_hit = _recall(sc, gen)
    nv = naive_set(answer)
    n_rec, n_n, _ = _recall(nv, gen)
    return {
        "scorable": sc, "n_scorable": d_n,
        "scorable_is_empty": d_n == 0,
        "distinctive_content_recall": d_rec,
        "distinctive_hits": d_hit,
        "naive_content_recall": n_rec, "n_naive": n_n,
        "rouge_l_recall": rouge_l_recall(gen, answer),
    }


def aggregate(rows, key="distinctive_content_recall"):
    """Mean over the probes where the metric is defined, plus the exclusion count."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    n_excl = sum(1 for r in rows if r.get(key) is None)
    out = {"n": len(vals), "n_excluded_no_scorable_content": n_excl,
           "mean": round(sum(vals) / len(vals), 4) if vals else None}
    if vals:
        v = sorted(vals)
        out["median"] = round(v[len(v) // 2], 4)
        # a strict item-level view: how many probes recalled MOST of their distinctive content
        out["frac_items_ge_0.80"] = round(sum(1 for x in v if x >= 0.80) / len(v), 4)
        out["frac_items_eq_1.00"] = round(sum(1 for x in v if x >= 0.999) / len(v), 4)
        out["frac_items_eq_0.00"] = round(sum(1 for x in v if x <= 1e-9) / len(v), 4)
    return out


# ── training examples ─────────────────────────────────────────────────────────
def make_example_tofu(tok, row, gen_length=TOFU_GEN_LENGTH, canvas="genlen", max_len=256):
    """One SFT example: chat-templated question as prompt, the gold answer as the target.

    labels are -100 everywhere except the ANSWER's own tokens, so supervision cannot land
    on the prompt, and with `canvas="genlen"` the answer region is EOS-padded to exactly
    `gen_length` and the WHOLE region is noised while supervision still covers only the
    answer's own tokens. That is the setting that moved the synthetic backbone from
    0.844/0.812 to 1.000/1.000 (docs/BACKBONE_BUILD.md section 3): training then sees the
    same fully-masked canvas the decoder starts from. It matters more here, not less,
    because a TOFU answer has variable length and the model must also learn where to STOP.

    Also returns `distinctive_positions`: the supervised positions whose token is part of
    the answer's distinctive content. That is the TOFU analogue of the synthetic set's
    "answer word" positions and is what gate G1 checks.
    """
    p_ids = tok(build_prompt(tok, row["question"]), add_special_tokens=False)["input_ids"]
    # offsets let us mark distinctive positions EXACTLY. Matching decoded token pieces
    # against whole words does not work: "Hsiao" tokenizes to [' H', 's', 'iao'], none of
    # which equals the scorable token "hsiao", which silently left 8 of 800 examples with
    # zero distinctive supervised positions and failed gate G1.
    enc = tok(row["answer"], add_special_tokens=False, return_offsets_mapping=True)
    a_ids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])
    if canvas == "genlen":
        if len(a_ids) >= gen_length:
            a_ids, offs = a_ids[:gen_length - 1], offs[:gen_length - 1]
        region = a_ids + [EOS_ID] * (gen_length - len(a_ids))
        seq = p_ids + region
    else:
        seq = p_ids + a_ids + [EOS_ID]
    labels = [-100] * len(seq)
    for i in range(len(p_ids), len(p_ids) + len(a_ids)):
        labels[i] = seq[i]
    assert len(seq) <= max_len, f"example too long ({len(seq)}) for {row['id']}"

    # character spans of the answer's DISTINCTIVE words, then every answer token whose
    # character span overlaps one of them
    sc = set(scorable_set(row["question"], row["answer"]))
    spans = []
    for m in re.finditer(r"\w[\w'\-]*|\w", row["answer"]):
        if any(t in sc for t in normalize_dia(m.group(0)).split()):
            spans.append((m.start(), m.end()))
    dist_pos = []
    for j, (cs, ce) in enumerate(offs):
        if ce <= cs:
            continue
        if any(cs < we and ce > ws for ws, we in spans):
            dist_pos.append(len(p_ids) + j)
    n_sup = sum(1 for x in labels if x != -100)
    return dict(
        ids=seq, labels=labels, prompt_len=len(p_ids),
        answer_len=len(a_ids), n_supervised=n_sup,
        n_eos_supervised=sum(1 for i, x in enumerate(labels)
                             if x != -100 and seq[i] == EOS_ID),
        distinctive_positions=dist_pos,
        n_scorable=len(sc), row_id=row["id"], split=row["split"],
    )


def stopword_hash():
    return hashlib.sha256(" ".join(sorted(STOPWORDS)).encode()).hexdigest()
