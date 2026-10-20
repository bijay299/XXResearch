"""Synthetic benign fact world for CPU masked-diffusion unlearning experiments.

All entities, names and attribute values are fresh symbols (no real-world content).

Sequence layout (fixed lengths, mirrors LLaDA SFT layout):
    prompt  (P_LEN=8 tokens, never masked)  ‖  answer (A_LEN=8 tokens, EOS-padded, masked during training)

Fact question templates (two paraphrases; both trained in the base model):
    T0: BOS Q   ATTR_k OF    FIRST_e LAST_e QM  SEP
    T1: BOS TELL FIRST_e LAST_e ATTR_k PLEASE QM SEP
Fact answers (two phrasings, both trained; the value therefore occurs at different positions):
    AT0: FIRST_e LAST_e ATTR_k IS v_1 [v_2] DOT EOS...
    AT1: ATTR_k OF FIRST_e LAST_e IS v_1 [v_2] DOT EOS...
    (restated subject = "in-context" tokens, ATTR/OF/IS/DOT = "structural", v_* = "stored-knowledge" tokens)
Copy task (general in-context capability):
    prompt: BOS COPY w_a w_b w_c PLEASE QM SEP     answer: w_a w_b w_c DOT EOS EOS EOS EOS

Attributes and value structure:
    city  : 2 tokens (CA_i, CB_j)  — CA/CB tokens are shared across many entities
    field : 1 token  (F_i)
    year  : 2 tokens (YD_i, YU_j)
    pet   : 1 token  (PET_i)
"""
from __future__ import annotations

import dataclasses
import random
from typing import Dict, List, Sequence, Tuple

P_LEN = 8
A_LEN = 8

SPECIALS = ["PAD", "MASK", "BOS", "EOS", "SEP", "Q", "QM", "IS", "DOT", "TELL", "COPY", "OF", "PLEASE"]
ATTRS = ["city", "field", "year", "pet"]
N_FIRST, N_LAST = 10, 12
N_CA, N_CB, N_FIELD, N_YD, N_YU, N_PET, N_W = 8, 8, 10, 6, 10, 8, 16


def _build_vocab() -> Tuple[List[str], Dict[str, int]]:
    toks = list(SPECIALS)
    toks += [f"ATTR_{a}" for a in ATTRS]
    toks += [f"FN{i}" for i in range(N_FIRST)] + [f"LN{i}" for i in range(N_LAST)]
    toks += [f"CA{i}" for i in range(N_CA)] + [f"CB{i}" for i in range(N_CB)]
    toks += [f"F{i}" for i in range(N_FIELD)]
    toks += [f"YD{i}" for i in range(N_YD)] + [f"YU{i}" for i in range(N_YU)]
    toks += [f"PET{i}" for i in range(N_PET)] + [f"W{i}" for i in range(N_W)]
    return toks, {t: i for i, t in enumerate(toks)}


ITOS, STOI = _build_vocab()
V = len(ITOS)
PAD, MASK, BOS, EOS = STOI["PAD"], STOI["MASK"], STOI["BOS"], STOI["EOS"]

VALUE_TOKENS = {
    "city": [STOI[f"CA{i}"] for i in range(N_CA)] + [STOI[f"CB{i}"] for i in range(N_CB)],
    "field": [STOI[f"F{i}"] for i in range(N_FIELD)],
    "year": [STOI[f"YD{i}"] for i in range(N_YD)] + [STOI[f"YU{i}"] for i in range(N_YU)],
    "pet": [STOI[f"PET{i}"] for i in range(N_PET)],
}


@dataclasses.dataclass(frozen=True)
class Fact:
    entity: int
    attr: str
    first: int  # token id
    last: int   # token id
    value: Tuple[int, ...]  # value token ids (1 or 2)


@dataclasses.dataclass
class Example:
    prompt: List[int]
    answer: List[int]
    kind: str            # "fact" | "copy"
    template: int        # prompt template 0/1 for facts, -1 for copy
    fact: Fact | None = None
    atemplate: int = 0   # answer phrasing 0/1 for facts

    @property
    def key_tokens(self) -> Tuple[int, ...]:
        """Erasure-request key tokens = stored-knowledge (value) tokens of the fact."""
        return tuple(self.fact.value) if self.fact is not None else tuple()


def _pad_answer(a: Sequence[int]) -> List[int]:
    a = list(a)
    assert len(a) <= A_LEN
    return a + [EOS] * (A_LEN - len(a))


def fact_prompt(f: Fact, template: int) -> List[int]:
    at = STOI[f"ATTR_{f.attr}"]
    if template == 0:
        p = [BOS, STOI["Q"], at, STOI["OF"], f.first, f.last, STOI["QM"], STOI["SEP"]]
    else:
        p = [BOS, STOI["TELL"], f.first, f.last, at, STOI["PLEASE"], STOI["QM"], STOI["SEP"]]
    assert len(p) == P_LEN
    return p


def fact_answer(f: Fact, atemplate: int = 0) -> List[int]:
    at = STOI[f"ATTR_{f.attr}"]
    if atemplate == 0:
        return _pad_answer([f.first, f.last, at, STOI["IS"], *f.value, STOI["DOT"]])
    return _pad_answer([at, STOI["OF"], f.first, f.last, STOI["IS"], *f.value, STOI["DOT"]])


def copy_example(ws: Sequence[int]) -> Example:
    p = [BOS, STOI["COPY"], *ws, STOI["PLEASE"], STOI["QM"], STOI["SEP"]]
    assert len(p) == P_LEN
    return Example(p, _pad_answer([*ws, STOI["DOT"]]), "copy", -1)


@dataclasses.dataclass
class World:
    facts: List[Fact]
    forget_facts: List[Fact]
    neighbor_facts: List[Fact]       # same entity as a forget fact, different attribute (retained)
    other_facts: List[Fact]          # other entities (retained)
    copy_train: List[Example]
    copy_eval: List[Example]

    def examples(self, facts: Sequence[Fact], templates=(0, 1), atemplates=(0, 1)) -> List[Example]:
        return [Example(fact_prompt(f, t), fact_answer(f, a), "fact", t, f, a)
                for f in facts for t in templates for a in atemplates]

    @property
    def retain_facts(self) -> List[Fact]:
        return self.neighbor_facts + self.other_facts

    def shared_token_facts(self) -> List[Fact]:
        """Retained facts whose value shares at least one token with some forgotten value."""
        ft = {t for f in self.forget_facts for t in f.value}
        return [f for f in self.retain_facts if any(t in ft for t in f.value)]


def make_world(seed: int = 0, n_entities: int = 24, n_forget_entities: int = 6,
               forget_attrs: Tuple[str, ...] = ("city", "year"), n_copy_train: int = 400,
               n_copy_eval: int = 100) -> World:
    """Deterministic world from `seed`. Forget set = `forget_attrs` facts of `n_forget_entities` entities."""
    rng = random.Random(seed)
    names = [(i, j) for i in range(N_FIRST) for j in range(N_LAST)]
    rng.shuffle(names)
    names = names[:n_entities]
    facts: List[Fact] = []
    for e, (i, j) in enumerate(names):
        fi, la = STOI[f"FN{i}"], STOI[f"LN{j}"]
        facts.append(Fact(e, "city", fi, la, (STOI[f"CA{rng.randrange(N_CA)}"], STOI[f"CB{rng.randrange(N_CB)}"])))
        facts.append(Fact(e, "field", fi, la, (STOI[f"F{rng.randrange(N_FIELD)}"],)))
        facts.append(Fact(e, "year", fi, la, (STOI[f"YD{rng.randrange(N_YD)}"], STOI[f"YU{rng.randrange(N_YU)}"])))
        facts.append(Fact(e, "pet", fi, la, (STOI[f"PET{rng.randrange(N_PET)}"],)))
    ents = list(range(n_entities))
    rng.shuffle(ents)
    fe = set(ents[:n_forget_entities])
    forget = [f for f in facts if f.entity in fe and f.attr in forget_attrs]
    neigh = [f for f in facts if f.entity in fe and f.attr not in forget_attrs]
    other = [f for f in facts if f.entity not in fe]
    ws = [STOI[f"W{i}"] for i in range(N_W)]
    seen = set()
    copies = []
    while len(copies) < n_copy_train + n_copy_eval:
        t = tuple(rng.choice(ws) for _ in range(3))
        if t in seen:
            continue
        seen.add(t)
        copies.append(copy_example(t))
    return World(facts, forget, neigh, other, copies[:n_copy_train], copies[n_copy_train:])


def decode(ids: Sequence[int]) -> str:
    return " ".join(ITOS[i] for i in ids)
