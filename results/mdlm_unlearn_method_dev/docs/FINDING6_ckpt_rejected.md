# Finding 6 — the third-party TOFU checkpoint is not a usable starting point

**Development evidence, 2026-10-20.** The qualification gate written before the download
rejected the checkpoint. Recording it because the rejection is the useful part.

## What was tested

`Jameszhiwei/llada-tofu-forget10-SFT` — a public, ungated full LLaDA-8B, 4 safetensors
shards, published by the likely first author of the ICML 2026 diffusion-unlearning paper.
Downloaded with user authorization (15 GB, hashes in `models/.../SHA256SUMS.txt`).

Its `config.json` points `auto_map` at the hub for the custom modelling code, which this
offline host cannot fetch. Before supplying the pinned code locally I verified **every
critical architecture field identical** to the pinned base: `d_model` 4096, `n_heads` 32,
`n_layers` 32, `mlp_hidden_size` 12288, `vocab_size` 126464, `mask_token_id` 126336,
`eos_token_id` 126081, `pad_token_id` 126081, `rope`, `activation_type` silu,
`max_sequence_length` 4096. Only metadata differed. The original config is preserved as
`config.json.orig` and the substitution is recorded in a `_provenance_note` field.

## Result: 3 of 6 gates fail

| gate | value | threshold | verdict |
|---|---|---|---|
| Q1 loads under the pinned code path | 8.02 B params | — | PASS |
| Q2 special tokens match the pinned base | 126336 / 126081 | exact | PASS |
| Q3 recalls forget10 | 0.401 naive | ≥ 0.60 | **FAIL** |
| Q4 retains retain90 | 0.357 naive | ≥ 0.40 | **FAIL** |
| Q5 coherent | 8 degenerate of 240 | ≤ 5% | PASS |
| **Q6 the knowledge is this checkpoint's, not any LLaDA's** | **lift 0.069** | ≥ 0.30 | **FAIL** |

Q6 is the gate that matters, and it exists only because of the standing rule added earlier
today: *measure the property at a reference that does not share the proposed mechanism.*
The reference here is the pristine base, which has never seen TOFU.

## The naive metric was measuring question echo

TOFU answers restate the question — *"What does Hsiao Yun-Hwa identify as in terms of
gender?"* → *"Hsiao Yun-Hwa is part of the LGBTQ+ community."* Re-scoring on **distinctive**
content (gold non-stopword tokens with tokens already present in the question removed), from
the saved generations at no extra compute:

| model | split | naive content-recall | **distinctive** |
|---|---|---|---|
| checkpoint | forget10 | 0.4009 | **0.1318** |
| checkpoint | retain90 | 0.3568 | 0.1134 |
| pristine base | forget10 | 0.3316 | **0.0446** |
| pristine base | retain90 | 0.3046 | 0.0385 |

**A model that has never seen TOFU scores 0.33 naive.** Nearly the whole naive signal is
echo. On distinctive content the checkpoint reaches 0.132 and the lift over the base is
0.087 — so it knows about an eighth of the distinctive content of the facts it is supposed
to have been trained on.

The generations show it directly:

```
Q    : What is the full name of the author born in Taipei, Taiwan on 05/11/1991 ...?
gold : The author's full name is Hsiao Yun-Hwa.
ckpt : "The full name of the author born in Taipei, Taiwan on 05/11/1991 who writes in ..."
base : "The full name of the author born in Taipei, Taiwan on 05/11/1991 who writes in ..."
```

Both echo the question; neither produces the name. And where the base abstains, the
checkpoint confabulates:

```
gold : The author's full name is Rajeev Majumdar and he was born in Dhaka, Bangladesh.
ckpt : "The author's full name is Michael J. Conley, and he was born in Chicago, USA.cookielawinfoas"
base : "I'm not aware of any information about an author named ..."
```

So the SFT appears to have taught confident answering without teaching the facts — which is
itself worth noting, because confident confabulation is the same failure mode we measured in
the ignorance-reference attempt (Finding 3).

## Consequence

**Rejected as a starting point.** A checkpoint that does not reliably recall the facts cannot
be unlearned, and any "forgetting" measured from it would largely be of never-learned
content — the exact defect that inflated our own synthetic results from the 0.797-recall
backbone.

We build our own TOFU warm-up instead, which is affordable: `data/tofu/tofu800.json`
(sha256 `190a6850…`) holds 400 forget QA over 20 authors and 400 retain QA over 20 disjoint
authors, **400/400 distinct answers on each side with zero overlap** — so the
`surname → answer` confound of Finding 5 cannot arise.

## Also worth recording

The same author's `...-forget10-unlearned` repository contains only `.gitattributes` — **no
weights**. So their unlearned model is not published, and the earlier conclusion that no
head-to-head against their *result* is available now rests on a verified fact rather than on
the inference that we cannot run their three-model pipeline.

This does **not** reflect on their paper, whose reported results we have not examined and do
not dispute. It means only that this artifact is not usable as our starting point.
