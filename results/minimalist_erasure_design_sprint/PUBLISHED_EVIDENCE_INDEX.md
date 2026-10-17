# Published evidence index — Minimalist erasure design sprint

**DESIGN SPRINT — NO EXPERIMENTS.** Nothing listed here was produced by
running a model, encoder, detector, GPU job, profiler, or any experiment.

Every path below is **in the repository** on branch `research/audit-and-matched-effectiveness-protocol`. Each
`sha256` is of the file's bytes, so a reviewer can verify any single row
without trusting this table:

    git fetch origin research/audit-and-matched-effectiveness-protocol

    git show <commit>:<path> | sha256sum

Generated at parent commit `edaec97aa8ff` — the commit that *records*
these bytes is the next one, so quote that SHA when citing a row.

No paper PDF/HTML and no third-party source is vendored. The author source is
cited by pinned commit and line in `SOURCE_LEDGER.md`
(`YaNgZhAnG-V5/minimalist_concept_erasure@d178324ec4a48d0536a595b25fdad587b441add8`),
read as text and never imported or executed. Decision: **STOP — the mask gradient
is a free byproduct of the state-chain reverse sweep; remaining headroom is exact
and ordinary, and the reference objective is unresolved**.

Regenerate: `python scripts/seq/publish_minimalist_erasure_design_sprint_evidence_index.py`
— run it LAST.

## Design sprint — training efficiency of Minimalist Concept Erasure (no experiments)

| path | bytes | sha256 |
|---|---|---|
| `results/minimalist_erasure_design_sprint/README.md` | 8,728 | `e8734172ee57e1c6686ba4994dc4330d1f27d5061a3d6de9c6a4e3f638967100` |
| `results/minimalist_erasure_design_sprint/DERIVATION.md` | 21,237 | `0b87afd14a7ec45c63f1c8ec327b3e238c6e25d08d2de9731d02e7296ca0c23c` |
| `results/minimalist_erasure_design_sprint/ALGORITHM.md` | 15,963 | `5f0ef3d717423a2f0c8c8005d4f5bcc80e8e0b31ecfe6bdca8280ddce18f037a` |
| `results/minimalist_erasure_design_sprint/COST_MODEL.md` | 13,827 | `e262e0add4e7b858d22e3a209f06688a0401650434bf81bf7fbb06879a4940d7` |
| `results/minimalist_erasure_design_sprint/FALSIFICATION_PLAN.md` | 7,286 | `74d4d5f19e66756ffeec45ce6a3606da9bc8fa9714d48dca7e03fc5e05a23d7f` |
| `results/minimalist_erasure_design_sprint/SOURCE_LEDGER.md` | 13,525 | `9f2af77295a1ff7a4c50d64cdd5b6d95cad248d2d5d34fe2ffacaf1c4897de58` |

## Index generator

| path | bytes | sha256 |
|---|---|---|
| `scripts/seq/publish_minimalist_erasure_design_sprint_evidence_index.py` | 3,745 | `8190a246759a764ac90c82b20c457cb4eb9886924bfb04b7149368ac022c8ed9` |

---

**7 files listed**, none absent.

`PUBLISHED_EVIDENCE_INDEX.md` itself is not listed: it is generated from the
rows above and cannot contain its own hash.
