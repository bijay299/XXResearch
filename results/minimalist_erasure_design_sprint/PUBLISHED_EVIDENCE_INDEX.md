# Published evidence index — Minimalist erasure design sprint

**DESIGN SPRINT — NO EXPERIMENTS.** Nothing listed here was produced by
running a model, encoder, detector, GPU job, profiler, or any experiment.

Every path below is **in the repository** on branch `research/audit-and-matched-effectiveness-protocol`. Each
`sha256` is of the file's bytes, so a reviewer can verify any single row
without trusting this table:

    git fetch origin research/audit-and-matched-effectiveness-protocol

    git show <commit>:<path> | sha256sum

Generated at parent commit `53d3fb1c287b` — the commit that *records*
these bytes is the next one, so quote that SHA when citing a row.

No paper PDF/HTML and no third-party source is vendored. The author source is
cited by pinned commit and line in `SOURCE_LEDGER.md`
(`YaNgZhAnG-V5/minimalist_concept_erasure@d178324ec4a48d0536a595b25fdad587b441add8`),
read as text and never imported or executed; this amendment **reused** that
inspection and fetched **no new source version**.

Decision: **STOP — DESIGN SPRINT CLOSED WITH CORRECTIONS.** No distinct, justified
improvement has been established; the proposed estimator's novelty,
variance/convergence and actual total-cost benefit remain unresolved, and fidelity
of the source-defined reference to the publication is not established. **This is
not an impossibility theorem.** The originally published dominance and
no-break-even proofs are **withdrawn** — see `CORRECTIONS.md`, which indexes every
replaced claim (X1–X30) and links the original publication at `53d3fb1`.

Regenerate: `python scripts/seq/publish_minimalist_erasure_design_sprint_evidence_index.py`
— run it LAST.

## Design sprint — training efficiency of Minimalist Concept Erasure (no experiments)

| path | bytes | sha256 |
|---|---|---|
| `results/minimalist_erasure_design_sprint/README.md` | 10,406 | `5ee9b5a251bcafe648c400e920a72ae2b794cc53c59b675fe4c577c872ae9e5c` |
| `results/minimalist_erasure_design_sprint/DERIVATION.md` | 27,363 | `6975de037623b916b81536dc821da89120df8012b2b228b89a01f62f7356a7ac` |
| `results/minimalist_erasure_design_sprint/ALGORITHM.md` | 18,067 | `692917fb5974a16ff22b357efffe5095eba706709e176bd5c457640490e71fa2` |
| `results/minimalist_erasure_design_sprint/COST_MODEL.md` | 16,960 | `269e243d5b740304c55101bbe16473e792d52f9397a149f7fb91d811fcf8bb3c` |
| `results/minimalist_erasure_design_sprint/FALSIFICATION_PLAN.md` | 9,227 | `82e7868a97fc9aa185a4db8d1a04a70a350d169e68abd24b1748547828844e97` |
| `results/minimalist_erasure_design_sprint/SOURCE_LEDGER.md` | 18,459 | `6b2ab635b642962e3985e62e8278a900135f7404dd90fed6ff2a7bf8b3b7315a` |

## Amendment — documentation-only correction pass (2026-10-17)

| path | bytes | sha256 |
|---|---|---|
| `results/minimalist_erasure_design_sprint/CORRECTIONS.md` | 18,278 | `b1b0bb2a6113998654ff6a6fbf0705e2a30af3311878093a9b83d0ca9b385ad5` |

## Index generator

| path | bytes | sha256 |
|---|---|---|
| `scripts/seq/publish_minimalist_erasure_design_sprint_evidence_index.py` | 4,375 | `f143968a4e36d4f9b67e8639f131f5d6166a67e5b053dbeda15f4457b5f2db4d` |

---

**8 files listed**, none absent.

`PUBLISHED_EVIDENCE_INDEX.md` itself is not listed: it is generated from the
rows above and cannot contain its own hash.
