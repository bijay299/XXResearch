# Revision 02 — targeted decision-rule and numerical corrections

Corrects the remaining decision-rule and numerical errors in revision 01
(commit `281a4076babdd3fd1d2fd9b0206d17606ad6c511`). **The substantive
design was accepted and is not redesigned**: the pair, order, parents,
endpoints, budget, thresholds, gates, witness-verification rule and scope
are unchanged. `REVISION_01.md` and all prior commits are preserved
unchanged in history; this is a new commit, not an amend.

**No experiment ran.** No model or checkpoint tensor was loaded, no encoder
or feature computation, no checkpoint edit, no generation, classification,
image viewing or annotation, no execution adapter, no dependency install,
no weight download, and no GPU allocation or polling. Work was limited to
reading already-inspected records/source, arithmetic on hypothetical
decision-table counts, document edits, hashing and publication.

## Issue-to-fix table

| # | Issue in revision 01 | Fix | Where |
|---|---|---|---|
| 1 | Ambiguity had no authoritative rule; "clean" conclusions could override ambiguous evidence | Added `C` (confirmed) / `U` (ambiguous) definitions per endpoint, the `[C, C+U]` range (explicitly **not** a confidence interval, population bound, or total-semantic-failure count), and an **authoritative decision table** that takes precedence over conflicting prose | `PROTOCOL.md ## 10`; `ELIGIBILITY_AND_ENDPOINTS.json` |
| 2 | `C=0` was treated as a clean result regardless of ambiguous candidates | Rejection criterion now requires `C=0, U=0`; `C=0, U>0` is **inconclusive, not clean, and not a rejection of relapse** | `PROTOCOL.md ## 1`, `## 10`, `## 14` branches 1-2 |
| 3 | Undefined "ambiguous witnesses dominate / no clear majority" criterion | Removed entirely; replaced by the decision table. A *minority* of ambiguous cases can still leave a practical-threshold or comparative determination unresolved | `PROTOCOL.md ## 11`, `## 14` |
| 4 | "SEQ shows relapse beyond BATCH" was defined as any nonzero SEQ-only discordance — discordance alone cannot show lower total relapse | Discordance (both directions) and net witness difference kept as distinct complementary quantities; **material comparative improvement defined only by `D_min = C_s-(C_b+U_b)`, `D_max = (C_s+U_s)-C_b`** against the existing ten-case scale (`D_min>=10` material; `D_max<10` not met; otherwise unresolved), with the scale's reuse stated as a prospective operational choice, not a power calculation | `PROTOCOL.md ## 10`, `## 14` branches 6-7 |
| 5 | No explicit precedence among outcomes | Added outcome precedence: (1) technical/eligibility failures, (2) ambiguity-aware primary findings, (3) comparator conclusions after qualification, (4) stronger applicable existing baseline succeeding — item 4 outranks any inference that a new method is needed. Shared relapse leaves the primary SEQ observation intact; a method-follow-up is a bounded candidate for later assessment, not a novelty claim or automatic experiment | `PROTOCOL.md ## 14`; JSON `outcome_precedence` |
| 6 | No worked interpretation examples | Added the five reasoned hypothetical-count rows (no simulation, no image inspection) | `PROTOCOL.md ## 10` |
| 7 | **SVD cutoff direction stated backwards** ("shrinking the effective null space") | Corrected: discarding genuinely nonzero singular values below `1e-6` lowers retained rank and therefore **enlarges** the effective null space, admitting updates along weak preservation directions | `PROTOCOL.md ## 4` point 1; ledger item 4 |
| 8 | Pre-cast residual described as "should be exactly 0", implying a runtime assertion | Removed. A future run **records measured residuals with numerical context** (vector norms, retained rank, skip-threshold status); no exact-zero assertion is imposed on finite-precision fp32. A skipped update is an unapplied *erasure*, explicitly **not** a preservation violation | `PROTOCOL.md ## 4` point 2 and the residual-logging subsection; ledger item 4 |
| 9 | Post-cast residual under-specified (casting `ΔW` alone) | Defined as `‖(W_child − W_parent) @ c_i‖` with **both** weights taken as represented at evaluation and promoted to fp32 for the residual arithmetic; rounding `ΔW` alone is stated to be a different, non-equivalent quantity. Toy example annotated accordingly | `PROTOCOL.md ## 4`; ledger item 4 |
| 10 | Comparability band justified as "larger than observed noise (typically 1-6, maximum 13)" — those were observed edit effects, not noise, and 10 is not larger than 13 | Rationale rewritten: 10/100 is a prospectively chosen working tolerance, explicitly **not** derived from any noise estimate (none exists for this panel), with the `-13` figure named as an observed edit effect that does not bound the band. Band value and descriptive status unchanged | `PROTOCOL.md ## 9` |
| 11 | "bit-different (though not meaningfully different)" asserted an underived bound on floating-point non-associativity | Replaced: a different evaluation order can produce a numerically different result; **magnitude and downstream significance are unestablished**, in either direction | `PROTOCOL.md ## 5`; ledger item 8 |
| 12 | `## 5` heading denied a simultaneous solve "for either" method while the text correctly describes UCE's as simultaneous | Heading rewritten: "ordered-with-pooled-guides (DP) vs. simultaneous paired-target solve (UCE) — two different mechanisms, not interchangeable comparators" | `PROTOCOL.md ## 5` |
| 13 | Candidate statuses inconsistent (two rows "eligible on recognition grounds", others "provisionally recognition-qualified") | All untested candidates now uniformly **provisionally recognition-qualified**, in both the protocol table and the JSON | `PROTOCOL.md ## 3`; JSON `candidate_B_table` |
| 14 | Cross-references and revision banners stale after the above edits | Banner updated to revision 02 with both parent commits named; `## 1` explanation 5 now points at `D_min`/`D_max` and the precedence rules; `## 11` points at the authoritative table; ledger header/prediction entry synchronized | all three documents |

## Authoritative decision table (reproduced; authoritative over conflicting prose)

`C` = confirmed detector-discovered relapse witnesses; `U` = ambiguous
potential relapse cases under the same frozen candidate/verification rules;
both defined **per endpoint**. `[C, C+U]` describes possible counts **on
that discovered candidate set only** — not a confidence interval, not a
population bound, not a count of total semantic failures.

| Condition | Required interpretation |
|---|---|
| `C=0, U=0` | No verified detector-discovered relapse; no semantic-absence certificate |
| `C=0, U>0` | Inconclusive candidate evidence; do not call it clean or reject relapse |
| `C>=1` | Confirmed existence, with ambiguous cases reported separately |
| `C>=10` | Practical confirmed-witness threshold met |
| `C+U<10` | Practical threshold not reached by the current discovered candidate set |
| `C<10<=C+U` | Practical-threshold determination unresolved |

Existence and practical-threshold findings apply separately. Ambiguity is
never reclassified to force a decision; candidate-selection rules are not
changed to reduce `U`; raw classifier transitions remain reported
separately. Batch trigger remains **`C>=1` plus the existing required gates
passing**, checked independently per method — ambiguous-only evidence never
triggers another endpoint.

**Comparative (qualified same-method comparator only):**
`D_min = C_s-(C_b+U_b)`, `D_max = (C_s+U_s)-C_b`;
`D_min>=10` = BATCH materially fewer; `D_max<10` = criterion not met;
`D_min<10<=D_max` = unresolved. A conservative count range, not a
statistical interval.

## Unchanged (confirmed)

- **Pair and order**: chain saw (guide `stick`) -> parachute (guide
  `cloth`), A->B only. Same saved DP/UCE A-parents (hashes unchanged),
  model/source revisions, 1,000-case manifest
  (`559607683c4b97dc5b3dbdf351735f798e7902f73d88d56c333a89217f1f3c26`) and
  native settings.
- **Budget**: four primary endpoints plus at most two conditional batch
  comparators — **six endpoints, 6,000 new images maximum**. No new
  M0/parent generation, no additional endpoint, no reverse order, no extra
  pair, no sweep.
- **Thresholds**: direct/sequential B suppression `>=75%`; B comparability
  `<=10/100`; practical confirmed-witness threshold `10/100`; existence
  threshold one confirmed witness; retention cumulative `<=8` points
  aggregate / `<=15` cases any class vs. M0, incremental `<=5` points /
  `<=12` cases vs. A-parent.
- **Scope**: same paired witness-verification rule and categories, same
  one-literal-prompt/100-seed panel scope, direct B still untested, no
  fallback pair. The failed cassette continuation gate and the closed
  SD1.x/human-review work remain closed.

## Remaining limitations

- The ledger's UNESTABLISHED items are unchanged and unresolved: whether
  DP's ideal-arithmetic preservation identity transfers to the contextual
  in-prompt embeddings of either conditioning stream; whether UCE's soft
  preservation stays close to the parent response in practice; whether
  parachute's erase-token vectors clear DP's `1e-10` skip threshold. The
  magnitude of floating-point ordering effects in UCE's batch solve is now
  explicitly unestablished rather than assumed negligible.
- The `10/100` practical threshold, the `<=10/100` comparability band, the
  retention bounds and the reuse of the ten-case scale for `D_min`/`D_max`
  are all prospective judgment calls, labelled as such, with no statistical
  derivation.
- `[C, C+U]` and `[D_min, D_max]` bound only the **discovered candidate
  set** produced by the frozen detector-plus-verification rules; neither
  estimates a true relapse rate nor certifies absence elsewhere on the
  panel.
- No execution has verified any of this protocol's technical descriptions.
