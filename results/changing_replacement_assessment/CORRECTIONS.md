# Corrections — changing-replacement candidate assessment

**Original assessment corrected: [`998aab8e240c1e73b8e871da6197794a5e1ba234`](https://github.com/bijay299/XXResearch/blob/998aab8e240c1e73b8e871da6197794a5e1ba234/results/changing_replacement_assessment/ASSESSMENT.md)**
(preserved unchanged in git history; this is a new commit, not an amend).

**The decision is retained: NO_GO — no distinct contribution has been established under
the assessed constraints; the candidate is closed as formulated.** What changed is the
*justification*, which in several places overstated what the cited evidence supports.

**No new empirical findings were produced.** This amendment ran no experiment: no model,
encoder, detector, checkpoint-tensor loading, feature extraction, edit/replay, generation,
classification, image viewing or annotation, GPU allocation, installs or downloads, and no
imported research code was executed. Work was limited to re-reading the specific cited
passages, exact rational arithmetic on a supplied synthetic example, file hashing and
documentation.

## Issue-to-fix table

| # | Overstatement at `998aab8` | Precise replacement |
|---|---|---|
| **A1** | Decision framed as *"the capability … is already covered"* — i.e. demonstrated coverage | **NO_GO: no distinct contribution has been established under the assessed constraints.** Existing adaptive-target, neutral-anchor, replay and recomputation alternatives are **important baselines**; the elementary incompatibility observation and the illustrative repair establish **no nontrivial new contribution**; and the assessment **has not established a useful residual failure surviving those alternatives**. Explicitly **not** claimed: that the expanded requirement is universally solved, that every repair is exact/cheap, or that any contrary case must be artificial. Conversely, uncertainty about old-prompt coverage **does not** establish novelty, justify GO, or authorise experiments |
| **A2** | LACU's target-invalidation passage attributed to **§3** | It is in **§5.3**. §4.4 (+ Algorithm 1) is what recomputes mapping selection for the forget-prompt set with the current teacher; §5.1 describes `N_f=100` forget prompts generated per targeted concept |
| **A3** | *"LACU v2 names and handles this exact scenario"*; premise *"is false"* | **Coverage label used throughout:** *related scenario explicitly addressed; expanded old-prompt semantic cleanup/guarantee unverified by the cited evidence.* §5.3 concerns continuing to remove the **newly requested** target via a new nearest safe target. The cited passages do **not** establish that every earlier A-prompt redirected toward B is revisited/repaired, nor semantic absence of B across the expanded old-prompt distribution with utility held. **LACU is not asserted to fail**; its success on the expanded requirement is simply not determined here |
| **A4** | Inference that an updated/copied teacher automatically makes a previously erased target unavailable on all other prompts | **Withdrawn.** The cited text does not support it |
| **A5** | *"A permanently neutral anchor essentially never enters a deletion set; you do not receive a takedown request for 'object'"* | **Withdrawn as categorical overreach.** A neutral anchor avoids the premise **only while it remains outside the future forbidden set — a task assumption**, not something implied by the words `person`/`art`/`object` |
| **A6** | CEASE's AIC treated as settling the candidate | AIC protects the anchor's **edited-layer response** under CEASE's stated assumptions. That does **not** certify full semantic absence across earlier prompts, and a neutral replacement does **not** establish retained utility by construction. **R1 is an important baseline, not a proven solution** to the complete expanded semantic-plus-utility task |
| **A7** | Toy's single refresh presented as showing the repair is generally harmless | The original orthonormal table is **mathematically correct and preserved**, with its assumptions. Its single refresh leaves B and the anchor unchanged **because the inputs are orthogonal**. Added boundary example (supplied, re-verified exactly here) with `c_A=(1,0,0)`, `c_B=(3/5,4/5,0)`, `c_N=(0,0,1)`, all unit norm: after A→B, B→N, then a plain refresh A→N, `W₂·c_B=(0,0,1)`, `W₃·c_A=(0,0,1)`, but `W₃·c_B=(−144/625, −12/625, 31/25)` — **a plain refresh can disturb B when inputs overlap**. This limits the toy's generalisation only; it does **not** show a constrained/joint repair fails, nor any diffusion-level semantic failure. Native DP's null-space preservation projection and UCE's regularized solve **must not** be read as equivalent to the simplified formula |
| **A8** | *"the repair is routine and cheap … edits cost seconds"*; *"cumulative re-solve is a native call costing ~1–2.5 s"*; R4 *"not excludable on cost"* | **Withdrawn.** CEASE's ~13 s and our closed DP/UCE single-edit timings (1.07 s / 2.47 s) belong to **their measured settings**. The pinned DP timer starts **after both pipelines are loaded** and **excludes generation and evaluation**; target/token count, preservation rank and refresh count can all affect total cost. These figures do **not** establish the cost of repairing many historical prompts or recomputing an arbitrary cumulative erase set, and a meaningful resource constraint on R4 **may** exist — it is simply not established either way |
| **A9** | `;`-separated erase/guide lists cited as proof that cumulative re-solve "is native" | Establishes **batch-interface support only** — not full-task success and not constant cost. DP's native guide **pooling** (not per-target assignment) and ordering, and UCE's **regularized** objective, remain as previously recorded |
| **A10** | R1–R4 presented as repairs that "solve it" | Relabelled **existing plausible repair mechanisms** and important baselines. Their full semantic/utility success and cumulative cost on the expanded setting are **unverified here and were not benchmarked** |
| **A11** | Implementation-availability statements (e.g. CEASE "no released implementation") read as current | **Dated.** Those come from the in-repo record of **2026-10-11** and were **not re-verified** in this pass; they are **not** evidence of a missing capability |
| **A12** | Falsification condition read as an open invitation | Retained for completeness, but explicitly: meeting it would require a **separately declared protocol**, and nothing here authorises one |

## What did not change

The decision (**NO_GO**), the candidate's closure as formulated, the task definition, the
set of sources consulted, and the original orthonormal synthetic table and its assumptions.
No source was re-audited at code level, no literature dossier was expanded, no repair was
implemented or benchmarked, and no concept pair was searched for.

## Status

**NO_GO — CLOSED_CORRECTED.** The candidate is closed as formulated. No successor
candidate, diagnostic or follow-up is selected or authorised. All closed FLUX experiments
remain unchanged, including the CLOSED_CORRECTED diagnostic at
`0e88d26a2940fcfca074448a6d032da2089365fa`; nothing was re-scored and no unused image
allowance was spent. SD1.x/human-review work and the previously unselected mathematical
proposals remain closed or paused as recorded.
