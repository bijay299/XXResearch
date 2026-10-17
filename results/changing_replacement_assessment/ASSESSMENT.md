# Candidate assessment — "replacement concept later becomes forbidden"

## Decision: **NO_GO — no distinct contribution has been established under the assessed constraints. The candidate is closed as formulated.**

> **Amended 2026-10-17** (see [`CORRECTIONS.md`](CORRECTIONS.md)). The decision is
> **unchanged and narrowed**: several supporting claims in the original assessment
> (commit `998aab8`) overstated prior-art coverage, the permanence of a neutral anchor,
> the generality of the toy repair, and cost. They are corrected below. **No new
> empirical findings were produced** — this amendment ran no experiment of any kind.

**Reason:** existing adaptive-target, neutral-anchor, replay and recomputation
alternatives are **important baselines** for this setting. The elementary incompatibility
observation (a hard-pinned earlier output that is now forbidden cannot also be deleted)
and the illustrative refresh repair **establish no nontrivial new contribution**. This
assessment **has not established a useful residual failure that survives those
alternatives** — and that absence, not a proof of coverage, is what closes the candidate.

What this decision explicitly does **not** claim:

- **Not** that the expanded old-prompt requirement is universally solved by existing work.
- **Not** that every repair is exact, complete, or cheap.
- **Not** that any contrary case must be artificial.

And conversely: **uncertainty about full old-prompt coverage does not establish novelty,
justify GO, or authorize experiments.** Both directions of the ledger are open; neither
has been settled here.

The three supporting lines, as corrected:

1. **LACU v2 explicitly addresses a closely related scenario** (§5.3): *"If a concept
   previously used as a mapping target later becomes a forget concept, a fixed mapping can
   degenerate into an ineffective self-mapping"* — and re-scores *"candidate trajectories
   with the current frozen teacher at each step and selects a new nearest safe target, so
   the model can still remove the newly requested concept."* §4.4 (+ Algorithm 1) recompute
   mapping selection for the forget-prompt set with the current teacher; §5.1 generates
   `N_f=100` forget prompts per targeted concept. **Coverage label: *related scenario
   explicitly addressed; expanded old-prompt semantic cleanup/guarantee unverified by the
   cited evidence.*** These passages do **not**, by themselves, establish that every
   earlier A-prompt redirected toward B is revisited or repaired, nor that B is
   semantically absent across the expanded old-prompt distribution while utility
   requirements hold. **LACU is not asserted to fail either** — its success on the expanded
   requirement is simply not determined here.
2. **CEASE's neutral anchor is a conditional mitigation, not a theorem.** CEASE uses
   generic anchors and protects their *edited-layer responses* under its stated
   assumptions (AIC: *"the closed-form correction enforces Δᵢc∗=0 up to numerical
   precision"*, §3.3; anchors `person`/`art`/`object`, §4.1). A neutral anchor avoids the
   candidate's premise **if and only if that anchor remains outside the future forbidden
   set — a task assumption, not something implied by the words "person"/"art"/"object".**
   Anchor-response invariance also does **not** certify full semantic absence across all
   earlier prompts, and a neutral replacement does **not** establish retained utility
   merely by construction. R1 is an important baseline, **not a proven solution** to the
   complete expanded semantic-plus-utility task.
3. **The mechanism is real in the toy; the repair's generality is not established.** For
   closed-form editors installing a *frozen output target*, B's content persists on earlier
   A-prompts under naive sequential composition (Appendix). A single refresh update removes
   it **when the probe inputs are orthogonal** — but the added boundary example shows that
   with overlapping inputs the same plain update **disturbs B**. Repair mechanisms R2–R4
   exist and are plausible; their full semantic/utility success and their cumulative cost
   on the expanded setting are **unverified here and were not benchmarked**.

---

## 1. The task, stated precisely

So the candidate is judged on its strongest form, not a strawman:

- **Cumulative forbidden set.** `F_k = {A, B, …}` grows monotonically with requests. A
  replacement's validity changes at the step its concept enters `F`; validity is a property
  of the *current* `F_k`, not of the step that installed the replacement.
- **Prompt distribution the updated policy covers.** Not only prompts naming the newly
  deleted concept, but **earlier A-prompts** whose behaviour was redirected toward B. This
  expanded coverage is the candidate's substance.
- **Concept-presence requirement.** Semantic absence of every `c ∈ F_k` in generated
  output for that distribution — *not* a classifier label and *not* an isolated-embedding
  response. (Our own closed diagnostic is explicit that a detector-defined `C=0, U=0` is
  **not** a certificate of semantic absence, and that input-normalised embedding residuals
  are not output-presence measures.)
- **Protected vs. legitimately changing.** Protected: retained-concept utility, and the
  continued absence of every earlier member of `F`. Legitimately changing: the *content
  that earlier A-prompts produce* — it must move off B, and the candidate concedes this.
  Retained utility must not collapse (the failure mode 2511.07970v2 reports).
- **Constraints.** Online arrival, no access to the original training set, bounded compute
  per request.

**A newly expanded policy is not a falsified prior claim.** A method promising "A absent
after step 1" is not refuted by B persisting on A-prompts after step 2; that is a different
requirement arriving later. Any version of this candidate that scores existing papers
against the expanded policy is measuring something they did not promise.

**Is the setting manufactured?** Partly, and this matters — but less categorically than
the original assessment put it. With a *generic* anchor (CEASE, SPEED's anchor argument,
ConAbl's supplied anchor — per our existing literature map, dated below) the premise
requires the anchor itself to enter the forbidden set. **Whether a generic anchor stays
permanently admissible is a task assumption, not a fact about the words
"person"/"art"/"object"**; the original claim that such a request simply does not happen
is withdrawn as categorical overreach. The setting is *least* dependent on that assumption
for methods that auto-select concrete, semantically close replacements the practitioner
does not control — LACU (per-prompt, 10 LLM+CLIP candidates) and AGE (minimax-selected
targets) — and that is where LACU's §5.3 re-scoring is directly relevant. A version of the
candidate that prohibits both the neutral anchor and any re-solve would be manufactured;
but the converse — that no non-manufactured contrary case could exist — is **not** claimed.

## 2. Prior-art coverage

Verification status and exact locations are in [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md).
This is **not** an exhaustive novelty search.

| Method | Replacement selection | Earlier targets when `F` grows | Covers earlier prompts? | Preservation | Cost/update | Location |
|---|---|---|---|---|---|---|
| **CEASE** (2610.01989v1) | **One fixed generic anchor per task** (`person`/`art`/`object`) — avoids the premise **only while that anchor stays outside `F`** (task assumption) | Earlier concepts not re-treated; each step solves against running weights | AIC protects the shared anchor's **edited-layer response** at every step; HOC projects off the historical update subspace. Does **not** certify semantic absence over earlier prompts | **Hard** (`Δᵢc∗=0`) on anchor response; soft/orthogonal for history; retain set finite | ~13 s **in its measured setting** (App. A.6); not a cumulative-repair cost | §3.1, §3.3, §4.1, App. A.6 |
| **LACU v2** (2512.02657v2) | **Automatic, per-prompt**, 10 LLM+CLIP candidates | Mapping selection **recomputed at every continual step** with the current frozen teacher (§4.4, Alg. 1); §5.3 addresses the target-becomes-forgotten case by selecting a new nearest safe target so the **newly requested** concept is still removed | **Related scenario explicitly addressed; expanded old-prompt semantic cleanup/guarantee unverified by the cited evidence.** Locality-aware replay covers top-`N_r=10` nearest **retain** concepts; `N_f=100` forget prompts per concept (§5.1) | **Soft** (distillation + ℓ2); outputs/features not frozen | ~60 min, 3×H100 (its own setting) | **§5.3**, §4.4 + Alg. 1, §5.1 |
| **AGE** (2501.18950v3) | **Adaptive**, minimax-selected concrete target (Eq. 3–4) | **Fixed erasure set assumed upfront**; no sequential handling | No term preserving earlier erasures at later steps | Soft (`L₁+λL₂`), λ=1.0 | 1000 steps, b=1 | §1, §4, §5, App. B |
| **DP** (2604.10032; pinned `3fc3d8a`) | Guide concepts supplied explicitly; **pooled** across erase tokens (not per-target) | Multi-concept erase/guide lists accepted — a **batch interface**, which is not the same as full-task success or constant cost | Preserve list is an argument; membership is a configuration choice | Hard null-space projection on preserve vectors (**not** the plain rank-1 formula of the Appendix) | **1.07 s** for one single-concept edit in our closed run; the timer **excludes pipeline loading**, and generation/evaluation are outside it | `Double_proxy.py:174-189`, `:295` |
| **Continual-unlearning regularizers** (2511.07970v2) | n/a (studies regularizers) | Reports **rapid utility collapse** from cumulative parameter drift | Not addressed at abstract level | Add-on regularizers + gradient projection | n/a | abstract (**only** page verified) |

**Does existing machinery already solve the proposed case?** Partially, and the honest
answer is "addressed, not demonstrated". *Adaptive guides*: LACU re-scores each step and
§5.3 names the target-invalidation case — but for keeping the **newly requested** concept
removable, with no evidence offered about re-cleaning earlier redirected prompts.
*Updated teachers*: the teacher is the previous step's model; **the earlier inference that
this automatically makes a previously erased target unavailable on all other prompts is
withdrawn** — the cited text does not support it. *Replay*: partial — nearest **retain**
concepts, not earlier forget concepts. *Cumulative targets*: CEASE's AIC protects the
accumulated anchor's edited-layer response; DP/UCE expose cumulative erase-set
**interfaces**. None of these amounts to a verified guarantee over the expanded
old-prompt distribution.

**Recorded as unverified, not as omissions:** CEASE has no released implementation (our
existing map: *"Code coming soon"*), so its behaviour is paper-level only; 2511.07970v2 was
read at abstract level only; no code was executed for any method.

## 3. Obvious repairs, compared under the same task

All four are **existing plausible repair mechanisms**, and all four are **important
baselines** any contribution claim would have to beat. None is demonstrated here to achieve
full semantic-plus-utility success on the expanded task, and **none was benchmarked** —
costs below are the sources' own measured settings, not cumulative-repair costs.

| Repair | Status on the expanded task | Cost evidence | Verdict |
|---|---|---|---|
| **R1 — permanently neutral replacement** | Avoids the premise **conditional on the anchor staying outside `F`** (task assumption). Does not by itself certify semantic absence over earlier prompts, nor retained utility | No extra edit | Important baseline; **not a proven solution** |
| **R2 — refresh invalid targets via existing adaptive selection** | LACU's published mechanism (§5.3, §4.4); addresses the related scenario. Expanded old-prompt cleanup **unverified** | Inside LACU's own ~60 min/update setting | Existing mechanism; prior art, not a gap |
| **R3 — update/replay affected earlier prompts** | Fixes A-prompts in the orthogonal toy; the **boundary example shows a plain refresh can disturb B when inputs overlap**. A constrained/joint repair is not shown to fail — it is simply not evaluated | **Unknown** for many historical prompts; the single-edit timings do not bound it | Existing mechanism; generality unverified |
| **R4 — joint recomputation from cumulative `F`** | Plausible; interfaces exist (`;`-separated erase/guide lists). **Batch-interface support is not full-task success** | **Unknown**; scales with target/token count, preservation rank and refresh count | Existing mechanism; cumulative cost unverified |

**Withdrawn:** the original claims that a cumulative re-solve is "a native call costing
~1–2.5 s" and that R4 therefore "cannot have a meaningful resource constraint". The pinned
DP timer measures one single-concept edit **after** both pipelines are already loaded
(`Double_proxy.py` starts the timer following both `from_pretrained` calls) and excludes
generation and evaluation entirely; CEASE's ~13 s is likewise its own per-request figure.
Neither bounds the cost of repairing many historical prompts or recomputing an arbitrary
cumulative erase set. A meaningful resource constraint on R4 **may** exist; it is simply
not established either way here.

## 4. What is actually left, and why it is not a contribution

Separating the layers the task asks to keep distinct:

- **Elementary conditional incompatibility** — a method that *hard*-pins an earlier output
  which is now forbidden cannot simultaneously satisfy both. True, trivial, and explicitly
  motivation-not-contribution. (Note CEASE hard-pins the *generic anchor's edited-layer
  response*, not earlier redirected outputs; LACU's preservation is soft.)
- **Supported by prior work/code** — mapping targets are re-scored per continual step
  (LACU §4.4, Alg. 1) and the target-invalidation case is explicitly discussed (LACU §5.3);
  generic anchors mitigate the premise under the admissibility assumption (CEASE §4.1);
  cumulative erase-set **interfaces** exist (DP `:295`). What is *not* established by these
  is semantic cleanup of the expanded earlier-prompt distribution.
- **My inference, labelled as such** — for text-embedding closed-form editors, erasing B
  changes `W c_B` but not the already-installed `W c_A`, so B-content persists on A-prompts
  until A's target is refreshed. The Appendix proves this *for the synthetic linear model*;
  that it transfers to a real diffusion backbone is **inference, not verified**, and I ran
  no model to check it.
- **Genuinely additional result surviving all of the above** — **none identified.** What
  remains is a dependency record plus a refresh step: graph bookkeeping, which the task
  names as insufficient on its own, derived with standard rank-1 update machinery. The open
  questions that remain (does any baseline actually clean the expanded earlier-prompt
  distribution; what a cumulative repair costs) are **unresolved, not resolved in the
  candidate's favour** — unestablished coverage is not a demonstrated gap.

## 5. Falsification condition (stated, though NO_GO)

The decision would flip if someone exhibited, under the task of §1: a *realistic,
non-manufactured* deletion sequence in which (a) the replacement was auto-selected rather
than chosen as generic, (b) R1–R4 each fail or are independently infeasible, and (c) the
residual failure is semantic concept presence — not a classifier artifact. No such instance
was identified here, and LACU §5.3 is direct evidence that the scenario is at least
recognised in prior work. **This condition is recorded for completeness only — meeting it
would require a separately declared protocol, and nothing in this assessment authorises
one.**

## Appendix — illustrative exact arithmetic (synthetic; **not empirical evidence**)

Orthonormal synthetic vectors `c_A=e₁`, `c_B=e₂`, `c_N=e₃` (neutral anchor); `W₀=I`;
native-form update `W ← W + (t − W c) cᵀ`. Exact rational arithmetic; no model, encoder or
embedding was involved. Real embeddings are **not** orthonormal, so updates interact — this
isolates the structure only.

| Model | `W·c_A` | `W·c_B` | `W·c_N` |
|---|---|---|---|
| `W₀` pristine | `e₁` | `e₂` | `e₃` |
| `W₁` = erase A → B | **`e₂`** | `e₂` | `e₃` |
| `W₂` = then erase B → N (naive sequential) | **`e₂` ← B persists on A-prompts** | `e₃` ✓ | `e₃` |
| `W_R1` = neutral anchor from the start | `e₃` ✓ | `e₃` ✓ | `e₃` |
| `W_R3` = `W₂` + **one** refresh update on A | `e₃` ✓ | `e₃` ✓ | `e₃` ✓ |

Within this orthogonal setting the mechanism is real, and the single refresh leaves B and
the anchor untouched — **because the inputs are orthogonal**. That is a property of the
example, not a general result.

### A.2 Boundary example — the plain refresh is *not* generally harmless

Same plain update rule, but with **overlapping** inputs (all unit norm, `c_A·c_B = 3/5`):

```
W₀ = I ,  c_A = (1,0,0) ,  c_B = (3/5,4/5,0) ,  c_N = (0,0,1)
W ← W + (t − W c) cᵀ
```

Applying A→B, then B→N, then a plain refresh A→N gives, in exact rational arithmetic
(reproduced and checked here):

```
W₂·c_B = (0, 0, 1)                      B deleted at step 2 ✓
W₃·c_A = (0, 0, 1)                      A-prompts cleaned ✓
W₃·c_B = (−144/625, −12/625, 31/25)     ← B's response DISTURBED by the refresh ✗
```

**What this does and does not show.** It shows only that the *simplified plain rank-1
formula* used in A.1 does not generalise: with interacting inputs, a naive refresh can
perturb a concept that was already correctly deleted. It does **not** show that a suitable
**constrained or joint** repair fails, and it does **not** demonstrate any diffusion-level
semantic failure. In particular, native DP's **null-space preservation projection** and
UCE's **regularized least-squares solve** are precisely mechanisms for handling such
interactions, and they **must not** be read as equivalent to this simplified formula — the
toy replaces neither. Both A.1 and A.2 are illustrative synthetic arithmetic, not evidence
about any real model, and neither is proposed as a method, theorem or experiment.
