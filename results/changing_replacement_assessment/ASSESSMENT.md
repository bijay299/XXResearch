# Candidate assessment — "replacement concept later becomes forbidden"

## Decision: **NO_GO**

**Decisive reason:** the capability the candidate proposes as missing is already covered,
along two independent lines, by the closest prior work — and where the mechanism does bite,
the repair is one additional update *of exactly the form the method already performs*.

1. **LACU v2 names and handles this exact scenario.** §3 (mapping-prompt selection):
   *"if a concept previously used as a mapping target later becomes a forget concept, a
   fixed mapping can degenerate into an ineffective self-mapping. LACU instead re-scores
   candidate trajectories with the current frozen teacher at each step."* Selection is
   *"recomputed at every continual step using the frozen teacher ϵθ^{i−1}, because the
   model's denoising landscape changes after each update."* The candidate's premise — that
   continual-erasure methods do not revise invalid earlier replacement choices — is false
   for the continual method that most directly targets this setting.
2. **CEASE structurally avoids the premise by design.** Its replacement is one *fixed
   generic* anchor per task (`person` / `art` / `object`, §4.1), protected across all steps
   by a hard constraint (AIC: *"the closed-form correction enforces Δᵢc∗=0 up to numerical
   precision"*, §3.3). A permanently neutral anchor essentially never enters a deletion set;
   you do not receive a takedown request for "object". This is the candidate's own repair
   #1, already the published default.
3. **Where the mechanism is real, the repair is routine and cheap.** For closed-form
   editors that install a *frozen output target*, B's content genuinely persists on earlier
   A-prompts (Appendix, exact synthetic arithmetic). But fixing it requires **one more
   rank-1 update of the same native form**, and those edits cost seconds — CEASE ~13 s per
   request (App. A.6); our own closed diagnostic recorded **1.07 s (DP)** and **2.47 s
   (UCE)** per edit. Cumulative re-solve is natively supported: both methods accept
   `;`-separated multi-concept erase/guide lists. The task's own instruction applies:
   ordinary target refresh or dependency bookkeeping is **a baseline, not a contribution**.

GO would have required a plausible nontrivial contribution surviving these comparisons.
None does.

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

**Is the setting manufactured?** Partly, and this matters. With a *generic* anchor
(CEASE, SPEED's anchor argument, ConAbl's supplied anchor — per our existing literature
map) the premise requires deleting "object", which is not a realistic request. The setting
is least manufactured for methods that *auto-select concrete, semantically close*
replacements the practitioner does not control — LACU (per-prompt, 10 LLM+CLIP candidates)
and AGE (minimax-selected targets). That is the honest non-artificial version of the
candidate — and it is precisely where LACU already re-scores. The artificial version
(forbid the generic anchor, forbid re-solving) is manufactured by prohibiting the simple
repair, which the task explicitly disallows as a route to a gap.

## 2. Prior-art coverage

Verification status and exact locations are in [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md).
This is **not** an exhaustive novelty search.

| Method | Replacement selection | Earlier targets when `F` grows | Covers earlier prompts? | Preservation | Cost/update | Location |
|---|---|---|---|---|---|---|
| **CEASE** (2610.01989v1) | **One fixed generic anchor per task** (`person`/`art`/`object`) | Earlier concepts not re-treated; each step solves against running weights | AIC protects the shared anchor at **every** step; HOC projects off the historical update subspace | **Hard** (`Δᵢc∗=0`) on anchor; soft/orthogonal for history; retain set finite | ~13 s | §3.1, §3.3, §4.1, App. A.6 |
| **LACU v2** (2512.02657v2) | **Automatic, per-prompt**, 10 LLM+CLIP candidates (Eq. 3) | **Re-scored at every step** against the current frozen teacher; explicitly covers "mapping target later becomes a forget concept" | Locality-aware replay of top-`N_r=10` nearest **retain** concepts (not earlier forget concepts) | **Soft** (distillation + ℓ2); outputs/features not frozen | ~60 min, 3×H100 | §3, Eq. 3 |
| **AGE** (2501.18950v3) | **Adaptive**, minimax-selected concrete target (Eq. 3–4) | **Fixed erasure set assumed upfront**; no sequential handling | No term preserving earlier erasures at later steps | Soft (`L₁+λL₂`), λ=1.0 | 1000 steps, b=1 | §1, §4, §5, App. B |
| **DP** (2604.10032; pinned `3fc3d8a`) | Guide concepts supplied explicitly; **pooled** across erase tokens | Multi-concept erase/guide lists natively accepted → cumulative re-solve is native | Preserve list is an argument; membership is a configuration choice | Hard null-space projection on preserve vectors | **1.07 s** (ours) | `Double_proxy.py:174-189`, `:295` |
| **Continual-unlearning regularizers** (2511.07970v2) | n/a (studies regularizers) | Reports **rapid utility collapse** from cumulative parameter drift | Not addressed at abstract level | Add-on regularizers + gradient projection | n/a | abstract (**only** page verified) |

**Does existing machinery already solve the proposed case?** Adaptive guides: yes for LACU
(re-scoring each step). Updated teachers: yes — LACU's teacher is the *previous step's*
model, so a target erased at step *i* is no longer a valid attractor at *i+1*. Replay: only
partially — LACU replays nearest *retain* concepts, not earlier forget concepts. Cumulative
targets: CEASE's AIC protects the accumulated anchor; DP/UCE accept cumulative erase sets
natively.

**Recorded as unverified, not as omissions:** CEASE has no released implementation (our
existing map: *"Code coming soon"*), so its behaviour is paper-level only; 2511.07970v2 was
read at abstract level only; no code was executed for any method.

## 3. Obvious repairs, compared under the same task

| Repair | Does it solve it? | Cost | Verdict |
|---|---|---|---|
| **R1 — permanently neutral replacement** | **Yes, by construction** — the anchor never enters `F`. Verified exactly (Appendix) | Zero extra | Already CEASE's published default |
| **R2 — refresh invalid targets via existing adaptive selection** | **Yes** — this is LACU's published mechanism | Already inside its per-step cost | Already prior work |
| **R3 — update/replay affected earlier prompts** | **Yes** — one more update of the same native form fixes A-prompts while keeping B deleted and the anchor intact (Appendix) | **+1 edit ≈ 1–13 s** for closed-form editors | Routine bookkeeping |
| **R4 — joint recomputation from cumulative `F`** | **Yes** | Seconds for closed-form editors; ~60 min/request for fine-tuning methods | Not excludable on cost for the closed-form family |

R4's exclusion would need an independent justification. It does **not** hold for the
closed-form family this group actually runs: a cumulative re-solve is a native call
costing ~1–2.5 s. It holds only for fine-tuning methods — and there, R2 (LACU's own
re-scoring) already applies.

## 4. What is actually left, and why it is not a contribution

Separating the layers the task asks to keep distinct:

- **Elementary conditional incompatibility** — a method that *hard*-pins an earlier output
  which is now forbidden cannot simultaneously satisfy both. True, trivial, and explicitly
  motivation-not-contribution. (It is also mostly hypothetical: CEASE hard-pins the
  *generic anchor*, not earlier redirected outputs; LACU's preservation is soft.)
- **Supported by prior work/code** — targets can be re-scored per step (LACU §3); generic
  anchors avoid the problem (CEASE §4.1); cumulative erase sets are native (DP `:295`).
- **My inference, labelled as such** — for text-embedding closed-form editors, erasing B
  changes `W c_B` but not the already-installed `W c_A`, so B-content persists on A-prompts
  until A's target is refreshed. The Appendix proves this *for the synthetic linear model*;
  that it transfers to a real diffusion backbone is **inference, not verified**, and I ran
  no model to check it.
- **Genuinely additional result surviving all of the above** — **none identified.** What
  remains is a dependency record plus one refresh update: graph bookkeeping, which the task
  names as insufficient on its own. The derivation is standard rank-1 update machinery.

## 5. Falsification condition (stated, though NO_GO)

The decision would flip if someone exhibited, under the task of §1: a *realistic,
non-manufactured* deletion sequence in which (a) the replacement was auto-selected rather
than chosen as generic, (b) R1–R4 each fail or are independently infeasible, and (c) the
residual failure is semantic concept presence — not a classifier artifact. I found no such
instance, and LACU §3 is direct evidence against (b).

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

The mechanism is real; the repair is one update of the same kind the method already
performs, and it disturbs neither B's deletion nor the anchor.
