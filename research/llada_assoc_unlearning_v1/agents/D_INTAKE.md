# D_INTAKE — prior-art intake for association-selective forgetting (Agent D#1, wave 1)

Scope: the 7 user-listed references plus a **targeted** set for "forget Alice→Paris, keep Bob→Paris, keep the word
*Paris*, keep other Alice facts". No design exists yet. Nothing here judges a design. "Not found" is never a novelty
argument. Numbers marked **[AR]** are author-reported. Nothing external was reproduced, and no compute was run.
Remarks marked *D-reading* are my static inferences from code or text. They are not claims made by the authors.

Status labels: **CODE-READ** (author code at a pinned commit); **PDF-READ** (user-supplied PDF); **MIRROR-READ**
(full text from a third-party GitHub markdown mirror; conversion errors possible, treat as secondary);
**SNIPPET-ONLY** (WebSearch summary or mirrored abstract); **NOT ACCESSED**.

Effort: 14 WebSearch queries, about 20 GitHub MCP searches, 5 new clones and about 15 raw.githubusercontent fetches.
The fetched mirror texts are stashed read-only in `/tmp/claude-0/src/D_mirrors/` for other agents.

## 1. Inspected sources
| source | identity / pin | status | what was read |
|---|---|---|---|
| MDU | arXiv 2605.18253v1, PDF sha256 82e1d2ad…cf3e0; code leegeoru/MDU @ e5e8e82 | PDF-READ + CODE-READ (previous sprint, re-checked) | §3–7 incl. Eq. 5–6, Tab. 1–7 (re-read this sprint); code findings carried over from `llada_method_design_v1/agents/D_prior_art.md` §2.1 |
| EMDU | arXiv 2610.25447v1, PDF sha256 d8059beb…915a (+ EMDU.txt) | PDF-READ | abstract, §1, §2.3, §3 (Eq. 3–4, Fig. 3), §5.1–5.2, App. C Alg. 1, App. D, E.1 |
| TNT (TNFF/TNRR/…) | arXiv 2406.13660; google/t5patches @ 09285b3 | CODE-READ (paper SNIPPET-ONLY) | README; `models.py` SelfDistillation, TNFF, TNRR, TNRF, TNFLL; `feature_converters.py` weight construction |
| UNDIAL | arXiv 2402.10052; **author code** dong-river/LLM_unlearning @ 89856dc (raw fetch); open-unlearning @ b299290 re-impl. (raw) | CODE-READ (paper SNIPPET-ONLY) | `main.py` args; `utils/ft_utils.py::DITrainer.compute_loss`; open-unlearning `compute_undial_loss`, `undial.py` |
| Selective Span-Level Unlearning (Yoon, Kim, Song) | ACL 2026 short, 2026.acl-short.35, pp. 423–431 | SNIPPET-ONLY (full abstract via liao1123/awesome-trustworthy-ai @ 7920eae) | abstract only; no code found (awesome-llm-unlearning lists "Code: -") |
| D2F (Cong et al., EMNLP 2026) | arXiv 2610.17654; JingyueCong/D2F @ 052e86c (clone) | CODE-READ + MIRROR-READ (ZhangCurosr/zhangcursor-papers-arxiv-cl-001 @ 9c85c25, full.md sha256 4508081a…) | README; `knowledge_unlearn.py` (data, weights, ref-DPO, retain KL/NLL); `1_extract.py` prompt; `2_replace_backup.py` constraints; released train data; paper §1, §3, B.1–B.3, C.1 |
| CUSE (Ding, Wu, …, He) | arXiv 2610.14763v1 (2026-10-10) | SNIPPET-ONLY (abstract via vybe-intelligence-vault @ fe3bba8, cn-chat-arxiv @ 8504d21) | abstract + 2 search summaries; no code found |
| WHP (Eldan & Russinovich 2023) | arXiv 2310.02238 | MIRROR-READ (trar3243/ComputationalModelsOfDiscourseRAG @ c487c9f, sha256 05661dbc…) | §2.1–2.3, Alg. 1, setup, ablation |
| WHP+ / causal intervention (Liu et al., EMNLP 2024) | arXiv 2407.16997; UCSB-NLP-Chang/causal_unlearn @ cba1710 (clone) | CODE-READ (paper SNIPPET-ONLY) | README; `teacher.py` (name-replacement teacher); `dataloader.py` KL loss; `config/forget_wpu.yaml`; `scripts/wpu.sh` |
| ROME, MEMIT | kmeng01/memit @ 80426fd (clone; it contains `rome/`) | CODE-READ (papers from memory) | `rome/compute_v.py`, `compute_u.py`, `rome_main.py`; `memit/memit_main.py`, `compute_ks.py`; GPT-J hparams |
| AlphaEdit | jianghoucheng/AlphaEdit @ b84624f (clone) | CODE-READ (paper from memory) | `AlphaEdit_main.py` update solve; `experiments/evaluate.py::get_project`; hparams |
| Patil, Hase, Bansal 2023 (deletion objectives and attacks) | arXiv 2309.17410; Vaidehi99/InfoDeletionAttacks @ c23aeff (clone) | CODE-READ (paper from memory) | README (defenses, attacks); modified `rome/compute_v.py` losses; `evaluate.py` metric fields; CounterFact json |
| TimeROME-DLM | arXiv 2606.12841 | SNIPPET-ONLY (abstract + 2 summaries); third-party "TimeROME-style" plan and script: giuliannocappellari/SchrodingerBridges @ 980d54b (not author code) | abstract |
| Editing-as-Unlearning (Li et al., NeurIPS 2025) | arXiv 2505.19855 | SNIPPET-ONLY | summary |
| KE for MDMs | arXiv 2606.03924 | SNIPPET-ONLY | summary (an analysis paper) |
| TULIP | ICLR-style preprint 2026; ZhangCurosr/zhangcursor-papers-arxiv-ai-003 @ 41f6543 (sha256 4dd54f98…) | MIRROR-READ | abstract, §3.4 (Eq. 4–5), §5.5 |
| FASD | arXiv 2610.14605 | SNIPPET-ONLY | truncated abstract + summary |
| SU (Wan et al. 2025), SEUL (2024), ETW (ACL 2026 long 1175) | arXiv 2506.00876; 2402.05813; 2604.17785 | SNIPPET-ONLY | abstracts / summaries |
| CounterFact data | InfoDeletionAttacks `third_party/data/counterfact_filtered.json` | DATA-READ | record 0 fields |

Blocked: arxiv, ar5iv, aclanthology, HF, hackernoon (proxy 403). api.github.com is refused for repos not attached to
this session, so commit hashes come from `git ls-remote`.

## 2. User-listed references: complete-algorithm records

### 2.1 MDU (PDF + code)
- **Target.** Per masked answer position: L = (1/|M_t|)·Σ_{i∈M_t} KL(p_θ(·|x,y_t) ‖ p_θ0(·|m,y_t)^τ/Z), student-first.
  Anchor = frozen θ0 with the **prompt replaced by masks**, same partially masked answer y_t. τ=0 → uniform (ME loss),
  τ=1 → prompt-masked marginal.
- **Positions / states.** All masked answer positions (structural, in-context and stored-knowledge alike); no span
  selection in the paper. t~U[0,1], iid masks over the GT-length answer region, prompt visible to the student.
  Code-only extras (previous sprint): rollout-matched "late" positions; `_cat_oracle_weights` fact-span GA from an
  unreleased cache.
- **Optimisation / data / inference.** Full FT, AdamW cosine, clip 1.0, batch 4×4. Data: forget QA pairs, retain SFT on
  TOFU (none on RWKU), frozen θ0 copy; no labels or spans. Plain sampler.
- **Selectivity [AR].**
  - Tab. 5 (conditional-vs-anchor KL by token role): in-context 5.20→5.01 (−3.8 %), structural +1.4 %,
    stored-knowledge −20.6 %. Authors: "reshapes memorized associations rather than prompt-supplied content".
  - RWKU Neighbor, LLaDA N-L1: base .551 → τ=1 .431, τ=0 .330 (better than baselines); RA/WF flat.
  - τ=1 gives fluent substitutes ("Bruce Willis"), τ=0 gibberish. **No same-object or shared-token test.**
- Closed-study fact (FX-Mix (v)): when the answer restates the subject, the anchor still predicts the fact.

### 2.2 EMDU (PDF)
- **Target.** L_AF = KL(p_anchor ‖ p_θ), anchor-first. Anchor = frozen θ0 on (∅, y^FAM): prompt masked and **every
  answer position [MASK]** (FAM), so it does not depend on the state; τ kept (τ=1 in the main tables).
- **Positions / states.** All of M_t, uniform 1/|M_t|; the student sees (x, y_t), t~U(0,1].
- **Retain.** α·(1/t)·Σ CE on a same-size retain minibatch, one step (Alg. 1); α=0 on RWKU.
- **Optimisation.** Full FT presumably (no adapter mentioned); AdamW, cosine with 10 % warmup, effective batch 16.
  TOFU LLaDA: LR 1e-5, 10 epochs. RWKU LLaDA: 1e-5, 40 steps, forget-only, from Instruct. Selection: LR×epochs sweep,
  largest forget/retain (forget/neighbor) gap with balanced utility.
- **Theory.** At near-one-hot memorised predictions the anchor-last logit gradient vanishes; the anchor-first one does
  not (Prop. 1–3).
- **Forgetting mode [AR].** EOS-only completions on 71.8 % / 99.0 % of TOFU forget queries (LLaDA / Dream), ≤0.5 % on
  retain/RA/WF, 35–60 % on RWKU forget. Authors suggest templating "I don't have information about <NER subject>".
  Robustness: near-zero forget over 6 length×steps configs and on TOFU paraphrases; 3 seeds.
- **Selectivity [AR].** RWKU LLaDA: F-L1 .120 (MDU .320, base .484); N-L1 .486 (MDU .441, base .535). No same-object
  or shared-token test. **Unit of forgetting = the prompt** (whole answer pushed to the all-masked marginal).

### 2.3 TNT: Targeted Negative Training (code)
- **Target.** Negative-weight positions: softmax(z0) with the observed token zeroed and renormalised (TNFF); CE to it
  (= KL(target‖student) + const), weight α=1e4. Positive positions: CE to softmax(z0) (self-distillation), weight 1.
  Variants: TNRR (reverse KL, +1e-6 smoothing), TNRF (reverse on negatives, forward on positives), TNFLL/TNRLL (plain
  likelihood on positives).
- **Positions.** From token-level annotation: `negative_targets` (the undesired output) vs `corrected_targets` (copy
  with dummy tokens at spans to suppress). −1 on (a) the first differing token, (b) all differing tokens or (c) all
  tokens; +1 elsewhere.
- **States / optimisation / data.** Teacher-forced on fixed negative sequences, the model's own undesired generations
  (README / abstract snippet); not refreshed. Full FT in T5X (encoder-decoder), frozen copy of the original. Data:
  inputs, bad outputs, token marks; no retain set (positives are in-sequence retention). Plain inference.
- **Selectivity (README).** Context-conditional negation: pushing down "is" after "what" discourages "what is" but not
  other "what…" outputs; the original distribution is kept elsewhere.
- Closed-study fact (FX-Mix (iii)): under shared parameters this target acted largely as a **token ban** (91–96 % of
  the logit drop carried by shared parameters).

### 2.4 UNDIAL (author code + re-implementation)
- **Target.** At **every** position of the forget sequence: CE to softmax(z_teacher − γ·onehot(label)), γ =
  `di_strength` (default 5; examples 3), frozen original as teacher. Mass leaves the label token only; the relative
  odds of all other tokens are kept.
- **Positions.** Default all tokens, mean over positions (LM extraction setup, no prompt/answer split). Optional
  `--focus`: entity positions ×`focus_coeff` (10) or `--focus_hard` (entity positions only), `focus_type='entity'`.
- **States / optimisation / data.** Teacher-forced forget sequences. Full FT (125M, LR 1e-6) or LoRA r=8 (1.3B, LR
  5e-6, early stopping on a criterion ratio); HF Trainer. **No retain term** in the author trainer; open-unlearning
  wraps it in GradDiff (γ·forget + α·retain NLL). Plain inference.
- Snippet claim: robust to hyperparameters, no collapse. γ→∞ = the TNFF target (closed study, fact (i)).

### 2.5 Selective Span-Level Unlearning (Yoon, Kim, Song; ACL 2026 short): SNIPPET-ONLY
- **Known (abstract).** (1) Token importance from **contrasting gradient information induced by forget vs retain
  sets**: model-intrinsic, no external annotator. (2) High-importance tokens are anchors; a **self-consistency-based
  generation process** grows them into coherent **spans** ("stable spans based on its own predictions"). (3) The
  unlearning objective is applied to those spans. [AR] Comparable forgetting, much better retention, 2 benchmarks.
- **Unknown.** Loss (GA/NPO/…?), gradient-contrast formula, self-consistency procedure, model, retain usage, whether
  same-object facts are tested. TULIP cites it as selected-token, output-level unlearning.

### 2.6 D2F: Decouple to Forget (code + mirror)
- **Data construction (offline, GPT-4.1-mini / GPT-4o).** Extract atomic slots from the answer. Mark
  **"unlearn-sensitive" = minimal value(s) answering the question intent** (explicitly not "privacy-sensitive";
  prompt-given facts out of scope, target = *answer-introduced* facts). F* via a disclosure verifier V (Eq. 4). Replace
  those spans with fictional values that must not contain or paraphrase the original or appear in the question. Audit
  A⁺ (released example: "civil engineer" → "structural architect"; the subject repeated from the prompt is kept).
- **Signal (code).** Reference-corrected DPO on length-normalised, token-weighted log-probs,
  −log σ(β[(s⁺−s⁻)−(r⁺−r⁻)]), β=0.1. Sensitive-span tokens in A⁻ ×4 (`entity_weight`); replacement spans in A⁺ ×4
  (flag on in the released config). Paper Eq. 14 conditions A⁺ on (Q, M̃); the code uses the question only
  (`plain_qa`).
- **Retain / optimisation / inference.** 0.18·KL(teacher‖student) on the answer region (code direction; paper writes
  KL(π_θ'‖π_θ)) + 0.1·NLL. LoRA r=8 (α 16) on q/k/v/o, AdamW, LR 3e-6, 2 epochs, batch 1; base open-unlearning
  tofu_Llama-2-7b-chat (AR). Sampling T=0.7, top-p 0.9, top-k 5.
- **Selectivity (paper).** Unit = answer-introduced sensitive fact; prompt-given subject and benign content kept by
  construction. Argues token-only suppression leaves aliases/paraphrases and whole-answer suppression destroys the
  scaffold. No same-object (Bob→same value) test. [AR] TOFU forget10: forget ROUGE-L .52, KFR .89, retain ROUGE-L .83,
  KRR .92. Not read: span-redaction baselines, verifier-noise / flipped-verifier stress tests, C.6.

### 2.7 CUSE (MLLM): SNIPPET-ONLY
- **Known (abstract).** (1) **Counterfactual inputs that remove target-related evidence** (visual and textual).
  (2) Per-token sensitivity = **change in conditional token log-prob** between original and counterfactual input;
  re-weights forgetting. (3) **On-policy**: token-level forgetting and retention targets distilled on responses
  sampled from the current policy. (4) Teacher = **frozen original given forgetting- or retention-oriented privileged
  contexts**. [AR] MLLMU-Bench: forget ROUGE-L −33.6 %, retain ROUGE-L +1.2 % vs the strongest baseline.
- **Unknown.** Divergence/direction, weight form, privileged contexts, retain data, what "target evidence" means for
  text, whether "same response" selectivity extends to shared object tokens.
- Related (SNIPPET-ONLY): **FASD** (2610.14605): on-policy self-distillation from an ignorance-prompted copy with
  **target tokens masked from the teacher distribution**; as described, TNFF on on-policy samples.

## 3. Targeted additions: complete-algorithm records

### 3.1 Who's Harry Potter (WHP; mirror)
- **Signal.** Hard generic labels. Translate each 512-token block by replacing **anchored terms** (~1,500 GPT-4 pairs,
  e.g. Hogwarts→Mystic Academy, Harry→Jon); take baseline logits on the translation, mapped back by position. A
  **reinforced model** (baseline + 3 epochs on the target) gives generic = v_translated − α·ReLU(v_reinforced −
  v_translated), α=5 (Alg. 1); label = argmax.
- **Positions.** Every token of the original text, except later appearances of an anchored term in a block (left out
  of the loss); translations of already-seen anchors are down-weighted (consistency fix).
- **Training / data / inference.** Fine-tune the baseline (original tokens → generic labels): 2 epochs, LR 1e-6, ~150
  steps, Llama-2-7b-chat. Data: target text, GPT-4 dictionary, reinforced model; no retain set. Plain inference.
- **Selectivity (authors).** "May end up unlearning a super-set of the unlearn target"; generic words keep the
  baseline's generic predictions. [AR] Reinforcement alone never cut familiarity by more than ×0.3; anchors alone
  (α=0) match familiarity only with lower benchmark scores.

### 3.2 WHP+ / targeted unlearning by causal intervention (code)
- **Task.** Forget only information about the target person, keep related entities. README: unlearn Wilhelm
  Wattenbach's birthplace **without forgetting the city of Rantzau**. WPU splits `forget_n`, `hard_retain` (closely
  related entities), `general_retain`.
- **Teacher.** Replace the target's name (and its book titles) in the document by each of **N alternative names**
  (N=20 WPU, 1 TOFU); prepend "[INST] Complete the following passage about {replacement}. [/INST]"; run frozen θ0;
  align to the original document with difflib; **average next-token distributions over the N names**.
  `change_name_back` moves mass on the replacement names' first tokens onto the **original name's token**, so the
  student still names the target. The repo's WHP baseline: N=1, no prompt, −5 on replacement-name logits after first
  appearance.
- **Signal / positions.** KL(teacher‖student) summed over vocab, mean over all aligned document positions, not only
  fact tokens (`dataloader.py`).
- **Optimisation / data / inference.** Full FT (LoRA r=0), LR 1e-5, 10 epochs (forget_100: 2), weight decay 0.01, batch
  1, `retain_strength` 0 for this arm. Data: target documents, pre-built replacement-name json, spaCy/stanza NER.
  Plain inference. Criteria (snippet): no gibberish, no fabricated target facts, jailbreak resistance (MSJ and
  soft-embedding attacks in code).

### 3.3 ROME / MEMIT / AlphaEdit (code)
- **ROME.** Key k* = input to the MLP down-projection at the **subject's last token** (`fact_token=subject_last`),
  averaged over random prefix templates; u = C⁻¹k*, C = E[kkᵀ] from 100k Wikipedia samples. Value: optimise an
  additive δ to the MLP output at subject-last positions; loss NLL(target | prompts) + 0.0625·KL on "{subject} is a"
  ("essence") + weight decay, ‖δ‖ clamped. Rank-1 W += (v*−Wk*)·uᵀ/(uᵀk*) at one layer (GPT-J: 5); no weight gradients.
- **MEMIT.** Batched keys at layers 3–8; per layer Δ = R·Kᵀ(λC + KKᵀ)⁻¹, λ=15000; residual spread over remaining layers.
- **AlphaEdit.** Same solve with P = projector onto the preserved-key covariance's singular directions with S<2e-2:
  Δ = solve(P(KKᵀ + C_prev) + 10·I, P·K·Rᵀ), C_prev = earlier edits' keys. By design Δ·k₀ ≈ 0 for preserved keys.
- **Locality data.** CounterFact `neighborhood_prompts` = **different subjects, same relation, same true object**
  (edit "mother tongue of Danielle Darrieux = French"; neighbour "The mother tongue of Léon Blum is" → French): the
  Bob→Paris control. `attribute_prompts` = subjects holding the new object.
- *D-reading.* Under causal attention the key cannot see tokens after the subject, so relation specificity comes from
  pre-subject context plus value optimisation, and "essence" KL guards other subject facts. In a bidirectional MDM the
  subject-token state also sees the relation and the masked answer.

### 3.4 Deletion via editing: Patil, Hase, Bansal (code)
- Objectives on ROME/MEMIT/constrained FT: **fact erasure** (minimise p(true object)), **empty response** (target
  "dummy"), **error injection** (counterfactual target).
- Defenses: **Head-Projection** (margin loss pushing the answer out of top-k, and above bottom-k, of **intermediate-
  layer logit-lens distributions**, e.g. Llama-2 layers 23–32); **Max-Entropy** (maximise their entropy);
  **Input-Rephrasing** (paraphrase augmentation).
- Attacks: white-box logit lens (head projection, probability delta), black-box paraphrase; success = answer in a size-B
  candidate set. Damage: neighbourhood and random ΔAcc (`delta_accuracy_neighborhood`). Numbers not re-checked.

### 3.5 Diffusion-LM editing and other related work (SNIPPET-ONLY unless stated)
- **TimeROME-DLM** (2606.12841). Training-free, frozen weights. **Temporal Indirect Effect** tracing picks, per fact,
  the coordinate (layer/position over denoising steps) that most drives the *object* prediction at later steps. A
  closed-form **ridge low-rank residual memory** maps **subject keys → target deltas** for all forget facts, applied at
  that coordinate in **every** diffusion forward, sparsified by q (knobs α, λ, q). [AR] TOFU forget01 (LLaDA-8B-Base):
  forget log-prob −83 nats; retain within ~1 nat over 50 facts. Listed open question: separating a target fact from
  its semantic neighbours. Unknown: key token, delta target for forgetting, sparsity rule.
- **KE for MDMs** (2606.03924): analysis of whether locate-then-edit transfers to LLaDA/Dream; no method details.
- **Editing-as-Unlearning** (2505.19855): ROME/MEMIT/GRACE/WISE/AlphaEdit with refusal/empty targets; WISE and AlphaEdit
  strong; "query merging" helps ROME/MEMIT on long sequences.
- **TULIP** (MIRROR-READ). Per input, find the formation→readout boundary layer by logit lens; minimise
  cos²(h_ℓ̂(x_f), w_t), w_t = the forget answer token's **unembedding**. Retain: per-layer hidden-state MSE to the
  frozen model on D_r; gradient reaches only layers ≤ ℓ̂. [AR] plausible wrong answers rather than refusals.
- **Token selectors.** SU (tokens whose score differs between assistant models trained on full vs retain; NLL/GA there
  plus a window), SEUL (GA on online/offline-selected spans), ETW (entropy-weighted forget loss).

## 4. Selectivity cross-walk (what each algorithm conditions on; *D-reading* from §2–3)
| method | conditioning of target / edit | tokens supervised | mechanism expected to keep **Bob→Paris** | to keep the word **Paris** | to keep **other Alice facts** | observable used by authors |
|---|---|---|---|---|---|---|
| MDU | prompt masked, same y_t | all masked answer pos. | retain SFT only | anchor = population prior (Paris kept if frequent) | not targeted | RWKU Neighbor, TOFU retain |
| EMDU | prompt + answer masked | all masked answer pos. | retain CE only | target state-independent → EOS | not targeted (prompt-level) | as MDU; % empty |
| TNT/TNFF | same context, token zeroed | annotated negatives (+ positives) | none beyond same-seq positives | the ban is context-specific in the objective; shared params → near token ban (closed study) | positives in same seq | — |
| UNDIAL | same context, −γ at label | all (opt. entity focus) | none | label-token logit shift at that context | none | — |
| Span-level (ACL'26) | grad contrast forget vs retain | self-consistent spans | via retain-contrast selection | unknown | unknown | 2 benchmarks |
| D2F | counterfactual object value | whole answer, ×4 on span | retain KL+NLL | positive keeps scaffold; span ×4 on A⁻ | prompt facts excluded by design | KFR/KRR, judges |
| CUSE | counterfactual input w/o evidence | on-policy tokens, sensitivity-weighted | retention targets | unknown | unknown | MLLMU-Bench |
| WHP | anchored-term translation | all non-repeated tokens | none (super-set forgotten) | generic labels | no (entity/book level) | familiarity |
| WHP+ | **subject name replaced** (avg N names) | all aligned doc tokens | — | **hard_retain related entities** | no (entity level) | WPU hard/general retain |
| ROME/MEMIT/AlphaEdit (+Patil) | **subject-keyed** MLP edit | none (closed form) | key separation (C⁻¹ / null space P) | lm_head untouched | "essence" KL; preserved keys | CounterFact neighborhood |
| TimeROME-DLM | subject keys at traced coordinate | none | ridge memory + sparsity | — | — | TOFU retain log-prob |
| TULIP | per-input hidden state ⟂ w_Paris | the answer token | retain hidden-state MSE | unembedding untouched | retain | TOFU/PISTOL/WMDP |

No inspected source tests the exact triple (forget s→o; keep s'→o; keep o as a word in other contexts; keep other
(s, r', o')). CounterFact neighbourhood (s'→o) and WPU hard_retain (related entities) are the closest existing splits.

## 5. Three sources most likely to be closest to an association-selective method
1. **WHP+ / causal_unlearn (CODE-READ).** Teacher = θ0's own distribution with the **subject replaced by other names**,
   averaged over N names, distilled at the same positions. For "Alice lives in ___" this is the generic-person prior,
   which keeps Paris at its population rate instead of banning it. Caveats: entity-level task, whole-document
   supervision, AR.
2. **The ROME family as unlearning** (Patil et al. objectives, AlphaEdit null-space, TimeROME-DLM as the MDM port):
   subject-keyed edits with an explicit Bob→Paris locality split (CounterFact neighbourhood). DLM instance SNIPPET-ONLY.
3. **CUSE (SNIPPET-ONLY)**, with code-verified neighbours **D2F** (counterfactual object, span-weighted DPO) and
   **TNT/TNFF** (same-context exclusion target). CUSE (counterfactual-input token sensitivity + on-policy
   self-distillation to a frozen privileged-context teacher) is the highest unresolved overlap for any
   "counterfactual-subject teacher on own samples" idea.

## 6. Unresolved (what would settle each)
- **CUSE full text** (main overlap risk): divergence, weight formula, privileged-context form, text-only use.
- **Span-level (ACL'26)**: loss and gradient-contrast formula. **TimeROME-DLM**: key token, delta target, sparsity,
  and whether the key at an MDM subject token is relation-conditioned.
- **TNT paper**: whether negatives are on-policy and refreshed. **UNDIAL paper**: intended tokens for TOFU-style QA.
- **D2F C.1/C.6** (span-redaction baselines, imperfect separation): in the mirror if needed. **WHP+ paper**:
  deconfounding derivation, TOFU N=1 results.
