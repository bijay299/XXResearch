# RESULTS — development vs locked final checks, outputs, failures and limitations

Everything here is CPU evidence on exact tabular models or a tiny synthetic masked-diffusion transformer with fresh
fictional facts. **Nothing here is evidence about LLaDA-8B.** Development numbers (§2) were seen while designing and
tuning; only §3 (locked, run by the independent verifier G after the lock) is final validation.

## 1. What is measured
- **Forgetting = reduced recall in natively generated answers** under the pinned LLaDA sampler semantics: `leak_any`
  (any value token of the fact at any answer position, incl. after EOS), `leak_full` (exact value span). Primary
  statistic: max over held-out configs H1–H5 (never used for training states); H6 (4 tokens/step) reported separately.
- **Not success by degeneration:** forget-answer well-formedness `F_wf` (one of the trained phrasings with the right
  subject/attribute and a value of the right type) is reported for every arm; the retrain-without-forget model is the
  reference (fluent hallucinations).
- **Retention:** mean retained-fact accuracy over the 9 configs (R), shared-value-token slice (S), copy accuracy.
- Toy configs (steps, block, remasking, T): pool P1 (8,8,lc,0), P2 (8,8,rnd,1), P3 (4,8,rnd,1); held-out H1 (8,2,lc,0),
  H2 (4,8,lc,0.5), H3 (8,8,rnd,0), H4 (8,8,lc,1), H5 (4,4,rnd,0.7), H6 (2,8,rnd,1).

## 2. Development evidence (pre-lock; used for design and tuning)

### 2.1 Exact tabular studies (no parameter sharing unless stated)
| finding | numbers | source |
|---|---|---|
| MDU's prompt-masked anchor still predicts the fact when the answer restates the subject | anchor p(fact) 0.21 (all-mask) → 0.44–0.47 once the subject is visible (reference 0.88–0.92); MDU τ=1 target model leaks 1.00 under greedy semi-AR, 0.41–0.57 under sampling | `agents/A_scripts/tabular_check_out.json` |
| Exact exclusion target imposed only on teacher-forced fact-free states still leaks; on all fact-free states it does not | 0.25 / 0.12 / 0.06 (C2/C5/C4) vs 0.000 under all six configs | same |
| TF-only states cannot reach config-robust forgetting in the zero-sharing class; a three-config on-policy pool can | TF + I-proj target plateau leak_max 0.496; single greedy rollout config 1.000; pool max 0.011 over 33 configs (30 held out) | `agents/B_scripts/tabular_commit_hazard_out.json` |
| A's original pool fails on held-out random-order parallel configs | in-pool 0.0004 vs held-out 0.20 | `agents/B_scripts/objection_A_coverage_out.json` |
| MSDA (C) with MDU's anchor loss cannot get below the anchor's fact mass | p_fact 0.328 vs anchor 0.325; max leak 1.00 at every λ; with an exclusion target 0.034 at retain TV 1e-4 (best case) | `agents/A_scripts/objection_msda_anchor_out.json` |
| Under shared parameters the exclusion edit is carried 91–96 % by shared parameters; retained shared-token recall follows one curve vs leakage for TF and on-policy states | R3 recall ≈ .875–.884 at leak ≈ .06 for all state measures; TF best .059 vs on-policy .038 | `agents/C_scripts/objection_fchmix_tokenban_out.json` |
| Proposition 2 (escape characterisation) holds exactly | 2 worlds × 11 configs, ALL_PASS; e = 0.908/0.883/0.875 (m=1) and 1 (m=2) for TF sets | `agents/E_scripts/prop2_check_out.json` |
| Counterexamples to earlier claims (A's KL→leak for 0<T<1; B's coverage constant; ε-floor leak) | e.g. π_0.5(F) = 0.47 > 0.40; true leak 0.81 vs claimed bound 0.053; floor 12 % at n = 128 for ε = 1e-3 | `agents/E_scripts/counterexamples_out.json` |

### 2.2 Coordinator probe on the neural toy (LoRA r=4, lr 2e-3, + retain SFT)
GD, NPO, MDU τ∈{0,1} all reach exact-value leakage 0 within 50 steps, but **every one produces 0 % well-formed forget
answers** (retrain reference: ≈100 % under D1–D3); MDU τ=1 keeps emitting value tokens (leak_any 0.10–0.54 vs the world-0
retrain reference's 0.08–0.13 per config; the retrain level is world-dependent, 0.17–0.26 held-out max in other worlds). `results/dev/coord_probe/`.

### 2.3 Neural-toy dev comparison (worlds 0 and 1, one unlearning seed per arm and LR)
Rule as later locked (VALIDATION_SPEC §5). Cut A = lowest held-out (H1–H5) max leak_any among checkpoints with
R ≥ R_base − 0.05 and copy ≥ base − 0.02; world 0: LR {5e-4,1e-3,2e-3} × 200 steps; world 1: LR {1e-3,2e-3,4e-3} × 300 steps.

| arm | w0 Cut A leak (R, F_wf) | w1 Cut A leak (R, F_wf) |
|---|---|---|
| FX-Mix | 0.036 (0.906, 0.37) | 0.021 (0.901, 0.29) |
| SC-TF | 0.047 (0.932, 0.37) | 0.016 (0.911, 0.37) |
| SC (MDU rollout flag) | 0.198 (0.925, 0.43) | 0.031 (0.899, 0.30) |
| FX-Mix-greedy | 0.010 (0.919, 0.39) | 0.010 (0.902, 0.34) |
| SC-TF-cm | 0.042 (0.929, 0.36) | 0.021 (0.917, 0.35) |
| EMDU | not run | 0.000 (0.911, **0.00**) |
| NPO | 0.016 (0.917, **0.00**) | 0.167 (0.890, **0.00**) |
| MDU τ=1 | INFEASIBLE | INFEASIBLE |
| *references* | base 1.00 / retrain 0.130 (F_wf 0.69) / decode-time ban 0.00 (F_wf 0.35) | base 1.00 / retrain 0.260 (F_wf 0.67) |

Dev reading (recorded before the lock, DECISION_LOG #17, #19): Δ_A(SC-TF) = +0.011 (w0), −0.005 (w1), far below the
0.03 margin later locked for R1; every arm sharing the exclusion target sits at its decode-time-ban fluency (≈0.3–0.4
mean F_wf; 0.58 under greedy vs retrain 1.00), while NPO, MDU and EMDU forget by degeneration (F_wf 0). No rescue pivot
was made. Files: `results/dev/compare_w{0,1}/`, `results/dev/DEV_LOG.md`.

## 3. Locked final validation (independent verifier G; worlds 2–4 × unlearning seeds 0–1)
Lock commit `13ec821` (`VALIDATION_SPEC.md`, `configs/LOCK.json`); run by G 05:36–06:25Z: 96/96 runs, 48/48 units, no
crash, no repair, R5 sanity passed in all three worlds. G re-implemented the §5 rules independently and reproduced every
selection and rule outcome. Raw outputs: `results/final/locked/`; report: `results/final/G_REPORT.md`.

**Decision (mechanical): `NO_DISTINCT_METHOD_AFTER_CONSTRUCTION`** — R1 failed.

| rule | result |
|---|---|
| R1 vs SC-TF | Δ_A per unit −.005, +.005, +.016, 0, +.010, +.016; **mean +0.0069** (needed ≥ 0.03); positive 4/6 (needed 5/6) → **FAIL** |
| R1 vs SC (MDU rollout flag) | mean +0.069, 6/6 → pass |
| R1 Cut-B condition | no infeasible Cut B for FX-Mix or SC-TF → pass |
| R2 (descriptive) | mean Δ_A(FX-Mix-greedy) = −0.009 (greedy-only pool slightly better than FX-Mix) |
| R2b (secondary, decision-irrelevant) | FX-Mix-greedy vs SC-TF mean +0.016 (6/6 positive, < 0.03) → FAIL; vs SC +0.078 → pass |
| R3 (mechanism) | 86 % of SC-TF's residual first-commit decision states (H1–H5, at Cut A) lie off the teacher-forced manifold (FX-Mix 82 %) → a coverage gap exists, yet SC-TF suppresses there as well as FX-Mix |
| R4 (degeneration) | mean forget well-formedness at Cut A: FX-Mix 0.359 vs NPO 0.000, MDU τ=1 0.000, EMDU 0.000 (gap 0.36 ≥ 0.2); SC-TF 0.348 and SC-TF-cm 0.361 are the same ⇒ a property of the exclusion target, not of FX-Mix; all far below retrain (0.66–0.69) and at the decode-time-ban level (0.33–0.36) |

Means over the 6 units at Cut A (held-out H1–H5 max leak_any / mean retained acc R / shared-token slice S / forget
well-formedness / H6 extrapolation leak), and Cut B retention (R at held-out leak ≤ the world's retrain level):

| arm | Cut A leak | R | S | F_wf | H6 | Cut B R | row-forwards/step |
|---|---|---|---|---|---|---|---|
| FX-Mix | 0.027 [0.016–0.036] | 0.896 | 0.824 | 0.36 | 0.051 | 0.905 | 136 |
| SC-TF | 0.034 [0.021–0.052] | 0.910 | 0.851 | 0.35 | 0.055 | 0.919 | 96 |
| SC-TF-cm (compute-matched) | 0.027 [0.021–0.036] | 0.911 | 0.852 | 0.36 | 0.044 | 0.915 | 136 |
| FX-Mix-greedy | 0.018 [0.010–0.031] | 0.907 | 0.846 | 0.35 | 0.046 | 0.910 | ≈144 |
| SC (MDU rollout flag) | 0.096 [0.052–0.120] | 0.910 | 0.847 | 0.39 | 0.240 | 0.916 | ≈226 |
| EMDU | 0.000 | 0.928 | 0.896 | **0.00** | 0.007 | 0.933 | 96 |
| NPO | 0.039 [0.000–0.167] | 0.919 | 0.873 | **0.00** | 0.030 | 0.924 | 96 |
| MDU τ=1 | 0.054 (3/6 feasible) | 0.888 | 0.811 | **0.00** | 0.017 | 0.874 | 96 |
| *references* | base 1.00; retrain 0.250 / 0.234 / 0.167 (w2/w3/w4), F_wf 0.66–0.69; decode-time F-ban F_wf 0.33–0.36 | | | | | | |

Reading:
1. The candidate's added operation (multi-config on-policy states with first-commit truncation, fact-free TF states)
   is **not supported**: the same target on plain random teacher-forced states (SC-TF) is within ≈1.3 forget samples of
   FX-Mix on held-out leakage, and compute-matched SC-TF equals it at higher retention. FX-Mix's edge over SC-TF is
   consistent with extra forget pressure (D's alternative explanation E2; SC-TF-cm Δ_A mean 0.000, 2/6 positive), and its retention is lower at matched leakage
   (Cut B 0.905 vs 0.919; shared-token slice 0.832 vs 0.869 at Cut B).
2. FX-Mix does beat the naive on-policy combination (MDU's single-greedy, random-stop rollout flag, which trains
   post-leak states and no TF states). Which difference causes that gain is not identified (no truncation or TF_ff
   ablation was locked), and plain SC-TF beats SC by a similar margin (0.034 vs 0.096).
3. EMDU attains the lowest leakage and the best retention in this toy, but by emitting no well-formed forget answer
   (consistent with its paper's EOS-only mode); NPO and MDU also forget by degeneration. Arms with the exclusion target
   are the only ones that keep partly well-formed answers (0.35–0.39 across SC-TF, FX-Mix-greedy, FX-Mix, SC-TF-cm and SC), still well below the retrain model.
4. Measurement granularity: one forget sample = 1/192 (stochastic configs), one prompt = 1/24 (greedy configs); the
   SC-TF difference is ≈1.3 samples.
5. Procedural issues (none affect a number): a post-lock edit of `experiments/run_locked.py` (bit-identical re-run;
   restored; DECISION_LOG #20); the locked §7 commands lacked `--label` (G added it); the decisive summariser has no
   unit test (G's independent recomputation matched). See `SPEC_ERRATA.md`.

## 4. Cost (measured CPU; LLaDA projections are arithmetic, not measurements)
Measured on the toy (`results/dev/COST.md`): per step SC-TF 96 row-forwards (64 student fwd + bwd, 32 reference); FX-Mix
136 (+40 no-grad rollout forwards, +42 %), SC 226 (MDU's per-step rollout). Projected for LLaDA-8B (F, `COST.md` §2):
FX-Mix ≈ 7.4e14 FLOPs/step vs SC-TF ≈ 1.8e14 (≈ 4.1×; rollouts dominate), LoRA r=16 optimiser state ≈ 0.67 GB, reference
via adapter-disable (no second copy), exclusion target ≈ 2–3 GB transient fp32 over (masked positions × 126,464).

## 5. Failures, deviations and limitations
- Container restart at ~05:14Z killed the dev world-1 shards (5/24 complete); only missing runs were relaunched.
- Post-lock edit of the locked runner by the implementation engineer (no numeric effect; restored) — DECISION_LOG #20.
- G ran up to 4 concurrent 1-thread processes (VALIDATION_SPEC §7) while G_BRIEF said ≤ 3; one 2-s audit script made 5.
- A thread-cap breach (3 concurrent threads for ≈1.5 min) during F's tests (DEV_LOG 04:50–04:56).
- THEORY.tex could not be compiled (no LaTeX toolchain obtainable offline); structural lint only (balanced environments,
  braces, math delimiters, labels).
- arxiv/HF/mlr.press blocked: LLaDA's modelling code (module names) and several papers were not inspectable (PRIOR_ART §1).
- The toy's exclusion-target fluency is limited by the reference's tail (decode-time ban 58 % well-formed under greedy);
  whether a pretrained 8B model's tail behaves better is unknown.
- Token-level F; leakage through unlisted surface forms is not measured in the toy (values are single symbols).
