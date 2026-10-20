# Objection C → B (FCH-Mix): strongest objection + secondary

## Strongest: under shared parameters the edit is a learned token ban, so the state design gives no retention lever

**Claim.** B §2's "retention lever" (only F-logits move relative to θ0 at visited states) is a function-space
statement about the target `r_ε` at the trained states, not about the deployed parameters.
- With sharing, the F-logit drop demanded at forget states is carried mostly by parameters active wherever F
  is predicted: F's output row, biases, shared relation and [MASK]-position directions.
- The result is a soft **token ban** that lowers F at retained facts sharing the value token.
- FCH-Mix's mixture and truncation change *where* the target is enforced, not *which parameters carry it*.
  They are a forgetting-robustness lever, not a retention mechanism.
- B's §7 update is tabular (no sharing), so it cannot show this; its sharing proxy retainF already worsened (.835 vs .857).

**Derivation (linear readout / log-linear).**
- With `z_f(s) = θ·φ_f(s)`, the forget gradient at θ0 is `∂KL(r_ε‖p)/∂z_f = p0(f)(1−ε/p0(F)) > 0` on F.
- Under gradient flow from θ0, or an L2 prox, a logit drop δ at state s is realised as
  `Δθ = −δφ_f(s)/‖φ_f(s)‖²`.
- So at a retained state `s_R`: `Δz_f(s_R) = −Σ_s μ(s)δ(s)⟨φ_f(s), φ_f(s_R)⟩/‖φ_f(s)‖²` (damage = *feature overlap*).
- Every μ on the forget prompt's states shares the same bias, output-row and relation features, so
  reweighting μ (all FCH-Mix does) leaves the overlap nearly unchanged. Only the retain term pushes back,
  and only on the contexts it samples.
- LLaDA analogue: F's unembedding rows plus key directions shared by all masked positions. In C's toy,
  74–82% of forget-key energy lies on the shared top retain principal component.

**Exact check.** `C_scripts/objection_fchmix_tokenban.py` (+ `_out.json`, `_stdout.txt`; ledger agent C, 72 s).
- Model: A's log-linear shared mask predictor (`A_scripts/sharing_check.py`, imported read-only), with the
  same `r_ε` target, retain-KL and prox in every arm.
- Measures:
  - **TF** = fact-free TF states.
  - **OP** = the student's own fact-free native states, 6 configs, 3 DAgger rounds; this is FCH-Mix's measure.
  - **OP-g** = OP with the all-masked state's weight capped (OP gives it .32; TF gives .12).
- R3 = "When was Z born?" → "born 1951" (base recall .952). "Shared" = fraction of the 1951-logit change at
  R3's masked states coming from the global bias, R[when] and W[·,·,MASK].

| arm | λ | max leak | retain TV | R3 recall | Δz_1951 at R3 | shared |
|---|---|---|---|---|---|---|
| TF | 30 | .094 | .028 | .906 | −0.94 | 95% |
| TF | 100 | .059 | .038 | .875 | −1.21 | 96% |
| OP | 10 | .079 | .030 | .898 | −0.81 | 94% |
| OP | 30 | .038 | .040 | .870 | −1.05 | 95% |
| OP-g | 30 | .035 | .038 | .873 | −1.00 | 91% |

**Reading**
- **(i)** Shared parameters carry 91–96% of the damage in every measure. R3's own parameters compensate only
  +0.04 to +0.12.
- **(ii)** R3 recall follows one curve against leakage, whatever the measure: TF .875 vs OP ≈ .884
  (interpolated) at leakage ≈ .06; OP .870 vs OP-g .873 at ≈ .035–.038.
- **(iii) Withdrawn:** "on-policy concentrates on generic states" (capping the all-masked state leaves R3 unchanged).
- **(iv)** OP buys **feasibility**: TF never reaches ≤ .05 for λ ≤ 100 (best .059); OP reaches .038 at λ = 30.
- So B's primary claim, *retention* at matched config-robust forgetting, has no mechanism in the method
  beyond "TF cannot reach that level at all".

**Falsifier (GPU).**
- Report retained facts that **share a value token** with forgotten ones as a separate subgroup (TOFU years,
  cities, professions; 26 such facts on the substrate). Compare SC-TF, FCH-Mix and a retain-set-augmented
  arm at matched max-over-G leakage.
- Predicted: SC-TF ≈ FCH-Mix on that subgroup; if FCH-Mix is clearly better there, this objection fails.
- Caveat: A's model has explicit one-hot bias/relation rows; LLaDA's shared fraction is unknown but
  estimable forward-only (masked-key overlap, F rows of the unembedding).

## Secondary (one line each)
1. **Paraphrase amplification.** `r(v) = (1−ε)p0(v)/(1−p0(F))` multiplies every non-F token by about 100× at
   p0(F) = .99, including sub-token or paraphrase forms of the fact missing from F. The target promotes
   surface variants exactly at the most confident states; it needs an F-closure or a cap on the tail tilt.
2. **"C = ∞ for TF" is tabular.** Under sharing, the binding coefficient is the feature-level
   `λ_max(Σ_μ⁻¹Σ_π)`, which is finite when deployed-state features lie in the TF span (A's model: TF .059 vs
   OP .038). A forward-only key diagnostic should decide whether rollouts are worth paying for.
3. **Cost is unmeasured.** About 96 no-grad forwards per rollout (gen 128, 3 configs), an R-step-stale pool,
   short truncated pools early in training, and a certificate valid only at the θ where it was measured.
   The "1.75× MDU" figure rests on untested J = 8 and R = 4.
