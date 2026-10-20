# E — check of THEORY.tex (Lemma 1, Prop 1, Prop 2, Remarks) and of SPEC v1.1 §2/§5 against it
**Exact check:** `E_scripts/prop2_check.py` → `prop2_check_out.json`, `prop2_check_stdout.txt` (ledger agent E, 4.5 s).
- Setup: L=4, |V|=5, exact DP of the pinned `generate` semantics (lc tie-break = lower position). 2 worlds (m=1, m=2)
  × 11 configs, including semi-AR b∈{1,2}, k_n=0 steps ((8,4,rnd,1) and (8,2,lc,0)), lc and rnd, T∈{0,.5,1}.
- **ALL_PASS.** For random p∈P_S: P(Λ) ≤ e, and P(σ_S<∞) = e exactly, so the law up to σ_S agrees. P^{p_δ}(Λ) = e
  (δ=1e-3, to 1e-9), and P^{r̄}(Λ) = 0.
- (b) With S = visitation of r̄: e = 0, and every p∈P_S visits exactly r̄'s states.
- (c) For S_TF with (4,4,rnd,T): e = 0.908 / 0.883 / 0.875 at T = 1 / .5 / 0 (≥ 1−1/L = 0.75) for m=1, and e = 1 for m=2.
- **ε>0:** the supremum is not e. It is P^{r̄_ε}(leak before escape, or escape): 0.9163 vs e = 0.9155 (m1, rnd, T=1,
  ε=.05); matched exactly.

## 1. Prop 2: verdict CORRECT
- The upper bound, the supremum, the infimum, and parts (b) and (c) all hold.
- "Same law up to σ_S" is right: the draws and the lc confidences p(x⁰_i|s_n,i), i∈E_n, use only the conditionals at
  the current state; the noise, k_n and the tie-break do not depend on the model.
- Edge cases are fine. If k_n=0, E_n=∅ and the state is unchanged. A fully committed state is excluded by M≠∅. For
  T=0 the supremum is attained. Prop 2 needs no T≤1 and holds for every T≥0.
- One statement in (c) is wrong as worded, and is why e exceeds 1−1/L: under r̄ the trajectory also escapes earlier,
  whenever a non-fact position gets a token ≠ y⋆.

## 2. Required edits to THEORY.tex (old → new)
1. l.140 `\sigma_S=\min\{n:s_n\notin S,\ \M(s_n)\neq\emptyset\}` → `\sigma_S=\min\{n\le N-1:\ s_n\notin S,\ \M(s_n)\neq\emptyset\}`
2. l.152–153 "so the trajectory escapes $S_{\mathrm{TF}}$ at the first step that commits a fact position while masked positions remain." → "so the trajectory escapes $S_{\mathrm{TF}}$ no later than the state following the first commit at a fact position, unless no masked position remains (it escapes earlier whenever a non-fact position receives a token $\neq y^\star_j$)."
3. l.154 "irrespective of $T$." → "irrespective of $T$ and of $p_0$."
4. l.158 "On $\{n<\sigma_S\}$ all states lie in $S$, where $p=r$" → "For $n<\sigma_S$, either $s_n\in S$, where $p=r$, or $\M(s_n)=\emptyset$ and nothing is committed; at states in $S$"
5. l.159–160 "(the selection rule depends only on the conditionals at visited states and on fresh noise)" → "(the draws and the lc confidences $p(x^0_i\mid s_n,i)$, $i\in E_n$, depend only on the conditionals at the current state; the noise, the schedule $k_n$ and the tie-break are model-independent)"
6. l.170 "no escape requires every fact position to be committed at the final step" → "no escape requires every fact position to be committed at the final commit step (and every earlier commit to equal $y^\star$)"
7. After l.155, add item (d): "(d) For $\varepsilon>0$ and $\mathcal P^\varepsilon_S$ (conditionals $=r_\varepsilon$ on $S$), the same argument gives $\sup_{p}\Prob^p_\pi(\Lambda_\F)=\Prob^{\bar r_\varepsilon}_\pi(\tau<\sigma_S\ \text{or}\ \sigma_S<\infty)$, since leakage before escape is then possible." The method's default ε=1e-4 is covered only by this item.
8. l.177–179 "Part (b) is the DAgger-type fixed point: FX-Mix's rollouts are drawn from the \emph{current student}, which equals $\bar r$ only at convergence; a finite pool…" → "Part (b) is self-referential: for $p\in\mathcal P_S$ the laws agree up to $\sigma_S$, so $S$ contains every state $p$ visits under $\pi$ iff it contains every state $\bar r$ visits. A student that matches the target on its \emph{own} visitation therefore does not leak under $\pi$ (DAgger-type fixed point). FX-Mix approximates this with a finite, stale pool and $\varepsilon>0$; a finite pool…"
9. l.110 after "$\pi_T(\F\mid s,i)$." add "($h_n=0$ when $k_n=0$, in which case $E_n=\emptyset$.)" The rnd factor is 0/0 there.
10. l.62 "equivalently $\mathrm{softmax}(\zeta_0-\gamma\mathbf 1_\F)$" → "for $\varepsilon>0$ equivalently $\mathrm{softmax}(\zeta_0-\gamma\mathbf 1_\F)$". l.63 "(for computation use the log-space form" → "(for computation use fp32 logits and the log-space form". This matches SPEC §2.
11. l.36 "divided by $L$" → "divided by the SFT answer length (incl.\ EOS padding)". The SFT answer length is not gen_length.
12. l.98 "AdamW step with clipping $1.0$" → "AdamW (weight decay 0) step with clipping $1.0$" (SPEC §2).
13. l.128–129 "is an unbiased estimate of an upper bound" → "is an unbiased estimate (report its Monte-Carlo standard error) of an upper bound" (SPEC §5).
14. l.131 "can raise $p(\F)$ from $0.05$ to $0.99$" → "can raise $p(\F)$ from $0.05$ to $0.99$ (scale 2, unconditional fact mass $0.001$)". Also "\texttt{logits\_eos\_inf} breaks (c,e)" → "\texttt{logits\_eos\_inf} violates the sampler definition and breaks (c,e) ($\rho=0.5$, yet greedy commits $f$)".
15. l.133–134 "and up to $\approx16\times$ loose for lc at $T=1$" → "and $\approx9\times$ (eligible-sum form) to $16\times$ ($\rho$ form) loose for lc at $T=1$ after training".
16. l.192 "$\E_\mu[H]$" is undefined. Insert "where $H(s)=\sum_{i\in\M(s)}\min\{1,\rho(s,i)\}$, $d^{\mathrm{ff}}_\pi(s)=\sum_n\Prob(s_n=s,\tau\ge n)$". At l.194 add "($\alpha$ = TF fraction of $\mu$)".
17. l.198 "certifying $5\%$ leakage through Lemma~\ref{lem:T}(iv)" → "certifying $5\%$ leakage at $T=1$ through Lemma~\ref{lem:T}(iv) with $\varepsilon=0$".
18. l.202 "The target's own fixed point leaks $1-(1-\varepsilon)^{n}$" → "The target's own fixed point leaks at least $1-(1-\varepsilon)^{n}$". l.204 "has no finite stationary point under Adam (the $\F$ logits drift sub-linearly)" → "has no finite minimiser; in a 1-D Adam example the $\F$-logit offset keeps drifting sub-linearly ($-8.2,-15.8,-26.5$ after 5k/20k/80k steps)".
Lemma 1, Prop 1 and Remarks R-joint and R-SFT are otherwise transcribed correctly: signs, the kl(ε‖p0F) value, the
gradient (p0F−ε)₊(p0|F−p0|F^c), x₀(κ)=1−e^{−κ}, and (1−ε_t^m)/((1−ε_t)L).

## 3. SPEC v1.1 vs THEORY.tex
- §5 Prop 2 bullet: add "(stated for ε=0, i.e. r_0 = decode-time ban; for the default ε>0 use item (d))".
- §4 "decode-time F-ban on the base model (inference-time, FX-Mix's fixed point" → "… FX-Mix's ε=0 fixed point".
- §5 R2 "coverage/non-identifiability of TF-only measures in the zero-sharing class" → "coverage constant C ≤
  |d^ff_k|/(w_k(1−α))". Non-identifiability is now Prop 2.
- §2 "(no finite stationary point under Adam)" → "(no finite minimiser; drift under Adam)".
- §2 J-sampling note ("not an unbiased sample of the visitation measure"): optionally mirror it at THEORY l.91.
- Otherwise §2 and §5 agree with THEORY.tex: target and log-space form, TF_ff, truncation plus s_τ, the mean over
  M(s), no 1/t weight, "loss is not a bound", Prop 1 scope, and the certificate in deployment dtype and tie-break.
