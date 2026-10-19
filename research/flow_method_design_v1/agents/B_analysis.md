# Agent B analysis: attraction/repulsion structure of relu(D+ - eta D-)

Reused (not redone): CPR algebra, centered radial control equivalence at v=b, eps_n all-ones shift artifact (FINAL_REPORT, AMENDMENT_01; read). CPR detector study is INCONCLUSIVE and untouched.
All numbers: `B_scripts/b_checks.py` -> `B_scripts/b_checks_output.json` (ledger `B_checks`, `B_checks_v2`). numpy only (torch absent).

## What I investigated
1. Zero set / spurious minimisers by eta (F1). eta<1 Apollonius ball (radius eta L/(1-eta^2), orthogonal slack up to eta L/sqrt(1-eta^2), exceeds L for eta>0.707), eta=1 half-space, eta>1 ball exterior. Upstream default eta=1.0 (VERIFIED c328fc6 erasure_gem.py:28).
2. eta=1 closed form: GEM = axial-margin x eccentricity gate (I1), gradient orthogonal to grad S (I2) => conservation of D+ + D-.
3. Free-output flow from offset start: planar, ends at midplane with r_end = sqrt(S0^2-L^2)/2 (T1); sqrt amplification of noise (slope 0.5003).
4. Counterexamples: anisotropic metric M, eta != 1, offset along axis (T10, T5).
5. Degeneracy catalogue below.

## Degeneracy catalogue (COMPUTED unless noted)
| id | degeneracy | evidence |
|---|---|---|
| D1 | v=b: direction set by eps_n*1 (native) -- CLOSED, not re-announced | AMENDMENT_01 |
| D2 | eta=1 flat tail: loss ~ L^2/(2r) as r->inf with no erasure; infimum 0 not attained | T7 |
| D3 | eta=1 axis ray beyond b: stationary plateau (gradient 0, loss L) | T5 theta=0 |
| D4 | zero set unbounded (eta>=1) / large ball (eta<1): collateral slack up to eta L/sqrt(1-eta^2) | T6 |
| D5 | hinge boundary: gradient jumps by |u+ - eta u-| (up to 1+eta; =0 only on the plateau) | T9 sign map |
| D6 | a=b (L=0): eta=1 loss identically 0, no anchoring; eta<1 reduces to (1-eta) anchor regression | T8 |
| D7 | small L: loss ~ L, gradient norm scale free => per-sample weight is 'normalised', loss weight proportional to L; with sum over samples, small-contrast samples contribute ~0 loss but O(1) gradient | T8 |
| D8 | Transverse sign: at eta=1 the whole active region repels transversally (descent increases r) | T9 |
| D9 | Noise amplification sqrt(L/(2 eps)) (H-B1), strongly metric dependent (r_end=4.86 L for stiff-transverse M) | T3, T10 |

## Discarded ideas and why
* **Projection into the span of contrasts across timesteps/latents (OC-GEM: GEM hinge on P_U coordinates + anchor on the complement).** Provably removes the orthogonal slack of D4 (minimiser set becomes k-dim Apollonius ball x {complement = 0}), but this is a change of variables plus a quadratic anchor: generic projection algebra, equivalent to anchor regression on the complement; and whether b-a's off-U component is concept-relevant cannot be determined without a real model. Discarded: no provable distinction that matters beyond D4.
* **Whitened/per-token metric.** GEM in whitened coordinates is GEM (change of variables); T10 shows M matters for dynamics but choosing M needs real-model Jacobian statistics. Not a candidate.
* **Capped repulsion relu(D+ - eta min(D-,L)).** Provably bounded zero set for all eta, equals GEM inside {D- <= L}; but flows started at b stay in that region (D- <= L at the stopping point since D- = S0/2 ~ L/2 at eta=1), so it changes nothing on the training path; matters only for overshoot. Subsumed by truncated-anchor control. Discarded.
* **Eta sweep as a knob.** eta<1 gives smaller transverse amplification (T10: 0.024 vs 0.071) but stops earlier (s_end=eta/(1+eta)=0.33 at eta=.5); this is a trade-off with no new mechanism.
* **Radial vs axial repulsion again.** Overlaps CPR; only kept as H-B1 because its mechanism (finite-offset noise amplification, I2 conservation) is not the closed eps_n/leading-drift analysis and also applies to the centered radial control. It is NOT claimed to make CPR better; the sign of the net effect on a real LoRA student depends on unmeasured M.

## Honest status
Nothing here is a new loss. I1/I2/T1 are exact elementary conic geometry; novelty UNVERIFIED (no literature search possible). Real-model relevance UNKNOWN. A reasonable use for the sprint: include H-B1 as one discriminating test between radial and axial repulsion with offset eps>0 and anisotropic M, and mention F1/D2 (eta=1 default has an unbounded zero set) as a limitation of native GEM.
