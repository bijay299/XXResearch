# RUN_MANIFEST — CPR-GEM CPU-only correctness prototype

**Frozen at:** 2026-10-19T02:42Z (UTC, `date -u`). Nothing below was changed
after this timestamp except where an explicitly timestamped correction says so.

**Status at freeze:** declared, not yet executed. The suite is executed only
after this file exists.

---

## 0. Scope and what this cannot establish

This is a **model-free, CPU-only correctness prototype**. It executes small
synthetic tensor calculations and tiny analytic shared-parameter maps.

**Hypothesis under test (corrected form).** Directional repulsion can reduce
leading allowed-output drift at equal positive first-order target progress
**under favourable shared-parameter geometry**. It can **lose** when allowed-output
responses align such that native GEM benefits from cancellation.

**Unknown and not addressed here:** whether FLUX satisfies the favourable
conditions; whether this is a useful or novel method; anything about semantic
erasure, preservation, image quality or paper readiness.

**Not performed:** no FLUX, no diffusers/transformers/PEFT, no model calls, no
downloaded weights, no generated images, no GPU, no old baselines, no reviewer
judgements, no literature search.

---

## 1. Environment and source pins

| Item | Pin |
|---|---|
| Host | Hyperplane, Linux 5.15.0-136-generic, x86-64 |
| Python | 3.10.12 |
| PyTorch | 2.4.0+cu121 (**run CPU-only**, `CUDA_VISIBLE_DEVICES=""`) |
| Interpreter | `/data/bijaypandey/cuig_pilot/venv-cuig/bin/python` (pre-existing; **nothing installed**) |
| GEM reference source | `multimodal-ai-lab/GEM` @ `c328fc6bec687efe60223498d1b6bed2e5fbcffd` |
| Local GEM checkout (read-only) | `/data/bijaypandey/cuig_pilot/modern_object_assets_v1/GEM_src` (verified `git rev-parse HEAD` == the pin) |
| Loss block read | `gem/operators/erasure/erasure_gem.py`, lines 233–249 |
| Repo | `bijay299/XXResearch`, branch `research/audit-and-matched-effectiveness-protocol` |
| Repo HEAD at start | `d21eb2c8fbb17649be0d8a556f911200584413fe` (clean tree, matches coordinator-accepted SHA) |

The GEM operator/pipeline modules are **never imported**. The loss arithmetic
was transcribed **by reading** the pinned file. Attribution is carried in the
module docstring of `cpr_gem/losses.py`.

### Verified upstream semantics

`torch.nn.functional.pairwise_distance(x, y, p=2, eps=1e-6)` computes
`||x - y + eps*ones||_2` — an **all-ones shift**, *not* a centered smoothed
norm. Verified on this environment before freeze:
`pairwise_distance(0, 0) == 1.7320508e-06 == eps_n*sqrt(3)`.
This shift is preserved exactly.

---

## 2. Exact formulas

Let `v` = student, `a` = **detached** anchor teacher, `b` = **detached** target
teacher. All three are cast to FP32 and all nonbatch dims flattened, exactly as
native GEM does. Teachers are detached **before** every norm, contrast and
direction calculation.

**Frozen constants:** `eps_n = 1e-6`, `tau_L = 1e-6`, `gamma_scale = 100.0`,
`n_latents = 5` (**independently supplied; deliberately != batch size 8**).

### Native baseline (reference)
```
D_plus  = F.pairwise_distance(v, a, p=2)      # = ||v - a + eps_n*ones||
D_minus = F.pairwise_distance(v, b, p=2)      # = ||v - b + eps_n*ones||
loss    = torch.relu(D_plus - eta*D_minus).sum()
loss   *= gamma_scale / n_latents
```
Argument order, FP32, `torch.relu`, `.sum()` and the scaling are preserved.
`torch.relu`'s boundary derivative is zero; `clamp` is **not** substituted.

### CPR candidate — changes ONLY `D_minus`
```
L      = ||b - a||                      # per sample, plain unshifted norm
e_tau  = (b - a)/L   if L > tau_L   else exactly 0
alpha  = (b - v) . e_tau
loss   = torch.relu(D_plus - eta*alpha).sum() * gamma_scale / n_latents
```
A **safe denominator** (`where(live, L, 1)`) is applied **before** masking, so
no eager `0/0` is evaluated. Native attraction, hinge, reduction and multiplier
are unchanged. At `L <= tau_L` repulsion is disabled but **attraction remains**.
`tau_L` is a **frozen design choice, not an empirically validated scale**.
FP32 normalization/projector identities are **approximate**.

### Controls
| Control | Formula | Domain |
|---|---|---|
| Native GEM | above | all |
| CPR | above | all |
| Centered smooth radial | replace ONLY `D_minus` by `sqrt(\|\|v-b\|\|^2 + delta_d^2)`, `delta_d = eps_n*sqrt(d)` | all |
| Ordinary anchor regression | native loss at `eta = 0` | all |
| Axial margin | `relu(L - (1+eta)*alpha)` | `L > tau_L`; else `NOT_APPLICABLE` |
| Margin-truncated anchor | `relu(D_plus - k_t)` | all; `k_t` detached |

`delta_d = eps_n*sqrt(d)` matches native repulsion's **value** at `v = b`
(native gives `||eps_n*ones|| = eps_n*sqrt(d)`) while removing its all-ones
**directional** shift. It is a **diagnostic control**, neither native GEM nor a
complete causal isolation.

### Margin-truncated thresholds
- **Ideal (unshifted collinear reference only):** `k_t = eta*L/(1+eta)`, which
  places CPR's stopping point at `alpha = L/(1+eta)`. Verified **only inside
  that domain**; it is **not** an exact native match off-axis.
- **Stronger native collinear match (`L > tau_L`):** walk `v(t) = b - t*e_tau`,
  `t in [0, L]`, and root-find the **actual CPR hinge interior**
  `phi(t) = D_plus(v(t)) - eta*((b - v(t)) . e_tau)` using the **same numerical
  direction and precision** (FP32, native shifted `D_plus`).
  **Frozen bisection policy:** strict bracket `phi(0) > 0 > phi(L)` required;
  `max_iter = 200`; `xtol = 1e-12`; accept only if
  `|residual| <= 1e-5` (`ROOT_RESIDUAL_TOL`). Then `k_t = eta*((b - v(t*)) . e_tau)`,
  **detached**. Otherwise report `NOT_BRACKETED` or `ROOT_UNRESOLVED`.
  This scalar calculation **defines a control; it is not a parameter search**.

### Rejection, not repair
Invalid shapes, nonfinite inputs and invalid divisors raise `ContractViolation`.
No silent repair. The FP64 oracle (`cpr_gem/oracle.py`) is kept **separate**
from the FP32 losses; **no naive double-precision gradcheck** is applied to a
function that forces FP32.

---

## 3. Corrected matched-first-order-progress comparison

At one common detached state/parameter point, with **compatible** kernel blocks
**derived from one common `(J_c, J_a, P)`** (never chosen independently):

```
K_cc = J_c P^-1 J_c^T          K_ac = J_a P^-1 J_c^T
g    = g_parallel*(e + rho*u)
k    = e^T K_cc e              gamma_cross = (e^T K_cc u)/k
z_e  = K_ac e                  z_u = K_ac u
A    = ||z_e||^2   B = z_e . z_u   C = ||z_u||^2

Delta_psi    = -p/[k*(1+rho*gamma_cross)] * P^-1 J_c^T (e + rho*u)
D_allow^(2)  = p^2/(2*k^2) * F(rho)
F(rho)       = (A + 2*B*rho + C*rho^2)/(1 + rho*gamma_cross)^2
```

The **`B` alignment term is essential**; sensitivity norms alone do not
determine the ordering.

**Common frame.** `e = e_tau = (1,0)`. The perpendicular `u` is the **ACTUAL**
transverse direction of the native gradient (`w + epsilon_perp`), **never a
silently idealized `w`**, and is shared by all arms.

**Frozen positive safety margins and locality bound:**

| Quantity | Floor | Rejection code |
|---|---|---|
| `\|g_parallel\|` | `1e-3` | `INACTIVE_OR_WEAK_GRADIENT` |
| `k` | `1e-6` | `REJECT_SMALL_K` |
| `1 + rho*gamma_cross` (positive side) | `1e-2` | `REJECT_NEAR_SINGULAR_DENOMINATOR` |
| `1 + rho*gamma_cross <= 0` | — | `POSITIVE_PROGRESS_NOT_FEASIBLE` |
| `\|\|Delta_psi\|\|` | `<= 1e-1` | `REJECT_STEP_TOO_LARGE` |
| transverse norm | `1e-9` | labelled `ZERO_TRANSVERSE` |

Infeasible matching is **rejected**. The gradient is never changed, no negative
step is taken, and no tiny denominator is divided by.

**Caveats.** This normalized step is an **analytical diagnostic, not a
replacement for native AdamW**. Finite nonlinear target progress differs from
`p` by a remainder. **Equal leading drift does not mean equal cumulative model
behaviour.**

---

## 4. Frozen fixtures, seeds, step sizes

**Caps:** <= 256 declared configurations (**actual: 92**), batch size 8,
flattened loss dimension 65,536, shared-map output dims <= 8, parameters <= 64,
LoRA rank <= 2. **No random search, no adaptive sweep.**

- **Seeds:** `20261018`, `20261019`, `20261020` (fixed `torch.Generator`).
- **Large fixtures:** `(8, 65536)` FP32; modes `generic` / `active` (`v = a + 0.05*z`)
  / `inactive` (`v = b + 0.05*z`).
- **Principal fixture:** fresh `d = 2`, `a = (0,0)`, `b = v = (1,0)`, `eta = 1/2`,
  `P = I`, `v(psi) = b + J_c psi`, `r(psi) = r0 + J_a psi`. **Gradients are
  obtained from the implemented losses**, not written by hand.
- **`eta`:** `1/2` for all principal paired coupling fixtures. `0.0`, `1.0`,
  `2.0` appear **only** in declared boundary/domain checks.
- **Common first-order target progress:** `p = 1e-3`.
- **Nonlinear remainder step sizes (frozen, shrinking):** `1e-2, 5e-3, 2.5e-3, 1.25e-3`;
  declared remainder constant `M = 2.0`.
- **LoRA toy:** rank 2, `in = 4`, `out = 3`; `A` seeded Kaiming (**nonzero**),
  `B` zeros — the **actual** LoRA initialization. Initializing **both** factors
  to zero is explicitly **not** the candidate. No PEFT, no AdamW trajectory.

### Shared-parameter maps (all `P = I`, `e = (1,0)`)

| Map | `J_c` | `J_a` | Expected outcome |
|---|---|---|---|
| `favorable` | `I` | `I` | matched-progress identity holds; **CPR leading drift LOWER** |
| `adverse_alignment` | `I` | `[1, 1]` | `B*rho < 0` cancellation; **NATIVE GEM leading drift LOWER** |
| `progress_rejection` | `[[1,0],[2,1]]` | `I` | native denominator **negative** → `POSITIVE_PROGRESS_NOT_FEASIBLE` |
| `near_singular` | `[[1,0],[1.8266,1]]` | `I` | native denominator `~1e-3` → `REJECT_NEAR_SINGULAR_DENOMINATOR` |
| `no_leading_advantage` | `I` | `[1, 0]` | `K_ac u = 0` **and** `gamma_cross = 0` → **EQUAL** leading drift |

`adverse_alignment` is a **required main-table row**. An adverse outcome is a
**passing check of a limitation**, not a reason to redesign CPR.

### Numerical tolerances (frozen)

| Comparison | Tolerance | Rationale |
|---|---|---|
| FP32 loss/grad vs FP64 oracle, `d = 65536` | rel `1e-4` | FP32 accumulation over 65k dims |
| FP32 vs FP64, small `d` | rel `1e-6` | FP32 epsilon |
| FP64 central differences vs FP64 analytic | rel `1e-7` | `h = 1e-6`; usable FP64 FD floor ~`1e-9` |
| matched-progress identity (`p`) | abs `1e-15` | FP64 identity |
| `F(rho)` drift vs `\|\|J_a Delta_psi\|\|^2/2` | rel `1e-10` | FP64 identity |
| bisection root residual | abs `1e-5` | FP32 hinge evaluation |

### Expected signs / rejections (declared before execution)

- `native_gem` at `v = b`: `g ≈ (0.6464, -0.3536)`, `g_parallel > 0`,
  `rho ≈ +0.5469` with `u = (0,-1)` — a **real** transverse component produced
  by the all-ones `eps` shift.
- `cpr` at `v = b`: `g ≈ (1.5, 1e-6)`, `rho ≈ 0` (not exactly 0; reported as measured).
- `centered` at `v = b`: `g ≈ (1.0, 1e-6)` — repulsion contributes **no**
  direction there, confirming value-match without directional shift.
- `favorable`: `F_cpr = A = 1 < F_native = 1 + rho^2 ≈ 1.299`.
- `adverse_alignment`: `F_native = (1-rho)^2 ≈ 0.205 < F_cpr = 1`.
- `no_leading_advantage`: `F(rho) ≡ A = 1` for every `rho` → **EQUAL**.
- Hinge-inactive cases must give loss `0` and gradient `0`.
- Degenerate `L <= tau_L` must give `alpha = 0`, `e_tau = 0`, finite outputs and
  gradients, with attraction intact.

---

## 5. Budget and enforcement controls

| Cap | Value | Enforcement actually in force |
|---|---|---|
| Elapsed | 6 h total; science stops by hour 5 | operator clock + ledger `elapsed_s` |
| CPU | 12 cumulative core-hours | `ulimit -t 43200` (shell rlimit) + in-process watchdog |
| Processes | one test process | suite is single-process |
| Threads | <= 4 | `torch.set_num_threads(4)`, `OMP/MKL/OPENBLAS_NUM_THREADS=4` |
| Resident memory | 8 GiB process tree | `ulimit -v 8388608` + in-process RSS watchdog |
| New disk | 10 GiB | **nothing installed**; artifacts are small text/JSON |
| GPU | zero | `CUDA_VISIBLE_DEVICES=""` asserted at process start (hard exit 2 otherwise); CPU tensors asserted; no CUDA API calls; **no GPU-idleness polling** |

**Three real watchdog layers, all unprivileged:**
1. `ulimit -v 8388608`, `ulimit -t 43200`, `ulimit -c 0` in `run_suite.sh`
   (verified to work with torch before freeze).
2. External `/usr/bin/timeout --signal=TERM --kill-after=30 1800`.
3. In-process `cpr_gem.ledger.Watchdog` thread polling wall/CPU/RSS every 2 s,
   `os._exit(97)` on breach.

**Append-only resource ledger:** `artifacts/resource_ledger.jsonl`, one JSON
line per event, covering elapsed time, process-tree CPU seconds, RSS, peak RSS,
each declared case, group boundaries, attempts and stop events.

**Disclosed enforcement gap:** `getrusage(RUSAGE_CHILDREN)` only accounts for
*reaped* children. This suite runs a single process, so the self figures are the
operative ones; no gap is material here, but the limitation is stated rather
than papered over.

**Repair policy:** at most **two** focused implementation-repair/retest cycles
after the initial suite, within the same caps. Failures and diffs are preserved.
**Fixtures, `eta`, `epsilon` and tolerances are not changed to manufacture
superiority.** Any legitimate contract correction is timestamped at the time it
is made and never backdated.

---

## 6. Required frozen checks (92 declared configurations)

| Group | Count | Content |
|---|---|---|
| A | 46 | source-faithful native forward parity; candidate/control autograd vs independent FP64 analytic agreement in smooth active regions; teacher detachment; FP32 promotion; shape/batch handling; independent divisor; scaling and reduction; rejection of bad shapes/nonfinite/invalid divisor |
| B | 3 | FP64 central finite differences away from hinges and norm cusps, with disclosed oracle precision |
| C | 6 | active / inactive / exactly-zero hinge; native shifted-norm minimum; `v = b`; idealized `v = a` within its domain |
| D | 8 | `L = 0`, below / at / above `tau_L`; finite fallback outputs and gradients; `epsilon` parallel, antiparallel, perpendicular to the contrast |
| E | 8 | centered-control value match at `v = b`; anchor regression == native at `eta = 0`; axial margin in-domain and `NOT_APPLICABLE`; ideal `k_t` and the CPR stopping-point identity; bisection-matched `k_t` with residual; `NOT_BRACKETED` |
| F | 15 | five shared-parameter maps x three arms: matched-progress identity, predicted vs observed drift, alignment, denominators, step norms, explicit rejection reasons |
| G | 4 | one small compatible nonlinear map with explicitly bounded quadratic perturbation; frozen shrinking steps; target-progress and allowed-drift remainder bounds; FP64 mathematics separated from FP32 accuracy |
| H | 2 | tiny factorized LoRA toy: zero effective adapter update, nonzero relevant first-order Jacobian |

**Every declared configuration must appear in the ledger and in
`artifacts/results.json`.** Missing or undeclared cases fail the completeness
gate.

Recorded per case: errors, actual and first-order progress, gradient ratios,
alignment, denominators, step norms, predicted and observed drift, and explicit
rejection reasons.

---

## 7. Verdict rule (frozen)

- `CPU_PROTOTYPE_VERIFIED` — **only** if every declared configuration is present
  and meets its declared correctness criterion, **including the adverse
  behaviour**.
- `CPU_PROTOTYPE_INCOMPLETE` — completeness or some checks unresolved.
- `CPU_PROTOTYPE_FAILED` — declared criteria violated.

**Correctness is separated from whether controls reproduce the advantage.**
CPR need not beat every control; **control-equivalent behaviour is an
informative result**. Passing establishes **executable conditional
mathematics** — not novelty, not semantic erasure or preservation, not FLUX
improvement, not paper readiness.

---

## 8. Reproduction

```bash
cd results/cpr_gem_cpu_prototype/code && ./run_suite.sh
```

## 9. Gemini note

**Not supplied.** No Gemini note was provided with this task, and none exists in
this repository or on this host. The only reference found is a negative one, in
`results/modern_flux_dp_uce/sequential_diagnostic_protocol/SOURCE_AND_ASSUMPTION_LEDGER.md`,
stating that no claim is made that Gemini's proposed statistical protocol was
evaluated or used. Coordinator-side Mac paths were **not** assumed to exist on
Hyperplane, and this task was **not** blocked on the note. Nothing in this
prototype depends on it.
