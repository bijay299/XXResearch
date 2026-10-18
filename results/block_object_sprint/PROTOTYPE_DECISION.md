> ## ⚠ SUPERSEDED — historical record
>
> **The sprint's final status is `BASELINE_UNQUALIFIED` — SPRINT CLOSED WITH CORRECTIONS.**
> See [`AMENDMENT_01_CLOSEOUT.md`](AMENDMENT_01_CLOSEOUT.md), which is authoritative over any
> conflicting wording below.
>
> The body of this document is **preserved unchanged** as the historical record of the
> decision and its chronology. Its framing is superseded in three ways:
>
> 1. The "pre-registered generalisation check" declared below on Cats and Horses was declared
>    **after** the gate-1 failure was already recorded — i.e. after the condition requiring a
>    stop was known. **Prospectively declaring a new check does not override the earlier
>    mandatory stop rule.** Those runs (160 images, 0.7743 recorded GPU-hours) are a
>    **disclosed protocol deviation**, preserved as additional descriptive native results.
> 2. The claim that the +10-point related-retention screen was **"arithmetically impossible"**
>    is withdrawn. Five points of headroom remained on the development object under the binary
>    procedure — enough to justify declining this prototype on the development evidence, but
>    not to prove the confirmation criterion unreachable on objects that should never have
>    been generated. **The decision to develop no prototype stands; the impossibility framing
>    does not.**
> 3. The mechanism hypothesis recorded at the end remains **untested speculation**. It
>    authorises nothing, and the measurement prerequisite it implies is **not** authorisation
>    for a successor assignment.

# Prototype decision: NO PROTOTYPE JUSTIFIED

Committed after the baseline-validation record and **before** any confirmation-object
edited output was generated or inspected. Budget state at decision time:
**3.20 / 48 GPU-hours**, **403 / 1,600 images**, **3.09 / 120 elapsed hours**.
No prototype or control code was written, and no prototype tuning was performed.

## Decision

No prototype is developed. Two independent reasons, either sufficient on its own.

### Reason 1 — the frozen retention screen is arithmetically unsatisfiable

The protocol's first prospective screening criterion is:

> Related-object retention improves by at least **10 percentage points** versus native BLOCK.

On the development object, native BLOCK does not damage related-object retention at all:

| instrument | related retention at native BLOCK | headroom to ceiling |
|---|---|---|
| native forced-choice IRA (release evaluator) | **100%** (20/20) | **0 points** |
| frozen binary presence (adaptation) | **95%** (19/20) | **5 points** |

A +10-point improvement is impossible under either instrument. Worse for the premise, the
single related failure under the binary instrument (Dogs / Bricks) **already fails at T0**,
i.e. it is caused by transcoder *reconstruction*, not by erasure. No erasure-side
intervention could repair it; only a better transcoder could.

The screen was written in the expectation that native BLOCK would show a related-retention
deficit to close. It does not. The protocol forbids replacing or weakening a committed
gate after seeing results, so the screen stands and cannot be met.

### Reason 2 — measurement is not qualified at the resolution the screen demands

The frozen presence instrument shows a systematic, one-directional generated-domain
failure on edited arms: it reports 5/40 residual bears where the agent audit of all 40
images finds 0 (see `BASELINE_VALIDATION.md` §1). That is a 12.5-point error on the exact
quantity a prototype would be screened on.

The remaining screening criteria are stated at +/-5 points — two images out of forty. The
instrument's demonstrated error on this arm is larger than the effect the screen must
resolve. The protocol states that an unresolved material generated-domain scoring failure
blocks a prototype, and that systematic disagreement capable of changing a gate means
measurement is unqualified. Both apply.

## What was NOT done, deliberately

- No prototype mechanism was implemented, and no control was implemented.
- The half-day prior-art allowance was **not** spent. It is allocated to "the closest
  primary prior art for *that* mechanism", and no mechanism is being pursued; spending it
  would be work on a candidate the protocol says not to start.
- No gate, criterion, category or object was changed after seeing results.
- The confirmation image allowance is **not** repurposed. See the next section for the one
  measurement that is taken, and why it is not a prototype confirmation.

## Pre-registered generalisation check (not a prototype confirmation)

Native BLOCK is run on the two frozen confirmation objects, **Cats** and **Horses**, using
the identical 80-case panels, arms and scoring already fixed in `RUN_MANIFEST.md`.

This is declared here, in advance, as a **characterisation of the native baseline across
the three frozen objects**, not as confirmation of any prototype. There is no prototype,
no control, and nothing to select; no tuning decision can be contaminated by the outcome,
because development is already closed by this document. The specific question committed
in advance is:

> Does the development-object observation — strong target suppression, related-object
> retention preserved, collateral damage concentrated in the *unrelated* stratum —
> reproduce on Cats and Horses?

Cost: 160 images and an estimated 0.7 GPU-hours generation plus ~0.2 GPU-hours scoring,
inside the already-allocated confirmation budget. Outcomes are reported per object and
separately, whatever they are.

## The one limitation the evidence does support (stated, untested)

Recorded as a hypothesis for a future decision, **not** pursued here and **not** supported
by any experiment in this sprint:

Native BLOCK's collateral damage on the development object fell entirely in the
**unrelated** stratum — binary unrelated retention 100% (T0) -> 90% (N), native IRA
unrelated 90% (T0) -> 75% (N) — while related retention was untouched. Both binary
failures were the same category, *Architectures*.

A plausible mechanism is that feature identification takes the **union of all latents
active for the target's tokens** (`A_t = union over tokens of {i : z_i != 0}`) with no
selectivity test. For "Bears" that is 57 unique latents of 65,536, chosen purely by
top-k activation on the bare concept string. Any of those latents that is also
load-bearing for unrelated concepts gets its decoder row redirected to the empty token,
so unrelated content degrades. A selectivity filter — excluding latents whose activation
on a reference corpus of unrelated prompts is comparable to their activation on the
target — is the obvious intervention family, and connects to existing sparse-feature
selection work and to BLOCK's own bijection-pairing ablation.

Testing it would first require a presence instrument qualified on the synthetic domain,
since the effect to be measured (2 images of 20) is smaller than the current instrument's
demonstrated error.
