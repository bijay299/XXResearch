# Preregistration precedence — what is verified, what is not, and the deviations

**Raised by:** external audit of snapshot `0b435427`, §7/§9.
**No preregistration has been edited.** This file records precedence and deviations *beside* the
plans, which stand exactly as written. That is the audit's instruction and it is the right one: a
plan amended after the fact is not a plan.

---

## 1. Precedence table

"Checked result" means the first committed artifact a reader can verify the plan against.

| prereg | plan first committed | checked result first committed | public commit order establishes precedence? | independent evidence |
|---|---|---|---|---|
| **H66** | 2026-10-23 16:17:56Z (+ mechanism addition 17:54:58Z) | 2026-10-23 18:20:11Z | **YES** | — |
| **H80** | 2026-10-24T18:41:37-04:00, commit `6caec48e` | **same commit** | **NO** | **see §2 — local mtime supports it** |
| **H81** | 2026-10-25 04:18:24Z | 2026-10-25 05:38:33Z | **YES** | — |
| **H82** | 2026-10-25 09:40:10Z | 2026-10-25 10:59:45Z | **YES** | — |
| **H84** | 2026-10-25 17:24:31Z | 2026-10-25 18:38:49Z | **YES** | — |
| **H85** | 2026-10-25T17:59:47-04:00, commit `ed1299f0` | **same commit** | **NO** | **see §2 — local mtime supports it** |
| **H90** | this remediation, before any v2 arm is built or attacked | not yet run | **YES, trivially** | the register names it as PENDING |

For H66, H81, H82 and H84 the plan commit precedes the checked result commit. That proves ordering
**against the committed result**, not against any unrecorded local observation — a weaker guarantee
than it looks, and stated as such.

## 2. The independent evidence I looked for, and found

The audit said: *"their asserted earlier local timing needs independent evidence, such as a
timestamped plan hash or job log."* There is **no** timestamped plan hash and **no** job log — those
were never produced, which is itself a process failure recorded in `RESOURCE_NOTE.md`'s family.

What does exist is **local filesystem mtimes** on the working copy:

| file | mtime | gap |
|---|---|---|
| `docs/PREREG_H80_meta_unlearning.md` | 2026-10-24 **15:59:25** −0400 | plan **32 minutes** before the first result |
| `outputs/meta_h80_meta_church_lam0.5.json` (earliest H80 artifact) | 2026-10-24 **16:31:11** −0400 | |
| `docs/PREREG_H85_cross_model.md` | 2026-10-25 **17:39:06** −0400 | plan **4 minutes** before the first result |
| `outputs/frp_frp_bksdm_church_r3_l05.json` (earliest cross-model artifact) | 2026-10-25 **17:43:17** −0400 | |

**This supports precedence in both cases**, and one detail strengthens it: mtime is *last*
modification. Because each plan's mtime is **earlier** than its first result, neither plan was
touched after its result existed — so the file was not edited to match an outcome.

**What it still does not establish**, stated plainly:

- mtime is **not tamper-evident**. It can be set arbitrarily, so it is weaker than a commit or a
  signed hash.
- It shows the *file* was not modified after the result; it does not prove the *content* was
  complete at creation.
- H85's **4-minute** gap is thin. It is consistent with writing the plan immediately before
  launching, which is what I believe happened, and also with writing it while the first build was
  already running.

**Therefore: H80 and H85 are labelled `PRECEDENCE SUPPORTED BY LOCAL MTIME ONLY, NOT PUBLICLY
VERIFIABLE`.** They are not claimed as preregistered on the same footing as H66/H81/H82/H84, and
nothing in report r5 leans on them being so. Going forward, each plan is committed and pushed in
its own commit **before** the run starts, so the question does not arise again.

## 3. Deviations, recorded separately rather than by amending plans

| prereg | commitment | what happened | status |
|---|---|---|---|
| **H80** | *"both concepts before any claim"* | only **church** was run | **UNMET.** `FINDING56`'s claims are church-only and are relabelled as such. Parachute was never built |
| **H80** | objective stated as the paper's | the implementation omits two of Eq. 8's three terms | **MISSTATED.** Corrected in `ERRATUM_meta_objective.md`; the plan is **not** rewritten |
| **H84** | one defence build per setting permitted *if clearly labelled* | single builds used and labelled | **COMPLIANT.** I previously described these as a protocol violation; that was wrong — H84's seed rule explicitly carries this exception. They are a **limitation**, not a breach |
| **H84** | inner process is "the actual full-UNet relearning attack" | it differed on seven axes | **MISSTATED.** `ERRATUM_utrap_inner_attacker.md`; plan not rewritten |
| **H85** | `PROTOCOL_FROZEN_v1`'s six-point grid | **three** attacks per arm, stopped on reaching the ceiling | **DEVIATION.** Defensible for counterexample-hunting — adding attacks cannot lower a maximum — but it is **not** compliance with an identical six-point comparative protocol, and `FINDING64` no longer claims it is |
| **H66** | λ predeclared before the run | honoured | **COMPLIANT** |
| **PROTOCOL_FROZEN_v1** | "unchanged throughout" | three-point subsets, extra attacks, new families, larger budgets all occurred | **NOT UNCHANGED.** The generated tables now separate fixed confirmatory grid / prespecified secondary / adaptive escalation, by budget, so the three are never pooled again |

## 4. Process changes adopted now

1. **Every new protocol is committed and pushed before the run starts, in a commit containing no
   results.** `PREREG_H90_utrap_v2.md` is the first under this rule.
2. Each plan states its **stopping rule**, **primary setting** and **falsifier** up front, and
   deviations go in a dated addendum section of *this* file, never into the plan body.
3. `PREREG_H90` additionally records the **matched-compute** definition before any arm is built, so
   the comparison cannot be rebalanced after seeing outcomes.
