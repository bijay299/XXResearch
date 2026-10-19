# Enforcement gaps — corrected current interpretation

*Amended 2026-10-19 under
[`AMENDMENT_01_EVALUATOR_FIDELITY_FREEZE_AND_ACCOUNTING.md`](AMENDMENT_01_EVALUATOR_FIDELITY_FREEZE_AND_ACCOUNTING.md).
Documentation only; `code/cprd/ledger.py` and every artifact are unchanged and no
mechanism was repaired. Attempt status: CLOSED —
`ENTRY_GATE_UNRESOLVED — NATIVE_DISPATCH_AGREEMENT_NOT_ESTABLISHED`.*

The original six-item text below is preserved verbatim. It listed real gaps but
**understated them**, and two of its sentences describe mechanisms that the
published source does not contain. Current interpretation (read from
`code/cprd/ledger.py` at the reviewed commit; no run was repeated):

1. **`settle()` can charge less than observed.** On an overrun it sets
   `charged = min(charged, reserved)` (`ledger.py:200`), contradicting its own
   docstring ("never charge less than observed"). It could silently under-charge an
   overrunning job. None of the three observed jobs overran (all below
   reservation), so no numeric under-charge is evidenced.
2. **Exception path without a confirmed reap.** In `supervise()`, an exception
   sends `SIGKILL` and the `finally` block settles without `wait()`/reaping, so the
   file's "charged through CONFIRMED TERMINATION, including exceptions" is not
   substantiated for that path (the timeout path does reap via `communicate()`).
3. **`reserve()` enforces only the GPU aggregate, the entry sub-cap and the
   protected evaluation reserve.** CPU core-hours, elapsed time, workers, RSS and
   storage appear only as constants in `CEILINGS` and are never compared with a
   measurement in the published source. Image and COCO counters add first and
   raise afterwards (detect, not prevent).
4. **No RSS/worker watchdog and no disk polling exist in the published source.**
   Original items 4 and 5 describe an "in-process watchdog of each child" and
   "polled" storage; those mechanisms are **not substantiated**. The statements that
   nothing is cgroup- or quota-enforced remain true.
5. **Original items 1–3 and 6 stand as written** (wall-clock × devices held; a
   dead parent over-charges at `reconcile_all`; unreaped-child CPU not counted;
   the 7-day cap is operator-enforced).

**What this does and does not change.** The three observed charges (58.685 s,
2.652 s, 106.541 s; total 167.8786 s = 0.04663 GPU-h on one device) are each below
their reservations and there are 0 open or unreconciled reservations. No numeric
CPU, storage, RSS, worker or elapsed breach is evidenced. But blanket
complete-enforcement assurances are **withdrawn**, and the CPU figure (207.19 s)
covers only the three supervised GPU children; whole-task CPU and peak storage are
unmeasured. The limitations are documented and **left unfixed**.

---

## HISTORICAL TEXT — original `ENFORCEMENT_GAPS.md` at `9940e94`, verbatim (superseded where it conflicts with the above)

---

# Actual enforcement gaps

Published verbatim from `cprd.ledger.gaps()`. These are the gaps that
really exist; no mechanism is claimed that is not implemented.

1. GPU cost is charged as wall-clock x devices HELD, not measured device utilisation, so allocated idle time inside a job is charged (intended) and another user's processes on the same device are invisible.
2. A child killed by the OS (OOM-killer, SIGKILL from outside) is charged by the parent's wall-clock measurement; if the PARENT itself dies, the reservation stays open until the next `reconcile_all`, which then charges the FULL reserved amount (deliberate over-charge, flagged in `unreconciled`).
3. CPU seconds are collected per supervised job via getrusage of the parent's reaped children; CPU burned by a process the parent did not reap is not counted.
4. RSS and worker-count ceilings are advisory: they are checked by the in-process watchdog of each child, not enforced by a cgroup.
5. Storage and the >=50 GiB free-space floor are polled, not quota-enforced.
6. The 7-day elapsed cap is operator-enforced against the ledger clock, not kernel-enforced.
