# Actual enforcement gaps

Published verbatim from `cprd.ledger.gaps()`. These are the gaps that
really exist; no mechanism is claimed that is not implemented.

1. GPU cost is charged as wall-clock x devices HELD, not measured device utilisation, so allocated idle time inside a job is charged (intended) and another user's processes on the same device are invisible.
2. A child killed by the OS (OOM-killer, SIGKILL from outside) is charged by the parent's wall-clock measurement; if the PARENT itself dies, the reservation stays open until the next `reconcile_all`, which then charges the FULL reserved amount (deliberate over-charge, flagged in `unreconciled`).
3. CPU seconds are collected per supervised job via getrusage of the parent's reaped children; CPU burned by a process the parent did not reap is not counted.
4. RSS and worker-count ceilings are advisory: they are checked by the in-process watchdog of each child, not enforced by a cgroup.
5. Storage and the >=50 GiB free-space floor are polled, not quota-enforced.
6. The 7-day elapsed cap is operator-enforced against the ledger clock, not kernel-enforced.
