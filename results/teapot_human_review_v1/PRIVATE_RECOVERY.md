# Private material — locations, verification and commitments

This file records **where** the private material lives and **that** it was
verified. It contains **no private contents**: no mapping value, no association,
no salt, no Task-B image reference.

## What is private

| material | why |
|---|---|
| Neutral-ID → source / condition / pair mapping (Task A and Task B) | the association under review |
| Randomization salt (32 bytes) | backs the published commitments |
| Populated Task-B package (20 pairs, 40 images, viewer) | not released; release is gated |
| Two quarantined QA screenshots | each showed a real image beside its neutral ID |
| Per-file integrity rows (case/condition → sha256; item ID → sha256) | jointly they would reveal the mapping |

## Locations

| | path |
|---|---|
| **Primary** | `<work-directory>/private/` |
| **Backup** | `/data/bijaypandey/cuig_pilot/_teapot_human_review_v1_private_backup/` |

The work directory is `/data/bijaypandey/cuig_pilot/teapot_human_review_v1`.
Neither location is inside the repository, and neither is reachable from Git
history or from any release asset.

## Verification — host-performed

[`scripts/private_backup.py`](scripts/private_backup.py) mirrored the primary
directory to the backup and compared both trees by sha256.

| check | result |
|---|---|
| Files | **49** |
| Total bytes | 15,357,769 |
| All files present in the backup | **yes** |
| All hashes match | **yes** (0 mismatches) |
| Permissions restricted (nothing group/world-readable) | **yes** |
| Verdict | **`PRIVATE_BACKUP_VERIFIED`** |

Contents by count only: 1 mapping file, 1 salt file, 40 Task-B images, 4 Task-B
viewer files, 2 quarantined screenshots (plus the private integrity rows).

> **This is host-performed verification.** The process that wrote the files also
> hashed and compared them, on this machine. It is a local integrity check — not
> an independent attestation, and not proof to a third party that the private
> material is what it claims to be. Full report:
> [`verify/private_backup.json`](verify/private_backup.json).

Directory modes are `0700` and file modes `0600` in both locations. That
restricts other users on this host; it does not protect against anyone with
root or physical access.

## Commitments — to the actual populated artifacts

[`protocol/COMMITMENTS.json`](protocol/COMMITMENTS.json) publishes
HMAC-SHA256 commitments, keyed by the private salt, over:

* the **actual 40-item Task-A mapping**,
* the **actual 20-pair Task-B mapping**,
* the three fixed order variants,
* the Task-A item ID list.

These commit to the **real, populated** material — **not** to empty placeholders
or to a schema. The committed counts (40 Task-A items, 20 Task-B pairs) are
published alongside.

**The salt is not published**, and no secret or association was printed at any
point during preparation.

### What a commitment does and does not prove

The commitments cover the **populated Task-A mapping, the populated Task-B
mapping, the three order variants and the item-ID list**. They do **not** cover
the whole Task-B package — its images, viewer files and layout are not
committed.

What they support is a **future opening anchored to public publication**: the
commitments were published in this repository at a commit anyone can inspect, so
a later-revealed mapping can be checked against what was publicly recorded at
that point. They do **not** independently prove a self-declared private freeze
timestamp — the `frozen_iso` field inside the artifacts is self-declared and is
not independently verifiable.

They also do **not** make the images or their provenance private: both are
already public in the `modern-object-assets-v1` release. They do not establish
that the mapping is *correct*, only that it is *unchanged*, and they cannot be
checked by anyone else until the salt is revealed.

## Task-B release gate

Task B stays private until **both** hold:

1. All planned Task-A collection is **closed** and the original Task-A responses
   are **locked**.
2. The user gives **later explicit authorization**.

No Task-B content, pair ID or association is embedded in the Task-A bundle, and
there is no unlockable B package inside A — the masking audit confirms zero
Task-B members in the published ZIP.

> **The offline viewer cannot enforce this gate.** It is a coordinator
> procedure, not a technical control, and nothing in the UI is claimed to
> prevent a holder of the files from opening them.
