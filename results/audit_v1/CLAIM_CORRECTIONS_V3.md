# CLAIM_CORRECTIONS_V3 — AUDIT-01d

Two guarantees AUDIT-01c claimed to have closed were still open, and two claims
were overstated. All four are corrected here.

**No measured value changed.** No model was trained, no image generated, no
detection run, no checkpoint downloaded, no coefficient swept, no experiment
launched. **No GPU work of any kind occurred in this round.** Everything below is
CPU: hashing, JSON, text and shell.

Prior rounds: [`CLAIM_CORRECTIONS.md`](CLAIM_CORRECTIONS.md) (AUDIT-01, nine
pilot claims), [`CLAIM_CORRECTIONS_V2.md`](CLAIM_CORRECTIONS_V2.md) (AUDIT-01b/c,
eight of the audit's own overstatements). Originals preserved in
[`preserved_originals/`](preserved_originals/) and git `a9de625` / `f7f4ba8`.

---

## G1 — a failed evaluation's images could be relabelled as a new checkpoint's

**What AUDIT-01c claimed.** That evaluations are bound to the model that
produced them: the launcher and the analysis path both recompute the on-disk
checkpoint hash and the validator requires the recorded hash to be supplied, to
be present, and to match.

**All of that is true, and it is not sufficient.** It binds the *metadata*. It
does not bind the *pixels*.

**The hole.** `run_seed.sh` quarantined `detections.jsonl` and
`detect_report.json` when an evaluation failed validation, and left the `images/`
directory and `image_report.json` in place. `generate_eval_images.py` then reused
every existing file **by pathname alone** — `if path.exists(): status="reused"` —
and wrote the **currently loaded** checkpoint's hash, the current manifest
identity and the current settings into a fresh report. The first, stale
evaluation was correctly rejected; the *recovery* then produced a result in which
every field the validator inspects was truthful about the current checkpoint
while the image bytes came from an older one. The validator could not see it,
because there was nothing left to see.

**Reproduced by the review**, driving the real generator under fake
torch/diffusers: one old placeholder image unchanged, **zero** generation calls,
the report claiming the current checkpoint hash — and the real validator then
accepting the result against synthetic detection metadata.

**Fix — reuse is gated on provenance, and the gate runs before any model.**
An existing image file is reused only when a prior report in that directory
proves it was produced under the same contract:

| bound | how |
|---|---|
| generating checkpoint | prior `checkpoint_load.sha256` == recomputed sha256 of the `delta.bin` now on disk (for M0: no delta applied, no `unet_ckpt`, no recorded delta hash, same `base_model_dir`) |
| prompt set | prior `manifest_sha256` == the **recomputed canonical content digest** of the manifest now in use |
| frozen identity | that content digest == `SEQ_EVAL_MANIFEST_SHA` from configuration |
| settings | prior `generation_settings` == the settings now in use, and the same image-seed rule |
| the bytes themselves | prior per-image recorded `sha256` == the digest of the file on disk **now**, and the same `prompt_id`, prompt text, `gen_seed` and derived image seed |

Any failure is **exit 3 before `import torch`**: nothing is written, the old
report is not overwritten, and no old image byte is reattributed. The one
pathname-only branch that remains is a fail-closed assertion that returns 3 if a
caller ever reaches it.

**Recovery is now decided, not drifted into.** `run_seed.sh` asks the
generator's own gate (`--provenance_check_only`, CPU, loads nothing) before
quarantining anything:

- images **provably** this checkpoint's → quarantine `detections.jsonl` and
  `detect_report.json` only, and re-run detection over the preserved images;
- otherwise → quarantine **the whole evaluation directory** so images and reports
  are preserved *together*, with a reason file, and generate into a fresh empty
  directory. A symlinked (shared M0) directory is refused rather than regenerated
  through.

**The launcher and the generator cannot disagree** about what provenance means:
there is one implementation and the launcher calls it.

**Regressions** — `scripts/seq/test_image_provenance.sh`, **57 checks**, real
launcher + real generator + real validator + real analysis gate, fake
torch/diffusers (`scripts/seq/testlib/fake_models/`), every pipeline construction
and generation call counted:

| case | asserted |
|---|---|
| valid same-contract resume | exit 0, **0** generations, **0** pipeline constructions, image bytes unchanged, images **not** quarantined, invalid detections quarantined, `n_reused` = every image |
| changed checkpoint under the same name, through the launcher | old evaluation quarantined **whole** (images *and* report, with a reason), quarantined bytes are the originals, fresh evaluation has **new** bytes, `n_reused` = 0, fresh report records the on-disk checkpoint |
| changed checkpoint, generator direct | exit 3, **0** pipeline constructions, old report byte-identical afterwards |
| changed prompt text, old filenames | exit 3; and the frozen-identity gate rejects it even earlier (exit 2) |
| changed generation settings, old filenames | exit 3, settings difference named |
| no prior image report | exit 3, nothing written |
| recorded image digest no longer matches its bytes | exit 3 |
| M0 report claiming a delta | exit 3 |
| **analysis** on the refused state | exit 1, "do not validate", nothing published |

The fake pipeline's output bytes depend on the loaded checkpoint's digest, so
"the images were really regenerated" is an assertion about bytes, not a
tautology.

---

## G2 — the legacy completion exception covered new runs, not saved artifacts

**What AUDIT-01c claimed.** That the legacy exception exists so a missing
historic counter cannot silently trigger retraining, and that "any **new** seed
is not on that list, so its counter is required".

**The hole.** The exception was a list of training-seed **numbers**,
`SEQ_LEGACY_TRAIN_SEEDS="17 29"`, and `seq_steps_evidence(seed)` took no
artifact, path or creation-state input at all. So:

- a **brand-new** run using seed 17 or 29 inherited the exception and could pass
  completion validation with **no counter**;
- the proposed new unregularised trajectories use **exactly those two seeds** —
  the exception would have covered the very runs whose completion must be proven;
- `run_seed.sh` computed the policy **once per seed** and applied it both before
  and after training;
- `${VAR:-default}` meant an explicitly **empty** configuration silently restored
  the default list;
- E2E case 15d "tested" the exception by resetting seed 29, creating fresh
  counterless mock reports and expecting success — a test of new output, not of
  saved reuse — and 15c **avoided** the default by overriding the list to 99
  rather than testing it.

**Fix — the exception is bound to the exact saved artifacts, by content.**
[`configs/legacy_training_artifacts.json`](../../configs/legacy_training_artifacts.json)
registers the ten saved pilot artifacts, each by its `train_report.json` sha256
**and** its `delta.bin` sha256, with provenance (commit `a9de625`, saved path,
reported parent/target/L2 weight/finish time, and
`carries_optimizer_steps_completed: false` for all ten). All ten in-repo report
copies under `results/seq{17,29}/train/` are **byte-identical** to the saved
reports, verified.

`scripts/seq/legacy_artifact_policy.py decide` hashes the artifact **on disk**
and matches it against that registry. Consequences:

- **every newly produced output requires the actual counter**, regardless of
  seed or directory label — it cannot match the registry, because its bytes are
  new;
- an **edited** report, or a checkpoint swapped under a legacy **name**, breaks
  the match and is held to the strict policy, with the near-miss named
  (`[legacy-miss] …`);
- a registered artifact keeps completion **UNVERIFIED** — structural validation
  in full — and is never retrained, moved, replaced or backfilled;
- a counter that is **present but short** fails in **both** policies;
- `SEQ_LEGACY_ARTIFACT_REGISTRY=""` removes every exception, as documented. The
  `${VAR-default}` form honours an explicitly empty value;
- a missing, unparseable or malformed registry, or an unreadable artifact,
  yields the strict policy. **Fail-closed in every direction.**

**Post-training validation is strict unconditionally.** `validate_train` takes a
forced policy, and `train()` passes `counter` immediately after a launch: nothing
this launcher just produced can reach the legacy branch even in principle.

**Saved checkpoints are protected from being overwritten.** Separately from the
completion policy, `run_seed.sh` refuses to retrain a checkpoint whose
`delta.bin` is registered saved evidence, **whatever state its report is in**. An
edited report loses the completion exception but the checkpoint beside it is
still the only copy of what produced the published rates; previously an
unvalidatable report would have led to retraining over it.

**Regressions** — in `scripts/seq/test_launcher_e2e.sh`:

| case | asserted |
|---|---|
| 15c | a **fresh seed-29** output with no counter is rejected **under defaults**, no override, and the legacy exception is **not** applied |
| 15c2 | a short counter is rejected under defaults |
| 15c3 | a **fresh seed-17** output with no counter is rejected under defaults, in an isolated eval directory |
| 15d | a **registered** artifact accepts an absent counter: completion UNVERIFIED, MA **not** retrained, nothing failed |
| 15d2 | a **changed** artifact bearing a registered name: identity difference named, and the launcher **refuses to retrain over the registered checkpoint** |
| 15d3 | the policy reads artifacts, not seed numbers: registered → `legacy_optional`; **explicitly empty** registry → `counter`; real registry vs mock artifact → `counter`; absent registry → `counter`; malformed registry → `counter`; unregistered sibling → `counter` |
| 15d4 | a short counter is rejected **even for a registered artifact** |

and in `scripts/seq/test_validate_stage.py`, the policy script is tested
directly against the **real** saved artifacts and the committed registry.

---

## W1 — `--verify_base_model` is a partial identity check, not backbone verification

**What AUDIT-01c said** (`CLAIM_CORRECTIONS_V2.md` §C1):

> "M0 validation asserts: … and — with `--verify_base_model` — the recomputed
> backbone identity."

**Why that is an overstatement.** `_base_model_cheap_identity` hashes five small
JSON config files and records the weight files' **byte lengths**. It never reads
the weight bytes. The contract's `weight_sha256` — the full UNet weight digest —
was never compared. **A weight mutation that preserves file length passes**, and
the review reproduced exactly that on CPU. Neither caller requests even the cheap
check by default.

**Corrected statement of the launch-time policy**, written into
`configs/base_model_contract.json` → `identity_policy` and into the validator's
own recorded verdict strings:

- At launch, M0's identity is asserted **structurally**: no delta applied, no
  `unet_ckpt`, **no** recorded delta hash, and `base_model_dir` equal to the
  contract's.
- The cheap identity is **declared from the contract and not recomputed** unless
  `--verify_base_model` is passed. The validator now records
  `base_model_identity = "DECLARED, not verified: …"` in that case, instead of
  implying a check happened.
- With `--verify_base_model` it records
  `"PARTIAL: config-file digests and weight-file byte lengths match … The weight
  CONTENTS were not hashed, so a same-size weight mutation would pass this
  check"`.
- The **full** content check is the new, explicit
  `--verify_base_model_weight_sha`, which hashes the weight files and compares
  them against the contract's `weight_sha256`.
- **Neither launcher enables it per evaluation**, and that is a deliberate,
  recorded decision rather than an omission: evaluation validation runs once per
  checkpoint per path, and re-hashing ~3.4 GB on each call would be a large
  repeated read for a backbone this pipeline never writes. No caching or hashing
  optimisation was built, and no model work was started.
- **Non-M0 checkpoints are different and stronger**: each is bound by the
  recomputed sha256 of its own `delta.bin`, which *is* a full content check of
  the delta.

**Residual limitation, stated.** Even the full weight digest checks the backbone
against **this contract**, whose values were derived from the on-disk backbone at
AUDIT-01c. It establishes that the backbone has not changed since then. It does
**not** establish that the backbone is the published Stable Diffusion v1.5
release: no upstream-published digest is compared anywhere.

---

## W2 — "no structured counter" is not "no evidence", and is not grounds to retrain

The ten saved reports carry no completed-step counter and training completion
**remains UNVERIFIED** for all of them. That is unchanged.

The terminal progress output in the saved logs was *reported* in an earlier round
but not verified. It now has a bounded CPU inspection with its provenance and
limits stated first:
[`TRAINING_LOG_INSPECTION.md`](TRAINING_LOG_INSPECTION.md) /
[`training_log_inspection.json`](training_log_inspection.json).

| measured | result |
|---|---|
| training logs found (2 seeds × 5 checkpoints) | **10 / 10** |
| logs showing a terminal `1000/1000` at `100%` | **10 / 10** |
| distinct progress-bar totals in any log | `[1000]` only |
| logs embedding the **registered `delta.bin` digest** of their artifact | **10 / 10** |

**This is kept clearly separate from direct counters.** A tqdm bar is terminal
display rendering its own configured total, written by the same process that
would have recorded a counter, never parsed or validated at the time, and not the
trainer's internal completed-step variable. No contract hashes these logs. So it
is **corroborating provenance** — this log belongs to this checkpoint — not a
completion measurement. **No counter was backfilled into any report**, and
`legacy_optional` still reports `training_completion_verified: false`.

**No L2 retraining follows.** The diagnostic compares **exact saved L2
endpoints** with new unregularised checkpoints at **comparable measured dog
suppression**. Equal verified training duration is not its matching variable, so
the logging asymmetry does not justify retraining the two L2 arms. The
~0.46 GPU-hour suggestion from an earlier round remains an **unapproved scope
expansion** and is excluded; it would also change the reference artifacts. The
asymmetry is disclosed in the protocol instead.

---

## W3 — "mildly optimistic" was an unsupported assurance

**What was said** (protocol §4, `disjointness_report.json`, `FREEZE.md`,
`AUDIT_REPORT.md` limitation 11):

> "Where prompt texts share a template the bootstrap's prompt clusters are not
> fully independent and intervals are mildly optimistic."

**Why it is unsupported.** Neither the direction nor the size of that effect was
estimated anywhere. "Mildly" was an assurance, not a result. **Withdrawn.**

**And the grouping it described was wrong about the real dependence.** Treating
each prompt *text* as one cluster ignores that `literal_i` and `paraphrase_i` of
a category are **the same scene worded two ways, by construction** — so it would
have counted a dependent pair as two independent observations.

**The protocol's uncertainty specification is restored in full and the grouping
is frozen from the actual manifests** —
[`draft_manifests/bootstrap_grouping.json`](draft_manifests/bootstrap_grouping.json),
identity `2ba0e8df5034d011…`, built by
`scripts/seq/freeze_bootstrap_grouping.py`:

| element | frozen value |
|---|---|
| resampling **unit** | the **scene cluster** `(set, category, prompt_index)` |
| the unit carries | **both** family wordings **and all four** generation seeds, together |
| **stratum** | **category**; with replacement within each stratum, that stratum's own cluster count per draw |
| pairing | one resampled cluster list indexes **both arms** and every endpoint; contrasts paired at identical `(prompt_id, gen_seed)` |
| draws | **10 000** |
| interval | 95% percentile |
| **RNG** | `numpy.random.default_rng(2026100901)`, recorded, with the draw order written down; distinct from the annotation sampler's `20261009` |

Verified from the manifests rather than assumed: **every** cluster of dev (10),
test (70) and the pilot set (35) carries **both** families and **all four**
generation seeds; strata are 10 clusters per category in dev and test, 5 in the
pilot set. Literal-only and paraphrase-only endpoints are recomputed on the
**same** resampled cluster list, which keeps them paired.

**The residual dependence is measured, and left un-characterised.** Clustering
absorbs within-scene dependence. A template shared *across* clusters is not
absorbed, and the test set has one: **all 70** literal prompts open
`"a photo of a"`, spanning all 70 clusters (pilot set: 3 such templates over 13
texts). **The direction and size of its effect on interval width are not
established by this design, and none is claimed.**

**Unchanged, deliberately**: the reference gates (≥30 pp dog suppression and
≤60% residue), the ≤5 pp match tolerance with earliest-step tie-break, the
infeasibility rule, all-category separate reporting with no averaging, the
non-target sandwich category, thresholds 0.3/0.5/0.7 with 0.5 primary,
per-training-seed intervals never pooled, the 10 pp material effect size and the
[−10, +10] equivalence margin, and every interpretation rule in §7.

---

## Scope, unchanged

Two new unregularised dog trajectories; **1,920** development + **3,360** test
images; estimated **3.095 GPU-hours**; proposed **four-hour** ceiling
([`protocol_cost_model.json`](protocol_cost_model.json)). No budget or launch was
approved and none is requested here.

**Still outstanding, and not addressed by this round:** actual blinded **human**
annotation — the packet is key-free and its label sheet is **empty**, 236 rows,
zero filled judgement cells — and the PI's concrete launch decision. No AI-filled
labels exist. No new research scope was added.
