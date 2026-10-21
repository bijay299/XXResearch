# Erratum — the source-hash provenance link for the three synthetic backbones is broken

**Status:** **CLOSED** — repaired by bitwise reproduction (`p096`); see §3.1
**Affects:** `memorized_v2_s20261101`, `memorized_v2_s20261102`, `memorized_v2_s20261103`
and the three reports `outputs/memorize_v2_s2026110{1,2,3}_report.json`
**Does not affect:** any conclusion drawn from those backbones, for the reasons in §4

---

## 1. The break

All three delivered backbone reports record

```
scripts/memorize_v2.py   951970d094c4bbfab6bf7721c4a293dd219ef836c25fa1f51b06e1351f45d1db
```

The file on disk is

```
scripts/memorize_v2.py   39a307f8e1590b20af24e703739ce281e5a8b784b9e87ed1dbc469f56b5b9ba2
```

mtime `2026-10-20 14:51:40`, which is after all three backbones were delivered. The change is
a `--train-split {all,forget,retain}` option added for the `ignref` attempt (`p046`), with a
`G4a_IGNORANCE_of_forget_facts` gate that `docs/BACKBONE_BUILD.md` does not list. The recorded
hash resolves to no file anywhere under `/data` (searched; the only other copy,
`wt/results/mdlm_unlearn_method_dev/scripts/memorize_v2.py`, is the same post-change file).

## 2. Reconstruction was attempted and FAILED

I reverted the three identifiable additions — the argparse block, the `TRAIN_FACTS`
indirection, and the retain-split gate branch — each matched verbatim before removal. The
result hashed

```
57cc1fb62ca1c4fe66b4214fceed280cb727c7e91b295702061a77f8313c7b50
```

which is neither hash. **At least one further change exists that I cannot identify**, so the
pre-change file cannot be reconstructed and the hash link is unrecoverable. I am recording
this failure rather than presenting the reconstruction as the original.

## 3. Repair by reproduction, not by hash

The substantive question is not which string the report contains but whether the script now on
disk produces the delivered artifact. That is testable and strictly stronger than a hash
match. `scripts/queue_provenance.sh` re-runs the on-disk script at the recorded seed and
hyperparameters

```
lr 2e-4  steps 1200  rank 32  seed 20261101  batch 4  t_low 0.0
answer_weight 1.0  norm pinned  canvas genlen  grad_clip 1.0  threshold 0.95  mode qualify
```

into a **new** adapter directory (`adapters/prov_repro_s20261101`; the delivered backbone is
never written to) and compares `adapter_model.safetensors` against the recorded

```
bafc5c86a9f8e7916e9adeb3c178ed0e358e4edc8d0be1a6c3ff284bc797690d
```

Two possible outcomes, both to be recorded in `outputs/prov_repair.json`:

| outcome | what it licenses |
|---|---|
| bitwise identical | the recorded script reproduces the artifact; provenance is restored in substance and this erratum closes with the new hash recorded alongside the old |
| not identical | the caveat is **permanent**: the three backbones remain usable as artifacts with verified weights, but no source file on record regenerates them |

I will not report the first outcome unless the hashes match exactly.

### 3.1 Result: bitwise reproduced

```
recorded    bafc5c86a9f8e7916e9adeb3c178ed0e358e4edc8d0be1a6c3ff284bc797690d
reproduced  bafc5c86a9f8e7916e9adeb3c178ed0e358e4edc8d0be1a6c3ff284bc797690d
MATCH
```

Directory hash also matches (`fc3efb207e3546fb4ea88c3e9410f5cb…`), and the reproduction passed
the same gates (`G2c` trainable params changed, `G2d` base weights unchanged, `G2e` loss
decreased, `G5c` coverage 32 blocks × 7 linears), with `degenerate_steps = 0/1200`.

**The script now on disk (`39a307f8…`) reproduces the delivered backbone
`memorized_v2_s20261101` bit for bit.** The source-hash *string* in the three reports is stale
and stays as recorded — the reports are not edited — but the provenance *claim* those hashes
exist to support (that a recorded script regenerates the artifact) is re-established, by a
check stronger than a hash comparison. Attempt `p096`, 227.5 s training, output
`outputs/prov_repair.json`, new adapter kept at `adapters/prov_repro_s20261101` as evidence.

Seeds 20261102 and 20261103 were not re-run: they differ from 20261101 only in `--seed`, share
the code path exercised here, and re-running them would cost ~0.6 GPU-h to test the same
proposition. This is stated as a scope limit, not as a result about those two.

## 4. What remains verified regardless of the outcome

The break is in the source link only. These are unaffected and were checked:

| artifact | status |
|---|---|
| adapter weights (3 backbones) | per-file sha256 and directory sha256 recorded in each report |
| `scripts/common_v2.py` | **byte-identical** to the recorded `8ca28c19f1beb250…` |
| `spex/scripts/common_big.py` | recorded hash, read-only, unmodified |
| data (`facts.json`, `forget.jsonl`, `retain.jsonl`, `frozen_spec.json`) | four recorded hashes, unchanged |
| base model revision | `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| save/reload | `max|Δlogit| = 0` bitwise, recorded per backbone |
| qualification gates | recorded per backbone, 1.000/1.000 recall |

The added code path is also behaviourally inert for these three runs: `--train-split` defaults
to `all`, under which `TRAIN_FACTS` is `list(FACTS)` and the gate branch takes the `elif
args.mode == "qualify"` arm — the pre-change behaviour. That is an argument, not evidence, so
it is not offered as the repair; the reproduction test in §3 is.

## 5. Process change adopted

The break happened because a later attempt edited a script that delivered artifacts were
already pinned to. Going forward, any script that has appeared in a delivered report's
`hashes.sources` is append-only: a new behaviour goes in a new file. `scripts/train_ord_pool.py`
was written as a new file for exactly this reason rather than as a flag on `train_ord.py`.
