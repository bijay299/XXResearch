# Execution manifest — revision-02 sequential-erasure diagnostic (chain saw -> parachute)

Written **before any new image generation**, per the protocol. Execution authorized
separately from the protocol's publication; this manifest records exactly what is
being run and against what.

## Authority and identities

| Item | Value |
|---|---|
| Protocol commit (authoritative) | `c2c0388da4b8a1e1f7b1f8c4cff5e1a029b6ed80` |
| Protocol path | `results/modern_flux_dp_uce/sequential_diagnostic_protocol/PROTOCOL.md` |
| Repo / branch | `bijay299/XXResearch` / `research/audit-and-matched-effectiveness-protocol` |
| Repo HEAD verified at execution start | `c2c0388da4b8a1e1f7b1f8c4cff5e1a029b6ed80` (clean tree) |
| Pinned source | `czhang024/CVPR_DouleProjection` @ `3fc3d8afe03a400dfa71aaaa34b647fef6550e0f` (already cloned, unmodified) |
| Model | `black-forest-labs/FLUX.1-schnell` rev `741f7c3ce8b383c54771c7003378a50191e9efe9` (already downloaded, hash-verified in `../RUN_MANIFEST.md`) |
| Environment | `/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/venv_flux_dp_uce` (Python 3.10.12, torch 2.4.0+cu121, diffusers 0.30.2, transformers 4.44.2) — reused, nothing installed |
| Work directory | `/data/bijaypandey/cuig_pilot/modern_flux_dp_uce/exec_rev02/` (all compute, scratch, logs, images on `/data`) |
| Hardware (**initial allocation**, as frozen pre-generation) | 1x NVIDIA A100-SXM4-40GB, `CUDA_VISIBLE_DEVICES=2`, verified 0% util / 14MiB before allocation; GPUs 0/1/3 untouched **at that moment** |
| Hardware (**post-execution correction C5** — what was actually used over the whole run) | GPU 2 ran every edit and all four generations. **GPU 3 was additionally used later in the run, concurrently with GPU 2**, for residual logging and the `DP_SEQ` classification pass; it was verified idle immediately before that first use. GPUs 0/1 were never used. The row above records the *initial* allocation only and must not be read as exclusive use of a single physical GPU throughout. |
| `/data` free at start | 652 GB |

## Frozen inputs verified by hash before use

| Artifact | Expected sha256 | Verified |
|---|---|---|
| DP A-parent `runs/DP_B/models/flux_double_proxy_ChainSaw.safetensors` | `d9e3d91c1cb2b544ca8a70353146ce68bdf2d330b795223d01caea62911cfd0a` | MATCH |
| UCE A-parent `runs/UCE_B/models/flux_UCE_proxy_ChainSaw.safetensors` | `bd21369aed67f69f199d037fce27041da300aab17ba2179f23a9f08f9155ee20` | MATCH |
| Case manifest `manifest/full_case_manifest.csv` | `559607683c4b97dc5b3dbdf351735f798e7902f73d88d56c333a89217f1f3c26` | MATCH |

Reused existing outputs (not regenerated): M0 (1,000 images), DP A-parent `DP_B`
(1,000), UCE A-parent `UCE_B` (1,000) = 3,000 existing images.

## Pair, order and frozen settings

- **A = chain saw, guide `stick`** (already erased; parents above). **B = parachute,
  guide `cloth`.** Order **A->B only**. No reverse order, no fallback pair.
- Preserve list for `*_B_direct` and `*_SEQ` (exact released order from
  `Flux/scripts/Double_Parachute_flux.sh`, unchanged, chain saw already present):
  `cassette player;chain saw;church;english springer;gas pump;tench;garbage truck;golf ball;french horn`
- Preserve list for `*_BATCH` (same order, both erase targets removed):
  `cassette player;church;english springer;gas pump;tench;garbage truck;golf ball;french horn`
- DP native: `concept_type=object`, `replace_indices=all`, `expand_prompts=false`,
  model/edit dtype **bf16** with fp32 internal math, all-token updates.
- UCE native: `concept_type=object`, `erase_scale=10`, `preserve_scale=1`, `lamb=0.1`,
  `expand_prompts=false`, edit dtype **fp32**, last-content-token only.
- Edit-time `max_sequence_length` = **256** (`'schnell' in model_id` branch);
  generation-time `max_sequence_length` = **256** (native default). Recorded separately.
- Generation (frozen, unchanged): 512x512, 4 steps, `guidance_scale=0.0`, bf16 inference,
  batch size 1, `global_seed=42`, `base_seed_offset=0`, per-image seed =
  `evaluation_seed + variation_index`, 1,000 cases from the frozen manifest.
- Evaluator (frozen): pinned `Flux/compare_accuracy.py` `ErasureAccuracyEvaluator`,
  ResNet50 `IMAGENET1K_V2`, its own semantic-equivalence mapping; strict and semantic
  scores reported separately.

## Implementation changes (the only ones)

1. **`adapter/run_edit_with_parent.py`** — minimal parent-loading adapter, used **only**
   for `DP_SEQ`/`UCE_SEQ`. It (a) verifies the parent file's sha256 before loading,
   (b) wraps `DiffusionPipeline.from_pretrained` so the parent's two tensors are merged
   into `pipe.transformer` **immediately on construction, strictly before** the pinned
   script scans `named_modules()` and deepcopies `original_modules` — so every reference,
   guide and preservation response is computed from the edited parent, (c) verifies the
   actually-loaded tensors are bitwise equal to the parent, exactly two expected keys,
   finite, native dtype (no silent casting — a dtype mismatch is a hard failure), and
   a save/reload check on the produced child (see the correction immediately below), and
   (d) for DP only, clones the final fp32 matrices at the moment of the native bf16 cast
   (shape+dtype filtered, observational only) for the protocol's pre-cast residual log.
   **The pinned source files are not modified.** Native arithmetic, update order, token
   handling, precision and guide semantics are untouched.

   > **Post-execution correction C8 — what `save_reload_bitwise_identical` actually
   > records.** The executed code (`run_edit_with_parent.py`, the `child2 = load_file(...)`
   > block) deserializes the **saved file a second time** and compares that against the
   > first deserialization of the same file. It is a **repeated-deserialization
   > determinism check**, *not* a documented comparison against the in-memory tensor as it
   > existed before saving. It therefore does not, on its own, rule out a
   > serialization-time discrepancy between the in-memory edit result and the stored file.
   > No further integrity experiment was run; the recorded field and its value are
   > unchanged, only its description is corrected.

2. `*_B_direct` and `*_BATCH` endpoints are run through the **native CLI unchanged**
   (no adapter, no patching) — the no-parent path remains the native reproduction.
3. Residual logging is computed **post hoc** from saved checkpoints plus cached text
   encodings; it does not alter any edit. **Post-execution correction C3:** that logger
   hardcodes **bf16** text encoders for both methods, while native UCE encodes in fp32 —
   so UCE's logged residuals are probes on bf16 encodings, not measurements of its native
   fp32 preservation vectors. The executed logger source and its outputs are preserved
   unchanged as historical artifacts; see `FINAL_REPORT.md` §3 and `CORRECTIONS.md`.

No tuning knob exists in any of the above, and nothing is adjusted in response to
residuals or to any intermediate result.

## Endpoint accounting and budget

Ceiling: **6 new endpoints, 6,000 newly generated images** (4 primary / 4,000 plus at
most 2 conditional comparators / 2,000). The ceiling is not a target. Failed or retried
generations count against it. Gated order:

1. `DP_B_direct`, `UCE_B_direct` (pristine M0) — edits complete, see below.
2. Per method, only after technical integrity **and** >=75% direct B suppression:
   `DP_SEQ` / `UCE_SEQ` from that method's verified A-parent.
3. Per method, only if that method's SEQ has >=1 **confirmed** relapse witness and its
   required technical, direct/sequential suppression and retention gates have been
   evaluated and passed (unknown != passed; ambiguous-only never triggers):
   `DP_BATCH` / `UCE_BATCH`.

A failed method stops; the other continues independently.

## Edits completed before generation (checkpoint provenance)

| Endpoint | Checkpoint sha256 | Keys | dtype | Finite | Save/reload | Integrity |
|---|---|---|---|---|---|---|
| `DP_B_direct` | `ba91e2443451836d9bae02e305ef67138dcc7297262a2670f17ac21ee8a68b35` | exactly the 2 expected | bf16 | yes | bitwise identical | PASS |
| `UCE_B_direct` | `badb20e3f0fbe9efb577882120b9ef0c89ad0c2bb5a80f746bdce598bcff2766` | exactly the 2 expected | fp32 | yes | bitwise identical | PASS |

DP native edit log for `DP_B_direct`: parachute tokenizes to **3 T5 tokens**, all three
updated (`‖x‖²` = 2.1055e+01, 2.2654e+01, 4.5781e+01 — all far above the `1e-10`
skip threshold, so no token was skipped), plus the CLIP-pooled module
(`‖x‖²` = 6.0479e+02). This resolves ledger item 21 for this pair.

## Exact commands

```
# DP_B_direct  (native CLI, no adapter)
python Double_proxy.py --edit_concepts "parachute" --guide_concepts "cloth" \
  --preserve_concepts "cassette player;chain saw;church;english springer;gas pump;tench;garbage truck;golf ball;french horn" \
  --concept_type object --replace_indices all --expand_prompts false \
  --save_dir <exec_rev02>/runs/DP_B_direct/models --exp_name flux_double_proxy_Parachute

# UCE_B_direct (native CLI, no adapter)
python UCE_original.py --edit_concepts "parachute" --guide_concepts "cloth" \
  --preserve_concepts "cassette player;chain saw;church;english springer;gas pump;tench;garbage truck;golf ball;french horn" \
  --concept_type object --erase_scale 10 --preserve_scale 1 --lamb 0.1 --expand_prompts false \
  --save_dir <exec_rev02>/runs/UCE_B_direct/models --exp_name flux_UCE_proxy_Parachute

# DP_SEQ / UCE_SEQ (adapter; parent verified by sha256 before load)
python adapter/run_edit_with_parent.py --method {DP|UCE} \
  --parent_checkpoint <A-parent .safetensors> --parent_sha256 <pinned hash> \
  --edit_concepts "parachute" --guide_concepts "cloth" \
  --preserve_concepts "cassette player;chain saw;church;english springer;gas pump;tench;garbage truck;golf ball;french horn" \
  --save_dir <exec_rev02>/runs/{DP,UCE}_SEQ/models --exp_name <name> \
  --verification_log <logs>/<name>.verification.json [--precast_dump <DP only>]

# generation (identical for every endpoint)
python generate_erased_flux.py --erased_model <child .safetensors> \
  --output_dir <endpoint>/images --prompt_csv <frozen CSV> \
  --num_variations 10 --global_seed 42 --resolution 512 --base_seed_offset 0 \
  --batch_size 1 --num_inference_steps 4
```
