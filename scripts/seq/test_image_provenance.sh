#!/bin/bash
# ---------------------------------------------------------------------------
# CPU regressions for image-reuse provenance.
#
# The bug these cover. A failed evaluation used to quarantine detections.jsonl
# and detect_report.json but leave the IMAGES and image_report.json in place.
# generate_eval_images.py then reused every existing file by PATHNAME alone and
# wrote the current checkpoint hash plus the current manifest and settings into a
# fresh report. The initial stale evaluation was rejected, but the recovery run
# relabelled the old image bytes as evidence from the new model, and nothing
# downstream could see it: every piece of metadata the validator inspects had
# been rewritten truthfully about the CURRENT checkpoint.
#
# What is REAL here and what is mocked:
#
#   REAL  scripts/seq/run_seed.sh          -- the actual launcher recovery path
#   REAL  scripts/seq/generate_eval_images.py -- the actual reuse branch
#   REAL  scripts/seq/validate_stage.py    -- the actual validator
#   REAL  scripts/seq/run_analysis.sh      -- the actual analysis gate
#   fake  torch / diffusers                -- scripts/seq/testlib/fake_models
#   mock  accelerate (training), detect.py, gpu_select.sh, nvidia-smi
#
# The reuse branch under test is NOT mocked away: the real generator decides
# reuse, and every pipeline construction and generation call is recorded in
# $FAKE_CALLS so the tests can assert that nothing was generated (or that
# something was). No GPU, no model, no network, no real image.
#
#   bash scripts/seq/test_image_provenance.sh
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=""

PASSN=0; FAILN=0
ok()  { PASSN=$((PASSN+1)); printf '  ok    %s\n' "$1"; }
bad() { FAILN=$((FAILN+1)); printf '  FAIL  %s\n' "$1"; [ -n "${2:-}" ] && printf '          %s\n' "$2"; }
want_rc()  { if [ "$2" = "$3" ]; then ok "$1 (exit $3)"; else bad "$1" "exit $3, wanted $2"; fi; }
want_out() { if grep -qF -- "$3" "$2"; then ok "$1"; else bad "$1" "output lacks: $3"; fi; }
want_not() { if grep -qF -- "$3" "$2"; then bad "$1" "output unexpectedly has: $3"; else ok "$1"; fi; }
want_eq()  { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1" "got '$2', wanted '$3'"; fi; }

ROOT="$(mktemp -d)"
if [ -z "${KEEP_PROV_ROOT:-}" ]; then trap 'rm -rf "$ROOT"' EXIT; else
  trap 'echo "sandbox kept: $ROOT"' EXIT; fi

REPO_AT_RISK="results/seq17 results/seq29 configs"
REPO_SNAPSHOT="${ROOT}/repo_before.sha"
# shellcheck disable=SC2086
( cd "$REPO" && find $REPO_AT_RISK -type f -exec sha256sum {} + 2>/dev/null | sort ) \
    > "$REPO_SNAPSHOT"

export CALL_LOG="${ROOT}/calls.log";        : > "$CALL_LOG"
export FAKE_MODEL_CALL_LOG="${ROOT}/fake_model_calls.log"; : > "$FAKE_MODEL_CALL_LOG"
FAKE_CALLS="$FAKE_MODEL_CALL_LOG"
FAKE_MODELS="${REPO}/scripts/seq/testlib/fake_models"

# Count calls of one kind in the fake-model log since a marked offset.
n_calls() {   # n_calls <kind> [from_line]
    local kind="$1" from="${2:-1}"
    tail -n "+${from}" "$FAKE_CALLS" 2>/dev/null | grep -cF "\"call\": \"${kind}\"" || true
}
calls_mark() { wc -l < "$FAKE_CALLS" | tr -d ' '; }

# --------------------------------------------------------------- mock harness
BIN="${ROOT}/bin"; mkdir -p "$BIN"

cat > "${BIN}/nvidia-smi" <<'EOF'
#!/bin/bash
echo "MOCK nvidia-smi $*" >> "$CALL_LOG"; echo "0, 41 MiB, 0 %"
EOF

cat > "${ROOT}/mock_gpu_select.sh" <<'EOF'
#!/bin/bash
echo "MOCK gpu_select $*" >> "$CALL_LOG"
for i in 0 1 2 3; do echo "GPU $i [mock] -> IDLE"; done
EOF
chmod +x "${ROOT}/mock_gpu_select.sh"
export GPU_SELECT="${ROOT}/mock_gpu_select.sh"

cat > "${BIN}/accelerate" <<'EOF'
#!/bin/bash
echo "MOCK accelerate $*" >> "$CALL_LOG"
name=""; out=""; l2="0"; seed=""; parent=""; target=""; anchor=""; iters="1000"
while [ $# -gt 0 ]; do
  case "$1" in
    --name) name="$2"; shift 2;; --output_dir) out="$2"; shift 2;;
    --l2sp_weight) l2="$2"; shift 2;; --seed) seed="$2"; shift 2;;
    --parent_name) parent="$2"; shift 2;; --target) target="$2"; shift 2;;
    --anchor_name) anchor="$2"; shift 2;; --iterations) iters="$2"; shift 2;;
    *) shift;;
  esac
done
"$MOCK_PY" "$MOCK_HELPER" make_train --out "$out" --name "$name" --l2 "$l2" \
    --seed "$seed" --parent "$parent" --target "$target" --anchor "$anchor" \
    --iters "$iters"
EOF

# The python shim. generate_eval_images.py is NOT intercepted: it runs for real
# with the fake model modules first on PYTHONPATH, which is the whole point of
# this suite. Everything else GPU-bound is mocked.
cat > "${BIN}/python" <<'EOF'
#!/bin/bash
for a in "$@"; do
  case "$a" in
    *generate_eval_images.py)
      echo "REAL generate_eval_images $*" >> "$CALL_LOG"
      exec env PYTHONPATH="${FAKE_MODELS}:${PYTHONPATH:-}" "$MOCK_PY" "$@" ;;
    *train_request.py|*detect.py|*param_movement.py)
      echo "MOCK python $*" >> "$CALL_LOG"
      exec "$MOCK_PY" "$MOCK_HELPER" gpu_entry "$@" ;;
  esac
done
exec "$MOCK_PY" "$@"
EOF
cp "${BIN}/python" "${BIN}/python3"
chmod +x "${BIN}"/*
export FAKE_MODELS

if [ -x /data/bijaypandey/cuig_pilot/venv-cuig/bin/python ]; then
    export MOCK_PY=/data/bijaypandey/cuig_pilot/venv-cuig/bin/python
else
    export MOCK_PY="$(command -v python3)"
fi
"$MOCK_PY" -c 'import torch' 2>/dev/null || {
    echo "FATAL: no interpreter with torch available (the validator needs it)" >&2
    exit 2; }
export MOCK_HELPER="${ROOT}/mock_helper.py"

cat > "$MOCK_HELPER" <<'PYEOF'
"""Mock the stages this suite is not testing: training and detection."""
import argparse, hashlib, json, os, sys
from pathlib import Path


def make_train(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import torch
    contract = json.loads(Path(os.environ["SEQ_CKPT_CONTRACT"]).read_text())
    fill = float(os.environ.get("MOCK_DELTA_FILL", "0.01"))
    sd = {k: torch.full(tuple(v["shape"]), fill) for k, v in contract["tensors"].items()}
    ck = out / "delta.bin"
    torch.save({contract["top_level_key"]: sd}, ck)
    h = hashlib.sha256(ck.read_bytes()).hexdigest()
    parent_sha = None
    if a.parent and a.parent != "M0":
        p = out.parent / a.parent / "delta.bin"
        if p.exists():
            parent_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    steps = int(a.iters)
    rep = {
        "checkpoint": a.name, "parent": a.parent, "training_seed": int(a.seed),
        "l2sp_weight": float(a.l2), "new_deletion_target": a.target,
        "anchor_concept": a.anchor,
        "child_checkpoint": {"sha256": h, "bytes": ck.stat().st_size, "tensors": len(sd)},
        "effective_hyperparameters": {"seed": int(a.seed), "iterations_requested": steps,
                                      "epoch_capacity_steps": 1250,
                                      "parameter_group": "kv-xattn"},
        "parent_verification": {"parent_sha256": parent_sha, "PASS": True,
                                "max_abs_deviation_after_load": 0.0},
        "l2sp_reference_check": {"PASS": True, "max_abs_deviation_from_parent": 0.0},
        "l2sp_trace": {"steps_recorded": steps if float(a.l2) > 0 else 0},
        "started_utc": "2026-10-09T00:00:00+00:00",
        "finished_utc": "2026-10-09T00:14:00+00:00",
        "runtime": {"train_seconds": steps * 0.83, "seconds_per_optimizer_step": 0.83,
                    "seconds_per_optimizer_step_is_derived": True,
                    "optimizer_steps_completed": steps,
                    "optimizer_steps_source": "mock (test fixture)"},
    }
    (out / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(f"mock trained {a.name}")


def gpu_entry(argv):
    script = next(x for x in argv if x.endswith(".py"))
    g, i = {}, 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv) and not argv[i+1].startswith("--"):
            g[argv[i][2:]] = argv[i+1]; i += 2
        else:
            i += 1
    if script.endswith("detect.py"):
        ir = json.loads(Path(g["image_report"]).read_text())
        man = json.loads(Path(os.environ["SEQ_EVAL_MANIFEST"]).read_text())
        rows = []
        for n, r in enumerate(man["records"]):
            ip = Path(ir["out_dir"]) / r["image_name"]
            rows.append({"checkpoint": g["checkpoint_name"], "prompt_id": r["prompt_id"],
                         "category": r["category"], "prompt_family": r["prompt_family"],
                         "prompt_index": r["prompt_index"], "prompt": r["prompt"],
                         "gen_seed": r["gen_seed"], "image_path": str(ip),
                         "image_sha256": hashlib.sha256(ip.read_bytes()).hexdigest()
                                         if ip.exists() else f"h{n}",
                         "detections": [], "max_score_target": 0.1,
                         "n_detections_target": {"0.3": 0, "0.5": 0, "0.7": 0},
                         "hit": {"0.3": False, "0.5": False, "0.7": False}})
        Path(g["out"]).write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        Path(g["report"]).write_text(json.dumps({
            "checkpoint": g["checkpoint_name"], "complete": True,
            "images_scored": len(rows), "images_expected": len(man["records"])}, indent=2))
        return 0
    if script.endswith("param_movement.py"):
        mr = Path(g["models_root"])
        cks = sorted(p.name for p in mr.iterdir() if (p / "delta.bin").is_file())
        Path(g["out"]).write_text(json.dumps({
            "note": "MOCK parameter movement (test fixture)",
            "checkpoints": {c: {"relative_l2_vs_M0": 0.01,
                                "relative_l2_vs_MA": 0.001} for c in cks}}, indent=2))
        return 0
    print(f"mock: unexpected GPU entry point {script}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if sys.argv[1] == "make_train":
        ap = argparse.ArgumentParser()
        for f in ("out", "name", "l2", "seed", "parent", "target", "anchor", "iters"):
            ap.add_argument(f"--{f}")
        make_train(ap.parse_args(sys.argv[2:]))
        sys.exit(0)
    sys.exit(gpu_entry(sys.argv[2:]))
PYEOF

# ---------------------------------------------------------------- environment
SEQ="${ROOT}/seq_pilot"
mkdir -p "$SEQ"/{models,analysis,logs,calib,anchor_caches/Horses,anchor_caches/Flowers}
export SEQ_ROOT="$SEQ"
export SEQ_REPO_OUT_OVERRIDE="${ROOT}/repo_out"; mkdir -p "$SEQ_REPO_OUT_OVERRIDE"
export PILOT_REPO_ROOT="$REPO"
export PILOT_VENV="${ROOT}/venv"; mkdir -p "${PILOT_VENV}/bin"
echo '# mock venv: a no-op so the mock PATH stays in front' > "${PILOT_VENV}/bin/activate"
export SEQ_ITERATIONS=1000
export SEQ_ACCEL_CFG="${ROOT}/accel.yaml"; echo "{}" > "$SEQ_ACCEL_CFG"
export SEQ_BASE_MODEL="${ROOT}/sd15"; mkdir -p "$SEQ_BASE_MODEL"
export GPU_A=2 GPU_B=3
export SEQ_GEN_SETTINGS_CONTRACT="${REPO}/configs/generation_settings.json"
export SEQ_VALIDATOR="${REPO}/scripts/seq/validate_stage.py"
export SEQ_EVAL_MANIFEST="${SEQ}/eval_manifest.json"

# A small frozen manifest carrying the REAL generation-settings contract, because
# the real generator reads its settings from the manifest and the validator
# compares them against that contract.
build_manifest() {   # build_manifest <out> [prompt_suffix] [steps_override]
    "$MOCK_PY" - "$1" "${2:-}" "${3:-}" <<'PY'
import json, sys, hashlib
from pathlib import Path
out, suffix, steps = sys.argv[1], sys.argv[2], sys.argv[3]
gs = json.loads(Path("configs/generation_settings.json").read_text())
gs = {k: v for k, v in gs.items() if not k.startswith("_")}
if steps:
    gs["num_inference_steps"] = int(steps)
recs = []
for c in ["cat", "dog", "sandwich", "horse", "bird", "chair", "bicycle"]:
    for fam in ["literal", "paraphrase"]:
        for s in (101, 202):
            recs.append({"prompt_id": f"{c}_{fam}_0", "category": c,
                         "prompt_family": fam, "prompt_index": 0,
                         "prompt": f"a photo of a {c} ({fam} 0){suffix}",
                         "gen_seed": s,
                         "image_name": f"{c}_{fam}_0_seed{s}.jpg"})
m = {"experiment": "mock", "frozen_before_any_editing": True, "records": recs,
     "gen_seeds": [101, 202], "images_per_checkpoint": len(recs),
     "generation_settings": gs}
sys.path.insert(0, "scripts/seq")
from validate_stage import canonical_manifest_digest
m["manifest_sha256"] = canonical_manifest_digest(m)
Path(out).write_text(json.dumps(m, indent=1))
print("manifest:", len(recs), "records,", m["manifest_sha256"][:12])
PY
}
( cd "$REPO" && build_manifest "$SEQ_EVAL_MANIFEST" ) || exit 2
export SEQ_EVAL_MANIFEST_SHA="$("$MOCK_PY" -c "
import json,sys; print(json.load(open(sys.argv[1]))['manifest_sha256'])" "$SEQ_EVAL_MANIFEST")"
N_RECORDS="$("$MOCK_PY" -c "
import json,sys; print(len(json.load(open(sys.argv[1]))['records']))" "$SEQ_EVAL_MANIFEST")"
echo "manifest identity ${SEQ_EVAL_MANIFEST_SHA:0:12}..., ${N_RECORDS} records"

# A tiny checkpoint contract, so mock deltas are bytes not 73 MiB.
cat > "${ROOT}/tiny_contract.json" <<'EOF'
{"contract":"TEST","top_level_key":"unet","n_tensors":2,"total_elements":12,
 "dtype":"torch.float32","expected_file_bytes":null,
 "tensors":{"a.attn2.to_k.weight":{"shape":[2,3],"dtype":"torch.float32"},
            "a.attn2.to_v.weight":{"shape":[2,3],"dtype":"torch.float32"}}}
EOF
export SEQ_CKPT_CONTRACT="${ROOT}/tiny_contract.json"

mkdir -p "${SEQ_BASE_MODEL}/unet"
echo '{"_class_name":"MockPipeline"}' > "${SEQ_BASE_MODEL}/model_index.json"
echo '{"mock":1}'                    > "${SEQ_BASE_MODEL}/unet/config.json"
head -c 128 /dev/zero               > "${SEQ_BASE_MODEL}/unet/w.safetensors"
export SEQ_BASE_MODEL_CONTRACT="${ROOT}/base_model_contract.json"
"$MOCK_PY" - "$SEQ_BASE_MODEL" "$SEQ_BASE_MODEL_CONTRACT" <<'MOCKBM'
import hashlib, json, sys
from pathlib import Path
base, out = Path(sys.argv[1]), Path(sys.argv[2])
cheap = {r: {"sha256": hashlib.sha256((base / r).read_bytes()).hexdigest(),
             "bytes": (base / r).stat().st_size}
         for r in ["model_index.json", "unet/config.json"]}
sizes = {"unet/w.safetensors": (base / "unet/w.safetensors").stat().st_size}
payload = {"cheap_identity_files": cheap, "weight_file_bytes": sizes}
out.write_text(json.dumps({
    "base_model_dir": str(base),
    "cheap_identity_sha256": hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest(),
    "cheap_identity_payload": payload,
    "weight_sha256": {"unet/w.safetensors": hashlib.sha256(
        (base / "unet/w.safetensors").read_bytes()).hexdigest()},
    "m0_policy": {"shared_evaluation": "generated once, counted once"},
}, indent=2))
MOCKBM

# A shared M0 evaluation, built with the REAL generator under the fake models.
mkdir -p "${SEQ}/eval/M0"
( cd "$REPO" && PATH="${BIN}:$PATH" python scripts/seq/generate_eval_images.py \
    --manifest "$SEQ_EVAL_MANIFEST" --expect_manifest_sha "$SEQ_EVAL_MANIFEST_SHA" \
    --base_model_dir "$SEQ_BASE_MODEL" --checkpoint_name M0 \
    --out_dir "${SEQ}/eval/M0/images" --report "${SEQ}/eval/M0/image_report.json" ) \
    > "${ROOT}/m0_gen.log" 2>&1 || { echo "FATAL: M0 generation failed"; cat "${ROOT}/m0_gen.log"; exit 2; }
( cd "$REPO" && PATH="${BIN}:$PATH" python scripts/seq/detect.py \
    --image_report "${SEQ}/eval/M0/image_report.json" --checkpoint_name M0 \
    --out "${SEQ}/eval/M0/detections.jsonl" \
    --report "${SEQ}/eval/M0/detect_report.json" ) >/dev/null 2>&1 || exit 2
echo "shared M0 evaluation built: $(wc -l < "${SEQ}/eval/M0/detections.jsonl") rows"
echo '{"positive_summary":{"cat":{"detection_rate":{"0.5":1.0}},"dog":{"detection_rate":{"0.5":1.0}},"sandwich":{"detection_rate":{"0.5":1.0}}},"negative_control_any_of_seven_rate":{"0.5":0.0}}' > "$SEQ/calib/calibration_report.json"
mkdir -p "$SEQ/eval/_reload_check"
echo '{"n_byte_identical":8,"n_sampled":8,"all_identical":true}' > "$SEQ/eval/_reload_check/report.json"
echo '{"anchor_seed":17}' > "$SEQ/anchor_caches/Horses/cache_meta.json"

run_seed()     { ( cd "$REPO" && PATH="${BIN}:$PATH" bash scripts/seq/run_seed.sh "$1" ) > "$2" 2>&1; }
run_analysis() { ( cd "$REPO" && PATH="${BIN}:$PATH" bash scripts/seq/run_analysis.sh "$1" ) > "$2" 2>&1; }
reset_seed()   { rm -rf "${SEQ}/models/seed$1" "${SEQ}/eval_seed$1" \
                        "${SEQ}/analysis/seed$1" "${SEQ}/logs/seed$1"; }
# Invoke the REAL generator directly, with the fake model modules in front.
gen() { ( cd "$REPO" && PATH="${BIN}:$PATH" python scripts/seq/generate_eval_images.py "$@" ); }

EVALR="${SEQ}/eval_seed29"
MODELS="${SEQ}/models/seed29"

echo
echo "=== 0. baseline: a clean seed-29 run with the real generator ==="
reset_seed 29
MARK=$(( $(calls_mark) + 1 ))   # the shared M0 set was generated during setup
run_seed 29 "${ROOT}/t0.log"; RC=$?
want_rc "clean run succeeds" 0 "$RC"
want_out "every stage validated" "${ROOT}/t0.log" "all stages validated"
want_eq "the real generator produced every image for 5 checkpoints" \
        "$(n_calls pipeline.generate "$MARK")" "$((N_RECORDS * 5))"
OLD_MA_SHA="$("$MOCK_PY" -c "
import hashlib,sys;from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" \
  "${EVALR}/MA/images/cat_literal_0_seed101.jpg")"
cp -r "${EVALR}/MA" "${ROOT}/MA_baseline"

echo
echo "=== 1. valid same-contract resume: images reused, nothing regenerated ==="
# Only the detections are broken. The images provably came from the checkpoint
# still on disk, so they must be kept and only detection re-run.
head -n 4 "${EVALR}/MA/detections.jsonl" > "${ROOT}/tmp.jsonl"
mv "${ROOT}/tmp.jsonl" "${EVALR}/MA/detections.jsonl"
MARK=$(( $(calls_mark) + 1 ))
run_seed 29 "${ROOT}/t1.log"; RC=$?
want_rc "resume succeeds" 0 "$RC"
want_out "launcher attributes the images to the checkpoint on disk" "${ROOT}/t1.log" \
    "images are provably from the checkpoint now on disk"
want_out "generator reports verified-provenance reuse" "${SEQ}/logs/seed29/gen_MA.log" \
    "already present with verified provenance"
want_eq "ZERO images generated during the resume" "$(n_calls pipeline.generate "$MARK")" "0"
want_eq "no model was loaded during the resume" "$(n_calls pipeline.from_pretrained "$MARK")" "0"
NEW_MA_SHA="$("$MOCK_PY" -c "
import hashlib,sys;from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" \
  "${EVALR}/MA/images/cat_literal_0_seed101.jpg")"
want_eq "image bytes untouched" "$NEW_MA_SHA" "$OLD_MA_SHA"
Q=$(find "${EVALR}" -maxdepth 1 -name 'MA.invalid.*' 2>/dev/null | wc -l)
want_eq "the valid evaluation directory was NOT quarantined" "$Q" "0"
Q=$(find "${EVALR}/MA" -name 'detections.jsonl.invalid.*' 2>/dev/null | wc -l)
if [ "$Q" -ge 1 ]; then ok "the invalid detections were quarantined, not deleted"
else bad "quarantine" "no quarantined detections.jsonl found"; fi
want_eq "the reused report records n_reused = every image" \
  "$("$MOCK_PY" -c "
import json,sys; print(json.load(open(sys.argv[1]))['n_reused'])" \
  "${EVALR}/MA/image_report.json")" "$N_RECORDS"

echo
echo "=== 2. changed checkpoint under the same name (launcher recovery path) ==="
# The exact attack: MA's delta.bin is replaced by a DIFFERENT model under the
# same name, its report is made consistent with the new bytes, the old images and
# old image_report are left in place, and the detections are broken so recovery
# runs. The old images must NOT be relabelled as the new checkpoint's.
"$MOCK_PY" - "$MODELS" "$SEQ_CKPT_CONTRACT" <<'RETRAIN'
import hashlib, json, sys, torch
from pathlib import Path
models, contract_p = Path(sys.argv[1]), Path(sys.argv[2])
c = json.loads(contract_p.read_text())
sd = {k: torch.full(tuple(v["shape"]), 0.077) for k, v in c["tensors"].items()}
ck = models / "MA" / "delta.bin"
torch.save({c["top_level_key"]: sd}, ck)
h = hashlib.sha256(ck.read_bytes()).hexdigest()
rp = models / "MA" / "train_report.json"
rep = json.loads(rp.read_text())
rep["child_checkpoint"]["sha256"] = h
rep["child_checkpoint"]["bytes"] = ck.stat().st_size
rp.write_text(json.dumps(rep, indent=2))
# MA's children recorded MA's OLD hash as their parent; drop them so this case
# is about MA's own evaluation and not about lineage.
for child in ("MAB", "MAB_L2", "MAC", "MAC_L2"):
    for f in ("train_report.json", "delta.bin", "_stage_complete.json"):
        (models / child / f).unlink(missing_ok=True)
print("MA replaced in place; new delta", h[:12])
RETRAIN
rm -f "${EVALR}/MA/_stage_complete.json"
head -n 4 "${ROOT}/MA_baseline/detections.jsonl" > "${EVALR}/MA/detections.jsonl"
MARK=$(( $(calls_mark) + 1 ))
run_seed 29 "${ROOT}/t2.log"; RC=$?
want_rc "recovery after a checkpoint swap succeeds" 0 "$RC"
want_out "launcher refuses to attribute the old images" "${ROOT}/t2.log" \
    "images CANNOT be attributed to the checkpoint now on disk"
want_out "the whole evaluation is preserved together" "${ROOT}/t2.log" \
    "preserving the entire evaluation (images and reports"
want_out "the reason names the changed model" "${ROOT}/t2.log" \
    "the model changed under the same name"
QD="$(find "${EVALR}" -maxdepth 1 -name 'MA.invalid.*' -type d 2>/dev/null | head -1)"
if [ -n "$QD" ]; then ok "the old evaluation directory was quarantined whole"
else bad "quarantine" "no quarantined MA evaluation directory"; fi
if [ -n "$QD" ]; then
    if [ -f "${QD}/image_report.json" ] && [ -d "${QD}/images" ]; then
        ok "images AND the old image report are preserved together"
    else bad "preservation" "quarantined directory is missing images or the report"; fi
    QSHA="$("$MOCK_PY" -c "
import hashlib,sys;from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" \
      "${QD}/images/cat_literal_0_seed101.jpg" 2>/dev/null)"
    want_eq "quarantined image bytes are the ORIGINAL bytes" "$QSHA" "$OLD_MA_SHA"
    if [ -f "${QD}.reason.txt" ]; then ok "a reason file accompanies the quarantine"
    else bad "quarantine reason" "no .reason.txt beside the quarantined directory"; fi
fi
FRESH_SHA="$("$MOCK_PY" -c "
import hashlib,sys;from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" \
  "${EVALR}/MA/images/cat_literal_0_seed101.jpg")"
# The fake pipeline's output bytes depend on the loaded checkpoint's digest, so
# different bytes here mean the image really was regenerated from the new model
# rather than carried over.
if [ "$FRESH_SHA" != "$OLD_MA_SHA" ]; then ok "the fresh evaluation holds NEW image bytes"
else bad "relabelling" "the new evaluation still holds the old image bytes"; fi
want_eq "MA's images were actually regenerated" \
        "$(n_calls pipeline.generate "$MARK")" "$N_RECORDS"
want_eq "the fresh report reused nothing" \
  "$("$MOCK_PY" -c "
import json,sys; print(json.load(open(sys.argv[1]))['n_reused'])" \
  "${EVALR}/MA/image_report.json")" "0"
want_eq "the fresh report records the checkpoint now on disk" \
  "$("$MOCK_PY" -c "
import json,sys,hashlib
from pathlib import Path
d=json.load(open(sys.argv[1]))
print(d['checkpoint_load']['sha256'] == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest())" \
  "${EVALR}/MA/image_report.json" "${MODELS}/MA/delta.bin")" "True"

echo
echo "=== 3. the generator refuses BEFORE any model setup ==="
# FAKE_PIPELINE_MUST_NOT_LOAD makes pipeline construction itself an error, so a
# refusal that happened after model setup would be visible as a different
# failure, not a clean exit 3.
PROBE="${ROOT}/probe3"; mkdir -p "${PROBE}/images"
cp -r "${ROOT}/MA_baseline/images/." "${PROBE}/images/"
cp "${ROOT}/MA_baseline/image_report.json" "${PROBE}/image_report.json"
"$MOCK_PY" - "${PROBE}/image_report.json" "${PROBE}/images" <<'REPATH'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); d = json.load(open(p)); newdir = sys.argv[2]
for r in d["images"]:
    r["image_path"] = str(Path(newdir) / r["image_name"])
d["out_dir"] = newdir
p.write_text(json.dumps(d, indent=2))
REPATH
BEFORE="$("$MOCK_PY" -c "
import hashlib,sys;from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" "${PROBE}/image_report.json")"
MARK=$(( $(calls_mark) + 1 ))
FAKE_PIPELINE_MUST_NOT_LOAD=1 gen \
    --manifest "$SEQ_EVAL_MANIFEST" --expect_manifest_sha "$SEQ_EVAL_MANIFEST_SHA" \
    --base_model_dir "$SEQ_BASE_MODEL" --unet_ckpt "${MODELS}/MA/delta.bin" \
    --checkpoint_name MA --out_dir "${PROBE}/images" \
    --report "${PROBE}/image_report.json" > "${ROOT}/t3.log" 2>&1; RC=$?
want_rc "changed checkpoint: generator exits 3" 3 "$RC"
want_out "names the changed model" "${ROOT}/t3.log" "the model changed under the same name"
want_out "says nothing was written" "${ROOT}/t3.log" "Nothing was written"
want_eq "no pipeline was constructed" "$(n_calls pipeline.from_pretrained "$MARK")" "0"
want_eq "no image was generated" "$(n_calls pipeline.generate "$MARK")" "0"
AFTER="$("$MOCK_PY" -c "
import hashlib,sys;from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" "${PROBE}/image_report.json")"
want_eq "the old image report is byte-identical afterwards" "$AFTER" "$BEFORE"

echo
echo "=== 4. changed prompt text, same filenames ==="
build_manifest_at() { ( cd "$REPO" && build_manifest "$@" ) >/dev/null; }
build_manifest_at "${ROOT}/man_prompt.json" " REWRITTEN"
MARK=$(( $(calls_mark) + 1 ))
FAKE_PIPELINE_MUST_NOT_LOAD=1 gen \
    --manifest "${ROOT}/man_prompt.json" \
    --base_model_dir "$SEQ_BASE_MODEL" --unet_ckpt "${ROOT}/MA_baseline_delta.bin" \
    --checkpoint_name MA --out_dir "${ROOT}/MA_baseline/images" \
    --report "${ROOT}/MA_baseline/image_report.json" > "${ROOT}/t4.log" 2>&1; RC=$?
want_rc "a changed prompt set is refused (missing ckpt is fatal first)" 2 "$RC"
# Supply the ORIGINAL checkpoint, so the only change is the prompt text.
cp "${ROOT}/MA_baseline/images/cat_literal_0_seed101.jpg" "${ROOT}/keep.jpg"
"$MOCK_PY" - "$MODELS" "$SEQ_CKPT_CONTRACT" "${ROOT}/orig_delta.bin" <<'ORIG'
import json, sys, torch
from pathlib import Path
c = json.loads(Path(sys.argv[2]).read_text())
sd = {k: torch.full(tuple(v["shape"]), 0.01) for k, v in c["tensors"].items()}
torch.save({c["top_level_key"]: sd}, Path(sys.argv[3]))
ORIG
MARK=$(( $(calls_mark) + 1 ))
FAKE_PIPELINE_MUST_NOT_LOAD=1 gen \
    --manifest "${ROOT}/man_prompt.json" \
    --base_model_dir "$SEQ_BASE_MODEL" --unet_ckpt "${ROOT}/orig_delta.bin" \
    --checkpoint_name MA --out_dir "${ROOT}/MA_baseline/images" \
    --report "${ROOT}/MA_baseline/image_report.json" > "${ROOT}/t4b.log" 2>&1; RC=$?
want_rc "changed prompt text with old filenames: exit 3" 3 "$RC"
want_out "names the manifest identity change" "${ROOT}/t4b.log" \
    "the prompt set or the generation settings changed"
want_eq "no pipeline was constructed" "$(n_calls pipeline.from_pretrained "$MARK")" "0"
# And the frozen-identity gate rejects it even earlier, before provenance.
FAKE_PIPELINE_MUST_NOT_LOAD=1 gen \
    --manifest "${ROOT}/man_prompt.json" --expect_manifest_sha "$SEQ_EVAL_MANIFEST_SHA" \
    --base_model_dir "$SEQ_BASE_MODEL" --unet_ckpt "${ROOT}/orig_delta.bin" \
    --checkpoint_name MA --out_dir "${ROOT}/MA_baseline/images" \
    --report "${ROOT}/MA_baseline/image_report.json" > "${ROOT}/t4c.log" 2>&1; RC=$?
want_rc "the frozen manifest identity gate rejects it first" 2 "$RC"
want_out "names the frozen identity" "${ROOT}/t4c.log" "not the frozen manifest"

echo
echo "=== 5. changed generation settings, same filenames ==="
build_manifest_at "${ROOT}/man_settings.json" "" 50
MARK=$(( $(calls_mark) + 1 ))
FAKE_PIPELINE_MUST_NOT_LOAD=1 gen \
    --manifest "${ROOT}/man_settings.json" \
    --base_model_dir "$SEQ_BASE_MODEL" --unet_ckpt "${ROOT}/orig_delta.bin" \
    --checkpoint_name MA --out_dir "${ROOT}/MA_baseline/images" \
    --report "${ROOT}/MA_baseline/image_report.json" > "${ROOT}/t5.log" 2>&1; RC=$?
want_rc "changed settings with old filenames: exit 3" 3 "$RC"
want_out "names the settings difference" "${ROOT}/t5.log" "generation settings differ"
want_eq "no pipeline was constructed" "$(n_calls pipeline.from_pretrained "$MARK")" "0"

echo
echo "=== 6. missing prior provenance ==="
PROBE6="${ROOT}/probe6"; mkdir -p "${PROBE6}/images"
cp -r "${ROOT}/MA_baseline/images/." "${PROBE6}/images/"
MARK=$(( $(calls_mark) + 1 ))
FAKE_PIPELINE_MUST_NOT_LOAD=1 gen \
    --manifest "$SEQ_EVAL_MANIFEST" --expect_manifest_sha "$SEQ_EVAL_MANIFEST_SHA" \
    --base_model_dir "$SEQ_BASE_MODEL" --unet_ckpt "${ROOT}/orig_delta.bin" \
    --checkpoint_name MA --out_dir "${PROBE6}/images" \
    --report "${PROBE6}/image_report.json" > "${ROOT}/t6.log" 2>&1; RC=$?
want_rc "images with no prior report: exit 3" 3 "$RC"
want_out "says provenance is absent" "${ROOT}/t6.log" "no prior image report"
want_eq "no pipeline was constructed" "$(n_calls pipeline.from_pretrained "$MARK")" "0"
if [ ! -f "${PROBE6}/image_report.json" ]; then ok "no report was written"
else bad "write-nothing" "a report was written despite the refusal"; fi

echo
echo "=== 7. a recorded image digest that no longer matches its bytes ==="
PROBE7="${ROOT}/probe7"; mkdir -p "${PROBE7}/images"
cp -r "${ROOT}/MA_baseline/images/." "${PROBE7}/images/"
cp "${ROOT}/MA_baseline/image_report.json" "${PROBE7}/image_report.json"
"$MOCK_PY" - "${PROBE7}/image_report.json" "${PROBE7}/images" <<'REPATH2'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); d = json.load(open(p)); newdir = sys.argv[2]
for r in d["images"]:
    r["image_path"] = str(Path(newdir) / r["image_name"])
d["out_dir"] = newdir
p.write_text(json.dumps(d, indent=2))
REPATH2
printf 'TAMPERED\n' >> "${PROBE7}/images/dog_literal_0_seed202.jpg"
MARK=$(( $(calls_mark) + 1 ))
FAKE_PIPELINE_MUST_NOT_LOAD=1 gen \
    --manifest "$SEQ_EVAL_MANIFEST" --expect_manifest_sha "$SEQ_EVAL_MANIFEST_SHA" \
    --base_model_dir "$SEQ_BASE_MODEL" --unet_ckpt "${ROOT}/orig_delta.bin" \
    --checkpoint_name MA --out_dir "${PROBE7}/images" \
    --report "${PROBE7}/image_report.json" > "${ROOT}/t7.log" 2>&1; RC=$?
want_rc "tampered image bytes: exit 3" 3 "$RC"
want_out "names the digest mismatch" "${ROOT}/t7.log" \
    "no longer match the digest recorded"
want_eq "no pipeline was constructed" "$(n_calls pipeline.from_pretrained "$MARK")" "0"

echo
echo "=== 8. M0: images claiming a delta are refused ==="
PROBE8="${ROOT}/probe8"; mkdir -p "${PROBE8}/images"
cp -r "${SEQ}/eval/M0/images/." "${PROBE8}/images/"
"$MOCK_PY" - "${SEQ}/eval/M0/image_report.json" "${PROBE8}" <<'M0TAMPER'
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads(src.read_text())
d["checkpoint_load"] = {"applied": True, "unet_ckpt": "/elsewhere/delta.bin",
                        "sha256": "dead" + "b" * 60}
for r in d["images"]:
    r["image_path"] = str(dst / "images" / r["image_name"])
d["out_dir"] = str(dst / "images")
(dst / "image_report.json").write_text(json.dumps(d, indent=2))
M0TAMPER
MARK=$(( $(calls_mark) + 1 ))
FAKE_PIPELINE_MUST_NOT_LOAD=1 gen \
    --manifest "$SEQ_EVAL_MANIFEST" --expect_manifest_sha "$SEQ_EVAL_MANIFEST_SHA" \
    --base_model_dir "$SEQ_BASE_MODEL" --checkpoint_name M0 \
    --out_dir "${PROBE8}/images" --report "${PROBE8}/image_report.json" \
    > "${ROOT}/t8.log" 2>&1; RC=$?
want_rc "an M0 report claiming a delta is refused" 3 "$RC"
want_out "names the delta claim" "${ROOT}/t8.log" "this request applies no delta"

echo
echo "=== 9. analysis rejects the refused state and publishes nothing ==="
# Leave MA's evaluation in the state a refusal produces: images present, report
# gone. Analysis must refuse to aggregate it.
rm -f "${EVALR}/MA/image_report.json" "${EVALR}/MA/_stage_complete.json"
rm -rf "${REPO}/results/seq29_prov_probe" "${SEQ}/analysis/seed29"
run_analysis 29 "${ROOT}/t9.log"; RC=$?
want_rc "analysis refuses to aggregate" 1 "$RC"
want_out "says the evaluations do not validate" "${ROOT}/t9.log" "do not validate"
if [ ! -d "${SEQ}/analysis/seed29" ]; then ok "nothing was published"
else bad "analysis isolation" "analysis/seed29 exists after a refused run"; fi

echo
echo "=== 10. no real GPU work, no repository mutation ==="
if grep -qE 'CUDA_VISIBLE_DEVICES=[0-9]' "$CALL_LOG"; then
    bad "device masking" "a mock saw a numeric CUDA_VISIBLE_DEVICES"
else ok "no mock was invoked with a numeric CUDA device"; fi
if grep -q "REAL generate_eval_images" "$CALL_LOG"; then
    ok "the real generator was exercised (not mocked away)"
else bad "coverage" "the real generator never ran"; fi
if [ "$(n_calls torch.load)" -gt 0 ]; then ok "the fake torch served the delta loads"
else bad "fake wiring" "the fake torch was never used"; fi
BIG=$(find "${SEQ}" -type f -size +200k 2>/dev/null | wc -l)
want_eq "no real-sized artifact was produced" "$BIG" "0"
if [ "${SEQ}" != "/data/bijaypandey/cuig_pilot/seq_pilot" ]; then
    ok "ran against a temporary SEQ_ROOT, not the real /data tree"
else bad "sandbox" "SEQ_ROOT points at the real pilot data"; fi
REPO_AFTER="${ROOT}/repo_after.sha"
# shellcheck disable=SC2086
( cd "$REPO" && find $REPO_AT_RISK -type f -exec sha256sum {} + 2>/dev/null | sort ) \
    > "$REPO_AFTER"
if diff -q "$REPO_SNAPSHOT" "$REPO_AFTER" >/dev/null; then
    ok "committed results/ and configs/ byte-identical after this suite"
else
    bad "repository isolation" "the suite modified a committed tree"
    diff "$REPO_SNAPSHOT" "$REPO_AFTER" | head -10 >&2
fi

echo
echo "-----------------------------------------------------------------"
echo "${PASSN} passed, ${FAILN} failed"
[ "$FAILN" -eq 0 ] || exit 1
echo "ALL IMAGE-PROVENANCE CHECKS PASSED (fake model modules; no GPU, no generation)"
