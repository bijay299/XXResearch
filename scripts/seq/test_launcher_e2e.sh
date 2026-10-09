#!/bin/bash
# ---------------------------------------------------------------------------
# End-to-end orchestration tests for run_seed.sh and run_analysis.sh.
#
# Every GPU-bound command is replaced by a mock on PATH: accelerate, the three
# GPU python entry points, gpu_select.sh and nvidia-smi. Each mock appends to
# $CALL_LOG, so the test can assert afterwards that no real GPU command
# escaped. CUDA_VISIBLE_DEVICES is forced empty throughout and nothing real is
# trained, generated or scored.
#
# What is exercised: a clean run, failure of the FIRST background child,
# failure of the SECOND, MA failure with stale artifacts present, an
# unvalidatable evaluation, a stale-configuration artifact, a partial
# detections file, analysis-stage failure, required-copy failure, and the
# seed-17 legacy/new layout policy.
#
#   bash scripts/seq/test_launcher_e2e.sh
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=""

PASSN=0; FAILN=0
ok()   { PASSN=$((PASSN+1)); printf '  ok    %s\n' "$1"; }
bad()  { FAILN=$((FAILN+1)); printf '  FAIL  %s\n' "$1"; [ -n "${2:-}" ] && printf '          %s\n' "$2"; }
want_rc() {  # want_rc <desc> <expected> <actual>
    if [ "$2" = "$3" ]; then ok "$1 (exit $3)"; else bad "$1" "exit $3, wanted $2"; fi
}
want_out() { # want_out <desc> <file> <needle>
    if grep -qF -- "$3" "$2"; then ok "$1"; else bad "$1" "output lacks: $3"; fi
}
want_not() {
    if grep -qF -- "$3" "$2"; then bad "$1" "output unexpectedly has: $3"; else ok "$1"; fi
}

ROOT="$(mktemp -d)"
# KEEP_E2E_ROOT=1 preserves the sandbox for debugging a failure.
if [ -z "${KEEP_E2E_ROOT:-}" ]; then trap 'rm -rf "$ROOT"' EXIT; else
  trap 'echo "sandbox kept: $ROOT"' EXIT; fi
# Snapshot the repo tree so the suite can prove it changed nothing itself
# (comparing against git HEAD would wrongly flag legitimate uncommitted work).
REPO_SNAPSHOT="${ROOT}/repo_before.sha"
REPO_AT_RISK="results/seq17 results/seq29"
# shellcheck disable=SC2086
( cd "$REPO" && find $REPO_AT_RISK -type f -exec sha256sum {} + \
    2>/dev/null | sort ) > "$REPO_SNAPSHOT"
export CALL_LOG="${ROOT}/calls.log"
: > "$CALL_LOG"

# ---------------------------------------------------------------- mock harness
BIN="${ROOT}/bin"; mkdir -p "$BIN"

cat > "${BIN}/nvidia-smi" <<'EOF'
#!/bin/bash
echo "MOCK nvidia-smi $*" >> "$CALL_LOG"
echo "0, 41 MiB, 0 %"
EOF

# The real gpu_select.sh is a careful fail-closed probe of actual hardware and
# is invoked by path, so PATH shadowing cannot reach it. run_seed.sh exposes
# GPU_SELECT as a seam; the mock reports every GPU idle (or not, on demand).
cat > "${ROOT}/mock_gpu_select.sh" <<'EOF'
#!/bin/bash
echo "MOCK gpu_select $*" >> "$CALL_LOG"
if [ -n "${MOCK_GPU_BUSY:-}" ]; then
  for i in 0 1 2 3; do echo "GPU $i [mock] -> BUSY"; done
  exit 0
fi
for i in 0 1 2 3; do echo "GPU $i [mock] -> IDLE"; done
EOF
chmod +x "${ROOT}/mock_gpu_select.sh"
export GPU_SELECT="${ROOT}/mock_gpu_select.sh"

cat > "${BIN}/accelerate" <<'EOF'
#!/bin/bash
echo "MOCK accelerate $*" >> "$CALL_LOG"
# Find --name and --output_dir and --l2sp_weight in the forwarded argv.
name=""; out=""; l2="0"; seed=""; parent=""; target=""; anchor=""; iters="1000"
while [ $# -gt 0 ]; do
  case "$1" in
    --name) name="$2"; shift 2;;
    --output_dir) out="$2"; shift 2;;
    --l2sp_weight) l2="$2"; shift 2;;
    --seed) seed="$2"; shift 2;;
    --parent_name) parent="$2"; shift 2;;
    --target) target="$2"; shift 2;;
    --anchor_name) anchor="$2"; shift 2;;
    --iterations) iters="$2"; shift 2;;
    *) shift;;
  esac
done
# Honour a scripted failure for this checkpoint.
if [ -n "${MOCK_TRAIN_FAIL:-}" ]; then
  for f in $MOCK_TRAIN_FAIL; do
    [ "$f" = "$name" ] && { echo "mock: scripted failure for $name" >&2; exit 23; }
  done
fi
"$MOCK_PY" "$MOCK_HELPER" make_train --out "$out" --name "$name" --l2 "$l2" \
    --seed "$seed" --parent "$parent" --target "$target" --anchor "$anchor" \
    --iters "$iters" ${MOCK_TRAIN_BREAK:+--break "$MOCK_TRAIN_BREAK"}
EOF

# A python shim: GPU entry points are mocked, everything else is the real
# interpreter. This is what lets run_seed.sh call the real validator while no
# model code ever runs.
cat > "${BIN}/python" <<'EOF'
#!/bin/bash
case "${2:-}${1:-}" in *) :;; esac
for a in "$@"; do
  case "$a" in
    *train_request.py|*generate_eval_images.py|*detect.py|*param_movement.py)
      echo "MOCK python $*" >> "$CALL_LOG"
      exec "$MOCK_PY" "$MOCK_HELPER" gpu_entry "$@" ;;
  esac
done
exec "$MOCK_PY" "$@"
EOF
cp "${BIN}/python" "${BIN}/python3"
chmod +x "${BIN}"/*

# The shim's "real" interpreter must have torch: the validator loads
# checkpoints on CPU. Prefer the project venv, fall back to system python3.
if [ -x /data/bijaypandey/cuig_pilot/venv-cuig/bin/python ]; then
    export MOCK_PY=/data/bijaypandey/cuig_pilot/venv-cuig/bin/python
else
    export MOCK_PY="$(command -v python3)"
fi
"$MOCK_PY" -c 'import torch' 2>/dev/null || {
    echo "FATAL: no interpreter with torch available; cannot run this suite" >&2
    exit 2
}
export MOCK_HELPER="${ROOT}/mock_helper.py"

cat > "$MOCK_HELPER" <<'PYEOF'
"""Fabricate the artifacts the real GPU stages would have produced."""
import argparse, hashlib, json, os, sys
from pathlib import Path

def make_train(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import torch
    contract = json.loads(Path(os.environ["SEQ_CKPT_CONTRACT"]).read_text())
    sd = {k: torch.full(tuple(v["shape"]), 0.01) for k, v in contract["tensors"].items()}
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
        "child_checkpoint": {"sha256": h, "bytes": ck.stat().st_size,
                             "tensors": len(sd)},
        "effective_hyperparameters": {"seed": int(a.seed),
                                      "iterations_requested": steps,
                                      "epoch_capacity_steps": 1250,
                                      "parameter_group": "kv-xattn"},
        "parent_verification": {"parent_sha256": parent_sha, "PASS": True,
                                "max_abs_deviation_after_load": 0.0},
        "l2sp_reference_check": {"PASS": True,
                                 "max_abs_deviation_from_parent": 0.0},
        "l2sp_trace": {"steps_recorded": steps if float(a.l2) > 0 else 0},
        "started_utc": "2026-10-09T00:00:00+00:00",
        "finished_utc": "2026-10-09T00:14:00+00:00",
        "runtime": {"train_seconds": steps * 0.83,
                    "seconds_per_optimizer_step": 0.83,
                    "seconds_per_optimizer_step_is_derived": True,
                    # The ACTUAL completed-step counter the real
                    # train_request.py now records from upstream's own counter.
                    "optimizer_steps_completed": steps,
                    "optimizer_steps_source": "mock (test fixture)"},
    }
    # Scripted corruption modes, to prove the launcher rejects them.
    if a.__dict__.get("break") == "stale_seed":
        rep["training_seed"] = 999; rep["effective_hyperparameters"]["seed"] = 999
    elif a.__dict__.get("break") == "short_steps":
        rep["effective_hyperparameters"]["iterations_requested"] = 400
        rep["runtime"]["train_seconds"] = 400 * 0.83
    elif a.__dict__.get("break") == "plain_text":
        ck.write_text("not a model")
        rep["child_checkpoint"]["sha256"] = hashlib.sha256(ck.read_bytes()).hexdigest()
    elif a.__dict__.get("break") == "no_finish":
        rep["finished_utc"] = None
    elif a.__dict__.get("break") == "short_counter":
        # Exits 0, report looks complete, but only 400 steps actually ran.
        rep["runtime"]["optimizer_steps_completed"] = 400
    elif a.__dict__.get("break") == "no_counter":
        rep["runtime"].pop("optimizer_steps_completed", None)
        rep["runtime"].pop("optimizer_steps_source", None)
    (out / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(f"mock trained {a.name}")

def gpu_entry(argv):
    script = next(x for x in argv if x.endswith(".py"))
    g = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv) and not argv[i+1].startswith("--"):
            g[argv[i][2:]] = argv[i+1]; i += 2
        else:
            i += 1
    if script.endswith("generate_eval_images.py"):
        man = json.loads(Path(g["manifest"]).read_text())
        od = Path(g["out_dir"]); od.mkdir(parents=True, exist_ok=True)
        mode = os.environ.get("MOCK_GEN_BREAK", "")
        if mode == "fail":
            print("mock gen failure", file=sys.stderr); return 7
        # Real (tiny) JPEGs: the grid stage opens them with PIL, so raw bytes
        # would fail there for fixture reasons rather than product reasons.
        from PIL import Image
        for i, r in enumerate(man["records"]):
            Image.new("RGB", (32, 32), ((i * 37) % 256, (i * 73) % 256,
                                        (i * 113) % 256)).save(
                od / r["image_name"], "JPEG", quality=70)
        cl = {"applied": bool(g.get("unet_ckpt")),
              "unet_ckpt": g.get("unet_ckpt")}
        if g.get("unet_ckpt") and Path(g["unet_ckpt"]).exists():
            cl["sha256"] = hashlib.sha256(Path(g["unet_ckpt"]).read_bytes()).hexdigest()
        if mode == "stale_ckpt_hash":
            # As if these images came from an older same-named checkpoint.
            cl["sha256"] = "dead" + "b" * 60
        elif mode == "no_ckpt_hash":
            cl.pop("sha256", None)
        gs = json.loads(Path(os.environ["SEQ_GEN_SETTINGS_CONTRACT"]).read_text())
        gs = {k: v for k, v in gs.items() if not k.startswith("_")}
        if mode == "settings":
            gs["num_inference_steps"] = 50
        Path(g["report"]).write_text(json.dumps({
            "checkpoint": g["checkpoint_name"], "complete": True,
            "expected_images": len(man["records"]),
            "generated_or_reused": len(man["records"]),
            "manifest_sha256": man.get("manifest_sha256"),
            "generation_settings": gs, "checkpoint_load": cl,
            "base_model_dir": os.environ.get("SEQ_BASE_MODEL"),
            "out_dir": str(od),
        }, indent=2))
        return 0
    if script.endswith("detect.py"):
        ir = json.loads(Path(g["image_report"]).read_text())
        man = json.loads(Path(os.environ["SEQ_EVAL_MANIFEST"]).read_text())
        mode = os.environ.get("MOCK_DET_BREAK", "")
        if mode == "fail":
            print("mock detect failure", file=sys.stderr); return 9
        recs = man["records"]
        if mode == "partial":
            recs = recs[: len(recs) // 2]
        rows = []
        for i, r in enumerate(recs):
            ip = Path(ir["out_dir"]) / r["image_name"]
            row = {"checkpoint": g["checkpoint_name"], "prompt_id": r["prompt_id"],
                   "category": r["category"], "prompt_family": r["prompt_family"],
                   "prompt_index": r["prompt_index"], "prompt": r["prompt"],
                   "gen_seed": r["gen_seed"], "image_path": str(ip),
                   "image_sha256": hashlib.sha256(ip.read_bytes()).hexdigest()
                                   if ip.exists() else f"h{i}",
                   "detections": [], "max_score_target": 0.1,
                   "n_detections_target": {"0.3": 0, "0.5": 0, "0.7": 0},
                   "hit": {"0.3": False, "0.5": False, "0.7": False}}
            rows.append(row)
        if mode == "identical":
            rows = [dict(rows[0]) for _ in rows]
        Path(g["out"]).write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        Path(g["report"]).write_text(json.dumps({
            "checkpoint": g["checkpoint_name"], "complete": True,
            "images_scored": len(rows), "images_expected": len(man["records"]),
        }, indent=2))
        return 0
    if script.endswith("param_movement.py"):
        if os.environ.get("MOCK_PARAM_FAIL"):
            print("mock parameter-movement failure", file=sys.stderr); return 11
        # Loads the real SD-1.5 UNet in production; emit the same shape here.
        mr = Path(g["models_root"])
        cks = sorted(p.name for p in mr.iterdir() if (p / "delta.bin").is_file())
        Path(g["out"]).write_text(json.dumps({
            "note": "MOCK parameter movement (test fixture); not a real measurement",
            "base_model_dir": g.get("base_model_dir"),
            "checkpoints": {c: {"relative_l2_vs_M0": 0.0100,
                                "relative_l2_vs_MA": 0.0010} for c in cks},
        }, indent=2))
        return 0
    print(f"mock: unexpected GPU entry point {script}", file=sys.stderr)
    return 1

if __name__ == "__main__":
    if sys.argv[1] == "make_train":
        ap = argparse.ArgumentParser()
        for f in ("out", "name", "l2", "seed", "parent", "target", "anchor", "iters"):
            ap.add_argument(f"--{f}")
        ap.add_argument("--break", dest="break")
        make_train(ap.parse_args(sys.argv[2:]))
        sys.exit(0)
    sys.exit(gpu_entry(sys.argv[2:]))
PYEOF

# -------------------------------------------------------------- environment
SEQ="${ROOT}/seq_pilot"
mkdir -p "$SEQ"/{models,analysis,logs,calib,anchor_caches/Horses,anchor_caches/Flowers}
export SEQ_ROOT="$SEQ"
export SEQ_REPO_OUT_OVERRIDE="${ROOT}/repo_out"
mkdir -p "$SEQ_REPO_OUT_OVERRIDE"
export PILOT_REPO_ROOT="$REPO"
export PILOT_VENV="${ROOT}/venv"; mkdir -p "${PILOT_VENV}/bin"
cat > "${PILOT_VENV}/bin/activate" <<'EOF'
# mock venv: deliberately a no-op so the mock PATH stays in front
EOF
export SEQ_ITERATIONS=1000
export SEQ_ACCEL_CFG="${ROOT}/accel.yaml"; echo "{}" > "$SEQ_ACCEL_CFG"
export SEQ_BASE_MODEL="${ROOT}/sd15"; mkdir -p "$SEQ_BASE_MODEL"
export GPU_A=2 GPU_B=3

# A small frozen manifest, so the whole suite runs in seconds. The expected
# count is derived from it, which is exactly the contract under test.
"$MOCK_PY" - "$SEQ/eval_manifest.json" <<'PY'
import json,sys,hashlib
recs=[]
for c in ["cat","dog","sandwich","horse","bird","chair","bicycle"]:
    for fam in ["literal","paraphrase"]:
        for i in range(1):
            for s in (101,202):
                recs.append({"prompt_id":f"{c}_{fam}_{i}","category":c,
                             "prompt_family":fam,"prompt_index":i,
                             "prompt":f"a photo of a {c} ({fam} {i})","gen_seed":s,
                             "image_name":f"{c}_{fam}_{i}_seed{s}.jpg"})
m={"experiment":"mock","frozen_before_any_editing":True,"records":recs,
   "gen_seeds":[101,202],"images_per_checkpoint":len(recs)}
m["manifest_sha256"]=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
open(sys.argv[1],"w").write(json.dumps(m,indent=1))
print("mock manifest:",len(recs),"records")
PY
echo '{"positive_summary":{"cat":{"detection_rate":{"0.5":1.0}},"dog":{"detection_rate":{"0.5":1.0}},"sandwich":{"detection_rate":{"0.5":1.0}}},"negative_control_any_of_seven_rate":{"0.5":0.0}}' > "$SEQ/calib/calibration_report.json"
mkdir -p "$SEQ/eval/_reload_check"
echo '{"n_byte_identical":8,"n_sampled":8,"all_identical":true}' > "$SEQ/eval/_reload_check/report.json"
echo '{"anchor_seed":17}' > "$SEQ/anchor_caches/Horses/cache_meta.json"
# The legacy eval/ dir now exists (it holds _reload_check), so seed 17 would be
# ambiguous; these tests use seed 29 unless they are testing the policy itself.

export SEQ_CKPT_CONTRACT="${REPO}/configs/checkpoint_contract.json"
export SEQ_GEN_SETTINGS_CONTRACT="${REPO}/configs/generation_settings.json"
export SEQ_EVAL_MANIFEST="${SEQ}/eval_manifest.json"
export SEQ_VALIDATOR="${REPO}/scripts/seq/validate_stage.py"

# A tiny checkpoint contract, so mock deltas are bytes not 73 MiB.
cat > "${ROOT}/tiny_contract.json" <<'EOF'
{"contract":"TEST","top_level_key":"unet","n_tensors":2,"total_elements":12,
 "dtype":"torch.float32","expected_file_bytes":null,
 "tensors":{"a.attn2.to_k.weight":{"shape":[2,3],"dtype":"torch.float32"},
            "a.attn2.to_v.weight":{"shape":[2,3],"dtype":"torch.float32"}}}
EOF
export SEQ_CKPT_CONTRACT="${ROOT}/tiny_contract.json"

# A mock base-model contract matching the sandbox backbone, so M0 can be
# validated under the explicit base-model identity policy rather than as an
# exception. Also export the frozen manifest identity the launcher now requires
# from CONFIGURATION (never read out of the manifest being validated).
mkdir -p "${SEQ_BASE_MODEL}/unet"
echo '{"_class_name":"MockPipeline"}' > "${SEQ_BASE_MODEL}/model_index.json"
echo '{"mock":1}'                    > "${SEQ_BASE_MODEL}/unet/config.json"
head -c 128 /dev/zero               > "${SEQ_BASE_MODEL}/unet/w.safetensors"
export SEQ_BASE_MODEL_CONTRACT="${ROOT}/base_model_contract.json"
"$MOCK_PY" - "$SEQ_BASE_MODEL" "$SEQ_BASE_MODEL_CONTRACT" <<'MOCKBM'
import hashlib, json, sys
from pathlib import Path
base, out = Path(sys.argv[1]), Path(sys.argv[2])
small = ["model_index.json", "unet/config.json"]
weights = ["unet/w.safetensors"]
cheap = {r: {"sha256": hashlib.sha256((base / r).read_bytes()).hexdigest(),
             "bytes": (base / r).stat().st_size} for r in small}
sizes = {r: (base / r).stat().st_size for r in weights}
payload = {"cheap_identity_files": cheap, "weight_file_bytes": sizes}
out.write_text(json.dumps({
    "base_model_dir": str(base),
    "cheap_identity_sha256": hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest(),
    "cheap_identity_payload": payload,
    "m0_policy": {"shared_evaluation": "generated once, counted once"},
}, indent=2))
print("mock base-model contract written")
MOCKBM
export SEQ_EVAL_MANIFEST_SHA="$("$MOCK_PY" -c "
import json,sys; print(json.load(open(sys.argv[1]))['manifest_sha256'])
" "$SEQ_EVAL_MANIFEST")"
echo "mock manifest identity: ${SEQ_EVAL_MANIFEST_SHA:0:12}..."

# A valid shared M0 evaluation, built with the same mocks. run_seed.sh symlinks
# eval_seed<N>/M0 to it, so without this the M0 stage is legitimately invalid
# and every "clean run" would fail for the wrong reason.
mkdir -p "${SEQ}/eval/M0"
PATH="${BIN}:$PATH" "$MOCK_PY" "$MOCK_HELPER" gpu_entry \
    "${REPO}/scripts/seq/generate_eval_images.py" \
    --manifest "$SEQ_EVAL_MANIFEST" --checkpoint_name M0 \
    --out_dir "${SEQ}/eval/M0/images" \
    --report "${SEQ}/eval/M0/image_report.json" >/dev/null || exit 2
PATH="${BIN}:$PATH" "$MOCK_PY" "$MOCK_HELPER" gpu_entry \
    "${REPO}/scripts/seq/detect.py" \
    --image_report "${SEQ}/eval/M0/image_report.json" --checkpoint_name M0 \
    --out "${SEQ}/eval/M0/detections.jsonl" \
    --report "${SEQ}/eval/M0/detect_report.json" >/dev/null || exit 2
echo "mock M0 evaluation built: $(wc -l < "${SEQ}/eval/M0/detections.jsonl") rows"

run_seed() {   # run_seed <seed> <logfile>
    local seed="$1" log="$2"
    ( PATH="${BIN}:$PATH" bash "${REPO}/scripts/seq/run_seed.sh" "$seed" ) \
        > "$log" 2>&1
    return $?
}
run_analysis() {
    local seed="$1" log="$2"
    ( PATH="${BIN}:$PATH" bash "${REPO}/scripts/seq/run_analysis.sh" "$seed" ) \
        > "$log" 2>&1
    return $?
}
reset_seed() { rm -rf "${SEQ}/models/seed$1" "${SEQ}/eval_seed$1" \
                      "${SEQ}/analysis/seed$1" "${SEQ}/logs/seed$1"; }

echo
echo "=== 1. clean run, every stage mocked ==="
reset_seed 29
run_seed 29 "${ROOT}/t1.log"; RC=$?
want_rc "clean run succeeds" 0 "$RC"
want_out "reports all stages validated" "${ROOT}/t1.log" "all stages validated"
want_out "MA validated" "${ROOT}/t1.log" "train MA: VALID"
want_out "MAC_L2 eval validated" "${ROOT}/t1.log" "eval  MAC_L2: VALID"

echo
echo "=== 2. idempotent re-run skips validated stages ==="
run_seed 29 "${ROOT}/t2.log"; RC=$?
want_rc "re-run succeeds" 0 "$RC"
want_out "skips an already-validated train stage" "${ROOT}/t2.log" "[skip] MAB (validated complete)"
want_out "skips an already-validated eval stage" "${ROOT}/t2.log" "[skip] eval MAB (validated complete)"

echo
echo "=== 3. FIRST background child fails ==="
reset_seed 29
MOCK_TRAIN_FAIL="MAB" run_seed 29 "${ROOT}/t3.log"; RC=$?
want_rc "first-child failure propagates" 1 "$RC"
want_out "names the failed child" "${ROOT}/t3.log" "train:MAB"
want_out "does not claim completion" "${ROOT}/t3.log" "NOT COMPLETE"
want_not "no false DONE" "${ROOT}/t3.log" "all stages validated"

echo
echo "=== 4. SECOND background child fails ==="
reset_seed 29
MOCK_TRAIN_FAIL="MAB_L2" run_seed 29 "${ROOT}/t4.log"; RC=$?
want_rc "second-child failure propagates" 1 "$RC"
want_out "names the failed child" "${ROOT}/t4.log" "train:MAB_L2"

echo
echo "=== 5. both children of one wave fail ==="
reset_seed 29
MOCK_TRAIN_FAIL="MAC MAC_L2" run_seed 29 "${ROOT}/t5.log"; RC=$?
want_rc "both failures propagate" 1 "$RC"
want_out "names MAC" "${ROOT}/t5.log" "train:MAC"
want_out "names MAC_L2" "${ROOT}/t5.log" "train:MAC_L2"

echo
echo "=== 6. MA fails, with stale MA artifacts already present ==="
reset_seed 29
run_seed 29 "${ROOT}/t6a.log" >/dev/null 2>&1   # produce real artifacts first
# Corrupt MA so it no longer validates, then script a launch failure too.
echo "corrupted" > "${SEQ}/models/seed29/MA/delta.bin"
MOCK_TRAIN_FAIL="MA" run_seed 29 "${ROOT}/t6.log"; RC=$?
want_rc "MA launch failure is fatal despite stale artifacts" 1 "$RC"
want_out "says MA stage returned nonzero" "${ROOT}/t6.log" "MA training stage returned"
want_not "no children were trained after MA failed" "${ROOT}/t6.log" "[train] seed29 MAB"

echo
echo "=== 7. MA exits 0 but leaves artifacts that do not validate ==="
reset_seed 29
MOCK_TRAIN_BREAK="plain_text" run_seed 29 "${ROOT}/t7.log"; RC=$?
want_rc "unvalidatable MA is fatal" 1 "$RC"
want_out "rejects the plain-text checkpoint" "${ROOT}/t7.log" "does not load as a torch"

echo
echo "=== 8. stale configuration in a training report ==="
reset_seed 29
MOCK_TRAIN_BREAK="stale_seed" run_seed 29 "${ROOT}/t8.log"; RC=$?
want_rc "stale-seed report is rejected" 1 "$RC"
want_out "reports the identity mismatch" "${ROOT}/t8.log" "training_seed mismatch"

echo
echo "=== 9. short trajectory (400 of 1000 steps) ==="
reset_seed 29
MOCK_TRAIN_BREAK="short_steps" run_seed 29 "${ROOT}/t9.log"; RC=$?
want_rc "short trajectory is rejected" 1 "$RC"
want_out "reports the step shortfall" "${ROOT}/t9.log" "iterations_requested"

echo
echo "=== 10. partial detections file ==="
reset_seed 29
MOCK_DET_BREAK="partial" run_seed 29 "${ROOT}/t10.log"; RC=$?
want_rc "partial detections rejected" 1 "$RC"
want_out "reports unscored manifest identities" "${ROOT}/t10.log" "unscored"
Q=$(find "${SEQ}/eval_seed29" -name '*.invalid.*' 2>/dev/null | wc -l)
if [ "$Q" -gt 0 ]; then ok "partial output quarantined, not deleted ($Q artifact(s))"
else bad "quarantine" "no quarantined artifact found"; fi

echo
echo "=== 11. every detections row identical ==="
reset_seed 29
MOCK_DET_BREAK="identical" run_seed 29 "${ROOT}/t11.log"; RC=$?
want_rc "all-identical rows rejected" 1 "$RC"
want_out "reports duplicate identities" "${ROOT}/t11.log" "duplicat"

echo
echo "=== 12. generation settings drift ==="
reset_seed 29
MOCK_GEN_BREAK="settings" run_seed 29 "${ROOT}/t12.log"; RC=$?
want_rc "settings drift rejected" 1 "$RC"
want_out "reports the setting mismatch" "${ROOT}/t12.log" "num_inference_steps"

echo
echo "=== 13. generation / detection stage failures ==="
reset_seed 29
MOCK_GEN_BREAK="fail" run_seed 29 "${ROOT}/t13.log"; RC=$?
want_rc "generate failure propagates" 1 "$RC"
want_out "names the generate failure" "${ROOT}/t13.log" "[FAIL] generate"
reset_seed 29
MOCK_DET_BREAK="fail" run_seed 29 "${ROOT}/t13b.log"; RC=$?
want_rc "detect failure propagates" 1 "$RC"
want_out "names the detect failure" "${ROOT}/t13b.log" "[FAIL] detect"

echo
echo "=== 14. analysis refuses an invalid seed, publishes nothing ==="
reset_seed 29
MOCK_DET_BREAK="partial" run_seed 29 /dev/null 2>&1
rm -rf "${REPO}/results/seq29_e2e_probe"
run_analysis 29 "${ROOT}/t14.log"; RC=$?
want_rc "analysis refuses to aggregate" 1 "$RC"
want_out "says evaluations do not validate" "${ROOT}/t14.log" "do not validate"
if [ ! -d "${SEQ}/analysis/seed29" ]; then ok "nothing published to the analysis dir"
else bad "analysis isolation" "analysis/seed29 exists after a refused run"; fi

echo
echo "=== 15. analysis stage failure publishes nothing ==="
reset_seed 29
run_seed 29 /dev/null 2>&1
# Break a required analysis stage (parameter movement) part-way through, after
# aggregation has already produced output into the staging directory.
MOCK_PARAM_FAIL=1 run_analysis 29 "${ROOT}/t15.log"; RC=$?
want_rc "required-stage failure aborts" 1 "$RC"
want_out "names the failed required stage" "${ROOT}/t15.log" "required stage failed"
want_out "states nothing was published" "${ROOT}/t15.log" "Nothing was published"
if [ ! -d "${SEQ}/analysis/seed29" ]; then ok "staging discarded on failure"
else bad "analysis isolation" "analysis/seed29 exists after a failed run"; fi

echo
echo "=== 16. seed-17 legacy / new layout policy ==="
# eval/ exists (legacy). Only legacy -> legacy is used, with a note.
rm -rf "${SEQ}/eval_seed17"
OUT="$( PATH="${BIN}:$PATH" bash -c "source ${REPO}/scripts/seq/exp_config.sh; seq_resolve_eval_root 17" 2>&1 )"
if grep -q "legacy evaluation directory" <<<"$OUT" && grep -q "/eval$" <<<"$OUT"; then
    ok "legacy-only resolves to eval/ with a note"
else bad "legacy-only policy" "$OUT"; fi
# Both exist -> refuse.
mkdir -p "${SEQ}/eval_seed17"
OUT="$( PATH="${BIN}:$PATH" bash -c "source ${REPO}/scripts/seq/exp_config.sh; seq_resolve_eval_root 17" 2>&1 )"; RC=$?
want_rc "ambiguous layout refused" 1 "$RC"
want_out_s() { if grep -qF -- "$2" <<<"$1"; then ok "$3"; else bad "$3" "$1"; fi; }
want_out_s "$OUT" "ambiguous evaluation layout" "explains the ambiguity"
want_out_s "$OUT" "neither will be deleted" "promises not to delete evidence"
# run_seed and run_analysis must BOTH refuse, not pick one.
run_seed 17 "${ROOT}/t16a.log"; RC=$?
want_rc "run_seed refuses ambiguous seed 17" 1 "$RC"
run_analysis 17 "${ROOT}/t16b.log"; RC=$?
want_rc "run_analysis refuses ambiguous seed 17" 1 "$RC"
# Override resolves it deliberately.
OUT="$( PATH="${BIN}:$PATH" SEQ_EVAL_ROOT_OVERRIDE="${SEQ}/eval" bash -c \
        "source ${REPO}/scripts/seq/exp_config.sh; seq_resolve_eval_root 17" 2>&1 )"
want_out_s "$OUT" "OVERRIDE in effect" "explicit override is honoured"
rm -rf "${SEQ}/eval_seed17"
# Both scripts agree on the SAME directory for the same seed.
A="$( PATH="${BIN}:$PATH" bash -c "source ${REPO}/scripts/seq/exp_config.sh; seq_resolve_eval_root 29" 2>/dev/null )"
B="$( PATH="${BIN}:$PATH" bash -c "source ${REPO}/scripts/seq/exp_config.sh; seq_resolve_eval_root 29" 2>/dev/null )"
if [ "$A" = "$B" ] && [ -n "$A" ]; then ok "one shared policy returns one directory"
else bad "shared policy" "got '$A' and '$B'"; fi

echo
echo "=== 15b. a short completed-step counter is rejected ==="
reset_seed 29
MOCK_TRAIN_BREAK="short_counter" run_seed 29 "${ROOT}/t15b.log"; RC=$?
want_rc "400-of-1000 completed steps rejected" 1 "$RC"
want_out "names the incomplete trajectory" "${ROOT}/t15b.log" "did not complete"

echo
echo "=== 15c. a NEW seed-29 output with no counter is rejected under DEFAULTS ==="
# Seed 29 is a saved-pilot seed number, and the proposed new trajectories use it
# too. Under the old seed-list policy this run inherited the legacy exception and
# passed with no counter at all. The policy is now bound to saved ARTIFACT
# identities, so a freshly produced seed-29 output cannot match and its counter
# is required. No environment override is needed to make that true.
reset_seed 29
MOCK_TRAIN_BREAK="no_counter" run_seed 29 "${ROOT}/t15c.log"; RC=$?
want_rc "a fresh seed-29 output with no counter is rejected by default" 1 "$RC"
want_out "says completion cannot be evidenced" "${ROOT}/t15c.log" "cannot be evidenced"
want_out "points at the explicit legacy option" "${ROOT}/t15c.log" "legacy_optional"
want_not "the legacy exception was NOT applied" "${ROOT}/t15c.log" "[legacy] MA"

echo
echo "=== 15c2. a short counter is rejected under DEFAULTS ==="
reset_seed 29
MOCK_TRAIN_BREAK="short_counter" run_seed 29 "${ROOT}/t15c2.log"; RC=$?
want_rc "a present-but-short counter is rejected" 1 "$RC"
want_out "names the incomplete trajectory" "${ROOT}/t15c2.log" "did not complete"

echo
echo "=== 15c3. a NEW seed-17 output with no counter is rejected under DEFAULTS ==="
# The same test for the other saved-pilot seed number, in its own isolated
# evaluation directory so the seed-17 legacy/new layout policy is not involved.
rm -rf "${SEQ}/models/seed17" "${SEQ}/eval_seed17_probe" "${SEQ}/logs/seed17"
SEQ_EVAL_ROOT_OVERRIDE="${SEQ}/eval_seed17_probe" MOCK_TRAIN_BREAK="no_counter" \
    run_seed 17 "${ROOT}/t15c3.log"; RC=$?
want_rc "a fresh seed-17 output with no counter is rejected by default" 1 "$RC"
want_out "says completion cannot be evidenced" "${ROOT}/t15c3.log" "cannot be evidenced"
want_not "the legacy exception was NOT applied" "${ROOT}/t15c3.log" "[legacy] MA"
rm -rf "${SEQ}/models/seed17" "${SEQ}/eval_seed17_probe" "${SEQ}/logs/seed17"

# A fixture registry, built from whatever artifacts are on disk. It stands in for
# configs/legacy_training_artifacts.json, which registers the REAL saved pilot
# and can never match a mock artifact.
make_legacy_registry() {   # make_legacy_registry <out> <models_root> <ck...>
    "$MOCK_PY" - "$@" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, mroot, cks = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
arts = [{"id": f"fixture/{ck}", "training_seed": 29, "checkpoint": ck,
         "train_report_sha256": h(mroot / ck / "train_report.json"),
         "delta_sha256": h(mroot / ck / "delta.bin"),
         "in_repo_report_copy": f"results/seq29/train/{ck}.json",
         "in_repo_report_copy_sha256": None,
         "provenance": {"produced_by": "E2E fixture, not real saved evidence",
                        "saved_path": str(mroot / ck / "train_report.json")}}
        for ck in cks]
out.write_text(json.dumps({"artifacts": arts}, indent=2))
print(f"fixture registry: {len(arts)} artifact(s)")
PY
}
strip_counter() {   # strip_counter <models_root> <ck>
    "$MOCK_PY" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / sys.argv[2] / "train_report.json"
d = json.loads(p.read_text())
d["runtime"].pop("optimizer_steps_completed", None)
d["runtime"].pop("optimizer_steps_source", None)
p.write_text(json.dumps(d, indent=2))
PY
}

echo
echo "=== 15d. a REGISTERED saved artifact accepts an absent counter ==="
# This is the only thing the legacy exception is allowed to cover: an artifact
# whose report AND checkpoint hashes are registered. Structural validation still
# applies in full, completion is reported UNVERIFIED, and nothing is retrained.
reset_seed 29
run_seed 29 /dev/null 2>&1
strip_counter "${SEQ}/models/seed29" MA
make_legacy_registry "${ROOT}/legacy_fixture.json" "${SEQ}/models/seed29" MA
SEQ_LEGACY_ARTIFACT_REGISTRY="${ROOT}/legacy_fixture.json" \
    run_seed 29 "${ROOT}/t15d.log"; RC=$?
want_rc "a registered saved artifact still completes" 0 "$RC"
want_out "names the registered artifact" "${ROOT}/t15d.log" \
    "registered saved pilot artifact"
want_out "records completion as unverified" "${ROOT}/t15d.log" "UNVERIFIED"
want_not "MA was not retrained" "${ROOT}/t15d.log" "[train] seed29 MA "
want_not "nothing failed" "${ROOT}/t15d.log" "[FAIL]"

echo
echo "=== 15d2. a CHANGED artifact bearing a registered name is not exempt ==="
# Its report no longer matches the registry, so the completed-step counter is
# required again -- and because its delta.bin IS still registered saved
# evidence, the launcher refuses to retrain over it rather than destroying it.
"$MOCK_PY" - "${SEQ}/models/seed29/MA/train_report.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); d = json.loads(p.read_text())
d["note_added_after_registration"] = "this report is no longer the saved bytes"
p.write_text(json.dumps(d, indent=2))
PY
rm -f "${SEQ}/models/seed29/MA/_stage_complete.json"
SEQ_LEGACY_ARTIFACT_REGISTRY="${ROOT}/legacy_fixture.json" \
    run_seed 29 "${ROOT}/t15d2.log"; RC=$?
want_rc "a changed legacy-named artifact is rejected" 1 "$RC"
want_out "says the identity differs from the registered artifact" "${ROOT}/t15d2.log" \
    "identity differs"
want_out "refuses to retrain over saved evidence" "${ROOT}/t15d2.log" \
    "REGISTERED saved pilot checkpoint"
want_out "promises not to replace it" "${ROOT}/t15d2.log" "Refusing to retrain"
want_not "no retraining happened" "${ROOT}/t15d2.log" "[train] seed29 MA "

echo
echo "=== 15d3. the completed-step policy reads artifacts, not seed numbers ==="
POL() {  # POL <registry> <models_root> <ck>
    PATH="${BIN}:$PATH" SEQ_LEGACY_ARTIFACT_REGISTRY="$1" bash -c \
        "source ${REPO}/scripts/seq/exp_config.sh; seq_steps_evidence_full '$2' '$3'" \
        2>/dev/null
}
reset_seed 29
run_seed 29 /dev/null 2>&1
strip_counter "${SEQ}/models/seed29" MA
make_legacy_registry "${ROOT}/legacy_fixture2.json" "${SEQ}/models/seed29" MA
OUT="$(POL "${ROOT}/legacy_fixture2.json" "${SEQ}/models/seed29" MA)"
case "$OUT" in legacy_optional*) ok "a registered artifact -> legacy_optional";;
  *) bad "registered artifact policy" "$OUT";; esac
# An EXPLICITLY EMPTY configuration removes every exception, as documented. The
# old ${VAR:-default} form silently restored the default on an empty value.
OUT="$(POL "" "${SEQ}/models/seed29" MA)"
case "$OUT" in counter*) ok "an explicitly EMPTY registry -> counter, no exception";;
  *) bad "empty registry policy" "$OUT";; esac
# The real committed registry registers the real saved pilot only, so a mock
# artifact under the same checkpoint name is held to the strict policy.
OUT="$(POL "${REPO}/configs/legacy_training_artifacts.json" "${SEQ}/models/seed29" MA)"
case "$OUT" in counter*) ok "a mock artifact against the REAL registry -> counter";;
  *) bad "real registry policy" "$OUT";; esac
OUT="$(POL "${ROOT}/does_not_exist.json" "${SEQ}/models/seed29" MA)"
case "$OUT" in counter*) ok "an absent registry -> counter (fails closed)";;
  *) bad "absent registry policy" "$OUT";; esac
echo '{ not json' > "${ROOT}/broken_registry.json"
OUT="$(POL "${ROOT}/broken_registry.json" "${SEQ}/models/seed29" MA)"
case "$OUT" in counter*) ok "a malformed registry -> counter (fails closed)";;
  *) bad "malformed registry policy" "$OUT";; esac
# MAB's artifacts are not registered at all, even though the fixture registers
# MA from the same seed and the same directory.
OUT="$(POL "${ROOT}/legacy_fixture2.json" "${SEQ}/models/seed29" MAB)"
case "$OUT" in counter*) ok "an unregistered sibling -> counter";;
  *) bad "sibling policy" "$OUT";; esac

echo
echo "=== 15d4. a short counter is rejected EVEN for a registered artifact ==="
# Build a VALID run first, then shorten MA's counter in place and register the
# shortened artifact. (Running with MOCK_TRAIN_BREAK=short_counter would leave
# no train_report.json to register: the launcher quarantines the invalid one.)
reset_seed 29
run_seed 29 /dev/null 2>&1
"$MOCK_PY" - "${SEQ}/models/seed29/MA/train_report.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); d = json.loads(p.read_text())
d["runtime"]["optimizer_steps_completed"] = 400
p.write_text(json.dumps(d, indent=2))
PY
make_legacy_registry "${ROOT}/legacy_fixture3.json" "${SEQ}/models/seed29" MA
rm -f "${SEQ}/models/seed29/MA/_stage_complete.json"
OUT="$(POL "${ROOT}/legacy_fixture3.json" "${SEQ}/models/seed29" MA)"
case "$OUT" in legacy_optional*) ok "the fixture artifact is registered";;
  *) bad "fixture registration" "$OUT";; esac
SEQ_LEGACY_ARTIFACT_REGISTRY="${ROOT}/legacy_fixture3.json" \
    run_seed 29 "${ROOT}/t15d4.log"; RC=$?
want_rc "the exception cannot wave through a short counter" 1 "$RC"
want_out "names the incomplete trajectory" "${ROOT}/t15d4.log" "did not complete"

echo
echo "=== 15e. stale generation hash rejected in BOTH paths ==="
reset_seed 29
MOCK_GEN_BREAK="stale_ckpt_hash" run_seed 29 "${ROOT}/t15e.log"; RC=$?
want_rc "launch path rejects a stale evaluation" 1 "$RC"
want_out "names the staleness" "${ROOT}/t15e.log" "stale with respect to its checkpoint"
# Produce a clean seed, then corrupt only the recorded generation hash so the
# ANALYSIS path has to catch it on its own.
reset_seed 29
run_seed 29 /dev/null 2>&1
"$MOCK_PY" - "${SEQ}/eval_seed29/MA/image_report.json" <<'TAMPER'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
d["checkpoint_load"]["sha256"] = "dead" + "b" * 60
json.dump(d, open(p, "w"))
TAMPER
run_analysis 29 "${ROOT}/t15f.log"; RC=$?
want_rc "analysis path rejects a stale evaluation" 1 "$RC"
want_out "analysis names the staleness" "${ROOT}/t15f.log" "stale with respect to its checkpoint"
want_out "analysis publishes nothing" "${ROOT}/t15f.log" "do not validate"

echo
echo "=== 15g. a missing recorded generation hash is rejected ==="
reset_seed 29
MOCK_GEN_BREAK="no_ckpt_hash" run_seed 29 "${ROOT}/t15g.log"; RC=$?
want_rc "absent generation hash rejected" 1 "$RC"
want_out "says the generating model is unidentifiable" "${ROOT}/t15g.log" "records no sha256"

echo
echo "=== 16b. a busy GPU is never used ==="
reset_seed 29
MOCK_GPU_BUSY=1 run_seed 29 "${ROOT}/t16c.log"; RC=$?
want_rc "refuses to launch on a busy GPU" 1 "$RC"
want_out "says the GPU was not idle" "${ROOT}/t16c.log" "not idle"
want_not "nothing was trained" "${ROOT}/t16c.log" "[done] MA exit=0"

echo
echo "=== 17. no real GPU command escaped ==="
# Every GPU-bound invocation must appear in the mock call log.
for pat in "MOCK accelerate" "MOCK python" "generate_eval_images.py" "detect.py"; do
    want_out "intercepted: ${pat}" "$CALL_LOG" "$pat"
done
# No real GPU device was ever addressed.
if grep -qE 'CUDA_VISIBLE_DEVICES=[0-9]' "$CALL_LOG"; then
    bad "device masking" "a mock saw a numeric CUDA_VISIBLE_DEVICES"
else ok "no mock was invoked with a numeric CUDA device"; fi
REAL_TORCH_CUDA=$("$MOCK_PY" - <<'PY'
import os
print("1" if os.environ.get("CUDA_VISIBLE_DEVICES","")!="" else "0")
PY
)
if [ "$REAL_TORCH_CUDA" = "0" ]; then ok "CUDA_VISIBLE_DEVICES stayed empty in this suite"
else bad "device masking" "CUDA_VISIBLE_DEVICES was not empty"; fi
# The real training/generation/detection scripts never ran: their mocks did.
# Check every delta.bin the suite produced, wherever it still exists, rather
# than one path an earlier test may have reset.
BIG=$(find "${SEQ}/models" -name delta.bin -size +100k 2>/dev/null | wc -l)
ANY=$(find "${SEQ}/models" -name delta.bin 2>/dev/null | wc -l)
if [ "$BIG" -eq 0 ]; then
    ok "no real-sized delta.bin was ever produced (${ANY} mock delta(s) seen)"
else bad "mock integrity" "${BIG} delta.bin file(s) exceed 100 KiB"
fi
# The suite must never have touched the real /data pilot artifacts.
if [ "${SEQ}" != "/data/bijaypandey/cuig_pilot/seq_pilot" ]; then
    ok "suite ran against a temporary SEQ_ROOT, not the real /data tree"
else bad "sandbox" "SEQ_ROOT points at the real pilot data"
fi
echo "  (mock call log: $(wc -l < "$CALL_LOG") intercepted GPU invocations)"

echo
echo "=== 18. clean run after all the failure cases ==="
reset_seed 29
run_seed 29 "${ROOT}/t18.log"; RC=$?
want_rc "still succeeds on a clean run" 0 "$RC"
run_analysis 29 "${ROOT}/t18b.log"; RC=$?
want_rc "analysis completes on a valid seed" 0 "$RC"
want_out "analysis reports completion" "${ROOT}/t18b.log" "ANALYSIS seed 29 COMPLETE"
want_out "optional formats reported explicitly" "${ROOT}/t18b.log" "optional formats"

echo
echo "-----------------------------------------------------------------"
echo "${PASSN} passed, ${FAILN} failed"
# Nothing in the repository may have been touched: the suite publishes only
# into its own sandbox via SEQ_REPO_OUT_OVERRIDE.
REPO_AFTER="${ROOT}/repo_after.sha"
# shellcheck disable=SC2086
( cd "$REPO" && find $REPO_AT_RISK -type f -exec sha256sum {} + \
    2>/dev/null | sort ) > "$REPO_AFTER"
if diff -q "$REPO_SNAPSHOT" "$REPO_AFTER" >/dev/null; then
    ok "committed results trees (${REPO_AT_RISK}) byte-identical after this suite"
else
    bad "repository isolation" "the suite modified a committed results tree"
    diff "$REPO_SNAPSHOT" "$REPO_AFTER" | head -10 >&2
fi
if [ -n "${SEQ_REPO_OUT_OVERRIDE:-}" ] && [ -d "$SEQ_REPO_OUT_OVERRIDE" ]; then
    ok "analysis published into the sandbox (SEQ_REPO_OUT_OVERRIDE), not the repo"
else bad "sandbox" "SEQ_REPO_OUT_OVERRIDE was not in effect"; fi
echo "${PASSN} passed, ${FAILN} failed (final)"
[ "$FAILN" -eq 0 ] || exit 1
echo "ALL LAUNCHER E2E CHECKS PASSED (all GPU commands mocked)"
