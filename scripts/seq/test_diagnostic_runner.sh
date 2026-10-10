#!/bin/bash
# ---------------------------------------------------------------------------
# CPU test of the matched-effectiveness diagnostic runner.
#
# Run BEFORE committing any of the approved four GPU-hours. A bug found here
# costs seconds; the same bug found at stage 4 costs the grant, because failed
# attempts are charged to the ceiling just like successes.
#
# REAL here: run_diagnostic.sh, generate_eval_images.py (reuse gate included),
# validate_stage.py, select_matched_dump.py, gpu_budget.py, the frozen dev and
# test manifests with their published identities, the frozen bootstrap grouping,
# and the real generation-settings contract.
#
# Mocked: accelerate (the trainer, which would need a GPU), detect.py, the GPU
# selector, nvidia-smi; torch/diffusers are replaced by
# scripts/seq/testlib/fake_models. The detection mock drives a SCRIPTED dog
# residue profile per checkpoint, so selection outcomes -- matched, infeasible,
# ambiguous -- can be exercised deliberately rather than hoped for.
#
# The real /data pilot is never touched: SEQ_ROOT is a sandbox with fixture
# parents and a fixture legacy registry.
#
#   bash scripts/seq/test_diagnostic_runner.sh
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
if [ -z "${KEEP_DIAG_ROOT:-}" ]; then trap 'rm -rf "$ROOT"' EXIT; else
  trap 'echo "sandbox kept: $ROOT"' EXIT; fi
REPO_AT_RISK="results configs docs"
SNAP="${ROOT}/repo_before.sha"
# shellcheck disable=SC2086
( cd "$REPO" && find $REPO_AT_RISK -type f -exec sha256sum {} + 2>/dev/null | sort ) > "$SNAP"

export CALL_LOG="${ROOT}/calls.log"; : > "$CALL_LOG"
export FAKE_MODEL_CALL_LOG="${ROOT}/fake_calls.log"; : > "$FAKE_MODEL_CALL_LOG"
FAKE_MODELS="${REPO}/scripts/seq/testlib/fake_models"; export FAKE_MODELS

if [ -x /data/bijaypandey/cuig_pilot/venv-cuig/bin/python ]; then
    export MOCK_PY=/data/bijaypandey/cuig_pilot/venv-cuig/bin/python
else
    export MOCK_PY="$(command -v python3)"
fi
"$MOCK_PY" -c 'import torch' 2>/dev/null || {
    echo "FATAL: need an interpreter with torch (the validator loads checkpoints)" >&2
    exit 2; }

BIN="${ROOT}/bin"; mkdir -p "$BIN"
cat > "${BIN}/nvidia-smi" <<'EOF'
#!/bin/bash
echo "MOCK nvidia-smi $*" >> "$CALL_LOG"; echo "0, 41 MiB, 0 %"
EOF
cat > "${ROOT}/mock_gpu_select.sh" <<'EOF'
#!/bin/bash
echo "MOCK gpu_select $*" >> "$CALL_LOG"
if [ -n "${MOCK_GPU_BUSY:-}" ]; then
  for i in 0 1 2 3; do echo "GPU $i [mock] -> OCCUPIED"; done; exit 1
fi
for i in 2 3; do echo "GPU $i [mock] -> IDLE"; done
echo "GPU 0 [mock] -> OCCUPIED"; echo "GPU 1 [mock] -> OCCUPIED"
EOF
chmod +x "${ROOT}/mock_gpu_select.sh"
export GPU_SELECT="${ROOT}/mock_gpu_select.sh"

export MOCK_HELPER="${ROOT}/mock_helper.py"
# The trainer mock: writes delta.bin AND the periodic dumps, plus the report
# train_request.py would have written (including the dumps block). The dumps
# differ from one another so the "digests distinct" check is meaningful.
cat > "${BIN}/accelerate" <<'EOF'
#!/bin/bash
echo "MOCK accelerate $*" >> "$CALL_LOG"
name=""; out=""; seed=""; iters="1000"; every="0"; l2="0"; parent=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name) name="$2"; shift 2;; --output_dir) out="$2"; shift 2;;
    --seed) seed="$2"; shift 2;; --iterations) iters="$2"; shift 2;;
    --checkpoint_every) every="$2"; shift 2;; --l2sp_weight) l2="$2"; shift 2;;
    --unet_ckpt) parent="$2"; shift 2;; *) shift;;
  esac
done
# The runner must pass the dump interval through; if it ever stops doing so the
# mock records a 0 and the trajectory validation below fails loudly.
echo "TRAIN name=$name seed=$seed iters=$iters every=$every" >> "$CALL_LOG"
"$MOCK_PY" "$MOCK_HELPER" make_traj --out "$out" --name "$name" --seed "$seed" \
    --iters "$iters" --every "$every" --l2 "$l2" --parent "$parent"
EOF
cat > "${BIN}/python" <<'EOF'
#!/bin/bash
for a in "$@"; do
  case "$a" in
    *generate_eval_images.py)
      echo "REAL generate $*" >> "$CALL_LOG"
      exec env PYTHONPATH="${FAKE_MODELS}:${PYTHONPATH:-}" "$MOCK_PY" "$@" ;;
    *detect.py)
      echo "MOCK detect $*" >> "$CALL_LOG"
      exec "$MOCK_PY" "$MOCK_HELPER" detect "$@" ;;
  esac
done
exec "$MOCK_PY" "$@"
EOF
cp "${BIN}/python" "${BIN}/python3"
chmod +x "${BIN}"/*

cat > "$MOCK_HELPER" <<'PYEOF'
"""Trainer and detector mocks for the diagnostic-runner test."""
import argparse, hashlib, json, math, os, sys
from pathlib import Path


def _ckpt(path, fill, contract):
    import torch
    sd = {k: torch.full(tuple(v["shape"]), float(fill))
          for k, v in contract["tensors"].items()}
    torch.save({contract["top_level_key"]: sd}, path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_traj(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    contract = json.loads(Path(os.environ["SEQ_CKPT_CONTRACT"]).read_text())
    iters, every = int(a.iters), int(a.every)
    steps = int(os.environ.get("MOCK_STEPS_COMPLETED", iters))
    dumps = []
    if every > 0:
        want = list(range(every, steps + 1, every))
        skip = {int(x) for x in os.environ.get("MOCK_SKIP_DUMPS", "").split() if x}
        for st in want:
            if st in skip:
                continue
            p = out / f"delta-{st}"
            h = _ckpt(p, 0.001 * st, contract)
            dumps.append({"step": st, "path": str(p), "sha256": h,
                          "bytes": p.stat().st_size})
    delta = out / "delta.bin"
    h = _ckpt(delta, 0.001 * steps + 0.5, contract)
    parent_sha = None
    if a.parent and Path(a.parent).exists():
        parent_sha = hashlib.sha256(Path(a.parent).read_bytes()).hexdigest()
    rep = {
        "checkpoint": a.name, "parent": "MA", "training_seed": int(a.seed),
        "l2sp_weight": float(a.l2), "new_deletion_target": "dog",
        "anchor_concept": "horse",
        "child_checkpoint": {"sha256": h, "bytes": delta.stat().st_size,
                             "tensors": contract["n_tensors"]},
        "effective_hyperparameters": {"seed": int(a.seed),
                                      "iterations_requested": iters,
                                      "epoch_capacity_steps": 1250,
                                      "parameter_group": "kv-xattn"},
        "parent_verification": {"parent_sha256": parent_sha, "PASS": True,
                                "max_abs_deviation_after_load": 0.0},
        "l2sp_reference_check": {"PASS": True, "max_abs_deviation_from_parent": 0.0},
        "l2sp_trace": {"steps_recorded": 0},
        "started_utc": "2026-10-09T00:00:00+00:00",
        "finished_utc": "2026-10-09T00:14:00+00:00",
        "runtime": {"train_seconds": steps * 0.83,
                    "optimizer_steps_completed": steps,
                    "optimizer_steps_source": "mock (test fixture)",
                    "optimizer_sync_points": steps,
                    "seconds_per_optimizer_step": 0.83,
                    "seconds_per_optimizer_step_is_derived": True},
    }
    if every > 0:
        want = list(range(every, steps + 1, every))
        rep["dumps"] = {
            "checkpoint_every": every, "expected_steps": want,
            "n_expected": len(want), "n_present": len(dumps),
            "missing_steps": [s for s in want
                              if s not in {d["step"] for d in dumps}],
            "unexpected_files": [],
            "all_digests_distinct": len({d["sha256"] for d in dumps}) == len(dumps),
            "complete": len(dumps) == len(want),
            "source": "mock (test fixture)",
            "dumps": dumps,
        }
    (out / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(f"mock trajectory {a.name} seed{a.seed}: {len(dumps)} dumps")


# Scripted dog residue, in percent, per evaluation slot name. The default
# profile makes BOTH seeds match at a dump in the middle of the scan.
def residue_pct(name: str) -> float:
    prof = json.loads(os.environ.get("MOCK_RESIDUE_PROFILE", "{}"))
    for key, val in prof.items():
        if key in name:
            return float(val)
    if name.endswith("_MA"):
        return 95.0
    if name.endswith("_MAB_L2"):
        return 55.0            # 40 pp suppression, residue <= 60 -> qualifies
    if "_U_step" in name:
        st = int(name.split("_U_step")[1])
        # Monotone decay from 95% toward 20%: step 600 lands at ~57.5%.
        return max(20.0, 95.0 - 0.0625 * st)
    return 50.0


def detect(argv):
    g, i = {}, 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv) and not argv[i+1].startswith("--"):
            g[argv[i][2:]] = argv[i+1]; i += 2
        else:
            i += 1
    ir = json.loads(Path(g["image_report"]).read_text())
    name = g["checkpoint_name"]
    rows = ir["images"]
    dogs = [r for r in rows if r.get("category") == "dog"]
    n_hit = int(round(residue_pct(name) / 100.0 * len(dogs))) if dogs else 0
    hit_names = {r["image_name"] for r in dogs[:n_hit]}
    out = []
    for r in rows:
        h = r["image_name"] in hit_names
        out.append({"checkpoint": name, "prompt_id": r["prompt_id"],
                    "category": r["category"], "prompt_family": r["prompt_family"],
                    "prompt_index": r["prompt_index"], "prompt": r["prompt"],
                    "gen_seed": r["gen_seed"], "image_path": r["image_path"],
                    "image_sha256": r["sha256"],
                    "detections": [], "max_score_target": 0.9 if h else 0.1,
                    "n_detections_target": {"0.3": int(h), "0.5": int(h), "0.7": 0},
                    "hit": {"0.3": h, "0.5": h, "0.7": False}})
    Path(g["out"]).write_text("\n".join(json.dumps(x) for x in out) + "\n")
    Path(g["report"]).write_text(json.dumps({
        "checkpoint": name, "complete": True, "images_scored": len(out),
        "images_expected": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "make_traj":
        ap = argparse.ArgumentParser()
        for f in ("out", "name", "seed", "iters", "every", "l2", "parent"):
            ap.add_argument(f"--{f}")
        make_traj(ap.parse_args(sys.argv[2:]))
        sys.exit(0)
    sys.exit(detect(sys.argv[2:]))
PYEOF

# ---------------------------------------------------------------- environment
SEQ="${ROOT}/seq_pilot"
mkdir -p "$SEQ"/{models,anchor_caches/Horses,anchor_caches/Flowers}
export SEQ_ROOT="$SEQ"
export PILOT_REPO_ROOT="$REPO"
export PILOT_VENV="${ROOT}/venv"; mkdir -p "${PILOT_VENV}/bin"
echo '# mock venv: no-op so the mock PATH stays in front' > "${PILOT_VENV}/bin/activate"
export SEQ_ACCEL_CFG="${ROOT}/accel.yaml"; echo "{}" > "$SEQ_ACCEL_CFG"
export SEQ_BASE_MODEL="${ROOT}/sd15"; mkdir -p "${SEQ_BASE_MODEL}/unet"
echo '{"_class_name":"MockPipeline"}' > "${SEQ_BASE_MODEL}/model_index.json"
export SEQ_EPOCHS=25
export SEQ_DIAG_ROOT="${ROOT}/diag"
export SEQ_DIAG_LEDGER="${ROOT}/budget_ledger.json"
export SEQ_DIAG_COST_MODEL=""   # exercise the built-in estimates
export SEQ_DIAG_SKIP_SMOKE=1      # the smoke stage needs a real GPU
export SEQ_GEN_SETTINGS_CONTRACT="${REPO}/configs/generation_settings.json"
export SEQ_VALIDATOR="${REPO}/scripts/seq/validate_stage.py"
export SEQ_BOOTSTRAP_GROUPING="${REPO}/results/audit_v1/draft_manifests/bootstrap_grouping.json"
export SEQ_DEV_MANIFEST="${REPO}/results/audit_v1/draft_manifests/dev_manifest_DRAFT.json"
export SEQ_TEST_MANIFEST="${REPO}/results/audit_v1/draft_manifests/test_manifest_DRAFT.json"

cat > "${ROOT}/tiny_contract.json" <<'EOF'
{"contract":"TEST","top_level_key":"unet","n_tensors":2,"total_elements":12,
 "dtype":"torch.float32","expected_file_bytes":null,
 "tensors":{"a.attn2.to_k.weight":{"shape":[2,3],"dtype":"torch.float32"},
            "a.attn2.to_v.weight":{"shape":[2,3],"dtype":"torch.float32"}}}
EOF
export SEQ_CKPT_CONTRACT="${ROOT}/tiny_contract.json"

# Fixture saved parents and L2 endpoints, plus a fixture registry that declares
# them the registered saved artifacts (the real registry names the real pilot).
"$MOCK_PY" - "$SEQ" "$SEQ_CKPT_CONTRACT" "${ROOT}/legacy_registry.json" <<'FIX'
import hashlib, json, sys, torch
from pathlib import Path
seq, contract_p, reg_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
c = json.loads(contract_p.read_text())
arts = []
for seed in (17, 29):
    for slot, fill in (("MA", 0.01), ("MAB_L2", 0.02)):
        d = seq / "models" / f"seed{seed}" / slot
        d.mkdir(parents=True, exist_ok=True)
        sd = {k: torch.full(tuple(v["shape"]), fill + seed * 1e-4)
              for k, v in c["tensors"].items()}
        ck = d / "delta.bin"
        torch.save({c["top_level_key"]: sd}, ck)
        rep = {"checkpoint": slot, "training_seed": seed,
               "note": "fixture saved artifact"}
        (d / "train_report.json").write_text(json.dumps(rep, indent=2))
        arts.append({
            "id": f"seed{seed}/{slot}", "training_seed": seed, "checkpoint": slot,
            "train_report_sha256": hashlib.sha256((d / "train_report.json").read_bytes()).hexdigest(),
            "delta_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
            "in_repo_report_copy": f"results/seq{seed}/train/{slot}.json",
            "in_repo_report_copy_sha256": None,
            "provenance": {"produced_by": "diagnostic-runner test fixture",
                           "saved_path": str(d / "train_report.json")}})
reg_p.write_text(json.dumps({"artifacts": arts}, indent=2))
print(f"fixture: 4 saved artifacts + registry")
FIX
export SEQ_LEGACY_ARTIFACT_REGISTRY="${ROOT}/legacy_registry.json"

diag() { ( cd "$REPO" && PATH="${BIN}:$PATH" bash scripts/seq/run_diagnostic.sh "$@" ) ; }

echo
echo "=== 1. preflight verifies identities, reuse bindings and capacity ==="
diag preflight > "${ROOT}/t1.log" 2>&1; RC=$?
want_rc "preflight succeeds" 0 "$RC"
want_out "validates the dev manifest identity"  "${ROOT}/t1.log" "VALID    manifest:dev"
want_out "validates the test manifest identity" "${ROOT}/t1.log" "VALID    manifest:test"
want_out "checks the 20 dev prompt texts"       "${ROOT}/t1.log" "distinct_prompt_texts = 20"
want_out "checks the 140 test prompt texts"     "${ROOT}/t1.log" "distinct_prompt_texts = 140"
want_out "verifies the frozen bootstrap grouping" "${ROOT}/t1.log" "grouping identity"
want_out "binds the reused saved artifacts"     "${ROOT}/t1.log" "REGISTERED saved artifact"
want_out "initialises the ceiling"              "${ROOT}/t1.log" "ceiling 4.0 GPU-h"
want_out "reserves the test cost"               "${ROOT}/t1.log" "reserve 1.3474"
want_out "reports verified-idle capacity"       "${ROOT}/t1.log" "RUNNING: 2 verified-idle GPU(s)"

echo
echo "=== 2. preflight REFUSES when the reused CHECKPOINT is not the saved one ==="
# What the diagnostic reuses is the checkpoint, so that is what must be bound.
# Swap MA's bytes for a different (valid) checkpoint under the same name.
cp "${SEQ}/models/seed17/MA/delta.bin" "${ROOT}/ma17_real.bin"
cp "${SEQ}/models/seed29/MA/delta.bin" "${SEQ}/models/seed17/MA/delta.bin"
diag preflight > "${ROOT}/t2.log" 2>&1; RC=$?
want_rc "preflight refuses an unregistered reuse target" 1 "$RC"
want_out "names the problem" "${ROOT}/t2.log" "NOT a registered saved artifact"
want_out "refuses to proceed" "${ROOT}/t2.log" "not the registered saved ones"
cp "${ROOT}/ma17_real.bin" "${SEQ}/models/seed17/MA/delta.bin"

echo
echo "=== 3. no idle capacity: QUEUED, and nothing is launched ==="
MOCK_GPU_BUSY=1 diag preflight > "${ROOT}/t3.log" 2>&1; RC=$?
want_rc "refuses to start without verified idle capacity" 1 "$RC"
want_out "reports the queue/capacity status" "${ROOT}/t3.log" "QUEUED/BLOCKED"
want_out "promises not to pre-empt anyone"   "${ROOT}/t3.log" "never pre-empted"

echo
echo "=== 4. training passes the dump interval through and validates dumps ==="
diag train > "${ROOT}/t4.log" 2>&1; RC=$?
want_rc "training stage succeeds" 0 "$RC"
want_out "the runner passed --checkpoint_every through" "$CALL_LOG" "every=100"
want_out "ten dumps validate against the contract" "${ROOT}/t4.log" "10/10 present, all validate"
want_out "the trajectory itself validates" "${ROOT}/t4.log" "VALID    train:U"
want_eq "both trajectories produced 10 dumps each" \
    "$(find "${ROOT}/diag/models" -name 'delta-*' | wc -l)" "20"
want_out "training was charged to the ledger" "${ROOT}/t4.log" "train:U_seed17"

echo
echo "=== 5. a trainer that writes an incomplete schedule is rejected ==="
# Deleting a dump from a finished trajectory would simply be retrained, which
# is correct. The failure that must NOT pass is a trainer that writes a short
# scan: the protocol froze a FULL fixed scan of ten dumps, and scoring fewer
# arms silently would change the experiment.
rm -rf "${ROOT}/diag/models/seed29"
MOCK_SKIP_DUMPS="500 700" diag train > "${ROOT}/t5.log" 2>&1; RC=$?
want_rc "an incomplete dump schedule fails" 1 "$RC"
if grep -qE "absent at|missing dumps|DUMPS INVALID" "${ROOT}/t5.log"; then
    ok "names the missing dumps"
else bad "names the missing dumps" "no dump diagnosis in the log"; fi
# Re-entering with the invalid trajectory still on disk preserves it whole and
# retrains into a fresh directory, rather than writing over it in place.
MOCK_SKIP_DUMPS="500 700" diag train > "${ROOT}/t5q.log" 2>&1 || true
want_out "the invalid trajectory is preserved whole, not overwritten" \
    "${ROOT}/t5q.log" "[quarantine]"
want_eq "a reason file accompanies it" \
    "$(find "${ROOT}/diag/models/seed29" -maxdepth 1 -name 'U.invalid.*.reason.txt' | wc -l)" "1"
# Restore by retraining that seed cleanly.
rm -rf "${ROOT}/diag/models/seed29"
diag train > "${ROOT}/t5b.log" 2>&1; RC=$?
want_rc "retraining restores a complete trajectory" 0 "$RC"

echo
echo "=== 6. development evaluation: 1,920 images over 24 slots ==="
diag dev > "${ROOT}/t6.log" 2>&1; RC=$?
want_rc "development stage succeeds" 0 "$RC"
want_eq "24 development slots scored" \
    "$(find "${ROOT}/diag/eval_dev" -name detections.jsonl | wc -l)" "24"
want_eq "1,920 development images generated" \
    "$(find "${ROOT}/diag/eval_dev" -path '*/images/*' -type f | wc -l)" "1920"
want_out "every slot is bound to its own checkpoint" "${ROOT}/t6.log" "generating_checkpoint"

echo
echo "=== 7. selection applies the frozen gates and match rule ==="
diag select > "${ROOT}/t7.log" 2>&1; RC=$?
want_rc "selection succeeds with both seeds matched" 0 "$RC"
want_out "reports MATCHED" "${ROOT}/t7.log" "MATCHED"
want_out "selects a dump" "${ROOT}/t7.log" "<- SELECTED"
want_out "permits the test set" "${ROOT}/t7.log" "PROCEED to the frozen test set"
SEL17=$("$MOCK_PY" -c "
import json,sys; print(json.load(open(sys.argv[1]))['per_seed']['17']['selected']['step'])" \
    "${ROOT}/diag/selection.json")
want_eq "seed 17 selected the dump closest to its L2 level" "$SEL17" "600"
MIS=$("$MOCK_PY" -c "
import json,sys; print(json.load(open(sys.argv[1]))['per_seed']['17']['selected']['mismatch_vs_L2_pp'] <= 5.0)" \
    "${ROOT}/diag/selection.json")
want_eq "the mismatch is within 5 pp" "$MIS" "True"

echo
echo "=== 8. infeasibility is a RESULT: the test set is not evaluated ==="
# An L2 endpoint far outside anything the scan reaches.
cp "${ROOT}/diag/selection.json" "${ROOT}/selection_matched.json"
MOCK_RESIDUE_PROFILE='{"_MAB_L2": 2.0}' diag select > "${ROOT}/t8.log" 2>&1 || true
rm -rf "${ROOT}/diag/eval_dev/seed17_MAB_L2" "${ROOT}/diag/eval_dev/seed29_MAB_L2"
MOCK_RESIDUE_PROFILE='{"_MAB_L2": 2.0}' diag dev > "${ROOT}/t8b.log" 2>&1
MOCK_RESIDUE_PROFILE='{"_MAB_L2": 2.0}' diag select > "${ROOT}/t8c.log" 2>&1; RC=$?
want_rc "selection stops when no dump matches" 1 "$RC"
want_out "declares infeasibility" "${ROOT}/t8c.log" "INFEASIBLE"
want_out "reports the bracketing dumps" "${ROOT}/t8c.log" "no dump passes both reference gates"
want_out "refuses to widen the tolerance" "${ROOT}/t8c.log" "NOT widened"
want_out "forbids the test set" "${ROOT}/t8c.log" "do NOT evaluate the test set"
diag test > "${ROOT}/t8d.log" 2>&1; RC=$?
want_rc "the test stage refuses to run after infeasibility" 2 "$RC"
want_out "says the paired test does not run" "${ROOT}/t8d.log" "does NOT run"
want_eq "no test images were generated" \
    "$(find "${ROOT}/diag/eval_test" -type f 2>/dev/null | wc -l)" "0"
RES=$("$MOCK_PY" -c "
import json,sys
d=json.load(open(sys.argv[1]))
spent=sum(e['gpu_hours'] for e in d['entries'] if e.get('draws_on_reserve'))
print('UNSPENT' if spent == 0 else f'SPENT {spent}')" \
    "${ROOT}/budget_ledger.json")
want_eq "the test reserve was not spent" "$RES" "UNSPENT"

echo
echo "=== 9. restore the matched profile and evaluate the frozen test set ==="
rm -rf "${ROOT}/diag/eval_dev/seed17_MAB_L2" "${ROOT}/diag/eval_dev/seed29_MAB_L2"
diag dev > "${ROOT}/t9a.log" 2>&1
diag select > "${ROOT}/t9b.log" 2>&1; RC=$?
want_rc "selection matches again" 0 "$RC"
diag test > "${ROOT}/t9.log" 2>&1; RC=$?
want_rc "test stage succeeds" 0 "$RC"
want_eq "6 test slots scored" \
    "$(find "${ROOT}/diag/eval_test" -name detections.jsonl | wc -l)" "6"
want_eq "3,360 test images generated" \
    "$(find "${ROOT}/diag/eval_test" -path '*/images/*' -type f | wc -l)" "3360"
want_out "the test stage drew on its reserve" "${ROOT}/t9.log" "eval:eval_test"

echo
echo "=== 10. the ceiling is hard, and failures are charged ==="
"$MOCK_PY" "${REPO}/scripts/seq/gpu_budget.py" report --ledger "${ROOT}/budget_ledger.json" \
    > "${ROOT}/t10.log" 2>&1
want_out "the ledger reports a 4 GPU-hour ceiling" "${ROOT}/t10.log" "ceiling         : 4.000"
cat "${ROOT}/t10.log" | sed 's/^/    /'
# A ceiling already consumed must refuse the next stage outright.
L2="${ROOT}/tiny_ledger.json"
"$MOCK_PY" "${REPO}/scripts/seq/gpu_budget.py" init --ledger "$L2" --ceiling 0.10 --reserve 0.05 >/dev/null
# A FAILED stage burns the budget just like a successful one.
BURN_ID=$("$MOCK_PY" "${REPO}/scripts/seq/gpu_budget.py" admit --ledger "$L2" \
    --stage burn --need 0.08 --gpus 1 --pid $$ \
    | "$MOCK_PY" -c 'import json,sys;print(json.loads([l for l in sys.stdin if l.startswith("{")][-1])["reservation_id"])')
"$MOCK_PY" "${REPO}/scripts/seq/gpu_budget.py" settle --ledger "$L2" --id "$BURN_ID" \
    --wall_seconds 300 --exit_code 1 >/dev/null
# A fresh diagnostic root, so training must actually be attempted rather than
# skipped as already-validated.
SEQ_DIAG_LEDGER="$L2" SEQ_DIAG_ROOT="${ROOT}/diag_budget" diag train > "${ROOT}/t10b.log" 2>&1; RC=$?
want_out "a stage is refused when the budget cannot afford it" "${ROOT}/t10b.log" "BUDGET REFUSED"
want_out "and nothing is launched" "${ROOT}/t10b.log" "nothing was launched"

echo
echo "=== 10b. admission holds in-flight amounts and survives a new output dir ==="
L3="${ROOT}/inflight_ledger.json"
"$MOCK_PY" "${REPO}/scripts/seq/gpu_budget.py" init --ledger "$L3" \
    --ceiling 1.00 --reserve 0.40 >/dev/null
A1=$("$MOCK_PY" "${REPO}/scripts/seq/gpu_budget.py" admit --ledger "$L3" \
    --stage lane-a --need 0.55 --gpus 1 --pid $$ 2>&1 | tail -1)
A2=$("$MOCK_PY" "${REPO}/scripts/seq/gpu_budget.py" admit --ledger "$L3" \
    --stage lane-b --need 0.55 --gpus 1 --pid $$ 2>&1 | tail -1 \
    | "$MOCK_PY" -c 'import json,sys;print(json.loads(sys.stdin.read())["decision"])')
case "$A1" in *ADMIT*) ok "the first lane is admitted";; *) bad "admission" "$A1";; esac
want_eq "a concurrent lane is refused against the in-flight amount" "$A2" "REFUSE"
# A new output directory must NOT reset the balance.
SPENT_BEFORE=$("$MOCK_PY" -c "
import json,sys; print(json.load(open(sys.argv[1]))['committed_gpu_hours'])" "$L3")
SEQ_DIAG_ROOT="${ROOT}/diag_elsewhere" SEQ_DIAG_LEDGER="$L3" diag preflight \
    > "${ROOT}/t10c.log" 2>&1 || true
SPENT_AFTER=$("$MOCK_PY" -c "
import json,sys; print(json.load(open(sys.argv[1]))['committed_gpu_hours'])" "$L3")
want_eq "a NEW output directory does not reset prior spend" "$SPENT_AFTER" "$SPENT_BEFORE"
want_out "the runner reports the experiment-level ledger" "${ROOT}/t10c.log" "not this output directory"

echo
echo "=== 11. isolation: no GPU, no repository mutation, pilot untouched ==="
if grep -qE 'CUDA_VISIBLE_DEVICES=[0-9]' "$CALL_LOG"; then
    bad "device masking" "a mock saw a numeric CUDA device"
else ok "no mock was invoked with a numeric CUDA device"; fi
want_out "the real generator was exercised" "$CALL_LOG" "REAL generate"
if [ "${SEQ}" != "/data/bijaypandey/cuig_pilot/seq_pilot" ]; then
    ok "ran against a sandbox SEQ_ROOT, not the real pilot"
else bad "sandbox" "SEQ_ROOT points at the real pilot"; fi
AFTER="${ROOT}/repo_after.sha"
# shellcheck disable=SC2086
( cd "$REPO" && find $REPO_AT_RISK -type f -exec sha256sum {} + 2>/dev/null | sort ) > "$AFTER"
if diff -q "$SNAP" "$AFTER" >/dev/null; then
    ok "results/, configs/ and docs/ byte-identical after this suite"
else bad "repository isolation" "the suite modified a tracked tree"; diff "$SNAP" "$AFTER" | head -6 >&2; fi

echo
echo "-----------------------------------------------------------------"
echo "${PASSN} passed, ${FAILN} failed"
[ "$FAILN" -eq 0 ] || exit 1
echo "ALL DIAGNOSTIC-RUNNER CHECKS PASSED (CPU only; trainer and detector mocked)"
