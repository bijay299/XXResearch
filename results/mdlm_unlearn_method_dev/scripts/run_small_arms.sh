#!/bin/bash
# H18 unlearning arms, scored against the EXACT retrain oracle.
#
# Objective with ground truth (FINDING21): take theta_ref from 0.6975 -> 0.1100 forget recall
# while holding retain at 0.6725. theta_oracle sits AT CHANCE on the forget facts (0.1100 vs
# 0.1135) because those authors never entered its pretraining or its fine-tune.
#
# Two metrics, both reported because they disagree (FINDING21 section 4):
#   ranking top-1 over the closed answer pool -> knowledge
#   infill six-cell profile + TV to the oracle -> behaviour
#
# Retain-side baseline is theta_ref, NOT the oracle: the oracle's retain recall is higher only
# because it had half as many authors to memorise.
set -u
cd /data/bijaypandey/mdlm_unlearn
export PATH=/data/bijaypandey/llada_smoke/venv_llada/bin:$PATH
source env.sh 2>/dev/null
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python
GPU="${GPU:-3}"
ORACLE=outputs/arm_h18_oracle.json

run () {                        # $1 = arm, $2 = lam_forget
  arm="$1"; lf="$2"
  lab="su_${arm}_f${lf/./p}"
  if [ ! -f "adapters/$lab/model.pt" ]; then
    echo "[sweep] train $lab"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_small_unlearn.py --arm "$arm" \
      --lam-forget "$lf" --lam-retain 1.0 --steps 1500 --log-every 500 \
      --out "/data/bijaypandey/mdlm_unlearn/adapters/$lab" \
      > "outputs/log_$lab.txt" 2>&1 || { echo "[sweep] TRAIN FAILED $lab"; return; }
  fi
  if [ ! -f "outputs/rank_$lab.json" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/rank_probe_small.py \
      --ckpt "adapters/$lab/model.pt" --label "$lab" \
      > "outputs/log_rank_$lab.txt" 2>&1 || echo "[sweep] RANK FAILED $lab"
  fi
  if [ ! -f "outputs/arm_$lab.json" ]; then
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_small.py --ckpt "adapters/$lab/model.pt" \
      --label "$lab" --infill --ref-profile "$ORACLE" \
      > "outputs/log_part_$lab.txt" 2>&1 || echo "[sweep] EVAL FAILED $lab"
  fi
  # reclaim weights once BOTH results exist; 0.46 GiB each and 15 arms is 7 GiB
  if [ -f "outputs/rank_$lab.json" ] && [ -f "outputs/arm_$lab.json" ]; then
    rm -f "adapters/$lab/model.pt"
  fi
}

for lf in 0.5 1.0 2.0; do
  for arm in afc mdu emdu mdu_fam ga; do
    run "$arm" "$lf"
  done
done

echo "[sweep] done"
$PY - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob('outputs/rank_su_*.json')):
    lab = json.load(open(f))["label"]
    r = json.load(open(f))["results"]
    af = f'outputs/arm_{lab}.json'
    a = json.load(open(af)) if os.path.exists(af) else {}
    rows.append((lab, r["forget"]["top1"], r["retain"]["top1"],
                 a.get("forget_recall"), a.get("retain_recall"),
                 a.get("forget_profile", {}).get("DEGENERATE"),
                 a.get("TV_forget_to_oracle")))
print(f"{'arm':22s} {'rankF':>7s} {'rankR':>7s} {'infF':>6s} {'infR':>6s} {'DEG':>6s} {'TVorc':>6s}")
print(f"{'theta_ref':22s} {0.6975:7.4f} {0.6725:7.4f} {0.2625:6.4f} {0.2725:6.4f} {0.0050:6.4f} {'-':>6s}")
print(f"{'theta_oracle (target)':22s} {0.1100:7.4f} {0.8875:7.4f} {0.0325:6.4f} {0.3625:6.4f} {0.0000:6.4f} {0.0:6.4f}")
for t in sorted(rows, key=lambda x: (x[1] is None, x[1])):
    f = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "  -   "
    print(f"{t[0]:22s} {f(t[1]):>7s} {f(t[2]):>7s} {f(t[3]):>6s} {f(t[4]):>6s} {f(t[5]):>6s} {f(t[6]):>6s}")
PY
