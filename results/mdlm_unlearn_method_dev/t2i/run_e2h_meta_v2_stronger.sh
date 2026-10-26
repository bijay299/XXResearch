#!/bin/bash
# E2h: does meta2_church_lt0.5's 25-step result survive the attacker that actually matters?
#
# FINDING78 S3 recorded that lt0.5 meets BOTH mandate halves at 25 steps (retain8 0.9000,
# max-over-valid H 0.0667) but is untested against the cheapest REPRODUCIBLE full-recovery attacker
# measured anywhere in this project: lr 5e-5 at 25 steps, which reached H=1.0000 on ALL THREE seeds
# with sd=0 (FINDING70 S1). That is the obvious falsifier and it had not been run.
#
# DECLARED BEFORE ANY RESULT: lr 5e-5 and 1e-4 x steps {25} x seeds {0,1,6} on lt0.5, plus lr 5e-5
# at steps 10 to locate the threshold. 12 cells. No escalation beyond these.
# PREDICTION: if lt0.5's 25-step robustness is real it should hold at 5e-5 too; if it was an
# artifact of a weak attacker at lr 1e-5, max-over-valid should exceed 0.40 at 5e-5.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/e2h_stronger.log
while :; do G=$(bash scripts/campaign_free.sh | head -1); [ -n "$G" ] && break; sleep 120; done
export CUDA_VISIBLE_DEVICES=$G
echo "=== E2h on GPU $G $(date -Is)" >>"$L"
A=meta2_church_lt0.5
for cfg in "5e-5 25" "1e-4 25" "5e-5 10" "5e-5 50"; do
  set -- $cfg; LR=$1; S=$2
  for SD in 0 1 6; do
    FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    [ "$FREE" -lt 5 ] && { echo "ABORT: ${FREE}GiB free" >>"$L"; break 2; }
    K="E2H_${A}_sd${SD}_lr${LR}_s${S}"
    [ -f "outputs/atk_${K}.json" ] && continue
    echo "=== $K $(date -Is)" >>"$L"
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr "$LR" --steps "$S" --label "$K" >>"$L" 2>&1
    rm -rf "adapters/_atk_$K" 2>/dev/null
  done
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json,glob,collections
print("\n=== E2h: meta2_church_lt0.5 (retain8 0.9000) vs STRONGER attackers ===")
print(f"  {'lr':>7} {'steps':>6} {'valid':>6} {'maxH VALID':>11} {'maxH UNGATED':>13}  bar<=0.40  per-seed")
c=collections.defaultdict(list)
for f in glob.glob("outputs/atk_E2[GH]_meta2_church_lt0.5_sd*.json"):
    d=json.load(open(f)); hp=d["hparams"]
    c[(hp["lr"],hp["steps"])].append((hp["seed"],d["attack_validity"]["valid"],
                                      round(d["recovery"]["H_after"],4)))
for k in sorted(c):
    v=sorted(c[k]); ok=[x[2] for x in v if x[1]]
    mv=max(ok) if ok else float('nan'); mu=max(x[2] for x in v)
    verdict = "MET" if (ok and mv<=0.40) else ("FAILED" if ok else "all invalid")
    print(f"  {k[0]:>7.0e} {k[1]:>6} {len(ok):>2}/{len(v):<3} {mv:>11.4f} {mu:>13.4f}  {verdict:>9}  "
          f"{[(x[0],'P' if x[1] else 'f',x[2]) for x in v]}")
PY
echo "E2H DONE $(date -Is)" >>"$L"
