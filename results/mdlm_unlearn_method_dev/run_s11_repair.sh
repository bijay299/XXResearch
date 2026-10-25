#!/bin/bash
# LABEL COLLISION, same failure mode as the t2i redundancy_sweep contamination: two of my own
# scripts both claimed the label mdu_l3_s11_n12 -- run_n12_mdu_gpu0.sh (GPU 0) and the chained MDU
# half of run_n12_extension.sh (GPU 3, which started when the RTT attack freed that card). One was
# TRAINING into adapters/mdu_l3_s11_n12 while the other was EVALUATING the same directory, so any
# arm_mdu_l3_s11_n12.json they produce is read off a checkpoint being rewritten underneath it.
#
# Neither job is killed. Both are allowed to finish, then their output is QUARANTINED rather than
# deleted (the project's standing practice for contaminated runs) and seed 11 is rebuilt once,
# alone, under an unambiguous new label. The existence guards mean nothing else re-enters the race.
set -u
cd /data/bijaypandey/mdlm_unlearn
source /data/bijaypandey/mdlm_unlearn/env.sh
PY=/data/bijaypandey/llada_smoke/venv_llada/bin/python

# wait for BOTH racers to exit on their own
while pgrep -f "mdu_l3_s11_n12" >/dev/null; do sleep 30; done

Q=outputs/CONTAMINATED_label_collision_s11
mkdir -p "$Q"
for f in outputs/arm_mdu_l3_s11_n12.json outputs/arm_mdu_l3_s11_n12_rows.json; do
  [ -f "$f" ] && mv "$f" "$Q/" && echo "quarantined $f" >>logs/s11_repair.log
done
cat > "$Q/README.md" <<'TXT'
Quarantined, not deleted. These rows were produced while two concurrent jobs of mine both held the
label `mdu_l3_s11_n12`: one training into `adapters/mdu_l3_s11_n12`, the other evaluating that same
directory. The evaluation therefore read a checkpoint that was being rewritten underneath it, so the
numbers are not attributable to any definite model state and must not be used in the n=12 aggregate.
The clean replacement is `mdu_l3_s11_n12b`, built alone.
TXT
rm -rf adapters/mdu_l3_s11_n12          # the raced checkpoint; its provenance is indeterminate

L=mdu_l3_s11_n12b
export CUDA_VISIBLE_DEVICES=0
$PY scripts/train_lam.py --arm mdu --lam-forget 3 --lam-retain 1 --seed 11 \
  --out "adapters/$L" >>"logs/$L.log" 2>&1
$PY scripts/eval_arm_partition.py --adapter "adapters/$L/ckpt_300" --label "$L" >>"logs/$L.log" 2>&1
echo "S11 REPAIR DONE $(date -Is)" >>logs/s11_repair.log
