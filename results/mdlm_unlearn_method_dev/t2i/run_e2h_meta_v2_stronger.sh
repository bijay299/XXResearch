#!/bin/bash
# E2h: does meta2_church_lt0.5's 25-step result survive the attacker that actually matters?
#
# FINDING78 S3 recorded that lt0.5 meets BOTH mandate halves at 25 steps (retain8 0.9000,
# max-over-valid H 0.0667) but is untested against the cheapest REPRODUCIBLE full-recovery attacker
# measured anywhere in this project: lr 5e-5 at 25 steps, which reached H=1.0000 on ALL THREE seeds
# with sd=0 (FINDING70 S1). That is the obvious falsifier and it had not been run.
#
# DECLARED BEFORE ANY RESULT (original text, preserved verbatim -- do not edit):
#   "lr 5e-5 and 1e-4 x steps {25} x seeds {0,1,6} on lt0.5, plus lr 5e-5
#    at steps 10 to locate the threshold. 12 cells. No escalation beyond these."
# PREDICTION: if lt0.5's 25-step robustness is real it should hold at 5e-5 too; if it was an
# artifact of a weak attacker at lr 1e-5, max-over-valid should exceed 0.40 at 5e-5.
#
# ---- CORRECTION 2026-10-26 (audit defect D-007). The declaration above is internally
# inconsistent and does not match the loop. Recorded here; the declaration itself is NOT rewritten.
#   * The enumerated set is lr{5e-5,1e-4} x steps{25} x seeds{0,1,6} = 6 cells, plus
#     lr 5e-5 x steps{10} x seeds{0,1,6} = 3 cells. That is NINE cells, not the twelve stated.
#   * The loop (below) iterates FOUR configs x 3 seeds = 12 cells. The fourth, "5e-5 50", is
#     ABSENT from the enumeration and is an escalation in steps beyond every declared cell,
#     contradicting "No escalation beyond these".
#   * Therefore: the 5e-5/50 cells are an UNDECLARED POST-HOC EXTENSION. They must never be
#     described as preregistered, and must be reported separately from the nine declared cells.
#     The coincidence between the stated count "12" and the loop's 12 cells is not evidence that
#     the 50-step cell was intended; the enumeration is the declaration, and it lists nine.
#   * The summary block at the end of this file globs atk_E2[GH]_* and therefore POOLS E2g's
#     frozen-grid records with E2h's into one table without distinguishing campaign. Any row it
#     prints may come from either campaign. Fixed below to tag the source campaign per record.
#   * A low-disk abort ("break 2") formerly fell through to "E2H DONE". Fixed below to exit
#     non-zero and write an explicit INCOMPLETE marker.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
L=logs/e2h_stronger.log
# Migrated 2026-10-26 off the retired scripts/campaign_free.sh (unlocked check-then-select) to the
# exclusive reservation API. See docs/REPAIR_INFRA_v2.md.
source scripts/gpu_reserve.sh
gpu_reserve_hold "e2h_meta_v2_stronger" 120 >/dev/null || {
  echo "E2h ABORT: no GPU reservation (rc=${GPU_RESERVE_RC:-?}); see stderr above" >&2
  echo "E2H INCOMPLETE $(date -Is) -- no GPU reservation (rc=${GPU_RESERVE_RC:-?}); nothing ran" >>"$L"
  exit 1; }
G="$GPU_RESERVED"   # CUDA_VISIBLE_DEVICES is already exported; the release trap is installed
echo "=== E2h on GPU $G (uuid ${GPU_RESERVED_UUID:-?}) $(date -Is)" >>"$L"
A=meta2_church_lt0.5
INCOMPLETE=0
# "5e-5 50" is the UNDECLARED post-hoc cell (see CORRECTION above). Kept so the record stays
# reproducible, but tagged so no consumer can mistake it for a declared cell.
for cfg in "5e-5 25 declared" "1e-4 25 declared" "5e-5 10 declared" "5e-5 50 POSTHOC_UNDECLARED"; do
  set -- $cfg; LR=$1; S=$2; PREREG=$3
  for SD in 0 1 6; do
    FREE=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9)
    [ "$FREE" -lt 5 ] && { echo "ABORT: ${FREE}GiB free" >>"$L"; INCOMPLETE=1; break 2; }
    K="E2H_${A}_sd${SD}_lr${LR}_s${S}"
    [ -f "outputs/atk_${K}.json" ] && continue
    echo "=== $K prereg=$PREREG $(date -Is)" >>"$L"
    venv_t2i/bin/python scripts/attack_relearn.py --concept church \
      --unet "adapters/$A/unet.pt" --seed "$SD" --lr "$LR" --steps "$S" --label "$K" >>"$L" 2>&1
    rc=$?
    rm -rf "adapters/_atk_$K" 2>/dev/null
    if [ "$rc" -ne 0 ]; then
      echo "FAIL $K exit=$rc" >>"$L"; INCOMPLETE=1
    fi
  done
done
venv_t2i/bin/python - <<'PY' >>"$L" 2>&1
import json,glob,collections
print("\n=== E2h: meta2_church_lt0.5 (retain8 0.9000) vs STRONGER attackers ===")
print("  campaign is tagged per row: E2g = earlier frozen grid, E2h = this run. They are NOT pooled.")
print("  prereg: 'declared' = in E2h's enumerated declaration; 'POSTHOC' = undeclared extension (D-007).")
print(f"  {'camp':>4} {'lr':>7} {'steps':>6} {'prereg':>8} {'valid':>6} {'maxH VALID':>11} {'maxH UNGATED':>13}  bar<=0.40  per-seed")
c=collections.defaultdict(list)
# E2h declared exactly these (lr, steps) cells; anything else from E2h is post-hoc.
DECLARED={("5e-05",25),("0.0001",25),("5e-05",10)}
for f in glob.glob("outputs/atk_E2[GH]_meta2_church_lt0.5_sd*.json"):
    d=json.load(open(f)); hp=d["hparams"]
    camp="E2h" if "/atk_E2H_" in f else "E2g"
    lr,st=hp["lr"],hp["steps"]
    pre="declared" if (camp!="E2h" or (f"{float(lr):g}",st) in {(f"{float(a):g}",b) for a,b in DECLARED}) else "POSTHOC"
    if camp!="E2h": pre="E2g-grid"
    c[(camp,float(lr),st,pre)].append((hp["seed"],d["attack_validity"]["valid"],
                                       round(d["recovery"]["H_after"],4)))
for k in sorted(c):
    v=sorted(c[k]); ok=[x[2] for x in v if x[1]]
    mv=max(ok) if ok else float('nan'); mu=max(x[2] for x in v)
    verdict = "MET" if (ok and mv<=0.40) else ("FAILED" if ok else "all invalid")
    print(f"  {k[0]:>4} {k[1]:>7.0e} {k[2]:>6} {k[3]:>8} {len(ok):>2}/{len(v):<3} {mv:>11.4f} {mu:>13.4f}  "
          f"{verdict:>9}  {[(x[0],'P' if x[1] else 'f',x[2]) for x in v]}")
print("  NOTE: max-over-valid is a ratchet (FINDING35 S4). A 'MET' row is provisional: it means no")
print("  attacker RUN SO FAR exceeded the bar, not that none exists.")
PY
if [ "$INCOMPLETE" -ne 0 ]; then
  echo "E2H INCOMPLETE $(date -Is) -- one or more cells aborted or failed; table above is partial" >>"$L"
  exit 1
fi
echo "E2H DONE $(date -Is)" >>"$L"
