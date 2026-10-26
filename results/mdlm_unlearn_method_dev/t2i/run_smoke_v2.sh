#!/bin/bash
# BOUNDED INTEGRATION VALIDATION of the three corrected implementations (audit completion item 2).
#
# This is NOT an experiment and produces NO scientific result. It runs each v2 script for a handful
# of steps on the real pinned model to prove it executes end-to-end, writes the artifact fields the
# repair promises, and that its telemetry (inner-attacker probe, divergence record, target-content
# check, trust-region stop) actually populates. Every checkpoint written here is labelled SMOKE and
# deleted immediately after its artifact is verified, because /data is at 100% capacity.
#
# Serial, one GPU at a time: /data has ~14 GiB free and the filesystem is shared with another
# project. Two concurrent 1.7 GiB writes plus HF cache is not worth the risk.
set -u
cd /data/bijaypandey/t2i_unlearn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs .labels
T=adapters/frp_church_r3_l05/unet.pt
L=logs/smoke_v2.log
: >"$L"

say () { echo "$1" | tee -a "$L"; }

# ---- 1. faithful WDR. 40 alternating steps; max-rms deliberately tiny so the trust-region stop
#         fires and we can see it recorded.
say "=== SMOKE 1/3: erase_wdr_v2.py (faithful direction) $(date -Is)"
venv_t2i/bin/python scripts/erase_wdr_v2.py --concept church --alpha 1.0 --max-rms 0.0005 \
  --steps 40 --n-retain 8 --log-every 10 --teacher "$T" \
  --label SMOKE_wdr2 --out adapters/SMOKE_wdr2 >>"$L" 2>&1
say "   exit=$? artifact=$([ -f outputs/wdr2_SMOKE_wdr2.json ] && echo yes || echo NO)"
rm -rf adapters/SMOKE_wdr2

# ---- 2. Meta v2 full Eq. 8. 12 steps, 8 images; target-source base so the content check runs on
#         base-model generations (variant A's pool came from the erased teacher).
say "=== SMOKE 2/3: meta_unlearn_v2.py (full Eq. 8) $(date -Is)"
venv_t2i/bin/python scripts/meta_unlearn_v2.py --teacher "$T" --concept church \
  --steps 12 --n-images 8 --log-every 4 --zeta 1.0 --lam-target 1.0 \
  --meta-scope xattn --attack-scope full --target-source base --require-target-acc 0.25 \
  --label SMOKE_meta2 --out adapters/SMOKE_meta2 >>"$L" 2>&1
say "   exit=$? artifact=$([ -f outputs/meta2_SMOKE_meta2.json ] && echo yes || echo NO)"
rm -rf adapters/SMOKE_meta2

# ---- 3. UTRAP v2 through the shared attack core. 6 outer steps, K=8, probe every step.
say "=== SMOKE 3/3: erase_utrap_v2.py (shared attack core) $(date -Is)"
venv_t2i/bin/python scripts/erase_utrap_v2.py --concept church --teacher "$T" \
  --lam 0.15 --inner-k 8 --steps 6 --n-images 8 --n-retain 8 --log-every 2 --probe-every 2 \
  --label SMOKE_utrap2 --out adapters/SMOKE_utrap2 >>"$L" 2>&1
say "   exit=$? artifact=$([ -f outputs/utrap2_SMOKE_utrap2.json ] && echo yes || echo NO)"
rm -rf adapters/SMOKE_utrap2

say "=== SMOKE DONE $(date -Is)"
