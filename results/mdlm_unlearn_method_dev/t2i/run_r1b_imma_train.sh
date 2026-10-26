#!/bin/bash
# R1b: the FULL official IMMA defence training, author-documented settings (README L50-66 verbatim):
#   --mixed_precision=no --resolution=512 --random_flip --train_batch_size=1 --num_train_epochs=50
#   --learning_rate_lora=1e-04 --learning_rate=1e-05 --lr_scheduler=constant --lr_warmup_steps=0
#   --seed=42 --report_to=tensorboard --max_train_samples 20 --inner_loop_steps=1 --outer_loop_steps=1
#
# AUDIT REPAIR 2026-10-26 (docs/AGENT2_BASELINES_FINDINGS.md A-1). The pre-audit version omitted
# --max_train_samples. That is NOT a no-op: the default is None, so it trained on all 100 images,
# i.e. 100*50 = 5000 steps where the README command resolves to 20*50 = 1000. Because the loop
# alternates inner(LoRA descent)/outer(UNet ascent) 1:1, the historical artifact received 2500
# gradient-ASCENT steps on the attention weights instead of 500 -- a 5x over-budget immunisation.
# --max_train_samples is now explicit AND the output directory is derived from it, so a
# README-faithful run can never silently overwrite the 5x artifact.
#   IMMA_MAX_SAMPLES=20   (default) -> REF_imma_vangogh_s20    == the README command
#   IMMA_MAX_SAMPLES=all            -> REF_imma_vangogh_sall   == the historical 5x setting
# The existing REF_imma_vangogh/ directory is the historical "all" run; it is left untouched.
#
# Concept: Van Gogh (concept-MATCHED to the only ESD checkpoints baulab publishes; /object/ 404s).
# Data: images from their eval/text2image.py at 'An artwork by Van Gogh', with NSFW-blanked
#       (all-black) frames excluded from metadata.csv by run_r1_imma_official.sh (findings A-3).
# Scope: the reference's substring match over to_q/to_k/to_v/to_out across ALL named_parameters,
#       verified = 160 tensors / 93.536640M params = 80 attn1 (SELF) + 80 attn2 (CROSS). The
#       "xatten" in the artifact name is a misnomer; half the trained scope is self-attention.
# Env: venv_imma, diffusers 0.24.0 + huggingface_hub 0.25.2 -- required because LoRAAttnProcessor's
#      signature changed in later diffusers (imports fine, but drops `hidden_size`).
set -euo pipefail
cd /data/bijaypandey/t2i_unlearn

if [ -f FREEZE_ACTIVE ] && [ "${R1_FREEZE_OVERRIDE:-0}" != "1" ]; then
  echo "FREEZE_ACTIVE is present; refusing to launch GPU work. Set R1_FREEZE_OVERRIDE=1 to run." >&2
  exit 1
fi

R=upstream_runs/imma
L=logs/r1b_imma_train.log
MAX_SAMPLES="${IMMA_MAX_SAMPLES:-20}"
case "$MAX_SAMPLES" in
  0|all|ALL) MS_ARG=(); TAG=sall ;;
  *[!0-9]*)  echo "IMMA_MAX_SAMPLES must be an integer or 'all'" >&2; exit 1 ;;
  *)         MS_ARG=(--max_train_samples "$MAX_SAMPLES"); TAG="s$MAX_SAMPLES" ;;
esac
A="REF_imma_vangogh_${TAG}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data/bijaypandey/t2i_unlearn/hf_home
export HF_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export HUGGINGFACE_HUB_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
export TRANSFORMERS_CACHE=/data/bijaypandey/t2i_unlearn/hf_home/hub
V=/data/bijaypandey/t2i_unlearn/venv_imma/bin/python
VT=/data/bijaypandey/t2i_unlearn/venv_t2i/bin/python

# ---- preflight: every input must exist, and the dataset must be the repaired one
[ -x "$V" ] || { echo "missing venv_imma" >&2; exit 1; }
[ -f "$R/diffusers-VanGogh-ESDx1-UNET.pt" ] || { echo "missing ESD ckpt" >&2; exit 1; }
[ -f "$R/data/vangogh/metadata.csv" ] || { echo "no dataset; run run_r1_imma_official.sh first" >&2; exit 1; }
ROWS=$(( $(wc -l < "$R/data/vangogh/metadata.csv") - 1 ))
[ "$ROWS" -ge 20 ] || { echo "dataset has only $ROWS rows" >&2; exit 1; }
if [ -e "$R/$A" ]; then
  echo "$R/$A already exists; refusing to overwrite an existing artifact" >&2; exit 1
fi

while :; do G=$(bash scripts/campaign_free.sh | head -1); [ -n "$G" ] && break; sleep 120; done
export CUDA_VISIBLE_DEVICES=$G
echo "=== R1b on GPU $G $(date -Is) out=$A max_train_samples=$MAX_SAMPLES dataset_rows=$ROWS" >>"$L"

rc=0
( cd upstream/IMMA && $V -m accelerate.commands.launch --num_processes=1 --mixed_precision=no \
    train/defend_text_to_image_lora.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
    --train_data_dir="/data/bijaypandey/t2i_unlearn/$R/data/vangogh" --caption_column=prompt \
    --resolution=512 --random_flip --train_batch_size=1 --num_train_epochs=50 \
    --learning_rate_lora=1e-04 --learning_rate=1e-05 --lr_scheduler=constant --lr_warmup_steps=0 \
    --seed=42 --report_to=tensorboard --inner_loop_steps=1 --outer_loop_steps=1 "${MS_ARG[@]}" \
    --delta_ckpt="/data/bijaypandey/t2i_unlearn/$R/diffusers-VanGogh-ESDx1-UNET.pt" \
    --output_dir="/data/bijaypandey/t2i_unlearn/$R/$A" ) >>"$L" 2>&1 || rc=$?
echo "  rc=$rc $(date -Is)" >>"$L"
[ "$rc" -eq 0 ] || { echo "R1b TRAINING FAILED rc=$rc" >>"$L"; exit 1; }

CK="$R/$A/imma_unet_xatten_layer.pt"
[ -s "$CK" ] || { echo "R1b PRODUCED NO CHECKPOINT" >>"$L"; exit 1; }
$VT - "$CK" "$R/diffusers-VanGogh-ESDx1-UNET.pt" <<'PY' >>"$L" 2>&1
import sys, torch
imma = torch.load(sys.argv[1], map_location="cpu")
esd  = torch.load(sys.argv[2], map_location="cpu")
esd = esd.get("unet", esd)
n = sum(v.numel() for v in imma.values())
a1 = sum(1 for k in imma if ".attn1." in k); a2 = sum(1 for k in imma if ".attn2." in k)
print(f"  ckpt: {len(imma)} tensors {n/1e6:.3f}M params attn1={a1} attn2={a2}")
assert len(imma) == 160 and a1 == 80 and a2 == 80, "unexpected defence scope"
assert n == 93_536_640, f"unexpected param count {n}"
assert all(torch.isfinite(v).all() for v in imma.values()), "non-finite tensors"
assert all(k in esd for k in imma), "checkpoint keys are not UNet parameter names"
d = sum((imma[k].float()-esd[k].float()).pow(2).sum() for k in imma).sqrt()
w = sum(esd[k].float().pow(2).sum() for k in imma).sqrt()
r = (d/w).item()
print(f"  global ||IMMA-ESD||/||ESD|| over the defence scope = {r:.4f}")
assert r > 1e-5, "the defence scope did not move at all -- training was a no-op"
PY
echo "R1b DONE $(date -Is) -> $R/$A" >>"$L"
