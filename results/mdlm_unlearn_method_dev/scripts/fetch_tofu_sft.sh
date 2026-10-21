#!/bin/bash
# Authorized by the user: download the public, ungated LLaDA-8B TOFU-forget10 SFT checkpoint
# (Jameszhiwei/llada-tofu-forget10-SFT). This is the WARM-UP model the ICML 2026 paper
# describes building, published by that paper's likely first author. It gives a third-party
# matched starting point, removing our own injection recipe as a confound.
# NOTE: the same author's "...-unlearned" repo contains only .gitattributes -- no weights --
# so their unlearned OUTPUT is not available and no head-to-head on their result is possible.
set -eu
D=/data/bijaypandey/mdlm_unlearn/models/llada-tofu-forget10-SFT
mkdir -p "$D"
REPO=Jameszhiwei/llada-tofu-forget10-SFT
for f in config.json generation_config.json model.safetensors.index.json \
         special_tokens_map.json tokenizer.json tokenizer_config.json \
         model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
         model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  if [ -s "$D/$f" ]; then echo "[have] $f"; continue; fi
  echo "[get ] $f"
  curl -sSL --retry 3 --max-time 3600 -o "$D/$f.part" \
    "https://huggingface.co/$REPO/resolve/main/$f"
  mv "$D/$f.part" "$D/$f"
done
echo "[done] $(du -sh $D | cut -f1)"
sha256sum "$D"/*.safetensors "$D"/config.json > "$D/SHA256SUMS.txt"
echo "[hashed] SHA256SUMS.txt written"
