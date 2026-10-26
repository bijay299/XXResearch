#!/bin/bash
# Isolated environments for the upstream reference implementations. v2, after a recorded failure.
#
# FIRST ATTEMPT FAILED and the errors are kept in the manifest:
#   venv_imma: `pip install datasets torchvision` resolved to torch 2.14.1+cu130 and
#              huggingface-hub 1.33.0, breaking transformers 4.46.3 (needs hub <1.0).
#   venv_uce:  `pip install diffusers>=0.34.0` pulled hub 1.33.0 and a diffusers that calls
#              torch.accelerator, absent in torch 2.4.1 ->
#              AttributeError: module 'torch' has no attribute 'accelerator'
# Root cause: unpinned installs resolving to latest. Fix: a constraints file + --no-deps.
#
# venv_t2i is never modified: every env is a COPY and all pip calls target the copy's interpreter.
set -u
cd /data/bijaypandey/t2i_unlearn
L=logs/upstream_envs.log
export PIP_CACHE_DIR=/data/bijaypandey/pipcache
printf 'torch==2.4.1\nhuggingface-hub<1.0\ntransformers==4.46.3\ndiffusers==0.31.0\n' > /tmp/upstream_constraints.txt
echo "=== v2 setup $(date -Is)" >>"$L"

# --- venv_uce: UCE's trainscripts import cleanly on the current stack (12 modules, 0 failures),
#     so its requirements.txt pin of diffusers>=0.34 is NOT needed. Unmodified copy.
if [ ! -d venv_uce ]; then
  cp -a venv_t2i venv_uce && sed -i "s|/venv_t2i|/venv_uce|g" venv_uce/pyvenv.cfg 2>/dev/null
  echo "venv_uce: unmodified copy (no installs needed)" >>"$L"
fi
venv_uce/bin/python -c "import diffusers,torch,transformers;print(f'  venv_uce OK torch={torch.__version__} diffusers={diffusers.__version__} transformers={transformers.__version__}')" >>"$L" 2>&1

# --- venv_imma: needs datasets + torchvision ONLY. Pin to avoid the torch upgrade.
if [ ! -d venv_imma ]; then
  cp -a venv_t2i venv_imma && sed -i "s|/venv_t2i|/venv_imma|g" venv_imma/pyvenv.cfg 2>/dev/null
fi
venv_imma/bin/python -m pip install --quiet --no-deps torchvision==0.19.1 >>"$L" 2>&1 \
  && echo "venv_imma: torchvision==0.19.1 (--no-deps, matches torch 2.4.1)" >>"$L" \
  || echo "venv_imma: torchvision INSTALL FAILED" >>"$L"
venv_imma/bin/python -m pip install --quiet -c /tmp/upstream_constraints.txt datasets >>"$L" 2>&1 \
  && echo "venv_imma: datasets (constrained)" >>"$L" \
  || echo "venv_imma: datasets INSTALL FAILED" >>"$L"
venv_imma/bin/python -c "
import torch,torchvision,datasets,diffusers,transformers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
print(f'  venv_imma OK torch={torch.__version__} tv={torchvision.__version__} datasets={datasets.__version__} diffusers={diffusers.__version__} transformers={transformers.__version__}')
print('  LoRAAttnProcessor + AttnProcsLayers import OK')" >>"$L" 2>&1
echo "ENVS v2 DONE $(date -Is)" >>"$L"
