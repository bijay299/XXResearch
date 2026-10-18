#!/bin/bash
set -x
cd /data/bijaypandey/cuig_pilot/block_object_sprint/env
export HF_HOME=/data/bijaypandey/cuig_pilot/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
python3.10 -m venv venv-llava
./venv-llava/bin/pip install -q --upgrade pip
./venv-llava/bin/pip install \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.1.2" "torchvision==0.16.2" "transformers==4.37.2" "tokenizers==0.15.1" \
  "sentencepiece==0.1.99" "accelerate==0.21.0" "shortuuid" "pydantic" "numpy==1.26.4" \
  "scikit-learn==1.2.2" "pillow==10.4.0" "pandas==2.2.2" "tqdm" "protobuf" "einops" \
  "einops-exts" "timm==0.6.13" "huggingface-hub==0.25.2" "bitsandbytes==0.41.0"
echo "PIP_EXIT=$?"
./venv-llava/bin/python -c "import torch,transformers;print('llava-env torch',torch.__version__,'tf',transformers.__version__)"
./venv-llava/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p=snapshot_download("liuhaotian/llava-v1.6-vicuna-7b",
                    allow_patterns=["*.json","*.model","*.safetensors","*.bin","*.txt"])
print("LLAVA_PATH="+p)
PY
echo "SETUP_DONE=$?"
