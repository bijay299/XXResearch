#!/bin/bash
W=/data/bijaypandey/llada_smoke
REV=08b83a6feb34df1a6011b80c3c00c7563e963b07
D=$W/model/LLaDA-8B-Instruct
BASE=https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/resolve/$REV
FILES="config.json configuration_llada.py generation_config.json modeling_llada.py model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer_config.json README.md"
for f in $FILES; do
  curl -sSL --retry 3 --fail "$BASE/$f" -o "$D/$f" || echo "FAIL $f"
done
echo "small files done"
# weights, 3 parallel streams
for i in 1 2 3 4 5 6; do
  f=$(printf "model-0000%d-of-00006.safetensors" $i)
  ( curl -sSL --retry 3 --fail "$BASE/$f" -o "$D/$f" || echo "FAIL $f" ) &
  if (( i % 3 == 0 )); then wait; fi
done
wait
echo "ALL DOWNLOADS COMPLETE"
du -sh $D
