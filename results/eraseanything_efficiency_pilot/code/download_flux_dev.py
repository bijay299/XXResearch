"""Download FLUX.1-dev diffusers-format files at a pinned revision. CPU/network only."""
import os, json, pathlib
from huggingface_hub import snapshot_download

REV = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
REPO = "black-forest-labs/FLUX.1-dev"
# token comes from the environment; the secrets file is sourced by the caller and is
# never read into, or echoed by, this script.
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
assert tok, "no token in environment"
# diffusers-format only: skip the single-file checkpoint, the raw AE and the grid jpeg
p = snapshot_download(
    REPO, revision=REV, token=tok,
    ignore_patterns=["flux1-dev.safetensors", "ae.safetensors", "dev_grid.jpg", "*.md"],
    max_workers=8,
)
print("SNAPSHOT", p)
print(json.dumps(sorted(os.path.relpath(str(q), p) for q in pathlib.Path(p).rglob("*") if q.is_file()), indent=1))
