#!/bin/bash
# ---------------------------------------------------------------------------
# fetch_assets.sh - download the UnlearnCanvas generator and classifiers.
#
# Follows CUIG's Checkpoints/README.md exactly (same Drive folder ids, same
# renames), but writes into PILOT_DATA_ROOT on /data instead of into the
# upstream checkout, because the filesystem holding /home is full.
#
# Assets are deliberately NOT tracked in git; see docs/ASSETS.md.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/pilot_config.sh"
# shellcheck disable=SC1091
source "${PILOT_VENV}/bin/activate"

CKPT_ROOT="${PILOT_DATA_ROOT}/Checkpoints"
GEN_DIR="${CKPT_ROOT}/Generators"
CLS_DIR="${CKPT_ROOT}/Classifiers"
mkdir -p "$GEN_DIR" "$CLS_DIR"

# --- Classifiers: Drive folder from Checkpoints/README.md -------------------
if [[ ! -f "${CLS_DIR}/UnlearnCanvas/style_classifier.pth" \
   || ! -f "${CLS_DIR}/UnlearnCanvas/object_classifier.pth" ]]; then
    echo "### Downloading UnlearnCanvas classifiers ###"
    cd "$CLS_DIR"
    gdown --folder https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61
    if [[ -d cls_model ]]; then
        mv cls_model UnlearnCanvas
    fi
    cd UnlearnCanvas
    rm -f style60.pth
    [[ -f style50.pth     ]] && mv -n style50.pth     style_classifier.pth
    [[ -f style50_cls.pth ]] && mv -n style50_cls.pth  object_classifier.pth
    cd "$CLS_DIR"
else
    echo "### Classifiers already present, skipping ###"
fi

# --- Generator: Drive folder from Checkpoints/README.md ---------------------
if [[ ! -f "${GEN_DIR}/UnlearnCanvas/model_index.json" ]]; then
    echo "### Downloading UnlearnCanvas generator ###"
    cd "$GEN_DIR"
    gdown --folder https://drive.google.com/drive/folders/18x40pLBcfNFyxBWZBGncTjqJTs_75SLx
    if [[ -d style50 ]]; then
        mv style50 UnlearnCanvas
    fi
else
    echo "### Generator already present, skipping ###"
fi

echo
echo "### Layout ###"
find "$CKPT_ROOT" -maxdepth 4 | sort
echo
echo "### Sizes ###"
du -sh "${GEN_DIR}/UnlearnCanvas" "${CLS_DIR}/UnlearnCanvas" 2>/dev/null || true
