#!/usr/bin/env bash
# =============================================================================
# Download JAAD (Joint Attention in Autonomous Driving) dataset
# =============================================================================
#
# JAAD is a pedestrian behaviour dataset recorded from a front-facing dashcam.
# Official page: https://data.nvision2.eecs.yorku.ca/JAAD_dataset/
#
# Dataset components
# ------------------
# 1. JAAD_clips/       — 346 short video clips (~5-10 s each), ~22 GB total
# 2. annotations/      — per-clip XML files with pedestrian tracks + behaviour
# 3. behavioral_data/  — optional per-pedestrian behaviour annotations
#
# Prerequisites
# -------------
#   sudo apt-get install -y aria2 wget
#
# NOTE: JAAD clips are hosted via the authors' website and require
#       registration/agreement.  The URLs below point to the GitHub release
#       data.  Adjust if the authors move the data.
# =============================================================================

set -euo pipefail

DATA_ROOT="${1:-/mnt/sda/edward/data_vcd/JAAD}"

echo "[JAAD] Target directory: $DATA_ROOT"
mkdir -p "$DATA_ROOT/JAAD_clips"
mkdir -p "$DATA_ROOT/annotations"
mkdir -p "$DATA_ROOT/behavioral_data"
mkdir -p "$DATA_ROOT/fast_results"
mkdir -p "$DATA_ROOT/slow_results"

# ---------------------------------------------------------------------------
# 1. Clone the JAAD repository (contains annotations + helper scripts)
# ---------------------------------------------------------------------------
JAAD_REPO="https://github.com/ykotseruba/JAAD.git"
if [ ! -d "$DATA_ROOT/JAAD_repo" ]; then
    echo "[JAAD] Cloning JAAD repository (annotations + scripts) ..."
    git clone --depth 1 "$JAAD_REPO" "$DATA_ROOT/JAAD_repo"
    echo "[JAAD] Linking annotations ..."
    ln -sfn "$DATA_ROOT/JAAD_repo/annotations" "$DATA_ROOT/annotations"
else
    echo "[JAAD] Repository already cloned — skipping."
fi

# ---------------------------------------------------------------------------
# 2. Download video clips
#    The JAAD authors provide a download script inside the repo.
# ---------------------------------------------------------------------------
CLIP_DOWNLOAD_SCRIPT="$DATA_ROOT/JAAD_repo/download_clips.sh"
if [ -f "$CLIP_DOWNLOAD_SCRIPT" ]; then
    echo "[JAAD] Running official clip download script ..."
    bash "$CLIP_DOWNLOAD_SCRIPT" "$DATA_ROOT/JAAD_clips"
else
    echo "[JAAD] WARNING: download script not found at $CLIP_DOWNLOAD_SCRIPT"
    echo "       Please manually download clips from:"
    echo "       https://data.nvision2.eecs.yorku.ca/JAAD_dataset/"
    echo "       and place .mp4 files under $DATA_ROOT/JAAD_clips/"
fi

# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------
CLIP_COUNT=$(find "$DATA_ROOT/JAAD_clips" -name "*.mp4" | wc -l)
XML_COUNT=$(find "$DATA_ROOT/annotations" -name "*.xml" 2>/dev/null | wc -l)

echo ""
echo "[JAAD] Download complete."
echo "  Clips found      : $CLIP_COUNT"
echo "  Annotation XMLs  : $XML_COUNT"
echo "  Data root        : $DATA_ROOT"
echo ""
echo "Next steps:"
echo "  1. Run fast-vision pipeline on clips → $DATA_ROOT/fast_results/"
echo "  2. Run slow-vision pipeline on clips → $DATA_ROOT/slow_results/"
echo "  3. Run fusion:  python VCD/fusion/test_fusion.py --jaad $DATA_ROOT"
