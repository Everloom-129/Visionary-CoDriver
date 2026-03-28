#!/usr/bin/env bash
# =============================================================================
# Download BDD100K dataset (detection + tracking subset)
# =============================================================================
#
# BDD100K is a large-scale driving dataset from UC Berkeley.
# Official page: https://bdd-data.berkeley.edu/
#
# Required components for this project
# --------------------------------------
# 1. videos/   — 100K driving videos (~1080p, 40s each)  [~1.8 TB for all]
#               For testing, 10K-video MOT subset is sufficient (~35 GB)
# 2. labels/   — Detection + tracking annotations (JSON)
#
# Download options
# ----------------
# A) via bdd100k Python package (recommended for structured download)
# B) via direct wget from the official CDN (requires login token)
#
# =============================================================================

set -euo pipefail

DATA_ROOT="${1:-/mnt/sda/edward/data_vcd/BDD100K}"
SPLIT="${2:-val}"   # train | val | test

echo "[BDD100K] Target directory: $DATA_ROOT"
echo "[BDD100K] Split: $SPLIT"

mkdir -p "$DATA_ROOT/videos/$SPLIT"
mkdir -p "$DATA_ROOT/labels/det_20"
mkdir -p "$DATA_ROOT/labels/box_track_20"
mkdir -p "$DATA_ROOT/fast_results"
mkdir -p "$DATA_ROOT/slow_results"

# ---------------------------------------------------------------------------
# Option A: bdd100k Python package
# ---------------------------------------------------------------------------
echo ""
echo "[BDD100K] Option A — using bdd100k Python package"
echo "  Install: pip install bdd100k"
echo "  Then run:"
echo ""
echo "    python -m bdd100k.common.utils.download \\"
echo "        --task det \\                # or: track, drivable, seg, ..."
echo "        --split $SPLIT \\"
echo "        --output $DATA_ROOT"
echo ""

# ---------------------------------------------------------------------------
# Option B: Direct download via official links
# ---------------------------------------------------------------------------
echo "[BDD100K] Option B — direct wget (requires login + token)"
echo "  1. Register at https://bdd-data.berkeley.edu/"
echo "  2. Download token from your account page"
echo "  3. Use wget with your token:"
echo ""
echo "    # MOT 2020 detection labels (~50 MB):"
echo "    wget -O $DATA_ROOT/bdd100k_det_20_labels_trainval.zip \\"
echo "        'https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip'"
echo ""
echo "    # Videos — MOT subset (val, ~4 GB):"
echo "    wget -O $DATA_ROOT/bdd100k_mot_20_images_val.zip \\"
echo "        'https://dl.cv.ethz.ch/bdd100k/data/bdd100k_mot_20_images_val.zip'"
echo ""

# ---------------------------------------------------------------------------
# Clone bdd100k toolkit (for annotation parsing utilities)
# ---------------------------------------------------------------------------
BDD_REPO="https://github.com/bdd100k/bdd100k.git"
if [ ! -d "$DATA_ROOT/bdd100k_repo" ]; then
    echo "[BDD100K] Cloning bdd100k toolkit ..."
    git clone --depth 1 "$BDD_REPO" "$DATA_ROOT/bdd100k_repo"
else
    echo "[BDD100K] Toolkit already cloned — skipping."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
VIDEO_COUNT=$(find "$DATA_ROOT/videos/$SPLIT" -name "*.mp4" 2>/dev/null | wc -l)
echo ""
echo "[BDD100K] Setup complete."
echo "  Videos found ($SPLIT): $VIDEO_COUNT"
echo "  Data root             : $DATA_ROOT"
echo ""
echo "Next steps:"
echo "  1. Run fast-vision pipeline → $DATA_ROOT/fast_results/"
echo "  2. Run slow-vision pipeline → $DATA_ROOT/slow_results/"
echo "  3. Run fusion pipeline on the results"
