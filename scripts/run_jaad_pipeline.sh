#!/usr/bin/env bash
# run_jaad_pipeline.sh — Full JAAD processing pipeline
#
# Stages (each is resumable — already-processed clips are skipped):
#   1. Copy fast results from latest YOLOX run → fast_results/
#   2. Run speed classifier on fast_results/
#   3. Extract frames from all JAAD clips → frames/
#   4. Run slow vision (GroundingDINO + SAM + DPT) → slow_results/
#   5. Run fusion for all clips with both fast + slow results → fusion_results/
#
# Usage:
#   bash scripts/run_jaad_pipeline.sh [--stages 1,2,3,4,5] [--clips video_0001,video_0002]
#
# Options:
#   --stages  Comma-separated list of stages to run (default: all)
#   --clips   Comma-separated clip names to process (default: all)
#
# Example (run only slow vision + fusion on two clips):
#   bash scripts/run_jaad_pipeline.sh --stages 4,5 --clips video_0001,video_0002

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
JAAD_ROOT="/mnt/sda/edward/data_vcd/JAAD"
CLIPS_DIR="${JAAD_ROOT}/JAAD_clips"
FRAMES_DIR="${JAAD_ROOT}/frames"
FAST_DIR="${JAAD_ROOT}/fast_results"
SLOW_DIR="${JAAD_ROOT}/slow_results"
FUSION_DIR="${JAAD_ROOT}/fusion_results"
YOLOX_BASE="YOLOX_outputs/yolox_x/track_vis"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
STAGES="1,2,3,4,5"
CLIPS_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --stages) STAGES="$2"; shift 2 ;;
        --clips)  CLIPS_FILTER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

run_stage() { echo "$STAGES" | tr ',' '\n' | grep -qx "$1"; }

cd "$PROJECT_ROOT"

echo "=================================================="
echo " Visionary-CoDriver — JAAD Pipeline"
echo " Stages : $STAGES"
echo " Root   : $JAAD_ROOT"
echo "=================================================="

# ---------------------------------------------------------------------------
# Stage 1: Copy latest YOLOX fast results → fast_results/
# ---------------------------------------------------------------------------
if run_stage 1; then
    echo ""
    echo "--- Stage 1: Copy fast results ---"
    mkdir -p "$FAST_DIR"

    LATEST_RUN=$(ls -t "${YOLOX_BASE}/" 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "ERROR: No YOLOX output found in ${YOLOX_BASE}/. Run 'pixi run fast-jaad' first."
        exit 1
    fi
    echo "Using YOLOX run: $LATEST_RUN"
    RUN_DIR="${YOLOX_BASE}/${LATEST_RUN}"

    copied=0; skipped=0
    for src in "${RUN_DIR}"/*_person.txt; do
        [ -f "$src" ] || continue
        name=$(basename "$src")
        dst="${FAST_DIR}/${name}"
        if [ ! -f "$dst" ]; then
            cp "$src" "$dst"
            ((copied++)) || true
        else
            ((skipped++)) || true
        fi
    done
    echo "  Copied ${copied} person files, skipped ${skipped} already present"
fi

# ---------------------------------------------------------------------------
# Stage 2: Speed classifier on all fast_results/
# ---------------------------------------------------------------------------
if run_stage 2; then
    echo ""
    echo "--- Stage 2: Speed classifier ---"
    pixi run python VCD/fast_vision/track_speed_classifier.py "$FAST_DIR"
    echo "  Speed classification done"
fi

# ---------------------------------------------------------------------------
# Stage 3: Extract frames from JAAD clips
# ---------------------------------------------------------------------------
if run_stage 3; then
    echo ""
    echo "--- Stage 3: Frame extraction ---"
    mkdir -p "$FRAMES_DIR"

    total=0; done_count=0; extracted=0
    for vid in "${CLIPS_DIR}"/*.mp4; do
        [ -f "$vid" ] || continue
        name=$(basename "$vid" .mp4)

        # Apply clip filter if specified
        if [ -n "$CLIPS_FILTER" ]; then
            echo "$CLIPS_FILTER" | tr ',' '\n' | grep -qx "$name" || continue
        fi

        outdir="${FRAMES_DIR}/${name}"
        ((total++)) || true

        if [ -d "$outdir" ] && [ "$(ls -A "$outdir" 2>/dev/null)" ]; then
            ((done_count++)) || true
            continue
        fi

        mkdir -p "$outdir"
        ffmpeg -i "$vid" -start_number 1 "${outdir}/%06d.jpg" \
               -y -loglevel error
        ((extracted++)) || true
        echo "  [${extracted}] Extracted: ${name}"
    done
    echo "  Done — extracted ${extracted} clips, ${done_count}/${total} already existed"
fi

# ---------------------------------------------------------------------------
# Stage 4: Slow vision (GroundingDINO + SAM + DPT)
# ---------------------------------------------------------------------------
if run_stage 4; then
    echo ""
    echo "--- Stage 4: Slow vision ---"
    mkdir -p "$SLOW_DIR"

    # If clip filter is set, process only those subdirs; otherwise process all.
    # DPT_analysis.py processes all subdirs of the input folder.
    # For selective processing we use a temp symlink dir.
    if [ -n "$CLIPS_FILTER" ]; then
        FRAMES_INPUT=$(mktemp -d)
        trap "rm -rf '$FRAMES_INPUT'" EXIT
        echo "$CLIPS_FILTER" | tr ',' '\n' | while read -r name; do
            [ -d "${FRAMES_DIR}/${name}" ] && ln -s "${FRAMES_DIR}/${name}" "${FRAMES_INPUT}/${name}"
        done
    else
        # Skip clips already processed (CSV exists) by building a temp dir of symlinks
        FRAMES_INPUT=$(mktemp -d)
        trap "rm -rf '$FRAMES_INPUT'" EXIT
        skipped_slow=0; queued=0
        for d in "${FRAMES_DIR}"/*/; do
            name=$(basename "$d")
            if [ -f "${SLOW_DIR}/${name}.csv" ]; then
                ((skipped_slow++)) || true
            else
                ln -s "${FRAMES_DIR}/${name}" "${FRAMES_INPUT}/${name}"
                ((queued++)) || true
            fi
        done
        echo "  Queued ${queued} clips, skipping ${skipped_slow} already processed"
    fi

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    pixi run python -m VCD.slow_vision.DPT_analysis \
        "$FRAMES_INPUT" \
        "$SLOW_DIR" \
        --downsample 15
fi

# ---------------------------------------------------------------------------
# Stage 5: Fusion
# ---------------------------------------------------------------------------
if run_stage 5; then
    echo ""
    echo "--- Stage 5: Fusion ---"
    mkdir -p "$FUSION_DIR"

    pixi run python - <<PYEOF
import os, glob
from VCD.fusion import run_fusion

fast_dir   = "${FAST_DIR}"
slow_dir   = "${SLOW_DIR}"
fusion_dir = "${FUSION_DIR}"
clip_filter = "${CLIPS_FILTER}"

clips = [c for c in clip_filter.split(',') if c] if clip_filter else None

fused_total = 0; skipped = 0; errors = 0
for fast_txt in sorted(glob.glob(f"{fast_dir}/*_person.txt")):
    name = os.path.basename(fast_txt).replace("_person.txt", "")
    if clips and name not in clips:
        continue
    slow_csv = f"{slow_dir}/{name}.csv"
    out_json = f"{fusion_dir}/{name}.json"
    if not os.path.exists(slow_csv):
        continue
    if os.path.exists(out_json):
        skipped += 1
        continue
    try:
        fused = run_fusion(fast_txt, slow_csv, out_json)
        matched = sum(1 for d in fused if d.match_iou is not None)
        print(f"  {name}: {len(fused)} records ({matched} matched)")
        fused_total += len(fused)
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        errors += 1

print(f"Fusion complete — {fused_total} total records, {skipped} skipped, {errors} errors")
PYEOF
fi

echo ""
echo "=================================================="
echo " Pipeline complete"
echo "  fast_results/   : $(ls ${FAST_DIR}/*.txt 2>/dev/null | wc -l) files"
echo "  slow_results/   : $(ls ${SLOW_DIR}/*.csv 2>/dev/null | wc -l) files"
echo "  fusion_results/ : $(ls ${FUSION_DIR}/*.json 2>/dev/null | wc -l) files"
echo "=================================================="
