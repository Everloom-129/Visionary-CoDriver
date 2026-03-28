# Visionary-CoDriver — Developer Reference

## Architecture Overview

```
Video Input (30 Hz)
       │
       ├─► Fast Vision (30 Hz)          VCD/fast_vision/
       │     YOLOX detection
       │     ByteTrack tracking
       │     Speed classification
       │     Output: MOT .txt files
       │
       ├─► Slow Vision (2 Hz, every 15th frame)   VCD/slow_vision/
       │     GSAM / DINOX segmentation
       │     DPT depth estimation
       │     Region + surface classification
       │     Output: DPT CSV files
       │
       └─► Fusion Module                VCD/fusion/
             TemporalAligner → nearest-slow-frame mapping
             DetectorFusion  → IoU-based bbox matching
             Output: fused JSON (FusedDetection records)
                     │
                     └─► LLM Agent     VCD/agent/
                           Scene understanding
                           Co-driver decisions
```

## Data Formats

### Fast Vision — MOT .txt (comma separated, no header)
```
frame_id,track_id,bb_left,bb_top,bb_width,bb_height,confidence,-1,-1,-1,speed_class
0,1,1394.25,651.75,93.00,241.50,0.90,-1,-1,-1,0
0,2,461.62,731.62,78.00,115.12,0.87,-1,-1,-1,1
```
- Columns 7-9 are `-1` (unused MOT world coords)
- `speed_class` at col[10]: 0 = slow/stationary, 1 = fast/moving (appended by `track_speed_classifier.py`)
- bytetracker outputs **separate files per class**: `video_0001_person.txt`, `video_0001_car.txt`
- Frame IDs are **0-indexed** (bytetracker starts at 0)

### Slow Vision — DPT CSV (with header row)
```
frame_id,person_id,x1,y1,x2,y2,confidence,avg_depth,angle,distance_level,region_category,surface_type
1,0,1396.06,657.12,1485.56,890.41,0.83,10.98,48.99,3,0,road
```
- `distance_level`: 1=very close (≤5m), 2=close (≤10m), 3=medium (≤15m), 4=far (>15m)
- `region_category`: 0=road, 1=sidewalk, 2=crosswalk, 3=parking, 4=grass, 5=unknown
- `surface_type`: free-text from RoadsideAnalyzer overlap analysis
- Frame IDs are **1-indexed** (extracted frames named `000001.jpg` → frame_id=1)
- Slow frames land at 1, 16, 31, 46 … (every 15th file from 1-indexed filenames)

### Fused Output — JSON (list of FusedDetection.to_dict())
```json
[
  {
    "frame_id": 0,
    "track_id": 1,
    "bbox_xywh": [1394.25, 651.75, 93.0, 241.5],
    "fast_confidence": 0.9,
    "speed_class": 0,
    "object_class": "person",
    "slow_frame_id": 1,
    "avg_depth": 10.98,
    "angle": 48.99,
    "distance_level": 3,
    "region_category": 0,
    "surface_type": "road",
    "slow_confidence": 0.83,
    "match_iou": 0.93
  }
]
```

## Key Source Files

### Fusion Module
| File | Role |
|------|------|
| `VCD/fusion/data_schema.py` | `FastDetection`, `SlowDetection`, `FusedDetection` dataclasses |
| `VCD/fusion/temporal_aligner.py` | Nearest-slow-frame alignment, MOT + CSV loaders |
| `VCD/fusion/detector_fusion.py` | IoU matrix, Hungarian matching, per-frame fusion |
| `VCD/fusion/dataset_loader.py` | `JAAdLoader`, `BDD100KLoader`, `FusionResultLoader` |
| `VCD/fusion/__init__.py` | `run_fusion()` convenience API |
| `VCD/fusion/test_fusion.py` | Synthetic unit tests (6 tests, all passing) |
| `VCD/utils/time_utils.py` | `frame_to_timestamp()`, `nearest_frame()`, timing decorator |
| `VCD/utils/visualization.py` | `Visualizer` class: boxes, masks, traces, heatmaps (uses supervision) |

### Fast Vision (`VCD/fast_vision/`)
| File | Role |
|------|------|
| `bytetracker.py` | Main entry: YOLOX+ByteTrack inference on video dir, writes MOT .txt to `./YOLOX_outputs/` |
| `track_speed_classifier.py` | Post-process MOT .txt → appends speed_class column (col[10]) |
| `visualize_tracking.py` | Overlay MOT results on video |
| `process_video_for_mot.py` | YOLOv8-based alternative tracker (xyxy format, different from bytetracker) |
| `ByteTrack/` | ByteTrack git submodule (YOLOX backbone, multi-size models) |

### Slow Vision (`VCD/slow_vision/`)
| File | Role |
|------|------|
| `DPT_analysis.py` | Main pipeline: GroundingDINO+SAM detection + DPT depth → DPT CSV |
| `gsam_detector.py` | `GSAMDetector`: GroundingDINO + SAM wrapper |
| `roadside_analyzer.py` | `RoadsideAnalyzer`: person-surface overlap, region categorisation |
| `depth_util.py` | `predict_depth()`, `get_distance_category()` |
| `GroundingDINO/` | Text-grounded object detection submodule |
| `DPT/` | Dense Prediction Transformer submodule |
| `segment-anything/` | Meta SAM submodule |

### Agent (`VCD/agent/`) — **stub, not yet implemented**
| File | Role |
|------|------|
| `llm.py` | (empty) LLM integration |
| `risk_analyzer.py` | (empty) Risk assessment |
| `prompt.md` | (empty) LLM prompt templates |

### Scripts & Config
| File | Role |
|------|------|
| `scripts/run_jaad_pipeline.sh` | Full JAAD pipeline: frames → slow → speed → fusion |
| `scripts/download_jaad.sh` | Download JAAD clips + annotations |
| `scripts/download_bdd100k.sh` | Download BDD100K split |
| `scripts/visualize_fusion.py` | Overlay fused JSON results on video; green=slow, red=fast, blue=slow-only |
| `config/config.py` | DDS cloud API token + model config (GDino1_5_Pro) |
| `main.py` | Video processing entry point (WIP — imports `VCD.vision` not yet exists) |

## Pixi Environment Setup

```bash
# Install pixi (one-time)
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc   # or restart your shell

# Install all dependencies + fast vision
pixi install
pixi run setup

# Install slow vision dependencies (one-time, compiles GroundingDINO CUDA extension)
pixi run setup-slow

# Run synthetic fusion tests
pixi run test
pixi run pytest
```

## Running Individual Stages

### Fast Vision (YOLOX + ByteTrack)
```bash
# Run on all JAAD clips → writes to YOLOX_outputs/yolox_x/track_vis/{timestamp}/
pixi run fast-jaad

# Then copy to fast_results/ and run speed classifier
YOLOX_RUN=$(ls -t YOLOX_outputs/yolox_x/track_vis/ | head -1)
cp YOLOX_outputs/yolox_x/track_vis/${YOLOX_RUN}/*_person.txt \
   /mnt/sda/edward/data_vcd/JAAD/fast_results/
pixi run python VCD/fast_vision/track_speed_classifier.py \
   /mnt/sda/edward/data_vcd/JAAD/fast_results/
```

### Slow Vision (GroundingDINO + SAM + DPT)
```bash
# Step 1: Extract frames from all clips (run once; skips existing)
for f in /mnt/sda/edward/data_vcd/JAAD/JAAD_clips/*.mp4; do
    name=$(basename "$f" .mp4)
    outdir=/mnt/sda/edward/data_vcd/JAAD/frames/${name}
    [ -d "$outdir" ] && continue
    mkdir -p "$outdir"
    ffmpeg -i "$f" -start_number 1 "$outdir/%06d.jpg" -y -loglevel error
done

# Step 2: Run slow vision (~9 s/frame × 40 frames/clip × 346 clips ≈ 34 h)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
pixi run python -m VCD.slow_vision.DPT_analysis \
    /mnt/sda/edward/data_vcd/JAAD/frames \
    /mnt/sda/edward/data_vcd/JAAD/slow_results \
    --downsample 15
```

### Fusion
```bash
pixi run python -c "
from VCD.fusion import run_fusion
import os, glob

fast_dir   = '/mnt/sda/edward/data_vcd/JAAD/fast_results'
slow_dir   = '/mnt/sda/edward/data_vcd/JAAD/slow_results'
output_dir = '/mnt/sda/edward/data_vcd/JAAD/fusion_results'
os.makedirs(output_dir, exist_ok=True)

for fast_txt in sorted(glob.glob(f'{fast_dir}/*_person.txt')):
    name = os.path.basename(fast_txt).replace('_person.txt', '')
    slow_csv = f'{slow_dir}/{name}.csv'
    out_json = f'{output_dir}/{name}.json'
    if not os.path.exists(slow_csv) or os.path.exists(out_json):
        continue
    fused = run_fusion(fast_txt, slow_csv, out_json)
    print(f'{name}: {len(fused)} records')
"
```

### Full Pipeline (all stages)
```bash
bash scripts/run_jaad_pipeline.sh
```

## Dataset Paths

| Dataset | Expected Root |
|---------|---------------|
| JAAD    | `/mnt/sda/edward/data_vcd/JAAD/` |
| BDD100K | `/mnt/sda/edward/data_vcd/BDD100K/` |

Expected subdirectories:
```
<root>/
  JAAD_clips/        # source .mp4 files (346 clips)
  annotations/       # original dataset labels
  frames/            # extracted JPEG frames, one subdir per clip
  fast_results/      # *_person.txt and *_car.txt from bytetracker + speed classifier
  slow_results/      # *.csv from DPT_analysis.py
  fusion_results/    # *.json from run_fusion()
```

## Temporal Alignment Details

- Fast stream: 30 Hz → frame ids 0, 1, 2, … (bytetracker is 0-indexed)
- Slow stream: 2 Hz → frame ids 1, 16, 31, 46 … (1-indexed filenames, step every 15 files)
- `slow_period = 15`
- Nearest-slow-frame: for fast frame `f`, find closest slow frame_id within `max_gap=15`
- Fast frame 0 → nearest slow frame 1 (gap=1 ✓)
- Fast frame 8 → nearest slow frame 1 (gap=7 ✓)
- Fast frame 15 → nearest slow frame 16 (gap=1 ✓)
- If gap > max_gap → carry-forward last valid slow frame

## MOT Format Auto-Detection

`load_fast_detections()` auto-detects three formats:

| Format | Source | Columns | bbox | Detection rule |
|--------|--------|---------|------|----------------|
| **Standard MOT** | `bytetracker.py` | 10-11 cols | xywh | col[7]==`-1` AND col[8]==`-1` |
| **Pipeline** | `process_video_for_mot.py` | 10-11 cols | xyxy → converted | col[7]≠`-1`, col[8]==`-1` |
| **Fusion spec** | manual / tests | ≤9 cols | xywh | all others |

For Standard MOT: `speed_class` read from col[10] if present, else defaults to 0. `object_class` defaults to `"person"` (bytetracker writes single-class files).

`load_slow_detections()` handles CSV column names case-insensitively and defaults `confidence=1.0`, `surface_type="unknown"` if missing.

## Slow Vision Setup Notes

Dependencies installed via `pixi run setup-slow`:
- `groundingdino` (from `VCD/slow_vision/GroundingDINO/`, requires CUDA compilation)
- `segment_anything` (from `VCD/slow_vision/segment-anything/`)
- `dpt` (from `VCD/slow_vision/DPT/`)
- `transformers>=4.30,<5.0` — **must pin below 5.0**; transformers 5.x removed `BertModel.get_head_mask` which GroundingDINO relies on

`DPT_analysis.py` expects **extracted frames in subdirectories** (not raw video):
```
frames/
  video_0001/
    000001.jpg   ← frame_id = 1
    000002.jpg
    ...
```

Memory notes (12 GB GPU):
- SAM ViT-H (~3 GB) + GroundingDINO (~1 GB) + DPT (~1.3 GB) loaded per-clip
- DPT model is loaded and freed per-frame (`del model; gc.collect(); torch.cuda.empty_cache()`)
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid fragmentation OOM

## Visualising Fusion Results

```bash
python scripts/visualize_fusion.py \
    --fused   /mnt/sda/edward/data_vcd/JAAD/fusion_results/video_0001.json \
    --video   /mnt/sda/edward/data_vcd/JAAD/JAAD_clips/video_0001.mp4 \
    --out     /mnt/sda/edward/data_vcd/JAAD/fusion_results/video_0001_annotated.mp4
```
Colour coding: **green** = slow/stationary, **red** = fast/moving, **blue** = slow-only (no fast match).
Solid box = matched (has depth), dashed = fast-only.

## Performance Reference (12 GB GPU, measured on JAAD video_0001)

| Module | Speed |
|--------|-------|
| GroundingDINO + SAM | ~3.7 s/frame |
| DPT depth | ~5.2 s/frame |
| Total slow vision | ~9 s/frame @ 2 Hz (40 frames/clip) |
| Fast vision (YOLOX+ByteTrack) | real-time |
| Fusion (CPU) | real-time |
| **Full dataset estimate** | ~34 h for all 346 JAAD clips (slow vision dominates) |

## Adding New Detection Classes

1. Add field to `FastDetection` or `SlowDetection` in `data_schema.py`
2. Update `load_fast_detections` / `load_slow_detections` in `temporal_aligner.py`
3. Propagate new field in `DetectorFusion._fast_only` / `_slow_only` / `fuse_frame`
4. Add field to `FusedDetection.to_dict()` and `FusionResultLoader._from_dict()`
