# Visionary CoDriver (VCD)
Tony Wang · 2025

## Overview

VCD is a dashcam co-driver system that fuses high-frequency object tracking with low-frequency semantic scene understanding, then feeds structured natural-language descriptions to an LLM for pedestrian risk reasoning.

```
Video Input (30 Hz)
       │
       ├─► Fast Vision (30 Hz)          VCD/fast_vision/
       │     YOLOX detection
       │     ByteTrack tracking
       │     Speed classification (slow / fast)
       │     Output: MOT .txt per class per clip
       │
       ├─► Slow Vision (2 Hz)           VCD/slow_vision/
       │     GroundingDINO + SAM segmentation
       │     DPT depth estimation
       │     Surface classification (road / sidewalk / crosswalk …)
       │     Output: DPT CSV per clip
       │
       └─► Fusion Module                VCD/fusion/
             Nearest-slow-frame alignment
             IoU-based bbox matching
             Output: fused JSON per clip
                     │
                     └─► Agent          VCD/agent/
                           GOFAI behavior inference
                           (crossing / approaching / parallel / stationary / receding)
                           Scene text assembly
                           GPT-3.5 co-driver reasoning
```

## Features

1. **Multi-object tracking** — YOLOX + ByteTrack at 30 Hz; globally unique track IDs across the clip
2. **Semantic surface segmentation** — GroundingDINO + SAM at 2 Hz; road / sidewalk / crosswalk / parking / grass
3. **Monocular depth estimation** — Dense Prediction Transformer (DPT); distance level and raw depth per pedestrian
4. **Slow-fast fusion** — temporal alignment + IoU matching; every fast frame gets depth and surface context from the nearest slow frame
5. **GOFAI behavior inference** — trajectory-based classifier over a 1-second sliding window; derives *what the pedestrian is doing* (crossing, approaching, etc.) rather than a single distance score
6. **LLM co-driver** — assembled natural-language scene description sent to GPT-3.5; outputs situation summary, per-person risk level, and recommended driver action
7. **Interactive dashboard** — Gradio app with frame scrubber, behavior-labelled bounding boxes, detection tables, and an LLM reasoning panel

## Installation

```bash
# Install pixi (one-time)
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# Install all dependencies
pixi install
pixi run setup          # installs OpenCV, ByteTrack, and tracker extras

# Slow vision dependencies (compiles GroundingDINO CUDA extension — one-time)
pixi run setup-slow

# LLM agent dependency
pixi run pip install openai

# Dashboard dependency
pixi run setup-dashboard
```

## Running the Pipeline

### Fast Vision (YOLOX + ByteTrack)

```bash
# Run on all JAAD clips → outputs to YOLOX_outputs/yolox_x/track_vis/<timestamp>/
pixi run fast-jaad

# Copy results and run speed classifier
YOLOX_RUN=$(ls -t YOLOX_outputs/yolox_x/track_vis/ | head -1)
cp YOLOX_outputs/yolox_x/track_vis/${YOLOX_RUN}/*_person.txt \
   /mnt/sda/edward/data_vcd/JAAD/fast_results/
pixi run python VCD/fast_vision/track_speed_classifier.py \
   /mnt/sda/edward/data_vcd/JAAD/fast_results/
```

### Slow Vision (GroundingDINO + SAM + DPT)

```bash
# Extract frames (run once — skips existing)
for f in /mnt/sda/edward/data_vcd/JAAD/JAAD_clips/*.mp4; do
    name=$(basename "$f" .mp4)
    outdir=/mnt/sda/edward/data_vcd/JAAD/frames/${name}
    [ -d "$outdir" ] && continue
    mkdir -p "$outdir"
    ffmpeg -i "$f" -start_number 1 "$outdir/%06d.jpg" -y -loglevel error
done

# Run slow vision (~9 s/frame; set expandable segments to avoid fragmentation OOM)
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

### Agent (GOFAI Behavior Inference + GPT-3.5)

```bash
export OPENAI_API_KEY="sk-..."

# Inspect assembled scene text without calling the LLM
python -m VCD.agent /mnt/sda/edward/data_vcd/JAAD/fusion_results/video_0001.json --scene-only

# Full co-driver analysis (prints scene text, then LLM response)
python -m VCD.agent /mnt/sda/edward/data_vcd/JAAD/fusion_results/video_0001.json --print-scene

# Finer temporal resolution (every slow keyframe instead of every second)
python -m VCD.agent /mnt/sda/edward/data_vcd/JAAD/fusion_results/video_0001.json --stride 15
```

From Python:

```python
from VCD.agent import run_agent, assemble_scene

# Scene text only (no API cost)
text = assemble_scene("fusion_results/video_0001.json", stride=30)
print(text)

# Full agent
response = run_agent("fusion_results/video_0001.json", print_scene=True)
print(response)
```

## Interactive Dashboard

```bash
pixi run dashboard   # → http://localhost:7888
```

Features:
- **Clip selector** with frame scrubber and play/pause (0.2×–2× speed)
- **Overlay layers**: fast bboxes (speed-coloured) · slow region fills (surface-coloured) · fused detections (behavior-coloured)
- **Behavior badges** on each fused bbox: CROSSING (red) · APPROACHING (orange) · PARALLEL (yellow) · STATIONARY on road (amber) · STATIONARY on sidewalk (green) · RECEDING (blue-grey)
- **Detection tables**: fast track data · nearest slow frame · fused with motion type and behavior description
- **Agent Reasoning panel**: paste an OpenAI API key, click "Run Agent" to get the GPT-3.5 co-driver narrative for the current clip; assembled scene text is viewable for inspection

## Dataset Paths

| Dataset | Root |
|---------|------|
| JAAD    | `/mnt/sda/edward/data_vcd/JAAD/` |
| BDD100K | `/mnt/sda/edward/data_vcd/BDD100K/` |

Expected subdirectories under each root:
```
JAAD_clips/        source .mp4 files (346 clips)
frames/            extracted JPEG frames, one sub-dir per clip
fast_results/      *_person.txt and *_car.txt from ByteTrack + speed classifier
slow_results/      *.csv from DPT_analysis.py
fusion_results/    *.json from run_fusion()
```

## Performance Reference (GTX TITAN X, measured on JAAD video_0001)

| Module | Speed |
|--------|-------|
| YOLOX + ByteTrack | real-time (≥30 fps) |
| GroundingDINO + SAM | ~3.7 s/frame |
| DPT depth | ~5.2 s/frame |
| Total slow vision | ~9 s/frame @ 2 Hz |
| Fusion (CPU) | real-time |
| GOFAI behavior inference | <1 ms/frame |
| GPT-3.5 query (per clip) | ~5–15 s |
| **Full JAAD estimate** | ~34 h (slow vision dominates) |

## Tests

```bash
pixi run pytest   # 6 synthetic fusion unit tests
```

## Known Issues

**Fast-result frame gap** — ByteTrack only writes rows for frames where ≥1 person is tracked. When all tracks are lost near the end of a clip, trailing frames have no overlay. In `video_0001` tracks end at frame 580; frames 581–599 render raw video with no bboxes. This is expected.

**Playback speed** — Timer intervals are calibrated to 30 fps. Actual display rate is bounded by rendering time (~10–30 ms/frame). Sequential ticks reuse an open `VideoCapture` to avoid H.264 keyframe-seek overhead.

**Slow-data carry-forward** — When a track has no slow-vision match in the current window (fast-only frames), the behavior analyzer carries forward depth and surface data from the last matched slow frame in the full track history.
