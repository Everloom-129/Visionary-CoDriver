# VCD Agent Prompt Templates

## System Prompt

```
You are a co-driver assistant integrated into a dashcam-equipped vehicle.
Your input is a structured scene report produced by a vision pipeline (YOLOX tracking +
depth estimation + surface segmentation). Pre-computed GOFAI risk hints are included.

Your task — think step by step:
1. Summarise the overall road situation in 2-3 sentences.
2. For each tracked person, state the risk level and explain why in one sentence.
3. Recommend a driver action (e.g. maintain speed, reduce speed, prepare to brake).

Output format for the risk list (required, one line per person):
  Person track#<id>: <low|medium|high> — <reason>

End with:
  Recommended action: <action>
```

## User Message Template

```
Video: {video_name}

{scene_text}

Provide your co-driver analysis now.
```

## Scene Text Format (assembled by SceneAssembler)

```
Scene report: video_0001
Dashcam co-driver system. Track IDs are globally unique across frames.
Positions use six screen zones: upper/lower × left/center/right.
GOFAI risk scores (pre-computed): distance + region + speed + angle alignment.

Frame 0 (t=0.0s):
  Surfaces: road, sidewalk
  Persons (2):
    Person track#1: lower-right | medium (≤15m) | 48.9° to the right | stationary/slow | on road  [GOFAI:MEDIUM score=4]
    Person track#2: lower-center | medium (≤15m) | 9.1° to the left | stationary/slow | on road  [GOFAI:MEDIUM score=4]

Frame 30 (t=1.0s):
  Surfaces: road
  Persons (3):
    Person track#1: lower-right | close (≤10m) | 35.2° to the right | moving fast | on road  [GOFAI:HIGH score=8]
    ...

Overall risk summary (worst seen across sampled frames):
  Person track#1: high
  Person track#2: medium
```

## GOFAI Risk Scoring Rules

| Factor | Value | Points |
|--------|-------|--------|
| distance_level | 1 (≤5m) | 4 |
| distance_level | 2 (≤10m) | 3 |
| distance_level | 3 (≤15m) | 1 |
| distance_level | 4 (>15m) | 0 |
| region_category | 0 road | 3 |
| region_category | 2 crosswalk | 3 |
| region_category | 1 sidewalk | 1 |
| region_category | 3/4/5 other | 0 |
| speed_class | 1 fast | 2 |
| speed_class | 0 slow | 0 |
| angle | \|angle\| < 15° | 1 |

**Risk level thresholds:** score ≥ 6 → HIGH · ≥ 3 → MEDIUM · < 3 → LOW

Maximum possible score: 10 (very close + road + fast + directly ahead)
