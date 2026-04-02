# VCD Agent Pipeline

## Design Rationale

Early versions of the system used a single distance-based risk score to label each pedestrian. This was found to be insufficient: a pedestrian 8 m away standing on the sidewalk poses very different risk from one 8 m away actively crossing the road into the vehicle's path.

The current design separates the problem into two stages:

1. **GOFAI behavior inference** — deterministic, trajectory-based classification of *what the pedestrian is doing*, performed over a 1-second sliding window of fused detection records.
2. **LLM reasoning** — the LLM receives a structured natural-language scene description and reasons about risk from the described behaviors, not from raw sensor numbers.

This mirrors how an experienced co-driver thinks: they describe the scene in human terms ("that person is crossing"), then apply judgment to recommend an action.

---

## Architecture

```
Fusion JSON (per clip)
       │
       ▼
load_fusion_json()
  → {frame_id: [raw detection dicts]}
       │
       ▼  (for each sampled keyframe)
analyze_scene(fused_by_frame, frame_id)          risk_analyzer.py
  │
  ├─ Build per-track histories (3× window look-back)
  ├─ For each active track: describe_track(records, window=30)
  │     ├─ Trajectory geometry: Δcx (lateral), Δheight (approach)
  │     ├─ Spatial context: depth, region, surface (carried forward from slow frames)
  │     ├─ Motion classification: crossing / approaching / receding / parallel / stationary
  │     └─ Situation sentence: natural-language description
  └─ Returns {track_id: BehaviorDescriptor}
       │
       ▼
SceneAssembler.video_to_text()                   llm.py
  → structured scene report (plain text)
       │
       ▼
LLMClient.query(scene_text)
  → co-driver narrative + per-person risk + recommended action
```

---

## Behavior Inference (`VCD/agent/risk_analyzer.py`)

### Sliding window

For each call to `analyze_scene(fused_by_frame, frame_id, window=30)`:
- History is collected from the last `3 × window` frames (90 frames = 3 s) to guarantee at least one slow-vision record is included per track even when the current window has only fast-only detections.
- Only tracks with a detection at `frame_id` are returned.
- Trajectory velocity is computed over the most recent `window` frames only.

### Trajectory signals

| Signal | Computation | Interpretation |
|--------|-------------|----------------|
| `delta_cx` | cx_last − cx_first over window | lateral screen motion (px) |
| `delta_h`  | h_last − h_first over window | bbox height change; + = approaching |

### Motion classification

| Class | Condition | Priority |
|-------|-----------|----------|
| `crossing`    | \|Δcx\| ≥ 80 px **and** lateral dominates over height | highest |
| `approaching` | Δheight ≥ 15 px | |
| `receding`    | Δheight ≤ −15 px | |
| `parallel`    | \|Δcx\| ≥ 40 px (weaker lateral without strong approach) | |
| `stationary`  | all others | lowest |

Crossing takes priority over approaching when `|Δcx| > 2 × Δheight` (pure lateral crossing without strong height growth).

### Spatial context carry-forward

Slow-vision data (depth, region category, surface type) arrives at 2 Hz (every 15 fast frames). Fast-only frames have no slow match. When the latest record is fast-only, the analyzer searches backward through the full track history to find the most recent slow-matched record and carries forward its depth and surface values. This prevents "unknown distance / unknown surface" labels on mid-trajectory frames.

### Situation sentence examples

```
# Crossing pedestrian
"Crossing the road from right to left (9.7m away), on road.
 directly into the vehicle's trajectory, moving slowly, to the far right of the vehicle."

# Approaching pedestrian
"Moving toward the vehicle and closing distance (8.3m away), on road.
 slightly to the right of the vehicle's path, bbox height growing by 21px."

# Stationary on road
"Stationary (10.1m away), on road, left side of view. Standing on the road is a potential hazard."

# Parallel walker
"Walking leftward parallel to traffic (12.9m away), on road.
 directly in the vehicle's forward path. Not heading toward the vehicle but remains on road."
```

---

## Scene Assembler (`VCD/agent/llm.py` — `SceneAssembler`)

`SceneAssembler.video_to_text(video_name, fused_by_frame, stride=30)` samples one keyframe every `stride` frames and builds a full scene report:

```
Scene report: video_0002
Dashcam co-driver system. Track IDs are globally unique across frames.
Behavior is inferred from trajectory analysis over a ~1-second sliding window: ...

Frame 30 (t=1.0s):
  Surfaces: road
  Persons (1, 0 untracked):
    Person track#58: Moving toward the vehicle and closing distance (9.3m away), on road.
                     to the far right of the vehicle, bbox height growing by 21px.
  !! Alert: Person track#58 is APPROACHING — requires immediate attention.

Frame 60 (t=2.0s):
  Surfaces: road
  Persons (2, 0 untracked):
    Person track#58: Crossing the road from right to left (9.7m away), on road.
                     directly into the vehicle's trajectory, moving slowly, to the far right.
    Person track#59: Moving toward the vehicle and closing distance (8.2m away), on road. ...
  !! Alert: Person track#58 is CROSSING — requires immediate attention.

...

Overall behavior summary across sampled frames:
  Person track#58: crossing
  Person track#59: crossing
```

**`stride` guidance:**
- `stride=30` (default): 1 keyframe per second — good balance for GPT-3.5 context budget
- `stride=15`: every slow-vision keyframe — more detail, ~2× longer prompt

---

## LLM Prompt (`VCD/agent/prompt.md`)

### System message

The system prompt instructs the model to interpret the five behavior categories with risk guidelines:
- `crossing` on road → HIGH
- `approaching` toward vehicle → MEDIUM–HIGH
- `parallel` on road → MEDIUM
- `stationary` on road → MEDIUM
- `stationary` on sidewalk → LOW
- `receding` → LOW

### Required output format

```
Person track#<id>: <low|medium|high> — <one sentence reason based on behavior>
...
Recommended action: <specific driver action>
```

### Example LLM output (video_0002)

```
The scene shows an active road crossing event. Two pedestrians (tracks #58 and #59) 
are crossing the road from right to left at close range (~9–10m). The vehicle is 
approaching them and they are not yet clear of the lane.

Person track#58: high — actively crossing the road at close range, directly in vehicle path
Person track#59: high — also crossing road, approaching vehicle from the right side

Recommended action: Reduce speed immediately and prepare to brake; yield to both 
pedestrians until the crossing is complete.
```

---

## Running the Agent

### CLI

```bash
# Preview assembled scene text (no API call)
python -m VCD.agent <fusion_json> --scene-only [--stride 30]

# Full run with GPT-3.5
export OPENAI_API_KEY="sk-..."
python -m VCD.agent <fusion_json> [--stride 30] [--print-scene]
```

### Python API

```python
from VCD.agent import run_agent, assemble_scene
from VCD.agent.risk_analyzer import analyze_scene, BehaviorDescriptor

# Scene text only
text = assemble_scene("fusion_results/video_0001.json", stride=30)

# Full agent
response = run_agent("fusion_results/video_0001.json", model="gpt-3.5-turbo")

# Low-level: per-frame behavior descriptors
from VCD.agent.llm import load_fusion_json
fused = load_fusion_json("fusion_results/video_0001.json")
behaviors = analyze_scene(fused, frame_id=60)  # {track_id: BehaviorDescriptor}
for tid, b in behaviors.items():
    print(f"Track #{tid}: {b.motion_type} — {b.situation}")
```

### Dashboard

The **Agent Reasoning** accordion at the bottom of the dashboard provides:
- API key input (password field)
- Stride slider (15–90 frames)
- **Run Agent** button → queries GPT-3.5 and shows the co-driver narrative
- **Show assembled scene text** checkbox → reveals the exact text sent to the LLM

Behavior badges are always shown on fused bounding boxes (no API key needed), using the same color scheme as the motion classification.

---

## Dashboard Color Scheme (Fused Bounding Boxes)

| Behavior | Color | Note |
|----------|-------|------|
| crossing    | Red     | Highest danger |
| approaching | Orange  | Closing distance |
| parallel    | Yellow  | Moving along road |
| stationary (on road) | Amber | Obstruction risk |
| stationary (off road) | Green | Low risk |
| receding   | Blue-grey | Moving away |

Box border is 3 px thick for `crossing` and `approaching`; 2 px otherwise.

---

## Key Source Files

| File | Role |
|------|------|
| `VCD/agent/risk_analyzer.py` | `BehaviorDescriptor`, `describe_track()`, `analyze_scene()` |
| `VCD/agent/llm.py` | `SceneAssembler`, `LLMClient`, `run_agent()`, `assemble_scene()` |
| `VCD/agent/__init__.py` | Public exports: `run_agent`, `assemble_scene`, `load_fusion_json` |
| `VCD/agent/__main__.py` | CLI entry point (`python -m VCD.agent`) |
| `VCD/agent/prompt.md` | Prompt templates and behavior category reference |
| `VCD/dashboard/app.py` | Dashboard; calls `analyze_scene()` for per-frame badges |
