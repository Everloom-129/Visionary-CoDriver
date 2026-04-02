"""GOFAI pedestrian behavior analyzer.

Design rationale:
  Distance alone is a poor risk signal. A pedestrian 15m away who is crossing
  the road is far more dangerous than one 5m away standing on a sidewalk.
  What the LLM needs is *what the pedestrian is doing* — their inferred
  intent and trajectory — not a collapsed scalar score.

This module analyzes each pedestrian's recent trajectory (sliding window of
the last ~1 second / 30 frames) and produces a natural-language behavioral
description suitable for LLM reasoning.

Behavior inference pipeline per track:
  1. Compute lateral velocity  (Δcx over window)   → crossing intent
  2. Compute approach velocity (Δheight over window) → closing distance
  3. Read region/surface context                    → road vs sidewalk
  4. Classify motion type                           → one of five categories
  5. Assemble a situation sentence                  → feeds directly into LLM

Motion categories:
  "crossing"   — significant lateral motion across the road
  "approaching"— bbox growing, heading toward the vehicle
  "receding"   — bbox shrinking, moving away
  "parallel"   — moving laterally without approaching (walking along road)
  "stationary" — minimal motion

Entry points:
  describe_track(records, window)             — single track
  analyze_scene(fused_by_frame, frame, window) — all tracks at a given frame
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_WIDTH  = 1920
IMAGE_HEIGHT = 1080

# Velocity thresholds measured per WINDOW_FRAMES frames
WINDOW_FRAMES = 30   # ~1 second at 30 fps

# Minimum lateral displacement (px) over the window to call it "crossing"
_CROSS_LAT_PX    = 80
# Minimum bbox height growth (px) to call it "approaching"
_APPROACH_DH_PX  = 15
# Minimum lateral displacement to call it "parallel walking" (weaker than crossing)
_PARALLEL_LAT_PX = 40

# Region category meanings (mirrors data_schema)
_ROAD_REGIONS     = {0, 2}   # road, crosswalk
_NONROAD_REGIONS  = {1, 3, 4}  # sidewalk, parking, grass

_DIST_LABEL = {
    1: "very close (≤5m)",
    2: "close (≤10m)",
    3: "medium distance (≤15m)",
    4: "far (>15m)",
}

_REGION_LABEL = {
    0: "road", 1: "sidewalk", 2: "crosswalk",
    3: "parking area", 4: "grass", 5: "unknown surface",
}


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class BehaviorDescriptor:
    """Natural-language behavioral description for one tracked pedestrian."""
    track_id: int
    motion_type: str                    # crossing | approaching | receding | parallel | stationary
    crossing_direction: Optional[str]   # left_to_right | right_to_left | None
    is_on_road: bool
    proximity_label: str                # very close | close | medium distance | far
    depth_m: Optional[float]
    lateral_side: str                   # left | center | right  (of ego vehicle view)
    situation: str                      # assembled natural-language sentence for LLM
    # Raw trajectory stats for debugging / table display
    delta_cx: float   # px/window  (lateral motion)
    delta_h:  float   # px/window  (height change, + = approaching)


# ---------------------------------------------------------------------------
# Core: single-track behavior inference
# ---------------------------------------------------------------------------

def describe_track(records: List[dict], window: int = WINDOW_FRAMES) -> BehaviorDescriptor:
    """Analyze the recent history of one track and return a behavioral description.

    Args:
        records: Chronologically ordered list of fused detection dicts for this track.
                 Should cover the last `window` frames or fewer.
        window:  Number of recent frames to consider (default 30 = 1 s).
    """
    # Use only the most recent `window` records
    recent = records[-window:]
    latest = recent[-1]

    tid = latest.get("track_id") or -1

    # ── Trajectory geometry ────────────────────────────────────────────────
    # Extract bbox centres and heights only where bbox_xywh is available
    bboxes = [(r["bbox_xywh"], r["frame_id"]) for r in recent if r.get("bbox_xywh")]

    if len(bboxes) >= 2:
        (x0, y0, w0, h0), _ = bboxes[0]
        (x1, y1, w1, h1), _ = bboxes[-1]
        cx0, cx1 = x0 + w0 / 2, x1 + w1 / 2
        delta_cx = cx1 - cx0    # positive = moved right on screen
        delta_h  = h1 - h0      # positive = bbox grew = approaching
    else:
        delta_cx = delta_h = 0.0

    # ── Spatial context: carry forward last known slow-data values ────────────
    # Fast-only frames have no depth/region. Search full records list (not just
    # recent) so a slow-vision record from 2-3 seconds ago still contributes.
    def _last_with(key):
        for r in reversed(records):
            v = r.get(key)
            if v is not None:
                return v
        return None

    region    = _last_with("region_category")
    surface   = _last_with("surface_type") or ""
    dist_lvl  = _last_with("distance_level") or 4
    depth_m   = _last_with("avg_depth")
    angle     = _last_with("angle")
    speed_cls = latest.get("speed_class") or 0

    is_on_road = region in _ROAD_REGIONS

    proximity_label = _DIST_LABEL.get(dist_lvl, "unknown distance")

    # Lateral screen position (left / center / right third of image)
    if bboxes:
        (xb, yb, wb, hb), _ = bboxes[-1]
        cx_latest = xb + wb / 2
    else:
        cx_latest = IMAGE_WIDTH / 2
    lateral_side = (
        "left" if cx_latest < IMAGE_WIDTH / 3
        else ("right" if cx_latest > 2 * IMAGE_WIDTH / 3 else "center")
    )

    # ── Motion classification ──────────────────────────────────────────────
    abs_lat  = abs(delta_cx)
    is_cross = abs_lat >= _CROSS_LAT_PX
    # Crossing is the dominant signal when lateral > height-based signal
    is_approach  = delta_h  >= _APPROACH_DH_PX
    is_recede    = delta_h  <= -_APPROACH_DH_PX
    is_parallel  = abs_lat  >= _PARALLEL_LAT_PX

    if is_cross and (not is_approach or abs_lat > delta_h * 2):
        motion_type = "crossing"
        crossing_direction = "left_to_right" if delta_cx > 0 else "right_to_left"
    elif is_approach:
        motion_type = "approaching"
        crossing_direction = None
    elif is_recede:
        motion_type = "receding"
        crossing_direction = None
    elif is_parallel:
        motion_type = "parallel"
        crossing_direction = None
    else:
        motion_type = "stationary"
        crossing_direction = None

    # ── Angle description ─────────────────────────────────────────────────
    if angle is not None:
        if abs(angle) < 8:
            angle_phrase = "directly in the vehicle's forward path"
        elif abs(angle) < 25:
            side = "right" if angle > 0 else "left"
            angle_phrase = f"slightly to the {side} of the vehicle's path"
        else:
            side = "right" if angle > 0 else "left"
            angle_phrase = f"to the far {side} of the vehicle"
    else:
        angle_phrase = ""

    # ── Location string ───────────────────────────────────────────────────
    if surface and surface not in ("unknown", ""):
        loc = f"on {surface}"
    elif region is not None:
        loc = f"on {_REGION_LABEL.get(region, 'unknown surface')}"
    else:
        loc = "location unknown"

    depth_str = f"{depth_m:.1f}m away" if depth_m is not None else proximity_label

    speed_str = "moving at speed" if speed_cls == 1 else "moving slowly"

    # ── Assemble situation sentence ────────────────────────────────────────
    # Helper: join non-empty phrase parts cleanly
    def _join(*parts):
        return ", ".join(p for p in parts if p)

    if motion_type == "crossing":
        dir_str   = "from left to right" if crossing_direction == "left_to_right" else "from right to left"
        road_warn = "directly into the vehicle's trajectory" if is_on_road else ""
        speed_str_c = "moving quickly" if speed_cls == 1 else "moving slowly"
        situation = (
            f"Crossing the road {dir_str} ({depth_str}), {loc}. "
            + _join(road_warn, speed_str_c, angle_phrase) + "."
        )

    elif motion_type == "approaching":
        situation = (
            f"Moving toward the vehicle and closing distance ({depth_str}), "
            f"{loc}. "
            + _join(angle_phrase, f"bbox height growing by {delta_h:.0f}px") + "."
        )

    elif motion_type == "receding":
        situation = (
            f"Moving away from the vehicle ({depth_str}), {loc}. "
            + (_join(angle_phrase) + ". " if angle_phrase else "")
            + "Pedestrian is moving to a safer distance."
        )

    elif motion_type == "parallel":
        par_dir = "rightward" if delta_cx > 0 else "leftward"
        on_road_note = (
            f"Not heading toward the vehicle but remains on {_REGION_LABEL.get(region, 'the road')}."
            if is_on_road
            else "Low immediate risk but monitor if they step onto the road."
        )
        situation = (
            f"Walking {par_dir} parallel to traffic ({depth_str}), {loc}. "
            + (_join(angle_phrase) + ". " if angle_phrase else "")
            + on_road_note
        )

    else:  # stationary
        on_road_warn = " Standing on the road is a potential hazard." if is_on_road else ""
        situation = (
            f"Stationary ({depth_str}), {loc}, {lateral_side} side of view.{on_road_warn}"
        )

    return BehaviorDescriptor(
        track_id=tid,
        motion_type=motion_type,
        crossing_direction=crossing_direction,
        is_on_road=is_on_road,
        proximity_label=proximity_label,
        depth_m=depth_m,
        lateral_side=lateral_side,
        situation=situation,
        delta_cx=delta_cx,
        delta_h=delta_h,
    )


# ---------------------------------------------------------------------------
# Scene-level analysis: all tracks at a given frame
# ---------------------------------------------------------------------------

def analyze_scene(
    fused_by_frame: Dict[int, List[dict]],
    current_frame: int,
    window: int = WINDOW_FRAMES,
) -> Dict[int, BehaviorDescriptor]:
    """Analyze all tracked pedestrians visible at or before current_frame.

    Two-pass history collection:
      - Slow data context: look back up to 3× window to guarantee we find
        at least one slow-vision frame (which runs at 2 Hz, every 15 fast frames).
      - Trajectory velocity: only the most recent `window` frames are used
        for motion classification (handled inside describe_track).

    Args:
        fused_by_frame: {frame_id: [detection_dicts]} — the full fusion result.
        current_frame:  Frame to analyze (inclusive upper bound).
        window:         Trajectory window in frames (default 30 = ~1 s).

    Returns:
        {track_id: BehaviorDescriptor} for every track present at current_frame.
    """
    # Extended look-back ensures we capture at least one slow-vision record
    ctx_start = max(0, current_frame - window * 3)
    histories: Dict[int, List[dict]] = defaultdict(list)

    for fid in range(ctx_start, current_frame + 1):
        for rec in fused_by_frame.get(fid, []):
            tid = rec.get("track_id")
            if tid is not None:
                histories[tid].append(rec)

    # Only return descriptors for tracks present at or near current_frame
    active_tracks = {
        rec.get("track_id")
        for rec in fused_by_frame.get(current_frame, [])
        if rec.get("track_id") is not None
    }

    return {
        tid: describe_track(recs, window)
        for tid, recs in histories.items()
        if tid in active_tracks
    }


# ---------------------------------------------------------------------------
# Convenience: worst motion type across the scene
# ---------------------------------------------------------------------------

_MOTION_SEVERITY = {
    "crossing":    4,
    "approaching": 3,
    "parallel":    2,
    "stationary":  1,
    "receding":    0,
}


def most_critical(descriptors: Dict[int, BehaviorDescriptor]) -> Optional[BehaviorDescriptor]:
    """Return the BehaviorDescriptor with the highest severity motion type."""
    if not descriptors:
        return None
    return max(descriptors.values(), key=lambda d: _MOTION_SEVERITY.get(d.motion_type, 0))
