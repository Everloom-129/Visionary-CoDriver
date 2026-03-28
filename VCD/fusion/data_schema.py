"""Data schemas for the slow-fast vision fusion module.

FastDetection  — from 30 Hz YOLOX + ByteTrack MOT output
SlowDetection  — from 2 Hz GSAM/DINOX + DPT output
FusedDetection — merged record used by downstream LLM agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class FastDetection:
    """Single detection from the fast-vision stream (30 Hz).

    MOT .txt format columns (1-indexed, space-separated):
        frame_id  track_id  bb_left  bb_top  bb_width  bb_height  conf  x  y  z
    The 'x y z' columns are unused (set to -1 in standard MOT).
    speed_class  0 = slow / stationary, 1 = fast / moving.
    object_class 'person' | 'car' | ...
    """

    frame_id: int
    track_id: int
    # (x, y, w, h) — top-left origin, pixel units
    bbox_xywh: Tuple[float, float, float, float]
    confidence: float
    speed_class: int = 0            # 0 = slow, 1 = fast
    object_class: str = "person"

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        x, y, w, h = self.bbox_xywh
        return (x, y, x + w, y + h)


@dataclass
class SlowDetection:
    """Single detection from the slow-vision stream (2 Hz).

    DPT CSV columns (header row expected):
        frame_id, person_id, x1, y1, x2, y2, confidence,
        avg_depth, angle, distance_level, region_category, surface_type
    """

    frame_id: int
    object_id: int
    # (x1, y1, x2, y2) — xyxy format
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    avg_depth: float                    # metres
    angle: float                        # degrees from camera centre
    distance_level: int = 1             # 1=very close … 4=far
    region_category: int = 0            # 0=road … 5=unknown
    surface_type: str = "unknown"

    @property
    def bbox_xywh(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1, y1, x2 - x1, y2 - y1)


@dataclass
class FusedDetection:
    """Union record produced by DetectorFusion for a single matched object.

    Fields from fast stream: frame_id, track_id, bbox_xywh, confidence,
                             speed_class, object_class
    Fields from slow stream: slow_frame_id, avg_depth, angle,
                             distance_level, region_category, surface_type
    Either side may be None if the detection was unmatched.
    """

    # --- fast-stream fields ---
    frame_id: int
    track_id: Optional[int] = None
    bbox_xywh: Optional[Tuple[float, float, float, float]] = None
    fast_confidence: Optional[float] = None
    speed_class: Optional[int] = None
    object_class: str = "person"

    # --- slow-stream fields ---
    slow_frame_id: Optional[int] = None
    avg_depth: Optional[float] = None
    angle: Optional[float] = None
    distance_level: Optional[int] = None
    region_category: Optional[int] = None
    surface_type: Optional[str] = None
    slow_confidence: Optional[float] = None

    # --- IoU of the matched pair (None if unmatched) ---
    match_iou: Optional[float] = None

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON output."""
        return {
            "frame_id": self.frame_id,
            "track_id": self.track_id,
            "bbox_xywh": list(self.bbox_xywh) if self.bbox_xywh else None,
            "fast_confidence": self.fast_confidence,
            "speed_class": self.speed_class,
            "object_class": self.object_class,
            "slow_frame_id": self.slow_frame_id,
            "avg_depth": self.avg_depth,
            "angle": self.angle,
            "distance_level": self.distance_level,
            "region_category": self.region_category,
            "surface_type": self.surface_type,
            "slow_confidence": self.slow_confidence,
            "match_iou": self.match_iou,
        }
