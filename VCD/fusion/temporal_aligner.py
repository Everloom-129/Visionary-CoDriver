"""Temporal alignment of fast (30 Hz) and slow (2 Hz) detection streams.

Strategy
--------
Slow frames are captured every SLOW_PERIOD fast frames (default 15).
For a given fast frame f, the nearest slow frame is::

    nearest = round(f / SLOW_PERIOD) * SLOW_PERIOD

If no slow detection exists at that frame, we carry forward the last valid
slow frame as long as the gap does not exceed MAX_GAP frames.
"""

from __future__ import annotations

import bisect
import csv
import os
from typing import Dict, Generator, List, Optional, Tuple

from VCD.fusion.data_schema import FastDetection, SlowDetection


# YOLO class-id → object class string (COCO labels used by YOLOv8)
_YOLO_CLASS_MAP: Dict[str, str] = {
    "0": "person", "1": "bicycle", "2": "car", "3": "motorcycle",
    "5": "bus", "7": "truck",
}


# ---------------------------------------------------------------------------
# MOT .txt loader
# ---------------------------------------------------------------------------

def load_fast_detections(mot_txt_path: str) -> Dict[int, List[FastDetection]]:
    """Parse a MOT-format .txt file into a dict keyed by frame_id.

    Supports three column layouts (auto-detected):

    **Standard MOT** from bytetracker.py (xywh, 10 cols, cols 7-9 all -1):
        frame_id, track_id, bb_left, bb_top, bb_width, bb_height,
        confidence, -1, -1, -1  [, speed_class]

    **Pipeline output** from process_video_for_mot.py (xyxy, 10-11 cols, col[7]=class_id):
        frame_id, track_id, x1, y1, x2, y2, confidence, class_id, -1, -1
        [, speed_class]

    **Fusion spec** (xywh, ≤9 cols):
        frame_id, track_id, bb_left, bb_top, bb_width, bb_height,
        confidence, [speed_class], [object_class]

    Columns beyond confidence are optional and default to 0 / 'person'.
    """
    detections: Dict[int, List[FastDetection]] = {}
    if not os.path.isfile(mot_txt_path):
        raise FileNotFoundError(f"Fast detection file not found: {mot_txt_path}")

    with open(mot_txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            frame_id = int(parts[0])
            track_id = int(parts[1])

            # --- format detection ---
            # Standard MOT (bytetracker.py): cols 7-9 are world coords, all -1 → xywh
            # Pipeline (process_video_for_mot.py): col[7]=class_id (not -1), col[8]="-1" → xyxy
            # Fusion spec: ≤9 cols with optional speed_class/object_class → xywh
            _is_neg1 = ("-1", "-1.0")
            is_standard_mot = (
                len(parts) >= 10
                and parts[7] in _is_neg1
                and parts[8] in _is_neg1
            )
            is_pipeline_fmt = (
                not is_standard_mot
                and len(parts) >= 9
                and parts[8] in _is_neg1
                and parts[7] not in _is_neg1
            )

            if is_standard_mot:
                # Standard MOT (bytetracker): cols 2-5 are already xywh
                bbox = (float(parts[2]), float(parts[3]),
                        float(parts[4]), float(parts[5]))
                confidence = float(parts[6]) if len(parts) > 6 else 1.0
                speed_class = int(parts[10]) if len(parts) > 10 else 0
                object_class = "person"  # bytetracker person/car files are single-class
            elif is_pipeline_fmt:
                # xyxy → xywh
                x1, y1 = float(parts[2]), float(parts[3])
                x2, y2 = float(parts[4]), float(parts[5])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                confidence = float(parts[6]) if len(parts) > 6 else 1.0
                speed_class = int(parts[10]) if len(parts) > 10 else 0
                object_class = _YOLO_CLASS_MAP.get(parts[7], "person")
            else:
                # Fusion spec: xywh
                bbox = (float(parts[2]), float(parts[3]),
                        float(parts[4]), float(parts[5]))
                confidence = float(parts[6]) if len(parts) > 6 else 1.0
                speed_class = int(parts[7]) if len(parts) > 7 else 0
                object_class = str(parts[8]) if len(parts) > 8 else "person"

            det = FastDetection(
                frame_id=frame_id,
                track_id=track_id,
                bbox_xywh=bbox,
                confidence=confidence,
                speed_class=speed_class,
                object_class=object_class,
            )
            detections.setdefault(frame_id, []).append(det)

    return detections


# ---------------------------------------------------------------------------
# DPT CSV loader
# ---------------------------------------------------------------------------

def load_slow_detections(dpt_csv_path: str) -> Dict[int, List[SlowDetection]]:
    """Parse a DPT CSV file into a dict keyed by frame_id.

    Accepts both the fusion spec column names and the actual DPT pipeline
    output column names (case-insensitive lookup with aliases):

    Fusion spec:
        frame_id, person_id, x1, y1, x2, y2, confidence,
        avg_depth, angle, distance_level, region_category, surface_type

    DPT pipeline output (DPT_analysis.py legacy names):
        Frame_id, Person_id, x1, y1, x2, y2, Average_Depth,
        Angle, Distance_Level, X, Y, catagory
        (confidence and surface_type absent → default to 1.0 / 'unknown')
    """
    detections: Dict[int, List[SlowDetection]] = {}
    if not os.path.isfile(dpt_csv_path):
        raise FileNotFoundError(f"Slow detection file not found: {dpt_csv_path}")

    def _get(row_lower: dict, *keys, default="0"):
        """Return the first matching key value (case-insensitive), else default."""
        for k in keys:
            if k in row_lower:
                return row_lower[k]
        return default

    with open(dpt_csv_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Normalise to lowercase keys for alias lookup
            r = {k.lower(): v for k, v in row.items()}

            frame_id = int(_get(r, "frame_id"))
            object_id = int(_get(r, "person_id"))
            bbox_xyxy = (
                float(_get(r, "x1")),
                float(_get(r, "y1")),
                float(_get(r, "x2")),
                float(_get(r, "y2")),
            )
            confidence = float(_get(r, "confidence", default="1.0"))
            avg_depth = float(_get(r, "avg_depth", "average_depth"))
            angle = float(_get(r, "angle"))
            distance_level = int(_get(r, "distance_level", default="1"))
            # region_category: use spec field if present; 'catagory' is a
            # screen-grid index (0-5), semantically different — skip it.
            region_category = int(_get(r, "region_category", default="5"))
            surface_type = str(_get(r, "surface_type", default="unknown"))

            det = SlowDetection(
                frame_id=frame_id,
                object_id=object_id,
                bbox_xyxy=bbox_xyxy,
                confidence=confidence,
                avg_depth=avg_depth,
                angle=angle,
                distance_level=distance_level,
                region_category=region_category,
                surface_type=surface_type,
            )
            detections.setdefault(frame_id, []).append(det)

    return detections


# ---------------------------------------------------------------------------
# TemporalAligner
# ---------------------------------------------------------------------------

class TemporalAligner:
    """Aligns fast and slow detection dicts by nearest-slow-frame logic.

    Parameters
    ----------
    fast_dets : dict[frame_id → List[FastDetection]]
    slow_dets : dict[frame_id → List[SlowDetection]]
    slow_period : int
        Number of fast frames between consecutive slow captures (default 15,
        corresponding to 30 Hz fast / 2 Hz slow).
    max_gap : int
        Maximum allowed frame distance between a fast frame and its nearest
        slow frame.  If the gap exceeds this, slow data is treated as missing.
    """

    def __init__(
        self,
        fast_dets: Dict[int, List[FastDetection]],
        slow_dets: Dict[int, List[SlowDetection]],
        slow_period: int = 15,
        max_gap: int = 15,
    ) -> None:
        self.fast_dets = fast_dets
        self.slow_dets = slow_dets
        self.slow_period = slow_period
        self.max_gap = max_gap

        # Sorted list of slow frame ids for bisect searches
        self._slow_frame_ids: List[int] = sorted(slow_dets.keys())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_nearest_slow_frame(self, fast_fid: int) -> Optional[int]:
        """Return the slow frame_id closest to *fast_fid*, or None if gap > max_gap."""
        if not self._slow_frame_ids:
            return None

        # Binary search
        idx = bisect.bisect_left(self._slow_frame_ids, fast_fid)

        candidates: List[int] = []
        if idx < len(self._slow_frame_ids):
            candidates.append(self._slow_frame_ids[idx])
        if idx > 0:
            candidates.append(self._slow_frame_ids[idx - 1])

        nearest = min(candidates, key=lambda s: abs(s - fast_fid))
        if abs(nearest - fast_fid) > self.max_gap:
            return None
        return nearest

    def align_sequences(self) -> Dict[int, Optional[int]]:
        """Return mapping fast_frame_id → nearest_slow_frame_id (or None).

        Uses carry-forward: if no slow frame is within max_gap, the last
        valid slow frame is reused until a closer one is available.
        """
        mapping: Dict[int, Optional[int]] = {}
        last_valid_slow: Optional[int] = None

        for fast_fid in sorted(self.fast_dets.keys()):
            nearest = self.get_nearest_slow_frame(fast_fid)
            if nearest is not None:
                last_valid_slow = nearest
                mapping[fast_fid] = nearest
            else:
                # carry-forward
                mapping[fast_fid] = last_valid_slow

        return mapping

    def iter_aligned_pairs(
        self,
    ) -> Generator[
        Tuple[int, Optional[int], List[FastDetection], List[SlowDetection]],
        None,
        None,
    ]:
        """Yield (fast_fid, slow_fid, fast_dets, slow_dets) for every fast frame."""
        alignment = self.align_sequences()
        for fast_fid in sorted(self.fast_dets.keys()):
            slow_fid = alignment[fast_fid]
            fast_d = self.fast_dets.get(fast_fid, [])
            slow_d = self.slow_dets.get(slow_fid, []) if slow_fid is not None else []
            yield fast_fid, slow_fid, fast_d, slow_d

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        mot_txt_path: str,
        dpt_csv_path: str,
        slow_period: int = 15,
        max_gap: int = 15,
    ) -> "TemporalAligner":
        fast = load_fast_detections(mot_txt_path)
        slow = load_slow_detections(dpt_csv_path)
        return cls(fast, slow, slow_period=slow_period, max_gap=max_gap)
