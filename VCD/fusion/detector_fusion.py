"""Detector fusion: IoU-based matching of fast and slow detections.

For each aligned frame pair the fusion proceeds as follows:

1. Build IoU matrix (N_fast × M_slow).
2. Greedy match: assign each fast detection to its highest-IoU slow detection,
   provided IoU > iou_threshold, and each slow detection is used at most once.
   (scipy.optimize.linear_sum_assignment is used when available; falls back to
   a pure-Python greedy approach.)
3. Produce FusedDetection for matched pairs, fast-only, and slow-only detections.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from VCD.fusion.data_schema import FastDetection, FusedDetection, SlowDetection


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _xyxy_iou(
    boxA: Tuple[float, float, float, float],
    boxB: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union of two xyxy boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0

    area_a = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    area_b = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def build_iou_matrix(
    fast_dets: List[FastDetection],
    slow_dets: List[SlowDetection],
) -> List[List[float]]:
    """Return N×M IoU matrix (list-of-lists)."""
    matrix: List[List[float]] = []
    for fd in fast_dets:
        row: List[float] = []
        for sd in slow_dets:
            row.append(_xyxy_iou(fd.bbox_xyxy, sd.bbox_xyxy))
        matrix.append(row)
    return matrix


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _greedy_match(
    iou_matrix: List[List[float]],
    iou_threshold: float,
) -> List[Tuple[int, int]]:
    """Greedy matching: assign each fast det to its best available slow det."""
    n = len(iou_matrix)
    m = len(iou_matrix[0]) if n > 0 else 0

    # Collect all (iou, i, j) pairs above threshold, sorted descending
    pairs = sorted(
        [
            (iou_matrix[i][j], i, j)
            for i in range(n)
            for j in range(m)
            if iou_matrix[i][j] >= iou_threshold
        ],
        key=lambda t: t[0],
        reverse=True,
    )

    matched_fast: set = set()
    matched_slow: set = set()
    matches: List[Tuple[int, int]] = []

    for iou, i, j in pairs:
        if i not in matched_fast and j not in matched_slow:
            matches.append((i, j))
            matched_fast.add(i)
            matched_slow.add(j)

    return matches


def _hungarian_match(
    iou_matrix: List[List[float]],
    iou_threshold: float,
) -> List[Tuple[int, int]]:
    """Optimal matching via scipy's Hungarian algorithm (preferred)."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _greedy_match(iou_matrix, iou_threshold)

    if not iou_matrix or not iou_matrix[0]:
        return []

    cost = np.array(iou_matrix)
    row_ind, col_ind = linear_sum_assignment(-cost)  # maximise IoU

    matches: List[Tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= iou_threshold:
            matches.append((int(r), int(c)))
    return matches


# ---------------------------------------------------------------------------
# DetectorFusion
# ---------------------------------------------------------------------------

class DetectorFusion:
    """Fuse fast and slow detections for a single aligned frame pair.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to consider a fast↔slow pair as a match (default 0.3).
    use_hungarian : bool
        Use optimal Hungarian matching when scipy is available (default True).
        Falls back to greedy if scipy is not installed.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        use_hungarian: bool = True,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.use_hungarian = use_hungarian

    def fuse_frame(
        self,
        frame_id: int,
        fast_dets: List[FastDetection],
        slow_dets: List[SlowDetection],
        slow_frame_id: Optional[int] = None,
    ) -> List[FusedDetection]:
        """Return fused detections for one frame pair.

        Parameters
        ----------
        frame_id     : fast-stream frame id
        fast_dets    : detections from fast stream at *frame_id*
        slow_dets    : detections from slow stream at *slow_frame_id*
        slow_frame_id: the slow frame that was aligned to *frame_id*
        """
        results: List[FusedDetection] = []

        # --- trivial cases ---
        if not fast_dets and not slow_dets:
            return results

        if not fast_dets:
            # slow-only
            for sd in slow_dets:
                results.append(self._slow_only(frame_id, sd, slow_frame_id))
            return results

        if not slow_dets:
            # fast-only
            for fd in fast_dets:
                results.append(self._fast_only(frame_id, fd, slow_frame_id))
            return results

        # --- compute IoU and match ---
        iou_matrix = build_iou_matrix(fast_dets, slow_dets)
        if self.use_hungarian:
            matches = _hungarian_match(iou_matrix, self.iou_threshold)
        else:
            matches = _greedy_match(iou_matrix, self.iou_threshold)

        matched_fast_idx = {i for i, _ in matches}
        matched_slow_idx = {j for _, j in matches}

        # matched pairs
        for i, j in matches:
            fd = fast_dets[i]
            sd = slow_dets[j]
            iou_val = iou_matrix[i][j]
            results.append(
                FusedDetection(
                    frame_id=frame_id,
                    track_id=fd.track_id,
                    bbox_xywh=fd.bbox_xywh,
                    fast_confidence=fd.confidence,
                    speed_class=fd.speed_class,
                    object_class=fd.object_class,
                    slow_frame_id=slow_frame_id,
                    avg_depth=sd.avg_depth,
                    angle=sd.angle,
                    distance_level=sd.distance_level,
                    region_category=sd.region_category,
                    surface_type=sd.surface_type,
                    slow_confidence=sd.confidence,
                    match_iou=iou_val,
                )
            )

        # unmatched fast-only
        for i, fd in enumerate(fast_dets):
            if i not in matched_fast_idx:
                results.append(self._fast_only(frame_id, fd, slow_frame_id))

        # unmatched slow-only
        for j, sd in enumerate(slow_dets):
            if j not in matched_slow_idx:
                results.append(self._slow_only(frame_id, sd, slow_frame_id))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fast_only(
        frame_id: int,
        fd: FastDetection,
        slow_frame_id: Optional[int],
    ) -> FusedDetection:
        return FusedDetection(
            frame_id=frame_id,
            track_id=fd.track_id,
            bbox_xywh=fd.bbox_xywh,
            fast_confidence=fd.confidence,
            speed_class=fd.speed_class,
            object_class=fd.object_class,
            slow_frame_id=slow_frame_id,
        )

    @staticmethod
    def _slow_only(
        frame_id: int,
        sd: SlowDetection,
        slow_frame_id: Optional[int],
    ) -> FusedDetection:
        x1, y1, x2, y2 = sd.bbox_xyxy
        return FusedDetection(
            frame_id=frame_id,
            bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
            slow_frame_id=slow_frame_id,
            avg_depth=sd.avg_depth,
            angle=sd.angle,
            distance_level=sd.distance_level,
            region_category=sd.region_category,
            surface_type=sd.surface_type,
            slow_confidence=sd.confidence,
        )
