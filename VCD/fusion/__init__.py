"""VCD Fusion module — slow-fast vision stream alignment and detection fusion.

Quick-start
-----------
>>> from VCD.fusion import run_fusion
>>> fused = run_fusion("path/to/fast.txt", "path/to/slow.csv", "output.json")

For programmatic use import the individual components::

    from VCD.fusion.data_schema import FastDetection, SlowDetection, FusedDetection
    from VCD.fusion.temporal_aligner import TemporalAligner, load_fast_detections, load_slow_detections
    from VCD.fusion.detector_fusion import DetectorFusion
    from VCD.fusion.dataset_loader import JAAdLoader, BDD100KLoader, FusionResultLoader
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from VCD.fusion.data_schema import FastDetection, FusedDetection, SlowDetection
from VCD.fusion.dataset_loader import BDD100KLoader, FusionResultLoader, JAAdLoader
from VCD.fusion.detector_fusion import DetectorFusion
from VCD.fusion.temporal_aligner import (
    TemporalAligner,
    load_fast_detections,
    load_slow_detections,
)

__all__ = [
    "run_fusion",
    "FastDetection",
    "SlowDetection",
    "FusedDetection",
    "TemporalAligner",
    "DetectorFusion",
    "JAAdLoader",
    "BDD100KLoader",
    "FusionResultLoader",
    "load_fast_detections",
    "load_slow_detections",
]


def run_fusion(
    fast_txt: str,
    slow_csv: str,
    output_json: Optional[str] = None,
    iou_threshold: float = 0.3,
    slow_period: int = 15,
    max_gap: int = 15,
) -> List[FusedDetection]:
    """Convenience function: align + fuse fast and slow detections.

    Parameters
    ----------
    fast_txt      : Path to MOT-format .txt file (fast-vision output).
    slow_csv      : Path to DPT CSV file (slow-vision output).
    output_json   : If given, write fused results as JSON to this path.
    iou_threshold : IoU threshold for fast↔slow bbox matching.
    slow_period   : Fast frames between consecutive slow frames (default 15).
    max_gap       : Max frame distance to accept a slow frame match.

    Returns
    -------
    List of FusedDetection objects (all frames concatenated).
    """
    aligner = TemporalAligner.from_files(
        fast_txt, slow_csv, slow_period=slow_period, max_gap=max_gap
    )
    fusioner = DetectorFusion(iou_threshold=iou_threshold)

    all_fused: List[FusedDetection] = []
    for fast_fid, slow_fid, fast_dets, slow_dets in aligner.iter_aligned_pairs():
        fused = fusioner.fuse_frame(fast_fid, fast_dets, slow_dets, slow_fid)
        all_fused.extend(fused)

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as fh:
            json.dump([d.to_dict() for d in all_fused], fh, indent=2)

    return all_fused
