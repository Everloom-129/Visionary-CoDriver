"""Synthetic test for the slow-fast fusion pipeline.

Run with:
    python VCD/fusion/test_fusion.py
    python VCD/fusion/test_fusion.py --jaad /mnt/sda/edward/data_vcd/JAAD

No real dataset is required for the default synthetic tests.
"""

from __future__ import annotations

import argparse
import sys
import os

# Ensure project root is on the path when run directly
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from VCD.fusion.data_schema import FastDetection, FusedDetection, SlowDetection
from VCD.fusion.detector_fusion import DetectorFusion, build_iou_matrix
from VCD.fusion.temporal_aligner import TemporalAligner


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_fast_dets(n_frames: int = 30, n_persons: int = 3):
    """Simulate 30 fast frames with up to n_persons tracked persons per frame."""
    dets = {}
    for fid in range(1, n_frames + 1):
        frame_dets = []
        for tid in range(1, n_persons + 1):
            # Simulate slight motion
            x = 100 + tid * 80 + fid * 2
            y = 200
            w, h = 60, 120
            frame_dets.append(
                FastDetection(
                    frame_id=fid,
                    track_id=tid,
                    bbox_xywh=(float(x), float(y), float(w), float(h)),
                    confidence=0.9,
                    speed_class=1 if tid % 2 == 0 else 0,
                    object_class="person",
                )
            )
        dets[fid] = frame_dets
    return dets


def make_slow_dets(n_frames: int = 30, n_persons: int = 3, slow_period: int = 15):
    """Simulate slow frames at multiples of slow_period (15, 30, ...)."""
    dets = {}
    for fid in range(slow_period, n_frames + 1, slow_period):
        frame_dets = []
        for pid in range(1, n_persons + 1):
            x1 = 100 + pid * 80 + fid * 2
            y1 = 200
            x2 = x1 + 60
            y2 = y1 + 120
            frame_dets.append(
                SlowDetection(
                    frame_id=fid,
                    object_id=pid,
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=0.85,
                    avg_depth=5.0 + pid * 1.5,
                    angle=float(pid * 5 - 7),
                    distance_level=2,
                    region_category=1,
                    surface_type="sidewalk",
                )
            )
        dets[fid] = frame_dets
    return dets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_iou():
    print("Test 1: IoU computation ... ", end="")
    boxA = (0.0, 0.0, 100.0, 100.0)
    boxB = (50.0, 50.0, 150.0, 150.0)

    from VCD.fusion.detector_fusion import _xyxy_iou
    iou = _xyxy_iou(boxA, boxB)
    expected = 2500.0 / (10000 + 10000 - 2500)
    assert abs(iou - expected) < 1e-6, f"IoU mismatch: {iou} vs {expected}"

    # Non-overlapping
    boxC = (200.0, 200.0, 300.0, 300.0)
    assert _xyxy_iou(boxA, boxC) == 0.0

    # Identical boxes
    assert _xyxy_iou(boxA, boxA) == 1.0
    print("PASS")


def test_temporal_alignment():
    print("Test 2: Temporal alignment ... ", end="")
    fast = make_fast_dets(30, 2)
    slow = make_slow_dets(30, 2, slow_period=15)

    aligner = TemporalAligner(fast, slow, slow_period=15, max_gap=15)
    mapping = aligner.align_sequences()

    # Slow frames are at 15 and 30 (multiples of slow_period)
    # Frame 1 → nearest is 15 (gap=14 ≤ max_gap=15)
    assert mapping[1] == 15, f"Expected 15, got {mapping[1]}"
    # Frame 8 → nearest is 15 (gap=7)
    assert mapping[8] == 15, f"Expected 15, got {mapping[8]}"
    # Frame 15 → nearest slow frame = 15
    assert mapping[15] == 15, f"Expected 15, got {mapping[15]}"
    # Frame 30 → nearest slow frame = 30
    assert mapping[30] == 30, f"Expected 30, got {mapping[30]}"

    pairs = list(aligner.iter_aligned_pairs())
    assert len(pairs) == 30, f"Expected 30 pairs, got {len(pairs)}"
    print("PASS")


def test_detector_fusion():
    print("Test 3: Detector fusion ... ", end="")
    fast = make_fast_dets(1, 2)[1]
    slow = make_slow_dets(2, 2, slow_period=1)[1]

    fusioner = DetectorFusion(iou_threshold=0.3)
    results = fusioner.fuse_frame(
        frame_id=1,
        fast_dets=fast,
        slow_dets=slow,
        slow_frame_id=1,
    )

    assert len(results) >= 1, "Expected at least one fused detection"
    for r in results:
        assert isinstance(r, FusedDetection)
        assert r.frame_id == 1

    # Matched detections should have both fast and slow fields
    matched = [r for r in results if r.match_iou is not None]
    for m in matched:
        assert m.track_id is not None
        assert m.avg_depth is not None
        assert m.surface_type is not None

    print("PASS")


def test_full_pipeline():
    print("Test 4: Full pipeline ... ", end="")
    fast = make_fast_dets(30, 3)
    slow = make_slow_dets(30, 3, slow_period=15)

    aligner = TemporalAligner(fast, slow, slow_period=15, max_gap=15)
    fusioner = DetectorFusion(iou_threshold=0.3)

    all_fused = []
    for fast_fid, slow_fid, fast_dets, slow_dets in aligner.iter_aligned_pairs():
        fused = fusioner.fuse_frame(fast_fid, fast_dets, slow_dets, slow_fid)
        all_fused.extend(fused)

    assert len(all_fused) >= 30, f"Expected ≥30 fused dets, got {len(all_fused)}"

    # Serialisation round-trip
    records = [d.to_dict() for d in all_fused]
    assert isinstance(records[0], dict)
    assert "frame_id" in records[0]
    assert "avg_depth" in records[0]

    print("PASS")
    return all_fused


def test_no_detections():
    print("Test 5: Empty detection edge cases ... ", end="")
    fusioner = DetectorFusion()
    # Both empty
    assert fusioner.fuse_frame(1, [], [], slow_frame_id=1) == []
    # Fast only
    fast_only = [FastDetection(1, 1, (0, 0, 50, 100), 0.9)]
    results = fusioner.fuse_frame(1, fast_only, [], slow_frame_id=None)
    assert len(results) == 1
    assert results[0].avg_depth is None
    # Slow only
    slow_only = [SlowDetection(1, 1, (0, 0, 50, 100), 0.8, 3.0, 5.0)]
    results = fusioner.fuse_frame(1, [], slow_only, slow_frame_id=1)
    assert len(results) == 1
    assert results[0].track_id is None
    print("PASS")


def test_loader_formats():
    """Test that loaders handle both pipeline and spec file formats."""
    import tempfile, csv as _csv, os as _os

    print("Test 6: Loader format compatibility ... ", end="")
    from VCD.fusion.temporal_aligner import load_fast_detections, load_slow_detections

    # --- Fast: pipeline format (xyxy, 10 cols + speed) ---
    mot_lines = [
        "1,1,100.0,200.0,160.0,320.0,0.92,0,-1,-1,0\n",  # car, slow
        "1,2,300.0,200.0,360.0,320.0,0.88,7,-1,-1,1\n",  # truck, fast
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.writelines(mot_lines)
        mot_path = f.name

    try:
        fast = load_fast_detections(mot_path)
        assert 1 in fast
        assert len(fast[1]) == 2
        d0, d1 = fast[1]
        # xyxy → xywh: (100,200,160,320) → (100,200,60,120)
        assert d0.bbox_xywh == (100.0, 200.0, 60.0, 120.0), f"Got {d0.bbox_xywh}"
        assert d0.object_class == "person"   # class_id 0
        assert d0.speed_class == 0
        assert d1.object_class == "truck"    # class_id 7
        assert d1.speed_class == 1
    finally:
        _os.unlink(mot_path)

    # --- Slow: DPT pipeline column names ---
    dpt_rows = [
        ["Frame_id", "Person_id", "x1", "y1", "x2", "y2",
         "Average_Depth", "Angle", "Distance_Level", "X", "Y", "catagory"],
        ["15", "0", "100", "200", "160", "320", "4.5", "-3.0", "2", "130", "260", "1"],
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = _csv.writer(f)
        writer.writerows(dpt_rows)
        csv_path = f.name

    try:
        slow = load_slow_detections(csv_path)
        assert 15 in slow
        assert len(slow[15]) == 1
        sd = slow[15][0]
        assert sd.avg_depth == 4.5
        assert sd.angle == -3.0
        assert sd.distance_level == 2
        assert sd.confidence == 1.0       # default
        assert sd.surface_type == "unknown"  # default
        assert sd.region_category == 5    # default (catagory not mapped)
    finally:
        _os.unlink(csv_path)

    # --- Slow: fusion spec column names ---
    dpt_spec_rows = [
        ["frame_id", "person_id", "x1", "y1", "x2", "y2",
         "confidence", "avg_depth", "angle", "distance_level",
         "region_category", "surface_type"],
        ["15", "1", "100", "200", "160", "320", "0.85", "4.5", "-3.0", "2", "1", "sidewalk"],
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = _csv.writer(f)
        writer.writerows(dpt_spec_rows)
        csv_path2 = f.name

    try:
        slow2 = load_slow_detections(csv_path2)
        sd2 = slow2[15][0]
        assert sd2.confidence == 0.85
        assert sd2.surface_type == "sidewalk"
        assert sd2.region_category == 1
    finally:
        _os.unlink(csv_path2)

    print("PASS")


# ---------------------------------------------------------------------------
# Optional real-data smoke test
# ---------------------------------------------------------------------------

def run_jaad_smoke_test(jaad_root: str):
    """Quick smoke-test against real JAAD data (if present)."""
    from VCD.fusion.dataset_loader import JAAdLoader
    loader = JAAdLoader(jaad_root)
    clips = list(loader.iter_complete_clips())
    if not clips:
        print(f"[JAAD] No complete clips found under {jaad_root} — skipping.")
        return
    clip_name, mot_txt, dpt_csv = clips[0]
    print(f"[JAAD] Testing clip: {clip_name}")
    from VCD.fusion import run_fusion
    fused = run_fusion(mot_txt, dpt_csv)
    print(f"[JAAD] Produced {len(fused)} fused detections across all frames. PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fusion module synthetic tests")
    parser.add_argument(
        "--jaad",
        metavar="DIR",
        default=None,
        help="JAAD dataset root — run real-data smoke test if provided",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Visionary-CoDriver Fusion Module — Synthetic Tests")
    print("=" * 60)

    test_iou()
    test_temporal_alignment()
    test_detector_fusion()
    all_fused = test_full_pipeline()
    test_no_detections()

    # Summary stats
    matched = [d for d in all_fused if d.match_iou is not None]
    fast_only = [d for d in all_fused if d.track_id is not None and d.avg_depth is None]
    slow_only = [d for d in all_fused if d.track_id is None and d.avg_depth is not None]
    print()
    print("Summary (30-frame synthetic sequence, 3 persons):")
    print(f"  Total fused dets   : {len(all_fused)}")
    print(f"  Matched (fast+slow): {len(matched)}")
    print(f"  Fast-only          : {len(fast_only)}")
    print(f"  Slow-only          : {len(slow_only)}")

    if args.jaad:
        print()
        run_jaad_smoke_test(args.jaad)

    print()
    print("All tests passed.")


if __name__ == "__main__":
    main()
