#!/usr/bin/env python3
"""Visualize fused slow-fast detections overlaid on the original video.

Usage
-----
# Single clip
python scripts/visualize_fusion.py \
    /mnt/sda/edward/data_vcd/JAAD/JAAD_clips/video_0001.mp4 \
    /mnt/sda/edward/data_vcd/JAAD/fused_results/video_0001.json \
    --output /tmp/video_0001_fused.mp4

# Batch: all JAAD clips that have a fused JSON
python scripts/visualize_fusion.py --batch \
    --clips-dir  /mnt/sda/edward/data_vcd/JAAD/JAAD_clips \
    --fused-dir  /mnt/sda/edward/data_vcd/JAAD/fused_results \
    --output-dir /mnt/sda/edward/data_vcd/JAAD/viz_results

Overlay legend
--------------
  GREEN box   — slow / stationary object
  RED box     — fast / moving object
  BLUE box    — slow-stream only (no fast track)
  Solid box   — matched (fast + slow data, has depth)
  Dashed box  — fast-only (no depth / surface info)
  Top label   — #{track_id} {class} | {depth}m @ {angle}°
  Bottom tag  — surface type  (road / sidewalk / …)
  "SLW" badge — frame is a slow-stream keyframe
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICKNESS = 1

# box colour by speed class
_COLOR_SLOW  = (50,  205,  50)   # lime green  (slow / stationary)
_COLOR_FAST  = (0,   80,  220)   # blue-ish red  (fast)
_COLOR_SLOW_ONLY = (200, 160,  0) # amber (no fast track)

_DIST_LABELS = {1: "very close", 2: "close", 3: "medium", 4: "far"}
_REGION_LABELS = {
    0: "road", 1: "sidewalk", 2: "crosswalk",
    3: "parking", 4: "grass", 5: "unknown",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fused(json_path: str) -> Dict[int, list]:
    """Return fused detections grouped by frame_id."""
    with open(json_path) as f:
        records = json.load(f)
    by_frame: Dict[int, list] = defaultdict(list)
    for r in records:
        by_frame[r["frame_id"]].append(r)
    return by_frame


def _draw_dashed_rect(img, pt1, pt2, color, thickness=1, gap=8):
    """Draw a dashed rectangle (fast-only detections)."""
    x1, y1 = pt1
    x2, y2 = pt2
    for start, end in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                        ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        length = max(1, int(np.hypot(dx, dy)))
        steps = length // (gap * 2) or 1
        for i in range(steps):
            t0 = i * 2 * gap / length
            t1 = min((i * 2 + 1) * gap / length, 1.0)
            p0 = (int(sx + t0 * dx), int(sy + t0 * dy))
            p1 = (int(sx + t1 * dx), int(sy + t1 * dy))
            cv2.line(img, p0, p1, color, thickness)


def _label_bg(img, text, x, y, color, text_color=(255, 255, 255)):
    """Draw a filled rectangle behind text for readability."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
    cv2.rectangle(img, (x, y - th - baseline - 2), (x + tw + 2, y + 2), color, -1)
    cv2.putText(img, text, (x + 1, y - baseline), _FONT, _FONT_SCALE, text_color, _THICKNESS, cv2.LINE_AA)


def _draw_detection(frame: np.ndarray, rec: dict, is_slow_frame: bool) -> None:
    """Overlay one FusedDetection record onto frame."""
    bbox = rec.get("bbox_xywh")
    if bbox is None:
        return
    x, y, w, h = [int(v) for v in bbox]
    x2, y2 = x + w, y + h

    track_id   = rec.get("track_id")
    obj_class  = rec.get("object_class") or "obj"
    speed      = rec.get("speed_class")       # 0=slow, 1=fast, None=slow-only
    avg_depth  = rec.get("avg_depth")
    angle      = rec.get("angle")
    surface    = rec.get("surface_type") or ""
    region_cat = rec.get("region_category")
    has_slow   = avg_depth is not None

    # Choose colour
    if track_id is None:
        color = _COLOR_SLOW_ONLY
    elif speed == 1:
        color = _COLOR_FAST
    else:
        color = _COLOR_SLOW

    # Box style: solid if matched, dashed if fast-only
    if has_slow:
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    else:
        _draw_dashed_rect(frame, (x, y), (x2, y2), color, thickness=2)

    # Top label — track / class / depth
    if track_id is not None:
        top_label = f"#{track_id} {obj_class}"
    else:
        top_label = obj_class
    if avg_depth is not None:
        top_label += f" | {avg_depth:.1f}m"
    if angle is not None:
        top_label += f" {angle:+.0f}\u00b0"

    _label_bg(frame, top_label, x, y - 2, color)

    # Bottom tag — surface / region
    if surface and surface != "unknown":
        surf_label = surface
    elif region_cat is not None and region_cat in _REGION_LABELS:
        surf_label = _REGION_LABELS[region_cat]
    else:
        surf_label = ""
    if surf_label:
        _label_bg(frame, surf_label, x, y2 + 14, (60, 60, 60))

    # "SLW" badge when this frame is a slow keyframe
    if is_slow_frame and has_slow:
        cv2.circle(frame, (x2 - 6, y + 6), 5, (255, 200, 0), -1)


def _is_slow_keyframe(frame_id: int, slow_period: int = 15) -> bool:
    return frame_id % slow_period == 0


# ---------------------------------------------------------------------------
# Core visualise function
# ---------------------------------------------------------------------------

def visualize_clip(
    video_path: str,
    fused_json_path: str,
    output_path: str,
    slow_period: int = 15,
    frame_id_offset: int = 0,
) -> None:
    """Render fused detections onto *video_path* and write to *output_path*.

    Parameters
    ----------
    frame_id_offset
        Add this to the video frame index to match the fused JSON frame_id.
        Use 1 if the pipeline uses 1-indexed frame IDs (default 0).
    """
    by_frame = _load_fused(fused_json_path)
    if not by_frame:
        print(f"  [WARN] No fused detections in {fused_json_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"  {clip_name}: {width}x{height} @ {fps:.1f} fps  ({total} frames)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fused_id   = frame_idx + frame_id_offset
        dets       = by_frame.get(fused_id, [])
        is_slow    = _is_slow_keyframe(fused_id, slow_period)

        for rec in dets:
            _draw_detection(frame, rec, is_slow)

        # HUD: frame counter + detection count
        hud = f"frame {fused_id}  |  {len(dets)} det"
        if is_slow:
            hud += "  [SLOW KEY]"
        cv2.putText(frame, hud, (8, 22), _FONT, 0.55, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, hud, (8, 22), _FONT, 0.55, (30, 30, 30),   1, cv2.LINE_AA)

        writer.write(frame)

        if frame_idx % 300 == 0:
            print(f"    {frame_idx}/{total} frames done …")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  -> saved: {output_path}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def visualize_batch(
    clips_dir: str,
    fused_dir: str,
    output_dir: str,
    slow_period: int = 15,
    frame_id_offset: int = 0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    video_exts = {".mp4", ".avi", ".mov"}

    clips = sorted(
        f for f in os.listdir(clips_dir)
        if os.path.splitext(f)[1].lower() in video_exts
    )
    if not clips:
        print(f"No video files found in {clips_dir}")
        return

    done = skipped = 0
    for fname in clips:
        stem = os.path.splitext(fname)[0]
        video_path = os.path.join(clips_dir, fname)
        json_path  = os.path.join(fused_dir,  stem + ".json")
        out_path   = os.path.join(output_dir, stem + "_fused.mp4")

        if not os.path.isfile(json_path):
            print(f"  [SKIP] no fused JSON for {stem}")
            skipped += 1
            continue

        print(f"Processing {stem} …")
        try:
            visualize_clip(video_path, json_path, out_path, slow_period, frame_id_offset)
            done += 1
        except Exception as exc:
            print(f"  [ERROR] {stem}: {exc}")
            skipped += 1

    print(f"\nDone: {done} visualized, {skipped} skipped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay fused slow-fast detections on JAAD/BDD100K video clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--batch", action="store_true",
                      help="Batch mode: process all clips in --clips-dir")

    # Single-clip args
    parser.add_argument("video",   nargs="?", help="Input video path (single mode)")
    parser.add_argument("fused",   nargs="?", help="Fused JSON path  (single mode)")
    parser.add_argument("--output", "-o", default=None, help="Output video path")

    # Batch args
    parser.add_argument("--clips-dir",  default=None)
    parser.add_argument("--fused-dir",  default=None)
    parser.add_argument("--output-dir", default=None)

    # Shared
    parser.add_argument("--slow-period",     type=int, default=15,
                        help="Fast frames between slow keyframes (default 15)")
    parser.add_argument("--frame-id-offset", type=int, default=0,
                        help="Add to video frame index to match JSON frame_id "
                             "(use 1 if pipeline is 1-indexed, default 0)")

    args = parser.parse_args()

    if args.batch:
        if not args.clips_dir or not args.fused_dir or not args.output_dir:
            parser.error("--batch requires --clips-dir, --fused-dir, --output-dir")
        visualize_batch(
            args.clips_dir, args.fused_dir, args.output_dir,
            args.slow_period, args.frame_id_offset,
        )
    else:
        if not args.video or not args.fused:
            parser.error("Provide VIDEO and FUSED arguments (or use --batch)")
        out = args.output or (os.path.splitext(args.video)[0] + "_fused.mp4")
        visualize_clip(args.video, args.fused, out, args.slow_period, args.frame_id_offset)


if __name__ == "__main__":
    main()
