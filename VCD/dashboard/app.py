"""Interactive Gradio dashboard for JAAD fusion result verification.

Launch via:
    python scripts/run_dashboard.py
    # → http://127.0.0.1:7860

Overlay layers (toggleable):
  Fast  — solid coloured boxes from MOT .txt (green=slow, red=fast)
  Slow  — semi-transparent region fills from DPT CSV (road/sidewalk/…)
  Fused — depth-labelled boxes from fusion JSON (solid=matched, dashed=fast-only)
"""
from __future__ import annotations

import bisect
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import gradio as gr
import gradio_client.utils as _gradio_client_utils

# Gradio 4.44 + Pydantic ≥2.11: JSON schemas use boolean `additionalProperties`
# (valid JSON Schema). gradio_client's `_json_schema_to_python_type` assumes it
# is always a dict, which breaks `/` page load when building API info.
_orig_json_schema_to_py = _gradio_client_utils._json_schema_to_python_type


def _json_schema_to_python_type_safe(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_py(schema, defs)


_gradio_client_utils._json_schema_to_python_type = _json_schema_to_python_type_safe

# ── Project root on path ─────────────────────────────────────────────────
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from VCD.fusion.temporal_aligner import load_fast_detections, load_slow_detections
from VCD.fusion.data_schema import FastDetection, SlowDetection

# ── Visual constants ─────────────────────────────────────────────────────
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICK      = 1

_COLOR_SLOW      = (50,  205,  50)   # lime green — slow/stationary
_COLOR_FAST      = (0,    80, 220)   # blue-red   — fast/moving
_COLOR_SLOW_ONLY = (200, 160,   0)   # amber      — slow-stream only

# BGR colours by region_category
_REGION_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (50,  200,  50),   # road      — green
    1: (200, 200,   0),   # sidewalk  — cyan
    2: (0,   200, 200),   # crosswalk — yellow
    3: (200,   0, 200),   # parking   — magenta
    4: (0,   100,   0),   # grass     — dark green
    5: (128, 128, 128),   # unknown   — gray
}
_REGION_LABELS = {
    0: "road", 1: "sidewalk", 2: "crosswalk",
    3: "parking", 4: "grass", 5: "unknown",
}
_DIST_LABELS = {1: "very close", 2: "close", 3: "medium", 4: "far"}


# ── Drawing helpers (ported from scripts/visualize_fusion.py) ────────────

def _label_bg(img: np.ndarray, text: str, x: int, y: int,
              bg_color, text_color=(255, 255, 255)) -> None:
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICK)
    cv2.rectangle(img, (x, y - th - baseline - 2), (x + tw + 2, y + 2), bg_color, -1)
    cv2.putText(img, text, (x + 1, y - baseline), _FONT, _FONT_SCALE,
                text_color, _THICK, cv2.LINE_AA)


def _draw_dashed_rect(img: np.ndarray, pt1, pt2, color, thickness=1, gap=8) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    for start, end in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                       ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        length = max(1, int(np.hypot(dx, dy)))
        steps  = length // (gap * 2) or 1
        for i in range(steps):
            t0 = i * 2 * gap / length
            t1 = min((i * 2 + 1) * gap / length, 1.0)
            p0 = (int(sx + t0 * dx), int(sy + t0 * dy))
            p1 = (int(sx + t1 * dx), int(sy + t1 * dy))
            cv2.line(img, p0, p1, color, thickness)


def _draw_fused(frame: np.ndarray, rec: dict, is_slow_key: bool) -> None:
    """Draw one FusedDetection record (replicates visualize_fusion._draw_detection)."""
    bbox = rec.get("bbox_xywh")
    if bbox is None:
        return
    x, y, w, h = [int(v) for v in bbox]
    x2, y2     = x + w, y + h
    track_id   = rec.get("track_id")
    obj_class  = rec.get("object_class") or "obj"
    speed      = rec.get("speed_class")
    avg_depth  = rec.get("avg_depth")
    angle      = rec.get("angle")
    surface    = rec.get("surface_type") or ""
    region_cat = rec.get("region_category")
    has_slow   = avg_depth is not None

    if track_id is None:
        color = _COLOR_SLOW_ONLY
    elif speed == 1:
        color = _COLOR_FAST
    else:
        color = _COLOR_SLOW

    if has_slow:
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    else:
        _draw_dashed_rect(frame, (x, y), (x2, y2), color, thickness=2)

    top = f"#{track_id} {obj_class}" if track_id is not None else obj_class
    if avg_depth is not None:
        top += f" | {avg_depth:.1f}m"
    if angle is not None:
        top += f" {angle:+.0f}\u00b0"
    _label_bg(frame, top, x, y - 2, color)

    surf = surface if (surface and surface != "unknown") else _REGION_LABELS.get(region_cat, "")
    if surf:
        _label_bg(frame, surf, x, y2 + 14, (60, 60, 60))

    if is_slow_key and has_slow:
        cv2.circle(frame, (x2 - 6, y + 6), 5, (255, 200, 0), -1)


# ── Slow-frame lookup (frame_idx is 0-indexed fast; slow IDs are 1-indexed) ──

def _nearest_slow(frame_idx: int, slow_frames: List[int], max_gap: int = 15) -> Optional[int]:
    """Return nearest slow frame_id for a 0-indexed fast frame_idx."""
    if not slow_frames:
        return None
    target = frame_idx + 1          # convert to 1-indexed for comparison
    pos    = bisect.bisect_left(slow_frames, target)
    candidates = slow_frames[max(0, pos - 1):pos + 2]
    best = min(candidates, key=lambda x: abs(x - target))
    return best if abs(best - target) <= max_gap else None


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class ClipInfo:
    stem: str
    video_path: str
    person_txt: str
    slow_csv: Optional[str] = None
    fusion_json: Optional[str] = None


@dataclass
class ClipData:
    stem: str
    video_path: str
    total_frames: int
    fps: float
    fast_by_frame: Dict[int, List[FastDetection]]
    slow_by_frame: Dict[int, List[SlowDetection]]
    slow_frames_sorted: List[int]        # sorted slow frame IDs (1-indexed)
    fused_by_frame: Dict[int, List[dict]]


# ── Clip discovery ───────────────────────────────────────────────────────

class ClipIndex:
    def __init__(self, jaad_root: str):
        self.clips: List[ClipInfo] = []
        fast_dir   = os.path.join(jaad_root, "fast_results")
        slow_dir   = os.path.join(jaad_root, "slow_results")
        fusion_dir = os.path.join(jaad_root, "fusion_results")
        clips_dir  = os.path.join(jaad_root, "JAAD_clips")

        if not os.path.isdir(fast_dir):
            print(f"[ClipIndex] fast_results not found: {fast_dir}")
            return

        for fname in sorted(os.listdir(fast_dir)):
            if not fname.endswith("_person.txt"):
                continue
            stem  = fname[: -len("_person.txt")]
            video = os.path.join(clips_dir, stem + ".mp4")
            if not os.path.isfile(video):
                continue
            slow  = os.path.join(slow_dir,   stem + ".csv")
            fused = os.path.join(fusion_dir, stem + ".json")
            self.clips.append(ClipInfo(
                stem        = stem,
                video_path  = video,
                person_txt  = os.path.join(fast_dir, fname),
                slow_csv    = slow  if os.path.isfile(slow)  else None,
                fusion_json = fused if os.path.isfile(fused) else None,
            ))

    @property
    def names(self) -> List[str]:
        return [c.stem for c in self.clips]

    def get(self, stem: str) -> Optional[ClipInfo]:
        for c in self.clips:
            if c.stem == stem:
                return c
        return None


# ── Data loading ─────────────────────────────────────────────────────────

def load_clip(info: ClipInfo) -> ClipData:
    fast = load_fast_detections(info.person_txt)

    slow: Dict[int, List[SlowDetection]] = {}
    if info.slow_csv:
        slow = load_slow_detections(info.slow_csv)
    slow_frames_sorted = sorted(slow.keys())

    fused_by_frame: Dict[int, List[dict]] = defaultdict(list)
    if info.fusion_json:
        with open(info.fusion_json) as f:
            for rec in json.load(f):
                fused_by_frame[rec["frame_id"]].append(rec)

    cap   = cv2.VideoCapture(info.video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    return ClipData(
        stem              = info.stem,
        video_path        = info.video_path,
        total_frames      = total,
        fps               = fps,
        fast_by_frame     = fast,
        slow_by_frame     = slow,
        slow_frames_sorted= slow_frames_sorted,
        fused_by_frame    = dict(fused_by_frame),
    )


# ── Frame rendering ───────────────────────────────────────────────────────

def render_frame(
    clip: ClipData,
    frame_idx: int,
    show_fast: bool = True,
    show_slow: bool = True,
    show_fused: bool = False,
) -> np.ndarray:
    """Return an annotated RGB frame as a numpy H×W×3 array."""
    frame_idx = int(np.clip(frame_idx, 0, clip.total_frames - 1))

    cap = cv2.VideoCapture(clip.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        cv2.putText(frame, f"Cannot read frame {frame_idx}", (20, 240),
                    _FONT, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    is_slow_key = (frame_idx % 15) == 0

    # ── Layer 1: slow region fills (semi-transparent) ────────────────────
    if show_slow and clip.slow_frames_sorted:
        slow_fid = _nearest_slow(frame_idx, clip.slow_frames_sorted)
        if slow_fid is not None:
            for det in clip.slow_by_frame.get(slow_fid, []):
                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                color   = _REGION_COLORS.get(det.region_category, (128, 128, 128))
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det.surface_type} | {det.avg_depth:.1f}m"
                _label_bg(frame, label, x1, y2 + 14, color)

    # ── Layer 2: fast bboxes (solid coloured borders, only when fused off) ─
    if show_fast and not show_fused:
        for det in clip.fast_by_frame.get(frame_idx, []):
            x, y, w, h = [int(v) for v in det.bbox_xywh]
            x2f, y2f   = x + w, y + h
            color       = _COLOR_FAST if det.speed_class == 1 else _COLOR_SLOW
            cv2.rectangle(frame, (x, y), (x2f, y2f), color, 2)
            speed_str   = "FST" if det.speed_class == 1 else "SLW"
            _label_bg(frame, f"#{det.track_id} {det.object_class} | {speed_str}", x, y - 2, color)

    # ── Layer 3: fused detections ─────────────────────────────────────────
    if show_fused:
        for rec in clip.fused_by_frame.get(frame_idx, []):
            _draw_fused(frame, rec, is_slow_key)

    # ── HUD ───────────────────────────────────────────────────────────────
    n_fast  = len(clip.fast_by_frame.get(frame_idx, []))
    slow_fid_hud = _nearest_slow(frame_idx, clip.slow_frames_sorted)
    n_slow  = len(clip.slow_by_frame.get(slow_fid_hud, [])) if slow_fid_hud else 0
    n_fused = len(clip.fused_by_frame.get(frame_idx, []))
    hud = f"frame {frame_idx}  |  fast:{n_fast}  slow:{n_slow}  fused:{n_fused}"
    if is_slow_key:
        hud += "  [SLW]"
    cv2.putText(frame, hud, (8, 22), _FONT, 0.55, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, hud, (8, 22), _FONT, 0.55, (30,  30,  30),  1, cv2.LINE_AA)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── Detection tables ──────────────────────────────────────────────────────

def make_tables(
    clip: ClipData, frame_idx: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame_idx = int(np.clip(frame_idx, 0, clip.total_frames - 1))

    # Fast table
    fast_rows = []
    for d in clip.fast_by_frame.get(frame_idx, []):
        x, y, w, h = d.bbox_xywh
        fast_rows.append({
            "track_id": d.track_id,
            "class": d.object_class,
            "speed": "FST" if d.speed_class == 1 else "SLW",
            "x": round(x, 1), "y": round(y, 1),
            "w": round(w, 1), "h": round(h, 1),
            "conf": round(d.confidence, 3),
        })
    df_fast = (pd.DataFrame(fast_rows)
               if fast_rows
               else pd.DataFrame(columns=["track_id", "class", "speed", "x", "y", "w", "h", "conf"]))

    # Slow table
    slow_rows = []
    slow_fid = _nearest_slow(frame_idx, clip.slow_frames_sorted)
    if slow_fid:
        for d in clip.slow_by_frame.get(slow_fid, []):
            x1, y1, x2, y2 = d.bbox_xyxy
            slow_rows.append({
                "id": d.object_id,
                "surface": d.surface_type,
                "region": _REGION_LABELS.get(d.region_category, "?"),
                "depth_m": round(d.avg_depth, 2),
                "angle_deg": round(d.angle, 1),
                "dist": _DIST_LABELS.get(d.distance_level, "?"),
                "slow_fid": slow_fid,
                "conf": round(d.confidence, 3),
            })
    df_slow = (pd.DataFrame(slow_rows)
               if slow_rows
               else pd.DataFrame(columns=["id", "surface", "region", "depth_m", "angle_deg", "dist", "slow_fid", "conf"]))

    # Fused table
    fused_rows = []
    for r in clip.fused_by_frame.get(frame_idx, []):
        fused_rows.append({
            "track_id": r.get("track_id"),
            "class": r.get("object_class"),
            "speed": "FST" if r.get("speed_class") == 1 else "SLW",
            "depth_m": round(r["avg_depth"], 2) if r.get("avg_depth") is not None else None,
            "angle_deg": round(r["angle"], 1) if r.get("angle") is not None else None,
            "surface": r.get("surface_type"),
            "iou": round(r["match_iou"], 3) if r.get("match_iou") is not None else None,
        })
    df_fused = (pd.DataFrame(fused_rows)
                if fused_rows
                else pd.DataFrame(columns=["track_id", "class", "speed", "depth_m", "angle_deg", "surface", "iou"]))

    return df_fast, df_slow, df_fused


# ── Gradio app ────────────────────────────────────────────────────────────

def build_app(index: ClipIndex) -> gr.Blocks:
    clip_names = index.names

    with gr.Blocks(title="VCD Fusion Dashboard") as app:
        gr.Markdown("## Visionary CoDriver — Fusion Dashboard")
        clip_state = gr.State(value=None)

        with gr.Row():
            # ── Left panel ────────────────────────────────────────────────
            with gr.Column(scale=3):
                clip_dd   = gr.Dropdown(
                    choices=clip_names,
                    value=clip_names[0] if clip_names else None,
                    label="Clip",
                )
                load_btn  = gr.Button("Load Clip", variant="primary")
                clip_info = gr.Markdown("_No clip loaded_")

                with gr.Accordion("Overlay Options", open=True):
                    chk_fast  = gr.Checkbox(label="Fast bboxes (MOT .txt)",    value=True)
                    chk_slow  = gr.Checkbox(label="Slow fills (DPT CSV)",       value=True)
                    chk_fused = gr.Checkbox(label="Fused detections (JSON)",    value=False)

                with gr.Accordion("Detection Tables", open=True):
                    gr.Markdown("**Fast** — MOT track at current frame")
                    tbl_fast  = gr.Dataframe(interactive=False)
                    gr.Markdown("**Slow** — nearest DPT frame")
                    tbl_slow  = gr.Dataframe(interactive=False)
                    gr.Markdown("**Fused** — IoU-matched")
                    tbl_fused = gr.Dataframe(interactive=False)

            # ── Right panel ───────────────────────────────────────────────
            with gr.Column(scale=7):
                frame_img = gr.Image(label="Frame", type="numpy", height=600)
                frame_sl  = gr.Slider(minimum=0, maximum=1, step=1, value=0, label="Frame")
                with gr.Row():
                    btn_m10 = gr.Button("-10",  scale=1)
                    btn_m1  = gr.Button("-1",   scale=1)
                    btn_p1  = gr.Button("+1",   scale=1)
                    btn_p10 = gr.Button("+10",  scale=1)

        # ── Event handlers ────────────────────────────────────────────────

        def _render_all(frame_idx, clip, show_fast, show_slow, show_fused):
            if clip is None:
                blank = np.full((480, 640, 3), 40, dtype=np.uint8)
                return blank, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            img          = render_frame(clip, int(frame_idx), show_fast, show_slow, show_fused)
            df_f, df_s, df_fu = make_tables(clip, int(frame_idx))
            return img, df_f, df_s, df_fu

        def _load(clip_name):
            info = index.get(clip_name)
            if info is None:
                return (None, "_Clip not found_",
                        gr.update(), None,
                        pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
            clip = load_clip(info)
            tags = []
            if info.slow_csv:    tags.append("slow: YES")
            else:                tags.append("slow: —")
            if info.fusion_json: tags.append("fused: YES")
            else:                tags.append("fused: —")
            info_md = (f"**{clip.stem}** — {clip.total_frames} frames "
                       f"@ {clip.fps:.1f} fps  |  " + "  |  ".join(tags))
            slider_upd = gr.update(maximum=clip.total_frames - 1, value=0)
            img, df_f, df_s, df_fu = _render_all(0, clip, True, True, False)
            return clip, info_md, slider_upd, img, df_f, df_s, df_fu

        def _step(frame_idx, clip, delta, show_fast, show_slow, show_fused):
            if clip is None:
                return 0, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            new_idx = int(np.clip(int(frame_idx) + delta, 0, clip.total_frames - 1))
            img, df_f, df_s, df_fu = _render_all(new_idx, clip, show_fast, show_slow, show_fused)
            return new_idx, img, df_f, df_s, df_fu

        # ── Wire events ───────────────────────────────────────────────────

        _render_ins  = [frame_sl, clip_state, chk_fast, chk_slow, chk_fused]
        _render_outs = [frame_img, tbl_fast, tbl_slow, tbl_fused]

        load_btn.click(
            _load,
            inputs=[clip_dd],
            outputs=[clip_state, clip_info, frame_sl, frame_img,
                     tbl_fast, tbl_slow, tbl_fused],
        )

        frame_sl.release(_render_all, inputs=_render_ins, outputs=_render_outs)
        for chk in [chk_fast, chk_slow, chk_fused]:
            chk.change(_render_all, inputs=_render_ins, outputs=_render_outs)

        for btn, delta in [(btn_m10, -10), (btn_m1, -1), (btn_p1, 1), (btn_p10, 10)]:
            btn.click(
                lambda fi, cl, sf, ss, sfu, d=delta: _step(fi, cl, d, sf, ss, sfu),
                inputs=[frame_sl, clip_state, chk_fast, chk_slow, chk_fused],
                outputs=[frame_sl, frame_img, tbl_fast, tbl_slow, tbl_fused],
            )

    return app


def main(jaad_root: str = "/mnt/sda/edward/data_vcd/JAAD") -> None:
    index = ClipIndex(jaad_root)
    if not index.clips:
        print(f"No clips found under {jaad_root}/fast_results/ — "
              "ensure *_person.txt files and matching .mp4 videos exist.")
        sys.exit(1)
    print(f"Found {len(index.clips)} clip(s): {index.names}")
    app = build_app(index)
    # 127.0.0.1 avoids some environments where `localhost` fails Gradio's url_ok check;
    # set GRADIO_SERVER_NAME=0.0.0.0 for LAN access.
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7888"))
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_api=False,
    )


if __name__ == "__main__":
    main()
