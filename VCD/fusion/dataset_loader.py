"""Dataset loaders for JAAD and BDD100K, plus a loader for saved fused JSON.

Directory layouts expected
--------------------------
JAAD::

    <root>/
        JAAD_clips/          # .mp4 or .avi video files
        annotations/         # .xml per-clip annotation files
        fast_results/        # MOT .txt files  (frame_id=clip_name)
        slow_results/        # DPT .csv files

BDD100K::

    <root>/
        videos/
            train/ val/ test/
        labels/
            det_20/          # detection labels (json)
        fast_results/        # MOT .txt files
        slow_results/        # DPT .csv files

Fused JSON::

    <path>   # single JSON file or directory of JSON files
    Each JSON file contains a list of dicts matching FusedDetection.to_dict()
"""

from __future__ import annotations

import json
import os
from typing import Generator, List, Optional, Tuple

from VCD.fusion.data_schema import FusedDetection


# ---------------------------------------------------------------------------
# JAAD
# ---------------------------------------------------------------------------

class JAAdLoader:
    """Iterate over JAAD clips and yield paths for fast/slow results.

    Parameters
    ----------
    root : str
        Root directory for the JAAD dataset (see module docstring).
    fast_subdir : str
        Sub-directory under *root* that holds MOT .txt files.
    slow_subdir : str
        Sub-directory under *root* that holds DPT .csv files.
    """

    def __init__(
        self,
        root: str,
        fast_subdir: str = "fast_results",
        slow_subdir: str = "slow_results",
    ) -> None:
        self.root = root
        self.fast_dir = os.path.join(root, fast_subdir)
        self.slow_dir = os.path.join(root, slow_subdir)

    def iter_clips(
        self,
    ) -> Generator[Tuple[str, Optional[str], Optional[str]], None, None]:
        """Yield (clip_name, mot_txt_path, dpt_csv_path).

        Either path is None if the file is missing.
        """
        clips_dir = os.path.join(self.root, "JAAD_clips")
        if not os.path.isdir(clips_dir):
            return

        for fname in sorted(os.listdir(clips_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in {".mp4", ".avi", ".mov"}:
                continue
            mot_path = os.path.join(self.fast_dir, stem + ".txt")
            csv_path = os.path.join(self.slow_dir, stem + ".csv")
            yield (
                stem,
                mot_path if os.path.isfile(mot_path) else None,
                csv_path if os.path.isfile(csv_path) else None,
            )

    def iter_complete_clips(
        self,
    ) -> Generator[Tuple[str, str, str], None, None]:
        """Yield only clips where both MOT and CSV files exist."""
        for clip, mot, csv_ in self.iter_clips():
            if mot and csv_:
                yield clip, mot, csv_


# ---------------------------------------------------------------------------
# BDD100K
# ---------------------------------------------------------------------------

class BDD100KLoader:
    """Iterate over BDD100K videos and yield paths for fast/slow results.

    Parameters
    ----------
    root : str
        Root directory for BDD100K (see module docstring).
    split : str
        One of 'train', 'val', 'test'.
    fast_subdir : str
        Sub-directory under *root* that holds MOT .txt files.
    slow_subdir : str
        Sub-directory under *root* that holds DPT .csv files.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        fast_subdir: str = "fast_results",
        slow_subdir: str = "slow_results",
    ) -> None:
        self.root = root
        self.split = split
        self.fast_dir = os.path.join(root, fast_subdir)
        self.slow_dir = os.path.join(root, slow_subdir)

    def iter_clips(
        self,
    ) -> Generator[Tuple[str, Optional[str], Optional[str]], None, None]:
        """Yield (video_name, mot_txt_path, dpt_csv_path)."""
        videos_dir = os.path.join(self.root, "videos", self.split)
        if not os.path.isdir(videos_dir):
            return

        for fname in sorted(os.listdir(videos_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in {".mp4", ".avi", ".mov"}:
                continue
            mot_path = os.path.join(self.fast_dir, stem + ".txt")
            csv_path = os.path.join(self.slow_dir, stem + ".csv")
            yield (
                stem,
                mot_path if os.path.isfile(mot_path) else None,
                csv_path if os.path.isfile(csv_path) else None,
            )

    def iter_complete_clips(
        self,
    ) -> Generator[Tuple[str, str, str], None, None]:
        """Yield only clips where both MOT and CSV files exist."""
        for clip, mot, csv_ in self.iter_clips():
            if mot and csv_:
                yield clip, mot, csv_


# ---------------------------------------------------------------------------
# FusionResultLoader
# ---------------------------------------------------------------------------

class FusionResultLoader:
    """Load previously saved fused detection JSON files.

    Parameters
    ----------
    path : str
        Either a single .json file or a directory of .json files.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def _json_paths(self) -> List[str]:
        if os.path.isfile(self.path):
            return [self.path]
        if os.path.isdir(self.path):
            return sorted(
                os.path.join(self.path, f)
                for f in os.listdir(self.path)
                if f.endswith(".json")
            )
        raise FileNotFoundError(f"FusionResultLoader: path not found: {self.path}")

    def load_all(self) -> List[FusedDetection]:
        """Return all FusedDetection objects from all JSON files."""
        results: List[FusedDetection] = []
        for p in self._json_paths():
            with open(p, "r") as fh:
                records = json.load(fh)
            for rec in records:
                results.append(self._from_dict(rec))
        return results

    def iter_frames(
        self,
    ) -> Generator[Tuple[str, List[FusedDetection]], None, None]:
        """Yield (source_file, fused_detections) per JSON file."""
        for p in self._json_paths():
            with open(p, "r") as fh:
                records = json.load(fh)
            dets = [self._from_dict(r) for r in records]
            yield os.path.basename(p), dets

    @staticmethod
    def _from_dict(d: dict) -> FusedDetection:
        bbox = tuple(d["bbox_xywh"]) if d.get("bbox_xywh") else None
        return FusedDetection(
            frame_id=d["frame_id"],
            track_id=d.get("track_id"),
            bbox_xywh=bbox,  # type: ignore[arg-type]
            fast_confidence=d.get("fast_confidence"),
            speed_class=d.get("speed_class"),
            object_class=d.get("object_class", "person"),
            slow_frame_id=d.get("slow_frame_id"),
            avg_depth=d.get("avg_depth"),
            angle=d.get("angle"),
            distance_level=d.get("distance_level"),
            region_category=d.get("region_category"),
            surface_type=d.get("surface_type"),
            slow_confidence=d.get("slow_confidence"),
            match_iou=d.get("match_iou"),
        )
