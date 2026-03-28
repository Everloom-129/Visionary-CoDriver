import bisect
import functools
import time
from typing import List, Optional


def run_time_decorator(func):
    """Decorator to measure and print execution time of functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper


def frame_to_timestamp(frame_id: int, fps: float = 30.0) -> float:
    """Convert a 1-indexed frame id to a timestamp in seconds.

    Parameters
    ----------
    frame_id : int  — 1-indexed frame number
    fps      : float — frames per second of the source video (default 30.0)

    Returns
    -------
    Timestamp in seconds (0-indexed, i.e. frame 1 → 0.0 s at 30 fps).
    """
    return (frame_id - 1) / fps


def nearest_frame(target_fid: int, candidate_fids: List[int]) -> Optional[int]:
    """Return the frame id in *candidate_fids* closest to *target_fid*.

    Uses binary search for O(log n) performance.
    Returns None if *candidate_fids* is empty.

    Parameters
    ----------
    target_fid     : frame id to match
    candidate_fids : sorted list of candidate frame ids
    """
    if not candidate_fids:
        return None

    idx = bisect.bisect_left(candidate_fids, target_fid)

    candidates = []
    if idx < len(candidate_fids):
        candidates.append(candidate_fids[idx])
    if idx > 0:
        candidates.append(candidate_fids[idx - 1])

    return min(candidates, key=lambda c: abs(c - target_fid))
