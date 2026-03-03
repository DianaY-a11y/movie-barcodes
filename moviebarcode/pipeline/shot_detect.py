"""
Shot boundary detection using HSV histogram distances with adaptive thresholding.

Algorithm:
  1. Decode video at low fps (default 2fps) and low resolution.
  2. Per frame: compute normalized HSV histogram.
  3. Compute L1 distance between consecutive histograms.
  4. Trigger a cut when distance > local_mean + multiplier * local_std
     using a rolling window for adaptivity.
  5. Enforce minimum shot duration to suppress noise cuts.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable, Optional

from moviebarcode.models import Shot
from moviebarcode.pipeline.letterbox import apply_crop
from moviebarcode.utils.av_helpers import iter_frames_at_fps


# Adaptive threshold multipliers per sensitivity setting
_SENSITIVITY_MAP = {
    "low":    4.5,   # only hard cuts
    "medium": 3.0,   # default — catches most cuts
    "high":   2.0,   # catches soft transitions too
}

_DETECTION_SCALE = (256, 144)

# HSV histogram bins: H=16, S=4, V=4 → 256-dimensional feature
_H_BINS = 16
_S_BINS = 4
_V_BINS = 4


def detect_shots(
    path: str | Path,
    detection_fps: float = 2.0,
    sensitivity: str = "medium",
    crop: Optional[tuple[float, float]] = None,
    min_shot_duration: float = 0.5,
    start: float = 0.0,
    end: Optional[float] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> list[Shot]:
    """
    Detect shot boundaries in a video file.

    Parameters
    ----------
    path               : video file path
    detection_fps      : frames per second to decode for analysis
    sensitivity        : 'low', 'medium', or 'high'
    crop               : (top_frac, bot_frac) from letterbox detection, or None
    min_shot_duration  : minimum shot length in seconds (prevents micro-shots)
    start              : start time in seconds
    end                : end time in seconds (None = full video)
    progress_callback  : called with a float in [0, 1] as frames are processed

    Returns
    -------
    List of Shot objects with start/end times set. Colors are not yet computed.
    """
    multiplier = _SENSITIVITY_MAP.get(sensitivity, 3.0)

    timestamps: list[float] = []
    histograms: list[np.ndarray] = []

    duration_hint = end or 1e9  # Used only for progress reporting

    for ts, frame in iter_frames_at_fps(
        path,
        target_fps=detection_fps,
        scale=_DETECTION_SCALE,
        start=start,
        end=end,
    ):
        if crop is not None:
            frame = apply_crop(frame, crop[0], crop[1])
            if frame.size == 0:
                continue

        hist = _hsv_histogram(frame)
        timestamps.append(ts)
        histograms.append(hist)

        if progress_callback is not None:
            progress = min((ts - start) / (duration_hint - start + 1e-9), 1.0)
            progress_callback(progress)

    if len(timestamps) < 2:
        # Entire video is one shot
        t_start = start
        t_end = end if end is not None else timestamps[-1] if timestamps else 0.0
        return [Shot(index=0, start=t_start, end=t_end)]

    distances = _compute_distances(histograms)
    cut_times = _adaptive_threshold(
        timestamps, distances, multiplier, min_shot_duration, start
    )

    t_end = end if end is not None else timestamps[-1]
    if cut_times[-1] < t_end - min_shot_duration:
        cut_times.append(t_end)

    shots = [
        Shot(index=i, start=s, end=e)
        for i, (s, e) in enumerate(zip(cut_times[:-1], cut_times[1:]))
    ]
    return shots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hsv_histogram(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a flat, normalized HSV histogram from an RGB frame.
    Returns shape (_H_BINS * _S_BINS * _V_BINS,).
    """
    from PIL import Image

    img = Image.fromarray(frame_rgb, "RGB").convert("HSV")
    hsv = np.asarray(img, dtype=np.float32)

    # Pillow HSV: H in [0,255], S in [0,255], V in [0,255]
    h_idx = np.floor(hsv[..., 0] / 256.0 * _H_BINS).astype(np.int32).clip(0, _H_BINS - 1)
    s_idx = np.floor(hsv[..., 1] / 256.0 * _S_BINS).astype(np.int32).clip(0, _S_BINS - 1)
    v_idx = np.floor(hsv[..., 2] / 256.0 * _V_BINS).astype(np.int32).clip(0, _V_BINS - 1)

    flat_idx = h_idx * (_S_BINS * _V_BINS) + s_idx * _V_BINS + v_idx
    n_bins = _H_BINS * _S_BINS * _V_BINS

    hist = np.bincount(flat_idx.ravel(), minlength=n_bins).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _compute_distances(histograms: list[np.ndarray]) -> np.ndarray:
    """Compute L1 distance between consecutive histograms."""
    arr = np.stack(histograms)  # (N, bins)
    diffs = np.abs(arr[1:] - arr[:-1])
    return diffs.sum(axis=1)    # (N-1,)


def _adaptive_threshold(
    timestamps: list[float],
    distances: np.ndarray,
    multiplier: float,
    min_shot_duration: float,
    start: float,
) -> list[float]:
    """
    Identify cut timestamps using a rolling adaptive threshold.
    Returns list of cut points (including the start).
    """
    cut_times = [start if start > 0.0 else timestamps[0]]
    window = 30  # frames on each side for local stats (~15s at 2fps)

    n = len(distances)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        local = distances[lo:hi]
        mean = local.mean()
        std = local.std()
        threshold = mean + multiplier * std

        if distances[i] > threshold:
            ts = timestamps[i + 1]  # distance[i] is between frame i and i+1
            if ts - cut_times[-1] >= min_shot_duration:
                cut_times.append(ts)

    return cut_times
