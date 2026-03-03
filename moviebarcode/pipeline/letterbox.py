"""
Letterbox (black bar) detection.

Samples N frames evenly across the video, computes per-row luminance,
and finds the stable crop rectangle via median across samples.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from moviebarcode.utils.av_helpers import iter_frames_at_fps


# Minimum fraction of the frame height a black bar must be to bother cropping.
_MIN_BAR_FRACTION = 0.03


def detect_letterbox(
    path: str | Path,
    n_samples: int = 8,
    threshold: float = 10 / 255,
    smoothing_window: int = 3,
) -> Tuple[float, float]:
    """
    Detect top and bottom black bars in the video.

    Parameters
    ----------
    path             : video file path
    n_samples        : number of frames to sample
    threshold        : per-row mean luminance above which content is present
    smoothing_window : moving-average window for row luminance (reduces noise)

    Returns
    -------
    (top_frac, bot_frac) : fractions of frame height to keep.
        top_frac=0.125 means the top 12.5% is black bars.
        Returns (0.0, 1.0) if no meaningful letterbox is detected.
    """
    # Sample frames at very low fps from 10%-90% of the video to avoid
    # opening/credits which are often all-black.
    # We use a low target fps and take the first n_samples frames we get.
    frames: list[np.ndarray] = []

    for _ts, frame in iter_frames_at_fps(path, target_fps=0.02, scale=(256, 144)):
        frames.append(frame)
        if len(frames) >= n_samples:
            break

    if not frames:
        return 0.0, 1.0

    top_fracs: list[float] = []
    bot_fracs: list[float] = []

    for frame in frames:
        top, bot = _find_bars(frame, threshold, smoothing_window)
        top_fracs.append(top)
        bot_fracs.append(bot)

    top = float(np.median(top_fracs))
    bot = float(np.median(bot_fracs))

    # Only apply crop if the detected bars are meaningful in size
    has_top_bar = top > _MIN_BAR_FRACTION
    has_bot_bar = bot < (1.0 - _MIN_BAR_FRACTION)

    if not has_top_bar and not has_bot_bar:
        return 0.0, 1.0

    return top, bot


def _find_bars(
    frame: np.ndarray,
    threshold: float,
    smoothing_window: int,
) -> Tuple[float, float]:
    """
    Given a single RGB frame, return (top_frac, bot_frac) where content starts/ends.
    """
    # BT.709 luminance
    lum = frame @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    lum = lum / 255.0  # shape: (H, W)

    row_means = lum.mean(axis=1)  # shape: (H,)

    # Smooth to reduce compression-artifact noise
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        row_means = np.convolve(row_means, kernel, mode="same")

    h = len(row_means)

    # Top bar: find first row from top where luminance > threshold
    top_frac = 0.0
    for i, v in enumerate(row_means):
        if v > threshold:
            top_frac = i / h
            break

    # Bottom bar: find first row from bottom where luminance > threshold
    bot_frac = 1.0
    for i, v in enumerate(reversed(row_means)):
        if v > threshold:
            bot_frac = 1.0 - i / h
            break

    return top_frac, bot_frac


def apply_crop(frame: np.ndarray, top_frac: float, bot_frac: float) -> np.ndarray:
    """
    Crop a frame array given fractional row bounds.

    Parameters
    ----------
    frame    : (H, W, 3) uint8
    top_frac : fraction of H to start from (e.g. 0.125 crops top 12.5%)
    bot_frac : fraction of H to end at    (e.g. 0.875 crops bottom 12.5%)

    Returns
    -------
    Cropped frame, same dtype.
    """
    h = frame.shape[0]
    y0 = int(round(top_frac * h))
    y1 = int(round(bot_frac * h))
    y0 = max(0, y0)
    y1 = min(h, y1)
    if y1 <= y0:
        return frame
    return frame[y0:y1]
