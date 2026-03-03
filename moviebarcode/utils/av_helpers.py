"""
PyAV helpers for video metadata, low-fps frame iteration, and seeking.
"""

from __future__ import annotations

import av
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple
from PIL import Image

from moviebarcode.models import VideoMeta


def open_video_meta(path: str | Path) -> VideoMeta:
    """
    Open a video file and return its metadata without holding the container open.

    Raises
    ------
    av.AVError
        If the file cannot be opened or has no video stream.
    """
    container = av.open(str(path), metadata_errors="ignore")
    try:
        stream = container.streams.video[0]
        duration = (
            float(container.duration / av.time_base)
            if container.duration is not None
            else float(stream.duration * stream.time_base)
            if stream.duration is not None
            else None
        )
        if duration is None:
            raise ValueError(f"Cannot determine duration of {path}")

        fps = float(stream.average_rate) if stream.average_rate else 24.0
        codec = stream.codec_context.name or "unknown"
        return VideoMeta(
            path=str(path),
            duration=duration,
            fps=fps,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            codec=codec,
        )
    finally:
        container.close()


def _resize_frame(arr: np.ndarray, scale: tuple[int, int]) -> np.ndarray:
    """Resize a (H, W, 3) uint8 array to (scale[1], scale[0]) using LANCZOS."""
    img = Image.fromarray(arr)
    img = img.resize(scale, Image.LANCZOS)
    return np.asarray(img)


def iter_frames_at_fps(
    path: str | Path,
    target_fps: float,
    scale: Optional[tuple[int, int]] = None,
    start: float = 0.0,
    end: Optional[float] = None,
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """
    Iterate through a video, yielding (timestamp_seconds, rgb_array) at
    approximately `target_fps` frames per second.

    Decodes every frame but only yields those that fall on the target grid.
    This is simpler and more robust than seeking for continuous streams.

    Parameters
    ----------
    path      : video file path
    target_fps: desired output frame rate
    scale     : (width, height) to resize each frame, or None for native res
    start     : start time in seconds
    end       : end time in seconds (None = full video)
    """
    container = av.open(str(path), metadata_errors="ignore")
    stream = container.streams.video[0]

    time_step = 1.0 / target_fps
    next_target = start

    try:
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            ts = float(frame.pts * stream.time_base)

            if ts < start:
                continue
            if end is not None and ts > end:
                break
            if ts < next_target:
                continue

            arr = frame.to_ndarray(format="rgb24")
            if scale is not None:
                arr = _resize_frame(arr, scale)

            yield ts, arr
            next_target = ts + time_step
    finally:
        container.close()


def seek_and_decode_frame(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    target_seconds: float,
    scale: Optional[tuple[int, int]] = None,
    seek_any: bool = False,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Seek to `target_seconds` in an already-open container and decode one frame.

    After a backward seek, decodes forward until a frame at or after the
    target timestamp is found.

    Parameters
    ----------
    container      : open av.InputContainer (caller manages lifetime)
    stream         : the video stream
    target_seconds : target timestamp in seconds
    scale          : optional (width, height) resize
    seek_any       : if True, allow returning any frame near the target

    Returns
    -------
    (actual_timestamp, rgb_array) or (None, None) on failure.
    """
    target_pts = int(target_seconds / float(stream.time_base))

    try:
        container.seek(target_pts, stream=stream, backward=True, any_frame=seek_any)
    except (av.AVError, Exception):
        return None, None

    max_frames = 300  # Safety cap to avoid infinite decode
    frames_decoded = 0
    last_arr: Optional[np.ndarray] = None
    last_ts: float = 0.0

    try:
        for packet in container.demux(stream):
            if packet.pts is None and packet.dts is None:
                # Flush packet — end of stream
                continue
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                ts = float(frame.pts * stream.time_base)
                arr = frame.to_ndarray(format="rgb24")
                if scale is not None:
                    arr = _resize_frame(arr, scale)

                frames_decoded += 1
                last_arr = arr
                last_ts = ts

                if ts >= target_seconds:
                    return ts, arr

                if frames_decoded >= max_frames:
                    # Return the closest frame we've decoded so far
                    return last_ts, last_arr

            if frames_decoded >= max_frames:
                break
    except (av.AVError, Exception):
        pass

    if last_arr is not None:
        return last_ts, last_arr
    return None, None
