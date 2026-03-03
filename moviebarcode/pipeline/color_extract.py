"""
Per-shot dominant color extraction.

Two modes:
  natural — K-means clustering in Lab space; returns the dominant cluster centroid.
  fast    — Trimmed mean in Lab space (no K-means, ~2× faster).

For each shot, sample_frames evenly-spaced timestamps are decoded via seeking,
then pixel samples are collected and processed.

Color extraction is embarrassingly parallel: each shot is independent.
Use concurrent.futures.ProcessPoolExecutor when workers > 1.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import av

from moviebarcode.models import Cluster, Shot
from moviebarcode.pipeline.letterbox import apply_crop
from moviebarcode.utils.av_helpers import seek_and_decode_frame
from moviebarcode.utils.color import lab_to_rgb, rgb_to_lab

# Resolution for color extraction frames (larger than detection = better colors)
_COLOR_SCALE = (480, 270)

# Random seed for reproducibility
_SEED = 42


def extract_shot_colors(
    path: str | Path,
    shots: list[Shot],
    mode: str = "pretty",
    crop: Optional[tuple[float, float]] = None,
    k: int = 3,
    sample_frames: int = 5,
    pixel_budget: int = 3000,
    workers: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[Shot]:
    """
    Fill color_lab, color_rgb, and clusters for each Shot.

    Parameters
    ----------
    path            : video file path
    shots           : list of Shot objects (start/end set, colors None)
    mode            : 'pretty' (K-means) or 'fast' (trimmed mean)
    crop            : (top_frac, bot_frac) letterbox crop, or None
    k               : number of K-means clusters (pretty mode only)
    sample_frames   : number of frames to sample per shot
    pixel_budget    : total pixels to sample per shot across all frames
    workers         : parallel worker processes (1 = sequential)
    progress_callback : called with (completed_count, total_count)

    Returns
    -------
    Same shots list with color fields populated in-place.
    """
    path = str(path)
    total = len(shots)

    args_list = [
        (path, i, shot.start, shot.end, mode, crop, k, sample_frames, pixel_budget)
        for i, shot in enumerate(shots)
    ]

    if workers <= 1:
        for idx, args in enumerate(args_list):
            result = _extract_single_shot(args)
            _apply_result(shots[idx], result)
            if progress_callback:
                progress_callback(idx + 1, total)
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_single_shot, a): a[1] for a in args_list}
            for future in as_completed(futures):
                shot_idx = futures[future]
                try:
                    result = future.result()
                    _apply_result(shots[shot_idx], result)
                except Exception:
                    pass  # Shot keeps color_lab=None; blending will skip it
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

    return shots


# ---------------------------------------------------------------------------
# Worker function — must be module-level to be picklable
# ---------------------------------------------------------------------------

def _extract_single_shot(args: tuple) -> dict:
    """
    Open the video, seek to sample timestamps, extract pixels, run color
    algorithm. Returns a dict with keys: color_lab, color_rgb, clusters.
    """
    (
        path, shot_idx, shot_start, shot_end,
        mode, crop, k, sample_frames, pixel_budget,
    ) = args

    duration = shot_end - shot_start
    if duration <= 0:
        return {}

    # Compute sample timestamps, skipping first/last 10% for long shots
    margin = 0.10 if duration >= 2.0 else 0.0
    t0 = shot_start + margin * duration
    t1 = shot_end   - margin * duration
    if t0 >= t1:
        t0, t1 = shot_start, shot_end

    if sample_frames == 1:
        sample_times = [0.5 * (t0 + t1)]
    else:
        sample_times = list(np.linspace(t0, t1, sample_frames))

    pixels_per_frame = max(1, pixel_budget // sample_frames)

    # Open container once for this shot's frames
    try:
        container = av.open(path, metadata_errors="ignore")
        stream = container.streams.video[0]
    except Exception:
        return {}

    rng = np.random.default_rng(_SEED + shot_idx)
    all_pixels: list[np.ndarray] = []  # list of (M, 3) float64 arrays

    try:
        for t in sample_times:
            _, frame = seek_and_decode_frame(container, stream, t, scale=_COLOR_SCALE)
            if frame is None:
                continue

            if crop is not None:
                frame = apply_crop(frame, crop[0], crop[1])
                if frame.size == 0:
                    continue

            flat = frame.reshape(-1, 3).astype(np.float64)
            if len(flat) > pixels_per_frame:
                idx = rng.choice(len(flat), pixels_per_frame, replace=False)
                flat = flat[idx]
            all_pixels.append(flat)
    finally:
        container.close()

    if not all_pixels:
        return {}

    pixels = np.concatenate(all_pixels, axis=0)  # (N, 3) RGB float64

    if mode == "natural":
        return _kmeans_dominant(pixels, k)
    elif mode == "vivid":
        return _vivid_dominant(pixels, k)
    else:
        return _trimmed_mean(pixels)


def _kmeans_dominant(pixels_rgb: np.ndarray, k: int) -> dict:
    """K-means in Lab space; return dominant cluster color."""
    from sklearn.cluster import MiniBatchKMeans

    pixels_lab = rgb_to_lab(pixels_rgb)

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=1024,
        max_iter=100,
        n_init=3,
        random_state=_SEED,
    )
    labels = kmeans.fit_predict(pixels_lab)

    weights = np.bincount(labels, minlength=k).astype(np.float64)
    weights /= weights.sum()
    centers_lab = kmeans.cluster_centers_  # (k, 3)

    # Sort clusters by weight descending
    order = np.argsort(weights)[::-1]
    weights = weights[order]
    centers_lab = centers_lab[order]

    dominant_lab = centers_lab[0]
    dominant_rgb = lab_to_rgb(dominant_lab)

    clusters = [
        Cluster(
            lab=centers_lab[i].tolist(),
            rgb=lab_to_rgb(centers_lab[i]).tolist(),
            weight=float(weights[i]),
        )
        for i in range(k)
    ]

    return {
        "color_lab": dominant_lab,
        "color_rgb": dominant_rgb,
        "clusters": clusters,
    }


def _vivid_dominant(pixels_rgb: np.ndarray, k: int) -> dict:
    """
    K-means in Lab space; return the most chromatically salient cluster.

    Instead of picking the largest cluster by pixel count (which would return
    black in a dark scene), scores each cluster by perceptual chroma:

        chroma   = sqrt(a² + b²)   — distance from the neutral grey axis in Lab
        salience = chroma * (weight ^ 0.25)

    The small weight exponent gives minority clusters a fair chance while
    preventing single-pixel color artifacts from winning outright.
    This means a small patch of vivid red beats a large field of black.
    """
    from sklearn.cluster import MiniBatchKMeans

    pixels_lab = rgb_to_lab(pixels_rgb)

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=1024,
        max_iter=100,
        n_init=3,
        random_state=_SEED,
    )
    labels = kmeans.fit_predict(pixels_lab)

    weights = np.bincount(labels, minlength=k).astype(np.float64)
    weights /= weights.sum()
    centers_lab = kmeans.cluster_centers_  # (k, 3)

    # Chroma = perceptual saturation (distance from grey axis in Lab)
    chroma = np.sqrt(centers_lab[:, 1] ** 2 + centers_lab[:, 2] ** 2)

    # Salience: strongly favour chromatic clusters, weakly favour larger ones
    salience = chroma * (weights ** 0.25)

    order = np.argsort(salience)[::-1]
    weights = weights[order]
    centers_lab = centers_lab[order]

    dominant_lab = centers_lab[0]
    dominant_rgb = lab_to_rgb(dominant_lab)

    clusters = [
        Cluster(
            lab=centers_lab[i].tolist(),
            rgb=lab_to_rgb(centers_lab[i]).tolist(),
            weight=float(weights[i]),
        )
        for i in range(k)
    ]

    return {
        "color_lab": dominant_lab,
        "color_rgb": dominant_rgb,
        "clusters": clusters,
    }


def _trimmed_mean(pixels_rgb: np.ndarray) -> dict:
    """Trimmed mean in Lab space: trim darkest 10% and brightest 2% by L."""
    pixels_lab = rgb_to_lab(pixels_rgb)
    L = pixels_lab[:, 0]

    low  = np.percentile(L, 10.0)
    high = np.percentile(L, 98.0)
    mask = (L >= low) & (L <= high)

    if mask.sum() == 0:
        mask = np.ones(len(L), dtype=bool)

    mean_lab = pixels_lab[mask].mean(axis=0)
    mean_rgb = lab_to_rgb(mean_lab)

    return {
        "color_lab": mean_lab,
        "color_rgb": mean_rgb,
        "clusters": [],
    }


def _apply_result(shot: Shot, result: dict) -> None:
    if not result:
        return
    shot.color_lab = result.get("color_lab")
    shot.color_rgb = result.get("color_rgb")
    shot.clusters  = result.get("clusters", [])
