"""
Shot-weighted Lab blending to produce per-stripe colors.

For each time stripe [i*D, (i+1)*D), every overlapping shot contributes
proportionally to the overlap duration. Blending is done in Lab space for
perceptual smoothness, then converted to RGB for rendering.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from moviebarcode.models import Shot, Stripe
from moviebarcode.utils.color import lab_to_rgb


def compute_stripes(
    shots: list[Shot],
    total_duration: float,
    stripe_duration: float,
    start: float = 0.0,
    end: Optional[float] = None,
) -> list[Stripe]:
    """
    Compute per-stripe colors by shot-weighted Lab blending.

    Parameters
    ----------
    shots            : List of Shot objects with color_lab set.
    total_duration   : Total video duration in seconds.
    stripe_duration  : Seconds per stripe.
    start            : Video start offset (for trimmed videos).
    end              : Video end offset (None = total_duration).

    Returns
    -------
    List of Stripe objects.
    """
    t_end = end if end is not None else total_duration
    t_start = start

    n_stripes = int(np.ceil((t_end - t_start) / stripe_duration))

    # Pre-filter shots that have a color and overlap the processed range
    colored_shots = [s for s in shots if s.color_lab is not None]

    stripes: list[Stripe] = []
    for i in range(n_stripes):
        bin_start = t_start + i * stripe_duration
        bin_end   = min(t_start + (i + 1) * stripe_duration, t_end)

        color_lab, contributing = _blend_bin(colored_shots, bin_start, bin_end, stripe_duration)
        color_rgb = lab_to_rgb(color_lab)

        stripes.append(Stripe(
            index=i,
            start=bin_start,
            end=bin_end,
            color_lab=color_lab,
            color_rgb=color_rgb,
            contributing_shots=contributing,
        ))

    return stripes


def _blend_bin(
    shots: list[Shot],
    bin_start: float,
    bin_end: float,
    stripe_duration: float,
) -> tuple[np.ndarray, list[dict]]:
    """
    Compute the weighted-mean Lab color for one stripe bin.

    Returns (lab_color, contributing_shots_list).
    """
    weighted_sum = np.zeros(3, dtype=np.float64)
    total_weight = 0.0
    contributing: list[dict] = []

    for shot in shots:
        overlap = max(0.0, min(shot.end, bin_end) - max(shot.start, bin_start))
        if overlap <= 0.0:
            continue

        w = overlap / stripe_duration
        weighted_sum += w * shot.color_lab
        total_weight += w
        contributing.append({"shot_index": shot.index, "overlap": round(overlap, 3)})

    if total_weight == 0.0:
        # No shot covers this bin — use neutral grey
        color_lab = np.array([50.0, 0.0, 0.0], dtype=np.float64)
    else:
        color_lab = weighted_sum / total_weight

    return color_lab, contributing
