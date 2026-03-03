"""Shared data models."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Cluster:
    lab: list[float]      # [L, a, b]
    rgb: list[int]        # [R, G, B] 0-255
    weight: float         # fraction of pixels in this cluster


@dataclass
class Shot:
    index: int
    start: float          # seconds
    end: float            # seconds
    color_lab: Optional[np.ndarray] = None   # shape (3,)
    color_rgb: Optional[np.ndarray] = None   # shape (3,) uint8
    clusters: list[Cluster] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Stripe:
    index: int
    start: float          # seconds
    end: float            # seconds
    color_lab: np.ndarray  # shape (3,)
    color_rgb: np.ndarray  # shape (3,) uint8
    contributing_shots: list[dict] = field(default_factory=list)
    # contributing_shots: [{"shot_index": int, "overlap": float}]


@dataclass
class VideoMeta:
    path: str
    duration: float       # seconds
    fps: float
    width: int
    height: int
    codec: str
