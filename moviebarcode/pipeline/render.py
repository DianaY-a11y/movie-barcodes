"""
Barcode image rendering.

Builds a numpy array of vertical color stripes and converts to a PIL Image.
Optional Gaussian blur for a smooth gradient look.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from moviebarcode.models import Stripe


def render_barcode(
    stripes: list[Stripe],
    stripe_width: int,
    height: int,
    smooth: int = 0,
) -> Image.Image:
    """
    Render a barcode image from a list of Stripe objects.

    Parameters
    ----------
    stripes      : ordered list of Stripe objects with color_rgb set
    stripe_width : pixel width of each vertical stripe
    height       : pixel height of the output image
    smooth       : Gaussian blur radius (0 = no blur)

    Returns
    -------
    PIL Image (RGB mode).
    """
    n = len(stripes)
    width = n * stripe_width

    # Build pixel array efficiently using numpy broadcasting
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    for i, stripe in enumerate(stripes):
        rgb = np.clip(np.round(stripe.color_rgb), 0, 255).astype(np.uint8)
        x0 = i * stripe_width
        x1 = x0 + stripe_width
        arr[:, x0:x1] = rgb  # broadcast (3,) across (height, stripe_width)

    img = Image.fromarray(arr, "RGB")

    if smooth > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=smooth))

    return img


def render_palette(
    stripes: list[Stripe],
    n_colors: int = 16,
    swatch_width: int = 80,
    swatch_height: int = 80,
) -> Image.Image:
    """
    Render a sorted dominant-color palette from the stripe colors.

    Colors are sorted by hue for a visually appealing layout.

    Parameters
    ----------
    stripes      : list of Stripe objects
    n_colors     : number of colors to show in the palette
    swatch_width : pixel width of each color swatch
    swatch_height: pixel height of each color swatch

    Returns
    -------
    PIL Image (RGB mode).
    """
    from sklearn.cluster import MiniBatchKMeans
    from moviebarcode.utils.color import rgb_to_lab, lab_to_rgb

    # Collect all stripe RGB colors
    colors = np.array([s.color_rgb for s in stripes], dtype=np.float64)

    # Cluster stripe colors down to n_colors
    k = min(n_colors, len(stripes))
    if k < 2:
        centers = colors
    else:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        km.fit(rgb_to_lab(colors))
        centers = lab_to_rgb(km.cluster_centers_)  # (k, 3) uint8

    # Sort by hue using HSV conversion
    centers_sorted = _sort_by_hue(centers)

    # Render swatches
    width = len(centers_sorted) * swatch_width
    arr = np.zeros((swatch_height, width, 3), dtype=np.uint8)
    for i, rgb in enumerate(centers_sorted):
        x0 = i * swatch_width
        arr[:, x0 : x0 + swatch_width] = rgb.astype(np.uint8)

    return Image.fromarray(arr, "RGB")


def _sort_by_hue(colors: np.ndarray) -> np.ndarray:
    """Sort (N, 3) uint8 RGB colors by hue."""
    from PIL import Image as _Image

    hues = []
    for rgb in colors:
        img = _Image.new("RGB", (1, 1), tuple(rgb.astype(int).tolist()))
        h, s, v = img.convert("HSV").getpixel((0, 0))
        hues.append(h)

    order = np.argsort(hues)
    return colors[order]
