"""
sRGB ↔ CIE Lab color space conversion.

Implemented manually to avoid a scikit-image dependency.
All conversions use the D65 illuminant.
"""

import numpy as np

# D65 illuminant reference white
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

# sRGB → XYZ (D65) linear transform matrix
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# XYZ → sRGB linear transform matrix (inverse of above)
_XYZ_TO_RGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float64)

_EPSILON = 0.008856   # (6/29)^3
_KAPPA   = 903.3      # (29/6)^3


def _linearize(c: np.ndarray) -> np.ndarray:
    """Remove sRGB gamma. Input/output in [0, 1]."""
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def _gamma(c: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma. Input/output in [0, 1]."""
    return np.where(c > 0.0031308, 1.055 * c ** (1.0 / 2.4) - 0.055, c * 12.92)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _EPSILON, t ** (1.0 / 3.0), (_KAPPA * t + 16.0) / 116.0)


def _f_inv(t: np.ndarray) -> np.ndarray:
    t3 = t ** 3
    return np.where(t3 > _EPSILON, t3, (116.0 * t - 16.0) / _KAPPA)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB to CIE Lab.

    Parameters
    ----------
    rgb : ndarray, shape (..., 3)
        uint8 [0-255] or float [0-1].

    Returns
    -------
    lab : ndarray, shape (..., 3), float64
        L in [0, 100], a/b roughly in [-128, 127].
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    linear = _linearize(rgb)
    xyz = linear @ _RGB_TO_XYZ.T
    xyz_n = xyz / _D65

    f = _f(xyz_n)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIE Lab to sRGB uint8.

    Parameters
    ----------
    lab : ndarray, shape (..., 3), float64

    Returns
    -------
    rgb : ndarray, shape (..., 3), uint8 [0-255]
    """
    lab = np.asarray(lab, dtype=np.float64)

    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0

    xyz = np.stack([
        _D65[0] * _f_inv(fx),
        _D65[1] * _f_inv(fy),
        _D65[2] * _f_inv(fz),
    ], axis=-1)

    linear = xyz @ _XYZ_TO_RGB.T
    linear = np.clip(linear, 0.0, 1.0)
    srgb = _gamma(linear)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)
