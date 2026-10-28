"""Measurements on an aperture: energy, area, average power, ISO beam size (X and Y) and ISO beam size angle (principal axes).

Ports of EnergyButton / AreaButton / PowerButton / SizeXYButton without any GUI, plus the new ISO beam size angle
(principal axes; it replaces Size45Button, whose 45 degree sizes were only a workaround for the lack of an arbitrary angle). ``mask`` is the
boolean aperture mask (``roi.roi_mask``) or None for "whole image". Positions in length units, pixel i at
``i*pixelsize``.

Deliberate differences from MATLAB (see matlab/MEASURE_PROFILES_FIXES.txt): SizeXY fallback centre swapped width and
height and crashed on a centroid outside the image or NaN; the average power divided by 0 for an empty aperture; the peak flux was taken on the whole image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .beam_metrics import PrincipalAxes, compute_beam_peak, compute_centroid, compute_principal_axes
from .profiles import matlab_round
from .roi import POLYGON, beam_diameters


def _cropped(image: np.ndarray, mask) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    return image if mask is None else image * np.asarray(mask, dtype=float)


def aperture_energy(image, mask, pixelsize: float = 1.0) -> float:
    """Sum of the pixels in the aperture (NaN ignored) times the pixel area."""
    return float(np.nansum(_cropped(image, mask)) * pixelsize ** 2)


def aperture_area(shape: tuple[int, int], mask, pixelsize: float = 1.0) -> float:
    """Number of pixels of the aperture (all the image if ``mask`` is None) times the pixel area."""
    n = shape[0] * shape[1] if mask is None else int(np.count_nonzero(mask))
    return n * pixelsize ** 2


def average_power(image, mask, pixelsize: float = 1.0) -> float:
    """Energy / area of the aperture; NaN for an empty aperture (MATLAB: 0/0 or Inf)."""
    area = aperture_area(np.shape(image), mask, pixelsize)
    return math.nan if area == 0 else aperture_energy(image, mask, pixelsize) / area


def roi_center(kind: str, pos) -> tuple[float, float]:
    """Centre of the bounding box of the aperture (``pos`` = [x y w h], or N x 2 vertices for a polygon)."""
    pos = np.asarray(pos, dtype=float)
    if kind == POLYGON:
        v = pos.reshape(-1, 2)
        return float((v[:, 0].min() + v[:, 0].max()) / 2), float((v[:, 1].min() + v[:, 1].max()) / 2)
    x, y, w, h = pos.ravel()[:4]
    return float(x + w / 2), float(y + h / 2)


@dataclass
class SizeResult:
    x0: float
    y0: float
    diam_h: float
    diam_v: float
    energy: float
    gauss_flux: float
    peak_flux: float

    def messages(self) -> list[str]:
        return [f"X0={self.x0:.4g}", f"Y0={self.y0:.4g}", f"DiamH={self.diam_h:.4g}", f"DiamV={self.diam_v:.4g}",
                f"Gauss Flux={self.gauss_flux:.4g}", f"Peak Flux={self.peak_flux:.4g}"]

    def clipboard_text(self) -> str:
        """What MATLAB copied: ``sprintf('%f\\t%f', DiamH, DiamV)``."""
        return f"{self.diam_h:f}\t{self.diam_v:f}"


def _aperture_centroid(cropped: np.ndarray, pixelsize: float, fallback_center: tuple[float, float]):
    """Centroid of the image under the aperture; ``fallback_center`` (centre of the aperture) when it is NaN, outside
    the image or on a null pixel."""
    with np.errstate(divide="ignore", invalid="ignore"):
        xc, yc = compute_centroid(cropped, pixelsize)
    rows, cols = cropped.shape
    ok = math.isfinite(xc) and math.isfinite(yc)
    if ok:
        r, c = int(matlab_round(yc / pixelsize)), int(matlab_round(xc / pixelsize))
        ok = 1 <= r <= rows and 1 <= c <= cols and cropped[r - 1, c - 1] != 0
    return (xc, yc) if ok else fallback_center


def size_xy(image, mask, method: int, pixelsize: float, fallback_center: tuple[float, float],
            peak_image=None) -> SizeResult:
    """``SizeXYButton_Callback``: ISO beam size of the part of ``image`` under ``mask``.

    ``fallback_center``: centre of the aperture, used when the centroid is NaN, outside the image or on a null pixel.
    ``peak_image``: image of the 'Peak Flux'. MATLAB used the whole filtered image ``rfilt``; the peak is now taken
    in the aperture like every other value (default), pass the whole image to get the old behaviour."""
    image = np.asarray(image, dtype=float)
    cropped = _cropped(image, mask)
    energy = aperture_energy(image, mask, pixelsize)
    xc, yc = _aperture_centroid(cropped, pixelsize, fallback_center)
    if method in (1, 2, 3, 4):
        with np.errstate(divide="ignore", invalid="ignore"):          # empty aperture: NaN results, no warning
            d = beam_diameters(cropped, method, pixelsize, centroid=(xc, yc))
        x0, y0, dh, dv = d.x0, d.y0, d.diam_h, d.diam_v
    else:
        x0 = y0 = dh = dv = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        gauss = energy / (2 * math.pi * (dh / 4) * (dv / 4)) if dh and dv else math.inf
    peak = compute_beam_peak(cropped if peak_image is None else np.asarray(peak_image, dtype=float), "smart")
    return SizeResult(float(x0), float(y0), float(dh), float(dv), energy, float(gauss), float(peak))


@dataclass
class AngleResult:
    x0: float
    y0: float
    principal: PrincipalAxes

    def messages(self) -> list[str]:
        p = self.principal
        return [f"X0={self.x0:.4g}", f"Y0={self.y0:.4g}", f"Major={p.d_major:.4g}", f"Minor={p.d_minor:.4g}",
                f"Angle={p.angle:.1f} deg", f"Ellipticity={p.ellipticity:.3f}"]


def size_angle(image, mask, pixelsize: float, fallback_center: tuple[float, float]) -> AngleResult:
    """ISO beam size angle (new): the ISO 11146-1 second-moment diameters along the PRINCIPAL axes of the beam under
    the aperture, the angle of the major axis (counter-clockwise on screen) and the ellipticity. Always the 4 sigma
    (second moment) definition, whatever the diameter menu says. Replaces the 45 degree sizes."""
    cropped = _cropped(image, mask)
    xc, yc = _aperture_centroid(cropped, pixelsize, fallback_center)
    with np.errstate(divide="ignore", invalid="ignore"):
        principal = compute_principal_axes(cropped, xc, yc, pixelsize)
    return AngleResult(float(xc), float(yc), principal)


def ellipse_points(x0: float, y0: float, diam_h: float, diam_v: float, angle_deg: float = 0.0, step: float = 0.01):
    """Points of the ellipse drawn after a size measurement (``theta = 0:0.01:2*pi``); the tilted ellipse of the
    principal axes uses ``angle_deg = principal.angle_xy`` (rotation in the axes, y downwards)."""
    t = np.arange(0.0, 2 * math.pi + 1e-12, step)
    a = math.radians(angle_deg)
    u, v = 0.5 * diam_h * np.cos(t), 0.5 * diam_v * np.sin(t)
    return x0 + u * math.cos(a) - v * math.sin(a), y0 + v * math.cos(a) + u * math.sin(a)
