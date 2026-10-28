"""Aperture (ROI) and crosshair geometry, without any GUI.

Conventions (same as the rest of pyCBA): pixel i (1-based) is centred at ``i*pixelsize``; the image
fills the axes limits ``[0.5, dim+0.5]*pixelsize``. A rectangle / ellipse position is ``[x, y, w, h]``
(corner + size), a polygon an N x 2 array.

Deliberate differences from the MATLAB code (bugs fixed, see matlab/ROI_CROSSHAIR_FIXES.txt):
auto aperture (lost width update, wrong bound, convergence test), default aperture (asymmetric size, 2
absolute units), diagonal crosshair lines (assumed axes starting at 0), L1L2 crosshair (division by
zero on horizontal / vertical arms, recursive constraints).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .beam_metrics import (BeamDiameters, compute_centroid, compute_d4sigma, compute_d_aperture,
                           compute_d_knife_edge, compute_dpeak, compute_peak_center)

RECT, SQUARE, ELLIPSE, CIRCLE, POLYGON = "rectangle", "square", "ellipse", "circle", "polygon"
ROI_KINDS = (RECT, SQUARE, ELLIPSE, CIRCLE, POLYGON)


def image_limits(shape: tuple[int, int], pixelsize: float):
    """Axes limits ((xmin, xmax), (ymin, ymax)) of an image of shape (rows, cols)."""
    return ((0.5 * pixelsize, (shape[1] + 0.5) * pixelsize), (0.5 * pixelsize, (shape[0] + 0.5) * pixelsize))


# ------------------------------------------------------------------ masks
def _in_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule; points exactly on an edge count as inside (like MATLAB's inpolygon)."""
    inside = np.zeros(x.shape, dtype=bool)
    on_edge = np.zeros(x.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        within = (np.minimum(x1, x2) - 1e-12 <= x) & (x <= np.maximum(x1, x2) + 1e-12) & \
                 (np.minimum(y1, y2) - 1e-12 <= y) & (y <= np.maximum(y1, y2) + 1e-12)
        on_edge |= (np.abs(cross) <= 1e-12 * max(1.0, abs(x2 - x1) + abs(y2 - y1))) & within
        if y1 != y2:
            crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
            inside ^= crosses
    return inside | on_edge


def roi_mask(shape: tuple[int, int], kind: str, pos, pixelsize: float = 1.0) -> np.ndarray:
    """``createMask``: boolean mask of the pixels whose centre is inside the aperture (edges included).
    Pixels exactly on a curved edge may differ by one pixel from MATLAB's poly2mask-based mask."""
    rows, cols = shape
    x = np.arange(1, cols + 1)[None, :] * pixelsize * np.ones((rows, 1))
    y = np.arange(1, rows + 1)[:, None] * pixelsize * np.ones((1, cols))
    pos = np.asarray(pos, dtype=float)
    if kind == POLYGON:
        return _in_polygon(x, y, pos.reshape(-1, 2))
    x0, y0, w, h = pos.ravel()[:4]
    tol = 1e-9 * max(abs(w), abs(h), pixelsize)
    if kind in (RECT, SQUARE):
        return (x >= x0 - tol) & (x <= x0 + w + tol) & (y >= y0 - tol) & (y <= y0 + h + tol)
    if kind in (ELLIPSE, CIRCLE):
        a, b = w / 2.0, h / 2.0
        if a <= 0 or b <= 0:
            return np.zeros(shape, dtype=bool)
        return ((x - (x0 + a)) / a) ** 2 + ((y - (y0 + b)) / b) ** 2 <= 1.0 + 1e-9
    raise ValueError(f"unknown aperture type {kind!r}")


# ------------------------------------------------------------------ default aperture
def default_roi(image: np.ndarray, pixelsize: float = 1.0) -> list[float]:
    """``createDefaultROI``: [x y w h] centred on the peak.

    MATLAB: ``width = max(2, |ncols*pixelsize - Xpeak|/2)``: looked at the right / bottom edge only (a peak near
    the right edge gave the minimum, one near the left edge a huge aperture) and 2 was in absolute units whatever
    the pixel size. Here: half the distance from the peak to the NEAREST edge (identical for a centred peak),
    at least 2 pixels."""
    image = np.asarray(image, dtype=float)
    xpeak, ypeak = compute_peak_center(image, pixelsize)
    (xmin, xmax), (ymin, ymax) = image_limits(image.shape, pixelsize)
    width = max(2 * pixelsize, min(xpeak - xmin, xmax - xpeak) / 2)
    height = max(2 * pixelsize, min(ypeak - ymin, ymax - ypeak) / 2)
    return [xpeak - width / 2, ypeak - height / 2, width, height]


# ------------------------------------------------------------------ auto aperture
def beam_diameters(image: np.ndarray, method: int, pixelsize: float = 1.0, centroid=None) -> BeamDiameters:
    """The 4 diameter methods of the diameter menu (1 = 4 sigma, 2 = 1/e2 of max, 3 = knife edge, 4 = aperture).
    ``centroid`` = (x, y) imposed for methods 1, 3 and 4 (default: centroid of ``image``)."""
    if method == 2:
        return compute_dpeak(image, pixelsize)
    xc, yc = compute_centroid(image, pixelsize) if centroid is None else centroid
    if method == 1:
        return compute_d4sigma(image, xc, yc, pixelsize)
    if method == 3:
        return compute_d_knife_edge(image, xc, yc, pixelsize)
    if method == 4:
        return compute_d_aperture(image, xc, yc, pixelsize)
    raise ValueError(f"unknown diameter method {method}")


MIN_AUTO_APERTURE = 4        # pixels: the auto aperture never shrinks below this width or height (no physical meaning, and
                             # an aperture a few pixels wide cannot be grabbed with the mouse)


def auto_aperture(image: np.ndarray, kind: str, pos, method: int, pixelsize: float = 1.0,
                  tol: float = 0.05, max_iter: int = 10,
                  diameters: Callable[..., BeamDiameters] = beam_diameters):
    """``AutoApertureButton_Callback``: resizes a rectangle / square / ellipse / circle to twice the beam
    diameter, centred where it is. Returns ``(pos, converged, iterations)``.

    Fixed MATLAB bugs: (1) when both the width and the height had to change, the second ``setPosition`` used the
    old position and cancelled the width change; (2) the aperture was bounded by ``ROIX0`` instead of
    ``2*min(distance to the left edge, distance to the right edge)`` and by 0 instead of the image edge, and
    only the left / top edge was considered; (3) the convergence test used the old width only and divided by 0
    for a null diameter; (4) the 'could not converge' message was shown when the last iteration converged;
    (5) polygons crashed (they have no width / height).
    Added: the width and the height never go below ``MIN_AUTO_APERTURE`` (4) pixels. With a background subtracted after
    an aperture that covered the whole image, the diameters computed on the negative noise were ~0 and the aperture
    collapsed to a line."""
    if kind == POLYGON:
        raise ValueError("auto aperture needs a rectangle, square, ellipse or circle")
    image = np.asarray(image, dtype=float)
    (xmin, xmax), (ymin, ymax) = image_limits(image.shape, pixelsize)
    iso = kind in (SQUARE, CIRCLE)
    x0, y0, w, h = (float(v) for v in np.asarray(pos, dtype=float).ravel()[:4])
    cx, cy = x0 + w / 2, y0 + h / 2
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        mask = roi_mask(image.shape, kind, [cx - w / 2, cy - h / 2, w, h], pixelsize)
        d = diameters(np.where(mask, image, 0.0), method, pixelsize)
        # largest aperture centred on (cx, cy) that stays in the image
        maxw = 2 * min(cx - xmin, xmax - cx)
        maxh = 2 * min(cy - ymin, ymax - cy)
        targets = [t for t in (2 * d.diam_h, maxw) if np.isfinite(t)]
        targeth = [t for t in (2 * d.diam_v, maxh) if np.isfinite(t)]
        new_w = max(min(targets), MIN_AUTO_APERTURE * pixelsize) if targets else w
        new_h = max(min(targeth), MIN_AUTO_APERTURE * pixelsize) if targeth else h
        if iso:
            new_w = new_h = min(new_w, new_h)
        ok_w = abs(w - new_w) <= tol * new_w
        ok_h = abs(h - new_h) <= tol * new_h
        if ok_w and ok_h:
            converged = True
            break
        w, h = new_w, new_h
    return [cx - w / 2, cy - h / 2, w, h], converged, it


# ------------------------------------------------------------------ crosshair
def _clip(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def cross_h(y: float, xlim, ylim) -> np.ndarray:
    """``crossHConstraintFcn``: horizontal line across the axes."""
    y = _clip(y, *ylim)
    return np.array([[xlim[0], y], [xlim[1], y]])


def cross_v(x: float, xlim, ylim) -> np.ndarray:
    """``crossVConstraintFcn``: vertical line across the axes."""
    x = _clip(x, *xlim)
    return np.array([[x, ylim[0]], [x, ylim[1]]])


def cross_diag_plus(x: float, y: float, xlim, ylim) -> np.ndarray:
    """``crossDiagPConstraintFcn``: the line x + y = c through (x, y), cut by the axes.
    (MATLAB assumed xlim(1) = ylim(1) = 0.)"""
    c = _clip(x + y, xlim[0] + ylim[0], xlim[1] + ylim[1])
    xa = max(xlim[0], c - ylim[1])
    xb = min(xlim[1], c - ylim[0])
    return np.array([[xa, c - xa], [xb, c - xb]])


def cross_diag_minus(x: float, y: float, xlim, ylim) -> np.ndarray:
    """``crossDiagMConstraintFcn``: the line x - y = d through (x, y), cut by the axes."""
    d = _clip(x - y, xlim[0] - ylim[1], xlim[1] - ylim[0])
    xa = max(xlim[0], ylim[0] + d)
    xb = min(xlim[1], ylim[1] + d)
    return np.array([[xa, xa - d], [xb, xb - d]])


def ray_to_box(center, direction, xlim, ylim) -> np.ndarray:
    """Point where the ray from ``center`` along ``direction`` leaves the axes (horizontal and vertical
    rays included: MATLAB's ``crossLCalculateConstrain`` divided by zero there)."""
    cx, cy = center
    dx, dy = direction
    tx = (xlim[1] - cx) / dx if dx > 0 else (xlim[0] - cx) / dx if dx < 0 else math.inf
    ty = (ylim[1] - cy) / dy if dy > 0 else (ylim[0] - cy) / dy if dy < 0 else math.inf
    t = min(tx, ty)
    return np.array([cx + t * dx, cy + t * dy]) if math.isfinite(t) else np.array([cx, cy])


def cross_l1l2(center, theta: float, xlim, ylim):
    """The '~' crosshair: two perpendicular lines through ``center``; line 1 makes the angle ``theta`` with the
    x axis, line 2 is perpendicular to it. Returns ``(center, line1, line2)``, the lines being 2 x 2 arrays
    ``[[x1, y1], [x2, y2]]`` cut by the axes.

    Replaces the pair of coupled, mutually recursive constraint functions of MATLAB: the state is just
    (center, angle), moving an arm changes ``theta`` (see ``angle_from_mouse``), moving the centre translates both."""
    c = np.array([_clip(center[0], *xlim), _clip(center[1], *ylim)])
    d1 = (math.cos(theta), math.sin(theta))
    d2 = (-d1[1], d1[0])
    line1 = np.array([ray_to_box(c, d1, xlim, ylim), ray_to_box(c, (-d1[0], -d1[1]), xlim, ylim)])
    line2 = np.array([ray_to_box(c, d2, xlim, ylim), ray_to_box(c, (-d2[0], -d2[1]), xlim, ylim)])
    return c, line1, line2


def angle_from_mouse(center, mouse, theta_old: float) -> float:
    """Angle of line 1 when its arm is dragged to ``mouse`` (unchanged if the mouse is on the centre)."""
    dx, dy = mouse[0] - center[0], mouse[1] - center[1]
    return theta_old if dx == 0 and dy == 0 else math.atan2(dy, dx)


def default_crosshair(kind: str, xlim, ylim):
    """Initial crosshair of ``createCrosshair``: '+' -> (hline, vline); 'X' -> (plus-diagonal, minus-diagonal);
    '~' -> (centre, line1, line2). All through the middle of the axes."""
    mx, my = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    if kind == "+":
        return cross_h(my, xlim, ylim), cross_v(mx, xlim, ylim)
    if kind == "X":
        return cross_diag_plus(mx, my, xlim, ylim), cross_diag_minus(mx, my, xlim, ylim)
    if kind == "~":
        return cross_l1l2((mx, my), math.pi / 4, xlim, ylim)
    raise ValueError(f"unknown crosshair {kind!r}")


@dataclass
class Crosshair:
    """State of the crosshair, in length units.

    kind '+': horizontal line at ``y`` and vertical line at ``x``;
    kind 'X': diagonal lines x + y = ``c`` and x - y = ``d``;
    kind '~': centre (``cx``, ``cy``) and angle ``theta`` (radians) of line 1, line 2 being perpendicular;
    kind 'O': no crosshair."""
    kind: str = "+"
    x: float = 0.0
    y: float = 0.0
    c: float = 0.0
    d: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    theta: float = math.pi / 4

    @classmethod
    def default(cls, kind: str, xlim, ylim) -> "Crosshair":
        """Crosshair through the middle of the axes (``createCrosshair`` without positions)."""
        mx, my = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
        return cls(kind, x=mx, y=my, c=mx + my, d=mx - my, cx=mx, cy=my)

    def center(self):
        """``GetProfilesCenter``, geometry only: (Xpos, Ypos) or None (kind 'O')."""
        if self.kind == "+":
            return self.x, self.y
        if self.kind == "X":
            return (self.c + self.d) / 2, (self.c - self.d) / 2
        if self.kind == "~":
            return self.cx, self.cy
        return None

    def clamp(self, xlim, ylim) -> "Crosshair":
        """Brings every element back inside the axes (what the ConstraintFcn did while dragging)."""
        self.x, self.y = _clip(self.x, *xlim), _clip(self.y, *ylim)
        self.c = _clip(self.c, xlim[0] + ylim[0], xlim[1] + ylim[1])
        self.d = _clip(self.d, xlim[0] - ylim[1], xlim[1] - ylim[0])
        self.cx, self.cy = _clip(self.cx, *xlim), _clip(self.cy, *ylim)
        return self

    def scaled(self, factor: float) -> "Crosshair":
        """Same crosshair after a change of pixel size (positions are in length units)."""
        return Crosshair(self.kind, self.x * factor, self.y * factor, self.c * factor, self.d * factor,
                         self.cx * factor, self.cy * factor, self.theta)

    def within(self, xlim, ylim) -> bool:
        tol = 1e-9 * max(abs(xlim[1]), abs(ylim[1]), 1.0)
        pts = {"+": [(self.x, self.y)], "~": [(self.cx, self.cy)],
               "X": [((self.c + self.d) / 2, (self.c - self.d) / 2)]}.get(self.kind, [])
        return all(xlim[0] - tol <= px <= xlim[1] + tol and ylim[0] - tol <= py <= ylim[1] + tol for px, py in pts)

    def l1l2_lines(self, xlim, ylim):
        """End points (2 x 2 arrays) of L1 and L2 for the '~' crosshair."""
        _, l1, l2 = cross_l1l2((self.cx, self.cy), self.theta, xlim, ylim)
        return l1, l2
