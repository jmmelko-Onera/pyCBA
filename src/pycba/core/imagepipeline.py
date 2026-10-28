"""Image pipeline: computation part of ``update_image``, ``plot_image`` (display scaling only),
``isROIWithin``, ``isCrossWithin`` and ``GetProfilesCenter``.

The drawing itself (imagesc, colorbar, ROI/crosshair objects, pixel info) belongs to the GUI
layer. ``IMAGES{n}`` and ``FILTERS`` become the two dataclasses below; image numbers 1..3
(A, B, C) become 0-based indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import warnings

import numpy as np

from .beam_metrics import compute_beam_peak


@dataclass
class ImageData:
    """One entry of IMAGES{n}."""
    raw: np.ndarray | None = None
    filt: np.ndarray | None = None
    deadmask: np.ndarray | float = 1.0       # 0 = dead pixel
    flicker: np.ndarray | float = 0.0        # additive row offset to remove (0 = no deflicker)
    bkg: np.ndarray | float = 0.0
    dimz: float = 0.0                        # full scale ("N-bit scale"), set by the Z scale menu
    znormalize: bool = False                # scale the display on the beam peak
    filepath: str = ""

    @classmethod
    def from_raw(cls, raw: np.ndarray, deflicker: bool = False, roi_mask=None, **kw) -> "ImageData":
        """What load_imagefile builds for a freshly loaded image: all pixels alive and, if
        ``deflicker``, the flicker estimated from the raw image (otherwise ZERO: MATLAB used
        ``ones``, which, subtracted, shifted every image by -1)."""
        raw = np.asarray(raw, dtype=float)
        flicker = compute_flicker(raw, roi_mask) if deflicker else np.zeros(raw.shape)
        return cls(raw=raw, deadmask=np.ones(raw.shape), flicker=flicker, **kw)

    @property
    def dim_y(self) -> int:
        return 0 if self.raw is None else self.raw.shape[0]

    @property
    def dim_x(self) -> int:
        return 0 if self.raw is None else self.raw.shape[1]


@dataclass
class Filters:
    """FILTERS (the parts used here)."""
    pixelsize: float = 1.0
    oldpixelsize: float = 1.0                # pixel size before the last change (ROI rescaling)
    energy: float = 1.0
    filtorder: int | None = None             # [] in MATLAB = no spatial filter
    log: bool = False


# --------------------------------------------------------------------- flicker
def compute_flicker(image: np.ndarray, roi_mask=None) -> np.ndarray:
    """computeFlicker: video flicker = an offset specific to each LINE, estimated as the mean of
    that line over the first and last ``ceil(5 %)`` columns, repeated over the width.

    ``roi_mask`` (rectangle/square/polygon ROI with no crosshair, a decision left to the GUI)
    multiplies the image first. MATLAB's right slice was ``cols-ceil(0.05*cols)-1:cols``, i.e.
    2 columns more than the left one (a typo): both slices have the same width here.
    NaN are not ignored (MATLAB's ``mean``): one NaN in those columns makes the whole line NaN.
    """
    image = np.asarray(image, dtype=float)
    if roi_mask is not None:
        image = image * np.asarray(roi_mask).astype(float)
    cols = image.shape[1]
    k = int(np.ceil(0.05 * cols))
    slices = np.r_[0:k, cols - k:cols]                 # 1:k and cols-k+1:cols (they overlap on tiny images)
    rowmeans = image[:, slices].mean(axis=1)
    return np.repeat(rowmeans[:, None], cols, axis=1)


# ------------------------------------------------------------------ box filter
_PADDING = {"zeros": ("constant", 0.0), "replicate": ("nearest", 0.0),
            "symmetric": ("reflect", 0.0), "circular": ("wrap", 0.0)}


def box_filter(image: np.ndarray, filtorder: int, padding: str = "replicate") -> np.ndarray:
    """``imboxfilt(image, floor(filtorder/2)*2+1)``: mean over a square window of odd size.

    ``padding`` is imboxfilt's 'Padding' option ('zeros' = numeric 0, 'replicate', 'symmetric',
    'circular'). Default 'replicate' = imboxfilt's default (confirmed against MATLAB). Windows touching a NaN/Inf give NaN (a plain running sum would spread the NaN
    over the whole line).
    """
    from scipy import ndimage

    image = np.asarray(image, dtype=float)
    size = int(np.floor(filtorder / 2) * 2 + 1)
    if size == 1:
        return image.copy()
    mode, cval = _PADDING[padding]
    bad = ~np.isfinite(image)
    if not bad.any():
        return ndimage.uniform_filter(image, size=size, mode=mode, cval=cval)
    out = ndimage.uniform_filter(np.where(bad, 0.0, image), size=size, mode=mode, cval=cval)
    touched = ndimage.uniform_filter(bad.astype(float), size=size, mode=mode, cval=0.0) > 1e-9
    out[touched] = np.nan
    return out


# --------------------------------------------------------------- normalisation
def normalize_image(method: int, a: np.ndarray, b: np.ndarray | None = None,
                    c: np.ndarray | None = None) -> np.ndarray:
    """``normalizeImage``: 1 = none, 2 = A-B, 3 = A/B, 4 = fringes (A-B-C)/(4 sqrt(B*C)).

    Division by 0 gives Inf/NaN as in MATLAB (display and metrics must cope, see ``clean_image``);
    for method 4 the pixels where B*C < 0 are set to 0, those where B*C == 0 stay Inf/NaN.
    An empty/missing B or an unknown method returns A with a warning (MATLAB: ``warning``).
    """
    if method == 1:
        return a
    if b is None or np.size(b) == 0:
        warnings.warn("internal error: empty image2")
        return a
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == 2:
            return a - b
        if method == 3:
            return a / b
        if method == 4:
            if c is None or np.size(c) == 0:
                warnings.warn("internal error: empty image3")
                return a
            denom = b * c
            out = (a - b - c) / (4.0 * np.sqrt(np.where(denom < 0, 0.0, denom)))   # real(sqrt(<0)) = 0
            out[denom < 0] = 0.0                                                    # avoids Inf
            return out
    warnings.warn(f"internal error: unknown method number {method}")
    return a


def clean_image(image: np.ndarray) -> np.ndarray:
    """``clean_image``: replaces Inf and NaN by the minimum of the finite pixels.

    MATLAB failed (assignment of an empty value) when no pixel was finite; 0 is used here.
    """
    image = np.array(image, dtype=float)
    finite = np.isfinite(image)
    image[~finite] = image[finite].min() if finite.any() else 0.0
    return image


# ----------------------------------------------------------------- update_image
def update_image(images: list[ImageData], index: int, method: int, filters: Filters,
                 normalizer: Callable | None = normalize_image, padding: str = "replicate") -> list[str]:
    """Recomputes ``images[index].filt`` from ``.raw`` and returns the messages for the text area.

    Steps: flicker subtraction, background subtraction, dead-pixel mask, energy scaling,
    normalisation (main image only; ``method`` 1 = none, 2..4 = A-B, A/B, fringes with C),
    spatial box filter. MATLAB masked the dead pixels BEFORE subtracting the background, which
    left them at ``-bkg`` instead of 0.

    ``normalizer(method, a, b, c)`` is ``normalizeImage`` (``normalize_image``). The size checks use
    the new image (MATLAB compared with the previous ``filt`` of A, which is empty the first time
    and stale when the size changes).
    """
    msgs: list[str] = []
    im = images[index]
    if im.raw is None or np.size(im.raw) == 0:
        return msgs

    rfilt = im.raw - im.flicker
    rfilt = rfilt - im.bkg
    rfilt = rfilt * im.deadmask
    rfilt = rfilt * filters.energy

    if index == 0 and method != 1:
        b, c = images[1].filt, images[2].filt
        dimy1, dimx1 = rfilt.shape
        dimy2, dimx2 = (0, 0) if b is None else b.shape
        dimy3, dimx3 = (0, 0) if c is None else c.shape
        if dimy2 == 0:
            msgs.append("No Ref. img -> skipping.")
        elif dimx1 != dimx2 or dimy1 != dimy2:
            msgs.append("Img A&B size mismatch -> skipping.")
        elif method == 4 and dimy3 == 0:
            msgs.append("No Ref C -> Skipping.")
        elif method == 4 and (dimx3 != dimx1 or dimy3 != dimy1):
            msgs.append("Img A&C size mismatch -> skipping.")
        else:
            rfilt = normalizer(method, rfilt, b, c)

    if filters.filtorder is not None:
        rfilt = box_filter(rfilt, filters.filtorder, padding)
    im.filt = rfilt
    return msgs


# ---------------------------------------------------------------- display scale
@dataclass
class DisplaySpec:
    data: np.ndarray            # what to show (linear, log10 or signed log10)
    vmin: float
    vmax: float
    extent: tuple               # (left, right, bottom, top) for matplotlib, pixel EDGES
    custom_colormap: bool       # the Spiricon palette must be used (colour limits include its padding)
    log: bool = False
    empty: bool = False


def _extent(dim_x: int, dim_y: int, pixelsize: float):
    """Pixel i (1-based) is centred at i*pixelsize, like in every computation of the analysis.

    MATLAB drew with ``imagesc([0 dimX]*pixelsize, [0 dimY]*pixelsize, ...)``, where the two
    values are the CENTRES of the first and last pixel: the image was drawn with a pitch of
    ``dimX/(dimX-1)*pixelsize`` and pixel i at ``(i-1)*pitch``, up to one pixel away from where the
    analysis (centroid, profiles) puts it. Use ``imagesc([1 dimX]*pixelsize, [1 dimY]*pixelsize, ...)``.
    """
    return (0.5 * pixelsize, (dim_x + 0.5) * pixelsize, (dim_y + 0.5) * pixelsize, 0.5 * pixelsize)


def display_scale(rfilt: np.ndarray | None, image_num: int, method: int, znormalize: bool, dimz: float,
                  log: bool, colormap_len: int, pixelsize: float) -> DisplaySpec:
    """Display limits of ``plot_image``. ``image_num`` is 1-based (1 = main image A),
    ``colormap_len`` the number of colours of the custom palette (0 = none)."""
    if rfilt is None or np.size(rfilt) == 0:
        return DisplaySpec(np.empty((0, 0)), 0.0, 100.0,
                           (0.0, 100.0 * pixelsize, 100.0 * pixelsize, 0.0), False, empty=True)
    rfilt = np.asarray(rfilt, dtype=float)
    dim_y, dim_x = rfilt.shape
    custom = colormap_len > 0

    # Zmax: avoids a false saturation effect
    if znormalize:
        if image_num == 1 and method == 3:        # A/B
            zmax = compute_beam_peak(rfilt, "smart")
        elif image_num == 1 and method == 4:      # fringes
            zmax = compute_beam_peak(rfilt, 3.5)
        else:                                     # reference image, A-B or no normalisation
            zmax = compute_beam_peak(rfilt, "max")
    else:
        zmax = dimz
    if not zmax > 0:                              # no positive signal (background too high, empty image, NaN)
        zmax = 1e-9                               # MATLAB only did this in log scale: the linear scale then ran from
                                                  # zmax < 0 to 0 and covered negative values

    extent = _extent(dim_x, dim_y, pixelsize)
    with np.errstate(divide="ignore", invalid="ignore"):
        if log:
            if method == 4 and image_num == 1:
                data = np.sign(rfilt) * np.log10(1 + np.abs(rfilt))
                vmax = np.log10(1 + zmax)
                vmin = -vmax
                if custom:
                    vmax = vmax + 2 * vmax / colormap_len
                    vmin = -vmax
            else:
                data = np.log10(np.maximum(rfilt, 0))
                vmin, vmax = np.log10(zmax / 10000), np.log10(zmax)
                if custom:
                    vmax = vmax + 2 * (vmax - vmin) / colormap_len
            return DisplaySpec(data, float(vmin), float(vmax), extent, custom, log=True)

        if custom:
            rescale = colormap_len / (colormap_len - 1)
            if method == 4 and image_num == 1:    # fringes: negative values
                vmax = zmax * rescale + 1e-9
                vmin = -vmax
            else:
                vmin = -0.5 * zmax / colormap_len
                vmax = zmax * rescale - vmin + 1e-9
        else:
            vmin, vmax = 0.0, zmax + 1e-9
    lo, hi = sorted((float(vmin), float(vmax)))
    return DisplaySpec(rfilt, lo, hi, extent, custom)


# ----------------------------------------------------- ROI / crosshair geometry
def _tol(xlim, ylim) -> float:
    return 1e-9 * max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]), 1.0)


def is_roi_within(xlim, ylim, pos) -> bool:
    """isROIWithin: is the ROI inside the axes? ``pos`` = ``[x y w h]`` (rectangle, ellipse) or an
    N x 2 array of vertices (polygon, freehand). Empty -> True. A tolerance of 1e-9 (relative)
    absorbs the rounding of positions rescaled when the pixel size changes.

    MATLAB used ``length(pos)``, so a polygon of 3 vertices was never "within" and one of 4
    vertices was read as ``[x y w h]``. Here any N x 2 array is a polygon."""
    if pos is None or np.size(pos) == 0:
        return True
    tol = _tol(xlim, ylim)
    pos = np.asarray(pos, dtype=float)
    if pos.ndim == 2 and pos.shape[1] == 2 and pos.shape[0] != 1:
        xs, ys = pos[:, 0], pos[:, 1]
        return bool(np.all(xs >= xlim[0] - tol) and np.all(xs <= xlim[1] + tol)
                    and np.all(ys >= ylim[0] - tol) and np.all(ys <= ylim[1] + tol))
    x, y, w, h = pos.ravel()[:4]
    return bool(x >= xlim[0] - tol and x + w <= xlim[1] + tol and y >= ylim[0] - tol and y + h <= ylim[1] + tol)


def is_cross_within(xlim, ylim, pos) -> bool:
    """isCrossWithin: is a crosshair element inside the axes? ``pos`` = 2 x 2 ``[x1 y1; x2 y2]``,
    ``[x1 y1 x2 y2]`` or a single point ``[x y]``. Empty -> True.

    End points ON the limits count as within (with a 1e-9 relative tolerance): the default
    crosshair lines span the whole axes (``createCrosshair`` puts their ends exactly on XLim/YLim),
    so with MATLAB's strict ``>``/``<`` the crosshair was never "within" and was reset to its
    default position every time the image was redrawn. MATLAB also indexed ``pos(1..4)`` in
    column order, which mixes x and y for a 2 x 2 matrix; all four coordinates are checked here.
    """
    if pos is None or np.size(pos) == 0:
        return True
    tol = _tol(xlim, ylim)
    arr = np.asarray(pos, dtype=float)
    p = arr.reshape(-1)
    if arr.ndim == 2 and arr.shape == (2, 2):         # [x1 y1; x2 y2]
        xs, ys = arr[:, 0], arr[:, 1]
    elif p.size == 2:                                 # a point [x y]
        xs, ys = p[[0]], p[[1]]
    else:                                             # [x1 y1 x2 y2]
        xs, ys = p[[0, 2]], p[[1, 3]]
    return bool(np.all(xs >= xlim[0] - tol) and np.all(xs <= xlim[1] + tol)
                and np.all(ys >= ylim[0] - tol) and np.all(ys <= ylim[1] + tol))


def crosshair_center(kind: str, pos_h=None, pos_v=None, pos_l1l2=None):
    """GetProfilesCenter, geometry only: (Xpos, Ypos) of the crosshair, or None.

    ``kind``: '+' (horizontal line + vertical line), 'X' (two diagonals) or '~' (L1/L2 point).
    ``pos_h``/``pos_v`` are the 2 x 2 ``[x1 y1; x2 y2]`` end points of the first / second line
    (handlers HPL1 / VML2), ``pos_l1l2`` the ``[x y]`` of the L1/L2 centre point.
    For 'X', the first line must be the anti-diagonal (y + x constant) and the second the
    diagonal (y - x constant): the formulas are their intersection. When None is returned the
    caller falls back on the centroid ("No crosshair defined.").
    """
    if kind == "+":
        h, v = np.asarray(pos_h, dtype=float), np.asarray(pos_v, dtype=float)
        return float(v[1, 0]), float(h[0, 1])
    if kind == "X":
        p, m = np.asarray(pos_h, dtype=float), np.asarray(pos_v, dtype=float)
        ypos = (p[0, 1] + m[0, 1] + p[0, 0] - m[0, 0]) / 2
        xpos = (p[0, 1] - m[0, 1] + p[0, 0] + m[0, 0]) / 2
        return float(xpos), float(ypos)
    if kind == "~" and pos_l1l2 is not None:
        p0 = np.asarray(pos_l1l2, dtype=float).ravel()
        return float(p0[0]), float(p0[1])
    return None
