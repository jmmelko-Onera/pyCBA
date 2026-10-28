"""Beam metrics: port of the ``Compute*`` functions of MAIN_CameraBeamAnalyzer.m

Faithful port, algorithms unchanged. Conventions kept from MATLAB so that results
are identical:

* positions are ``(1-based pixel index) * pixelsize`` (the first pixel is at
  ``pixelsize``, not 0). Diameters are unaffected by this offset;
* images are indexed ``image[row, col]``; X is the column axis, Y the row axis;
* NaNs are ignored like ``'omitnan'`` / MATLAB ``max``/``min`` do (except where
  noted ``MATLAB quirk``);
* where MATLAB relies on column-major order (``find``, linear indices, ties), the
  same order is reproduced.

``FILTERS.pixelsize`` (a global in MATLAB) is an explicit ``pixelsize`` argument.
"""
from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np


class BeamDiameters(NamedTuple):
    x0: float
    y0: float
    diam_h: float
    diam_v: float


# ----------------------------------------------------------------- helpers
def _energy(image: np.ndarray, pixelsize: float) -> float:
    return np.nansum(image) * pixelsize**2


def _vectors(image: np.ndarray, pixelsize: float):
    """Xvector (columns) and Yvector (lines), 1-based like MATLAB's cumsum(ones)."""
    nrows, ncols = image.shape
    return np.arange(1, ncols + 1) * pixelsize, np.arange(1, nrows + 1) * pixelsize


def _real_sqrt(v: float) -> float:
    """real(sqrt(v)): MATLAB gives 0 (real part of an imaginary root) for v < 0."""
    if v < 0:
        return 0.0
    return float(np.sqrt(v))


def _nanargmax(a: np.ndarray, axis=None):
    """argmax ignoring NaN, first occurrence (MATLAB max semantics)."""
    return np.argmax(np.where(np.isnan(a), -np.inf, a), axis=axis)


def _nanargmin(a: np.ndarray, axis=None):
    return np.argmin(np.where(np.isnan(a), np.inf, a), axis=axis)


def _colon_count(start: float, step: float, stop: float) -> int:
    """Number of elements of MATLAB's start:step:stop (floating-point tolerant)."""
    return int(math.floor((stop - start) / step + 1e-10)) + 1


# --------------------------------------------------------------- centroid
def compute_centroid(image: np.ndarray, pixelsize: float = 1.0) -> tuple[float, float]:
    """ComputeCentroid -> (Xcentroid, Ycentroid)."""
    image = np.asarray(image, dtype=float)
    energy = _energy(image, pixelsize)
    xvec, yvec = _vectors(image, pixelsize)
    xsum = np.nansum(image, axis=0) * pixelsize   # sum along Y
    ysum = np.nansum(image, axis=1) * pixelsize   # sum along X
    xc = (np.nansum(xsum * xvec) * pixelsize) / energy
    yc = (np.nansum(ysum * yvec) * pixelsize) / energy
    return float(xc), float(yc)


def compute_peak_center(image: np.ndarray, pixelsize: float = 1.0) -> tuple[float, float]:
    """ComputePeakCenter -> (Xpeak, Ypeak) position of the maximum pixel."""
    image = np.asarray(image, dtype=float)
    maxpos = _nanargmax(image, axis=0)                       # row of the max of each column
    maxincol = np.where(np.isnan(image), -np.inf, image)[maxpos, np.arange(image.shape[1])]
    colmax = int(np.argmax(maxincol))
    rowmax = int(maxpos[colmax])
    return (colmax + 1) * pixelsize, (rowmax + 1) * pixelsize


# -------------------------------------------------------------- diameters
def compute_d4sigma(image, xcentroid, ycentroid, pixelsize: float = 1.0) -> BeamDiameters:
    """ComputeD4sigma: second-moment diameter, 4*sigma on each axis."""
    image = np.asarray(image, dtype=float)
    energy = _energy(image, pixelsize)
    xvec, yvec = _vectors(image, pixelsize)
    xsum = np.nansum(image, axis=0) * pixelsize
    ysum = np.nansum(image, axis=1) * pixelsize
    xsigma = _real_sqrt((np.nansum(xsum * (xvec - xcentroid) ** 2) * pixelsize) / energy)
    ysigma = _real_sqrt((np.nansum(ysum * (yvec - ycentroid) ** 2) * pixelsize) / energy)
    return BeamDiameters(xcentroid, ycentroid, 4 * xsigma, 4 * ysigma)


class PrincipalAxes(NamedTuple):
    x0: float
    y0: float
    d_major: float          # 4 sqrt(largest eigenvalue of the covariance matrix)
    d_minor: float
    angle: float            # degrees, major axis from the X axis, counter-clockwise AS SEEN ON SCREEN (y axis downwards), in (-90, 90]
    ellipticity: float      # d_minor / d_major (ISO 11146: > 0.87 = "circular")
    d_x: float              # 4 sigma along X and Y: same as compute_d4sigma
    d_y: float

    @property
    def angle_xy(self) -> float:
        """The same angle in the (x, y) coordinates of the image arrays (y downwards): rotation of the major axis
        from +X towards +Y, i.e. clockwise on screen. What ``measure.ellipse_points`` takes."""
        return -self.angle


def compute_principal_axes(image, xcentroid, ycentroid, pixelsize: float = 1.0) -> PrincipalAxes:
    """ISO 11146-1 second-moment diameters along the PRINCIPAL axes of the beam (new: ``ComputeD4sigma`` only gives the
    widths along X and Y, which for a tilted elliptical beam are neither the major nor the minor diameter).

    Covariance matrix of the intensity about ``(xcentroid, ycentroid)``: ``[[sxx, sxy], [sxy, syy]]`` with
    ``sxy = sum(I (x-x0)(y-y0)) / sum(I)``. Eigenvalues ``l1 >= l2``: ``d_major = 4 sqrt(l1)``, ``d_minor = 4 sqrt(l2)``
    (a negative eigenvalue, possible with negative pixels, gives 0 like ``_real_sqrt``). The major axis makes the angle
    ``atan2(2 sxy, sxx - syy) / 2`` with +X in image coordinates; the angle returned is counter-clockwise on screen
    (the image is displayed with y downwards). It is arbitrary (0) for a round beam.
    The second moments are as sensitive to the background as ``ComputeD4sigma``.
    """
    image = np.asarray(image, dtype=float)
    xvec, yvec = _vectors(image, pixelsize)
    w = np.where(np.isfinite(image), image, 0.0)
    total = w.sum()
    dx = (xvec - xcentroid)[None, :]
    dy = (yvec - ycentroid)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        sxx = float((w * dx**2).sum() / total)
        syy = float((w * dy**2).sum() / total)
        sxy = float((w * dx * dy).sum() / total)
    half_sum = 0.5 * (sxx + syy)
    radius = math.hypot(0.5 * (sxx - syy), sxy)
    l1, l2 = half_sum + radius, half_sum - radius
    d_major, d_minor = 4 * _real_sqrt(l1), 4 * _real_sqrt(l2)
    phi = 0.5 * math.atan2(2 * sxy, sxx - syy) if np.isfinite(radius) else math.nan     # (-90, 90] degrees, in xy coords
    angle = -math.degrees(phi)
    if angle <= -90.0:
        angle += 180.0
    ellipticity = d_minor / d_major if d_major > 0 else math.nan
    return PrincipalAxes(xcentroid, ycentroid, d_major, d_minor, angle, ellipticity,
                         4 * _real_sqrt(sxx), 4 * _real_sqrt(syy))


def _width_at_level(profile: np.ndarray, ic: int, level: float) -> float:
    """Width (in pixels) of the peak of a 1-D profile at ``level``, from the peak pixel ``ic`` outwards.

    On each side the profile is followed from the peak until the first pixel that is not above ``level``; the
    crossing is then interpolated linearly between that pixel and the previous one. The width is the distance
    between the two crossings. A side that never goes down to ``level`` (beam cut by the image edge) or that meets a
    NaN pixel is missing: the width is then twice the distance from the peak to the other crossing (the old
    symmetric assumption), and 0 if both are missing (or if the peak is not positive: there is no 1/e^2 level).
    """
    n = len(profile)
    if not profile[ic] > 0:
        return 0.0
    i = ic - 1
    while i >= 0 and profile[i] > level:
        i -= 1
    j = ic + 1
    while j < n and profile[j] > level:
        j += 1
    left = right = math.nan
    if i >= 0 and np.isfinite(profile[i]):
        left = i + (level - profile[i]) / (profile[i + 1] - profile[i])
    if j < n and np.isfinite(profile[j]):
        right = (j - 1) + (profile[j - 1] - level) / (profile[j - 1] - profile[j])
    if math.isnan(left) and math.isnan(right):
        return 0.0
    if math.isnan(left):
        return 2 * (right - ic)
    if math.isnan(right):
        return 2 * (ic - left)
    return right - left


def compute_dpeak(image: np.ndarray, pixelsize: float = 1.0) -> BeamDiameters:
    """ComputeDpeak: 1/e^2 diameter measured on the row and the column through the peak.

    SUB-PIXEL since 23/10/2026 (see ``_width_at_level``). The old version took, on the whole row, the pixel closest to
    the 1/e^2 level and doubled its distance to the peak: an even number of pixels (+-1 pixel of error), and a noisy
    pixel far from the beam could be picked instead of the beam edge."""
    image = np.asarray(image, dtype=float)
    maxpos = _nanargmax(image, axis=0)
    maxincol = np.where(np.isnan(image), -np.inf, image)[maxpos, np.arange(image.shape[1])]
    colmax = int(np.argmax(maxincol))
    themax = maxincol[colmax]
    rowmax = int(maxpos[colmax])
    level = themax / np.exp(2)
    return BeamDiameters(
        (colmax + 1) * pixelsize,
        (rowmax + 1) * pixelsize,
        _width_at_level(image[rowmax, :], colmax, level) * pixelsize,
        _width_at_level(image[:, colmax], rowmax, level) * pixelsize,
    )


# Fractions of the energy at -1 and +1 sigma of a Gaussian: 1/2 (1 -/+ erf(1/sqrt 2)) = 0.158655 and 0.841345
KNIFE_LOW = 0.5 * math.erfc(1 / math.sqrt(2))
KNIFE_HIGH = 0.5 * math.erfc(-1 / math.sqrt(2))


def _knife_edge_axis(sums: np.ndarray, energy: float, pixelsize: float, subpixel: bool = True) -> tuple[float, float]:
    """One axis of ComputeDKnifeEdge: positions of the 15.8655 % and 84.1345 % points.

    ``sums`` are the line (or column) sums; ``cum`` is the cumulative energy up to and including each pixel.

    ``subpixel=True`` (default): the positions are interpolated linearly inside the pixel where the cumulative
    energy crosses the threshold. Pixel ``k`` (1-based) spans ``[k - 0.5, k + 0.5] * pixelsize`` and ``cum`` is the
    energy up to its right edge, so the crossing is at ``(k - 0.5 + (target - cum[k-1]) / (cum[k] - cum[k-1])) * pixelsize``
    (``cum[0] = 0``; the fraction is clamped to [0, 1], it only matters for an image with a negative total energy, where
    the thresholds are negative and the old integer result was already meaningless). The 84 % search starts at the same pixel as the 16 % one (a single pixel may hold both).

    ``subpixel=False``: MATLAB's original result, the position of the CENTRE of the first pixel whose cumulative energy
    exceeds the threshold (integer-pixel resolution: up to +-1.5 pixel of error on a diameter of 20-30 pixels), the
    second search continuing from the next pixel.
    """
    cum = np.cumsum(sums * pixelsize**2)
    low = high = 0.0
    low_target, high_target = energy * KNIFE_LOW, energy * KNIFE_HIGH
    hit = np.nonzero(cum > low_target)[0]
    if hit.size == 0:
        return low, high                                       # loop ended: i = n, no 2nd search
    i = int(hit[0])
    if not subpixel:
        low = (i + 1) * pixelsize
        hit2 = np.nonzero(cum[i + 1:] > high_target)[0]
        if hit2.size:
            high = (i + 1 + int(hit2[0]) + 1) * pixelsize
        return low, high

    def crossing(k: int, target: float) -> float:
        before = cum[k - 1] if k > 0 else 0.0
        frac = (target - before) / (cum[k] - before)          # in [0, 1) for a normal (positive) beam
        return (k + 0.5 + min(max(frac, 0.0), 1.0)) * pixelsize   # clamped: negative energy (garbage image) stays in the pixel

    with np.errstate(divide="ignore", invalid="ignore"):
        low = float(crossing(i, low_target))
        hit2 = np.nonzero(cum[i:] > high_target)[0]
        if hit2.size:
            high = float(crossing(i + int(hit2[0]), high_target))
    return low, high


def compute_d_knife_edge(image, xcentroid, ycentroid, pixelsize: float = 1.0, subpixel: bool = True) -> BeamDiameters:
    """ComputeDKnifeEdge: 2*(x84.13 - x15.87) on each axis, interpolated inside the pixel (``subpixel=False``: MATLAB's
    original integer-pixel positions, see ``_knife_edge_axis``)."""
    image = np.asarray(image, dtype=float)
    energy = _energy(image, pixelsize)
    xlow, xhigh = _knife_edge_axis(np.nansum(image, axis=0), energy, pixelsize, subpixel)
    ylow, yhigh = _knife_edge_axis(np.nansum(image, axis=1), energy, pixelsize, subpixel)
    return BeamDiameters(xcentroid, ycentroid, 2 * (xhigh - xlow), 2 * (yhigh - ylow))


def compute_d_aperture(image, xcentroid, ycentroid, pixelsize: float = 1.0,
                       matlab_compat: bool = False) -> BeamDiameters:
    """ComputeDAperture: diameter of the circle (centred on the centroid) holding 86.47 %
    of the energy, by a false-position iteration (max 100 steps).

    ``matlab_compat=True`` reproduces the original MATLAB quirk: the inner ``sum`` has no
    'omitnan' (and NaN*0 = NaN), so every column containing a NaN/Inf was dropped from the
    enclosed energy. Default (False): NaN pixels are simply ignored.
    """
    image = np.asarray(image, dtype=float)
    x0, y0 = xcentroid, ycentroid
    xvec, yvec = _vectors(image, pixelsize)
    energy = _energy(image, pixelsize)
    nrows, ncols = image.shape
    dist = np.sqrt((yvec[:, None] - y0) ** 2 + (xvec[None, :] - x0) ** 2)

    drmin = pixelsize * np.sqrt(2)
    dfmin = (1 / 1000) * energy
    rb = np.float64(0.5 * max(nrows, ncols) * pixelsize)
    ra = np.float64(1 * pixelsize)

    def enclosed(r) -> np.float64:
        masked = image * (dist < r)
        if matlab_compat:   # sum(sum(image.*mask),'omitnan'): columns with a NaN are dropped
            return np.nansum(np.sum(masked, axis=0)) * pixelsize**2 - energy * 0.8647
        return np.nansum(masked) * pixelsize**2 - energy * 0.8647   # sum(sum(...,'omitnan'),'omitnan')

    k = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(1, 101):
            fa, fb = enclosed(ra), enclosed(rb)
            rc = ra - ((ra - rb) / (fa - fb)) * fa
            if np.sign(fa) == np.sign(fb):
                ra = rc
            else:
                rb = rc
            if abs(fb - fa) <= dfmin:
                break
            if abs(rb - ra) <= drmin:
                break
    if k == 100:  # (also true if it converged exactly at the last step, as in MATLAB)
        warnings.warn("Could not calculate diameter. Consider using a different aperture "
                      "or background substraction.", RuntimeWarning, stacklevel=2)
    return BeamDiameters(x0, y0, float(2 * ra), float(2 * ra))


# ------------------------------------------------------------- beam peak
def _argmax_colmajor(r: np.ndarray) -> tuple[float, int, int]:
    """max(r,[],[1 2],'linear','omitnan') + ind2sub: ties go to the first pixel in
    COLUMN-major order (numpy's default would be row-major)."""
    flat = np.where(np.isnan(r), -np.inf, r).ravel(order="F")
    idx = int(np.argmax(flat))
    row, col = np.unravel_index(idx, r.shape, order="F")
    return float(r[row, col]), int(row), int(col)


def _find_colmajor(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[X,Y] = find(mask) (0-based), listed in column-major order like MATLAB."""
    cols, rows = np.nonzero(mask.T)
    return rows, cols


def _first_enough(r, themax, smooth_start, n_iter, nthreshold):
    """Loop 'for smooth = start:start:end' of ComputeBeamPeak. Like MATLAB, the
    variables of the last iteration are kept if the loop is never broken."""
    X = Y = np.empty(0, dtype=int)
    for i in range(n_iter):
        smooth = smooth_start + i * smooth_start
        X, Y = _find_colmajor(r > themax * (1 - smooth))
        if len(X) >= nthreshold:
            break
    return X, Y


def _valid_points(X, Y, X0, Y0):
    d = np.sqrt((X - X0).astype(float) ** 2 + (Y - Y0).astype(float) ** 2)
    idx = np.argsort(d, kind="stable")
    steps = np.diff(d[idx])
    # MATLAB: idx(steps<=sqrt(2)) with a logical index one element shorter than idx.
    # NB: it does NOT stop at the first gap > sqrt(2); every point followed by a small
    # gap is kept.
    return idx[: len(steps)][steps <= np.sqrt(2)]


HOT_PIXEL_RATIO = 3.0


def _hot_pixels(r: np.ndarray, threshold: float, mdev: float):
    """Hot pixels = pixels that stand >= HOT_PIXEL_RATIO times above their 3rd brightest
    neighbour (of 8) AND more than ``threshold`` MADs above it.

    A beam sampled by pixels cannot do that (even a sub-pixel spot is < 3x its third
    neighbour), whereas an isolated hot pixel, a hot pair or a hot 3-pixel line can.
    A hot group of 4 pixels or more cannot be told from a beam. Returns (mask, values),
    the values being the 3rd-brightest-neighbour levels to put in place of the hot pixels.
    """
    from scipy import ndimage
    domain = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    rf = np.where(np.isnan(r), -np.inf, r)
    # MATLAB: ordfilt2(rf, 6, [1 1 1;1 0 1;1 1 1], 'zeros')  (6th smallest of 8 = 3rd largest)
    kth = ndimage.rank_filter(rf, rank=5, footprint=domain, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore"):
        mask = (rf > HOT_PIXEL_RATIO * kth) & ((rf - kth) > threshold * mdev)
    return mask, kth[mask]


def compute_beam_peak(r: np.ndarray, mode: str | float = "smart", return_info: bool = False):
    """ComputeBeamPeak: beam maximum robust against hot pixels.

    ``mode``: ``'smart'`` (default, threshold 5.8), ``'max'`` (plain maximum) or a number
    (threshold, in MADs, of the hot-pixel contrast test; non-finite = plain maximum).

    The peak is the mean of the pixels within 0.2..2 % of the maximum that lie close to it.
    When too few pixels do (narrow or noisy beam, hot pixel on top), hot pixels are
    removed first (see ``_hot_pixels``) and the search is repeated. The original MATLAB
    code clipped every pixel more than 5.8 MADs from the median at that point, which also
    erased the beam itself and returned a noise-level "peak".

    ``return_info=True`` returns ``(peak, info)`` with ``info = {'themax', 'fallback',
    'n_points', 'nthreshold'}`` (``fallback``: hot-pixel removal was needed).
    """
    threshold = 5.8
    smooth_start = 0.2 / 100
    r = np.array(r, dtype=float)             # copy: MATLAB modifies its local copy only
    nthreshold = max(4, int(np.floor(r.size * 1e-4 + 0.5)))   # round() = half away from 0
    r[np.isinf(r)] = 0
    themax, X0, Y0 = _argmax_colmajor(r)
    if isinstance(mode, str):
        if mode.lower() == "max":
            threshold = np.inf
        elif mode.lower() != "smart":
            raise ValueError(f"unknown mode {mode!r}")
    else:
        threshold = float(mode)
    info = {"themax": themax, "fallback": False, "n_points": 0, "nthreshold": nthreshold}

    def out(peak):
        return (peak, info) if return_info else peak

    if not np.isfinite(threshold):
        return out(themax)

    smooth_end = min(10 * smooth_start, 0.2)
    n1 = _colon_count(smooth_start, smooth_start, smooth_end)
    X, Y = _first_enough(r, themax, smooth_start, n1, nthreshold)
    valid = _valid_points(X, Y, X0, Y0)
    themax_final = themax

    if len(valid) < nthreshold:
        info["fallback"] = True
        med = np.nanmedian(r)
        mdev = np.nanmedian(np.abs(r - med))
        mask, values = _hot_pixels(r, threshold, mdev)
        r[mask] = values
        themax_final, X0, Y0 = _argmax_colmajor(r)
        n2 = _colon_count(smooth_start, smooth_start, 2 * smooth_end)
        X, Y = _first_enough(r, themax_final, smooth_start, n2, nthreshold)
        valid = _valid_points(X, Y, X0, Y0)

    info["n_points"] = int(valid.size)
    if valid.size:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN -> NaN, as MATLAB
            return out(float(np.nanmean(r[X[valid], Y[valid]])))
    return out(themax_final)
