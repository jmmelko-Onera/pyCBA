"""Gaussian fits of 1D beam profiles: port of ``gaussianfit``, ``PlotGaussianFit``
(computation only, no plotting) and Utils/myGaussFit.m.

Two models, like in MATLAB:

* ``method='gauss1'`` (Curve Fitting Toolbox branch): ``a1*exp(-((x-b1)/c1)^2) + d1``
  fitted by least squares. The baseline ``d1`` is new: MATLAB's built-in ``gauss1`` has
  none, which biases the width of profiles sitting on a background.
  ``waist = sqrt(2)*|c1|`` is the 1/e^2 radius.
* ``method='legacy'`` (the branch used when the toolbox is absent): ``my_gauss_fit``,
  a Gauss-Newton fit of a normalised Gaussian with a baseline taken from the profile
  edges. ``waist = 2*sigma`` (same quantity: the 1/e^2 radius).

The diameter shown in the GUI is ``2*waist`` (= 4 sigma).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GaussianFit:
    xfit: np.ndarray = field(default_factory=lambda: np.empty(0))   # empty if the fit failed
    yfit: np.ndarray = field(default_factory=lambda: np.empty(0))
    waist: float = 0.0                                              # 1/e^2 radius, 0 = failed
    params: tuple | None = None   # gauss1: (a1, b1, c1, d1); legacy: (sigma, mu, amp, y0)

    @property
    def diameter(self) -> float:
        return 2 * self.waist


@dataclass
class ProfileFits:
    h: GaussianFit
    v: GaussianFit
    messages: list[str]


# ---------------------------------------------------------------- myGaussFit
def _pdf(x, sigma, mu):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def my_gauss_fit(x, y, sigma0=None, mu0=None, nmax: int = 50):
    """Utils/myGaussFit.m (corrected version) -> (sigma, mu, s, y0).

    Fits ``y - y0`` by ``s * pdf(x; mu, sigma)`` (normalised Gaussian). ``y0`` is the mean
    of the 5 first/last points, ``s`` the trapezoid area of ``y - y0``; the data are always
    divided by ``s`` so that the returned ``s`` is the true amplitude of the fit.
    Gauss-Newton with step halving: a step is only taken if it keeps ``sigma > 0`` and does
    not increase the residual, so the iteration cannot diverge. Returns zeros if x and y
    differ in length or are empty, and sigma = mu = 0 if the area is null.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    n = x.size
    order = np.lexsort((y, x))                       # sortrows([x y])
    x, y = x[order], y[order]

    y0 = (y[:5].mean() + y[-5:].mean()) / 2 if n > 13 else (y[0] + y[-1]) / 2
    y = y - y0
    s = float(np.sum(np.diff(x) * 0.5 * (y[:-1] + y[1:])))
    if s == 0 or not np.isfinite(s):
        return 0.0, 0.0, s, float(y0)
    y = y / s

    with np.errstate(all="ignore"):
        index = int(np.argmax(np.where(np.isnan(y), -np.inf, y)))
        mu = x[index]
        sigma = 1 / (np.sqrt(2 * np.pi) * y[index])
        if sigma0 is not None:
            sigma = sigma0
        if mu0 is not None:
            mu = mu0

        k = np.sqrt(2 * np.pi)
        for _ in range(nmax):
            e = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
            dfdsigma = -1 / (k * sigma**2) * e + 1 / (k * sigma) * e * ((x - mu) ** 2 / sigma**3)
            dfdmu = 1 / (k * sigma) * e * (x - mu) / (sigma**2)
            F = np.column_stack([dfdsigma, dfdmu])
            res = y - _pdf(x, sigma, mu)
            r0 = np.sum(res**2)
            try:
                step = np.linalg.solve(F.T @ F, F.T @ res)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)):
                break
            t, accepted = 1.0, False
            for _ in range(10):                      # step halving
                sn, mn = sigma + t * step[0], mu + t * step[1]
                if sn > 0 and np.sum((y - _pdf(x, sn, mn)) ** 2) <= r0:
                    accepted = True
                    break
                t /= 2
            if not accepted:
                break
            done = abs(sn - sigma) <= 1e-12 * sigma and abs(mn - mu) <= 1e-12 * sigma
            sigma, mu = sn, mn
            if done:
                break
    return float(sigma), float(mu), s, float(y0)


# ------------------------------------------------------------------ gauss1
def _gauss1(x, a, b, c, d):
    """a*exp(-((x-b)/c)^2) + d : MATLAB's gauss1 plus a free baseline d."""
    return a * np.exp(-(((x - b) / c) ** 2)) + d


def _gauss1_start(x, y):
    """Start point [a b c d]; the same recipe is used in the MATLAB version."""
    d = np.sort(y)[:5].mean()                        # baseline: mean of the 5 lowest values
    i = int(np.argmax(y))
    a, b = y[i] - d, x[i]
    above = np.nonzero(y >= d + a / 2)[0]
    c = abs(x[above[-1]] - x[above[0]]) / (2 * np.sqrt(np.log(2)))     # FWHM -> c
    if not np.isfinite(c) or c <= 0:
        c = abs(x[-1] - x[0]) / 4 or 1.0
    return a, b, c, d


def _kept(linemask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean array of the points used by the fit.

    MATLAB ``'Exclude',[indices(linemask==0) 1 2 n-1 n]``: the points outside the ROI
    and the first two and last two points of the profile are not used.
    """
    n = x.size
    keep = (np.asarray(linemask) != 0) & np.isfinite(x) & np.isfinite(y)
    for i in (1, 2, n - 1, n):                       # 1-based MATLAB indices
        if 1 <= i <= n:
            keep[i - 1] = False
    return keep


TIGHT_SIGMAS = 2.5     # fit window narrower than +-2.5 sigma: the baseline d is not determined by the data
PRIOR_POINTS = 3       # strength of the "baseline ~ 0" prior, in equivalent data points


def _fit_gauss1(x, y, linemask):
    """Gaussian + baseline fit, in two steps when the window is tight.

    1. Plain least squares on ``a*exp(-((x-b)/c)^2) + d`` (what MATLAB's ``fit`` does).
    2. If the kept points span less than +-``TIGHT_SIGMAS`` sigma around the peak (tight aperture), the baseline ``d``
       and the width ``c`` cannot be told apart by the data: step 1 is then very noisy. The fit is redone with a soft
       prior ``d ~ 0`` (the image background is subtracted: Bkg field), worth ``PRIOR_POINTS`` extra data points. It is
       a weight on one virtual point, far from the beam, where the model is exactly ``d`` (a ``'Weights'`` vector in
       MATLAB's ``fit``). The core is not touched; wide profiles go through step 1 only, unchanged.
    """
    from scipy.optimize import OptimizeWarning, curve_fit

    keep = _kept(linemask, x, y)
    xk, yk = x[keep], y[keep]
    if xk.size < 4:                                  # 4 coefficients
        raise RuntimeError("not enough points left after exclusion")
    a0, b0, c0, d0 = _gauss1_start(xk, yk)
    half_width = 0.5 * (xk.max() - xk.min())
    tight = half_width < TIGHT_SIGMAS * c0 / np.sqrt(2)       # c0/sqrt(2) = sigma (FWHM recipe, robust)
    with warnings.catch_warnings():                  # the covariance is not used
        warnings.simplefilter("ignore", OptimizeWarning)
        if not tight:
            p, _ = curve_fit(_gauss1, xk, yk, p0=(a0, b0, c0, d0), maxfev=20000)
        else:
            xf = np.append(xk, b0 + 10 * c0)         # virtual point: gaussian = 0 there, so the model equals d
            yf = np.append(yk, 0.0)
            sf = np.append(np.ones(xk.size), 1 / np.sqrt(PRIOR_POINTS))
            p, _ = curve_fit(_gauss1, xf, yf, p0=(a0, b0, c0, d0), sigma=sf, maxfev=20000)
    return float(p[0]), float(p[1]), float(abs(p[2])), float(p[3])


# ------------------------------------------------------------- gaussianfit
def gaussian_fit(x, y, linemask=None, method: str = "gauss1") -> GaussianFit:
    """Port of ``gaussianfit(x, y, linemask)``.

    ``linemask`` (1 = use the point, e.g. inside the ROI). On failure an empty
    ``GaussianFit`` (``waist = 0``) is returned with a warning, where MATLAB's ``fit``
    would raise an error.
    """
    if method not in ("gauss1", "legacy"):
        raise ValueError(f"unknown method {method!r}")
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    linemask = np.ones(x.size, dtype=bool) if linemask is None else np.asarray(linemask).ravel()
    res = GaussianFit()

    if method == "gauss1":
        try:
            a1, b1, c1, d1 = _fit_gauss1(x, y, linemask)
        except (RuntimeError, ValueError, TypeError) as exc:
            warnings.warn(f"Gaussian fit failed: {exc}", RuntimeWarning, stacklevel=2)
            return res
        res.waist = np.sqrt(2) * c1
        res.params = (a1, b1, c1, d1)
        f = lambda t: _gauss1(t, a1, b1, c1, d1)     # noqa: E731
    else:
        m = linemask == 1
        sigma, mu, amp, y0 = my_gauss_fit(x[m], y[m])
        res.waist = 2 * sigma
        res.params = (sigma, mu, amp, y0)
        with np.errstate(all="ignore"):
            f = lambda t: y0 + (amp / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((t - mu) ** 2) / (2 * sigma**2))  # noqa: E731

    if res.waist != 0 and x.size > 1:
        res.xfit = np.linspace(x[0], x[-1], 10 * (x.size - 1) + 1)   # 0.1 * the sampling step
        res.yfit = f(res.xfit)
    return res


def fit_profiles(hvector, hprofile, hmask, hname, vvector, vprofile, vmask, vname,
                 method: str = "gauss1") -> ProfileFits:
    """Computation part of ``PlotGaussianFit``: fits both profiles and builds the
    messages shown in the GUI. Plotting is left to the GUI layer."""
    messages = []
    out = []
    for vec, prof, mask, name, sign in ((hvector, hprofile, hmask, hname, "+"),
                                        (vvector, vprofile, vmask, vname, "-")):
        mask = np.asarray(mask)
        if mask.sum() < mask.size * 1 / 100:
            warnings.warn(f"{sign} profile does not cross ROI", RuntimeWarning, stacklevel=2)
        fit = gaussian_fit(vec, prof, mask, method)
        messages.append(f"{name} diam.={2 * fit.waist:.4g}")
        out.append(fit)
    return ProfileFits(out[0], out[1], messages)


# =====================================================================================
# Profile extraction: computation part of ProfileButton_Callback, Profile45Button_Callback,
# ProfileL1L2Button_Callback, plus smoothOutliers and ComputeProfilesMinMax.
# Figures, file names, messages boxes and crosshair handling belong to the GUI layer.
# =====================================================================================
@dataclass
class Profile:
    vector: np.ndarray     # position along the profile (length units)
    values: np.ndarray
    mask: np.ndarray       # 1 = valid point (inside the ROI), 0 = ignored


@dataclass
class ProfilePair:
    first: Profile
    second: Profile
    info: dict = field(default_factory=dict)   # linenum/colnum, angle ...


def matlab_round(x: float) -> float:
    """MATLAB round(): halves go away from zero (Python's round() goes to even)."""
    return float(np.sign(x) * np.floor(abs(x) + 0.5))


def _cropped(image, roi_mask):
    """``croppedimg = rfilt.*mask`` (NaN/Inf outside the ROI stay NaN) and the mask."""
    image = np.asarray(image, dtype=float)
    if roi_mask is None:
        return image, np.ones(image.shape)
    mask = np.asarray(roi_mask).astype(float)
    return image * mask, mask


def extract_hv_profiles(image, xpos: float, ypos: float, pixelsize: float = 1.0,
                        roi_mask=None) -> ProfilePair:
    """ProfileButton_Callback: horizontal cut at row ``round(ypos/pixelsize)`` and vertical
    cut at column ``round(xpos/pixelsize)`` (1-based pixels, positions in length units).
    ``first`` = H profile, ``second`` = V profile. Raises IndexError if the crosshair is
    outside the image (MATLAB would fail with an index error; Python would wrap around)."""
    img, mask = _cropped(image, roi_mask)
    nrows, ncols = img.shape
    linenum = int(matlab_round(ypos / pixelsize))
    colnum = int(matlab_round(xpos / pixelsize))
    if not (1 <= linenum <= nrows and 1 <= colnum <= ncols):
        raise IndexError(f"profile centre (line {linenum}, column {colnum}) is outside the "
                         f"{nrows}x{ncols} image")
    h = Profile(np.arange(1, ncols + 1) * pixelsize, img[linenum - 1, :].copy(), mask[linenum - 1, :].copy())
    v = Profile(np.arange(1, nrows + 1) * pixelsize, img[:, colnum - 1].copy(), mask[:, colnum - 1].copy())
    return ProfilePair(h, v, {"linenum": linenum, "colnum": colnum})


def spdiags_column(A: np.ndarray, d: int) -> np.ndarray:
    """MATLAB ``spdiags(A, d)`` for ONE diagonal, as a vector of length ``min(m, n)``.

    The diagonal is placed in the vector according to its COLUMN index when A is tall
    (m >= n) and its ROW index when A is wide (m < n); the remaining entries are zeros.
    """
    A = np.asarray(A)
    m, n = A.shape
    out = np.zeros(min(m, n), dtype=A.dtype if A.dtype.kind in "fc" else float)
    if m >= n:
        i = np.arange(max(1, 1 + d), min(n, m + d) + 1)
    else:
        i = np.arange(max(1, 1 - d), min(m, n - d) + 1)
    diag = np.diagonal(A, offset=d)
    if i.size:
        out[i - 1] = diag
    return out


def extract_diagonal_profiles(image, xpos: float, ypos: float, pixelsize: float = 1.0,
                              roi_mask=None) -> ProfilePair:
    """Profile45Button_Callback: the two diagonals through the crosshair.
    ``first`` = 'minus' ('\\' diagonal), ``second`` = 'plus' ('/' diagonal).

    Like MATLAB's spdiags, each profile has ``min(nrows, ncols)`` points; the part of
    the vector not covered by the diagonal is 0 and its mask is 0 (so it is ignored by
    the fits). The position of a point is ``index*pixelsize*sqrt(2)``.
    """
    img, mask = _cropped(image, roi_mask)
    nrows = img.shape[0]
    linenum = int(matlab_round(ypos / pixelsize))
    colnum = int(matlab_round(xpos / pixelsize))
    d_minus = colnum - linenum
    d_plus = colnum - (nrows - linenum) - 1
    n = min(img.shape)
    vec = np.arange(1, n + 1) * pixelsize * np.sqrt(2)
    minus = Profile(vec.copy(), spdiags_column(img, d_minus), spdiags_column(mask, d_minus))
    plus = Profile(vec.copy(), spdiags_column(np.flipud(img), d_plus), spdiags_column(np.flipud(mask), d_plus))
    return ProfilePair(minus, plus, {"linenum": linenum, "colnum": colnum})


def _line_profile(img: np.ndarray, roi: np.ndarray | None, pos, pixelsize: float) -> Profile:
    """Profile of ``img`` along the segment ``pos = [[x1, y1], [x2, y2]]`` (length units; the
    pixel (row, col), 1-based, is at (col*pixelsize, row*pixelsize)).

    The image is sampled every ``pixelsize`` along the segment by bilinear interpolation, so
    the positions are the distance from the first end point. ``mask`` is 0 for the samples that
    fall outside the image or the ROI, or that are not finite (their value is set to 0).
    """
    from scipy import ndimage

    nrows, ncols = img.shape
    pos = np.asarray(pos, dtype=float).reshape(2, 2)
    p1, p2 = pos[0] / pixelsize, pos[1] / pixelsize           # [col, row], 1-based pixels
    length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    n = int(np.floor(length + 1e-9)) + 1
    t = np.arange(n, dtype=float)
    if length > 0:
        cols = p1[0] + t * (p2[0] - p1[0]) / length
        rows = p1[1] + t * (p2[1] - p1[1]) / length
    else:
        cols, rows = np.full(1, p1[0]), np.full(1, p1[1])
    c0, r0 = cols - 1, rows - 1                               # 0-based array indices
    eps = 1e-9
    inside = (c0 >= -eps) & (c0 <= ncols - 1 + eps) & (r0 >= -eps) & (r0 <= nrows - 1 + eps)
    vals = ndimage.map_coordinates(img, [r0, c0], order=1, mode="nearest")
    bad = ~inside | ~np.isfinite(vals)
    vals = np.where(bad, 0.0, vals)
    good = ~bad
    if roi is not None:
        good &= ndimage.map_coordinates(np.asarray(roi, dtype=float), [r0, c0], order=0, mode="nearest") > 0.5
    return Profile(t * pixelsize, vals, good.astype(float))


def extract_l1l2_profiles(image, pos_l1, pos_l2, roi_mask=None, pixelsize: float = 1.0) -> ProfilePair:
    """ProfileL1L2Button_Callback: profiles along the two lines L1 and L2 of the crosshair.

    ``pos_l1`` / ``pos_l2`` are the 2x2 ``[x1 y1; x2 y2]`` end points in length units.

    MATLAB rotated the image so that L1 became horizontal, rotated the two line masks with
    ``imrotate`` and summed the image under them. The rotated 1-pixel masks come out deformed
    and with holes, which made the profiles irregular (-19 % on the width of the narrow axis
    at some angles). The lines being known exactly, the image is sampled directly along them
    here (bilinear, one sample per pixel): error on the beam widths < 0.25 % at any angle,
    no rotation of the image, no mask, and L1 and L2 need not be perpendicular.
    ``first`` = L1, ``second`` = L2; ``info['angle']`` is the angle of L1 in degrees.
    """
    img, _ = _cropped(image, roi_mask)
    roi = None if roi_mask is None else np.asarray(roi_mask).astype(float)
    pos_l1 = np.asarray(pos_l1, dtype=float).reshape(2, 2)
    angle = np.arctan2(pos_l1[1, 1] - pos_l1[0, 1], pos_l1[1, 0] - pos_l1[0, 0]) * 180 / np.pi
    return ProfilePair(_line_profile(img, roi, pos_l1, pixelsize),
                       _line_profile(img, roi, pos_l2, pixelsize),
                       {"angle": float(angle)})


# ------------------------------------------------------------ smoothOutliers & min/max
def smooth_outliers(y, threshold: float = 5.0) -> np.ndarray:
    """Utils/smoothOutliers.m: values farther than ``threshold`` MADs from the median are
    replaced by the median (non-finite values are left alone and ignored by the statistics).

    The MATLAB version computed the outlier test on the FINITE values only and applied it to
    the whole vector, which mis-aligns the replaced elements as soon as one value is NaN/Inf.
    Here the test is applied to the right elements.
    """
    y = np.array(y, dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        return y
    med = np.median(y[finite])
    d = np.abs(y - med)
    mdev = np.median(d[finite])
    if mdev != 0:
        with np.errstate(invalid="ignore"):
            y[finite & (d / mdev > threshold)] = med
    return y


def _clean_profile(p: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Outlier removal used before taking the min/max of a profile.

    High side: hot points only, i.e. points >= 3x their lesser neighbour AND more than
    ``threshold`` MADs above it (the same idea as the beam-peak fallback, in 1D). Isolated
    hot points and hot pairs are caught. The MATLAB code replaced EVERY point more than 5 MADs
    above the median, which erases the beam itself (the reported maximum was 6 to 22 % of the
    true one unless the beam filled most of the profile).
    Low side: unchanged (points more than ``threshold`` MADs below the median -> median).
    """
    p = np.array(p, dtype=float)
    fin = np.isfinite(p)
    if fin.sum() == 0:
        return p
    med = np.median(p[fin])
    mdev = np.median(np.abs(p[fin] - med))
    if p.size >= 3:
        left = np.empty_like(p)
        right = np.empty_like(p)
        left[1:], left[0] = p[:-1], p[1]                 # end points: their only neighbour
        right[:-1], right[-1] = p[1:], p[-2]
        k = np.fmin(left, right)                         # lesser neighbour (NaN ignored)
        with np.errstate(invalid="ignore"):
            hot = fin & (p > 3 * k) & ((p - k) > threshold * mdev)
        p[hot] = k[hot]
    if mdev != 0:
        with np.errstate(invalid="ignore"):
            p[fin & (med - p > threshold * mdev)] = med
    return p


def _min_max(profile, mask, smooth: int = 2, matlab_compat: bool = False) -> tuple[float, float]:
    p = np.array(profile, dtype=float)[np.asarray(mask) == 1]
    p[np.isinf(p)] = 0
    p = np.sort(smooth_outliers(p, 5) if matlab_compat else _clean_profile(p, 5))
    n = p.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # empty -> NaN, like MATLAB
        lo = np.nanmean(p[: min(smooth, n)])
        # MATLAB's Hprofile2(max(nH-smooth,1):end) took smooth+1 = 3 points for the maximum while the
        # minimum takes 2: an off-by-one, corrected (same number of points on both sides).
        # (matlab_compat=True keeps the literal 3 points, to reproduce the original code)
        hi = np.nanmean(p[max(n - smooth, 1) - 1:] if matlab_compat else p[max(n - smooth, 0):])
    return float(lo), float(hi)


def profiles_min_max(profile_a, mask_a, profile_b, mask_b,
                     matlab_compat: bool = False) -> tuple[float, float, float, float]:
    """ComputeProfilesMinMax -> (Amin, Amax, Bmin, Bmax): robust min and max of two profiles
    (mean of the lowest / highest values after removing outliers, ROI points only).
    ``matlab_compat=True`` uses MATLAB's smoothOutliers (which erases the beam peak)."""
    amin, amax = _min_max(profile_a, mask_a, matlab_compat=matlab_compat)
    bmin, bmax = _min_max(profile_b, mask_b, matlab_compat=matlab_compat)
    return amin, amax, bmin, bmax


@dataclass
class ProfileAnalysis:
    fits: ProfileFits
    minmax: tuple[float, float, float, float]
    messages: list[str]


def analyze_profile_pair(pair: ProfilePair, name_a: str, name_b: str,
                         method: str = "gauss1", matlab_compat: bool = False) -> ProfileAnalysis:
    """What the three Profile*Button callbacks do after extracting the profiles: Gaussian fits
    of both profiles, robust min/max, and the messages for the text area, in the same order.

    The min/max messages of the 45 degree and L1/L2 callbacks printed the values of the OTHER
    profile (labels crossed); here each label carries its own profile's values.
    """
    a, b = pair.first, pair.second
    fits = fit_profiles(a.vector, a.values, a.mask, name_a, b.vector, b.values, b.mask, name_b, method)
    amin, amax, bmin, bmax = profiles_min_max(a.values, a.mask, b.values, b.mask, matlab_compat)
    messages = fits.messages + [f"{name_a} min/max=[{amin:.3g},{amax:.3g}]",
                                f"{name_b} min/max=[{bmin:.3g},{bmax:.3g}]"]
    return ProfileAnalysis(fits, (amin, amax, bmin, bmax), messages)
