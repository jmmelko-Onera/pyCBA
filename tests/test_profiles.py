"""Tests of pycba.core.profiles (Gaussian fits of 1D profiles).

1. analytical tests (synthetic profiles);  2. MATLAB comparison, needs
matlab_reference/profile_*.mat from export_profiles_reference.m.
"""
import os
import sys
import warnings
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pytest

from pycba.core import profiles
from pycba.core import (analyze_profile_pair, extract_diagonal_profiles, extract_hv_profiles,
                        extract_l1l2_profiles, fit_profiles, gaussian_fit, matlab_round, my_gauss_fit,
                        profiles_min_max, smooth_outliers, spdiags_column)

REF = HERE / "matlab_reference"


def profile(n=120, b=300.0, c=80.0, a=800.0, offset=0.0, noise=0.0, seed=0):
    x = np.arange(1, n + 1) * 5.5
    y = a * np.exp(-(((x - b) / c) ** 2)) + offset
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, n)
    return x, y


# ------------------------------------------------------------------ gauss1
def test_gauss1_recovers_parameters_and_waist():
    x, y = profile(b=300, c=80, a=800, offset=50)
    fit = gaussian_fit(x, y)
    assert fit.params == pytest.approx((800, 300, 80, 50), rel=1e-6)
    assert fit.waist == pytest.approx(np.sqrt(2) * 80, rel=1e-6)       # 1/e^2 radius
    assert fit.diameter == pytest.approx(4 * 80 / np.sqrt(2), rel=1e-6)  # = 4 sigma, sigma = c/sqrt2


def test_gauss1_baseline_does_not_bias_the_width():
    """MATLAB's built-in gauss1 has no baseline: a 20 % background made it overestimate."""
    x, y = profile(b=330, c=60, a=800, offset=160, noise=5)
    assert gaussian_fit(x, y).waist == pytest.approx(np.sqrt(2) * 60, rel=0.02)


def test_gauss1_curve_is_10x_oversampled():
    x, y = profile(offset=20)
    fit = gaussian_fit(x, y)
    assert fit.xfit.size == 10 * (x.size - 1) + 1
    assert fit.xfit[0] == x[0] and fit.xfit[-1] == pytest.approx(x[-1])
    np.testing.assert_allclose(fit.yfit[::10], y, atol=1e-6 * 800)


def _plain_gauss1(x, y):
    """The unregularised fit (step 1 alone)."""
    from scipy.optimize import curve_fit
    p, _ = curve_fit(profiles._gauss1, x, y, p0=profiles._gauss1_start(x, y), maxfev=20000)
    return abs(p[2])


def test_gauss1_tight_window_is_stabilised_and_wide_window_is_unchanged():
    """Tight aperture: baseline and width are degenerate; the soft prior d~0 removes most of the scatter.
    Wide aperture: the result is exactly the plain fit."""
    rng = np.random.default_rng(3)
    sigma = 10.0
    c_true = np.sqrt(2) * sigma
    x = np.arange(-150, 151, 1.0)
    ones = np.ones(x.size)
    plain, two_step = [], []
    for _ in range(40):
        y = 1000 * np.exp(-(x / c_true) ** 2) + rng.normal(0, 10, x.size)
        m = np.abs(x) <= 1.5 * sigma
        plain.append(_plain_gauss1(x[m], y[m]))
        two_step.append(profiles._fit_gauss1(x[m], y[m], np.ones(m.sum()))[2])
        w = np.abs(x) <= 4 * sigma
        assert profiles._fit_gauss1(x[w], y[w], np.ones(w.sum()))[2] == pytest.approx(_plain_gauss1(x[w][2:-2], y[w][2:-2]))
    assert np.std(two_step) < 0.5 * np.std(plain)
    assert abs(np.mean(two_step) / c_true - 1) < 0.02


def test_gauss1_ignores_masked_points():
    x, y = profile(noise=3)
    mask = np.ones(x.size)
    mask[80:100] = 0
    ref = gaussian_fit(x, y, mask)                    # masked points are simply not used
    y2 = y.copy()
    y2[80:100] = 5000.0                               # garbage outside the ROI
    assert gaussian_fit(x, y2, mask).waist == pytest.approx(ref.waist, rel=1e-9)
    assert gaussian_fit(x, y).waist != pytest.approx(ref.waist, rel=1e-9)   # the mask matters


# A Gaussian is ~0 at the edges, so an outlier there has no leverage. Use beams centred
# NEAR an edge so that the edge points really influence the fit.
RIGHT = dict(b=640.0, c=80.0)      # x runs from 5.5 to 660
LEFT = dict(b=25.0, c=80.0)
N = 120


def _waist_with_outlier(kw, idx):
    x, y = profile(noise=3, **kw)
    y2 = y.copy()
    y2[idx] += 400.0                                   # mild enough for the fit to converge
    ref, out = gaussian_fit(x, y).waist, gaussian_fit(x, y2).waist
    assert ref > 0 and out > 0                         # both fits converged
    return ref, out


@pytest.mark.parametrize("kw,idx", [(LEFT, 0), (LEFT, 1), (RIGHT, N - 2), (RIGHT, N - 1)])
def test_gauss1_two_first_and_two_last_points_are_excluded(kw, idx):
    """MATLAB 1-based [1 2 n-1 n] = 0-based 0, 1, n-2, n-1."""
    ref, with_outlier = _waist_with_outlier(kw, idx)
    assert with_outlier == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize("kw,idx", [(LEFT, 2), (RIGHT, N - 3)])
def test_gauss1_next_points_are_used(kw, idx):
    ref, with_outlier = _waist_with_outlier(kw, idx)
    assert with_outlier != pytest.approx(ref, rel=1e-3)


def test_gauss1_failure_returns_empty_fit_with_warning():
    x, y = profile(n=4)
    with pytest.warns(RuntimeWarning, match="fit failed"):
        fit = gaussian_fit(x, y)
    assert fit.waist == 0 and fit.xfit.size == 0 and fit.yfit.size == 0


def test_unknown_method():
    with pytest.raises(ValueError):
        gaussian_fit([1, 2, 3], [1, 2, 3], method="bogus")


# ------------------------------------------------------------ my_gauss_fit
def test_my_gauss_fit_recovers_gaussian_with_baseline():
    x = np.linspace(-10, 10, 201)
    sigma, mu, amp, y0 = 2.0, 0.7, 1000.0, 20.0
    y = y0 + amp / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    s_, m_, area, base = my_gauss_fit(x, y)
    assert (s_, m_, base) == pytest.approx((sigma, mu, y0), rel=1e-3)
    assert area == pytest.approx(amp, rel=1e-3)                  # trapezoid area of y - y0


def test_my_gauss_fit_degenerate_inputs_return_zeros():
    assert my_gauss_fit([], []) == (0.0, 0.0, 0.0, 0.0)
    assert my_gauss_fit([1, 2, 3], [1, 2]) == (0.0, 0.0, 0.0, 0.0)


def test_my_gauss_fit_area_between_half_and_1p5_is_not_a_special_case():
    """The original did not normalise when 0.5 <= area <= 1.5, which gave a wrong sigma."""
    x = np.linspace(-10, 10, 401)
    sigma = 2.0
    y = 1.2 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x**2) / (2 * sigma**2))   # area 1.2
    s_, m_, area, _ = my_gauss_fit(x, y)
    assert s_ == pytest.approx(sigma, rel=1e-3) and area == pytest.approx(1.2, rel=1e-3)


def test_my_gauss_fit_sigma0_and_mu0_are_independent():
    x = np.linspace(-10, 10, 201)
    y = 500 / (np.sqrt(2 * np.pi) * 2) * np.exp(-((x - 1) ** 2) / 8) + 5
    ref = my_gauss_fit(x, y)
    np.testing.assert_allclose(my_gauss_fit(x, y, sigma0=2.5), ref, rtol=1e-6)
    np.testing.assert_allclose(my_gauss_fit(x, y, sigma0=2.5, mu0=0.5), ref, rtol=1e-6)


def test_my_gauss_fit_never_diverges_on_garbage():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = np.linspace(0, 100, int(rng.integers(5, 80)))
        sigma, mu, s, y0 = my_gauss_fit(x, rng.normal(0, 1, x.size))
        assert np.isfinite(sigma) and sigma >= 0 and np.isfinite(mu)


def test_my_gauss_fit_null_area_returns_zero_sigma():
    assert my_gauss_fit([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])[:2] == (0.0, 0.0)


def test_my_gauss_fit_sorts_by_x():
    x = np.linspace(-10, 10, 201)
    y = 500 / (np.sqrt(2 * np.pi) * 2) * np.exp(-((x - 1) ** 2) / 8) + 5
    perm = np.random.default_rng(0).permutation(x.size)
    np.testing.assert_allclose(my_gauss_fit(x[perm], y[perm]), my_gauss_fit(x, y), rtol=1e-12)


def test_legacy_method_gives_same_diameter_definition():
    x = np.linspace(-10, 10, 201)
    sigma = 2.0
    y = 500 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x**2) / (2 * sigma**2)) + 10
    fit = gaussian_fit(x, y, method="legacy")
    assert fit.waist == pytest.approx(2 * sigma, rel=1e-3)       # 1/e^2 radius = 2 sigma
    assert fit.diameter == pytest.approx(4 * sigma, rel=1e-3)
    gfit = gaussian_fit(x, y)                                    # gauss1 now has a baseline too
    assert gfit.waist == pytest.approx(fit.waist, rel=1e-3)


# ------------------------------------------------------------ fit_profiles
def test_fit_profiles_messages_and_roi_warning():
    x, y = profile()
    mask_ok = np.ones(x.size)
    mask_bad = np.zeros(x.size)
    mask_bad[0] = 1                                              # < 1 % of the profile
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = fit_profiles(x, y, mask_ok, "H", x, y, mask_bad, "V")
    assert res.messages[0] == f"H diam.={2 * res.h.waist:.4g}"
    assert any("- profile does not cross ROI" in str(i.message) for i in w)
    assert res.v.waist == 0 and res.messages[1] == "V diam.=0"


# ----------------------------------------------------------- profile extraction
def test_matlab_round_halves_go_away_from_zero():
    assert [matlab_round(v) for v in (0.5, 1.5, 2.5, -0.5, -2.5, 2.4, 2.6)] == [1, 2, 3, -1, -3, 2, 3]


def test_hv_profiles_row_and_column_with_1_based_positions():
    img = np.arange(1, 25, dtype=float).reshape(4, 6)           # 4 rows, 6 columns
    p = extract_hv_profiles(img, xpos=3 * 5.5, ypos=2 * 5.5, pixelsize=5.5)
    assert (p.info["linenum"], p.info["colnum"]) == (2, 3)
    np.testing.assert_array_equal(p.first.values, img[1, :])      # row 2
    np.testing.assert_array_equal(p.second.values, img[:, 2])     # column 3
    np.testing.assert_allclose(p.first.vector, np.arange(1, 7) * 5.5)
    np.testing.assert_allclose(p.second.vector, np.arange(1, 5) * 5.5)
    assert p.first.mask.all() and p.second.mask.all()


def test_hv_profiles_round_half_up_and_roi_mask():
    img = np.ones((6, 6))
    roi = np.zeros((6, 6), bool)
    roi[2:5, 2:5] = True
    p = extract_hv_profiles(img, xpos=2.5, ypos=3.5, pixelsize=1.0, roi_mask=roi)
    assert (p.info["linenum"], p.info["colnum"]) == (4, 3)        # 3.5 -> 4, 2.5 -> 3 (not 2)
    np.testing.assert_array_equal(p.first.mask, [0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(p.first.values, [0, 0, 1, 1, 1, 0])


@pytest.mark.parametrize("x,y", [(0, 3), (3, 0), (99, 3), (3, 99)])
def test_hv_profiles_centre_outside_image_raises(x, y):
    with pytest.raises(IndexError):
        extract_hv_profiles(np.ones((6, 6)), x, y, 1.0)


def test_spdiags_column_matches_matlab_layout():
    tall = np.arange(1, 13, dtype=float).reshape(4, 3)            # aligned by COLUMN index
    np.testing.assert_array_equal(spdiags_column(tall, 0), [1, 5, 9])
    np.testing.assert_array_equal(spdiags_column(tall, -1), [4, 8, 12])
    np.testing.assert_array_equal(spdiags_column(tall, 1), [0, 2, 6])
    wide = np.arange(1, 13, dtype=float).reshape(3, 4)            # aligned by ROW index
    np.testing.assert_array_equal(spdiags_column(wide, 0), [1, 6, 11])
    np.testing.assert_array_equal(spdiags_column(wide, 1), [2, 7, 12])
    np.testing.assert_array_equal(spdiags_column(wide, -1), [0, 5, 10])
    np.testing.assert_array_equal(spdiags_column(wide, 7), [0, 0, 0])   # diagonal does not exist


def test_diagonal_profiles_square_image():
    img = np.arange(25, dtype=float).reshape(5, 5)
    p = extract_diagonal_profiles(img, xpos=3, ypos=3, pixelsize=1.0)
    np.testing.assert_array_equal(p.first.values, [0, 6, 12, 18, 24])     # '\' diagonal
    np.testing.assert_array_equal(p.second.values, [20, 16, 12, 8, 4])    # '/' diagonal, bottom-left first
    np.testing.assert_allclose(p.first.vector, np.arange(1, 6) * np.sqrt(2))
    assert p.first.mask.all()


def test_diagonal_profiles_pad_with_zeros_and_mask_the_padding():
    img = np.arange(1, 25, dtype=float).reshape(4, 6)                      # wide: 4 points
    p = extract_diagonal_profiles(img, xpos=5, ypos=1, pixelsize=1.0)      # '\' diagonal d = 4
    np.testing.assert_array_equal(p.first.values, [5, 12, 0, 0][:2] + [0, 0])
    np.testing.assert_array_equal(p.first.mask, [1, 1, 0, 0])              # padding is masked out
    assert p.first.values.size == p.first.vector.size == 4


def _elongated_beam(theta_deg, su=14.0, sv=6.0, n=220, c=110.0):
    """Gaussian whose axes are at angle theta (image coordinates, y down); c = centre (0-based)."""
    th = np.radians(theta_deg)
    y, x = np.mgrid[0:n, 0:n]
    u = (x - c) * np.cos(th) + (y - c) * np.sin(th)
    v = -(x - c) * np.sin(th) + (y - c) * np.cos(th)
    return 1000 * np.exp(-(u**2) / (2 * su**2) - (v**2) / (2 * sv**2)), th, c


def _phys(p0, ps):
    """0-based (x=col, y=row) pixel coordinates -> length units (pixel (row, col) 1-based is at col*ps, row*ps)."""
    return (np.asarray(p0, float) + 1.0) * ps


@pytest.mark.parametrize("theta", [0.0, 10.0, 30.0, -25.0, 45.0, 60.0, 89.0, 135.0])
def test_l1l2_profiles_recover_the_beam_axes(theta):
    """L1 along the long axis, L2 along the short one; error is < 0.25 % at any angle."""
    ps = 5.5
    img, th, c = _elongated_beam(theta)
    L = 80
    p1 = _phys([[c - L * np.cos(th), c - L * np.sin(th)], [c + L * np.cos(th), c + L * np.sin(th)]], ps)
    p2 = _phys([[c + L * np.sin(th), c - L * np.cos(th)], [c - L * np.sin(th), c + L * np.cos(th)]], ps)
    p = extract_l1l2_profiles(img, p1, p2, pixelsize=ps)
    assert p.info["angle"] == pytest.approx(theta, abs=1e-9)
    d1 = gaussian_fit(p.first.vector, p.first.values).diameter      # 4 sigma along L1, in length units
    d2 = gaussian_fit(p.second.vector, p.second.values).diameter
    assert d1 == pytest.approx(4 * 14.0 * ps, rel=0.01)
    assert d2 == pytest.approx(4 * 6.0 * ps, rel=0.01)


def test_l1l2_profile_is_exact_on_a_linear_image():
    """Bilinear interpolation reproduces a linear image exactly, whatever the direction."""
    ps = 2.0
    rows, cols = np.mgrid[0:30, 0:40]
    img = 3.0 * cols + 7.0 * rows                                 # 0-based indices
    p1 = _phys([[2.3, 4.1], [30.7, 20.2]], ps)
    p2 = _phys([[35.0, 3.0], [5.5, 26.5]], ps)
    p = extract_l1l2_profiles(img, p1, p2, pixelsize=ps)
    for prof, pos in ((p.first, p1), (p.second, p2)):
        x0, y0 = pos[0] / ps - 1
        x1, y1 = pos[1] / ps - 1
        length = np.hypot(x1 - x0, y1 - y0)
        s = prof.vector / ps                                      # distance in pixels
        expected = 3.0 * (x0 + s * (x1 - x0) / length) + 7.0 * (y0 + s * (y1 - y0) / length)
        np.testing.assert_allclose(prof.values, expected, atol=1e-9)
        assert prof.mask.all()


def test_l1l2_sampling_step_and_number_of_points():
    img = np.ones((20, 20))
    p = extract_l1l2_profiles(img, [[2 * 5.5, 3 * 5.5], [12.5 * 5.5, 3 * 5.5]],       # 10.5 pixels long
                              [[5 * 5.5, 2 * 5.5], [5 * 5.5, 2 * 5.5]], pixelsize=5.5)  # zero-length line
    np.testing.assert_allclose(p.first.vector, np.arange(11) * 5.5)
    assert p.second.vector.size == 1 and p.second.values.size == 1


def test_l1l2_points_outside_the_image_or_roi_are_masked():
    img = np.ones((10, 10))
    roi = np.zeros((10, 10), bool)
    roi[:, 3:8] = True                                            # columns 4..8 (1-based)
    # L1: row 6 (1-based), columns -1 .. 12 (1-based): leaves the image on both sides
    p = extract_l1l2_profiles(img, [[-1.0, 6.0], [12.0, 6.0]], [[1.0, 1.0], [1.0, 1.0]], roi_mask=roi)
    cols = np.arange(-1, 13)                                      # 1-based column of each sample
    inside_img = (cols >= 1) & (cols <= 10)
    in_roi = (cols >= 4) & (cols <= 8)
    np.testing.assert_array_equal(p.first.mask.astype(bool), inside_img & in_roi)
    assert np.all(p.first.values[~inside_img] == 0)


def test_l1l2_not_finite_pixels_are_masked_with_their_interpolation_neighbours():
    img = np.ones((10, 10))
    img[5, 5] = np.nan                                            # row 6, column 6 (1-based)
    p = extract_l1l2_profiles(img, [[1.0, 6.0], [10.0, 6.0]], [[1.0, 1.0], [1.0, 1.0]])
    assert p.first.mask[5] == 0 and p.first.values[5] == 0        # the NaN pixel itself
    assert p.first.mask[[0, 1, 2, 3, 7, 8, 9]].all()              # samples >= 2 pixels away are valid
    assert np.isfinite(p.first.values).all()


def test_l1l2_lines_need_not_be_perpendicular_and_roi_cropping_is_applied():
    img, th, c = _elongated_beam(0.0)
    roi = np.zeros(img.shape, bool)
    roi[:, : int(c)] = True                                       # left half only
    p = extract_l1l2_profiles(img, _phys([[c - 50, c], [c + 50, c]], 1.0), _phys([[c - 40, c + 40], [c + 40, c - 40]], 1.0),
                              roi_mask=roi, pixelsize=1.0)
    assert p.first.mask[:50].all() and not p.first.mask[60:].any()    # right of the centre: outside the ROI
    assert p.first.values[60:].sum() == 0                            # image cropped by the ROI


# ------------------------------------------------------ smoothOutliers & min/max
def test_smooth_outliers_replaces_outliers_by_the_median():
    y = np.array([1, 2, 1, 2, 100, 1, 2, 1, 2, 1], float)
    out = smooth_outliers(y)
    assert out[4] == np.median(y) and np.delete(out, 4).tolist() == np.delete(y, 4).tolist()
    assert y[4] == 100                                           # input untouched


def test_smooth_outliers_null_mad_changes_nothing():
    y = np.array([5, 5, 5, 5, 9], float)
    np.testing.assert_array_equal(smooth_outliers(y), y)


def test_smooth_outliers_nan_do_not_misalign_the_replaced_elements():
    y = np.array([np.nan, 1, 2, 1, 2, 100, 1], float)
    out = smooth_outliers(y)
    assert np.isnan(out[0]) and out[5] == np.median([1, 2, 1, 2, 100, 1])
    assert out[1:5].tolist() == [1, 2, 1, 2] and out[6] == 1


def test_profiles_min_max_uses_the_same_number_of_points_on_both_sides():
    v = np.arange(1, 11, dtype=float)
    amin, amax, bmin, bmax = profiles_min_max(v, np.ones(10), v[::-1], np.ones(10))
    assert amin == pytest.approx(1.5)             # mean of the 2 lowest
    assert amax == pytest.approx(9.5)             # mean of the 2 highest (MATLAB took 3: off-by-one)
    assert (bmin, bmax) == (amin, amax)
    assert profiles_min_max(v, np.ones(10), v, np.ones(10), matlab_compat=True)[1] == pytest.approx(9.0)  # literal


def test_profiles_min_max_tiny_profiles():
    one = profiles_min_max(np.array([4.0]), np.ones(1), np.array([4.0, 6.0]), np.ones(2))
    assert one == (4.0, 4.0, 5.0, 5.0)        # 2 points: min and max are both their mean


def test_profiles_min_max_uses_the_mask_and_handles_empty():
    v = np.arange(1, 11, dtype=float)
    mask = np.array([0] * 5 + [1] * 5)
    amin, amax, _, _ = profiles_min_max(v, mask, v, mask)
    assert amin == pytest.approx(6.5) and amax == pytest.approx(9.5)
    nan_min, nan_max, _, _ = profiles_min_max(v, np.zeros(10), v, np.zeros(10))
    assert np.isnan(nan_min) and np.isnan(nan_max)


def _beam_profile(sigma=10.0, n=1000, seed=0, peak=1000.0, bg=50.0, noise=5.0):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    return bg + peak * np.exp(-((x - n / 2) ** 2) / (2 * sigma**2)) + rng.normal(0, noise, n)


@pytest.mark.parametrize("sigma", [1.5, 3.0, 10.0, 40.0])
def test_profiles_min_max_keeps_the_beam_peak(sigma):
    y = _beam_profile(sigma)
    ones = np.ones(y.size)
    _, hi, _, _ = profiles_min_max(y, ones, y, ones)
    assert hi == pytest.approx(np.sort(y)[-2:].mean(), rel=0.02)


@pytest.mark.parametrize("n_hot", [1, 2])
def test_profiles_min_max_ignores_hot_points_inside_and_outside_the_beam(n_hot):
    y = _beam_profile(10.0)
    ones = np.ones(y.size)
    ref = profiles_min_max(y, ones, y, ones)[1]
    for pos in (100, 500, 800):                                  # outside, on the peak, outside
        y2 = y.copy()
        y2[pos:pos + n_hot] = 3e5
        assert profiles_min_max(y2, ones, y2, ones)[1] == pytest.approx(ref, rel=0.03), pos


def test_profiles_min_max_cold_point_does_not_set_the_minimum():
    y = _beam_profile(10.0)
    ones = np.ones(y.size)
    ref = profiles_min_max(y, ones, y, ones)[0]
    y[300] = -5000.0
    assert profiles_min_max(y, ones, y, ones)[0] == pytest.approx(ref, abs=15)


def test_profiles_min_max_matlab_compat_erases_the_peak():
    """Documents the MATLAB behaviour: smoothOutliers(.,5) replaces the beam itself."""
    y = _beam_profile(10.0)
    ones = np.ones(y.size)
    _, hi, _, _ = profiles_min_max(y, ones, y, ones, matlab_compat=True)
    assert hi < 0.2 * y.max()


def test_analyze_profile_pair_labels_carry_their_own_values():
    x, ya = profile(a=100, noise=1)
    _, yb = profile(a=500, noise=1)
    from pycba.core import Profile, ProfilePair
    pair = ProfilePair(Profile(x, ya, np.ones(x.size)), Profile(x, yb, np.ones(x.size)))
    res = analyze_profile_pair(pair, "A", "B")
    assert res.messages[0].startswith("A diam.=")
    amin, amax, bmin, bmax = res.minmax
    assert res.messages[2] == f"A min/max=[{amin:.3g},{amax:.3g}]"
    assert res.messages[3] == f"B min/max=[{bmin:.3g},{bmax:.3g}]"
    assert bmax > 4 * amax                                       # B is the brighter profile


# ------------------------------------------------------------------ MATLAB
@pytest.mark.parametrize("ref", sorted(REF.glob("profile_*.mat")) or [None])
def test_against_matlab_profiles(ref):
    scipy_io = pytest.importorskip("scipy.io")
    if ref is None:
        pytest.skip("no MATLAB profile reference exported yet (export_profiles_reference.m)")
    m = scipy_io.loadmat(ref)
    f = lambda k: np.ravel(m[k]).astype(float)        # noqa: E731
    x, y, mask = f("x"), f("y"), f("mask")

    # myGaussFit and the legacy branch are deterministic: must match tightly
    np.testing.assert_allclose(my_gauss_fit(x[mask == 1], y[mask == 1]), f("mygauss"),
                               rtol=1e-6, atol=1e-9)
    lg = gaussian_fit(x, y, mask, method="legacy")
    np.testing.assert_allclose(lg.waist, float(f("waist_legacy")[0]), rtol=1e-6)
    np.testing.assert_allclose(lg.yfit, f("yfit_legacy"), rtol=1e-6, atol=1e-6 * np.nanmax(np.abs(y)))

    # gauss1: iterative fits in two different implementations -> looser tolerance
    if float(f("gauss1_ok")[0]) == 0 or float(f("waist_gauss1")[0]) == 0:
        pytest.skip("MATLAB gauss1 fit failed on this profile")
    g1 = gaussian_fit(x, y, mask)
    a1, b1, c1, d1 = g1.params
    ra, rb, rc, rd = f("coefs_gauss1")
    scale = np.nanmax(np.abs(y))
    np.testing.assert_allclose([b1, c1], [rb, abs(rc)], rtol=2e-3)
    np.testing.assert_allclose([a1, d1], [ra, rd], rtol=2e-3, atol=2e-3 * scale)
    np.testing.assert_allclose(g1.waist, float(f("waist_gauss1")[0]), rtol=2e-3)
    np.testing.assert_allclose(g1.yfit, f("yfit_gauss1"), atol=2e-3 * np.nanmax(np.abs(y)))


@pytest.mark.parametrize("ref", sorted(REF.glob("extract_*.mat")) or [None])
def test_against_matlab_extraction(ref):
    """Needs extract_*.mat from export_extraction_reference.m (real spdiags/imrotate)."""
    scipy_io = pytest.importorskip("scipy.io")
    if ref is None:
        pytest.skip("no MATLAB extraction reference exported yet (export_extraction_reference.m)")
    m = scipy_io.loadmat(ref)
    f = lambda k: np.ravel(m[k]).astype(float)        # noqa: E731
    img, ps = m["image"].astype(float), float(f("pixelsize")[0])
    roi = m["roimask"].astype(bool) if float(f("has_roi")[0]) else None
    xpos, ypos = float(f("xpos")[0]), float(f("ypos")[0])

    hv = extract_hv_profiles(img, xpos, ypos, ps, roi)
    for name, prof in (("H", hv.first), ("V", hv.second)):
        np.testing.assert_allclose(prof.vector, f(f"{name}vector"), rtol=1e-12)
        np.testing.assert_allclose(prof.values, f(f"{name}profile"), rtol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(prof.mask, f(f"{name}mask"))

    dg = extract_diagonal_profiles(img, xpos, ypos, ps, roi)
    for name, prof in (("minus", dg.first), ("plus", dg.second)):
        np.testing.assert_allclose(prof.vector, f(f"{name}Vector"), rtol=1e-12)
        np.testing.assert_allclose(prof.values, f(f"{name}Profile"), rtol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(prof.mask, f(f"{name}Mask"))

    # min/max computed on MATLAB's own profiles: new function (default) and original (compat)
    for kind, (pa, ma, pb, mb) in (("hv", ("Hprofile", "Hmask", "Vprofile", "Vmask")),
                                   ("diag", ("minusProfile", "minusMask", "plusProfile", "plusMask"))):
        mm = profiles_min_max(f(pa), f(ma), f(pb), f(mb))
        np.testing.assert_allclose(mm, f(f"minmax_{kind}_new"), rtol=1e-12, equal_nan=True)
        if not np.isnan(f(pa)).any() and not np.isnan(f(pb)).any():   # original smoothOutliers mis-aligns NaN
            mmc = profiles_min_max(f(pa), f(ma), f(pb), f(mb), matlab_compat=True)
            np.testing.assert_allclose(mmc, f(f"minmax_{kind}_orig"), rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("ref", sorted(REF.glob("l1l2_*.mat")) or [None])
def test_against_matlab_l1l2(ref):
    """Needs l1l2_*.mat from export_extraction_reference.m (lineProfile.m, interp2 'linear')."""
    scipy_io = pytest.importorskip("scipy.io")
    if ref is None:
        pytest.skip("no MATLAB L1/L2 reference exported yet (export_extraction_reference.m)")
    m = scipy_io.loadmat(ref)
    f = lambda k: np.ravel(m[k]).astype(float)        # noqa: E731
    roi = m["roimask"].astype(bool) if float(f("has_roi")[0]) else None
    p = extract_l1l2_profiles(m["image"].astype(float), m["posL1"], m["posL2"], roi, float(f("pixelsize")[0]))
    assert p.info["angle"] == pytest.approx(float(f("angle")[0]), abs=1e-9)
    for prof, name in ((p.first, "L1"), (p.second, "L2")):
        np.testing.assert_allclose(prof.vector, f(f"{name}Vector"), rtol=1e-12)
        assert prof.values.size == f(f"{name}Profile").size
        ref_mask = f(f"{name}Mask")
        assert np.sum(prof.mask != ref_mask) <= 1                      # ROI ties at .5 may differ
        ok = (prof.mask == 1) & (ref_mask == 1)
        np.testing.assert_allclose(prof.values[ok], f(f"{name}Profile")[ok], rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v", "-p", "no:cacheprovider"]))
