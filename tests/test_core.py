"""Tests of pycba.core.beam_metrics.

1. Analytical tests on synthetic Gaussian beams (independent of MATLAB).
2. MATLAB-convention tests (1-based positions, column-major ties).
3. Comparison with MATLAB (needs core_*.mat from matlab_reference/export_core_reference.m).
"""
import os
import sys
import warnings
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pytest

from pycba.core import (compute_beam_peak, compute_centroid, compute_d4sigma, compute_d_aperture,
                        compute_d_knife_edge, compute_dpeak, compute_peak_center)

REF = HERE / "matlab_reference"
PS = 5.5  # pixel size (um) used in the tests


def gaussian(shape=(200, 240), cx=120.3, cy=95.7, sx=10.0, sy=6.0, amp=1000.0):
    """cx, cy are 1-based pixel coordinates (MATLAB convention)."""
    y, x = np.mgrid[1 : shape[0] + 1, 1 : shape[1] + 1]
    return amp * np.exp(-((x - cx) ** 2) / (2 * sx**2) - ((y - cy) ** 2) / (2 * sy**2))


# --------------------------------------------------------------- analytical
def test_centroid_is_1_based_position():
    img = gaussian()
    xc, yc = compute_centroid(img, PS)
    assert xc == pytest.approx(120.3 * PS, rel=1e-9)
    assert yc == pytest.approx(95.7 * PS, rel=1e-9)


def test_d4sigma_gaussian():
    img = gaussian(sx=10, sy=6)
    xc, yc = compute_centroid(img, PS)
    r = compute_d4sigma(img, xc, yc, PS)
    assert r.diam_h == pytest.approx(4 * 10 * PS, rel=1e-6)
    assert r.diam_v == pytest.approx(4 * 6 * PS, rel=1e-6)
    assert (r.x0, r.y0) == (xc, yc)


def test_d4sigma_negative_variance_is_zero_like_matlab_real_sqrt():
    # a constant negative image gives a negative variance: real(sqrt(<0)) = 0
    img = -np.ones((5, 5))
    xc, yc = 3 * PS, 3 * PS
    r = compute_d4sigma(img, xc, yc, PS)
    assert r.diam_h >= 0 and r.diam_v >= 0


def test_dpeak_gaussian_is_subpixel():
    img = gaussian(cx=120, cy=96, sx=10.3, sy=6.1)
    r = compute_dpeak(img, PS)
    assert (r.x0, r.y0) == (120 * PS, 96 * PS)
    assert r.diam_h == pytest.approx(4 * 10.3 * PS, abs=0.1 * PS)       # was +-1 pixel (even number of pixels)
    assert r.diam_v == pytest.approx(4 * 6.1 * PS, abs=0.1 * PS)


def test_dpeak_removes_the_pixel_quantisation():
    """The old result was an even integer number of pixels, whatever the true size."""
    yy, xx = np.mgrid[1:201, 1:201]
    errors = []
    for w in np.linspace(8.0, 30.0, 23):                               # 1/e^2 radius, not an integer
        img = np.exp(-2 * ((xx - 100.37) ** 2 + (yy - 99.81) ** 2) / w**2)
        r = compute_dpeak(img, 1.0)
        errors += [r.diam_h - 2 * w, r.diam_v - 2 * w]
    assert np.max(np.abs(errors)) < 0.15
    assert abs(np.mean(errors)) < 0.1


def test_dpeak_pixel_size_scales_the_result():
    img = gaussian(cx=60, cy=50, sx=8.3, sy=8.3)
    a, b = compute_dpeak(img, 1.0), compute_dpeak(img, 5.5)
    assert b.diam_h == pytest.approx(5.5 * a.diam_h, rel=1e-12) and b.diam_v == pytest.approx(5.5 * a.diam_v, rel=1e-12)


def test_dpeak_ignores_a_noisy_pixel_far_from_the_beam():
    """The old search took, on the whole row, the pixel closest to the level: a pixel at the level far from the beam won."""
    img = gaussian(cx=100, cy=100, sx=10, sy=10)
    ref = compute_dpeak(img, 1.0)
    level = img.max() / np.exp(2)
    img[100 - 1, 190] = level                                          # exactly at the level, 90 px from the peak (row 100 = index 99)
    img[20, 100 - 1] = level
    r = compute_dpeak(img, 1.0)
    assert r.diam_h == pytest.approx(ref.diam_h, rel=1e-12) and r.diam_v == pytest.approx(ref.diam_v, rel=1e-12)


def test_dpeak_is_the_distance_between_the_two_crossings():
    """An asymmetric profile: the old result was twice the distance of ONE side."""
    x = np.arange(1, 201, dtype=float)
    p = np.where(x < 100, np.exp(-2 * ((x - 100) / 10) ** 2), np.exp(-2 * ((x - 100) / 30) ** 2))
    img = np.tile(p, (9, 1))
    r = compute_dpeak(img, 1.0)
    assert r.diam_h == pytest.approx(10 + 30, abs=0.05)


def test_dpeak_beam_cut_by_the_edge_and_degenerate_images():
    x = np.arange(1, 101, dtype=float)
    img = np.tile(np.exp(-2 * ((x - 95) / 10) ** 2), (9, 1))          # right side never reaches the level
    r = compute_dpeak(img, 1.0)
    assert r.diam_h == pytest.approx(2 * 10, abs=0.05)                 # twice the left half-width
    assert compute_dpeak(np.zeros((5, 5)), 1.0).diam_h == 0
    assert compute_dpeak(-np.ones((5, 5)), 1.0).diam_v == 0


def test_dpeak_nan_on_a_crossing_makes_that_side_missing():
    x = np.arange(1, 201, dtype=float)
    p = np.where(x < 100, np.exp(-2 * ((x - 100) / 10) ** 2), np.exp(-2 * ((x - 100) / 30) ** 2))   # crossings at -10 and +30
    img = np.tile(p, (9, 1))
    assert compute_dpeak(img, 1.0).diam_h == pytest.approx(40, abs=0.05)
    img[:, 99 + 29: 99 + 32] = np.nan                                   # NaN where the right edge is
    assert compute_dpeak(img, 1.0).diam_h == pytest.approx(2 * 10, abs=0.05)      # twice the left half-width


def test_knife_edge_gaussian_within_pixel_resolution():
    img = gaussian(sx=10, sy=6)
    xc, yc = compute_centroid(img, PS)
    r = compute_d_knife_edge(img, xc, yc, PS)
    assert r.diam_h == pytest.approx(4 * 10 * PS, abs=0.2 * PS)       # sub-pixel: 4 sigma to a fraction of a pixel
    assert r.diam_v == pytest.approx(4 * 6 * PS, abs=0.2 * PS)
    old = compute_d_knife_edge(img, xc, yc, PS, subpixel=False)       # MATLAB's integer-pixel positions
    assert old.diam_h == pytest.approx(4 * 10 * PS, abs=2.5 * PS)


def test_knife_edge_subpixel_removes_the_pixel_quantisation():
    """A beam moved by fractions of a pixel: the interpolated diameter stays put, MATLAB's one jumps by whole pixels."""
    sig = 7.3
    y, x = np.mgrid[1:101, 1:121].astype(float)
    new, old = [], []
    for dx in (0.0, 0.2, 0.4, 0.5, 0.7, 0.9):
        img = np.exp(-0.5 * (((x - 60 - dx) / sig) ** 2 + ((y - 50) / (0.6 * sig)) ** 2))
        xc, yc = compute_centroid(img, 1.0)
        new.append(compute_d_knife_edge(img, xc, yc, 1.0).diam_h)
        old.append(compute_d_knife_edge(img, xc, yc, 1.0, subpixel=False).diam_h)
    assert np.ptp(new) < 0.05 and np.max(np.abs(np.array(new) - 4 * sig)) < 0.15
    assert np.ptp(old) >= 1.9 and np.max(np.abs(np.array(old) - 4 * sig)) > 1.0


def test_knife_edge_subpixel_pixel_size_and_single_pixel_beam():
    img = np.zeros((9, 9)); img[4, 4] = 1.0                           # all the energy in one pixel
    r = compute_d_knife_edge(img, 5.0, 5.0, 1.0)
    assert r.diam_h == pytest.approx(2 * (0.8413447 - 0.1586553)) and r.diam_v == pytest.approx(r.diam_h)
    big = gaussian(sx=10, sy=6)
    a = compute_d_knife_edge(big, 0, 0, 1.0).diam_h
    assert compute_d_knife_edge(big, 0, 0, 0.5).diam_h == pytest.approx(0.5 * a)      # scales with the pixel size


def test_aperture_circular_gaussian():
    img = gaussian(shape=(240, 240), cx=120.5, cy=120.5, sx=12, sy=12)
    xc, yc = compute_centroid(img, PS)
    r = compute_d_aperture(img, xc, yc, PS)
    assert r.diam_h == r.diam_v
    assert r.diam_h == pytest.approx(4 * 12 * PS, abs=3 * PS)   # 86.47 % circle: D = 4 sigma


def test_aperture_ignores_nan_pixels():
    img = gaussian(shape=(240, 240), cx=120.5, cy=120.5, sx=12, sy=12)
    img[120, 119] = np.nan                                  # one NaN in the beam core
    clean = img.copy()
    clean[120, 119] = 0.0                                   # NaN ignored == pixel at 0
    xc, yc = compute_centroid(clean, PS)
    fixed = compute_d_aperture(img, xc, yc, PS)
    assert fixed.diam_h == compute_d_aperture(clean, xc, yc, PS).diam_h
    legacy = compute_d_aperture(img, xc, yc, PS, matlab_compat=True)   # whole column dropped
    assert legacy.diam_h != fixed.diam_h


def test_beam_peak_info_flags_fallback():
    clean = gaussian(sx=10, sy=10)
    _, info = compute_beam_peak(clean, return_info=True)
    assert info["fallback"] is False and info["n_points"] >= info["nthreshold"]
    hot = clean.copy()
    hot[10, 10] = 1e6                                       # isolated hot pixel is the maximum
    _, info = compute_beam_peak(hot, return_info=True)
    assert info["fallback"] is True


def _noisy_beam(sx, seed=0, shape=(200, 240)):
    rng = np.random.default_rng(seed)
    img = gaussian(shape=shape, sx=sx, sy=sx) + rng.normal(0, 5, shape) + 50
    return img, img.max()


@pytest.mark.parametrize("sx", [1.0, 2.0, 6.0, 15.0])
@pytest.mark.parametrize("where", ["outside", "inside"])
def test_beam_peak_ignores_hot_pixels(sx, where):
    img, true_peak = _noisy_beam(sx)
    pos = (10, 10) if where == "outside" else (96, 120)      # (96,120) = beam centre
    img[pos] = 300 * true_peak
    assert compute_beam_peak(img) == pytest.approx(true_peak, rel=0.05)


@pytest.mark.parametrize("pts", [[(0, 0), (0, 1)], [(0, 0), (0, 1), (0, 2)], [(0, 0), (1, 1)]])
def test_beam_peak_ignores_small_hot_groups_inside_the_beam(pts):
    img, true_peak = _noisy_beam(6.0)
    for dy, dx in pts:
        img[96 + dy, 120 + dx] = 300 * true_peak
    assert compute_beam_peak(img) == pytest.approx(true_peak, rel=0.05)


def test_beam_peak_hot_group_of_4_cannot_be_told_from_a_beam():
    img, true_peak = _noisy_beam(6.0)
    img[96:98, 120:122] = 300 * true_peak
    assert compute_beam_peak(img) > 10 * true_peak           # documented limit


def test_beam_peak_noise_level_regression():
    """MATLAB returned median + 5.8*MAD (~7 % of the maximum) for a speckled ROI beam."""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:256, 0:320]
    img = 1000 * np.exp(-((x - 150) ** 2 + (y - 120) ** 2) / (2 * 10**2)) * rng.gamma(8, 1 / 8, (256, 320))
    img[((x - 160) / 130) ** 2 + ((y - 128) / 100) ** 2 > 1] = 0.0       # zero outside the ROI
    p, info = compute_beam_peak(img, return_info=True)
    assert info["fallback"] and p > 0.9 * img.max()


def test_beam_peak_unchanged_when_no_fallback_needed():
    img = gaussian(sx=15, sy=15)                                           # smooth, wide beam
    p, info = compute_beam_peak(img, return_info=True)
    assert info["fallback"] is False
    assert 0.97 * img.max() <= p <= img.max()


def test_knife_edge_subpixel_stays_in_the_pixel_for_a_negative_image():
    """Negative total energy (garbage image): negative thresholds, the interpolation would extrapolate wildly."""
    y, x = np.mgrid[1:101, 1:121].astype(float)
    img = -np.exp(-0.5 * (((x - 60) / 9) ** 2 + ((y - 50) / 6) ** 2))
    new = compute_d_knife_edge(img, 60, 50, 1.0)
    old = compute_d_knife_edge(img, 60, 50, 1.0, subpixel=False)
    assert np.isfinite([new.diam_h, new.diam_v]).all()
    assert abs(new.diam_h - old.diam_h) <= 3 and abs(new.diam_v - old.diam_v) <= 3


def test_knife_edge_returns_zero_when_never_reached():
    r = compute_d_knife_edge(np.zeros((10, 10)), 0, 0, 1.0)
    assert (r.diam_h, r.diam_v) == (0.0, 0.0)


# -------------------------------------------------------------- conventions
def test_peak_center_ties_row_first_within_column_then_first_column():
    img = np.zeros((6, 6))
    img[4, 1] = 5.0
    img[2, 4] = 5.0
    assert compute_peak_center(img, 1.0) == (2.0, 5.0)    # column 2 (1-based), row 5


def test_beam_peak_ties_use_column_major_order():
    from pycba.core.beam_metrics import _argmax_colmajor
    img = np.zeros((6, 6))
    img[4, 1] = 5.0
    img[2, 4] = 5.0
    assert _argmax_colmajor(img) == (5.0, 4, 1)           # numpy default would give (2, 4)


def test_beam_peak_modes_and_no_input_mutation():
    img = gaussian(sx=10, sy=10)
    img[3, 3] = np.inf
    keep = img.copy()
    pmax = compute_beam_peak(img, "max")
    assert pmax == pytest.approx(1000.0, rel=1e-3)
    psmart = compute_beam_peak(img)
    assert 0.97 * pmax <= psmart <= pmax
    assert compute_beam_peak(img, np.inf) == pmax
    np.testing.assert_array_equal(img, keep)               # inf kept in the caller's array
    with pytest.raises(ValueError):
        compute_beam_peak(img, "bogus")


def test_nan_are_ignored():
    img = gaussian()
    ref = compute_centroid(img, PS)
    img2 = img.copy()
    img2[0, 0] = np.nan
    assert compute_centroid(img2, PS) == pytest.approx(ref, rel=1e-9)


# ------------------------------------------------------------------- MATLAB
@pytest.mark.parametrize("ref", sorted(REF.glob("core_*.mat")) or [None])
def test_against_matlab_core(ref):
    scipy_io = pytest.importorskip("scipy.io")
    if ref is None:
        pytest.skip("no MATLAB core reference exported yet (export_core_reference.m)")
    m = scipy_io.loadmat(ref)
    img, ps = m["img"], float(np.ravel(m["pixelsize"])[0])
    f = lambda k: np.ravel(m[k]).astype(float)
    xc, yc = compute_centroid(img, ps)
    np.testing.assert_allclose([xc, yc], f("centroid"), rtol=1e-12)
    np.testing.assert_allclose(compute_peak_center(img, ps), f("peakcenter"), rtol=1e-12)
    np.testing.assert_allclose(compute_d4sigma(img, xc, yc, ps), f("d4sigma"), rtol=1e-10)
    np.testing.assert_allclose(compute_dpeak(img, ps), f("dpeak"), rtol=1e-12)
    try:
        np.testing.assert_allclose(compute_d_knife_edge(img, xc, yc, ps), f("dknife"), rtol=1e-12)
    except AssertionError as exc:
        ref_sizes = f("dknife")[2:] / ps
        if np.allclose(ref_sizes, np.round(ref_sizes), rtol=0, atol=1e-9):
            raise AssertionError(
                "dknife of the reference are whole numbers of pixels: it was exported with the OLD integer-pixel "
                "ComputeDKnifeEdge. The exporting MATLAB must use the sub-pixel one (Utils/knifeEdgeAxis.m on the path "
                "and export_core_reference.m calling it).\n" + str(exc)) from None
        raise
    with warnings.catch_warnings():       # "Could not calculate diameter" is also raised by MATLAB
        warnings.simplefilter("ignore", RuntimeWarning)
        np.testing.assert_allclose(compute_d_aperture(img, xc, yc, ps, matlab_compat=True),
                                   f("daperture"), rtol=1e-10)
        if "daperture_fixed" in m:        # MATLAB copy with 'omitnan' added to the inner sum
            np.testing.assert_allclose(compute_d_aperture(img, xc, yc, ps),
                                       f("daperture_fixed"), rtol=1e-10)
    peak, info = compute_beam_peak(img, return_info=True)
    np.testing.assert_allclose(peak, f("peak_smart"), rtol=1e-12)
    if "peak_smart_old" in m and not info["fallback"]:       # no-fallback path must be unchanged
        np.testing.assert_allclose(peak, f("peak_smart_old"), rtol=1e-12)
    np.testing.assert_allclose(compute_beam_peak(img, "max"), f("peak_max"), rtol=1e-12)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v", "-p", "no:cacheprovider"]))
