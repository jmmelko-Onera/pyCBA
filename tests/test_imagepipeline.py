"""Tests of pycba.core.imagepipeline: update_image, box filter, display scale, ROI/crosshair geometry.

MATLAB comparison needs matlab_reference/pipeline_*.mat from export_pipeline_reference.m.
"""
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pytest

from pycba.core import (Filters, ImageData, box_filter, compute_beam_peak, compute_flicker, crosshair_center, display_scale,
                        is_cross_within, is_roi_within, update_image)

REF = HERE / "matlab_reference"


def naive_box(a, order, padding):
    """Reference box filter written with explicit loops."""
    k = int(np.floor(order / 2) * 2 + 1)
    h = k // 2
    pad = {"zeros": dict(mode="constant", constant_values=0), "replicate": dict(mode="edge"),
           "symmetric": dict(mode="symmetric"), "circular": dict(mode="wrap")}[padding]
    ap = np.pad(a, h, **pad)
    out = np.empty_like(a, dtype=float)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = ap[i:i + k, j:j + k].mean()
    return out


# ----------------------------------------------------------------- box filter
@pytest.mark.parametrize("padding", ["zeros", "replicate", "symmetric", "circular"])
@pytest.mark.parametrize("order", [1, 2, 3, 5, 6])
def test_box_filter_matches_naive_loops(order, padding):
    a = np.random.default_rng(order).uniform(0, 1000, (17, 23))
    np.testing.assert_allclose(box_filter(a, order, padding), naive_box(a, order, padding), rtol=1e-11)


def test_box_filter_window_size_is_odd():
    a = np.random.default_rng(0).uniform(size=(9, 9))
    np.testing.assert_allclose(box_filter(a, 2), box_filter(a, 3))        # floor(2/2)*2+1 = 3
    np.testing.assert_array_equal(box_filter(a, 1), a)                    # size 1: identity


def test_box_filter_nan_only_touches_its_window():
    a = np.ones((9, 9))
    a[4, 4] = np.nan
    out = box_filter(a, 3, "replicate")            # no edge effect: isolates the effect of the NaN
    window = np.zeros((9, 9), bool)
    window[3:6, 3:6] = True
    assert np.isnan(out[window]).all()             # every window containing the NaN
    np.testing.assert_allclose(out[~window], 1.0)  # nothing else, in particular not the rest of the line


# ---------------------------------------------------------------- update_image
def make(raw, **kw):
    return ImageData(raw=np.asarray(raw, float), **kw)


def test_update_image_steps_and_order():
    raw = np.full((4, 5), 100.0)
    dead = np.ones((4, 5)); dead[1, 2] = 0
    im = make(raw, flicker=2.0, deadmask=dead, bkg=10.0)
    update_image([im, ImageData(), ImageData()], 0, 1, Filters(energy=3.0))
    expected = np.full((4, 5), (100 - 2 - 10) * 3.0)
    expected[1, 2] = 0.0                               # a dead pixel ends at 0 (MATLAB: -bkg*energy = -30)
    np.testing.assert_allclose(im.filt, expected)


def test_default_flicker_is_zero_so_the_image_is_not_shifted():
    """MATLAB set flicker = ones and subtracted it: every image was shifted by -1."""
    raw = np.full((3, 3), 10.0)
    for im in (ImageData(raw=raw), ImageData.from_raw(raw)):
        update_image([im, ImageData(), ImageData()], 0, 1, Filters())
        np.testing.assert_array_equal(im.filt, raw)


def test_image_data_from_raw_builds_a_clean_entry():
    raw = np.random.default_rng(0).uniform(0, 100, (6, 40))
    im = ImageData.from_raw(raw)
    assert im.deadmask.shape == raw.shape and im.deadmask.all() and not im.flicker.any()
    imf = ImageData.from_raw(raw, deflicker=True)
    np.testing.assert_array_equal(imf.flicker, compute_flicker(raw))
    assert (imf.dim_y, imf.dim_x) == (6, 40)


# ------------------------------------------------------------------- flicker
def test_compute_flicker_is_the_row_mean_of_the_edge_columns_repeated():
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 100, (8, 45))                      # cols=45 -> k=3
    fl = compute_flicker(img)
    assert fl.shape == img.shape
    rows = img[:, [0, 1, 2, 42, 43, 44]].mean(axis=1)                # 3 columns on each side
    np.testing.assert_allclose(fl, np.repeat(rows[:, None], 45, axis=1))


def test_compute_flicker_removes_a_row_offset():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.1, (10, 100))
    offsets = rng.uniform(-5, 5, (10, 1))
    img = base + offsets                                     # beam-free edges: flicker == offsets
    fl = compute_flicker(img)
    np.testing.assert_allclose(fl[:, 0], offsets[:, 0], atol=0.2)
    assert np.abs(img - fl).max() < 0.5


def test_compute_flicker_edge_slices_have_the_same_width():
    """MATLAB's right slice had 2 columns more than the left one (typo): both have ceil(5 %) here."""
    img = np.tile(np.arange(60, dtype=float), (3, 1))        # column index as value; cols=60 -> k=3
    expected = np.mean([0, 1, 2, 57, 58, 59])
    np.testing.assert_allclose(compute_flicker(img)[0, 0], expected)


def test_compute_flicker_roi_mask_and_narrow_images():
    img = np.ones((4, 40))
    roi = np.zeros((4, 40)); roi[:, 10:30] = 1               # edge columns outside the ROI
    assert not compute_flicker(img, roi).any()
    for cols in (1, 2, 3, 10):                                # tiny images work (slices may overlap)
        np.testing.assert_allclose(compute_flicker(np.full((3, cols), 7.0)), 7.0)


def test_update_image_spatial_filter_applied_last():
    raw = np.zeros((7, 7)); raw[3, 3] = 9.0
    im = make(raw, flicker=0.0)
    update_image([im, ImageData(), ImageData()], 0, 1, Filters(filtorder=3), padding="zeros")
    assert im.filt[3, 3] == pytest.approx(1.0) and im.filt[2, 2] == pytest.approx(1.0) and im.filt[0, 0] == 0


def test_update_image_empty_raw_is_a_noop():
    im = ImageData()
    assert update_image([im, ImageData(), ImageData()], 0, 1, Filters()) == [] and im.filt is None


def _set(a_shape, b_shape=None, c_shape=None):
    a = make(np.ones(a_shape), flicker=0.0)
    b = ImageData(filt=None if b_shape is None else np.ones(b_shape))
    c = ImageData(filt=None if c_shape is None else np.ones(c_shape))
    return [a, b, c]


@pytest.mark.parametrize("method,shapes,msg", [
    (2, ((4, 5), None, None), "No Ref. img -> skipping."),
    (2, ((4, 5), (4, 6), None), "Img A&B size mismatch -> skipping."),
    (4, ((4, 5), (4, 5), None), "No Ref C -> Skipping."),
    (4, ((4, 5), (4, 5), (3, 5)), "Img A&C size mismatch -> skipping."),
])
def test_update_image_normalization_checks(method, shapes, msg):
    imgs = _set(*shapes)
    calls = []
    out = update_image(imgs, 0, method, Filters(), normalizer=lambda *a: calls.append(a))
    assert out == [msg] and not calls
    np.testing.assert_array_equal(imgs[0].filt, np.ones(shapes[0]))        # image left un-normalised


def test_update_image_normalizer_is_called_for_the_main_image_only():
    imgs = _set((4, 5), (4, 5), (4, 5))
    out = update_image(imgs, 0, 3, Filters(energy=2.0), normalizer=lambda m, a, b, c: a / b)
    assert out == [] and imgs[0].filt.shape == (4, 5) and np.allclose(imgs[0].filt, 2.0)
    imgs[1].raw = np.full((4, 5), 6.0)
    imgs[1].flicker = 0.0
    update_image(imgs, 1, 3, Filters(), normalizer=lambda *a: pytest.fail("not for a reference image"))
    assert np.allclose(imgs[1].filt, 6.0)


def test_update_image_uses_normalize_image_by_default():
    imgs = _set((4, 5), (4, 5))                    # A, B, C filled by the helper
    update_image(imgs, 1, 1, Filters())
    update_image(imgs, 0, 2, Filters())
    np.testing.assert_allclose(imgs[0].filt, imgs[0].raw - imgs[1].filt)


def test_update_image_size_check_uses_the_new_image_not_the_stale_filt():
    imgs = _set((4, 5), (4, 5))
    imgs[0].filt = np.ones((9, 9))                 # stale filtered A from a previous (other size) image
    assert update_image(imgs, 0, 3, Filters(), normalizer=lambda m, a, b, c: a) == []


# --------------------------------------------------------------- display scale
def beam(shape=(40, 60), peak=1000.0):
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    return peak * np.exp(-((x - 30) ** 2 / (2 * 8**2) + (y - 20) ** 2 / (2 * 5**2))) + 20 + rng.normal(0, 2, shape)


def spec(r, **kw):
    a = dict(image_num=1, method=1, znormalize=True, dimz=4095.0, log=False, colormap_len=0, pixelsize=5.5)
    a.update(kw)
    return display_scale(r, **a)


def test_display_linear_without_palette():
    r = beam()
    s = spec(r)
    assert (s.vmin, s.vmax) == (0.0, pytest.approx(r.max() + 1e-9))
    s = spec(r, znormalize=False)
    assert s.vmax == pytest.approx(4095.0 + 1e-9) and s.data is not None and not s.custom_colormap


def test_display_linear_with_palette():
    r, n = beam(), 256
    z = compute_beam_peak(r, "max")
    s = spec(r, colormap_len=n)
    vmin = -0.5 * z / n
    assert s.vmin == pytest.approx(vmin) and s.vmax == pytest.approx(z * n / (n - 1) - vmin + 1e-9) and s.custom_colormap


def test_display_peak_mode_depends_on_method_and_image_number():
    r = beam()
    r[3, 3] = 1e6                                                    # hot pixel
    assert spec(r, method=1).vmax == pytest.approx(1e6 + 1e-9)                       # 'max'
    assert spec(r, method=3).vmax == pytest.approx(compute_beam_peak(r, "smart") + 1e-9)
    assert spec(r, method=3).vmax < 2000                                              # hot pixel ignored
    assert spec(r, method=3, image_num=2).vmax == pytest.approx(1e6 + 1e-9)           # reference image: 'max'
    assert spec(r, method=4).vmax == pytest.approx(compute_beam_peak(r, 3.5) + 1e-9)


def test_display_fringes_are_symmetric():
    r = beam() - 500
    s = spec(r, method=4, colormap_len=64)
    z = compute_beam_peak(r, 3.5)
    assert s.vmax == pytest.approx(z * 64 / 63 + 1e-9) and s.vmin == pytest.approx(-s.vmax)
    s = spec(r, method=4)
    assert s.vmin == 0.0                                              # no palette: [0, Zmax], as MATLAB


def test_display_log():
    r = beam()
    z = compute_beam_peak(r, "max")
    s = spec(r, log=True)
    assert s.log and s.vmin == pytest.approx(np.log10(z / 10000)) and s.vmax == pytest.approx(np.log10(z))
    np.testing.assert_allclose(s.data, np.log10(np.maximum(r, 0)))
    n = 100
    s = spec(r, log=True, colormap_len=n)
    lo, hi = np.log10(z / 10000), np.log10(z)
    assert s.vmax == pytest.approx(hi + 2 * (hi - lo) / n) and s.vmin == pytest.approx(lo)


def test_display_symlog_for_fringes_and_log_of_nonpositive_zmax():
    r = beam() - 500
    z = compute_beam_peak(r, 3.5)
    s = spec(r, method=4, log=True)
    np.testing.assert_allclose(s.data, np.sign(r) * np.log10(1 + np.abs(r)))
    assert s.vmax == pytest.approx(np.log10(1 + z)) and s.vmin == pytest.approx(-s.vmax)
    neg = -np.ones((5, 5))
    s = spec(neg, log=True)                                           # MATLAB: complex log10
    assert np.isfinite([s.vmin, s.vmax]).all() and s.vmax == pytest.approx(-9.0)


def test_display_empty_image_and_extent_conventions():
    s = display_scale(None, 1, 1, True, 4095, False, 0, 5.5)
    assert s.empty and (s.vmin, s.vmax) == (0.0, 100.0)
    r = np.ones((10, 20))
    assert spec(r).extent == (0.5 * 5.5, 20.5 * 5.5, 10.5 * 5.5, 0.5 * 5.5)       # pixel i centred at i*ps
    # the analysis puts the centroid of a one-pixel image at pixel (3, 5) at (5*ps, 3*ps): the drawing must agree
    r2 = np.zeros((10, 20)); r2[2, 4] = 1.0
    s = spec(r2)
    l, rr, b, t_ = s.extent
    col_center = l + (4 + 0.5) * (rr - l) / 20
    row_center = t_ + (2 + 0.5) * (b - t_) / 10
    assert (col_center, row_center) == (pytest.approx(5 * 5.5), pytest.approx(3 * 5.5))


# ---------------------------------------------------------- ROI / crosshair
XL, YL = (0.0, 100.0), (0.0, 50.0)


@pytest.mark.parametrize("pos,expected", [
    ([10, 10, 20, 20], True), ([0, 0, 100, 50], True), ([-1, 0, 10, 10], False), ([95, 10, 10, 10], False),
    ([10, 45, 10, 10], False), (None, True), ([], True),
])
def test_is_roi_within_rectangle(pos, expected):
    assert is_roi_within(XL, YL, pos) is expected


@pytest.mark.parametrize("n", [3, 4, 6])
def test_is_roi_within_polygons_of_any_size(n):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inside = np.c_[50 + 20 * np.cos(t), 25 + 20 * np.sin(t)]
    outside = inside.copy(); outside[0, 0] = 120
    assert is_roi_within(XL, YL, inside) is True and is_roi_within(XL, YL, outside) is False


def test_is_cross_within_matrix_and_vector_agree_and_check_all_coordinates():
    m = np.array([[10.0, 40.0], [60.0, 5.0]])                       # line going up: y2 < y1
    assert is_cross_within(XL, YL, m) is True
    assert is_cross_within(XL, YL, m.ravel()[[0, 1, 2, 3]]) is True
    assert is_cross_within(XL, YL, [10.0, 40.0, 60.0, 5.0]) is True
    assert is_cross_within(XL, YL, np.array([[10.0, 60.0], [60.0, 5.0]])) is False   # y1 outside
    assert is_cross_within(XL, YL, np.array([[0.0, 10.0], [60.0, 5.0]])) is True     # on the limit: within
    assert is_cross_within(XL, YL, np.array([[-0.001, 10.0], [60.0, 5.0]])) is False
    assert is_cross_within(XL, YL, np.array([[-1e-12, 10.0], [100.0 + 1e-12, 5.0]])) is True   # rounding after a rescale
    assert is_cross_within(XL, YL, None) is True
    assert is_cross_within(XL, YL, [50.0, 25.0]) is True                     # a single point [x y]
    assert is_cross_within(XL, YL, [150.0, 25.0]) is False


def test_default_crosshair_spanning_the_axes_is_within():
    """createCrosshair puts the ends of the '+' lines exactly on XLim/YLim."""
    h = np.array([[XL[0], 25.0], [XL[1], 25.0]])
    v = np.array([[50.0, YL[0]], [50.0, YL[1]]])
    assert is_cross_within(XL, YL, h) and is_cross_within(XL, YL, v)
    scale = 1.7 / 1.3                                             # pixel size change: positions and limits rescaled
    xl2, yl2 = tuple(np.array(XL) * scale), tuple(np.array(YL) * scale)
    assert is_cross_within(xl2, yl2, h * 1.7 / 1.3 * (1 + 1e-14))


def test_crosshair_center():
    assert crosshair_center("+", [[10, 20], [90, 20]], [[30, 5], [30, 45]]) == (30.0, 20.0)
    anti = np.array([[10.0, 50.0], [40.0, 20.0]])                   # y + x = 60
    diag = np.array([[10.0, 20.0], [40.0, 50.0]])                   # y - x = 10
    assert crosshair_center("X", anti, diag) == (25.0, 35.0)        # intersection
    assert crosshair_center("~", None, None, [12.5, 7.5]) == (12.5, 7.5)
    assert crosshair_center("?", None, None) is None
    assert crosshair_center("~", None, None, None) is None


# ------------------------------------------------------------------ MATLAB
@pytest.mark.parametrize("ref", sorted(REF.glob("pipeline_*.mat")) or [None])
def test_against_matlab_pipeline(ref):
    scipy_io = pytest.importorskip("scipy.io")
    if ref is None:
        pytest.skip("no MATLAB pipeline reference exported yet (export_pipeline_reference.m)")
    m = scipy_io.loadmat(ref)
    f = lambda k: np.ravel(m[k]).astype(float)        # noqa: E731
    img = m["image"].astype(float)

    # 1. every padding option of imboxfilt, checked against MY implementation of that option
    for order in (1, 2, 3, 5, 6):
        for pad in ("zeros", "replicate", "symmetric", "circular"):
            np.testing.assert_allclose(box_filter(img, order, pad), m[f"box_o{order}_{pad}"], rtol=1e-10,
                                       err_msg=f"order {order}, padding {pad}")
    # 2. which padding is imboxfilt's DEFAULT?
    matches = [p for p in ("zeros", "replicate", "symmetric", "circular")
               if np.allclose(m["box_o3_default"], m[f"box_o3_{p}"], rtol=1e-10)]
    # order 3 cannot tell 'replicate' from 'symmetric' (1-pixel border): MATLAB documents 'replicate' as the default
    assert "replicate" in matches and "zeros" not in matches, \
        f"imboxfilt default padding corresponds to {matches}: change update_image's default"
    if "box_o5_default" in m:        # order 5 tells 'replicate' from 'symmetric' (2-pixel border)
        matches5 = [p for p in ("zeros", "replicate", "symmetric", "circular")
                    if np.allclose(m["box_o5_default"], m[f"box_o5_{p}"], rtol=1e-10)]
        assert matches5 == ["replicate"], f"order 5: imboxfilt default padding corresponds to {matches5}"
    # 3. NaN handling
    np.testing.assert_allclose(box_filter(m["image_nan"].astype(float), 3), m["box_nan_o3_default"],
                               rtol=1e-10, equal_nan=True)

    # 4. computeFlicker, then update_image (method 1) with the corrected order
    np.testing.assert_allclose(compute_flicker(m["raw"].astype(float)), m["flicker"], rtol=1e-12)
    np.testing.assert_allclose(compute_flicker(m["raw"].astype(float), m["roimask"].astype(float)),
                               m["flicker_roi"], rtol=1e-12)
    im = ImageData(raw=m["raw"].astype(float), flicker=m["flicker"].astype(float), deadmask=m["deadmask"].astype(float),
                   bkg=float(f("bkg")[0]))
    update_image([im, ImageData(), ImageData()], 0, 1, Filters(energy=float(f("energy")[0]), filtorder=int(f("filtorder")[0])))
    np.testing.assert_allclose(im.filt, m["rfilt_ref"], rtol=1e-10)

    # 5. display limits
    rd = m["disp_image"].astype(float)
    for row, lim, chk in zip(m["disp_params"], m["disp_lim"], np.ravel(m["disp_check"])):
        num, method, zn, dimz, lg, cm = row
        s = display_scale(rd, int(num), int(method), bool(zn), float(dimz), bool(lg), int(cm), 5.5)
        np.testing.assert_allclose([s.vmin, s.vmax], lim, rtol=1e-10, err_msg=str(row))
        fin = np.isfinite(s.data)
        assert s.data[fin].sum() == pytest.approx(chk, rel=1e-10), str(row)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v", "-p", "no:cacheprovider"]))
