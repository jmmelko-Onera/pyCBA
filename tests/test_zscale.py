"""Tests of normalize_image, clean_image and of the Z scale logic (pycba.core.zscale).

MATLAB comparison needs matlab_reference/zscale_all.mat from export_zscale_reference.m.
"""
import os
import sys
import warnings
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pytest

from pycba.core import (Filters, ImageData, clean_image, load_image, load_image_file, normalize_image, refresh_zscale, reinit_zscale,
                        select_zscale, update_image, zscale_for_peak)

REF = HERE / "matlab_reference"
ZS = [1, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]     # the real Zscales


# -------------------------------------------------------------- normalize_image
def test_normalize_methods():
    a, b, c = np.full((2, 3), 10.0), np.full((2, 3), 4.0), np.full((2, 3), 1.0)
    assert normalize_image(1, a, b, c) is a
    np.testing.assert_allclose(normalize_image(2, a, b), 6.0)
    np.testing.assert_allclose(normalize_image(3, a, b), 2.5)
    np.testing.assert_allclose(normalize_image(4, a, b, c), (10 - 4 - 1) / (4 * 2.0))


def test_normalize_division_by_zero_gives_inf_nan_like_matlab():
    a = np.array([[1.0, 0.0, -1.0]])
    b = np.zeros((1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")               # no numpy RuntimeWarning leaks
        out = normalize_image(3, a, b)
    assert out[0, 0] == np.inf and np.isnan(out[0, 1]) and out[0, 2] == -np.inf


def test_normalize_fringes_negative_denominator_is_zero_zero_denominator_is_not():
    a = np.array([[5.0, 5.0, 5.0]])
    b = np.array([[4.0, -4.0, 0.0]])
    c = np.array([[1.0, 1.0, 1.0]])
    out = normalize_image(4, a, b, c)
    assert out[0, 0] == pytest.approx((5 - 4 - 1) / 8.0)
    assert out[0, 1] == 0.0                           # B*C < 0
    assert not np.isfinite(out[0, 2])                 # B*C == 0: Inf/NaN as in MATLAB


def test_normalize_warns_and_returns_a_when_reference_missing_or_method_unknown():
    a = np.ones((2, 2))
    with pytest.warns(UserWarning, match="empty image2"):
        assert normalize_image(3, a, None) is a
    with pytest.warns(UserWarning, match="unknown method"):
        assert normalize_image(9, a, a) is a


# ------------------------------------------------------------------ clean_image
def test_clean_image_replaces_non_finite_by_the_finite_minimum():
    a = np.array([[3.0, np.nan, np.inf], [-2.0, -np.inf, 7.0]])
    out = clean_image(a)
    np.testing.assert_array_equal(out, [[3, -2, -2], [-2, -2, 7]])
    assert np.isnan(a[0, 1])                          # input untouched


def test_clean_image_without_any_finite_pixel_does_not_fail():
    np.testing.assert_array_equal(clean_image(np.full((2, 2), np.nan)), 0.0)


# ------------------------------------------------------------------- Z scale
@pytest.mark.parametrize("rmax, dimz", [
    (0.3, 1), (1.4, 256), (3, 256), (100, 256), (127, 256), (200, 256), (254, 256), (255, 256),
    (256, 512), (1000, 1024), (2000, 2048), (4095, 4096), (4096, 8192), (5000, 8192), (60000, 65536),
    (1e6, 131072),
])
def test_zscale_for_peak_is_the_first_scale_not_below_the_next_power_of_two(rmax, dimz):
    z = zscale_for_peak(rmax, ZS)
    assert z.dimz == dimz and not z.znormalize and ZS[z.menu_index - 1] == dimz


def test_zscale_small_peaks_are_not_saturated():
    """MATLAB's 'closest entry' gave scale 1 for any peak up to 127 (128 is closer to 1 than to 256)."""
    for peak in (2, 10, 100, 127):
        assert zscale_for_peak(peak, ZS).dimz >= peak


def test_zscale_short_lists():
    assert zscale_for_peak(300, [255, 1023]).dimz == 1023
    assert zscale_for_peak(5000, [255, 1023]).dimz == 1023          # larger than all: the largest


@pytest.mark.parametrize("rmax", [0.0, -3.0, np.nan])
def test_zscale_non_positive_peak_normalizes(rmax):
    z = zscale_for_peak(rmax, ZS)
    assert z.menu_index == 0 and z.znormalize and z.dimz == 1e-9


def test_reinit_zscale_sets_the_image_and_returns_menu_index():
    im = ImageData.from_raw(np.full((6, 8), 3000.0))
    im.znormalize = True                              # stale value must not survive
    idx = reinit_zscale(im, ZS)
    assert (im.dimz, im.znormalize, idx) == (4096.0, False, 6)
    im.filt = np.zeros((6, 8))
    assert reinit_zscale(im, ZS, "filt") == 0 and im.znormalize
    with pytest.raises(ValueError):
        reinit_zscale(im, ZS, "bad")


def test_refresh_and_select_zscale_roundtrip():
    im = ImageData.from_raw(np.ones((3, 3)))
    select_zscale(im, 4, ZS)
    assert (im.znormalize, im.dimz) == (False, 1024.0) and refresh_zscale(im, ZS) == 4
    im.dimz = 20000                                   # not a menu value: closest entry (16384)
    assert refresh_zscale(im, ZS) == 8
    select_zscale(im, 0, ZS)
    assert im.znormalize and refresh_zscale(im, ZS) == 0


# ------------------------------------------------------------------ load_image
def test_load_image_main_resets_energy_and_updates():
    imgs = [ImageData(), ImageData(), ImageData()]
    flt = Filters(energy=5.0)
    raw = np.full((5, 6), 100.0)
    res = load_image(imgs, 0, raw, flt, 1, ZS, filepath="a.csv")
    assert flt.energy == 1.0 and res.reset_energy and res.zscale_index == 2 and res.messages == []
    np.testing.assert_allclose(imgs[0].filt, raw)
    assert imgs[0].filepath == "a.csv"


def test_load_image_reference_refreshes_the_normalized_main_image():
    imgs = [ImageData(), ImageData(), ImageData()]
    flt = Filters()
    load_image(imgs, 0, np.full((4, 4), 10.0), flt, 1, ZS)
    res = load_image(imgs, 1, np.full((4, 4), 4.0), flt, 2, ZS)      # method 2: A - B
    assert not res.reset_energy
    np.testing.assert_allclose(imgs[0].filt, 6.0)
    load_image(imgs, 1, np.full((3, 3), 4.0), flt, 2, ZS)            # other size
    assert imgs[0].filt.shape == (4, 4)


def test_load_image_cleans_non_finite_pixels_first():
    imgs = [ImageData(), ImageData(), ImageData()]
    raw = np.full((4, 5), 50.0); raw[1, 1] = np.nan; raw[2, 2] = np.inf; raw[0, 0] = 10.0
    load_image(imgs, 0, raw, Filters(), 1, ZS)
    assert np.isfinite(imgs[0].raw).all() and imgs[0].raw[1, 1] == 10.0 and imgs[0].raw[2, 2] == 10.0


def test_load_image_keeps_the_background_when_it_still_fits():
    imgs = [ImageData(), ImageData(), ImageData()]
    imgs[0].bkg = 7.0
    load_image(imgs, 0, np.full((4, 5), 50.0), Filters(), 1, ZS)
    assert imgs[0].bkg == 7.0 and np.allclose(imgs[0].filt, 43.0)
    imgs[0].bkg = np.full((4, 5), 2.0)                       # background image of the same size: kept
    load_image(imgs, 0, np.full((4, 5), 50.0), Filters(), 1, ZS)
    assert np.allclose(imgs[0].filt, 48.0)
    res = load_image(imgs, 0, np.full((6, 6), 50.0), Filters(), 1, ZS)   # other size: reset, no crash
    assert np.allclose(imgs[0].filt, 50.0) and "Background reset (image size changed)." in res.messages


@pytest.mark.filterwarnings("ignore::pycba.io.AsciiFormatWarning")
def test_load_image_file_messages_and_errors(tmp_path):
    imgs = [ImageData(), ImageData(), ImageData()]
    flt = Filters()
    res = load_image_file(imgs, 0, tmp_path / "missing.csv", flt, 1, ZS)
    assert res.messages == ["File not found."] and res.cleared and not res.ok and imgs[0].raw is None

    one_col = tmp_path / "single.txt"
    one_col.write_text("1\n2\n3\n")
    res = load_image_file(imgs, 0, one_col, flt, 1, ZS, 2, 3)
    assert res.messages == ["single.txt may not be a 2D ASCII file."]
    assert not res.ok and not res.cleared

    good = tmp_path / "beam.csv"
    good.write_text("\n".join("\t".join(f"{(i + 1) * (j + 1) * 1.5:.3f}" for j in range(8)) for i in range(6)) + "\n")
    res = load_image_file(imgs, 0, good, flt, 1, ZS, 1, 1)
    assert res.ok and res.messages[0] == "Image opened."
    assert imgs[0].raw.shape == (6, 8) and imgs[0].filepath == str(good)
    assert imgs[0].raw[0, 1] == pytest.approx(3.0)


# ------------------------------------------------------------ MATLAB comparison
def test_against_matlab_zscale():
    scipy_io = pytest.importorskip("scipy.io")
    ref = REF / "zscale_all.mat"
    if not ref.exists():
        pytest.skip("no MATLAB reference exported yet (export_zscale_reference.m)")
    m = scipy_io.loadmat(ref)
    for k in range(int(m["n_norm"].item())):
        a, b, c = (m[f"n{k}_{x}"].astype(float) for x in "abc")
        method = int(m[f"n{k}_method"].item())
        np.testing.assert_allclose(normalize_image(method, a, b, c), m[f"n{k}_out"], rtol=1e-12,
                                   equal_nan=True, err_msg=f"normalize case {k}")
    np.testing.assert_array_equal(clean_image(m["clean_in"].astype(float)), m["clean_out"])
    zs = np.ravel(m["zscales"]).astype(float)
    for rmax, dimz, idx in zip(np.ravel(m["rmax"]), np.ravel(m["dimz"]), np.ravel(m["menu_value"])):
        if rmax > 0:                                   # (non-positive peaks: documented differences)
            z = zscale_for_peak(float(rmax), zs)
            assert (z.dimz, z.menu_index + 1) == (dimz, idx), f"rmax={rmax}"


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v", "-p", "no:cacheprovider"]))
