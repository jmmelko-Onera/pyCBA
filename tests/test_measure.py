"""Tests of pycba.core.measure and pycba.core.profile_files."""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))

import numpy as np
import pytest

from pycba.core import measure, profile_files, roi


def gauss(shape=(100, 120), cx=60.0, cy=50.0, sx=8.0, sy=5.0, amp=100.0):
    y, x = np.mgrid[1:shape[0] + 1, 1:shape[1] + 1]
    return amp * np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2)))


def test_energy_area_power_whole_image_and_mask():
    img = np.full((4, 5), 2.0)
    assert measure.aperture_energy(img, None, 0.5) == pytest.approx(2.0 * 20 * 0.25)
    assert measure.aperture_area(img.shape, None, 0.5) == pytest.approx(20 * 0.25)
    mask = np.zeros((4, 5), bool); mask[1:3, 1:4] = True
    assert measure.aperture_energy(img, mask, 0.5) == pytest.approx(2.0 * 6 * 0.25)
    assert measure.aperture_area(img.shape, mask, 0.5) == pytest.approx(6 * 0.25)
    assert measure.average_power(img, mask, 0.5) == pytest.approx(2.0)


def test_energy_ignores_nan_and_empty_aperture_power_is_nan():
    img = np.ones((3, 3)); img[1, 1] = np.nan
    assert measure.aperture_energy(img, None) == 8.0
    assert math.isnan(measure.average_power(img, np.zeros((3, 3), bool)))


def test_roi_center():
    assert measure.roi_center("rectangle", [2, 4, 10, 6]) == (7.0, 7.0)
    assert measure.roi_center("polygon", np.array([[0, 0], [10, 2], [4, 8]])) == (5.0, 4.0)


def test_size_xy_on_gaussian():
    img = gauss()
    r = measure.size_xy(img, None, 1, 1.0, (60, 50))
    assert r.x0 == pytest.approx(60, abs=0.01) and r.y0 == pytest.approx(50, abs=0.01)
    assert r.diam_h == pytest.approx(4 * 8, rel=0.01) and r.diam_v == pytest.approx(4 * 5, rel=0.01)
    assert r.gauss_flux == pytest.approx(100, rel=0.02)           # E/(2 pi sx sy) = amplitude of a Gaussian
    assert r.peak_flux == pytest.approx(100, rel=0.05)
    assert r.messages()[0] == "X0=60" and len(r.messages()) == 6
    assert r.clipboard_text().count("\t") == 1


def test_size_xy_in_an_aperture_and_fallback_centre():
    img = gauss()
    mask = roi.roi_mask(img.shape, "rectangle", [40, 35, 40, 30])
    r = measure.size_xy(img, mask, 1, 1.0, (60, 50))
    assert r.diam_h == pytest.approx(32, rel=0.08)          # the aperture (+-2.5 sigma) truncates the wings
    # centroid on a null pixel (aperture far from the beam): fallback centre is used, no crash
    far = roi.roi_mask(img.shape, "rectangle", [2, 2, 10, 10])
    r2 = measure.size_xy(np.zeros(img.shape) + 1e-9 * img, far, 1, 1.0, (7, 7))
    assert r2.energy >= 0
    zero = measure.size_xy(np.zeros(img.shape), None, 1, 1.0, (60, 50))   # energy 0: NaN centroid -> fallback
    assert zero.energy == 0


def test_peak_flux_is_taken_in_the_aperture():
    img = gauss()
    img[2:14, 2:14] = 500.0                                        # a bright spot outside the aperture (not a hot pixel)
    mask = roi.roi_mask(img.shape, "rectangle", [40, 35, 40, 30])
    inside = measure.size_xy(img, mask, 1, 1.0, (60, 50))
    whole = measure.size_xy(img, mask, 1, 1.0, (60, 50), peak_image=img)
    assert inside.peak_flux == pytest.approx(100, rel=0.1) and whole.peak_flux > 400


def test_nan_pixels_count_in_the_area_but_not_in_the_energy():
    img = np.ones((10, 10)); img[:5] = np.nan
    assert measure.aperture_area(img.shape, None) == 100 and measure.aperture_energy(img, None) == 50
    assert measure.average_power(img, None) == 0.5               # under-estimated on purpose: the pixels exist


def test_size_xy_every_method_and_unknown_method():
    img = gauss()
    for m in (1, 2, 3, 4):
        r = measure.size_xy(img, None, m, 1.0, (60, 50))
        assert r.diam_h == pytest.approx(32, rel=0.2), m
    r = measure.size_xy(img, None, 9, 1.0, (60, 50))
    assert (r.x0, r.diam_h, r.diam_v) == (0, 0, 0)


def test_size_xy_has_no_principal_axes_any_more():
    r = measure.size_xy(gauss(), None, 1, 1.0, (60, 50))
    assert not hasattr(r, "principal") and [m.split("=")[0] for m in r.messages()] == [
        "X0", "Y0", "DiamH", "DiamV", "Gauss Flux", "Peak Flux"]


def test_size_angle_on_a_tilted_gaussian_and_its_drawing():
    """The ellipse drawn with ``angle_xy`` has its long axis on the measured direction (here the (1, 1) diagonal:
    down and to the right on screen, i.e. a NEGATIVE angle)."""
    y, x = np.mgrid[1:101, 1:101]
    u = ((x - 50) + (y - 50)) / math.sqrt(2)
    v = ((x - 50) - (y - 50)) / math.sqrt(2)
    img = 1000 * np.exp(-u ** 2 / (2 * 12 ** 2) - v ** 2 / (2 * 4 ** 2))
    r = measure.size_angle(img, None, 1.0, (50, 50))
    p = r.principal
    assert p.d_major == pytest.approx(48, rel=0.01) and p.d_minor == pytest.approx(16, rel=0.01)
    assert p.angle == pytest.approx(-45, abs=0.3) and p.ellipticity == pytest.approx(1 / 3, rel=0.01)
    assert r.x0 == pytest.approx(50, abs=0.01) and r.y0 == pytest.approx(50, abs=0.01)
    ex, ey = measure.ellipse_points(r.x0, r.y0, p.d_major, p.d_minor, p.angle_xy)
    far = np.argmax(np.hypot(ex - r.x0, ey - r.y0))
    assert (ex[far] - r.x0) * (ey[far] - r.y0) > 0                 # long axis on the (1, 1) diagonal
    assert r.messages() == ["X0=50", "Y0=50", "Major=48", "Minor=16", "Angle=-45.0 deg", "Ellipticity=0.333"]


def test_size_angle_in_an_aperture_and_fallback_centre():
    img = gauss()
    mask = np.zeros(img.shape, bool); mask[:, :55] = True            # only the left part of the beam
    r = measure.size_angle(img, mask, 1.0, (30, 40))
    assert 0 < r.x0 < 55 and math.isfinite(r.principal.d_major)
    empty = measure.size_angle(img, np.zeros(img.shape, bool), 1.0, (30, 40))   # nothing under the aperture
    assert (empty.x0, empty.y0) == (30, 40) and math.isnan(empty.principal.ellipticity)


def test_ellipse_points_closed_and_axis_aligned():
    x, y = measure.ellipse_points(10, 20, 6, 4)
    assert x.max() == pytest.approx(13, abs=1e-3) and y.min() == pytest.approx(18, abs=1e-3)
    assert x[0] == pytest.approx(x[-1], abs=1e-2)


# ------------------------------------------------------------------ profile files
def test_profile_paths_naming(tmp_path):
    img = tmp_path / "beam.txt"
    p = profile_files.profile_paths("hv", img, 30, 40)
    assert p["Horizontal"].name == "Hprofile_at_Vpx30.txt" and p["Vertical"].name == "Vprofile_at_Hpx40.txt"
    assert p["Horizontal"].parent == tmp_path
    p = profile_files.profile_paths("45", img, 30, 40)
    assert p["Minus"].name == "minusProfile_at_Hpx40_Vpx30.txt"
    p = profile_files.profile_paths("l1l2", img, 30, 40, 12.34)
    assert p["L1"].name == "L1Profile_at_Hpx40_Vpx30_A12.3.txt"
    assert profile_files.profile_paths("hv", "", 1, 1) == {}              # image not from a file
    assert profile_files.profile_paths("hv", "C:\\data\\beam.txt", 1, 1)["Horizontal"].name == "Hprofile_at_Vpx1.txt"


def test_save_profile_roundtrip_and_unwritable(tmp_path):
    f = tmp_path / "p.txt"
    msg = profile_files.save_profile(f, [1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
    assert msg.startswith("Saved")
    data = np.loadtxt(f)
    assert data.shape == (3, 2) and data[2, 1] == pytest.approx(2.5)
    assert profile_files.save_profile(tmp_path / "no_such_dir" / "p.txt", [1], [1]).startswith("/!\\Could not save")


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v", "-p", "no:cacheprovider"]))
