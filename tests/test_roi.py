"""Tests of pycba.core.roi (aperture masks, default / automatic aperture, crosshair geometry)."""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))

import numpy as np
import pytest

from pycba.core import roi


def beam(shape=(120, 160), cx=80.0, cy=60.0, sx=10.0, sy=7.0, amp=1000.0):
    y, x = np.mgrid[1:shape[0] + 1, 1:shape[1] + 1]
    return amp * np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2)))


# ------------------------------------------------------------------ masks
def test_rect_mask_pixel_centres():
    m = roi.roi_mask((10, 10), "rectangle", [2, 3, 4, 2])          # x 2..6, y 3..5 (centres i*ps)
    ys, xs = np.nonzero(m)
    assert xs.min() + 1 == 2 and xs.max() + 1 == 6 and ys.min() + 1 == 3 and ys.max() + 1 == 5


def test_mask_scales_with_pixelsize():
    a = roi.roi_mask((20, 20), "ellipse", [3, 4, 10, 8], 1.0)
    b = roi.roi_mask((20, 20), "ellipse", [6, 8, 20, 16], 2.0)
    assert np.array_equal(a, b)


def test_ellipse_and_circle_masks():
    m = roi.roi_mask((41, 41), "circle", [11, 11, 20, 20])
    assert m[20, 20] and not m[0, 0] and not m[10, 10]            # centre in, corner of the bounding box out
    area = m.sum()
    assert abs(area - math.pi * 100) < 0.05 * math.pi * 100


def test_polygon_mask_triangle():
    m = roi.roi_mask((20, 20), "polygon", np.array([[2.0, 2.0], [18.0, 2.0], [2.0, 18.0]]))
    assert m[4, 4] and m[2, 2] and not m[16, 16] and not m[0, 0]  # edge points count as inside
    assert abs(m.sum() - 0.5 * 16 * 16) < 40


# ------------------------------------------------------------------ default aperture
def test_default_roi_centred_on_peak_and_symmetric():
    img = beam(cx=40, cy=30)
    x, y, w, h = roi.default_roi(img)
    assert abs(x + w / 2 - 40) < 1e-9 and abs(y + h / 2 - 30) < 1e-9
    # the same beam at the mirrored position gives the same size (MATLAB looked at the right edge only)
    x2, y2, w2, h2 = roi.default_roi(img[:, ::-1])
    assert abs(w - w2) < 1e-9 and abs(h - h2) < 1e-9
    (xmin, xmax), (ymin, ymax) = roi.image_limits(img.shape, 1.0)
    assert x >= xmin and x + w <= xmax and y >= ymin and y + h <= ymax


def test_default_roi_minimum_size_in_pixels():
    img = beam(cx=1, cy=1)
    _, _, w, h = roi.default_roi(img, 5.5)
    assert w >= 2 * 5.5 and h >= 2 * 5.5


# ------------------------------------------------------------------ auto aperture
@pytest.mark.parametrize("kind", ["rectangle", "ellipse"])
def test_auto_aperture_converges_to_twice_the_diameter(kind):
    img = beam(sx=8, sy=5)
    start = roi.default_roi(img)
    pos, ok, n = roi.auto_aperture(img, kind, start, 1)
    assert ok and n < 10
    x, y, w, h = pos
    assert abs(w - 2 * 4 * 8) < 0.1 * 2 * 4 * 8 and abs(h - 2 * 4 * 5) < 0.1 * 2 * 4 * 5
    assert abs(x + w / 2 - 80) < 1e-9 and abs(y + h / 2 - 60) < 1e-9     # centre kept


def test_auto_aperture_changes_width_and_height_together():
    """MATLAB lost the width change when the height changed too."""
    img = beam(sx=12, sy=4)
    pos, ok, _ = roi.auto_aperture(img, "rectangle", [70, 55, 20, 20], 1)
    assert ok and pos[2] > 1.5 * pos[3]


def test_auto_aperture_square_stays_square_and_inside():
    img = beam(cx=30, cy=60, sx=15, sy=15)                 # big beam near the left edge
    pos, _, _ = roi.auto_aperture(img, "square", roi.default_roi(img), 1)
    x, y, w, h = pos
    assert abs(w - h) < 1e-9
    (xmin, xmax), (ymin, ymax) = roi.image_limits(img.shape, 1.0)
    assert x >= xmin - 1e-9 and x + w <= xmax + 1e-9 and y >= ymin - 1e-9 and y + h <= ymax + 1e-9


def test_auto_aperture_not_converged_reported():
    img = beam(cx=30, cy=60, sx=40, sy=40)                 # target larger than any allowed aperture
    pos, ok, n = roi.auto_aperture(img, "ellipse", [20, 50, 10, 10], 1, max_iter=2)
    assert not ok and n == 2


@pytest.mark.parametrize("kind", ["rectangle", "ellipse", "square", "circle"])
@pytest.mark.parametrize("ps", [1.0, 5.5])
def test_auto_aperture_never_collapses_below_4_pixels(kind, ps):
    """A diameter of 0 (what the 4sigma method can give on a noisy / negative image) used to shrink the aperture to a
    one-pixel line, which cannot be grabbed with the mouse."""
    from pycba.core.beam_metrics import BeamDiameters
    zero = lambda img, method, pixelsize: BeamDiameters(0.0, 0.0, 0.0, 0.0)
    img = beam()
    start = [30 * ps, 20 * ps, 60 * ps, 40 * ps]
    pos, ok, _ = roi.auto_aperture(img, kind, start, 1, ps, diameters=zero)
    assert roi.MIN_AUTO_APERTURE == 4
    assert pos[2] == pytest.approx(4 * ps) and pos[3] == pytest.approx(4 * ps)
    assert ok                                                               # stable at the minimum, not an endless shrink
    nan = lambda img, method, pixelsize: BeamDiameters(0.0, 0.0, math.nan, math.nan)
    pos, _, _ = roi.auto_aperture(img, kind, start, 1, ps, diameters=nan)
    assert pos[2] >= 4 * ps - 1e-9 and pos[3] >= 4 * ps - 1e-9


def test_auto_aperture_above_the_minimum_is_unchanged():
    img = beam(sx=10, sy=7)
    pos, ok, _ = roi.auto_aperture(img, "rectangle", [60, 45, 40, 30], 1)
    assert ok and pos[2] > 4 and pos[3] > 4


def test_auto_aperture_rejects_polygon():
    with pytest.raises(ValueError):
        roi.auto_aperture(beam(), "polygon", np.zeros((3, 2)), 1)


# ------------------------------------------------------------------ crosshair
XL, YL = (0.5, 160.5), (0.5, 120.5)


def test_h_and_v_lines_clamped():
    assert np.allclose(roi.cross_h(50, XL, YL), [[0.5, 50], [160.5, 50]])
    assert np.allclose(roi.cross_h(-9, XL, YL), [[0.5, 0.5], [160.5, 0.5]])
    assert np.allclose(roi.cross_v(1e9, XL, YL), [[160.5, 0.5], [160.5, 120.5]])


@pytest.mark.parametrize("x,y", [(80, 60), (20, 100), (150, 10), (5, 5)])
def test_diagonals_lie_on_the_line_and_on_the_box(x, y):
    p = roi.cross_diag_plus(x, y, XL, YL)
    assert np.allclose(p.sum(axis=1), x + y)
    q = roi.cross_diag_minus(x, y, XL, YL)
    assert np.allclose(q[:, 0] - q[:, 1], x - y)
    for pts in (p, q):
        for px, py in pts:
            assert XL[0] - 1e-9 <= px <= XL[1] + 1e-9 and YL[0] - 1e-9 <= py <= YL[1] + 1e-9
            on_border = min(abs(px - XL[0]), abs(px - XL[1]), abs(py - YL[0]), abs(py - YL[1])) < 1e-9
            assert on_border


def test_diagonal_far_outside_is_clamped_to_a_corner():
    p = roi.cross_diag_plus(1e6, 1e6, XL, YL)
    assert np.allclose(p, [[160.5, 120.5], [160.5, 120.5]])


@pytest.mark.parametrize("theta", [0.0, math.pi / 2, math.pi / 4, 2.0, -1.0, math.pi])
def test_l1l2_lines_are_perpendicular_through_the_centre(theta):
    c, l1, l2 = roi.cross_l1l2((70, 50), theta, XL, YL)
    assert np.allclose(c, (70, 50))
    for line in (l1, l2):
        a, b = line - c
        assert abs(a[0] * b[1] - a[1] * b[0]) < 1e-9 and a @ b < 0       # collinear, on both sides of the centre
        for px, py in line:
            assert min(abs(px - XL[0]), abs(px - XL[1]), abs(py - YL[0]), abs(py - YL[1])) < 1e-9
    d1, d2 = l1[0] - c, l2[0] - c
    assert abs(d1 @ d2) < 1e-9


def test_l1l2_horizontal_arm_no_division_by_zero():
    c, l1, l2 = roi.cross_l1l2((70, 50), 0.0, XL, YL)
    assert np.all(np.isfinite(l1)) and np.all(np.isfinite(l2))
    assert np.allclose(sorted(l1[:, 0]), [0.5, 160.5]) and np.allclose(l1[:, 1], 50)


def test_l1l2_centre_clamped_and_angle_from_mouse():
    c, _, _ = roi.cross_l1l2((1e6, -5), 0.3, XL, YL)
    assert np.allclose(c, (160.5, 0.5))
    assert abs(roi.angle_from_mouse((0, 0), (1, 1), 0.2) - math.pi / 4) < 1e-12
    assert roi.angle_from_mouse((3, 3), (3, 3), 0.2) == 0.2


@pytest.mark.parametrize("kind", ["+", "X", "~"])
def test_default_crosshair_is_centred(kind):
    out = roi.default_crosshair(kind, XL, YL)
    assert all(np.all(np.isfinite(o)) for o in out)


def test_crosshair_centres_and_scaling():
    ch = roi.Crosshair.default("+", XL, YL)
    assert ch.center() == (80.5, 60.5)
    for kind in ("X", "~"):
        c = roi.Crosshair.default(kind, XL, YL)
        c.kind = kind
        assert np.allclose(c.center(), (80.5, 60.5))
    assert roi.Crosshair("O").center() is None
    s = ch.scaled(2.0)
    assert s.center() == (161.0, 121.0) and ch.center() == (80.5, 60.5)


def test_crosshair_X_center_is_the_intersection_of_the_diagonals():
    ch = roi.Crosshair("X", c=100 + 40, d=100 - 40)               # x + y = 140, x - y = 60 -> (100, 40)
    assert np.allclose(ch.center(), (100, 40))
    p = roi.cross_diag_plus(100, 40, XL, YL)
    q = roi.cross_diag_minus(100, 40, XL, YL)
    assert np.allclose(p.sum(axis=1), 140) and np.allclose(q[:, 0] - q[:, 1], 60)


def test_crosshair_clamp_and_within():
    ch = roi.Crosshair("+", x=1e6, y=-5)
    assert not ch.within(XL, YL)
    ch.clamp(XL, YL)
    assert ch.center() == (160.5, 0.5) and ch.within(XL, YL)


def test_crosshair_l1l2_lines():
    ch = roi.Crosshair("~", cx=70, cy=50, theta=math.pi / 3)
    l1, l2 = ch.l1l2_lines(XL, YL)
    assert np.all(np.isfinite(l1)) and np.all(np.isfinite(l2))


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v", "-p", "no:cacheprovider"]))
