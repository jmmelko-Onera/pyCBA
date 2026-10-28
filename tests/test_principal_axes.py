"""Tests of the ISO 11146 principal axes (core.compute_principal_axes) and of the ISO Beam size angle button."""
import math
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from pycba.core import compute_centroid, compute_d4sigma, compute_principal_axes, measure

ROWS, COLS = 200, 260


def tilted_beam(sigma_major, sigma_minor, angle_ccw_deg, x0=130.0, y0=100.0, amp=1000.0, noise=0.0, seed=0):
    """Gaussian beam whose major axis makes ``angle_ccw_deg`` with the X axis, counter-clockwise ON SCREEN
    (y axis downwards: the direction vector is (cos a, -sin a) in image coordinates). Pixel i is at i (1-based)."""
    y, x = np.mgrid[1:ROWS + 1, 1:COLS + 1].astype(float)
    a = math.radians(angle_ccw_deg)
    u = (x - x0) * math.cos(a) - (y - y0) * math.sin(a)           # along the major axis
    v = (x - x0) * math.sin(a) + (y - y0) * math.cos(a)
    img = amp * np.exp(-0.5 * (u / sigma_major) ** 2 - 0.5 * (v / sigma_minor) ** 2)
    return img + np.random.default_rng(seed).normal(0, noise, img.shape) if noise else img


@pytest.mark.parametrize("angle", [0, 20, 45, 70, -30, -60, 90])
def test_tilted_gaussian_gives_its_axes_and_angle(angle):
    img = tilted_beam(24, 9, angle)
    xc, yc = compute_centroid(img)
    p = compute_principal_axes(img, xc, yc)
    assert p.d_major == pytest.approx(4 * 24, rel=0.01) and p.d_minor == pytest.approx(4 * 9, rel=0.01)
    assert p.ellipticity == pytest.approx(9 / 24, rel=0.01)
    wrapped = (p.angle - angle + 90) % 180 - 90                  # 90 and -90 are the same axis
    assert abs(wrapped) < 0.3
    assert -90 < p.angle <= 90 and p.angle_xy == -p.angle
    assert (p.x0, p.y0) == (xc, yc)


def test_screen_direction_of_the_angle():
    """A beam running from the lower left to the upper right of the displayed image has a POSITIVE angle."""
    img = tilted_beam(24, 9, 0)
    img = np.zeros((ROWS, COLS))
    for t in np.linspace(-60, 60, 400):                           # a thick line along (+x, -y)
        r, c = int(round(100 - t * 0.7)), int(round(130 + t * 0.7))
        img[r - 2:r + 3, c - 2:c + 3] = 1.0
    xc, yc = compute_centroid(img)
    assert 40 < compute_principal_axes(img, xc, yc).angle < 50    # y decreases when x increases: up and to the right


def test_x_and_y_widths_are_those_of_compute_d4sigma():
    img = tilted_beam(24, 9, 30)
    xc, yc = compute_centroid(img)
    p, d = compute_principal_axes(img, xc, yc), compute_d4sigma(img, xc, yc)
    assert p.d_x == pytest.approx(d.diam_h, rel=1e-9) and p.d_y == pytest.approx(d.diam_v, rel=1e-9)
    assert p.d_major >= max(p.d_x, p.d_y) - 1e-9 and p.d_minor <= min(p.d_x, p.d_y) + 1e-9   # axes bracket X / Y widths
    assert p.d_major**2 + p.d_minor**2 == pytest.approx(p.d_x**2 + p.d_y**2, rel=1e-9)         # trace invariant


def test_round_beam_and_degenerate_inputs():
    img = tilted_beam(15, 15, 0)
    xc, yc = compute_centroid(img)
    p = compute_principal_axes(img, xc, yc)
    assert p.ellipticity == pytest.approx(1.0, abs=1e-6) and p.d_major == pytest.approx(60, rel=0.01)
    empty = compute_principal_axes(np.zeros((30, 30)), 15, 15)             # no energy: NaN, no exception
    assert math.isnan(empty.angle) and math.isnan(empty.ellipticity)
    img2 = img.copy(); img2[10, 10] = np.nan                               # NaN pixels are ignored
    assert compute_principal_axes(img2, xc, yc).d_major == pytest.approx(p.d_major, rel=1e-3)
    neg = -np.abs(img) * 1e-3 + img                                        # (tiny) negative pixels: no crash, no NaN axis
    assert np.isfinite(compute_principal_axes(neg, xc, yc).d_minor)


def test_size_angle_reports_the_principal_axes_whatever_the_diameter_menu():
    img = tilted_beam(24, 9, 35)
    r = measure.size_angle(img, None, 1.0, (130, 100))
    assert r.principal.angle == pytest.approx(35, abs=0.3) and (r.x0, r.y0) == pytest.approx((130, 100), abs=0.01)
    p = r.principal
    assert r.messages()[-4:] == [f"Major={p.d_major:.4g}", f"Minor={p.d_minor:.4g}",
                                 f"Angle={p.angle:.1f} deg", f"Ellipticity={p.ellipticity:.3f}"]
    assert measure.size_xy(img, None, 1, 1.0, (130, 100)).clipboard_text().count("\t") == 1   # H and V sizes unchanged


@pytest.mark.parametrize("angle", [30, -30, 70, -70])
def test_rotating_the_image_by_minus_the_angle_aligns_the_beam_with_X(angle):
    """Independent check of the sign convention: scipy rotates counter-clockwise on screen (y downwards), so rotating
    the image by -angle brings a beam measured at +angle onto the X axis: angle ~ 0 and D4sigma along X = major."""
    from scipy import ndimage
    img = tilted_beam(24, 9, angle)
    p = measure.size_angle(img, None, 1.0, (130, 100)).principal
    assert p.angle == pytest.approx(angle, abs=0.3)
    rot = ndimage.rotate(img, -p.angle, reshape=True, order=1, mode="constant", cval=0.0)
    xc, yc = compute_centroid(rot)
    d = compute_d4sigma(rot, xc, yc)
    q = compute_principal_axes(rot, xc, yc)
    assert abs(q.angle) < 0.5 and d.diam_h == pytest.approx(p.d_major, rel=0.02) and d.diam_v == pytest.approx(p.d_minor, rel=0.03)


def test_principal_axes_with_an_aperture_and_noise():
    img = tilted_beam(24, 9, 25, noise=3.0)
    mask = np.zeros(img.shape, bool); mask[40:161, 40:221] = True
    p = measure.size_angle(img, mask, 1.0, (130, 100)).principal
    assert p.d_major == pytest.approx(96, rel=0.05) and p.d_minor == pytest.approx(36, rel=0.08)
    assert p.angle == pytest.approx(25, abs=2)


# ---------------------------------------------------------------- the window draws the tilted ellipse
def test_window_angle_button_draws_the_ellipse_of_the_principal_axes(tmp_path):
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtWidgets
    from pycba.gui.main_window import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    f = tmp_path / "tilted.txt"
    np.savetxt(f, tilted_beam(24, 9, 40), delimiter="\t", fmt="%.4f")
    w = MainWindow()
    w.show()
    w.open_path(str(f))
    app.processEvents()
    w.on_size_angle()
    app.processEvents()
    ell = [it for it in w.marks.items if len(it.getData()[0]) > 100][0]     # the ellipse (the other item is the '+')
    xs, ys = ell.getData()
    pts = np.vstack([xs - xs.mean(), ys - ys.mean()])
    vals, vecs = np.linalg.eigh(np.cov(pts))                                 # principal directions of the drawn curve
    major = vecs[:, 1]
    ang = math.degrees(math.atan2(-major[1], major[0])) % 180                # counter-clockwise on screen
    msgs = w.msgbox.toPlainText().splitlines()                               # the default aperture cuts the beam: compare
    reported = float(next(m for m in msgs if m.startswith("Angle=")).split("=")[1].split()[0])   # with what is reported
    maj = float(next(m for m in msgs if m.startswith("Major=")).split("=")[1])
    mino = float(next(m for m in msgs if m.startswith("Minor=")).split("=")[1])
    assert ang == pytest.approx(reported % 180, abs=0.3) and 20 < reported < 60
    assert np.sqrt(vals[1] / vals[0]) == pytest.approx(maj / mino, rel=0.02)  # axis ratio of the drawn ellipse
    assert any(m.startswith("Ellipticity=") for m in w.msgbox.toPlainText().splitlines())
    w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
