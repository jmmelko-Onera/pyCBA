"""Tests of the automatic background estimate (pycba.core.background) and of the 'Auto' button (Session.auto_bkg)."""
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from pycba.core import estimate_background
from pycba.gui.session import Session

pytestmark = pytest.mark.filterwarnings("ignore::pycba.io.AsciiFormatWarning")

BKG = 50.0


def _image(rows=100, cols=120, cx=None, seed=0, hot=0.0, dead=0.0, quadrant=False, w=10):
    r = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols]
    cx = cols / 2 if cx is None else cx
    img = 1000 * np.exp(-(((x - cx) / w) ** 2 + ((y - rows / 2) / w) ** 2)) + BKG + r.normal(0, 2, (rows, cols))
    m = r.random((rows, cols))
    img[m < hot] = 4000                                        # hot pixels
    img[(m >= hot) & (m < hot + dead)] = 0                     # dead pixels
    if quadrant:                                               # a quadrant damaged by a laser impact
        img[: rows // 2, : cols // 2] = 300 + r.normal(0, 2, (rows // 2, cols // 2))
    return img


def test_estimate_on_a_clean_image():
    est = estimate_background(_image())
    assert abs(est.value - BKG) < 0.5 and not est.non_uniform
    assert len(est.sides) == 4 and all(abs(s - BKG) < 1 for s in est.sides)


def test_estimate_ignores_hot_dead_pixels_and_nan():
    img = _image(hot=0.10, dead=0.02)
    img[0, :] = np.nan                                         # dead pixels given as NaN are skipped
    est = estimate_background(img)
    assert abs(est.value - BKG) < 1.5
    assert abs(estimate_background(_image(hot=0.10, dead=0.02)).value - np.mean(img[np.isfinite(img)])) > 50   # the mean would be far off


def test_estimate_survives_a_damaged_quadrant_and_says_so():
    est = estimate_background(_image(quadrant=True))
    assert abs(est.value - BKG) < 3                            # the median of the whole frame holds
    assert est.non_uniform                                     # but the sides disagree: the user is warned


def test_beam_reaching_the_border_is_flagged_and_a_touching_beam_is_not_a_problem():
    touching = estimate_background(_image(cx=3))                # a beam only touching one edge: the median ignores it
    assert abs(touching.value - BKG) < 1 and not touching.non_uniform
    clean_high = estimate_background(_image()).high
    huge = estimate_background(_image(w=40))                   # wings everywhere on the border
    assert huge.high and huge.value > BKG + 20 and not clean_high
    assert not touching.high


def test_estimate_errors():
    with pytest.raises(ValueError, match="too small"):
        estimate_background(np.ones((8, 8)))
    img = np.full((50, 50), np.nan)
    img[10:40, 10:40] = 1.0
    with pytest.raises(ValueError, match="no valid pixel"):
        estimate_background(img)
    with pytest.raises(ValueError):
        estimate_background(np.ones((4, 4, 3)))


# ---------------------------------------------------------------- the button (Session, no Qt)
def _session(tmp_path, img=None):
    f = tmp_path / "bkg.txt"
    np.savetxt(f, _image() if img is None else img, delimiter="\t", fmt="%.6f")
    s = Session()
    s.open_file(f, 0, 1, 1)
    s.messages.clear()
    return s


def test_auto_bkg_without_image():
    s = Session()
    u = s.auto_bkg()
    assert s.messages == ["/!\\No image."] and u.bkg_text is None


def test_auto_bkg_fills_the_field_and_subtracts(tmp_path):
    s = _session(tmp_path)
    before = np.median(s.image.filt[:5, :])
    u = s.auto_bkg()
    assert abs(s.image.bkg - BKG) < 1 and u.bkg_text == f"{s.image.bkg:.6g}"
    assert float(u.bkg_text) == s.image.bkg                    # the value applied is the value shown
    assert abs(np.median(s.image.filt[:5, :])) < 1 < before - 40
    assert s.messages == [f"Bkg = {u.bkg_text}: median of the image border."]
    s.messages.clear()
    s.auto_aperture()
    assert "/!\\ Auto-aperture without bkg substract. likely to fail" not in s.messages


def test_auto_bkg_after_flicker_removal_finds_what_is_left(tmp_path):
    s = _session(tmp_path)
    s.set_deflicker(True)                                      # the line offsets (background included) are gone
    s.messages.clear()
    u = s.auto_bkg()
    assert abs(s.image.bkg) < 1 and "after flicker removal" in s.messages[0] and u.bkg_text is not None


def test_auto_bkg_ignores_killed_pixels(tmp_path):
    img = _image()
    img[0, 10:60] = 4000
    s = _session(tmp_path, img)
    for c in range(10, 60):
        s.toggle_pixel(0, c)
    s.messages.clear()
    s.auto_bkg()
    assert abs(s.image.bkg - BKG) < 0.5


def test_auto_bkg_warns_when_the_border_is_not_uniform(tmp_path):
    s = _session(tmp_path, _image(quadrant=True))
    s.auto_bkg()
    assert any(m.startswith("/!\\Background not uniform") for m in s.messages)


def test_auto_bkg_warns_when_the_border_is_high(tmp_path):
    s = _session(tmp_path, _image(w=40))
    s.auto_bkg()
    assert any(m.startswith("/!\\Background is ") and "% of the peak" in m for m in s.messages)


def test_auto_bkg_on_a_tiny_image_says_why(tmp_path):
    s = _session(tmp_path, np.ones((8, 8)))
    u = s.auto_bkg()
    assert s.messages[-1].startswith("/!\\image too small") and u.bkg_text is None and s.image.bkg == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
