"""Tests of pycba.io.converters (RAW / PRW / screenshot -> image arrays).

Run it either way:
  * Spyder: open this file and press "Run file" (F5)
  * terminal: python -m pytest tests -v
"""
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pytest

from pycba.io import AsciiFormatError, converters as cv, read_ascii, read_palette

DATA = HERE / "data"


def _sample(name):
    """Sample binary in tests/data (the test is skipped when the file is not there)."""
    if (DATA / name).is_file():
        return DATA / name
    pytest.skip(f"{name} not found in tests/data")
PAL = HERE.parent / "src" / "pycba" / "resources" / "Spiricon_128bit_RIFF.pal"


@pytest.fixture(scope="module")
def cmap():
    return read_palette(PAL)


# ---------------------------------------------------------------- Pyrocam RAW
def test_raw_matches_header_peak_location():
    img = cv.read_pyrocam_raw(_sample("sample_beam.RAW"))
    hdr = cv.read_pyrocam_header(_sample("sample_beam.RAW"))
    assert img.shape == (int(hdr["Height"]), int(hdr["Width"])) == (124, 124)
    row, col = np.unravel_index(np.argmax(img), img.shape)
    assert (col, row) == (int(hdr["PeakLocx"]), int(hdr["PeakLocy"]))      # no transposition error
    assert img.min() < 0                                                   # signed data: the background is subtracted


def test_raw_signed_and_color_zoom():
    a = cv.read_pyrocam_raw(_sample("sample_beam.RAW"), color_zoom=1)
    b = cv.read_pyrocam_raw(_sample("sample_beam.RAW"))                    # "auto": from the header
    np.testing.assert_allclose(a / 4, cv.read_pyrocam_raw(_sample("sample_beam.RAW"), color_zoom=4))
    # header Peak=1825,5 / Valley=-117,5 / FrameTotal=-55007,5 are the raw max / min / sum divided by 2 (not 4)
    assert cv.pyrocam_color_zoom(_sample("sample_beam.RAW")) == 2
    np.testing.assert_allclose(a / 2, b)
    assert a.max() / 2 == 1825.5 and b.max() == 1825.5 and b.min() == -117.5
    u = cv.read_pyrocam_raw(_sample("sample_beam.RAW"), signed=False, color_zoom=1)
    assert u.max() == 65535                                                # unsigned reading would be wrong


def test_raw_color_zoom_auto_falls_back_to_the_matlab_value(tmp_path):
    p = tmp_path / "x.RAW"
    head = b"[Header]\r\nWidth=4\r\nHeight=4\r\nPeak=7\r\n"                  # one number only: not enough to decide
    p.write_bytes(head + b"\0" * (16704 - len(head)) + np.arange(16, dtype="<i2").tobytes())
    with pytest.warns(UserWarning, match="colour zoom"):
        img = cv.read_pyrocam_raw(p)
    np.testing.assert_allclose(img, np.arange(16.0).reshape(4, 4) / 4)
    assert cv.pyrocam_color_zoom(p) is None
    np.testing.assert_allclose(cv.read_pyrocam_raw(p, color_zoom=1), np.arange(16.0).reshape(4, 4))   # forced
    with pytest.raises(ValueError):
        cv.read_pyrocam_raw(p, color_zoom="two")


def test_raw_truncated(tmp_path):
    p = tmp_path / "x.RAW"
    p.write_bytes(b"\0" * 20000)
    with pytest.raises(AsciiFormatError):
        cv.read_pyrocam_raw(p)


# ---------------------------------------------------------------- myimresize
def test_resize_image_enlarges_pixels_like_myimresize():
    img = np.arange(12.0).reshape(3, 4)
    np.testing.assert_array_equal(cv.resize_image(img, 4), np.kron(img, np.ones((4, 4))))
    # MATLAB fallback (no Image Processing Toolbox), transcribed: same result as the nearest-neighbour path
    X1, Y1, s = img.shape[0], img.shape[1], 3
    X2, Y2 = X1 * s, Y1 * s
    ref = np.zeros((X2, Y2))
    for i in range(Y1):
        for j in range(s):
            ref[0:X2:s, i * s + j] = img[:, i]
    for i in range(1, s):
        ref[i:X2:s, :] = ref[0:X2:s, :]
    np.testing.assert_array_equal(cv.resize_image(img, s), ref)
    assert cv.resize_image(img, 2.5).shape == (8, 10)                   # ceil(n * scale)


def test_resize_image_unchanged_cases_and_reduction():
    img = np.arange(12.0).reshape(3, 4)
    assert cv.resize_image(img, 1) is img or np.array_equal(cv.resize_image(img, 1), img)
    tiny = np.ones((2, 2))
    np.testing.assert_array_equal(cv.resize_image(tiny, 5), tiny)       # length(img) <= 2
    np.testing.assert_array_equal(cv.resize_image(np.empty((0, 3)), 5), np.empty((0, 3)))
    big = np.random.default_rng(0).random((40, 60))
    small = cv.resize_image(big, 0.5)
    assert small.shape == (20, 30) and abs(small.mean() - big.mean()) < 0.02
    with pytest.raises(ValueError):
        cv.resize_image(big, -1)


# ---------------------------------------------------------------- Photon PRW
@pytest.mark.filterwarnings("ignore::pycba.io.AsciiFormatWarning")      # integer-only text: no decimal separator to find
def test_prw_header_gives_size_and_matches_matlab_export():
    img = cv.read_photon_prw(_sample("sample_beam.prw"))
    assert img.shape == (1024, 1280)
    ref = read_ascii(_sample("sample_beam.prw.txt"))                         # written by MAIN_PhotonPRW_to_ASCII.m (raw values, no 0..255 normalisation since 22/10/2026)
    assert ref.shape == img.shape
    np.testing.assert_array_equal(img, ref)


def test_prw_wrong_size_asks_for_dimensions(tmp_path):
    p = tmp_path / "x.prw"
    p.write_bytes(b"\x05\x00\x04\x00" + b"\x01\x00" * 10)                  # header 5x4 but 10 pixels
    with pytest.raises(AsciiFormatError, match="candidates"):
        cv.read_photon_prw(p)
    assert cv.read_photon_prw(p, width=5, height=2).shape == (2, 5)


def test_deinterlace_even_and_odd_heights():
    for h in (6, 7):
        img = np.arange(h)[:, None] * np.ones((1, 3))
        out = cv.deinterlace(img)
        for i in range(0, h - 1, 2):
            np.testing.assert_array_equal(out[i], img[i + 1])             # incl. the last pair
        np.testing.assert_array_equal(out[1::2], img[1::2])
    assert cv.deinterlace(np.ones((1, 3))).shape == (1, 3)


def test_divisors_and_normalize():
    np.testing.assert_array_equal(cv.divisors(12), [1, 2, 3, 4, 6, 12])
    n = cv.normalize_z(np.array([[0.0, 5.0], [10.0, 2.5]]), 255)
    assert n.max() == 255 and n[1, 1] == 64                                # round(63.75)
    assert cv.median_filter(np.arange(9.0).reshape(3, 3), 1).tolist() == np.arange(9.0).reshape(3, 3).tolist()


# ---------------------------------------------------------------- screenshot
def _shot(tmp_path, rgb, name="s.png"):
    from PIL import Image
    p = tmp_path / name
    Image.fromarray(rgb).save(p)
    return p


def test_colorize_then_import_round_trip(tmp_path, cmap):
    yy, xx = np.mgrid[0:40, 0:50]
    z = 12000 * np.exp(-((xx - 20) ** 2 + (yy - 15) ** 2) / 60) - 100
    rgb = cv.colorize(z, cmap, color_zoom=1)
    res = cv.import_screenshot(_shot(tmp_path, rgb), cmap, zoom=1, crosshair=cv.CrosshairClean(enable=False))
    assert res.values.shape == z.shape
    quantum = 16383 / 127
    assert np.all(np.abs(res.values - np.clip(z, -1, None)) <= quantum + 1)
    assert res.index.min() == 0 and res.index.max() > 90


def test_palette_exact_colours_and_nearest():
    cm = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    rgb = np.array([[[0, 0, 0], [250, 10, 0]], [[3, 240, 9], [0, 0, 255]]], np.uint8)
    np.testing.assert_array_equal(cv.rgb_to_index(rgb, cm), [[0, 1], [2, 3]])


def test_alpha_and_indexed_screenshots(tmp_path, cmap):
    from PIL import Image
    rgb = cv.colorize(np.linspace(0, 8000, 20 * 20).reshape(20, 20), cmap, color_zoom=1)
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    p = tmp_path / "a.png"
    Image.fromarray(rgba).save(p)
    np.testing.assert_array_equal(cv.load_rgb(p), rgb)
    q = tmp_path / "p.png"
    Image.fromarray(rgb).convert("P").save(q)
    assert cv.load_rgb(q).shape == rgb.shape


def test_pixel_grid_recovers_camera_pixels(tmp_path, cmap):
    rng = np.random.default_rng(0)
    cam = rng.integers(0, 120, size=(30, 40))
    zoom = 4
    big = np.kron(cam, np.ones((zoom, zoom), int))
    rgb = np.round(cmap[big] * 255).astype(np.uint8)
    res = cv.import_screenshot(_shot(tmp_path, rgb), cmap, zoom=zoom, crosshair=cv.CrosshairClean(enable=False))
    n = (res.index.shape[0], res.index.shape[1])
    assert n[0] >= 28 and n[1] >= 38                       # the last pixels near the border may be dropped (as in MATLAB)
    np.testing.assert_array_equal(res.index[:5, :5], cam[:5, :5])


def test_pixel_grid_is_not_uint8_saturated():
    """MATLAB subtracted uint8 columns: a negative difference became 0. A decreasing step must be seen too."""
    row = np.array([9, 9, 9, 1, 1, 1, 9, 9, 9, 1, 1, 1, 9, 9, 9, 1, 1, 1, 9, 9, 9, 1, 1, 1])
    idx = np.tile(row, (6, 1))
    xs, _ = cv.pixel_grid(idx, 3)
    assert xs[:4] == [1, 4, 7, 10]


def test_crosshair_removal_std_and_threshold(tmp_path, cmap):
    yy, xx = np.mgrid[0:100, 0:120]
    z = 10000 * np.exp(-((xx - 60) ** 2 + (yy - 50) ** 2) / 600) + 500 * ((xx // 7 + yy // 5) % 2)
    rgb = cv.colorize(z, cmap, color_zoom=1)
    clean = rgb.copy()
    rgb[:, 29] = (255, 255, 255)                          # 1-px white lines (1-based column 30, row 70):
    rgb[69, :] = (255, 255, 255)                          # the 'std' heuristic is centred on the strongest step
    p = _shot(tmp_path, rgb)
    res = cv.import_screenshot(p, cmap, crosshair=cv.CrosshairClean(method="std", thickness=3))
    assert abs(res.crosshair[0] - 30) <= 1 and abs(res.crosshair[1] - 70) <= 1
    assert not (res.rgb_clean == 255).all(axis=2).any()   # the white lines are gone
    res2 = cv.import_screenshot(p, cmap, crosshair=cv.CrosshairClean(method="threshold", threshold=2.9))
    assert 30 in res2.vlines and 70 in res2.hlines        # field name bug of the MATLAB script is gone
    # image untouched when disabled
    res3 = cv.import_screenshot(p, cmap, crosshair=cv.CrosshairClean(enable=False))
    np.testing.assert_array_equal(res3.rgb_clean, rgb)
    assert (clean != rgb).any()


def test_crosshair_auto_removes_only_near_complete_lines(tmp_path, cmap):
    yy, xx = np.mgrid[0:100, 0:120]
    z = 12000 * np.exp(-((xx - 60) ** 2 + (yy - 50) ** 2) / 600)
    clean = cv.colorize(z, cmap, color_zoom=1)
    rgb = clean.copy()
    rgb[69, :] = (255, 0, 255)                              # row 70: complete line
    rgb[:, 29] = (255, 0, 255)                              # column 30 ...
    rgb[45:55, 29] = clean[45:55, 28]                       # ... with a gap in the middle (80 % left)
    rgb[:, 30] = (255, 0, 255)                              # a 2-pixel thick vertical line
    for y, x in ((10, 100), (80, 5), (50, 60)):             # isolated off-palette pixels (saturation...): not touched
        rgb[y, x] = (1, 2, 3)
    res = cv.import_screenshot(_shot(tmp_path, rgb), cmap, crosshair=cv.CrosshairClean())
    assert res.hlines == [70] and res.vlines == [30, 31]
    assert not (res.rgb_clean == (255, 0, 255)).all(axis=2).any()
    for y, x in ((10, 100), (80, 5), (50, 60)):
        assert tuple(res.rgb_clean[y, x]) == (1, 2, 3)


def test_crosshair_auto_finds_a_dashed_black_and_white_line(tmp_path, cmap):
    """Real case: a dashed NOIR-BLANC crosshair; white is in the Spiricon palette, so only half of the line was off-palette."""
    assert not any((np.round(cmap * 255) == 0).all(axis=1))                 # no black in the palette
    assert any((np.round(cmap * 255) == 255).all(axis=1))                   # pure white is its last colour
    yy, xx = np.mgrid[0:100, 0:120]
    z = 12000 * np.exp(-((xx - 60) ** 2 + (yy - 50) ** 2) / 600)
    clean = cv.colorize(z, cmap, color_zoom=1)
    for period, black, white in ((2, 1, 1), (6, 3, 3), (5, 3, 2)):          # dash patterns
        rgb = clean.copy()
        pattern = np.array([0] * black + [255] * white, dtype=np.uint8)
        dashes = np.resize(pattern, rgb.shape[1])
        rgb[49, :, :] = dashes[:, None]                                     # horizontal, over the beam
        rgb[50, :, :] = dashes[:, None]
        res = cv.import_screenshot(_shot(tmp_path, rgb, f"d{period}.png"), cmap, crosshair=cv.CrosshairClean())
        assert res.hlines == [50, 51] and res.vlines == []
        assert not ((res.rgb_clean[49] == 0).all(axis=1)).any()              # the black dashes are gone ...
    rgb = clean.copy()
    rgb[:, 70, :] = np.resize(np.array([0, 255], dtype=np.uint8), rgb.shape[0])[:, None]      # dashed vertical
    res = cv.import_screenshot(_shot(tmp_path, rgb, "v.png"), cmap, crosshair=cv.CrosshairClean())
    assert res.vlines == [71] and res.hlines == []


def test_crosshair_auto_ignores_saturated_white_and_black_regions(tmp_path, cmap):
    """A wide saturated beam (rows mostly white, but with nothing off-palette) is not a crosshair."""
    rgb = cv.colorize(np.random.default_rng(5).random((60, 80)) * 4000, cmap, color_zoom=1)
    rgb[20:40, 5:75] = 255                                                   # 20 rows, 87 % white
    res = cv.import_screenshot(_shot(tmp_path, rgb, "sat.png"), cmap, crosshair=cv.CrosshairClean())
    assert not res.hlines and not res.vlines
    np.testing.assert_array_equal(res.rgb_clean, rgb)
    rgb2 = rgb.copy(); rgb2[0:2, :, :] = 255                                 # solid white lines are NOT found (documented)
    res2 = cv.import_screenshot(_shot(tmp_path, rgb2, "sat2.png"), cmap, crosshair=cv.CrosshairClean())
    assert not res2.hlines


def test_crosshair_auto_leaves_an_image_without_crosshair_alone(tmp_path, cmap):
    rgb = cv.colorize(np.random.default_rng(2).random((40, 50)) * 9000, cmap, color_zoom=1)
    rgb[3, 4] = (255, 255, 254)
    res = cv.import_screenshot(_shot(tmp_path, rgb), cmap, crosshair=cv.CrosshairClean())
    np.testing.assert_array_equal(res.rgb_clean, rgb)
    assert not res.hlines and not res.vlines and any("No crosshair detected" in m for m in res.messages)
    from PIL import Image
    blurred = np.asarray(Image.fromarray(rgb).resize((200, 160), Image.BICUBIC))
    res2 = cv.import_screenshot(_shot(tmp_path, blurred, "b.png"), cmap, crosshair=cv.CrosshairClean())
    assert any("Most colours are outside the palette" in m for m in res2.messages)
    np.testing.assert_array_equal(res2.rgb_clean, blurred)
    assert not res2.hlines and not res2.vlines                          # nothing is removed


def test_crosshair_next_to_border_does_not_crash(tmp_path, cmap):
    rgb = cv.colorize(np.random.default_rng(1).random((30, 30)) * 9000, cmap, color_zoom=1)
    cv.import_screenshot(_shot(tmp_path, rgb), cmap, crosshair=cv.CrosshairClean(method="std", image_borders=0.0, thickness=7))


def test_sample_screenshot(cmap):
    auto = cv.import_screenshot(_sample("sample_beam2.tiff"), cmap, zoom=1)          # default: crosshair 'auto'
    assert not auto.hlines and not auto.vlines                                      # the 'std' method found one here
    res = cv.import_screenshot(_sample("sample_beam2.tiff"), cmap, zoom=1, crosshair=cv.CrosshairClean(enable=False))
    np.testing.assert_array_equal(auto.index, res.index)
    assert res.values.shape == (320, 320)
    assert set(np.unique(res.index)) <= set(range(128))
    row, col = np.unravel_index(np.argmax(res.values), res.values.shape)
    assert 100 < col < 300 and 50 < row < 200             # the beam seen on the picture


# ---------------------------------------------------------------- output
def test_write_ascii_is_readable_by_read_ascii(tmp_path):
    img = np.arange(12.0).reshape(3, 4) * 1.25 - 3
    for delim, fmt in (("\t", "%.7e"), (",", "%.16e"), ("\t", "%g")):
        out = cv.write_ascii(tmp_path / "o.txt", img, delimiter=delim, fmt=fmt)
        np.testing.assert_allclose(read_ascii(out), img, rtol=1e-6)
    assert cv.default_output_path("a/b/s.RAW").name == "s.RAW.txt"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
