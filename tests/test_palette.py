"""Tests of pycba.io.read_palette (pal2colormap) on the Spiricon 128-colour RIFF palette.

MATLAB comparison needs matlab_reference/palette_all.mat from export_palette_reference.m.
"""
import colorsys
import os
import struct
import sys
import warnings
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import pytest

from pycba.io import AsciiFormatError, PaletteWarning, read_palette, try_read_palette

PAL = HERE / "data" / "Spiricon_128bit_RIFF.pal"
REF = HERE / "matlab_reference"


def write_pal(path, colors, extra=0):
    """Minimal RIFF palette writer (independent of the reader)."""
    n = len(colors)
    body = struct.pack("<HH", 0x300, n) + b"".join(bytes([*c, 0]) for c in colors)
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + 8 + len(body) + extra) + b"PAL data"
                     + struct.pack("<I", len(body) + extra) + body + bytes(extra))


def test_spiricon_palette():
    cm = read_palette(PAL, 128)
    assert cm.shape == (128, 3) and cm.min() >= 0 and cm.max() <= 1
    np.testing.assert_allclose(cm[0] * 255, [64, 64, 64])          # lowest level: grey
    np.testing.assert_allclose(cm[1] * 255, [40, 0, 60])
    np.testing.assert_allclose(cm[126] * 255, [255, 0, 0])         # top of the scale: red
    np.testing.assert_allclose(cm[127] * 255, [255, 255, 255])     # last entry: white (saturation)


def test_forced_size_equal_to_file_size_changes_nothing():
    np.testing.assert_array_equal(read_palette(PAL, 128), read_palette(PAL))


def test_second_output_is_the_255_version():
    cm, rgb = read_palette(PAL, return_rgb255=True)
    np.testing.assert_allclose(rgb, 255 * cm)
    assert rgb[1].tolist() == [40, 0, 60]


@pytest.mark.filterwarnings("ignore::pycba.io.PaletteWarning")
def test_forced_size_interpolates_in_hsv_like_matlab(tmp_path):
    colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255)]
    f = tmp_path / "p4.pal"
    write_pal(f, colors)
    got = read_palette(f, 7)
    hsv = np.array([colorsys.rgb_to_hsv(*(np.array(c) / 255)) for c in colors])
    xi = np.arange(7) / 6
    x = np.arange(4) / 3
    expected = np.array([colorsys.hsv_to_rgb(*[np.interp(t, x, hsv[:, k]) for k in range(3)]) for t in xi])
    np.testing.assert_allclose(got, expected, atol=1e-12)
    np.testing.assert_allclose(got[0], [1, 0, 0]); np.testing.assert_allclose(got[-1], [0, 0, 1])


def test_warnings(tmp_path):
    f = tmp_path / "p3.pal"
    write_pal(f, [(0, 0, 0), (128, 128, 128), (255, 255, 255)])
    with pytest.warns(PaletteWarning, match="2\\^n"):
        read_palette(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PaletteWarning)
        assert read_palette(f, 5).shape == (5, 3)
    with pytest.warns(PaletteWarning, match="longer"):
        read_palette(PAL, 256)


def test_chunk_sizes_are_not_trusted(tmp_path):
    f = tmp_path / "pad.pal"
    write_pal(f, [(1, 2, 3), (4, 5, 6)], extra=4)               # declared size 4 bytes too long, like the Spiricon file
    np.testing.assert_allclose(read_palette(f) * 255, [[1, 2, 3], [4, 5, 6]])


def test_not_a_palette_or_truncated(tmp_path):
    bad = tmp_path / "x.pal"
    bad.write_bytes(b"hello world, definitely not RIFF")
    with pytest.raises(AsciiFormatError):
        read_palette(bad)
    cut = tmp_path / "cut.pal"
    cut.write_bytes(PAL.read_bytes()[:100])
    with pytest.raises(AsciiFormatError):
        read_palette(cut)


def test_try_read_palette_returns_none_with_a_warning(tmp_path):
    with pytest.warns(PaletteWarning, match="Cannot find/open"):
        assert try_read_palette(tmp_path / "missing.pal", 128) is None
    assert try_read_palette(PAL, 128).shape == (128, 3)


@pytest.mark.parametrize("ref", sorted(REF.glob("palette_*.mat")) or [None])
def test_against_matlab(ref):
    scipy_io = pytest.importorskip("scipy.io")
    if ref is None:
        pytest.skip("no MATLAB palette reference exported yet (export_palette_reference.m)")
    m = scipy_io.loadmat(ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PaletteWarning)
        for size in np.ravel(m["sizes"]).astype(int):
            np.testing.assert_allclose(read_palette(PAL, int(size)), m[f"cm_{size}"], atol=1e-12,
                                       err_msg=f"forced size {size}")


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v", "-p", "no:cacheprovider"]))
