"""Tests of pycba.io against the sample camera files.

Run it either way:
  * Spyder: open this file and press "Run file" (F5)
  * terminal: python -m pytest tests -v

Expected values were established by inspecting the raw files independently of
the parser (see `naive_read`), not by running the parser itself.
"""
import re
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent          # .../pyCBA/tests
sys.path.insert(0, str(HERE.parent / "src"))           # no need to pip install

import numpy as np
import pytest

from pycba.io import AsciiFormatWarning, NoImageDataError, analyze_ascii, check_get_file, read_ascii


DATA = HERE / "data"          # the tests only read tests/data (never SampleBeams)
if not (DATA / "sample_beam_SP928.csv").is_file():
    raise FileNotFoundError(f"Sample files not found: put sample_*.csv/.txt and LFCR.txt in {DATA}")


def _p(name):
    return DATA / name


REF = HERE / "matlab_reference"

pytestmark = pytest.mark.filterwarnings("ignore::pycba.io.AsciiFormatWarning")

# name: (numheaders, numcols, coldel, lastcoldel, decsep, linedel)
# numlines is NOT hard-coded: files may be truncated or full-size; it is checked
# against an independent count of the data rows instead.
EXPECTED = {
    "sample_beam_SP928.csv":                  (0, 1928, ";", ";", ".", "\r\n"),
    "sample_beam_SP928_comma.csv":  (0, 1928, ";", ";", ",", "\r\n"),
    "sample_beam_beamgage.txt":               (8, 2048, ";", ";", ".", "\r\n"),
    "sample_beam.prw.txt":                    (0, 1280, ",", "",  ".", "\n"),
    "sample_beam_tab.txt":                    (0, 496, "\t", "", ".", "\r\n"),
    "sample_emptylines.csv":                  (10, 161, "\t", "", ".", "\r\n"),
    "sample_nosep.csv":                       (0, 1928, ";", ";", None, "\r\n"),  # integers only: decsep not meaningful
    "sample_perfectgaussianbeam.csv":         (0, 161, "\t", "", ".", "\r\n"),
    "sample_pyrocam.csv":                     (0, 320, ";", ";", ",", "\r\n"),
    "sample_spiriconSP1550.spc":              (0, 512, " ", "", ".", "\n"),   # blank-aligned columns
    "sample_beam_WinCamD.wct":                (0, 680, ",", ";", None, "\r\n"),  # integers, trailing " ; "
    "sample_roi.csv":                         (0, 319, ";", ";", ",", "\r\n"),   # 10 empty ROI rows on top are IMAGE rows
}


def naive_read(path, skip, sep):
    """Independent reference reader (not used by the package)."""
    rows = []
    for line in path.read_bytes().decode("latin1").splitlines()[skip:]:
        if not line.strip():
            break
        line = line.strip()
        if sep != ";" and line.endswith(";"):      # WinCamD: trailing " ; " marker
            line = line[:-1]
        cells = [c.strip().replace(",", ".") if sep != "," else c.strip()
                 for c in line.strip().split(sep if sep not in " \t" else None)]
        if cells and cells[-1] == "":
            cells.pop()
        rows.append([float(c) if c else 0.0 for c in cells])
    return np.array(rows)


@pytest.mark.parametrize("name", EXPECTED)
def test_analyze(name):
    nh, nc, cd, lcd, ds, ld = EXPECTED[name]
    f = analyze_ascii(_p(name))
    got = (f.numheaders, f.numcols, f.coldel, f.lastcoldel, f.decsep, f.linedel)
    expected = (nh, nc, cd, lcd, ds, ld)
    if ds is None:                       # decimal separator undetermined by design
        got, expected = got[:4] + got[5:], expected[:4] + expected[5:]
    assert got == expected
    assert f.numlines == naive_read(_p(name), nh, cd).shape[0]


@pytest.mark.parametrize("name", EXPECTED)
def test_read_matches_independent_reader(name):
    nh, _, cd, _, _, _ = EXPECTED[name]
    r = read_ascii(_p(name))
    ref = naive_read(_p(name), nh, cd)
    assert r.shape == ref.shape
    np.testing.assert_array_equal(r, ref)


def test_roi_structure():
    r = read_ascii(_p("sample_roi.csv"))
    assert r.shape == (320, 319)                    # 10 empty rows (outside the ROI) + 300 data rows + 10 empty rows
    mask = r != 0
    assert mask.sum() > 0
    assert not mask[:, -1].any()                    # no data in the last column
    assert not mask[:10].any() and not mask[-10:].any()
    assert mask.any(axis=1).sum() == 300            # 300 rows carry data
    assert mask[10:310].any(axis=1).all()


def test_empty_cell_rows_before_the_data_are_not_headers(tmp_path):
    """Only lines made of exactly numcols empty cells, right before the first data line, are given back to the image."""
    body = "\r\n".join(";".join(f"{i + j}" for j in range(4)) + ";" for i in range(3)) + "\r\n"
    f = tmp_path / "roi_like.csv"
    f.write_bytes((";;;;\r\n;;;;\r\n" + body).encode())
    fmt = analyze_ascii(f)
    assert fmt.numheaders == 0 and fmt.numlines == 5
    r = read_ascii(f)
    assert r.shape == (5, 4) and not r[:2].any() and r[2, 1] == 1
    g = tmp_path / "with_text_header.csv"                     # a real header stays a header, the empty rows after it do not
    g.write_bytes(("Camera export\r\n;;;;\r\n" + body).encode())
    fmt = analyze_ascii(g)
    assert fmt.numheaders == 1 and read_ascii(g).shape == (4, 4)
    h = tmp_path / "wrong_width.csv"                          # an empty line with another width is a header
    h.write_bytes((";;\r\n" + body).encode())
    assert analyze_ascii(h).numheaders == 1


def test_last_column_value():
    r = read_ascii(_p("sample_beam_SP928.csv"))
    assert r[0, -1] == -10.015625 and r[0, 0] == 53.328125


def test_lfcr_detection():
    assert analyze_ascii(_p("LFCR.txt")).lineskip == 2


def test_single_column_is_not_an_image():
    with pytest.raises(NoImageDataError):
        read_ascii(_p("LFCR.txt"))


def test_no_decsep_warns():
    with pytest.warns(AsciiFormatWarning, match="decimal separator"):
        analyze_ascii(_p("sample_beam.prw.txt"))


def test_check_get_file(tmp_path):
    assert check_get_file(tmp_path / "missing.txt") is None
    info = check_get_file(_p("LFCR.txt"))
    assert info.name_ext == "LFCR.txt" and Path(info.dir_name) == _p("LFCR.txt").parent
    chosen = check_get_file(None, chooser=lambda title: _p("LFCR.txt"))
    assert chosen.name_ext == "LFCR.txt"


# Files for which the MATLAB reference itself is wrong (Python verified against the raw file):
# {name: reason}. Empty now that analyzeASCII / readAsciiFromFile carry the fixes of matlab/ASCII_ROI_FIXES.txt.
KNOWN_MATLAB_BUGS = {}
_MATLAB_PARAMS = [
    pytest.param(n, marks=pytest.mark.xfail(reason=KNOWN_MATLAB_BUGS[n], strict=False))
    if n in KNOWN_MATLAB_BUGS else n
    for n in EXPECTED
]


@pytest.mark.parametrize("name", _MATLAB_PARAMS)
def test_against_matlab_reference(name):
    """Full comparison with MATLAB (needs ref_<name>.mat from export_reference.m)."""
    scipy_io = pytest.importorskip("scipy.io")
    ref = REF / ("ref_" + re.sub(r"\W", "_", name) + ".mat")   # same rule as export_reference.m: sample_beam.prw.txt -> ref_sample_beam_prw_txt.mat
    if not ref.exists():
        pytest.skip("no MATLAB reference exported yet")
    m = scipy_io.loadmat(ref)
    r = read_ascii(_p(name))
    assert r.shape == m["r"].shape
    np.testing.assert_array_equal(r, m["r"])


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v", "-p", "no:cacheprovider"]))
