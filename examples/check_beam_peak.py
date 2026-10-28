"""Diagnostic: on which images does ComputeBeamPeak('smart') need its hot-pixel fallback,
and how far is the result from the plain maximum (smart/max should be ~1, or a bit below)?

Usage:  python check_beam_peak.py [folder_or_files ...]   (default: ../tests/data)
"""
import os
import sys
import warnings
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))

from pycba.core import compute_beam_peak
from pycba.io import AsciiFormatError, read_ascii

warnings.simplefilter("ignore")
args = sys.argv[1:] or [str(HERE.parent / "tests" / "data")]
files = []
for a in args:
    p = Path(a)
    files += sorted(f for f in p.iterdir() if f.is_file()) if p.is_dir() else [p]

print(f"{'file':42s} {'max':>12s} {'smart':>12s} {'smart/max':>9s} {'fallback':>8s} {'pts':>5s}")
for f in files:
    try:
        img = read_ascii(f)
    except (AsciiFormatError, FileNotFoundError, UnicodeError):
        continue
    peak, info = compute_beam_peak(img, return_info=True)
    mx = compute_beam_peak(img, "max")
    flag = "YES" if info["fallback"] else "-"
    print(f"{f.name:42s} {mx:12.5g} {peak:12.5g} {peak / mx if mx else float('nan'):9.4f} {flag:>8s} "
          f"{info['n_points']:>2d}/{info['nthreshold']}")
