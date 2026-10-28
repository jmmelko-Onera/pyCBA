"""Port of Utils/readAsciiFromFile.m"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

from ._text import deblank, read_text_lines
from .analyze_ascii import AsciiFormat, analyze_ascii
from .check_get_file import check_get_file
from .errors import AsciiFormatError, AsciiFormatWarning, NoImageDataError


def read_ascii(
    path: str | Path | None = None,
    fmt: AsciiFormat | None = None,
    chooser: Callable[[str], str | Path | None] | None = None,
) -> np.ndarray:
    """Read an ASCII camera image into a 2D float64 array.

    ``fmt`` may be given to skip the automatic analysis (MATLAB's optional
    arguments). Raises NoImageDataError for files that are not 2D images
    (MATLAB returned [] with a warning) and AsciiFormatError for malformed rows.

    Empty cells are read as 0 when one of the first two data rows starts with
    a delimiter or contains two consecutive ones (ROI exports from camera
    software), as in MATLAB.
    """
    info = check_get_file(path, "ASCII file", chooser)
    if info is None:
        raise FileNotFoundError("No file chosen. Exiting read_ascii.")
    if fmt is None:
        fmt = analyze_ascii(info.full_path)

    numcols, numlines = fmt.numcols or 0, fmt.numlines or 0
    coldel = fmt.coldel or ("," if numcols >= 2 else "")
    lastcoldel = fmt.lastcoldel
    decsep = fmt.decsep
    lineskip = fmt.lineskip or 1
    if numcols < 2 or numlines < 2:
        raise NoImageDataError(
            f"{info.name_ext}: less than 2 columns or less than 2 lines "
            "(not a 2D image). Skipping."
        )
    if not decsep:
        warnings.warn("Decimal separator not provided. Assuming '.'", AsciiFormatWarning, stacklevel=2)
        decsep = "."
    elif decsep != ".":
        warnings.warn(f"Found '{decsep}' to be the decimal separator.", AsciiFormatWarning, stacklevel=2)

    lines = read_text_lines(info.full_path)
    first = fmt.numheaders * lineskip
    rows = lines[first : first + numlines * lineskip : lineskip]
    if len(rows) < numlines:
        raise AsciiFormatError(f"{info.name_ext}: expected {numlines} data rows, found {len(rows)}.")

    is_ws = coldel in (" ", "\t")
    roi = False
    if not is_ws:  # ROI exports: empty cells outside the region of interest
        pat = re.compile("^" + re.escape(coldel) + "|" + re.escape(coldel) * 2)
        roi = any(pat.search(r) for r in rows[:2])

    out = np.zeros((numlines, numcols))
    for k, raw in enumerate(rows):
        s = deblank(raw)
        if decsep != ".":
            s = s.replace(decsep, ".")
        if lastcoldel and s.endswith(lastcoldel):
            s = s[: -len(lastcoldel)]
        tokens = s.split() if is_ws else s.split(coldel)
        try:
            try:
                vals = np.array(tokens, dtype=float)
            except ValueError:
                if not roi:
                    raise
                vals = np.array([t if t.strip() else "0" for t in tokens], dtype=float)
        except ValueError as exc:
            raise AsciiFormatError(
                f"{info.name_ext}: non-numeric value in data row {k + 1} "
                f"(file line {first + k * lineskip + 1}). Is it a text file with numbers?"
            ) from exc
        if vals.size != numcols:
            raise AsciiFormatError(
                f"{info.name_ext}: data row {k + 1} has {vals.size} values, expected {numcols}."
            )
        out[k] = vals
    return out
