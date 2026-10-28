"""Port of Utils/analyzeASCII.m

Detects the structure of a camera ASCII export (headers, delimiters, decimal
separator, size). The logic follows the MATLAB version step by step; the
deliberate deviations are marked ``DEVIATION`` and listed in the README.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from ._text import deblank, read_text_lines
from .errors import AsciiFormatWarning

# CAUTION (as in MATLAB): tab must be tested before space.
POSSIBLE_DELIMITERS = (";", ",", "\t", " ")
MAX_LINES_TO_FIND_COLDEL = 300

_REPEATED_BLANKS = re.compile(r"(\s)\1+")
_FLOAT = re.compile(r"[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf|nan)\Z", re.I)


@dataclass
class AsciiFormat:
    numheaders: int = 0       # header lines before the data
    numlines: int = 0         # number of data rows
    numcols: int = 0
    linedel: str = ""         # line delimiter ("\n", "\r\n", ...)
    lineskip: int = 1         # 2 => LF+CR files, 1 line out of 2 to skip
    coldel: str = ""          # column delimiter
    lastcoldel: str = ""      # trailing delimiter after the last column, if any
    decsep: str = ""          # decimal separator
    fixedcolsep: bool = True  # False if repeated blanks were found (unused)
    numcolsperline: list[int] = field(default_factory=list)  # debug only

    @property
    def shape(self) -> tuple[int, int]:
        return (self.numlines, self.numcols)


def _scan_numbers(text: str, coldel: str) -> tuple[int, str]:
    """Equivalent of sscanf(text, ['%f' coldel]): returns (count, errmsg)."""
    if text == "":
        return 0, ""
    tokens = text.split() if coldel in (" ", "\t") else text.split(coldel)
    n = 0
    for tok in tokens:
        if not _FLOAT.match(tok.strip()):
            return n, "sscanf: format failed to match"
        n += 1
    return n, ""


def _scanline(strline: str, coldel: str, lastcoldel: str):
    """Counts the cells of one line for a candidate column delimiter.

    Returns (emptycount, fullcount, decsep, errmsg).
    """
    is_ws = coldel in (" ", "\t")
    decsep = ""
    emptycount = 0
    start, end = 0, len(strline)
    if lastcoldel and strline.endswith(lastcoldel):
        end -= len(lastcoldel)

    if not is_ws:  # leading / trailing empty cells (ROI exports)
        seg = deblank(strline[start:end])
        lead = len(seg) - len(seg.lstrip(coldel))
        emptycount += lead
        start += lead
        seg = deblank(strline[start:end])
        trail = len(seg) - len(seg.rstrip(coldel))
        emptycount += trail
        end -= trail
    # DEVIATION: MATLAB also counted a leading blank as an "empty cell" for
    # blank/tab delimiters, which inflated numcols by one for space-aligned
    # files. Blanks around whitespace-delimited data are now ignored.

    sanitized = deblank(strline[start:end])
    count, errmsg = _scan_numbers(sanitized, coldel)

    # Attempts to detect comma as decimal separator
    if errmsg and coldel != "," and decsep != ",":
        count, errmsg = _scan_numbers(sanitized.replace(",", "."), coldel)
        if not errmsg:
            decsep = ","
    # Confirms that '.' is the decimal separator
    if "." in sanitized and coldel != "." and decsep != ",":
        decsep = "."

    return emptycount, count + emptycount, decsep, errmsg


def analyze_ascii(filepath: str | Path, debug: bool = False) -> AsciiFormat:
    """Analyse the structure of an ASCII image file (see AsciiFormat)."""
    fmt = AsciiFormat()
    filepath = Path(filepath)
    if not filepath.is_file():
        warnings.warn("File not found.", AsciiFormatWarning, stacklevel=2)
        return fmt

    lines = read_text_lines(filepath)

    # Tests whether line delimiter is LF+CR (instead of CR+LF)
    if len(lines) >= 3:
        if deblank(lines[0]) and deblank(lines[2]) and lines[1][:1] == "\r":
            fmt.lineskip = 2

    # -------------------------- Find delimiters -----------------------
    numlines = 0
    numempty = 0
    decsep = ""
    errmsg = ""
    strline = ""
    idx = 0
    while idx < len(lines) and not fmt.coldel and numlines < MAX_LINES_TO_FIND_COLDEL:
        numlines += 1
        strline = lines[idx]
        idx += 1

        # Sanitizes line: removes repeated blank characters
        if _REPEATED_BLANKS.search(strline):
            fmt.fixedcolsep = False
            strline = _REPEATED_BLANKS.sub(r"\1", strline)

        # 1. Line delimiter (LF, CR+LF)
        if strline:
            fmt.linedel = strline[-1]
            if len(strline) > 1 and ord(strline[-2]) < 20:
                fmt.linedel = strline[-2:]
        else:
            fmt.linedel = ""

        strline = deblank(strline)
        if not strline:
            numempty += 1

        # 2. Column delimiter
        for dlm in POSSIBLE_DELIMITERS:
            if strline and strline[-1] == dlm:
                fmt.lastcoldel = dlm
            emptycount, fullcount, decsep, errmsg = _scanline(strline, dlm, fmt.lastcoldel)
            if dlm in (" ", "\t"):
                delnum = strline.strip(dlm).count(dlm)
            else:
                delnum = strline.count(dlm)
                if fmt.lastcoldel == dlm:
                    delnum -= 1
            if debug:
                print(f"coldel={dlm!r} emptycount={emptycount} fullcount={fullcount} "
                      f"decsep={decsep!r} delnum={delnum} lastcoldel={fmt.lastcoldel!r} errmsg={errmsg}")
            if not errmsg and delnum + 1 == fullcount:
                fmt.coldel = dlm
                fmt.numheaders = numlines - 1
                fmt.numcols = fullcount
                break

    # ROI exports: the rows outside the camera ROI are lines made only of empty cells (";;;;"). They are
    # image rows (zeros), not headers: MATLAB's detector only recognises a data line when it holds at least
    # one number, so it counted them as headers. Only the lines just before the first data line are given
    # back, and only if they hold exactly numcols empty cells.
    if fmt.coldel and fmt.coldel not in (" ", "\t") and fmt.numheaders > 0 and fmt.lineskip == 1:
        n_delims = fmt.numcols if fmt.lastcoldel == fmt.coldel else fmt.numcols - 1
        while fmt.numheaders > 0:
            row = deblank(lines[fmt.numheaders - 1])
            if row and set(row) == {fmt.coldel} and len(row) == n_delims:
                fmt.numheaders -= 1
            else:
                break

    # Counts the number of real lines (stops at the first blank line)
    tline: str | None = strline
    while True:
        if not decsep:  # decimal separator not on the 1st line (rounded numbers)
            if fmt.coldel != ",":
                if tline is not None and "," in tline:
                    decsep = ","
            elif "." in (tline or ""):
                decsep = "."
        if debug and tline is not None:
            fmt.numcolsperline.append(tline.count(fmt.coldel) if fmt.coldel else 0)
        if idx >= len(lines):
            break
        tline = lines[idx]
        idx += 1
        if not deblank(tline):
            break
        numlines += 1

    # `numlines` is now the index of the last data line. DEVIATION: MATLAB
    # computed (numlines-1-numheaders+lineskip-1)/lineskip, which for
    # lineskip=1 gives one row LESS than the file contains. Here it is exact.
    ls = fmt.lineskip
    fmt.numlines = (numlines - fmt.numheaders + ls - 1) // ls
    fmt.decsep = decsep

    # Warning messages
    def warn(msg: str) -> None:
        warnings.warn(msg, AsciiFormatWarning, stacklevel=3)

    if errmsg and not fmt.coldel:
        warn(errmsg)
    if not fmt.decsep:
        warn("Could not find decimal separator (integers only?). Assuming it is '.'")
        fmt.decsep = "."
    if not fmt.coldel:
        warn("Column delimiter not found.")
    elif re.search(r"[a-zA-Z0-9]", fmt.coldel):
        warn("Found column delimiter is not a special character.")
    if not fmt.linedel:
        warn("Line delimiter not found.")
    if fmt.lineskip > 1:
        warn("File has empty lines, or LF and CR terminators are reversed.")
    if fmt.numlines == 0:
        warn("Found 0 lines in file.")
    if fmt.numcols == 0:
        warn("Found 0 columns in file.")
        if fmt.numheaders > 50:
            warn(f"More than 50 headers found. Are you sure '{filepath}' contains numerical data?")
    if debug:
        warn(f"Debug mode: {fmt}")
    return fmt
