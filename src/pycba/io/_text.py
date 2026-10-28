"""Low-level text helpers shared by analyze_ascii and read_ascii."""
from __future__ import annotations

from pathlib import Path

_BLANKS = " \t\n\r\v\f\x00"


def deblank(s: str) -> str:
    """MATLAB deblank: strips TRAILING whitespace and NULs only."""
    return s.rstrip(_BLANKS)


def read_text_lines(path: str | Path) -> list[str]:
    """Emulates a MATLAB fgets loop: split on LF ONLY, terminator kept.

    A CR is therefore left at the end of a CR+LF line (or at the start of the
    next line for LF+CR files), which is what analyzeASCII's detection relies on.
    Decoded as latin-1 so camera headers with odd bytes never raise.
    """
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
        raw = raw[3:]
    parts = raw.decode("latin-1").split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines
