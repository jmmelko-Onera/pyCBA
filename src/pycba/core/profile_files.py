"""Names and writing of the profile text files saved next to the image by the Profile buttons.

Convention (harmonised, MATLAB mixed them): ``Hpx`` = pixel along X (column), ``Vpx`` = pixel along Y (row).
A horizontal profile is taken AT a row, hence ``Hprofile_at_Vpx<row>``; a vertical one at a column,
``Vprofile_at_Hpx<column>``. MATLAB named the horizontal profile with the column and the vertical one with the row
(crossed), and the 45 degree files with ``Hpx<row>_Vpx<column>``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _folder(image_path) -> Path | None:
    """Folder of the image file; None when the image does not come from a file (MATLAB then built a wrong path)."""
    if not image_path:
        return None
    return Path(str(image_path).replace("\\", "/")).parent


def profile_paths(kind: str, image_path, linenum: int, colnum: int, angle: float | None = None):
    """``{label: Path}`` of the two files of a profile pair, or ``{}`` without image file.

    kind: 'hv' (H / V), '45' (minus / plus), 'l1l2' (L1 / L2, the angle of L1 is added to the name)."""
    folder = _folder(image_path)
    if folder is None:
        return {}
    if kind == "hv":
        return {"Horizontal": folder / f"Hprofile_at_Vpx{linenum}.txt",
                "Vertical": folder / f"Vprofile_at_Hpx{colnum}.txt"}
    tail = f"_at_Hpx{colnum}_Vpx{linenum}"
    if kind == "45":
        return {"Minus": folder / f"minusProfile{tail}.txt", "Plus": folder / f"plusProfile{tail}.txt"}
    if kind == "l1l2":
        a = "" if angle is None else f"_A{angle:.1f}"
        return {"L1": folder / f"L1Profile{tail}{a}.txt", "L2": folder / f"L2Profile{tail}{a}.txt"}
    raise ValueError(f"unknown profile kind {kind!r}")


def save_profile(path, vector, values) -> str:
    """``save(path,[vector values],'-ascii')`` (8 digits, space separated). Returns a message for the text area;
    a folder that cannot be written (read-only share...) no longer aborts the whole callback."""
    data = np.column_stack([np.asarray(vector, dtype=float).ravel(), np.asarray(values, dtype=float).ravel()])
    try:
        np.savetxt(path, data, fmt="%.8e", delimiter=" ")
    except OSError as exc:
        return f"/!\\Could not save {Path(path).name}: {exc.strerror or exc}"
    return f"Saved: {path}"
