"""Port of Utils/checkGetFile.m"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple


class FileInfo(NamedTuple):
    name_ext: str
    dir_name: str
    full_path: Path


def check_get_file(
    path: str | Path | None = None,
    filetype: str = "",
    chooser: Callable[[str], str | Path | None] | None = None,
) -> FileInfo | None:
    """Check that *path* exists, otherwise ask for a file through *chooser*.

    MATLAB opened ``uigetfile`` directly. The I/O layer must stay GUI-free, so
    the dialog is injected: ``chooser(title)`` returns a path or None
    (the GUI passes a QFileDialog wrapper). Returns None if no file results
    (MATLAB returned FullPath = -1).
    """
    if not path or not Path(path).is_file():
        if chooser is None:
            return None
        path = chooser(f"Select the first {filetype} file...")
        if not path:
            return None
    full = Path(path)
    return FileInfo(full.name, str(full.parent), full)
