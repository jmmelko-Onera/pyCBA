"""Reader of Microsoft RIFF palette files (``.pal``): ``pal2colormap``."""
from __future__ import annotations

import colorsys
import struct
import warnings
from pathlib import Path

import numpy as np

from .errors import AsciiFormatError, PaletteWarning


def _resample_hsv(colormap: np.ndarray, size: int) -> np.ndarray:
    """MATLAB: ``hsv2rgb(interp1(x, rgb2hsv(colormap), xi, 'linear'))`` with x, xi evenly spaced in [0, 1].

    The interpolation is done on H, S and V separately, so the hue is interpolated linearly across
    the hue circle (and through the arbitrary hue 0 of greys): this is MATLAB's behaviour, kept as is.
    """
    n = len(colormap)
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in colormap])
    x = np.arange(n) / (n - 1)
    xi = np.arange(size) / (size - 1)
    out = np.column_stack([np.interp(xi, x, hsv[:, k]) for k in range(3)])
    return np.array([colorsys.hsv_to_rgb(*c) for c in out])


def read_palette(path, forced_size: int | None = None, return_rgb255: bool = False):
    """``pal2colormap``: colours of a RIFF palette as an ``(n, 3)`` float array in [0, 1] (R, G, B).

    Layout: ``RIFF <size> 'PAL ' 'data' <size> <version:u16> <n:u16> n * (R, G, B, flags)``.
    The chunk sizes are not trusted (the Spiricon file declares 4 bytes more than it holds); only the
    entry count ``n`` (at byte 22) is used and the file must hold that many entries.

    ``forced_size`` resamples the palette to that many colours (HSV interpolation, like MATLAB; no
    change when it equals ``n``). A palette whose size is not a power of two gives a
    ``PaletteWarning``, as in MATLAB. With ``return_rgb255`` the 0-255 version (MATLAB's second
    output ``RGBpalette``, not rounded) is returned too.
    """
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 24 or data[:4] != b"RIFF" or data[8:12] != b"PAL ":
        raise AsciiFormatError(f"{path.name} is not a RIFF palette file")
    n = struct.unpack_from("<H", data, 22)[0]
    if len(data) < 24 + 4 * n:
        raise AsciiFormatError(f"{path.name}: truncated palette ({n} entries announced)")
    rgb = np.frombuffer(data, dtype=np.uint8, count=4 * n, offset=24).reshape(n, 4)[:, :3]
    colormap = rgb.astype(float) / 255.0
    if n == 0 or n & (n - 1):
        warnings.warn("Number of colors in palette is not 2^n. Are you sure it is a valid color palette?",
                      PaletteWarning, stacklevel=2)
    if forced_size is not None and forced_size != n:
        if forced_size < 2 or n < 2:
            raise ValueError("resampling needs at least 2 colours in the palette and in the result")
        colormap = _resample_hsv(colormap, int(forced_size))
        if forced_size > n:
            warnings.warn("Forced color depth is longer than original palette depth. Interpolation may fail.",
                          PaletteWarning, stacklevel=2)
    return (colormap, 255.0 * colormap) if return_rgb255 else colormap


def try_read_palette(path, forced_size: int | None = None):
    """MATLAB's behaviour on a missing or invalid palette: a warning and an empty result (``None``)."""
    try:
        return read_palette(path, forced_size)
    except (OSError, AsciiFormatError, TypeError) as exc:
        warnings.warn(f"Cannot find/open {path}: {exc}", PaletteWarning, stacklevel=2)
        return None
