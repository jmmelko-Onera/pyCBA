"""Converters from camera binaries / screenshots to image arrays (ASCII-exportable).

Ports of the three stand-alone MATLAB scripts that are *not* called by MAIN_CameraBeamAnalyzer:

* ``MAIN_SpiriconRAW_to_ASCII.m``       -> :func:`read_pyrocam_raw`, :func:`colorize`
* ``MAIN_PhotonPRW_to_ASCII.m``         -> :func:`read_photon_prw`, :func:`deinterlace`, :func:`normalize_z`
* ``MAIN_Import_screenshot_as_ASCII.m`` -> :func:`import_screenshot`

plus :func:`write_ascii`. The I/O layer stays GUI-free: no dialog, no figure. Images are returned
as ``(rows, cols)`` float arrays, i.e. exactly what ``read_ascii`` would give for the exported text.

Differences from the MATLAB scripts (bugs fixed in both languages, see ``matlab/CONVERTERS_FIXES.txt``):
the PRW image size is read from the file header instead of being guessed, the last pair of lines is
de-interlaced too, the screenshot pixel-grid detection no longer suffers from uint8 saturation, the
``threshold`` crosshair method uses an existing field name, and paths are portable.
"""
from __future__ import annotations

import re
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .errors import AsciiFormatError

# --------------------------------------------------------------------------------------------------
# Spiricon Pyrocam III .RAW
# --------------------------------------------------------------------------------------------------

PYROCAM_HEADER_BYTES = 16704       # MATLAB: headerskip
PYROCAM_COLOR_ZOOM = 4             # MATLAB: pyrocolorzoom ("Pyrocam III multiplies by Color Zoom when exporting")


def read_pyrocam_header(path) -> dict:
    """The ``[Header]`` INI block at the start of a Pyrocam .RAW file, as ``{key: str}``.

    Informative only (``Width``, ``Height``, ``PixelBits``, ``Peak``, ``PeakLocx`` ...): values keep the
    decimal comma of the French locale. Returns ``{}`` when the file does not start with ``[Header]``.
    """
    with open(path, "rb") as fh:
        head = fh.read(PYROCAM_HEADER_BYTES)
    if not head.startswith(b"[Header]"):
        return {}
    text = head.split(b"\x00", 1)[0].decode("latin-1")
    return {k.strip(): v.strip() for k, v in re.findall(r"^([A-Za-z0-9_]+)=(.*)$", text, re.M)}


def _header_float(hdr: dict, key: str):
    try:
        return float(hdr[key].replace(",", "."))
    except (KeyError, ValueError):
        return None


def _zoom_from_header(raw: np.ndarray, hdr: dict):
    """Colour zoom implied by the header: ``raw max / Peak``, ``raw min / Valley``, ``raw sum / FrameTotal``.

    The camera software writes these three numbers in counts, the data in the file are ``zoom`` times larger.
    Returns the zoom when at least two of the ratios exist and agree within 1 % and are close to an integer,
    else ``None`` (header missing, empty image, inconsistent values...).
    """
    ratios = []
    for key, value in (("Peak", raw.max()), ("Valley", raw.min()), ("FrameTotal", raw.sum())):
        ref = _header_float(hdr, key)
        if ref and np.isfinite(ref) and value != 0:
            ratios.append(value / ref)
    if len(ratios) < 2 or min(ratios) <= 0:
        return None
    mean = float(np.mean(ratios))
    if max(ratios) - min(ratios) > 0.01 * mean or abs(mean - round(mean)) > 0.01 * mean or round(mean) < 1:
        return None
    return float(round(mean))


def pyrocam_color_zoom(path, width: int | None = None, height: int | None = None,
                       header_bytes: int = PYROCAM_HEADER_BYTES):
    """Colour zoom of a Pyrocam .RAW file deduced from its header (see :func:`_zoom_from_header`), or ``None``."""
    raw = read_pyrocam_raw(path, width, height, header_bytes, color_zoom=1)
    return _zoom_from_header(raw, read_pyrocam_header(path))


def read_pyrocam_raw(path, width: int | None = None, height: int | None = None,
                     header_bytes: int = PYROCAM_HEADER_BYTES, color_zoom: float | str = "auto",
                     signed: bool = True) -> np.ndarray:
    """Pyrocam III .RAW -> ``(height, width)`` float array, in counts (raw value / ``color_zoom``).

    The data are *signed* little-endian 16-bit integers (MATLAB ``bit16``: negative values do occur,
    the camera background is subtracted) stored line by line, after ``header_bytes`` bytes.
    ``width``/``height`` default to the ``Width``/``Height`` of the ``[Header]`` block, then to 124.

    ``color_zoom="auto"`` (default) takes the zoom from the header (``Peak``, ``Valley`` and ``FrameTotal`` of the
    sample file are exactly the raw max, min and sum divided by 2, not 4). When the header does not allow it,
    ``PYROCAM_COLOR_ZOOM`` (4, the MATLAB value) is used and a warning says so. A number forces the divisor.
    """
    path = Path(path)
    hdr = read_pyrocam_header(path)
    if width is None or height is None:
        width = width or int(hdr.get("Width", 124))
        height = height or int(hdr.get("Height", 124))
    n = width * height
    with open(path, "rb") as fh:
        fh.seek(header_bytes)
        data = np.fromfile(fh, dtype="<i2" if signed else "<u2", count=n)
    if data.size < n:
        raise AsciiFormatError(f"{path.name}: {data.size} pixels after the {header_bytes}-byte header, "
                               f"{n} ({width}x{height}) expected")
    raw = data.astype(float).reshape(height, width)
    if isinstance(color_zoom, str):
        if color_zoom != "auto":
            raise ValueError("color_zoom must be a number or 'auto'")
        color_zoom = _zoom_from_header(raw, hdr)
        if color_zoom is None:
            color_zoom = PYROCAM_COLOR_ZOOM
            warnings.warn(f"{path.name}: colour zoom cannot be deduced from the header, using {color_zoom:g}",
                          UserWarning, stacklevel=2)
    return raw / color_zoom


def colorize(values: np.ndarray, colormap: np.ndarray, color_zoom: float = 4.0, bpp: int = 14) -> np.ndarray:
    """MATLAB TIFF export: ``ind2rgb(uint8((zoom*z + offset)*Zscale), colormap)`` -> ``(rows, cols, 3)`` uint8.

    ``colormap`` is the ``(n, 3)`` [0, 1] palette; its first colour is for z < 0.
    """
    n = len(colormap)
    zscale = (n - 1) / (2 ** bpp - 1)
    offset = 0.51 * (2 ** bpp - 1) / (n - 1)
    x = np.clip((color_zoom * np.asarray(values, float) + offset) * zscale, 0, 255)
    idx = np.clip(np.floor(x + 0.5).astype(int), 0, n - 1)           # uint8() rounds, ind2rgb clamps
    return np.round(np.asarray(colormap)[idx] * 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------------
# Pixel enlargement (myimresize)
# --------------------------------------------------------------------------------------------------
def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Port of ``Utils/myimresize.m``: enlarges (``scale`` > 1) or reduces (``scale`` < 1) an image.

    * ``scale`` > 1: nearest neighbour, like ``imresize(img, scale, 'nearest')`` (an integer scale just repeats
      each pixel ``scale`` times along both axes). The output has ``ceil(n * scale)`` rows and columns.
    * ``scale`` < 1: bicubic with anti-aliasing (Pillow). MATLAB's ``imresize`` is not bit-identical to it
      (slightly different kernel); MATLAB without the Image Processing Toolbox leaves the image unchanged.
    * ``scale`` == 1, an empty image, or one whose longest side is <= 2 (``length(img) <= 2`` in MATLAB):
      returned unchanged.
    """
    img = np.asarray(img)
    if img.size == 0 or max(img.shape) <= 2 or scale == 1:
        return img
    if scale <= 0:
        raise ValueError("scale must be > 0")
    n_rows, n_cols = (int(np.ceil(n * scale)) for n in img.shape[:2])
    if scale > 1:
        def src(n_out, n_in):                       # MATLAB: u = (x - 0.5) / scale + 0.5 (1-based), nearest
            return np.clip(np.floor((np.arange(n_out) + 0.5) / scale).astype(int), 0, n_in - 1)
        return img[np.ix_(src(n_rows, img.shape[0]), src(n_cols, img.shape[1]))]
    from PIL import Image
    out = Image.fromarray(np.asarray(img, dtype=np.float32), mode="F").resize((n_cols, n_rows), Image.BICUBIC)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------------------------------
# Photon .PRW
# --------------------------------------------------------------------------------------------------

def divisors(n: int) -> np.ndarray:
    """Sorted divisors of ``n`` (what MATLAB's ``divisor.m`` is used for: size guessing)."""
    n = int(n)
    return np.array([d for d in range(1, n + 1) if n % d == 0])


def _guess_sizes(npix: int, max_aspect: float = 2.39):
    """All ``(width, height)`` with ``width*height == npix`` and aspect ratio within ``1/max..max``."""
    out = []
    for w in divisors(npix):
        h = npix // int(w)
        if 1 / max_aspect < w / h < max_aspect:
            out.append((int(w), h))
    return out


def read_photon_prw(path, width: int | None = None, height: int | None = None) -> np.ndarray:
    """Photon .PRW -> ``(height, width)`` float array of raw counts (no normalisation).

    Layout: 2 x uint16 little-endian header (width, height), then ``width*height`` uint16, line by line.
    The MATLAB script skipped 4 bytes and *guessed* the size by asking the user; the header makes this
    unnecessary. ``width``/``height`` override the header (they are required for a file without a valid one).
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 4:
        raise AsciiFormatError(f"{path.name}: too short for a PRW file")
    hw, hh = struct.unpack_from("<HH", raw, 0)
    data = np.frombuffer(raw, dtype="<u2", offset=4)
    if width is None and height is None:
        if hw * hh != data.size:
            sizes = _guess_sizes(data.size)
            raise AsciiFormatError(
                f"{path.name}: header says {hw}x{hh} but the file holds {data.size} pixels; "
                f"give width and height (candidates: {sizes})")
        width, height = hw, hh
    elif width is None:
        width = data.size // height
    elif height is None:
        height = data.size // width
    if width * height != data.size:
        raise AsciiFormatError(f"{path.name}: {data.size} pixels, not {width}x{height}")
    return data.astype(float).reshape(height, width)


def deinterlace(image: np.ndarray) -> np.ndarray:
    """Replaces the odd field (1st, 3rd, 5th ... line) by a copy of the following line (MATLAB 'détramage').

    The MATLAB loop stopped one pair too early (``while i < dimY-1``): the last two lines were left
    interlaced. Here every odd line that has a successor is replaced.
    """
    out = np.array(image, copy=True)
    out[0:-1:2] = out[1::2][: len(out[0:-1:2])]
    return out


def normalize_z(image: np.ndarray, dim_z: float = 255) -> np.ndarray:
    """MATLAB: ``round(r*dimZ/max(r))``: *each image* is rescaled to ``0..dim_z`` (lossy, per-image gain)."""
    m = float(np.max(image))
    return np.floor(np.asarray(image, float) * dim_z / m + 0.5) if m > 0 else np.asarray(image, float)


def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    """``medfilt2(r, [size size])`` (zero padding, like MATLAB); ``size`` 1 does nothing."""
    if size <= 1:
        return np.array(image, float)
    from scipy.ndimage import median_filter as _mf
    return _mf(np.asarray(image, float), size=size, mode="constant", cval=0.0)


# --------------------------------------------------------------------------------------------------
# Screenshot of the camera software
# --------------------------------------------------------------------------------------------------

@dataclass
class CrosshairClean:
    """Options of the crosshair removal (MATLAB ``CROSSHAIRCLEAN`` structure)."""
    enable: bool = True
    thickness: int = 3
    threshold: float = 1.2            # 'threshold' method only
    image_borders: float = 0.08       # 'std' method: ignore the borders (fraction of the size)
    method: str = "auto"              # 'auto' (default), 'std' or 'threshold' (the two MATLAB methods)
    line_fraction: float = 0.6        # 'auto': a row (column) is a crosshair line if this fraction of it is off-palette


@dataclass
class ScreenshotResult:
    index: np.ndarray                 # (rows, cols) palette index (0-based) at camera resolution
    values: np.ndarray                # (rows, cols) pyrocam counts: ``(index*16383 - 127)/127``
    rgb_clean: np.ndarray             # (rows, cols, 3) uint8 screenshot after crosshair removal
    crosshair: tuple | None = None    # (x, y) 1-based found by the 'std' method, else None
    hlines: list = field(default_factory=list)
    vlines: list = field(default_factory=list)
    shape_in: tuple = ()
    messages: list = field(default_factory=list)


def load_rgb(path) -> np.ndarray:
    """Image file -> ``(rows, cols, 3)`` uint8 (alpha dropped, palette images expanded)."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")           # drops alpha / expands indexed / replicates grey
        return np.asarray(im, dtype=np.uint8).copy()


def rgb_to_index(rgb: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    """Nearest palette entry (Euclidean in RGB, MATLAB ``rgb2ind(...,'nodither')``), 0-based, ``(rows, cols)``."""
    pal = np.round(np.asarray(colormap) * 255).astype(np.int64)
    flat = rgb.reshape(-1, 3).astype(np.int64)
    code = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    pcode = (pal[:, 0] << 16) | (pal[:, 1] << 8) | pal[:, 2]
    uniq, inv = np.unique(code, return_inverse=True)
    ucol = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1)
    first = {}
    for i, c in enumerate(pcode):
        first.setdefault(int(c), i)                  # duplicate palette colours: first entry wins
    lut = np.empty(len(uniq), dtype=np.int64)
    for k, (u, c) in enumerate(zip(uniq, ucol)):
        j = first.get(int(u))
        lut[k] = j if j is not None else int(np.argmin(((pal - c) ** 2).sum(axis=1)))
    return lut[inv].reshape(rgb.shape[:2])


def _std_crosshair(idx: np.ndarray, borders: float):
    """MATLAB 'std' method: 1-based position of the strongest change of the column (row) dispersion."""
    def one(lines_axis_len, stats):
        d = np.abs(np.diff(stats))
        d = np.where(np.isnan(d), np.inf, d)                         # MATLAB 'descend' puts NaN first
        order = np.argsort(-d, kind="stable") + 1                    # 1-based positions
        m = int(np.floor(borders * lines_axis_len + 0.5))
        ok = np.flatnonzero((order > m) & (order < lines_axis_len - m))
        return int(order[ok[0]]) + 1 if ok.size else m

    def disp(lines):                                                  # std(L/mean(L)) - 1, N-1 normalisation
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = lines.mean(axis=1)
            return (lines / mean[:, None]).std(axis=1, ddof=1) - 1

    ymax, xmax = idx.shape
    x = one(xmax, disp(idx.T.astype(float)))
    y = one(ymax, disp(idx.astype(float)))
    return x, y


DASH_MIN_OFF_FRACTION = 0.4        # of line_fraction: a dashed black / white line must still be this much off-palette


def _auto_crosshair(rgb: np.ndarray, colormap: np.ndarray, line_fraction: float):
    """Rows and columns (1-based) made, for the most part, of colours that are not in the palette.

    The camera software draws its crosshair in a colour of its own over the palette image, so on a screenshot that
    was not resampled the crosshair is the only thing outside the palette. Only (almost) complete horizontal and
    vertical lines count: isolated off-palette pixels (saturation, dead pixels...) are left alone, and so is a
    crosshair at 45 degrees.

    DASHED crosshair (black / white / black / white...): the black dashes are outside the palette but the white ones
    are not (pure white is the last colour of the Spiricon palette), so only ~50 % of the line was off-palette. Pure
    black and pure white pixels therefore also count towards ``line_fraction``, provided that at least
    ``DASH_MIN_OFF_FRACTION * line_fraction`` of the line is really off-palette: a row of saturated (white) pixels
    inside the beam has none, and is not taken for a crosshair. A solid white crosshair cannot be told from
    saturation by its colour and is not found.
    Returns ``(hlines, vlines, off_palette_fraction)``.
    """
    pal = np.round(np.asarray(colormap) * 255).astype(np.int64)
    code = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2].astype(np.int64)
    pcode = (pal[:, 0] << 16) | (pal[:, 1] << 8) | pal[:, 2]
    off = ~np.isin(code, pcode)
    dash = off | (code == 0xFFFFFF) | (code == 0)
    min_off = DASH_MIN_OFF_FRACTION * line_fraction
    hl = (np.flatnonzero((dash.mean(axis=1) >= line_fraction) & (off.mean(axis=1) >= min_off)) + 1).tolist()
    vl = (np.flatnonzero((dash.mean(axis=0) >= line_fraction) & (off.mean(axis=0) >= min_off)) + 1).tolist()
    return hl, vl, float(off.mean())


def _threshold_crosshair(rgb: np.ndarray, threshold: float):
    """MATLAB 'threshold' method (black -> white, then lines whose sum exceeds ``threshold*size``)."""
    img = rgb.astype(float) / 255.0
    black = (img == 0).all(axis=2)
    img[black] = 1.0
    ymax, xmax = img.shape[:2]
    s = img.sum(axis=2)
    # a row sum has xmax terms, a column sum ymax terms (MATLAB compared them with ymax and xmax resp.:
    # same thing for a square image, wrong otherwise)
    hl = (np.flatnonzero(s.sum(axis=1) > xmax * threshold) + 1).tolist()
    vl = (np.flatnonzero(s.sum(axis=0) > ymax * threshold) + 1).tolist()
    return hl, vl


def pixel_grid(idx: np.ndarray, zoom: float):
    """Pixel distortion correction: 1-based start positions ``(xcoord, ycoord)`` of each camera pixel.

    Port of the MATLAB loop: from each pixel start, the next one is where the difference with the
    start column (row) jumps most between consecutive offsets 0..round(1.3*zoom). The MATLAB version
    did the differences in uint8 (negative differences saturated at 0): here they are true absolute values.
    """
    offmax = int(np.floor(1.3 * zoom + 0.5))
    data = idx.astype(np.int64)

    def scan(get, n):
        coords = [1]
        i = 1
        while i + offmax <= n:
            res = np.zeros(offmax + 1)
            for off in range(1, offmax + 1):
                res[off] = np.abs(get(i + off) - get(i)).sum()
            d = np.diff(res)
            i += int(np.flatnonzero(d == d.max())[-1]) + 1          # MATLAB: last of the (stable) ascending sort
            coords.append(i)
        return coords

    xcoord = scan(lambda c: data[:, c - 1], data.shape[1])
    ycoord = scan(lambda r: data[r - 1, :], data.shape[0])
    return xcoord, ycoord


def import_screenshot(path, colormap: np.ndarray, zoom: float = 1,
                      crosshair: CrosshairClean | None = None, bpp: int = 14) -> ScreenshotResult:
    """Screenshot (TIFF/PNG/BMP...) of the camera window -> palette indices and camera counts.

    ``zoom`` is the ratio between screen pixels and camera pixels (1: nothing to resample).
    ``values = (index*(2**bpp-1) - (n-1))/(n-1)`` for an ``n``-colour palette (pyrocam specific:
    ``(index*16383-127)/127`` for the 128-colour palette): a **0-based** index, so the first palette
    colour (z < 0) gives -1.
    """
    crosshair = crosshair or CrosshairClean()
    msgs: list[str] = []
    rgb = load_rgb(path)
    idx = rgb_to_index(rgb, colormap)
    ymax, xmax = idx.shape
    hlines: list = []
    vlines: list = []
    found = None
    if crosshair.enable:
        if crosshair.method == "auto":
            hlines, vlines, off_fraction = _auto_crosshair(rgb, colormap, crosshair.line_fraction)
            if off_fraction > 0.5 or len(hlines) > 0.1 * ymax or len(vlines) > 0.1 * xmax:
                hlines, vlines = [], []                         # a resampled screenshot: everything is off-palette
                msgs.append("Most colours are outside the palette (resized or interpolated screenshot?): "
                            "automatic crosshair detection impossible, image unchanged. Use method 'std' or 'threshold'.")
            elif not hlines and not vlines:
                msgs.append("No crosshair detected (no line of colours outside the palette): image unchanged.")
        elif crosshair.method == "threshold":
            hlines, vlines = _threshold_crosshair(rgb, crosshair.threshold)
        elif crosshair.method == "std":
            xc, yc = _std_crosshair(idx, crosshair.image_borders)
            found = (xc, yc)
            half = int(crosshair.thickness) // 2
            vlines = [xc + k for k in range(-half, half + 1)]
            hlines = [yc + k for k in range(-half, half + 1)]
        else:
            raise ValueError(f"unknown crosshair method {crosshair.method!r}")
        # lines outside the image (crosshair next to a border) are ignored; MATLAB raised an index error
        if hlines or vlines:
            rgb = rgb.copy()
            for y in hlines:
                if 1 <= y <= ymax:
                    rgb[y - 1] = rgb[max(1, y - 1) - 1]
            for x in reversed(vlines):
                if 1 <= x <= xmax:
                    rgb[:, x - 1] = rgb[:, min(xmax, x + 1) - 1]
            idx = rgb_to_index(rgb, colormap)
            msgs.append(f"Found crosshair at X={vlines}, Y={hlines}")
    if zoom > 1:
        xcoord, ycoord = pixel_grid(idx, zoom)
        idx_cam = idx[np.ix_(np.array(ycoord) - 1, np.array(xcoord) - 1)]
    else:
        idx_cam = idx
    n = len(colormap)
    values = (idx_cam.astype(float) * (2 ** bpp - 1) - (n - 1)) / (n - 1)
    msgs.append(f"Zoom is set to {zoom:g}: image file has {xmax}x{ymax} px, camera image has "
                f"{idx_cam.shape[1]}x{idx_cam.shape[0]} px.")
    return ScreenshotResult(idx_cam, values, rgb, found, hlines, vlines, (ymax, xmax), msgs)


# --------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------

def write_ascii(path, image: np.ndarray, delimiter: str = "\t", fmt: str = "%.7e") -> Path:
    """Text export readable by ``read_ascii``: one line per image row.

    MATLAB: ``save(...,'-ASCII','-single','-tabs')`` = ``%.7e`` + tab (RAW converter);
    ``'-double'`` = ``%.16e`` (screenshot importer); ``dlmwrite(...,',')`` (PRW converter) = comma and
    ``%.5g``-like text. Pass ``fmt="%.16e"`` or ``fmt="%g"`` and ``delimiter=","`` to choose.
    """
    path = Path(path)
    np.savetxt(path, np.asarray(image), delimiter=delimiter, fmt=fmt)
    return path


def default_output_path(path, extension: str = ".txt") -> Path:
    """``sample.RAW`` -> ``sample.RAW.txt`` (the MATLAB scripts append the extension to the full name)."""
    path = Path(path)
    return path.with_name(path.name + extension)


__all__ = [
    "PYROCAM_HEADER_BYTES", "PYROCAM_COLOR_ZOOM", "read_pyrocam_header", "read_pyrocam_raw", "pyrocam_color_zoom", "colorize",
    "resize_image", "divisors", "read_photon_prw", "deinterlace", "normalize_z", "median_filter",
    "CrosshairClean", "ScreenshotResult", "load_rgb", "rgb_to_index", "pixel_grid", "import_screenshot",
    "write_ascii", "default_output_path",
]
