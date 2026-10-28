"""Z scale of the display: ``reinitZScale``, ``refreshZScale``, ``ZscaleMenu_Callback`` and the
computation part of ``load_imagefile``.

The menu of MATLAB is ``['Normalize', Zscales...]`` (1-based ``Value``). Here ``menu_index`` is
0-based: 0 = "Normalize", ``k + 1`` = ``zscales[k]``. The widgets themselves belong to the GUI layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .beam_metrics import compute_beam_peak
from .imagepipeline import Filters, ImageData, clean_image, update_image


@dataclass
class ZScale:
    menu_index: int          # entry to select in the Z scale menu (0 = Normalize)
    dimz: float
    znormalize: bool


def _nearest(zscales: Sequence[float], value: float) -> int:
    """Index of the closest entry; ties go to the first one (MATLAB ``sort`` is stable)."""
    return int(np.argsort(np.abs(np.asarray(zscales, dtype=float) - value), kind="stable")[0])


def _first_not_below(zscales: Sequence[float], value: float) -> int:
    """Index of the first scale >= value (the scales are sorted); the last one if none is large enough."""
    z = np.asarray(zscales, dtype=float)
    above = np.flatnonzero(z >= value)
    return int(above[0]) if above.size else int(np.argmax(z))


def zscale_for_peak(rmax: float, zscales: Sequence[float]) -> ZScale:
    """Scale chosen for an image whose peak is ``rmax``.

    ``rmax > 0``: the smallest of ``zscales`` that is >= 2**n, n = number of bits of round(rmax)
    (peak 254 -> 256, 256 -> 512), the largest one if none is. MATLAB took the entry CLOSEST to 2**n,
    which is the same thing when ``zscales`` holds every power of two but not near the ends of the
    list (peak 100 -> 128 -> closest of [1 256 ...] is 1: saturated display). Otherwise (zero, negative or NaN peak): "Normalize" with the
    tiny scale 1e-9 (MATLAB kept ``rmax`` itself, possibly negative or NaN; unused while normalising).
    """
    if rmax > 0:
        nbits = int(np.floor(rmax + 0.5)).bit_length()          # bitget(...) without MATLAB's 32-bit limit
        k = _first_not_below(zscales, 2.0 ** nbits)
        return ZScale(k + 1, float(zscales[k]), False)
    return ZScale(0, 1e-9, True)


def reinit_zscale(image: ImageData, zscales: Sequence[float], image_type: str = "raw") -> int:
    """``reinitZScale``: picks the best Z scale for ``image.raw`` (or ``.filt``), stores it in
    ``image.dimz`` / ``image.znormalize`` and returns the menu index to select.

    MATLAB left ``Znormalize`` untouched when a scale was found, so a stale ``true`` contradicted the
    menu; it is set to False here.
    """
    kind = image_type.lower()
    if kind not in ("raw", "filt"):
        raise ValueError(f"Unknown field {image_type}")
    r = image.raw if kind == "raw" else image.filt
    z = zscale_for_peak(compute_beam_peak(r, "max"), zscales)
    image.dimz, image.znormalize = z.dimz, z.znormalize
    return z.menu_index


def refresh_zscale(image: ImageData, zscales: Sequence[float]) -> int:
    """``refreshZScale``: menu index matching the internal scaling."""
    if image.znormalize:
        return 0
    return _nearest(zscales, image.dimz) + 1


def select_zscale(image: ImageData, menu_index: int, zscales: Sequence[float]) -> None:
    """``ZscaleMenu_Callback`` (the redraw is left to the caller)."""
    if menu_index == 0:
        image.znormalize = True
    else:
        image.znormalize = False
        image.dimz = float(zscales[menu_index - 1])


@dataclass
class LoadResult:
    messages: list[str] = field(default_factory=list)
    ok: bool = True                 # an image was loaded
    cleared: bool = False           # file not found: the slot was emptied (main image: the GUI removes the axes)
    zscale_index: int = 0
    reset_energy: bool = False      # the GUI must clear the energy field (main image loaded)


def load_image(images: list[ImageData], index: int, raw: np.ndarray, filters: Filters, method: int,
               zscales: Sequence[float], filepath: str = "", deflicker: bool = False, roi_mask=None,
               clean: bool = True, padding: str = "replicate") -> LoadResult:
    """Computation part of ``load_imagefile`` once the file is read, in the same order: ``clean_image``,
    new entry (all pixels alive, flicker estimated if ``deflicker`` else 0), energy reset when the main
    image is loaded, Z scale, ``update_image`` and, if a reference changed, the normalised main image.
    Drawing (``plot_image``) is up to the GUI.

    The background of the slot survives a reload, as in MATLAB (only raw, size, deadmask, flicker and
    file path are rewritten there); if its size no longer matches the new image it is reset to 0 (MATLAB
    then failed at the next subtraction) and a message says so.
    """
    res = LoadResult()
    raw = clean_image(raw) if clean else np.asarray(raw, dtype=float)
    bkg = images[index].bkg
    if np.ndim(bkg) > 0 and np.shape(bkg) != raw.shape:
        bkg = 0.0
        res.messages.append("Background reset (image size changed).")
    images[index] = ImageData.from_raw(raw, deflicker=deflicker, roi_mask=roi_mask, filepath=filepath, bkg=bkg)
    if index == 0:
        filters.energy = 1.0
        res.reset_energy = True
    res.zscale_index = reinit_zscale(images[index], zscales, "raw")
    res.messages += update_image(images, index, method, filters, padding=padding)
    if method != 1 and index != 0:
        res.messages += update_image(images, 0, method, filters, padding=padding)
    return res


def load_image_file(images: list[ImageData], index: int, path, filters: Filters, method: int,
                    zscales: Sequence[float], cur_file: int = 1, num_files: int = 1, deflicker: bool = False,
                    padding: str = "replicate") -> LoadResult:
    """``load_imagefile`` without the widgets: messages, reading, then ``load_image``."""
    from pathlib import Path

    from ..io import AsciiFormatError, read_ascii

    path = Path(path)
    if not path.is_file():
        images[index].raw = None
        images[index].filt = None
        return LoadResult(["File not found."], ok=False, cleared=True)
    res = LoadResult()
    try:
        raw = read_ascii(path)
    except AsciiFormatError:
        res.messages.append(f"{path.name} may not be a 2D ASCII file.")
        res.ok = False
        return res
    res.messages.append("Image opened." if num_files <= 1 else f"Image opened ({cur_file}/{num_files}).")
    loaded = load_image(images, index, raw, filters, method, zscales, filepath=str(path), deflicker=deflicker,
                        padding=padding)
    loaded.messages = res.messages + loaded.messages
    return loaded
