"""Geometry of the main window, copied from the ``%% POSITIONS`` / ``%% GUI OBJECTS`` section of
MAIN_CameraBeamAnalyzer.m.

Rectangles are ``(x, y, w, h)`` in MATLAB figure pixels: origin at the BOTTOM-left, window
800 x 600. MATLAB switches every control to normalized units, so the whole layout scales with the
window: ``to_qt`` converts a rectangle for a window of any size (origin top-left, as Qt wants).
"""
from __future__ import annotations

from typing import NamedTuple

WIN_W, WIN_H = 800, 600          # design size of the MATLAB layout (reference of every rectangle)
DEFAULT_W, DEFAULT_H = 1200, 900  # size at startup (same 4:3 ratio, fits a 1920x1080 screen); the layout scales


class Rect(NamedTuple):
    x: float
    y: float
    w: float
    h: float


def _compute() -> dict[str, Rect]:
    b_w, b_h = 120, 25                       # right hand side buttons
    b_hpos = WIN_W - b_w - 15
    p_w, p_h = WIN_W - 50 - 30, 20           # path box
    p_hpos, p_vpos = 5, WIN_H - p_h
    field_h = p_h
    fields_v = p_vpos - p_h                  # first line of fields
    fields_v2 = fields_v - 2 * field_h       # second line
    fields_v3 = fields_v2 - field_h          # third line
    m_w, m_h = WIN_W - b_hpos - 10, WIN_H / 2.1       # message box
    b_vpos = p_vpos - p_h - 10

    r: dict[str, Rect] = {}
    r["path"] = Rect(p_hpos, p_vpos, p_w, p_h)
    r["browse"] = Rect(p_hpos + p_w, p_vpos, 20, p_h)
    r["left_file"] = Rect(p_hpos + p_w + 25, p_vpos, 20, p_h)
    r["right_file"] = Rect(p_hpos + p_w + 50, p_vpos, 20, p_h)

    x, field_w = p_hpos, 60
    r["zscale_text"] = Rect(x, fields_v + 2, 42, 13)
    r["zscale"] = Rect(x, fields_v - field_h, field_w, field_h)
    r["log_check"] = Rect(p_hpos + 5, fields_v2, 12, 12)
    r["log_text"] = Rect(p_hpos + 20, fields_v2, 50, 13)
    r["deflicker_check"] = Rect(p_hpos + 5, fields_v3, 12, 12)
    r["deflicker_text"] = Rect(p_hpos + 20, fields_v3, 150, 13)

    x += field_w + 20
    w_txt, field_w = 70, 35
    r["bkg_text"] = Rect(x, fields_v, w_txt, 15)
    r["bkg"] = Rect(x, fields_v - field_h, field_w, field_h)
    r["bkg_image"] = Rect(x + field_w + 3, fields_v - field_h * 0.85, 20, 15)
    r["bkg_auto"] = Rect(x + field_w + 25, fields_v - field_h, 28, field_h)
    x += max(field_w, w_txt) + 20
    w_txt, field_w = 65, 35
    r["filter_text"] = Rect(x, fields_v, w_txt, 15)
    r["filter"] = Rect(x, fields_v - field_h, field_w, field_h)
    x += max(field_w, w_txt) + 20
    w_txt, field_w = 50, 40
    r["pixelsize_text"] = Rect(x, fields_v, w_txt, 15)
    r["pixelsize"] = Rect(x, fields_v - field_h, field_w, field_h)
    x += max(field_w, w_txt) + 20
    w_txt, field_w = 90, 60
    r["energy_text"] = Rect(x, fields_v, w_txt, 15)
    r["energy"] = Rect(x, fields_v - field_h, field_w, field_h)
    x += max(field_w, w_txt) + 20 / 2
    w_txt, field_w = 0, 40
    r["pixkill"] = Rect(x, fields_v - 20, field_w, field_h)
    x += max(field_w, w_txt) + 20
    w_txt, field_w = 70, 70
    r["diam_text"] = Rect(x, fields_v, w_txt, 15)
    r["diam"] = Rect(x, fields_v - field_h, field_w, field_h)
    x += max(field_w, w_txt) + 20
    w_txt, field_w = 80, 80
    r["norm_text"] = Rect(x, fields_v, w_txt, 15)
    r["norm"] = Rect(x, fields_v - field_h, field_w, field_h)

    r["help"] = Rect(p_hpos + 1 * p_w / 4, fields_v2 - 0.35 * p_h, 100, p_h)
    r["image_num"] = Rect(p_hpos + 2 * p_w / 4, fields_v2 - 0.35 * p_h, 100, p_h)

    for k, name in enumerate(["energy_button", "area_button", "power_button", "size_xy_button",
                              "size_angle_button", "roi_menu"]):
        r[name] = Rect(b_hpos, b_vpos - k * (b_h + 5), b_w, b_h)
    r["crosshair"] = Rect(b_hpos, b_vpos - 6 * (b_h + 5), 30, b_h)
    r["profiles"] = Rect(b_hpos + 50, b_vpos - 6 * (b_h + 5), 70, b_h)
    r["auto_aperture"] = Rect(b_hpos, b_vpos - 7 * (b_h + 5), b_w, b_h)
    r["msgbox"] = Rect(b_hpos, b_vpos - 7.5 * (b_h + 5) - m_h, m_w, m_h)

    # Image area (ImagePlotBoundaries): up to the message box on the right, under the controls.
    right = r["msgbox"].x
    bottom = 22                                  # leaves a line for the "Pixel info" text
    top = r["deflicker_check"].y - 5            # under the check boxes (and the image menu)
    r["plot"] = Rect(0, bottom, right, top - bottom)
    r["pixel_info"] = Rect(5, 2, right - 5, 18)
    return r


RECTS = _compute()


def to_qt(rect: Rect, width: float = WIN_W, height: float = WIN_H) -> tuple[int, int, int, int]:
    """``(x, y, w, h)`` in Qt pixels (origin top-left) of ``rect`` for a window ``width`` x ``height``."""
    sx, sy = width / WIN_W, height / WIN_H
    return (round(rect.x * sx), round((WIN_H - rect.y - rect.h) * sy), round(rect.w * sx), round(rect.h * sy))
