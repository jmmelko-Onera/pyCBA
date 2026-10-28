"""Aperture (ROI), crosshair and measurement marks drawn over the image (pyqtgraph items).

The Session is the source of truth: ``sync()`` rebuilds the items from it, and the items write back what the user
does with the mouse (no modal "draw the aperture" step like MATLAB's imrect(gca, []): the aperture appears at once
around the beam peak with handles to move and resize it)."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from ..core import roi as geom
from ..core.measure import ellipse_points

ROI_PEN = pg.mkPen("y", width=2)
CROSS_PEN = pg.mkPen("c", width=1)                      # thin line...
CROSS_HOVER = pg.mkPen((0, 255, 255, 80), width=13)     # ...but a wide grab zone (+-7 px), highlighted under the mouse
GRAB_PX = 10                                             # aperture: only a strip this wide (inside its border) reacts to the mouse
ROI_HOVER = pg.mkPen((255, 255, 0, 70), width=2 * GRAB_PX)   # grey-ish band shown under the mouse (clipped to the inside)
CROSS_Z, ROI_Z = 20, 10                                  # the crosshair lines stay above the aperture
TITLE_PLACEHOLDER = "\u00a0"                             # keeps the row of the ROI text reserved: the image does not jump


def set_title(plot, text: str) -> None:
    plot.setTitle(text or TITLE_PLACEHOLDER, size="9pt")


def eight_handles(item, lock_aspect: bool) -> None:
    """Handles at the corners and (unless the aspect is locked) at the middle of the sides, like MATLAB's imrect /
    imellipse; no rotation handle and no diagonal scaler."""
    for h in list(item.getHandles()):
        item.removeHandle(h)
        if h.scene() is None:          # removeHandle only deletes the graphics item of a ROI that is already in a scene:
            h.setParentItem(None)      # otherwise the default handles (circle at 0 deg, diamond at -45 deg) stay drawn
    for x, y in ((0, 0), (1, 0), (1, 1), (0, 1)):
        item.addScaleHandle([x, y], [1 - x, 1 - y])
    if not lock_aspect:
        for x, y in ((0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)):
            item.addScaleHandle([x, y], [1 - x if x != 0.5 else 0.5, 1 - y if y != 0.5 else 0.5])
MARK_PEN = pg.mkPen("m", width=2)


class _BorderOnly:
    """Mixin: an aperture is transparent to the mouse except in a strip of ``GRAB_PX`` pixels along its border (inside it),
    so what lies below (the crosshair, the pixel killer) stays reachable. Hit zone = hover band = that strip."""

    def _fill_path(self) -> QtGui.QPainterPath:
        raise NotImplementedError

    def _grab(self) -> float:
        """GRAB_PX in item units."""
        try:
            px = [v.length() for v in self.pixelVectors() if v is not None]
        except Exception:
            px = []
        if px:
            return GRAB_PX * sum(px) / len(px)
        w, h = self.state["size"]
        return 0.03 * max(abs(w), abs(h))

    def shape(self):
        fill = self._fill_path()
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(2 * self._grab())
        return stroker.createStroke(fill).intersected(fill)

    def paint(self, p, *args):
        if self.mouseHovering:
            fill = self._fill_path()
            p.save()
            p.setClipPath(fill)
            p.setPen(ROI_HOVER)
            p.drawPath(fill)
            p.restore()
        super().paint(p, *args)


class _RectROI(_BorderOnly, pg.RectROI):
    def _fill_path(self):
        path = QtGui.QPainterPath()
        path.addRect(QtCore.QRectF(0, 0, *self.state["size"]).normalized())
        return path


class _EllipseROI(_BorderOnly, pg.EllipseROI):
    def _fill_path(self):
        path = QtGui.QPainterPath()
        path.addEllipse(QtCore.QRectF(0, 0, *self.state["size"]).normalized())
        return path


class _PolyROI(_BorderOnly, pg.PolyLineROI):
    def _fill_path(self):
        path = QtGui.QPainterPath()
        pts = [h["item"].pos() for h in self.handles]
        if pts:
            path.moveTo(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
            path.closeSubpath()
        return path


class RoiOverlay:
    def __init__(self, plot: pg.PlotItem, session) -> None:
        self.plot, self.session = plot, session
        self.item = None
        self._busy = False

    def clear(self) -> None:
        if self.item is not None:
            self.plot.removeItem(self.item)
            self.item = None
        set_title(self.plot, "")

    def sync(self) -> None:
        """Rebuilds the item from ``session.roi_kind`` / ``roi_pos``."""
        self.clear()
        s = self.session
        lim = s.limits()
        if not s.roi_kind or s.roi_pos is None or lim is None:
            return
        kind, pos = s.roi_kind, np.asarray(s.roi_pos, dtype=float)
        bounds = QtCore.QRectF(lim[0][0], lim[1][0], lim[0][1] - lim[0][0], lim[1][1] - lim[1][0])
        common = dict(pen=ROI_PEN, rotatable=False)
        aspect_locked = kind in (geom.SQUARE, geom.CIRCLE)
        if kind in (geom.RECT, geom.SQUARE):
            item = _RectROI(pos[:2], pos[2:4], aspectLocked=kind == geom.SQUARE, maxBounds=bounds, **common)
        elif kind in (geom.ELLIPSE, geom.CIRCLE):
            item = _EllipseROI(pos[:2], pos[2:4], aspectLocked=kind == geom.CIRCLE, maxBounds=bounds, **common)
        else:
            item = _PolyROI([tuple(p) for p in pos.reshape(-1, 2)], closed=True, pen=ROI_PEN, hoverPen=ROI_HOVER)
        if not isinstance(item, pg.PolyLineROI):
            eight_handles(item, aspect_locked)
        item.setZValue(ROI_Z)
        item.sigRegionChanged.connect(self._changed)
        self.plot.addItem(item)
        self.item = item
        set_title(self.plot, s.roi_text())

    def read(self):
        """Position of the item in the same form as ``session.roi_pos``."""
        item = self.item
        if isinstance(item, pg.PolyLineROI):
            pts = [item.mapToParent(h.pos()) for h in item.getHandles()]
            return np.array([[p.x(), p.y()] for p in pts])
        return np.array([item.pos().x(), item.pos().y(), item.size().x(), item.size().y()])

    def _changed(self, *_args) -> None:
        if self.item is None:
            return
        self.session.set_roi(self.session.roi_kind, self.read())
        set_title(self.plot, self.session.roi_text())


class CrosshairOverlay:
    def __init__(self, plot: pg.PlotItem, session) -> None:
        self.plot, self.session = plot, session
        self.items: list = []
        self._busy = False

    def clear(self) -> None:
        for it in self.items:
            self.plot.removeItem(it)
        self.items = []

    def _add(self, item):
        item.setZValue(CROSS_Z)
        self.plot.addItem(item)
        self.items.append(item)
        return item

    def sync(self) -> None:
        self.clear()
        s = self.session
        lim = s.limits()
        ch = s.crosshair
        if lim is None or ch.kind == "O":
            return
        (x0, x1), (y0, y1) = lim
        if ch.kind == "+":
            self.hline = self._add(pg.InfiniteLine(pos=ch.y, angle=0, movable=True, pen=CROSS_PEN, hoverPen=CROSS_HOVER, bounds=(y0, y1)))
            self.vline = self._add(pg.InfiniteLine(pos=ch.x, angle=90, movable=True, pen=CROSS_PEN, hoverPen=CROSS_HOVER, bounds=(x0, x1)))
            self.hline.sigPositionChanged.connect(self._plus_moved)
            self.vline.sigPositionChanged.connect(self._plus_moved)
        elif ch.kind == "X":
            self.pline = self._add(pg.InfiniteLine(pos=(ch.c / 2, ch.c / 2), angle=-45, movable=True, pen=CROSS_PEN, hoverPen=CROSS_HOVER))
            self.mline = self._add(pg.InfiniteLine(pos=(ch.d, 0.0), angle=45, movable=True, pen=CROSS_PEN, hoverPen=CROSS_HOVER))
            self.pline.sigPositionChanged.connect(self._diag_moved)
            self.mline.sigPositionChanged.connect(self._diag_moved)
        else:
            self.l1 = self._add(pg.InfiniteLine(pos=(ch.cx, ch.cy), angle=math.degrees(ch.theta), movable=False,
                                                pen=CROSS_PEN))
            self.l2 = self._add(pg.InfiniteLine(pos=(ch.cx, ch.cy), angle=math.degrees(ch.theta) + 90, movable=False,
                                                pen=CROSS_PEN))
            self.center = self._add(pg.TargetItem(pos=(ch.cx, ch.cy), size=16, symbol="+", pen=pg.mkPen("c", width=2), movable=True))
            arm = geom.ray_to_box((ch.cx, ch.cy), (math.cos(ch.theta), math.sin(ch.theta)), *lim)
            self.arm = self._add(pg.TargetItem(pos=tuple(arm), size=14, symbol="o", pen=pg.mkPen("c", width=2), movable=True))
            self.center.sigPositionChanged.connect(self._center_moved)
            self.arm.sigPositionChanged.connect(self._arm_moved)

    def _guard(self):
        return self._busy

    def _plus_moved(self, *_a) -> None:
        ch = self.session.crosshair
        ch.x, ch.y = self.vline.value(), self.hline.value()

    def _diag_moved(self, *_a) -> None:
        if self._busy:
            return
        ch = self.session.crosshair
        lim = self.session.limits()
        p, m = self.pline.pos(), self.mline.pos()
        ch.c, ch.d = p.x() + p.y(), m.x() - m.y()
        ch.clamp(*lim)
        self._busy = True
        try:
            self.pline.setPos((ch.c / 2, ch.c / 2))
            self.mline.setPos((ch.d, 0.0))
        finally:
            self._busy = False

    def _update_l(self) -> None:
        ch = self.session.crosshair
        lim = self.session.limits()
        self._busy = True
        try:
            for ln, extra in ((self.l1, 0), (self.l2, 90)):
                ln.setPos((ch.cx, ch.cy))
                ln.setAngle(math.degrees(ch.theta) + extra)
            self.center.setPos((ch.cx, ch.cy))
            self.arm.setPos(tuple(geom.ray_to_box((ch.cx, ch.cy), (math.cos(ch.theta), math.sin(ch.theta)), *lim)))
        finally:
            self._busy = False

    def _center_moved(self, *_a) -> None:
        if self._busy:
            return
        ch = self.session.crosshair
        p = self.center.pos()
        ch.cx, ch.cy = p.x(), p.y()
        ch.clamp(*self.session.limits())
        self._update_l()

    def _arm_moved(self, *_a) -> None:
        if self._busy:
            return
        ch = self.session.crosshair
        p = self.arm.pos()
        ch.theta = geom.angle_from_mouse((ch.cx, ch.cy), (p.x(), p.y()), ch.theta)
        self._update_l()


class MarksOverlay:
    """Centroid '+' and diameter ellipse drawn by the ISO beam size buttons (removed at the next redraw)."""

    def __init__(self, plot: pg.PlotItem) -> None:
        self.plot = plot
        self.items: list = []

    def clear(self) -> None:
        for it in self.items:
            self.plot.removeItem(it)
        self.items = []

    def draw(self, x0, y0, diam_h, diam_v, angle_deg: float = 0.0) -> None:
        self.clear()
        if not (np.isfinite(x0) and np.isfinite(y0)):
            return
        self.items.append(self.plot.plot([x0], [y0], pen=None, symbol="+", symbolSize=14, symbolPen=MARK_PEN))
        if np.isfinite(diam_h) and np.isfinite(diam_v):
            xs, ys = ellipse_points(x0, y0, diam_h, diam_v, angle_deg)
            self.items.append(self.plot.plot(xs, ys, pen=MARK_PEN))
