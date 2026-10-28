"""Image toolbar (zoom in, zoom out, pan, home): equivalent of the MATLAB axes toolbar.

Zoom in : drag a rectangle, or click to zoom x2 around the click.
Zoom out: click to zoom out x2 around the click.
Pan     : drag the image.
None    : mouse does nothing (a pressed button is released by clicking it again).
Home    : back to the full image.
"""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets   # the binding already loaded (PyQt5 in Anaconda), else PySide6

ZOOM_IN, ZOOM_OUT, PAN, NONE = "zoom_in", "zoom_out", "pan", "none"


def _icon(kind: str) -> QtGui.QIcon:
    pm = QtGui.QPixmap(32, 32)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(40, 40, 40), 2.4)
    p.setPen(pen)
    if kind in (ZOOM_IN, ZOOM_OUT):
        p.drawEllipse(QtCore.QRectF(4, 4, 18, 18))
        p.drawLine(20, 20, 28, 28)
        p.drawLine(9, 13, 17, 13)
        if kind == ZOOM_IN:
            p.drawLine(13, 9, 13, 17)
    elif kind == PAN:
        c = 16
        p.drawLine(c, 4, c, 28)
        p.drawLine(4, c, 28, c)
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            x, y = c + 12 * dx, c + 12 * dy
            p.drawLine(x, y, x - 3 * dx + 3 * dy, y - 3 * dy + 3 * dx)
            p.drawLine(x, y, x - 3 * dx - 3 * dy, y - 3 * dy - 3 * dx)
    else:  # home
        p.drawPolyline([QtCore.QPointF(5, 16), QtCore.QPointF(16, 6), QtCore.QPointF(27, 16)])
        p.drawPolyline([QtCore.QPointF(8, 14), QtCore.QPointF(8, 27), QtCore.QPointF(24, 27), QtCore.QPointF(24, 14)])
    p.end()
    return QtGui.QIcon(pm)


class ImageToolbar(QtCore.QObject):
    """Floating tool buttons laid over ``host`` (top right) and acting on a pyqtgraph PlotItem."""

    def __init__(self, host: QtWidgets.QWidget, plot) -> None:
        super().__init__(host)
        self.host, self.plot, self.vb = host, plot, plot.vb
        self.home_rect: tuple[float, float, float, float] | None = None    # left, right, top, bottom
        self.mode = NONE
        self.panel = QtWidgets.QWidget(host)
        self.panel.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        lay = QtWidgets.QHBoxLayout(self.panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.buttons: dict[str, QtWidgets.QToolButton] = {}
        tips = {ZOOM_IN: "Zoom in", ZOOM_OUT: "Zoom out", PAN: "Pan", "home": "Restore original view"}
        for key, tip in tips.items():
            b = QtWidgets.QToolButton(self.panel)
            b.setIcon(_icon(key))
            b.setIconSize(QtCore.QSize(18, 18))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.setFixedSize(24, 24)
            if key == "home":
                b.clicked.connect(self.home)
            else:
                b.setCheckable(True)
                b.clicked.connect(lambda checked, k=key: self.set_mode(k if checked else NONE))
            self.buttons[key] = b
            lay.addWidget(b)
        self.panel.adjustSize()
        host.installEventFilter(self)
        self._place()
        self.panel.raise_()
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self.set_mode(NONE)

    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self.host and event.type() == QtCore.QEvent.Resize:
            self._place()
        return False

    def _place(self) -> None:
        self.panel.move(max(0, self.host.width() - self.panel.width() - 6), 4)
        self.panel.raise_()

    def set_home(self, left: float, right: float, top: float, bottom: float) -> None:
        self.home_rect = (left, right, top, bottom)

    def home(self) -> None:
        if self.home_rect is not None:
            left, right, top, bottom = self.home_rect
            self.plot.setRange(xRange=(left, right), yRange=(top, bottom), padding=0)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        for k, b in self.buttons.items():
            if k != "home":
                b.blockSignals(True)
                b.setChecked(k == mode)
                b.blockSignals(False)
        pan = mode == PAN
        self.vb.setMouseEnabled(pan, pan)
        self.vb.setMouseMode(self.vb.RectMode if mode == ZOOM_IN else self.vb.PanMode)
        if mode == ZOOM_IN:
            self.vb.setMouseEnabled(True, True)
        elif mode == ZOOM_OUT:
            self.vb.setMouseEnabled(False, False)

    def zoom_at(self, x: float, y: float, factor: float) -> None:
        """Scales the view by ``factor`` (<1 zooms in) keeping (x, y) at the same relative position."""
        (x0, x1), (y0, y1) = self.vb.viewRange()
        fx, fy = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
        w, h = (x1 - x0) * factor, (y1 - y0) * factor
        self.plot.setRange(xRange=(x - fx * w, x - fx * w + w), yRange=(y - fy * h, y - fy * h + h), padding=0)

    def _clicked(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton or self.mode not in (ZOOM_IN, ZOOM_OUT):
            return
        if not self.vb.sceneBoundingRect().contains(event.scenePos()):
            return
        p = self.vb.mapSceneToView(event.scenePos())
        self.zoom_at(p.x(), p.y(), 0.5 if self.mode == ZOOM_IN else 2.0)
