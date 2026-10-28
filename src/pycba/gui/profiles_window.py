"""Window of the profiles (``figure('Name',...)`` of the Profile*Button callbacks): the two profiles and their
Gaussian fits."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


class ProfilesWindow(QtWidgets.QWidget):
    def __init__(self, run, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(run.title)
        self.resize(800, 520)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        lay.addWidget(self.plot)
        pi = self.plot.getPlotItem()
        pi.addLegend()
        pi.showGrid(x=True, y=True, alpha=0.2)
        pi.setLabel("bottom", "Position")
        pi.setLabel("left", "Intensity")
        first, second = run.pair.first, run.pair.second
        fits = run.analysis.fits
        spec = ((first, run.names[0], "o", "r", fits.h), (second, run.names[1], "+", "b", fits.v))
        for prof, name, symbol, color, fit in spec:
            pi.plot(prof.vector, prof.values, pen=None, symbol=symbol, symbolSize=7, symbolPen=color,
                    symbolBrush=None if symbol == "+" else None, name=name)
            if np.size(fit.xfit):
                pi.plot(fit.xfit, fit.yfit, pen=pg.mkPen(color, width=2), name=f"{name} fit")
        pi.autoRange()
        self.run = run
