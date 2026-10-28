"""Main window of pyCBA (PyQt5 or PySide6, with pyqtgraph).

Widgets, positions, labels and tooltips are those of ``MAIN_CameraBeamAnalyzer.m``. Like MATLAB's
``Units='normalized'`` they follow the window size (``resizeEvent``). Only the callbacks whose
MATLAB code was provided are wired; the other controls post a "not ported yet" message.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets   # the binding already loaded (PyQt5 in Anaconda), else PySide6

from .imagetoolbar import ImageToolbar
from .overlays import CrosshairOverlay, MarksOverlay, RoiOverlay, set_title
from .profiles_window import ProfilesWindow
from . import layout
from .session import DIAM_METHODS, IMAGE_NAMES, ROI_TYPES, PathPlan, Session

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

TITLE = "Laser beam analyzer from 2D cameras images, with aperture tool"


def _fallback_colormap() -> pg.ColorMap:
    """Used when the Spiricon palette cannot be read (MATLAB then keeps its default colormap)."""
    return pg.colormap.get("viridis")


def _palette_to_colormap(rgb: np.ndarray) -> pg.ColorMap:
    return pg.ColorMap(np.linspace(0, 1, len(rgb)), (np.clip(rgb, 0, 1) * 255).astype(np.uint8))


HELP_TEXT = "\n".join([
    "1/ Open an ASCII camera image",
    "It must be a text file containing numbers arranged in rows (separated by line feed or carriage return)",
    "and columns separated by commas, semicolons, spaces, tabs...",
    "",
    "2/ Evaluate the background power and substract it to the image:",
    "- Choose an aperture",
    "- Select a large region without the laser beam",
    "- Click Average Power",
    "- Enter this value in the Bkg substract field",
    "Or click Auto, next to the field: the background is the median of a 5 % frame along the image edges",
    "(not reliable if the beam reaches the edges: then use the method above).",
    "",
    "3/ Compute the diameter",
    "- Enter the pixel size (in any unit you want)",
    "- Choose a beam diameter computation method (default is ISO1146 4sigma)",
    "- Click on ISO beam size",
    "",
    "4/ Compute the energy density",
    "- Choose an aperture",
    "- Encircle tightly the laser beam",
    "- Enter the beam energy in real world unit (U)",
    "- Enter the pixel size in real world unit (h)",
    "- click on Encircled Energy: the result is in U/h\u00b2",
    "",
    "5/ Compute the 1-D profiles",
    "- Click on the + cross button",
    "- Click again to change the profiles directions",
    "- Move the lines, one by one, over the beam",
    "- Click the Profiles button",
])

CONFIRM_CLOSE = True      # asks before closing (myCloseReq); the tests switch it off


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, session: Session | None = None, path: str | None = None):
        super().__init__()
        self.session = session or Session()
        self.setWindowTitle(TITLE)
        self._build_widgets()
        self._build_plot()
        self.resize(layout.DEFAULT_W, layout.DEFAULT_H)
        self._set_enabled(False)
        self.left_btn.setEnabled(False)               # enabled once several files are found
        self.right_btn.setEnabled(False)
        if self.session.colormap is None and not self.session.messages:
            self.session.load_palette()
        if path:
            self.path_edit.setText(str(path))
            self.on_path_entered()
        else:
            self.session.say("No file found.")
        self._flush_messages()

    # ------------------------------------------------------------------ widgets
    def _build_widgets(self) -> None:
        self.canvas = QtWidgets.QWidget(self)
        self.setCentralWidget(self.canvas)
        W = self.canvas
        self.rects: dict[str, QtWidgets.QWidget] = {}

        def add(key, widget):
            widget.setParent(W)
            self.rects[key] = widget
            return widget

        def label(key, text, bold=False):
            w = add(key, QtWidgets.QLabel(text))
            if bold:
                f = w.font(); f.setBold(True); w.setFont(f)
            return w

        def button(key, text, slot, tip=""):
            w = add(key, QtWidgets.QPushButton(text))
            w.setToolTip(tip)
            w.clicked.connect(slot)
            return w

        def edit(key, text, slot, tip=""):
            w = add(key, QtWidgets.QLineEdit(text))
            w.setToolTip(tip)
            w.editingFinished.connect(slot)
            return w

        def combo(key, items, slot, tip=""):
            w = add(key, QtWidgets.QComboBox())
            w.addItems(items)
            w.setToolTip(tip)
            w.activated.connect(slot)
            return w

        def check(key, slot, tip=""):
            w = add(key, QtWidgets.QCheckBox())
            w.setToolTip(tip)
            w.toggled.connect(slot)
            return w

        n = IMAGE_NAMES
        self.path_edit = add("path", QtWidgets.QLineEdit())
        self.path_edit.editingFinished.connect(self.on_path_entered)
        self.browse_btn = button("browse", "...", self.on_browse,
                                 "Select an image file, in 2D ASCII format.\n"
                                 "The program automatically searches for similar files in the same folder.")
        self.left_btn = button("left_file", "<--", lambda: self.step_file(-1), "Previous file")
        self.right_btn = button("right_file", "-->", lambda: self.step_file(+1), "Next file")

        label("zscale_text", "Z scale")
        self.zscale_menu = combo("zscale", ["Normalize"] + [str(z) for z in self.session.zscales],
                                 self.on_zscale,
                                 "Color scale in raw camera units.\nChoose 'normalized' to auto-adjust to the maximum value.")
        self.log_check = check("log_check", self.on_log, "Switches color scale to log to better see low signals.")
        label("log_text", "Log scale")
        self.deflicker_check = check("deflicker_check", self.on_deflicker,
                                     "Removes the horizontal frame noise: a constant offset per line, estimated on "
                                     "the left & right edges, is subtracted from each line.")
        label("deflicker_text", "Flicker removal")

        label("bkg_text", "Bkg substract")
        self.bkg_edit = edit("bkg", "0", self.on_bkg,
                             "Enter background value and press enter to substract the background.\n"
                             "Integer value in camera file scale.")
        self.bkg_image_label = label("bkg_image", f"({n[0]})")
        self.bkg_auto_btn = button("bkg_auto", "Auto", self.on_auto_bkg,
                                   "Estimates the background: median of a frame 5 % wide along the 4 edges of the image\n"
                                   "(what remains after flicker removal), and writes it in the field.\n"
                                   "Not reliable if the beam reaches the edges: then enter the value by hand.")
        label("filter_text", "Filter param.")
        self.filter_edit = edit("filter", "1", self.on_filter,
                                "Enter an integer value >1 and press enter to smooth the image")
        label("pixelsize_text", "Pixel size")
        self.pixelsize_edit = edit("pixelsize", "1", self.on_pixelsize,
                                   "Pixel size in real-world units (µm, mm, cm). Affects beam size and energy "
                                   "densities computations.\nPyrocam III (124x124) = 85 µm\n"
                                   "Pyrocam IIIHR / IV (320x320) = 80 µm\nDataray WinCamD-FIR2-16 (392x296) = 35 µm\n"
                                   "USBeamPro CCD (1280x1024) = 6.7 µm\nGentec Beamage 3.0 CMOS (2048x1088) = 5.5 µm\n"
                                   "Spiricon SP928 CCD / Newport LBP2-HR-VIS2 (1448x1928) = 3.69 µm\n"
                                   "Spiricon SP-1550m (640x480) = 8.4x9.8 µm\nWinCamD-LCM (2048x2048) = 5.5 µm\n"
                                   "WinCamD-IR-BB () = 17 µm")
        label("energy_text", "Energy in aperture")
        self.energy_edit = edit("energy", "1", self.on_energy,
                                "Energy in the aperture, in real-world units (J, mJ, W)")
        self.pixkill_btn = button("pixkill", "Pix Kill", self.on_pixkill,
                                  "Click to switch to pixel killing mode.\nShooting a pixel twice restores it.\n"
                                  "Reload image to restore pixels.")
        self.pixkill_btn.setCheckable(True)
        label("diam_text", "Diam. method")
        self.diam_menu = combo("diam", DIAM_METHODS, self.on_diam,
                               "Switches between different measurements of the diameter.")
        label("norm_text", "Norm. method")
        self.norm_menu = combo("norm", ["(none)", f"{n[0]}-{n[1]}", f"{n[0]}/{n[1]}",
                                        f"({n[0]}-{n[1]}-{n[2]})/4*sqrt({n[1]}*{n[2]})"],
                               self.on_norm, "IMAGE/Reference normalisation operation")
        self.help_btn = button("help", "HELP!", self.on_help, "Push for help")
        self.image_menu = combo("image_num", [f"IMAGE ({n[0]})", f"Ref. ({n[1]})", f"Ref. ({n[2]})"],
                                self.on_image_menu, "Switches between image and reference/background image")

        self.energy_btn = button("energy_button", "Encircled Energy", self.on_energy_button,
                                 "Computes the energy within the aperture (or the whole image if undefined).\n"
                                 "Uses the 'pixel size' and 'energy' fields.")
        self.area_btn = button("area_button", "Encircled Area", self.on_area_button,
                               "Computes the area of the aperture (or the whole image if undefined).\n"
                               "Uses the 'pixel size' field.")
        self.power_btn = button("power_button", "Average Power", self.on_power_button,
                                "Computes the power=energy/area, within the aperture (or the whole image if undefined).\n"
                                "Uses the 'pixel size' and 'energy' fields.")
        self.size_xy_btn = button("size_xy_button", "ISO Beam size", self.on_size_xy,
                                  "Computes the 4 sigma ISO beam diameters along X and Y\n + the energy density if the "
                                  "beam were Gaussian\n+ the maximum energy density (hot spots)\n"
                                  "Uses the 'pixel size' and 'energy' fields.")
        self._size_angle_tip = ("Computes the ISO 11146 beam diameters along the PRINCIPAL axes (4 sigma, second "
                                "moments):\nmajor and minor diameters, angle of the major axis (counter-clockwise "
                                "as seen on screen) and ellipticity.\nUses the 'pixel size' field.")
        self.size_angle_btn = button("size_angle_button", "ISO Beam size angle", self.on_size_angle, self._size_angle_tip)
        self.roi_menu = combo("roi_menu", ROI_TYPES, self.on_roi_menu,
                              "Draw an aperture on the camera image to ignore other pixels during computations")
        self.crosshair_btn = button("crosshair", "+", self.on_crosshair,
                                    "Place a crosshair to plot 2D profiles.\n+ : straight crosshairs\n"
                                    "X : 45° diagonals\n~: diagonals\nO : deletes crosshair")
        self.profiles_btn = button("profiles", "Profiles", self.on_profiles,
                                   "Plot 2D profiles along crosshair and Gaussian fits (in a new window).")
        self.auto_btn = button("auto_aperture", "Auto Aperture", self.on_auto_aperture,
                               "Auto Aperture\n")
        self.msgbox = add("msgbox", QtWidgets.QPlainTextEdit())
        self.msgbox.setMaximumBlockCount(500)          # replaces updatemsgtxt: appends, oldest lines drop out
        self.msgbox.setReadOnly(True)
        self.pixel_info = label("pixel_info", "Pixel info:  (X, Y)  Intensity", bold=True)

        self._measure_buttons = [self.energy_btn, self.area_btn, self.power_btn, self.size_xy_btn, self.size_angle_btn,
                                 self.auto_btn, self.crosshair_btn, self.profiles_btn, self.roi_menu]
        self._controls = [self.energy_btn, self.area_btn, self.power_btn, self.size_xy_btn, self.size_angle_btn,
                          self.auto_btn, self.crosshair_btn, self.profiles_btn, self.roi_menu, self.log_check,
                          self.rects["log_text"], self.deflicker_check, self.rects["deflicker_text"], self.bkg_edit, self.bkg_auto_btn,
                          self.filter_edit, self.energy_edit, self.pixelsize_edit, self.pixkill_btn,
                          self.zscale_menu, self.diam_menu, self.image_menu, self.norm_menu]

    def _build_plot(self) -> None:
        self.graphics = pg.GraphicsLayoutWidget(self.canvas)
        self.rects["plot"] = self.graphics
        self.graphics.setBackground(self.palette().color(QtGui.QPalette.Window))
        self.plot = self.graphics.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.hideButtons()
        self.plot.showAxis("top", False)
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self.colorbar = pg.ColorBarItem(values=(0, 100), colorMap=self._colormap(), width=22, interactive=False)
        self.colorbar.setImageItem(self.image_item, insert_in=self.plot)
        self._proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)
        self.plot.setRange(xRange=(0, 100), yRange=(0, 100), padding=0)
        set_title(self.plot, "")                         # room for the ROI text from the start
        self.toolbar = ImageToolbar(self.graphics, self.plot)
        self.roi_overlay = RoiOverlay(self.plot, self.session)
        self.cross_overlay = CrosshairOverlay(self.plot, self.session)
        self.marks = MarksOverlay(self.plot)
        self.profile_windows: list = []
        self.plot.scene().sigMouseClicked.connect(self._image_clicked)
        self.toolbar.set_home(0, 100, 0, 100)

    def closeEvent(self, event) -> None:  # noqa: N802
        """``myCloseReq``: closing by mistake loses the work, so it is confirmed (default answer: No)."""
        if CONFIRM_CLOSE:
            answer = QtWidgets.QMessageBox.question(
                self, "Close", "Do you really want to close the main window?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        for win in self.profile_windows:
            win.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------ geometry
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        w, h = self.canvas.width(), self.canvas.height()
        for key, widget in self.rects.items():
            x, y, ww, hh = layout.to_qt(layout.RECTS[key], w, h)
            widget.setGeometry(x, y, ww, hh)

    # ------------------------------------------------------------------ helpers
    def _colormap(self) -> pg.ColorMap:
        cm = self.session.colormap
        return _fallback_colormap() if cm is None else _palette_to_colormap(cm)

    def _set_enabled(self, on: bool) -> None:
        for w in self._controls:
            w.setEnabled(on)
        self.image_menu.setEnabled(True)               # always enabled, like in MATLAB
        if on:
            self._refresh_size_angle_button(True)

    def _refresh_size_angle_button(self, has_image: bool) -> None:
        """The principal axes are 4 sigma (second moments) only: the button is greyed out for the other diameter
        methods, and its tooltip says why (a tooltip is still shown on a disabled button)."""
        is_4sigma = self.session.diam_method == 0
        self.size_angle_btn.setEnabled(has_image and is_4sigma)
        self.size_angle_btn.setToolTip(self._size_angle_tip if is_4sigma else
                                       "Not available: the principal axes (major / minor / angle) are computed with "
                                       "the 4sigma method.\nSelect '4sigma' in the diameter menu to use this button.")

    def _flush_messages(self) -> None:
        if self.session.messages:
            self.msgbox.appendPlainText("\n".join(self.session.messages))
            self.session.messages.clear()

    def stub(self, name: str) -> None:
        self.session.say(f"{name}: not ported yet.")
        self._flush_messages()

    # ------------------------------------------------------------------ drawing
    def redraw(self, keep_view: bool = False) -> None:
        """``plot_image(imageNum, keepAxes)``: ``keep_view`` keeps the zoom (same image extent only)."""
        s = self.session
        spec = s.display() if s.has_image() else None
        self.zscale_menu.blockSignals(True)
        self.zscale_menu.setCurrentIndex(s.zscale_menu_index())
        self.zscale_menu.blockSignals(False)
        self.log_check.blockSignals(True)
        self.log_check.setChecked(s.filters.log)
        self.log_check.blockSignals(False)
        self.bkg_image_label.setText(f"({IMAGE_NAMES[s.current]})")
        self.colorbar.setColorMap(self._colormap())
        if spec is None or spec.empty:
            self.image_item.clear()
            self._sync_overlays()
            self.colorbar.setLevels(low=0, high=100)
            self.plot.setRange(xRange=(0, 100), yRange=(0, 100), padding=0)
            self.toolbar.set_home(0, 100, 0, 100)
            self._last_extent = None
            return
        left, right, bottom, top = spec.extent
        data = np.where(np.isfinite(spec.data), spec.data, spec.vmin)     # -Inf (log of 0) -> lowest colour
        self.image_item.setImage(data, autoLevels=False)
        self.image_item.setRect(QtCore.QRectF(left, top, right - left, bottom - top))
        self.colorbar.setLevels(low=spec.vmin, high=spec.vmax)
        if not (keep_view and getattr(self, "_last_extent", None) == spec.extent):
            self.plot.setRange(xRange=(left, right), yRange=(top, bottom), padding=0)
        self._last_extent = spec.extent
        self.toolbar.set_home(left, right, top, bottom)
        self._sync_overlays()

    def _sync_overlays(self) -> None:
        """Aperture and crosshair follow the Session (and the shown image)."""
        s = self.session
        s.check_geometry()
        self.marks.clear()
        self.roi_overlay.sync()
        self.cross_overlay.sync()
        self.roi_menu.blockSignals(True)
        self.roi_menu.setCurrentIndex(s.roi_menu_index())
        self.roi_menu.blockSignals(False)
        self.crosshair_btn.setText(s.next_crosshair_kind())      # the button shows what the next click does
        has = s.has_image() and s.has_image(0) and self.session.limits() is not None
        for b in self._measure_buttons:                  # nothing to measure on an empty image (B or C not loaded)
            b.setEnabled(bool(has))
        self._refresh_size_angle_button(bool(has))
        s.geometry_changed = False
        self._flush_messages()

    # ------------------------------------------------------------------ aperture, measurements, profiles
    def _after_geometry_call(self) -> None:
        if self.session.geometry_changed:
            self._sync_overlays()
        self._flush_messages()

    def on_help(self) -> None:
        """``HelpButton_Callback``."""
        QtWidgets.QMessageBox.information(self, "HELP", HELP_TEXT)

    def on_energy_button(self) -> None:
        self.session.measure_energy()
        self._flush_messages()

    def on_area_button(self) -> None:
        self.session.measure_area()
        self._flush_messages()

    def on_power_button(self) -> None:
        self.session.measure_power()
        self._flush_messages()

    def on_size_xy(self) -> None:
        res = self.session.measure_size_xy()
        if res is not None:
            self._sync_overlays() if self.session.geometry_changed else None
            self.marks.draw(res.x0, res.y0, res.diam_h, res.diam_v)
            QtWidgets.QApplication.clipboard().setText(res.clipboard_text())
            self.session.say("ISO beam sizes copied to clipboard")
        self._flush_messages()

    def on_size_angle(self) -> None:
        res = self.session.measure_size_angle()
        if res is not None:
            self._sync_overlays() if self.session.geometry_changed else None
            p = res.principal
            if np.isfinite(p.angle):                              # the ellipse of the principal axes
                self.marks.draw(res.x0, res.y0, p.d_major, p.d_minor, p.angle_xy)
            else:
                self.marks.draw(res.x0, res.y0, p.d_x, p.d_y)
        self._flush_messages()

    def on_roi_menu(self, index: int) -> None:
        self.session.choose_roi(index)
        self._after_geometry_call()

    def on_auto_aperture(self) -> None:
        self.session.auto_aperture()
        self._after_geometry_call()

    def on_crosshair(self) -> None:
        self.session.set_crosshair(self.session.next_crosshair_kind())
        self._after_geometry_call()

    def on_profiles(self) -> None:
        run = self.session.run_profiles()
        self._flush_messages()
        if run is not None:
            win = ProfilesWindow(run)
            win.show()
            self.profile_windows.append(win)

    def on_pixkill(self, checked: bool) -> None:
        """Pix Kill is a mode: while it is pressed, a click on the image kills / restores the pixel."""
        if checked:
            self.toolbar.set_mode("none")

    def _image_clicked(self, event) -> None:
        if not self.pixkill_btn.isChecked() or event.button() != QtCore.Qt.LeftButton:
            return
        if self.toolbar.mode != "none" or not self.plot.vb.sceneBoundingRect().contains(event.scenePos()):
            return
        p = self.plot.vb.mapSceneToView(event.scenePos())
        ps = self.session.filters.pixelsize
        if self.session.toggle_pixel(int(round(p.y() / ps)), int(round(p.x() / ps))):
            self.redraw(keep_view=True)
        self._flush_messages()

    def _mouse_moved(self, args) -> None:
        pos = args[0]
        s = self.session
        if not s.has_image() or s.image.filt is None or not self.plot.sceneBoundingRect().contains(pos):
            return
        p = self.plot.vb.mapSceneToView(pos)
        ps = s.filters.pixelsize
        col, row = int(round(p.x() / ps)), int(round(p.y() / ps))
        filt = s.image.filt
        if 1 <= col <= filt.shape[1] and 1 <= row <= filt.shape[0]:
            self.pixel_info.setText(f"Pixel info:  ({col * ps:g}, {row * ps:g})  {filt[row - 1, col - 1]:g}")

    # ------------------------------------------------------------------ files
    def open_path(self, path, index: int | None = None, cur_file: int = 1, num_files: int = 1) -> None:
        """``load_imagefile`` + what surrounds it in the callbacks (path box, widgets)."""
        s = self.session
        index = s.current if index is None else index
        res = s.open_file(path, index, cur_file, num_files)
        if res.ok:
            self._set_enabled(True)
            s.current = index
            self.image_menu.setCurrentIndex(index)
            if index == 0:
                self.energy_edit.setText("")
            self.path_edit.setText(str(path))
            self.redraw()
        elif res.cleared and index == 0:
            self._set_enabled(False)
            self.roi_menu.setCurrentIndex(0)
            self.redraw()
        self._flush_messages()

    def _apply_plan(self, plan: PathPlan | None) -> None:
        if plan is None:
            return
        self.left_btn.setEnabled(plan.nav_enabled)
        self.right_btn.setEnabled(plan.nav_enabled)
        if plan.path_text is not None:
            self.path_edit.setText(plan.path_text)
        if plan.load:
            self.open_path(*plan.load)
        self._flush_messages()

    def on_path_entered(self) -> None:
        """``path_Callback``."""
        self._apply_plan(self.session.enter_path(self.path_edit.text()))

    def on_browse(self) -> None:
        """``BrowseButton_Callback`` (MATLAB also left a ``profile on`` there: not reproduced)."""
        name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose file...", self.session.browse_start_dir())
        if name:                                       # cancelled: nothing (MATLAB only warned in the console)
            self.path_edit.setText(name)
            self.on_path_entered()

    def step_file(self, step: int) -> None:
        """``LeftButton_Callback`` / ``RightButton_Callback``."""
        self._apply_plan(self.session.step_file(step))

    # ------------------------------------------------------------------ callbacks
    def apply_update(self, u, keep_view: bool = True) -> None:
        """Refreshes the widgets a Session callback asked for, shows its messages and redraws."""
        for edit_, text in ((self.path_edit, u.path_text), (self.bkg_edit, u.bkg_text),
                            (self.filter_edit, u.filter_text), (self.pixelsize_edit, u.pixelsize_text),
                            (self.energy_edit, u.energy_text)):
            if text is not None:
                edit_.setText(text)
        if u.nav_enabled is not None:
            self.left_btn.setEnabled(u.nav_enabled)
            self.right_btn.setEnabled(u.nav_enabled)
        if u.image_index is not None:
            self.image_menu.setCurrentIndex(u.image_index)
        if u.deflicker_checked is not None:
            self.deflicker_check.blockSignals(True)
            self.deflicker_check.setChecked(u.deflicker_checked)
            self.deflicker_check.blockSignals(False)
        self._flush_messages()
        self.redraw(keep_view=keep_view)

    def on_zscale(self, menu_index: int) -> None:
        self.session.choose_zscale(menu_index)
        self.redraw(keep_view=True)

    def on_log(self, checked: bool) -> None:
        self.apply_update(self.session.set_log(checked))

    def on_deflicker(self, checked: bool) -> None:
        self.apply_update(self.session.set_deflicker(checked))

    def on_bkg(self) -> None:
        self.apply_update(self.session.set_bkg(self.bkg_edit.text()))

    def on_auto_bkg(self) -> None:
        self.apply_update(self.session.auto_bkg())

    def on_filter(self) -> None:
        self.apply_update(self.session.set_filter(self.filter_edit.text()))

    def on_pixelsize(self) -> None:
        self.apply_update(self.session.set_pixelsize(self.pixelsize_edit.text(), self.energy_edit.text()), keep_view=False)

    def on_energy(self) -> None:
        self.apply_update(self.session.set_energy(self.energy_edit.text()))

    def on_diam(self, index: int) -> None:
        self.apply_update(self.session.choose_diam(index))
        has = self.session.has_image() and self.session.has_image(0) and self.session.limits() is not None
        self._refresh_size_angle_button(bool(has))

    def on_norm(self, index: int) -> None:
        self.apply_update(self.session.choose_norm(index), keep_view=self.session.current == 0)

    def on_image_menu(self, index: int) -> None:
        self.apply_update(self.session.switch_image(index), keep_view=False)


_OPEN_WINDOWS: list = []          # keeps the window alive when main() returns inside an IPython console (Spyder)


def _ipython():
    try:
        from IPython import get_ipython
        return get_ipython()
    except Exception:             # IPython not installed
        return None


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    win = MainWindow(path=argv[1] if len(argv) > 1 else None)
    try:                                   # movegui(mainf,'center')
        geo = win.screen().availableGeometry()
        win.move(geo.center() - win.rect().center())
    except Exception:
        pass
    win.show()
    _OPEN_WINDOWS.append(win)
    ip = _ipython()
    if ip is not None:            # Spyder / IPython: the console runs the Qt event loop, do not block it
        try:
            ip.run_line_magic("gui", "qt")
        except Exception:
            pass
        return 0
    return app.exec() if not hasattr(sys, "ps1") else 0
