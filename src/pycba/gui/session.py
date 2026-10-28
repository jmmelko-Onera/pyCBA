"""State of the application (the globals ``IMAGES``, ``FILTERS``, ... of the MATLAB program) without any Qt."""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core import (estimate_background, is_roi_within, DisplaySpec, Filters, ImageData, compute_flicker, display_scale, load_image_file, matlab_round,
                    refresh_zscale, reinit_zscale, select_zscale, update_image)
from ..core import measure, profile_files, roi as roi_geom
from ..core.profiles import (Profile, ProfileAnalysis, ProfilePair, analyze_profile_pair, extract_diagonal_profiles,
                             extract_hv_profiles, extract_l1l2_profiles)
from ..core.beam_metrics import compute_centroid
from ..core.zscale import LoadResult
from ..io import try_read_palette
from .numparse import str2num

ZSCALES = [1, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
SAMPLE_DIR = Path(__file__).resolve().parents[3] / "SampleBeams"
PALETTE_FILE = Path(__file__).resolve().parent.parent / "resources" / "Spiricon_128bit_RIFF.pal"
IMAGE_NAMES = ("A", "B", "C")
DIAM_METHODS = ["4sigma", "1/e2", "Knife-edge", "86% diaphram"]
ROI_TYPES = ["Choose aperture...", "Rectangle", "Square", "Ellipse", "Circle", "Polygon"]


def natural_key(name: str):
    """Sort key putting ``img2`` before ``img10`` (case-insensitive; ties broken by the plain name)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)], name


def list_files(path) -> tuple[Path | None, list[str]]:
    """``listFiles(path)`` -> (folder, file names).

    * a folder: all its files (sub-folders are removed);
    * an existing file: the files of its folder with the SAME extension (case-insensitive); a file without
      extension gives the files of its folder that have no extension either;
    * anything else (empty text, path that does not exist): nothing.

    Natural order (``img2`` before ``img10``). MATLAB's ``listFiles`` listed Windows' ``dir`` order, took an
    empty text for "every file of the current folder", read a non-existent text as an extension filter on
    the current folder, and a file without extension as "no filter": all four changed on purpose.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("Invalid input path.")
    text = os.fspath(path)
    if text.strip() and os.path.isdir(text):
        folder, ext = Path(text), None
    elif text.strip() and os.path.isfile(text):
        folder = Path(text).parent if str(Path(text).parent) != "" else Path.cwd()
        ext = Path(text).suffix.lower()
    else:
        return None, []
    names = [p.name for p in folder.iterdir()
             if p.is_file() and (ext is None or p.suffix.lower() == ext)]
    return folder, sorted(names, key=natural_key)


@dataclass
class FileList:
    """``IMAGES{n}.filesDir / filesNames / curFileIndex``: every image (A, B, C) has its own."""
    folder: Path | None = None
    names: list = field(default_factory=list)
    index: int = 0                                    # 0-based

    def path_at(self, i: int) -> Path:
        return self.folder / self.names[i]


@dataclass
class PathPlan:
    """What ``path_Callback`` / ``LeftButton`` / ``RightButton`` decided."""
    nav_enabled: bool = False                         # left / right buttons
    path_text: str | None = None                      # new content of the path box (None: unchanged)
    load: tuple | None = None                         # (path, image index, cur_file, num_files)


@dataclass
class Update:
    """Widgets to refresh after a callback (None = leave as is). The image is always redrawn."""
    path_text: str | None = None
    bkg_text: str | None = None
    filter_text: str | None = None
    pixelsize_text: str | None = None
    energy_text: str | None = None
    nav_enabled: bool | None = None                   # left / right buttons
    image_index: int | None = None                    # selection of the image menu
    deflicker_checked: bool | None = None


def _num_text(x) -> str:
    """``num2str`` of a scalar (an array cannot come from the GUI)."""
    return f"{float(x):.10g}" if np.ndim(x) == 0 else "0"


@dataclass
class Session:
    images: list[ImageData] = field(default_factory=lambda: [ImageData(), ImageData(), ImageData()])
    filters: Filters = field(default_factory=lambda: Filters(filtorder=1))
    method: int = 1                                   # normalisation: 1 = none ... 4 = fringes (menu value)
    diam_method: int = 0                              # index in DIAM_METHODS
    zscales: list = field(default_factory=lambda: list(ZSCALES))
    colormap: np.ndarray | None = None                # (n, 3) in [0, 1] or None
    deflicker: list = field(default_factory=lambda: [False, False, False])
    current: int = 0                                  # image shown (0 = A)
    lists: list = field(default_factory=lambda: [FileList(), FileList(), FileList()])
    messages: list = field(default_factory=list)
    roi_mask: np.ndarray | None = None                # explicit aperture mask (tests); normally computed from roi_kind / roi_pos
    roi_kind: str = ""                                # '' = no aperture, else one of roi.ROI_KINDS
    roi_pos=None                                      # [x y w h] (rectangle, square, ellipse, circle) or N x 2 vertices
    crosshair: roi_geom.Crosshair = field(default_factory=lambda: roi_geom.Crosshair("O"))   # none at start
    geometry_changed: bool = False                    # set when Session itself moved the aperture / crosshair (GUI re-syncs)

    def say(self, *msgs: str) -> None:
        self.messages.extend(msgs)

    def load_palette(self, path=PALETTE_FILE) -> None:
        self.colormap = try_read_palette(path, 128)
        if self.colormap is not None:
            self.say("custom colormap loaded.")

    @property
    def image(self) -> ImageData:
        return self.images[self.current]

    def has_image(self, index: int | None = None) -> bool:
        im = self.images[self.current if index is None else index]
        return im.raw is not None and im.raw.size > 0

    def open_file(self, path, index: int | None = None, cur_file: int = 1, num_files: int = 1) -> LoadResult:
        index = self.current if index is None else index
        res = load_image_file(self.images, index, path, self.filters, self.method, self.zscales,
                              cur_file, num_files, deflicker=self.deflicker[index])
        self.say(*res.messages)
        return res

    def enter_path(self, text: str, index: int | None = None) -> PathPlan:
        """``path_Callback``: lists the files around the path of image ``index``, then (re)opens it."""
        index = self.current if index is None else index
        folder, names = list_files(text)
        fl = self.lists[index]
        fl.folder, fl.names = folder, names
        if not names:
            self.say("No file found.")
            return PathPlan(False)
        self.say(f"{len(names)} files found.")
        plan = PathPlan(nav_enabled=len(names) > 1)
        path = Path(text)
        if not path.is_file():                        # the user gave a folder name: first file of the folder
            path = folder / names[0]
            plan.path_text = str(path)
        # MATLAB's find() was empty for a name missing from the list; the first file is used here
        fl.index = names.index(path.name) if path.name in names else 0
        plan.load = (path, index, fl.index + 1, len(names))
        return plan

    def step_file(self, step: int, index: int | None = None) -> PathPlan | None:
        """``LeftButton_Callback`` (-1) / ``RightButton_Callback`` (+1): previous / next file of image
        ``index``, cycling at both ends."""
        index = self.current if index is None else index
        fl = self.lists[index]
        if len(fl.names) <= 1:
            return None
        fl.index = (fl.index + step) % len(fl.names)
        path = fl.path_at(fl.index)
        return PathPlan(True, str(path), (path, index, fl.index + 1, len(fl.names)))

    def browse_start_dir(self) -> str:
        """Folder of the first image that has a file (``BrowseButton_Callback``); '' if none exists."""
        for im in self.images:
            if im.filepath:
                folder = Path(im.filepath).parent
                return str(folder) if folder.is_dir() else ""
        return str(SAMPLE_DIR) if SAMPLE_DIR.is_dir() else ""        # <project>/SampleBeams

    # ------------------------------------------------------------------ updating the images
    def update_all(self) -> None:
        """``for i=length(IMAGES):-1:1 update_image(i)``: references first, the normalised main image last."""
        for i in (2, 1, 0):
            self.say(*update_image(self.images, i, self.method, self.filters))

    def update_chain(self, index: int) -> None:
        """``update_image(index)``, then the normalised main image if ``index`` is a reference it depends on
        (MATLAB left it stale until the next update)."""
        self.say(*update_image(self.images, index, self.method, self.filters))
        if index != 0 and self.method != 1:
            self.say(*update_image(self.images, 0, self.method, self.filters))

    def toggle_pixel(self, row: int, col: int) -> bool:
        """``pixelKiller``: kills (deadmask 0) or restores (1) the pixel (row, col), 1-based as in the pixel info.
        Returns False, without any change, when the pixel is outside the image (MATLAB raised an error)."""
        im = self.image
        if not self.has_image() or im.raw is None:
            return False
        h, w = im.raw.shape
        if not (1 <= row <= h and 1 <= col <= w):
            return False
        if not isinstance(im.deadmask, np.ndarray) or im.deadmask.shape != im.raw.shape:
            im.deadmask = np.ones(im.raw.shape)
        killed = im.deadmask[row - 1, col - 1] == 1
        im.deadmask[row - 1, col - 1] = 0.0 if killed else 1.0
        self.say(f"Pix ({row},{col}) {'killed' if killed else 'restored'}.")
        self.update_chain(self.current)
        return True

    # ------------------------------------------------------------------ callbacks of the settings fields
    def set_bkg(self, text: str) -> Update:
        """``bkg_Callback``. MATLAB stored ``str2num(text)``, an empty array for invalid text, which made the
        next subtraction fail; 0 is used instead and the field is reset."""
        vals = str2num(text)
        valid = len(vals) == 1
        self.image.bkg = vals[0] if valid else 0.0
        self.update_chain(self.current)
        return Update(bkg_text=None if valid else "0")

    def auto_bkg(self) -> Update:
        """Button 'Auto' next to the background field (no MATLAB counterpart): single value = median of a 5 % frame
        along the image edges, estimated on what remains to subtract (raw image minus flicker, dead pixels ignored)
        and written in the Bkg field. The aperture plays no part in it."""
        n = self.current
        im = self.images[n]
        if not self.has_image(n):
            self.say("/!\\No image.")
            return Update()
        data = np.where(np.asarray(im.deadmask) == 0, np.nan, im.raw - im.flicker)
        try:
            est = estimate_background(data)
        except ValueError as exc:
            self.say(f"/!\\{exc}")
            return Update()
        text = f"{est.value:.6g}"
        im.bkg = float(text)                             # the value applied is the value shown
        self.update_chain(n)
        self.say(f"Bkg = {text}: median of the image border" + (" after flicker removal." if self.deflicker[n] else "."))
        if est.high and not est.non_uniform:
            peak = float(np.nanmax(data))
            self.say(f"/!\\Background is {100 * est.value / peak:.0f} % of the peak: beam wings on the image "
                     "border? Check the value.")
        if est.non_uniform:
            self.say("/!\\Background not uniform along the image border (beam reaching the border, damaged "
                     "area, gradient?): check the value.")
        return Update(bkg_text=text)

    def set_filter(self, text: str) -> Update:
        """``filter_Callback``: integer in [1, 100] (empty or invalid text: 1), then every image is recomputed."""
        vals = str2num(text)
        order = vals[0] if vals else 1.0
        order = min(max(order, 1.0), 100.0)
        self.filters.filtorder = int(matlab_round(order))
        self.update_all()
        return Update(filter_text=str(self.filters.filtorder))

    def set_pixelsize(self, text: str, energy_text: str) -> Update:
        """``pixelSize_Callback``: new pixel size (empty, invalid or 0: 1; sign dropped). If an energy is set
        the energy scaling keeps the total energy constant, otherwise it is reset."""
        f = self.filters
        f.oldpixelsize = f.pixelsize
        vals = str2num(text)
        ps = abs(vals[0]) if vals else 1.0
        ps = ps if ps != 0 else 1.0
        f.pixelsize = ps
        self.rescale_geometry(ps / f.oldpixelsize)           # aperture and crosshair keep their place on the image
        user_energy = str2num(energy_text)
        if not user_energy or user_energy[0] == 0:
            f.energy = 1.0
        else:
            f.energy = f.energy / (ps / f.oldpixelsize) ** 2
            self.image.znormalize = True
        # MATLAB recomputed the current image only, leaving the others with the old energy scaling
        self.update_all()
        return Update(pixelsize_text=f"{ps:g}")

    def set_energy(self, text: str) -> Update:
        """``energy_Callback``: scales the images so that the energy in the aperture (the whole image if none)
        equals the value typed. An image with no signal (zero energy) cannot be scaled: ignored with a message
        (MATLAB divided by zero and ruined every image)."""
        f = self.filters
        vals = str2num(text)
        if not vals:
            f.energy = 1.0
            self.update_all()
            return Update()
        if vals[0] == 0:
            f.energy = 1.0
            self.update_all()
            return Update(energy_text="")
        user = abs(vals[0])
        shown = f"{user:.3g}"
        filt = self.image.filt
        if filt is None:
            self.say("/!\\No image.")
            return Update(energy_text=shown)
        mask = self.current_mask()
        if mask is None:
            self.say("Aperture undefined: taking whole img.")
            cropped = filt
        else:
            cropped = filt * mask
        total = float(np.nansum(cropped)) * f.pixelsize ** 2
        if total == 0 or not np.isfinite(total):
            self.say("/!\\No energy in the aperture: ignoring.")
            return Update(energy_text=shown)
        f.energy = f.energy * user / total
        self.image.znormalize = True
        self.update_all()
        return Update(energy_text=shown)

    def set_deflicker(self, checked: bool) -> Update:
        """``hDeflickerCheck_Callback`` (the flicker is estimated without aperture, as in MATLAB)."""
        n = self.current
        self.deflicker[n] = bool(checked)
        im = self.images[n]
        if self.has_image(n):
            im.flicker = compute_flicker(im.raw) if checked else np.zeros(im.raw.shape)
            self.update_chain(n)
        return Update()

    def set_log(self, checked: bool) -> Update:
        """``hLogCheck_Callback``."""
        self.filters.log = bool(checked)
        return Update()

    def choose_norm(self, menu_index: int) -> Update:
        """``NormMenu_Callback`` (menu_index 0..3 = method 1..4).

        Changes: choosing "(none)" is always accepted (MATLAB complained about missing references even then);
        when it falls back to image A the other widgets follow (MATLAB only switched the menu).
        """
        self.method = menu_index + 1
        n = self.current
        a, b, c = self.images[0], self.images[1], self.images[2]
        if self.method != 1 and b.filt is None and c.filt is None:
            self.say("/!\\No reference image(s): ignoring.")
            self.say(*update_image(self.images, 0, self.method, self.filters))
            return self.switch_image(0)
        if a.filt is None:
            if self.method != 1:
                self.say("/!\\No main image: ignoring.")
            return Update()
        if self.method in (3, 4):
            a.znormalize = True
            self.say(*update_image(self.images, 0, self.method, self.filters))
        else:
            a.znormalize = False
            self.say(*update_image(self.images, 0, self.method, self.filters))
            if n == 0:
                reinit_zscale(a, self.zscales, "filt")
        return Update()

    def choose_diam(self, menu_index: int) -> Update:
        """``DiamMenu_Callback`` does nothing: the method is read when a size is computed."""
        self.diam_method = menu_index
        return Update()

    def switch_image(self, index: int) -> Update:
        """``ImageNumMenu_Callback``: shows image ``index`` with its own path, background and flicker option.
        The left/right buttons follow the file list of that image (MATLAB kept the state of the previous one)."""
        self.current = index
        im = self.images[index]
        return Update(path_text=im.filepath, bkg_text=_num_text(im.bkg),
                      nav_enabled=len(self.lists[index].names) > 1, image_index=index,
                      deflicker_checked=self.deflicker[index])

    def zscale_menu_index(self) -> int:
        return refresh_zscale(self.image, self.zscales)

    def choose_zscale(self, menu_index: int) -> None:
        select_zscale(self.image, menu_index, self.zscales)

    def display(self, index: int | None = None) -> DisplaySpec:
        index = self.current if index is None else index
        im = self.images[index]
        return display_scale(im.filt, index + 1, self.method, im.znormalize, im.dimz, self.filters.log,
                             0 if self.colormap is None else len(self.colormap), self.filters.pixelsize)

    # ------------------------------------------------------------------ aperture (ROI) and crosshair
    def limits(self):
        """Axes limits ((xmin, xmax), (ymin, ymax)) of the shown image, None without image."""
        filt = self.image.filt
        if filt is None or np.size(filt) == 0:
            return None
        return roi_geom.image_limits(filt.shape, self.filters.pixelsize)

    def current_mask(self):
        """Mask of the aperture on the shown image (None = no aperture): ``createMask``."""
        if self.roi_mask is not None:
            return self.roi_mask
        filt = self.image.filt
        if not self.roi_kind or self.roi_pos is None or filt is None or np.size(filt) == 0:
            return None
        return roi_geom.roi_mask(filt.shape, self.roi_kind, self.roi_pos, self.filters.pixelsize)

    def set_roi(self, kind: str, pos) -> None:
        self.roi_kind = kind
        self.roi_pos = None if pos is None else np.array(pos, dtype=float)

    def clear_roi(self) -> None:
        self.roi_kind, self.roi_pos = "", None

    def default_roi_pos(self, kind: str):
        """Default aperture of ``createDefaultROI`` (squares and circles take the smaller side)."""
        x, y, w, h = roi_geom.default_roi(self.image.filt, self.filters.pixelsize)
        if kind in (roi_geom.SQUARE, roi_geom.CIRCLE):
            side = min(w, h)
            x, y, w, h = x + w / 2 - side / 2, y + h / 2 - side / 2, side, side
        if kind == roi_geom.POLYGON:
            return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        return np.array([x, y, w, h])

    def choose_roi(self, menu_index: int) -> None:
        """``ROImenu_Callback``: deletes the aperture, creates the chosen one (index 0 = none)."""
        self.clear_roi()
        if menu_index > 0 and self.limits() is not None:
            kind = ROI_TYPES[menu_index].lower()
            self.set_roi(kind, self.default_roi_pos(kind))
        self.geometry_changed = True

    def ensure_roi(self, message: str) -> None:
        """Creates the default (ellipse) aperture when there is none (``createDefaultROI('')``)."""
        if not self.roi_kind:
            self.say(message)
            self.set_roi(roi_geom.ELLIPSE, self.default_roi_pos(roi_geom.ELLIPSE))
            self.geometry_changed = True

    def roi_menu_index(self) -> int:
        return 0 if not self.roi_kind else [t.lower() for t in ROI_TYPES].index(self.roi_kind)

    def roi_text(self) -> str:
        """``ROIPosition_Callback``: the text above the image."""
        if self.roi_pos is None:
            return ""
        pos = np.asarray(self.roi_pos, dtype=float)
        if pos.ndim == 2:
            x0, y0 = pos[:, 0].min(), pos[:, 1].min()
            dx, dy = pos[:, 0].max() - x0, pos[:, 1].max() - y0
        else:
            x0, y0, dx, dy = pos.ravel()[:4]
        return f"ROI: X={x0:.4g}, Y={y0:.4g}, \u0394X={dx:.4g} ; \u0394Y={dy:.4g}"

    def set_crosshair(self, kind: str) -> None:
        """Replaces the crosshair by a new one of type ``kind`` ('+', 'X', '~', 'O') through the middle of the image."""
        lim = self.limits()
        self.crosshair = roi_geom.Crosshair.default(kind, *lim) if lim else roi_geom.Crosshair(kind)
        self.crosshair.kind = kind
        self.geometry_changed = True

    def next_crosshair_kind(self) -> str:
        """What the crosshair button does when pressed (and what it shows): + -> X -> ~ -> O (delete) -> + ..."""
        kinds = ["+", "X", "~", "O"]
        return kinds[(kinds.index(self.crosshair.kind) + 1) % 4]

    def rescale_geometry(self, factor: float) -> None:
        """Pixel size changed: aperture and crosshair keep their place on the image (positions are in length units)."""
        if self.roi_pos is not None:
            self.roi_pos = np.asarray(self.roi_pos, dtype=float) * factor
        self.crosshair = self.crosshair.scaled(factor)
        self.geometry_changed = True

    def check_geometry(self) -> None:
        """After an image change: an aperture / crosshair that no longer fits in the axes is deleted / recentred."""
        lim = self.limits()
        if lim is None:
            return
        if self.roi_kind and not is_roi_within(lim[0], lim[1], self.roi_pos):
            self.say("/!\\Aperture outside the image: default aperture created.")      # as MATLAB's plot_image
            self.set_roi(self.roi_kind, self.default_roi_pos(self.roi_kind))
            self.geometry_changed = True
        if self.crosshair.kind != "O" and not self.crosshair.within(*lim):
            self.say("/!\\Crosshair outside the image: recentred.")
            self.set_crosshair(self.crosshair.kind)

    # ------------------------------------------------------------------ measurements
    def _filt(self):
        filt = self.image.filt
        return None if filt is None or np.size(filt) == 0 else filt

    def measure_energy(self) -> None:
        filt = self._filt()
        if filt is None:
            return self.say("/!\\No image.")
        mask = self.current_mask()
        if mask is None:
            self.say("Aperture undefined: taking whole img.")
        self.say(f"energy={measure.aperture_energy(filt, mask, self.filters.pixelsize):.8g}")

    def measure_area(self) -> None:
        filt = self._filt()
        if filt is None:
            return self.say("/!\\No image.")
        mask = self.current_mask()
        if mask is None:
            self.say("Aperture undefined: taking whole img.")
        self.say(f"area={measure.aperture_area(filt.shape, mask, self.filters.pixelsize):.4g}")

    def measure_power(self) -> None:
        filt = self._filt()
        if filt is None:
            return self.say("/!\\No image.")
        mask = self.current_mask()
        if mask is None:
            self.say("Aperture undefined: taking whole img.")
        ps = self.filters.pixelsize
        if measure.aperture_area(filt.shape, mask, ps) == 0:
            return self.say("/!\\Empty aperture.")
        self.say(f"Ave. power={measure.average_power(filt, mask, ps):.4g}")

    @property
    def diam_number(self) -> int:
        return self.diam_method + 1

    def measure_size_xy(self):
        """``SizeXYButton_Callback``. Returns the SizeResult (the window draws the centroid and the ellipse)."""
        filt = self._filt()
        if filt is None:
            return self.say("/!\\No image.")
        self.ensure_roi("/!\\ No aperture defined!")
        mask = self.current_mask()
        res = measure.size_xy(filt, mask, self.diam_number, self.filters.pixelsize,
                              measure.roi_center(self.roi_kind, self.roi_pos))
        self.say(*res.messages())
        return res

    def measure_size_angle(self):
        """ISO beam size angle (replaces ``Size45Button_Callback``): principal axes, angle and ellipticity of the beam
        under the aperture. Returns the AngleResult (the window draws the centroid and the tilted ellipse)."""
        filt = self._filt()
        if filt is None:
            return self.say("/!\\No image.")
        self.ensure_roi("/!\\ No aperture defined!")
        if self.diam_method != 0:
            self.say("/!\\Principal axes always use the 4sigma method, whatever the diameter menu says.")
        res = measure.size_angle(filt, self.current_mask(), self.filters.pixelsize,
                                 measure.roi_center(self.roi_kind, self.roi_pos))
        self.say(*res.messages())
        return res

    def auto_aperture(self) -> None:
        """``AutoApertureButton_Callback``."""
        filt = self._filt()
        if filt is None:
            return self.say("/!\\No image.")
        if self.image.bkg == 0 and not self.deflicker[self.current]:
            self.say("/!\\ Auto-aperture without bkg substract. likely to fail")
        self.ensure_roi("/!\\No aperture defined: creating default.")
        if self.roi_kind == roi_geom.POLYGON:
            return self.say("/!\\Auto aperture is not available for a polygon.")
        pos, ok, n = roi_geom.auto_aperture(filt, self.roi_kind, self.roi_pos, self.diam_number,
                                            self.filters.pixelsize)
        self.set_roi(self.roi_kind, pos)
        self.geometry_changed = True
        if not ok:
            self.say(f"/!\\Could not converge after {n} iterations")
        ps = self.filters.pixelsize
        if min(pos[2], pos[3]) <= roi_geom.MIN_AUTO_APERTURE * ps * (1 + 1e-9):
            self.say(f"/!\\Auto-aperture stopped at its minimum size ({roi_geom.MIN_AUTO_APERTURE} pixels): "
                     "check the background subtraction.")

    # ------------------------------------------------------------------ profiles
    def run_profiles(self):
        """The three ``Profile*Button_Callback`` according to the crosshair type. Returns a ProfileRun or None."""
        filt = self._filt()
        if filt is None:
            return self.say("/!\\No image.")
        ps = self.filters.pixelsize
        mask = self.current_mask()
        center = self.crosshair.center()
        if center is None:                                  # no crosshair: the centroid (GetProfilesCenter)
            self.say("No crosshair defined: using centroid.")
            with np.errstate(divide="ignore", invalid="ignore"):
                center = compute_centroid(filt if mask is None else filt * mask, ps)
            if not all(np.isfinite(center)):
                return self.say("/!\\Could not find profiles center")
        xpos, ypos = center
        kind = {"+": "hv", "O": "hv", "X": "45", "~": "l1l2"}[self.crosshair.kind]
        try:
            if kind == "hv":
                pair, names = extract_hv_profiles(filt, xpos, ypos, ps, mask), ("Hline", "Vline")
            elif kind == "45":
                pair, names = extract_diagonal_profiles(filt, xpos, ypos, ps, mask), ("-", "+")
            else:
                l1, l2 = self.crosshair.l1l2_lines(*self.limits())
                pair, names = extract_l1l2_profiles(filt, l1, l2, mask, ps), ("L1", "L2")
        except IndexError:
            return self.say("/!\\Crosshair outside the image.")
        line, col = int(matlab_round(ypos / ps)), int(matlab_round(xpos / ps))
        pair.info.setdefault("linenum", line)
        pair.info.setdefault("colnum", col)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analysis = analyze_profile_pair(pair, *names)
        for w in caught:
            self.say(f"/!\\{w.message}")
        self.say(*analysis.messages)
        cuts = f"Cuts at Y={ypos:.4g} (Hline {line}) and X={xpos:.4g} (Vcol {col})"
        angle = pair.info.get("angle")
        title = cuts if kind == "hv" else "45\u00b0 " + cuts if kind == "45" else f"{angle:.1f}\u00b0 " + cuts
        paths = profile_files.profile_paths(kind, self.image.filepath, line, col, angle)
        for path, prof in zip(paths.values(), (pair.first, pair.second)):
            self.say(profile_files.save_profile(path, prof.vector, prof.values))
        return ProfileRun(kind, pair, analysis, names, title)


@dataclass
class ProfileRun:
    kind: str                     # 'hv', '45' or 'l1l2'
    pair: ProfilePair
    analysis: ProfileAnalysis
    names: tuple
    title: str
