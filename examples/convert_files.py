"""Converts camera binaries / screenshots to text images readable by pyCBA (Spyder: "Run file").

Edit the USER PARAMETERS, run, pick a file. Batch mode converts every file with the same extension in the folder.
Equivalent of the MATLAB MAIN_SpiriconRAW_to_ASCII / MAIN_PhotonPRW_to_ASCII / MAIN_Import_screenshot_as_ASCII.
"""
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np

from pycba.io import converters as cv, read_palette

# ---------------------------------------------------------------- USER PARAMETERS
KIND = "raw"            # "raw" (Spiricon Pyrocam III), "prw" (Photon) or "screenshot"
BATCH = False           # True: every file with the same extension as the chosen one
PATH = ""               # "" = ask with a dialog
DELIMITER = "\t"
# raw
COLOR_ZOOM = "auto"     # raw value / COLOR_ZOOM. "auto": read in the file header (4 if it does not say); or a number
PX_SCALE = 1            # enlarges each pixel PX_SCALE times (MATLAB pxscale; 1 = off). The pixel size to enter in pyCBA is divided by it
# prw
DIM_Z = None            # None = keep the raw counts (recommended: the display scale of pyCBA adapts the colours);
                        # 255 = MATLAB (each image rescaled to 0..255, information is lost)
DEINTERLACE = False
MEDIAN = 1              # median filter size, 1 = off
# screenshot
ZOOM = 1                # screen pixels / camera pixels
CLEAN_CROSSHAIR = False  # True: remove the camera software crosshair (lines made of colours outside the palette)
PALETTE = HERE.parent / "src" / "pycba" / "resources" / "Spiricon_128bit_RIFF.pal"
# -------------------------------------------------------------------------------


def convert(path):
    if KIND == "raw":
        img, fmt = cv.read_pyrocam_raw(path, color_zoom=COLOR_ZOOM), "%.7e"
        if COLOR_ZOOM == "auto":
            zoom = cv.pyrocam_color_zoom(path)
            print(f"{Path(path).name}: colour zoom {zoom:g} (from the header)" if zoom
                  else f"{Path(path).name}: colour zoom {cv.PYROCAM_COLOR_ZOOM:g} (the header does not say)")
    elif KIND == "prw":
        img, fmt = cv.read_photon_prw(path), "%g"
        img = cv.median_filter(img, MEDIAN)
        if DEINTERLACE:
            img = cv.deinterlace(img)
        if DIM_Z:
            img = cv.normalize_z(img, DIM_Z)
    else:
        res = cv.import_screenshot(path, read_palette(PALETTE), zoom=ZOOM,
                                   crosshair=cv.CrosshairClean(enable=CLEAN_CROSSHAIR))
        print("\n".join(res.messages))
        img, fmt = res.values, "%.16e"
    out = cv.write_ascii(cv.default_output_path(path), img, delimiter=DELIMITER, fmt=fmt)
    print(f"{Path(path).name} -> {out.name}  ({img.shape[1]} x {img.shape[0]})")


def main():
    path = PATH
    if not path:
        from tkinter import Tk, filedialog
        root = Tk(); root.withdraw()
        path = filedialog.askopenfilename(title="File to convert")
        root.destroy()
    if not path:
        return
    path = Path(path)
    files = sorted(path.parent.glob("*" + path.suffix)) if BATCH else [path]
    files = [f for f in files if not f.name.endswith(".txt")]
    for f in files:
        try:
            convert(f)
        except Exception as exc:                           # keep going in batch mode
            print(f"/!\\ {f.name}: {exc}")


if __name__ == "__main__":
    main()
