# pyCBA — Camera Beam Analyzer (Python)

Python GUI to measure the diameter and profile of a laser beam from camera images (ASCII/CSV exported by Spiricon, WinCamD, BeamGage, Pyrocam, Photon…), using the ISO 11146 algorithms.

* Python ≥ 3.10, Windows / Linux / macOS
* User interface: PyQt5 (Anaconda default) or PySide6, via `pyqtgraph`
* No installation of pyCBA required: run the script from the unzipped folder

## 1. Requirements

With **Anaconda** (Spyder included), everything is already there except possibly `pyqtgraph`:

```
conda install -c conda-forge pyqtgraph
```

Packages used: `numpy`, `scipy`, `Pillow`, `pyqtgraph` (≥ 0.13), `PyQt5` or `PySide6`.
Without Anaconda: `pip install numpy scipy pillow pyqtgraph PyQt5`.

## 2. Running the program

Unzip the `pyCBA` folder wherever you like (e.g. `C:\Users\you\pyCBA`), then choose one of:

| Method | How |
|---|---|
| **Double-click** (Windows) | `run_cba.bat` — finds Anaconda, activates it and starts the program. You can also drag an image onto the `.bat` file. |
| **Spyder** | open `run_cba.py` and "Run file" (F5). The window stays open and the Spyder console remains usable. |
| **Terminal** | `python run_cba.py [image_file]` |

If `run_cba.bat` cannot find Anaconda, open it in a text editor and set the `ACT` variable
(path to `Scripts\activate.bat`).

## 3. Usage (summary)

1. **Open**: choose an image; the other files in the same folder with the same extension can be browsed with the `<--` / `-->` buttons.
   The format (separator, decimal mark, header, empty cells from an ROI export) is detected automatically.
2. **Background / filter / energy / de-flicker / Z scale**: same settings as in the MATLAB version, plus an **Auto** button
   next to the Bkg field (median of a frame covering 5 % along the image edges, written into the field);
   the zoom is kept on a simple refresh and reset when a new image is loaded.
3. **Aperture (ROI)**: rectangle, square, ellipse, circle or polygon, with handles; automatic aperture;
   **Energy / Area / Power**; **ISO Beam size** (ISO diameters along the horizontal and vertical axes) and **ISO Beam size angle**
   (ISO 11146 major and minor diameters along the beam's principal axes, angle of the major axis — positive counter-clockwise
   on screen — and ellipticity; always the 4σ definition, and the tilted ellipse is drawn).
4. **Crosshair** (`+`, `X`, `~`, `O`): defines the profile lines; **Profiles** opens the profile window
   (data + fits) and saves the profile files next to the image.
5. **Pixel killer**: toggle button; clicking a pixel excludes it from the calculation.
6. **Help**: in-program reminder of the functions.

## 4. Converters (outside the GUI)

`examples/convert_files.py` (to be run in Spyder, parameters at the top of the file) converts to text readable by pyCBA:

* Spiricon **Pyrocam III `.RAW`**, Photon **`.PRW`**, and **screenshots** (TIFF/PNG) from the camera software.

The functions are in `pycba.io.converters`. Default choices: the RAW "Color Zoom" factor is read from the file header
(4 otherwise), PRW in raw counts (scaling to 0–255 remains an option), the crosshair in screenshots is detected by its
out-of-palette colours (see `matlab/CONVERTERS_FIXES.txt`).

## 5. Folder contents

```
run_cba.bat / run_cba.py      launchers
src/pycba/io/                 file reading (ASCII, palette, converters)
src/pycba/core/               computations (ISO metrics, profiles, ROI, measurements) — no GUI
src/pycba/gui/                Qt/pyqtgraph interface (callbacks only handle UI logic)
src/pycba/resources/          Spiricon_128bit_RIFF.pal palette
SampleBeams/                  your sample images (the Open button starts there by default)
examples/                     convert_files.py, check_beam_peak.py
tests/                        automated tests + sample files (tests/data)
matlab/                       fixes to carry over into the MATLAB code (see §7)
docs/FUNCTION_MAP.md          MATLAB function → Python function mapping
```