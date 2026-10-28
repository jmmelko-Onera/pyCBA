"""Starts pyCBA. Spyder: "Run file" (F5); no installation needed. Needs PyQt5 or PySide6 and pyqtgraph (all in Anaconda except PySide6).

Optional command-line argument: path of an image file to open at start.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent / "src"))

from pycba.gui.main_window import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
