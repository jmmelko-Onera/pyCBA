"""Tests of the GUI layer: geometry and session (no Qt needed), then the window itself (PySide6 + pyqtgraph,
run off-screen; skipped when they are not installed)."""
import os
import sys
import warnings
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest

from pycba.gui import layout
from pycba.core import is_roi_within
from pycba.gui.session import Session, list_files

pytestmark = pytest.mark.filterwarnings("ignore::pycba.io.AsciiFormatWarning")


# ----------------------------------------------------------------------- layout
def test_layout_matches_the_matlab_positions():
    R = layout.RECTS
    assert R["path"] == (5, 580, 720, 20) and R["browse"] == (725, 580, 20, 20)
    assert R["left_file"] == (750, 580, 20, 20) and R["right_file"] == (775, 580, 20, 20)
    assert R["zscale"] == (5, 540, 60, 20) and R["bkg"] == (85, 540, 35, 20)
    assert R["filter"] == (175, 540, 35, 20) and R["pixelsize"] == (260, 540, 40, 20)
    assert R["energy"] == (330, 540, 60, 20) and R["pixkill"] == (430, 540, 40, 20)
    assert R["diam"] == (490, 540, 70, 20) and R["norm"] == (580, 540, 80, 20)
    assert R["energy_button"] == (665, 550, 120, 25) and R["roi_menu"] == (665, 400, 120, 25)
    assert R["crosshair"] == (665, 370, 30, 25) and R["profiles"] == (715, 370, 70, 25)
    assert R["auto_aperture"] == (665, 340, 120, 25)
    assert R["msgbox"].x == 665 and R["msgbox"].w == 125 and R["msgbox"].h == pytest.approx(600 / 2.1)
    assert R["msgbox"].y == pytest.approx(550 - 7.5 * 30 - 600 / 2.1)


def test_layout_to_qt_flips_y_and_scales():
    assert layout.to_qt(layout.RECTS["path"]) == (5, 0, 720, 20)
    assert layout.to_qt(layout.RECTS["energy_button"], 1600, 1200) == (1330, 50, 240, 50)


def test_image_area_does_not_cover_any_control():
    R = layout.RECTS
    plot = R["plot"]
    assert plot.x + plot.w <= R["msgbox"].x
    for k, r in R.items():
        if k in ("plot", "pixel_info") or r.x >= plot.x + plot.w:
            continue
        assert r.y >= plot.y + plot.h or r.y + r.h <= plot.y, f"{k} overlaps the image area"


# ---------------------------------------------------------------------- session
def test_session_palette_and_display_spec(tmp_path):
    s = Session()
    s.load_palette()
    assert s.colormap.shape == (128, 3) and s.messages == ["custom colormap loaded."]
    f = tmp_path / "b.txt"
    f.write_text("\n".join("\t".join(f"{(i + 1) * (j + 1) / 3:.3f}" for j in range(9)) for i in range(7)) + "\n")
    res = s.open_file(f)
    assert res.ok
    spec = s.display()
    assert spec.custom_colormap and spec.data.shape == (7, 9) and not spec.empty
    assert spec.extent == (0.5, 9.5, 7.5, 0.5)


def _make_files(tmp_path, names=("img10.txt", "img2.txt", "img1.txt", "other.csv")):
    for n in names:
        (tmp_path / n).write_text("1\t2\t3\n4\t5\t6\n")


def test_list_files_follows_listFiles_with_the_agreed_changes(tmp_path):
    _make_files(tmp_path, ("img10.txt", "img2.txt", "img1.txt", "other.csv", "NOEXT", "OTHER2"))
    (tmp_path / "subdir").mkdir()
    folder, names = list_files(tmp_path / "img2.txt")
    assert folder == tmp_path and names == ["img1.txt", "img2.txt", "img10.txt"]      # same extension, natural order
    assert list_files(tmp_path)[1] == ["img1.txt", "img2.txt", "img10.txt", "NOEXT", "OTHER2", "other.csv"]   # folder: every file
    assert list_files(tmp_path / "NOEXT")[1] == ["NOEXT", "OTHER2"]                  # no extension: only such files


def test_list_files_order_and_extension_ignore_case(tmp_path):
    _make_files(tmp_path, ("b.txt", "A.txt", "C.TXT", "d.csv"))
    assert list_files(tmp_path / "b.txt")[1] == ["A.txt", "b.txt", "C.TXT"]


def test_list_files_empty_or_missing_paths_give_nothing(tmp_path, monkeypatch):
    _make_files(tmp_path, ("a.txt", "b.csv"))
    monkeypatch.chdir(tmp_path)                                  # MATLAB listed this folder for '' and for '.txt'
    for text in ("", "   ", ".txt", str(tmp_path / "missing.txt")):
        assert list_files(text) == (None, [])
    with pytest.raises(TypeError):
        list_files(3)


def test_enter_path_follows_path_callback(tmp_path):
    _make_files(tmp_path)
    s = Session()
    plan = s.enter_path(str(tmp_path / "img2.txt"), 0)
    assert s.messages == ["3 files found."] and plan.nav_enabled and plan.path_text is None
    assert plan.load == (tmp_path / "img2.txt", 0, 2, 3) and s.lists[0].index == 1      # img1, img2, img10
    s.messages.clear()
    plan = s.enter_path(str(tmp_path), 0)                         # a folder: its first file, shown in the path box
    assert plan.path_text == str(tmp_path / "img1.txt") and plan.load[2:] == (1, 4)
    s.messages.clear()
    plan = s.enter_path(str(tmp_path / "nothing.txt"), 0)
    assert s.messages == ["No file found."] and not plan.nav_enabled and plan.load is None


def test_single_file_disables_navigation(tmp_path):
    _make_files(tmp_path, ("only.txt",))
    s = Session()
    assert not s.enter_path(str(tmp_path / "only.txt"), 0).nav_enabled
    assert s.step_file(+1, 0) is None


def test_stepping_cycles_and_every_image_has_its_own_list(tmp_path):
    _make_files(tmp_path)
    sub = tmp_path / "sub"; sub.mkdir()
    _make_files(sub, ("r1.txt", "r2.txt"))
    s = Session()
    s.enter_path(str(tmp_path / "img1.txt"), 0)
    s.enter_path(str(sub / "r1.txt"), 1)
    assert s.step_file(-1, 0).load == (tmp_path / "img10.txt", 0, 3, 3)       # before the first: the last
    assert s.step_file(+1, 0).load == (tmp_path / "img1.txt", 0, 1, 3)        # after the last: the first
    assert s.step_file(+1, 1).load == (sub / "r2.txt", 1, 2, 2)               # image B moves in its own folder
    assert s.lists[0].index == 0 and s.lists[1].index == 1


def test_browse_start_dir_is_the_folder_of_the_first_loaded_image(tmp_path):
    s = Session()
    from pycba.gui.session import SAMPLE_DIR
    assert s.browse_start_dir() == (str(SAMPLE_DIR) if SAMPLE_DIR.is_dir() else "")   # SampleBeams by default
    s.images[1].filepath = str(tmp_path / "x.txt")
    assert s.browse_start_dir() == str(tmp_path)
    s.images[0].filepath = str(tmp_path / "gone" / "y.txt")                  # first non-empty path decides
    assert s.browse_start_dir() == ""


# ------------------------------------------------------------- settings callbacks
def _beam_image(shape=(30, 40), peak=200.0, base=20.0):
    y, x = np.mgrid[1:shape[0] + 1, 1:shape[1] + 1]
    return base + peak * np.exp(-2 * (((x - 20) / 5) ** 2 + ((y - 15) / 4) ** 2))


def _loaded(tmp_path, n_images=1, shape=(30, 40)):
    """Session with the first ``n_images`` slots loaded (A = beam, B / C = other images of the same size)."""
    s = Session()
    raws = [_beam_image(shape), 3.0 + _beam_image(shape, 10, 1), 5.0 + _beam_image(shape, 5, 2)]
    for i in range(n_images):
        f = tmp_path / f"im{i}.txt"
        np.savetxt(f, raws[i], fmt="%.6f", delimiter="\t")
        assert s.open_file(f, i).ok
        raws[i] = np.loadtxt(f, delimiter="\t")
    s.current = 0
    s.messages.clear()
    return s, raws


def test_set_bkg_subtracts_and_resets_invalid_text(tmp_path):
    s, raw = _loaded(tmp_path)
    u = s.set_bkg("2*5")
    assert u.bkg_text is None and s.images[0].bkg == 10 and np.allclose(s.images[0].filt, raw[0] - 10)
    for bad in ("abc", "", "1 2", "nan"):
        u = s.set_bkg(bad)
        assert u.bkg_text == "0" and s.images[0].bkg == 0 and np.allclose(s.images[0].filt, raw[0])


def test_set_bkg_of_a_reference_refreshes_the_normalised_main_image(tmp_path):
    s, raw = _loaded(tmp_path, 2)
    s.choose_norm(1)                                             # A - B
    assert np.allclose(s.images[0].filt, raw[0] - raw[1])
    s.current = 1
    s.set_bkg("5")
    assert np.allclose(s.images[1].filt, raw[1] - 5)
    assert np.allclose(s.images[0].filt, raw[0] - (raw[1] - 5))            # MATLAB left A stale here


def test_set_filter_clamps_rounds_and_updates_every_image(tmp_path):
    from pycba.core import box_filter
    s, raw = _loaded(tmp_path, 2)
    assert s.set_filter("3").filter_text == "3" and s.filters.filtorder == 3
    np.testing.assert_allclose(s.images[0].filt, box_filter(raw[0], 3))
    np.testing.assert_allclose(s.images[1].filt, box_filter(raw[1], 3))         # the reference too
    for text, order in (("0", 1), ("-4", 1), ("500", 100), ("2.5", 3), ("2.4", 2), ("x", 1), ("", 1)):
        assert s.set_filter(text).filter_text == str(order) and s.filters.filtorder == order
    assert np.allclose(s.images[0].filt, raw[0])                                  # order 1 = no smoothing


def test_set_energy_scales_the_whole_image_or_the_aperture(tmp_path):
    s, raw = _loaded(tmp_path, 2)
    u = s.set_energy("123.456")
    assert u.energy_text == "123" and "Aperture undefined: taking whole img." in s.messages
    assert s.images[0].filt.sum() * s.filters.pixelsize ** 2 == pytest.approx(123.456)
    assert s.images[0].znormalize
    np.testing.assert_allclose(s.images[1].filt, raw[1] * s.filters.energy)          # references scale too
    s.messages.clear()
    mask = np.zeros(raw[0].shape); mask[10:20, 10:30] = 1
    s.roi_mask = mask
    s.set_energy("50")
    assert "Aperture undefined: taking whole img." not in s.messages
    assert (s.images[0].filt * mask).sum() == pytest.approx(50)


def test_set_energy_special_values(tmp_path):
    s, raw = _loaded(tmp_path)
    s.set_energy("10")
    assert s.filters.energy != 1
    assert s.set_energy("0").energy_text == "" and s.filters.energy == 1
    s.set_energy("10"); s.set_energy("")
    assert s.filters.energy == 1
    assert s.set_energy("-7").energy_text == "7"                                       # sign dropped
    s.images[0].raw[:] = 0                                                              # no signal at all
    s.set_energy("5")
    s.filters.energy = 1.0
    s.update_all(); s.messages.clear()
    s.set_energy("5")
    assert "/!\\No energy in the aperture: ignoring." in s.messages and s.filters.energy == 1.0
    assert np.isfinite(s.images[0].filt).all()                                          # MATLAB: Inf / NaN everywhere


def test_set_pixelsize_keeps_the_total_energy_and_the_energy_scaling(tmp_path):
    s, raw = _loaded(tmp_path, 2)
    s.set_energy("100")
    u = s.set_pixelsize("2", "100")
    f = s.filters
    assert u.pixelsize_text == "2" and f.pixelsize == 2 and f.oldpixelsize == 1
    assert s.images[0].filt.sum() * f.pixelsize ** 2 == pytest.approx(100)             # total energy unchanged
    np.testing.assert_allclose(s.images[1].filt, raw[1] * f.energy)                    # the reference follows (MATLAB: stale)
    u = s.set_pixelsize("3", "")                                                       # no energy typed: scaling reset
    assert f.energy == 1 and f.oldpixelsize == 2


@pytest.mark.parametrize("text, expected", [("abc", "1"), ("", "1"), ("0", "1"), ("-3", "3"), ("2.5", "2.5"),
                                            ("0.0037", "0.0037"), ("1 2", "1")])
def test_set_pixelsize_text_rules(tmp_path, text, expected):
    s, _ = _loaded(tmp_path)
    assert s.set_pixelsize(text, "").pixelsize_text == expected


def test_set_deflicker_removes_the_row_offsets_and_is_per_image(tmp_path):
    s = Session()
    rng = np.random.default_rng(3)
    raw = rng.normal(100, 0.2, (20, 60)) + rng.uniform(-5, 5, (20, 1))
    f = tmp_path / "flk.txt"
    np.savetxt(f, raw, fmt="%.6f", delimiter="\t")
    s.open_file(f, 0)
    raw = np.loadtxt(f, delimiter="\t")
    s.set_deflicker(True)
    assert s.deflicker == [True, False, False] and np.abs(s.images[0].filt - 0).max() < 8
    spread = lambda a: a.mean(axis=1).std()                                             # noqa: E731
    assert spread(s.images[0].filt) < 0.3 * spread(raw)
    s.set_deflicker(False)
    assert np.allclose(s.images[0].filt, raw) and not s.images[0].flicker.any() and s.deflicker[0] is False


def test_set_deflicker_without_image_only_stores_the_choice():
    s = Session()
    s.current = 1
    s.set_deflicker(True)
    assert s.deflicker == [False, True, False]


def test_choose_norm_follows_NormMenu_callback(tmp_path):
    s, raw = _loaded(tmp_path)
    s.current = 1
    s.images[1].znormalize = False
    u = s.choose_norm(2)                                                                # A/B, but no reference loaded
    assert "/!\\No reference image(s): ignoring." in s.messages and s.current == 0 and u.image_index == 0
    s.messages.clear()
    assert s.choose_norm(0).image_index is None and s.messages == []                  # "(none)": no complaint
    s2, raw2 = _loaded(tmp_path, 2)
    s2.choose_norm(2)                                                                   # A/B
    assert s2.method == 3 and s2.images[0].znormalize
    np.testing.assert_allclose(s2.images[0].filt, raw2[0] / raw2[1])
    s2.choose_norm(1)                                                                   # A-B: scale chosen on the result
    assert not s2.images[0].znormalize and s2.images[0].dimz in s2.zscales
    assert s2.images[0].dimz >= np.nanmax(s2.images[0].filt) or s2.images[0].dimz == max(s2.zscales)
    s3 = Session()
    s3.images[1].filt = np.ones((2, 2))
    s3.choose_norm(2)
    assert "/!\\No main image: ignoring." in s3.messages


def test_switch_image_refreshes_path_background_flicker_and_navigation(tmp_path):
    s, _ = _loaded(tmp_path, 2)
    s.images[1].bkg = 7.5
    s.deflicker[1] = True
    s.lists[1].names = ["a", "b"]
    u = s.switch_image(1)
    assert s.current == 1 and u.image_index == 1 and u.bkg_text == "7.5" and u.deflicker_checked is True
    assert u.path_text == s.images[1].filepath and u.nav_enabled is True
    u = s.switch_image(0)
    assert u.bkg_text == "0" and u.deflicker_checked is False and u.nav_enabled is False


# ----------------------------------------------------------------------- window
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtWidgets
    from pycba.gui import main_window
    main_window.CONFIRM_CLOSE = False            # no modal question in the tests (see test_close_confirmation)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    # no window must survive until interpreter shutdown (pyqtgraph then resizes deleted items and prints errors)
    import gc
    main_window.CONFIRM_CLOSE = False
    for widget in app.topLevelWidgets():
        if isinstance(widget, main_window.MainWindow):     # only ours (Spyder's own windows share the QApplication)
            try:
                widget.close()
                widget.deleteLater()
            except RuntimeError:                           # already deleted
                pass
    app.processEvents()
    app.sendPostedEvents(None, 0)
    gc.collect()


def _beam_file(tmp_path, name="beam.txt", shape=(60, 80)):
    y, x = np.mgrid[1:shape[0] + 1, 1:shape[1] + 1]
    img = 20 + 180 * np.exp(-2 * (((x - 40) / 8) ** 2 + ((y - 30) / 6) ** 2))
    f = tmp_path / name
    np.savetxt(f, img, fmt="%.2f", delimiter="\t")
    return f, img


def test_window_opens_without_image(qapp):
    from pycba.gui.main_window import MainWindow
    w = MainWindow()
    w.show(); qapp.processEvents()
    assert "custom colormap loaded." in w.msgbox.toPlainText() and "No file found." in w.msgbox.toPlainText()
    assert not w.energy_btn.isEnabled() and w.image_menu.isEnabled() and not w.left_btn.isEnabled()
    assert w.zscale_menu.itemText(0) == "Normalize" and w.zscale_menu.count() == 12
    assert [w.diam_menu.itemText(i) for i in range(4)] == ["4sigma", "1/e2", "Knife-edge", "86% diaphram"]
    assert w.norm_menu.itemText(3) == "(A-B-C)/4*sqrt(B*C)"
    w.close()


def test_window_children_follow_the_matlab_geometry(qapp):
    from pycba.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(800, 600); w.show(); qapp.processEvents()
    for key, widget in w.rects.items():
        assert tuple(widget.geometry().getRect()) == layout.to_qt(layout.RECTS[key], w.canvas.width(), w.canvas.height())
    w.resize(1000, 750); qapp.processEvents()
    g = w.energy_btn.geometry()
    assert g.x() == pytest.approx(665 * w.canvas.width() / 800, abs=1)
    w.close()


def test_window_loads_and_displays_an_image(qapp, tmp_path):
    from pycba.gui.main_window import MainWindow
    f, img = _beam_file(tmp_path)
    w = MainWindow(path=str(f))
    w.show(); qapp.processEvents()
    assert w.energy_btn.isEnabled() and w.path_edit.text() == str(f) and w.energy_edit.text() == ""
    assert w.image_item.image.shape == img.shape
    assert w.zscale_menu.currentText() == "256"             # peak 200 -> 2**8 = 256
    low, high = w.colorbar.levels()
    spec = w.session.display()
    assert (low, high) == pytest.approx((spec.vmin, spec.vmax))
    text = w.msgbox.toPlainText()
    assert "Image opened." in text
    w.close()


def test_window_zscale_log_and_image_menu(qapp, tmp_path):
    from pycba.gui.main_window import MainWindow
    f, _ = _beam_file(tmp_path)
    w = MainWindow(path=str(f))
    w.show(); qapp.processEvents()
    w.zscale_menu.setCurrentIndex(3); w.on_zscale(3)          # 512
    assert not w.session.image.znormalize and w.session.image.dimz == 512
    w.zscale_menu.setCurrentIndex(0); w.on_zscale(0)
    assert w.session.image.znormalize
    w.log_check.setChecked(True); qapp.processEvents()
    assert w.session.filters.log and w.session.display().log
    w.image_menu.setCurrentIndex(1); w.on_image_menu(1)       # reference B: empty
    assert w.session.current == 1 and w.image_item.image is None and w.bkg_image_label.text() == "(B)"
    w.close()


def test_window_path_box_lists_files_and_steps_cyclically(qapp, tmp_path):
    from pycba.gui.main_window import MainWindow
    paths = [_beam_file(tmp_path, f"shot{i}.txt")[0] for i in (1, 2, 10)]
    w = MainWindow()
    w.path_edit.setText(str(paths[0])); w.on_path_entered()
    text = w.msgbox.toPlainText()
    assert text.index("3 files found.") < text.index("Image opened (1/3).") and w.right_btn.isEnabled()
    w.step_file(-1)                                          # before the first: the last
    assert w.path_edit.text() == str(paths[2]) and "Image opened (3/3)." in w.msgbox.toPlainText()
    w.step_file(+1)                                          # after the last: the first
    assert w.path_edit.text() == str(paths[0])
    w.path_edit.setText(str(tmp_path)); w.on_path_entered()  # a folder name: first file
    assert w.path_edit.text() == str(paths[0])
    w.close()


def test_window_loads_reference_images_with_their_own_file_list(qapp, tmp_path):
    from pycba.gui.main_window import MainWindow
    main = [_beam_file(tmp_path, f"a{i}.txt")[0] for i in (1, 2)]
    ref_dir = tmp_path / "ref"; ref_dir.mkdir()
    ref = [_beam_file(ref_dir, f"b{i}.txt")[0] for i in (1, 2, 3)]
    w = MainWindow(path=str(main[0]))
    w.image_menu.setCurrentIndex(1); w.on_image_menu(1)
    w.path_edit.setText(str(ref[0])); w.on_path_entered()    # loads into B
    assert w.session.has_image(1) and w.session.images[1].filepath == str(ref[0]) and w.session.current == 1
    w.step_file(+1)
    assert w.session.images[1].filepath == str(ref[1]) and w.session.images[0].filepath == str(main[0])
    assert [len(l.names) for l in w.session.lists[:2]] == [2, 3]
    w.close()


def test_window_missing_file_clears_the_main_image(qapp, tmp_path):
    from pycba.gui.main_window import MainWindow
    f, _ = _beam_file(tmp_path)
    w = MainWindow(path=str(f))
    w.open_path(tmp_path / "gone.txt", 0)
    assert "File not found." in w.msgbox.toPlainText() and not w.energy_btn.isEnabled()
    assert w.image_item.image is None
    w.close()


def test_window_settings_fields_are_wired(qapp, tmp_path):
    from pycba.gui.main_window import MainWindow
    f, img = _beam_file(tmp_path)
    w = MainWindow(path=str(f))
    w.show(); qapp.processEvents()
    raw = w.session.images[0].raw
    w.bkg_edit.setText("10"); w.on_bkg()
    assert np.allclose(w.session.images[0].filt, raw - 10)
    w.bkg_edit.setText("oops"); w.on_bkg()
    assert w.bkg_edit.text() == "0"
    w.filter_edit.setText("500"); w.on_filter()
    assert w.filter_edit.text() == "100" and w.session.filters.filtorder == 100
    w.filter_edit.setText("1"); w.on_filter()
    w.energy_edit.setText("123.456"); w.on_energy()
    assert w.energy_edit.text() == "123" and "Aperture undefined: taking whole img." in w.msgbox.toPlainText()
    assert w.zscale_menu.currentText() == "Normalize"                                  # the scale follows the energy change
    w.pixelsize_edit.setText("-2"); w.on_pixelsize()
    assert w.pixelsize_edit.text() == "2"
    w.deflicker_check.setChecked(True); qapp.processEvents()
    assert w.session.deflicker[0] is True
    w.close()


def test_window_image_menu_and_norm_menu(qapp, tmp_path):
    from pycba.gui.main_window import MainWindow
    fa, _ = _beam_file(tmp_path, "a.txt")
    fb, _ = _beam_file(tmp_path, "b.txt")
    w = MainWindow(path=str(fa))
    w.show(); qapp.processEvents()
    w.norm_menu.setCurrentIndex(2); w.on_norm(2)                                         # A/B without reference
    assert "/!\\No reference image(s): ignoring." in w.msgbox.toPlainText() and w.image_menu.currentIndex() == 0
    w.image_menu.setCurrentIndex(1); w.on_image_menu(1)
    assert w.bkg_image_label.text() == "(B)" and w.path_edit.text() == "" and not w.left_btn.isEnabled()
    w.path_edit.setText(str(fb)); w.on_path_entered()
    w.bkg_edit.setText("4"); w.on_bkg()
    w.image_menu.setCurrentIndex(0); w.on_image_menu(0)
    assert w.path_edit.text() == str(fa) and w.bkg_edit.text() == "0" and w.bkg_image_label.text() == "(A)"
    w.close()


def test_unported_controls_say_so(qapp):
    from pycba.gui.main_window import MainWindow
    w = MainWindow()
    w.stub("Profiles")
    assert "Profiles: not ported yet." in w.msgbox.toPlainText()
    w.close()

# ---------------------------------------------------------------- image toolbar
def test_toolbar_modes_and_home(qapp):
    from pycba.gui.main_window import MainWindow
    w = MainWindow()
    tb = w.toolbar
    assert tb.mode == "none" and not w.plot.vb.state["mouseEnabled"][0]
    tb.buttons["pan"].click()
    assert tb.mode == "pan" and w.plot.vb.state["mouseEnabled"] == [True, True]
    tb.buttons["zoom_in"].click()                      # exclusive modes
    assert tb.mode == "zoom_in" and not tb.buttons["pan"].isChecked()
    tb.buttons["zoom_in"].click()                      # second click releases the tool
    assert tb.mode == "none" and not tb.buttons["zoom_in"].isChecked()
    tb.set_home(0, 80, 0, 60)
    tb.home()
    (a0, a1), _ = w.plot.vb.viewRange()
    tb.zoom_at(40, 30, 0.5)
    (x0, x1), _ = w.plot.vb.viewRange()
    assert abs((x1 - x0) - (a1 - a0) / 2) < 1e-6
    tb.buttons["home"].click()
    (x0, x1), _ = w.plot.vb.viewRange()
    assert abs((x1 - x0) - (a1 - a0)) < 1e-6
    w.close()


def test_toolbar_zoom_keeps_point_fixed(qapp):
    from pycba.gui.main_window import MainWindow
    w = MainWindow()
    tb = w.toolbar
    tb.set_home(0, 100, 0, 100)
    tb.home()
    tb.zoom_at(25, 75, 0.5)
    (x0, x1), (y0, y1) = w.plot.vb.viewRange()
    assert abs((25 - x0) / (x1 - x0) - 0.25) < 1e-6 and abs((75 - y0) / (y1 - y0) - 0.75) < 1e-6
    tb.zoom_at(25, 75, 2.0)
    (x0, x1), _ = w.plot.vb.viewRange()
    assert abs((x1 - x0) - 100) < 1e-6
    w.close()


def test_pixel_killer(tmp_path):
    _make_files(tmp_path)
    s = Session()
    s.open_file(tmp_path / "img1.txt", 0, 1, 1)
    assert s.has_image()
    s.messages.clear()
    before = s.image.filt[0, 1]
    assert s.toggle_pixel(1, 2)                                        # (row, col), 1-based
    assert s.messages == ["Pix (1,2) killed."] and s.image.filt[0, 1] == 0
    s.messages.clear()
    assert s.toggle_pixel(1, 2)                                        # a second shot restores it
    assert s.messages == ["Pix (1,2) restored."] and s.image.filt[0, 1] == before
    s.messages.clear()
    assert not s.toggle_pixel(0, 1) and not s.toggle_pixel(1, 10_000) and s.messages == []


def test_pixkill_click_in_window(qapp, tmp_path):
    from pycba.gui.main_window import MainWindow
    _make_files(tmp_path)
    w = MainWindow()
    w.open_path(tmp_path / "img1.txt")
    w.pixkill_btn.click()
    assert w.pixkill_btn.isChecked() and w.toolbar.mode == "none"
    w.session.toggle_pixel(1, 1); w.redraw(); w._flush_messages()
    assert "Pix (1,1) killed." in w.msgbox.toPlainText()
    w.close()


def test_close_confirmation(qapp, monkeypatch):
    from pyqtgraph.Qt import QtWidgets
    from pycba.gui import main_window
    monkeypatch.setattr(main_window, "CONFIRM_CLOSE", True)
    w = main_window.MainWindow()
    w.show()
    Box = QtWidgets.QMessageBox
    monkeypatch.setattr(Box, "question", staticmethod(lambda *a, **k: Box.No))
    w.close()
    assert w.isVisible()                                   # refused: the window stays
    monkeypatch.setattr(Box, "question", staticmethod(lambda *a, **k: Box.Yes))
    w.close()
    assert not w.isVisible()
    monkeypatch.setattr(main_window, "CONFIRM_CLOSE", False)


# ---------------------------------------------------------------- aperture, crosshair, measurements (Session, no Qt)
def _session_with_beam(tmp_path):
    f, _ = _beam_file(tmp_path)
    s = Session()
    s.open_file(f, 0, 1, 1)
    s.messages.clear()
    return s


def test_choose_roi_creates_an_aperture_around_the_peak(tmp_path):
    s = _session_with_beam(tmp_path)
    s.choose_roi(3)                                                # Ellipse
    assert s.roi_kind == "ellipse" and s.roi_menu_index() == 3 and s.geometry_changed
    x, y, w, h = s.roi_pos
    assert abs(x + w / 2 - 40) < 1 and abs(y + h / 2 - 30) < 1     # centred on the peak (40, 30)
    assert s.current_mask().sum() > 0 and s.roi_text().startswith("ROI: X=")
    s.choose_roi(5)
    assert s.roi_kind == "polygon" and np.asarray(s.roi_pos).shape == (4, 2)
    s.choose_roi(2)                                                # Square: really square
    assert s.roi_pos[2] == s.roi_pos[3]
    s.choose_roi(0)
    assert s.roi_kind == "" and s.current_mask() is None


def test_energy_area_power_messages(tmp_path):
    s = _session_with_beam(tmp_path)
    s.measure_energy(); s.measure_area(); s.measure_power()
    assert s.messages.count("Aperture undefined: taking whole img.") == 3
    assert s.messages[1].startswith("energy=") and s.messages[3] == "area=4800"
    s.messages.clear()
    s.choose_roi(1)
    s.measure_area()
    assert s.messages[0].startswith("area=") and "undefined" not in s.messages[0]
    s.messages.clear()
    s.set_roi("rectangle", [1000, 1000, 5, 5])                     # no pixel inside
    s.measure_power()
    assert s.messages == ["/!\\Empty aperture."]


def test_size_xy_creates_a_default_aperture(tmp_path):
    s = _session_with_beam(tmp_path)
    res = s.measure_size_xy()
    assert s.roi_kind == "ellipse" and "/!\\ No aperture defined!" in s.messages
    assert res.diam_h > 0 and any(m.startswith("DiamH=") for m in s.messages) and any(m.startswith("Peak Flux=") for m in s.messages)


def test_size_angle_creates_a_default_aperture_and_ignores_the_diameter_menu(tmp_path):
    s = _session_with_beam(tmp_path)
    res = s.measure_size_angle()
    assert s.roi_kind == "ellipse" and "/!\\ No aperture defined!" in s.messages
    assert res.principal.d_major > 0 and any(m.startswith("Major=") for m in s.messages)
    assert not any(m.startswith("/!\\Principal axes") for m in s.messages)       # 4sigma selected: no remark
    s.messages.clear()
    s.choose_diam(2)                                                              # knife edge
    s.measure_size_angle()
    assert s.messages[0].startswith("/!\\Principal axes always use the 4sigma method")
    assert s.measure_size_angle() is not None
    empty = Session()
    assert empty.measure_size_angle() is None and empty.messages == ["/!\\No image."]


def test_auto_aperture_at_the_minimum_size_says_so(tmp_path, monkeypatch):
    from pycba.core import roi as roi_geom
    s = _session_with_beam(tmp_path)
    s.ensure_roi("x")
    s.messages.clear()
    monkeypatch.setattr(roi_geom, "auto_aperture", lambda *a, **k: ([10.0, 10.0, 4.0, 4.0], True, 2))
    s.auto_aperture()
    assert any("minimum size (4 pixels)" in m and "background" in m for m in s.messages)
    s.messages.clear()
    monkeypatch.setattr(roi_geom, "auto_aperture", lambda *a, **k: ([10.0, 10.0, 40.0, 4.0], True, 2))
    s.auto_aperture()
    assert any("minimum size" in m for m in s.messages)                       # either side at the minimum
    s.messages.clear()
    monkeypatch.setattr(roi_geom, "auto_aperture", lambda *a, **k: ([10.0, 10.0, 40.0, 30.0], True, 2))
    s.auto_aperture()
    assert not any("minimum size" in m for m in s.messages)


def test_auto_aperture_session(tmp_path):
    s = _session_with_beam(tmp_path)
    s.auto_aperture()
    assert s.roi_kind == "ellipse" and "/!\\No aperture defined: creating default." in s.messages
    x, y, w, h = s.roi_pos
    assert w > h                                                   # the beam is wider than high (8 x 6 sigma)
    s.set_roi("polygon", [[1, 1], [9, 1], [5, 9]])
    s.messages.clear()
    s.auto_aperture()
    assert s.messages[-1] == "/!\\Auto aperture is not available for a polygon."


def test_auto_aperture_warns_without_background_subtraction(tmp_path):
    warning = "/!\\ Auto-aperture without bkg substract. likely to fail"
    s = _session_with_beam(tmp_path)
    s.auto_aperture()
    assert warning in s.messages                                   # bkg == 0 and no deflicker
    s.messages.clear()
    s.set_bkg("5")
    s.auto_aperture()
    assert warning not in s.messages                               # a background is subtracted
    s.set_bkg("0")
    s.set_deflicker(True)
    s.messages.clear()
    s.auto_aperture()
    assert warning not in s.messages                               # the deflicker removes the background offsets
    s.set_deflicker(False)
    s.messages.clear()
    s.auto_aperture()
    assert warning in s.messages


def test_crosshair_cycle_rescale_and_check(tmp_path):
    s = _session_with_beam(tmp_path)
    assert s.crosshair.kind == "O" and s.next_crosshair_kind() == "+"
    s.set_crosshair("+"); assert s.next_crosshair_kind() == "X"
    s.set_crosshair("~")
    assert s.crosshair.kind == "~" and s.crosshair.center() == (40.5, 30.5)
    s.choose_roi(1)
    s.rescale_geometry(2.0)
    assert s.crosshair.center() == (81.0, 61.0) and s.roi_pos[2] > 0
    s.filters.pixelsize = 2.0
    s.check_geometry()
    assert s.roi_kind == "rectangle"                               # still inside the (rescaled) image
    s.set_roi("rectangle", [1e5, 1e5, 10, 10])
    s.check_geometry()
    assert s.roi_kind == "rectangle" and "/!\\Aperture outside the image: default aperture created." in s.messages
    assert is_roi_within(*s.limits(), s.roi_pos)


def test_profiles_every_crosshair_kind_and_files(tmp_path):
    s = _session_with_beam(tmp_path)
    for kind, names in (("+", "Hline"), ("X", "-"), ("~", "L1"), ("O", "Hline")):
        s.set_crosshair(kind)
        s.messages.clear()
        run = s.run_profiles()
        assert run is not None and run.names[0] == names, kind
        assert any(m.startswith(f"{names} diam.=") for m in s.messages), (kind, s.messages)
        assert any(m.startswith(f"{names} min/max=") for m in s.messages)
        assert "Cuts at Y=" in run.title
        if kind == "O":
            assert "No crosshair defined: using centroid." in s.messages
    assert (tmp_path / "Hprofile_at_Vpx30.txt").exists() or any(tmp_path.glob("Hprofile_at_Vpx*.txt"))
    assert any(tmp_path.glob("L1Profile_at_*_A*.txt"))
    s.set_crosshair("+")
    s.crosshair.x = 1e6                                            # outside the image
    s.messages.clear()
    assert s.run_profiles() is None and s.messages == ["/!\\Crosshair outside the image."]


# ---------------------------------------------------------------- window: aperture, crosshair, measurements, profiles
def _window_with_beam(qapp, tmp_path):
    from pycba.gui.main_window import MainWindow
    f, _ = _beam_file(tmp_path)
    w = MainWindow()
    w.show()
    w.open_path(f)
    qapp.processEvents()
    w.session.messages.clear(); w.msgbox.clear()
    return w


def test_window_roi_menu_creates_a_movable_aperture(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    for index, kind in ((1, "rectangle"), (2, "square"), (3, "ellipse"), (4, "circle"), (5, "polygon")):
        w.on_roi_menu(index)
        assert w.session.roi_kind == kind and w.roi_overlay.item is not None
        assert w.plot.titleLabel.text.startswith("ROI: X=")
    w.on_roi_menu(1)
    item = w.roi_overlay.item
    item.setPos((10, 12))                                          # what a drag does
    assert tuple(np.round(w.session.roi_pos[:2], 6)) == (10.0, 12.0)
    w.on_roi_menu(0)
    assert w.session.roi_kind == "" and w.roi_overlay.item is None
    w.close()


def test_window_buttons_measure_and_show_messages(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    w.energy_btn.click(); w.area_btn.click(); w.power_btn.click()
    txt = w.msgbox.toPlainText()
    assert "energy=" in txt and "area=4800" in txt and "Ave. power=" in txt
    w.size_xy_btn.click()
    assert w.session.roi_kind == "ellipse" and w.roi_menu.currentIndex() == 3        # default aperture created and shown
    assert len(w.marks.items) == 2 and "ISO beam sizes copied to clipboard" in w.msgbox.toPlainText()
    assert "\t" in QtWidgets_clipboard()
    w.size_angle_btn.click()
    assert len(w.marks.items) == 2 and "Major=" in w.msgbox.toPlainText() and "Ellipticity=" in w.msgbox.toPlainText()
    w.auto_btn.click()
    assert w.session.roi_kind == "ellipse"
    w.close()


def test_window_size_angle_button_follows_the_diameter_method(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    assert w.size_angle_btn.isEnabled() and w.size_xy_btn.isEnabled()
    tip = w.size_angle_btn.toolTip()
    assert "PRINCIPAL axes" in tip
    for index in (1, 2, 3):                                                   # 1/e2, knife-edge, 86 % diaphragm
        w.diam_menu.setCurrentIndex(index); w.on_diam(index)                  # `activated` is emitted by the user only
        assert not w.size_angle_btn.isEnabled()
        assert "4sigma" in w.size_angle_btn.toolTip() and "diameter menu" in w.size_angle_btn.toolTip()
        assert w.size_xy_btn.isEnabled() and w.energy_btn.isEnabled()         # the other buttons are not concerned
        n = len(w.msgbox.toPlainText())
        w.size_angle_btn.click()                                              # a disabled button does nothing
        assert len(w.msgbox.toPlainText()) == n
    w.diam_menu.setCurrentIndex(0); w.on_diam(0)
    assert w.size_angle_btn.isEnabled() and w.size_angle_btn.toolTip() == tip
    w.close()


def test_window_size_angle_button_stays_off_after_an_update_with_another_method(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    w.diam_menu.setCurrentIndex(1); w.on_diam(1)
    w.on_roi_menu(3)                                                          # an update that re-evaluates the buttons
    w.energy_btn.click()
    assert not w.size_angle_btn.isEnabled()
    w.close()


def QtWidgets_clipboard():
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.clipboard().text()


def test_window_crosshair_cycle_and_profiles_windows(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    assert w.crosshair_btn.text() == "+" and w.session.crosshair.kind == "O"     # no crosshair at start
    assert "Crosshair outside" not in w.msgbox.toPlainText()
    seen = []
    for expected in ("+", "X", "~", "O"):
        w.crosshair_btn.click()
        seen.append(w.crosshair_btn.text())                        # the button shows what the next click does
        assert w.session.crosshair.kind == expected
        assert len(w.cross_overlay.items) == {"X": 2, "~": 4, "O": 0, "+": 2}[expected]
    assert seen == ["X", "~", "O", "+"]
    w.crosshair_btn.click(); w.crosshair_btn.click()               # X
    w.profiles_btn.click()
    assert len(w.profile_windows) == 1 and "45" in w.profile_windows[0].windowTitle()
    w.close()
    assert not w.profile_windows[0].isVisible()


def test_window_crosshair_lines_follow_the_mouse_and_stay_inside(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    w.crosshair_btn.click()                                         # '+'
    w.cross_overlay.vline.setValue(20.0); w.cross_overlay.hline.setValue(10.0)
    assert w.session.crosshair.center() == (20.0, 10.0)
    w.crosshair_btn.click(); w.crosshair_btn.click()                # 'X' then '~'
    ov = w.cross_overlay
    ov.center.setPos((1e6, 1e6))                                    # dragged outside: clamped to the image corner
    lim = w.session.limits()
    assert w.session.crosshair.center() == (lim[0][1], lim[1][1])
    ov.arm.setPos((60.0, 30.0))
    assert abs(np.degrees(w.session.crosshair.theta) - np.degrees(np.arctan2(30 - w.session.crosshair.cy, 60 - w.session.crosshair.cx))) < 1e-6
    w.close()


def test_window_empty_reference_disables_measure_buttons(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    assert w.energy_btn.isEnabled()
    w.on_image_menu(1)                                              # B never loaded
    assert not w.energy_btn.isEnabled() and not w.profiles_btn.isEnabled() and not w.size_xy_btn.isEnabled()
    w.on_image_menu(0)
    assert w.energy_btn.isEnabled() and w.size_xy_btn.isEnabled()
    w.close()


def test_window_pixelsize_keeps_aperture_on_the_image(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    w.on_roi_menu(1)
    before = w.session.roi_pos.copy()
    w.pixelsize_edit.setText("2"); w.on_pixelsize()
    assert np.allclose(w.session.roi_pos, before * 2) and w.session.roi_kind == "rectangle"
    assert np.allclose(w.roi_overlay.read(), before * 2)
    w.close()


def test_linear_scale_never_covers_negative_values_when_the_peak_is_not_positive():
    from pycba.core import display_scale
    img = np.full((5, 5), -50.0)                                    # background subtracted too high
    for log in (False, True):
        spec = display_scale(img, 1, 1, True, 256.0, log, 128, 1.0)
        assert spec.vmax > spec.vmin >= (-1e-9 if not log else -20)       # palette padding: tiny negative offset by design
    lin = display_scale(img, 1, 1, True, 256.0, False, 0, 1.0)
    assert lin.vmin == 0.0 and lin.vmax > 0
    assert display_scale(np.full((3, 3), np.nan), 1, 1, True, 256.0, False, 0, 1.0).vmin == 0.0


def test_session_scale_after_a_too_high_background(tmp_path):
    s = _session_with_beam(tmp_path)
    s.set_bkg("1000")
    s.choose_zscale(0)                                              # Normalize
    spec = s.display()
    assert spec.vmin >= 0 and spec.vmax > spec.vmin


def test_window_reserves_the_roi_text_row_and_has_no_useless_handles(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    h0 = w.plot.vb.sceneBoundingRect().height()
    for index, n_handles in ((1, 8), (2, 4), (3, 8), (4, 4)):
        w.on_roi_menu(index)
        qapp.processEvents()
        assert len(w.roi_overlay.item.getHandles()) == n_handles, index
        assert all(hd.typ == "s" for hd in w.roi_overlay.item.getHandles())      # no rotation handle
        drawn = [c for c in w.roi_overlay.item.childItems() if isinstance(c, pg.graphicsItems.ROI.Handle)]
        assert len(drawn) == n_handles, index                                    # no stray graphics item left drawn
        assert abs(w.plot.vb.sceneBoundingRect().height() - h0) < 1.0             # the image does not jump
    w.on_roi_menu(0)
    qapp.processEvents()
    assert abs(w.plot.vb.sceneBoundingRect().height() - h0) < 1.0
    w.close()


def test_aperture_is_transparent_to_the_mouse_except_along_its_border(qapp, tmp_path):
    """Clicks inside an aperture reach what lies below (crosshair, pixel killer); only a strip along its border grabs."""
    from pyqtgraph.Qt import QtCore
    w = _window_with_beam(qapp, tmp_path)
    w.crosshair_btn.click()                       # '+' crosshair, drawn over the image
    qapp.processEvents()
    for index in (1, 2, 3, 4, 5):                 # rectangle, square, ellipse, circle, polygon
        w.on_roi_menu(index)
        qapp.processEvents()
        item = w.roi_overlay.item
        shape = item.shape()
        r = item.mapRectToParent(item.boundingRect()) if index < 5 else item.mapRectToParent(shape.boundingRect())
        cx, cy = r.center().x(), r.center().y()
        centre = item.mapFromParent(QtCore.QPointF(cx, cy))
        assert not shape.contains(centre), index                              # inside: transparent
        g = item._grab()
        top = item.mapFromParent(QtCore.QPointF(cx, r.top() + 0.4 * g))       # just inside the top border
        if index in (1, 2):
            assert shape.contains(top), index
        assert shape.boundingRect().width() > 0
        # what pyqtgraph's mouse dispatch finds under the pointer: at the centre the crosshair line, not the aperture...
        w.cross_overlay.vline.setPos(cx)
        qapp.processEvents()

        class Ev:                                   # minimal stand-in of a mouse event
            def __init__(self, scene_pos):
                self.sp = scene_pos

            def scenePos(self):
                return self.sp

        def under(x, y):
            return w.plot.scene().itemsNearEvent(Ev(w.plot.vb.mapViewToScene(QtCore.QPointF(x, y))), hoverable=True)

        hits = under(cx, cy)
        assert w.cross_overlay.vline in hits and item not in hits, index
        # ... while a point of the border grabs the aperture
        if index in (1, 2, 3, 4):
            assert item in under(cx, r.top() + 0.4 * g), index
    assert w.cross_overlay.items and all(it.zValue() > w.roi_overlay.item.zValue() for it in w.cross_overlay.items)
    w.close()


def test_window_crosshair_has_a_wide_grab_zone(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    w.crosshair_btn.click()
    line = w.cross_overlay.hline
    assert line.pen.width() <= 1 and line.hoverPen.width() >= 10
    w.close()


def test_help_button_shows_the_help_text(qapp, monkeypatch, tmp_path):
    from pyqtgraph.Qt import QtWidgets
    from pycba.gui import main_window
    w = _window_with_beam(qapp, tmp_path)
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(a[2])))
    w.help_btn.click()
    assert shown and shown[0] == main_window.HELP_TEXT and "5/ Compute the 1-D profiles" in shown[0]
    w.close()


def test_zoom_is_kept_by_updates_and_reset_by_loads(qapp, tmp_path):
    """``plot_image(imageNum, keepAxes)``: bkg, filter, log, Z scale, pixel kill keep the zoom; a load, the image menu and
    the pixel size reset it."""
    w = _window_with_beam(qapp, tmp_path)
    def zoom():
        w.plot.setRange(xRange=(10, 30), yRange=(10, 25), padding=0)
        qapp.processEvents()
        return tuple(np.round(w.plot.vb.viewRange()[0], 3))
    z = zoom()
    w.bkg_edit.setText("5"); w.on_bkg()
    w.log_check.setChecked(True); w.on_log(True)
    w.on_zscale(3)
    w.filter_edit.setText("3"); w.on_filter()
    assert tuple(np.round(w.plot.vb.viewRange()[0], 3)) == z
    w.session.toggle_pixel(12, 12); w.redraw(keep_view=True)
    assert tuple(np.round(w.plot.vb.viewRange()[0], 3)) == z
    w.pixelsize_edit.setText("2"); w.on_pixelsize()
    assert tuple(np.round(w.plot.vb.viewRange()[0], 3)) != z                  # full image again
    zoom(); w.on_image_menu(0)
    assert w.plot.vb.viewRange()[0][1] > 40
    zoom(); w.open_path(w.session.image.filepath)
    assert w.plot.vb.viewRange()[0][1] > 40
    w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v", "-p", "no:cacheprovider"]))


def test_window_auto_bkg_button_and_flicker_legend(qapp, tmp_path):
    w = _window_with_beam(qapp, tmp_path)
    assert w.rects["deflicker_text"].text() == "Flicker removal"
    assert w.bkg_auto_btn.text() == "Auto" and w.bkg_auto_btn.isEnabled()
    a, b, f = w.bkg_auto_btn.geometry(), w.bkg_edit.geometry(), w.filter_edit.geometry()
    assert not a.intersects(b) and not a.intersects(f) and not a.intersects(w.rects["bkg_image"].geometry())
    w.bkg_auto_btn.click()
    qapp.processEvents()
    assert float(w.bkg_edit.text()) == w.session.image.bkg
    assert w.msgbox.toPlainText().startswith("Bkg = ")
