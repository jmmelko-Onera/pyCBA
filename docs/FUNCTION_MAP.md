# MATLAB function -> pyCBA (state of the port)

| MATLAB (MAIN_CameraBeamAnalyzer.m / Utils) | pyCBA | Remark |
|---|---|---|
| `CameraBeamAnalyzer` (init, uicontrols) | `gui/main_window.py`, `gui/layout.py`, `gui/session.py` | layout and widgets ported; start-up values partly from the user's IMAGES init |
| `HelpButton_Callback` | `MainWindow.on_help` (`HELP_TEXT`) | |
| `deleteROI`, `abortROIcallback` | `Session.clear_roi` | Java robot / ESC no longer needed (no modal drawing) |
| `createROI`, `createDefaultROI` | `Session.choose_roi`, `core.roi.default_roi`, `gui/overlays.RoiOverlay` | aperture appears at once around the peak |
| `backupROI`, `backupCrosshair` | (not needed) | the Session is the single source of truth. `backupCrosshair` code was never supplied |
| `ROIPosition_Callback` | `Session.roi_text` | |
| `gaussianfit`, `PlotGaussianFit`, `myGaussFit` | `core.profiles` (`gaussian_fit`, `fit_profiles`, `my_gauss_fit`) | |
| `updatemsgtxt` | message box (`session.say`) | redesigned: appends, 500 lines |
| `BrowseButton`, `Left/RightButton`, `path_Callback`, `listFiles`, `checkGetFile` | `Session`, `io.check_get_file` | |
| `ROImenu`, `Energy/Area/PowerButton` | `Session.choose_roi`, `measure_energy/area/power`, `core.measure` | |
| `ComputeCentroid`, `ComputePeakCenter`, `ComputeD4sigma`, `ComputeDpeak`, `ComputeDKnifeEdge`, `ComputeDAperture`, `ComputeBeamPeak` | `core.beam_metrics` | validated against MATLAB exports |
| `SizeXYButton`, `DiamMenu` | `core.measure.size_xy`, `Session.measure_size_xy`, `choose_diam` | |
| `Size45Button` | **removed**: replaced by `SizeAngleButton` (below) | the 45 degree sizes were a workaround for the lack of an arbitrary angle |
| `AutoApertureButton` | `core.roi.auto_aperture`, `Session.auto_aperture` | corrected algorithm |
| `cross*ConstraintFcn`, `crossLCalculateConstrain`, `createCrosshair`, `CrosshairButton` | `core.roi` (`Crosshair`, `cross_*`), `gui/overlays.CrosshairOverlay`, `Session.set_crosshair` | |
| `GetProfilesCenter` | `Crosshair.center`, `Session.run_profiles` (centroid if none) | |
| `ComputeProfilesMinMax`, `smoothOutliers`, `lineProfile` | `core.profiles` | |
| `ProfileButton`, `Profile45Button`, `ProfileL1L2Button` | `core.profiles`, `Session.run_profiles`, `gui/profiles_window.py`, `core.profile_files` | |
| `bkg`, `filter`, `energy`, `pixelSize` callbacks | `Session.set_bkg/set_filter/set_energy/set_pixelsize` | |
| `pixelKiller` | `Session.toggle_pixel`, Pix Kill mode button | `pixelKiller_Callback` (mode switching) code never supplied: designed as a toggle button |
| `hDeflickerCheck`, `refreshDeflickerCheck`, `computeFlicker` | `Session.set_deflicker`, `core.compute_flicker` | |
| `hLogCheck`, `refreshZScale`, `reinitZScale`, `ZscaleMenu` | `Session.set_log`, `core.zscale` | |
| `ImageNumMenu`, `NormMenu`, `normalizeImage` | `Session.switch_image`, `choose_norm`, `core.normalize_image` | |
| `create_axes` | (layout handled by Qt / `layout.py`) | |
| `clean_image`, `load_imagefile`, `update_image` | `core.clean_image`, `core.zscale.load_image(_file)`, `core.update_image` | |
| `isROIWithin`, `isCrossWithin` | `core.is_roi_within`, `core.is_cross_within`, `Session.check_geometry` | |
| `plot_image` | `MainWindow.redraw` (keeps the zoom like `keepAxes`), `core.display_scale`, overlays | |
| `readAsciiFromFile`, `analyzeASCII` | `io.read_ascii`, `io.analyze_ascii` | |
| `myCloseReq` | `MainWindow.closeEvent` (confirmation) | |
| `divisor` | `io.converters.divisors` | only used by the PRW size guess (no longer needed: the PRW header holds the size) |
| `myimresize` | `io.converters.resize_image` | nearest for scale > 1 (identical to MATLAB's fallback); scale < 1 bicubic via Pillow (not bit-identical to `imresize`). Used by `examples/convert_files.py` (`PX_SCALE`, RAW only, as in MATLAB) |
| `MAIN_SpiriconRAW_to_ASCII` | `io.converters.read_pyrocam_raw`, `colorize`, `write_ascii` | outside MAIN; GUI integration to be decided |
| `MAIN_PhotonPRW_to_ASCII` | `io.converters.read_photon_prw`, `deinterlace`, `normalize_z`, `median_filter` | outside MAIN; GUI integration to be decided |
| `MAIN_Import_screenshot_as_ASCII` | `io.converters.import_screenshot` | outside MAIN; GUI integration to be decided |
| (new) `estimateBackground` | `core.background.estimate_background`, `Session.auto_bkg`, 'Auto' button | median of a 5 % border frame; MATLAB version in `matlab/estimateBackground.m` |
| (new) `SizeAngleButton`, `computePrincipalAxes` | `core.beam_metrics.compute_principal_axes`, `core.measure.size_angle`, `Session.measure_size_angle`, 'ISO Beam size angle' button | ISO 11146-1 principal axes, angle (counter-clockwise on screen) and ellipticity (4 sigma, second moments); MATLAB: `matlab/computePrincipalAxes.m`, `matlab/PRINCIPAL_AXES.txt` |
| `ComputeDKnifeEdge` (sub-pixel) | `core.beam_metrics.compute_d_knife_edge(subpixel=True)` | linear interpolation of the cumulative energy; `subpixel=False` = MATLAB's integer pixel result; MATLAB: `matlab/knifeEdgeAxis.m`, `matlab/KNIFE_EDGE_SUBPIXEL.txt` |
| `ComputeDpeak` (1/e², sub-pixel) | `core.beam_metrics.compute_dpeak` (`_width_at_level`) | search outwards from the peak, linear interpolation, D = distance between the two crossings (a missing side = twice the other half-width); MATLAB: `matlab/widthAtLevel.m`, `matlab/DPEAK_SUBPIXEL.txt` |
| `AutoApertureButton_Callback` (min. size) | `core.roi.auto_aperture`, `MIN_AUTO_APERTURE = 4` | width and height never below 4 pixels; the session says so (`/!\Auto-aperture stopped at its minimum size`); MATLAB: `matlab/ROI_CROSSHAIR_FIXES.txt` §1 |
| `ISO Beam size angle` button | `gui.main_window._refresh_size_angle_button` | greyed out (with an explanatory tooltip) unless the diameter menu is on 4sigma |
