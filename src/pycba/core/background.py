"""Automatic background estimate (new feature, no MATLAB counterpart).

The background is a single value, estimated as the **median of a frame** ``FRAME_FRACTION`` (5 %, like the
flicker columns) wide along the four edges of the image. The aperture plays no part in it: the aperture surrounds
the beam, the background is measured far from it. The median ignores isolated hot / dead pixels and a damaged area
that covers up to about a quarter of the frame (a laser impact on one quadrant of the sensor). Simulated on
124 x 124 images (noise sigma = 5): 10 % of hot pixels move the estimate by 0.7 count, a damaged quadrant by
2.4 counts (the mean is off by 400 and 150). The median of the four *sides* is NOT used: with a damaged quadrant two
sides are half contaminated and it is off by 120 counts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FRAME_FRACTION = 0.05
MIN_SIDE = 20                   # pixels: below, a 1-pixel frame holds too few pixels to mean anything
HIGH_FRACTION = 0.10             # background above this fraction of the peak: probably beam wings on the border
NON_UNIFORM_FRACTION = 0.05      # a side median further than this fraction of (peak - background) from the value


@dataclass(frozen=True)
class BackgroundEstimate:
    value: float
    sides: tuple                 # median of the top, bottom, left and right bands
    non_uniform: bool            # the bands disagree: beam reaching the border, damaged area, strong gradient...
    high: bool                   # the value is above HIGH_FRACTION of the peak (wide beam: wings on the whole border)
    frame_pixels: int            # number of valid pixels used


def estimate_background(image: np.ndarray, frame: float = FRAME_FRACTION) -> BackgroundEstimate:
    """Median of the border frame of ``image`` (NaN = dead pixels are ignored). ``ValueError`` if it cannot be done."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("2D image expected")
    rows, cols = image.shape
    kr, kc = max(1, math.ceil(frame * rows)), max(1, math.ceil(frame * cols))
    if min(rows, cols) < MIN_SIDE or 2 * kr >= rows or 2 * kc >= cols:
        raise ValueError(f"image too small to estimate the background on its border (minimum {MIN_SIDE} x {MIN_SIDE} pixels)")
    mask = np.zeros(image.shape, dtype=bool)
    mask[:kr] = mask[-kr:] = True
    mask[:, :kc] = mask[:, -kc:] = True
    vals = image[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("no valid pixel on the image border")
    value = float(np.median(vals))
    sides = tuple(float(np.nanmedian(s)) if np.isfinite(s).any() else float("nan")
                  for s in (image[:kr], image[-kr:], image[:, :kc], image[:, -kc:]))
    peak = np.nanmax(image[np.isfinite(image)]) if np.isfinite(image).any() else value
    signal = peak - value
    non_uniform = bool(signal > 0 and np.nanmax(np.abs(np.array(sides) - value)) > NON_UNIFORM_FRACTION * signal)
    high = bool(peak > 0 and value > HIGH_FRACTION * peak)
    return BackgroundEstimate(value, sides, non_uniform, high, int(vals.size))
