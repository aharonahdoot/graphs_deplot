"""Regression: a page/figure border line outside the plot must NOT be taken as
an axis spine. Before the cream-region constraint in find_spines, the lowest long
dark horizontal line won (`hrows.max()`), so a rule below the axis title became
the x-axis -- inflating the box and masking the tick labels as spurious markers
(see experiments/notes/detection-failure-investigation.md). Self-contained: it
renders a minimal chart, so it needs no external (local-only) images.

Run: .venv/bin/python tests/test_find_spines_border.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import cv2
from markers import find_spines


def _chart_with_border():
    H, W = 600, 800
    img = np.full((H, W, 3), 255, np.uint8)          # white page
    l, t, r, b = 100, 40, 760, 500
    img[t:b, l:r] = (238, 244, 250)                  # cream plot interior (high V, low-mid S)
    axis = (70, 70, 75)
    cv2.line(img, (l, t), (l, b), axis, 1)           # left spine
    cv2.line(img, (l, b), (r, b), axis, 1)           # true x-axis baseline (row b)
    # the offending out-of-plot rule: a long dark line near the image bottom,
    # below where the axis title would sit (~80px below the real baseline)
    cv2.line(img, (0, H - 6), (W - 1, H - 6), (60, 60, 60), 1)
    return img, (l, t, r, b)


if __name__ == "__main__":
    img, (l, t, r, b) = _chart_with_border()
    L, T, R, B = find_spines(img)
    print(f"find_spines -> bottom={B} (true baseline={b}, page border at {img.shape[0]-6})")
    assert abs(B - b) <= 5, \
        f"bottom spine {B} latched onto the page border, not the baseline {b}"
    print("PASS")
