"""Regression: a plot CUT OFF at the top image edge (cream runs to the top with
no white margin or frame) must get its box top at the cut edge, not one gridline
below it. Otherwise the box ends a tick early and the highest data point -- which
lives in that top tick band -- is excluded and missed (see
experiments/notes/origin-spine-recovery-investigation.md §6). Self-contained.

Run: .venv/bin/python tests/test_find_spines_cut_top.py
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import cv2
import markers as M
from markers import find_spines, detect_markers


def _chart_cut_at_top():
    H, W = 420, 640
    l, b, r = 80, 380, 600
    img = np.full((H, W, 3), 255, np.uint8)
    img[0:b, l:r] = (238, 244, 250)               # cream runs to the TOP edge (row 0) -> cut off
    for gy in range(28, b, 60):                    # faint gridlines; the first (row 28) is a tick down
        cv2.line(img, (l, gy), (r, gy), (212, 214, 220), 1)
    color = (170, 80, 150)                         # purple series
    pts = [(l + 12, b - 6), (250, 250), (430, 120), (560, 12)]   # last point in the top tick band
    cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)
    cv2.line(img, (l, 0), (l, b), (70, 70, 75), 1)   # left spine (to the top)
    cv2.line(img, (l, b), (r, b), (70, 70, 75), 1)   # bottom spine
    return img, (l, b, r), pts[-1]


if __name__ == "__main__":
    img, (l, b, r), peak = _chart_cut_at_top()

    M._RECOVER_CUT_TOP = True
    top_on = find_spines(img)[1]
    mk_on, _ = detect_markers(img)
    M._RECOVER_CUT_TOP = False
    top_off = find_spines(img)[1]
    mk_off, _ = detect_markers(img)
    M._RECOVER_CUT_TOP = True

    found = lambda mks: any(math.hypot(x - peak[0], y - peak[1]) <= 16 for x, y in mks)
    print(f"box top: with cut-top fix={top_on}, without={top_off}  (image top=0)")
    print(f"peak {peak} detected: with={found(mk_on)}, without={found(mk_off)}")
    assert top_on <= 6, f"box top should reach the cut edge, got {top_on}"
    assert top_off > 20, "test invalid: without the fix the top should land on the first gridline"
    assert found(mk_on), "the top-band peak was not detected after the cut-top fix"
    print("PASS")
