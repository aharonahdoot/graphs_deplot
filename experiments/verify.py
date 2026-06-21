"""End-to-end verification: for each image draw the CALIBRATED grid (lines at
tick values computed from the fitted transform) + detected markers annotated
with their extracted (x,y) values. If the calibrated grid lands on the image's
printed gridlines/ticks, calibration is correct; if circles sit on markers,
detection is correct."""
import cv2, glob, os, numpy as np, sys
from collections import Counter
sys.path.insert(0, "src")
from markers import detect_markers
from calibrate import calibrate, pixel_to_data

os.makedirs("output/verify", exist_ok=True)
files = sorted(glob.glob("Growth_Curves_NCI_BRCA/*.jpeg"))

def tick_step(vals):
    """Most common spacing between consecutive distinct OCR'd tick values."""
    u = sorted(set(round(v) for v in vals))
    diffs = [b - a for a, b in zip(u, u[1:]) if b > a]
    return Counter(diffs).most_common(1)[0][0] if diffs else 1

def regular_ticks(vals, slope, intercept, p_lo, p_hi):
    """Full regular tick sequence spanning the plotted pixel range, derived from
    the fitted transform -- independent of which labels OCR happened to read.
    Ticks are PHASE-ALIGNED to the real tick values (e.g. 8,38,68 with step 30),
    not just multiples of the step from zero."""
    u = sorted(set(round(v) for v in vals))
    step = tick_step(u)
    phase = Counter(int(v) % step for v in u).most_common(1)[0][0]  # majority offset
    v_lo, v_hi = sorted([slope * p_lo + intercept, slope * p_hi + intercept])
    import math
    k0 = math.ceil((v_lo - phase) / step)
    k1 = math.floor((v_hi - phase) / step)
    return [phase + k * step for k in range(k0, k1 + 1)]

for f in files:
    bgr = cv2.imread(f); H, W = bgr.shape[:2]
    markers, _ = detect_markers(bgr)
    cal = calibrate(bgr)
    ov = bgr.copy()
    (xs, xi), (ys, yi) = cal["x"][:2], cal["y"][:2]
    l, t, r, b = cal["box"]
    # complete calibrated vertical grid (x ticks) from the fit
    for v in regular_ticks([p[1] for p in cal["x_pts"]], xs, xi, l, r):
        px = int(round((v - xi) / xs))
        if l <= px <= r:
            cv2.line(ov, (px, t), (px, b), (0, 200, 0), 1)
    # complete calibrated horizontal grid (y ticks) from the fit
    for v in regular_ticks([p[1] for p in cal["y_pts"]], ys, yi, t, b):
        py = int(round((v - yi) / ys))
        if t <= py <= b:
            cv2.line(ov, (l, py), (r, py), (0, 200, 0), 1)
    # markers + value labels
    for px, py in markers:
        x, y = pixel_to_data(cal, px, py)
        p = (int(round(px)), int(round(py)))
        cv2.circle(ov, p, 6, (0, 0, 255), 2)
        cv2.putText(ov, f"{x:.0f},{y:.0f}", (p[0] + 6, p[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 0), 1)
    cv2.imwrite(f"output/verify/{os.path.splitext(os.path.basename(f))[0]}.png", ov)
print("wrote", len(files), "verification overlays to output/verify/")
