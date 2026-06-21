"""Automated end-to-end verification of the extraction system on all 44 images.

Asserts, with no hand-labelled ground truth:
  * every image calibrates (both axes fit);
  * calibration residuals are small (tick fit rms within tolerance);
  * enough tick labels are used per axis (>=3 inliers);
  * every extracted marker value lies within the chart's own plotted axis
    range (the spine-to-spine box extent, +/- a small margin) -- i.e. no
    marker maps to an impossible value from a broken calibration;
  * a non-trivial number of markers is found per image.
Exits non-zero on any failure.
"""
import glob, os, sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from markers import detect_markers
from calibrate import calibrate, pixel_to_data

files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..",
                                      "Growth_Curves_NCI_BRCA", "*.jpeg")))
assert files, "no images found"

fails, total_markers = [], 0
print(f"{'image':24s} {'mk':>3} {'xrms':>6} {'yrms':>6} {'x-range':>16} {'y-range':>16}")
for f in files:
    name = os.path.basename(f)
    bgr = cv2.imread(f)
    markers, _ = detect_markers(bgr)
    cal = calibrate(bgr)
    problems = []

    if not (cal["x"] and cal["y"]):
        problems.append("not calibrated")
    else:
        xs, xi, xn, xrms = cal["x"]
        ys, yi, yn, yrms = cal["y"]
        if xn < 3 or yn < 3:
            problems.append(f"few inliers x={xn} y={yn}")
        if xrms > 5 or yrms > 8:
            problems.append(f"high rms x={xrms:.2f} y={yrms:.2f}")
        # True plotted range = the spine-to-spine box extent (markers are
        # detected inside the box, so legitimate values must map within it).
        l, t, r, b = cal["box"]
        xlo, xhi = sorted([xs * l + xi, xs * r + xi])
        ylo, yhi = sorted([ys * t + yi, ys * b + yi])
        mx, my = 0.03 * (xhi - xlo) + 1, 0.03 * (yhi - ylo) + 1
        out_of_range = 0
        for px, py in markers:
            x, y = pixel_to_data(cal, px, py)
            if not (xlo - mx <= x <= xhi + mx) or not (ylo - my <= y <= yhi + my):
                out_of_range += 1
        xv = sorted(v for _, v in cal["x_pts"])
        yv = sorted(v for _, v in cal["y_pts"])
        if out_of_range:
            problems.append(f"{out_of_range} values out of axis range")
        if len(markers) < 3:
            problems.append(f"only {len(markers)} markers")
        print(f"{name:24s} {len(markers):3d} {xrms:6.2f} {yrms:6.2f} "
              f"{xv[0]:7.0f}..{xv[-1]:<7.0f} {yv[0]:7.0f}..{yv[-1]:<7.0f}"
              + ("  <-- " + "; ".join(problems) if problems else ""))
    total_markers += len(markers)
    if problems:
        fails.append((name, problems))

print(f"\n{len(files)} images, {total_markers} markers total, {len(fails)} failures")
if fails:
    for n, p in fails:
        print("  FAIL", n, p)
    sys.exit(1)
print("ALL CHECKS PASSED")
