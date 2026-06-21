"""End-to-end verification for a single image: draw the CALIBRATED grid (lines at
tick values computed from the fitted transform) + detected markers annotated
with their extracted (x,y) values.
"""
import argparse
import cv2
import math
import os
import sys
from collections import Counter

sys.path.insert(0, "src")
from markers import detect_markers
from calibrate import calibrate, pixel_to_data


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
    k0 = math.ceil((v_lo - phase) / step)
    k1 = math.floor((v_hi - phase) / step)
    return [phase + k * step for k in range(k0, k1 + 1)]


def main():
    parser = argparse.ArgumentParser(description="Generate verification overlay for one image.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("-o", "--out-dir", default="output/verify", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: file not found: {args.image_path}")
        sys.exit(1)

    print(f"Processing: {args.image_path}")
    bgr = cv2.imread(args.image_path)
    if bgr is None:
        print(f"Error: could not read image: {args.image_path}")
        sys.exit(1)

    H, W = bgr.shape[:2]
    markers, _ = detect_markers(bgr)
    cal = calibrate(bgr)

    # Copy the image for overlay
    ov = bgr.copy()

    # Check if calibration succeeded
    if cal["x"] and cal["y"]:
        (xs, xi), (ys, yi) = cal["x"][:2], cal["y"][:2]
        l, t, r, b = cal["box"]

        # complete calibrated vertical grid (x ticks) from the fit
        if cal["x_pts"]:
            for v in regular_ticks([p[1] for p in cal["x_pts"]], xs, xi, l, r):
                px = int(round((v - xi) / xs))
                if l <= px <= r:
                    cv2.line(ov, (px, t), (px, b), (0, 200, 0), 1)

        # complete calibrated horizontal grid (y ticks) from the fit
        if cal["y_pts"]:
            for v in regular_ticks([p[1] for p in cal["y_pts"]], ys, yi, t, b):
                py = int(round((v - yi) / ys))
                if t <= py <= b:
                    cv2.line(ov, (l, py), (r, py), (0, 200, 0), 1)

        # Draw box outline
        cv2.rectangle(ov, (l, t), (r, b), (0, 150, 0), 1)

        # markers + value labels
        for px, py in markers:
            x, y = pixel_to_data(cal, px, py)
            p = (int(round(px)), int(round(py)))
            cv2.circle(ov, p, 6, (0, 0, 255), 2)
            cv2.putText(ov, f"{x:.0f},{y:.0f}", (p[0] + 6, p[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 0), 1)
        
        print(f"Calibration succeeded. Detected {len(markers)} markers.")
    else:
        print("Warning: Calibration failed or incomplete. Just drawing markers on the overlay.")
        for px, py in markers:
            p = (int(round(px)), int(round(py)))
            cv2.circle(ov, p, 6, (0, 0, 255), 2)

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(args.image_path))[0] + ".png"
    out_path = os.path.join(args.out_dir, out_name)
    cv2.imwrite(out_path, ov)
    print(f"Wrote verification overlay to: {out_path}")


if __name__ == "__main__":
    main()
