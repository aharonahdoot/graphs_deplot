import cv2, sys
sys.path.insert(0, "src")
from calibrate import calibrate

imgs = [
    "171881-019-R_P0.jpeg",   # Y 0..600 step50, X 9..349 step20
    "352319-331-R_P1.jpeg",   # Y 0..2400 step200 (commas), X 324..624 step30
    "K33807-207-R_P3.jpeg",   # Y 0..1600 step200, big font
    "337426-197-R_P0.jpeg",   # Y 0..950 step50, X 15..145 step10
    "556579-094-R P2.jpeg",   # Y 0..2400, X 209..? step3
    "824345-141-R_P0.jpeg",
]
for fn in imgs:
    bgr = cv2.imread("Growth_Curves_NCI_BRCA/" + fn)
    c = calibrate(bgr)
    y, x = c["y"], c["x"]
    print(f"\n=== {fn}  box={c['box']}")
    if y:
        s, i, n, rms = y
        print(f"  Y: value = {s:.4f}*py + {i:.2f}   inliers={n}/{len(c['y_pts'])} rms={rms:.3f}")
    else:
        print(f"  Y: FAILED  pts={c['y_pts']}")
    if x:
        s, i, n, rms = x
        print(f"  X: value = {s:.4f}*px + {i:.2f}   inliers={n}/{len(c['x_pts'])} rms={rms:.3f}")
    else:
        print(f"  X: FAILED  pts={c['x_pts']}")
