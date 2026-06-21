import cv2, glob, os, sys
sys.path.insert(0, "src")
from calibrate import calibrate

files = sorted(glob.glob("Growth_Curves_NCI_BRCA/*.jpeg"))
flags = []
for f in files:
    bgr = cv2.imread(f)
    c = calibrate(bgr)
    name = os.path.splitext(os.path.basename(f))[0]
    y, x = c["y"], c["x"]
    yr = f"y={y[0]:+.4f}*p{y[1]:+.1f} n={y[2]}/{len(c['y_pts'])} rms={y[3]:.2f}" if y else f"y=FAIL pts={len(c['y_pts'])}"
    xr = f"x={x[0]:+.4f}*p{x[1]:+.1f} n={x[2]}/{len(c['x_pts'])} rms={x[3]:.2f}" if x else f"x=FAIL pts={len(c['x_pts'])}"
    bad = (not y) or (not x) or (y and (y[3] > 5 or y[2] < 3)) or (x and (x[3] > 5 or x[2] < 3))
    print(f"{'!!' if bad else '  '} {name:22s} | {yr:42s} | {xr}")
    if bad:
        flags.append(name)
print(f"\n{len(files)} images, {len(flags)} flagged: {flags}")
