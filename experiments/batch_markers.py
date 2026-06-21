import cv2, glob, os, sys
sys.path.insert(0, "src")
from markers import detect_markers, draw_overlay

files = sorted(glob.glob("Growth_Curves_NCI_BRCA/*.jpeg"))
os.makedirs("experiments/out/markers", exist_ok=True)
print(f"{len(files)} images\n")
for f in files:
    bgr = cv2.imread(f)
    markers, diag = detect_markers(bgr)
    name = os.path.splitext(os.path.basename(f))[0]
    ov = draw_overlay(bgr, markers, diag["box"])
    cv2.imwrite(f"experiments/out/markers/{name}.png", ov)
    print(f"{name:22s} n={len(markers):3d}  dmax={diag['dmax']:.1f} lw={diag['lw']:.1f} h={diag['h']:.1f} box={diag['box']}")
