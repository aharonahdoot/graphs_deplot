"""Measure background, series, and frame pixel values to design adaptive thresholds."""
import cv2, numpy as np, sys
files = [
    ("orange", "Growth_Curves_NCI_BRCA/171881-019-R_P0.jpeg"),
    ("olive",  "Growth_Curves_NCI_BRCA/171881-019-R_P3.jpeg"),
    ("red",    "Growth_Curves_NCI_BRCA/352319-331-R_P1.jpeg"),
    ("black5", "Growth_Curves_NCI_BRCA/556579-094-R P2.jpeg"),
    ("black",  "Growth_Curves_NCI_BRCA/337426-197-R_P0.jpeg"),
]
for tag, f in files:
    bgr = cv2.imread(f); H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:,:,1].astype(int), hsv[:,:,2].astype(int)
    # background = brightest cluster
    bgV = np.percentile(V, 80)
    bgS = np.percentile(S, 50)
    # candidate "series" = darker than bg by >D OR saturation high
    # Show distribution of how-dark and saturation among the darker pixels
    diff = bgV - V
    darkish = diff > 8
    print(f"\n=== {tag}  {f.split('/')[-1]} ({W}x{H})")
    print(f"  bgV(p80)={bgV:.0f}  bgS(p50)={bgS:.0f}  Vp5={np.percentile(V,5):.0f}  Sp95={np.percentile(S,95):.0f}")
    # among darkish pixels, what V and S?
    if darkish.sum():
        print(f"  darkish(diff>8): frac={darkish.mean():.3f}  their V: p5={np.percentile(V[darkish],5):.0f} med={np.median(V[darkish]):.0f}  their S: med={np.median(S[darkish]):.0f} p90={np.percentile(S[darkish],90):.0f}")
    # frame detection test: relative darkness long lines
    for D in (8, 20, 40):
        m = (diff > D).astype(np.uint8)*255
        vk = cv2.getStructuringElement(cv2.MORPH_RECT,(1,H//4))
        hk = cv2.getStructuringElement(cv2.MORPH_RECT,(W//4,1))
        cols = np.where(cv2.erode(m,vk).sum(axis=0)>0)[0]
        rows = np.where(cv2.erode(m,hk).sum(axis=1)>0)[0]
        cx = f"{cols.min()}..{cols.max()}" if len(cols) else "none"
        rx = f"{rows.min()}..{rows.max()}" if len(rows) else "none"
        print(f"  D={D:2d}: vlines x[{cx}] hlines y[{rx}]")
