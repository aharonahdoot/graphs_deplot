"""Sweep the h-maxima prominence threshold on representative images."""
import cv2, numpy as np, sys
sys.path.insert(0, "src")
from markers import series_mask
from skimage.morphology import h_maxima
from scipy import ndimage

imgs = {
    "orange_clean": "171881-019-R_P0.jpeg",   # ~32 expected, thin line, no staircase issue
    "orange_steep": "171881-019-R_P1.jpeg",    # steep zigzags -> staircase risk
    "olive_dense":  "352319-331-R_P2.jpeg",    # dense overlapping -> want MORE
    "red_tri_dense":"352319-331-R_P1.jpeg",    # dense triangles
    "teal_thick":   "K33807-207-R_P4.jpeg",    # thick line -> false-peak risk
    "magenta_thick":"K33807-207-R_P3.jpeg",
}
hs = [1.3, 1.5, 1.6, 1.7, 2.0]
print(f"{'image':16s} dmax  lw  " + "  ".join(f"h={h}" for h in hs))
for tag, fn in imgs.items():
    bgr = cv2.imread("Growth_Curves_NCI_BRCA/" + fn)
    mask, box = series_mask(bgr)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5); dmax = dist.max()
    d = dist[dist > 0.5]; lw = np.sort(d)[int(0.6 * d.size)]
    counts = []
    for h in hs:
        prom = h_maxima(dist, h) & (dist >= max(2.1, lw + 0.8))
        _, n = ndimage.label(prom)
        counts.append(n)
    print(f"{tag:16s} {dmax:4.1f} {lw:3.1f}  " + "  ".join(f"{c:4d}" for c in counts))
