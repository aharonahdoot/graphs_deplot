"""Probe color structure + detect plot frame on one image."""
import cv2, numpy as np, sys

path = sys.argv[1] if len(sys.argv) > 1 else "Growth_Curves_NCI_BRCA/171881-019-R_P0.jpeg"
bgr = cv2.imread(path)
H, W = bgr.shape[:2]
print(f"image {path}  size WxH = {W}x{H}")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Sample center region (likely background gradient) vs whole image stats
print("\n-- whole image HSV means --")
print("H mean/med:", hsv[:,:,0].mean(), np.median(hsv[:,:,0]))
print("S mean/med/p95:", hsv[:,:,1].mean(), np.median(hsv[:,:,1]), np.percentile(hsv[:,:,1],95))
print("V mean/med/p5 :", hsv[:,:,2].mean(), np.median(hsv[:,:,2]), np.percentile(hsv[:,:,2],5))

# Frame detection via long dark lines
# dark mask: low value
dark = (gray < 130).astype(np.uint8)*255
print("\ndark pixel fraction:", dark.mean()/255)

# vertical lines: erode with tall thin kernel
vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, H//6)))
vlines = cv2.erode(dark, vk)
col_sum = vlines.sum(axis=0)  # per-column
hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, W//6), 1))
hlines = cv2.erode(dark, hk)
row_sum = hlines.sum(axis=0)  # wrong axis below; recompute
row_sum = hlines.sum(axis=1)

cols = np.where(col_sum > 0)[0]
rows = np.where(row_sum > 0)[0]
print("\nvertical-line columns (x):", cols[:50] if len(cols) else "none")
print("horizontal-line rows (y):", rows[:50] if len(rows) else "none")
if len(cols):
    print("left spine x ~", cols.min(), " right frame x ~", cols.max())
if len(rows):
    print("top frame y ~", rows.min(), " bottom spine y ~", rows.max())
