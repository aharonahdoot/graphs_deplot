"""Segment series + isolate markers via distance transform. Save overlay."""
import cv2, numpy as np, sys, os

path = sys.argv[1] if len(sys.argv) > 1 else "Growth_Curves_NCI_BRCA/171881-019-R_P0.jpeg"
name = os.path.splitext(os.path.basename(path))[0]
bgr = cv2.imread(path)
H, W = bgr.shape[:2]
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
S = hsv[:,:,1]; V = hsv[:,:,2]

# --- plot box via dark spines ---
dark = (gray < 130).astype(np.uint8)*255
vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H//6))
hk = cv2.getStructuringElement(cv2.MORPH_RECT, (W//6, 1))
cols = np.where(cv2.erode(dark, vk).sum(axis=0) > 0)[0]
rows = np.where(cv2.erode(dark, hk).sum(axis=1) > 0)[0]
left = cols.min() if len(cols) else int(0.08*W)
bottom = rows.max() if len(rows) else int(0.92*H)
right = cols.max() if len(cols) and cols.max() > left+50 else W-3
top = rows.min() if len(rows) and rows.min() < bottom-50 else 3
print(f"plot box: x[{left},{right}] y[{top},{bottom}]")

# --- series mask ---
series = ((S > 60) | (V < 110)).astype(np.uint8)
# restrict to plot interior (exclude spines)
box = np.zeros((H,W), np.uint8)
box[top+2:bottom-2, left+2:right-2] = 1
series &= box
series *= 255
print("series pixel count:", int((series>0).sum()))

# --- distance transform to isolate markers ---
dist = cv2.distanceTransform(series, cv2.DIST_L2, 5)
print(f"dist max={dist.max():.2f}  (marker core radius)")
# threshold relative to max marker thickness; line is thin
T = max(2.2, 0.45*dist.max())
cores = (dist >= T).astype(np.uint8)
n, lbl, stats, cent = cv2.connectedComponentsWithStats(cores, connectivity=8)
markers = []
for i in range(1, n):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 1: continue
    cx, cy = cent[i]
    markers.append((cx, cy, area))
print(f"T={T:.2f}  markers found: {len(markers)}")

# --- overlay ---
ov = bgr.copy()
for cx, cy, area in markers:
    cv2.circle(ov, (int(round(cx)), int(round(cy))), 7, (0,0,255), 2)
    cv2.drawMarker(ov, (int(round(cx)), int(round(cy))), (255,0,0), cv2.MARKER_CROSS, 6, 1)
cv2.rectangle(ov, (left,top), (right,bottom), (0,255,0), 1)
out = f"experiments/out/{name}_overlay.png"
cv2.imwrite(out, ov)
cv2.imwrite(f"experiments/out/{name}_mask.png", series)
print("wrote", out)
