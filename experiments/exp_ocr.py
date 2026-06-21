"""Test OCR of axis-label strips."""
import cv2, numpy as np, sys, re
sys.path.insert(0, "src")
from markers import find_spines
import pytesseract

f = sys.argv[1] if len(sys.argv) > 1 else "Growth_Curves_NCI_BRCA/171881-019-R_P0.jpeg"
bgr = cv2.imread(f); H, W = bgr.shape[:2]
V = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
left, top, right, bottom = find_spines(V)
print(f"{f}  box=({left},{top},{right},{bottom})  WxH={W}x{H}")

CFG = "--psm 11 -c tessedit_char_whitelist=0123456789,.-"

def ocr(crop, scale=4):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    d = pytesseract.image_to_data(g, config=CFG, output_type=pytesseract.Output.DICT)
    out = []
    for i in range(len(d["text"])):
        t = d["text"][i].strip()
        if t and int(d["conf"][i]) > 20:
            cx = (d["left"][i] + d["width"][i] / 2) / scale
            cy = (d["top"][i] + d["height"][i] / 2) / scale
            out.append((t, cx, cy, int(d["conf"][i])))
    return out

# Y labels: strip left of the left spine
ystrip = bgr[top:bottom + 5, 0:left]
print("\n-- Y labels (text, x, y_in_strip, conf) --")
for t, cx, cy, c in ocr(ystrip):
    print(f"  {t!r:12s} y_pixel={cy+top:6.1f} conf={c}")

# X labels: strip below the bottom spine
xstrip = bgr[bottom:H, left:right]
print("\n-- X labels (text, x_in_strip, y, conf) --")
for t, cx, cy, c in ocr(xstrip):
    print(f"  {t!r:8s} x_pixel={cx+left:6.1f} conf={c}")
