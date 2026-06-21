"""Axis calibration: read tick labels with OCR and fit robust linear
pixel -> data transforms for the X (Study Days) and Y (Tumor Volume) axes.

The axes are linear with evenly-spaced ticks, so two anchors suffice; we read
many and fit with RANSAC (largest consensus set of collinear labels, then a
least-squares refit on the inliers), which ignores the occasional OCR misread
(e.g. "550.00" -> "250.00") as long as most labels are correct.
"""
import re
import cv2
import numpy as np
import pytesseract

from markers import find_spines

_CFG = "--psm 11 -c tessedit_char_whitelist=0123456789,.-"
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _parse(text):
    """Extract the numeric value from an OCR token, tolerating adjacent junk
    such as a tick mark read as a trailing '-' (e.g. '2,400.00-')."""
    matches = _NUM.findall(text.replace(" ", ""))
    if not matches:
        return None
    best = max(matches, key=len)         # the full number, not a fragment
    try:
        return float(best.replace(",", ""))
    except ValueError:
        return None


def _ocr_tokens(crop, scale=4):
    """OCR a strip; return list of (text, cx, cy) in original-strip pixels.

    scale=4 is the default used by the primary path. Pass scale=None for the
    ADAPTIVE factor used by the fallback: it caps the image handed to tesseract
    at ~2400px on its longest side, since over-enlarging a large strip degrades
    image_to_data and can lose every label."""
    h, w = crop.shape[:2]
    if scale is None:
        scale = float(np.clip(2400.0 / max(1, max(h, w)), 1.5, 4.0))
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Feed grayscale (tesseract does its own thresholding). Global OTSU erased
    # thin strokes -- notably leading '1' digits ('1,600'->'600'), the single
    # biggest source of axis-label misreads.
    d = pytesseract.image_to_data(g, config=_CFG, output_type=pytesseract.Output.DICT)
    out = []
    for i in range(len(d["text"])):
        if d["text"][i].strip() and int(d["conf"][i]) > 20:
            cx = (d["left"][i] + d["width"][i] / 2) / scale
            cy = (d["top"][i] + d["height"][i] / 2) / scale
            out.append((d["text"][i], cx, cy))
    return out


def _fit(pixels, values):
    """Robust linear fit value = slope*pixel + intercept via RANSAC: the line
    through some pair of labels that agrees with the most other labels (within a
    quarter tick step) wins, then refit by least squares on those inliers.

    RANSAC is used instead of Theil-Sen because dropped-digit OCR misreads
    (e.g. '165' -> '65') can corrupt a majority of pairwise slopes and break
    Theil-Sen's median, whereas a clean run of collinear labels still forms the
    largest consensus set. For clean calibrations every label is an inlier, so
    the result matches a plain least-squares fit. Returns
    (slope, intercept, inliers, rms)."""
    px = np.asarray(pixels, float)
    vv = np.asarray(values, float)
    n = len(px)
    if n < 2 or np.ptp(px) == 0:
        return None
    tol = max(1e-6, 0.25 * _expected_step(vv))
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            if px[j] == px[i]:
                continue
            s = (vv[j] - vv[i]) / (px[j] - px[i])
            b = vv[i] - s * px[i]
            inl = np.abs(vv - (s * px + b)) <= tol
            if best is None or inl.sum() > best.sum():
                best = inl
    if best is None or best.sum() < 2 or np.ptp(px[best]) == 0:
        return None
    slope, intercept = np.polyfit(px[best], vv[best], 1)
    resid = np.abs(vv[best] - (slope * px[best] + intercept))
    return float(slope), float(intercept), int(best.sum()), float(np.sqrt(np.mean(resid ** 2)))


def _expected_step(values):
    u = np.unique(np.round(values, 6))
    if len(u) < 2:
        return 1.0
    return float(np.median(np.diff(np.sort(u))))


def _collect(crop, axis, off_x, off_y, scale=4):
    """OCR a region; return (pixels, values) for one axis. `axis` selects which
    label coordinate (cx for x, cy for y) is the pixel position."""
    pix, val = [], []
    for text, cx, cy in _ocr_tokens(crop, scale):
        v = _parse(text)
        if v is not None:
            pix.append(cx + off_x if axis == "x" else cy + off_y)
            val.append(v)
    return pix, val


def calibrate(bgr):
    """Return dict with x and y transforms: {'x':(slope,intercept,n,rms),
    'y':(...), 'box':(l,t,r,b)}. slope/intercept map pixel -> data value."""
    H, W = bgr.shape[:2]
    left, top, right, bottom = find_spines(bgr)

    # PRIMARY: Y labels left of the y-spine, X labels below the x-spine. Trim a
    # few px before the y-spine so tick marks (read as '-') don't corrupt tokens;
    # skip the tick row just under the x-spine. The X strip extends LEFT of the
    # spine because the first x-tick label is centred on the origin, so its
    # leading digit sits at x<left and would otherwise be clipped (e.g. '230'
    # read as '30'). Below the spine there are no y-labels, so this is safe.
    ytop = max(0, top - 4)
    xleft = max(0, left - max(22, int(0.03 * W)))
    ypix, yval = _collect(bgr[ytop:bottom + 6, 0:max(1, left - 4)], "y", 0, ytop)
    xpix, xval = _collect(bgr[bottom + 4:H, xleft:right + 4], "x", xleft, 0)
    yfit, xfit = _fit(ypix, yval), _fit(xpix, xval)

    # FALLBACK (only when the primary fails): some charts carry a title or a
    # rotated axis name that misleads spine detection, so the strip lands in the
    # wrong place. OCR a generous label band at the adaptive scale; the robust
    # fit discards any stray numbers (e.g. an ID in the title). This never runs
    # when the primary already succeeds, so it cannot regress working charts.
    if yfit is None:
        y0, x1 = max(0, top - 4), max(int(0.20 * W), left)
        p, v = _collect(bgr[y0:bottom + 6, 0:x1], "y", 0, y0, scale=None)
        f = _fit(p, v)
        if f:
            yfit, ypix, yval = f, p, v
    if xfit is None:
        y0 = min(bottom, int(0.80 * H))
        p, v = _collect(bgr[y0:H, left:right + 4], "x", left, 0, scale=None)
        f = _fit(p, v)
        if f:
            xfit, xpix, xval = f, p, v

    return {
        "box": (left, top, right, bottom),
        "y": yfit, "x": xfit,
        "y_pts": list(zip(ypix, yval)),
        "x_pts": list(zip(xpix, xval)),
    }


def pixel_to_data(cal, px, py):
    xs, xi, *_ = cal["x"]
    ys, yi, *_ = cal["y"]
    return xs * px + xi, ys * py + yi
