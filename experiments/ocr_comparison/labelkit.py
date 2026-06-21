"""Synthetic axis-label rendering + real-strip segmentation.

Shared by the synthetic dataset generator, the GUI labeler, and the optimizer.
Everything operates on a single tick-LABEL crop (one number on white), which is
both what the optimizer scores and what the GUI shows.

Grounded in the real data (see tools/analyze_misreads.py findings):
  * labels are clean black sans-serif (Arial-like) on white;
  * Y labels are formatted "{:,.2f}" (e.g. "1,200.00"), often with a trailing
    tick dash; X labels are plain integers;
  * source images are mostly PNG (clean) with a few JPEG, at widely varying
    resolution -> glyph height ~11-30 px;
  * real OCR failures are digit-insertion (900.00->9300.00) and decimal/comma
    confusion -> the synthetic degradations target exactly these regimes.
"""
import io, os, random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
FONTS = [f for f in FONTS if os.path.exists(f)]
_SS = 4  # supersample factor for antialiasing that mimics chart rendering


def format_label(value, axis):
    """Render a value the way the real charts do."""
    if axis == "y":
        return f"{value:,.2f}"          # 1200 -> "1,200.00"
    return str(int(round(value)))         # x: plain integer


def render_label(text, font_px, font_path=None, tick=False, ink=10, paper=255,
                 pad=6, rng=random):
    """Render `text` as a tight black-on-white crop at ~font_px glyph height.
    Returns a BGR uint8 image. Supersampled then downsized for realistic AA."""
    font_path = font_path or rng.choice(FONTS)
    big = max(8, int(font_px * _SS))
    font = ImageFont.truetype(font_path, big)
    # real y-ticks glue a small dash/dot a short gap to the right of the number;
    # vary the form so the optimizer sees the real "tick-junk" regime.
    s = text + (rng.choice([" -", " ·", " -", "  -"]) if tick else "")
    # measure
    tmp = ImageDraw.Draw(Image.new("L", (10, 10)))
    l, t, r, b = tmp.textbbox((0, 0), s, font=font)
    w, h = r - l, b - t
    P = pad * _SS
    img = Image.new("L", (w + 2 * P, h + 2 * P), paper)
    d = ImageDraw.Draw(img)
    d.text((P - l, P - t), s, font=font, fill=ink)
    arr = np.array(img)
    # downsize by supersample factor -> smooth edges
    arr = cv2.resize(arr, (max(1, arr.shape[1] // _SS), max(1, arr.shape[0] // _SS)),
                     interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def degrade(bgr, blur=0.0, jpeg=None, noise=0.0, rng=random):
    """Apply realistic capture degradations to a rendered crop."""
    out = bgr
    if blur > 0:
        k = max(1, int(blur * 3) | 1)
        out = cv2.GaussianBlur(out, (k, k), blur)
    if noise > 0:
        n = rng.gauss
        g = np.array([[n(0, noise) for _ in range(out.shape[1])] for _ in range(out.shape[0])])
        out = np.clip(out.astype(float) + g[..., None], 0, 255).astype(np.uint8)
    if jpeg is not None:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def sample(axis, rng=random):
    """Generate one (crop_bgr, truth_text) synthetic label, styled like the data."""
    if axis == "y":
        step = rng.choice([50, 100, 200, 250, 300, 500])
        value = step * rng.randint(0, 20)
    else:
        value = rng.choice([rng.randint(0, 60), rng.randint(0, 450)])
    text = format_label(value, axis)
    font_px = rng.randint(11, 30)
    tick = axis == "y" and rng.random() < 0.5
    crop = render_label(text, font_px, tick=tick,
                        ink=rng.randint(0, 45), paper=rng.randint(248, 255), rng=rng)
    # mostly-PNG dataset: clean often, mild degradation sometimes
    crop = degrade(crop,
                   blur=rng.choice([0, 0, 0.4, 0.7]),
                   jpeg=rng.choice([None, None, None, 75, 90]),
                   noise=rng.choice([0, 0, 2.0]), rng=rng)
    return crop, text


def _runs(on, min_gap):
    """Index ranges of consecutive True, bridging gaps shorter than min_gap."""
    bands, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or (j + min_gap < n and on[j:j + min_gap].any())):
                j += 1
            bands.append((i, j)); i = j
        else:
            i += 1
    return bands


def segment_strip(strip_bgr, axis, pad=3):
    """Cut a real axis strip into individual label crops via projection profile.
    Returns list of (crop_bgr, cx, cy) where (cx,cy) is the crop centre in the
    strip's pixel coordinates. Detection-free: relies on the blank gaps between
    labels, so it does not depend on the OCR we are trying to evaluate.

    Y axis: labels are stacked vertically; the rotated title/spine are vertical
    structures (high column-coverage) and are masked first, then each row band is
    one label. X axis: labels share one row with the axis title below; we keep
    the row band nearest the axis (the labels) and split it into numbers with a
    gap sized to the glyph height, so inter-digit gaps don't fragment a number."""
    g = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
    ink = (g < 128).astype(np.uint8)             # dark text on light paper
    H, W = g.shape
    out = []

    if axis == "y":
        if H > 20:
            ink[:, ink.mean(axis=0) > 0.45] = 0   # drop vertical title / spine
        for a, b in _runs(ink.sum(axis=1) > 0, min_gap=2):
            cols = np.where(ink[a:b].sum(axis=0) > 0)[0]
            if len(cols) == 0:
                continue
            y0, y1 = max(0, a - pad), min(H, b + pad)
            x0, x1 = max(0, cols[0] - pad), min(W, cols[-1] + pad)
            crop = strip_bgr[y0:y1, x0:x1]
            if crop.shape[0] >= 5 and crop.shape[1] >= 5:
                out.append((crop, (x0 + x1) / 2, (y0 + y1) / 2))
        return out

    # X axis: rows below the spine are, top to bottom, tick marks -> number
    # labels -> axis title ("Study Days"). The number row spans the whole axis
    # with many digits, so it carries far more ink than the thin ticks or the
    # short title; pick the row band with the most ink.
    row_bands = _runs(ink.sum(axis=1) > 0, min_gap=2)
    if not row_bands:
        return out
    a, b = max(row_bands, key=lambda ab: int(ink[ab[0]:ab[1]].sum()))
    band_h = b - a
    y0, y1 = max(0, a - pad), min(H, b + pad)
    # bridge inter-digit gaps (~fraction of glyph height) but not inter-number
    # gaps; clamp so it works across the wide resolution range in the dataset.
    gap = max(3, int(0.45 * band_h))
    col_on = ink[a:b].sum(axis=0) > 0
    for c, d in _runs(col_on, min_gap=gap):
        x0, x1 = max(0, c - pad), min(W, d + pad)
        crop = strip_bgr[y0:y1, x0:x1]
        if crop.shape[0] >= 5 and crop.shape[1] >= 5:
            out.append((crop, (x0 + x1) / 2, (y0 + y1) / 2))
    return out
