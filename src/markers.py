"""Robust data-marker detection for tumor growth-curve line charts.

Works across series colors (orange, red, black, olive/desaturated) and marker
shapes (circle, square, triangle). Pipeline:

  1. Estimate the smooth cream background via a large morphological closing of
     the brightness channel; `darkness = bg - V` is a top-hat that isolates
     anything drawn on top of the background, independent of the gradient.
  2. Series mask = significantly-darker-than-background OR high-saturation.
     Catches dark (black), desaturated (olive), and saturated (orange/red) series
     while rejecting the faint gridlines.
  3. Remove the plot frame, axis spines, and full-width gridlines: they are long
     straight full-span lines; data markers are compact blobs and survive.
  4. Find the left spine and bottom spine to drop the axis tick labels.
  5. Isolate markers from the thin connecting line with a distance transform
     (markers are locally thick; the line is thin), separating touching markers
     via local maxima of the distance transform.
"""
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import h_maxima


# ---------------------------------------------------------------- segmentation
def _background_darkness(V):
    """Top-hat: how much darker each pixel is than the local cream background."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    bg = cv2.morphologyEx(V, cv2.MORPH_CLOSE, k)
    return cv2.subtract(bg, V)  # >=0, large where something dark sits on bg


def _plot_bbox(S, V):
    """Bounding box of the cream/tan plotting area (distinct from the white
    page around it), used only as a fallback when spine detection is degenerate.
    Returns (l, t, r, b) or None."""
    cream = ((V > 205) & (S > 10) & (S < 90)).astype(np.uint8)
    cream = cv2.morphologyEx(cream, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cream, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
    w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    if w < 0.4 * V.shape[1] or h < 0.4 * V.shape[0]:
        return None
    return int(x), int(y), int(x + w - 1), int(y + h - 1)


_RECOVER_CUT_TOP = True   # extend the box top to the cut edge on top-clipped plots


def find_spines(bgr):
    """Return (left, top, right, bottom) of the plot box from the strong long
    dark axis spines."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    H, W = V.shape
    dk = _background_darkness(V)
    strong = (dk > 20).astype(np.uint8) * 255
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H // 4))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 4, 1))
    vcols = np.where(cv2.erode(strong, vk).sum(axis=0) > 0)[0]
    hrows = np.where(cv2.erode(strong, hk).sum(axis=1) > 0)[0]
    # ignore image-border artefacts
    vcols = vcols[(vcols > 4) & (vcols < W - 4)]
    hrows = hrows[(hrows > 4) & (hrows < H - 4)]

    # Constrain spine candidates to the cream PLOTTING region. A page/figure
    # border line OUTSIDE the plot (e.g. a rule below the "Study Days" title) is a
    # long dark horizontal line that would otherwise win `hrows.max()` and be
    # taken as the x-axis -- dragging the box, and the label-masking band, down
    # over the tick labels (which then detect as a spurious row of markers). The
    # cream bbox bounds the real plot, so drop any line beyond it (small margin).
    plot = _plot_bbox(S, V)
    if plot:
        px0, py0, px1, py1 = plot
        mw, mh = max(10, int(0.03 * W)), max(10, int(0.03 * H))
        hin = hrows[(hrows >= py0 - mh) & (hrows <= py1 + mh)]
        vin = vcols[(vcols >= px0 - mw) & (vcols <= px1 + mw)]
        if len(hin):
            hrows = hin
        if len(vin):
            vcols = vin

    left = int(vcols.min()) if len(vcols) else int(0.09 * W)
    bottom = int(hrows.max()) if len(hrows) else int(0.92 * H)
    right = int(vcols.max()) if len(vcols) and vcols.max() > left + 50 else W - 2
    top = int(hrows.min()) if len(hrows) and hrows.min() < bottom - 50 else 2
    # Guarded fallback: if a title or rotated axis name fooled the spines into a
    # degenerate box, fall back to the cream plot-area bounding box. Only fires
    # on degenerate boxes, so non-degenerate (working) charts are unaffected.
    if right - left < 0.4 * W or bottom - top < 0.4 * H:
        if plot:
            left, top, right, bottom = plot

    # If the plot is CUT OFF at the top image edge (the cream interior extends
    # above the detected top, with no white margin or frame above it), the topmost
    # strong line is the first GRIDLINE -- one tick BELOW the true top -- so the box
    # ends a tick early and clips the highest data point (which then either goes
    # undetected or shows up as a stub at the box edge). Walk up while the row is
    # still plot interior (cream) to recover the true top (the cut edge). Charts
    # with a white margin / top frame stop immediately and are unaffected.
    if _RECOVER_CUT_TOP:
        cream = (V > 205) & (S > 10) & (S < 90)
        xa, xb = left + 5, max(left + 6, right - 5)
        while top - 1 >= 0 and cream[top - 1, xa:xb].mean() > 0.5:
            top -= 1
    return left, top, right, bottom


def series_mask(bgr):
    """Binary mask (uint8 0/255) of the data series, with axis tick labels
    excluded. The plot frame, axis spines and gridlines are intentionally LEFT
    in: they are thin and are rejected downstream by the distance transform,
    whereas erasing them would also destroy data markers that sit on the y=0
    baseline (a common case for early/zero tumour volumes). Returns spine box."""
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    left, top, right, bottom = find_spines(bgr)

    dk = _background_darkness(V)
    mask = ((dk > 30) | (S > 70)).astype(np.uint8) * 255

    # Keep only the plot interior so axis tick labels (left of the y-spine,
    # below the x-spine) don't become spurious blobs. Keep one marker-radius
    # below the bottom spine so value-0 markers centred on it survive; the thin
    # tick marks there are rejected by the distance transform. The LEFT bound is
    # the spine itself (not left+3): a growth curve's day-0 marker sits ON the
    # spine, so the 3 columns just right of it are the THICKEST part of that
    # half-clipped marker -- erasing them was dropping it. Tick labels are at
    # x < left, so they stay excluded; the thin spine is rejected downstream.
    keep = np.zeros((H, W), np.uint8)
    keep[top + 2:bottom + 8, left:right + 2] = 1
    mask = cv2.bitwise_and(mask, keep * 255)
    return mask, (left, top, right, bottom)


# ----------------------------------------------------------------- detection
def detect_markers(bgr, debug=False):
    """Return list of (cx, cy) marker centers in pixel coords + diagnostics.

    A marker is a local maximum of the distance transform that rises clearly
    above the ~1px connecting line. The threshold is fixed just above the line
    half-width (not relative to the largest marker) so small/thin markers
    (e.g. triangle apexes, baseline-occluded points) are not lost. Touching
    markers are resolved by adaptive non-maximum suppression whose radius is
    derived from the typical marker size in this image."""
    mask, box = series_mask(bgr)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dmax = float(dist.max())

    # Estimate the connecting-line half-width from the bulk of the distance
    # values (the line dominates pixel count; its half-width varies per chart:
    # ~1px on thin charts, ~2px on bold ones).
    d = dist[dist > 0.5]
    lw = float(np.sort(d)[int(0.6 * d.size)]) if d.size else 1.0

    # PROMINENCE-based peaks (h-maxima): a marker is a regional maximum of the
    # distance transform rising >= h above its surroundings.
    #
    #  * The flat line RIDGE between two markers ascends toward them, so it is
    #    never a regional maximum -- no false peaks along thick/bold lines.
    #  * Line rasterisation makes small "staircase" bumps on steep segments with
    #    prominence ~1.2px; OVERLAPPING markers keep a prominence ~2px (their
    #    shared saddle still dips ~2px below each centre). h=1.5 sits in that
    #    gap: it rejects the staircase yet still SPLITS overlapping markers
    #    (which a larger h would merge). Each regional-maximum blob is one
    #    marker; its dist-weighted centroid is the centre.
    h = 1.5
    prom = h_maxima(dist, h) & (dist >= max(2.1, lw + 0.8))
    lbl, n = ndimage.label(prom)
    if n == 0:
        return [], dict(box=box, dmax=dmax, lw=lw, h=h, mask=mask, dist=dist)
    centers = ndimage.center_of_mass(dist, lbl, range(1, n + 1))  # (y, x)
    pts = [(float(x), float(y)) for (y, x) in centers]

    # Merge duplicate seeds: one marker can yield two regional maxima when its
    # flat top is split by noise. Real (even overlapping) markers sit >= a
    # marker radius apart, so merging centres closer than ~0.6*radius removes
    # duplicates without collapsing distinct markers.
    mr = float(np.median(dist[prom > 0])) if prom.any() else 3.0
    merge = max(2.0, 0.6 * mr)
    chosen = _merge_close(pts, merge)

    # A marker clipped by the right plot edge (the final study-day point, often
    # the peak volume) is recovered because the mask now includes the frame
    # column (see series_mask), giving the half-symbol enough body for the
    # prominence test. The steep connecting line running along the edge can,
    # however, leave a second weaker peak there. A curve exits the right edge at
    # most once, so within the edge zone keep only the strongest marker.
    chosen = _dedupe_right_edge(chosen, box, mr, dist)

    # A marker on the LEFT SPINE (the day-0 reading, present in every growth
    # curve) is bisected by the spine -- and, when its value is ~0, quartered by
    # the spine AND the baseline. The surviving fragment fuses with the rising
    # connecting line, so it often fails the generic prominence test. Using the
    # domain prior that this marker always exists, recover it from the spine band.
    chosen = _recover_left_spine(chosen, box, mr, dist, lw)

    # Markers on the y=0 BASELINE (early/zero tumour volumes) are bisected by the
    # bottom spine; the surviving top half fuses with the horizontal spine + the
    # near-horizontal connecting line, so it rarely forms a prominent regional
    # maximum (the dominant miss mode). Recover them with an INDEPENDENT signal:
    # the series is coloured/dark while the spine is mid-grey, so a colour mask
    # drops the spine; a vertical opening drops the thin connecting line; what
    # survives near the baseline as a thick, roundish blob is a marker.
    #
    # SCOPE: recovery is restricted to the ORIGIN corner (x ~= x_min). Extending
    # it across the whole baseline raises synthetic recall ~2 pts but spawns false
    # positives at curve-LIFTOFF junctions on real images -- which we cannot
    # measure (no real marker ground truth), so we will not pay an unbounded
    # precision cost for a tiny recall gain. The interior y=0 baseline therefore
    # stays at its honest floor; only the always-present day-0 point is recovered.
    chosen = _recover_baseline(bgr, chosen, box, mr, lw, dist)

    diag = dict(box=box, dmax=dmax, lw=lw, h=h, mr=mr, n=len(chosen), mask=mask, dist=dist)
    return chosen, diag


def _recover_baseline(bgr, markers, box, mr, lw, dist):
    """Recover the y=0 day-0 marker at the ORIGIN corner that the generic
    prominence test missed, without inflating false positives. Three INDEPENDENT
    gates must all agree, each rejecting one confounder: (1) a colour mask
    (saturated OR clearly darker than the ~grey axis) removes the spine; (2) a
    vertical opening removes the thin horizontal connecting line, keeping only
    blobs as tall as a marker; (3) a distance-transform peak confirms genuine
    thickness, and a roundness check rejects flat line fragments. A bare
    spine/line therefore recovers nothing. The search is confined to the origin
    corner (see caller: extending across the baseline adds unmeasurable real FPs)."""
    l, t, r, b = box
    H, W = dist.shape
    corner_x = l + int(round(2.5 * mr))                  # FP-safe origin-corner zone
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    dk = _background_darkness(V)
    series = ((S > 60) | (dk > 55)).astype(np.uint8) * 255   # series ink, not grey spine
    ksz = max(3, int(round(1.6 * lw)) + 1)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksz))
    opened = cv2.morphologyEx(series, cv2.MORPH_OPEN, vk)     # drop thin horizontal line
    band = np.zeros_like(opened)
    bt = max(t, int(b - 2.4 * mr))
    cx1 = min(W, max(l + 3, corner_x))                   # confine to the origin corner
    band[bt:min(H, b + 2), l + 2:cx1] = opened[bt:min(H, b + 2), l + 2:cx1]
    n, lbl, stats, cent = cv2.connectedComponentsWithStats((band > 0).astype(np.uint8), 8)
    hmin = max(3, 0.55 * mr); wmin = max(2, 0.4 * mr); wmax = 2.6 * mr
    dgate = max(2.0, lw + 1.0)            # tight: a real (clipped) marker is thick
    out = list(markers)
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        cx, cy = cent[i]
        if hh < hmin or w < wmin or w > wmax or area < 0.6 * mr * mr:
            continue
        if (y + hh) < b - mr:                # blob must reach the baseline
            continue
        if w > 1.3 * hh:                     # flat -> line fragment, not a marker
            continue
        sub = dist[y:y + hh, x:x + w]
        if sub.size == 0 or sub.max() < dgate:   # not genuinely thick
            continue
        # Reject liftoff junctions: where the curve RISES off the baseline, the
        # marker + steep line spawn extra blobs. An isolated baseline marker has
        # only the thin near-horizontal line above it; a liftoff has the rising
        # curve, i.e. series ink well above the band directly overhead. A fixed
        # dedupe radius (independent of the tiny mr on low-res charts) then keeps
        # one detection per junction.
        dr = max(8.0, 2.0 * mr)
        if any(abs(cx - mx) <= dr and abs(cy - my) <= 2.4 * mr for mx, my in out):
            continue                          # already detected / clustered here
        out.append((float(cx), float(min(cy + 0.2 * mr, b))))
    return out


def _recover_left_spine(markers, box, mr, dist, lw):
    """Recover the day-0 marker that sits ON the left spine when the generic
    detector missed it. Domain prior: a growth curve always has a leftmost marker
    at x = x_min. If none was detected in a narrow band just right of the spine,
    take the thickest distance-transform blob there -- but only if it clears a
    relaxed threshold (so a bare spine/line, with no marker, recovers nothing)."""
    l, t, r, b = box
    bw = int(np.ceil(1.4 * mr)) + 2
    band_x = l + bw
    if any(l - 1 <= x <= band_x for x, y in markers):
        return markers                       # a spine marker is already present
    H, W = dist.shape
    y0b, y1b = t, min(H, b + 6)
    sub = dist[y0b:y1b, l:min(W, band_x + 1)]
    if sub.size == 0:
        return markers
    fy, fx = np.unravel_index(int(np.argmax(sub)), sub.shape)
    peak = float(sub[fy, fx])
    # the thin spine alone reads ~0.5-1px; the connecting line ~lw. A real
    # (clipped) marker is a thicker bump on top of that. Require clearance.
    if peak < max(1.8, lw + 0.4):
        return markers
    # dist-weighted centroid of the marker blob within the band; snap x to the
    # spine because the prior fixes its data-x at x_min (the surviving fragment's
    # own centroid would bias a few px to the right).
    yy = fy + y0b
    ys0 = int(round(max(y0b, yy - 2 * mr)))
    ys1 = int(round(min(y1b, yy + 2 * mr + 1)))
    blob = dist[ys0:ys1, l:min(W, band_x + 1)]
    m = blob >= 0.5 * peak
    ys, xs = np.nonzero(m)
    w = blob[ys, xs]
    cy = float((ys * w).sum() / w.sum()) + ys0
    return markers + [(float(l), cy)]


def _dedupe_right_edge(markers, box, mr, dist):
    l, t, r, b = box
    edge_x = r - 1.2 * mr
    edge = [(x, y) for x, y in markers if x >= edge_x]
    if len(edge) <= 1:
        return markers
    H, W = dist.shape
    strength = lambda p: dist[min(H - 1, max(0, int(p[1]))), min(W - 1, max(0, int(p[0])))]
    keep = max(edge, key=strength)
    return [m for m in markers if m[0] < edge_x] + [keep]


def _merge_close(pts, thresh):
    """Greedily merge points closer than `thresh` (averaging their positions)."""
    pts = sorted(pts)
    out = []
    for p in pts:
        for i, q in enumerate(out):
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= thresh * thresh:
                out[i] = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                break
        else:
            out.append(p)
    return out


def draw_overlay(bgr, markers, box):
    ov = bgr.copy()
    for cx, cy in markers:
        p = (int(round(cx)), int(round(cy)))
        cv2.circle(ov, p, 7, (0, 0, 255), 2)
        cv2.drawMarker(ov, p, (255, 0, 0), cv2.MARKER_CROSS, 6, 1)
    cv2.rectangle(ov, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    return ov
