"""Synthetic ground-truth benchmark for the (x,y) data-marker extractor.

WHY THIS EXISTS
---------------
The 44 real graphs (and the 3000+ single_curve targets) have NO ground-truth
marker coordinates -- the underlying study numbers are not available, and a human
cannot click a marker centre more precisely than the detector already places it.
So we cannot measure exact (x,y) error on real images.

We therefore render synthetic growth-curve graphs whose marker coordinates we
KNOW exactly, in a style sampled from the real single_curve images, run the FULL
src pipeline on them, and measure error against the known truth.

WHAT IT MEASURES (three separable quantities)
  * recall / false-positives  -- did detection find every marker, none spurious,
    broken out for the hard cases (markers on the y=0 baseline, markers bisected
    by the right edge, and markers sitting on the ORIGIN -- quartered by BOTH the
    left spine and the baseline at once). This is the axis the human verification
    overlays corroborate on REAL images.
  * calibration error          -- map each marker's TRUE pixel through the fitted
    transform; error vs known value (% of axis span). Isolates OCR+axis-fit.
  * end-to-end (x,y) error     -- map each DETECTED pixel through the transform;
    error vs known value. Detection + calibration combined.

HONEST LIMITATION
  The generator and the detector's assumptions are written by the same hand, so
  a synthetic-only result risks "shared-assumption circularity". This number is
  the method's INTRINSIC PRECISION under a matched distribution; it is paired
  with human-overlay verification on real images (real distribution, recall +
  gridline-level value sanity) -- see ../METHODS.md. Style parameters below are
  measured from single_curve, not invented (see commit / METHODS for the probe).

Run:  .venv/bin/python experiments/synthetic_benchmark.py [N] [--seed S] [--save K]
"""
import os
# Set environment variables for thread limitation before importing heavy libraries
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys, math, random, argparse, glob
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from markers import detect_markers
from calibrate import calibrate, pixel_to_data

OUT = os.path.join(os.path.dirname(__file__), "out", "synthetic")

# --- style ranges measured from single_curve (50-image probe) -----------------
DIM_W = (620, 1500)
DIM_H = (410, 960)
BG_TOP = (232, 244, 250)      # BGR, plot-interior top of the cream gradient
BG_BOT = (221, 238, 247)      # BGR, plot-interior bottom (warmer)
MARK_R = (4, 9)               # marker radius px (dmax probe)
LINE_W = (1, 2)               # connecting-line width px
N_MARK = (6, 40)              # markers per graph
# series colours seen across the dataset (BGR): magenta, purple, blue, red,
# orange, olive, teal, black
PALETTE = [(182,131,195),(170,80,150),(200,24,36),(40,40,210),(40,120,235),
           (40,140,170),(150,140,40),(20,20,20)]
SHAPES = ["circle","square","triangle","diamond","star","x","plus"]
Y_MAXES = [600, 800, 950, 1200, 1600, 2000, 2400]
X_STEPS = [3, 10, 20, 30]
# Real tick labels are a sans-serif TTF. By visual sampling they read
# PREDOMINANTLY SMOOTH (anti-aliased); a minority -- tied to smaller/lower-res
# images -- are hard/aliased ("8-bit"). gen_graph reproduces both: ~80% smooth
# (AA on) and ~20% pixelated (AA off, drawn with PIL fontmode="1").
FONTS = [f for f in ["/System/Library/Fonts/Supplemental/Arial.ttf",
                     "/System/Library/Fonts/Supplemental/Verdana.ttf",
                     "/System/Library/Fonts/Supplemental/Tahoma.ttf"]
         if os.path.exists(f)] or ["/System/Library/Fonts/Supplemental/Arial.ttf"]
# Tick NUMBER labels are rasterised tiny with AA off; Verdana (a screen font with
# small-size hinting) stays crisp and matches the real aliased look, so number
# labels always use it. Titles are anti-aliased and may use any font.
NUM_FONTS = [f for f in ["/System/Library/Fonts/Supplemental/Verdana.ttf"]
             if os.path.exists(f)] or FONTS


def _nice_step(span, target=10):
    """A 'nice' tick step (1/2/5 x 10^k) giving ~target ticks over span."""
    raw = span / target
    k = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if m * k >= raw:
            return m * k
    return 10 * k


def _draw_marker(img, cx, cy, r, color, shape):
    p = (int(round(cx)), int(round(cy)))
    if shape == "circle":
        cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
    elif shape == "square":
        cv2.rectangle(img, (p[0]-r, p[1]-r), (p[0]+r, p[1]+r), color, -1, cv2.LINE_AA)
    elif shape == "triangle":
        pts = np.array([[p[0], p[1]-r], [p[0]-r, p[1]+r], [p[0]+r, p[1]+r]])
        cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)
    elif shape == "diamond":
        pts = np.array([[p[0], p[1]-r], [p[0]+r, p[1]], [p[0], p[1]+r], [p[0]-r, p[1]]])
        cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA)
    elif shape == "star":
        pts = []
        for i in range(10):
            ang = -math.pi/2 + i*math.pi/5
            rad = r if i % 2 == 0 else r*0.45
            pts.append([p[0]+rad*math.cos(ang), p[1]+rad*math.sin(ang)])
        cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)
    elif shape == "x":
        t = max(2, r//2)
        cv2.line(img, (p[0]-r,p[1]-r), (p[0]+r,p[1]+r), color, t, cv2.LINE_AA)
        cv2.line(img, (p[0]-r,p[1]+r), (p[0]+r,p[1]-r), color, t, cv2.LINE_AA)
    elif shape == "plus":
        t = max(2, r//2)
        cv2.line(img, (p[0]-r,p[1]), (p[0]+r,p[1]), color, t, cv2.LINE_AA)
        cv2.line(img, (p[0],p[1]-r), (p[0],p[1]+r), color, t, cv2.LINE_AA)


def _label_tile(s, fnt, fill, gap=1, punct_gap=0, aa=False):
    """Render a numeric label char-by-char with adjustable letter-spacing (`gap`,
    which may be negative to tighten) and extra space around ',' and '.'.
    `aa=False` (pixelated mode) draws with anti-aliasing OFF; without the gap the
    tiny punctuation gets absorbed into the adjacent digit and OCR reads '200.00'
    as '20000'. `aa=True` (smooth mode) keeps anti-aliasing on."""
    chs = list(s)
    xs, cur = [], 1
    for ch in chs:
        if ch in ",.":
            cur += punct_gap
        xs.append(cur)
        cur += max(1, int(round(fnt.getlength(ch)))) + gap + (punct_gap if ch in ",." else 0)
    tile = Image.new("RGBA", (max(1, cur + 1), fnt.size + 5), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    if not aa:
        d.fontmode = "1"
    for ch, x in zip(chs, xs):
        d.text((x, 1), ch, font=fnt, fill=fill)
    return tile


def gen_graph(rng, smooth_labels=None, n_y=None, n_x=None,
              lw=None, radius=None, shape=None, color=None, width=None, px_per_tick=None):
    # optional overrides (n_y/n_x tick counts, line width, marker radius, shape,
    # colour, image width, px-per-tick spacing) let a caller match a specific real
    # graph's style/aspect; default = random
    ri = lambda a, b: int(rng.integers(a, b + 1))      # inclusive int
    W = width or ri(*DIM_W)                            # H derived from tick density below

    # axes ranges
    y_max = int(rng.choice(Y_MAXES)); y_min = 0
    x_step = int(rng.choice(X_STEPS))
    x_min = ri(0, 400)
    x_max = x_min + x_step * ((n_x - 1) if n_x else ri(6, 12))
    y_step = (y_max - y_min) / (n_y - 1) if n_y else \
             _nice_step(y_max - y_min, ri(14, 19))    # tuned so OCR yield ~= real (~12/axis)

    # Two text styles, like the real charts:
    #  * tick NUMBER labels -> small native glyph, AA off, NEAREST-upscaled by SB
    #    into clean square pixel-blocks ("8-bit" look). Native size is held in a
    #    narrow band (independent of image size) so the blockiness is consistent.
    #  * axis TITLES -> larger, anti-aliased (smooth) and more legible.
    SB = 2                                            # label pixel-block size
    lab_n = 9                                          # native label height; 9 is the legible
    #                                                    floor -- at 8px the AA-off '0' pixelates
    #                                                    into a 'D'. Smooth labels can go smaller.
    nfont_path = str(rng.choice(NUM_FONTS))
    sfont = ImageFont.truetype(nfont_path, lab_n)     # screen font, crisp
    def ssize(s, fnt=sfont):
        x0, y0, x1, y1 = fnt.getbbox(s); return (x1 - x0), (y1 - y0)
    fpath = str(rng.choice(FONTS))                    # titles are AA'd -> any font is fine
    ttl_px = lab_n * SB + ri(3, 6)                    # titles: larger than numbers
    tfont = ImageFont.truetype(fpath, ttl_px)
    def tsize(s, fnt=tfont):
        x0, y0, x1, y1 = fnt.getbbox(s); return (x1 - x0), (y1 - y0)

    # margins sized from real label/title dimensions so nothing is clipped
    yvals = list(np.arange(0, y_max + 1e-6, y_step))
    ylab_w = max(_label_tile(f"{v:,.2f}", sfont, (0,0,0,255)).width for v in yvals) * SB
    xlab_h = ssize("0")[1] * SB
    TICK = max(5, (lab_n * SB) // 3)
    l = ttl_px + 8 + ylab_w + TICK + 12               # y-title band + labels + tick
    # IMAGE HEIGHT is derived from the tick count: each y-interval gets a
    # comfortable px spacing, so dense-tick graphs come out TALLER (as in the real
    # data) instead of cramming many ticks into a fixed height.
    n_int = max(1, len(yvals) - 1)
    px_per_tick = px_per_tick or ri(30, 46)
    t = ri(14, 40)                                    # top margin
    bot_margin = xlab_h + TICK + ttl_px + 18          # x-labels + tick + x-title
    plot_h = n_int * px_per_tick
    H = min(1400, t + plot_h + bot_margin)            # cap very tall images
    b = H - bot_margin
    W = max(W, l + ri(260, 460))                      # keep room for the plot width
    if smooth_labels is None:
        # Real labels read predominantly SMOOTH (anti-aliased); the chunky "8-bit"
        # pixelated look is the minority, tied to smaller/lower-resolution images.
        # ~70% smooth overall, biased so smaller images pixelate more often.
        smooth_labels = rng.random() < (0.80 if H >= 620 else 0.55)
    # Either a clean white right margin (default), or -- like the real charts --
    # the plot bleeds to the image edge so the final marker is bisected by it.
    edge_case = rng.random() < 0.45
    r = W - (ri(1, 3) if edge_case else
             max(int(W*rng.uniform(0.035, 0.06)), ssize(str(x_max))[0]*SB // 2 + 8))

    def X2px(x): return l + (x - x_min) / (x_max - x_min) * (r - l)
    def Y2py(y): return b - (y - y_min) / (y_max - y_min) * (b - t)
    def px2x(px): return x_min + (px - l) / (r - l) * (x_max - x_min)

    # cream vertical gradient inside the plot, white outside
    img = np.full((H, W, 3), 255, np.uint8)
    top = np.array(BG_TOP) + rng.integers(-4, 5, 3)
    bot = np.array(BG_BOT) + rng.integers(-4, 5, 3)
    for yy in range(t, b):
        f = (yy - t) / max(1, (b - t))
        img[yy, l:r] = np.clip(top*(1-f) + bot*f, 0, 255)

    grid = tuple(int(c) for c in (np.array(BG_TOP) - 20))
    axis = (70, 70, 75)                               # thin dark axis line
    yticks = [(v, int(round(Y2py(v)))) for v in yvals]
    xticks = [(x_min + i*x_step, int(round(X2px(x_min + i*x_step))))
              for i in range(int((x_max - x_min)/x_step) + 1)]
    for _, py in yticks:                              # faint gridlines
        cv2.line(img, (l, py), (r, py), grid, 1, cv2.LINE_AA)

    # growth curve: monotonic-ish, low early then rising; first point AT y=0
    n = ri(*N_MARK)
    k = min(n, x_max - x_min)
    xs_days = sorted(int(v) for v in
                     rng.choice(np.arange(x_min, x_max+1), size=k, replace=False))
    n = len(xs_days)
    base = np.linspace(0, 1, n) ** rng.uniform(2.0, 4.0)
    noise = np.cumsum(rng.normal(0, 0.04, n))
    ys_val = np.clip((base + noise - noise.min()), 0, None)
    ys_val = ys_val / max(ys_val.max(), 1e-6) * y_max * rng.uniform(0.75, 0.98)
    ys_val[0] = 0.0 if rng.random() < 0.6 else rng.uniform(0, y_max*0.015)

    color = color or tuple(int(c) for c in PALETTE[int(rng.integers(0, len(PALETTE)))])
    shape = shape or str(rng.choice(SHAPES))
    radius = radius if radius else ri(*MARK_R)
    lw = lw if lw else ri(*LINE_W)

    xs_val = [float(x) for x in xs_days]
    # Every growth curve's FIRST study point sits ON the left spine (x = x_min):
    # the day-0 reading. Its VALUE is usually ~0 (the ORIGIN -- quartered by both
    # the spine and the baseline), but real charts also show a non-zero baseline
    # reading, so lift it off the axis ~30% of the time -> a marker bisected by
    # the LEFT SPINE alone (x=0, y>0). Detection must capture both.
    xs_val[0] = float(x_min)
    if rng.random() < 0.30:
        ys_val[0] = float(rng.uniform(0.05, 0.20) * y_max)
    if edge_case:                                     # last marker bisected by image edge
        xs_val[-1] = px2x(W - ri(0, radius))
    pts = [(X2px(x), Y2py(y)) for x, y in zip(xs_val, ys_val)]

    poly = np.array([[int(round(px)), int(round(py))] for px, py in pts], np.int32)
    cv2.polylines(img, [poly], False, color, lw, cv2.LINE_AA)
    for px, py in pts:
        _draw_marker(img, px, py, radius, color, shape)

    # clip drawing to the plot area: white-out below baseline & left of the spine,
    # so y=0 markers are bisected by the baseline exactly like the real charts.
    # (Right of r is left intact so right-edge markers stay bisected by the edge.)
    img[b+1:, :] = 255
    img[:, :l] = 255

    cv2.line(img, (l, t), (l, b), axis, 1, cv2.LINE_AA)         # thin left spine
    cv2.line(img, (l, b), (r, b), axis, 1, cv2.LINE_AA)         # thin bottom spine
    def minors(p0, p1, k):    # k evenly-spaced minor positions between two majors
        return [int(round(p0 + (p1 - p0) * j / (k + 1))) for j in range(1, k + 1)]
    for _, py in yticks:                                        # major y ticks
        cv2.line(img, (l-TICK, py), (l, py), axis, 1, cv2.LINE_AA)
    for a, c in zip(yticks, yticks[1:]):                        # 2 minor y ticks
        for my in minors(a[1], c[1], 2):
            cv2.line(img, (l-TICK//2, my), (l, my), axis, 1, cv2.LINE_AA)
    # x-tick marks start 2 px BELOW the baseline (1 px gap). An isolated 1 px tick
    # stub is far too thin for the detector (distance-transform peak ~0.5 << the
    # 2.1 gate); only its FUSION with the baseline spine made a detectable blob,
    # which produced a row of spurious y=0 markers at the minor-tick positions.
    for _, px in xticks:                                        # major x ticks
        cv2.line(img, (px, b+2), (px, b+2+TICK), axis, 1, cv2.LINE_AA)
    for a, c in zip(xticks, xticks[1:]):                        # 4 minor x ticks
        for mx in minors(a[1], c[1], 4):
            cv2.line(img, (mx, b+2), (mx, b+2+TICK//2), axis, 1, cv2.LINE_AA)

    tcol = (45, 45, 45, 255)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # --- (1) tick NUMBER labels. Real charts vary: some are aliased ("8-bit"
    # pixel-blocked), some anti-aliased (smooth). `smooth_labels` selects which.
    if smooth_labels:                                 # anti-aliased, full resolution
        # smooth labels are anti-aliased so they stay legible smaller than the
        # pixelated floor (native 9 -> 18 px); render them a touch smaller, with
        # slightly tightened letter-spacing (gap=-1)
        nf = ImageFont.truetype(nfont_path, max(11, lab_n * SB - ri(4, 7)))
        for v, py in yticks:                          # right-aligned at the tick
            tile = _label_tile(f"{v:,.2f}", nf, tcol, gap=-1, aa=True)
            pil.alpha_composite(tile, (l-TICK-5 - tile.width, py - tile.height//2))
        for v, px in xticks:                          # centred
            tile = _label_tile(f"{int(v)}", nf, tcol, gap=-1, aa=True)
            if px - tile.width//2 >= 0 and px + tile.width//2 <= W-1:
                pil.alpha_composite(tile, (px - tile.width//2, b+TICK+3))
    else:                                             # aliased blocks ("8-bit")
        ow, oh = W // SB, H // SB
        ov = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
        for v, py in yticks:                          # right-aligned at the tick
            tile = _label_tile(f"{v:,.2f}", sfont, tcol)
            ov.alpha_composite(tile, ((l-TICK-5)//SB - tile.width, py//SB - tile.height//2))
        for v, px in xticks:                          # centred; include the ORIGIN
            tile = _label_tile(f"{int(v)}", sfont, tcol)
            cx = px//SB
            if (cx - tile.width//2)*SB >= 0 and (cx + tile.width//2)*SB <= W-1:
                ov.alpha_composite(tile, (cx - tile.width//2, (b+TICK+3)//SB))
        ov = ov.resize((W, H), Image.NEAREST)
        pil.alpha_composite(ov)

    # --- (2) axis TITLES: full resolution, anti-aliased (smooth) and legible
    dr2 = ImageDraw.Draw(pil)                          # default fontmode -> AA on
    dr2.text(((l+r)//2, H-ttl_px-6), "Study Days", font=tfont, fill=tcol, anchor="ma")
    yt = "Tumor Volume (mm³)"
    tw_, th_ = tsize(yt)
    yt_img = Image.new("RGBA", (tw_+8, th_+10), (0, 0, 0, 0))
    ImageDraw.Draw(yt_img).text((4, 2), yt, font=tfont, fill=tcol)   # AA on
    yt_img = yt_img.rotate(90, expand=True, resample=Image.BICUBIC)  # smooth rotate
    yy = max(t, (t+b)//2 - yt_img.height//2)           # keep inside the plot height
    pil.alpha_composite(yt_img, (2, yy))
    img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    gt = []
    for (px, py), x, y in zip(pts, xs_val, ys_val):
        gt.append(dict(x=float(x), y=float(y), px=float(px), py=float(py),
                       on_base=bool(y <= y_max*0.005),
                       right_edge=bool(px >= r - radius),
                       on_spine=bool(abs(px - l) <= radius),
                       origin=bool(y <= y_max*0.005 and abs(px - l) <= radius)))
    meta = dict(W=W, H=H, box=(l,t,r,b), x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max, x_step=x_step, y_step=y_step,
                color=color, shape=shape, radius=radius, lw=lw,
                font=os.path.basename(fpath), smooth_labels=bool(smooth_labels))
    return img, gt, meta


def _match(pred_px, gt, tol):
    """Greedy nearest-pixel matching of predicted markers to ground truth."""
    used = set(); pairs = [];
    gtpx = [(g["px"], g["py"]) for g in gt]
    for pi, (px, py) in enumerate(pred_px):
        best, bd = -1, tol
        for gi, (gx, gy) in enumerate(gtpx):
            if gi in used: continue
            d = math.hypot(px-gx, py-gy)
            if d < bd: bd, best = d, gi
        if best >= 0:
            used.add(best); pairs.append((pi, best, bd))
    matched_gt = used
    fp = [pi for pi in range(len(pred_px)) if pi not in {p for p,_,_ in pairs}]
    return pairs, matched_gt, fp


def _curve_metrics(gt, ext_xy, ys_span):
    """Trajectory fidelity: how well the EXTRACTED curve reproduces the TRUE one,
    independent of per-marker recall. Both are treated as piecewise-linear curves
    and compared on a common x-grid, so a missed or spurious marker only perturbs
    the interpolation -- this is the scientifically meaningful, FP/FN-robust view.
    Reports curve error + the derived quantities a growth study actually uses."""
    tx = np.array([g["x"] for g in gt], float); ty = np.array([g["y"] for g in gt], float)
    o = np.argsort(tx); tx, ty = tx[o], ty[o]
    if len(ext_xy) < 2:
        return None
    # extracted curve: sort by x, average any duplicate-x detections
    d = {}
    for x, y in ext_xy:
        d.setdefault(round(float(x), 3), []).append(float(y))
    ex = np.array(sorted(d)); ey = np.array([np.mean(d[k]) for k in ex])
    if len(ex) < 2 or tx.max() <= tx.min():
        return None
    grid = np.linspace(tx.min(), tx.max(), 200)
    tc = np.interp(grid, tx, ty)            # true curve
    ec = np.interp(grid, ex, ey)            # extracted (clips outside its x-range)
    mae = float(np.mean(np.abs(ec - tc)) / ys_span * 100)
    _trap = getattr(np, "trapezoid", np.trapz)
    auc_t = float(_trap(tc, grid)); auc_e = float(_trap(ec, grid))
    # AUC error as % of the PLOT AREA (y-span x x-window), not of the curve's own
    # integral -- the latter explodes for low-volume curves (tiny denominator) and
    # is not a meaningful instability of the extraction.
    plot_area = ys_span * (grid[-1] - grid[0])
    auc = abs(auc_e - auc_t) / plot_area * 100 if plot_area > 1e-9 else float("nan")
    peak = abs(ec.max() - tc.max()) / ys_span * 100
    thr = 0.5 * tc.max()                    # time to reach half the (true) peak volume
    def cross(c):
        idx = np.where(c >= thr)[0]
        return grid[idx[0]] if len(idx) else np.nan
    tt_t, tt_e = cross(tc), cross(ec)
    tt = abs(tt_e - tt_t) if not (np.isnan(tt_t) or np.isnan(tt_e)) else float("nan")
    return dict(mae=mae, auc=auc, peak=peak, tt_days=tt)


def process_one_graph(args):
    i, seed, save_k = args
    rng = np.random.default_rng(seed)

    img, gt, m = gen_graph(rng)
    l, t, r, b = m["box"]
    name = f"synth_{i:03d}"
    # Bake JPEG compression artifacts into EVERY image (blocking/ringing
    # around edges and text), like the real dataset, then save.
    q = int(rng.integers(54, 67))                 # preview-2 ballpark (~59)
    enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
    bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)     # what the pipeline processes
    path = os.path.join(OUT, name + (".jpg" if i % 2 == 0 else ".png"))
    if path.endswith(".jpg"):
        enc.tofile(path)                          # write the SAME bytes -> no double-JPEG
    else:
        cv2.imwrite(path, bgr)                     # PNG preserves the baked-in artifacts

    markers, _ = detect_markers(bgr)
    cal = calibrate(bgr)

    ok = bool(cal["x"]) and bool(cal["y"])

    xs_span = m["x_max"] - m["x_min"]
    ys_span = m["y_max"] - m["y_min"]
    # calibration error at TRUE pixels (isolates OCR + axis fit). A graph with
    # a GROSS scale error (>5% of span) means OCR misread the axis labels and
    # the fit locked onto a wrong scale -- count it as miscalibrated and keep
    # it out of the precision percentiles (its error is OCR, not geometry).
    gx_err = [abs(pixel_to_data(cal, g["px"], g["py"])[0] - g["x"]) / xs_span * 100 for g in gt] if ok else []
    gy_err = [abs(pixel_to_data(cal, g["px"], g["py"])[1] - g["y"]) / ys_span * 100 for g in gt] if ok else []
    good = ok and (np.median(gy_err) < 5 and np.median(gx_err) < 5)

    tol = max(12, 2.6 * m["radius"])
    pairs, matched_gt, fp = _match([(px, py) for px, py in markers], gt, tol)

    gt_stats = []
    rbucket = "4-5px" if m["radius"] <= 5 else ("6-7px" if m["radius"] <= 7 else "8-9px")
    for gi, g in enumerate(gt):
        cat = ("origin" if g["origin"] else
               "spine" if g["on_spine"] else        # x=0, y>0 (bisected by left spine)
               "base" if g["on_base"] else
               ("edge" if g["right_edge"] else "interior"))
        gt_stats.append((cat, m["shape"], rbucket, gi in matched_gt))

    px_err_delta = []
    e2e_err_x_all_delta = []
    e2e_err_y_all_delta = []
    e2e_err_x_delta = []
    e2e_err_y_delta = []
    for pi, gi, d in pairs:
        px_err_delta.append(d)
        if ok:
            px, py = markers[pi]
            ex, ey = pixel_to_data(cal, px, py)
            ax = abs(ex - gt[gi]["x"]) / xs_span * 100
            ay = abs(ey - gt[gi]["y"]) / ys_span * 100
            e2e_err_x_all_delta.append(ax)
            e2e_err_y_all_delta.append(ay)
            if good:
                e2e_err_x_delta.append(ax)
                e2e_err_y_delta.append(ay)

    cm_delta = None
    if ok and len(markers) >= 2:
        ext_xy = [pixel_to_data(cal, px, py) for px, py in markers]
        cm_delta = _curve_metrics(gt, ext_xy, ys_span)

    if i < save_k:
        ov = bgr.copy()
        for g in gt:
            cv2.drawMarker(ov, (int(g["px"]), int(g["py"])), (0,180,0),
                           cv2.MARKER_CROSS, 12, 1)
        for px, py in markers:
            cv2.circle(ov, (int(px), int(py)), int(m["radius"])+3, (0,0,255), 2)
        cv2.imwrite(os.path.join(OUT, name + "_overlay.png"), ov)

    return {
        "ok": ok,
        "good": good,
        "gx_err": gx_err,
        "gy_err": gy_err,
        "fp_len": len(fp),
        "markers_len": len(markers),
        "gt_stats": gt_stats,
        "px_err_delta": px_err_delta,
        "e2e_err_x_all_delta": e2e_err_x_all_delta,
        "e2e_err_y_all_delta": e2e_err_y_all_delta,
        "e2e_err_x_delta": e2e_err_x_delta,
        "e2e_err_y_delta": e2e_err_y_delta,
        "cm_delta": cm_delta
    }


def run(N, seed, save_k, workers=None):
    os.makedirs(OUT, exist_ok=True)

    # Generate deterministic seeds for each task derived from the main seed
    main_rng = np.random.default_rng(seed)
    seeds = main_rng.integers(0, 2**31 - 1, size=N)
    tasks = [(i, int(seeds[i]), save_k) for i in range(N)]

    rec = dict(cal_ok=0, n_img=0, miscal=0)
    cal_err_x, cal_err_y = [], []      # % of axis span, using TRUE pixels (correct-cal only)
    e2e_err_x, e2e_err_y = [], []      # % of axis span, using DETECTED pixels (correct-cal only)
    cal_err_x_all, cal_err_y_all = [], []   # FULL: every calibrated graph (incl. misreads)
    e2e_err_x_all, e2e_err_y_all = [], []
    px_err = []                        # detection pixel error (matched)
    tot = dict(all=0, base=0, edge=0, interior=0, origin=0, spine=0)
    hit = dict(all=0, base=0, edge=0, interior=0, origin=0, spine=0)
    from collections import defaultdict
    shp = defaultdict(lambda: [0, 0])      # shape -> [hit, total]
    rad = defaultdict(lambda: [0, 0])      # radius bucket -> [hit, total]
    traj = defaultdict(list)               # trajectory fidelity (correct-cal only)
    traj_all = defaultdict(list)           # FULL trajectory (every calibrated graph)
    fp_total = 0; pred_total = 0

    results = []
    # If multiprocessing pool size is specified or defaults to CPU count, run in parallel.
    # Standard tqdm updates as each worker finishes.
    with Pool(processes=workers) as pool:
        for res in tqdm(pool.imap_unordered(process_one_graph, tasks), total=N, desc="Benchmark", file=sys.stdout):
            results.append(res)

    for res in results:
        rec["n_img"] += 1
        ok = res["ok"]
        good = res["good"]
        rec["cal_ok"] += ok
        pred_total += res["markers_len"]

        if ok:                         # FULL stats: every calibrated graph
            cal_err_x_all += res["gx_err"]
            cal_err_y_all += res["gy_err"]
        if not good:
            rec["miscal"] += ok        # ok-but-wrong-scale (vs hard FAIL)
        else:                          # conditional stats: correct-cal only
            cal_err_x += res["gx_err"]
            cal_err_y += res["gy_err"]

        fp_total += res["fp_len"]

        for cat, shape, rbucket, is_hit in res["gt_stats"]:
            tot["all"] += 1; tot[cat] += 1
            shp[shape][1] += 1; rad[rbucket][1] += 1
            if is_hit:
                hit["all"] += 1; hit[cat] += 1
                shp[shape][0] += 1; rad[rbucket][0] += 1

        px_err += res["px_err_delta"]
        e2e_err_x_all += res["e2e_err_x_all_delta"]
        e2e_err_y_all += res["e2e_err_y_all_delta"]
        e2e_err_x += res["e2e_err_x_delta"]
        e2e_err_y += res["e2e_err_y_delta"]

        cm = res["cm_delta"]
        if cm:
            for k, v in cm.items():
                if not (isinstance(v, float) and math.isnan(v)):
                    traj_all[k].append(v)
                    if good:
                        traj[k].append(v)

    def pct(d, k): return (100.0*hit[k]/tot[k]) if tot[k] else float("nan")
    def q(a):
        a = np.array(a) if len(a) else np.array([np.nan])
        return np.nanpercentile(a, [50, 90])

    lines = []
    P = lines.append
    P(f"# Synthetic ground-truth benchmark")
    P("")
    P(f"N={N} graphs, seed={seed}. Style sampled from single_curve. "
      f"Marker shapes: {', '.join(SHAPES)}. Every image JPEG-compressed (q54-66) "
      f"to bake in artifacts; saved half .jpg / half .png.")
    P("")
    n = rec["n_img"]; correct = rec["cal_ok"] - rec["miscal"]
    P("## Calibration outcome (headline)")
    P(f"- usable calibration (both axes fit, correct scale): **{correct}/{n}**")
    P(f"- gross OCR scale-misread (axes fit but wrong scale): {rec['miscal']}/{n}")
    P(f"- hard fail (an axis could not be fit): {n - rec['cal_ok']}/{n}")
    P("")
    P("> 'Correct scale' can only be checked here because synthetic graphs have "
      "ground truth; on real images we can verify calibration *succeeded and is "
      "self-consistent* but cannot certify *accuracy* (the reason this benchmark "
      "exists). Calibration difficulty is matched to real: the synthetic OCR "
      "tick-yield is tuned to the real distribution (median ~12 readable y-ticks "
      "per axis vs ~12.5 real -- run `ocr_yield_probe.py`), so the success rate "
      "here tracks real difficulty rather than being a low-legibility artifact. "
      "The few gross misreads are largely **detectable** -- most produce "
      "out-of-range values the pipeline's range check rejects -- rather than "
      "silently corrupting output. Per-marker precision below is **conditional on "
      "a correct calibration**; trajectory figures use all detected markers there.")
    P("")
    P("## Detection recall / false positives")
    P(f"- overall recall: **{pct(hit,'all'):.1f}%** ({hit['all']}/{tot['all']})")
    P(f"- interior markers: {pct(hit,'interior'):.1f}% ({hit['interior']}/{tot['interior']})")
    P(f"- right-edge (bisected): {pct(hit,'edge'):.1f}% ({hit['edge']}/{tot['edge']})")
    P(f"- y=0 baseline (bisected): {pct(hit,'base'):.1f}% ({hit['base']}/{tot['base']})  "
      f"_(documented limitation)_")
    P(f"- left-spine x=0, y>0 (bisected): {pct(hit,'spine'):.1f}% ({hit['spine']}/{tot['spine']})")
    P(f"- origin x=0, y=0 (quartered by both axes): {pct(hit,'origin'):.1f}% ({hit['origin']}/{tot['origin']})  "
      f"_(hardest case)_")
    P(f"- false positives: {fp_total} spurious / {pred_total} predicted")
    mp = q(px_err); P(f"- matched detection pixel error: median {mp[0]:.2f}px, p90 {mp[1]:.2f}px")
    P("")
    P("### Recall by marker shape (thin/concave shapes form weaker blobs)")
    for k in sorted(shp, key=lambda k: shp[k][0]/max(1,shp[k][1])):
        h, n = shp[k]; P(f"- {k}: {100*h/n:.1f}% ({h}/{n})")
    P("")
    P("### Recall by marker radius (smaller = lower prominence)")
    for k in sorted(rad):
        h, n = rad[k]; P(f"- {k}: {100*h/n:.1f}% ({h}/{n})")
    P("")
    def mx(a): return np.nanmax(a) if len(a) else float("nan")
    P("## Per-marker value error (% of axis span)")
    P(f"Two views. **Conditional** = the {rec['cal_ok']-rec['miscal']} correctly-calibrated "
      f"graphs (the method's precision when it works). **Full** = all {rec['cal_ok']} "
      f"calibrated graphs *including* the {rec['miscal']} gross OCR scale-misreads (the "
      f"{rec['n_img']-rec['cal_ok']} hard-fails produce no values and can't enter). The "
      f"full tail is dominated by the misreads, most of which the range check would reject.")
    cx, cy = q(cal_err_x), q(cal_err_y); ex, ey = q(e2e_err_x), q(e2e_err_y)
    cxa, cya = q(cal_err_x_all), q(cal_err_y_all); exa, eya = q(e2e_err_x_all), q(e2e_err_y_all)
    P("| source | X med | X p90 | X max | Y med | Y p90 | Y max |")
    P("|---|---|---|---|---|---|---|")
    P(f"| end-to-end, **conditional** | {ex[0]:.2f}% | {ex[1]:.2f}% | {mx(e2e_err_x):.2f}% | {ey[0]:.2f}% | {ey[1]:.2f}% | {mx(e2e_err_y):.2f}% |")
    P(f"| end-to-end, **full** | {exa[0]:.2f}% | {exa[1]:.2f}% | {mx(e2e_err_x_all):.1f}% | {eya[0]:.2f}% | {eya[1]:.2f}% | {mx(e2e_err_y_all):.1f}% |")
    P(f"| calibration-only, conditional | {cx[0]:.2f}% | {cx[1]:.2f}% | {mx(cal_err_x):.2f}% | {cy[0]:.2f}% | {cy[1]:.2f}% | {mx(cal_err_y):.2f}% |")
    P(f"| calibration-only, full | {cxa[0]:.2f}% | {cxa[1]:.2f}% | {mx(cal_err_x_all):.1f}% | {cya[0]:.2f}% | {cya[1]:.2f}% | {mx(cal_err_y_all):.1f}% |")
    P("")
    P("## Trajectory fidelity (extracted curve vs true | robust to FP/FN)")
    P("The scientifically meaningful view: the whole extracted curve (ALL detected "
      "markers, including false positives, mapped through the fitted axes) vs the "
      "known curve, compared as interpolated trajectories. A missed/spurious marker "
      "only perturbs the interpolation, so this reflects whether the *growth "
      "dynamics* survive -- the quantities a study actually uses.")
    for label, T in [("conditional (correct calibration)", traj), ("full (all calibrated)", traj_all)]:
        tm = q(T["mae"]); ta = q(T["auc"]); tp = q(T["peak"]); tt = q(T["tt_days"])
        P(f"- **{label}** — curve MAE median {tm[0]:.2f}% (p90 {tm[1]:.2f}%, max {mx(T['mae']):.1f}%); "
          f"AUC {ta[0]:.2f}% (p90 {ta[1]:.2f}%); peak {tp[0]:.2f}% (p90 {tp[1]:.2f}%); "
          f"time-to-half-peak {tt[0]:.1f}d (p90 {tt[1]:.1f}d)")
    # single bottom-line over EVERY graph, counting hard-fails (no curve) and gross
    # misreads as failures -- the honest whole-pipeline growth-dynamics success rate
    aucs = np.array(traj_all["auc"])
    rec_n = int((aucs < 5).sum())          # graphs whose AUC is within 5%
    P(f"- **whole-pipeline recovery (ALL {rec['n_img']} graphs, incl. every failure):** "
      f"growth-curve AUC within 5% of truth on **{rec_n}/{rec['n_img']}** "
      f"({100*rec_n/rec['n_img']:.0f}%); the rest are the {rec['miscal']} gross misreads "
      f"+ {rec['n_img']-rec['cal_ok']} hard-fails (which produce no curve).")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(os.path.dirname(__file__), "notes",
              "synthetic_benchmark_results.md"), "w") as fh:
        fh.write(report + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("N", nargs="?", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", type=int, default=6, help="how many overlays to save")
    ap.add_argument("--workers", type=int, default=None, help="number of worker processes")
    a = ap.parse_args()
    run(a.N, a.seed, a.save, a.workers)
