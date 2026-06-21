"""Scientific A/B of OCR backends for axis-label calibration -- no ground truth.

For each backend, over the SAME random image sample, we measure:

  ACCURACY (label-free):
    * misread rate  -- fraction of OCR'd tick labels the robust fit rejects as
      outliers (a clean digit read lands on the line; a dropped-digit read does
      not). Lower is better. This is the metric that exposed Tesseract's
      leading-'1' loss.
    * LOO error     -- leave-one-out held-out tick prediction error, as a % of
      the tick step (how well calibration generalises when each engine's reads
      are fed to the identical fit). Lower is better.
    * #ticks used   -- mean inlier ticks per axis; more redundancy = safer fit.

  SPEED:
    * sec/image for the OCR-dominated calibrate() call, and images/sec.

Run:
  .venv/bin/python tests/ab_ocr.py [N] [backend ...]
  e.g.  .venv/bin/python tests/ab_ocr.py 30 tesseract rapidocr rapidocr-coreml
"""
import glob, os, random, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))   # sibling backends/pipeline modules
import calibrate
from ocr_backends import get_backend

# The shipping src/calibrate.py deliberately has no pluggable-backend hook -- the
# A/B machinery is kept out of the clean production code. This experiment installs
# the hook at runtime by swapping the module's token function, so src/ is untouched.
_ORIG_OCR_TOKENS = calibrate._ocr_tokens
def _set_ocr_backend(backend):
    calibrate._ocr_tokens = _ORIG_OCR_TOKENS if backend is None else backend
calibrate.set_ocr_backend = _set_ocr_backend

random.seed(0)
argv = sys.argv[1:]
N = int(argv[0]) if argv and argv[0].isdigit() else 25
backends = [a for a in argv if not a.isdigit()] or ["tesseract", "rapidocr"]

_d = os.path.join(os.path.dirname(__file__), "..", "..", "single_curve")
imgs = sorted(glob.glob(os.path.join(_d, "*.png")) + glob.glob(os.path.join(_d, "*.jpg")))
sample = random.sample(imgs, min(N, len(imgs)))
crops = [cv2.imread(f) for f in sample]
crops = [c for c in crops if c is not None]
print(f"Sample: {len(crops)} images\n")


def loo_rel(pts):
    """Leave-one-out held-out errors (as fraction of tick step) over the inlier
    ticks; plus the misread rate and inlier count for the full label set."""
    if len(pts) < 3:
        return [], None, 0
    px = np.array([p for p, _ in pts], float)
    vv = np.array([v for _, v in pts], float)
    u = np.unique(np.round(vv))
    step = np.median(np.diff(np.sort(u))) if len(u) > 1 else 1.0
    s, b = np.polyfit(px, vv, 1)
    inl = np.abs(vv - (s * px + b)) <= 0.25 * step
    misread = 1.0 - inl.mean()
    px, vv = px[inl], vv[inl]
    errs = []
    for k in range(len(px)):
        m = np.ones(len(px), bool); m[k] = False
        if m.sum() < 2 or np.ptp(px[m]) == 0:
            continue
        ss, bb = np.polyfit(px[m], vv[m], 1)
        errs.append(abs(vv[k] - (ss * px[k] + bb)) / step if step else 0)
    return errs, misread, int(inl.sum())


results = {}
for name in backends:
    try:
        calibrate.set_ocr_backend(get_backend(name))
    except Exception as e:
        print(f"[skip {name}] {e}"); continue
    # warm up (model load / EP compile) so timing reflects steady state
    try:
        calibrate.calibrate(crops[0])
    except Exception as e:
        print(f"[skip {name}] failed on warmup: {e}"); continue

    rel, misreads, ninl, t0, ncal = [], [], [], time.time(), 0
    for c in crops:
        cal = calibrate.calibrate(c)
        ncal += bool(cal["x"] and cal["y"])
        for key in ("x_pts", "y_pts"):
            e, mr, ni = loo_rel(cal[key])
            rel += e
            if mr is not None:
                misreads.append(mr); ninl.append(ni)
    dt = time.time() - t0
    results[name] = dict(
        rel=np.array(rel) if rel else np.array([0.0]),
        misread=np.mean(misreads) if misreads else float("nan"),
        ninl=np.mean(ninl) if ninl else 0,
        spi=dt / len(crops), ncal=ncal,
    )
    print(f"  {name:18s} done  ({dt:.1f}s)")

calibrate.set_ocr_backend(None)  # restore production default

print(f"\n{'backend':18s} {'calib':>6} {'misread':>8} {'LOO med':>8} {'LOO p90':>8} "
      f"{'ticks':>6} {'sec/img':>8} {'img/s':>7}")
print("-" * 80)
for name, r in results.items():
    print(f"{name:18s} {r['ncal']:3d}/{len(crops):<2d} "
          f"{r['misread']*100:6.1f}% "
          f"{np.median(r['rel'])*100:6.1f}% {np.percentile(r['rel'],90)*100:6.1f}% "
          f"{r['ninl']:6.1f} {r['spi']:7.3f}s {1/r['spi']:6.1f}")
print("\nmisread/LOO: lower=better. calib: images both axes fit. "
      "ticks: mean inlier labels/axis.")
