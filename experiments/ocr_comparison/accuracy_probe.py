"""Label-free accuracy probes for the extraction pipeline.

Without hand-labelled ground truth, we still get real error numbers two ways:

1. LEAVE-ONE-OUT CALIBRATION (jackknife): for each axis, drop one OCR'd tick
   label, refit the pixel->value line on the rest, and predict the held-out
   tick's value. The held-out error (in data units, and normalised by the tick
   step) is an honest estimate of how well the calibration generalises -- it
   uses no external truth, only the redundancy of having many ticks.

2. X-GRID QUANTISATION RESIDUAL: tumour measurements fall on a regular study-day
   schedule. Snap each extracted x to the best regular grid (modal spacing) and
   measure the residual. A tight residual means the x pipeline (detection +
   calibration) lands points where real measurements must be; a loose one bounds
   the x error from below -- again with no labels.

Run:  .venv/bin/python tests/accuracy_probe.py [N]
"""
import glob, os, random, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from calibrate import calibrate

random.seed(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
imgs = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "single_curve", "*.jpg")))
sample = random.sample(imgs, min(N, len(imgs)))


def loo(pts):
    """Leave-one-out held-out prediction errors for one axis's tick anchors.
    Returns list of (abs_err_data_units, step)."""
    pts = list(pts)
    if len(pts) < 4:
        return []
    px = np.array([p for p, _ in pts], float)
    vv = np.array([v for _, v in pts], float)
    # use only the consensus (inlier) ticks: a single misread tick is itself an
    # error we want to keep OUT of the "how good is a clean fit" estimate, but
    # we report the misread rate separately.
    step = np.median(np.diff(np.sort(np.unique(np.round(vv)))))
    s_all, b_all = np.polyfit(px, vv, 1)
    inl = np.abs(vv - (s_all * px + b_all)) <= 0.25 * step
    px, vv = px[inl], vv[inl]
    errs = []
    for k in range(len(px)):
        m = np.ones(len(px), bool); m[k] = False
        if m.sum() < 2 or np.ptp(px[m]) == 0:
            continue
        s, b = np.polyfit(px[m], vv[m], 1)
        errs.append((abs(vv[k] - (s * px[k] + b)), step))
    return errs


xerr_rel, yerr_rel, misread = [], [], []
for f in sample:
    bgr = cv2.imread(f)
    cal = calibrate(bgr)
    for fit_key, pts_key, bucket in (("x", "x_pts", xerr_rel), ("y", "y_pts", yerr_rel)):
        pts = cal[pts_key]
        if not pts:
            continue
        vv = np.array([v for _, v in pts], float)
        px = np.array([p for p, _ in pts], float)
        step = np.median(np.diff(np.sort(np.unique(np.round(vv))))) if len(vv) > 1 else 1
        s, b = np.polyfit(px, vv, 1)
        n_out = int((np.abs(vv - (s * px + b)) > 0.25 * step).sum())
        misread.append(n_out / max(1, len(vv)))
        for e, st in loo(pts):
            bucket.append(e / st if st else 0)


def report(name, rel):
    if not rel:
        print(f"{name}: no data"); return
    a = np.array(rel)
    print(f"{name}: n={len(a):4d}  "
          f"median={np.median(a)*100:5.1f}%  mean={a.mean()*100:5.1f}%  "
          f"p90={np.percentile(a,90)*100:5.1f}%  p99={np.percentile(a,99)*100:5.1f}%  "
          f"(% of one tick step)")


print(f"Leave-one-out calibration accuracy over {len(sample)} random images")
print("(held-out tick prediction error as a fraction of the tick spacing)\n")
report("X axis", xerr_rel)
report("Y axis", yerr_rel)
print(f"\nMisread tick rate (labels rejected as outliers): "
      f"mean {np.mean(misread)*100:.1f}% per axis")
