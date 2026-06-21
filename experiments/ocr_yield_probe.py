"""OCR tick-yield probe — how many numeric axis-tick labels the calibrator
actually reads per image, for REAL single_curve images vs the SYNTHETIC
benchmark graphs.

This is the ground-truth-free check behind the benchmark's claim that synthetic
calibration difficulty should track real difficulty: RANSAC axis-fitting needs a
healthy number of correctly-read tick labels to form a consensus, so if synthetic
images yield far fewer readable ticks than real ones, the synthetic
calibration-failure rate would be an artifact rather than a property of the
method. We tune the synthetic tick density (see synthetic_benchmark.gen_graph,
`y_step`) so the synthetic yield lands in the real range.

Run:  .venv/bin/python experiments/ocr_yield_probe.py [N_real] [N_synth]
Needs single_curve/ locally for the real side (not redistributed).
"""
import sys, os, glob, random
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from calibrate import calibrate
import synthetic_benchmark as sb


def yield_real(n, seed=1):
    fs = sorted(glob.glob("single_curve/*.png") + glob.glob("single_curve/*.jpg"))
    random.seed(seed)
    fs = random.sample(fs, min(n, len(fs)))
    return [len(calibrate(cv2.imread(f))["y_pts"]) for f in fs]


def yield_synth(n, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        img, gt, m = sb.gen_graph(rng)
        q = int(rng.integers(54, 67))
        bgr = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                           cv2.IMREAD_COLOR)
        out.append(len(calibrate(bgr)["y_pts"]))
    return out


def _stat(name, ys):
    ys = np.array(ys)
    print(f"  {name:10s} n={len(ys):3d}  median={np.median(ys):5.1f}  "
          f"p10={np.percentile(ys,10):4.1f}  min={ys.min()}  max={ys.max()}")


if __name__ == "__main__":
    n_real = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_syn = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    print("OCR y-tick yield (numeric tick labels read per axis):")
    _stat("REAL", yield_real(n_real))
    _stat("SYNTHETIC", yield_synth(n_syn))
    print("\nGoal: synthetic median should sit in the real range, so calibration "
          "difficulty on synthetic tracks real rather than being an artifact.")
