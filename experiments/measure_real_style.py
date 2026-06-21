"""Measure the visual style of the real single_curve images, to ground the
synthetic-benchmark generator constants (DIM_*, BG_TOP/BG_BOT, MARK_R, LINE_W,
N_MARK, PALETTE in synthetic_benchmark.py). Everything the generator mimics is
sourced from this probe rather than invented.

Run:  .venv/bin/python experiments/measure_real_style.py [N]
Needs single_curve/ locally (not redistributed).
"""
import sys, os, glob, random
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from markers import detect_markers, find_spines


def main(n=50, seed=3):
    random.seed(seed)
    fs = sorted(glob.glob("single_curve/*.png") + glob.glob("single_curve/*.jpg"))
    fs = random.sample(fs, min(n, len(fs)))
    dims, dmaxs, lws, ncs, bg_top, bg_bot, ser = [], [], [], [], [], [], []
    for f in fs:
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]; dims.append((W, H))
        try:
            l, t, r, b = find_spines(bgr)
        except Exception:
            continue
        interior = bgr[t+3:b-3, l+3:r-3]
        ih = interior.shape[0]
        if ih < 20:
            continue
        def bgmed(px):                      # background = the bright (non-ink) pixels
            px = px.reshape(-1, 3); m = px.min(axis=1) > 200
            return np.median(px[m], axis=0) if m.sum() > 10 else np.median(px, axis=0)
        bg_top.append(bgmed(interior[:ih//6])); bg_bot.append(bgmed(interior[-ih//6:]))
        mk, diag = detect_markers(bgr)
        ncs.append(len(mk)); dmaxs.append(diag.get("dmax", np.nan)); lws.append(diag.get("lw", np.nan))
        mask = diag.get("mask")
        if mask is not None:
            ys, xs = np.where(mask > 0)
            if len(xs):
                sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[ys, xs, 1]
                sel = sat > 60                          # the saturated series ink
                if sel.sum() > 20:
                    ser.append(np.median(bgr[ys, xs][sel], axis=0))
    dims = np.array(dims)
    print(f"sampled {len(dims)} images\n")
    print("DIM_W  (W p5/50/95):", np.percentile(dims[:,0], [5,50,95]).astype(int))
    print("DIM_H  (H p5/50/95):", np.percentile(dims[:,1], [5,50,95]).astype(int))
    print("BG_TOP (interior top, BGR):  ", np.median(bg_top, axis=0).astype(int))
    print("BG_BOT (interior bottom, BGR):", np.median(bg_bot, axis=0).astype(int))
    print("MARK_R (marker radius dmax p5/50/95):", np.round(np.nanpercentile(dmaxs, [5,50,95]), 1))
    print("LINE_W (line width lw p5/50/95):     ", np.round(np.nanpercentile(lws, [5,50,95]), 1))
    print("N_MARK (markers/image p5/50/95):     ", np.percentile(ncs, [5,50,95]).astype(int))
    print("PALETTE (sample of series colours, BGR):")
    for c in np.array(ser).astype(int)[:10]:
        print("   ", c)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
