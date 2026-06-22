#!/usr/bin/env python3
"""Render a review overlay for every image where the CUT-TOP box fix recovered a
data point -- i.e. a plot cut off at the top image edge whose box used to end one
gridline too early, clipping the highest marker(s). Each overlay shows the
detected markers (thin red) and the RECOVERED marker(s) highlighted (green ring +
yellow crosshair), so they can be eyeballed in the viewer (review_gui.py).

The recovery is the `find_spines` top walk-up gated by markers._RECOVER_CUT_TOP;
this tool toggles that flag to diff with/without it.

Usage:
  python tools/build_recovery_review.py [--src single_curve] [--out experiments/out/recovery_review] [--workers N]
"""
import os, sys, argparse, glob
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import markers as M
from markers import detect_markers

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _process(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    M._RECOVER_CUT_TOP = True
    on, diag = detect_markers(bgr)
    M._RECOVER_CUT_TOP = False                     # process-local: box without the top fix
    off, _ = detect_markers(bgr)
    M._RECOVER_CUT_TOP = True
    gained = [p for p in on if p not in off]
    if not gained:
        return None

    ov = bgr.copy()
    l, t, r, b = diag["box"]
    cv2.rectangle(ov, (l, t), (r, b), (0, 200, 0), 1)
    for x, y in off:
        cv2.circle(ov, (int(x), int(y)), 6, (0, 0, 255), 1)
    for x, y in gained:
        p = (int(round(x)), int(round(y)))
        cv2.circle(ov, p, 14, (0, 200, 0), 3)
        cv2.drawMarker(ov, p, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
    return os.path.basename(path), ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="single_curve")
    ap.add_argument("--out", default=os.path.join("experiments", "out", "recovery_review"))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    files = sorted(f for e in IMG_EXT for f in glob.glob(os.path.join(a.src, f"*{e}")))
    if not files:
        ap.error(f"no images in {a.src}")
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with Pool(processes=a.workers) as pool:
        for res in tqdm(pool.imap_unordered(_process, files), total=len(files),
                        desc="Rendering", unit="img", file=sys.stdout):
            if res is None:
                continue
            name, ov = res
            cv2.imwrite(os.path.join(a.out, os.path.splitext(name)[0] + ".png"), ov)
            n += 1
    print(f"\n{n} cut-top-recovered overlays -> {a.out}/")
    print(f"View with:  python tools/review_gui.py {a.out}")


if __name__ == "__main__":
    main()
