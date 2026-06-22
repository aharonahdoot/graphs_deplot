#!/usr/bin/env python3
"""Extract (x, y) data-marker values from line-graph images.

Pipeline (see markers.py and calibrate.py):
  1. detect data markers as pixel coordinates (colour-agnostic, shape-agnostic,
     robust to thin/bold lines and dense overlapping markers);
  2. calibrate the axes from their tick labels (OCR + robust linear fit);
  3. map each marker pixel to (x, y) data values and emit CSV / JSON.

Usage:
  python src/extract.py IMAGE [IMAGE ...]        # one or more images
  python src/extract.py DIR                       # every *.jpeg/*.png in DIR
  python src/extract.py DIR -o out --overlay      # also write overlay PNGs
  python src/extract.py DIR --format json         # json instead of csv
  python src/extract.py DIR --workers 8           # cap worker processes
  python src/extract.py DIR --workers 1           # force sequential (debugging)

A whole directory is processed in parallel across CPU cores with a tqdm progress
bar; per-image outputs are written by the workers, the aggregate files by the
parent.

Outputs (in the output dir, default ./output):
  <name>.csv / <name>.json   per-image extracted markers
  all_markers.csv            every marker from every image, one row each
  summary.csv                per-image status (n markers, calibration quality)
  overlays/<name>.png        optional visual check
"""
import os
# Limit each process to a single math/OCR thread BEFORE numpy/cv2/tesseract load,
# so N worker processes don't oversubscribe the CPU (the win comes from process
# parallelism across images, not threads within one image).
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import glob
import json
import sys
from collections import Counter
from multiprocessing import Pool

import cv2
from tqdm import tqdm

cv2.setNumThreads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markers import detect_markers, draw_overlay
from calibrate import calibrate, pixel_to_data

IMG_EXT = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")


def extract_image(path):
    """Return a dict with markers as (x, y) data values + calibration metadata."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"could not read image: {path}")
    markers, mdiag = detect_markers(bgr)
    cal = calibrate(bgr)

    rows = []
    if cal["x"] and cal["y"]:
        for px, py in markers:
            x, y = pixel_to_data(cal, px, py)
            rows.append({"x": round(x, 2), "y": round(y, 2),
                         "px": round(px, 1), "py": round(py, 1)})
        rows.sort(key=lambda r: r["x"])

    def quality(fit, pts):
        """Raw, un-combined calibration-confidence components (inliers, total
        labels, tick step, fit rms) so callers can derive relative-rms or
        inlier-fraction however they like in post-processing."""
        if not fit:
            return None
        vals = sorted({round(v) for _, v in pts})
        diffs = [b - a for a, b in zip(vals, vals[1:]) if b > a]
        # MODE of the gaps, not the min: a dropped-digit OCR misread (e.g.
        # '13'->'3') creates a spurious small gap that min-diff would report as
        # the step, falsely inflating rel_rms = rms/step.
        step = Counter(diffs).most_common(1)[0][0] if diffs else None
        return {"inliers": fit[2], "n_labels": len(pts), "step": step, "rms": round(fit[3], 3)}

    return {
        "image": os.path.basename(path),
        "n_markers": len(rows),
        "calibrated": bool(cal["x"] and cal["y"]),
        "x_axis": quality(cal["x"], cal["x_pts"]),
        "y_axis": quality(cal["y"], cal["y_pts"]),
        "markers": rows,
        "_bgr": bgr, "_pixels": markers, "_box": cal["box"],
    }


def write_csv(result, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "pixel_x", "pixel_y"])
        for r in result["markers"]:
            w.writerow([r["x"], r["y"], r["px"], r["py"]])


def write_json(result, path):
    out = {k: result[k] for k in
           ("image", "n_markers", "calibrated", "x_axis", "y_axis", "markers")}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def gather(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for e in IMG_EXT:
                files += glob.glob(os.path.join(p, f"*{e}"))
        elif os.path.isfile(p):
            files.append(p)
    return sorted(set(files))


def _process_one(task):
    """Worker: extract one image, write its per-image outputs, and return only a
    small aggregate payload. The full image array stays in the worker (never
    pickled back to the parent), which is what makes the parallelism cheap."""
    path, out, fmt, overlay = task
    name = os.path.splitext(os.path.basename(path))[0]
    base = os.path.basename(path)
    empty = {"image": base, "n_detected": 0, "n_markers": 0, "calibrated": False,
             "x_rms": "", "x_inliers": "", "x_labels": "", "x_step": "",
             "y_rms": "", "y_inliers": "", "y_labels": "", "y_step": ""}
    try:
        res = extract_image(path)
    except Exception as e:  # keep going across the dataset
        return {"image": base, "error": str(e), "rows": [], "summary": empty}

    if fmt in ("csv", "both"):
        write_csv(res, os.path.join(out, name + ".csv"))
    if fmt in ("json", "both"):
        write_json(res, os.path.join(out, name + ".json"))
    if overlay:
        ov = draw_overlay(res["_bgr"], res["_pixels"], res["_box"])
        cv2.imwrite(os.path.join(out, "overlays", name + ".png"), ov)

    xa, ya = res["x_axis"], res["y_axis"]
    return {
        "image": res["image"], "error": None,
        "rows": [[res["image"], r["x"], r["y"]] for r in res["markers"]],
        "summary": {
            "image": res["image"],
            "n_detected": len(res["_pixels"]),   # markers found (drawn on the overlay)
            "n_markers": res["n_markers"],         # markers written to the CSV
            "calibrated": res["calibrated"],
            "x_rms": xa["rms"] if xa else "", "x_inliers": xa["inliers"] if xa else "",
            "x_labels": xa["n_labels"] if xa else "", "x_step": xa["step"] if xa else "",
            "y_rms": ya["rms"] if ya else "", "y_inliers": ya["inliers"] if ya else "",
            "y_labels": ya["n_labels"] if ya else "", "y_step": ya["step"] if ya else "",
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract data-marker (x,y) values from line graphs.")
    ap.add_argument("inputs", nargs="+", help="image file(s) or directory")
    ap.add_argument("-o", "--out", default="output", help="output directory (default: output)")
    ap.add_argument("--format", choices=["csv", "json", "both"], default="csv")
    ap.add_argument("--overlay", action="store_true", help="also write overlay PNGs")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: all CPU cores; 1 = sequential)")
    args = ap.parse_args(argv)

    files = gather(args.inputs)
    if not files:
        ap.error("no images found in the given inputs")
    os.makedirs(args.out, exist_ok=True)
    if args.overlay:
        os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)

    tasks = [(p, args.out, args.format, args.overlay) for p in files]
    n_workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(tasks)))

    results = []
    if n_workers == 1:                                   # sequential (single image / debugging)
        for t in tqdm(tasks, desc="Extracting", unit="img"):
            results.append(_process_one(t))
    else:
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap_unordered(_process_one, tasks),
                            total=len(tasks), desc="Extracting", unit="img", file=sys.stdout):
                results.append(res)

    results.sort(key=lambda r: r["image"])               # deterministic output order
    all_rows = [row for r in results for row in r["rows"]]
    summary = [r["summary"] for r in results]
    errors = [r for r in results if r.get("error")]

    with open(os.path.join(args.out, "all_markers.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "x", "y"]); w.writerows(all_rows)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "n_detected", "n_markers", "calibrated",
                                          "x_rms", "x_inliers", "x_labels", "x_step",
                                          "y_rms", "y_inliers", "y_labels", "y_step"])
        w.writeheader(); w.writerows(summary)

    ncal = sum(1 for s in summary if s["calibrated"])
    print(f"\n{len(files)} images -> {args.out}/  "
          f"({ncal}/{len(files)} calibrated, {sum(s['n_markers'] for s in summary)} markers total)")
    uncal = [s["image"] for s in summary if not s["calibrated"] and s["image"] not in {e["image"] for e in errors}]
    if uncal:
        print(f"  not calibrated ({len(uncal)}): {', '.join(uncal)}")
    for e in errors:
        print(f"  ERROR {e['image']}: {e['error']}")


if __name__ == "__main__":
    main()
