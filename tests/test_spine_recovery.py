"""Detection regression test for markers on the LEFT SPINE (x = x_min).

Every growth curve's first study point sits on the left spine; its value is
usually ~0 (the ORIGIN, quartered by both axes) but sometimes a non-zero
baseline reading (bisected by the spine alone). Both are routinely clipped by
the axis and were being dropped by the generic prominence test. This test pins
the recall for those two categories so the fix cannot regress.

Run: .venv/bin/python tests/test_spine_recovery.py
"""
import os, sys, math
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
import numpy as np
import cv2

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sb", os.path.join(HERE, "..", "experiments", "synthetic_benchmark.py"))
sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)
from markers import detect_markers


def measure(n=80, seed=7):
    rng_seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n)
    tot = {"origin": 0, "spine": 0}
    hit = {"origin": 0, "spine": 0}
    for s in rng_seeds:
        rng = np.random.default_rng(int(s))
        img, gt, m = sb.gen_graph(rng)
        # same JPEG path the pipeline actually processes
        enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 59])[1]
        bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        markers, _ = detect_markers(bgr)
        tol = max(12, 2.6 * m["radius"])
        for g in gt:
            if not g["on_spine"]:
                continue
            cat = "origin" if g["origin"] else "spine"
            tot[cat] += 1
            if any(math.hypot(x - g["px"], y - g["py"]) < tol for x, y in markers):
                hit[cat] += 1
    return tot, hit


if __name__ == "__main__":
    tot, hit = measure()
    for cat in ("origin", "spine"):
        r = hit[cat] / tot[cat] if tot[cat] else float("nan")
        print(f"{cat:8s} recall: {hit[cat]:3d}/{tot[cat]:<3d} = {100*r:.1f}%")
    overall = (hit["origin"] + hit["spine"]) / (tot["origin"] + tot["spine"])
    print(f"left-spine overall: {100*overall:.1f}%")
    # Requirement: left-spine markers (the day-0 point that is ALWAYS present in a
    # growth curve) must be captured at least as reliably as other hard cases.
    assert overall >= 0.90, f"left-spine recall too low: {100*overall:.1f}% (< 90%)"
    print("PASS")
