"""Verify that 'good overlay = good extraction' across an output folder.

The overlay draws every detected marker (n_detected); the CSV holds every
extracted (x,y) (n_markers). They match exactly when the image calibrated, since
each detected marker then becomes one CSV row. This asserts:
  * n_detected == n_markers for every image (no overlay-without-data mismatch);
  * the per-image CSV row count equals n_markers (files agree with the summary).
Reports any exceptions; exits non-zero if any remain.

Usage: python tests/verify_overlays_match.py [output_dir]
"""
import csv, glob, os, sys

out = sys.argv[1] if len(sys.argv) > 1 else "output_single_curve"
rows = list(csv.DictReader(open(os.path.join(out, "summary.csv"))))
assert rows, "empty summary"

mismatch, uncal, filebad = [], [], []
for r in rows:
    nd, nm = int(r["n_detected"]), int(r["n_markers"])
    if r["calibrated"] != "True":
        uncal.append(r["image"])
    if nd != nm:
        mismatch.append((r["image"], nd, nm))
    # cross-check the actual CSV file row count
    csvf = os.path.join(out, os.path.splitext(r["image"])[0] + ".csv")
    if os.path.exists(csvf):
        with open(csvf) as f:
            ncsv = sum(1 for _ in f) - 1  # minus header
        if ncsv != nm:
            filebad.append((r["image"], ncsv, nm))

print(f"images: {len(rows)}")
print(f"uncalibrated (overlay has markers, CSV empty): {len(uncal)}")
print(f"n_detected != n_markers mismatches:           {len(mismatch)}")
print(f"per-file CSV count disagrees with summary:    {len(filebad)}")
for im, nd, nm in mismatch[:20]:
    print(f"  MISMATCH {im}: detected={nd} extracted={nm}")
for im in uncal[:20]:
    print(f"  UNCALIBRATED {im}")

ok = not mismatch and not uncal and not filebad
print("\n" + ("ALL OVERLAYS MATCH EXTRACTION ✓" if ok else "DISCREPANCIES REMAIN ✗"))
sys.exit(0 if ok else 1)
