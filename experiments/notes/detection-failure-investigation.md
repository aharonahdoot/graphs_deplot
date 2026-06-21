# Investigation: which graph/marker styles the pipeline fails on

Driven by the synthetic ground-truth benchmark (`../synthetic_benchmark.py`),
which renders growth-curve graphs in the measured single_curve style with KNOWN
marker coordinates. Two questions: (A) which marker/graph **styles** lose
detection recall, and (B) the cause of occasional gross **scale** errors.
Headline numbers in `synthetic_benchmark_results.md` (N=80, seed 7).

## A. Marker-detection recall — failure modes

Detection precision is excellent where a marker is found: **median 0.62 px**
(p90 2.71 px) off the true centre. Recall is where the failures live.

Recall by category (N=500, seed 7): interior **91.9%**, right-edge bisected
**94.0%**, **left-spine day-0 (x=0, y>0) 98.8%**, **origin (x=0, y=0) 95.7%**,
**interior-x y=0 baseline 66.7%**. By shape: **X 84.3%**, star/square/triangle/
plus ~91–93%, circle/diamond 95–96%. By radius: **4–5 px 90.2%** vs 6–7 px 94.7%.

> **Update (2026-06):** the origin / left-spine cases below were the worst
> failure modes (origin recall was **33%**). They are now recovered FP-free —
> root cause was `series_mask` cropping the spine marker's thickest columns plus
> the prominence step failing on the clipped/fused blob. Full record and the
> rejected full-baseline experiments in
> `origin-spine-recovery-investigation.md`. The interior-x y=0 baseline remains
> at its ~67% floor (recovering it re-admits the axis line as false positives).

**Root cause (one mechanism explains all of it).** `detect_markers`
(`src/markers.py`) finds markers as **prominence peaks of the distance
transform** — `h_maxima(dist, 1.5) & (dist >= max(2.1, lw+0.8))`. It keys on
*local blob thickness*. Every failure mode is a weak distance-transform peak:
- **Thin-stroke shapes (X, plus)** are 1–2 px strokes, not fat blobs → low `dist`
  → fail the prominence/threshold. X is the worst (two thin diagonals).
- **Concave shapes (star)** have thin points; only the core survives the test.
- **Small radius (4–5 px)** → low `dist`, close to the `lw+0.8` line-width gate.
- **y=0 baseline** markers fuse with the axis line; their thickness ≈ the
  line/baseline thickness, so there is no prominence to detect (this is the
  documented limitation, now quantified at ~65% recall — the rest are accepted
  as low-signal points at volume ≈ 0).

These compound: a small thin marker on a thick line at y=0 is the worst case.

**Confirmed relevant to real data, not a synthetic artifact:** sampling real
single_curve images shows the dataset *does* use star, square, and bold-line
styles (not only circles), so the shape/line-weight sensitivity is real.

**Caveat on the per-shape numbers.** Shape is assigned per graph (~10 graphs per
shape at N=80), so each shape's recall is confounded by that sample's random
radius / line-weight / density / y=0 count; the numbers move run-to-run (e.g.
triangle 84% here vs 95% at another seed). The *robust* signal is the ordering:
thin-stroke + small-radius + baseline-fused are the hard cases. Run larger N to
tighten per-shape estimates.

## B. Occasional gross scale error — root cause found and fixed

Symptom: a few graphs calibrated to a wildly wrong Y scale (value error ≫100% of
axis span) while X was fine.

Hypotheses considered:
1. **Genuine calibration fragility.** *Rejected* — real single_curve images
   calibrate 25/25 with correct values; the production fit is robust.
2. **Detection feeding bad points to calibration.** *Rejected* — calibration
   reads axis *labels*, independent of marker detection; the error reproduced
   with true pixels.
3. **OCR misreading the axis labels.** *Confirmed.* The misreads strip
   punctuation: `200.00` → `20000`, `1,600.00` → `1600000` (~100× inflation).
   X-labels are plain integers, so X stays clean — exactly the asymmetry seen.

Causal chain: the synthetic labels were rendered as tiny AA-off pixel blocks; the
comma/period (1 px) got absorbed into the adjacent digit, so OCR dropped them.

Fix (generator fidelity, `_label_tile`): render labels per-character with 1 px
letter-spacing so punctuation stays separable, at realistic (not exaggerated)
kerning. This removed the ×100 inflation.

## C. Does the robust fit (RANSAC) actually handle the failures?

Note: `calibrate._fit` **already uses RANSAC** (pairwise consensus: the line
through some label pair that the most other labels agree with, ±¼ tick, then a
least-squares refit on those inliers). (The README's "Theil-Sen" description is
stale — the code was switched to RANSAC precisely because dropped-digit misreads
corrupt Theil-Sen's median of pairwise slopes.)

Tested on the N=80 synthetic set, classifying each graph:
- **RANSAC recovered 18 graphs** by rejecting an isolated misread outlier; **44
  were clean** (every label an inlier).
- **14 miscalibrated**, and they expose RANSAC's two hard limits, neither of
  which RANSAC *can* fix:
  1. **Too few labels read** (often only 2 of ~10 ticks survive OCR on the hard
     pixel-blocked text). With 2 points there is no consensus to test — any
     single misread or mis-localisation defines a wrong line.
  2. **A spurious outlier among too few good labels** (e.g. a stray "3300" with
     only 2–3 real labels): the consensus set is too small to outvote it.
- **4 hard FAILs** (<2 numeric labels read at all).

Conclusion: RANSAC works **given enough correctly-read labels** — it needs
redundancy to form a consensus. The residual synthetic failures are driven by
**low OCR yield** on the deliberately-hard pixel-blocked labels, not by a fit
deficiency.

### Is the "pessimistic vs real" claim justified without ground truth?

Care is needed: real images calibrate **and pass the consistency checks** at a
high rate (25/25 random sample, 44/44 curated), but "pass" means *succeeded +
self-consistent*, **not** *certified accurate* — we have no real ground truth,
which is the whole reason for this benchmark. Two ground-truth-free probes
support the "pessimistic vs real" claim without any accuracy assumption:

1. **OCR tick yield** (measurable directly): real images yield a **median of 12
   readable y-tick labels per axis (min 6)** vs the synthetic **median of 4 (often
   just 2)**. RANSAC needs consensus redundancy; real data supplies ~3× more, so
   it fails far less often there. The synthetic labels are *visually* matched but
   *functionally harder to OCR* than real — so the synthetic calibration-failure
   rate overstates real risk.
2. **Catchability**: of the gross-misread synthetic graphs, ~3 of 5 produce
   marker values **outside the plotted axis range**, which `verify_system`'s
   range check rejects — so most gross miscalibrations are **detectable, not
   silent** (≈2/5 could still slip past, a real residual risk).

## D. Trajectory-metric regression (false positives + an unstable AUC %)

When image height was made to scale with tick count (so dense-tick graphs are
taller, like the real data), the trajectory-fidelity p90 appeared to regress and
false positives jumped 25 → 97. Investigation:
- **All the extra FPs sat on the y=0 baseline at the 4 minor-x-tick positions.**
  An isolated 1 px tick stub is far too thin to detect (distance-transform peak
  ~0.5 ≪ the 2.1 gate); only its *fusion* with the 1 px baseline spine made a
  detectable junction blob. Real graphs don't show this (their near-baseline
  detections track the curve). Fix: draw x-ticks with a 1 px gap below the
  baseline → FP 97 → 27, visually unchanged.
- The remaining inflated AUC p90 was **a metric artifact**: AUC error normalised
  by the curve's *own integral* explodes for low-volume curves (one graph: cal
  fine, recall 32/33, yet AUC 177% because its true integral is tiny). Fix:
  normalise by plot area → AUC p90 6.7% → 1.5%.
- After both fixes the worst trajectory graphs have **FP = 0** and are
  **X-marker graphs with low recall** (markers missed in the steep growth
  region) — the documented thin-shape weakness (§A), not a new defect.

What we therefore *can* claim: calibration **accuracy** is established only where
there is ground truth — this benchmark (sub-1% value error when calibration is
correct) — plus the human-overlay gridline checks on real images. Calibration
**success rate** is reported separately and is pessimistic on synthetic for the
yield reason above. Gross-misread graphs are stratified out of the precision
percentiles (not deleted): the success rate is reported as a headline metric.
