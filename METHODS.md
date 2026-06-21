# Methods, limitations, and validity

This document describes how the extractor works, how it was developed, its known
limitations, and — most importantly — how its accuracy was established given that
the real graphs have no ground-truth coordinates. It is written for readers
evaluating the tool as the data-extraction step of a study.

## 1. What the tool does

It reads images of tumour growth-curve line graphs (the `single_curve` dataset:
3,000+ machine-rendered single-series charts; and a 44-image multi-series set)
and extracts the **(x, y) values of the plotted data markers** — the measured
points — ignoring the interpolating line between them. It is designed to be
robust across the dataset's variation in background gradient, series colour,
marker shape (circle / square / triangle / star / ✕ / …), line weight, axis
ranges, and image resolution.

## 2. How the code works (`src/`, ~520 LOC)

The pipeline is deliberately **classical, deterministic, and inspectable** — no
trained model — so its behaviour and failure modes can be reasoned about and
reproduced exactly. Three stages:

### 2.1 Marker detection (`src/markers.py`)
1. **Plot frame** (`find_spines`): the axes are the outermost long dark
   vertical/horizontal lines; this gives the plot box.
2. **Series mask**: the series is segmented as *significantly darker than the
   cream background* **OR** *high-saturation* (`(dk > 30) | (S > 70)`). This is
   colour-agnostic — it captures black, coloured, and pale series without
   per-colour tuning. Gridlines and tick labels are excluded (low saturation /
   outside the box).
3. **Marker isolation by distance-transform prominence**: the connecting line is
   a thin ridge; markers are fat blobs. Markers are found as **h-maxima** of the
   distance transform — regional maxima rising `h = 1.5` above their surroundings
   and thicker than the line (`dist ≥ max(2.1, line_width + 0.8)`). The
   threshold `h = 1.5` sits in the gap between line-rasterisation "staircase"
   noise (~1.2) and the prominence of genuinely overlapping markers (~2.0), so it
   rejects the staircase yet still **splits overlapping markers**. Each blob's
   distance-weighted centroid is the marker centre.
4. **Edge-clipped marker recovery.** Markers bisected by a plot boundary lose the
   thickness the generic prominence test needs. Three boundary cases are handled:
   the **right edge** (the series mask keeps the frame column so the half-symbol
   survives, then a dedupe keeps one marker per edge exit); the **left spine** —
   a growth curve's day-0 point always sits at `x = x_min`, so if none was
   detected in a narrow band right of the spine its thickest blob is recovered
   (domain prior, FP-free by construction: it adds at most one marker where ink
   exists); and the **origin corner** (`x = x_min, y ≈ 0`, quartered by *both*
   axes) — recovered from a colour mask (drops the grey spine) + vertical opening
   (drops the thin line) + a thickness/roundness gate, confined to the corner so
   no false positives are added across the baseline (see §4).

### 2.2 Axis calibration (`src/calibrate.py`)
1. **OCR the tick labels** with tesseract (`--psm 11`, numeric whitelist,
   confidence > 20), feeding grayscale directly (a global OTSU binarisation
   erased thin strokes and lost labels).
2. **Robust linear fit per axis by RANSAC** (`_fit`): the line through some pair
   of tick labels that the most other labels agree with (within a quarter tick
   step) wins, then a least-squares refit on those inliers. RANSAC — not
   Theil-Sen — because a dropped-digit misread (`165 → 65`) corrupts a majority
   of pairwise slopes and breaks Theil-Sen's median, whereas a clean run of
   collinear labels still forms the largest consensus set. For clean
   calibrations every label is an inlier and the result equals plain
   least-squares.

### 2.3 Extraction (`src/extract.py`)
Map each detected marker pixel through the fitted per-axis transforms to (x, y);
write per-image CSV/JSON, a combined table, and optional overlays.

## 3. How it was developed

The design was arrived at empirically; the full record is in `experiments/`
(see `experiments/README.md`), including approaches that were **tried and
rejected**:

- Series segmentation, frame detection, and the colour-agnostic threshold were
  designed from pixel-value probes across deliberately different images
  (`exp1_probe.py`, `exp3_probe_colors.py`).
- The marker prominence threshold `h = 1.5` was chosen by a sweep across images
  that stress opposite failure modes (`sweep_h.py`).
- **OCR alternatives were rejected.** A multi-engine A/B (tesseract vs
  RapidOCR/CoreML) and an Optuna hyperparameter search over preprocessing
  (`experiments/ocr_comparison/`) did not beat the simple tesseract baseline
  end-to-end — the robust axis fit already absorbs residual OCR error, so OCR is
  not the accuracy bottleneck. The simpler, dependency-light baseline ships.
- **A y=0 baseline-marker detector was designed and rejected**
  (`experiments/notes/y0-baseline-marker-recovery-design.md`) — see §4.

## 4. Known limitations

- **Markers on the y = 0 baseline** are flattened against the x-axis and fuse
  with the horizontal spine + near-horizontal connecting line, so the surviving
  top half rarely forms a regional maximum (the dominant miss mode — confirmed:
  18/19 misses have sufficient distance but fail the prominence step, not the
  thickness gate). Two sub-cases differ sharply:
  - The **origin** point (`x = x_min, y ≈ 0`), present in every growth curve, is
    now **recovered FP-free** by the corner detector (§2.1.4): recall rose from
    **33% → ~96%** with **zero** added false positives on synthetic *or* the 44
    real images (verified by controlled before/after).
  - **Interior-x y = 0 markers** remain at a **~65% floor**. The still-missed ones
    are tiny (r ≤ 5 px) and information-theoretically near-identical to the axis
    line; any recovery aggressive enough to catch them re-admits the axis line and
    spawns false positives at curve-**liftoff** junctions (visually confirmed on
    real images, and *unmeasurable* there for lack of marker ground truth). A
    full-baseline detector was prototyped and **rejected** for exactly this — a
    ~2 pt recall gain is not worth an unbounded, unverifiable precision cost.
    Full record (root cause, the four rejected recovery experiments, and the
    controlled before/after): `experiments/notes/origin-spine-recovery-investigation.md`.
  Because a marker at y ≈ 0 is a **low-value, low-signal** point (tumour volume
  ≈ 0), the residual interior misses barely affect the curve (§5, trajectory
  fidelity).
- **Thin / concave marker shapes** (✕, star) and **very small markers** (4–5 px)
  form weaker distance-transform blobs and have lower recall (§5).
- **A marker fused with the very top frame line** can rarely be missed.
- **Calibration depends on legible tick labels**; an image where OCR reads too
  few ticks may fail to calibrate. Gross scale-misreads are largely caught by the
  range check (values out of the plotted axis range are rejected).

## 5. Validity — how accuracy was established

### 5.1 The core problem, stated honestly
The real images have **no ground-truth marker coordinates** — the underlying
study numbers are not available, and a human cannot click a marker centre more
precisely than the detector already places it. The internal check
(`tests/verify_system.py`: every image calibrates, residuals small, values
within the plotted range, non-trivial marker count) confirms **success and
self-consistency, not accuracy** — passing it (44/44 on the curated set) does
*not* prove the values are correct.

One important distinction: **axis calibration *can* be checked on real images,
but marker positions cannot.** The chart prints its own tick labels and
gridlines, so the fitted axis transform can be verified against that printed
"ground truth" (§5.5); and a RANSAC fit in which ~12 *independently* OCR'd tick
numbers agree on one line is itself strong evidence the scale is right (a wrong
scale would require many labels to misread mutually-collinearly — the one such
mode, uniform ×100 punctuation loss, is caught by the range check). Marker (x,y)
values have no printed counterpart, which is why their accuracy is established on
synthetic data. Accuracy is therefore established two ways, each used only where
it is trustworthy.

### 5.2 Synthetic ground-truth benchmark (`experiments/synthetic_benchmark.py`)
We render growth-curve graphs whose marker coordinates we **know exactly**, in a
style **measured from** the real `single_curve` images (full accounting in
`experiments/notes/generative-fidelity.md`, with a side-by-side figure;
parameters reproduced by `experiments/measure_real_style.py` and
`experiments/ocr_yield_probe.py`), run the **full pipeline**, and measure error
against truth. Latest run (N = 500, seed 7; `notes/synthetic_benchmark_results.md`):

- **Detection precision:** matched markers land **median 0.70 px** (p90 3.15 px)
  from the true centre.
- **Recall:** 92.0% overall (10219/11109); by shape **✕ 84.3% (worst) →
  diamond 96.4% / circle 95.3%**; small radius (4–5 px) 90.2%. By boundary case:
  right-edge bisected **94.0%**, **left-spine day-0 point (x=0, y>0) 98.8%**,
  **origin (x=0, y=0, quartered by both axes) 95.7%** (was 33% before the spine /
  corner recovery, §2.1.4), and **interior-x y = 0 baseline 66.7%** (the residual
  documented limitation — §4). False positives 253 / 10472.
- **Per-marker value error**, reported two ways. *Conditional* (the 439
  correctly-calibrated graphs — the method's precision when it works): end-to-end
  **X median 0.11 %, Y median 0.34 %** of axis span (p90 0.35 % / 0.73 %). *Full*
  (all 475 calibrated graphs, *including* the 36 gross misreads): medians
  essentially unchanged (X 0.12 %, Y 0.35 %), p90 0.52 % / 0.85 %, but **X max
  blows up to ~10⁹ %** — i.e. the gross-misread tail is real but rare (36/475)
  and barely moves the median/p90.
- **Calibration outcome** (reported as a first-class metric, *not* folded into
  precision): **439/500 correct, 36 gross misread, 25 hard fail.** The hard-fails
  produce no curve at all (§5.6) and are flagged `calibrated: False` in the
  output — a visible failure, not silent bad data.

### 5.3 Trajectory fidelity — the scientifically meaningful result
Per-marker recall is the wrong yardstick for scientific viability: a missed
y ≈ 0 point or a spurious marker barely moves the **growth trajectory**, because
the connecting line interpolates. Comparing the whole extracted curve (using
**all** detected markers, including false positives) to the known curve, again
conditional vs full:

- *conditional (correct calibration):* curve mean-absolute error **0.48 %** of
  y-span (p90 1.02 %); AUC error **0.41 %** of plot area; peak **0.40 %**;
  time-to-half-peak **0.2 days**.
- *full (all calibrated, incl. the 36 misreads):* curve MAE **0.49 %** (p90
  3.20 %, max 496 %); AUC **0.43 %** (p90 2.14 %); peak **0.42 %**.
- **whole-pipeline recovery over ALL 500 graphs, counting every failure** (the 36
  gross misreads *and* the 25 hard-fails that produce no curve): the growth-curve
  AUC is within 5 % of truth on **443/500 (89 %)**.

So although per-marker recall is 92.0 % (✕-markers 84 %, interior baseline 67 %), the
extracted **growth dynamics** — AUC, peak, timing — are reproduced to well under
1 % on the graphs that calibrate, and the *whole pipeline including all failure
modes* recovers the growth curve on 89 % of graphs. The recall limitations are
**scientifically negligible** for the quantities a growth study uses; this is the
strongest single validity result and is robust to false positives/negatives by
construction.

### 5.4 Why the synthetic number is trustworthy (and its limits)
- **Generative fidelity:** every generator parameter is sampled from the real
  images, not invented, and matched by an iterative render→compare→fix loop
  (`generative-fidelity.md`).
- **Functional fidelity, not just visual:** the synthetic OCR tick-yield is tuned
  to the real distribution (median ≈12 vs ≈12.5 readable y-ticks/axis), so
  calibration *difficulty* tracks real difficulty rather than being an artifact
  of over- or under-degraded labels.
- **Honest residual circularity:** the generator and the detector's assumptions
  share an author, so a synthetic-only result risks validating assumptions
  against themselves. The synthetic number is therefore the method's *intrinsic
  precision under a matched distribution*, not a standalone proof.

### 5.5 Human-overlay verification on real images
To close the circularity, the synthetic result is paired with human review of
**verification overlays on the real images** (`experiments/verify.py`): each
image is drawn with the *calibrated* grid (tick lines from the fitted transform)
and the detected markers labelled with their extracted (x, y). A human confirms,
on the **real distribution**, the two things a human is reliable at — that the
grid lands on the printed gridlines (value sanity) and that markers are found and
sit on the plotted points (recall/correspondence) — *not* sub-pixel precision,
where the detector beats the eye. Across representative samples the overlays
agree with the printed graphs.

### 5.6 Summary of the validity argument
| Question | Evidence | Limit |
|---|---|---|
| Sub-pixel **precision** / exact (x,y) error | synthetic benchmark (has ground truth) | modelled distribution; circularity |
| **Recall / value sanity** on the real distribution | human-overlay verification | qualitative; gridline-level |
| **Scientific viability** (growth dynamics) | trajectory fidelity on synthetic | inherits synthetic caveats |
| Does it run/self-consistent on all real images | `tests/verify_system.py` (44/44) | success ≠ accuracy |

No single line of evidence is sufficient alone; together they bound the method's
accuracy from complementary directions, with each used only where it is reliable.
