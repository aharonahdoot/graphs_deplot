# Experiments

These scripts are the **method-development record** for the extractor in `../src`.
They are kept verbatim for transparency: they show how each design decision and
parameter in the shipping pipeline was arrived at empirically — **including the
approaches that were tried and rejected**. They are not part of the shipping
pipeline and are not required to run it (see `../README.md`).

Scripts assume they are run from the repository root and read images from
`Growth_Curves_NCI_BRCA/` (the 44-image study dataset) or `single_curve/`
(single-curve crops). Those image sets and the run outputs are **local-only and
not redistributed** (see `../.gitignore`); the scripts and result summaries are
kept here as the record. Outputs go under `experiments/out/` (gitignored).

```bash
.venv/bin/python experiments/<script>.py [optional/image/path.jpeg]
```

## How the decisions map to the shipping code

| Decision in `src/` | Evidence script(s) |
|---|---|
| Series segmentation by darkness **or** saturation | `exp1_probe.py`, `exp3_probe_colors.py` |
| Plot-frame detection via long dark lines | `exp1_probe.py`, `exp2_markers.py` |
| Marker isolation via distance-transform prominence | `exp2_markers.py` |
| Prominence threshold **h = 1.5** | `sweep_h.py` |
| Left-spine / origin-corner recovery of the clipped day-0 marker | `notes/origin-spine-recovery-investigation.md`, `../tests/test_spine_recovery.py` |
| OCR config (`--psm 11`, char-whitelist, grayscale-to-tesseract) | `exp_ocr.py`, `ocr_comparison/` |
| Keep tesseract over RapidOCR / tuned configs | `ocr_comparison/` (**rejected** alternatives) |
| Exclude *interior-x* y≈0 markers rather than force-detect them | `notes/y0-baseline-marker-recovery-design.md`, `notes/origin-spine-recovery-investigation.md` (**rejected** full-baseline recovery; corner recovery adopted) |
| Robust pixel→value calibration | `test_calib.py`, `batch_calib.py` |
| Whole-dataset robustness | `batch_markers.py`, `batch_calib.py`, `verify.py` |

---

## Exploration / probes

### `exp1_probe.py`
Probes a single image: dumps whole-image HSV statistics and attempts plot-frame
detection by eroding a dark-pixel mask with tall/wide line kernels.
**Outcome — adopted.** Confirmed the cream background sits at high V, the series
is either darker or more saturated, and the frame is recoverable as the
outermost long dark vertical/horizontal lines → `find_spines` in
`src/markers.py`. (Contains a deliberately-shown false start — a `row_sum`
computed on the wrong axis then recomputed — left in to document the fix.)

### `exp3_probe_colors.py`
Measures background, series, and frame pixel values across five deliberately
different images (orange / olive / red / two blacks) to design **adaptive,
color-agnostic** thresholds rather than hard-coded color ranges. Sweeps the
relative-darkness threshold `D ∈ {8,20,40}`.
**Outcome — adopted.** A fixed darkness cutoff fails across the color range, but
"darker-than-background-by-a-margin **OR** high-saturation" separates every
series → the `(S > 60) | (V < 110)` style mask in `src/markers.py`.

### `exp2_markers.py`
First single-image prototype: frame → series mask → distance transform →
threshold the transform → connected components → overlay.
**Outcome — partially adopted, then superseded.** Validated the
distance-transform idea (the line is a thin ridge; markers are fat blobs), but
its **fixed** threshold `T = max(2.2, 0.45·dmax)` was brittle on bold lines and
dense overlapping markers. This motivated the prominence (h-maxima) approach.

## Parameter tuning

### `sweep_h.py`
Sweeps the h-maxima **prominence** threshold `h ∈ {1.3,1.5,1.6,1.7,2.0}` on six
images chosen to stress opposite failure modes (thin clean, steep zig-zags,
dense overlapping, thick lines).
**Outcome — adopted, fixed h = 1.5.** `h = 1.5` sits in the stable band between
staircase rasterization noise (~1.2) and the prominence of genuinely
overlapping markers (~2.0): it neither invents peaks on bold/steep lines nor
merges adjacent real markers.

## OCR

### `exp_ocr.py`
Tests OCR of the axis-label strips with tesseract (`--psm 11`, numeric
whitelist); reports each token's text, pixel position, and confidence.
**Outcome — adopted as the shipping config.** Two findings carried into
`src/calibrate.py`: (1) feed **grayscale straight to tesseract** — a global OTSU
binarization erased thin label strokes and lost labels; (2) `--psm 11` + numeric
whitelist + a `conf > 20` floor is enough, because the downstream calibration
fit is robust to the occasional misread.

### `ocr_comparison/` — rejected OCR alternatives (kept as evidence)
A separate effort asked whether a better recognizer or a tuned preprocessing
config would improve calibration. **Both were rejected; the simple baseline
ships.** We keep the code and results deliberately — the simple method won an
honest comparison, and that is worth showing.

- `ab_ocr.py` — label-free A/B of OCR backends (tesseract vs RapidOCR /
  RapidOCR-CoreML) over the same image sample, scored by misread rate,
  leave-one-out tick-prediction error (% of tick step), inlier-tick count, and
  speed. Installs a runtime backend hook by swapping `calibrate._ocr_tokens`, so
  `src/` stays clean.
- `optimize_ocr.py` — Optuna (TPE) search over preprocessing/`psm`/`oem` to
  maximise exact-read accuracy, scored on **synthetic train / synthetic val /
  real hand-labelled holdout** kept separate to expose overfitting. The winning
  config is `../data/best_ocr_cfg.json` (local).
- `ocr_pipeline.py`, `ocr_backends.py`, `labelkit.py`, `accuracy_probe.py` —
  supporting infrastructure (configurable single-label pipeline, pluggable
  backends, synthetic label generator, production-path accuracy probe).

Illustrative `ab_ocr.py` run (12 single-curve crops, this branch):

| backend | calib | misread | LOO med | LOO p90 | ticks | sec/img |
|---|---|---|---|---|---|---|
| tesseract | 12/12 | 23.6% | 0.5% | 1.1% | 10.2 | 0.17 |
| rapidocr  | 12/12 | 17.4% | 0.6% | 1.5% | 11.1 | 1.11 |

**Reading:** RapidOCR reads slightly more labels correctly (lower misread), but
the **downstream calibration error is the same** (~0.5% of a tick step, LOO) —
the robust Theil-Sen + outlier-rejection fit absorbs tesseract's higher misread
rate — while RapidOCR is ~6× slower and pulls in a heavy ONNX runtime
dependency. The Optuna search likewise improved per-crop reads on synthetic data
but **did not beat the baseline on the real holdout** end-to-end. Conclusion:
**OCR is not the accuracy bottleneck**; the simple, dependency-light tesseract
baseline ships. (Numbers above are a small fresh sample for illustration; they
reproduce the direction of the larger development runs, not a fixed benchmark.)

> Run requirements: `ab_ocr.py` / `accuracy_probe.py` need `single_curve/`
> (local); `optimize_ocr.py` needs `data/holdout/truth.csv` + the synthetic
> generator and `optuna`/`tqdm`; RapidOCR backends need `rapidocr-onnxruntime`.

## Batch runs / whole-dataset robustness

### `batch_calib.py`
Runs `calibrate` over all 44 images and flags any whose calibration fails or
looks weak (rms > 5 or < 3 tick inliers per axis).
**Outcome — diagnostic.** Drove the robust calibrator (Theil-Sen slope,
median-residual intercept, outlier rejection) in `src/calibrate.py`.

### `batch_markers.py`
Runs `detect_markers` over all 44 images, writes an annotated overlay per image
to `experiments/out/markers/`, prints per-image diagnostics.
**Outcome — diagnostic.** Primary regression-spotting tool during iteration.

### `cmp.py`
Diffs two `batch_markers.py` runs by per-image marker count and flags changes.
**Outcome — tooling.** A/B harness to confirm a change helped target images
without regressing others.

### `verify.py`
End-to-end visual verification: draws the **calibrated grid** (tick lines at
values from the *fitted* transform, phase-aligned to the real ticks) plus
detected markers labelled with extracted `(x,y)`. If the drawn grid lands on the
printed gridlines and circles sit on markers, calibration and detection are both
correct. Writes to `output/verify/`.
**Outcome — primary human-verification method.** This is the overlay a human
reviews to judge accuracy without marker ground truth. **Caveat:** it shows
internal consistency and visual agreement, *not* measured error against known
coordinates — see the synthetic benchmark and `../METHODS.md`.

### `test_calib.py`
Spot-checks `calibrate` on six images with hand-noted expected axis ranges/steps
(comma thousands, large fonts, small steps).
**Outcome — regression test.** Confirms the fitted transforms match hand-read
axis specs on known-tricky cases.

### `montage.py`
Assembles per-image overlays into paginated montages for whole-dataset
eyeballing. **Outcome — tooling.** Review convenience only.

## Synthetic ground-truth benchmark & validity

Real marker coordinates are unobtainable at the needed precision (a human can't
click a centre more precisely than the detector), so accuracy is measured on
synthetic graphs with KNOWN coordinates, rendered in the measured real style.

### `synthetic_benchmark.py`
Renders growth-curve graphs (style sampled from `single_curve`) with known
(x,y) markers, runs the full `src` pipeline, and reports detection recall (by
shape/radius/category), per-marker value error, and **trajectory fidelity**
(curve / AUC / peak / time-to-threshold error — robust to false positives and
negatives). **Outcome — primary accuracy benchmark.** Results in
`notes/synthetic_benchmark_results.md`.

### `measure_real_style.py`
Samples `single_curve` and reports the dimensions, background gradient, marker
radius, line width, markers/image, and series colours. **Outcome — provenance:**
reproduces the values behind the generator's style constants.

### `ocr_yield_probe.py`
Compares OCR tick-yield (readable tick labels per axis) on real vs synthetic
images. **Outcome — provenance:** used to tune synthetic tick density so
calibration difficulty matches real (median ≈12 ticks).

### `notes/generative-fidelity.md` + `notes/real_vs_synthetic.png`
How each synthetic element was engineered from the originals, with a side-by-side
real-vs-synthetic figure.

### `notes/detection-failure-investigation.md`
Root-cause analysis of detection misses (weak distance-transform prominence on
thin/small/baseline-fused markers), the OCR scale-misread (punctuation
absorption), whether RANSAC handles it, and the trajectory-metric regression
(baseline-tick false positives + an unstable AUC normalisation).

## Notes

### `notes/y0-baseline-marker-recovery-design.md` — rejected approach
Design for recovering markers sitting exactly on the **y = 0** baseline, where a
marker is flattened against the x-axis and has no excess thickness for the
distance-transform/prominence test to fire on. An alternative "vertical extent
above the baseline" detector was designed and prototyped.
**Outcome — rejected; we accept excluding y≈0 markers.** The approach was
finicky and risked re-admitting the baseline/connecting line as false positives.
Because a marker at y ≈ 0 is a **low-value, low-signal** point (tumour volume
≈ 0), excluding it is acceptable and far safer than degrading detection
everywhere to chase it. This is recorded as a **known limitation** in
`../README.md` and `../METHODS.md` rather than shipped.
