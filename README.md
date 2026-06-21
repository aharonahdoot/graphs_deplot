# Line-graph data-marker extractor

Reads images of tumour growth-curve line graphs and extracts the **(x, y) data
values of the data markers** (the plotted measurement points), ignoring the
interpolating line between them. Designed to be reliable across the dataset
despite different background colours, line colours, marker symbols (circle /
square / triangle) and line weights (thin / bold).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install numpy opencv-python-headless pillow scipy scikit-image pytesseract
brew install tesseract            # OCR engine for axis labels (macOS)
```

## Usage

```bash
# one image, a list, or a whole directory
.venv/bin/python src/extract.py Growth_Curves_NCI_BRCA --out output --format both --overlay
.venv/bin/python src/extract.py path/to/chart.jpeg
```

Outputs (in `--out`, default `output/`):

| file | contents |
|------|----------|
| `<name>.csv` / `<name>.json` | extracted markers for one image: `x, y, pixel_x, pixel_y` |
| `all_markers.csv` | every marker from every image (`image, x, y`) |
| `summary.csv` | per-image status: marker count, calibration rms |
| `overlays/<name>.png` | detected markers drawn on the image (with `--overlay`) |

## How it works

1. **Marker detection** (`src/markers.py`) — segment the series by darkness
   relative to the cream background OR by saturation (colour-agnostic), then
   find markers as **prominence peaks (h-maxima) of the distance transform**.
   The connecting line is a thin ridge that ascends to its endpoint markers, so
   it is never a regional maximum and produces no false peaks — even for bold
   lines. The prominence threshold (h=1.5) sits between line-rasterisation
   "staircase" noise (~1.2) and overlapping-marker prominence (~2.0), so dense
   overlapping markers are separated.
2. **Axis calibration** (`src/calibrate.py`) — OCR the axis tick labels
   (tesseract) and fit a robust linear pixel→value transform per axis by
   **RANSAC** (the line through some pair of tick labels that the most other
   labels agree with, within a quarter tick step, then a least-squares refit on
   those inliers), tolerant of occasional OCR misreads.
3. **Extraction** (`src/extract.py`) — map each marker pixel to (x, y).

## Verification

```bash
.venv/bin/python tests/verify_system.py    # asserts all 44 images pass
```

Checks every image calibrates, calibration residuals are small, all marker
values fall within the plotted axis range, and a non-trivial number of markers
is found. Result: **44 images, 739 markers, 0 failures**.

`experiments/verify.py` produces human-readable **grid overlays**: it redraws
each image with the *calibrated* grid (green lines placed at tick values from the
fitted transform) and the detected markers (red circles labelled with their
extracted `x,y`). If the green grid lands on the chart's own printed gridlines
the calibration is right; if the red circles sit on the plotted points the
detection is right. Example on a `single_curve` image:

![Calibrated grid + detected markers on a single_curve image](experiments/notes/single_curve_verify_overlay.png)

The green calibrated grid coincides with the printed gridlines and the red
markers land on the plotted points; the near-zero baseline points are the
documented y=0 limitation below.

Markers bisected by a plot boundary lose the thickness the prominence test needs
and are recovered with three boundary-specific passes (see `METHODS.md` §2.1):
- **Right edge** (final study-day point): `series_mask` keeps the frame column so
  the half-symbol survives, and a de-dupe keeps the strongest peak there.
- **Left spine** (day-0 point, `x = x_min`): always present in a growth curve, so
  if none was detected in the spine band its thickest blob is recovered.
- **Origin corner** (`x = x_min, y ≈ 0`, quartered by *both* axes): recovered via
  a colour mask (drops the grey spine) + vertical opening (drops the thin line) +
  a thickness/roundness gate, confined to the corner so no false positives are
  added. This lifted origin recall **33 % → ~96 %** with zero added FPs (verified
  on synthetic and the 44 real images).

### Known limitations
- **Interior-x markers on the y=0 baseline** (mid-plot, volume ≈ 0) are detected
  ~67 % of the time. The still-missed ones are tiny and near-identical to the
  axis line; recovering them re-admits the line as false positives at curve
  lift-off points, so they are accepted as low-value noise (the day-0 baseline
  point *is* recovered, above). See `experiments/notes/origin-spine-recovery-investigation.md`.
- A marker fused with the very top frame line (a point at the chart's ceiling)
  can be missed in rare cases.

## Accuracy

Because the real graphs have no ground-truth coordinates, accuracy is
established on a **synthetic benchmark** whose style is sampled from the real
`single_curve` images and whose marker coordinates are known exactly. The
generator is matched to the real distribution (left = real, right = synthetic):

![Real vs. synthetic graph style](experiments/notes/real_vs_synthetic.png)

On N=500 synthetic graphs the full pipeline recovers the **growth trajectory**
(AUC, peak, timing) to well under 1 % of axis span on graphs that calibrate, and
recovers the growth-curve AUC within 5 % of truth on 89 % of all graphs counting
every failure mode. See [`METHODS.md`](METHODS.md) for the full validity argument
and `experiments/notes/synthetic_benchmark_results.md` for the raw numbers.
