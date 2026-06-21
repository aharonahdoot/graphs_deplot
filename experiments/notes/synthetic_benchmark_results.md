# Synthetic ground-truth benchmark

N=500 graphs, seed=7. Style sampled from single_curve. Marker shapes: circle, square, triangle, diamond, star, x, plus. Every image JPEG-compressed (q54-66) to bake in artifacts; saved half .jpg / half .png.

## Calibration outcome (headline)
- usable calibration (both axes fit, correct scale): **439/500**
- gross OCR scale-misread (axes fit but wrong scale): 36/500
- hard fail (an axis could not be fit): 25/500

> 'Correct scale' can only be checked here because synthetic graphs have ground truth; on real images we can verify calibration *succeeded and is self-consistent* but cannot certify *accuracy* (the reason this benchmark exists). Calibration difficulty is matched to real: the synthetic OCR tick-yield is tuned to the real distribution (median ~12 readable y-ticks per axis vs ~12.5 real -- run `ocr_yield_probe.py`), so the success rate here tracks real difficulty rather than being a low-legibility artifact. The few gross misreads are largely **detectable** -- most produce out-of-range values the pipeline's range check rejects -- rather than silently corrupting output. Per-marker precision below is **conditional on a correct calibration**; trajectory figures use all detected markers there.

## Detection recall / false positives
- overall recall: **92.0%** (10219/11109)
- interior markers: 91.9% (9330/10150)
- right-edge (bisected): 94.0% (327/348)
- y=0 baseline (bisected): 66.7% (70/105)  _(documented limitation)_
- left-spine x=0, y>0 (bisected): 98.8% (245/248)
- origin x=0, y=0 (quartered by both axes): 95.7% (247/258)  _(hardest case)_
- false positives: 253 spurious / 10472 predicted
- matched detection pixel error: median 0.70px, p90 3.15px

### Recall by marker shape (thin/concave shapes form weaker blobs)
- x: 84.3% (1398/1659)
- star: 91.3% (1534/1680)
- square: 92.6% (1273/1375)
- triangle: 92.7% (1902/2052)
- plus: 92.7% (1460/1575)
- circle: 95.3% (1384/1453)
- diamond: 96.4% (1268/1315)

### Recall by marker radius (smaller = lower prominence)
- 4-5px: 90.2% (3185/3531)
- 6-7px: 94.7% (3453/3646)
- 8-9px: 91.1% (3581/3932)

## Per-marker value error (% of axis span)
Two views. **Conditional** = the 439 correctly-calibrated graphs (the method's precision when it works). **Full** = all 475 calibrated graphs *including* the 36 gross OCR scale-misreads (the 25 hard-fails produce no values and can't enter). The full tail is dominated by the misreads, most of which the range check would reject.
| source | X med | X p90 | X max | Y med | Y p90 | Y max |
|---|---|---|---|---|---|---|
| end-to-end, **conditional** | 0.11% | 0.35% | 6.83% | 0.34% | 0.73% | 13.95% |
| end-to-end, **full** | 0.12% | 0.52% | 992577452.5% | 0.35% | 0.85% | 1058.3% |
| calibration-only, conditional | 0.08% | 0.26% | 6.83% | 0.32% | 0.57% | 13.82% |
| calibration-only, full | 0.09% | 0.42% | 993199514.1% | 0.33% | 0.62% | 1059.5% |

## Trajectory fidelity (extracted curve vs true | robust to FP/FN)
The scientifically meaningful view: the whole extracted curve (ALL detected markers, including false positives, mapped through the fitted axes) vs the known curve, compared as interpolated trajectories. A missed/spurious marker only perturbs the interpolation, so this reflects whether the *growth dynamics* survive -- the quantities a study actually uses.
- **conditional (correct calibration)** — curve MAE median 0.48% (p90 1.02%, max 18.2%); AUC 0.41% (p90 0.76%); peak 0.40% (p90 1.96%); time-to-half-peak 0.2d (p90 1.7d)
- **full (all calibrated)** — curve MAE median 0.49% (p90 3.20%, max 495.9%); AUC 0.43% (p90 2.14%); peak 0.42% (p90 10.47%); time-to-half-peak 0.2d (p90 1.8d)
- **whole-pipeline recovery (ALL 500 graphs, incl. every failure):** growth-curve AUC within 5% of truth on **443/500** (89%); the rest are the 36 gross misreads + 25 hard-fails (which produce no curve).
