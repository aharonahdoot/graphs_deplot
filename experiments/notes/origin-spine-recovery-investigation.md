# Investigation: origin / left-spine / y=0 baseline marker recovery

Record of the investigation that fixed detection of markers clipped by the plot
axes. Driven by the synthetic ground-truth benchmark
(`../synthetic_benchmark.py`); the shipping changes are in `src/markers.py`, and
the regression guard is `tests/test_spine_recovery.py`. Numbers below are from
the controlled experiments run during the investigation (seeds derived from
`np.random.default_rng(7)`, the benchmark's canonical seed) and the headline
`synthetic_benchmark_results.md` (N=500, seed 7).

---

## 0. Motivation

Markers that sit **on an axis** are bisected/quartered by the spine and lose the
local thickness the distance-transform prominence test keys on. The right-edge
case was already handled; the **left spine** and especially the **origin corner**
(`x = x_min, y ≈ 0`, quartered by *both* axes) were not. The user also supplied a
domain prior: **every growth curve has a marker at `x = 0`** (the day-0 reading),
though its `y` is not necessarily 0.

### 0.1 Synthetic coverage added (generator, `synthetic_benchmark.py`)

To measure the problem we first made the generator reflect reality:

- The **first study point is always pinned to the left spine** (`xs_val[0] =
  x_min`); its value is ~0 most of the time (origin) but lifted to a non-zero
  baseline reading ~30 % of the time (`0.05–0.20 × y_max`) → a marker bisected by
  the spine *alone*.
- New ground-truth flags `on_spine` (`|px − l| ≤ radius`) and `origin`
  (`on_spine ∧ y ≈ 0`), and new recall categories **`spine`** (x=0, y>0) and
  **`origin`** (x=0, y=0), reported separately from `base`/`edge`/`interior`.

**Baseline measurement (before any fix), N=200 seed 7:** origin recall **33.3 %**
(41/123) — confirmed visually: the origin marker survives the axis clipping as a
~quarter-blob fused with the connecting line, with no detection on it.

---

## 1. Root cause (evidence, not inference)

`detect_markers` finds markers as **h-maxima of the distance transform**
(`h_maxima(dist, 1.5) & (dist ≥ max(2.1, lw+0.8))`). Two distinct kill modes:

**(a) `series_mask` over-cropped the left.** The keep-window started at `left+3`,
erasing the **3 thickest columns** of any marker on the spine (the image is white
left of the spine, so those columns are the densest part of the surviving half).
Instrumenting one missed origin (`synth_000`, r=7): the surviving blob's local
`dist.max()` was only **3.0**, barely above the gate `lw+0.8 = 2.8`.

**(b) The prominence step, not the distance gate, is the dominant baseline kill.**
Per-marker attribution over 400 graphs:
- **Origin markers:** all **17/17** with local `dist ≤ gate` were missed (gate
  kill), and 10/36 with `dist > gate` were *still* missed (the quarter-blob fused
  with the rising line never forms a prominent regional maximum).
- **Interior y=0 baseline markers:** of 19 misses, **only 1** was a distance-gate
  kill; **18** had sufficient distance but **failed `h_maxima`** — fused with the
  horizontal spine + near-horizontal connecting line, they are not regional
  maxima. Misses concentrate at small radius (13/19 are r ≤ 5 px).

---

## 2. The fix (three parts, `src/markers.py`)

| # | change | rationale | FP cost |
|---|---|---|---|
| 1 | `series_mask` keep-window `left+3` → `left` | restores the 3 thickest columns of the spine marker; tick labels are at `x < left`, so they stay excluded | none (FP went *down*) |
| 2 | `_recover_left_spine` | domain prior: a curve always has a day-0 marker at `x=x_min`; if none was detected in the spine band, recover the thickest blob there (one marker, only where ink exists) | none by construction |
| 3 | `_recover_baseline` (origin corner only) | colour mask (drops grey spine) + vertical opening (drops thin line) + thickness/roundness gate | ~0 (corner-confined) |

### Stepwise effect on left-spine overall recall (origin + spine), N=80 seed 7
(`tests/test_spine_recovery.py`):

| stage | origin | spine (x=0,y>0) | overall |
|---|---|---|---|
| baseline (before) | 38.5 % | 56.1 % | 47.5 % |
| + step 1 (mask `left`) | 61.5 % | 87.8 % | 75.0 % |
| + step 2 (`_recover_left_spine`) | 89.7 % | 100 % | 95.0 % |
| + step 3 (`_recover_baseline`, corner) | 94.9 % | 100 % | 97.5 % |

---

## 3. The y=0 baseline question — what was tried, and why most was rejected

Goal: raise interior-x y=0 baseline recall **without adding false positives**.
Because the misses fail the *prominence* step (not the distance gate), simply
lowering the gate cannot help. Four recovery strategies were prototyped and
measured over 400 graphs (recovered = matches a previously-missed marker;
FP = matches no ground-truth marker):

| approach | idea | recovered | false positives | verdict |
|---|---|---|---|---|
| **1-D prominence** | `find_peaks` on the column-wise max-distance profile along the baseline | 32 | **3724** | rejected — the line/baseline ridge is peaky everywhere |
| **mirror below baseline** | reflect the strip above the baseline downward so a half-marker becomes a full disk | (88 % recall) | **6637** | rejected — doubles the connecting line too |
| **vertical opening** | open the mask with a vertical element to keep tall blobs, drop the thin horizontal line | 9–14 | 445–3296 | rejected — small half-markers ≈ spine fragments (height ~2–4 px) |
| **colour + morphology** | colour mask drops the **grey spine**; vertical opening drops the colour **line**; distance + roundness gates confirm a real marker | 12 | **6** | viable, but see §3.1 |

The colour mask is the key discriminator the geometry-only methods lacked: the
series is saturated/dark, the axis spine is mid-grey (`V ≈ 75`, low `S`), so a
`(S>60) | (dk>55)` mask removes the dominant FP source (spine fragments).

### 3.1 Why even the viable approach is corner-restricted

A full-baseline sweep of the colour+morphology recovery, controlled before/after
on 400 graphs (identical greedy matcher):

| variant | interior baseline | origin | synth FP | real-image markers (44 imgs) |
|---|---|---|---|---|
| OFF | 61.9 % | 92.2 % | 192 | 760 |
| **corner-only (`x ≤ l + 2.5·mr`)** | 61.9 % | **95.1 %** | **192** | **760** |
| full baseline | 64.3 % | 95.1 % | 192 (+12 by benchmark matcher) | 777 (**+17**) |

Two decisive facts:
1. **Interior-x y=0 baseline is at a ~62–67 % floor.** The full sweep buys only
   **+2 markers / 84** over OFF — the still-missed ones are tiny (r ≤ 5 px) and
   information-theoretically near-identical to the axis line.
2. **The full sweep adds real-image false positives we cannot measure.** On real
   images it added **+17 markers**; visual inspection of `824345-141-R_P2.jpeg`
   showed several are genuine baseline data points but ≥1 is a **false positive at
   the curve lift-off junction** (the rising line spawns an extra blob). Real
   images have **no marker ground truth**, so this precision cost is unbounded and
   unverifiable.

**Decision:** confine `_recover_baseline` to the origin corner. It captures the
full origin gain (the always-present day-0 point) at **zero** added FPs on
synthetic *and* real (760 → 760), and leaves the interior-x baseline at its honest
floor rather than paying an unverifiable precision cost for ~2 pts of recall. This
independently re-confirms the earlier rejection in
`y0-baseline-marker-recovery-design.md`.

---

## 4. Result (headline, N=500 seed 7)

| category | before | after |
|---|---|---|
| origin (x=0, y=0) | 33 % | **95.7 %** |
| left-spine (x=0, y>0) | ~47 % | **98.8 %** |
| interior-x y=0 baseline | ~66 % | **66.7 %** (unchanged — by design) |
| overall recall | 90.7 % | **92.0 %** |
| false positives | — | **no net increase** (synthetic 192→192, real 760→760, controlled) |

Verification: `tests/test_spine_recovery.py` (passes ≥ 90 %, currently 97.5 %);
`tests/verify_system.py` (44 real images, 0 failures, every value in range);
visual corner crops of synthetic and real overlays.

---

## 5. Reproduce

```bash
# headline benchmark (writes synthetic_benchmark_results.md)
.venv/bin/python experiments/synthetic_benchmark.py 500 --seed 7
# focused regression for the spine/origin categories
.venv/bin/python tests/test_spine_recovery.py
# real-image end-to-end sanity (no ground truth; success + range checks)
.venv/bin/python tests/verify_system.py
```

---

## 6. The top / final-point miss — two wrong fixes, then the right one

Symptom (user-reported, frequent): on plots cropped tight at the top, the
connecting line appears to "terminate early" and the final / highest point is not
detected.

**Wrong fix #1 — pad the top (proposed, rejected).** Hypothesis: too little white
padding above the plot. Implemented faithfully (detect insufficient top-white →
prepend white + a fresh top spine); tested on the confirmed cases: **detection
count unchanged.** Where a marker is genuinely clipped, its pixels were discarded
at render time, so adding canvas can't restore them; and two of the test images
already had white top-padding yet still missed the point. Padding is a *correlate*
of the problem, not a lever on it.

**Wrong fix #2 — a `_recover_terminus` pass (built, then reverted).** It took the
topmost *thick* blob in the right portion of the plot and recovered it. On the
synthetic set it doubled FPs (92→186) by picking the left-spine top; restricting
to the right portion zeroed that *on synthetic*. But a manual review of the ~700
real "recoveries" showed **almost none sat on an actual marker glyph** — the pass
was picking the point where the rising line simply **exits the top of the plot**
(an off-chart point whose value exceeds the y-axis maximum) and asserting a marker
there. A line terminus is not a data point. Reverted.

**Root cause (the real one).** `find_spines` takes the box top as the topmost
strong horizontal line. On a plot **cut off at the top image edge** (cream runs to
the top, no margin/frame), that line is the first **gridline** — one tick *below*
the true top. The box ends a tick early, so a real marker in that top tick band is
excluded by the `series_mask` keep region and missed; and the cut line stub at the
too-low box edge is what the terminus pass was hallucinating on.

**Fix (`find_spines`, gated by `_RECOVER_CUT_TOP`).** When the cream interior
extends above the detected top, walk the top up through the cream to the cut edge.
The highest marker is then inside the box and detected **normally** — and because
the normal detector requires a prominent glyph, a line merely *exiting* the top
(off-chart) yields nothing, so no hallucination. Framed / margined charts stop at
the white row above the frame and are unaffected.

Impact (single_curve, 3133 imgs): box-top correction recovers a real top marker on
**144** images (+1 each, 1 fewer; recovered points verified as actual glyphs);
reverting the terminus pass removed **~597** hallucinated points. Synthetic FP back
to 92 (terminus gone); real 44-image set and the spine/origin/border regressions
unchanged. Guard: `tests/test_find_spines_cut_top.py`. Review tooling:
`tools/build_recovery_review.py` + `tools/review_gui.py`.

Lesson: a connecting-line endpoint is *not* evidence of a data point. Recovery
must key on an actual marker glyph (thickness/prominence), and the fix for a
clipped peak is to get the **box** right, not to guess points at line ends.
