# Baseline (y=0) Marker Recovery — Design

## Problem

Data markers sitting at **y = 0** are flattened against the horizontal x-axis
baseline. Example: `579494-129-T_P3_gc2988.png`, the day-476 point. They are
missed by the current detector, and this recurs across the dataset wherever a
curve touches the baseline.

### Root cause (evidence)

`detect_markers` keys on the **distance transform** (local thickness) and a
**prominence** test (`h_maxima`, h=1.5): a marker must rise ≥1.5 above its
surroundings. A marker fused with the baseline has no excess thickness — at the
579494 origin the distance transform peaks at **2.2**, essentially identical to
the bare baseline / connecting-line value (~2.0) running alongside it. So
`h_maxima` correctly fires **0 peaks** there. The signal genuinely is not in the
distance transform; lowering the threshold would re-admit the whole baseline as
false positives.

A genuine y=0 marker differs from the bare axis in two measurable ways:
1. **Ink above the baseline** — a filled symbol leaves a localized vertical cap
   rising above the line at one x. The bare line is uniformly ~2px; tick marks
   sit *below* the line; tick labels sit in a separate band further down.
2. **The data curve descends into that x.**

We switch the measured dimension from *local thickness* to *vertical extent
above the baseline*, which is exactly where a baseline marker stands out.

## Approach A — baseline bump-profile

A self-contained function `_recover_baseline_markers(mask, dist, box, mr, lw)`
in `markers.py`, called at the end of `detect_markers` after the existing
`chosen` list is built. It **only adds** points in a thin band around the bottom
spine, so existing detection is untouched (additive ⇒ zero regression by
construction).

### Algorithm

1. **Band**: rows strictly above the bottom spine `b`: `[b - bandH, b)`, with
   `bandH = max(8, round(3*mr))` (adaptive to this chart's marker scale).
   Ignoring the spine row and below means descending tick marks and tick labels
   never contribute.
2. **Profile**: small vertical closing in the band (bridges 1px symbol gaps),
   then for each column `x ∈ (l, r)`, `h(x)` = height of the contiguous ink run
   rising from just above the spine.
   - bare axis line → `h ≈ 1`
   - connecting line skimming the baseline → low, smooth **ridge**
   - filled marker cap → localized **peak** above that ridge
3. **Peak detection (prominence principle, in 1-D)**:
   `scipy.signal.find_peaks(h, prominence=P, distance=D, height=Hmin)`
   - `prominence=P≈2` — peak must rise above the local connecting-line ridge,
     so a shallow ridge is rejected exactly as `h_maxima` rejects ridges in 2-D.
   - `distance=D≈mr` — no splitting one symbol into two.
   - `height=Hmin` ≈ ridge + 2 — kills 1px noise.
4. **Output**: each surviving peak → center `(x_peak, ~b)`. Dedupe against
   existing `chosen` via the existing `_merge_close` radius, then append.

### False-positive defenses (the core requirement)

1. Measure ink **only above** the spine → ticks/labels excluded.
2. **Reject any column with `h(x) ≥ bandH`** — a full-band-height run is a
   *vertical line* (y-axis spine, vertical gridline), not a marker cap. This
   cleanly separates the origin-corner marker from the y-spine beside it.
3. Restrict scan to plot interior `(l, r)`.
4. Optional guard (kept unless it costs recall): require the cap to connect to
   series ink rather than float free.

Charts whose curve never reaches the baseline yield an empty band → nothing
added.

## Validation (verification-grounding loop)

Correctness only shows when run, so:

- Run `extract.py` over a representative sample of `single_curve/` **before vs.
  after**; diff per-image `n_detected` in `summary.csv`.
- **Invariant: counts only increase, and every added point lies on the
  baseline.** Any image that loses a marker, or gains one *not* on the baseline,
  is a regression to investigate.
- Visually inspect the overlay of every changed image; confirm 579494 now picks
  up the day-476 / y≈0 star.
- User visually inspects overlays as the final acceptance gate.

## Scope

~30–40 lines, additive. Reuses existing `mr`/`lw` scale estimates and
`_merge_close`. No change to calibration or the existing 2-D detection path.

---

## Outcome (2026-06) — Approach A rejected; corner recovery adopted instead

Approach A above (1-D bump-profile along the whole baseline) was implemented and
benchmarked. **Rejected:** on 400 synthetic graphs it recovered ~32 markers but
added **3724 false positives** — the connecting line + baseline ridge is peaky
everywhere, so a 1-D prominence test fires across the whole baseline. Mirroring
and vertical-opening variants failed the same way (6637 / 445–3296 FPs). The
missing discriminator was **colour**: the series is saturated/dark, the spine is
mid-grey, so a colour mask removes the spine (the dominant FP source).

A colour + vertical-opening + thickness/roundness recovery brought FPs down to ~6
/ 400, but a full-baseline sweep still:
- bought only **+2 markers / 84** for interior-x y=0 (a genuine ~65 % floor: the
  residual misses are tiny markers ≈ the axis line), and
- added **unmeasurable real-image FPs at curve lift-off junctions** (no real
  marker ground truth to bound the precision cost).

**What shipped instead:** the recovery is confined to the **origin corner**
(`_recover_baseline` in `src/markers.py`), recovering the always-present day-0
point at **zero** added FPs (synthetic and real), while the interior-x baseline
stays at its honest floor. Full record:
`origin-spine-recovery-investigation.md`. This confirms the original instinct in
this doc — a general baseline force-detector re-admits the axis line — and bounds
exactly where recovery *is* safe (the corner, via the domain prior that a curve
always starts at `x = x_min`).
