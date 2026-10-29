# Level 3 observation contract (PROPOSED, not implemented, not validated)

Status: **proposal for review.** Nothing here is measured. No Bayesian/log-odds update may be implemented or fed into
`PriorityPlanner` until the calibration data in section 6 exists and the numbers in section 5 are measured at *this*
observation level. `planning/priority.py` stays the placeholder (`max(confidence)` plus repeat bonus).

Why this exists: the Level 2 numbers are box-level (recall of ground-truth boxes, mAP) and image-level (false-positive rate on
empty images). A map update needs the likelihood of a *cell observation* given the cell's true state. Those are different
quantities, and neither can stand in for the other: a box-level recall of 0.95 says nothing about how often a *cell inside the
camera footprint that contains smoke* is reported smoke, and an image-level FP rate of 0.14 says nothing about the per-cell false
alarm rate (one false box can raise several cells, and one empty image contains many cells).

## 1. Observation unit

One **cell observation** is the tuple `(frame f, cell c)` where `c = (i, j)` lies inside the footprint of frame `f` on the
`N x N` coverage grid (bit index `i*N + j`, as everywhere else).

- **Camera frame.** One RGB image rendered/captured at UAV pose `p_f` (lat, lon, alt, heading, pitch) and time `t_f`. The
  detector sees only the image, never the pose or ground truth.
- **Footprint** `F(f)`: the set of cells whose centre projects inside the image **and** that are *resolvable*: the cell centre
  lies inside the image with a margin `m` pixels from every edge, and the cell's ground sample distance is below `gsd_max`
  (cells near the horizon are excluded rather than scored). Computed by `sim/camera.py` (the same projection that `synthgen/`
  uses; never forked). Cells outside `F(f)` produce **no** observation, which is information too (no update, not "negative").
  `m` and `gsd_max` are fixed before calibration and recorded in the run manifest.
- **Detector output** `D(f)`: the set of boxes with `conf >= tau` after NMS (IoU 0.7, max_det 300, imgsz 640, as in the harness).
  `tau` is a *single fixed operating threshold per model*, chosen on the calibration split (section 6), not on test.

## 2. Mapping boxes to cell observations

For each box `b in D(f)`, project the box's ground footprint to the grid with `sim/camera.py` (box bottom edge and sides are
projected through the pinhole model; smoke is a flat billboard, so the box footprint is the ground region under the plume as
seen, not a volume). Define for each `c in F(f)`:

- `y_hat(f, c) = 1` iff some `b in D(f)` has a projected footprint covering at least a fraction `a` of cell `c`'s area
  (default rule: cell centre inside the projected polygon; `a` is fixed in advance and the choice is a sensitivity analysis
  item, not tuned on test).
- `s(f, c) = max conf over covering boxes` (kept for diagnostics and for a possible later calibrated score; not used by the
  binary contract).

This is **cell-level matching**, not IoU matching. No tolerance radius is added: `Fog._is_correct` currently accepts a detection
within `detection_truth_radius` cells of a burning cell; that is a convenience metric for `detections_<tag>.csv`, not a
likelihood, and must not be used for calibration.

## 3. Ground truth for a cell observation

The detector has a single class, `smoke`, and the existing fog ground truth is the *burning* state. They are different sets
(smoke is advected and can sit tens of metres to hundreds of metres from the burning cells). Calibration therefore needs a
**smoke-presence truth** `z(t, c) in {0, 1}`:

- Synthetic Domain: `z(c) = 1` iff the cell lies inside the smoke footprint the generator itself placed at ground
  coordinates (the plume polygon, `target_box_xyxy`-equivalent in ground coordinates), at the render time. This exists by
  construction in `synthgen/` and is the same projection as the footprint code.
- Real Domain (Boreal): there is no ground georeferencing, so **cell-level truth does not exist for Boreal** and cell-level
  sensitivity/specificity cannot be measured on real data from the current artifacts. Boreal gives only image/box truth.
- Level 3 closed loop (SITL): `z` must be emitted by the simulator that places the smoke, with the plume polygon logged at
  each frame time. The cellular-automaton burning state alone is not smoke truth. A cell is labelled **ambiguous**
  (excluded from sensitivity/specificity, counted and reported) if it lies within `d_amb` of the plume boundary.

Truth is evaluated **at frame time** (`t_f`), never later (same rule as `Fog.ingest_detection`).

## 4. Temporal dependence

Observations are not independent draws:

- Consecutive frames of one pass see the same smoke with highly correlated detector errors (Boreal frames from one video are
  correlated; the Level 2 video-cluster bootstrap already shows this).
- Overlapping cells within one frame share the same box.
- Repeat visits to a cell by one or several UAVs under the same plume, wind and lighting are correlated through the scene.

Consequences for the contract: (a) the unit of independence for uncertainty is the **flight pass / scene / video**, not the
observation: all intervals are cluster bootstraps over passes (and, for synthetic data, over texture-bank slot and sprite bank,
which are shared; see `level2_v2.md` section 1.3); (b) a naive product of per-frame likelihoods for the same cell overcounts
evidence, so any map update must either (i) take at most one observation per cell per pass (with a configurable minimum
inter-observation time), or (ii) discount repeated observations by an effective-sample-size factor estimated from the
calibration data. Which of (i)/(ii) is allowed is itself a calibration outcome, not an assumption.

## 5. Quantities to measure (per model, per seed, per domain)

At a fixed `tau`, over resolvable, non-ambiguous observations `(f, c)`:

- Cell sensitivity `Se = P(y_hat = 1 | z = 1)` and cell specificity `Sp = P(y_hat = 0 | z = 0)`, with the false-positive
  likelihood `1 - Sp` stratified by what the empty cell contains: empty forest, burnt ground, cell adjacent to a plume,
  cells in a frame that contains another cell with smoke vs. frames with no smoke at all. A single pooled `Sp` is not allowed to
  stand in for the adjacent-to-plume stratum.
- Dependence of `Se` on slant range / GSD, altitude and plume size (cell footprint vs. box coverage), because `Se` is not
  constant over the footprint.
- Pass-cluster bootstrap 95% intervals (clusters = flight pass or scene; n clusters reported) and the across-seed spread; the
  two are reported separately and never merged.
- Calibration check of `s(f, c)` (reliability by bin) only if a continuous score is later wanted.

Level 2 box-level recall, image-level FP rate and mAP are **not** inputs to these quantities and must not be substituted for them,
even as "approximate" values.

## 6. Data needed (held out from everything used so far)

All new, none drawn from existing frozen test sets, with a calibration/test split frozen before any model is scored:

1. **Synthetic cell-observation set (feasible with existing generator):** a new dataset version (not `synth_smoke_v1`) rendered
   by the 2.5D generator along simulated UAV passes (varying altitude, heading, pitch, wind, and plume position), saving per frame
   the pose, the plume polygon in ground coordinates and the render time, and including many all-negative frames plus frames with
   smoke in a *different* part of the footprint (to measure cell-level specificity near plumes). Fresh texture and sprite banks for
   the held-out calibration passes, disjoint from any training bank.
2. **SITL passes with a known plume (Level 3 proper):** UAV passes over the sim with logged plume polygon and burning state;
   the same logging, disjoint seeds/scenarios from calibration.
3. **Real negatives (specificity on real forest):** Boreal currently has no usable test negatives (one unlabeled frame) and the
   split is frozen. A separate real-negative source is required (new empty-forest footage from sources not in Boreal train/val/test,
   with video-level provenance recorded). Until it exists, real-domain `Sp` is unmeasured; the train-split `Empty-Images` frames are a
   diagnostic for Model A only and, as documented in `level2_v2_audit.md`, share clips with val/test.
4. **Real positives with geometry (optional):** real frames with a known pose and a ground-truth smoke region would let `Se`
   be measured on real data. If unavailable, real-domain cell-level `Se` is declared unmeasured and Level 3 claims about Model B
   are restricted accordingly.

## 7. What may be implemented before calibration

Allowed: the footprint/box-to-cell projection function and its tests (`sim/camera.py` reuse), logging of `(f, c, y_hat, s)` and of
`z` in SITL runs, and the evaluation script that computes section 5 from those logs. Not allowed until section 5 is measured and
reviewed: any `mapping/probability_map.py` update rule, any use of `Se`/`Sp` values, and any change to `PriorityPlanner.score()`.
