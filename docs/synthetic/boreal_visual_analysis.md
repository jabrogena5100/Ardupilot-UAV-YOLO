# Boreal reference analysis (design input for the synthetic smoke generator)

Status: analysis only. No Boreal pixels are stored in this repository or used by the generator.

## What was analysed

- 14 Boreal frames (12 positives, 2 negatives), all drawn from the **train** split. Selection rule: 4 sites
  (Evo, Heinola, Karkkila, Ruokolahti) x about P10/P50/P90 of smoke-box area, distinct videos preferred;
  2 `Empty-Images` negatives. Exact list: `selection_manifest.json` in the local sample folder.
- `boreal_label_stats.json`: label statistics for **train and val only**. The test split was deliberately excluded
  so that test data does not influence the synthetic design.
- The frames are stored outside the repo (`~/boreal_design_sample/`), used only for visual analysis, and are
  never read by the generator.

Caveat: 14 frames is a small sample of 4,253 train+val images. Observations below are qualitative and describe
what the sample shows. The statistics are over all train/val labels.

## Quantitative label statistics (train: 3,479 images / 3,304 boxes; val: 774 images / 845 boxes)

| Statistic | Train | Val |
|---|---|---|
| Background (no-box) images | 260 (7.47%) | 0 (0%) |
| Images with exactly 1 box | 3,208 (92.2% of all, 99.6% of positives) | 764 (98.7%) |
| Images with 4-12 boxes | 11 images | 10 images |
| Image sizes | 4096x2160 (2584), 3840x2160 (802), 1920x1080 (93) | 4096x2160 (640), 3840x2160 (134) |
| Mean image aspect | 1.866 (about 16:9) | 1.876 |
| Box area / image area, P5 / P25 / P50 / P75 / P95 | 0.12 / 0.22 / 0.34 / 0.71 / 0.81 | 0.02 / 0.38 / 0.51 / 0.61 / 0.84 |
| Box width fraction, P5 / P50 / P95 | 0.23 / 0.62 / 0.85 | 0.10 / 0.61 / 0.89 |
| Box height fraction, P5 / P50 / P95 | 0.36 / 0.63 / 0.98 | 0.17 / 0.82 / 0.98 |
| Box centre x, P5 / P50 / P95 | 0.30 / 0.58 / 0.69 | 0.34 / 0.66 / 0.77 |
| Box centre y, P5 / P50 / P95 | 0.25 / 0.40 / 0.51 | 0.23 / 0.45 / 0.53 |
| Pixel aspect w/h, P10 / P50 / P90 | 1.01 / 1.62 / 2.56 | 1.21 / 1.49 / 2.04 |
| Boxes touching or clipped by the image edge | 30.4% | 43.0% |

Per-site median box area (train): Ruokolahti 0.76, Evo 0.33, Heinola 0.27, Karkkila 0.22.
Per-site share of train boxes: Ruokolahti 36%, Karkkila 23%, Heinola 22%, Evo 19%.
The pooled distribution is therefore a **mixture** of quite different scene types, not one population. Val has a
different mixture again (Karkkila median 0.59, Heinola 0.26), which is why val and test box sizes differ from train.

## Visual characteristics (from the 14 frames)

**Viewpoint**
- All frames are **oblique drone views**, not nadir. Two regimes appear in the sample:
  1. Near-horizon / high-oblique (Karkkila, Heinola P10 and P50, Evo P50): the horizon is visible and sky takes
     roughly 30-50% of the frame. Seen in 6 of the 12 positive frames and in both negatives.
  2. Steeper oblique looking down (Evo P10 and P90, Heinola P90, Ruokolahti): no horizon, ground fills the frame.
- A renderer that cannot show a horizon and sky would miss about half of the sample.

**Smoke appearance**
- Volumetric plumes with cumulus-like billowing cores, not flat sprites. Colours range from bright white through
  blue-grey to tan/beige (Evo P50, Heinola P50, Karkkila). The tan/white mixture occurs inside one plume.
- Opacity varies a lot: dense opaque cores plus thin translucent edges through which trees, ground and roads are
  clearly visible (Heinola P90, Ruokolahti).
- Shape depends on distance and wind:
  - tall near-vertical columns (Karkkila; box width/height about 0.6),
  - plumes leaning or drifting diagonally (Evo P10 and P90),
  - ground-hugging curtains of low smoke streaming off a burn line (Ruokolahti), with plume size ranging from a
    small column to most of the frame.
- Smoke is often seen against a cloudy sky (Evo P50 has cumulus clouds of similar brightness right beside the plume).

**Fire / flames**
- Flames are small and sparse when visible (orange specks along the burn line in Ruokolahti and a few in Evo P10);
  in most frames none is visible. They are **not labelled** (no fire class), which matches the project decision
  to render flames only as an optional, unlabelled, default-off feature.

**Background / context**
- Dense dark-green conifer forest with strong texture; lakes (dark blue, with sun glints); clearings and logging
  areas with brown ground, tracks and roads (Ruokolahti); distant forest turns blue-grey (atmospheric haze).
- Lighting is summer daylight. Exposure varies a lot, including backlit scenes where the forest is almost black
  against a bright sky (Evo P50, Heinola P50).

**Negatives (2 frames, both from the same video)**
- Pure forest plus sky or lake, with thin cloud and haze in one. Clouds and haze are natural hard negatives: a
  synthetic set without clouds could teach a detector that "bright white blob in the sky = smoke".

**Annotation convention (important for the label definition)**
- Boxes are **axis-aligned, loose, and judgement-based**. They enclose the visible smoke mass including thin haze and
  usually reach down to the burn base; they frequently extend over large areas that contain no smoke
  (Karkkila P90: the box spans far to the left of a narrow column, over empty sky and forest; Ruokolahti boxes run
  to the frame edge).
- A tight box around the visible smoke would therefore differ systematically from Boreal labels. This is a
  label-convention gap that is independent of image appearance.

## Implications for the generator (what is matched vs only approximate)

All parameters below will be exposed in `synthgen/configs/domain_rand.yaml` rather than hard-coded.

| Property | Plan | Match level |
|---|---|---|
| Classes | `0 = smoke` only | exact |
| Image size / aspect | 1280x720 (16:9) | exact aspect; resolution differs from the originals, but training is at 640 either way |
| Smoke instances per positive image | exactly 1 | matched (99.6% of Boreal positives); multi-box images (0.3%) not generated |
| Background images | about 7.5% in train; also in val/test | matched to Boreal train; Boreal val/test have none, so synthetic val/test will include some |
| Box area distribution | sample smoke size to approach the Boreal pooled percentiles (median about 0.34, wide upper tail); checked in validation | approximate |
| Box centre / edge clipping | placement chosen to approach cx median about 0.58, cy median about 0.40 and about 30% edge-touching boxes | approximate |
| Camera regimes | oblique, with and without a visible horizon | approximate; no real altitude/pitch metadata exists for Boreal |
| Scene mixture | archetypes: distant column near the horizon, mid-range wide smoke, steep close range with large smoke; weights set from the per-site box shares above | approximate; site is only a proxy for archetype |
| Smoke appearance | procedural billowing plumes with a variable-opacity edge, white/grey/tan colour range | **approximate and intentionally weaker**: flat billboards, no true volume or 3D occlusion |
| Background | procedural conifer-canopy ground texture, sky gradient, procedural clouds, haze, lake patches, exposure variation | approximate; the weakest realism point |
| Flames | optional unlabelled sprites, off by default | not matched by default |
| Box definition | threshold on the smoke alpha mask plus optional padding; both configurable (defaults to be chosen from the Boreal box-looseness seen here) | **not matched**: Boreal boxes are loose and subjective, synthetic boxes are exact; reported as a known label-convention gap |

## Open questions to flag

1. **Level 3 camera vs Boreal camera.** The Level 3 UAV config (`fov_deg: 78`, `tilt_deg: 30`, altitude 25 m)
   looks down at a 100 m grid cell from low altitude, while Boreal frames are oblique views from far higher up.
   This gap exists regardless of how the synthetic set is made. The POC targets Boreal-like geometry for
   comparability, and the camera model supports the Level 3 geometry as well.
2. **Meaning of `tilt_deg` and `fov_deg`.** Whether `tilt_deg: 30` is measured from nadir or from the horizon, and
   whether `fov_deg: 78` is horizontal or diagonal, is not stated in the repo. `sim/camera.py` makes both explicit
   parameters; the defaults must be confirmed before Level 3 use.
3. **Annotator noise.** Boreal labels are loose. Whether to emulate that (box jitter and padding) is an
   experimental choice, not a rendering one. Defaulting to exact boxes keeps the label noise out of the comparison;
   an ablation could add it later.
