# Synthetic smoke generator: POC report

Date: 2026-10-05. Status: **POC complete and validated; full-scale generation and Model A training NOT started.**

Scope: build the 2.5D synthetic smoke pipeline (`sim/camera.py`, `synthgen/`, `eval/validate_dataset.py`), generate a
180-image proof of concept, validate it automatically and visually, and measure throughput. No YOLO training, no
RunPod training, no change to the frozen Boreal split or `real_boreal_v1`.

## What was built

| Piece | File | Role |
|---|---|---|
| Projection | `sim/camera.py` | One pinhole camera: ground <-> pixel, horizon, footprint, cells in footprint, bbox base -> grid cell. Used by the renderer now and by `perception/detector.py` later. Builds on `mission/geo.py` and `fog/state.cell_index` (no copies). |
| Scene sampling | `synthgen/scene_params.py` | Seeded sampling of one scene from `configs/domain_rand.yaml`; scene = f(config, seed, split, index, attempt). |
| Renderer | `synthgen/render.py` | Procedural ground (tileable conifer-canopy / clearcut / lake textures), sky, clouds, haze, and billboard smoke puffs, all through `sim/camera.py`. Smoke size is calibrated to the sampled target box. |
| Labels | `synthgen/annotate.py` | YOLO line `0 cx cy w h` from the smoke alpha mask (threshold 0.08, 2% padding). |
| Dataset writer | `synthgen/generate_dataset.py` | Same YOLO layout as the frozen Boreal split + `data.yaml`, `metadata.jsonl` (full scene params, target vs achieved box), `manifest.json` (hashes, seeds, versions). |
| Validator | `eval/validate_dataset.py` | One validator for any YOLO dataset (unified annotation format) with a side-by-side against Boreal statistics. |
| Tests | `tests/test_camera.py`, `tests/test_synthgen.py` | 31 `unittest` tests: projection round trips, determinism, label/mask agreement, end-to-end generate -> validate -> reproduce, validator failure cases. |

All generation parameters are in `synthgen/configs/domain_rand.yaml`; nothing is hard-coded in the renderer.
No real imagery is read by the generator (a test guards against image-loading calls in `render.py`).

## POC dataset

`~/synth_poc_v1/` (outside the repo, not committed): seed 0, 1280x720, class `0 = smoke`.

| Split | Images | Positives | Background | Archetype mix (positives) |
|---|---|---|---|---|
| train | 120 | 111 | 9 (7.5%) | see `manifest.json` |
| val | 30 | 28 | 2 | |
| test | 30 | 28 | 2 | |

Each split uses its own seed space and its own ground-texture bank, so no ground texture is shared across splits.

### Automatic validation: PASSED (0 problems)

Checked: every image decodes and is 1280x720; every image has a label file and vice versa; every label line is
`cls cx cy w h` with class in `{0}`, values in [0, 1], positive size, box inside the image; no byte-identical image
across splits; manifest file-list and label hashes match; generation is byte-for-byte reproducible from the seed
(tested). Visual inspection of two contact sheets (`poc_v1_contact_sheet.jpg` here): boxes align with the visible
smoke, plumes lean/drift and billow, negatives include horizon/sky/cloud scenes.

### Label statistics vs Boreal train (informational)

| Statistic | Synthetic train | Boreal train | Verdict |
|---|---|---|---|
| Background fraction | 0.075 | 0.075 | matched |
| Boxes per positive image | 1 | 1 (99.6%) | matched |
| Image aspect | 16:9 | about 16:9 | matched |
| Box area / image, P5 / P50 / P95 | 0.135 / 0.374 / 0.775 | 0.121 / 0.338 / 0.815 | close |
| Box width frac, P5 / P50 / P95 | 0.25 / 0.58 / 0.83 | 0.23 / 0.62 / 0.85 | close |
| Box height frac, P5 / P50 / P95 | 0.41 / 0.69 / 1.00 | 0.36 / 0.63 / 0.98 | close |
| Box centre x, P50 | 0.59 | 0.58 | close |
| Box centre y, P50 | 0.45 | 0.40 | slightly low |
| Pixel aspect w/h, P50 / P95 | 1.41 / 2.25 | 1.62 / 3.12 | synthetic boxes less wide |
| **Boxes touching an image edge** | **0.81** | **0.30** | **NOT matched** |

Calibration of plume size to the sampled target box works: rendered box area is a median of 0.93x the target
(P10-P90 0.71-1.11). The edge-touch mismatch is mostly inherited from the sampled targets themselves, which already
touch an edge in most scenes (curtain 34 of 39, column 23 of 26). Boreal's per-scene-type edge-touch rates are unknown
(only the pooled 0.30 was computed), so it is not yet clear which archetype is wrong.

### Throughput (measured; single CPU core, 1.9 GB RAM machine)

| Item | Value |
|---|---|
| 180 images | 188 s, 0.96 img/s (includes building 12 ground textures, about 10 s) |
| Per-image render time | mean 1.02 s, median 0.80 s, P95 2.11 s, max 5.62 s |
| Rejected/re-sampled scenes | 3 of 180 needed more than one attempt |
| Peak memory | 429 MB |
| Disk | 27 MB for 180 images (about 0.15 MB each) |

Extrapolation: a Boreal-sized dataset (3,479 + 774 + 701 = 4,954 images) is about 85 minutes on this machine and about
0.75 GB. Scenes are independent, so generation can be split by index range across cores if needed (not implemented).

## Recommendation for the full dataset (needs approval; nothing generated yet)

Size: **3,479 train / 774 val / 701 test**, the same counts as the frozen Boreal split, with 7.5% background images in
each split. Same train size and the same 100 epochs at batch 32 give Model A the same number of gradient steps as
Model B, keeping the comparison fair. About 85 minutes locally.

Suggested configuration changes before generating:
1. Increase the ground-texture bank (train 6 -> 12, val/test 3 -> 4), because 6 textures reused over 3,479 images limits
   background diversity. Cost: about 10 s.
2. Decide the edge-touch question (below) and, if wanted, adjust archetype box placement in `domain_rand.yaml`.
3. Keep box definition (alpha threshold 0.08, pad 2%) unless an ablation on label looseness is planned.
4. Freeze split and seeds, record the config hash in `manifest.json` (already automatic), and add the dataset to
   `models/MODELS.md` once trained.

Decisions needed from you:
1. Approve the full-dataset counts above (or choose a different size).
2. Edge-touch: leave as a documented mismatch, or spend a cheap pod run to compute per-site edge-touch from the Boreal
   labels (labels only, a few cents) and retune placement before generating?
3. Whether to pull the Boreal-trained checkpoint's predictions on this POC (Real->Synth preview) before committing to the
   full set. That would need torch and ultralytics and a pod or another machine, so it is optional.

## Known limitations (state in any write-up)

- **Smoke is a set of flat billboards**, not a true volume; there is no 3D occlusion; flat ground with texture instead of
  trees; no lens distortion; no physically based light transport.
- **Renderer realism is the dominant risk**: the terrain reads as flight-simulator ground and the smoke as a cloud of
  puffs. A weak Synth->Real result may reflect renderer weakness rather than a domain-transfer finding.
- **Label convention differs**: Boreal boxes are loose, hand-drawn and sometimes include large empty areas; synthetic
  boxes are exact boxes of the smoke alpha at a threshold (with 2% padding). Not matched.
- **Edge-touching boxes**: 81% vs Boreal's 30% (see above).
- **Camera geometry is assumed**: Boreal has no pose metadata; `hfov` 72-86 degrees is a plausible wide-angle value. The
  repo's Level 3 config (`tilt_deg: 30`, `fov_deg: 78`, 25 m altitude) looks down from low altitude, which is a
  different regime from the oblique Boreal views. `tilt_deg` and `fov_deg` meanings are explicit parameters in
  `sim/camera.py` (pitch below horizon; horizontal or diagonal FOV) and must be confirmed before Level 3 use.
- **Multi-box images** (about 0.3% of Boreal positives) and **flames** (unlabelled in Boreal, off by default here) are
  not generated.
- **Smoke colour**: Boreal smoke is mostly white/grey with some tan; the tan share here is 10% by config and visibly
  more saturated than in the Boreal sample.
- **Throughput** is for a single core; textures are tileable procedural noise, which repeats at long range (hidden by haze).
