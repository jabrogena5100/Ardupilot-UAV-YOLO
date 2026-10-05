# synth_smoke_v1: generation and validation record

Status (2026-10-05): **generation COMPLETE and validated.** No Model A training has started. The dataset is stored
outside the repository and is never committed. Design and rationale: `docs/synthetic/synth_smoke_v1_spec.md`.

## Identity

| Item | Value |
|---|---|
| Dataset ID | `synth_smoke_v1` (condition `Boreal-prior-matched`) |
| Frozen generator | commit `9cb0f1a006905dd80ba552a7a6767fba9cf62bfb`, annotated tag `synth-v1` (points at `9cb0f1a`) |
| Config | `synthgen/configs/domain_rand.yaml`, sha256 `95b2ed6d4812efe02cfdeea2800777ac2af6180a0ce27e06870c46b8b3ad4c62` |
| Base seed | 1001 |
| Image size / class | 1280x720, single class `smoke` (smoke only; no fire class, as in Boreal) |
| `metadata.jsonl` sha256 (from `manifest.json`) | `506e9050206a0f1cd5f0194f37b796630928cd404f84cf93578aa08f00566ccc` |
| Tarball sha256 | `264894b7e291211a4440a6aa8e07c0227fcb45ff9d9c8cd69e243419da28d159` (`synth_smoke_v1.tar`, 787,220,480 bytes) |
| Per-file hashes | `synth_smoke_v1.SHA256SUMS` (10,616 files); all verified on the pod and again after transfer |
| Environment | Python 3.10.18, numpy 2.2.6, opencv 4.12.0, pillow 11.3.0 (RunPod CPU pod, 2 vCPU) |

`manifest.json` records the commit (`versions.git.commit = 9cb0f1a...`, `synthgen_sim_dirty: false`). It does **not**
record `--max-attempts`; that value is documented here and in `gen_cmd.txt`.

## Split counts

| Split | Smoke | Background | Images | Labels |
|---|---|---|---|---|
| train | 3,219 | 260 | 3,479 | 3,479 |
| val | 774 | 0 | 774 | 774 |
| test | 701 | 350 | 1,051 | 1,051 |
| total | 4,694 | 610 | 5,304 | 5,304 |

Test views (reported separately, never merged): `test_pos` (701 smoke, `data_test_pos.yaml`) and `test_all`
(701 + 350 background, `data_test_all.yaml`).

## Exact generation command

Run on the pod from a checkout of `9cb0f1a` (clean tree), inside tmux, log to `/root/gen_full.log`:

```
cd /root/synth-v1 && /root/venv/bin/python -m synthgen.generate_dataset \
  --config /root/synth-v1/synthgen/configs/domain_rand.yaml --out /root/datasets/synth_smoke_v1 \
  --counts train=3219:260,val=774:0,test=701:350 --seed 1001 --dataset-id synth_smoke_v1 \
  --contact-sheet 24 --max-attempts 500
```

Started 2026-10-05 09:55 UTC, finished in 2,781 s (1.91 img/s), exit code 0.

### `--max-attempts 500` is a runtime parameter, not a code change

`--max-attempts` is an existing CLI argument of the frozen `9cb0f1a` generator (default 40, `generate_dataset.py`).
No generator, config or `sim/` code was modified. The first full run used the default 40 and aborted (below). For
`max_attempts >= 11` the edge-gate setting for every attempt is identical (`enforce = attempt < min(max_attempts-1, 10)`),
and scene sampling depends only on `(seed, split, index, attempt)`, so raising the ceiling changes only whether a hard
image is allowed to keep retrying; it does not alter any image that already succeeded within 40 attempts.

## First run failure and clean rerun

- Run 1 (default `--max-attempts 40`) generated train (3,479) and 737 val images, then aborted at
  `val_000737`: `no valid scene after 40 attempts`. No manifest was written; the output was not a valid dataset.
- Diagnosis (read-only, on the pod): a `wide_steep` scene. In attempts 0-39, 25 were rejected by `dense_frac_too_low`
  (`min_dense_frac` 0.04) and 15 by `no_box` (`min_alpha_pixels` 400); the edge and area gates never rejected anything.
  The first accepted attempt is index 65 (`attempts_used = 66`). It is a rare deterministic case, not a rendering fault.
- Run 2 (clean rerun, `--max-attempts 500`, output moved aside first) completed. `val_000737` succeeded with
  `attempts_used = 66`. This is also the **maximum** `attempts_used` in the dataset.
- `attempts_used` histogram: 1: 5,079; 2: 162; 3: 27; 4: 13; 5: 3; 6: 6; 8: 4; 10: 1; 11: 1; 12: 3; 14: 1; 16: 1;
  19: 1; 37: 1; 66: 1. Nine images are flagged `edge_gate_fallback` (generated without the edge gate); all hard images
  seen were archetype `wide_steep`.

## Deterministic comparison with the failed partial run

All 4,216 image/label pairs written before the crash (3,479 train + 737 val) are **byte-identical** in the rerun,
including `train_002142` (37 attempts). Their `metadata.jsonl` rows are identical except for `render_seconds`. A fresh
renderer also re-rendered `train_002142` at attempt index 36 to the identical JPEG bytes. The partial output was deleted
with the pod after this comparison.

## Validation

`eval/validate_dataset.py` returned `VALIDATION PASSED` on `data.yaml`, `data_test_pos.yaml` and `data_test_all.yaml`,
on the pod (original paths) and locally (path-rewritten scratch view, see caveat below). Among other things it checks that
image directories exist, that images decode, and that the test views agree with the manifest. The reports are in the
run-record directory (`validate_*.json`).

| Split | Images | Background | Boxes | Box area P50 | Edge-touch fraction |
|---|---|---|---|---|---|
| train | 3,479 | 260 | 3,219 | 0.351 | 0.760 |
| val | 774 | 0 | 774 | 0.361 | 0.756 |
| test | 1,051 | 350 | 701 | 0.358 | 0.773 |

All images are 1280x720. Local transfer verification: tarball hash matched, all 10,616 per-file hashes matched, and the
local per-split image/label counts equal the table above.

## Locations

| What | Where |
|---|---|
| Dataset (local, not in git) | `~/datasets/synth_smoke_v1/` |
| Run records | `~/datasets/synth_smoke_v1_run_records/` : `gen_cmd.txt`, `gen_full.log`, `gen_failed_run1.log`, `validate_*.json`, `synth_smoke_v1.tar`, `.tar.sha256`, `.SHA256SUMS` |
| Transfer leftovers | `~/datasets/_incoming/` (tarball copy, safe to delete) |
| RunPod pod `ju9568emxzxr1a` (`synth-v1-cpu-gen`) | **deleted** 2026-10-05; `list-pods` returned empty afterwards |
| RunPod volume | dataset **not yet uploaded** (needs separate approval with cost stated first) |

## Caveat: absolute paths in generated YAML and list files

The generator writes the pod's absolute paths into `data.yaml` / `data_test_*.yaml` (`path: /root/datasets/synth_smoke_v1`)
and into `lists/test_pos.txt` / `lists/test_all.txt`. The local copy was left byte-identical to the verified files, so
the YAMLs and lists do **not** resolve on this machine and will not on the RunPod volume either. Before local use or
upload, make path-rewritten copies (as done for local validation: replace the prefix in the three YAMLs and the two
lists) or generate relative ones; do not edit the hashed originals in place. Validation locally used scratch copies.

## Known limitation: edge-touch rate (needs review before Model A training)

Observed fraction of boxes touching an image edge is about **0.76 (train), 0.76 (val), 0.77 (test)**, against about
**0.30 in Boreal train** (POC v2 measured 0.80). The box-area distribution matches Boreal closely (P50 about 0.35 vs 0.34).
The spec (section 4, Option A) already recorded this mismatch as an accepted, documented domain gap of v1, with a
pre-declared diagnostic (Model A on Boreal test grouped by edge-touching vs not) and the rule that the result will not
be used to modify or regenerate `synth_smoke_v1`; any generator change is v2, reported as post-hoc. The spec's
geometric estimate was about 44% forced touching, and the measured rate is higher than that, so the project owner asked
for a review of the edge-touch distribution versus Boreal before any GPU money is spent on training. That review has
**not** been done, and nothing about edge-touch behavior was investigated or changed in this session.

## Limitations (carry into any write-up)

Smoke is a flat billboard, not a true volume; no real 3D occlusion. Synthetic labels are exact alpha-mask boxes, while
Boreal boxes are loose and human-drawn.
