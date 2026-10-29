# Boreal v2: clip-disjoint split and Level 2 re-evaluation (experiment record)

Date: 2026-10-29. Status: **split frozen and hashed (free work complete); GPU training and evaluation NOT yet run** (see section 6 for the blocker). Boreal v1 (`docs/experiments/real_boreal_v1.md`, `level2_v1*.md`, `level2_v2*.md`) is untouched and remains the historical baseline. **v1 and v2 numbers must not be compared as if they measured the same thing: train and test populations differ.**

## 1. Why a new split
The frozen `boreal_v1` split shares 3 source clips between train and test and 2 between train and val, through background-only `Empty-Images` frames (`level2_v2_audit.md`). v2 assigns whole **normalised source clips** to partitions, so no clip contributes frames to two partitions.

## 2. Clip identity and ambiguities
- Identity = file name with the site prefix and `_frameN` removed, case-folded. Empty-Images frames join the clip they came from. **31 clips** in total (4,954 frames); no normalised id occurs under two real sites.
- Only 31 clips exist, so every partition has few clusters (test: 8). Clustered uncertainty will be wide; this is a property of the data, not fixable by the split.
- Ambiguity found: different clips of one session are near-identical at frame level (e.g. `karkkila_dji_0007`/`0008`: 2,590 cross-clip frame pairs within 4 dHash bits; `ruokolahti_dji_0087`/`0088`: 114). They are probably consecutive segments of one recording. This cannot be resolved from filenames or 64-bit dHash alone (UNKNOWN: pixel-level truth). The search therefore **penalises cross-partition near-duplicate pairs**; the result is reported in section 4.
- Unlabeled non-Empty frames (5, possibly unlabeled smoke): kept with their clips as in v1 (train 5, val 0, test 0 in the table below); the test set contains none, so test metrics are not affected by them.

## 3. Construction (deterministic, no model output used)
Seeded steepest-descent with 400 random restarts over single-clip moves and pair swaps (seed 20261029); 392 distinct local optima; chosen objective 0.1193, next best 0.1195. The objective uses only frame counts, sites, background frames, box counts and cross-partition near-duplicate pairs.
Constraints: val and test 600-900 frames, >= 3 sites and >= 3 clips each, every site present in val or test, true negatives (Empty-Images frames) train >= 100 / val >= 30 / test >= 50, test negatives from >= 2 clips including >= 1 'mixed' clip (boxes and >= 10 background frames), no clip > 40% of test.
**Design history (honest record):** candidate 1 was generated with the first four constraint groups and its composition inspected. It was rejected because Evo was absent from val and test and 59 of its 61 test negatives came from one all-background clip. The last three constraints were added and the search rerun to give this split. No model was trained or evaluated on candidate 1. Alternatives with near-equal objective exist (top test-clip sets in `split_assignment.json`), so this split is one defensible choice, not the unique optimum.

## 4. Frozen split

| Partition | Frames | Clips | With boxes | Boxes | True negatives (Empty-Images) | Unlabeled non-Empty | Sites (frames) |
|---|---|---|---|---|---|---|---|
| train | 3462 | 18 | 3325 | 3480 | 132 | 5 | evo 642, heinola 762, karkkila 361, ruokolahti 1697 |
| val | 748 | 5 | 689 | 692 | 59 | 0 | evo 289, heinola 59, karkkila 400 |
| test | 744 | 8 | 679 | 690 | 65 | 0 | heinola 302, karkkila 372, ruokolahti 70 |

- **train** clips: `evodji_0001` (314), `evodji_0007` (328), `heinola_dji_0025` (201), `heinola_dji_0028` (267), `heinola_dji_0029` (106), `heinola_dji_0030` (143), `heinola_dji_0036` (45), `karkkila_dji_0004` (71), `karkkila_dji_0005` (171), `karkkila_dji_0006` (119), `ruokolahti_dji_0080` (341), `ruokolahti_dji_0081` (128), `ruokolahti_dji_0084` (23), `ruokolahti_dji_0085` (178), `ruokolahti_dji_0086` (328), `ruokolahti_dji_0087` (282), `ruokolahti_dji_0088` (328), `ruokolahti_dji_0089` (89)
- **val** clips: `evodji_0009` (289), `heinola_dji_0024` (59), `karkkila_dji_0007` (171), `karkkila_dji_0008` (171), `karkkila_dji_0010` (58)
- **test** clips: `heinola_dji_0033` (230), `heinola_dji_0038` (10), `heinola_dji_0048` (62), `karkkila_dji_0001` (69), `karkkila_dji_0002` (125), `karkkila_dji_0003` (44), `karkkila_dji_0020` (134), `ruokolahti_dji_0082` (70)

File-list SHA-256 (sorted image names joined by newline; materialised by `eval/runpod/boreal_v2_materialize.py`, which aborts on mismatch):

| Partition | SHA-256 |
|---|---|
| test | `9fa6dbc1d0eb9fa437e989be29afadc9a459f7850502f1c18c350379afcddc8f` |
| train | `5b23b199965dfc47372e6100f4d5bfd41d57ef5b0b16b5ed83e91db056245c25` |
| val | `8ef9e902bb24a720112bef32d20248d6aa33c17d67efc9771643daa87d252e2f` |

Harness views (pinned in `eval/datasets_boreal_v2.yaml`): `boreal_v2_test` 744 frames; `boreal_v2_test_pos` 679 frames with boxes; `boreal_v2_test_neg` 65 held-out true-negative frames (from `heinola_dji_0048` 47 and `karkkila_dji_0003` 18).

## 5. Integrity checks (recomputed from per-file records, `split_audit.json`, tests in `tests/test_boreal_v2.py`)
- Video-disjoint: **yes** (0 shared normalised clips in all three pairs).
- Exact (MD5) duplicates spanning partitions: **0** (dataset-wide duplicate groups: 0).
- Near duplicates (64-bit dHash, cross-partition frame pairs) by threshold: {"train&test": {"<=0": 0, "<=2": 3, "<=4": 22, "<=8": 381}, "train&val": {"<=0": 0, "<=2": 0, "<=4": 0, "<=8": 83}, "val&test": {"<=0": 0, "<=2": 0, "<=4": 0, "<=8": 16}}.
- Cross-partition pairs <= 4 bits by clip pair: {"heinola_dji_0028|heinola_dji_0048": 15, "heinola_dji_0048|ruokolahti_dji_0089": 7} (same-site recordings of one session: `heinola_dji_0028` train vs `heinola_dji_0048` test, `heinola_dji_0048` test vs `ruokolahti_dji_0089` train). Small, but a documented residual leak risk for those test clips.
- dHash is a coarse 64-bit hash of a 9x8 grayscale thumbnail: it flags candidates, it does not prove identity. Pixel-level confirmation is UNKNOWN.
- The scan (sha256 `8a4db032...0e8e`) is the committed integrity scan of `/workspace/boreal/boreal_yolo`; the pod re-verifies size, MD5, label presence and box counts of every source file against it before creating the split.

## 6. What was NOT done and why
- No GPU training or evaluation has run. Models required: **3 new Model B seeds (0,1,2) trained on `boreal_v2` train** (names `real_boreal_v2_seed{N}`, checkpoints `real_boreal_v2_best_seed{N}.pt`); **no Model A retraining** (its synthetic training data does not depend on the Boreal split; the existing pinned seeds 0-2 are only re-evaluated on the v2 test set); the v1 Model B checkpoints are not used for v2 claims (they were trained on clips that now sit in v2 test).
- **Blocker for completing the research from the cloud sandbox:** results live on the RunPod volume and a pod; this sandbox cannot read them back (REST API has no log or file endpoint; `*.proxy.runpod.net` is blocked by the network policy; the only RunPod-S3-looking credentials in the environment have unknown origin and were deliberately not used). A paid run whose results this session cannot read is not started without the owner choosing a return path (see handoff).

## 7. Planned run (ready, not launched)
One RTX 4090 pod in US-IL-1, volume `wildfire-datasets`, job `eval/runpod/boreal_v2_train_eval.sh` under `pod_supervisor.py` (stop-not-delete), GitHub watchdog loop run as the independent stop. Protocol per seed unchanged from the handoff (YOLOv8s COCO, 100 epochs, patience 100, imgsz 640, batch 32, deterministic, workers 8, AdamW lr0 0.002 momentum 0.9 warmup_bias_lr 0.0, ultralytics 8.4.173). Outputs under `/workspace/runs/boreal_v2/` (new; the script aborts if it exists): checkpoints, `train_config.json` per seed, harness cells with 1000-resample image and **clip-cluster** bootstrap, saved predictions and ground truth, `seed_aggregation.{md,json}`, `INTEGRITY.json` (existing checkpoints re-hashed, unchanged), `results_bundle.tgz`.
Estimated cost: about 2.5 h at <= $1.00/h = about $2.5 expected; worst case $5.53 (18,000 s hard timeout + 900 s boot + 120 s watchdog latency, $0.25 margin). Limits: $20 per experiment, $100 allowance (ledger about $9.3 spent). Exact dry-run command is in the handoff.

## 8. Limitations
- 31 clips only; test = 8 clips, so clustered intervals will be wide and the result depends on this one clip assignment (alternatives exist).
- Real-domain specificity is measurable only on 65 held-out background frames from 2 clips; train still contains 132.
- Cross-partition near-duplicate residue (section 5); same-session consecutive clips cannot be proven separate.
- Edge-touch mismatch of `synth_smoke_v1` vs Boreal is unchanged (Model A is a baseline).
- Pod inference provenance: the pod runs a bundled copy of the repo; `git.commit` is recorded from the launch environment but the commit is not pushed (by instruction), so independent reviewers need the offline artifacts.

