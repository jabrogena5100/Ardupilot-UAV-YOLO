# FASDD_UAV external evaluation: readiness assessment (2026-11-09)

Scope of this step: documentation and tooling only. Nothing was downloaded from the dataset, no model was run on external data, no RunPod resource was used, no checkpoint or Level 3 artifact was touched
(`python3 eval/freeze_baseline.py --verify` passes after these changes). Read with `dataset_audit.md` and `protocol.md`.

## 1. Download availability, license, integrity metadata
| check | result |
|---|---|
| Record resolves | yes: DOI redirects (HTTP 302) to the Science Data Bank page (HTTP 200) |
| License | CC BY-SA 4.0, verified from the record (details and obligations: `dataset_audit.md`) |
| Public / open access | `conditionsOfAccess: PUBLIC` |
| Archive listing, per-file size, checksum | **not verifiable without a login**: the file-list endpoint returns `User Not Login`. No checksum is published on the public page, so integrity can only be (a) the site's own size figure once logged in, (b) zip CRC/`unzip -t` after download, and (c) our own SHA-256 recorded at first download and frozen in `eval/datasets_external.yaml`. This is weaker than a publisher checksum; say so in any write-up. |
| Download itself | **untested** (needs the owner's Science Data Bank session; not attempted) |

## 2. Storage
- Verified: whole FASDD = 82.1 GB (four zips). `FASDD_UAV.zip` alone is not published. Planning estimate **about 17 GB** by image share (25,097 / 122,634 x 82.1 GB), plausible range 8-35 GB; this is an estimate until the logged-in page shows the real size.
- Verified free space: `df` shows **17 GB free** on `/` (58 GB total, 40 GB used; `local_data` is 7.4 GB).
- Requirement: archive + extraction of the test subset + tool outputs is roughly archive size + 3 GB (test split about 4,183 of 25,097 images) + 1 GB headroom. At 17 GB that is about **21 GB > 17 GB free: not sufficient**. Extracting everything would need about 2x the archive size.
- Action for the owner: free or attach at least **30 GB** (more if the real size exceeds 17 GB), then point `local_data/external_eval/` there (symlink). Nothing was deleted by this assessment.

## 2a. Storage re-audit (2026-11-10, read-only: df, lsblk, du; nothing resized or deleted)
- `/` is now `/dev/mapper/ubuntu--vg-ubuntu--lv`, 144G total, 40G used, **98G free** (disk `sda` 147.4G, LVM on `sda3` 145.4G). The 58 GB / 17 GB-free figures in section 2 are superseded.
- `local_data/` is 7.4 GB; `local_data/external_eval/` does not exist yet. Same filesystem as the repo, so no symlink is needed.
- Requirement: archive (estimated 17 GB, range 8-35 GB) + test-subset extraction ~3 GB + outputs/headroom ~1 GB. Even at the 35 GB upper bound with a full extraction (~2x archive = 70 GB) the 98 GB is enough, but keep >= 15 GB free afterwards. **G0 storage: pass (98 GB >= 30 GB).**
- Recommended destination: `local_data/external_eval/raw/` (new empty directory, gitignored). Extract only the test split plus label files, with a Python `-I` script.
- Download probe: the file-list API still returns `User Not Login` (HTTP 200, code 20001). **G0b remains blocked on the owner's Science Data Bank login**; no workaround attempted.

## 3. Class mapping
- Record: two object labels, "fire" and "smoke". Target class for comparison = **smoke only** (our detectors have the single class `0: smoke`); fire boxes are dropped, never merged into smoke.
- Tool: `eval/external_prep.py remap` reads the dataset's own class list, **refuses** anything but exactly {fire, smoke}, writes smoke as class 0, drops fire lines, and reports counts (tested with both id orders).
- Consequences: the image-name prefix `Fire` means fire without smoke by the authors' scheme. Whether such images contain unlabeled smoke is exactly what the screening (section 6) measures; until then they are **not** counted as negatives. Prefix vs label consistency (e.g. `Smoke` images with no smoke box) is reported as a data-quality statistic.

## 4. Official split and leakage
- Verified: random split 1/2 : 1/3 : 1/6 per label format. Unknown: how membership is expressed, and whether the three label formats share one split (must be checked: the YOLO test list must equal the COCO and VOC test lists, else the "test" set is ambiguous).
- Because nobody trains on FASDD here, split leakage cannot inflate either model's score. It matters for statistics: near-duplicate frames in the test split inflate the effective sample size. Primary set = **the official test split** (authors' pre-existing partition; no selection by us). Procedure: compute dHash64 within the test split, form clusters as connected components at Hamming <= 4, report cluster count and size distribution, use clustered bootstrap by these groups. If the cluster count after grouping is below 20 (gate G4) the intervals are labeled image-level only.
- Also report the train-vs-test near-duplicate rate within FASDD as a descriptive data-quality number, but it does not change the evaluation.

## 5. Overlap with Boreal (exact and perceptual)
- Reference: `local_data/integrity/boreal_scan.json.gz` (sha256 `8a4db032...e8e`, recorded in `docs/experiments/level2_v2_audit.md`): MD5 and 64-bit dHash for all 4,954 Boreal images. **No Boreal images are needed locally.**
- Tool: `eval/external_prep.py overlap`. Exact: MD5. Perceptual: dHash64 (9x8 grayscale), minimum Hamming distance over three resampling filters and over each reference hash's bitwise complement (guards an unknown bit convention), flag at <= 4 bits; sensitivity at 6 and 8 bits reported. Flagged images are removed from the candidate list *before* inference and counted by Boreal split.
- **Known limits (must appear in any claim of "no overlap"):** (a) the original Boreal scan code and its resampling filter are not in the repo, so the dHash definition is matched by tolerance, not proven identical; (b) MD5 will not match re-encoded or resized copies; (c) dHash does not catch crops, rotations or heavy aspect changes; (d) FASDD JPEG recompression shifts hashes slightly. Precondition P1 (recommended, cheap): obtain about 50 original Boreal frames (public dataset, CC BY 4.0), recompute dHash with the tool, and confirm distance 0-2 against the stored scan values; this validates the convention. Not done.
- Temporal argument (Boreal published 2025; FASDD V9 record dates from 2022-2025) is not evidence of disjointness.

## 6. Annotation screening procedure (100 images, 30 spot-checked)
Reproducible and independent of model outputs.
1. Candidates = names in the official **test** split after G3/G5 exclusions.
2. `python3 eval/external_prep.py screen --test-list TEST.txt --seed 20261109 --out docs/external_eval/screening_sheet.csv` draws by SHA-256 ranking (input-order independent, tested): **Smoke 20, FireAndSmoke 20, Fire 30, NeitherFireNorSmoke 30** = 100. The 30-image spot-check subset (6/6/9/9) is drawn from those with an independent salt and marked in the sheet **before** anyone looks at an image. If a stratum has too few candidates the tool aborts (no padding).
3. For each image the screener records: smoke visible (y/n/unsure); every visible smoke region labeled (y/n/n.a.); verdict in {correct, missed smoke, phantom box, loose/tight box, unsure}. Smoke = any visible smoke plume, not only dense smoke; fire-only glow is not smoke; fog/cloud/dust ambiguous cases = unsure (reported separately, not forced).
4. Screener: Claude reads the image with the label overlay. **Owner spot-check: the 30 marked images**, independently, without seeing the screener's verdicts; report agreement as raw percent and Cohen's kappa on {correct, not correct}. Kappa below 0.6 means the screening is not trusted and the owner reviews more.
5. Output: per-stratum rates with Wilson 95% intervals. Quantity that decides the gates: visible-but-unlabeled smoke rate in `Fire` + `NeitherFireNorSmoke` (negative-labelled strata), reported with its upper bound.
6. The sheet, seed, list hash, and tool commit go in `docs/external_eval/gates.md`. No model is run during screening.

## 7. Frozen baseline
Unchanged and re-verified (`eval/freeze_baseline.py --verify`). New files live under `docs/external_eval/`, `eval/external_prep.py`, `tests/test_external_prep.py`; they are outside the pinned sets.

## Remaining go/no-go gates
| gate | state | what decides it |
|---|---|---|
| G1 License | **pass on record** (CC BY-SA 4.0 verified). Obligation: no redistribution of derived data; cite paper | none outstanding unless the owner wants to redistribute |
| G0 Storage | **blocked** | >= 30 GB free at `local_data/external_eval/` |
| G0b Download/integrity | **untested** | owner logs in, records `FASDD_UAV.zip` size (and any checksum shown); after download `unzip -t` passes and SHA-256 is recorded |
| G2 Label compatibility | not run | screening of section 6; negatives usable only if the unlabeled-smoke upper bound is acceptable (proposed <= 5%); otherwise report box metrics on positives only |
| G3 Boreal overlap | tooling ready, not run | zero (or removed-and-counted) hits at <= 4 bits; P1 convention check recommended |
| G4 Sufficiency | not run | >= 500 smoke-positive test images and >= 20 dHash clusters; else intervals labeled image-level |
| G5 RGB/UAV only | not run | confirm FASDD_UAV contains no thermal/IR or satellite images by file inspection (rule documented, never by detector output) |
| Split consistency | not run | YOLO/COCO/VOC test lists identical |

## Precise next action to begin external testing
1. **Owner:** sign in to Science Data Bank on the FASDD record, note the listed size of `FASDD_UAV.zip` (and checksum if shown), free or attach >= 30 GB, and download `FASDD_UAV.zip` into a new empty directory `local_data/external_eval/raw/` (untrusted data: extract only with a Python `-I` script, never run anything from the archive).
2. Then (Claude, no spend): `unzip -t`, record SHA-256, list structure, read the class file, derive the test list and check split consistency, run `remap`, `overlap`, cluster step and `screen`, write `gates.md`.
3. Only if all gates pass: freeze `eval/datasets_external.yaml` (file-list hash) and run the eval-only harness on the 6 checkpoints (CPU is sufficient for about 4,000 images at 640 px; no RunPod needed) via `eval/run_matrix.py` unchanged.
