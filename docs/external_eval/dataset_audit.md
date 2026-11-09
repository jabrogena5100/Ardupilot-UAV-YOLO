# External UAV wildfire dataset audit (pre-registration stage, 2026-11-09)

Status: **desk audit only. No external data has been downloaded, no model has been run on it.** Every fact is marked
*verified* (read from the dataset's own page by a fetch on 2026-11-09), *unverified* (secondary source or summary) or *open* (not found).
The existing Level 1-3 baseline (`docs/baseline/`, tag `baseline-L1-3-v1`) is untouched; all external outputs go under `docs/external_eval/` and `local_data/external_eval/`.

What our frozen detectors can be scored on: single class `0: smoke`, axis-aligned boxes, RGB, 640 px inference (`eval/metrics.py`). A candidate is only usable
if it offers smoke **boxes** (or masks reducible to boxes) with smoke annotation that is complete enough to count a missed or extra detection as an error.

## FLAME (Northern Arizona pile burns, Shamsoshoara et al.)

| item | FLAME 1 (doi 10.21227/qad6-r683) | FLAME 2 (doi 10.21227/swyw-6j78) |
|---|---|---|
| Labels | frame-level Fire / Non-Fire (39,375 train, 8,617 test) and pixel masks of **fire** for 2,003 frames (verified) | frame-level Fire/NoFire and Smoke/NoSmoke per RGB/IR frame pair; no boxes or masks stated (verified) |
| Smoke boxes/masks | **none** (verified) | **none**; Smoke label rule (search snippet, unverified): applied when at least half the RGB frame is filled with smoke |
| License | CC BY 4.0 per page (verified) | CC BY 4.0 per page; "requires an IEEE DataPort Subscription" (verified) |
| Size / scenes | seven raw videos, Arizona pile burns; page lists internal inconsistencies in sizes and in which video is FLIR vs DJI (verified) | 53,541 frame pairs claimed, RGB counts sum to 53,451 (page inconsistency, verified); seven video pairs of one prescribed fire |

**Decision: do not use FLAME for box-level (mAP, localisation) evaluation.** Reasons:
1. FLAME 1 has no smoke annotation. Using it as smoke-negative data would be invalid: pile-burn frames contain unlabeled smoke, so "FLAME 1 fire frame, no smoke label" is not a true negative and every detection there would be miscounted as a false positive.
2. FLAME 2's smoke label is frame-level and rule-defined (heavy smoke), not "any visible smoke". It cannot score localisation, and the label threshold differs from our box-level notion of smoke.
3. All FLAME frames come from a few burns (7 video pairs of one fire in FLAME 2): the effective number of independent scenes is single digits, so any interval would be dominated by clustering and a one-site shift.
A frame-level secondary analysis on FLAME 2 (max detection confidence per frame vs Smoke/NoSmoke, AUROC with video-clustered bootstrap) is possible but answers a different question, needs a subscription, and is **not** part of the minimal protocol. It may be added later as clearly labeled exploratory work.

## FASDD_UAV (Flame and Smoke Detection Dataset V9; FASDD = FASDD_CV + FASDD_UAV + FASDD_RS)

Primary source read on 2026-11-09: the Science Data Bank record `https://doi.org/10.57760/sciencedb.j00104.00103` (redirects to `scidb.cn/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda`;
the record's machine-readable JSON-LD and description were parsed from the served HTML, saved only in the session scratch area). The published paper
(Wang, Yue, Jiang, Yu, Tuo, Li, *Geo-spatial Information Science* 28(2), 511-526, 2025; citation as given by the record) **could not be read** (publisher/DOAJ page returned 403; the earlier ESSD preprint is withdrawn). Everything below marked *verified* is from the record text; paper-only details stay *open*.

| item | status |
|---|---|
| Version / access | V9 (`version 9.0.0`), first published 2022-08-02, `conditionsOfAccess: PUBLIC` (verified) |
| **License** | **CC BY-SA 4.0** (record JSON-LD `license` and the page's license text, verified). ShareAlike: any adapted material we *distribute* (re-labelled annotations, derived images) must carry CC BY-SA 4.0 with attribution and a change notice. Plan: keep images and remapped labels local and gitignored; commit only file-name lists, hashes and aggregate metrics. Cite the paper. |
| Conflicting third-party licenses | A Roboflow mirror of FASDD_CV lists CC BY 4.0 (unverified, mirror, different subset). A Hugging Face derivative corpus cites CC BY-SA 4.0. **The official record governs; mirrors are not used as data sources** (they may be re-split or re-processed). |
| UAV subset | `FASDD_UAV.zip`; 25,097 images, 36,308 fire instances, **17,222 smoke instances**; aspect ratios mainly 4:3 to 16:9 (verified). Whole-dataset size on the record: 82,125,550,221 bytes (**82.1 GB, all four zips**); the size of `FASDD_UAV.zip` alone and any checksum are **not on the public page** (file list API answers "User Not Login"; not bypassed) |
| Annotation formats | YOLO, VOC, COCO, TDML, in an `annotations` folder next to `images` (verified) |
| Classes | "fire" and "smoke" object labels (verified). Numeric ids in the YOLO files **open** until the class file is read; `external_prep.py remap` reads the ids and refuses unexpected class sets |
| Scene-level names | file names are prefixed `Fire`, `Smoke`, `FireAndSmoke`, `NeitherFireNorSmoke` (verified). Total dataset: 70,581 positive and 52,073 negative samples (UAV subset split **open**) |
| Official split | **random** train/val/test = 1/2, 1/3, 1/6 "within each label format" (verified); membership mechanism (folders vs list) **open**. Implies about 4,183 UAV test images |
| Same-scene / near-duplicate leakage | A random image-level split of web/video-derived data offers no protection against near-identical frames in different splits (inference, **not measured**). Matters here only for effective sample size (no model trains on FASDD); see readiness section 4 |
| Image sources of the UAV subset | record says "Internet, social media and open-access fire datasets" sources overall; UAV-specific source list **open** |
| Box convention | "minimum bounding rectangle" (reviewer comment on the withdrawn preprint, **unverified**); Boreal boxes derive from SAM masks, so box tightness may differ |

**Decision: conditionally proceed with FASDD_UAV only** (still the only candidate with smoke boxes). Gates and the exact next action: `readiness.md`.

## Source overlap
- vs Boreal (Hakala et al., Scientific Data, Aug 2025, CC BY 4.0, Finland, DJI Phantom 4, 2022 burns; 4,954 images, 31 clips in our audit): no FASDD source list names it, and the temporal order (FASDD preprint 2022-2023) makes inclusion unlikely, but this is **not evidence**; it must be tested by pixel comparison (gate G3).
- vs synthetic `synth_smoke_v1`: none by construction (own generator, own textures); only domain-style similarity is possible.
- FASDD vs FLAME: FASDD claims "open-access fire datasets" as a source; FLAME may be in it. Irrelevant for the minimal protocol (FLAME is not used) but means "FASDD plus FLAME" would not be independent; the two must not be pooled.

## Domain differences to report as limitations
Heterogeneous sensors and altitudes; web/social-media imagery (compression, watermarks, framing); possibly urban or non-forest scenes and fire-dominant images; different smoke density and annotation style from Boreal's controlled restoration burns. Both models are **out of domain** on FASDD_UAV, so the comparison measures relative robustness to a third, uncontrolled domain shift, not "synthetic vs real" in isolation (Model B's training data is also drone smoke from a different distribution).
