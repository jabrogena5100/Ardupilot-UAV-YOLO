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

## FASDD_UAV (Flame and Smoke Detection Dataset, subset of FASDD_CV / FASDD_UAV / FASDD_RS)

| item | status |
|---|---|
| Classes | "fire" and "smoke" boxes; UAV subset reported as 25,097 samples (search snippets, **unverified**) |
| Annotation formats | "four common dataset formats" (preprint abstract); names **open** |
| Box convention | "minimum bounding rectangle" (reviewer comment, **unverified**). Boreal boxes derive from SAM masks, so box tightness may differ and mAP50-95 would be affected |
| Image sources | "Internet, social media, and open-access fire datasets" and drones etc. (preprint abstract, **verified as abstract text**); specific datasets **open**. The preprint is **withdrawn**; the published version (Geo-spatial Information Science 2024) could not be fetched (HTTP 403) |
| License | Science Data Bank record not readable by fetch (JS page); the preprint page's CC BY 4.0 notice covers the journal website, not the data; a Roboflow mirror of FASDD_CV says CC BY 4.0 (**unverified**); authors' comments mention conditions for sharing a portion (**unverified**). **Open: must be read on the record before download** |
| Splits | **open** |
| Resolution, negatives, clustering | **open**; near-duplicate video frames are plausible for web/video-derived data |

**Decision: conditionally proceed with FASDD_UAV only**, because it is the only candidate known to carry smoke boxes. Proceeding is gated (below); if any gate fails, the external evaluation is **not run** and the failure is reported instead.

## Source overlap
- vs Boreal (Hakala et al., Scientific Data, Aug 2025, CC BY 4.0, Finland, DJI Phantom 4, 2022 burns; 4,954 images, 31 clips in our audit): no FASDD source list names it, and the temporal order (FASDD preprint 2022-2023) makes inclusion unlikely, but this is **not evidence**; it must be tested by pixel comparison (gate G3).
- vs synthetic `synth_smoke_v1`: none by construction (own generator, own textures); only domain-style similarity is possible.
- FASDD vs FLAME: FASDD claims "open-access fire datasets" as a source; FLAME may be in it. Irrelevant for the minimal protocol (FLAME is not used) but means "FASDD plus FLAME" would not be independent; the two must not be pooled.

## Domain differences to report as limitations
Heterogeneous sensors and altitudes; web/social-media imagery (compression, watermarks, framing); possibly urban or non-forest scenes and fire-dominant images; different smoke density and annotation style from Boreal's controlled restoration burns. Both models are **out of domain** on FASDD_UAV, so the comparison measures relative robustness to a third, uncontrolled domain shift, not "synthetic vs real" in isolation (Model B's training data is also drone smoke from a different distribution).
