Archived one-off RunPod job scripts of the external-evaluation preparation (2026-11-10/11). Not part of the evaluation harness and not
unit-tested; each was run once on a short CPU pod under a once-only marker file and a hard in-container timeout, and its output is quoted
in docs/external_eval/readiness.md (sections 2b-2h) and dataset_audit.md. Paths (/workspace/...) are volume paths.

| script | what it did |
|---|---|
| fasdd_inspect.py | read-only zip inspection (size, CRC, SHA-256, structure, labels) |
| fasdd_validate.py | split, label and edge-touch statistics from the zip |
| fasdd_extract.py | extracted the 4,181 official test images + smoke-only labels (new directory, sealed after verification) |
| fasdd_screen.py | v1 Boreal overlap (used a hash definition different from the stored scan: superseded), screening sample, sheets |
| fasdd_overlap2.py | hash-definition study, identical hashing on both sides, pixel check, screening v2, clustering sensitivity |
| vol_cleanup.py | volume usage report and guarded deletion of FASDD_UAV.zip |
| fireman_pixcheck_zenodo_FAILED_blocked.py | tried to fetch FireMan frames from Zenodo on the pod; produced no output (the pod's address was effectively throttled) |
| fireman_refexport.py | exported 32x32 grey thumbnails and 160x120 JPEGs of all Boreal and FASDD test images for a local pixel comparison |

The results server used to bring files back is the one in eval/runpod/results_server.py (routes) with eval/runpod/fetch_results.py as the client.
