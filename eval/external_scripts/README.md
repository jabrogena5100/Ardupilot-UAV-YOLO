Local one-off scripts of the FireMan preparation (2026-11-11). Not tested; they carry hard-coded local_data paths. The tested code is
eval/fireman_prep.py and eval/external_stats.py.

  rangefile.py     seekable file over HTTP Range requests (reads a few MB of a 33 GB Zenodo zip)
  sampler.py       fetch evenly spaced FireMan frames, MD5 + dHash64 + 32x32 grey thumbnail (resumes; backs off on HTTP 429)
  analyze.py       overlap against Boreal and FASDD hashes, cross-video similarity, frame correlation
  screen_builder.py  100-frame stratified screening sample and contact sheets (seed 20261109)
  pixcheck_local.py  32x32 correlation of all sampled FireMan frames against all Boreal and FASDD thumbnails; pair sheets

Typical order: sampler.py -> analyze.py -> (pod: eval/runpod/external_eval/fireman_refexport.py, fetched with eval/runpod/fetch_results.py) -> pixcheck_local.py -> screen_builder.py
