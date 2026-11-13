#!/usr/bin/env python3
"""
eval/level3_scenarios.py — the Level 3 scenario sets, split once, before any detector or fire tuning.

A scenario seed fixes: the rendered environment (ground texture, sun, haze, exposure, grade: `FrameSynth(seed)`), the FireSim
random seed, the ignition cell and the agents' seeds. Two sets:

  calibration   used for the camera/threshold/fire-spread calibration and for the pilot; may be debugged against
  heldout       sealed: used ONLY by the final paired evaluation; calibration tools refuse these seeds (`require_calibration`)

Leakage controls (checked by tests/test_level3_scenarios.py and by `verify()`):
  * disjoint seeds,
  * disjoint GROUND-TEXTURE slots: the test-split texture bank (6 slots) is partitioned, calibration slots {0,1,2},
    held-out {3,4,5}, so no ground texture is rendered in both sets,
  * disjoint ignition cells between sets (calibration: i+j even, held-out: i+j odd) and distinct within a set, all at Chebyshev distance >= 3 from the takeoff cell,
  * the held-out list is hash-pinned in `level3_scenarios.json`.
Not separated (stated limitation): the 16 test-split smoke sprites, the sky, and the detector weights are shared (the detectors
were trained on neither renderer split's test images; sprites are test-split by construction).

    python3 eval/level3_scenarios.py            # regenerate/verify (never silently rewrites the pinned file)
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PATH = os.path.join(ROOT, "eval", "level3_scenarios.json")
CAL_SLOTS, HELD_SLOTS = (0, 1, 2), (3, 4, 5)
N_CAL, N_HELD = 4, 8
# Calibration ignition cells, version 2 (2026-11-02). Version 1 (random cells >= 3 from takeoff, i+j even) put every fire outside the
# camera's reach: the pilot logged 0 camera-visible frames of burning cells (docs/experiments/level3_methodology.md section 10).
# Rule fixed BEFORE the post-fix pilot was run, applied to the geometry of one recorded BASELINE flight (the baseline ignores fire, so its
# flight does not depend on the ignition; eval/level3_visibility.py, local_data/level3/pilot2/vis_baseline_postfix.json):
#   eligible = even parity (keeps the cells disjoint from the held-out odd-parity cells), Chebyshev distance >= 2 from the takeoff cell
#   (not at takeoff), first geometrically visible at 60-300 s (early patrol, not in the first minute) and >= 4 perception frames in
#   which the plume base is inside the image (a realistic opportunity). Exactly four cells qualify; they are assigned to the four
#   calibration seeds in order of first-visible time. The ignition is simulation-side only: no agent sees it.
# The held-out ignitions are NOT changed here (pinned, hash-checked); they must be re-derived by the same rule from a baseline flight
# before any final evaluation (a separate, explicit step).
CAL_IGNITION_V2 = {1001: [2, 2], 1002: [3, 1], 1003: [1, 1], 1005: [0, 0]}
# Held-out ignition cells, version 2 (amendment 2026-11-09, made BEFORE any held-out run, from the same recorded baseline flight and the same
# visibility rule as above, parity odd): first geometrically visible 60-400 s (window widened from 300 s: only 3 odd cells qualify at 300 s),
# >= 4 visible frames, Chebyshev distance >= 2 from takeoff. Exactly four cells qualify: (3,2) 84 s/11 frames, (2,1) 172 s/4, (1,0) 196 s/11,
# (0,1) 320 s/12. Eight held-out scenarios therefore REUSE each cell twice, with different fire seeds and (by construction below) different
# texture slots; this relaxes the "distinct ignition cells within the set" control (cells stay disjoint from calibration by parity).
# Version 1 (random cells, pinned hash 558f7282...) is superseded and was never flown.
HELD_IGNITION_V2 = {2001: [3, 2], 2005: [3, 2], 2002: [2, 1], 2012: [2, 1], 2004: [1, 0], 2014: [1, 0], 2006: [0, 1], 2015: [0, 1]}
TAKEOFF_CELL = (5, 5)          # the SITL home (grid origin) is the corner of cells (4..5, 4..5); (5,5) is the reference cell
LAT0, LON0 = 21.2970, -157.8170


def texture_slot(seed: int) -> int:
    """Texture-bank slot of a scenario, without building any texture."""
    from synthgen.render import Renderer, bank_kinds
    from synthgen.scene_params import load_config, sample_scene
    from sim.frame_synth import DEFAULT_CFG_PATH

    cfg = load_config(DEFAULT_CFG_PATH)
    n = int(cfg["background"]["bank_size"]["test"])
    kinds = bank_kinds(n)

    class _Slot(Renderer):
        def _bank(self, split):
            return [(k, None) for k in kinds]

    sp = sample_scene(cfg, "test", int(seed), 0, is_positive=False)
    return _Slot(cfg, 0).bank_slot(sp)


def ignition_cell(seed: int, parity: int):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 31337]))
    while True:
        i, j = (int(x) for x in rng.integers(1, 9, size=2))
        if (i + j) % 2 == parity and max(abs(i - TAKEOFF_CELL[0]), abs(j - TAKEOFF_CELL[1])) >= 3:
            return [i, j]


def build():
    out = {"calibration": [], "heldout": []}
    for split, base, slots, n, parity in (("calibration", 1001, CAL_SLOTS, N_CAL, 0), ("heldout", 2001, HELD_SLOTS, N_HELD, 1)):
        seed = base
        while len(out[split]) < n:
            slot = texture_slot(seed)
            ign = (CAL_IGNITION_V2 if split == "calibration" else HELD_IGNITION_V2).get(seed)
            if ign is None:
                seed += 1
                continue
            if slot in slots and (split == "heldout" or ign not in [s["ignition"] for s in out[split]]):
                out[split].append({"seed": seed, "split": split, "texture_slot": slot, "ignition": ign, "fire_seed": seed})
            seed += 1
    return out


def digest(sc: dict) -> str:
    return hashlib.sha256(json.dumps(sc["heldout"], sort_keys=True).encode()).hexdigest()


def verify(sc: dict) -> None:
    cal, held = sc["calibration"], sc["heldout"]
    assert not ({s["seed"] for s in cal} & {s["seed"] for s in held}), "seed overlap"
    assert not ({s["texture_slot"] for s in cal} & {s["texture_slot"] for s in held}), "texture slot overlap"
    assert not ({tuple(s["ignition"]) for s in cal} & {tuple(s["ignition"]) for s in held}), "ignition overlap"
    for c in {tuple(s["ignition"]) for s in held}:       # a reused held-out cell must come with different texture slots and fire seeds
        grp = [s for s in held if tuple(s["ignition"]) == c]
        assert len({s["texture_slot"] for s in grp}) == len(grp) and len({s["fire_seed"] for s in grp}) == len(grp), "reused cell not varied"
    for s in cal + held:
        assert texture_slot(s["seed"]) == s["texture_slot"]


def load() -> dict:
    return json.load(open(PATH))


def require_calibration(seeds) -> None:
    """Calibration tools call this: it refuses any seed that belongs to the sealed held-out set."""
    sc = load()
    held = {s["seed"] for s in sc["heldout"]}
    bad = held & set(int(x) for x in seeds)
    if bad:
        raise PermissionError(f"held-out scenario seeds {sorted(bad)} may not be used for calibration or debugging")


def main():
    sc = build()
    verify(sc)
    doc = {"calibration": sc["calibration"], "heldout": sc["heldout"], "heldout_sha256": digest(sc),
           "rules": __doc__.split("Leakage controls")[1].split("Not separated")[0].strip()}
    if os.path.exists(PATH):
        old = load()
        if old["heldout_sha256"] != doc["heldout_sha256"] or old["calibration"] != doc["calibration"]:
            sys.exit("pinned scenario file differs from a regeneration: refusing to overwrite it")
        print("verified: pinned file matches regeneration")
        return
    json.dump(doc, open(PATH, "w"), indent=1)
    print("wrote", PATH)
    for s in sc["calibration"] + sc["heldout"]:
        print(s)


if __name__ == "__main__":
    main()
