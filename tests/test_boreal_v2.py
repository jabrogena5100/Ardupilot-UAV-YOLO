"""Boreal v2 split: frozen-artifact consistency, reproducibility of the construction, and the pod-side materialiser (fake tree)."""
import gzip
import hashlib
import json
import os
import sys
import tempfile
import unittest

import yaml

from eval import boreal_v2_split as B
from eval.boreal_split_audit import parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "docs", "experiments", "boreal_v2")
SCAN = os.path.join(ROOT, "docs", "experiments", "level2_v2_audit", "boreal_scan.json.gz")
sys.path.insert(0, os.path.join(ROOT, "eval", "runpod"))
import boreal_v2_materialize as MAT  # noqa: E402


def j(name):
    with open(os.path.join(D, name)) as f:
        return json.load(f)


class FrozenArtifacts(unittest.TestCase):
    def test_scan_is_the_pinned_one(self):
        self.assertEqual(hashlib.sha256(open(SCAN, "rb").read()).hexdigest(), j("split_assignment.json")["construction"]["scan_sha256_expected"])

    def test_filelists_match_audit_hashes_and_cover_every_file_once(self):
        fl, audit = j("filelists.json"), j("split_audit.json")
        scan = json.load(gzip.open(SCAN))
        allnames = sorted(n for s in scan.values() for n in s)
        got = sorted(e["name"] for p in fl.values() for e in p)
        self.assertEqual(got, allnames)
        self.assertEqual(len(set(got)), len(got))
        for p in ("train", "val", "test"):
            self.assertEqual(MAT.list_hash([e["name"] for e in fl[p]]), audit["filelist_sha256"][p])

    def test_video_disjoint_recomputed_independently(self):
        fl = j("filelists.json")
        vids = {p: {parse(e["name"])[0] for e in fl[p]} for p in fl}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertEqual(vids[a] & vids[b], set(), (a, b))
        self.assertEqual(sum(len(v) for v in vids.values()), 31)

    def test_assignment_matches_filelists_and_constraints_hold(self):
        asg = j("split_assignment.json")["assignment"]
        fl = j("filelists.json")
        for p in fl:
            for e in fl[p]:
                self.assertEqual(asg[parse(e["name"])[0]], p)
        scan = json.load(gzip.open(SCAN))
        clips = B.clip_table(scan)
        st = B.stats(asg, clips)
        self.assertEqual(B.feasible(st, sum(c["frames"] for c in clips.values())), [])
        self.assertGreaterEqual(st["test"]["true_neg"], 50)
        self.assertEqual(set(st["val"]["sites"]) | set(st["test"]["sites"]), {"evo", "heinola", "karkkila", "ruokolahti"})

    def test_no_exact_duplicates_across_partitions(self):
        self.assertEqual(j("split_audit.json")["md5_groups_spanning_partitions"], 0)

    def test_harness_config_pins_match_frozen_test_lists(self):
        cfg = yaml.safe_load(open(os.path.join(ROOT, "eval", "datasets_boreal_v2.yaml")))
        t = j("filelists.json")["test"]
        views = {"boreal_v2_test": [e["name"] for e in t], "boreal_v2_test_pos": [e["name"] for e in t if e["n_boxes"] > 0],
                 "boreal_v2_test_neg": [e["name"] for e in t if e["name"].startswith("Empty-Images__")]}
        for k, names in views.items():
            self.assertEqual(cfg["datasets"][k]["n_images"], len(names), k)
            self.assertEqual(cfg["datasets"][k]["filelist_sha256"], MAT.list_hash(names), k)
        v1 = yaml.safe_load(open(os.path.join(ROOT, "eval", "datasets.yaml")))
        for m in ("model_a", "model_a_seed1", "model_a_seed2"):
            self.assertEqual(cfg["models"][m], v1["models"][m])          # Model A checkpoints are the existing, pinned ones
        for s in (0, 1, 2):
            n = "model_bv2" if s == 0 else f"model_bv2_seed{s}"
            self.assertIn(f"boreal_v2/real_boreal_v2_seed{s}/real_boreal_v2_best_seed{s}.pt", cfg["models"][n]["checkpoint"])
            self.assertNotIn("real_boreal_best_v1", cfg["models"][n]["checkpoint"])   # never collides with v1 names

    def test_v1_baseline_config_untouched(self):
        v1 = yaml.safe_load(open(os.path.join(ROOT, "eval", "datasets.yaml")))
        self.assertEqual(v1["datasets"]["boreal_test"]["filelist_sha256"], "d2a610deddfd0053e046efb42e9aaa9e22158164d7b89a0c63e4789d159fc41b")
        self.assertEqual(v1["models"]["model_b"]["sha256"], "1dc35ef293b6b35bb2dadaf5b6d5887ece89ca9d55a445bfdde258286c8355b6")

    def test_recheck_matches_audit_and_documents_limitations(self):
        r = j("recheck.json")
        self.assertEqual(r["n_pairs_le4"], sum(j("split_audit.json")["dhash_cross_partition_pairs_<=4_by_clip_pair"].values()))
        self.assertEqual(r["n_pairs_le4"], 22)
        self.assertEqual(len(r["test_frames_with_train_or_val_neighbour_le4"]), 6)
        self.assertIn("NOT established", r["interpretation"])

    def test_construction_is_reproducible(self):
        scan = json.load(gzip.open(SCAN))
        clips = B.clip_table(scan)
        near = B.near_pairs_by_clip(clips)
        asg, obj = B.search(clips, near)[0]
        self.assertEqual(asg, j("split_assignment.json")["assignment"])
        self.assertAlmostEqual(obj, j("split_assignment.json")["construction"]["objective_value"], places=9)


class BundleSize(unittest.TestCase):
    def test_boreal_v2_bundle_fits_runpod_env_limit_and_contains_everything_needed(self):
        import base64, io, tarfile
        import launch_job as L
        chunks, _ = L.make_bundle([], L.BOREAL_V2_REPO_PATHS)
        self.assertLess(sum(len(c) for c in chunks), L.BOREAL_V2_MAX_BUNDLE_B64)
        names = tarfile.open(fileobj=io.BytesIO(base64.b64decode("".join(chunks))), mode="r:gz").getnames()
        for p in L.BOREAL_V2_REPO_PATHS:
            self.assertIn("repo/" + p, names)
        # every module the job imports is shipped
        src = open(os.path.join(ROOT, "eval", "run_matrix.py")).read() + open(os.path.join(ROOT, "eval", "aggregate_seeds.py")).read()
        self.assertIn("from . import metrics", src)

    def test_retry_tag_isolates_outputs(self):
        import argparse
        import launch_job as L
        ns = L.boreal_v2_args(argparse.Namespace(run_name="", ports=[], run_tag="boreal_v2_try2"))
        self.assertEqual(ns.run_name, "boreal_v2_try2_train_eval")
        self.assertEqual(ns.extra_env, {"RUN_TAG": "boreal_v2_try2"})
        self.assertTrue(all("/boreal_v2_try2/" in a for a in ns.artifact))
        with self.assertRaises(SystemExit):
            L.boreal_v2_args(argparse.Namespace(run_name="", ports=[], run_tag="real_boreal"))

    def test_preset_fills_the_job_definition(self):
        import argparse
        import launch_job as L
        ns = L.boreal_v2_args(argparse.Namespace(run_name="", ports=[]))
        self.assertEqual((ns.run_name, ns.hard_timeout_s, ns.ports), ("boreal_v2_train_eval", 16200, ["8000/http"]))
        self.assertTrue(any("real_boreal_v2_best_seed" in a for a in ns.artifact))


class JobScriptLint(unittest.TestCase):
    """Static guarantees about eval/runpod/boreal_v2_train_eval.sh: everything it writes goes to NEW v2 paths; nothing existing is touched."""

    def setUp(self):
        self.src = open(os.path.join(ROOT, "eval", "runpod", "boreal_v2_train_eval.sh")).read()
        self.code = "\n".join(l for l in self.src.splitlines() if not l.strip().startswith("#"))

    def test_never_deletes_moves_or_overwrites(self):
        import re
        code = self.code.replace("mv -f $EXP/status.tmp $EXP/status.json", "")   # the one allowed rename: atomic status file inside the new export dir
        for pat in (r"\brm\b", r"\bmv\b", r"exist_ok=True", r"\bdd\b", r"mkfs", r"git (push|reset|checkout)", r"--force"):
            self.assertIsNone(re.search(pat, code), pat)

    def test_aborts_if_targets_exist(self):
        self.assertIn('[ -e "$p" ] && { echo "ABORT: $p exists (refusing to overwrite)"; exit 1; }', self.code)
        self.assertIn('for p in "$OUT" "$SPL2.partial"; do', self.code)
        # the frozen split may be reused only after its hashes are re-verified (read-only); it is never rewritten
        self.assertIn("verifying against the frozen hashes", self.code)
        self.assertIn("MISMATCH", self.code)

    def test_writes_only_under_v2_paths(self):
        import re
        self.assertIn("OUT=$RUNS/$TAG", self.code)
        self.assertEqual(re.findall(r"TAG=\$\{RUN_TAG:-(\w+)\}", self.code), ["boreal_v2"])
        for m in re.finditer(r"\bcp\s+(.*)", self.code):
            self.assertRegex(m.group(1), r"\$(OUT|RUN|EXP|SPL2)", m.group(0))
        self.assertNotIn("splits/boreal_v1", self.code)                 # v1 split is never even referenced
        self.assertNotIn("real_boreal_best_v1", self.code)
        self.assertIn("project=$OUT name=$NAME exist_ok=False", self.code)

    def test_protocol_is_unchanged(self):
        for flag in ("model=yolov8s.pt", "epochs=100", "patience=100", "imgsz=640", "batch=32", "deterministic=True", "workers=8",
                     "optimizer=AdamW", "lr0=0.002", "momentum=0.9", "warmup_bias_lr=0.0", "data=$SPL2/data.yaml", '"ultralytics==8.4.173"'):
            self.assertIn(flag, self.code)
        self.assertIn("for S in 0 1 2; do train_one $S; done", self.code)

    def test_all_artifact_classes_are_saved_and_hashed(self):
        for need in ("sha256sum $RUN/$CK", "train_config.json", "results.csv", "args.yaml", "results_bundle.tgz", "INTEGRITY.json",
                     "--bootstrap 1000", "split_manifest_materialised.json", "pip_freeze", "seed_aggregation"):
            self.assertIn(need, self.code + self.src)
        self.assertIn("--exclude='*.pt'", self.code)                    # checkpoints stay on the volume, never exported

    def test_serving_is_guarded_by_token_and_only_exports(self):
        self.assertIn('[ -z "$NO_SERVE" ] && [ -n "$RESULTS_TOKEN" ]', self.code)
        self.assertIn("results_server.py --root $EXP", self.code)
        self.assertNotIn("--root $OUT", self.code)
        self.assertNotIn("--root /workspace", self.code)


class Materialise(unittest.TestCase):
    def build(self, td):
        src = os.path.join(td, "src")
        scan = {"train": {}, "val": {}, "test": {}}
        fl = {"train": [], "val": [], "test": []}
        files = [("train", "Karkkila-Images__k_DJI_0001_frame1.jpg", b"aaa", "0 0.5 0.5 0.2 0.2\n"),
                 ("val", "Empty-Images__k_DJI_0002_frame3.jpg", b"bbb", None),
                 ("test", "Heinola-Images__h_DJI_0003_frame2.jpg", b"ccc", "0 0.4 0.4 0.1 0.1\n0 0.6 0.6 0.1 0.1\n")]
        for sp, n, data, lab in files:
            os.makedirs(os.path.join(src, "images", sp), exist_ok=True)
            os.makedirs(os.path.join(src, "labels", sp), exist_ok=True)
            open(os.path.join(src, "images", sp, n), "wb").write(data)
            if lab:
                open(os.path.join(src, "labels", sp, n[:-4] + ".txt"), "w").write(lab)
            nl = -1 if not lab else len(lab.splitlines())
            scan[sp][n] = [hashlib.md5(data).hexdigest(), len(data), [10, 10], "0" * 16, nl]
            fl[sp].append({"name": n, "source_split": sp, "has_label_file": nl >= 0, "n_boxes": max(nl, 0)})
        exp = {p: MAT.list_hash([e["name"] for e in fl[p]]) for p in fl}
        return src, scan, fl, exp

    def test_success_builds_symlinks_manifest_and_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            src, scan, fl, exp = self.build(td)
            dest = os.path.join(td, "out", "boreal_v2")
            os.makedirs(os.path.dirname(dest))
            rc, errs = MAT.materialize(src, dest, fl, scan, exp)
            self.assertEqual((rc, errs), (0, []))
            self.assertTrue(os.path.islink(os.path.join(dest, "images", "test", "Heinola-Images__h_DJI_0003_frame2.jpg")))
            self.assertTrue(os.path.exists(os.path.join(dest, "labels", "test", "Heinola-Images__h_DJI_0003_frame2.txt")))
            self.assertFalse(os.listdir(os.path.join(dest, "labels", "val")))        # background frame: no label file
            m = json.load(open(os.path.join(dest, "manifest.json")))
            self.assertEqual(m["partitions"]["test"]["boxes"], 2)
            self.assertIn("test: images/test", open(os.path.join(dest, "data.yaml")).read())
            self.assertFalse(os.path.exists(dest + ".partial"))

    def test_refuses_existing_destination_and_never_touches_it(self):
        with tempfile.TemporaryDirectory() as td:
            src, scan, fl, exp = self.build(td)
            dest = os.path.join(td, "boreal_v2")
            os.makedirs(dest)
            open(os.path.join(dest, "keep.txt"), "w").write("x")
            rc, _ = MAT.materialize(src, dest, fl, scan, exp)
            self.assertEqual(rc, 4)
            self.assertEqual(os.listdir(dest), ["keep.txt"])

    def test_detects_source_corruption_label_and_hash_mismatch(self):
        for case in ("md5", "box", "hash", "labelpresence"):
            with tempfile.TemporaryDirectory() as td:
                src, scan, fl, exp = self.build(td)
                n = "Heinola-Images__h_DJI_0003_frame2"
                if case == "md5":
                    open(os.path.join(src, "images", "test", n + ".jpg"), "wb").write(b"ccd")
                if case == "box":
                    open(os.path.join(src, "labels", "test", n + ".txt"), "w").write("0 0.4 0.4 0.1 0.1\n")
                if case == "hash":
                    exp["train"] = "0" * 64
                if case == "labelpresence":
                    os.remove(os.path.join(src, "labels", "test", n + ".txt"))
                rc, errs = MAT.materialize(src, os.path.join(td, "boreal_v2"), fl, scan, exp)
                self.assertEqual(rc, 3, case)
                self.assertTrue(errs)
                self.assertFalse(os.path.exists(os.path.join(td, "boreal_v2")), case)   # nothing published on failure

    def test_never_writes_into_source(self):
        with tempfile.TemporaryDirectory() as td:
            src, scan, fl, exp = self.build(td)
            before = sorted(os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs)
            MAT.materialize(src, os.path.join(td, "boreal_v2"), fl, scan, exp)
            self.assertEqual(before, sorted(os.path.join(d, f) for d, _, fs in os.walk(src) for f in fs))


if __name__ == "__main__":
    unittest.main()
