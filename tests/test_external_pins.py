"""Consistency of the pinned external-evaluation files (eval/datasets_external.yaml and everything it points at). Nothing here runs a model."""
import hashlib, json, os, tempfile, unittest

import yaml

from eval import external_materialize as M
from eval import fireman_prep as P
from eval import run_matrix as RM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_DIR = os.path.join(ROOT, "local_data", "external_eval", "fireman", "annotations")
CFG = yaml.safe_load(open(os.path.join(ROOT, "eval", "datasets_external.yaml")))


def sha(path):
    return hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()


class TestPins(unittest.TestCase):
    def test_pinned_files_match_the_yaml(self):
        d = CFG["datasets"]
        for name in ("fireman_primary", "fireman_secondary"):
            self.assertEqual(sha(d[name]["manifest"]), d[name]["manifest_file_sha256"], name)
            self.assertEqual(sha(d[name]["video_metadata"]), d[name]["video_metadata_file_sha256"], name)
        f = d["fasdd_smoke_positive"]
        self.assertEqual(sha(f["image_list"]), f["image_list_file_sha256"])
        self.assertEqual(sha(f["clusters"]), f["clusters_file_sha256"])

    def test_filelist_pins_follow_the_runmatrix_definition(self):
        d = CFG["datasets"]
        for name in ("fireman_primary", "fireman_secondary"):
            man = json.load(open(os.path.join(ROOT, d[name]["manifest"])))
            names = [P.frame_name(v, f) + ".jpg" for v, f, *_ in man["frames"]]
            self.assertEqual(len(names), d[name]["n_images"]); self.assertEqual(M.filelist_sha256(names), d[name]["filelist_sha256"])
            self.assertEqual(P.names_sha256(n[:-4] for n in names), man["names_sha256"])
        f = d["fasdd_smoke_positive"]
        names = [x.strip() for x in open(os.path.join(ROOT, f["image_list"])) if x.strip()]
        self.assertEqual(len(names), f["n_images"]); self.assertEqual(len(set(names)), len(names))
        self.assertEqual(M.filelist_sha256(names), f["filelist_sha256"])
        self.assertTrue(all(n.startswith(("smoke_", "bothFireAndSmoke_")) for n in names))

    def test_models_are_the_six_frozen_checkpoints(self):
        base = json.load(open(os.path.join(ROOT, "docs", "baseline", "baseline_v1.json")))["checkpoints"]
        want = {"model_a": "A_seed0 ", "model_a_seed1": "A_seed1 ", "model_a_seed2": "A_seed2 ", "model_bv2": "B_v2_seed0 ", "model_bv2_seed1": "B_v2_seed1 ", "model_bv2_seed2": "B_v2_seed2 "}
        self.assertEqual(sorted(CFG["models"]), sorted(want))
        for m, prefix in want.items():
            (entry,) = [v for k, v in base.items() if k.startswith(prefix)]
            self.assertEqual(CFG["models"][m]["sha256"], entry["sha256"], m)
            self.assertTrue(CFG["models"][m]["checkpoint"].endswith(os.path.basename(entry["path"])), m)
        self.assertEqual(sorted(CFG["cells"]["models"]), sorted(want))

    def test_inference_settings_equal_the_2x2(self):
        base = yaml.safe_load(open(os.path.join(ROOT, "eval", "datasets.yaml")))["settings"]
        ext = dict(CFG["settings"])
        self.assertEqual(ext.pop("batch"), 1)                      # the one documented deviation (rectangular letterbox per image, 2 GB RAM)
        base.pop("batch")
        self.assertEqual(ext, base)

    def test_datasets_separate_and_secondary_not_in_default_cells(self):
        self.assertEqual(CFG["cells"]["datasets"], ["fireman_primary", "fasdd_smoke_positive"])
        self.assertNotIn("fireman_secondary", CFG["cells"]["datasets"])
        self.assertFalse(CFG["datasets"]["fireman_primary"]["inference_allowed"])
        self.assertEqual(CFG["status"], "used_unfrozen_not_preregistered")
        self.assertIn("preregistration", CFG["protocol_record"]["not_claimed"])

    def test_fireman_tiers_in_the_manifests(self):
        meta = json.load(open(os.path.join(ROOT, "docs/external_eval/fireman_video_metadata.json")))["videos"]
        for name, tier in (("fireman_primary", "verified_2023"), ("fireman_secondary", "unverified")):
            man = json.load(open(os.path.join(ROOT, CFG["datasets"][name]["manifest"])))
            self.assertTrue(man["videos"]); self.assertTrue(all(meta[v]["tier"] == tier for v in man["videos"]), name)
        prim = json.load(open(os.path.join(ROOT, CFG["datasets"]["fireman_primary"]["manifest"])))
        sec = json.load(open(os.path.join(ROOT, CFG["datasets"]["fireman_secondary"]["manifest"])))
        self.assertFalse(set(prim["videos"]) & set(sec["videos"]))
        self.assertEqual(len(prim["videos"]) + len(sec["videos"]), len(meta))

    def test_fasdd_cluster_file_covers_the_list_and_platform_counts(self):
        f = CFG["datasets"]["fasdd_smoke_positive"]
        names = [x.strip() for x in open(os.path.join(ROOT, f["image_list"])) if x.strip()]
        cl = json.load(open(os.path.join(ROOT, f["clusters"])))["images"]
        self.assertEqual(sorted(cl), sorted(names))
        counts = {}
        for v in cl.values():
            counts[v["platform_by_size"]] = counts.get(v["platform_by_size"], 0) + 1
        pl = f["platform_labels"]
        self.assertEqual(counts, {"tower_fixed_camera": pl["tower_fixed_camera"]["n"], "uav_nadir_1080p": pl["uav_nadir_1080p"]["n"], "ground_or_low_video_720p": pl["ground_or_low_video_720p"]["n"]})
        self.assertEqual(len({v["cluster_r6"] for v in cl.values()}), 680)

    def test_gate_parameters_are_sane(self):
        g = CFG["analysis"]["gates"]
        self.assertGreater(g["min_clusters_for_interval"], g["min_clusters_for_exact_test"] > 0)

    @unittest.skipUnless(os.path.isdir(XML_DIR), "FireMan annotation XML not present (local_data)")
    def test_manifests_reproduce_exactly_from_the_xml(self):
        meta = json.load(open(os.path.join(ROOT, "docs/external_eval/fireman_video_metadata.json")))["videos"]
        for name in ("fireman_primary", "fireman_secondary"):
            man = json.load(open(os.path.join(ROOT, CFG["datasets"][name]["manifest"])))
            self.assertEqual(P.verify_manifest(man, XML_DIR, meta), [])
            tiers = None if man["tiers"] == "all" else man["tiers"]
            again = P.build_manifest(XML_DIR, meta, man["per_video"], tiers)
            self.assertEqual(json.dumps(again, indent=1), json.dumps(man, indent=1))   # byte-identical serialisation

    def test_harness_accepts_the_view_layout(self):
        # a FASDD-style view built by the materializer loads through run_matrix.load_dataset with a matching pin
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src"); os.makedirs(src + "/images"); os.makedirs(src + "/labels_smoke_only")
            names = ["smoke_UAV1.jpg", "bothFireAndSmoke_UAV2.jpg"]
            for n in names:
                open(f"{src}/images/{n}", "wb").write(b"x"); open(f"{src}/labels_smoke_only/{n[:-4]}.txt", "w").write("0 0.5 0.5 0.2 0.2\n")
            lst = os.path.join(d, "l.txt"); open(lst, "w").write("\n".join(names) + "\n")
            r = M.fasdd_view(src, lst, os.path.join(d, "view"))
            cfg = {"images_dir": os.path.join(d, "view/images/test"), "n_images": 2, "filelist_sha256": r["filelist_sha256"]}
            ds = RM.load_dataset("t", cfg)
            self.assertEqual(ds["n_images"], 2); self.assertTrue(ds["pinned"])


if __name__ == "__main__":
    unittest.main()
