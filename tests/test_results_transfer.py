"""Results endpoint (server + fetch client) tests on loopback; no network, no RunPod."""
import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RP = os.path.join(ROOT, "eval", "runpod")
sys.path.insert(0, RP)
import fetch_results as F  # noqa: E402
import launch_job as L  # noqa: E402
import results_server as RS  # noqa: E402

TOKEN = "t" * 40


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Served:
    def __init__(self, td, files, secret_files=()):
        self.root = os.path.join(td, "export")
        os.makedirs(self.root)
        for n, data in files.items():
            open(os.path.join(self.root, n), "wb").write(data)
        for n, data in secret_files:   # present on disk but NOT in the manifest
            open(os.path.join(self.root, n), "wb").write(data)
        RS.write_manifest(self.root, list(files))
        self.port = free_port()
        self.res = {}
        self.t = threading.Thread(target=lambda: self.res.update(r=RS.serve(self.root, self.port, TOKEN, 20, host="127.0.0.1", poll=0.05)))
        self.t.start()
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        self.base = f"http://127.0.0.1:{self.port}"

    def req(self, path, token=TOKEN, method="GET"):
        h = {"Authorization": "Bearer " + token} if token is not None else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + path, headers=h, method=method,
                                                               data=b"" if method == "POST" else None), timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def stop(self):
        self.req("/ack", method="POST")
        self.t.join(10)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.s = Served(self.td.name, {"results_bundle.tgz": b"x" * 5000, "note.txt": b"hello"}, [("secret.txt", b"nope"), ("model.pt", b"w")])

    def tearDown(self):
        self.s.stop()
        self.td.cleanup()

    def test_requires_token(self):
        self.assertEqual(self.s.req("/manifest", token=None)[0], 401)
        self.assertEqual(self.s.req("/manifest", token="wrong" * 8)[0], 401)
        self.assertEqual(self.s.req("/f/note.txt", token=None)[0], 401)
        self.assertEqual(self.s.req("/manifest")[0], 200)

    def test_serves_only_manifest_files(self):
        self.assertEqual(self.s.req("/f/note.txt"), (200, b"hello"))
        self.assertEqual(self.s.req("/f/secret.txt")[0], 404)       # on disk, not in the manifest
        self.assertEqual(self.s.req("/f/model.pt")[0], 404)
        self.assertEqual(self.s.req("/f/export_manifest.json")[0], 404)

    def test_no_traversal_listing_or_root(self):
        for p in ("/", "/f/", "/f/..", "/f/../export_manifest.json", "/f/%2e%2e/etc/passwd", "/f/../../etc/passwd", "/etc/passwd", "/list", "//"):
            self.assertIn(self.s.req(p)[0], (404,), p)

    def test_read_only_methods(self):
        self.assertEqual(self.s.req("/f/note.txt", method="POST")[0], 404)
        for m in ("PUT", "DELETE", "PATCH"):
            try:
                urllib.request.urlopen(urllib.request.Request(self.s.base + "/f/note.txt", method=m, data=b"x",
                                                              headers={"Authorization": "Bearer " + TOKEN}), timeout=10)
                self.fail(m)
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 405, m)

    def test_token_in_query_works_and_manifest_hashes_are_right(self):
        code, body = self.s.req("/manifest?token=" + TOKEN, token=None)
        self.assertEqual(code, 200)
        m = json.loads(body)["files"]
        self.assertEqual(m["note.txt"]["sha256"], hashlib.sha256(b"hello").hexdigest())


class DynamicAndStatus(unittest.TestCase):
    def test_files_published_after_start_are_served_and_status_is_fixed_route(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "export")
            os.makedirs(root)
            RS.write_manifest(root, [])
            port = free_port()
            res = {}
            t = threading.Thread(target=lambda: res.update(r=RS.serve(root, port, TOKEN, 20, host="127.0.0.1", poll=0.05)))
            t.start()
            time.sleep(0.5)
            s = Served.__new__(Served)
            s.base = f"http://127.0.0.1:{port}"
            self.assertEqual(Served.req(s, "/status")[0], 404)                     # no status yet
            open(os.path.join(root, "status.json"), "w").write('{"phase": "train_seed_0"}')
            self.assertEqual(Served.req(s, "/status"), (200, b'{"phase": "train_seed_0"}'))
            self.assertEqual(Served.req(s, "/status", token=None)[0], 401)
            self.assertEqual(Served.req(s, "/f/late.txt")[0], 404)
            open(os.path.join(root, "late.txt"), "w").write("late")
            RS.write_manifest(root, ["late.txt"])
            self.assertEqual(Served.req(s, "/f/late.txt"), (200, b"late"))
            self.assertEqual(Served.req(s, "/f/status.json")[0], 404)              # status is not a downloadable manifest file
            open(os.path.join(root, "export_manifest.json"), "w").write("{broken")  # corrupt allowlist fails closed
            self.assertEqual(Served.req(s, "/f/late.txt")[0], 404)
            Served.req(s, "/ack", method="POST")
            t.join(10)

    def test_status_symlink_and_size_cap(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "export")
            os.makedirs(root)
            RS.write_manifest(root, [])
            open(os.path.join(td, "secret.txt"), "w").write("secret")
            os.symlink(os.path.join(td, "secret.txt"), os.path.join(root, "status.json"))
            port = free_port()
            t = threading.Thread(target=lambda: RS.serve(root, port, TOKEN, 20, host="127.0.0.1", poll=0.05))
            t.start()
            time.sleep(0.5)
            s = Served.__new__(Served)
            s.base = f"http://127.0.0.1:{port}"
            self.assertEqual(Served.req(s, "/status")[0], 404)
            os.remove(os.path.join(root, "status.json"))
            open(os.path.join(root, "status.json"), "w").write("x" * 200000)
            code, body = Served.req(s, "/status")
            self.assertEqual((code, len(body)), (200, RS.STATUS_MAX))
            Served.req(s, "/ack", method="POST")
            t.join(10)


class ManifestValidation(unittest.TestCase):
    def test_refuses_checkpoint_symlink_traversal_and_size_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "e")
            os.makedirs(root)
            open(os.path.join(td, "outside.txt"), "w").write("o")
            os.symlink(os.path.join(td, "outside.txt"), os.path.join(root, "link.txt"))
            open(os.path.join(root, "best.pt"), "wb").write(b"w")
            open(os.path.join(root, "ok.txt"), "w").write("ok")
            for name, ent in (("link.txt", {"bytes": 1, "sha256": "0"}), ("best.pt", {"bytes": 1, "sha256": "0"}),
                              ("../outside.txt", {"bytes": 1, "sha256": "0"}), ("ok.txt", {"bytes": 999, "sha256": "0"}),
                              ("sub/x.txt", {"bytes": 1, "sha256": "0"}), ("weights.safetensors", {"bytes": 1, "sha256": "0"})):
                json.dump({"files": {name: ent}}, open(os.path.join(root, "export_manifest.json"), "w"))
                with self.assertRaises(ValueError, msg=name):
                    RS.load_manifest(root)

    def test_short_token_refused(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "e"))
            RS.write_manifest(os.path.join(td, "e"), [])
            with self.assertRaises(SystemExit):
                RS.serve(os.path.join(td, "e"), free_port(), "short", 1)


class FetchTests(unittest.TestCase):
    def test_round_trip_verifies_hashes_and_acks(self):
        with tempfile.TemporaryDirectory() as td:
            s = Served(td, {"a.dat": os.urandom(300_000), "b.txt": b"b"})
            dest = os.path.join(td, "got")
            m = F.fetch(s.base, dest, TOKEN)
            s.t.join(10)
            self.assertEqual(s.res.get("r"), "acked")
            self.assertEqual(sorted(os.listdir(dest)), ["a.dat", "b.txt"])
            self.assertEqual(F.hashlib.sha256(open(os.path.join(dest, "a.dat"), "rb").read()).hexdigest(), m["a.dat"]["sha256"])

    def test_refuses_existing_dest_and_wrong_token_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            s = Served(td, {"a.dat": b"abc" * 100})
            os.makedirs(os.path.join(td, "exists"))
            with self.assertRaises(FileExistsError):
                F.fetch(s.base, os.path.join(td, "exists"), TOKEN)
            with self.assertRaises(ConnectionError):
                F.fetch(s.base, os.path.join(td, "x"), "w" * 40, retries=1, sleep=lambda x: None)
            open(os.path.join(s.root, "a.dat"), "wb").write(b"ZZZ" * 100)   # corrupt after the manifest was written (same size)
            with self.assertRaises(Exception):
                F.fetch(s.base, os.path.join(td, "y"), TOKEN)
            self.assertFalse(os.path.exists(os.path.join(td, "y")))          # nothing published on failure
            s.stop()


class LauncherPieces(unittest.TestCase):
    def a(self):
        from tests.test_runpod_orchestration import LauncherTests
        return LauncherTests().a(confirm_paid="t_run", ports=["8000/http"], results_token="R" * 43)

    def test_spec_exposes_port_and_token_only_in_env_and_redacts(self):
        from tests.test_runpod_orchestration import FakeAPI
        api = FakeAPI()
        rc, out = L.launch(api, self.a(), "abc")
        spec = api.calls[0][1]
        self.assertEqual(spec["ports"], ["8000/http"])
        self.assertEqual(spec["env"]["RESULTS_TOKEN"], "R" * 43)
        self.assertNotIn("R" * 43, json.dumps(out))                    # printed spec is redacted
        self.assertEqual(out["proxy_urls"], {"8000": "https://new1-8000.proxy.runpod.net"})

    def test_transfer_preset_is_disposable_and_token_file_is_outside_repo(self):
        import argparse
        ns = L.transfer_args(argparse.Namespace(run_name="", mount="/workspace", ports=[]))
        self.assertTrue(ns.run_name.startswith("transfer_test_"))
        self.assertEqual((ns.hard_timeout_s, ns.ports), (600, ["8000/http"]))
        with self.assertRaises(SystemExit):
            L.transfer_args(argparse.Namespace(run_name="real_boreal_x", mount="/workspace", ports=[]))
        with self.assertRaises(SystemExit):
            L.load_token(os.path.join(ROOT, "token.txt"))
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "tok")
            t1 = L.load_token(p)
            self.assertEqual(L.load_token(p), t1)
            self.assertEqual(oct(os.stat(p).st_mode & 0o777), "0o600")
            self.assertGreaterEqual(len(t1), 32)

    def test_transfer_job_refuses_other_paths(self):
        import subprocess
        for bad in ("/workspace/runs/real_boreal", "/tmp/transfer_test_x"):
            r = subprocess.run([sys.executable, os.path.join(RP, "transfer_test_job.py"), "--out", bad], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(os.path.exists(bad))


if __name__ == "__main__":
    unittest.main()
