"""Mock-only lifecycle tests for eval/runpod orchestration. No network, no RunPod, no credentials."""
import argparse
import json
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RP = os.path.join(ROOT, "eval", "runpod")
sys.path.insert(0, RP)
import launch_job as L  # noqa: E402
import pod_supervisor as S  # noqa: E402
import rp_common as C  # noqa: E402

SECRET = "SECRET-KEY-DO-NOT-LEAK"


class FakeAPI:
    def __init__(self, pods=None, billed=5.0, stop_failures=0, stays_running=False, price=0.89):
        self.pods = {p["id"]: dict(p) for p in (pods or [])}
        self.billed, self.stop_failures, self.stays_running, self.price = billed, stop_failures, stays_running, price
        self.calls = []

    def list_pods(self):
        return list(self.pods.values())

    def get_pod(self, pid):
        return dict(self.pods[pid])

    def create_pod(self, spec):
        self.calls.append(("create", spec))
        self.pods["new1"] = {"id": "new1", "name": spec["name"], "desiredStatus": "RUNNING"}
        return {"id": "new1", "costPerHr": self.price}

    def stop_pod(self, pid):
        self.calls.append(("stop", pid))
        if pid not in self.pods:
            raise C.RunPodError("POST stop -> HTTP 500", code=500, detail="stop pod: Attempted to stop pod that does not exist.")
        if self.stop_failures > 0:
            self.stop_failures -= 1
            raise C.RunPodError("POST stop -> HTTP 500")
        if not self.stays_running:
            self.pods[pid]["desiredStatus"] = "EXITED"
        return {}

    def billed_total(self):
        return self.billed

    def list_network_volumes(self):
        return [{"id": "sej40ez3ia", "name": "wildfire-datasets"}]

    def stops(self):
        return [c for c in self.calls if c[0] == "stop"]


def args(td, **kw):
    d = dict(run_name="t_run", result_dir=os.path.join(td, "res"), hard_timeout_s=60, finish_margin_s=0, artifact=[])
    d.update(kw)
    return argparse.Namespace(**d)


class Env:
    def __enter__(self):
        self.old = dict(os.environ)
        os.environ.update(RUNPOD_POD_ID="pod1", RUNPOD_API_KEY=SECRET, NOTIFY_URL="")
        return self

    def __exit__(self, *a):
        os.environ.clear()
        os.environ.update(self.old)


def read(path):
    with open(path) as f:
        return f.read()


class BudgetTests(unittest.TestCase):
    def test_ok_one_minute_test(self):
        ok, why, worst = C.check_budget(price_per_hr=0.89, hard_timeout_s=60, billed_usd=9.0, ledger_floor_usd=9.3)
        self.assertTrue(ok, why)
        self.assertAlmostEqual(worst, 0.89 / 60 + 0.25, 4)

    def test_per_experiment_limit(self):
        ok, why, _ = C.check_budget(price_per_hr=0.89, hard_timeout_s=9000 * 10, billed_usd=0)
        self.assertFalse(ok)
        self.assertTrue(any("per-experiment" in r for r in why))

    def test_allowance_uses_ledger_floor_when_billing_lags(self):
        ok, why, _ = C.check_budget(price_per_hr=1.0, hard_timeout_s=3600, billed_usd=1.0, ledger_floor_usd=99.0)
        self.assertFalse(ok)
        self.assertTrue(any("allowance" in r for r in why))

    def test_one_running_pod_blocks(self):
        ok, why, _ = C.check_budget(price_per_hr=0.5, hard_timeout_s=60, billed_usd=0, other_running=1)
        self.assertFalse(ok)


class LauncherTests(unittest.TestCase):
    def a(self, **kw):
        d = dict(run_name="t_run", job="bash x.sh", hard_timeout_s=60, max_hourly_price=0.89, ledger_floor_usd=9.3,
                 artifact=["/workspace/runs/t_run/*.pt"], repo_dir="/workspace/repo_orch", confirm_paid="",
                 mount="/workspace", **{k: v for k, v in L.DEFAULTS.items() if k in ("gpu", "image", "volume_id")},
                 data_center="US-IL-1", container_disk_gb=30, boot_allowance_s=600, bundle_files=[], watch=False)
        d.update(kw)
        return argparse.Namespace(**d)

    def test_dry_run_creates_nothing_and_redacts(self):
        api = FakeAPI()
        rc, out = L.launch(api, self.a(), "https://ntfy.sh/secret-topic", "abc123")
        self.assertEqual(rc, 0)
        self.assertFalse(out["created"])
        self.assertEqual(api.calls, [])
        self.assertNotIn("secret-topic", json.dumps(out))

    def test_wrong_confirmation_creates_nothing(self):
        api = FakeAPI()
        rc, out = L.launch(api, self.a(confirm_paid="other"), "", "abc")
        self.assertFalse(out["created"])
        self.assertEqual(api.calls, [])

    def test_confirmed_launch_creates_one_pod_with_supervisor(self):
        api = FakeAPI()
        rc, out = L.launch(api, self.a(confirm_paid="t_run"), "", "abc")
        self.assertTrue(out["created"])
        spec = api.calls[0][1]
        cmd = spec["dockerStartCmd"][-1]
        self.assertIn("pod_supervisor.py", cmd)
        self.assertIn("--hard-timeout-s 60", cmd)
        self.assertEqual(spec["networkVolumeId"], "sej40ez3ia")
        self.assertNotIn("RUNPOD_API_KEY", json.dumps(spec))

    def test_refused_when_pod_running(self):
        api = FakeAPI(pods=[{"id": "p0", "name": "other", "desiredStatus": "RUNNING"}])
        rc, out = L.launch(api, self.a(confirm_paid="t_run"), "", "abc")
        self.assertEqual(rc, 3)
        self.assertFalse(out["created"])

    def test_stopped_pods_do_not_block(self):
        api = FakeAPI(pods=[{"id": "p0", "name": "old", "desiredStatus": "EXITED"}])
        rc, out = L.launch(api, self.a(), "", "abc")
        self.assertEqual(rc, 0)

    def test_refused_over_budget(self):
        rc, out = L.launch(FakeAPI(), self.a(hard_timeout_s=36000 * 3), "", "abc")
        self.assertEqual(rc, 3)

    def test_price_above_cap_stops_pod_immediately(self):
        api = FakeAPI(price=2.5)
        rc, out = L.launch(api, self.a(confirm_paid="t_run"), "", "abc")
        self.assertEqual(rc, 3)
        self.assertEqual(api.stops(), [("stop", "new1")])

    def test_bad_run_name_refused(self):
        rc, out = L.launch(FakeAPI(), self.a(run_name="../evil"), "", "abc")
        self.assertEqual(rc, 3)

    def test_monitor_stops_past_deadline(self):
        api = FakeAPI(pods=[{"id": "pod1", "name": "n", "desiredStatus": "RUNNING"}])
        t = {"now": 1000.0}
        res = L.monitor(api, "pod1", 1000.0, "", grace_s=60, poll_s=30,
                        sleep=lambda s: t.__setitem__("now", t["now"] + s), now=lambda: t["now"])
        self.assertEqual(res, "stopped")
        self.assertGreaterEqual(len(api.stops()), 1)
        self.assertGreater(t["now"], 1060)  # did not stop early


class InfraTests(unittest.TestCase):
    def a(self, **kw):
        return LauncherTests.a(LauncherTests(), **kw)

    def test_boot_allowance_in_worst_case(self):
        _, _, w0 = C.check_budget(price_per_hr=0.89, hard_timeout_s=60, billed_usd=0)
        _, _, w1 = C.check_budget(price_per_hr=0.89, hard_timeout_s=60, billed_usd=0, boot_allowance_s=600)
        self.assertGreater(w1, w0)

    def test_deadline_includes_boot_allowance(self):
        api = FakeAPI()
        rc, out = L.launch(api, self.a(confirm_paid="t_run"), "", "abc", now=lambda: 1000.0)
        self.assertEqual(out["deadline_epoch"], 1000.0 + 600 + 60)

    def test_bundle_needs_no_repo_files_on_fresh_pod(self):
        """Run the pod start command in an empty dir with no repo: it must unpack the bundle and start the supervisor."""
        import subprocess
        api = FakeAPI()
        L.launch(api, self.a(confirm_paid="t_run"), "", "abc")
        spec = api.calls[0][1]
        with tempfile.TemporaryDirectory() as td:
            env = {"PATH": os.environ["PATH"], "HOME": td, "TMPDIR": td, "ORCH_BUNDLE_B64": spec["env"]["ORCH_BUNDLE_B64"]}
            r = subprocess.run(["bash", "-c", spec["dockerStartCmd"][-1]], cwd=td, env=env, capture_output=True, text=True,
                               timeout=60)
        self.assertEqual(r.returncode, 5, r.stderr)  # reached the supervisor; no key => refuses to run the job
        self.assertIn("refusing to run a job that cannot self-stop", r.stderr)

    def test_bundle_is_deterministic_and_small(self):
        b1, b2 = L.make_bundle(), L.make_bundle()
        self.assertEqual(b1, b2)
        self.assertLess(len(b1), L.MAX_BUNDLE_B64)

    def test_bundle_rejects_files_outside_eval_runpod(self):
        with self.assertRaises(ValueError):
            L.make_bundle(["../../CLAUDE.md"])

    def test_lifecycle_preset_is_disposable(self):
        ns = L.lifecycle_args(argparse.Namespace(run_name="", mount="/workspace"))
        self.assertTrue(ns.run_name.startswith("lifecycle_test_"))
        self.assertEqual(ns.hard_timeout_s, 60)
        self.assertIn("lifecycle_test_job.py", ns.job)
        with self.assertRaises(SystemExit):
            L.lifecycle_args(argparse.Namespace(run_name="real_boreal_seed3", mount="/workspace"))

    def test_lifecycle_job_refuses_other_locations(self):
        import subprocess
        for bad in ("/workspace/runs/real_boreal", "/tmp/lifecycle_test_x", "/workspace/models/lifecycle_test_x"):
            r = subprocess.run([sys.executable, os.path.join(RP, "lifecycle_test_job.py"), "--out", bad, "--sleep-s", "0"],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertFalse(os.path.exists(bad))

    def test_finish_margin_capped_for_short_timeouts(self):
        with tempfile.TemporaryDirectory() as td:
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}])
            a = args(td, hard_timeout_s=8, finish_margin_s=300)
            with Env():
                S.supervise(a, [sys.executable, "-c", "import time; time.sleep(0.5)"], api,
                            spawn_watchdog=lambda *_: None, sleep=lambda s: None)
            self.assertEqual(json.loads(read(os.path.join(a.result_dir, "DONE.json")))["status"], "success")

    def test_sweep_stops_only_matching_running_pods(self):
        api = FakeAPI(pods=[{"id": "a", "name": "lifecycle_test_1", "desiredStatus": "RUNNING"},
                            {"id": "b", "name": "real_boreal_seed3", "desiredStatus": "RUNNING"},
                            {"id": "c", "name": "lifecycle_test_0", "desiredStatus": "EXITED"}])
        self.assertEqual(L.sweep(api, "lifecycle_test_", ""), ["a"])
        self.assertEqual(api.stops(), [("stop", "a")])
        with self.assertRaises(ValueError):
            L.sweep(api, "", "")

    def test_monitor_returns_gone_on_404(self):
        class Gone(FakeAPI):
            def get_pod(self, pid):
                raise C.RunPodError("x", code=404)
        self.assertEqual(L.monitor(Gone(), "p", 0, "", sleep=lambda s: None), "gone")

    def test_monitor_gives_up_loudly(self):
        api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}], stays_running=True)
        t = {"now": 0.0}
        res = L.monitor(api, "pod1", 0.0, "", grace_s=10, poll_s=30, give_up_s=100,
                        sleep=lambda s: t.__setitem__("now", t["now"] + s), now=lambda: t["now"])
        self.assertEqual(res, "UNCONFIRMED")

    def test_check_access_report(self):
        rep = L.check_access(FakeAPI())
        self.assertEqual(rep["volume_present"], True)
        self.assertTrue(rep["stop_probe"].startswith("authorized"))

        class NoScope(FakeAPI):
            def stop_pod(self, pid):
                raise C.RunPodError("x", code=403)
        self.assertEqual(L.check_access(NoScope())["stop_probe"], "NO WRITE SCOPE")

    def test_ntfy_format_plain_text_with_title_header(self):
        import http.server
        import threading
        got = {}

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                got.update(body=self.rfile.read(int(self.headers["Content-Length"])).decode(), title=self.headers["Title"],
                           prio=self.headers["Priority"], ctype=self.headers["Content-Type"])
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_port}/topic"
            self.assertTrue(C.notify(url, {"title": "Job success", "status": "success", "urgent": True}))
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(got["title"], "Job success")
        self.assertEqual(got["prio"], "urgent")
        self.assertIn("status: success", got["body"])
        self.assertTrue(got["ctype"].startswith("text/plain"))


class SupervisorTests(unittest.TestCase):
    def run_sup(self, cmd, api, td, **kw):
        a = args(td, **kw)
        with Env():
            rc = S.supervise(a, cmd, api, spawn_watchdog=lambda *_: None, sleep=lambda s: None)
        return rc, a.result_dir

    def done(self, d):
        return json.loads(read(os.path.join(d, "DONE.json")))

    def test_success_stops_pod_and_hashes_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            ck = os.path.join(td, "best.pt")
            with open(ck, "wb") as f:
                f.write(b"weights")
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}])
            rc, d = self.run_sup([sys.executable, "-c", "print('ok')"], api, td, artifact=[ck])
            self.assertEqual(rc, 0)
            done = self.done(d)
            self.assertEqual(done["status"], "success")
            self.assertEqual(done["artifacts"][ck]["sha256"], C.sha256_file(ck))
            self.assertEqual(api.pods["pod1"]["desiredStatus"], "EXITED")
            self.assertEqual(open(ck, "rb").read(), b"weights")  # artifact untouched

    def test_failure_still_stops_pod(self):
        with tempfile.TemporaryDirectory() as td:
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}])
            rc, d = self.run_sup([sys.executable, "-c", "import sys; sys.exit(7)"], api, td)
            self.assertEqual(rc, 1)
            done = self.done(d)
            self.assertEqual((done["status"], done["exit_code"]), ("failed", 7))
            self.assertEqual(api.pods["pod1"]["desiredStatus"], "EXITED")

    def test_missing_command_still_stops_pod(self):
        with tempfile.TemporaryDirectory() as td:
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}])
            rc, d = self.run_sup(["/nonexistent/binary"], api, td)
            self.assertEqual(rc, 1)
            self.assertEqual(api.pods["pod1"]["desiredStatus"], "EXITED")

    def test_timeout_kills_job_and_stops_pod(self):
        with tempfile.TemporaryDirectory() as td:
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}])
            rc, d = self.run_sup([sys.executable, "-c", "import time; time.sleep(60)"], api, td, hard_timeout_s=2)
            self.assertEqual(self.done(d)["status"], "timeout")
            self.assertEqual(api.pods["pod1"]["desiredStatus"], "EXITED")

    def test_stop_is_retried(self):
        with tempfile.TemporaryDirectory() as td:
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}], stop_failures=2)
            rc, d = self.run_sup([sys.executable, "-c", "pass"], api, td)
            self.assertEqual(api.pods["pod1"]["desiredStatus"], "EXITED")
            self.assertGreaterEqual(len(api.stops()), 3)

    def test_stop_failure_is_logged_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}], stays_running=True)
            rc, d = self.run_sup([sys.executable, "-c", "pass"], api, td)
            self.assertIn("STOP_FAILED", read(os.path.join(d, "events.jsonl")))

    def test_existing_result_dir_aborts_without_stopping(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "res"))
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}])
            rc, _ = self.run_sup([sys.executable, "-c", "pass"], api, td)
            self.assertEqual(rc, 4)
            self.assertEqual(api.stops(), [])

    def test_done_json_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "DONE.json")
            C.write_json_new(p, {"a": 1})
            with self.assertRaises(FileExistsError):
                C.write_json_new(p, {"a": 2})
            self.assertEqual(json.loads(read(p)), {"a": 1})

    def test_no_credentials_in_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}], stop_failures=1)
            rc, d = self.run_sup([sys.executable, "-c", "print('hi')"], api, td)
            for name in os.listdir(d):
                self.assertNotIn(SECRET, read(os.path.join(d, name)))


class WatchdogTests(unittest.TestCase):
    def test_fires_at_deadline_independently_of_job(self):
        with tempfile.TemporaryDirectory() as td:
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}])
            t = {"now": 0.0}
            ok = S.run_watchdog(api, "pod1", 100.0, os.path.join(td, "wd.jsonl"), "",
                                sleep=lambda s: t.__setitem__("now", t["now"] + s), now=lambda: t["now"])
            self.assertTrue(ok)
            self.assertGreaterEqual(t["now"], 100.0)
            self.assertEqual(api.pods["pod1"]["desiredStatus"], "EXITED")

    def test_keeps_retrying_stop(self):
        with tempfile.TemporaryDirectory() as td:
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}], stop_failures=5)
            ok = S.run_watchdog(api, "pod1", 0.0, os.path.join(td, "wd.jsonl"), "", sleep=lambda s: None,
                                now=lambda: 1.0)
            self.assertTrue(ok)


class ClientTests(unittest.TestCase):
    """RunPodAPI against a throwaway localhost server: right verbs/paths, UA set, key never leaks into errors."""

    def setUp(self):
        import http.server
        import threading
        seen = self.seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def _do(self):
                seen.append((self.command, self.path, self.headers.get("User-Agent"), self.headers.get("Authorization")))
                code = 500 if self.path.endswith("/fail/stop") else 200
                body = b"[]" if self.path == "/pods" else b"{}"
                self.send_response(code)
                self.end_headers()
                self.wfile.write(body)
            do_GET = do_POST = _do

            def log_message(self, *a):
                pass
        self.srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.api = C.RunPodAPI(key=SECRET, base=f"http://127.0.0.1:{self.srv.server_port}")

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_verbs_and_user_agent(self):
        self.assertEqual(self.api.list_pods(), [])
        self.api.stop_pod("abc")
        self.assertEqual([c[:2] for c in self.seen], [("GET", "/pods"), ("POST", "/pods/abc/stop")])
        self.assertTrue(all(c[2] == C.UA for c in self.seen))

    def test_error_does_not_leak_key(self):
        with self.assertRaises(C.RunPodError) as cm:
            self.api.stop_pod("fail/stop".replace("/stop", ""))
        self.assertNotIn(SECRET, str(cm.exception))
        with self.assertRaises(C.RunPodError):
            C.RunPodAPI(key=SECRET, base="http://127.0.0.1:9").list_pods()

    def test_missing_key_refused(self):
        with Env():
            os.environ.pop("RUNPOD_API_KEY")
            with self.assertRaises(C.RunPodError):
                C.RunPodAPI()


class SafetyTests(unittest.TestCase):
    def test_notify_never_raises_and_empty_url_is_noop(self):
        self.assertFalse(C.notify("", {"x": 1}))

        def boom(url, payload):
            raise OSError("down")
        self.assertFalse(C.notify("https://x", {"x": 1}, post=boom))
        self.assertTrue(C.notify("https://x", {"x": 1}, post=lambda u, p: None))
        self.assertFalse(C.notify("http://127.0.0.1:9/none", {"x": 1}, timeout=1))

    def test_no_delete_or_terminate_in_orchestration_code(self):
        for fn in ("rp_common.py", "launch_job.py", "pod_supervisor.py"):
            src = read(os.path.join(RP, fn))
            self.assertIsNone(re.search(r"\"DELETE\"|terminate_pod|delete_pod|remove pod", src), fn)

    def test_no_literal_credentials_in_sources(self):
        for fn in ("rp_common.py", "launch_job.py", "pod_supervisor.py"):
            self.assertIsNone(re.search(r"rpa_[A-Za-z0-9]{10,}", read(os.path.join(RP, fn))))


if __name__ == "__main__":
    unittest.main()
