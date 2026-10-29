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
import watchdog as W  # noqa: E402
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
        os.environ.update(RUNPOD_POD_ID="pod1", RUNPOD_API_KEY=SECRET)
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
                 data_center="US-IL-1", container_disk_gb=30, boot_allowance_s=600, watchdog_latency_s=600, watchdog_armed=True,
                 bundle_files=[])
        d.update(kw)
        return argparse.Namespace(**d)

    def test_dry_run_creates_nothing_and_redacts(self):
        api = FakeAPI()
        rc, out = L.launch(api, self.a(), "abc123")
        self.assertEqual(rc, 0)
        self.assertFalse(out["created"])
        self.assertEqual(api.calls, [])
        self.assertEqual(out["spec"]["env"]["ORCH_BUNDLE_B64"], "<set>")
        self.assertRegex(out["spec"]["name"], r"^t_run__dl\d{10}$")

    def test_wrong_confirmation_creates_nothing(self):
        api = FakeAPI()
        rc, out = L.launch(api, self.a(confirm_paid="other"), "abc")
        self.assertFalse(out["created"])
        self.assertEqual(api.calls, [])

    def test_confirmed_launch_creates_one_pod_with_supervisor(self):
        api = FakeAPI()
        rc, out = L.launch(api, self.a(confirm_paid="t_run"), "abc")
        self.assertTrue(out["created"])
        spec = api.calls[0][1]
        cmd = spec["dockerStartCmd"][-1]
        self.assertIn("pod_supervisor.py", cmd)
        self.assertIn("--hard-timeout-s 60", cmd)
        self.assertEqual(spec["networkVolumeId"], "sej40ez3ia")
        self.assertNotIn("RUNPOD_API_KEY", json.dumps(spec))

    def test_refused_when_pod_running(self):
        api = FakeAPI(pods=[{"id": "p0", "name": "other", "desiredStatus": "RUNNING"}])
        rc, out = L.launch(api, self.a(confirm_paid="t_run"), "abc")
        self.assertEqual(rc, 3)
        self.assertFalse(out["created"])

    def test_stopped_pods_do_not_block(self):
        api = FakeAPI(pods=[{"id": "p0", "name": "old", "desiredStatus": "EXITED"}])
        rc, out = L.launch(api, self.a(), "abc")
        self.assertEqual(rc, 0)

    def test_refused_over_budget(self):
        rc, out = L.launch(FakeAPI(), self.a(hard_timeout_s=36000 * 3), "abc")
        self.assertEqual(rc, 3)

    def test_price_above_cap_stops_pod_immediately(self):
        api = FakeAPI(price=2.5)
        rc, out = L.launch(api, self.a(confirm_paid="t_run"), "abc")
        self.assertEqual(rc, 3)
        self.assertEqual(api.stops(), [("stop", "new1")])

    def test_bad_run_name_refused(self):
        rc, out = L.launch(FakeAPI(), self.a(run_name="../evil"), "abc")
        self.assertEqual(rc, 3)

    def test_unarmed_watchdog_refuses_paid_launch(self):
        api = FakeAPI()
        rc, out = L.launch(api, self.a(confirm_paid="t_run", watchdog_armed=False), "abc")
        self.assertEqual(rc, 3)
        self.assertEqual(api.calls, [])


class InfraTests(unittest.TestCase):
    def a(self, **kw):
        return LauncherTests.a(LauncherTests(), **kw)

    def test_boot_allowance_in_worst_case(self):
        _, _, w0 = C.check_budget(price_per_hr=0.89, hard_timeout_s=60, billed_usd=0)
        _, _, w1 = C.check_budget(price_per_hr=0.89, hard_timeout_s=60, billed_usd=0, boot_allowance_s=600)
        self.assertGreater(w1, w0)

    def test_deadline_includes_boot_allowance(self):
        api = FakeAPI()
        rc, out = L.launch(api, self.a(confirm_paid="t_run"), "abc", now=lambda: 1700000000.0)
        self.assertEqual(out["deadline_epoch"], 1700000000 + 600 + 60)
        self.assertEqual(C.parse_deadline(api.calls[0][1]["name"]), 1700000660)

    def test_bundle_needs_no_repo_files_on_fresh_pod(self):
        """Run the pod start command in an empty dir with no repo: it must unpack the bundle and start the supervisor."""
        import subprocess
        api = FakeAPI()
        L.launch(api, self.a(confirm_paid="t_run"), "abc")
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

    def test_check_access_report(self):
        rep = L.check_access(FakeAPI())
        self.assertEqual(rep["volume_present"], True)
        self.assertTrue(rep["stop_probe"].startswith("authorized"))

        class NoScope(FakeAPI):
            def stop_pod(self, pid):
                raise C.RunPodError("x", code=403)
        self.assertEqual(L.check_access(NoScope())["stop_probe"], "NO WRITE SCOPE")


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
            ok = S.run_watchdog(api, "pod1", 100.0, os.path.join(td, "wd.jsonl"),
                                sleep=lambda s: t.__setitem__("now", t["now"] + s), now=lambda: t["now"])
            self.assertTrue(ok)
            self.assertGreaterEqual(t["now"], 100.0)
            self.assertEqual(api.pods["pod1"]["desiredStatus"], "EXITED")

    def test_keeps_retrying_stop(self):
        with tempfile.TemporaryDirectory() as td:
            api = FakeAPI(pods=[{"id": "pod1", "desiredStatus": "RUNNING"}], stop_failures=5)
            ok = S.run_watchdog(api, "pod1", 0.0, os.path.join(td, "wd.jsonl"), sleep=lambda s: None,
                                now=lambda: 1.0)
            self.assertTrue(ok)


class EnforceTests(unittest.TestCase):
    """Off-pod stateless watchdog (what GitHub Actions runs)."""

    def run_enforce(self, api, now, **kw):
        ev = []
        rc = W.enforce(api, lambda n, **f: ev.append((n, f)), now=lambda: now, sleep=lambda s: None, **kw)
        return rc, [e[0] for e in ev], ev

    def pods(self, *specs):
        return [{"id": i, "name": n, "desiredStatus": st} for i, n, st in specs]

    def test_stops_only_overdue_tagged_running_pods(self):
        api = FakeAPI(pods=self.pods(("a", C.tag_name("r1", 1700000000), "RUNNING"), ("b", C.tag_name("r2", 9_000_000_000), "RUNNING"),
                                     ("c", C.tag_name("r3", 1700000000), "EXITED")))
        rc, names, _ = self.run_enforce(api, 1700001000)
        self.assertEqual(rc, 0)
        self.assertEqual(api.stops(), [("stop", "a")])
        self.assertIn("pod_stopped", names)
        self.assertIn("heartbeat_running_within_deadline", names)

    def test_untagged_pod_reported_not_stopped_by_default(self):
        api = FakeAPI(pods=self.pods(("a", "manual-debug", "RUNNING")))
        rc, names, _ = self.run_enforce(api, 1700001000)
        self.assertEqual((rc, names, api.stops()), (0, ["untagged_running_pod"], []))

    def test_untagged_pod_stopped_with_flag(self):
        api = FakeAPI(pods=self.pods(("a", "manual-debug", "RUNNING")))
        self.run_enforce(api, 1700001000, stop_untagged=True)
        self.assertEqual(api.stops(), [("stop", "a")])

    def test_failed_stop_is_logged_and_nonzero(self):
        api = FakeAPI(pods=self.pods(("a", C.tag_name("r1", 1700000000), "RUNNING")), stays_running=True)
        rc, names, _ = self.run_enforce(api, 1700001000)
        self.assertEqual(rc, 2)
        self.assertIn("STOP_FAILED", names)
        self.assertEqual(len(api.stops()), 3)  # retried within the pass

    def test_api_error_is_logged_and_nonzero(self):
        class Down(FakeAPI):
            def list_pods(self):
                raise C.RunPodError("GET /pods -> HTTP 503", code=503)
        rc, names, _ = self.run_enforce(Down(), 1700001000)
        self.assertEqual((rc, names), (2, ["api_error"]))

    def test_quiet_when_nothing_running(self):
        rc, names, _ = self.run_enforce(FakeAPI(pods=self.pods(("c", C.tag_name("r3", 1700000000), "EXITED"))), 1700001000)
        self.assertEqual((rc, names), (0, []))

    def test_idempotent_second_pass_does_nothing(self):
        api = FakeAPI(pods=self.pods(("a", C.tag_name("r1", 1700000000), "RUNNING")))
        self.run_enforce(api, 1700001000)
        _, names, _ = self.run_enforce(api, 1700001060)
        self.assertEqual(names, [])
        self.assertEqual(len(api.stops()), 1)

    def test_deadline_tag_roundtrip_and_limits(self):
        self.assertEqual(C.parse_deadline(C.tag_name("lifecycle_test_x", 1761720000)), 1761720000)
        self.assertIsNone(C.parse_deadline("no tag"))
        with self.assertRaises(ValueError):
            C.tag_name("x" * 60, 1761720000)

    def test_cli_writes_append_only_log(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            log = os.path.join(td, "e.jsonl")
            with open(log, "w") as f:
                f.write('{"event": "earlier"}\n')
            env = {"PATH": os.environ["PATH"]}  # no key: must fail loudly, must not truncate the log
            r = subprocess.run([sys.executable, os.path.join(RP, "watchdog.py"), "--log", log], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertEqual(read(log), '{"event": "earlier"}\n')

    def test_workflow_file_is_sane(self):
        import yaml
        path = os.path.join(ROOT, ".github", "workflows", "runpod-watchdog.yml")
        text = read(path)
        wf = yaml.safe_load(text)
        on = wf.get("on", wf.get(True))
        self.assertEqual(set(on), {"schedule", "workflow_dispatch"})
        self.assertEqual(on["workflow_dispatch"]["inputs"]["selftest"]["default"], False)  # never pull_request / pull_request_target
        self.assertEqual(wf["permissions"], {"contents": "write"})
        steps = wf["jobs"]["enforce"]["steps"]
        by_secret = [s_["name"] for s_ in steps if "secrets.RUNPOD_API_KEY" in json.dumps(s_)]
        by_token = [s_["name"] for s_ in steps if "github.token" in json.dumps(s_)]
        self.assertEqual(by_secret, ["Enforce pod deadlines"])  # secret reaches exactly one step
        self.assertEqual(by_token, ["Append events to runpod-logs branch (append-only)"])  # write token reaches exactly one step
        self.assertEqual(text.count("secrets."), text.count("secrets.RUNPOD_API_KEY"))
        checkout = [s_ for s_ in steps if str(s_.get("uses", "")).startswith("actions/checkout")][0]
        self.assertIs(checkout["with"]["persist-credentials"], False)
        code = "\n".join(s_.get("run", "") for s_ in steps)  # executable shell only, not comments
        for bad in ("--force", "push -f", "--force-with-lease", "origin main", "pip install", "npm ", "curl ", "wget "):
            self.assertNotIn(bad, code)
        self.assertNotIn("${{", code)  # token only via env, not inlined
        uses = [s_["uses"] for s_ in steps if "uses" in s_]
        self.assertTrue(all(u.startswith("actions/") for u in uses), uses)  # first-party actions only

    def test_watchdog_uses_only_read_and_stop_calls(self):
        """The off-pod watchdog may call list_pods, get_pod, stop_pod and nothing else on the API object."""
        allowed = {"list_pods", "get_pod", "stop_pod"}
        used = set()

        class Guard(FakeAPI):
            def __getattribute__(self, name):
                if not name.startswith("_") and name in ("create_pod", "list_network_volumes", "billed_total", "list_pods",
                                                          "get_pod", "stop_pod"):
                    used.add(name)
                return super().__getattribute__(name)
        api = Guard(pods=self.pods(("a", C.tag_name("r1", 1_700_000_000), "RUNNING")))
        W.enforce(api, lambda n, **f: None, now=lambda: 1_700_001_000, sleep=lambda s: None)
        self.assertTrue(used <= allowed, used)

    def test_api_client_has_no_delete_capability(self):
        methods = [m for m in dir(C.RunPodAPI) if not m.startswith("_")]
        self.assertFalse([m for m in methods if "delete" in m or "terminate" in m or "remove" in m], methods)
        src = read(os.path.join(RP, "rp_common.py"))
        self.assertEqual(sorted(set(re.findall(r'self\._call\("([A-Z]+)"', src))), ["GET", "POST"])
        self.assertEqual(sorted(set(re.findall(r'self\._call\("POST", f?"([^"]+)"', src))), ["/pods", "/pods/{pod_id}/stop"])

    def test_loop_stops_pod_within_one_interval_after_deadline(self):
        B = 1_700_000_000
        api = FakeAPI(pods=self.pods(("a", C.tag_name("r1", B + 300), "RUNNING")))
        t = {"now": float(B)}
        ev = []
        rc = W.run_loop(api, lambda n, **f: ev.append((t["now"], n)), minutes=30, interval_s=60,
                        now=lambda: t["now"], sleep=lambda s: t.__setitem__("now", t["now"] + max(s, 0.001)))
        self.assertEqual(rc, 0)
        stop_t = [x for x, n in ev if n == "pod_stopped"]
        self.assertEqual(len(stop_t), 1)
        self.assertLessEqual(stop_t[0] - (B + 300), 70)  # about one interval of latency (plus enforce's own 5 s waits)
        self.assertEqual(ev[0][1], "loop_start")
        self.assertEqual(ev[-1][1], "loop_end")
        self.assertEqual(len(api.stops()), 1)

    def test_loop_throttles_heartbeats(self):
        B = 1_700_000_000
        api = FakeAPI(pods=self.pods(("a", C.tag_name("r1", B + 100000), "RUNNING")))
        t = {"now": float(B)}
        ev = []
        W.run_loop(api, lambda n, **f: ev.append(n), minutes=60, interval_s=60, now=lambda: t["now"],
                   sleep=lambda s: t.__setitem__("now", t["now"] + s))
        hb = ev.count("heartbeat_running_within_deadline")
        self.assertTrue(5 <= hb <= 8, hb)  # about 1 per 10 min over 60 min, not 60
        self.assertEqual(api.stops(), [])

    def test_loop_survives_transient_api_errors(self):
        B = 1_700_000_000

        class Flaky(FakeAPI):
            n = 0

            def list_pods(self):
                Flaky.n += 1
                if Flaky.n <= 2:
                    raise C.RunPodError("GET /pods -> HTTP 503", code=503)
                return super().list_pods()
        api = Flaky(pods=self.pods(("a", C.tag_name("r1", B + 10), "RUNNING")))
        t = {"now": float(B)}
        ev = []
        rc = W.run_loop(api, lambda n, **f: ev.append(n), minutes=10, interval_s=60, now=lambda: t["now"],
                        sleep=lambda s: t.__setitem__("now", t["now"] + s))
        self.assertIn("api_error", ev)
        self.assertIn("pod_stopped", ev)
        self.assertEqual(rc, 0)

    def test_loop_failed_stop_is_sticky(self):
        B = 1_700_000_000
        api = FakeAPI(pods=self.pods(("a", C.tag_name("r1", B), "RUNNING")), stays_running=True)
        t = {"now": float(B + 100)}
        rc = W.run_loop(api, lambda n, **f: None, minutes=3, interval_s=60, now=lambda: t["now"],
                        sleep=lambda s: t.__setitem__("now", t["now"] + s))
        self.assertEqual(rc, 2)

    def test_main_end_to_end_with_pod_name_field(self):
        """Regression: the CLI logger once collided with the `name=<pod name>` field and crashed on the first real pod."""
        B = 1_700_000_000
        api = FakeAPI(pods=self.pods(("a", C.tag_name("r1", B), "RUNNING")))
        import time as _t
        old_cls, old_sleep = C.RunPodAPI, _t.sleep
        C.RunPodAPI = lambda: api
        _t.sleep = lambda s: None
        try:
            with tempfile.TemporaryDirectory() as td:
                log = os.path.join(td, "e.jsonl")
                rc = W.main(["--log", log])
                lines = [json.loads(x) for x in read(log).splitlines()]
        finally:
            C.RunPodAPI, _t.sleep = old_cls, old_sleep
        self.assertEqual(rc, 0)
        self.assertIn("pod_stopped", [x["event"] for x in lines])
        self.assertTrue(any(x.get("name", "").startswith("r1__dl") for x in lines))
        self.assertEqual(api.pods["a"]["desiredStatus"], "EXITED")

    def test_selftest_cli_requires_key_and_never_touches_pods(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run([sys.executable, os.path.join(RP, "watchdog.py"), "--log", os.path.join(td, "e.jsonl"), "--selftest"],
                               env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)

    def test_status_reports_pods_and_overdue(self):
        api = FakeAPI(pods=self.pods(("a", C.tag_name("r1", 1700000000), "RUNNING"), ("b", "x", "EXITED")))
        out = L.status(api, now=lambda: 1700000500)
        self.assertEqual(out["running"], 1)
        self.assertEqual(out["pods"][0]["overdue_s"], 500)
        self.assertEqual(out["billed_total_usd"], 5.0)


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
    def test_no_delete_or_terminate_in_orchestration_code(self):
        for fn in ("rp_common.py", "launch_job.py", "pod_supervisor.py", "watchdog.py"):
            src = read(os.path.join(RP, fn))
            self.assertIsNone(re.search(r"\"DELETE\"|terminate_pod|delete_pod|remove pod", src), fn)

    def test_no_literal_credentials_in_sources(self):
        for fn in ("rp_common.py", "launch_job.py", "pod_supervisor.py", "watchdog.py"):
            self.assertIsNone(re.search(r"rpa_[A-Za-z0-9]{10,}", read(os.path.join(RP, fn))))


if __name__ == "__main__":
    unittest.main()
