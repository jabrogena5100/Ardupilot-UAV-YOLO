"""Paired Level 3 plan: arms differ only in the perception arm; scenarios come from the pinned split; held-out is guarded."""
import copy
import json
import os
import tempfile
import unittest

import yaml

from eval import level3_pairs as L
from eval import level3_scenarios as SC
from perception import detector as D


class TestPairs(unittest.TestCase):
    def setUp(self):
        self._thr = L.THR_PATH
        self.tmp = tempfile.TemporaryDirectory()
        L.THR_PATH = os.path.join(self.tmp.name, "thr.json")          # an isolated thresholds file per test

    def tearDown(self):
        L.THR_PATH = self._thr
        self.tmp.cleanup()

    def write_thr(self, taus):
        json.dump({"arms": {a: {"tau": t} for a, t in taus.items()}}, open(L.THR_PATH, "w"))

    def test_plan_calibration_pairs_every_arm_and_never_uses_heldout(self):
        self.write_thr({"A0": 0.4, "B2s0": 0.3})
        rows = L.plan("calibration", list(L.ARMS), os.path.join(self.tmp.name, "p"))
        cal = {s["seed"] for s in SC.load()["calibration"]}
        held = {s["seed"] for s in SC.load()["heldout"]}
        self.assertEqual({r["scenario"] for r in rows}, cal)
        self.assertEqual(len(rows), len(cal) * len(L.ARMS))
        for seed in cal:
            cfgs = {r["arm"]: yaml.safe_load(open(r["config"])) for r in rows if r["scenario"] == seed}
            L.assert_paired(cfgs)
            self.assertEqual({c["fire_model"]["random_seed"] for c in cfgs.values()}, {seed})
            self.assertEqual({tuple(map(tuple, c["fire_model"]["initial_ignitions"])) for c in cfgs.values()},
                             {tuple(map(tuple, [next(s for s in SC.load()["calibration"] if s["seed"] == seed)["ignition"]]))})
            self.assertEqual(cfgs["A0"]["perception"]["conf_threshold"], 0.4)
            self.assertEqual(cfgs["B2s0"]["perception"]["conf_threshold"], 0.3)
            self.assertFalse(cfgs["baseline"]["perception"]["enabled"])
            self.assertFalse(any(c["fire_model"]["oracle_hazards"] for c in cfgs.values()))
            self.assertTrue(all(c["perception"]["camera"]["yaw_to_track"] for c in cfgs.values()))   # identical flight in every arm
            self.assertTrue(all(c["fire_model"]["start_gate"]["agents"] == 1 for c in cfgs.values()))
        self.assertFalse(held & {r["scenario"] for r in rows})

    def test_heldout_is_refused_without_the_flag_and_before_freezing(self):
        with self.assertRaises(PermissionError):
            L.plan("heldout", ["baseline"], os.path.join(self.tmp.name, "h"))
        with self.assertRaises(PermissionError):                 # flag given but thresholds not frozen yet
            L.plan("heldout", ["baseline"], os.path.join(self.tmp.name, "h"), final=True)
        self.write_thr({"A0": 0.4})
        rows = L.plan("heldout", ["baseline", "A0"], os.path.join(self.tmp.name, "h"), final=True)
        self.assertEqual({r["scenario"] for r in rows}, {s["seed"] for s in SC.load()["heldout"]})

    def test_unpaired_difference_or_uncalibrated_tau_is_refused(self):
        self.write_thr({"A0": 0.4, "B2s0": 0.3})
        base = L.build_base()
        sc = SC.load()["calibration"][0]
        a, b = L.make_config(base, "A0", sc), L.make_config(base, "B2s0", sc)
        L.assert_paired({"A0": a, "B2s0": b})
        c = copy.deepcopy(b)
        c["perception"]["rate_hz"] = 0.5
        with self.assertRaises(ValueError):
            L.assert_paired({"A0": a, "B2s0": c})
        d = copy.deepcopy(b)
        d["perception"]["conf_threshold"] = 0.55                   # not the calibrated value
        with self.assertRaises(ValueError):
            L.assert_paired({"A0": a, "B2s0": d})
        e = copy.deepcopy(b)
        e["vehicle_defaults"]["alt"] = 50
        with self.assertRaises(ValueError):
            L.assert_paired({"A0": a, "B2s0": e})

    def test_scenarios_differ_and_agent_seeds_follow_the_scenario(self):
        self.write_thr({})
        base = L.build_base()
        s1, s2 = SC.load()["calibration"][:2]
        a, b = L.make_config(base, "A0", s1), L.make_config(base, "A0", s2)
        self.assertNotEqual(L.strip_arm(a), L.strip_arm(b))
        self.assertEqual(L.make_config(base, "A0", s1)["agents"][0]["seed"], L.make_config(base, "B2s0", s1)["agents"][0]["seed"])

    def test_arm_checkpoints_are_registry_files(self):
        for arm, v in L.ARMS.items():
            if v is not None:
                self.assertIn(os.path.basename(v[0]), D.KNOWN_SHA256, arm)

    def test_base_uses_the_frozen_fire_and_the_documented_camera(self):
        base = L.build_base()
        fire = L.load_fire()
        self.assertEqual((base["fire_model"]["p_ignite"], base["fire_model"]["burn_steps"]), (fire["p_ignite"], fire["burn_steps"]))
        self.assertEqual((base["perception"]["camera"]["pitch_deg"], base["perception"]["camera"]["hfov_deg"]), (60, 78))
        self.assertFalse(base["fire_model"]["oracle_hazards"])
        self.assertEqual(base["fog_tracker"]["max_seconds"], fire["T_s"])

    def test_process_matcher_ignores_shells_that_only_mention_names(self):
        self.assertEqual(L._pids({"definitely_not_a_running_program.py"}), set())


class TestParallelSlots(unittest.TestCase):
    """Parallel slots must isolate every endpoint and leave the scientific config untouched; sequential stays the default."""

    def cfg(self, arm="A0"):
        return L.make_config(L.build_base(), arm, SC.load()["heldout"][0], {})

    def test_slot_plans_are_pairwise_disjoint(self):
        plans = [L.slot_plan(k) for k in range(6)]
        for key in ("gossip_port", "camera_port", "instance_base", "sysid_base", "tmux_socket", "session"):
            vals = [p[key] for p in plans]
            self.assertEqual(len(set(vals)), len(vals), key)
        flat = [x for p in plans for x in p["agent_ports"]]
        self.assertEqual(len(set(flat)), len(flat))
        sitl = [5760 + 10 * p["instance_base"] for p in plans]
        self.assertEqual(len(set(sitl)), len(sitl))
        self.assertNotIn(14550, flat)                              # the QGC port is never an agent port

    def test_slot_config_changes_only_endpoints(self):
        c = self.cfg()
        s = L.slot_config(c, L.slot_plan(2))
        self.assertEqual(L.unslot(c), L.unslot(s))
        self.assertNotEqual(c, s)
        self.assertEqual(s["comms"]["port"], 5007)
        self.assertEqual(s["perception"]["camera_service"]["port"], 8767)
        self.assertTrue(s["perception"]["frame_source"]["url"].endswith(":8767"))
        self.assertEqual(s["agents"][0]["conn"], "udp:127.0.0.1:14575")
        for key in ("fire_model", "timing", "vehicle_defaults", "grid", "origin", "fog_tracker"):
            self.assertEqual(c[key], s[key])
        self.assertEqual(c["perception"]["conf_threshold"], s["perception"]["conf_threshold"])
        self.assertEqual(c["agents"][0]["seed"], s["agents"][0]["seed"])

    def test_slot_config_does_not_mutate_input_and_keeps_pairing(self):
        a, b = self.cfg("A0"), self.cfg("B2s0")
        before = copy.deepcopy(a)
        sa, sb = L.slot_config(a, L.slot_plan(1)), L.slot_config(b, L.slot_plan(2))
        self.assertEqual(a, before)
        self.assertEqual(L.unslot(L.strip_arm(sa)), L.unslot(L.strip_arm(sb)))   # different slots, same science

    def test_slot_zero_plan_starts_from_the_sequential_ports(self):
        p0 = L.slot_plan(0)
        self.assertEqual((p0["gossip_port"], p0["camera_port"], p0["agent_ports"][0], p0["instance_base"]), (5005, 8765, 14555, 1))

    def test_start_check_only_refuses_on_own_ports(self):
        import socket
        busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        try:
            plan = L.slot_plan(3)
            plan["camera_port"] = port                             # this slot's own camera port is taken
            self.assertIn(("camera/tcp", port), L.slot_ports_in_use(plan))
            other = L.slot_plan(4)                                  # another slot's ports are free: no refusal
            other["gossip_port"], other["camera_port"] = 45871, 45872
            other["agent_ports"] = [45873]
            other["instance_base"] = 4000                           # sitl tcp 45760+... never in use
            self.assertEqual(L.slot_ports_in_use(other), [])
        finally:
            busy.close()

    def test_start_check_ignores_time_wait_but_not_a_live_listener(self):
        import socket
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0)); srv.listen(1)
        port = srv.getsockname()[1]
        cli = socket.create_connection(("127.0.0.1", port))
        conn, _ = srv.accept()
        self.assertFalse(L._bind_free(port, socket.SOCK_STREAM))          # live listener: refused
        conn.close(); cli.close(); srv.close()                             # server side closes first -> TIME_WAIT on `port`
        self.assertTrue(L._bind_free(port, socket.SOCK_STREAM))           # TIME_WAIT only: allowed

    def test_start_check_sees_a_multicast_listener_on_the_gossip_port(self):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        port = s.getsockname()[1]
        try:
            plan = L.slot_plan(5)
            plan["gossip_port"] = port
            self.assertIn(("gossip/udp", port), L.slot_ports_in_use(plan))
        finally:
            s.close()

    def test_tagged_pids_find_only_this_runs_children(self):
        import subprocess
        import sys
        import time
        tag = f"unit-{os.getpid()}"
        mine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], env=dict(os.environ, L3_RUN_TAG=tag))
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], env=dict(os.environ, L3_RUN_TAG=tag + "-x"))
        try:
            time.sleep(0.5)
            found = L._tagged_pids(tag)
            self.assertEqual(found, {mine.pid})                    # not the other tag, not an unrelated process, not a prefix match
            L._signal_pids(found, 15)
            mine.wait(timeout=5)
            self.assertIsNone(other.poll())                        # the neighbour survived the cleanup
        finally:
            mine.kill()
            other.kill()

    def test_queue_skips_clean_runs_orders_by_scenario_and_ignores_unclean(self):
        from eval import level3_queue as Q
        with tempfile.TemporaryDirectory() as d:
            rows = [{"scenario": sc, "split": "heldout", "arm": arm, "config": os.path.join(d, f"l3-{arm}-{sc}.yaml")}
                    for sc in (2002, 2001) for arm in ("A0", "baseline")]
            json.dump({"runs": rows}, open(os.path.join(d, "matrix.json"), "w"))
            for arm, sc, meta in (("baseline", 2001, {"fog_exit_code": 0, "leftover_after_cleanup": []}),
                                  ("A0", 2001, {"fog_exit_code": 0, "leftover_after_cleanup": [123]}),
                                  ("baseline", 2002, {"fog_exit_code": 1, "leftover_after_cleanup": []})):
                os.makedirs(os.path.join(d, "results", f"l3-{arm}-{sc}"))
                json.dump(meta, open(os.path.join(d, "results", f"l3-{arm}-{sc}", "run_meta.json"), "w"))
            jobs = Q.build_jobs(d, repeats=["baseline:2001"])
            self.assertEqual([j["tag"] for j in jobs], ["l3-A0-2001", "l3-baseline-2002", "l3-A0-2002", "l3-baseline-2001"])
            self.assertEqual(jobs[-1]["kind"], "repeat")
            self.assertIn("results_repeat", jobs[-1]["out"])


if __name__ == "__main__":
    unittest.main()
