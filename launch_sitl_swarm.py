#!/usr/bin/env python3
"""
launch_sitl_swarm.py
Launch or clean up multiple ArduCopter SITL drones (WITH MAVProxy) in tmux.

Each instance outputs to:
  - UDP:<QGC_IP>:14550 (for QGroundControl which ever ip --qgc <QGC_IP> is set)
  - UDP:127.0.0.1:14550 (for QGroundControl default)
  - UDP:127.0.0.1:(14555 + i) (for swarm_agent.py)

Usage:
  Launch drones (default UH location):
      python3 launch_sitl_swarm.py --count 3

  Random spawn within a 5-mile radius around UH (reproducible with seed):
      python3 launch_sitl_swarm.py --count 3 --randomspawn --seed 42

  Change radius (miles):
      python3 launch_sitl_swarm.py --count 4 --randomspawn --radius-miles 2.5

  Change QGroundControl IP (remote machine)
      python3 launch_sitl_swarm.py --count 3 --qgc 198.15.2.2

  Attach to tmux (view tmux terminal):
      tmux attach -t sitl-swarm

  Kill all running drones:
      python3 launch_sitl_swarm.py --killdrone
"""

import argparse, subprocess, shutil, sys, time, random, math

# Reference for -L UH
UH_LAT = 21.297
UH_LON = -157.817
DEFAULT_ALT_M = 150.0
DEFAULT_HDG_DEG = 90.0

def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def run_quiet(cmd: list):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def kill_all_drones(session: str):
    print(f"[🧹] Cleaning up session '{session}' and SITL processes...")
    run_quiet(["tmux", "kill-session", "-t", session])
    run_quiet(["pkill", "-f", "ArduCopter"])
    run_quiet(["pkill", "-f", "sim_vehicle.py"])
    run_quiet(["pkill", "-f", "mavproxy.py"])
    print("[✅] All SITL/MAVProxy sessions terminated.\n")

def tmux(cmd: list, check=True):
    return subprocess.run(["tmux", *cmd], check=check)

def meters_to_deg_offsets(lat_deg: float, north_m: float, east_m: float):
    dlat = north_m / 111_320.0
    dlon = east_m  / (111_320.0 * math.cos(math.radians(lat_deg)))
    return dlat, dlon

def random_offset_m(rng: random.Random, radius_m: float):
    theta = rng.uniform(0.0, 2.0 * math.pi)
    r = radius_m * math.sqrt(rng.random())
    north = r * math.cos(theta)
    east  = r * math.sin(theta)
    return north, east

def build_location_arg(use_random: bool, rng: random.Random | None, radius_m: float):
    """Return either '-L UH' or '--custom-location=lat,lon,alt,hdg'"""
    if not use_random:
        return "-L UH"
    north_m, east_m = random_offset_m(rng, radius_m)
    dlat, dlon = meters_to_deg_offsets(UH_LAT, north_m, east_m)
    lat = UH_LAT + dlat
    lon = UH_LON + dlon
    return f'--custom-location="{lat:.7f},{lon:.7f},{DEFAULT_ALT_M:.1f},{DEFAULT_HDG_DEG:.1f}"'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2, help="Number of SITL drones to launch")
    ap.add_argument("--session", default="sitl-swarm", help="tmux session name (default: sitl-swarm)")
    ap.add_argument("--vehicle", default="ArduCopter", help="Vehicle type (default: ArduCopter)")
    ap.add_argument("--loc", default="UH", help="ArduPilot location label (default: UH)")
    ap.add_argument("--base-port", dest="base_port", type=int, default=14555, help="Base UDP port for swarm_agent connections")
    ap.add_argument("--delay", type=float, default=15.0, help="Seconds between drone launches")
    ap.add_argument("--killdrone", action="store_true", help="Kill all SITL and MAVProxy processes, then exit")

    # New QGroundControl IP option
    ap.add_argument("--qgc", default="127.0.0.1",
                    help="IP address of QGroundControl (default: 127.0.0.1)")

    # Random spawn controls
    ap.add_argument("--randomspawn", action="store_true",
                    help="Spawn each drone randomly within a radius around UH")
    ap.add_argument("--seed", type=int, default=None, help="Seed for reproducible random spawns")
    ap.add_argument("--radius-miles", type=float, default=5.0,
                    help="Random spawn radius in miles (default 5.0)")
    args = ap.parse_args()

    if args.killdrone:
        kill_all_drones(args.session)
        sys.exit(0)

    if not have("tmux"):
        print("ERROR: tmux not found. Install: sudo apt install tmux", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed) if args.randomspawn else None
    radius_m = args.radius_miles * 1609.344  # miles to meters

    # Kill old session and create new one
    run_quiet(["tmux", "kill-session", "-t", args.session])
    tmux(["new-session", "-d", "-s", args.session, "-n", "SITL"])

    def send(target, line):
        tmux(["send-keys", "-t", target, line, "C-m"])

    for i in range(args.count):
        inst  = i + 1
        sysid = i + 1
        port  = args.base_port + i

        loc_arg = build_location_arg(args.randomspawn, rng, radius_m) if args.randomspawn else f"-L {args.loc}"

        # <-- Updated here: QGC IP can be changed -->
        cmd = (
            f"sim_vehicle.py -v {args.vehicle} "
            f"-I {inst} --sysid {sysid} {loc_arg} "
            f"--no-rebuild "
            f"--out=udp:{args.qgc}:14550 "    # QGroundControl IP
            f"--out=udp:127.0.0.1:{port}"     # local brain port
        )

        if i == 0:
            target = f"{args.session}:SITL.0"
            send(target, f"printf '\\033]0;DRONE_{inst}\\007'")
            send(target, cmd)
        else:
            tmux(["split-window", "-t", f"{args.session}:SITL", "-v"])
            tmux(["select-layout", "-t", f"{args.session}:SITL", "tiled"])
            send(f"{args.session}:SITL", f"printf '\\033]0;DRONE_{inst}\\007'")
            send(f"{args.session}:SITL", cmd)

        time.sleep(args.delay)

    tmux(["select-layout", "-t", f"{args.session}:SITL", "tiled"])

    print(f"\n✅ Launched {args.count} SITL drone(s) in tmux session '{args.session}'.")
    if args.randomspawn:
        print(f"   Random spawn: radius {args.radius_miles} miles around UH  (seed={args.seed})")
    print(f"   QGroundControl IP: {args.qgc}")
    print("Attach tmux:\n  tmux attach -t", args.session)
    print("Detach (leave running): Ctrl+b then d\n")
    print("Each drone outputs to:")
    for i in range(args.count):
        print(f"  UDP {args.qgc}:14550 (QGroundControl)")
        print(f"  UDP 127.0.0.1:{args.base_port + i} (for swarm_agent)")
    print("\nQuick start examples:")
    print("  # Launch 4 SITL + swarm + fog (fire demo config):")
    print("  python3 launch_sitl_swarm.py --count 4 --base-port 14555")
    print("  python3 launch_experiment.py --config config/exp_swarm_fire.yaml\n")
    print("  # Direct agent connect (single):")
    print(f"  python3 core/swarm_agent.py --conn udp:127.0.0.1:{args.base_port} "
          f"--origin-lat {UH_LAT} --origin-lon {UH_LON} --mission grid_frontier")
    print("\nCleanup:\n  python3 launch_sitl_swarm.py --killdrone\n")

if __name__ == "__main__":
    main()
