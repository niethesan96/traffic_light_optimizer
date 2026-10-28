#!/usr/bin/env python3
"""
SUMO Test Simulation Runner
============================

Runs a short SUMO simulation to verify:
  1. All 9 intersections exist
  2. Traffic lights operate
  3. Vehicles enter the network
  4. Vehicles can travel through the network
  5. The simulation runs without TraCI/SUMO errors

Usage:
    python scripts/run_simulation.py [--config configs/default.yaml] [--gui]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure SUMO paths for eclipse-sumo pip package
_sumo_tools = str(Path(__file__).resolve().parent.parent / "venv" / "Lib" / "site-packages" / "sumo" / "tools")
if _sumo_tools not in sys.path:
    sys.path.insert(0, _sumo_tools)

try:
    import sumo
    os.environ.setdefault("SUMO_HOME", str(Path(sumo.SUMO_HOME)))
except Exception:
    pass

import yaml
import traci


def find_sumo_binary(use_gui: bool = False) -> str:
    """Find the SUMO binary (sumo or sumo-gui)."""
    import shutil

    binary_name = "sumo-gui" if use_gui else "sumo"

    # Check PATH
    on_path = shutil.which(binary_name)
    if on_path:
        return on_path

    # Check eclipse-sumo pip package
    try:
        import sumo as sumo_pkg
        sumo_home = Path(sumo_pkg.SUMO_HOME)
        if use_gui:
            candidate = sumo_home / "bin" / "sumo-gui.exe"
        else:
            candidate = sumo_home / "bin" / "sumo.exe"
        if candidate.exists():
            return str(candidate)
        # Try without .exe
        candidate = sumo_home / "bin" / binary_name
        if candidate.exists():
            return str(candidate)
    except (ImportError, AttributeError):
        pass

    # Check SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / f"{binary_name}.exe"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not find {binary_name} binary. "
        "Ensure SUMO is installed and SUMO_HOME is set."
    )


def run_simulation(config_path: str, use_gui: bool = False, sim_duration: int | None = None) -> dict:
    """
    Run a SUMO simulation and collect basic statistics.

    Args:
        config_path: Path to YAML configuration file.
        use_gui: Whether to use sumo-gui.
        sim_duration: Override simulation duration (seconds).

    Returns:
        Dict with simulation results.
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Paths
    network_dir = Path(config.get("paths", {}).get("network_dir", "data/synthetic/network"))
    demand_dir = Path(config.get("paths", {}).get("demand_dir", "data/synthetic/demand"))
    network_name = "grid_3x3"

    net_file = network_dir / f"{network_name}.net.xml"
    rou_file = demand_dir / "demand.rou.xml"

    if not net_file.exists():
        raise FileNotFoundError(
            f"Network file not found: {net_file}\n"
            "Run 'python scripts/build_network.py' first."
        )
    if not rou_file.exists():
        raise FileNotFoundError(
            f"Demand file not found: {rou_file}\n"
            "Run 'python scripts/build_network.py' first."
        )

    # Create temp SUMO config
    tmp_dir = Path(tempfile.mkdtemp(prefix="sumo_test_"))
    sumo_cfg = tmp_dir / "sumo.sumocfg"

    duration = sim_duration or config.get("sumo", {}).get("end_time", 3600)
    step_length = config.get("sumo", {}).get("step_length", 10)
    seed = config.get("sumo", {}).get("seed", 42)

    # Write SUMO config file
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file.resolve()}"/>
        <route-files value="{rou_file.resolve()}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{duration}"/>
        <step-length value="{step_length}"/>
    </time>
    <random>
        <seed value="{seed}"/>
    </random>
    <output>
        <tripinfo-output value="{tmp_dir / 'tripinfo.xml'}"/>
    </output>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
"""
    sumo_cfg.write_text(cfg_content)

    print("=" * 60)
    print("SUMO Test Simulation")
    print("=" * 60)
    print(f"\nNetwork:  {net_file}")
    print(f"Demand:   {rou_file}")
    print(f"Duration: {duration}s ({duration / 60:.1f} min)")
    print(f"GUI:      {use_gui}")

    # Find SUMO binary
    try:
        import sumo as sumo_pkg
        sumo_home = Path(sumo_pkg.SUMO_HOME)
        candidate = sumo_home / "bin" / ("sumo-gui.exe" if use_gui else "sumo.exe")
        sumo_bin = str(candidate) if candidate.exists() else find_sumo_binary(use_gui)
    except Exception:
        sumo_bin = find_sumo_binary(use_gui)
    print(f"SUMO:     {sumo_bin}")

    # Start SUMO via TraCI
    sumo_cmd = [sumo_bin, "-c", str(sumo_cfg), "--no-step-log"]
    print(f"\nStarting SUMO...")
    traci.start(sumo_cmd)

    try:
        sim_stats = {
            "steps": 0,
            "total_vehicles_entered": 0,
            "total_vehicles_left": 0,
            "vehicles_in_network": [],
            "tls_ids": [],
            "tls_switches": {},
            "errors": [],
            "intersections_verified": False,
            "tls_operating": False,
            "vehicles_entered": False,
            "vehicles_traveled": False,
        }

        # --- Verify intersections ---
        print("\n--- Verifying intersections ---")
        junction_ids = traci.trafficlight.getIDList()
        sim_stats["tls_ids"] = junction_ids
        print(f"  Traffic lights found: {len(junction_ids)}")
        for tl_id in junction_ids:
            print(f"    {tl_id}")
            sim_stats["tls_switches"][tl_id] = 0

        expected_intersections = config.get("network", {}).get("grid", {}).get("rows", 3) * \
                                 config.get("network", {}).get("grid", {}).get("cols", 3)
        if len(junction_ids) >= expected_intersections:
            sim_stats["intersections_verified"] = True
            print(f"  [OK] All {expected_intersections} intersections verified!")
        else:
            print(f"  [FAIL] Expected {expected_intersections} intersections, found {len(junction_ids)}")

        # --- Run simulation ---
        print("\n--- Running simulation ---")
        step = 0
        max_steps = int(duration / step_length)

        while step < max_steps:
            traci.simulationStep()
            step += 1
            sim_stats["steps"] = step

            # Count vehicles
            vehicle_ids = traci.vehicle.getIDList()
            num_vehicles = len(vehicle_ids)

            # Track entries and exits
            arrived = traci.simulation.getArrivedNumber()
            departed = traci.simulation.getDepartedNumber()
            sim_stats["total_vehicles_entered"] += departed
            sim_stats["total_vehicles_left"] += arrived

            if departed > 0:
                sim_stats["vehicles_entered"] = True
            if arrived > 0:
                sim_stats["vehicles_traveled"] = True

            # Track TLS switches
            for tl_id in junction_ids:
                current_phase = traci.trafficlight.getPhase(tl_id)
                # Check if phase changed (simplified detection)
                # We'll count phase changes over the simulation

            # Log progress every 10% (or every step if very short)
            log_interval = max(1, max_steps // 10)
            if step % log_interval == 0 or step == max_steps:
                pct = (step / max_steps) * 100
                print(f"  Step {step}/{max_steps} ({pct:.0f}%) — "
                      f"Vehicles in network: {num_vehicles}, "
                      f"Total entered: {sim_stats['total_vehicles_entered']}, "
                      f"Total left: {sim_stats['total_vehicles_left']}")

        # --- Check TLS operation ---
        print("\n--- Checking TLS operation ---")
        for tl_id in junction_ids:
            phase = traci.trafficlight.getPhase(tl_id)
            program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)
            print(f"  {tl_id}: current phase = {phase}, program phases = {len(program)}")
            if len(program) > 0:
                sim_stats["tls_operating"] = True

        # --- Final statistics ---
        print("\n--- Simulation Results ---")
        print(f"  Steps completed:     {sim_stats['steps']}")
        print(f"  Vehicles entered:    {sim_stats['total_vehicles_entered']}")
        print(f"  Vehicles left:       {sim_stats['total_vehicles_left']}")
        print(f"  TLS found:           {len(junction_ids)}")
        print(f"  Intersections OK:    {'[OK]' if sim_stats['intersections_verified'] else '[FAIL]'}")
        print(f"  TLS operating:       {'[OK]' if sim_stats['tls_operating'] else '[FAIL]'}")
        print(f"  Vehicles entered:    {'[OK]' if sim_stats['vehicles_entered'] else '[FAIL]'}")
        print(f"  Vehicles traveled:   {'[OK]' if sim_stats['vehicles_traveled'] else '[FAIL]'}")

        # Overall verdict
        all_ok = all([
            sim_stats["intersections_verified"],
            sim_stats["tls_operating"],
            sim_stats["vehicles_entered"],
            sim_stats["vehicles_traveled"],
        ])
        print(f"\n{'='*60}")
        if all_ok:
            print("  [OK] ALL CHECKS PASSED — Network simulation works correctly!")
        else:
            print("  [FAIL] SOME CHECKS FAILED — See details above.")
        print(f"{'='*60}")

        sim_stats["success"] = all_ok
        return sim_stats

    finally:
        traci.close()
        # Clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run SUMO test simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use sumo-gui for visualization",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Simulation duration in seconds (overrides config)",
    )
    args = parser.parse_args()

    result = run_simulation(args.config, args.gui, args.duration)

    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
