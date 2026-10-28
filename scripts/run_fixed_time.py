#!/usr/bin/env python3
"""
Fixed-Time Baseline Simulation Runner
=======================================

Runs a SUMO simulation with fixed-time traffic signal control.
Collects network-level and per-intersection metrics for baseline comparison
against future RL controllers.

Usage:
    python scripts/run_fixed_time.py --config configs/default.yaml
    python scripts/run_fixed_time.py --config configs/default.yaml --duration 300
    python scripts/run_fixed_time.py --config configs/default.yaml --gui
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path# Add project root to path
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

from sumo_network.grid_builder import GridBuilder, GridConfig
from sumo_network.signal_plans import SignalPlanGenerator, SignalConfig
from sumo_network.traffic_demand import TrafficDemandGenerator, DemandConfig
from evaluation.metrics import (
    MetricsCollector,
    SimulationResults,
    compute_travel_times_from_tripinfo,
    save_results_csv,
    save_results_json,
    print_summary,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_sumo_binary(use_gui: bool = False) -> str:
    """Find the SUMO binary (sumo or sumo-gui)."""
    import shutil as _shutil

    binary_name = "sumo-gui" if use_gui else "sumo"

    on_path = _shutil.which(binary_name)
    if on_path:
        return on_path

    try:
        import sumo as sumo_pkg
        sumo_home = Path(sumo_pkg.SUMO_HOME)
        candidate = sumo_home / "bin" / (f"{binary_name}.exe" if os.name == "nt" else binary_name)
        if candidate.exists():
            return str(candidate)
    except (ImportError, AttributeError):
        pass

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        suffix = ".exe" if os.name == "nt" else ""
        candidate = Path(sumo_home) / "bin" / f"{binary_name}{suffix}"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not find {binary_name} binary. "
        "Ensure SUMO is installed and SUMO_HOME is set."
    )


def run_fixed_time_simulation(
    config_path: str,
    duration: int | None = None,
    use_gui: bool = False,
    output_dir: str | None = None,
) -> SimulationResults:
    """
    Run a fixed-time baseline SUMO simulation.

    Args:
        config_path: Path to YAML configuration file.
        duration: Override simulation duration (seconds). None = use config.
        use_gui: Whether to use sumo-gui for visualization.
        output_dir: Override output directory for results.

    Returns:
        SimulationResults with all collected metrics.
    """
    import traci

    config = load_config(config_path)

    # --- Resolve paths ---
    network_dir = Path(config.get("paths", {}).get("network_dir", "data/synthetic/network"))
    demand_dir = Path(config.get("paths", {}).get("demand_dir", "data/synthetic/demand"))
    network_name = config.get("network", {}).get("name", "grid_3x3")
    net_file = network_dir / f"{network_name}.net.xml"
    rou_file = demand_dir / "demand.rou.xml"

    if not net_file.exists():
        raise FileNotFoundError(f"Network file not found: {net_file}\nRun 'python scripts/build_network.py' first.")
    if not rou_file.exists():
        raise FileNotFoundError(f"Demand file not found: {rou_file}\nRun 'python scripts/build_network.py' first.")

    # --- Simulation settings ---
    sim_end_time = duration or config.get("sumo", {}).get("end_time", 3600)
    step_length = config.get("sumo", {}).get("step_length", 1)
    seed = config.get("sumo", {}).get("seed", 42)
    tls_type = config.get("sumo", {}).get("tls_type", "static")

    print("=" * 60)
    print("  Fixed-Time Baseline Simulation")
    print("=" * 60)
    print(f"  Config:   {config_path}")
    print(f"  Network:  {net_file}")
    print(f"  Demand:   {rou_file}")
    print(f"  Duration: {sim_end_time}s ({sim_end_time / 60:.1f} min)")
    print(f"  Step:     {step_length}s")
    print(f"  Seed:     {seed}")
    print(f"  TLS type: {tls_type}")
    print(f"  GUI:      {use_gui}")
    print()

    # --- Setup SUMO config ---
    tmp_dir = Path(tempfile.mkdtemp(prefix="sumo_ft_"))
    sumo_cfg = tmp_dir / "sumo.sumocfg"
    tripinfo_path = tmp_dir / "tripinfo.xml"

    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file.resolve()}"/>
        <route-files value="{rou_file.resolve()}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{sim_end_time}"/>
        <step-length value="{step_length}"/>
    </time>
    <random>
        <seed value="{seed}"/>
    </random>
    <output>
        <tripinfo-output value="{tripinfo_path}"/>
    </output>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
"""
    sumo_cfg.write_text(cfg_content)

    # --- Determine intersection info for per-intersection metrics ---
    grid_config = GridConfig(
        rows=config.get("network", {}).get("grid", {}).get("rows", 3),
        cols=config.get("network", {}).get("grid", {}).get("cols", 3),
        block_length=config.get("network", {}).get("block_length", 300.0),
        lanes_per_direction=config.get("network", {}).get("lanes_per_direction", 1),
        speed_limit=config.get("network", {}).get("speed_limit", 8.33),
        bidi=config.get("network", {}).get("bidi", True),
    )
    grid = GridBuilder(grid_config)

    # Build mapping: intersection node_id -> incoming edge IDs
    # Use edge topology (edges pointing TO the intersection node) rather than
    # connections, because corner intersections may have no explicit connections
    # but SUMO still routes vehicles through them.
    incoming_edges: dict[str, list[str]] = {iid: [] for iid in grid.intersection_nodes}
    for edge in grid.edges:
        if edge.to_node in incoming_edges:
            if edge.id not in incoming_edges[edge.to_node]:
                incoming_edges[edge.to_node].append(edge.id)

    # --- Setup metrics collector ---
    collector = MetricsCollector()
    collector.setup(
        intersection_ids=grid.intersection_ids if hasattr(grid, 'intersection_ids') else grid.intersection_nodes,
        incoming_edges=incoming_edges,
        config={
            "config_path": config_path,
            "duration": sim_end_time,
            "step_length": step_length,
            "seed": seed,
            "tls_type": tls_type,
            "demand_level": config.get("demand", {}).get("demand_level", "medium"),
            "flow_rate": config.get("demand", {}).get("flow_rate", 300),
        },
    )

    # --- Find SUMO binary and start simulation ---
    try:
        import sumo as sumo_pkg
        sumo_home = Path(sumo_pkg.SUMO_HOME)
        candidate = sumo_home / "bin" / ("sumo-gui.exe" if use_gui else "sumo.exe")
        sumo_bin = str(candidate) if candidate.exists() else find_sumo_binary(use_gui)
    except Exception:
        sumo_bin = find_sumo_binary(use_gui)

    print(f"  SUMO bin: {sumo_bin}")
    print()

    sumo_cmd = [sumo_bin, "-c", str(sumo_cfg), "--no-step-log"]
    print("  Starting SUMO...")
    traci.start(sumo_cmd)

    try:
        collector.start()

        # Verify traffic lights
        tl_ids = traci.trafficlight.getIDList()
        expected = config.get("network", {}).get("grid", {}).get("rows", 3) * \
                   config.get("network", {}).get("grid", {}).get("cols", 3)
        print(f"  Traffic lights found: {len(tl_ids)} / {expected} expected")
        if len(tl_ids) < expected:
            print(f"  WARNING: Fewer traffic lights than expected!")

        # Run simulation loop
        max_steps = int(sim_end_time / step_length)
        print(f"  Running {max_steps} steps...\n")

        teleports_started = 0
        teleports_ended = 0

        for step in range(max_steps):
            traci.simulationStep()
            sim_time = (step + 1) * step_length

            teleports_started += traci.simulation.getStartingTeleportNumber()
            teleports_ended += traci.simulation.getEndingTeleportNumber()

            collector.collect_step(traci, sim_time)

            # Progress log every 10%
            if (step + 1) % max(1, max_steps // 10) == 0 or step == max_steps - 1:
                pct = ((step + 1) / max_steps) * 100
                veh = traci.vehicle.getIDCount()
                entered = collector._total_vehicles_entered
                left = collector._total_vehicles_left
                print(f"    Step {step + 1}/{max_steps} ({pct:.0f}%) — "
                      f"Active: {veh}, Entered: {entered}, Left: {left}")

        # Finalize metrics (without tripinfo first — it's written on traci.close())
        results = collector.finalize(traci)

    finally:
        traci.close()

    # Now re-finalize with tripinfo for accurate per-vehicle stats
    # (traci.close() writes the tripinfo file)
    if tripinfo_path.exists():
        results = collector.finalize(None, tripinfo_path=tripinfo_path)
        print(f"\n  Parsed {results.throughput} completed trips from tripinfo.xml")
    else:
        print("\n  WARNING: tripinfo.xml not found, using step-based metrics")

    results.teleports = teleports_started

    # --- Save results ---
    out = Path(output_dir) if output_dir else Path("results/baseline")
    out.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = save_results_json(results, out / "baseline_results.json")
    print(f"\n  Results JSON: {json_path}")

    # Save CSVs
    csv_paths = save_results_csv(results, out)
    for name, path in csv_paths.items():
        print(f"  Results CSV:  {path}")

    # Copy tripinfo if it exists
    if tripinfo_path.exists():
        tripinfo_dest = out / "tripinfo.xml"
        shutil.copy2(tripinfo_path, tripinfo_dest)
        print(f"  Tripinfo:     {tripinfo_dest}")

    # --- Print summary ---
    print_summary(results)

    # --- Cleanup ---
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


# Need numpy for avg_travel_time
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run fixed-time baseline simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_fixed_time.py --config configs/default.yaml
  python scripts/run_fixed_time.py --config configs/default.yaml --duration 300
  python scripts/run_fixed_time.py --config configs/default.yaml --output-dir results/baseline
        """,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Simulation duration in seconds (overrides config)",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Use sumo-gui for visualization",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: results/baseline)",
    )
    args = parser.parse_args()

    results = run_fixed_time_simulation(
        config_path=args.config,
        duration=args.duration,
        use_gui=args.gui,
        output_dir=args.output_dir,
    )

    # Exit with success if any vehicles entered
    if results.vehicles_entered == 0:
        print("\n  WARNING: No vehicles entered the network!")
        sys.exit(1)


if __name__ == "__main__":
    main()
