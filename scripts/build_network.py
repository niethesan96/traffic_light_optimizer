#!/usr/bin/env python3
"""
Build SUMO Network and Traffic Demand
======================================

CLI script to generate the complete SUMO simulation environment:
  1. 3x3 grid network with 9 signalized intersections
  2. Fixed-time signal plans for all intersections
  3. Configurable traffic demand

Usage:
    python scripts/build_network.py [--config configs/default.yaml] [--output-dir data/synthetic]

Output files in output-dir/network/:
    grid_3x3.nod.xml   — Node definitions
    grid_3x3.edg.xml   — Edge definitions
    grid_3x3.con.xml   — Connection definitions
    grid_3x3.add.xml   — Additional objects (TLS)
    grid_3x3.net.xml   — Compiled SUMO network

Output files in output-dir/demand/:
    demand.rou.xml     — Vehicle flows
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from sumo_network.grid_builder import GridBuilder, GridConfig
from sumo_network.traffic_demand import TrafficDemandGenerator, DemandConfig
from sumo_network.signal_plans import SignalPlanGenerator, SignalConfig


def _set_netxml_phase_durations(net_path: str | Path, plan) -> None:
    """
    Override phase durations in a compiled .net.xml with the desired values.

    netconvert auto-generates TLS programs with equal-duration phases.
    This function patches each phase's duration to match the signal plan,
    preserving the auto-generated phase states (which correctly match the
    number of controlled links per intersection).
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(net_path))
    root = tree.getroot()

    # Build a mapping from plan phase index → desired duration.
    # We match phases by their index (0..N-1) since netconvert preserves
    # the same phase structure from the connection topology.
    for tl in root.findall(".//tlLogic"):
        phases = tl.findall("phase")
        n_plan = len(plan.phases)
        n_net = len(phases)

        if n_plan == n_net:
            # Direct 1:1 mapping
            for i, phase_elem in enumerate(phases):
                phase_elem.set("duration", str(int(plan.phases[i].duration)))
        elif n_plan > n_net and n_net >= 2:
            # netconvert merged some phases (e.g., merged yellow+all_red)
            # Map green phases by index, leave others as-is
            plan_green = [i for i, p in enumerate(plan.phases)
                          if "green" in p.name.lower()]
            net_green = [i for i, p in enumerate(phases)
                         if any(ch == "G" for ch in p.get("state", ""))]
            for pi, ni in zip(plan_green, net_green):
                phases[ni].set("duration", str(int(plan.phases[pi].duration)))

    tree.write(str(net_path), xml_declaration=True)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_network(config_path: str, output_dir: str, network_name: str | None = None, flow_rate: int | None = None) -> dict:
    """
    Build the complete SUMO network and traffic demand.

    Args:
        config_path: Path to the YAML configuration file.
        output_dir: Output directory for generated files.
        network_name: Name prefix for network files.
        flow_rate: Optional override for demand flow_rate (veh/hr).

    Returns:
        Dict with paths to all generated files.
    """
    config = load_config(config_path)
    network_name = network_name or config.get("network", {}).get("name", "grid_3x3")
    output_path = Path(output_dir)

    print("=" * 60)
    print("Multi-Agent Traffic Network — Build Script")
    print("=" * 60)

    # --- Step 1: Build grid network ---
    print("\n[1/4] Building grid network...")
    grid_config = GridConfig(
        rows=config.get("network", {}).get("grid", {}).get("rows", 3),
        cols=config.get("network", {}).get("grid", {}).get("cols", 3),
        block_length=config.get("network", {}).get("block_length", 300.0),
        lanes_per_direction=config.get("network", {}).get("lanes_per_direction", 1),
        speed_limit=config.get("network", {}).get("speed_limit", 8.33),
        bidi=config.get("network", {}).get("bidi", True),
        network_name=network_name,
        boundary_ring=config.get("network", {}).get("boundary_ring", False),
    )
    grid = GridBuilder(grid_config)

    # Print network summary
    print(grid.summary())

    # --- Step 2: Generate signal plans ---
    print("\n[2/4] Generating signal plans...")
    signal_config = SignalConfig(
        cycle_length=config.get("signals", {}).get("cycle_length", 90.0),
        yellow_time=config.get("signals", {}).get("yellow_time", 3.0),
        all_red_time=config.get("signals", {}).get("all_red_time", 1.0),
    )

    # Load phases from config if available
    raw_phases = config.get("signals", {}).get("phases", [])
    if raw_phases:
        signal_config.phases = [
            {
                "name": p.get("name", f"phase_{i}"),
                "duration": p.get("duration", 40),
                "state": p.get("signal_state", p.get("state", "GGrr")),
            }
            for i, p in enumerate(raw_phases)
        ]

    sig_gen = SignalPlanGenerator(signal_config)

    # Create signal plan and generate TLS XML
    plan = sig_gen.create_plan_from_config() if signal_config.phases else sig_gen.create_default_2phase_plan()

    print(f"  Signal plan: {plan.cycle_length}s cycle, {plan.num_phases} phases")
    for i, phase in enumerate(plan.phases):
        print(f"    Phase {i}: {phase.name} ({phase.duration}s) — {phase.state}")

    # Generate .add.xml with per-intersection TLS programs.
    # Each intersection may have a different number of controlled links,
    # so we build state strings from scratch based on the NS/EW link layout.
    from collections import Counter
    network_dir = output_path / "network"
    network_dir.mkdir(parents=True, exist_ok=True)

    # Classify each connection as NS or EW at each intersection.
    # NS edges run vertically (same column, different row),
    # EW edges run horizontally (same row, different column).
    ns_link_count: dict[str, int] = {nid: 0 for nid in grid.intersection_nodes}
    ew_link_count: dict[str, int] = {nid: 0 for nid in grid.intersection_nodes}
    for conn in grid.connections:
        if conn.tl is None:
            continue
        # Determine direction: same column → NS, same row → EW
        from_parts = conn.from_edge.split("_to_")[0].split("_")  # ['E', 'r1', 'c1']
        to_parts = conn.from_edge.split("_to_")[0].split("_")
        # Edge format: E_fromR_fromC_to_toR_toC
        ef = conn.from_edge.split("_to_")
        from_r, from_c = int(ef[0].split("_")[1]), int(ef[0].split("_")[2])
        to_r, to_c = int(ef[1].split("_")[0]), int(ef[1].split("_")[1])
        if from_c == to_c:  # same column → vertical → NS
            ns_link_count[conn.tl] += 1
        else:  # same row → horizontal → EW
            ew_link_count[conn.tl] += 1

    # Build per-intersection TLS XML with correct state strings.
    import xml.etree.ElementTree as ET
    root = ET.Element("additional")
    for node_id in grid.intersection_nodes:
        ns_n = ns_link_count.get(node_id, 0)
        ew_n = ew_link_count.get(node_id, 0)
        total = ns_n + ew_n
        if total == 0:
            continue  # no controlled links

        tls = ET.SubElement(root, "tlLogic", id=node_id, type="static",
                            programID="0", offset="0")
        for phase in plan.phases:
            # Build state: NS links first, then EW links
            if "green" in phase.name.lower() and "ew" not in phase.name.lower():
                state = "G" * ns_n + "r" * ew_n
            elif "green" in phase.name.lower() and "ew" in phase.name.lower():
                state = "r" * ns_n + "G" * ew_n
            elif "yellow" in phase.name.lower() and "ew" not in phase.name.lower():
                state = "y" * ns_n + "r" * ew_n
            elif "yellow" in phase.name.lower() and "ew" in phase.name.lower():
                state = "r" * ns_n + "y" * ew_n
            else:  # all_red or other
                state = "r" * total
            ET.SubElement(tls, "phase",
                          duration=str(int(phase.duration)),
                          state=state)

    tls_path = network_dir / f"{network_name}.add.xml"
    tls_tree = ET.ElementTree(root)
    ET.indent(tls_tree, space="  ")
    tls_tree.write(tls_path, xml_declaration=True)
    print(f"  Written: {tls_path}")

    # --- Step 3: Build all network files ---
    print("\n[3/4] Generating network XML files...")
    files = grid.build_all(network_dir, network_name)

    # Post-process the net.xml to set phase durations from our config.
    # netconvert auto-generates TLS with equal durations; we override them.
    _set_netxml_phase_durations(files["network"], plan)

    print(f"  Nodes:       {files['nodes']}")
    print(f"  Edges:       {files['edges']}")
    print(f"  Connections: {files['connections']}")
    print(f"  Additional:  {files['additional']}")
    print(f"  Network:     {files['network']}")

    # --- Step 4: Generate traffic demand ---
    print("\n[4/4] Generating traffic demand...")
    demand_config = DemandConfig(
        duration=config.get("demand", {}).get("duration", 3600),
        seed=config.get("demand", {}).get("seed", 42),
        flow_rate=flow_rate if flow_rate is not None else config.get("demand", {}).get("flow_rate", 300),
        demand_level=config.get("demand", {}).get("demand_level", "medium"),
        demand_presets=config.get("demand", {}).get("demand_presets", {"low": 100, "medium": 300, "high": 600}),
        depart_position=config.get("demand", {}).get("depart_position", "random_free"),
        depart_speed=config.get("demand", {}).get("depart_speed", "max"),
        min_gap=config.get("demand", {}).get("min_gap", 2.5),
        corridor_weights=config.get("demand", {}).get("corridor_weights"),
        platoon_period=config.get("demand", {}).get("platoon_period"),
        platoon_on_time=config.get("demand", {}).get("platoon_on_time",30),
        demand_jitter=config.get("demand", {}).get("demand_jitter",0.1),
    )

    demand_gen = TrafficDemandGenerator(demand_config)
    print(demand_gen.summary())

    demand_dir = output_path / "demand"
    demand_dir.mkdir(parents=True, exist_ok=True)

    demand_type = config.get("demand", {}).get("demand_type", "boundary")
    if demand_type == "network_wide":
        rou_path = demand_gen.generate_network_wide_with_grid_builder(
            grid, demand_dir / "demand.rou.xml"
        )
    else:
        rou_path = demand_gen.generate_with_grid_builder(
            grid, demand_dir / "demand.rou.xml"
        )
    print(f"  Written: {rou_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print(f"\nNetwork files:  {network_dir}")
    print(f"Demand files:   {demand_dir}")
    print(f"\nIntersections:  {len(grid.intersection_nodes)}")
    print(f"Edges:          {len(grid.edges)}")
    print(f"Connections:    {len(grid.connections)}")
    print(f"\nTo run simulation:")
    print(f"  python scripts/run_simulation.py --config {config_path}")
    print(f"\nTo run tests:")
    print(f"  python -m pytest tests/ -v")

    return {
        "network_dir": str(network_dir),
        "demand_dir": str(demand_dir),
        "net_file": str(files["network"]),
        "rou_file": str(rou_path),
        "intersection_count": len(grid.intersection_nodes),
        "edge_count": len(grid.edges),
        "connection_count": len(grid.connections),
    }


def main():
    parser = argparse.ArgumentParser(description="Build SUMO network and traffic demand")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--network-name",
        type=str,
        default=None,
        help="Name prefix for network files",
    )
    parser.add_argument(
        "--flow-rate",
        type=int,
        default=None,
        help="Override demand flow_rate (veh/hr) from the config",
    )
    args = parser.parse_args()

    result = build_network(args.config, args.output_dir, args.network_name, flow_rate=args.flow_rate)
    return result


if __name__ == "__main__":
    main()
