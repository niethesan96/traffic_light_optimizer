"""
SUMO Simulation Metrics Collector
==================================

Collects network-level and per-intersection metrics from a running
SUMO simulation via TraCI.

Network-level metrics:
  - total_waiting_time
  - avg_waiting_time
  - avg_queue_length
  - max_queue_length
  - avg_travel_time
  - throughput (vehicles completed)
  - vehicles_entered
  - vehicles_remaining
  - simulation_duration

Per-intersection metrics:
  - queue_length
  - waiting_time
  - vehicle_count
  - mean_speed
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StepSnapshot:
    """Metrics collected at a single simulation step."""

    time: float
    num_vehicles: int
    total_waiting_time: float  # sum of all vehicles' waiting times this step
    avg_waiting_time: float
    queue_lengths: dict[str, int]  # per-intersection queue length
    vehicle_speeds: dict[str, float]  # per-intersection mean speed
    vehicles_per_intersection: dict[str, int]  # per-intersection vehicle count


@dataclass
class IntersectionMetrics:
    """Accumulated per-intersection metrics over the full simulation."""

    intersection_id: str
    total_queue_length: int = 0
    total_waiting_time: float = 0.0
    total_vehicle_count: int = 0
    step_count: int = 0
    max_queue_length: int = 0
    speed_samples: list[float] = field(default_factory=list)

    @property
    def avg_queue_length(self) -> float:
        return self.total_queue_length / max(self.step_count, 1)

    @property
    def avg_waiting_time(self) -> float:
        return self.total_waiting_time / max(self.step_count, 1)

    @property
    def avg_vehicle_count(self) -> float:
        return self.total_vehicle_count / max(self.step_count, 1)

    @property
    def avg_speed(self) -> float:
        return float(np.mean(self.speed_samples)) if self.speed_samples else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "intersection_id": self.intersection_id,
            "avg_queue_length": round(self.avg_queue_length, 4),
            "max_queue_length": self.max_queue_length,
            "avg_waiting_time": round(self.avg_waiting_time, 4),
            "avg_vehicle_count": round(self.avg_vehicle_count, 4),
            "avg_speed": round(self.avg_speed, 4),
        }


@dataclass
class SimulationResults:
    """Complete results of a simulation run."""

    # Network-level
    total_waiting_time: float = 0.0
    avg_waiting_time: float = 0.0
    avg_queue_length: float = 0.0
    max_queue_length: int = 0
    avg_travel_time: float = 0.0
    throughput: int = 0  # vehicles completed route
    vehicles_entered: int = 0
    vehicles_remaining: int = 0
    teleports: int = 0  # teleport events during simulation
    simulation_duration: float = 0.0  # actual wall-clock time
    sim_end_time: float = 0.0  # SUMO end time

    # Per-intersection
    intersection_metrics: dict[str, IntersectionMetrics] = field(default_factory=dict)

    # Step-level time series
    steps: list[StepSnapshot] = field(default_factory=list)

    # Config metadata
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "network_level": {
                "total_waiting_time": round(self.total_waiting_time, 4),
                "avg_waiting_time": round(self.avg_waiting_time, 4),
                "avg_queue_length": round(self.avg_queue_length, 4),
                "max_queue_length": self.max_queue_length,
                "avg_travel_time": round(self.avg_travel_time, 4),
                "throughput": self.throughput,
                "vehicles_entered": self.vehicles_entered,
                "vehicles_remaining": self.vehicles_remaining,
                "teleports": self.teleports,
                "simulation_duration": round(self.simulation_duration, 4),
                "sim_end_time": self.sim_end_time,
            },
            "per_intersection": {
                iid: m.to_dict() for iid, m in self.intersection_metrics.items()
            },
            "config": self.config,
        }


class MetricsCollector:
    """
    Collects metrics from a running SUMO simulation via TraCI.

    Usage:
        collector = MetricsCollector()
        collector.setup(intersection_ids, ...)

        # At each step:
        collector.collect_step(traaci, sim_time)

        # At end:
        results = collector.finalize(traaci)
    """

    def setup(
        self,
        intersection_ids: list[str],
        incoming_edges: dict[str, list[str]],
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the collector.

        Args:
            intersection_ids: List of TLS intersection IDs (e.g. ['N_1_1', ...]).
            incoming_edges: Dict mapping intersection_id to list of incoming edge IDs.
            config: Optional config metadata to store in results.
        """
        self.intersection_ids = intersection_ids
        self.incoming_edges = incoming_edges
        self.config = config or {}

        self.steps: list[StepSnapshot] = []
        self.intersection_metrics: dict[str, IntersectionMetrics] = {
            iid: IntersectionMetrics(intersection_id=iid) for iid in intersection_ids
        }

        # Accumulators for network-level metrics
        self._total_vehicles_entered = 0
        self._total_vehicles_left = 0
        self._sum_waiting_times = 0.0  # sum across ALL steps (each step = per-veh sum)
        self._sum_queue_lengths = 0.0
        self._max_queue_length = 0
        self._travel_times: list[float] = []
        self._start_time: float | None = None

    def start(self) -> None:
        """Mark simulation start."""
        import time
        self._start_time = time.time()

    def collect_step(self, traci, sim_time: float) -> StepSnapshot:
        """
        Collect metrics for one simulation step.

        Args:
            traci: The traci module.
            sim_time: Current SUMO simulation time in seconds.

        Returns:
            StepSnapshot with this step's metrics.
        """
        # --- Network-level per-vehicle metrics ---
        vehicle_ids = traci.vehicle.getIDList()
        num_vehicles = len(vehicle_ids)

        total_wait = 0.0
        for vid in vehicle_ids:
            total_wait += traci.vehicle.getWaitingTime(vid)

        avg_wait = total_wait / max(num_vehicles, 1)

        # --- Per-intersection metrics ---
        queue_lengths: dict[str, int] = {}
        vehicle_speeds: dict[str, float] = {}
        vehicles_per_intersection: dict[str, int] = {}

        for iid in self.intersection_ids:
            edges = self.incoming_edges.get(iid, [])
            ql = 0
            speeds: list[float] = []
            vcount = 0

            for edge_id in edges:
                try:
                    # Vehicles on this edge with speed < threshold = queued
                    edge_vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
                    vcount += len(edge_vehicles)
                    for vid in edge_vehicles:
                        spd = traci.vehicle.getSpeed(vid)
                        speeds.append(spd)
                        # Queue = vehicles waiting (speed < 0.1 m/s and have been stopped)
                        if spd < 0.1:
                            ql += 1
                except Exception:
                    pass

            queue_lengths[iid] = ql
            vehicle_speeds[iid] = float(np.mean(speeds)) if speeds else 0.0
            vehicles_per_intersection[iid] = vcount

            # Accumulate intersection metrics
            im = self.intersection_metrics[iid]
            im.total_queue_length += ql
            im.total_waiting_time += avg_wait
            im.total_vehicle_count += vcount
            im.step_count += 1
            if ql > im.max_queue_length:
                im.max_queue_length = ql
            if speeds:
                im.speed_samples.extend(speeds)

        # --- Accumulate network-level ---
        self._total_vehicles_entered += traci.simulation.getDepartedNumber()
        self._total_vehicles_left += traci.simulation.getArrivedNumber()
        self._sum_waiting_times += total_wait
        self._sum_queue_lengths += sum(queue_lengths.values())
        current_max = sum(queue_lengths.values())
        if current_max > self._max_queue_length:
            self._max_queue_length = current_max

        # --- Record step snapshot ---
        snapshot = StepSnapshot(
            time=sim_time,
            num_vehicles=num_vehicles,
            total_waiting_time=total_wait,
            avg_waiting_time=avg_wait,
            queue_lengths=queue_lengths,
            vehicle_speeds=vehicle_speeds,
            vehicles_per_intersection=vehicles_per_intersection,
        )
        self.steps.append(snapshot)
        return snapshot

    def finalize(
        self,
        traci,
        tripinfo_path: str | Path | None = None,
    ) -> SimulationResults:
        """
        Finalize and return the complete results.

        Args:
            traci: The traci module.
            tripinfo_path: Optional path to SUMO tripinfo.xml for accurate
                          per-vehicle waiting time and travel time.

        Returns:
            SimulationResults with all collected metrics.
        """
        import time as time_module

        total_steps = len(self.steps)

        # --- Compute queue metrics from step data (correct: per-step average) ---
        avg_queue = self._sum_queue_lengths / max(total_steps, 1)
        max_queue = self._max_queue_length

        # --- Compute waiting time and travel time from tripinfo if available ---
        # This is the authoritative source for per-vehicle metrics.
        total_waiting = 0.0
        avg_waiting = 0.0
        avg_travel = 0.0
        num_completed = self._total_vehicles_left

        if tripinfo_path is not None:
            from pathlib import Path as _P
            tp = _P(tripinfo_path)
            if tp.exists():
                waiting_times = compute_waiting_times_from_tripinfo(tp)
                travel_times = compute_travel_times_from_tripinfo(tp)
                if waiting_times:
                    total_waiting = float(np.sum(waiting_times))
                    avg_waiting = float(np.mean(waiting_times))
                if travel_times:
                    avg_travel = float(np.mean(travel_times))
                # Update throughput from actual completed trips
                num_completed = len(travel_times)

        # Fallback: if no tripinfo, use step-based estimate (less accurate)
        if tripinfo_path is None and total_steps > 0:
            total_waiting = self._sum_waiting_times
            avg_waiting = self._sum_waiting_times / max(num_completed, 1)
            if self._travel_times:
                avg_travel = float(np.mean(self._travel_times))

        wall_time = time_module.time() - self._start_time if self._start_time else 0.0

        results = SimulationResults(
            total_waiting_time=total_waiting,
            avg_waiting_time=avg_waiting,
            avg_queue_length=avg_queue,
            max_queue_length=max_queue,
            avg_travel_time=avg_travel,
            throughput=num_completed,
            vehicles_entered=self._total_vehicles_entered,
            vehicles_remaining=self._total_vehicles_entered - num_completed,
            simulation_duration=wall_time,
            sim_end_time=self.steps[-1].time if self.steps else 0.0,
            intersection_metrics=self.intersection_metrics,
            steps=self.steps,
            config=self.config,
        )
        return results

    def record_trip_time(self, travel_time: float) -> None:
        """Record a completed vehicle's travel time."""
        self._travel_times.append(travel_time)


def compute_travel_times_from_tripinfo(tripinfo_path: str | Path) -> list[float]:
    """
    Parse a SUMO tripinfo XML file to extract travel times.

    Args:
        tripinfo_path: Path to the tripinfo.xml output file.

    Returns:
        List of travel times in seconds.
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(str(tripinfo_path))
    except ET.ParseError:
        return []
    root = tree.getroot()
    travel_times = []
    for trip in root.findall(".//tripinfo"):
        duration = trip.get("duration")
        if duration is not None:
            travel_times.append(float(duration))
    return travel_times


def compute_waiting_times_from_tripinfo(tripinfo_path: str | Path) -> list[float]:
    """
    Parse a SUMO tripinfo XML file to extract per-vehicle waiting times.

    The SUMO 'waitingTime' attribute is the total time the vehicle spent
    stopped (speed below junction threshold) during its trip.

    Args:
        tripinfo_path: Path to the tripinfo.xml output file.

    Returns:
        List of waiting times in seconds (one per completed vehicle).
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(str(tripinfo_path))
    except ET.ParseError:
        return []
    root = tree.getroot()
    waiting_times = []
    for trip in root.findall(".//tripinfo"):
        wt = trip.get("waitingTime")
        if wt is not None:
            waiting_times.append(float(wt))
    return waiting_times


def save_results_json(results: SimulationResults, output_path: str | Path) -> Path:
    """Save results to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    return path


def save_results_csv(results: SimulationResults, output_dir: str | Path) -> dict[str, Path]:
    """
    Save results as CSV files.

    Saves:
      - network_summary.csv: single-row network-level metrics
      - per_intersection.csv: one row per intersection
      - step_metrics.csv: one row per simulation step

    Returns:
        Dict mapping name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # --- Network summary ---
    net_path = output_dir / "network_summary.csv"
    with open(net_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(results.to_dict()["network_level"].keys())
        writer.writerow(results.to_dict()["network_level"].values())
    paths["network_summary"] = net_path

    # --- Per intersection ---
    int_path = output_dir / "per_intersection.csv"
    with open(int_path, "w", newline="") as f:
        writer = csv.writer(f)
        if results.intersection_metrics:
            first = list(results.intersection_metrics.values())[0]
            writer.writerow(first.to_dict().keys())
            for m in results.intersection_metrics.values():
                writer.writerow(m.to_dict().values())
    paths["per_intersection"] = int_path

    # --- Step metrics ---
    step_path = output_dir / "step_metrics.csv"
    with open(step_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "num_vehicles", "total_waiting_time", "avg_waiting_time", "total_queue_length"])
        for s in results.steps:
            total_q = sum(s.queue_lengths.values())
            writer.writerow([
                s.time, s.num_vehicles, s.total_waiting_time,
                s.avg_waiting_time, total_q,
            ])
    paths["step_metrics"] = step_path

    return paths


def print_summary(results: SimulationResults) -> str:
    """
    Create a human-readable summary of simulation results.

    Returns:
        Formatted summary string.
    """
    d = results.to_dict()["network_level"]
    lines = [
        "",
        "=" * 60,
        "  FIXED-TIME BASELINE — Simulation Results",
        "=" * 60,
        "",
        "  Network-Level Metrics:",
        f"    Vehicles entered:          {d['vehicles_entered']}",
        f"    Vehicles completed:        {d['throughput']}",
        f"    Vehicles remaining:        {d['vehicles_remaining']}",
        f"    Vehicles teleported:       {d['teleports']}",
        f"    Total waiting time:        {d['total_waiting_time']:.1f} s",
        f"    Avg waiting time/vehicle:  {d['avg_waiting_time']:.2f} s",
        f"    Avg queue length:          {d['avg_queue_length']:.2f}",
        f"    Max queue length:          {d['max_queue_length']}",
        f"    Avg travel time:           {d['avg_travel_time']:.2f} s",
        f"    Simulation wall time:      {d['simulation_duration']:.1f} s",
        f"    Simulation end time:       {d['sim_end_time']:.0f} s",
        "",
        "  Per-Intersection Metrics:",
    ]

    if results.intersection_metrics:
        # Header
        lines.append(f"    {'Intersection':<15} {'Avg Queue':>10} {'Max Queue':>10} {'Avg Wait':>10} {'Avg Speed':>10}")
        lines.append(f"    {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for iid, m in results.intersection_metrics.items():
            lines.append(
                f"    {iid:<15} {m.avg_queue_length:>10.2f} {m.max_queue_length:>10} "
                f"{m.avg_waiting_time:>10.2f} {m.avg_speed:>10.2f}"
            )

    lines.append("")
    lines.append("=" * 60)
    summary = "\n".join(lines)
    print(summary)
    return summary
