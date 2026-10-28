"""
Fixed-Time Signal Plans for SUMO Traffic Lights
================================================

Generates traffic light signal programs for the grid network intersections.
Each intersection has a simple 2-phase signal:
  - Phase 0: North-South green
  - Phase 1: East-West green
With yellow and all-red transitions between phases.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Phase:
    """A single signal phase."""

    name: str
    duration: float  # seconds
    state: str  # signal state string (G=green, y=yellow, r=red)
    min_duration: float = 0.0
    max_duration: float = 0.0


@dataclass
class SignalPlan:
    """A complete signal plan for one or more intersections."""

    name: str
    cycle_length: float
    phases: list[Phase]
    offset: float = 0.0  # offset from cycle start (for coordination)

    @property
    def num_phases(self) -> int:
        return len(self.phases)

    def get_state_at_time(self, time: float) -> str:
        """Get the signal state at a given time within the cycle."""
        t = time % self.cycle_length
        cumulative = 0.0
        for phase in self.phases:
            cumulative += phase.duration
            if t < cumulative:
                return phase.state
        return self.phases[-1].state


@dataclass
class SignalConfig:
    """Configuration for signal plans."""

    cycle_length: float = 90.0  # seconds
    yellow_time: float = 3.0
    all_red_time: float = 1.0
    phases: list[dict] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> SignalConfig:
        """Load signal configuration from a YAML file."""
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        sig = raw.get("signals", {})
        phases_raw = sig.get("phases", [])
        phases = [
            {"name": p.get("name", f"phase_{i}"), "duration": p.get("duration", 40), "state": p.get("state", "GGrrrrrrrr")}
            for i, p in enumerate(phases_raw)
        ]
        return cls(
            cycle_length=sig.get("cycle_length", 90.0),
            yellow_time=sig.get("yellow_time", 3.0),
            all_red_time=sig.get("all_red_time", 1.0),
            phases=phases,
        )


class SignalPlanGenerator:
    """
    Generates fixed-time signal plans for grid intersections.

    For a simple 2-phase NS/EW intersection with N connections:
      - The state string has one character per connection link
      - Phase 0 (NS green):  G for NS links, r for EW links
      - Phase 1 (EW green):  r for NS links, G for EW links
    """

    def __init__(self, config: SignalConfig | None = None):
        if config is None:
            config = SignalConfig()
        self.config = config

    def create_default_2phase_plan(
        self,
        cycle_length: float | None = None,
        yellow_time: float | None = None,
        all_red_time: float | None = None,
    ) -> SignalPlan:
        """
        Create a default 2-phase signal plan (NS then EW).

        The state string length depends on the number of connection links.
        For our grid: each intersection has 4 incoming edges, each with 1 lane,
        connecting to 4 outgoing edges. With tl_link indices:
          - Links 0, 1, 2, 3 = NS and EW connections
        Actually, in our connection XML we use tl_link 0 for NS and tl_link 1 for EW.
        SUMO creates 4 internal links per intersection (one per incoming lane).
        The state string needs 4 characters for 4 incoming lanes.

        For a 2-lane NS/EW intersection:
          - NS links: positions 0, 1
          - EW links: positions 2, 3
        """
        cl = cycle_length or self.config.cycle_length
        yt = yellow_time or self.config.yellow_time
        art = all_red_time or self.config.all_red_time

        # Green time per phase = (cycle - 2*yellow - 2*all_red) / 2
        green_time = (cl - 2 * yt - 2 * art) / 2
        if green_time < 5:
            raise ValueError(
                f"Cycle length {cl}s is too short for {yt}s yellow and {art}s all-red. "
                f"Minimum cycle length: {2 * yt + 2 * art + 10}s"
            )

        # 4 links: NS-Straight(0), NS-Left(1), EW-Straight(2), EW-Left(3)
        # For simple straight-only: links 0,1 = NS, links 2,3 = EW
        phases = [
            Phase(name="NS_green", duration=green_time, state="GGrr"),
            Phase(name="NS_yellow", duration=yt, state="yyrr"),
            Phase(name="all_red_1", duration=art, state="rrrr"),
            Phase(name="EW_green", duration=green_time, state="rrGG"),
            Phase(name="EW_yellow", duration=yt, state="rryy"),
            Phase(name="all_red_2", duration=art, state="rrrr"),
        ]

        return SignalPlan(
            name="default_2phase",
            cycle_length=cl,
            phases=phases,
        )

    def create_plan_from_config(self) -> SignalPlan:
        """Create a signal plan from the loaded YAML config phases."""
        phases = []
        for p in self.config.phases:
            phases.append(
                Phase(
                    name=p.get("name", "unknown"),
                    duration=float(p.get("duration", 40)),
                    state=p.get("state", "GGrr"),
                )
            )
        return SignalPlan(
            name="config_based",
            cycle_length=self.config.cycle_length,
            phases=phases,
        )

    def generate_tls_add_xml(
        self,
        intersection_ids: list[str],
        output_path: str | Path,
        plan: SignalPlan | None = None,
        offsets: dict[str, float] | None = None,
        num_links: dict[str, int] | None = None,
    ) -> Path:
        """
        Generate a .add.xml file with traffic light definitions.

        Args:
            intersection_ids: List of intersection IDs (e.g., ['N_1_1', 'N_1_2', ...])
            output_path: Path to write the .add.xml file.
            plan: Signal plan to use. If None, creates default 2-phase plan.
            offsets: Per-intersection offset dict for coordination. If None, all offset=0.
            num_links: Dict mapping intersection ID to number of controlled links.
                      If None, uses a default of 2 links per intersection.

        Returns:
            Path to the generated .add.xml file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if plan is None:
            plan = self.create_default_2phase_plan()

        if offsets is None:
            offsets = {}
        if num_links is None:
            num_links = {}

        root = ET.Element("additional")

        for int_id in intersection_ids:
            # Skip intersections with 0 links (no valid connections)
            if num_links and num_links.get(int_id, 0) == 0:
                continue

            offset = offsets.get(int_id, 0.0)
            tls = ET.SubElement(
                root,
                "tlLogic",
                id=int_id,
                type="static",
                programID="0",
                offset=str(int(offset)),
            )

            for phase in plan.phases:
                state = phase.state
                # If num_links provided, adjust state string to match
                if num_links and int_id in num_links:
                    n = num_links[int_id]
                    if len(state) != n:
                        if len(state) < n:
                            state = (state * (n // len(state) + 1))[:n]
                        else:
                            state = state[:n]
                ET.SubElement(
                    tls,
                    "phase",
                    duration=str(int(phase.duration)),
                    state=state,
                )

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, xml_declaration=True)
        return path

    @staticmethod
    def compute_offsets_linear(
        intersection_ids: list[str],
        grid_rows: int,
        grid_cols: int,
        block_length: float,
        wave_speed: float = 8.33,
    ) -> dict[str, float]:
        """
        Compute linear offset offsets for green wave coordination.

        Uses a simple linear progression from corner (1,1) outward.

        Args:
            intersection_ids: List of intersection IDs.
            grid_rows: Number of grid rows.
            grid_cols: Number of grid cols.
            block_length: Block length in meters.
            wave_speed: Wave speed in m/s (default: speed limit).

        Returns:
            Dict mapping intersection_id to offset in seconds.
        """
        offsets = {}
        for int_id in intersection_ids:
            parts = int_id.split("_")
            row = int(parts[1])
            col = int(parts[2])
            # Linear offset from top-left corner
            dist = (row - 1) * block_length + (col - 1) * block_length
            offsets[int_id] = dist / wave_speed
        return offsets


def create_signal_plan_summary(plan: SignalPlan) -> str:
    """Create a human-readable summary of a signal plan."""
    lines = [
        f"Signal Plan: {plan.name}",
        f"  Cycle length: {plan.cycle_length}s",
        f"  Number of phases: {plan.num_phases}",
        f"  Phases:",
    ]
    for i, phase in enumerate(plan.phases):
        lines.append(f"    {i}: {phase.name} ({phase.duration}s) — {phase.state}")
    return "\n".join(lines)
