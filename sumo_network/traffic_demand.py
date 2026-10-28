"""
SUMO Traffic Demand Generator
=============================

Generates realistic traffic demand as .rou.xml files for SUMO simulation.

Supports:
  - Configurable flow rates per edge
  - Multiple vehicle types
  - Time-varying demand patterns
  - Random seed for reproducibility
  - Demand level presets (low/medium/high)
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Demand level presets (vehicles per hour per edge)
DEMAND_PRESETS = {
    "low": 100,
    "medium": 300,
    "high": 600,
}


@dataclass
class DemandConfig:
    """Configuration for traffic demand generation."""

    duration: int = 3600  # seconds
    seed: int = 42
    flow_rate: int = 300  # vehicles per hour per edge (default)
    demand_level: str = "medium"
    demand_presets: dict[str, int] = field(default_factory=lambda: dict(DEMAND_PRESETS))
    # Teleportation mitigation: "random_free" inserts vehicles only at a free
    # random position (no overlap -> no collision teleports), and "max" depart
    # speed prevents rear-end catch-up collisions on short entry edges.
    depart_position: str = "random_free"
    depart_speed: str = "max"
    min_gap: float = 2.5
    corridor_weights: list[float] | None = None
    platoon_period: int | None = None
    platoon_on_time: int = 30
    demand_jitter: float = 0.1
    vehicle_types: dict[str, dict] = field(
        default_factory=lambda: {
            "passenger": {"probability": 1.0, "speed_factor": 1.0}
        }
    )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> DemandConfig:
        """Load demand configuration from a YAML file."""
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        dem = raw.get("demand", {})
        return cls(
            duration=dem.get("duration", 3600),
            seed=dem.get("seed", 42),
            flow_rate=dem.get("flow_rate", 300),
            demand_level=dem.get("demand_level", "medium"),
            demand_presets=dem.get("demand_presets", dict(DEMAND_PRESETS)),
            depart_position=dem.get("depart_position", "random_free"),
            depart_speed=dem.get("depart_speed", "max"),
            min_gap=dem.get("min_gap", 2.5),
            corridor_weights=dem.get("corridor_weights"),
            platoon_period=dem.get("platoon_period"),
            platoon_on_time=dem.get("platoon_on_time",30),
            demand_jitter=dem.get("demand_jitter",0.1),
            vehicle_types=dem.get("vehicle_types", {"passenger": {"probability": 1.0, "speed_factor": 1.0}}),
        )

    def get_flow_rate(self) -> int:
        """Get the effective flow rate based on demand level."""
        if self.demand_level in self.demand_presets:
            return self.demand_presets[self.demand_level]
        return self.flow_rate


class TrafficDemandGenerator:
    """
    Generates SUMO traffic demand files (.rou.xml).

    The generator creates vehicle flows on boundary edges of the grid network
    so that vehicles enter from the edges of the grid and travel through the
    signalized intersections.
    """

    def __init__(self, config: DemandConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.flow_rate = config.get_flow_rate()

    def identify_boundary_edges(
        self,
        edge_ids: list[str],
        grid_rows: int = 3,
        grid_cols: int = 3,
    ) -> dict[str, list[str]]:
        """
        Identify boundary edges (edges that start or end at the grid boundary).

        Args:
            edge_ids: List of all edge IDs in the network.
            grid_rows: Number of intersection rows (determines node grid rows+1).
            grid_cols: Number of intersection columns (determines node grid cols+1).

        Returns:
            Dict with keys 'entry' (edges entering the grid) and 'exit' (edges leaving).
        """
        entry_edges = []
        exit_edges = []

        # Node grid dimensions: (grid_rows+1) x (grid_cols+1)
        max_row = grid_rows
        max_col = grid_cols

        for edge_id in edge_ids:
            parts = edge_id.split("_to_")
            if len(parts) != 2:
                continue

            # Edge format: E_fromRow_fromCol_to_toRow_toCol
            from_part = parts[0]  # E_fromRow_fromCol
            to_part = parts[1]    # toRow_toCol

            from_parts = from_part.split("_")
            to_parts = to_part.split("_")

            if len(from_parts) != 3 or len(to_parts) != 2:
                continue

            from_row = int(from_parts[1])
            from_col = int(from_parts[2])
            to_row = int(to_parts[0])
            to_col = int(to_parts[1])

            from_boundary = (
                from_row == 0 or from_row == max_row
                or from_col == 0 or from_col == max_col
            )
            to_boundary = (
                to_row == 0 or to_row == max_row
                or to_col == 0 or to_col == max_col
            )

            # Entry edges: from boundary inward (from is boundary, to is interior)
            if from_boundary and not to_boundary:
                entry_edges.append(edge_id)
            # Exit edges: from interior to boundary
            if not from_boundary and to_boundary:
                exit_edges.append(edge_id)

        return {"entry": entry_edges, "exit": exit_edges}

    def _extract_edge_coords(self, edge_id: str) -> tuple[int, int, int, int] | None:
        """
        Extract coordinates from an edge ID.

        Returns:
            (from_row, from_col, to_row, to_col) or None if parsing fails.
        """
        parts = edge_id.split("_to_")
        if len(parts) != 2:
            return None

        from_parts = parts[0].split("_")  # ['E', 'fromRow', 'fromCol']
        to_parts = parts[1].split("_")    # ['toRow', 'toCol']

        if len(from_parts) != 3 or len(to_parts) != 2:
            return None

        return (
            int(from_parts[1]),
            int(from_parts[2]),
            int(to_parts[0]),
            int(to_parts[1]),
        )

    def generate_rou_xml(
        self,
        edge_ids: list[str],
        output_path: str | Path,
        demand_scale: float = 1.0,
        grid_rows: int = 3,
        grid_cols: int = 3,
    ) -> Path:
        """
        Generate a .rou.xml file with vehicle flows.

        Args:
            edge_ids: List of all edge IDs in the network.
            output_path: Path to write the .rou.xml file.
            demand_scale: Multiplier on the flow rate (1.0 = normal).
            grid_rows: Number of intersection rows.
            grid_cols: Number of intersection columns.

        Returns:
            Path to the generated .rou.xml file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rng = random.Random(self.config.seed)
        boundaries = self.identify_boundary_edges(edge_ids, grid_rows, grid_cols)

        root = ET.Element("routes")

        # --- Vehicle type definition ---
        vtype = ET.SubElement(
            root,
            "vType",
            id="passenger",
            accel="2.6",
            decel="4.5",
            sigma="0.5",
            length="5.0",
            minGap=str(self.config.min_gap),
            maxSpeed="50",
            speedFactor="1.0",
            speedDev="0.0",
        )

        # --- Flows on entry edges ---
        flow_id_counter = 0
        effective_flow = int(self.flow_rate * demand_scale)

        # Build flows, then sort by begin_time for SUMO compatibility
        flow_data = []
        for edge_id in boundaries.get("entry", []):
            coords = self._extract_edge_coords(edge_id)
            if coords is None:
                continue
            begin_time = rng.randint(0, min(300, self.config.duration // 10))
            flow_data.append((begin_time, edge_id))
        flow_data.sort(key=lambda x: x[0])

        for fid, (begin_time, edge_id) in enumerate(flow_data):
            attrs = {
                "id": f"flow_{fid}",
                "type": "passenger",
                "from": edge_id,
                "vehsPerHour": str(effective_flow),
                "begin": str(begin_time),
                "end": str(self.config.duration),
                "departPos": self.config.depart_position,
                "departSpeed": self.config.depart_speed,
            }
            if self.config.platoon_period is None:
                ET.SubElement(root, "flow", **attrs)
            else:
                period = int(self.config.platoon_period)
                on_time = int(self.config.platoon_on_time)
                if period <= 0 or not 0 < on_time <= period:
                    raise ValueError('platoon_on_time must be in (0, platoon_period]')
                # Opposing axes release at different offsets; seed-dependent
                # jitter varies the platoon phase without leaking test seeds.
                offset = (fid % 4)*period/4 + self.rng.uniform(0,5)
                for burst in range(-1, int(self.config.duration/period)+1):
                    begin = max(0.0,burst*period+offset)
                    end = min(float(self.config.duration),burst*period+offset+on_time)
                    if end <= begin:
                        continue
                    burst_attrs = dict(attrs,id=f'flow_{fid}_burst{burst+1}',
                        begin=f'{begin:.3f}',end=f'{end:.3f}',
                        vehsPerHour=f'{float(attrs["vehsPerHour"])*period/on_time:.6f}')
                    ET.SubElement(root,'flow',**burst_attrs)

        # SUMO requires departure specifications in nondecreasing time order.
        flows=list(root.findall('flow'))
        for flow in flows:
            root.remove(flow)
        for flow in sorted(flows,key=lambda f:float(f.get('begin','0'))):
            root.append(flow)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, xml_declaration=True)
        return path

    def generate_rou_xml_with_od(
        self,
        entry_edges: list[str],
        exit_edges: list[str],
        output_path: str | Path,
        demand_scale: float = 1.0,
    ) -> Path:
        """
        Generate a .rou.xml file with origin-destination based demand.

        Creates vehicles with specific routes from entry edges to exit edges.

        Args:
            entry_edges: List of entry edge IDs.
            exit_edges: List of exit edge IDs.
            output_path: Path to write the .rou.xml file.
            demand_scale: Multiplier on the flow rate.

        Returns:
            Path to the generated .rou.xml file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rng = random.Random(self.config.seed)
        root = ET.Element("routes")

        # Vehicle type
        ET.SubElement(
            root,
            "vType",
            id="passenger",
            accel="2.6",
            decel="4.5",
            sigma="0.5",
            length="5.0",
            minGap=str(self.config.min_gap),
            maxSpeed="50",
        )

        # For each OD pair, create a flow
        flow_id = 0
        effective_flow = int(self.flow_rate * demand_scale)

        for entry in sorted(entry_edges):
            for exit_ in sorted(exit_edges):
                if entry == exit_:
                    continue

                begin_time = rng.randint(0, min(300, self.config.duration // 10))

                attrs = {
                    "id": f"flow_{flow_id}",
                    "type": "passenger",
                    "from": entry,
                    "to": exit_,
                    "vehsPerHour": str(max(1, effective_flow // len(exit_edges))),
                    "begin": str(begin_time),
                    "end": str(self.config.duration),
                    "departPos": self.config.depart_position,
                    "departSpeed": self.config.depart_speed,
                }
                ET.SubElement(root, "flow", **attrs)
                flow_id += 1

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, xml_declaration=True)
        return path

    def generate_with_grid_builder(
        self,
        grid_builder,
        output_path: str | Path,
        demand_scale: float = 1.0,
    ) -> Path:
        """
        Generate demand using a GridBuilder instance.

        Uses boundary-only entry flows (original Phase 2 pattern).

        Args:
            grid_builder: A GridBuilder instance with the network definition.
            output_path: Path to write the .rou.xml file.
            demand_scale: Multiplier on the flow rate.

        Returns:
            Path to the generated .rou.xml file.
        """
        edge_ids = [edge.id for edge in grid_builder.edges]
        return self.generate_rou_xml(
            edge_ids, output_path, demand_scale,
            grid_rows=grid_builder.config.rows,
            grid_cols=grid_builder.config.cols,
        )

    def generate_network_wide_with_grid_builder(
        self,
        grid_builder,
        output_path: str | Path,
        demand_scale: float = 1.0,
    ) -> Path:
        """
        Generate demand that covers ALL intersections in the grid.

        Uses explicit origin-destination flows so that SUMO routes vehicles
        THROUGH the grid (not just to the nearest exit).  Each OD pair
        consists of a boundary entry edge and a boundary exit edge on the
        OPPOSITE side of the grid, forcing cross-grid travel through
        interior intersections.

        For a 3×3 grid with bidi edges there are 6 entry edges (top + left
        boundaries) and 6 exit edges (bottom + right boundaries via bidi),
        yielding 36 opposite-side OD pairs.  ``flow_rate`` is the TOTAL
        network demand in veh/hr, divided evenly across the OD pairs, and
        every flow spans the full simulation (begin=0, end=duration) so the
        inserted vehicle count scales linearly with the requested total.
        The expected total vehicle count is printed when generating.

        Args:
            grid_builder: A GridBuilder instance with the network definition.
            output_path: Path to write the .rou.xml file.
            demand_scale: Multiplier on the flow rate.

        Returns:
            Path to the generated .rou.xml file.
        """
        grid_rows = grid_builder.config.rows
        grid_cols = grid_builder.config.cols
        edge_set = {e.id for e in grid_builder.edges}

        def _eid(fr, fc, tr, tc):
            return f"E_{fr}_{fc}_to_{tr}_{tc}"

        # Build OD pairs that cross the grid diagonally, ensuring each route
        # passes through multiple interior intersections.
        #
        # Strategy: entry from top/left boundary, exit to opposite bottom/right.
        # Only create a subset of OD pairs to keep total volume manageable.
        entry_edges = []
        for c in range(1, grid_cols + 1):
            e = _eid(0, c, 1, c)
            if e in edge_set:
                entry_edges.append((0, c, e))  # (from_row, from_col, edge_id)
        for r in range(1, grid_rows + 1):
            e = _eid(r, 0, r, 1)
            if e in edge_set:
                entry_edges.append((r, 0, e))

        exit_edges = []
        for c in range(1, grid_cols + 1):
            e = _eid(grid_rows, c, grid_rows - 1, c)
            if e in edge_set:
                exit_edges.append((grid_rows, c, e))
        for r in range(1, grid_rows + 1):
            e = _eid(r, grid_cols, r, grid_cols - 1)
            if e in edge_set:
                exit_edges.append((r, grid_cols, e))

        # Create OD pairs: entry from top → exit at bottom/right (long routes)
        # entry from left → exit at right/bottom (long routes)
        od_pairs = []
        for er, ec, entry_e in entry_edges:
            for xr, xc, exit_e in exit_edges:
                # Only pair entries and exits that are on OPPOSITE sides
                # This ensures routes traverse the full grid
                same_top_bottom = (er == 0 and xr == grid_rows)
                same_left_right = (ec == 0 and xc == grid_cols)
                cross_top_right = (er == 0 and xc == grid_cols)
                cross_left_bottom = (ec == 0 and xr == grid_rows)
                if same_top_bottom or same_left_right or cross_top_right or cross_left_bottom:
                    od_pairs.append((entry_e, exit_e))

        # If no cross-pairs found, fall back to all pairs
        if not od_pairs:
            od_pairs = [(ie, xe) for _, _, ie in entry_edges for _, _, xe in exit_edges]

        if grid_builder.config.boundary_ring:
            # Straight bidirectional corridors, no implicit turning paths or
            # boundary detours. Every route crosses a row/column of signals.
            od_pairs = []
            for c in range(1, grid_cols + 1):
                od_pairs.extend([
                    (_eid(0, c, 1, c), _eid(grid_rows, c, grid_rows+1, c)),
                    (_eid(grid_rows+1, c, grid_rows, c), _eid(1, c, 0, c))])
            for r in range(1, grid_rows + 1):
                od_pairs.extend([
                    (_eid(r, 0, r, 1), _eid(r, grid_cols, r, grid_cols+1)),
                    (_eid(r, grid_cols+1, r, grid_cols), _eid(r, 1, r, 0))])

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("routes")

        # Vehicle type
        ET.SubElement(
            root, "vType",
            id="passenger",
            accel="2.6",
            decel="4.5",
            sigma="0.5",
            length="5.0",
            minGap=str(self.config.min_gap),
            maxSpeed="50",
        )

        # flow_rate is TOTAL network veh/hr, divided evenly across all OD
        # pairs.  Float split (SUMO accepts fractional vehsPerHour): an
        # integer floor would collapse small totals to ~1 vehicle per pair.
        effective_flow = float(self.flow_rate) * demand_scale
        per_od_flow = effective_flow / max(1, len(od_pairs))

        # Seed-dependent per-OD jitter (±10 %): the generation itself is
        # otherwise fully deterministic (fixed begin/end times), so without
        # this every demand seed would produce an identical file. The jitter
        # keeps the mean demand identical while varying it per seed.
        if not 0 <= self.config.demand_jitter <= 1:
            raise ValueError('demand_jitter must be in [0, 1]')
        jitters = [1.0 + self.rng.uniform(-self.config.demand_jitter,self.config.demand_jitter)
                   for _ in range(len(od_pairs))]
        weights = self.config.corridor_weights
        if weights is not None:
            if (len(weights) != len(od_pairs) or any(w < 0 for w in weights)
                    or sum(weights) <= 0):
                raise ValueError('corridor_weights must match OD pairs, be nonnegative, and sum positive')
            total_weight = sum(w*j for w,j in zip(weights,jitters))
            flow_rates = [effective_flow*w*j/total_weight for w,j in zip(weights,jitters)]
        else:
            flow_rates = [per_od_flow*j for j in jitters]

        expected_total = effective_flow * (self.config.duration / 3600.0)
        print(
            f"  Network-wide demand: {effective_flow:.1f} veh/hr total across "
            f"{len(od_pairs)} OD pairs ({per_od_flow:.2f} veh/hr each) -> "
            f"expected {expected_total:.0f} vehicles over {self.config.duration}s"
        )

        # All flows span the full simulation (begin=0, end=duration) so the
        # inserted vehicle count scales linearly with the requested total.
        # (Randomized begin times previously forced >= 1 vehicle per flow,
        # breaking count-vs-demand scaling at low flow rates.)
        for fid, (entry, exit_) in enumerate(od_pairs):
            attrs = {
                "id": f"flow_{fid}",
                "type": "passenger",
                "from": entry,
                "to": exit_,
                "vehsPerHour": f"{flow_rates[fid]:.6f}",
                "begin": "0",
                "end": str(self.config.duration),
                "departPos": self.config.depart_position,
                "departSpeed": self.config.depart_speed,
            }
            if self.config.platoon_period is None:
                ET.SubElement(root, "flow", **attrs)
            else:
                period = int(self.config.platoon_period)
                on_time = int(self.config.platoon_on_time)
                if period <= 0 or not 0 < on_time <= period:
                    raise ValueError('platoon_on_time must be in (0, platoon_period]')
                offset = (fid % 4)*period/4 + self.rng.uniform(0,5)
                for burst in range(-1,int(self.config.duration/period)+1):
                    begin = max(0.0,burst*period+offset)
                    end = min(float(self.config.duration),burst*period+offset+on_time)
                    if end>begin:
                        ET.SubElement(root,'flow',**dict(attrs,id=f'flow_{fid}_burst{burst+1}',
                            begin=f'{begin:.3f}',end=f'{end:.3f}',
                            vehsPerHour=f'{flow_rates[fid]*period/on_time:.6f}'))

        flows=list(root.findall('flow'))
        for flow in flows:
            root.remove(flow)
        for flow in sorted(flows,key=lambda f:float(f.get('begin','0'))):
            root.append(flow)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, xml_declaration=True)
        return path

    def summary(self) -> str:
        """Return a human-readable summary of the demand configuration."""
        lines = [
            f"Traffic Demand Configuration:",
            f"  Duration: {self.config.duration} s ({self.config.duration / 60:.1f} min)",
            f"  Demand level: {self.config.demand_level}",
            f"  Flow rate: {self.flow_rate} veh/hr/edge",
            f"  Seed: {self.config.seed}",
            f"  Depart position: {self.config.depart_position}",
            f"  Depart speed: {self.config.depart_speed}",
            f"  Min gap: {self.config.min_gap} m",
            f"  Vehicle types: {list(self.config.vehicle_types.keys())}",
        ]
        return "\n".join(lines)
