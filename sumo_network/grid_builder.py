"""
SUMO Grid Network Builder
=========================

Generates a configurable grid network for SUMO traffic simulation.

For a rows×cols grid, the network has:
  - (rows+1) × (cols+1) nodes (junctions)
  - rows × (cols+1) horizontal edges (×2 for bidi)
  - (rows+1) × cols vertical edges (×2 for bidi)
  - rows × cols signalized intersections

Output files:
  - {name}.nod.xml  — node definitions
  - {name}.edg.xml  — edge definitions
  - {name}.add.xml  — additional objects (TLS definitions)
  - {name}.con.xml  — connection definitions (implicit via netconvert)
  - {name}.net.xml  — compiled network (via netconvert)
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GridConfig:
    """Configuration for the grid network."""

    rows: int = 3
    cols: int = 3
    block_length: float = 300.0  # meters
    lanes_per_direction: int = 1
    speed_limit: float = 8.33  # m/s (~30 km/h)
    bidi: bool = True
    network_name: str = "grid_3x3"
    boundary_ring: bool = False

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> GridConfig:
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        net_cfg = raw.get("network", {}).get("grid", {})
        return cls(
            rows=net_cfg.get("rows", 3),
            cols=net_cfg.get("cols", 3),
            block_length=raw.get("network", {}).get("block_length", 300.0),
            lanes_per_direction=raw.get("network", {}).get("lanes_per_direction", 1),
            speed_limit=raw.get("network", {}).get("speed_limit", 8.33),
            bidi=raw.get("network", {}).get("bidi", True),
            network_name=raw.get("network", {}).get("name", "grid_3x3"),
            boundary_ring=raw.get("network", {}).get("boundary_ring", False),
        )


@dataclass
class Node:
    """Represents a SUMO node (junction)."""

    id: str
    x: float
    y: float
    node_type: str = "priority"
    # Intersection IDs are the internal junction nodes (for signalized intersections)
    is_intersection: bool = False
    # For signalized intersections: the actual junction ID used by TLS
    junction_id: str | None = None


@dataclass
class Edge:
    """Represents a SUMO edge (road segment)."""

    id: str
    from_node: str
    to_node: str
    num_lanes: int
    speed: float
    priority: int = 1


@dataclass
class Connection:
    """Represents a connection between edges at an intersection."""

    from_edge: str
    to_edge: str
    from_lane: int = 0
    to_lane: int = 0
    tl: str | None = None
    tl_link: int | None = None


class GridBuilder:
    """
    Builds a grid network for SUMO traffic simulation.

    Grid layout for rows=3, cols=3 → 3×3 = 9 signalized intersections:

        N0---N1---N2---N3       (4 nodes wide)
        |    |    |    |
        N4---I0---I1---N7       (intersections at I_1_1, I_1_2, I_1_3)
        |    |    |    |
        N8---I2---I3---N11      (intersections at I_2_1, I_2_2, I_2_3)
        |    |    |    |
        N12--I4---I5---N15      (intersections at I_3_1, I_3_2, I_3_3)

    In our coordinate system:
    - Nodes are at integer grid positions (i, j) for i in 0..rows, j in 0..cols
    - Intersections are at (i, j) for i in 1..rows, j in 1..cols
      These are the interior nodes that have roads from all 4 directions.
    - rows and cols specify the NUMBER OF INTERSECTIONS per dimension.
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.connections: list[Connection] = []
        self.intersection_nodes: list[str] = []  # IDs of signalized intersections
        self._build()

    def _node_id(self, row: int, col: int) -> str:
        """Generate a node ID from grid coordinates."""
        return f"N_{row}_{col}"

    def _edge_id(self, from_row: int, from_col: int, to_row: int, to_col: int) -> str:
        """Generate an edge ID from the two endpoint grid coordinates."""
        return f"E_{from_row}_{from_col}_to_{to_row}_{to_col}"

    def _intersection_id(self, row: int, col: int) -> str:
        """Generate an intersection (TLS) ID from grid coordinates."""
        return f"I_{row}_{col}"

    def _build(self) -> None:
        """Build all nodes, edges, connections, and identify intersections."""
        rows = self.config.rows
        cols = self.config.cols
        bl = self.config.block_length
        lanes = self.config.lanes_per_direction
        speed = self.config.speed_limit
        # Optional complete boundary gives every signal four approaches.
        node_rows = rows + int(self.config.boundary_ring)
        node_cols = cols + int(self.config.boundary_ring)

        # --- Nodes ---
        for i in range(node_rows + 1):
            for j in range(node_cols + 1):
                node_id = self._node_id(i, j)
                x = j * bl
                y = i * bl
                # Determine if this node is an intersection (has roads from all 4 dirs)
                is_interior = 0 < i <= rows and 0 < j <= cols
                node_type = "traffic_light" if is_interior else "priority"
                junction_id = self._intersection_id(i, j) if is_interior else None

                self.nodes.append(
                    Node(
                        id=node_id,
                        x=x,
                        y=y,
                        node_type=node_type,
                        is_intersection=is_interior,
                        junction_id=junction_id,
                    )
                )
                if is_interior:
                    self.intersection_nodes.append(node_id)

        # --- Horizontal edges (left-to-right and right-to-left) ---
        for i in range(node_rows + 1):
            for j in range(node_cols):
                # Forward edge
                fwd_id = self._edge_id(i, j, i, j + 1)
                self.edges.append(
                    Edge(id=fwd_id, from_node=self._node_id(i, j), to_node=self._node_id(i, j + 1), num_lanes=lanes, speed=speed)
                )
                # Backward (bidi) edge
                if self.config.bidi:
                    bwd_id = self._edge_id(i, j + 1, i, j)
                    self.edges.append(
                        Edge(id=bwd_id, from_node=self._node_id(i, j + 1), to_node=self._node_id(i, j), num_lanes=lanes, speed=speed)
                    )

        # --- Vertical edges (top-to-bottom and bottom-to-top) ---
        for i in range(node_rows):
            for j in range(node_cols + 1):
                # Forward edge (downward)
                fwd_id = self._edge_id(i, j, i + 1, j)
                self.edges.append(
                    Edge(id=fwd_id, from_node=self._node_id(i, j), to_node=self._node_id(i + 1, j), num_lanes=lanes, speed=speed)
                )
                # Backward (bidi) edge (upward)
                if self.config.bidi:
                    bwd_id = self._edge_id(i + 1, j, i, j)
                    self.edges.append(
                        Edge(id=bwd_id, from_node=self._node_id(i + 1, j), to_node=self._node_id(i, j), num_lanes=lanes, speed=speed)
                    )

        # --- Connections at signalized intersections ---
        self._build_connections()

    def _build_connections(self) -> None:
        """
        Build connections at each signalized intersection.

        At each intersection (i, j) where 1 <= i <= rows and 1 <= j <= cols,
        we create straight-through connections for NS and EW traffic.
        Only connections where BOTH the source and destination edges exist
        in the network are created (boundary intersections may not have
        edges in all directions).
        """
        rows = self.config.rows
        cols = self.config.cols

        # Build a set of valid edge IDs for quick lookup
        valid_edges = {edge.id for edge in self.edges}

        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                int_id = self._node_id(i, j)
                link = 0

                # NS straight connections
                for from_edge, to_edge in [
                    (self._edge_id(i - 1, j, i, j), self._edge_id(i, j, i + 1, j)),
                    (self._edge_id(i + 1, j, i, j), self._edge_id(i, j, i - 1, j)),
                ]:
                    if from_edge in valid_edges and to_edge in valid_edges:
                        for lane in range(self.config.lanes_per_direction):
                            self.connections.append(
                                Connection(
                                    from_edge=from_edge,
                                    to_edge=to_edge,
                                    from_lane=lane,
                                    to_lane=lane,
                                    tl=int_id,
                                    tl_link=link,
                                )
                            )
                link += 1

                # EW straight connections
                for from_edge, to_edge in [
                    (self._edge_id(i, j - 1, i, j), self._edge_id(i, j, i, j + 1)),
                    (self._edge_id(i, j + 1, i, j), self._edge_id(i, j, i, j - 1)),
                ]:
                    if from_edge in valid_edges and to_edge in valid_edges:
                        for lane in range(self.config.lanes_per_direction):
                            self.connections.append(
                                Connection(
                                    from_edge=from_edge,
                                    to_edge=to_edge,
                                    from_lane=lane,
                                    to_lane=lane,
                                    tl=int_id,
                                    tl_link=link,
                                )
                            )
                link += 1

    # --- XML Generation ---

    def generate_nod_xml(self, output_path: str | Path) -> Path:
        """Generate the .nod.xml file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("nodes")
        for node in self.nodes:
            attribs = {
                "id": node.id,
                "x": f"{node.x:.2f}",
                "y": f"{node.y:.2f}",
                "type": node.node_type,
            }
            if node.junction_id:
                attribs["name"] = node.junction_id
            ET.SubElement(root, "node", attribs)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, xml_declaration=True)
        return path

    def generate_edg_xml(self, output_path: str | Path) -> Path:
        """Generate the .edg.xml file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("edges")
        for edge in self.edges:
            attribs = {
                "id": edge.id,
                "from": edge.from_node,
                "to": edge.to_node,
                "numLanes": str(edge.num_lanes),
                "speed": f"{edge.speed:.2f}",
                "priority": str(edge.priority),
            }
            ET.SubElement(root, "edge", attribs)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, xml_declaration=True)
        return path

    def generate_con_xml(self, output_path: str | Path) -> Path:
        """Generate the .con.xml file with connections."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("connections")
        for conn in self.connections:
            attribs = {
                "from": conn.from_edge,
                "to": conn.to_edge,
                "fromLane": str(conn.from_lane),
                "toLane": str(conn.to_lane),
            }
            if conn.tl:
                attribs["tl"] = conn.tl
                attribs["tlLink"] = str(conn.tl_link) if conn.tl_link is not None else "0"
            ET.SubElement(root, "connection", attribs)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, xml_declaration=True)
        return path

    def generate_add_xml(self, output_path: str | Path, tls_phases: list[dict] | None = None) -> Path:
        """
        Generate the .add.xml file with traffic light definitions.

        Args:
            output_path: Path to write the .add.xml file.
            tls_phases: List of phase dicts with keys 'duration', 'state'.
                       If None, generates a default 2-phase program.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if tls_phases is None:
            # Default 2-phase: NS green, then EW green
            # State format: one char per connection link
            # At each intersection we have 2 links: 0=NS, 1=EW
            tls_phases = [
                {"duration": "40", "state": "GG"},
                {"duration": "3", "state": "yy"},
                {"duration": "40", "state": "rrGG"},
                {"duration": "3", "state": "rryy"},
            ]

        root = ET.Element("additional")
        # Use node IDs as TLS IDs (they match the junction IDs netconvert creates)
        for node_id in self.intersection_nodes:
            tls = ET.SubElement(root, "tlLogic", id=node_id, type="static", programID="0", offset="0")
            for phase in tls_phases:
                ET.SubElement(tls, "phase", duration=phase["duration"], state=phase["state"])

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, xml_declaration=True)
        return path

    def _intersection_id_from_node(self, node_id: str) -> str:
        """Extract the intersection ID from a node ID like N_1_2 -> I_1_2."""
        parts = node_id.split("_")
        return f"I_{parts[1]}_{parts[2]}"

    def run_netconvert(self, output_dir: str | Path, network_name: str | None = None) -> Path:
        """
        Run netconvert to produce the final .net.xml file.

        Args:
            output_dir: Directory containing .nod.xml, .edg.xml, .con.xml, .add.xml
            network_name: Name prefix for the output .net.xml file

        Returns:
            Path to the generated .net.xml file.
        """
        output_dir = Path(output_dir)
        name = network_name or self.config.network_name

        nod_file = output_dir / f"{name}.nod.xml"
        edg_file = output_dir / f"{name}.edg.xml"
        con_file = output_dir / f"{name}.con.xml"
        add_file = output_dir / f"{name}.add.xml"
        net_file = output_dir / f"{name}.net.xml"

        # Find netconvert binary
        netconvert = self._find_netconvert()

        cmd = [
            netconvert,
            "-n", str(nod_file.resolve()),
            "-e", str(edg_file.resolve()),
            "-x", str(con_file.resolve()),
            "-o", str(net_file.resolve()),
            "--no-internal-links", "false",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"netconvert failed (exit code {result.returncode}):\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )
        return net_file

    def _find_netconvert(self) -> str:
        """Find the netconvert binary."""
        # Check if netconvert is on PATH
        import shutil

        on_path = shutil.which("netconvert")
        if on_path:
            return on_path

        # Try eclipse-sumo pip package location
        try:
            import sumo

            sumo_home = Path(sumo.SUMO_HOME)
            candidate = sumo_home / "bin" / "netconvert.exe"
            if candidate.exists():
                return str(candidate)
            # Try without .exe extension (Linux/Mac)
            candidate = sumo_home / "bin" / "netconvert"
            if candidate.exists():
                return str(candidate)
        except (ImportError, AttributeError):
            pass

        # Try SUMO_HOME env var
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            candidate = Path(sumo_home) / "bin" / "netconvert.exe"
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            "Could not find netconvert binary. "
            "Ensure SUMO is installed and either:\n"
            "  1. netconvert is on your PATH, or\n"
            "  2. SUMO_HOME environment variable is set, or\n"
            "  3. eclipse-sumo pip package is installed."
        )

    def build_all(self, output_dir: str | Path, network_name: str | None = None) -> dict[str, Path]:
        """
        Generate all XML files and run netconvert.

        Returns:
            Dict mapping file type to output path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = network_name or self.config.network_name

        nod_path = self.generate_nod_xml(output_dir / f"{name}.nod.xml")
        edg_path = self.generate_edg_xml(output_dir / f"{name}.edg.xml")
        con_path = self.generate_con_xml(output_dir / f"{name}.con.xml")
        # Note: .add.xml (TLS) must be generated separately by the caller
        # (e.g., signal_plans module) to ensure phase states match the actual
        # number of links at each intersection.
        add_path = output_dir / f"{name}.add.xml"
        net_path = self.run_netconvert(output_dir, name)

        return {
            "nodes": nod_path,
            "edges": edg_path,
            "connections": con_path,
            "additional": add_path,
            "network": net_path,
        }

    def get_intersection_info(self) -> list[dict[str, Any]]:
        """
        Get information about each signalized intersection.

        Returns:
            List of dicts with keys: node_id, intersection_id, x, y,
            incoming_edges, outgoing_edges.
        """
        info = []
        rows = self.config.rows
        cols = self.config.cols

        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                node_id = self._node_id(i, j)
                int_id = self._intersection_id(i, j)

                incoming = [
                    self._edge_id(i - 1, j, i, j),
                    self._edge_id(i + 1, j, i, j),
                    self._edge_id(i, j - 1, i, j),
                    self._edge_id(i, j + 1, i, j),
                ]
                outgoing = [
                    self._edge_id(i, j, i - 1, j),
                    self._edge_id(i, j, i + 1, j),
                    self._edge_id(i, j, i, j - 1),
                    self._edge_id(i, j, i, j + 1),
                ]

                info.append(
                    {
                        "node_id": node_id,
                        "intersection_id": int_id,
                        "grid_position": (i, j),
                        "x": j * self.config.block_length,
                        "y": i * self.config.block_length,
                        "incoming_edges": incoming,
                        "outgoing_edges": outgoing,
                        "num_incoming": len(incoming),
                        "num_outgoing": len(outgoing),
                    }
                )
        return info

    def get_adjacency(self) -> dict[str, list[str]]:
        """
        Get the adjacency map of intersections (for message passing later).

        Returns:
            Dict mapping intersection_id to list of neighboring intersection_ids.
        """
        rows = self.config.rows
        cols = self.config.cols
        adj = {}

        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                int_id = self._intersection_id(i, j)
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 1 <= ni <= rows and 1 <= nj <= cols:
                        neighbors.append(self._intersection_id(ni, nj))
                adj[int_id] = neighbors
        return adj

    def summary(self) -> str:
        """Return a human-readable summary of the network."""
        lines = [
            f"Grid Network: {self.config.rows}x{self.config.cols}",
            f"  Nodes: {len(self.nodes)}",
            f"  Edges: {len(self.edges)}",
            f"  Signalized Intersections: {len(self.intersection_nodes)}",
            f"  Connections: {len(self.connections)}",
            f"  Block length: {self.config.block_length} m",
            f"  Lanes per direction: {self.config.lanes_per_direction}",
            f"  Speed limit: {self.config.speed_limit} m/s ({self.config.speed_limit * 3.6:.1f} km/h)",
            f"  Bidi: {self.config.bidi}",
            f"",
            f"  Intersections:",
        ]
        for info in self.get_intersection_info():
            lines.append(f"    {info['intersection_id']} at ({info['grid_position']}) -- {info['num_incoming']} incoming, {info['num_outgoing']} outgoing")
        lines.append(f"")
        lines.append(f"  Adjacency:")
        adj = self.get_adjacency()
        for int_id, neighbors in adj.items():
            lines.append(f"    {int_id} -> {neighbors}")
        return "\n".join(lines)
