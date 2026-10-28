"""
Tests for the SUMO Grid Builder module.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sumo_network.grid_builder import GridBuilder, GridConfig


class TestGridConfig:
    """Tests for GridConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GridConfig()
        assert config.rows == 3
        assert config.cols == 3
        assert config.block_length == 300.0
        assert config.lanes_per_direction == 1
        assert config.speed_limit == 8.33
        assert config.bidi is True
        assert config.network_name == "grid_3x3"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GridConfig(rows=4, cols=5, block_length=200.0, lanes_per_direction=2)
        assert config.rows == 4
        assert config.cols == 5
        assert config.block_length == 200.0
        assert config.lanes_per_direction == 2


class TestGridBuilder:
    """Tests for GridBuilder class.

    Convention: rows=3, cols=3 means 3×3 = 9 signalized intersections.
    Node grid is (rows+1) × (cols+1) = 4×4 = 16 nodes.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.config_3x3 = GridConfig(rows=3, cols=3)
        self.grid_3x3 = GridBuilder(self.config_3x3)

    def test_node_count_3x3(self):
        """A 3×3 intersection grid has (3+1)×(3+1)=16 nodes."""
        assert len(self.grid_3x3.nodes) == 16

    def test_node_count_2x2(self):
        """A 2×2 intersection grid has 3×3=9 nodes."""
        config = GridConfig(rows=2, cols=2)
        grid = GridBuilder(config)
        assert len(grid.nodes) == 9

    def test_node_count_4x4(self):
        """A 4×4 intersection grid has 5×5=25 nodes."""
        config = GridConfig(rows=4, cols=4)
        grid = GridBuilder(config)
        assert len(grid.nodes) == 25

    def test_intersection_count_3x3(self):
        """A 3×3 grid should have 9 signalized intersections."""
        assert len(self.grid_3x3.intersection_nodes) == 9

    def test_intersection_count_2x2(self):
        """A 2×2 grid should have 4 signalized intersections."""
        config = GridConfig(rows=2, cols=2)
        grid = GridBuilder(config)
        assert len(grid.intersection_nodes) == 4

    def test_intersection_positions(self):
        """Check that intersections are at correct grid positions (1..rows, 1..cols)."""
        for node_id in self.grid_3x3.intersection_nodes:
            parts = node_id.split("_")
            row, col = int(parts[1]), int(parts[2])
            assert 1 <= row <= 3
            assert 1 <= col <= 3

    def test_edge_count_3x3_bidi(self):
        """A 3×3 intersection grid with bidi edges:
        Horizontal: (rows+1) × cols × 2 = 4 × 3 × 2 = 24
        Vertical: rows × (cols+1) × 2 = 3 × 4 × 2 = 24
        Total: 48
        """
        assert len(self.grid_3x3.edges) == 48

    def test_edge_count_3x3_no_bidi(self):
        """A 3×3 intersection grid without bidi edges:
        Horizontal: (rows+1) × cols = 4 × 3 = 12
        Vertical: rows × (cols+1) = 3 × 4 = 12
        Total: 24
        """
        config = GridConfig(rows=3, cols=3, bidi=False)
        grid = GridBuilder(config)
        assert len(grid.edges) == 24

    def test_edge_ids_unique(self):
        """All edge IDs should be unique."""
        edge_ids = [edge.id for edge in self.grid_3x3.edges]
        assert len(edge_ids) == len(set(edge_ids))

    def test_node_ids_unique(self):
        """All node IDs should be unique."""
        node_ids = [node.id for node in self.grid_3x3.nodes]
        assert len(node_ids) == len(set(node_ids))

    def test_node_positions(self):
        """Check node positions match grid coordinates."""
        for node in self.grid_3x3.nodes:
            parts = node.id.split("_")
            row, col = int(parts[1]), int(parts[2])
            expected_x = col * 300.0
            expected_y = row * 300.0
            assert abs(node.x - expected_x) < 0.01
            assert abs(node.y - expected_y) < 0.01

    def test_boundary_nodes_not_intersections(self):
        """Boundary nodes should not be signalized intersections."""
        for node in self.grid_3x3.nodes:
            if not node.is_intersection:
                parts = node.id.split("_")
                row, col = int(parts[1]), int(parts[2])
                # At least one coordinate should be on the boundary (0 or 3)
                assert row == 0 or row == 3 or col == 0 or col == 3

    def test_connections_exist(self):
        """Connections should be generated for signalized intersections."""
        assert len(self.grid_3x3.connections) > 0

    def test_connection_count(self):
        """Connections depend on which edges exist at each intersection.
        Boundary intersections may have fewer than 4 connections.
        For a 3×3 intersection grid: 24 connections total.
        """
        assert len(self.grid_3x3.connections) == 24

    def test_adjacency(self):
        """Test intersection adjacency map."""
        adj = self.grid_3x3.get_adjacency()
        assert len(adj) == 9
        # Corner intersections have 2 neighbors
        assert len(adj["I_1_1"]) == 2
        assert len(adj["I_3_3"]) == 2
        # Center intersection has 4 neighbors
        assert len(adj["I_2_2"]) == 4
        # Edge intersections (non-corner) have 3 neighbors
        assert len(adj["I_1_2"]) == 3
        assert len(adj["I_2_1"]) == 3

    def test_adjacency_symmetry(self):
        """If A is neighbor of B, then B should be neighbor of A."""
        adj = self.grid_3x3.get_adjacency()
        for node, neighbors in adj.items():
            for neighbor in neighbors:
                assert node in adj[neighbor], f"{node} is neighbor of {neighbor}, but not vice versa"

    def test_intersection_info(self):
        """Test intersection info generation."""
        info = self.grid_3x3.get_intersection_info()
        assert len(info) == 9
        for item in info:
            assert "intersection_id" in item
            assert "incoming_edges" in item
            assert "outgoing_edges" in item
            assert item["num_incoming"] == 4
            assert item["num_outgoing"] == 4

    def test_xml_generation_nod(self):
        """Test .nod.xml file generation."""
        tmp_dir = tempfile.mkdtemp()
        try:
            path = self.grid_3x3.generate_nod_xml(os.path.join(tmp_dir, "test.nod.xml"))
            assert path.exists()
            content = path.read_text()
            assert '<?xml version' in content
            assert 'id="N_0_0"' in content
            assert 'id="N_3_3"' in content
        finally:
            shutil.rmtree(tmp_dir)

    def test_xml_generation_edg(self):
        """Test .edg.xml file generation."""
        tmp_dir = tempfile.mkdtemp()
        try:
            path = self.grid_3x3.generate_edg_xml(os.path.join(tmp_dir, "test.edg.xml"))
            assert path.exists()
            content = path.read_text()
            assert '<?xml version' in content
            assert 'from="' in content
            assert 'to="' in content
        finally:
            shutil.rmtree(tmp_dir)

    def test_xml_generation_con(self):
        """Test .con.xml file generation."""
        tmp_dir = tempfile.mkdtemp()
        try:
            path = self.grid_3x3.generate_con_xml(os.path.join(tmp_dir, "test.con.xml"))
            assert path.exists()
            content = path.read_text()
            assert '<?xml version' in content
            assert 'tl="' in content
        finally:
            shutil.rmtree(tmp_dir)

    def test_xml_generation_add(self):
        """Test .add.xml file generation."""
        tmp_dir = tempfile.mkdtemp()
        try:
            path = self.grid_3x3.generate_add_xml(os.path.join(tmp_dir, "test.add.xml"))
            assert path.exists()
            content = path.read_text()
            assert '<?xml version' in content
            assert 'tlLogic' in content
            assert 'phase' in content
        finally:
            shutil.rmtree(tmp_dir)

    def test_summary(self):
        """Test summary generation."""
        summary = self.grid_3x3.summary()
        assert "3x3" in summary
        assert "16" in summary  # nodes
        assert "48" in summary  # edges
        assert "9" in summary   # intersections

    def test_custom_block_length(self):
        """Test with different block length."""
        config = GridConfig(rows=3, cols=3, block_length=500.0)
        grid = GridBuilder(config)
        for node in grid.nodes:
            parts = node.id.split("_")
            col = int(parts[2])
            expected_x = col * 500.0
            assert abs(node.x - expected_x) < 0.01


class TestGridBuilderEdgeCases:
    """Tests for edge cases."""

    def test_1x1_grid(self):
        """A 1×1 intersection grid has 2×2=4 nodes and 1 intersection."""
        config = GridConfig(rows=1, cols=1)
        grid = GridBuilder(config)
        assert len(grid.nodes) == 4
        assert len(grid.intersection_nodes) == 1

    def test_2x2_grid(self):
        """A 2×2 intersection grid has 3×3=9 nodes and 4 intersections."""
        config = GridConfig(rows=2, cols=2)
        grid = GridBuilder(config)
        assert len(grid.nodes) == 9
        assert len(grid.intersection_nodes) == 4


class TestGridBuilderBuildAll:
    """Tests for the build_all method."""

    def test_build_all_creates_files(self):
        """Test that build_all creates all required files."""
        tmp_dir = tempfile.mkdtemp()
        try:
            config = GridConfig(rows=3, cols=3, network_name="test_grid")
            grid = GridBuilder(config)
            # No add.xml — netconvert auto-generates TLS from node types
            files = grid.build_all(tmp_dir, "test_grid")

            assert "nodes" in files
            assert "edges" in files
            assert "connections" in files
            assert "additional" in files
            assert "network" in files

            for key, path in files.items():
                # additional file may not exist if not generated separately
                if key != "additional":
                    assert path.exists(), f"File {key} not created: {path}"

            net_file = files["network"]
            assert net_file.stat().st_size > 0
        finally:
            shutil.rmtree(tmp_dir)
