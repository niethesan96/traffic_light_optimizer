"""
Tests for the Traffic Demand Generator module.
"""

import os
import sys
import tempfile
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sumo_network.traffic_demand import TrafficDemandGenerator, DemandConfig, DEMAND_PRESETS
from sumo_network.grid_builder import GridBuilder, GridConfig


class TestDemandConfig:
    """Tests for DemandConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DemandConfig()
        assert config.duration == 3600
        assert config.seed == 42
        assert config.flow_rate == 300
        assert config.demand_level == "medium"
        assert config.depart_position == "random_free"
        assert config.depart_speed == "max"
        assert config.min_gap == 2.5

    def test_demand_presets(self):
        """Test demand level presets."""
        assert DEMAND_PRESETS["low"] == 100
        assert DEMAND_PRESETS["medium"] == 300
        assert DEMAND_PRESETS["high"] == 600

    def test_get_flow_rate_medium(self):
        """Test flow rate for medium demand."""
        config = DemandConfig(demand_level="medium")
        assert config.get_flow_rate() == 300

    def test_get_flow_rate_low(self):
        """Test flow rate for low demand."""
        config = DemandConfig(demand_level="low")
        assert config.get_flow_rate() == 100

    def test_get_flow_rate_high(self):
        """Test flow rate for high demand."""
        config = DemandConfig(demand_level="high")
        assert config.get_flow_rate() == 600

    def test_get_flow_rate_custom(self):
        """Test flow rate for custom demand level."""
        config = DemandConfig(demand_level="custom", flow_rate=500)
        assert config.get_flow_rate() == 500


class TestTrafficDemandGenerator:
    """Tests for TrafficDemandGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.grid = GridBuilder(GridConfig(rows=3, cols=3, bidi=True))
        self.edge_ids = [edge.id for edge in self.grid.edges]
        self.config = DemandConfig(seed=42)
        self.generator = TrafficDemandGenerator(self.config)

    def test_network_wide_total_demand_scaling(self):
        """flow_rate must be TOTAL network veh/hr, evenly split over OD pairs.

        Regression: randomized begin times forced >= 1 vehicle per flow, so
        low totals (20/40/60) all produced ~36/72/72 departures regardless of
        the requested demand. The generated demand must represent the
        requested total vehicle count within 5%.
        """
        total = 1000
        # demand_level="custom" so flow_rate is used verbatim (a preset
        # level would override the requested total)
        config = DemandConfig(
            duration=3600, seed=42, flow_rate=total, demand_level="custom"
        )
        gen = TrafficDemandGenerator(config)

        tmp_dir = tempfile.mkdtemp()
        try:
            path = gen.generate_network_wide_with_grid_builder(
                self.grid, os.path.join(tmp_dir, "nw.rou.xml")
            )
            root = ET.parse(path).getroot()
            flows = root.findall(".//flow")
            assert len(flows) > 0

            # Every flow spans the full simulation; per-OD rates carry the
            # seeded ±10% jitter (Stage 3: demand seeds must be able to vary
            # the file content), so rates are near-even but not identical.
            assert all(f.get("begin") == "0" for f in flows)
            assert all(f.get("end") == "3600" for f in flows)
            rates = [float(f.get("vehsPerHour")) for f in flows]
            base_rate = total / len(rates)
            assert all(
                0.9 * base_rate - 2e-6 <= r <= 1.1 * base_rate + 2e-6 for r in rates
            ), "per-OD flow left its ±10% seeded jitter band"

            # Same seed → identical file content (deterministic per seed;
            # fresh generator so the jitter RNG is unconsumed)
            gen2 = TrafficDemandGenerator(
                DemandConfig(
                    duration=3600, seed=42, flow_rate=total,
                    demand_level="custom",
                )
            )
            path2 = gen2.generate_network_wide_with_grid_builder(
                self.grid, os.path.join(tmp_dir, "nw2.rou.xml")
            )
            with open(path) as fa, open(path2) as fb:
                assert fa.read() == fb.read()

            # Vehicle count represented by the generated demand within 5%
            expected = sum(
                float(f.get("vehsPerHour"))
                * (float(f.get("end")) - float(f.get("begin"))) / 3600.0
                for f in flows
            )
            assert abs(expected - total) / total <= 0.05, (
                f"Requested {total} vehicles but demand represents {expected:.1f}"
            )
        finally:
            shutil.rmtree(tmp_dir)

    def test_identify_boundary_edges(self):
        """Test boundary edge identification."""
        boundaries = self.generator.identify_boundary_edges(self.edge_ids, grid_rows=3, grid_cols=3)
        assert "entry" in boundaries
        assert "exit" in boundaries
        assert len(boundaries["entry"]) > 0
        assert len(boundaries["exit"]) > 0

    def test_boundary_edges_are_bidi(self):
        """Boundary edges should come in pairs (entry and exit)."""
        boundaries = self.generator.identify_boundary_edges(self.edge_ids, grid_rows=3, grid_cols=3)
        # 3×3 intersections → 4×4 node grid
        # Each of 4 top-row nodes has 1 entry from above = 4 entries
        # Each of 4 bottom-row nodes has 1 exit to below = 4 exits
        # Each of 4 left-column nodes has 1 entry from left = 4 entries
        # Each of 4 right-column nodes has 1 exit to right = 4 exits
        # Total entries: 8, Total exits: 8
        assert len(boundaries["entry"]) == 8
        assert len(boundaries["exit"]) == 8

    def test_generate_rou_xml(self):
        """Test .rou.xml file generation."""
        tmp_dir = tempfile.mkdtemp()
        try:
            path = self.generator.generate_rou_xml(
                self.edge_ids, os.path.join(tmp_dir, "test.rou.xml")
            )
            assert path.exists()
            content = path.read_text()
            assert '<?xml version' in content
            assert '<routes' in content
            assert 'vType' in content
            assert 'flow' in content
        finally:
            shutil.rmtree(tmp_dir)

    def test_generate_rou_xml_with_od(self):
        """Test .rou.xml file generation with OD pairs."""
        tmp_dir = tempfile.mkdtemp()
        try:
            boundaries = self.generator.identify_boundary_edges(self.edge_ids, grid_rows=3, grid_cols=3)
            path = self.generator.generate_rou_xml_with_od(
                boundaries["entry"],
                boundaries["exit"],
                os.path.join(tmp_dir, "test_od.rou.xml"),
            )
            assert path.exists()
            content = path.read_text()
            assert '<?xml version' in content
            assert 'from="' in content
            assert 'to="' in content
        finally:
            shutil.rmtree(tmp_dir)

    def test_generate_with_grid_builder(self):
        """Test generation using grid builder reference."""
        tmp_dir = tempfile.mkdtemp()
        try:
            path = self.generator.generate_with_grid_builder(
                self.grid, os.path.join(tmp_dir, "test_grid.rou.xml")
            )
            assert path.exists()
            content = path.read_text()
            assert 'flow' in content
        finally:
            shutil.rmtree(tmp_dir)

    def test_flow_rate_scaling(self):
        """Test that demand scale multiplier works."""
        tmp_dir = tempfile.mkdtemp()
        try:
            path_half = self.generator.generate_rou_xml(
                self.edge_ids,
                os.path.join(tmp_dir, "half.rou.xml"),
                demand_scale=0.5,
            )
            path_double = self.generator.generate_rou_xml(
                self.edge_ids,
                os.path.join(tmp_dir, "double.rou.xml"),
                demand_scale=2.0,
            )

            tree_half = ET.parse(path_half)
            tree_double = ET.parse(path_double)

            flows_half = tree_half.findall(".//flow")
            flows_double = tree_double.findall(".//flow")

            assert len(flows_half) == len(flows_double)

            for f_half, f_double in zip(flows_half, flows_double):
                rate_half = int(f_half.get("vehsPerHour", "0"))
                rate_double = int(f_double.get("vehsPerHour", "0"))
                assert rate_double == rate_half * 4
        finally:
            shutil.rmtree(tmp_dir)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same output."""
        config1 = DemandConfig(seed=42)
        config2 = DemandConfig(seed=42)
        gen1 = TrafficDemandGenerator(config1)
        gen2 = TrafficDemandGenerator(config2)

        tmp_dir = tempfile.mkdtemp()
        try:
            path1 = gen1.generate_rou_xml(
                self.edge_ids, os.path.join(tmp_dir, "test1.rou.xml")
            )
            path2 = gen2.generate_rou_xml(
                self.edge_ids, os.path.join(tmp_dir, "test2.rou.xml")
            )
            content1 = path1.read_text()
            content2 = path2.read_text()
            assert content1 == content2
        finally:
            shutil.rmtree(tmp_dir)

    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different output."""
        config1 = DemandConfig(seed=42)
        config2 = DemandConfig(seed=123)
        gen1 = TrafficDemandGenerator(config1)
        gen2 = TrafficDemandGenerator(config2)

        tmp_dir = tempfile.mkdtemp()
        try:
            path1 = gen1.generate_rou_xml(
                self.edge_ids, os.path.join(tmp_dir, "test1.rou.xml")
            )
            path2 = gen2.generate_rou_xml(
                self.edge_ids, os.path.join(tmp_dir, "test2.rou.xml")
            )
            content1 = path1.read_text()
            content2 = path2.read_text()
            assert content1 != content2
        finally:
            shutil.rmtree(tmp_dir)

    def test_vehicle_type_definition(self):
        """Test that vehicle type is properly defined."""
        tmp_dir = tempfile.mkdtemp()
        try:
            path = self.generator.generate_rou_xml(
                self.edge_ids, os.path.join(tmp_dir, "test.rou.xml")
            )
            tree = ET.parse(path)
            vtypes = tree.findall(".//vType")
            assert len(vtypes) == 1
            vtype = vtypes[0]
            assert vtype.get("id") == "passenger"
            assert vtype.get("length") == "5.0"
            assert vtype.get("minGap") == "2.5"
        finally:
            shutil.rmtree(tmp_dir)

    def test_flow_attributes(self):
        """Test that flow elements have required attributes."""
        tmp_dir = tempfile.mkdtemp()
        try:
            path = self.generator.generate_rou_xml(
                self.edge_ids, os.path.join(tmp_dir, "test.rou.xml")
            )
            tree = ET.parse(path)
            flows = tree.findall(".//flow")
            assert len(flows) > 0

            for flow in flows:
                assert flow.get("id") is not None
                assert flow.get("type") == "passenger"
                assert flow.get("from") is not None
                assert flow.get("vehsPerHour") is not None
                assert flow.get("begin") is not None
                assert flow.get("end") is not None
        finally:
            shutil.rmtree(tmp_dir)

    def test_depart_position_config(self):
        """Test that depart position is configurable."""
        config = DemandConfig(depart_position="free", depart_speed="max")
        gen = TrafficDemandGenerator(config)

        tmp_dir = tempfile.mkdtemp()
        try:
            path = gen.generate_rou_xml(
                self.edge_ids, os.path.join(tmp_dir, "test.rou.xml")
            )
            tree = ET.parse(path)
            flows = tree.findall(".//flow")
            for flow in flows:
                assert flow.get("departPos") == "free"
                assert flow.get("departSpeed") == "max"
        finally:
            shutil.rmtree(tmp_dir)

    def test_summary(self):
        """Test summary generation."""
        summary = self.generator.summary()
        assert "Traffic Demand Configuration" in summary
        assert "3600" in summary  # duration
        assert "300" in summary   # flow rate


class TestTrafficDemandIntegration:
    """Integration tests with the grid builder."""

    def test_full_pipeline(self):
        """Test full pipeline: grid builder -> demand generator -> XML files."""
        config = GridConfig(rows=3, cols=3)
        grid = GridBuilder(config)

        demand_config = DemandConfig(seed=42, demand_level="medium")
        gen = TrafficDemandGenerator(demand_config)

        tmp_dir = tempfile.mkdtemp()
        try:
            path = gen.generate_with_grid_builder(
                grid, os.path.join(tmp_dir, "demand.rou.xml")
            )
            assert path.exists()

            tree = ET.parse(path)
            root = tree.getroot()
            assert root.tag == "routes"

            flows = root.findall(".//flow")
            assert len(flows) > 0

            vtypes = root.findall(".//vType")
            assert len(vtypes) == 1
        finally:
            shutil.rmtree(tmp_dir)

    def test_demand_with_2x2_grid(self):
        """Test demand generation with 2x2 grid."""
        config = GridConfig(rows=2, cols=2)
        grid = GridBuilder(config)
        edge_ids = [edge.id for edge in grid.edges]

        demand_config = DemandConfig(seed=42)
        gen = TrafficDemandGenerator(demand_config)

        tmp_dir = tempfile.mkdtemp()
        try:
            path = gen.generate_rou_xml(
                edge_ids, os.path.join(tmp_dir, "demand.rou.xml")
            )
            assert path.exists()
            tree = ET.parse(path)
            flows = tree.findall(".//flow")
            assert len(flows) > 0
        finally:
            shutil.rmtree(tmp_dir)

    def test_demand_with_custom_flow_rate(self):
        """Test demand with custom flow rate."""
        config = GridConfig(rows=3, cols=3)
        grid = GridBuilder(config)
        edge_ids = [edge.id for edge in grid.edges]

        demand_config = DemandConfig(seed=42, demand_level="custom", flow_rate=500)
        gen = TrafficDemandGenerator(demand_config)

        tmp_dir = tempfile.mkdtemp()
        try:
            path = gen.generate_rou_xml(
                edge_ids, os.path.join(tmp_dir, "demand.rou.xml")
            )
            tree = ET.parse(path)
            flows = tree.findall(".//flow")
            for flow in flows:
                assert flow.get("vehsPerHour") == "500"
        finally:
            shutil.rmtree(tmp_dir)
