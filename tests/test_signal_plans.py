"""
Tests for the Signal Plans module.
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

from sumo_network.signal_plans import (
    SignalPlanGenerator,
    SignalConfig,
    SignalPlan,
    Phase,
    create_signal_plan_summary,
)


class TestSignalConfig:
    """Tests for SignalConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SignalConfig()
        assert config.cycle_length == 90.0
        assert config.yellow_time == 3.0
        assert config.all_red_time == 1.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SignalConfig(cycle_length=120.0, yellow_time=4.0, all_red_time=2.0)
        assert config.cycle_length == 120.0
        assert config.yellow_time == 4.0
        assert config.all_red_time == 2.0


class TestPhase:
    """Tests for Phase dataclass."""

    def test_phase_creation(self):
        """Test phase creation."""
        phase = Phase(name="NS_green", duration=40.0, state="GGrr")
        assert phase.name == "NS_green"
        assert phase.duration == 40.0
        assert phase.state == "GGrr"


class TestSignalPlan:
    """Tests for SignalPlan dataclass."""

    def test_plan_creation(self):
        """Test plan creation."""
        phases = [
            Phase(name="NS_green", duration=40.0, state="GGrr"),
            Phase(name="EW_green", duration=40.0, state="rrGG"),
        ]
        plan = SignalPlan(name="test", cycle_length=90.0, phases=phases)
        assert plan.num_phases == 2
        assert plan.cycle_length == 90.0

    def test_get_state_at_time(self):
        """Test getting signal state at different times."""
        phases = [
            Phase(name="NS_green", duration=40.0, state="GGrr"),
            Phase(name="EW_green", duration=40.0, state="rrGG"),
        ]
        plan = SignalPlan(name="test", cycle_length=80.0, phases=phases)

        # Time 0 should be NS_green
        assert plan.get_state_at_time(0) == "GGrr"
        # Time 20 should be NS_green
        assert plan.get_state_at_time(20) == "GGrr"
        # Time 45 should be EW_green
        assert plan.get_state_at_time(45) == "rrGG"
        # Time 70 should be EW_green
        assert plan.get_state_at_time(70) == "rrGG"

    def test_get_state_wraps_around(self):
        """Test that state wraps around the cycle."""
        phases = [
            Phase(name="NS_green", duration=40.0, state="GGrr"),
            Phase(name="EW_green", duration=40.0, state="rrGG"),
        ]
        plan = SignalPlan(name="test", cycle_length=80.0, phases=phases)

        # Time 80 should wrap to 0
        assert plan.get_state_at_time(80) == "GGrr"
        # Time 100 should wrap to 20
        assert plan.get_state_at_time(100) == "GGrr"


class TestSignalPlanGenerator:
    """Tests for SignalPlanGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SignalConfig()
        self.generator = SignalPlanGenerator(self.config)

    def test_create_default_2phase_plan(self):
        """Test default 2-phase plan creation."""
        plan = self.generator.create_default_2phase_plan()
        assert plan.num_phases == 6  # green, yellow, all-red, green, yellow, all-red
        assert plan.cycle_length == 90.0

    def test_plan_phases_correct_duration(self):
        """Test that phases have correct durations."""
        plan = self.generator.create_default_2phase_plan()
        # Green: (90 - 2*3 - 2*1) / 2 = (90 - 8) / 2 = 41s
        total_duration = sum(p.duration for p in plan.phases)
        assert abs(total_duration - 90.0) < 0.01

    def test_plan_phases_correct_states(self):
        """Test that phases have correct signal states."""
        plan = self.generator.create_default_2phase_plan()
        assert plan.phases[0].state == "GGrr"  # NS green
        assert plan.phases[1].state == "yyrr"  # NS yellow
        assert plan.phases[2].state == "rrrr"  # all red
        assert plan.phases[3].state == "rrGG"  # EW green
        assert plan.phases[4].state == "rryy"  # EW yellow
        assert plan.phases[5].state == "rrrr"  # all red

    def test_create_plan_from_config(self):
        """Test plan creation from config with phases."""
        config = SignalConfig(
            phases=[
                {"name": "NS_green", "duration": 30, "state": "GGrr"},
                {"name": "EW_green", "duration": 30, "state": "rrGG"},
            ]
        )
        generator = SignalPlanGenerator(config)
        plan = generator.create_plan_from_config()
        assert plan.num_phases == 2
        assert plan.phases[0].name == "NS_green"

    def test_generate_tls_add_xml(self):
        """Test TLS add XML generation."""
        intersection_ids = ["I_1_1", "I_1_2", "I_2_1", "I_2_2"]
        tmp_dir = tempfile.mkdtemp()
        try:
            path = self.generator.generate_tls_add_xml(
                intersection_ids, os.path.join(tmp_dir, "test.add.xml")
            )
            assert path.exists()
            content = path.read_text()
            assert '<?xml version' in content
            assert 'tlLogic' in content
            assert 'phase' in content

            # Verify all intersections are in the file
            tree = ET.parse(path)
            tl_elements = tree.findall(".//tlLogic")
            assert len(tl_elements) == 4

            # Verify each has phases
            for tl in tl_elements:
                phases = tl.findall(".//phase")
                assert len(phases) > 0
        finally:
            shutil.rmtree(tmp_dir)

    def test_generate_tls_add_xml_with_custom_plan(self):
        """Test TLS add XML with custom signal plan."""
        plan = SignalPlan(
            name="custom",
            cycle_length=120.0,
            phases=[
                Phase(name="NS_green", duration=50.0, state="GGrr"),
                Phase(name="EW_green", duration=50.0, state="rrGG"),
            ],
        )
        intersection_ids = ["I_1_1"]
        tmp_dir = tempfile.mkdtemp()
        try:
            path = self.generator.generate_tls_add_xml(
                intersection_ids, os.path.join(tmp_dir, "custom.add.xml"),
                plan=plan, num_links={"I_1_1": 4},
            )
            tree = ET.parse(path)
            tl = tree.find(".//tlLogic")
            phases = tl.findall(".//phase")
            assert len(phases) == 2
            assert phases[0].get("duration") == "50"
            assert phases[0].get("state") == "GGrr"
        finally:
            shutil.rmtree(tmp_dir)

    def test_compute_offsets_linear(self):
        """Test linear offset computation."""
        intersection_ids = ["I_1_1", "I_1_2", "I_2_1", "I_2_2"]
        offsets = SignalPlanGenerator.compute_offsets_linear(
            intersection_ids, grid_rows=3, grid_cols=3, block_length=300.0
        )
        assert "I_1_1" in offsets
        assert "I_1_2" in offsets
        assert "I_2_1" in offsets
        assert "I_2_2" in offsets

        # I_1_1 should have offset 0
        assert offsets["I_1_1"] == 0.0
        # I_1_2 should be offset by 300m / 8.33m/s ≈ 36s
        assert abs(offsets["I_1_2"] - 36.0) < 0.1


class TestSignalPlanSummary:
    """Tests for signal plan summary."""

    def test_summary(self):
        """Test summary generation."""
        phases = [
            Phase(name="NS_green", duration=40.0, state="GGrr"),
            Phase(name="EW_green", duration=40.0, state="rrGG"),
        ]
        plan = SignalPlan(name="test", cycle_length=80.0, phases=phases)
        summary = create_signal_plan_summary(plan)
        assert "test" in summary
        assert "80" in summary
        assert "NS_green" in summary
        assert "EW_green" in summary
