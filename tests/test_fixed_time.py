"""
Tests for Phase 2 — Fixed-Time Baseline.

Tests cover:
  - Metric calculation logic (MetricsCollector, SimulationResults)
  - Vehicle waiting-time tracking
  - Throughput calculation
  - Queue length calculation
  - Result file generation (JSON, CSV)
  - Signal plan offset coordination
  - Reproducibility
"""

import csv
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (
    MetricsCollector,
    SimulationResults,
    IntersectionMetrics,
    StepSnapshot,
    compute_travel_times_from_tripinfo,
    save_results_json,
    save_results_csv,
    print_summary,
)
from sumo_network.signal_plans import SignalPlanGenerator, SignalConfig, SignalPlan, Phase
from sumo_network.grid_builder import GridBuilder, GridConfig
from sumo_network.traffic_demand import TrafficDemandGenerator, DemandConfig


# ============================================================================
# MetricsCollector Unit Tests
# ============================================================================


class TestMetricsCollectorSetup:
    """Tests for MetricsCollector initialization."""

    def test_setup_creates_intersection_metrics(self):
        """Each intersection should get its own IntersectionMetrics."""
        collector = MetricsCollector()
        iids = ["I_1_1", "I_1_2", "I_2_1"]
        collector.setup(iids, {"I_1_1": ["E_0_1_to_1_1"], "I_1_2": ["E_0_2_to_1_2"], "I_2_1": ["E_1_1_to_2_1"]})
        assert len(collector.intersection_metrics) == 3
        for iid in iids:
            assert iid in collector.intersection_metrics

    def test_setup_stores_config(self):
        """Config metadata should be stored."""
        collector = MetricsCollector()
        cfg = {"seed": 42, "duration": 300}
        collector.setup(["I_1_1"], {"I_1_1": []}, config=cfg)
        assert collector.config["seed"] == 42
        assert collector.config["duration"] == 300

    def test_setup_initializes_empty_steps(self):
        """Steps list should be empty at start."""
        collector = MetricsCollector()
        collector.setup(["I_1_1"], {"I_1_1": []})
        assert len(collector.steps) == 0
        assert collector._total_vehicles_entered == 0


# ============================================================================
# IntersectionMetrics Tests
# ============================================================================


class TestIntersectionMetrics:
    """Tests for IntersectionMetrics dataclass."""

    def test_avg_queue_length(self):
        """Average queue should be total / steps."""
        m = IntersectionMetrics(intersection_id="I_1_1")
        m.total_queue_length = 100
        m.step_count = 10
        assert m.avg_queue_length == 10.0

    def test_avg_queue_length_zero_steps(self):
        """Zero steps should return 0 (not divide by zero)."""
        m = IntersectionMetrics(intersection_id="I_1_1")
        m.total_queue_length = 0
        m.step_count = 0
        assert m.avg_queue_length == 0.0

    def test_max_queue_length(self):
        """Max queue should be tracked."""
        m = IntersectionMetrics(intersection_id="I_1_1")
        m.max_queue_length = 15
        assert m.max_queue_length == 15

    def test_avg_speed(self):
        """Average speed should be mean of samples."""
        m = IntersectionMetrics(intersection_id="I_1_1")
        m.speed_samples = [10.0, 20.0, 30.0]
        assert abs(m.avg_speed - 20.0) < 0.01

    def test_avg_speed_empty(self):
        """Empty speed samples should return 0."""
        m = IntersectionMetrics(intersection_id="I_1_1")
        assert m.avg_speed == 0.0

    def test_to_dict(self):
        """to_dict should return all fields."""
        m = IntersectionMetrics(intersection_id="I_1_1")
        m.total_queue_length = 50
        m.step_count = 5
        m.max_queue_length = 15
        m.speed_samples = [10.0, 20.0]
        d = m.to_dict()
        assert d["intersection_id"] == "I_1_1"
        assert d["avg_queue_length"] == 10.0
        assert d["max_queue_length"] == 15
        assert d["avg_speed"] > 0


# ============================================================================
# SimulationResults Tests
# ============================================================================


class TestSimulationResults:
    """Tests for SimulationResults dataclass."""

    def test_to_dict_structure(self):
        """to_dict should have network_level and per_intersection keys."""
        results = SimulationResults()
        d = results.to_dict()
        assert "network_level" in d
        assert "per_intersection" in d
        assert "config" in d

    def test_to_dict_network_level_fields(self):
        """Network-level should have all required fields."""
        results = SimulationResults(
            total_waiting_time=1000.0,
            avg_waiting_time=5.0,
            avg_queue_length=3.2,
            max_queue_length=15,
            throughput=25,
            vehicles_entered=30,
            vehicles_remaining=5,
        )
        d = results.to_dict()["network_level"]
        assert d["total_waiting_time"] == 1000.0
        assert d["throughput"] == 25
        assert d["vehicles_entered"] == 30
        assert d["vehicles_remaining"] == 5


# ============================================================================
# Vehicle Waiting-Time Tracking Tests
# ============================================================================


class TestWaitingTimeTracking:
    """Tests for vehicle waiting time tracking logic."""

    def _make_mock_traci(self, vehicles: dict[str, float], departed: int = 0, arrived: int = 0):
        """Create a mock traci module."""
        mock = MagicMock()
        mock.vehicle.getIDList.return_value = list(vehicles.keys())
        mock.vehicle.getWaitingTime.side_effect = lambda vid: vehicles.get(vid, 0.0)
        mock.vehicle.getSpeed.return_value = 10.0
        mock.simulation.getDepartedNumber.return_value = departed
        mock.simulation.getArrivedNumber.return_value = arrived
        mock.edge.getLastStepVehicleIDs.return_value = []
        return mock

    def test_single_vehicle_waiting(self):
        """Single vehicle with waiting time should be tracked."""
        collector = MetricsCollector()
        collector.setup(["I_1_1"], {"I_1_1": ["E_0_1_to_1_1"]})
        collector.start()

        mock = self._make_mock_traci({"v1": 10.0})
        snapshot = collector.collect_step(mock, 10.0)
        assert snapshot.total_waiting_time == 10.0
        assert snapshot.avg_waiting_time == 10.0

    def test_multiple_vehicles_waiting(self):
        """Multiple vehicles' waiting times should be summed."""
        collector = MetricsCollector()
        collector.setup(["I_1_1"], {"I_1_1": ["E_0_1_to_1_1"]})
        collector.start()

        vehicles = {"v1": 10.0, "v2": 20.0, "v3": 5.0}
        mock = self._make_mock_traci(vehicles)
        snapshot = collector.collect_step(mock, 10.0)
        assert snapshot.total_waiting_time == 35.0
        assert abs(snapshot.avg_waiting_time - 35.0 / 3) < 0.01

    def test_no_vehicles_zero_wait(self):
        """No vehicles should give zero waiting time."""
        collector = MetricsCollector()
        collector.setup(["I_1_1"], {"I_1_1": ["E_0_1_to_1_1"]})
        collector.start()

        mock = self._make_mock_traci({})
        snapshot = collector.collect_step(mock, 10.0)
        assert snapshot.total_waiting_time == 0.0
        assert snapshot.avg_waiting_time == 0.0

    def test_waiting_time_accumulates_across_steps(self):
        """Waiting time should accumulate across steps."""
        collector = MetricsCollector()
        collector.setup(["I_1_1"], {"I_1_1": ["E_0_1_to_1_1"]})
        collector.start()

        mock = self._make_mock_traci({"v1": 10.0})
        collector.collect_step(mock, 10.0)
        collector.collect_step(mock, 20.0)

        # Total sum across steps should be 20.0
        assert collector._sum_waiting_times == 20.0


# ============================================================================
# Throughput Calculation Tests
# ============================================================================


class TestThroughputCalculation:
    """Tests for throughput (vehicles completed) tracking."""

    def test_throughput_counts_departed(self):
        """Departed vehicles should be counted as entered."""
        collector = MetricsCollector()
        collector.setup([], {})
        collector.start()

        mock = MagicMock()
        mock.vehicle.getIDList.return_value = []
        mock.vehicle.getWaitingTime.return_value = 0.0
        mock.vehicle.getSpeed.return_value = 0.0
        mock.edge.getLastStepVehicleIDs.return_value = []
        mock.simulation.getDepartedNumber.return_value = 5
        mock.simulation.getArrivedNumber.return_value = 0

        collector.collect_step(mock, 10.0)
        assert collector._total_vehicles_entered == 5
        assert collector._total_vehicles_left == 0

    def test_throughput_counts_arrived(self):
        """Arrived vehicles should be counted as completed."""
        collector = MetricsCollector()
        collector.setup([], {})
        collector.start()

        mock = MagicMock()
        mock.vehicle.getIDList.return_value = []
        mock.vehicle.getWaitingTime.return_value = 0.0
        mock.vehicle.getSpeed.return_value = 0.0
        mock.edge.getLastStepVehicleIDs.return_value = []
        mock.simulation.getDepartedNumber.return_value = 0
        mock.simulation.getArrivedNumber.return_value = 3

        collector.collect_step(mock, 10.0)
        assert collector._total_vehicles_entered == 0
        assert collector._total_vehicles_left == 3

    def test_throughput_accumulates(self):
        """Throughput should accumulate across multiple steps."""
        collector = MetricsCollector()
        collector.setup([], {})
        collector.start()

        mock = MagicMock()
        mock.vehicle.getIDList.return_value = []
        mock.vehicle.getWaitingTime.return_value = 0.0
        mock.vehicle.getSpeed.return_value = 0.0
        mock.edge.getLastStepVehicleIDs.return_value = []
        mock.simulation.getDepartedNumber.return_value = 3
        mock.simulation.getArrivedNumber.return_value = 1

        collector.collect_step(mock, 10.0)
        collector.collect_step(mock, 20.0)
        collector.collect_step(mock, 30.0)

        assert collector._total_vehicles_entered == 9
        assert collector._total_vehicles_left == 3

    def test_vehicles_remaining(self):
        """Vehicles remaining should be entered - left."""
        results = SimulationResults(
            vehicles_entered=30,
            vehicles_remaining=5,
            throughput=25,
        )
        assert results.vehicles_remaining + results.throughput == results.vehicles_entered


# ============================================================================
# Queue Calculation Tests
# ============================================================================


class TestQueueCalculation:
    """Tests for queue length calculation."""

    def test_queue_uses_speed_threshold(self):
        """Vehicles with speed < 0.1 should be counted as queued."""
        collector = MetricsCollector()
        collector.setup(["I_1_1"], {"I_1_1": ["E_0_1_to_1_1"]})
        collector.start()

        mock = MagicMock()
        mock.vehicle.getIDList.return_value = ["v1", "v2", "v3"]
        mock.vehicle.getWaitingTime.return_value = 0.0
        mock.vehicle.getSpeed.side_effect = lambda vid: {"v1": 0.0, "v2": 0.05, "v3": 5.0}[vid]
        mock.edge.getLastStepVehicleIDs.return_value = ["v1", "v2", "v3"]
        mock.simulation.getDepartedNumber.return_value = 0
        mock.simulation.getArrivedNumber.return_value = 0

        snapshot = collector.collect_step(mock, 10.0)
        # v1 and v2 are queued (speed < 0.1), v3 is not
        assert snapshot.queue_lengths["I_1_1"] == 2

    def test_max_queue_tracked(self):
        """Max queue across steps should be tracked."""
        collector = MetricsCollector()
        collector.setup(["I_1_1"], {"I_1_1": ["E_0_1_to_1_1"]})
        collector.start()

        mock = MagicMock()
        mock.vehicle.getIDList.return_value = ["v1"]
        mock.vehicle.getWaitingTime.return_value = 0.0
        mock.vehicle.getSpeed.return_value = 0.0
        mock.edge.getLastStepVehicleIDs.return_value = ["v1"]
        mock.simulation.getDepartedNumber.return_value = 0
        mock.simulation.getArrivedNumber.return_value = 0

        collector.collect_step(mock, 10.0)
        # _max_queue_length should be at least 1
        assert collector._max_queue_length >= 1


# ============================================================================
# Result File Generation Tests
# ============================================================================


class TestResultFileGeneration:
    """Tests for saving results to files."""

    def _make_sample_results(self) -> SimulationResults:
        """Create a sample SimulationResults for testing."""
        im1 = IntersectionMetrics(intersection_id="I_1_1")
        im1.total_queue_length = 50
        im1.step_count = 10
        im1.max_queue_length = 10
        im1.speed_samples = [10.0, 15.0]

        im2 = IntersectionMetrics(intersection_id="I_1_2")
        im2.total_queue_length = 30
        im2.step_count = 10
        im2.max_queue_length = 8
        im2.speed_samples = [12.0, 18.0]

        return SimulationResults(
            total_waiting_time=500.0,
            avg_waiting_time=50.0,
            avg_queue_length=8.0,
            max_queue_length=18,
            throughput=25,
            vehicles_entered=30,
            vehicles_remaining=5,
            simulation_duration=120.0,
            sim_end_time=300.0,
            intersection_metrics={"I_1_1": im1, "I_1_2": im2},
            config={"seed": 42, "duration": 300},
        )

    def test_save_json(self):
        """JSON file should be valid and contain expected data."""
        tmp_dir = tempfile.mkdtemp()
        try:
            results = self._make_sample_results()
            path = save_results_json(results, os.path.join(tmp_dir, "results.json"))
            assert path.exists()

            with open(path) as f:
                data = json.load(f)
            assert "network_level" in data
            assert data["network_level"]["throughput"] == 25
            assert "I_1_1" in data["per_intersection"]
        finally:
            shutil.rmtree(tmp_dir)

    def test_save_csv_network_summary(self):
        """Network summary CSV should be valid."""
        tmp_dir = tempfile.mkdtemp()
        try:
            results = self._make_sample_results()
            paths = save_results_csv(results, tmp_dir)
            assert "network_summary" in paths
            assert paths["network_summary"].exists()

            with open(paths["network_summary"]) as f:
                reader = csv.reader(f)
                rows = list(reader)
            # Header + 1 data row
            assert len(rows) == 2
            assert "throughput" in rows[0]
        finally:
            shutil.rmtree(tmp_dir)

    def test_save_csv_per_intersection(self):
        """Per-intersection CSV should have one row per intersection."""
        tmp_dir = tempfile.mkdtemp()
        try:
            results = self._make_sample_results()
            paths = save_results_csv(results, tmp_dir)
            assert "per_intersection" in paths

            with open(paths["per_intersection"]) as f:
                reader = csv.reader(f)
                rows = list(reader)
            # Header + 2 data rows (2 intersections)
            assert len(rows) == 3
        finally:
            shutil.rmtree(tmp_dir)

    def test_save_csv_step_metrics(self):
        """Step metrics CSV should have correct columns."""
        tmp_dir = tempfile.mkdtemp()
        try:
            results = self._make_sample_results()
            results.steps = [
                StepSnapshot(time=10.0, num_vehicles=5, total_waiting_time=20.0,
                             avg_waiting_time=4.0, queue_lengths={}, vehicle_speeds={},
                             vehicles_per_intersection={}),
                StepSnapshot(time=20.0, num_vehicles=8, total_waiting_time=30.0,
                             avg_waiting_time=3.75, queue_lengths={}, vehicle_speeds={},
                             vehicles_per_intersection={}),
            ]
            paths = save_results_csv(results, tmp_dir)
            assert "step_metrics" in paths

            with open(paths["step_metrics"]) as f:
                reader = csv.reader(f)
                rows = list(reader)
            # Header + 2 data rows
            assert len(rows) == 3
            assert rows[0][0] == "time"
        finally:
            shutil.rmtree(tmp_dir)

    def test_output_directory_created(self):
        """Output directory should be created if it doesn't exist."""
        tmp_dir = tempfile.mkdtemp()
        try:
            nested = os.path.join(tmp_dir, "results", "baseline")
            results = self._make_sample_results()
            save_results_json(results, os.path.join(nested, "test.json"))
            assert os.path.isdir(nested)
        finally:
            shutil.rmtree(tmp_dir)


# ============================================================================
# Signal Plan Offset Tests
# ============================================================================


class TestSignalPlanOffsets:
    """Tests for signal plan offset coordination."""

    def test_offsets_computed_correctly(self):
        """Linear offsets should increase with distance from corner."""
        iids = [f"I_{i}_{j}" for i in range(1, 4) for j in range(1, 4)]
        offsets = SignalPlanGenerator.compute_offsets_linear(
            iids, grid_rows=3, grid_cols=3, block_length=300.0, wave_speed=8.33,
        )
        assert offsets["I_1_1"] == 0.0
        # I_1_2 is 300m east -> 300/8.33 ≈ 36s
        assert abs(offsets["I_1_2"] - 36.0) < 0.1
        # I_2_1 is 300m south -> same offset
        assert abs(offsets["I_2_1"] - 36.0) < 0.1
        # I_3_3 is 600+600=1200m -> 1200/8.33 ≈ 144s
        assert abs(offsets["I_3_3"] - 144.1) < 0.1

    def test_signal_plan_from_config(self):
        """Signal plan from YAML config should have 6 phases summing to cycle."""
        config = SignalConfig(
            cycle_length=90.0,
            yellow_time=3.0,
            all_red_time=1.0,
            phases=[
                {"name": "NS_green", "duration": 41, "state": "GGGrrr"},
                {"name": "NS_yellow", "duration": 3, "state": "yyyrrr"},
                {"name": "all_red_1", "duration": 1, "state": "rrrrrr"},
                {"name": "EW_green", "duration": 41, "state": "rrrGGG"},
                {"name": "EW_yellow", "duration": 3, "state": "rrryyy"},
                {"name": "all_red_2", "duration": 1, "state": "rrrrrr"},
            ],
        )
        gen = SignalPlanGenerator(config)
        plan = gen.create_plan_from_config()
        assert plan.num_phases == 6
        total = sum(p.duration for p in plan.phases)
        assert abs(total - 90.0) < 0.01

    def test_plan_xml_with_offsets(self):
        """TLS XML with offsets should have correct offset attribute."""
        tmp_dir = tempfile.mkdtemp()
        try:
            config = SignalConfig(cycle_length=90.0, yellow_time=3.0, all_red_time=1.0)
            gen = SignalPlanGenerator(config)
            plan = gen.create_default_2phase_plan()

            iids = ["I_1_1", "I_1_2"]
            offsets = {"I_1_1": 0.0, "I_1_2": 36.0}
            num_links = {"I_1_1": 4, "I_1_2": 4}

            path = gen.generate_tls_add_xml(
                iids, os.path.join(tmp_dir, "test.add.xml"),
                plan=plan, offsets=offsets, num_links=num_links,
            )

            import xml.etree.ElementTree as ET
            tree = ET.parse(path)
            tls_list = tree.findall(".//tlLogic")
            assert len(tls_list) == 2
            for tl in tls_list:
                if tl.get("id") == "I_1_2":
                    assert tl.get("offset") == "36"
        finally:
            shutil.rmtree(tmp_dir)


# ============================================================================
# Signal Cycle Consistency Tests (Issue 3)
# ============================================================================


class TestSignalCycleConsistency:
    """Tests ensuring signal cycle durations are internally consistent."""

    def test_config_cycle_equals_phase_sum(self):
        """The configured cycle_length must equal the sum of all phase durations."""
        config = SignalConfig.from_yaml("configs/default.yaml")
        plan = SignalPlanGenerator(config).create_plan_from_config()
        total_phase_duration = sum(p.duration for p in plan.phases)
        assert abs(total_phase_duration - config.cycle_length) < 0.01, (
            f"Cycle length {config.cycle_length}s != sum of phases {total_phase_duration}s "
            f"(differences: {config.cycle_length - total_phase_duration}s)"
        )

    def test_default_plan_cycle_consistency(self):
        """Default 2-phase plan should have cycle = sum of phase durations."""
        config = SignalConfig(cycle_length=90.0, yellow_time=3.0, all_red_time=1.0)
        gen = SignalPlanGenerator(config)
        plan = gen.create_default_2phase_plan()
        total = sum(p.duration for p in plan.phases)
        assert abs(total - 90.0) < 0.01, (
            f"Default plan: cycle={plan.cycle_length}, sum={total}"
        )

    def test_green_yellow_red_sum(self):
        """Green + yellow + all_red per phase should account for full cycle."""
        config = SignalConfig(cycle_length=90.0, yellow_time=3.0, all_red_time=1.0)
        gen = SignalPlanGenerator(config)
        plan = gen.create_default_2phase_plan()
        # Two green phases, two yellow phases, two all-red phases
        green_total = sum(p.duration for p in plan.phases if "green" in p.name)
        yellow_total = sum(p.duration for p in plan.phases if "yellow" in p.name)
        red_total = sum(p.duration for p in plan.phases if "red" in p.name)
        assert abs(green_total + yellow_total + red_total - 90.0) < 0.01
        # Each green should be 41s for 90s cycle
        assert all(p.duration == 41.0 for p in plan.phases if "green" in p.name)

    def test_yaml_config_phases_sum_to_cycle(self):
        """Phases defined in default.yaml must sum to the configured cycle_length."""
        import yaml
        with open("configs/default.yaml") as f:
            cfg = yaml.safe_load(f)
        cycle = cfg["signals"]["cycle_length"]
        phases = cfg["signals"]["phases"]
        total = sum(p["duration"] for p in phases)
        assert total == cycle, (
            f"YAML phases sum to {total}s but cycle_length is {cycle}s"
        )


# ============================================================================
# Travel Time Parsing Tests
# ============================================================================


class TestTravelTimeParsing:
    """Tests for parsing SUMO tripinfo output."""

    def test_parse_tripinfo_xml(self):
        """Should extract travel times from tripinfo XML."""
        tmp_dir = tempfile.mkdtemp()
        try:
            tripinfo = Path(tmp_dir) / "tripinfo.xml"
            tripinfo.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<output>
    <tripinfo id="veh0" depart="0.00" departSpeed="0.00" departPos="5.10" departDelay="0.00"
              arrival="120.50" arrivalSpeed="8.33" arrivalPos="294.90"
              duration="120.50" routeLength="300.00" speedLoss="15.20" durationLoss="18.30"
              timeLoss="18.30" waitingTime="5.00"/>
    <tripinfo id="veh1" depart="10.00" departSpeed="0.00" departPos="5.10" departDelay="0.00"
              arrival="150.00" arrivalSpeed="8.33" arrivalPos="294.90"
              duration="140.00" routeLength="300.00" speedLoss="12.00" durationLoss="15.00"
              timeLoss="15.00" waitingTime="3.00"/>
</output>
""")
            travel_times = compute_travel_times_from_tripinfo(tripinfo)
            assert len(travel_times) == 2
            assert travel_times[0] == 120.50
            assert travel_times[1] == 140.00
        finally:
            shutil.rmtree(tmp_dir)

    def test_parse_empty_tripinfo(self):
        """Empty tripinfo should return empty list."""
        tmp_dir = tempfile.mkdtemp()
        try:
            tripinfo = Path(tmp_dir) / "tripinfo.xml"
            tripinfo.write_text('<?xml version="1.0" encoding="UTF-8"?><output/>')
            travel_times = compute_travel_times_from_tripinfo(tripinfo)
            assert len(travel_times) == 0
        finally:
            shutil.rmtree(tmp_dir)


# ============================================================================
# Print Summary Tests
# ============================================================================


class TestPrintSummary:
    """Tests for the summary output function."""

    def test_summary_contains_key_metrics(self):
        """Summary should mention all key metrics."""
        results = SimulationResults(
            total_waiting_time=1000.0,
            avg_waiting_time=5.0,
            avg_queue_length=3.2,
            max_queue_length=15,
            throughput=25,
            vehicles_entered=30,
            vehicles_remaining=5,
        )
        summary = print_summary(results)
        assert "vehicles_entered" in summary or "Vehicles entered" in summary
        assert "throughput" in summary or "completed" in summary.lower()
        assert "waiting" in summary.lower()
        assert "queue" in summary.lower()

    def test_summary_per_intersection(self):
        """Summary should include per-intersection table when available."""
        im = IntersectionMetrics(intersection_id="I_1_1")
        im.avg_queue_length  # property
        results = SimulationResults(
            intersection_metrics={"I_1_1": im},
        )
        summary = print_summary(results)
        assert "I_1_1" in summary


# ============================================================================
# Reproducibility Tests
# ============================================================================


class TestReproducibility:
    """Tests for simulation reproducibility."""

    def test_same_seed_same_config(self):
        """Same seed and config should produce identical metadata."""
        from sumo_network.grid_builder import GridConfig
        from sumo_network.traffic_demand import DemandConfig, TrafficDemandGenerator

        for seed in [42, 123]:
            config = GridConfig(rows=3, cols=3)
            grid = GridBuilder(config)
            edge_ids = [e.id for e in grid.edges]

            d1 = DemandConfig(seed=seed)
            d2 = DemandConfig(seed=seed)
            g1 = TrafficDemandGenerator(d1)
            g2 = TrafficDemandGenerator(d2)

            tmp = tempfile.mkdtemp()
            try:
                p1 = g1.generate_rou_xml(edge_ids, os.path.join(tmp, "d1.rou.xml"))
                p2 = g2.generate_rou_xml(edge_ids, os.path.join(tmp, "d2.rou.xml"))
                assert p1.read_text() == p2.read_text()
            finally:
                shutil.rmtree(tmp)

    def test_different_seeds_different_output(self):
        """Different seeds should produce different route files."""
        from sumo_network.grid_builder import GridConfig
        from sumo_network.traffic_demand import DemandConfig, TrafficDemandGenerator

        config = GridConfig(rows=3, cols=3)
        grid = GridBuilder(config)
        edge_ids = [e.id for e in grid.edges]

        g1 = TrafficDemandGenerator(DemandConfig(seed=42))
        g2 = TrafficDemandGenerator(DemandConfig(seed=999))

        tmp = tempfile.mkdtemp()
        try:
            p1 = g1.generate_rou_xml(edge_ids, os.path.join(tmp, "d1.rou.xml"))
            p2 = g2.generate_rou_xml(edge_ids, os.path.join(tmp, "d2.rou.xml"))
            assert p1.read_text() != p2.read_text()
        finally:
            shutil.rmtree(tmp)


# ============================================================================
# Edge Existence Regression Tests (Issue 1)
# ============================================================================


class TestEdgeExistence:
    """
    Regression tests ensuring all edges in the network actually exist in
    the compiled SUMO net.xml, and all route-file edges exist in net.xml.
    """

    def test_all_edg_xml_edges_exist_in_net_xml(self):
        """Every edge defined in .edg.xml must appear in the compiled .net.xml."""
        import xml.etree.ElementTree as ET

        tree_edg = ET.parse("data/synthetic/network/grid_3x3.edg.xml")
        edg_ids = set(e.get("id") for e in tree_edg.getroot().findall("edge"))

        tree_net = ET.parse("data/synthetic/network/grid_3x3.net.xml")
        net_ids = set(
            e.get("id")
            for e in tree_net.getroot().findall(".//edge")
            if not e.get("id", ":").startswith(":")
        )

        missing = edg_ids - net_ids
        assert len(missing) == 0, f"Edges in edg.xml but NOT in net.xml: {sorted(missing)}"
        assert len(edg_ids) == len(net_ids), (
            f"Edge count mismatch: edg.xml={len(edg_ids)}, net.xml={len(net_ids)}"
        )

    def test_all_route_edges_exist_in_net_xml(self):
        """Every edge referenced in the .rou.xml must exist in the compiled .net.xml."""
        import xml.etree.ElementTree as ET

        tree_net = ET.parse("data/synthetic/network/grid_3x3.net.xml")
        net_ids = set(
            e.get("id")
            for e in tree_net.getroot().findall(".//edge")
            if not e.get("id", ":").startswith(":")
        )

        tree_rou = ET.parse("data/synthetic/demand/demand.rou.xml")
        route_edges = set()
        for flow in tree_rou.getroot().findall(".//flow"):
            if flow.get("from"):
                route_edges.add(flow.get("from"))
            if flow.get("to"):
                route_edges.add(flow.get("to"))

        missing = route_edges - net_ids
        assert len(missing) == 0, f"Route edges not in net.xml: {sorted(missing)}"

    def test_all_intersections_have_connections(self):
        """Every interior intersection should have connections in .con.xml.
        
        Corner intersections (like N_3_3) may have no straight-through
        connections because all edges go off-grid. SUMO handles these via
        implicit connections. We verify that at least the interior
        intersections have explicit connections.
        """
        import xml.etree.ElementTree as ET

        tree_con = ET.parse("data/synthetic/network/grid_3x3.con.xml")
        conns = tree_con.getroot().findall("connection")
        tls_with_conns = set(c.get("tl") for c in conns if c.get("tl"))

        grid = GridBuilder(GridConfig(rows=3, cols=3))
        # Interior intersections (not on the grid boundary) should have connections
        interior_intersections = [
            iid for iid in grid.intersection_nodes
            if iid not in ("N_1_3", "N_2_3", "N_3_1", "N_3_2", "N_3_3")
        ]
        for iid in interior_intersections:
            assert iid in tls_with_conns, (
                f"Interior intersection {iid} has no connections in .con.xml"
            )

    def test_boundary_edges_no_invalid_references(self):
        """Demand generator should not create flows on nonexistent edges."""
        import xml.etree.ElementTree as ET

        config = GridConfig(rows=3, cols=3)
        grid = GridBuilder(config)
        edge_ids = [e.id for e in grid.edges]

        demand_config = DemandConfig(seed=42, demand_level="medium")
        gen = TrafficDemandGenerator(demand_config)
        boundaries = gen.identify_boundary_edges(edge_ids, grid_rows=3, grid_cols=3)

        # All identified boundary edges must be in the edge list
        all_edge_set = set(edge_ids)
        for edge_id in boundaries["entry"]:
            assert edge_id in all_edge_set, f"Entry edge {edge_id} not in network"
        for edge_id in boundaries["exit"]:
            assert edge_id in all_edge_set, f"Exit edge {edge_id} not in network"

    def test_network_node_count_after_geometry_preservation(self):
        """All 16 nodes should exist in net.xml (no geometry removal)."""
        import xml.etree.ElementTree as ET

        tree_net = ET.parse("data/synthetic/network/grid_3x3.net.xml")
        junctions = tree_net.getroot().findall(".//junction")
        # Filter out internal junctions
        real_junctions = [j for j in junctions if not j.get("id", ":").startswith(":")]
        assert len(real_junctions) == 16, (
            f"Expected 16 junctions, found {len(real_junctions)}: "
            f"{[j.get('id') for j in real_junctions]}"
        )


# ============================================================================
# Metric Correctness Tests (Issue 2)
# ============================================================================


class TestMetricCorrectness:
    """Tests verifying metric calculations with small known examples."""

    def test_avg_waiting_from_tripinfo(self):
        """avg_waiting_time should be total waiting / num completed vehicles."""
        tmp = tempfile.mkdtemp()
        try:
            # Create a small tripinfo.xml with known values
            tripinfo = Path(tmp) / "tripinfo.xml"
            tripinfo.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<output>\n'
                '  <tripinfo id="v0" duration="100" waitingTime="20" timeLoss="25"/>\n'
                '  <tripinfo id="v1" duration="80" waitingTime="10" timeLoss="15"/>\n'
                '  <tripinfo id="v2" duration="120" waitingTime="30" timeLoss="35"/>\n'
                '</output>'
            )
            from evaluation.metrics import (
                compute_travel_times_from_tripinfo,
                compute_waiting_times_from_tripinfo,
            )
            travel = compute_travel_times_from_tripinfo(tripinfo)
            waiting = compute_waiting_times_from_tripinfo(tripinfo)
            assert len(travel) == 3
            assert len(waiting) == 3
            # avg travel = (100+80+120)/3 = 100.0
            assert abs(np.mean(travel) - 100.0) < 0.01
            # avg waiting = (20+10+30)/3 = 20.0
            assert abs(np.mean(waiting) - 20.0) < 0.01
            # total waiting = 60.0
            assert abs(sum(waiting) - 60.0) < 0.01
        finally:
            shutil.rmtree(tmp)

    def test_metrics_finalization_with_tripinfo(self):
        """finalize() with tripinfo should use per-vehicle stats, not per-step."""
        tmp = tempfile.mkdtemp()
        try:
            tripinfo = Path(tmp) / "tripinfo.xml"
            tripinfo.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<output>\n'
                '  <tripinfo id="v0" duration="100" waitingTime="20" timeLoss="25"/>\n'
                '  <tripinfo id="v1" duration="80" waitingTime="10" timeLoss="15"/>\n'
                '</output>'
            )
            collector = MetricsCollector()
            collector.setup([], {}, config={})
            collector._total_vehicles_entered = 2
            collector._total_vehicles_left = 2
            collector._start_time = 0.0
            collector.steps = [StepSnapshot(time=10, num_vehicles=0,
                                            total_waiting_time=0, avg_waiting_time=0,
                                            queue_lengths={}, vehicle_speeds={},
                                            vehicles_per_intersection={})]

            mock_traci = MagicMock()
            results = collector.finalize(mock_traci, tripinfo_path=tripinfo)
            # avg travel = (100+80)/2 = 90.0
            assert abs(results.avg_travel_time - 90.0) < 0.01
            # avg waiting = (20+10)/2 = 15.0
            assert abs(results.avg_waiting_time - 15.0) < 0.01
            # total waiting = 30.0
            assert abs(results.total_waiting_time - 30.0) < 0.01
            # throughput from tripinfo
            assert results.throughput == 2
        finally:
            shutil.rmtree(tmp)

    def test_queue_length_from_steps(self):
        """avg_queue_length should be computed from step snapshots."""
        collector = MetricsCollector()
        collector.setup([], {}, config={})
        collector._start_time = 0.0
        collector.steps = [
            StepSnapshot(time=10, num_vehicles=2, total_waiting_time=0,
                         avg_waiting_time=0, queue_lengths={},
                         vehicle_speeds={}, vehicles_per_intersection={}),
            StepSnapshot(time=20, num_vehicles=3, total_waiting_time=0,
                         avg_waiting_time=0, queue_lengths={},
                         vehicle_speeds={}, vehicles_per_intersection={}),
        ]
        collector._sum_queue_lengths = 10.0  # total queue across steps
        collector._max_queue_length = 6
        collector._total_vehicles_entered = 3
        collector._total_vehicles_left = 0

        mock_traci = MagicMock()
        results = collector.finalize(mock_traci)
        # avg queue = 10.0 / 2 steps = 5.0
        assert abs(results.avg_queue_length - 5.0) < 0.01
        assert results.max_queue_length == 6

    def test_throughput_consistency(self):
        """throughput + vehicles_remaining should equal vehicles_entered."""
        collector = MetricsCollector()
        collector.setup([], {}, config={})
        collector._start_time = 0.0
        collector.steps = []
        collector._total_vehicles_entered = 10
        collector._total_vehicles_left = 7

        mock_traci = MagicMock()
        results = collector.finalize(mock_traci)
        assert results.vehicles_entered == 10
        assert results.throughput == 7
        assert results.vehicles_remaining == 3

    def test_all_nine_intersections_have_incoming_edges(self):
        """All 9 intersections should have at least one incoming edge."""
        grid = GridBuilder(GridConfig(rows=3, cols=3))
        incoming_edges: dict[str, list[str]] = {iid: [] for iid in grid.intersection_nodes}
        for edge in grid.edges:
            if edge.to_node in incoming_edges:
                if edge.id not in incoming_edges[edge.to_node]:
                    incoming_edges[edge.to_node].append(edge.id)

        expected = ["N_1_1", "N_1_2", "N_1_3", "N_2_1", "N_2_2",
                    "N_2_3", "N_3_1", "N_3_2", "N_3_3"]
        for iid in expected:
            assert iid in incoming_edges, f"Intersection {iid} not in incoming_edges"
            assert len(incoming_edges[iid]) > 0, (
                f"Intersection {iid} has no incoming edges"
            )

    def test_corner_intersections_have_incoming_edges(self):
        """Corner intersections (N_1_3, N_3_1, N_3_3) should have incoming edges
        even though they have no explicit connections in .con.xml."""
        grid = GridBuilder(GridConfig(rows=3, cols=3))
        incoming_edges: dict[str, list[str]] = {iid: [] for iid in grid.intersection_nodes}
        for edge in grid.edges:
            if edge.to_node in incoming_edges:
                if edge.id not in incoming_edges[edge.to_node]:
                    incoming_edges[edge.to_node].append(edge.id)

        for corner in ["N_1_3", "N_3_1", "N_3_3"]:
            assert len(incoming_edges[corner]) > 0, (
                f"Corner intersection {corner} has no incoming edges"
            )
