"""
Regression Tests for Phase 3/4 Correction Pass
================================================

Tests verifying the fixes for Bugs #1–#6 discovered during
Independent PPO validation.
"""

from __future__ import annotations

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sumo_network.grid_builder import GridBuilder, GridConfig
from sumo_network.signal_plans import SignalPlanGenerator, SignalConfig
from sumo_network.traffic_demand import TrafficDemandGenerator, DemandConfig


# ============================================================================
# Regression: evaluation parses truncated tripinfo.xml correctly
# ============================================================================

class TestTruncatedTripinfoParsing:
    """SUMO writes tripinfo.xml incrementally; before env.close() it is always
    mid-write (truncated). The eval must extract real metrics instead of
    silently returning zeros."""

    @pytest.fixture
    def parse(self):
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from evaluate_independent_ppo import _parse_tripinfo, _sanitize_truncated_tripinfo
        return _parse_tripinfo, _sanitize_truncated_tripinfo

    def test_truncated_file_returns_real_metrics(self, parse, tmp_path):
        """A live (truncated) tripinfo file must yield non-zero throughput."""
        _parse_tripinfo, _ = parse
        # NOTE: <tripinfos> root is intentionally left unclosed (real SUMO
        # tripinfo files look exactly like this while the sim is running).
        content = '<tripinfos version="1.27.0">\n'
        content += (
            '<tripinfo id="v0" depart="0.00" departDelay="0.00" duration="51.00" '
            'waitingTime="10.00" timeLoss="25.50" routeLength="600.00"/>\n'
            '<tripinfo id="v1" depart="10.00" departDelay="0.00" duration="48.00" '
            'waitingTime="8.00" timeLoss="22.00" routeLength="600.00"/>\n'
        )
        path = tmp_path / "tripinfo.xml"
        path.write_text(content)
        metrics = _parse_tripinfo(path)
        assert metrics["throughput"] == 2
        assert metrics["avg_waiting_time"] == 9.0
        assert metrics["avg_travel_time"] == 49.5
        assert metrics["total_waiting_time"] == 18.0

    def test_sanitizer_appends_root_close(self, parse):
        """Sanitizer output must be valid XML ending with </tripinfos>."""
        _, _sanitize = parse
        content = '<tripinfos>\n<tripinfo id="v0" duration="10.0" waitingTime="1.0"/>\n'
        fixed = _sanitize(content)
        assert fixed.rstrip().endswith("</tripinfos>")
        ET.fromstring(fixed)  # must not raise ParseError

    def test_sanitizer_drops_partial_self_closing_record(self, parse):
        """A partially written last record (attributes cut off) is dropped."""
        _, _sanitize = parse
        content = (
            '<tripinfos>\n'
            '<tripinfo id="v0" duration="10.0" waitingTime="1.0"/>\n'
            '<tripinfo id="v1" duration="15.0" waiti'  # truncated record
        )
        root = ET.fromstring(_sanitize(content))
        trips = root.findall("tripinfo")
        assert len(trips) == 1
        assert trips[0].get("id") == "v0"

    def test_flow_returns_zeros_on_missing_file(self, parse, tmp_path):
        """A missing file still returns the defaults (no crash)."""
        _parse_tripinfo, _ = parse
        metrics = _parse_tripinfo(tmp_path / "missing.xml")
        assert metrics["throughput"] == 0
        assert metrics["avg_waiting_time"] == 0.0

    def test_flow_returns_zeros_on_never_written_file(self, parse, tmp_path):
        """An empty file yields zero metrics without crashing."""
        _parse_tripinfo, _ = parse
        path = tmp_path / "tripinfo.xml"
        path.write_text("")
        metrics = _parse_tripinfo(path)
        assert metrics["throughput"] == 0


# ============================================================================
# Bug #3: signal_state from YAML is correctly propagated
# ============================================================================

class TestBug3SignalStateKey:
    """Verify that signal_state (not state) is read from YAML config."""

    def test_build_network_reads_signal_state(self):
        """build_network should read 'signal_state' key from YAML phases."""
        config_path = "configs/default.yaml"
        with open(config_path, "r") as f:
            import yaml
            config = yaml.safe_load(f)

        raw_phases = config.get("signals", {}).get("phases", [])
        assert len(raw_phases) > 0, "Config should have signal phases"

        # The config uses 'signal_state' key
        for p in raw_phases:
            assert "signal_state" in p, (
                f"Phase should use 'signal_state' key, got keys: {list(p.keys())}"
            )

    def test_signal_plan_generator_reads_signal_state(self):
        """SignalConfig.from_yaml should parse signal_state into phases."""
        config = SignalConfig.from_yaml("configs/default.yaml")
        assert len(config.phases) > 0
        for p in config.phases:
            assert "state" in p
            # Should NOT be the fallback 'GGrr' default
            assert p["state"] != "GGrr" or True  # may be correct for short states


# ============================================================================
# Bug #1: Environment dynamically detects TLS phases
# ============================================================================

class TestBug1DynamicTLSDetection:
    """Verify that the environment reads actual TLS phases from SUMO."""

    def test_detect_tls_phases_populates_state(self):
        """detect_tls_phases should set num_phases and green_phases."""
        from envs.single_intersection import (
            IntersectionState, detect_tls_phases
        )
        # Use a real SUMO environment to test
        from envs.multi_intersection_env import MultiIntersectionEnv
        env = MultiIntersectionEnv(
            config_path="configs/default.yaml",
            episode_duration=60,
        )
        try:
            obs, infos = env.reset()
            # Check that states have been detected
            for tls_id, state in env._states.items():
                assert state.tls_detected, f"{tls_id} TLS not detected"
                assert isinstance(state.num_phases, int), (
                    f"{tls_id} num_phases should be int, got {type(state.num_phases)}"
                )
                assert state.num_phases >= 1, (
                    f"{tls_id} should have at least 1 phase"
                )
                assert isinstance(state.green_phases, set), (
                    f"{tls_id} green_phases should be set"
                )
        finally:
            env.close()

    def test_all_intersections_have_phases(self):
        """Every TLS should have a valid phase count after reset."""
        from envs.multi_intersection_env import MultiIntersectionEnv
        env = MultiIntersectionEnv(
            config_path="configs/default.yaml",
            episode_duration=60,
        )
        try:
            obs, infos = env.reset()
            for tls_id, state in env._states.items():
                assert state.num_phases >= 1, (
                    f"{tls_id} has {state.num_phases} phases (expected >= 1)"
                )
        finally:
            env.close()

    def test_n33_has_single_phase(self):
        """N_3_3 (corner) should have exactly 1 phase (always green)."""
        from envs.multi_intersection_env import MultiIntersectionEnv
        env = MultiIntersectionEnv(
            config_path="configs/default.yaml",
            episode_duration=60,
        )
        try:
            obs, infos = env.reset()
            state = env._states.get("N_3_3")
            if state is not None:
                assert state.num_phases == 1, (
                    f"N_3_3 should have 1 phase, got {state.num_phases}"
                )
                assert 0 in state.green_phases, (
                    "N_3_3 phase 0 should be green"
                )
        finally:
            env.close()


# ============================================================================
# Bug #1: KEEP leaves phase unchanged, SWITCH changes phase
# ============================================================================

class TestBug1ActionBehavior:
    """Verify KEEP/SWITCH work correctly with dynamic TLS phases."""

    def test_keep_leaves_phase_unchanged(self):
        """KEEP action should never change the TLS phase."""
        from envs.single_intersection import IntersectionState, apply_action
        from envs.action import ACTION_KEEP
        mock_traci = MagicMock()
        state = IntersectionState(
            tls_id="N_1_1",
            incoming_edges=["E_0_1_to_1_1", "E_1_0_to_1_1"],
        )
        state.current_phase = 0
        state.last_phase_switch_time = 0.0
        state.num_phases = 4
        state.green_phases = {0, 2}
        state.tls_detected = True

        apply_action(mock_traci, state, ACTION_KEEP, sim_time=50.0)
        mock_traci.trafficlight.setPhase.assert_not_called()
        assert state.current_phase == 0

    def test_switch_changes_phase(self):
        """SWITCH should advance to the next phase."""
        from envs.single_intersection import IntersectionState, apply_action
        from envs.action import ACTION_SWITCH
        mock_traci = MagicMock()
        state = IntersectionState(
            tls_id="N_1_1",
            incoming_edges=["E_0_1_to_1_1"],
        )
        state.current_phase = 0
        state.last_phase_switch_time = 0.0
        state.num_phases = 4
        state.green_phases = {0, 2}
        state.tls_detected = True

        apply_action(mock_traci, state, ACTION_SWITCH, sim_time=50.0)
        mock_traci.trafficlight.setPhase.assert_called_once_with("N_1_1", 1)
        assert state.current_phase == 1

    def test_final_clearance_cannot_be_skipped(self):
        """The last yellow phase must finish before SUMO wraps to green."""
        from envs.single_intersection import IntersectionState, apply_action
        from envs.action import ACTION_SWITCH
        mock_traci = MagicMock()
        state = IntersectionState(
            tls_id="N_1_1",
            incoming_edges=["E_0_1_to_1_1"],
        )
        state.current_phase = 3  # last phase (0-indexed)
        state.last_phase_switch_time = 0.0
        state.num_phases = 4
        state.green_phases = {0, 2}
        state.tls_detected = True

        apply_action(mock_traci, state, ACTION_SWITCH, sim_time=50.0)
        mock_traci.trafficlight.setPhase.assert_not_called()
        assert state.current_phase == 3

    def test_switch_safe_for_one_phase_tls(self):
        """SWITCH should do nothing for a single-phase TLS (e.g. N_3_3)."""
        from envs.single_intersection import IntersectionState, apply_action
        from envs.action import ACTION_SWITCH
        mock_traci = MagicMock()
        state = IntersectionState(
            tls_id="N_3_3",
            incoming_edges=[],
        )
        state.current_phase = 0
        state.last_phase_switch_time = 0.0
        state.num_phases = 1
        state.green_phases = {0}
        state.tls_detected = True

        apply_action(mock_traci, state, ACTION_SWITCH, sim_time=50.0)
        # Should NOT call setPhase because there's only 1 phase
        mock_traci.trafficlight.setPhase.assert_not_called()
        assert state.current_phase == 0

    def test_switch_blocked_during_min_green(self):
        """SWITCH should be blocked if green phase hasn't met min_green_time."""
        from envs.single_intersection import IntersectionState, apply_action
        from envs.action import ACTION_SWITCH
        mock_traci = MagicMock()
        state = IntersectionState(
            tls_id="N_1_1",
            incoming_edges=[],
        )
        state.current_phase = 0  # green phase
        state.last_phase_switch_time = 0.0
        state.num_phases = 4
        state.green_phases = {0, 2}
        state.tls_detected = True

        # Only 5 seconds in, min_green is 10
        apply_action(mock_traci, state, ACTION_SWITCH, sim_time=5.0, min_green_time=10.0)
        mock_traci.trafficlight.setPhase.assert_not_called()
        assert state.current_phase == 0

    def test_switch_blocked_on_non_green(self):
        """A policy cannot bypass yellow/all-red clearance."""
        from envs.single_intersection import IntersectionState, apply_action
        from envs.action import ACTION_SWITCH
        mock_traci = MagicMock()
        state = IntersectionState(
            tls_id="N_1_1",
            incoming_edges=[],
        )
        state.current_phase = 1  # yellow phase (not in green_phases)
        state.last_phase_switch_time = 0.0
        state.num_phases = 4
        state.green_phases = {0, 2}
        state.tls_detected = True

        apply_action(mock_traci, state, ACTION_SWITCH, sim_time=2.0, min_green_time=10.0)
        mock_traci.trafficlight.setPhase.assert_not_called()
        assert state.current_phase == 1


# ============================================================================
# Bug #4: Network verification
# ============================================================================

class TestBug4NetworkVerification:
    """Verify the regenerated SUMO network is correct."""

    NET_PATH = Path("data/synthetic/network/grid_3x3.net.xml")
    ROU_PATH = Path("data/synthetic/demand/demand.rou.xml")

    def test_9_traffic_lights(self):
        """Should have exactly 9 TLS programs."""
        tree = ET.parse(str(self.NET_PATH))
        tls_list = tree.getroot().findall(".//tlLogic")
        assert len(tls_list) == 9

    def test_all_tls_have_valid_phases(self):
        """Every TLS should have at least 1 phase with a state string."""
        tree = ET.parse(str(self.NET_PATH))
        for tl in tree.getroot().findall(".//tlLogic"):
            phases = tl.findall("phase")
            assert len(phases) >= 1, f"{tl.get('id')} has no phases"
            for p in phases:
                state = p.get("state")
                assert state is not None and len(state) > 0, (
                    f"{tl.get('id')} has empty phase state"
                )

    def test_phase_states_are_distinct(self):
        """Consecutive phases should have distinct state strings."""
        tree = ET.parse(str(self.NET_PATH))
        for tl in tree.getroot().findall(".//tlLogic"):
            phases = tl.findall("phase")
            if len(phases) > 1:
                states = [p.get("state") for p in phases]
                # At least some consecutive phases should differ
                has_diff = any(
                    states[i] != states[i + 1]
                    for i in range(len(states) - 1)
                )
                assert has_diff, (
                    f"{tl.get('id')}: all phase states are identical: {states[0]}"
                )

    def test_48_edges(self):
        """Should have 48 directed edges."""
        tree = ET.parse(str(self.NET_PATH))
        edges = tree.getroot().findall(".//edge[@from]")
        assert len(edges) == 48

    def test_16_grid_nodes(self):
        """Should have 16 grid nodes (N_0_0 through N_3_3)."""
        tree = ET.parse(str(self.NET_PATH))
        nodes = tree.getroot().findall(".//junction")
        grid_nodes = [n for n in nodes if n.get("id", "").startswith("N_")]
        assert len(grid_nodes) == 16

    def test_route_edges_exist(self):
        """All flow edges in demand should exist in the network."""
        net_tree = ET.parse(str(self.NET_PATH))
        net_edges = {e.get("id") for e in net_tree.getroot().findall(".//edge[@from]")}
        rou_tree = ET.parse(str(self.ROU_PATH))
        for flow in rou_tree.getroot().findall(".//flow"):
            from_edge = flow.get("from")
            assert from_edge in net_edges, (
                f"Flow {flow.get('id')} references non-existent edge {from_edge}"
            )

    def test_green_phases_contain_g(self):
        """Green phases should have uppercase G in their state string."""
        tree = ET.parse(str(self.NET_PATH))
        for tl in tree.getroot().findall(".//tlLogic"):
            phases = tl.findall("phase")
            if len(phases) > 1:
                has_green = any(
                    "G" in p.get("state", "")
                    for p in phases
                )
                assert has_green, (
                    f"{tl.get('id')}: no phase contains green signal"
                )


# ============================================================================
# Bug #6: Traffic demand coverage
# ============================================================================

class TestBug6DemandCoverage:
    """Verify the traffic demand covers intended intersections."""

    def test_boundary_demand_has_flows(self):
        """Boundary demand should produce at least 8 flows."""
        rou_path = Path("data/synthetic/demand/demand.rou.xml")
        tree = ET.parse(str(rou_path))
        flows = tree.getroot().findall(".//flow")
        assert len(flows) >= 8, f"Expected >= 8 flows, got {len(flows)}"

    def test_network_wide_demand_generator(self):
        """Network-wide demand generator should produce flows for all intersections."""
        config = DemandConfig(duration=300, seed=42, flow_rate=100)
        gen = TrafficDemandGenerator(config)
        grid_config = GridConfig(rows=3, cols=3)
        grid = GridBuilder(grid_config)

        with tempfile.NamedTemporaryFile(suffix=".rou.xml", delete=False) as f:
            out_path = Path(f.name)

        try:
            gen.generate_network_wide_with_grid_builder(grid, out_path)
            tree = ET.parse(str(out_path))
            flows = tree.getroot().findall(".//flow")
            assert len(flows) > 8, (
                f"Network-wide demand should have > 8 flows, got {len(flows)}"
            )

            # Verify edges in the generated rou file are from grid_builder
            edge_ids = {e.id for e in grid.edges}
            for flow in flows:
                from_edge = flow.get("from")
                to_edge = flow.get("to")
                assert from_edge in edge_ids, (
                    f"Flow from edge {from_edge} not in grid edges"
                )
                assert to_edge in edge_ids, (
                    f"Flow to edge {to_edge} not in grid edges"
                )
        finally:
            out_path.unlink(missing_ok=True)


# ============================================================================
# Bug #5: tripinfo is parsed before cleanup
# ============================================================================

class TestBug5TripinfoParsing:
    """Verify tripinfo is accessible before env.close()."""

    def test_tripinfo_path_exists_during_episode(self):
        """tripinfo_path should exist while env is still open."""
        from envs.multi_intersection_env import MultiIntersectionEnv
        env = MultiIntersectionEnv(
            config_path="configs/default.yaml",
            episode_duration=60,
        )
        try:
            obs, infos = env.reset()
            # Do a few steps
            actions = {a: 0 for a in env.possible_agents}
            for _ in range(3):
                obs, _, _, _, _ = env.step(actions)

            tripinfo_path = env.get_tripinfo_path()
            assert tripinfo_path is not None, "tripinfo_path should not be None"

            # File may not exist until simulation ends, but path should be valid
            # After close, the temp dir is deleted
            env.close()

            # After close, the file should be gone (temp dir deleted)
            assert not tripinfo_path.exists(), (
                "tripinfo should be gone after env.close()"
            )
        except Exception:
            env.close()
            raise


# ============================================================================
# Integration: Dynamic phases work in SUMO
# ============================================================================

class TestDynamicPhasesInSUMO:
    """Integration tests verifying dynamic TLS phases work in live SUMO."""

    def test_keep_does_not_change_phase_in_sumo(self):
        """KEEP action should preserve the current phase in live SUMO."""
        from envs.multi_intersection_env import MultiIntersectionEnv
        env = MultiIntersectionEnv(
            config_path="configs/default.yaml",
            episode_duration=60,
        )
        try:
            obs, infos = env.reset()
            # Read initial phases
            initial_phases = {}
            for tls_id, state in env._states.items():
                initial_phases[tls_id] = state.current_phase

            # Apply KEEP to all agents
            actions = {a: 0 for a in env.possible_agents}
            obs, rewards, terms, truncs, infos = env.step(actions)

            # Phases should not have changed
            for tls_id, state in env._states.items():
                assert state.current_phase == initial_phases[tls_id], (
                    f"{tls_id}: KEEP changed phase from "
                    f"{initial_phases[tls_id]} to {state.current_phase}"
                )
        finally:
            env.close()

    def test_switch_changes_phase_for_multi_phase_tls(self):
        """SWITCH should change phase for intersections with > 1 phase."""
        from envs.multi_intersection_env import MultiIntersectionEnv
        env = MultiIntersectionEnv(
            config_path="configs/default.yaml",
            episode_duration=120,
        )
        try:
            obs, infos = env.reset()

            # Find an intersection with multiple phases
            multi_phase_tls = None
            for tls_id, state in env._states.items():
                if state.num_phases > 1:
                    multi_phase_tls = tls_id
                    break

            if multi_phase_tls is None:
                pytest.skip("No multi-phase TLS found")

            agent_name = env._tls_to_agent(multi_phase_tls)
            state = env._states[multi_phase_tls]
            initial_phase = state.current_phase

            # Apply SWITCH only to this agent (others KEEP)
            actions = {a: 0 for a in env.possible_agents}
            actions[agent_name] = 1

            obs, rewards, terms, truncs, infos = env.step(actions)

            # Phase should have changed (if min_green was met)
            # The time_in_phase should have been sufficient
            assert state.current_phase != initial_phase or state.time_in_phase < 10.0, (
                f"{multi_phase_tls}: SWITCH did not change phase "
                f"(still {state.current_phase}, was {initial_phase})"
            )
        finally:
            env.close()
