"""
Tests for Phase 3 — Multi-Agent PettingZoo Environment.

Covers:
  - Environment construction and agent naming
  - Observation space shape and bounds
  - Action space validity
  - Reward function (pure logic, no SUMO)
  - Single-intersection helpers
  - Observation / action module correctness
  - Full SUMO integration (reset, step, close)
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.single_intersection import (
    IntersectionState,
    collect_observation,
    apply_action,
    compute_reward,
    compute_local_waiting_time,
    compute_local_queue_length,
    build_info,
    GREEN_PHASES,
    NUM_PHASES,
    get_observation_dim,
    DEFAULT_MIN_GREEN_TIME,
)
from envs.observation import observation_space, observation_dim, scale_observation
from envs.action import (
    action_space,
    is_valid_action,
    ACTION_KEEP,
    ACTION_SWITCH,
    NUM_ACTIONS,
)
from envs.reward import RewardConfig, compute_reward as reward_compute, reward_from_config
from envs.multi_intersection_env import MultiIntersectionEnv


# ============================================================================
# Observation Module Tests
# ============================================================================

class TestObservationModule:
    """Tests for envs/observation.py."""

    def test_observation_dim_default(self):
        """Default 4 incoming edges × 1 lane → 2 + 4*4 = 18."""
        assert observation_dim() == 18

    def test_observation_dim_custom(self):
        """Custom edges and lanes should scale correctly."""
        assert observation_dim(2, 1) == 10  # 2 + 2*4
        assert observation_dim(4, 2) == 34  # 2 + 8*4

    def test_observation_space_shape(self):
        """Space shape should match observation_dim."""
        space = observation_space()
        assert space.shape == (18,)

    def test_observation_space_dtype(self):
        """Space dtype should be float32."""
        space = observation_space()
        assert space.dtype == np.float32

    def test_observation_space_low_zero(self):
        """Low bounds should be all zeros."""
        space = observation_space()
        np.testing.assert_array_equal(space.low, np.zeros(18))

    def test_observation_space_high_inf(self):
        """High bounds should be inf (unbounded above)."""
        space = observation_space()
        assert np.all(np.isinf(space.high))

    def test_scale_observation_shape(self):
        """Scaled obs should have same length."""
        obs = np.zeros(18, dtype=np.float32)
        scaled = scale_observation(obs)
        assert len(scaled) == obs.shape[0]

    def test_scale_observation_range_andConstants(self):
        """Fixed constants: phase/3, time/60, queue/8, wait/8, count/15,
        speed/12, all clipped to [0, 5]. Layout: [phase, time,
        queue×4, waiting×4, count×4, speed×4]."""
        obs = np.array(
            [3, 120,
             8, 8, 8, 8,          # queue  → ÷8 → 1
             100, 200, 150, 240,  # waiting → clipped at 5
             5, 5, 5, 5,          # count → 5/15
             100, 200, 150, 240], # speed → clipped at 5
            dtype=np.float32,
        )
        scaled = np.array(scale_observation(obs))
        assert scaled[0] == pytest.approx(1.0)   # 3 / 3
        assert scaled[1] == pytest.approx(2.0)   # 120 / 60 (clipped at 5 stays 2)
        np.testing.assert_allclose(scaled[2:6], 1.0)
        np.testing.assert_allclose(scaled[6:10], 5.0)
        np.testing.assert_allclose(scaled[10:14], 5.0 / 15.0)
        np.testing.assert_allclose(scaled[14:18], 5.0)
        assert np.all(scaled >= 0.0) and np.all(scaled <= 5.0)


# ============================================================================
# Action Module Tests
# ============================================================================

class TestActionModule:
    """Tests for envs/action.py."""

    def test_action_space_size(self):
        """Action space should have 2 actions."""
        space = action_space()
        assert space.n == NUM_ACTIONS == 2

    def test_valid_actions(self):
        """Actions 0 and 1 should be valid."""
        assert is_valid_action(ACTION_KEEP)
        assert is_valid_action(ACTION_SWITCH)

    def test_invalid_actions(self):
        """Actions outside {0, 1} should be invalid."""
        assert not is_valid_action(-1)
        assert not is_valid_action(2)
        assert not is_valid_action(100)

    def test_action_descriptions(self):
        """Actions should have readable descriptions."""
        from envs.action import action_to_description
        assert action_to_description(0) == "keep"
        assert action_to_description(1) == "switch"
        assert "invalid" in action_to_description(99)


# ============================================================================
# Reward Module Tests
# ============================================================================

class TestRewardModule:
    """Tests for envs/reward.py."""

    def test_positive_reward_on_improvement(self):
        """Reward should be positive when waiting time decreases."""
        r = reward_compute(
            prev_waiting_time=100.0,
            current_waiting_time=80.0,
            prev_queue_length=10,
            current_queue_length=8,
        )
        # Δwait = 20, Δqueue = 2 → r = 20 + 2 = 22
        assert r == 22.0

    def test_negative_reward_on_worsening(self):
        """Reward should be negative when waiting time increases."""
        r = reward_compute(
            prev_waiting_time=50.0,
            current_waiting_time=70.0,
            prev_queue_length=5,
            current_queue_length=8,
        )
        # Δwait = -20, Δqueue = -3 → r = -23
        assert r == -23.0

    def test_zero_reward_on_no_change(self):
        """Reward should be zero when nothing changes."""
        r = reward_compute(
            prev_waiting_time=50.0,
            current_waiting_time=50.0,
            prev_queue_length=5,
            current_queue_length=5,
        )
        assert r == 0.0

    def test_custom_weights(self):
        """Custom weights should scale the reward components."""
        config = RewardConfig(waiting_weight=2.0, queue_weight=0.5)
        r = reward_compute(
            prev_waiting_time=100.0,
            current_waiting_time=80.0,
            prev_queue_length=10,
            current_queue_length=8,
            config=config,
        )
        # 2.0 * 20 + 0.5 * 2 = 40 + 1 = 41
        assert r == 41.0

    def test_reward_from_config_dict(self):
        """reward_from_config should parse YAML-style dict."""
        cfg = {"reward_weights": {"waiting_weight": 3.0, "queue_weight": 0.2}}
        rc = reward_from_config(cfg)
        assert rc.waiting_weight == 3.0
        assert rc.queue_weight == 0.2

    def test_reward_from_empty_config(self):
        """Missing keys should use defaults."""
        rc = reward_from_config({})
        assert rc.waiting_weight == 1.0
        assert rc.queue_weight == 1.0


# ============================================================================
# Single-Intersection Helper Tests
# ============================================================================

class TestSingleIntersectionHelpers:
    """Tests for envs/single_intersection.py helper functions."""

    def test_intersection_state_creation(self):
        """IntersectionState should initialise with correct fields."""
        state = IntersectionState(
            tls_id="N_1_1",
            incoming_edges=["E_0_1_to_1_1", "E_2_1_to_1_1"],
        )
        assert state.tls_id == "N_1_1"
        assert len(state.incoming_edges) == 2
        assert state.current_phase == 0
        assert state.num_lanes == 1

    def test_get_observation_dim(self):
        """get_observation_dim should match observation_dim."""
        assert get_observation_dim(4, 1) == 18
        assert get_observation_dim(2, 2) == 18  # 2 + (2*2)*4

    def test_collect_observation_with_mock(self):
        """collect_observation should build a correct SCALED vector from mock traci."""
        state = IntersectionState(
            tls_id="N_1_1",
            incoming_edges=["E_0_1_to_1_1", "E_1_0_to_1_1"],
        )
        state.last_phase_switch_time = 5.0

        mock_traci = MagicMock()
        mock_traci.trafficlight.getPhase.return_value = 3
        mock_traci.trafficlight.getNextSwitch.return_value = 46.0
        mock_traci.lane.getLastStepHaltingNumber.return_value = 2
        mock_traci.lane.getLastStepVehicleIDs.return_value = ["v1", "v2"]
        mock_traci.vehicle.getWaitingTime.side_effect = lambda vid: 5.0 if vid == "v1" else 0.0
        mock_traci.vehicle.getSpeed.side_effect = lambda vid: 0.0 if vid == "v1" else 8.0

        obs = collect_observation(mock_traci, state, sim_time=15.0)

        # 2 incoming edges × 1 lane × 4 metrics + 2 header = 10
        expected_dim = 2 + 2 * 1 * 4
        assert len(obs) == expected_dim
        assert obs[0] == pytest.approx(1.0)      # phase 3 / 3 = 1
        assert obs[1] == pytest.approx(10.0 / 60.0)  # time_in_phase 10 / 60
        # queue_lengths: [2, 2] → obs[2], obs[3] = 2/8
        assert obs[2] == pytest.approx(2.0 / 8.0)
        assert obs[3] == pytest.approx(2.0 / 8.0)

    def test_apply_action_keep(self):
        """Action KEEP should not change phase."""
        state = IntersectionState(tls_id="N_1_1", incoming_edges=[])
        state.current_phase = 0
        state.last_phase_switch_time = 0.0

        mock_traci = MagicMock()
        apply_action(mock_traci, state, ACTION_KEEP, sim_time=10.0)
        mock_traci.trafficlight.setPhase.assert_not_called()

    def test_apply_action_switch_after_min_green(self):
        """Action SWITCH should change phase if min green time elapsed."""
        state = IntersectionState(tls_id="N_1_1", incoming_edges=[])
        state.current_phase = 0  # NS_green
        state.last_phase_switch_time = 0.0

        mock_traci = MagicMock()
        apply_action(mock_traci, state, ACTION_SWITCH, sim_time=15.0, min_green_time=10.0)
        mock_traci.trafficlight.setPhase.assert_called_once_with("N_1_1", 1)
        assert state.current_phase == 1

    def test_apply_action_switch_blocked_by_min_green(self):
        """Action SWITCH should be blocked if min green time not elapsed."""
        state = IntersectionState(tls_id="N_1_1", incoming_edges=[])
        state.current_phase = 0  # NS_green
        state.last_phase_switch_time = 5.0

        mock_traci = MagicMock()
        apply_action(mock_traci, state, ACTION_SWITCH, sim_time=10.0, min_green_time=10.0)
        # time_in_phase = 5 < 10 → blocked
        mock_traci.trafficlight.setPhase.assert_not_called()
        assert state.current_phase == 0

    def test_apply_action_switch_on_non_green(self):
        """Yellow/all-red clearance must complete without policy shortcuts."""
        state = IntersectionState(tls_id="N_1_1", incoming_edges=[])
        state.current_phase = 1  # NS_yellow (not green)
        state.last_phase_switch_time = 0.0

        mock_traci = MagicMock()
        apply_action(mock_traci, state, ACTION_SWITCH, sim_time=2.0, min_green_time=10.0)
        mock_traci.trafficlight.setPhase.assert_not_called()
        assert state.current_phase == 1

    def test_compute_reward_local(self):
        """compute_reward should return the difference."""
        state = IntersectionState(tls_id="N_1_1", incoming_edges=[])
        state.prev_waiting_time = 100.0
        state.prev_queue_length = 10

        r = compute_reward(state, current_waiting_time=80.0, current_queue_length=8)
        assert r == 22.0  # (100-80) + (10-8)

    def test_build_info(self):
        """build_info should contain expected keys."""
        state = IntersectionState(tls_id="N_1_1", incoming_edges=[])
        state.current_phase = 0
        state.time_in_phase = 15.0
        state.vehicle_counts = [3, 2]
        state.mean_speeds = [5.0, 8.0]

        info = build_info(state, reward=1.0, current_waiting_time=20.0,
                          current_queue_length=5, sim_time=100.0)
        assert info["intersection_id"] == "N_1_1"
        assert info["current_phase"] == 0
        assert info["local_queue_length"] == 5
        assert info["local_waiting_time"] == 20.0
        assert info["local_vehicle_count"] == 5
        assert info["reward"] == 1.0

    def test_green_phases_constant(self):
        """GREEN_PHASES should be {0, 3}."""
        assert GREEN_PHASES == {0, 3}

    def test_num_phases_constant(self):
        """NUM_PHASES should be 6."""
        assert NUM_PHASES == 6


# ============================================================================
# Environment Construction Tests (no SUMO)
# ============================================================================

class TestEnvConstruction:
    """Tests for MultiIntersectionEnv construction (no SUMO needed)."""

    def test_construction(self):
        """Environment should construct from default config."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        assert env is not None
        env.close()

    def test_agent_count(self):
        """Should have exactly 9 agents."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        assert env.num_agents == 9
        assert len(env.possible_agents) == 9
        env.close()

    def test_agent_names(self):
        """Agent names should follow the agent_X_Y pattern."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        expected = [
            "agent_1_1", "agent_1_2", "agent_1_3",
            "agent_2_1", "agent_2_2", "agent_2_3",
            "agent_3_1", "agent_3_2", "agent_3_3",
        ]
        assert sorted(env.possible_agents) == sorted(expected)
        env.close()

    def test_observation_spaces_all_agents(self):
        """All 9 agents should have identical observation spaces."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        shapes = [env.observation_spaces[a].shape for a in env.possible_agents]
        assert all(s == shapes[0] for s in shapes)
        assert shapes[0] == (18,)  # 2 + 4*4
        env.close()

    def test_action_spaces_all_agents(self):
        """All 9 agents should have Discrete(2) action spaces."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        for agent in env.possible_agents:
            assert env.action_spaces[agent].n == 2
        env.close()

    def test_observation_dim(self):
        """get_observation_dim should return 18."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        assert env.get_observation_dim() == 18
        env.close()

    def test_cycle_length(self):
        """Cycle length should be 90 seconds."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        assert env.get_cycle_length() == 90.0
        env.close()

    def test_num_phases(self):
        """Number of phases should be 6."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        assert env.get_num_phases() == 6
        env.close()

    def test_agent_to_tls_mapping(self):
        """Agent names should map correctly to TLS IDs."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        mapping = env.get_agent_intersection_map()
        assert mapping["agent_1_1"] == "N_1_1"
        assert mapping["agent_3_3"] == "N_3_3"
        assert len(mapping) == 9
        env.close()

    def test_incoming_edges_all_agents_have_edges(self):
        """All agents should have at least 1 incoming edge."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        incoming = env.get_incoming_edges()
        for agent, edges in incoming.items():
            assert len(edges) > 0, f"{agent} has no incoming edges"
        env.close()

    def test_render_mode_default(self):
        """Default render mode should be None."""
        env = MultiIntersectionEnv(config_path="configs/default.yaml")
        assert env._render_mode is None
        env.close()


# ============================================================================
# SUMO Integration Tests
# ============================================================================

class TestEnvIntegration:
    """Full SUMO integration tests — reset, step, close."""

    def _make_env(self, **kwargs) -> MultiIntersectionEnv:
        """Create an environment with short episode for testing."""
        defaults = {
            "config_path": "configs/default.yaml",
            "episode_duration": 120,   # 2 minutes
            "step_duration": 10,
            "seed": 42,
        }
        defaults.update(kwargs)
        return MultiIntersectionEnv(**defaults)

    def test_reset_returns_all_agents(self):
        """reset() should return observations and infos for all 9 agents."""
        env = self._make_env()
        try:
            obs, infos = env.reset()
            assert len(obs) == 9
            assert len(infos) == 9
            for agent in env.possible_agents:
                assert agent in obs
                assert agent in infos
        finally:
            env.close()

    def test_reset_observation_shape(self):
        """Each observation should have the correct shape."""
        env = self._make_env()
        try:
            obs, _ = env.reset()
            for agent, o in obs.items():
                assert o.shape == (18,), f"{agent} obs shape {o.shape}"
        finally:
            env.close()

    def test_reset_observation_finite(self):
        """Observations should contain no NaN or inf."""
        env = self._make_env()
        try:
            obs, _ = env.reset()
            for agent, o in obs.items():
                assert np.all(np.isfinite(o)), f"{agent} has non-finite obs"
        finally:
            env.close()

    def test_step_returns_all_agents(self):
        """step() should return results for all 9 agents."""
        env = self._make_env()
        try:
            env.reset()
            actions = {a: 0 for a in env.possible_agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            assert len(obs) == 9
            assert len(rewards) == 9
            assert len(terminations) == 9
            assert len(truncations) == 9
            assert len(infos) == 9
        finally:
            env.close()

    def test_step_observation_shape(self):
        """step() observations should have correct shape."""
        env = self._make_env()
        try:
            env.reset()
            actions = {a: 0 for a in env.possible_agents}
            obs, _, _, _, _ = env.step(actions)
            for agent, o in obs.items():
                assert o.shape == (18,)
        finally:
            env.close()

    def test_step_rewards_are_finite(self):
        """Rewards should be finite floats."""
        env = self._make_env()
        try:
            env.reset()
            actions = {a: 0 for a in env.possible_agents}
            _, rewards, _, _, _ = env.step(actions)
            for agent, r in rewards.items():
                assert np.isfinite(r), f"{agent} reward is not finite: {r}"
        finally:
            env.close()

    def test_episode_terminates(self):
        """Episode should truncate after max_steps."""
        # 120s / 10s = 12 steps; reset does 1, so 11 more steps
        env = self._make_env(episode_duration=120, step_duration=10)
        try:
            env.reset()
            actions = {a: 0 for a in env.possible_agents}
            for i in range(11):
                _, _, _, truncations, _ = env.step(actions)
                if i < 10:
                    assert not any(truncations.values()), f"Truncated too early at step {i}"
            # After 11 steps (12 total including reset), episode should be done
            assert env._episode_done, "Episode should be done after 12 steps"
            # Stepping again should raise RuntimeError
            with pytest.raises(RuntimeError):
                env.step(actions)
        finally:
            env.close()

    def test_step_after_done_raises(self):
        """Stepping after episode end should raise RuntimeError."""
        env = self._make_env(episode_duration=20, step_duration=10)
        try:
            env.reset()
            # 20/10 = 2 steps; reset does 1, so 1 more
            actions = {a: 0 for a in env.possible_agents}
            env.step(actions)  # step 2 → done
            with pytest.raises(RuntimeError):
                env.step(actions)
        finally:
            env.close()

    def test_multiple_resets(self):
        """Environment should support multiple reset() calls."""
        env = self._make_env()
        try:
            for _ in range(3):
                obs, infos = env.reset()
                assert len(obs) == 9
                actions = {a: 0 for a in env.possible_agents}
                env.step(actions)
            # Final close
            env.close()
        except Exception:
            env.close()
            raise

    def test_close_cleans_up(self):
        """close() should clean up SUMO and temp files."""
        env = self._make_env()
        env.reset()
        tmp_dir = env._tmp_dir
        assert tmp_dir is not None
        assert tmp_dir.exists()
        env.close()
        assert env._traci is None
        assert env._tmp_dir is None

    def test_random_actions_run(self):
        """Running with random actions should not crash."""
        env = self._make_env(episode_duration=60, step_duration=10)
        try:
            env.reset()
            rng = np.random.RandomState(42)
            for _ in range(5):
                actions = {
                    a: int(rng.randint(0, 2))
                    for a in env.possible_agents
                }
                obs, rewards, terms, truncs, infos = env.step(actions)
                assert len(obs) == 9
                # Check observations are finite
                for o in obs.values():
                    assert np.all(np.isfinite(o))
        finally:
            env.close()

    def test_step_advances_sim_time(self):
        """Info dicts should show advancing sim_time."""
        env = self._make_env(episode_duration=120, step_duration=10)
        try:
            obs, infos = env.reset()
            t0 = infos["agent_1_1"]["sim_time"]
            actions = {a: 0 for a in env.possible_agents}
            _, _, _, _, infos2 = env.step(actions)
            t1 = infos2["agent_1_1"]["sim_time"]
            assert t1 > t0, f"sim_time did not advance: {t0} → {t1}"
        finally:
            env.close()

    def test_info_contains_expected_keys(self):
        """Info dicts should have all documented keys."""
        env = self._make_env()
        try:
            env.reset()
            actions = {a: 0 for a in env.possible_agents}
            _, _, _, _, infos = env.step(actions)
            expected_keys = {
                "intersection_id", "current_phase", "time_in_phase",
                "local_queue_length", "local_waiting_time",
                "local_vehicle_count", "local_mean_speed",
                "reward", "sim_time",
            }
            for agent, info in infos.items():
                assert expected_keys.issubset(info.keys()), (
                    f"{agent} missing keys: {expected_keys - info.keys()}"
                )
        finally:
            env.close()

    def test_incoming_edges_populated(self):
        """get_incoming_edges should return edges for all agents."""
        env = self._make_env()
        try:
            incoming = env.get_incoming_edges()
            assert len(incoming) == 9
            for agent, edges in incoming.items():
                assert isinstance(edges, list)
                assert len(edges) > 0
        finally:
            env.close()
