"""
Tests for Stage 3 corrections
==============================

Covers:

  * Truncation bootstrapping in the PPO buffer (hand-computed GAE numbers).
  * Reward scaling / clipping (``RewardConfig.reward_scale`` / ``reward_clip``).
  * Fixed observation scaling (single place, constants, clipping, padding).
  * Per-episode demand seeds with cached .rou.xml files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.independent_ppo.buffer import RolloutBuffer


# ============================================================================
# Truncation bootstrapping (hand-computed example)
# ============================================================================

class TestTruncationBootstrap:
    """On truncation, gamma * V(s_final) is added to the last reward while
    ``done`` still cuts the GAE recursion. gamma=0.9, values V=[2,3,4],
    next value V(s_final)=5."""

    GAMMA = 0.9
    VALUES = [2.0, 3.0, 4.0]
    REWARDS = [1.0, 0.0, 2.0]
    LAST_VALUE = 5.0

    def _buffer(self, terminated, truncated, done):
        buf = RolloutBuffer(
            buffer_size=3, obs_dim=4, gamma=self.GAMMA, gae_lambda=0.95,
        )
        obs = np.arange(4, dtype=np.float32)
        for t in range(3):
            buf.add(
                obs=obs, action=0,
                reward=self.REWARDS[t],
                value=self.VALUES[t],
                log_prob=-0.5,
                done=done[t],
                terminated=terminated[t],
                truncated=truncated[t],
            )
            # Overwrite obs with distinct rows so batches differ
            buf.observations[t] = t
        return buf

    def test_truncated_final_step_bootstrapped(self):
        buf = self._buffer(
            terminated=[False, False, False],
            truncated=[False, False, True],
            done=[False, False, True],
        )
        buf.compute_gae(self.LAST_VALUE)

        # rewards[2] += 0.9 * 5.0 = 4.5 → 6.5
        # A2 = 6.5 - 4 = 2.5 (recursion cut by done)
        # A1 = (0 + 0.9*4 - 3) + 0.9*0.95*2.5 = 0.6 + 2.1375 = 2.7375
        # A0 = (1 + 0.9*3 - 2) + 0.9*0.95*2.7375 = 1.7 + 2.34056 = 4.0405625
        assert buf.rewards[2] == pytest.approx(6.5)
        assert buf.advantages[2] == pytest.approx(2.5)
        assert buf.advantages[1] == pytest.approx(2.7375)
        assert buf.advantages[0] == pytest.approx(4.0405625)
        # returns = A + V
        assert buf.returns[2] == pytest.approx(6.5)
        assert buf.returns[1] == pytest.approx(5.7375)
        assert buf.returns[0] == pytest.approx(6.0405625)

    def test_terminated_final_step_not_bootstrapped(self):
        buf = self._buffer(
            terminated=[False, False, True],
            truncated=[False, False, False],
            done=[False, False, True],
        )
        buf.compute_gae(self.LAST_VALUE)

        # No bootstrap: A2 = 2 - 4 = -2
        # A1 = 0.6 + 0.9*0.95*(-2) = -1.11
        # A0 = 1.7 + 0.9*0.95*(-1.11) = 0.75095
        assert buf.rewards[2] == pytest.approx(2.0)
        assert buf.advantages[2] == pytest.approx(-2.0)
        assert buf.advantages[1] == pytest.approx(-1.11)
        assert buf.advantages[0] == pytest.approx(0.75095)

    def test_legacy_done_only_behaviour_unchanged(self):
        """Callers passing only ``done`` keep the old terminal semantics."""
        buf = self._buffer(
            terminated=[None, None, None],  # will be replaced; see _buffer_raw
            truncated=[None, None, None],
            done=[False, False, True],
        )
        buf.compute_gae(self.LAST_VALUE)
        # Same numbers as the terminated case
        assert buf.rewards[2] == pytest.approx(2.0)
        assert buf.advantages[0] == pytest.approx(0.75095)

    def test_mid_rollout_bootstrap_untouched(self):
        """A full, non-truncated rollout still bootstraps via last_value."""
        buf = self._buffer(
            terminated=[False, False, False],
            truncated=[False, False, False],
            done=[False, False, False],
        )
        buf.compute_gae(self.LAST_VALUE)
        # t=2: delta = 2 + 0.9*5 - 4 = 2.5; A2 = 2.5 (same as truncated case
        # numerically, but via the standard value-term path)
        assert buf.advantages[2] == pytest.approx(2.5)
        assert buf.returns[2] == pytest.approx(2.5 + 4.0)


# ============================================================================
# Reward scaling / clipping
# ============================================================================

class TestRewardScaleClip:

    def test_scale_and_clip_applied_after_sum(self):
        from envs.reward import RewardConfig, compute_reward
        cfg = RewardConfig(reward_scale=0.02, reward_clip=10.0)
        r = compute_reward(
            prev_waiting_time=100.0, current_waiting_time=80.0,
            prev_queue_length=10, current_queue_length=8,
            config=cfg,
        )
        # raw = 22 → 22 * 0.02 = 0.44, inside ±10, no clipping
        assert r == pytest.approx(0.44)

        r_big = compute_reward(
            prev_waiting_time=1.0, current_waiting_time=2000.0,
            prev_queue_length=0, current_queue_length=0,
            config=cfg,
        )
        # raw = -1999 → -39.98 → clipped to -10
        assert r_big == pytest.approx(-10.0)

        r_big_pos = compute_reward(
            prev_waiting_time=2000.0, current_waiting_time=1.0,
            prev_queue_length=0, current_queue_length=0,
            config=cfg,
        )
        assert r_big_pos == pytest.approx(10.0)

    def test_defaults_keep_legacy_behaviour(self):
        from envs.reward import RewardConfig, compute_reward
        r = compute_reward(
            prev_waiting_time=100.0, current_waiting_time=80.0,
            prev_queue_length=10, current_queue_length=8,
            config=RewardConfig(),
        )
        assert r == pytest.approx(22.0)

    def test_reward_from_config_parses_scale_clip(self):
        from envs.reward import reward_from_config
        rc = reward_from_config({
            "reward_weights": {
                "reward_scale": 0.02, "reward_clip": 10.0,
                "waiting_weight": 2.0, "queue_weight": 0.5,
            }
        })
        assert rc.reward_scale == pytest.approx(0.02)
        assert rc.reward_clip == pytest.approx(10.0)
        rc2 = reward_from_config({})
        assert rc2.reward_scale == pytest.approx(1.0)
        assert rc2.reward_clip is None

    def test_network_wide_yaml_has_reward_keys(self):
        import yaml
        with open("configs/network_wide.yaml") as f:
            cfg = yaml.safe_load(f)
        rw = cfg["reward_weights"]
        assert rw["reward_scale"] == pytest.approx(0.02)
        assert rw["reward_clip"] == pytest.approx(10.0)


# ============================================================================
# Observation scaling
# ============================================================================

class TestObservationScaling:

    def test_env_returns_scaled_observations(self):
        from envs.multi_intersection_env import MultiIntersectionEnv

        env = MultiIntersectionEnv(
            config_path="configs/default.yaml", episode_duration=60,
        )
        try:
            obs, _ = env.reset()
            for agent, o in obs.items():
                assert o.max() <= 5.0, f"{agent}: obs max {o.max()} > 5"
                assert o.min() >= 0.0
        finally:
            env.close()

    def test_padding_rows_stay_zero(self):
        from envs.observation import scale_observation
        # 2-lane natural length padded later: extra tail uses queue divisor
        scaled = scale_observation([1, 60, 8, 8, 15, 12], num_lanes_total=1)
        assert len(scaled) == 6
        # Additional padded values scale to 0
        padded = scale_observation([1, 60, 8, 8, 15, 12, 0], num_lanes_total=1)
        assert padded[-1] == pytest.approx(0.0)


# ============================================================================
# Per-episode demand seeds (cached rou files)
# ============================================================================

class TestDemandSeeds:

    @staticmethod
    def _reset_env(seed):
        from envs.multi_intersection_env import MultiIntersectionEnv
        env = MultiIntersectionEnv(
            config_path="configs/network_wide.yaml",
            episode_duration=120,
            demand_seed=seed,
        )
        try:
            env.reset()
            return env
        finally:
            env.close()

    def test_same_seed_same_file(self, tmp_path):
        cache_dir = Path("data/network_wide/demand/cache")
        f1000 = cache_dir / "demand_seed1000_dur120.rou.xml"
        if f1000.exists():
            f1000.unlink()

        self._reset_env(1000)
        assert f1000.exists()
        content_1 = f1000.read_bytes()

        self._reset_env(1000)
        assert f1000.read_bytes() == content_1

    def test_different_seed_different_file(self):
        cache_dir = Path("data/network_wide/demand/cache")
        f1000 = cache_dir / "demand_seed1000_dur120.rou.xml"
        f1001 = cache_dir / "demand_seed1001_dur120.rou.xml"
        if f1001.exists():
            f1001.unlink()

        self._reset_env(1000)
        self._reset_env(1001)
        assert f1000.exists() and f1001.exists()
        assert f1000.read_bytes() != f1001.read_bytes()

    def test_no_demand_seed_uses_static_file(self):
        from envs.multi_intersection_env import MultiIntersectionEnv
        env = MultiIntersectionEnv(
            config_path="configs/network_wide.yaml", episode_duration=120,
        )
        try:
            env.reset()
            # Without demand_seed the static demand file must be selected
            # (assertion: sumocfg was written and points at demand.rou.xml)
            cfg_text = env._sumo_cfg_path.read_text()
            assert "demand.rou.xml" in cfg_text
        finally:
            env.close()
