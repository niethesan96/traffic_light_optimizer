"""
Tests for Stage 1 training logging
===================================

Verifies:

  * ``TrainingLogger`` writes one CSV row per update with an immediate flush.
  * ``write_json()`` rewrites training_logs.json (including prior history).
  * Header conflicts rotate the old CSV to ``*.bak`` instead of deleting it.
  * ``compute_gae_and_update`` reports critic ``explained_variance``.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.independent_ppo.trainer import (
    IndependentPPOTrainer,
    TrainingLogger,
    _explained_variance,
)
from agents.independent_ppo.ppo import PPOConfig


AGENT_NAMES = [f"agent_{r}_{c}" for r in (1, 2) for c in (1, 2)]


class TestTrainingLogger:
    """Unit tests for agents/independent_ppo/trainer.py::TrainingLogger."""

    def test_csv_row_per_log_with_flush(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        logger.log({"global_step": 360, "value_loss": 10.0, "per_agent_reward": {"agent_1_1": -1.0}})
        logger.log({"global_step": 720, "value_loss": 9.0, "per_agent_reward": {"agent_1_1": -2.0}})
        # CSV content must already be on disk (flush per row) while open.
        rows = list(csv.reader(open(tmp_path / "training_metrics.csv", newline="")))
        assert rows[0] == ["global_step", "value_loss", "per_agent_reward_agent_1_1"]
        assert rows[1] == ["360", "10.0", "-1.0"]
        assert rows[2] == ["720", "9.0", "-2.0"]
        logger.close()

    def test_json_rewrite_keeps_update_and_history(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        logger.log({"update_count": 1, "value_loss": 10.0})
        logger.write_json()
        logger.log({"update_count": 2, "value_loss": 9.0})
        logger.write_json()
        with open(tmp_path / "training_logs.json") as f:
            data = json.load(f)
        assert [row["update_count"] for row in data] == [1, 2]
        logger.close()

    def test_header_conflict_rotates_to_bak(self, tmp_path):
        # Simulate an old run with a different schema.
        old_path = tmp_path / "training_metrics.csv"
        with open(old_path, "w", newline="") as f:
            csv.writer(f).writerow(["old_col"])
            csv.writer(f).writerow([1])

        logger = TrainingLogger(tmp_path)
        logger.log({"global_step": 1, "value_loss": 2.0})
        logger.close()

        # Old data preserved under .bak — nothing deleted.
        bak = tmp_path / "training_metrics.csv.bak"
        assert bak.exists()
        bak_rows = list(csv.reader(open(bak, newline="")))
        assert bak_rows[0] == ["old_col"] and bak_rows[1] == ["1"]

        new_rows = list(csv.reader(open(old_path, newline="")))
        assert new_rows[0] == ["global_step", "value_loss"]
        assert new_rows[1] == ["1", "2.0"]

    def test_same_header_appends_without_rotation(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        logger.log({"a": 1})
        logger.close()

        logger2 = TrainingLogger(tmp_path)
        logger2.log({"a": 2})
        logger2.close()

        rows = list(csv.reader(open(tmp_path / "training_metrics.csv", newline="")))
        assert rows == [["a"], ["1"], ["2"]]
        assert not (tmp_path / "training_metrics.csv.bak").exists()

    def test_json_history_preserved_across_logger_instances(self, tmp_path):
        logger = TrainingLogger(tmp_path)
        logger.log({"update_count": 1})
        logger.write_json()
        logger.close()

        logger2 = TrainingLogger(tmp_path)
        logger2.log({"update_count": 2})
        logger2.write_json()
        with open(tmp_path / "training_logs.json") as f:
            data = json.load(f)
        assert [row["update_count"] for row in data] == [1, 2]
        logger2.close()


class TestCriticExplainedVariance:
    """compute_gae_and_update must report explained_variance per agent."""

    def _make_trainer(self, rollout_steps=6) -> IndependentPPOTrainer:
        config = PPOConfig(
            update_epochs=1,
            target_kl=None,
            hidden_size=16,
            minibatch_size=4,
        )
        return IndependentPPOTrainer(
            agent_names=AGENT_NAMES,
            obs_dim=18,
            num_actions=2,
            config=config,
            rollout_steps=rollout_steps,
        )

    def _run_one_rollout(self, trainer) -> None:
        rng = np.random.RandomState(0)
        for _ in range(trainer.rollout_steps):
            obs = {
                name: rng.rand(18).astype(np.float32) * 10 for name in AGENT_NAMES
            }
            actions = trainer.get_actions(obs)
            rewards = {name: float(rng.randn()) for name in AGENT_NAMES}
            dones = {name: False for name in AGENT_NAMES}
            trainer.store_transition(obs, actions, rewards, dones)

    def test_losses_carry_explained_variance(self):
        trainer = self._make_trainer()
        self._run_one_rollout(trainer)
        obs = {name: np.zeros(18, dtype=np.float32) for name in AGENT_NAMES}
        losses = trainer.compute_gae_and_update(obs)
        assert len(losses) == len(AGENT_NAMES)
        for name, loss in losses.items():
            assert hasattr(loss, "explained_variance")
            assert not math.isnan(loss.explained_variance)
            assert loss.explained_variance <= 1.0

    def test_explained_variance_helper(self):
        # Perfect predictions → EV = 1
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        assert _explained_variance(y_true, y_true.copy()) == pytest.approx(1.0)
        # Noisy predictions → EV in (0, 1)
        y_pred = y_true + np.array([0.5, -0.5, 0.5, -0.5])
        ev = _explained_variance(y_true, y_pred)
        assert 0.0 < ev < 1.0
        # Zero-variance returns → NaN (no signal), not a fake 1.0
        assert math.isnan(_explained_variance(np.zeros(5), np.zeros(5)))
