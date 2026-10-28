"""
Tests for Stage 4 — checkpoint resume
======================================

``IndependentPPOTrainer.load_checkpoint`` must restore networks, optimizer
state and the global_step / update counters exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.independent_ppo.trainer import IndependentPPOTrainer
from agents.independent_ppo.ppo import PPOConfig

AGENT_NAMES = [f"agent_{r}_{c}" for r in (1, 2) for c in (1, 2)]


def _make_trainer(rollout_steps=4) -> IndependentPPOTrainer:
    config = PPOConfig(
        learning_rate=3e-4,
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


def _run_rollouts(trainer, n_rollouts=2) -> None:
    rng = np.random.RandomState(7)
    for _ in range(n_rollouts):
        for _ in range(trainer.rollout_steps):
            obs = {name: rng.rand(18).astype(np.float32) for name in AGENT_NAMES}
            actions = trainer.get_actions(obs)
            rewards = {name: float(rng.randn()) for name in AGENT_NAMES}
            dones = {name: False for name in AGENT_NAMES}
            trainer.store_transition(obs, actions, rewards, dones)
        terminal_obs = {
            name: rng.rand(18).astype(np.float32) for name in AGENT_NAMES
        }
        trainer.compute_gae_and_update(terminal_obs)


class TestCheckpointResume:

    def _save_and_load(self, tmp_path):
        trainer = _make_trainer()
        _run_rollouts(trainer, n_rollouts=1)
        assert trainer.global_step == trainer.rollout_steps * len(AGENT_NAMES)

        ckpt_dir = tmp_path / "ckpt"
        trainer.save_checkpoint(
            ckpt_dir, config={"timesteps": 999, "seed": 7, "episode_index": 5}
        )
        assert (ckpt_dir / "metadata.json").exists()
        for name in AGENT_NAMES:
            assert (ckpt_dir / f"{name}.pt").exists()

        resumed = IndependentPPOTrainer.load_checkpoint(ckpt_dir)
        return trainer, resumed

    def test_scalars_and_agent_state_restored(self, tmp_path):
        trainer, resumed = self._save_and_load(tmp_path)

        assert resumed.global_step == trainer.global_step
        assert resumed.update_count == trainer.update_count
        assert resumed.seed == trainer.seed
        assert list(resumed.agent_names) == list(trainer.agent_names)

        # Networks identical (bit-exact)
        for name in AGENT_NAMES:
            for (k1, p1), (k2, p2) in zip(
                trainer.networks[name].state_dict().items(),
                resumed.networks[name].state_dict().items(),
            ):
                assert k1 == k2
                assert torch.equal(p1, p2), f"{name}.{k1} differs after resume"

        # Optimizer state restored (Adam moments + step counters)
        for name in AGENT_NAMES:
            orig = trainer.ppo_algorithms[name].optimizer.state_dict()
            load = resumed.ppo_algorithms[name].optimizer.state_dict()
            assert load["state"][0]["step"].item() == orig["state"][0]["step"].item()
            assert torch.equal(load["state"][0]["exp_avg"], orig["state"][0]["exp_avg"])
            assert torch.equal(
                load["state"][0]["exp_avg_sq"], orig["state"][0]["exp_avg_sq"]
            )
            # Learning rate matches for equal configs
            assert load["param_groups"][0]["lr"] == pytest.approx(
                orig["param_groups"][0]["lr"]
            )

    def test_extra_config_roundtrip(self, tmp_path):
        trainer, resumed = self._save_and_load(tmp_path)
        with open(tmp_path / "ckpt" / "metadata.json") as f:
            meta = json.load(f)
        assert meta["extra_config"]["episode_index"] == 5

        # train script reads the same key to continue demand seeds
        episode_index = int(meta.get("extra_config", {}).get("episode_index", 0))
        assert episode_index == 5

    def test_resumed_training_continues(self, tmp_path):
        """A resumed trainer can run another update and keeps counting."""
        trainer, resumed = self._save_and_load(tmp_path)
        _run_rollouts(resumed, n_rollouts=1)

        assert resumed.global_step == trainer.global_step + trainer.rollout_steps * len(AGENT_NAMES)
        assert resumed.update_count == trainer.update_count + 1

    def test_minibatch_size_persisted_in_metadata(self, tmp_path):
        trainer, _ = self._save_and_load(tmp_path)
        with open(tmp_path / "ckpt" / "metadata.json") as f:
            meta = json.load(f)
        assert meta["ppo_config"]["minibatch_size"] == trainer.config.minibatch_size
