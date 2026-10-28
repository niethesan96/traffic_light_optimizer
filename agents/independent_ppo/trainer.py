"""
Independent PPO Trainer
========================

Manages 9 independent PPO agents, one per intersection.

Key design:
    - Each agent has its own ActorCritic network (NO parameter sharing)
    - Each agent has its own RolloutBuffer
    - Each agent has its own PPO optimizer
    - All 9 agents interact with the SAME SUMO environment simultaneously
    - Agents choose actions from their LOCAL observations only

Training loop:
    1. env.reset() → observations for all 9 agents
    2. For each agent: collect action from its local policy
    3. env.step(all_actions) → new observations, rewards, truncations
    4. Store transitions in each agent's buffer
    5. After rollout_steps: compute GAE, PPO update for each agent
    6. Repeat

There is NO centralized critic, NO shared parameters, NO cross-agent
information flow.
"""

from __future__ import annotations

import csv
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

from agents.independent_ppo.network import ActorCritic
from agents.independent_ppo.buffer import RolloutBuffer
from agents.independent_ppo.ppo import PPOAlgorithm, PPOConfig, PPOLoss


def _explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Explained variance of the critic.

    EV = 1 - Var(y_true - y_pred) / Var(y_true)

    Returns NaN when the returns have zero variance (no signal).
    """
    if len(y_true) == 0:
        return float("nan")
    var_true = float(np.var(y_true))
    if var_true <= 0.0:
        return float("nan")
    return 1.0 - float(np.var(y_true - y_pred)) / var_true


# ---------------------------------------------------------------------------
# Training logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Append-per-update CSV with immediate flush + periodic JSON rewrite.

    Guarantees:

    * One CSV row per PPO update; the row is flushed to disk immediately so
      a crash or KeyboardInterrupt never loses completed updates.
    * ``training_logs.json`` is rewritten in full by ``write_json()``
      (call every N updates and once in a ``finally:`` block).
    * If the CSV already exists from a previous run with a different
      header, the old file is preserved by renaming it to ``*.bak``
      (nothing is deleted).
    """

    def __init__(
        self,
        output_dir: str | Path,
        csv_name: str = "training_metrics.csv",
        json_name: str = "training_logs.json",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / csv_name
        self.json_path = self.output_dir / json_name
        self.json_logs: List[Dict[str, Any]] = []

        self._columns: List[str] = []
        self._needs_header = True
        self._csv_file = None

        # Preserve JSON history across resumed runs when parseable.
        if self.json_path.exists():
            try:
                with open(self.json_path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    self.json_logs = loaded
            except (OSError, ValueError):
                self.json_logs = []

        # An existing non-empty CSV means we append after it (and rotate it
        # if its header no longer matches).
        self._append_existing = (
            self.csv_path.exists() and self.csv_path.stat().st_size > 0
        )

    # -- internals --

    def _open_file(self, columns: List[str]) -> None:
        """Open the CSV handle, rotating any incompatible existing file."""
        if self._append_existing:
            with open(self.csv_path, "r", newline="") as f:
                existing_header = next(csv.reader(f), None)
            if existing_header == columns:
                self._needs_header = False
            else:
                # Different schema: keep the old data under a .bak name.
                bak = self.csv_path.with_suffix(".csv.bak")
                try:
                    if bak.exists():
                        bak.unlink()
                    self.csv_path.rename(bak)
                except OSError:
                    pass  # fall through: overwritten by truncate below
        self._csv_file = open(self.csv_path, "a", newline="")

    @staticmethod
    def _flatten(entry: Dict[str, Any]) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for k, v in entry.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[f"{k}_{sub_k}"] = sub_v
            else:
                flat[k] = v
        return flat

    # -- public API --

    def log(self, entry: Dict[str, Any]) -> None:
        """Append one flattened row per PPO update and flush immediately."""
        flat = self._flatten(entry)
        columns = list(flat.keys())

        if not self._columns:
            self._open_file(columns)
            self._columns = columns
        elif self._columns != columns:
            # Schema changed mid-run: rotate and restart the file.
            if self._csv_file is not None:
                self._csv_file.close()
            bak = self.csv_path
            try:
                target = self.csv_path.with_suffix(".csv.bak")
                if target.exists():
                    target.unlink()
                self.csv_path.rename(target)
            except OSError:
                pass
            self._append_existing = False
            self._needs_header = True
            self._open_file(columns)
            self._columns = columns

        writer = csv.writer(self._csv_file)
        if self._needs_header:
            writer.writerow(self._columns)
            self._needs_header = False
        writer.writerow([flat.get(k, "") for k in self._columns])
        self._csv_file.flush()

        self.json_logs.append(entry)

    def write_json(self) -> None:
        """Rewrite training_logs.json in full (call every 10 updates + finally)."""
        with open(self.json_path, "w") as f:
            json.dump(self.json_logs, f, indent=2)

    def close(self) -> None:
        """Flush pending CSV data and release the file handle."""
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
            except OSError:
                pass
            self._csv_file.close()
            self._csv_file = None


class IndependentPPOTrainer:
    """
    Trainer for 9 independent PPO agents in the multi-intersection environment.

    Parameters
    ----------
    agent_names : list[str]
        List of agent names (e.g. ["agent_1_1", ..., "agent_3_3"]).
    obs_dim : int
        Observation dimensionality for each agent.
    num_actions : int
        Number of discrete actions.
    config : PPOConfig
        PPO hyperparameters (shared across agents for simplicity).
    rollout_steps : int
        Number of steps to collect before each PPO update.
        (Legacy default was 128; the unified default is now 360 — one full
        3600 s episode at the standard 10 s decision interval.)
    device : torch.device
        Compute device.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        agent_names: List[str],
        obs_dim: int = 18,
        num_actions: int = 2,
        config: PPOConfig | None = None,
        rollout_steps: int = 360,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        self.agent_names = agent_names
        self.num_agents = len(agent_names)
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.config = config or PPOConfig()
        self.rollout_steps = rollout_steps
        self.device = device
        self.seed = seed

        # Create one independent ActorCritic per agent (NO parameter sharing)
        self.networks: Dict[str, ActorCritic] = {}
        self.ppo_algorithms: Dict[str, PPOAlgorithm] = {}
        self.buffers: Dict[str, RolloutBuffer] = {}

        for name in agent_names:
            net = ActorCritic(
                obs_dim=obs_dim,
                num_actions=num_actions,
                hidden_size=self.config.hidden_size,
            ).to(device)
            self.networks[name] = net

            self.ppo_algorithms[name] = PPOAlgorithm(
                network=net,
                config=self.config,
                device=device,
            )

            self.buffers[name] = RolloutBuffer(
                buffer_size=rollout_steps,
                obs_dim=obs_dim,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                device=device,
            )

        # Training statistics
        self.global_step = 0
        self.update_count = 0
        self.episode_rewards: Dict[str, List[float]] = {name: [] for name in agent_names}
        self.episode_lengths: List[int] = []

        # Current rollout storage
        self._current_obs: Dict[str, np.ndarray] = {}
        self._current_log_probs: Dict[str, float] = {}
        self._current_values: Dict[str, float] = {}
        self._episode_rewards_buffer: Dict[str, float] = {name: 0.0 for name in agent_names}
        self._episode_lengths_buffer: Dict[str, int] = {name: 0 for name in agent_names}

    @torch.no_grad()
    def get_actions(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, int]:
        """
        Get actions for all agents from their local observations.

        Parameters
        ----------
        observations : dict
            Mapping from agent_name to observation array (obs_dim,).

        Returns
        -------
        dict
            Mapping from agent_name to action (int).
        """
        actions = {}
        for name in self.agent_names:
            obs = observations[name]
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.networks[name].eval()
            action, log_prob, _, value = self.networks[name].get_action_and_value(obs_tensor)
            self.networks[name].train()

            actions[name] = action.item()
            self._current_log_probs[name] = log_prob.item()
            self._current_values[name] = value.item()
            self._current_obs[name] = obs.copy()

        return actions

    def store_transition(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        terminations: Dict[str, bool] | None = None,
        truncations: Dict[str, bool] | None = None,
        next_observations: Dict[str, np.ndarray] | None = None,
    ) -> None:
        """
        Store transitions in each agent's buffer.

        Parameters
        ----------
        observations : dict
            New observations (after step).
        actions : dict
            Actions that were taken.
        rewards : dict
            Rewards received.
        dones : dict
            Whether each agent's episode ended (terminated OR truncated).
        terminations : dict, optional
            True where the episode ended for a genuine environment reason.
        truncations : dict, optional
            True where the episode hit the time limit. Truncated final
            transitions get ``gamma * V(s_final)`` added to their reward
            (bootstrapping) while still cutting the GAE recursion.
        """
        for name in self.agent_names:
            terminated = (
                terminations[name] if terminations is not None else None
            )
            truncated = (
                truncations[name] if truncations is not None else None
            )
            next_value = None
            if truncated and next_observations is not None:
                with torch.no_grad():
                    next_value = self.networks[name].get_value(torch.as_tensor(
                        next_observations[name], dtype=torch.float32,
                        device=self.device).unsqueeze(0)).item()
            self.buffers[name].add(
                obs=self._current_obs[name],
                action=actions[name],
                reward=rewards[name],
                value=self._current_values[name],
                log_prob=self._current_log_probs[name],
                done=dones[name],
                terminated=terminated,
                truncated=truncated,
                next_value=next_value,
            )

            # Track episode reward and length
            self._episode_rewards_buffer[name] += rewards[name]
            self._episode_lengths_buffer[name] += 1
            self.global_step += 1

            if dones[name]:
                self.episode_rewards[name].append(self._episode_rewards_buffer[name])
                self._episode_rewards_buffer[name] = 0.0

        if any(dones.values()):
            self.episode_lengths.append(
                max(self._episode_lengths_buffer.values())
            )
            for name in self.agent_names:
                self._episode_lengths_buffer[name] = 0

    def should_update(self) -> bool:
        """Check if all buffers are full and ready for update."""
        return all(buf.is_full for buf in self.buffers.values())

    @torch.no_grad()
    def compute_bootstrap_values(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute bootstrap value V(s_{T+1}) for each agent's last observation.

        Parameters
        ----------
        observations : dict
            Current observations for all agents.

        Returns
        -------
        dict
            Mapping from agent_name to bootstrap value.
        """
        bootstrap = {}
        for name in self.agent_names:
            obs = observations[name]
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            value = self.networks[name].get_value(obs_tensor)
            bootstrap[name] = value.item()
        return bootstrap

    def compute_gae_and_update(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, PPOLoss]:
        """
        Compute GAE for all agents and perform PPO updates.

        Parameters
        ----------
        observations : dict
            Current observations (used for bootstrap value estimation).

        Returns
        -------
        dict
            Mapping from agent_name to PPOLoss.
        """
        bootstrap = self.compute_bootstrap_values(observations)

        losses = {}
        for name in self.agent_names:
            # Compute GAE
            self.buffers[name].compute_gae(last_value=bootstrap[name])
            self.buffers[name].normalize_advantages()

            # Explained variance of the critic, measured on this rollout
            # *before* the update changes the network.
            ev = _explained_variance(
                self.buffers[name].values[: self.buffers[name].buffer_size],
                self.buffers[name].returns,
            )

            # PPO update
            loss = self.ppo_algorithms[name].update(self.buffers[name])
            loss.explained_variance = ev
            losses[name] = loss

            # Clear buffer
            self.buffers[name].clear()

        self.update_count += 1
        return losses

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current training statistics.

        Returns
        -------
        dict with keys:
            global_step, update_count,
            mean_episode_reward, per_agent_mean_reward,
            mean_episode_length
        """
        mean_rewards = {}
        for name in self.agent_names:
            if self.episode_rewards[name]:
                mean_rewards[name] = float(np.mean(self.episode_rewards[name][-100:]))
            else:
                mean_rewards[name] = 0.0

        all_recent_rewards = []
        for name in self.agent_names:
            all_recent_rewards.extend(self.episode_rewards[name][-100:])

        return {
            "global_step": self.global_step,
            "update_count": self.update_count,
            "mean_episode_reward": float(np.mean(all_recent_rewards)) if all_recent_rewards else 0.0,
            "per_agent_mean_reward": mean_rewards,
            "mean_episode_length": float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0,
        }

    def save_checkpoint(
        self, path: str | Path, config: dict | None = None
    ) -> Path:
        """
        Save checkpoint for all 9 agents.

        Parameters
        ----------
        path : str or Path
            Directory to save checkpoint files.
        config : dict, optional
            Additional config to save.

        Returns
        -------
        Path
            Directory where checkpoint was saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each agent's network
        for name in self.agent_names:
            agent_path = path / f"{name}.pt"
            torch.save(
                {
                    "network_state_dict": self.networks[name].state_dict(),
                    "optimizer_state_dict": self.ppo_algorithms[name].optimizer.state_dict(),
                },
                agent_path,
            )

        # Save training state
        metadata = {
            "global_step": self.global_step,
            "update_count": self.update_count,
            "agent_names": self.agent_names,
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "seed": self.seed,
            "rollout_steps": self.rollout_steps,
            "ppo_config": self.ppo_algorithms[self.agent_names[0]].get_config_dict(),
            "episode_rewards": {name: rewards[-100:] for name, rewards in self.episode_rewards.items()},
            "episode_lengths": self.episode_lengths[-100:],
        }
        if config:
            metadata["extra_config"] = config

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: torch.device = torch.device("cpu"),
    ) -> "IndependentPPOTrainer":
        """
        Load checkpoint and restore trainer state.

        Parameters
        ----------
        path : str or Path
            Directory containing checkpoint files.
        device : torch.device
            Device to load models onto.

        Returns
        -------
        IndependentPPOTrainer
            Restored trainer.
        """
        path = Path(path)

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        config = PPOConfig(**metadata["ppo_config"])

        trainer = cls(
            agent_names=metadata["agent_names"],
            obs_dim=metadata["obs_dim"],
            num_actions=metadata["num_actions"],
            config=config,
            rollout_steps=metadata.get("rollout_steps", 360),
            device=device,
            seed=metadata["seed"],
        )

        trainer.global_step = metadata["global_step"]
        trainer.update_count = metadata["update_count"]

        for name in metadata["agent_names"]:
            checkpoint = torch.load(path / f"{name}.pt", map_location=device)
            trainer.networks[name].load_state_dict(checkpoint["network_state_dict"])
            trainer.ppo_algorithms[name].optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        return trainer

    def save_training_log(self, path: str | Path, logs: List[Dict[str, Any]]) -> Path:  # noqa: D212
        """
        Save training logs to JSON.

        Parameters
        ----------
        path : str or Path
            File path for the log.
        logs : list of dict
            Training log entries.

        Returns
        -------
        Path
            Path to saved log file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(logs, f, indent=2)
        return path
