"""
Rollout Buffer with GAE for PPO
================================

Stores trajectories collected from the environment and computes
Generalized Advantage Estimation (GAE) and returns.

Storage per transition:
    - observation (obs_dim,)
    - action (int)
    - reward (float)
    - value (float)
    - log_prob (float)
    - done (bool)

Computed after rollout:
    - advantages (computed via GAE)
    - returns (computed as advantages + values)

GAE Formula:
    δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
    A_t  = δ_t + γ · λ · A_{t+1}

    where A_{T} = δ_T for the last step (or 0 if terminal).

Returns:
    R_t = A_t + V(s_t)

References:
    Schulman et al., "High-Dimensional Continuous Control Using
    Generalized Advantage Estimation", ICLR 2016.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Iterator, Tuple


class RolloutBuffer:
    """
    Fixed-size rollout buffer for one PPO agent.

    Parameters
    ----------
    buffer_size : int
        Maximum number of transitions to store (= rollout_steps).
    obs_dim : int
        Dimensionality of the observation space.
    gamma : float
        Discount factor for GAE.
    gae_lambda : float
        Lambda parameter for GAE.
    device : torch.device
        Device for tensor storage.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Pre-allocate storage
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size + 1, dtype=np.float32)  # +1 for bootstrap
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        # Keep terminated and truncated separate so compute_gae can bootstrap
        # truncated final transitions while still cutting at time limits.
        self.terminated = np.zeros(buffer_size, dtype=np.float32)
        self.truncated = np.zeros(buffer_size, dtype=np.float32)
        self.next_values = np.full(buffer_size, np.nan, dtype=np.float32)

        # Computed after compute_gae()
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        terminated: bool | None = None,
        truncated: bool | None = None,
        next_value: float | None = None,
    ) -> None:
        """
        Add a single transition to the buffer.

        Parameters
        ----------
        obs : np.ndarray of shape (obs_dim,)
        action : int
        reward : float
        value : float
            Value estimate V(s) at the time of this transition.
        log_prob : float
            Log probability of the taken action.
        done : bool
            Whether this transition ends the episode (terminated OR truncated).
            Legacy callers that pass only ``done`` keep the old
            truncation-as-terminal semantics.
        terminated : bool, optional
            True when the episode ended for a genuine environment reason.
            Defaults to ``done and not truncated`` when omitted.
        truncated : bool, optional
            True when the episode was cut by a time limit. On truncation the
            final transition is bootstrapped with ``gamma * V(s_final)`` in
            ``compute_gae`` instead of being treated as terminal.
        """
        if self.ptr >= self.buffer_size:
            raise BufferError(
                f"Buffer is full ({self.buffer_size} transitions). "
                "Call compute_gae() and clear() first."
            )

        if terminated is None:
            terminated = bool(done) and not bool(truncated)
        if truncated is None:
            truncated = False

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done or terminated or truncated)
        self.terminated[self.ptr] = float(bool(terminated))
        self.truncated[self.ptr] = float(bool(truncated))
        self.next_values[self.ptr] = np.nan if next_value is None else next_value
        self.ptr += 1

        if self.ptr == self.buffer_size:
            self.full = True

    def compute_gae(self, last_value: float) -> None:
        """
        Compute GAE advantages and returns using stored transitions.

        Must be called after the buffer is filled (ptr == buffer_size).
        The bootstrap value V(s_{T+1}) is provided as last_value.

        Parameters
        ----------
        last_value : float
            Value estimate V(s_{T+1}) for the state after the last transition.
            Use 0 if the last transition was terminal.
        """
        self.values[self.ptr] = last_value

        # --- Truncation bootstrapping -------------------------------------
        # If the rollout ended because the episode hit a TIME LIMIT (truncated,
        # not truly terminal), the state after the last transition is a real
        # state whose value must not be treated as zero. Add gamma * V(s_next)
        # to the last reward of that transition; ``done`` stays True so the
        # GAE recursion below still cuts at the episode boundary.
        last = self.ptr - 1
        if (last >= 0 and self.truncated[last] and not self.terminated[last]
                and np.isnan(self.next_values[last])):
            self.rewards[last] += self.gamma * last_value

        last_gae_lam = 0.0
        for t in reversed(range(self.buffer_size)):
            if self.dones[t]:
                next_non_terminal = 0.0
            else:
                next_non_terminal = 1.0

            delta = (
                self.rewards[t]
                + self.gamma * self.values[t + 1] * next_non_terminal
                - self.values[t]
            )
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            if (self.truncated[t] and not self.terminated[t]
                    and np.isfinite(self.next_values[t])):
                # Bootstrap an internal episode time limit from its final
                # observation, while the done mask cuts off the next episode.
                last_gae_lam += self.gamma * self.next_values[t]
            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + self.values[: self.buffer_size]

    def compute_returns_simple(self) -> None:
        """
        Compute discounted returns (without GAE) as a fallback.

        R_t = r_t + γ · r_{t+1} + γ² · r_{t+2} + ...
        """
        running_return = 0.0
        for t in reversed(range(self.buffer_size)):
            if self.dones[t]:
                running_return = 0.0
            running_return = self.rewards[t] + self.gamma * running_return
            self.returns[t] = running_return
        self.advantages = self.returns - self.values[: self.buffer_size]

    def normalize_advantages(self) -> None:
        """
        Normalize advantages to have zero mean and unit variance.

        This is standard practice in PPO to stabilize training.
        """
        mean = np.mean(self.advantages)
        std = np.std(self.advantages) + 1e-8
        self.advantages = (self.advantages - mean) / std

    def get_batches(
        self, minibatch_size: int, shuffle: bool = True
    ) -> Iterator[dict[str, torch.Tensor]]:
        """
        Yield minibatches for PPO update.

        Parameters
        ----------
        minibatch_size : int
            Size of each minibatch.
        shuffle : bool
            Whether to shuffle indices.

        Yields
        ------
        dict with keys: obs, actions, log_probs, advantages, returns, values
        """
        assert self.ptr == self.buffer_size, (
            f"Buffer not full: {self.ptr}/{self.buffer_size}"
        )

        indices = np.arange(self.buffer_size)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.buffer_size, minibatch_size):
            end = start + minibatch_size
            batch_indices = indices[start:end]

            yield {
                "obs": torch.tensor(
                    self.observations[batch_indices], dtype=torch.float32, device=self.device
                ),
                "actions": torch.tensor(
                    self.actions[batch_indices], dtype=torch.long, device=self.device
                ),
                "old_log_probs": torch.tensor(
                    self.log_probs[batch_indices], dtype=torch.float32, device=self.device
                ),
                "advantages": torch.tensor(
                    self.advantages[batch_indices], dtype=torch.float32, device=self.device
                ),
                "returns": torch.tensor(
                    self.returns[batch_indices], dtype=torch.float32, device=self.device
                ),
                "old_values": torch.tensor(
                    self.values[batch_indices], dtype=torch.float32, device=self.device
                ),
            }

    def clear(self) -> None:
        """Reset the buffer for a new rollout."""
        self.ptr = 0
        self.full = False
        self.terminated[:] = 0.0
        self.truncated[:] = 0.0
        self.next_values[:] = np.nan

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self.ptr

    @property
    def is_full(self) -> bool:
        """Whether the buffer is full."""
        return self.full
