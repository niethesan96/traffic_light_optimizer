"""
PPO Algorithm — Clipped Surrogate Objective
=============================================

Implements the Proximal Policy Optimization algorithm from:

    Schulman et al., "Proximal Policy Optimization Algorithms", 2017.

PPO Objective
-------------

The clipped surrogate objective:

    L_CLIP = E_t [
        min(
            r_t(θ) · A_t,
            clip(r_t(θ), 1 - ε, 1 + ε) · A_t
        )
    ]

where:
    r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
    A_t     = advantage estimate (from GAE)
    ε       = clip epsilon (default 0.2)

Total Loss
----------

    L = - L_CLIP + c_1 · L_VF - c_2 · H[π]

where:
    L_VF = (V_θ(s_t) - R_t)²        (value function loss)
    H[π] = - Σ π(a|s) log π(a|s)    (entropy bonus)
    c_1   = value_loss_coefficient    (default 0.5)
    c_2   = entropy_coefficient       (default 0.01)

Approximate KL Divergence
--------------------------

    KL ≈ E_t [ log(π_θ(a_t|s_t) / π_θ_old(a_t|s_t)) ]

    Used for early stopping if KL divergence gets too large.

Clip Fraction
-------------

    clip_frac = E_t [ |r_t(θ) - 1| > ε ]

    Fraction of samples where the clipping was active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from agents.independent_ppo.network import ActorCritic
from agents.independent_ppo.buffer import RolloutBuffer


@dataclass
class PPOConfig:
    """Configuration for the PPO algorithm."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 8
    target_kl: float | None = 0.015
    hidden_size: int = 64
    minibatch_size: int = 64

    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.gamma <= 1.0, f"gamma must be in (0, 1], got {self.gamma}"
        assert 0 <= self.gae_lambda <= 1.0, f"gae_lambda must be in [0, 1], got {self.gae_lambda}"
        assert self.clip_epsilon >= 0, f"clip_epsilon must be >= 0, got {self.clip_epsilon}"
        assert self.entropy_coefficient >= 0, f"entropy_coefficient must be >= 0"
        assert self.value_loss_coefficient >= 0, f"value_loss_coefficient must be >= 0"
        assert self.update_epochs >= 1, f"update_epochs must be >= 1, got {self.update_epochs}"
        assert self.minibatch_size >= 1, f"minibatch_size must be >= 1, got {self.minibatch_size}"


@dataclass
class PPOLoss:
    """Container for PPO loss components."""

    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float = 0.0


class PPOAlgorithm:
    """
    PPO algorithm for a single agent.

    This class manages one ActorCritic network and performs PPO updates
    on its rollout buffer.

    Parameters
    ----------
    network : ActorCritic
        The agent's neural network.
    config : PPOConfig
        PPO hyperparameters.
    device : torch.device
        Compute device.
    """

    def __init__(
        self,
        network: ActorCritic,
        config: PPOConfig | None = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.network = network
        self.config = config or PPOConfig()
        self.device = device

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )

        # Running statistics for logging
        self._update_count = 0

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute the clipped PPO policy loss.

        L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

        Parameters
        ----------
        log_probs : Tensor
            Log probabilities of actions under current policy.
        old_log_probs : Tensor
            Log probabilities of actions under old policy.
        advantages : Tensor
            GAE advantage estimates.

        Returns
        -------
        policy_loss : Tensor
            Negative clipped surrogate objective (for minimization).
        ratio : Tensor
            Probability ratio r_t(θ).
        clip_fraction : float
            Fraction of samples where clipping was active.
        """
        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        ) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # Clip fraction: fraction of samples where |r_t - 1| > epsilon
        with torch.no_grad():
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.config.clip_epsilon).float().mean().item()
            )

        return policy_loss, ratio, clip_fraction

    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value function loss (MSE).

        L_VF = (V_θ(s) - R)²

        Parameters
        ----------
        values : Tensor
            Value estimates from current network.
        returns : Tensor
            Target returns.

        Returns
        -------
        value_loss : Tensor
        """
        return 0.5 * ((values - returns) ** 2).mean()

    def update(
        self,
        buffer: RolloutBuffer,
    ) -> PPOLoss:
        """
        Perform PPO update on the collected rollout.

        Runs multiple epochs of minibatch SGD on the stored data.

        Parameters
        ----------
        buffer : RolloutBuffer
            Filled rollout buffer with computed GAE and returns.

        Returns
        -------
        PPOLoss
            Aggregate loss components from the update.
        """
        self.network.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        for epoch in range(self.config.update_epochs):
            for batch in buffer.get_batches(minibatch_size=self.config.minibatch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Forward pass
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    obs, actions
                )

                # Policy loss (clipped surrogate)
                policy_loss, ratio, clip_frac = self.compute_policy_loss(
                    new_log_probs, old_log_probs, advantages
                )

                # Value loss
                value_loss = self.compute_value_loss(new_values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coefficient * value_loss
                    + self.config.entropy_coefficient * entropy_loss
                )

                # Gradient update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = (
                        (old_log_probs - new_log_probs).mean().item()
                    )

                # Accumulate for logging
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                total_approx_kl += approx_kl
                total_clip_fraction += clip_frac
                num_updates += 1

            # Early stopping if KL divergence is too large
            if self.config.target_kl is not None:
                avg_kl = total_approx_kl / max(num_updates, 1)
                if avg_kl > self.config.target_kl:
                    break

        self._update_count += 1

        n = max(num_updates, 1)
        return PPOLoss(
            policy_loss=total_policy_loss / n,
            value_loss=total_value_loss / n,
            entropy=total_entropy / n,
            total_loss=(
                total_policy_loss + self.config.value_loss_coefficient * total_value_loss
            ) / n,
            approx_kl=total_approx_kl / n,
            clip_fraction=total_clip_fraction / n,
        )

    def get_config_dict(self) -> Dict[str, Any]:
        """Return config as a dictionary for serialization."""
        return {
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "gae_lambda": self.config.gae_lambda,
            "clip_epsilon": self.config.clip_epsilon,
            "entropy_coefficient": self.config.entropy_coefficient,
            "value_loss_coefficient": self.config.value_loss_coefficient,
            "max_grad_norm": self.config.max_grad_norm,
            "update_epochs": self.config.update_epochs,
            "target_kl": self.config.target_kl,
            "hidden_size": self.config.hidden_size,
            "minibatch_size": self.config.minibatch_size,
        }
