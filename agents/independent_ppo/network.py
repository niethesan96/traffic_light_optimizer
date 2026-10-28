"""
Actor-Critic Network for Independent PPO
==========================================

Each agent has its own independent ActorCritic network with NO parameter
sharing between agents.

Architecture:
    observation (18-dim float32)
        ↓
    Shared feature MLP (input_dim → hidden_size → hidden_size)
        ↓
    ├── Actor head: linear → num_actions logits
    └── Critic head: linear → scalar V(s)

Total parameters per agent: ~2,300 (with default hidden_size=64).

Mathematical formulation:
    φ(s) = MLP(s)                    # shared features
    π(a|s) = softmax(W_actor · φ(s)) # policy (actor)
    V(s) = W_critic · φ(s)           # value (critic)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple


class ActorCritic(nn.Module):
    """
    Independent Actor-Critic network for one agent.

    Parameters
    ----------
    obs_dim : int
        Observation dimensionality (default: 18).
    num_actions : int
        Number of discrete actions (default: 2).
    hidden_size : int
        Width of hidden layers (default: 64).
    """

    def __init__(
        self,
        obs_dim: int = 18,
        num_actions: int = 2,
        hidden_size: int = 64,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        # --- Shared feature extractor ---
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        # --- Actor head (policy) ---
        self.actor = nn.Linear(hidden_size, num_actions)

        # --- Critic head (value function) ---
        self.critic = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

        # Actor: smaller gain for action head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        # Critic: sqrt(2) gain for value head
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through shared features.

        Parameters
        ----------
        obs : Tensor of shape (batch_size, obs_dim)

        Returns
        -------
        logits : Tensor of shape (batch_size, num_actions)
        value : Tensor of shape (batch_size, 1)
        """
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate an action and compute log_prob + value.

        Parameters
        ----------
        obs : Tensor of shape (batch_size, obs_dim)
            Observation.
        action : Tensor of shape (batch_size,), optional
            If provided, evaluate log_prob of this action (for training).
            If None, sample a new action (for rollout collection).

        Returns
        -------
        action : Tensor of shape (batch_size,)
            Selected action.
        log_prob : Tensor of shape (batch_size,)
            Log probability of the action.
        entropy : Tensor of shape (batch_size,)
            Entropy of the policy distribution.
        value : Tensor of shape (batch_size,)
            Critic value estimate V(s).
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate only (for GAE computation on last observation).

        Parameters
        ----------
        obs : Tensor of shape (batch_size, obs_dim)

        Returns
        -------
        value : Tensor of shape (batch_size,)
        """
        _, value = self.forward(obs)
        return value.squeeze(-1)

    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities (for deterministic evaluation or analysis).

        Parameters
        ----------
        obs : Tensor of shape (batch_size, obs_dim)

        Returns
        -------
        probs : Tensor of shape (batch_size, num_actions)
        """
        logits, _ = self.forward(obs)
        return torch.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
