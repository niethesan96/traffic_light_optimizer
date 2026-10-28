"""CTDE policy: one-hop neighbor messages for actors, joint state for critic."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


def neighbors(agent_names):
    positions = [tuple(map(int, name.split('_')[-2:])) for name in agent_names]
    return [[j for j, q in enumerate(positions)
             if abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1]
            for p in positions]


class MAPPOModel(nn.Module):
    def __init__(self, agent_names, obs_dim=18, hidden_size=64, num_actions=2):
        super().__init__()
        self.agent_names = list(agent_names)
        self.obs_dim = obs_dim
        self.adjacency = neighbors(agent_names)
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh())
        # Ordered N/E/S/W slots retain directional information. Missing neighbors
        # send zero messages; actor identity is local and fixed at deployment.
        self.actor = nn.Sequential(
            nn.Linear(hidden_size * 5 + len(agent_names), hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, num_actions))
        self.critic = nn.Sequential(
            nn.Linear(len(agent_names) * obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, len(agent_names)))

    def actor_logits(self, local_observation, neighbor_observations, agent_index):
        """Deployment API: only local state and four immediate neighbor slots."""
        own = self.encoder(local_observation)
        messages = self.encoder(neighbor_observations)
        # Zero absent messages including the encoder bias.
        present = neighbor_observations.abs().sum(-1, keepdim=True) != 0
        messages = (messages * present).flatten(-2)
        identity = torch.nn.functional.one_hot(
            torch.as_tensor(agent_index, device=own.device), len(self.agent_names)
        ).to(own.dtype)
        identity = identity.expand(*own.shape[:-1], len(self.agent_names))
        return self.actor(torch.cat((own, messages, identity), -1))

    def logits(self, observations):
        positions = [tuple(map(int, n.split('_')[-2:])) for n in self.agent_names]
        lookup = {p: j for j, p in enumerate(positions)}
        outputs = []
        for i, (r, c) in enumerate(positions):
            slots = [observations[..., lookup[p], :] if p in lookup
                     else torch.zeros_like(observations[..., i, :])
                     for p in ((r-1, c), (r, c+1), (r+1, c), (r, c-1))]
            outputs.append(self.actor_logits(observations[..., i, :],
                                            torch.stack(slots, -2), i))
        return torch.stack(outputs, -2)

    def values(self, observations):
        return self.critic(observations.flatten(-2))

    def distribution(self, observations):
        return Categorical(logits=self.logits(observations))


class ResidualMAPPOModel(MAPPOModel):
    """Independent local experts plus a learned neighbor-dependent correction.

    A zero initial correction supports a controlled warm start: actions and
    values initially match independent PPO, with equal pretraining budgets.
    """
    def __init__(self, agent_names, obs_dim=18, hidden_size=64, num_actions=2):
        super().__init__(agent_names, obs_dim, hidden_size, num_actions)
        from agents.independent_ppo.network import ActorCritic
        self.local_experts = nn.ModuleList([
            ActorCritic(obs_dim, num_actions, hidden_size) for _ in agent_names])
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def actor_logits(self, local_observation, neighbor_observations, agent_index):
        local_logits, _ = self.local_experts[int(agent_index)](local_observation)
        return local_logits + super().actor_logits(
            local_observation, neighbor_observations, agent_index)

    def values(self, observations):
        local_values = torch.stack([expert.get_value(observations[..., i, :])
            for i, expert in enumerate(self.local_experts)], -1)
        return local_values + super().values(observations)
