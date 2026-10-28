"""Shared-actor MAPPO with per-agent centralized values and boundary-safe GAE."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from agents.independent_ppo.ppo import PPOConfig
from agents.mappo.model import MAPPOModel, ResidualMAPPOModel


def compute_gae(rewards, values, next_values, terminated, truncated,
                gamma=0.99, gae_lambda=0.95):
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * gae_lambda * (~(terminated[t] | truncated[t])) * carry
        advantages[t] = carry
    return advantages, advantages + values


class MAPPOTrainer:
    def __init__(self, agent_names, obs_dim=18, config=None, model_version=1):
        self.agent_names = list(agent_names)
        self.obs_dim = obs_dim
        self.config = config or PPOConfig()
        self.model_version = model_version
        model_class = ResidualMAPPOModel if model_version == 2 else MAPPOModel
        self.model = model_class(agent_names, obs_dim, self.config.hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.global_step = 0

    def tensor(self, observations):
        return torch.as_tensor(np.stack([observations[n] for n in self.agent_names]),
                               dtype=torch.float32)

    @torch.no_grad()
    def actions(self, observations, deterministic=False):
        state = self.tensor(observations)
        distribution = self.model.distribution(state)
        actions = distribution.logits.argmax(-1) if deterministic else distribution.sample()
        return dict(zip(self.agent_names, actions.tolist()))

    def collect(self, env, observations, steps):
        storage = {k: [] for k in ('obs', 'actions', 'log_probs', 'rewards', 'values',
                                  'next_values', 'terminated', 'truncated')}
        for _ in range(steps):
            state = self.tensor(observations)
            with torch.no_grad():
                dist = self.model.distribution(state)
                action = dist.sample()
                value = self.model.values(state)
                log_prob = dist.log_prob(action)
            following, rewards, terminated, truncated, _ = env.step(
                dict(zip(self.agent_names, action.tolist())))
            with torch.no_grad():
                next_value = self.model.values(self.tensor(following))
            row = dict(obs=state, actions=action, log_probs=log_prob, values=value,
                       next_values=next_value,
                       rewards=torch.tensor([rewards[n] for n in self.agent_names]),
                       terminated=torch.tensor([terminated[n] for n in self.agent_names]),
                       truncated=torch.tensor([truncated[n] for n in self.agent_names]))
            for key, item in row.items():
                storage[key].append(item)
            self.global_step += 1
            observations = following
            if any(terminated.values()) or any(truncated.values()):
                observations, _ = env.reset()
        return {k: torch.stack(v) for k, v in storage.items()}, observations

    def update(self, rollout):
        cfg = self.config
        advantages, returns = compute_gae(
            rollout['rewards'], rollout['values'], rollout['next_values'],
            rollout['terminated'], rollout['truncated'], cfg.gamma, cfg.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        records = []
        for _ in range(cfg.update_epochs):
            stop = False
            for indices in torch.randperm(len(advantages)).split(cfg.minibatch_size):
                state = rollout['obs'][indices]
                dist = self.model.distribution(state)
                log_prob = dist.log_prob(rollout['actions'][indices])
                log_ratio = log_prob - rollout['log_probs'][indices]
                ratio = log_ratio.exp()
                adv = advantages[indices]
                policy = -torch.minimum(ratio * adv,
                    ratio.clamp(1-cfg.clip_epsilon, 1+cfg.clip_epsilon) * adv).mean()
                value = (self.model.values(state) - returns[indices]).square().mean()
                entropy = dist.entropy().mean()
                loss = policy + cfg.value_loss_coefficient * value - cfg.entropy_coefficient * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite MAPPO loss')
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                kl = ((ratio - 1) - log_ratio).mean().detach().item()
                records.append([policy.item(), value.item(), entropy.item(), kl])
                if cfg.target_kl is not None and kl > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        return dict(zip(('policy_loss', 'value_loss', 'entropy', 'approx_kl'),
                        np.mean(records, axis=0).tolist()),
                    mean_reward=rollout['rewards'].mean().item(), step=self.global_step)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                        agent_names=self.agent_names, obs_dim=self.model.obs_dim,
                        config=asdict(self.config), global_step=self.global_step,
                        model_version=self.model_version), path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        trainer = cls(checkpoint['agent_names'], checkpoint['obs_dim'],
                      PPOConfig(**checkpoint['config']), checkpoint.get('model_version', 1))
        trainer.model.load_state_dict(checkpoint['model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.global_step = checkpoint['global_step']
        return trainer
