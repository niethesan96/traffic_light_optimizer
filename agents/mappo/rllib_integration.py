"""Ray 2.59 new-stack MAPPO module and SUMO multi-agent adapter.

The actor slices only local/one-hop features. Joint state is reserved for
ValueFunctionAPI.compute_values, which the PPO learner uses during training.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from agents.mappo.model import MAPPOModel
from envs.multi_intersection_env import MultiIntersectionEnv


def pack_observations(observations, names, disable_messages=False, include_joint=True):
    positions = [tuple(map(int, n.split('_')[-2:])) for n in names]
    lookup = dict(zip(positions, names))
    joint = (np.concatenate([observations[n] for n in names]) if include_joint
             else np.zeros(len(names)*len(observations[names[0]]),dtype=np.float32))
    result = {}
    for index, (name, (r, c)) in enumerate(zip(names, positions)):
        own = observations[name]
        messages = [observations[lookup[p]] if p in lookup and not disable_messages
                    else np.zeros_like(own)
                    for p in ((r-1,c), (r,c+1), (r+1,c), (r,c-1))]
        result[name] = np.concatenate([own, *messages, np.eye(len(names))[index], joint]).astype(np.float32)
    return result


class TrafficMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.traffic = MultiIntersectionEnv(config.get('config_path', 'configs/cooperative.yaml'),
            seed=config.get('seed', 42), episode_duration=config.get('episode_duration', 3600),
            demand_seed=config.get('demand_seed_start', 2000))
        self.demand_seed_start = config.get('demand_seed_start', 2000)
        self.episode_index = -1
        self.team_weight = config.get('team_weight', 0.25)
        self.possible_agents = self.traffic.possible_agents
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.agents)
        dim = self.traffic.get_observation_dim()
        packed_dim = dim * (5 + len(self.agents)) + len(self.agents)
        self.observation_spaces = {n: spaces.Box(-np.inf, np.inf, (packed_dim,), np.float32) for n in self.agents}
        self.action_spaces = {n: spaces.Discrete(2) for n in self.agents}
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)

    def reset(self, *, seed=None, options=None):
        self.episode_index += 1
        self.agents = list(self.possible_agents)
        obs, infos = self.traffic.reset(seed=seed, options={
            'demand_seed': self.demand_seed_start+self.episode_index})
        return pack_observations(obs, self.possible_agents), infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.traffic.step(actions)
        mean = float(np.mean(list(rewards.values())))
        rewards = {n: (1-self.team_weight)*r+self.team_weight*mean for n,r in rewards.items()}
        terminated['__all__'] = all(terminated.values())
        truncated['__all__'] = all(truncated.values())
        return pack_observations(obs, self.possible_agents), rewards, terminated, truncated, infos

    def close(self):
        self.traffic.close()


class TrafficPPOModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        self.names = self.model_config['agent_names']
        self.local_dim = self.model_config['local_dim']
        self.network = MAPPOModel(self.names, self.local_dim, self.model_config.get('hidden_size',64))

    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS]
        dim, count = self.local_dim, len(self.names)
        local = obs[..., :dim]
        messages = obs[..., dim:5*dim].reshape(*obs.shape[:-1],4,dim)
        identity = obs[..., 5*dim:5*dim+count].argmax(-1)
        return {Columns.ACTION_DIST_INPUTS: self.network.actor_logits(local, messages, identity)}

    def compute_values(self, batch, embeddings=None):
        obs = batch[Columns.OBS]
        dim, count = self.local_dim, len(self.names)
        identity = obs[..., 5*dim:5*dim+count].argmax(-1)
        joint = obs[..., 5*dim+count:].reshape(*obs.shape[:-1],count,dim)
        values = self.network.values(joint)
        return values.gather(-1, identity.unsqueeze(-1)).squeeze(-1)


class RLlibController:
    def __init__(self, path):
        self.module = RLModule.from_checkpoint(str(Path(path).resolve()))
        self.agent_names = self.module.names
        self.obs_dim = self.module.local_dim
        self.global_step = 0
        manifest=Path(path).parent/'manifest.json'
        if manifest.exists():
            metadata=json.loads(manifest.read_text())
            self.global_step=metadata.get('joint_env_steps',
                metadata.get('iterations',0)*metadata.get('batch_size',0))

    @torch.no_grad()
    def actions(self, observations, deterministic=True, disable_messages=False):
        packed = pack_observations(observations, self.agent_names, disable_messages, include_joint=False)
        tensor = torch.as_tensor(np.stack([packed[n] for n in self.agent_names]))
        logits = self.module.forward_inference({Columns.OBS:tensor})[Columns.ACTION_DIST_INPUTS]
        actions = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        return dict(zip(self.agent_names, actions.tolist()))

    @torch.no_grad()
    def local_action(self, local_observation, neighbor_observations, agent_index):
        """Single deployed actor: four one-hop message slots, no joint state."""
        obs=np.concatenate([np.asarray(local_observation).reshape(-1),
            np.asarray(neighbor_observations).reshape(-1),
            np.eye(len(self.agent_names))[agent_index],
            np.zeros(len(self.agent_names)*self.obs_dim)]).astype(np.float32)
        logits=self.module.forward_inference({Columns.OBS:torch.from_numpy(obs).unsqueeze(0)})[Columns.ACTION_DIST_INPUTS]
        return int(logits.argmax(-1).item())
