import numpy as np
import torch
from pathlib import Path
import pytest

from agents.mappo.model import MAPPOModel, neighbors
from agents.mappo.trainer import MAPPOTrainer, compute_gae
from agents.independent_ppo.ppo import PPOConfig
from sumo_network.grid_builder import GridBuilder, GridConfig

NAMES = [f'agent_{r}_{c}' for r in range(1, 4) for c in range(1, 4)]


def test_actor_has_only_one_hop_information():
    torch.manual_seed(2)
    model = MAPPOModel(NAMES)
    obs = torch.rand(9, 18)
    original = model.logits(obs)[0].detach().clone()
    obs[8] += 20  # opposite corner is outside the local communication graph
    torch.testing.assert_close(model.logits(obs)[0], original)
    obs[1] += 20
    assert not torch.allclose(model.logits(obs)[0], original)
    assert len(neighbors(NAMES)[4]) == 4


def test_deployment_actor_matches_batched_actor():
    model = MAPPOModel(NAMES)
    obs = torch.rand(9, 18)
    messages = torch.stack([torch.zeros(18), obs[1], obs[3], torch.zeros(18)])
    torch.testing.assert_close(model.actor_logits(obs[0], messages, 0), model.logits(obs)[0])


def test_gae_bootstraps_truncation_without_crossing_episode():
    reward = torch.tensor([[1.], [100.]])
    values = torch.zeros_like(reward)
    next_values = torch.tensor([[10.], [50.]])
    terminated = torch.tensor([[False], [True]])
    truncated = torch.tensor([[True], [False]])
    advantages, returns = compute_gae(reward, values, next_values,
                                      terminated, truncated, gamma=0.9)
    torch.testing.assert_close(advantages, torch.tensor([[10.], [100.]]))
    torch.testing.assert_close(returns, advantages)


def test_update_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    trainer = MAPPOTrainer(NAMES, config=PPOConfig(update_epochs=1, minibatch_size=2))
    obs = torch.rand(3, 9, 18)
    with torch.no_grad():
        dist = trainer.model.distribution(obs)
        actions = dist.sample()
        values = trainer.model.values(obs)
    rollout = dict(obs=obs, actions=actions, log_probs=dist.log_prob(actions).detach(),
                   rewards=torch.rand(3, 9), values=values, next_values=values,
                   terminated=torch.zeros(3, 9, dtype=torch.bool),
                   truncated=torch.zeros(3, 9, dtype=torch.bool))
    before = [p.detach().clone() for p in trainer.model.parameters()]
    metrics = trainer.update(rollout)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))
    trainer.save(tmp_path / 'model.pt')
    restored = MAPPOTrainer.load(tmp_path / 'model.pt')
    torch.testing.assert_close(restored.model.logits(obs), trainer.model.logits(obs))


def test_independent_ppo_bootstraps_internal_time_limit():
    from agents.independent_ppo.buffer import RolloutBuffer
    buffer = RolloutBuffer(2, 1, gamma=0.9)
    buffer.add(np.zeros(1), 0, 1, 0, 0, True, terminated=False,
               truncated=True, next_value=10)
    buffer.add(np.zeros(1), 0, 100, 0, 0, True, terminated=True)
    buffer.compute_gae(0)
    np.testing.assert_allclose(buffer.advantages, [10, 100])


def test_boundary_ring_has_nine_four_approach_intersections():
    grid = GridBuilder(GridConfig(boundary_ring=True))
    assert len(grid.intersection_nodes) == 9
    for junction in grid.intersection_nodes:
        assert len([e for e in grid.edges if e.to_node == junction]) == 4
        assert len([c for c in grid.connections if c.tl == junction]) == 4


def test_coordination_simulator_timing_and_all_lane_observations():
    from envs.multi_intersection_env import MultiIntersectionEnv
    if not Path('data/coordination/network/grid_coordination.net.xml').exists():
        pytest.skip('Build configs/coordination.yaml before running integration check')
    env = MultiIntersectionEnv('configs/coordination.yaml', episode_duration=30)
    try:
        obs, _ = env.reset()
        assert env._traci.simulation.getTime() == 10
        assert all(v.shape == (50,) for v in obs.values())
        assert all(env.get_num_phases(n) == 6 for n in env.possible_agents)
        env.step({n: 0 for n in env.possible_agents})
        assert env._traci.simulation.getTime() == 20
    finally:
        env.close()
