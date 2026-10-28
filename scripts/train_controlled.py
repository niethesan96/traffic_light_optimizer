"""Train both algorithms with identical episode demand schedules and budgets."""
from pathlib import Path
import argparse
from dataclasses import asdict
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from agents.mappo.trainer import MAPPOTrainer
from agents.independent_ppo.trainer import IndependentPPOTrainer
from agents.independent_ppo.ppo import PPOConfig
from envs.multi_intersection_env import MultiIntersectionEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm', choices=['mappo', 'independent'], required=True)
    p.add_argument('--config', default='configs/cooperative.yaml')
    p.add_argument('--steps', type=int, default=36000)
    p.add_argument('--rollout-steps', type=int, default=360)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--demand-seed-start', type=int, default=2000)
    p.add_argument('--initialize-from', help='Independent PPO pretraining checkpoint shared by both algorithms')
    p.add_argument('--resume',help='Continue this algorithm checkpoint for the additional --steps budget')
    p.add_argument('--team-weight', type=float, default=0.25)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.resume and a.initialize_from:
        p.error('--resume and --initialize-from are mutually exclusive')
    if a.steps <= 0 or a.rollout_steps <= 0 or a.steps % a.rollout_steps:
        p.error('--steps must be a positive multiple of --rollout-steps')
    if not 0 <= a.team_weight <= 1:
        p.error('--team-weight must be in [0, 1]')
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    raw = yaml.safe_load(Path(a.config).read_text())
    cfg = PPOConfig(**{k: v for k, v in raw.get('ppo', {}).items()
                      if k in PPOConfig.__dataclass_fields__})
    env = MultiIntersectionEnv(a.config, seed=a.seed, demand_seed=a.demand_seed_start)
    names, dim = env.possible_agents, env.get_observation_dim()
    pretrained = IndependentPPOTrainer.load_checkpoint(a.initialize_from) if a.initialize_from else None
    if pretrained and (pretrained.agent_names != names or pretrained.obs_dim != dim):
        raise ValueError('Pretraining checkpoint does not match scenario')
    if a.algorithm == 'mappo':
        trainer = MAPPOTrainer.load(a.resume) if a.resume else MAPPOTrainer(names, dim, cfg, model_version=2)
        if pretrained:
            for name, expert in zip(names, trainer.model.local_experts):
                expert.load_state_dict(pretrained.networks[name].state_dict())
    else:
        trainer = (IndependentPPOTrainer.load_checkpoint(a.resume) if a.resume else
                   IndependentPPOTrainer(names, dim, config=cfg, rollout_steps=a.rollout_steps))
        if pretrained:
            for name in names:
                trainer.networks[name].load_state_dict(pretrained.networks[name].state_dict())
    if trainer.agent_names!=names or trainer.obs_dim!=dim:
        raise ValueError('Checkpoint does not match environment')
    starting_steps=(trainer.global_step if a.algorithm=='mappo' else trainer.global_step//len(names))
    episode=0
    if a.resume:
        parent=Path(a.resume).parent
        previous_logs=parent/'training_logs.json'
        if previous_logs.exists():
            episode=int(json.loads(previous_logs.read_text())[-1]['episode'])+1
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(vars(a), scenario=raw, ppo=asdict(cfg),
                    pretraining_agent_transitions=pretrained.global_step if pretrained else 0,
                    starting_joint_steps=starting_steps,starting_episode=episode,
                    algorithm_budget_unit='joint_environment_decisions')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    obs, _ = env.reset(options={'demand_seed':a.demand_seed_start+episode})
    logs, storage = [], []
    started = time.perf_counter()
    try:
        for step in range(1, a.steps + 1):
            if a.algorithm == 'mappo':
                state = trainer.tensor(obs)
                with torch.no_grad():
                    dist = trainer.model.distribution(state)
                    action = dist.sample()
                    values = trainer.model.values(state)
                    log_probs = dist.log_prob(action)
                actions = dict(zip(names, action.tolist()))
            else:
                actions = trainer.get_actions(obs)
            following, rewards, terminated, truncated, infos = env.step(actions)
            if a.algorithm == 'mappo':
                r = torch.tensor([rewards[n] for n in names])
                r = (1-a.team_weight)*r + a.team_weight*r.mean()
                with torch.no_grad():
                    next_values = trainer.model.values(trainer.tensor(following))
                storage.append(dict(obs=state, actions=action, log_probs=log_probs,
                    values=values, next_values=next_values, rewards=r,
                    terminated=torch.tensor([terminated[n] for n in names]),
                    truncated=torch.tensor([truncated[n] for n in names])))
                trainer.global_step = starting_steps+step
            else:
                done = {n: terminated[n] or truncated[n] for n in names}
                trainer.store_transition(obs, actions, rewards, done,
                    terminations=terminated, truncations=truncated, next_observations=following)
            obs = following
            if step % a.rollout_steps == 0:
                if a.algorithm == 'mappo':
                    rollout = {k: torch.stack([r[k] for r in storage]) for k in storage[0]}
                    record = trainer.update(rollout)
                    storage.clear()
                    trainer.save(output / 'latest.pt')
                else:
                    losses = trainer.compute_gae_and_update(obs)
                    record = {k: float(np.mean([getattr(v, k) for v in losses.values()]))
                              for k in ('policy_loss', 'value_loss', 'entropy', 'approx_kl')}
                    trainer.save_checkpoint(output / 'latest', config=dict(joint_steps=starting_steps+step, episode_index=episode))
                record.update(joint_steps=starting_steps+step, additional_steps=step,
                              episode=episode, wall_seconds=time.perf_counter()-started)
                logs.append(record)
                (output / 'training_logs.json').write_text(json.dumps(logs, indent=2))
                print(json.dumps(record), flush=True)
            if any(terminated.values()) or any(truncated.values()):
                episode += 1
                obs, _ = env.reset(options={'demand_seed': a.demand_seed_start+episode})
        if a.algorithm == 'mappo':
            trainer.save(output / 'final.pt')
        else:
            trainer.save_checkpoint(output / 'final', config=dict(joint_steps=starting_steps+a.steps, episode_index=episode))
    finally:
        env.close()


if __name__ == '__main__':
    main()
