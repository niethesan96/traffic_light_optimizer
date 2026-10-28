"""Train MAPPO on the same SUMO scenario as independent PPO."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from agents.independent_ppo.ppo import PPOConfig
from agents.mappo.trainer import MAPPOTrainer
from envs.multi_intersection_env import MultiIntersectionEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/coordination.yaml')
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--rollout-steps', type=int, default=360)
    parser.add_argument('--episode-duration', type=int, default=3600)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='results/mappo')
    parser.add_argument('--resume')
    args = parser.parse_args()
    if min(args.timesteps, args.rollout_steps, args.episode_duration) <= 0:
        parser.error('step counts and duration must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = PPOConfig(**{k: v for k, v in raw.get('ppo', {}).items()
                       if k in PPOConfig.__dataclass_fields__})
    env = MultiIntersectionEnv(args.config, seed=args.seed, episode_duration=args.episode_duration)
    trainer = (MAPPOTrainer.load(args.resume) if args.resume else
               MAPPOTrainer(env.possible_agents, env.get_observation_dim(), cfg))
    if trainer.agent_names != env.possible_agents or trainer.model.obs_dim != env.get_observation_dim():
        raise ValueError('Checkpoint does not match environment')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / 'training_logs.json'
    logs = json.loads(log_path.read_text()) if args.resume and log_path.exists() else []
    manifest = dict(vars(args), algorithm='native_pytorch_mappo',
                    communication='one-hop directional neighbor messages',
                    ppo_config=vars(trainer.config), scenario=raw)
    (output / 'run_config.json').write_text(json.dumps(manifest, indent=2))
    try:
        obs, _ = env.reset(seed=args.seed)
        while trainer.global_step < args.timesteps:
            rollout, obs = trainer.collect(env, obs, min(args.rollout_steps,
                                         args.timesteps - trainer.global_step))
            record = trainer.update(rollout)
            logs.append(record)
            print(json.dumps(record), flush=True)
            trainer.save(output / 'latest.pt')
            log_path.write_text(json.dumps(logs, indent=2))
        trainer.save(output / 'final.pt')
    finally:
        env.close()


if __name__ == '__main__':
    main()
