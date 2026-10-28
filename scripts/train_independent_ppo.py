#!/usr/bin/env python3
"""
Independent PPO Training Script
================================

Trains 9 independent PPO agents — one per intersection.
Each agent has its own ActorCritic network with NO parameter sharing.

Usage:
    python scripts/train_independent_ppo.py
    python scripts/train_independent_ppo.py --config configs/default.yaml --timesteps 10000
    python scripts/train_independent_ppo.py --timesteps 500 --episode-duration 120
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
from pathlib import Path

# Add project root and SUMO tools to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

_sumo_tools = str(_project_root / "venv" / "Lib" / "site-packages" / "sumo" / "tools")
if os.path.isdir(_sumo_tools) and _sumo_tools not in sys.path:
    sys.path.insert(0, _sumo_tools)

try:
    import sumo as _sumo_pkg
    os.environ.setdefault("SUMO_HOME", str(Path(_sumo_pkg.SUMO_HOME)))
except Exception:
    pass

import numpy as np
import torch
import yaml

from envs.multi_intersection_env import MultiIntersectionEnv
from agents.independent_ppo.trainer import IndependentPPOTrainer, TrainingLogger
from agents.independent_ppo.ppo import PPOConfig


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train independent PPO agents")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="CPU threads for small policy networks (default: 1)")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--timesteps", type=int, default=500, help="Total environment timesteps")
    parser.add_argument("--episode-duration", type=int, default=None,
                        help="Episode duration in seconds (overrides config)")
    parser.add_argument("--rollout-steps", type=int, default=None,
                        help="Steps per rollout before PPO update "
                             "(default: configs <ppo>.rollout_steps, else 360)")
    parser.add_argument("--minibatch-size", type=int, default=None,
                        help="PPO minibatch size "
                             "(default: configs <ppo>.minibatch_size, else 64)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate "
                             "(default: configs <ppo>.learning_rate, else 3e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value-loss-coeff", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--update-epochs", type=int, default=None,
                        help="PPO update epochs "
                             "(default: configs <ppo>.update_epochs, else 8)")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="results/independent_ppo", help="Output directory")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N timesteps")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint directory to resume from: restores all "
                             "agent networks, optimizers and global_step. "
                             "Hyperparameters come from the checkpoint metadata; "
                             "pass a larger --timesteps than the original run.")
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)

    # --- Resolve PPO hyperparameters: CLI flag > YAML <ppo> key > new default ---
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    ppo_yaml = yaml_cfg.get("ppo", {}) or {}

    def _resolve(cli_value, yaml_key, default):
        return cli_value if cli_value is not None else ppo_yaml.get(yaml_key, default)

    rollout_steps = _resolve(args.rollout_steps, "rollout_steps", 360)
    minibatch_size = _resolve(args.minibatch_size, "minibatch_size", 64)
    update_epochs = _resolve(args.update_epochs, "update_epochs", 8)
    learning_rate = _resolve(args.learning_rate, "learning_rate", 3e-4)

    # --- Setup ---
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Create PPO config ---
    ppo_config = PPOConfig(
        learning_rate=learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coefficient=args.entropy_coeff,
        value_loss_coefficient=args.value_loss_coeff,
        update_epochs=update_epochs,
        hidden_size=args.hidden_size,
        minibatch_size=minibatch_size,
    )

    # --- Create environment ---
    env_kwargs = {"config_path": args.config, "seed": args.seed}
    if args.episode_duration is not None:
        env_kwargs["episode_duration"] = args.episode_duration

    # --- Resume (Stage 4): restore networks, optimizers, global_step ---
    episode_index = 0
    if args.resume is not None:
        import json as _json
        meta_path = Path(args.resume) / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Not a training checkpoint directory (missing metadata.json): {args.resume}"
            )
        with open(meta_path, "r") as fh:
            _meta = _json.load(fh)
        episode_index = int(_meta.get("extra_config", {}).get("episode_index", 0))

    print("=" * 70)
    print("  Independent PPO Training")
    print("=" * 70)
    print(f"  Config:          {args.config}")
    print(f"  Target timesteps: {args.timesteps}")
    print(f"  Rollout steps:   {rollout_steps}")
    print(f"  Minibatch size:  {minibatch_size}")
    print(f"  Update epochs:   {update_epochs}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Episode duration: {env_kwargs.get('episode_duration', 'default')}s")
    print(f"  Seed:            {args.seed}")
    print(f"  Hidden size:     {args.hidden_size}")
    print()

    # --- Create trainer ---
    # We need to know agent names and obs_dim, so create a dummy env first
    dummy_env = MultiIntersectionEnv(**env_kwargs)
    agent_names = list(dummy_env.possible_agents)
    obs_dim = dummy_env.get_observation_dim()
    num_actions = 2  # KEEP / SWITCH
    dummy_env.close()

    print(f"  Agents: {len(agent_names)}")
    print(f"  Obs dim: {obs_dim}")
    print(f"  Num actions: {num_actions}")
    print()

    if args.resume is not None:
        # Restore full training state (networks + optimizers + counters)
        trainer = IndependentPPOTrainer.load_checkpoint(
            args.resume, device=torch.device("cpu")
        )
        if list(trainer.agent_names) != agent_names:
            raise ValueError(
                f"Checkpoint agents {trainer.agent_names} do not match the "
                f"environment agents {agent_names} — wrong config for resume?"
            )
        if trainer.obs_dim != obs_dim:
            raise ValueError(
                f"Checkpoint obs_dim {trainer.obs_dim} != environment "
                f"obs_dim {obs_dim}"
            )
        global_timestep_start = trainer.global_step
        print(f"  Resumed from:    {args.resume}")
        print(f"  global_step:     {trainer.global_step}")
        print(f"  update_count:    {trainer.update_count}")
        print()
    else:
        trainer = IndependentPPOTrainer(
            agent_names=agent_names,
            obs_dim=obs_dim,
            num_actions=num_actions,
            config=ppo_config,
            rollout_steps=rollout_steps,
            device=torch.device("cpu"),
            seed=args.seed,
        )
        global_timestep_start = 0

    # Count parameters
    total_params = sum(net.count_parameters() for net in trainer.networks.values())
    per_agent_params = list(trainer.networks.values())[0].count_parameters()
    print(f"  Parameters per agent: {per_agent_params:,}")
    print(f"  Total parameters (9 agents): {total_params:,}")
    print()

    # --- Training loop ---
    logger = TrainingLogger(output_dir)
    env = None
    global_timestep = global_timestep_start
    episodes_completed = 0
    episode_reward_accum: dict[str, float] = {name: 0.0 for name in agent_names}
    returns_since_last_log: list[float] = []     # one entry per completed episode
    step_reward_sum: dict[str, float] = {name: 0.0 for name in agent_names}
    step_reward_count = 0
    start_time = time.time()

    try:
        while global_timestep < args.timesteps:
            # Start new episode — vary the demand seed per episode
            # (training demand seeds 1000+; the SUMO sim seed stays args.seed)
            episode_env_kwargs = dict(env_kwargs)
            episode_env_kwargs["demand_seed"] = 1000 + episode_index
            env = MultiIntersectionEnv(**episode_env_kwargs)
            observations, infos = env.reset()
            episode_done = False
            episode_reward_accum = {name: 0.0 for name in agent_names}

            while not episode_done and global_timestep < args.timesteps:
                # Collect actions from all agents
                actions = trainer.get_actions(observations)

                # Step environment
                new_observations, rewards, terminations, truncations, infos = env.step(actions)

                # Check if episode ended
                dones = {
                    name: terminations[name] or truncations[name]
                    for name in agent_names
                }
                episode_done = any(dones.values())

                # Accumulate step rewards (for per-agent mean step reward)
                for name in agent_names:
                    episode_reward_accum[name] += rewards[name]
                    step_reward_sum[name] += rewards[name]
                step_reward_count += 1

                # Store transitions (terminated vs truncated kept separate
                # so truncated final steps get bootstrapped, not zeroed)
                trainer.store_transition(
                    observations, actions, rewards, dones,
                    terminations=terminations, truncations=truncations,
                    next_observations=new_observations,
                )

                observations = new_observations
                global_timestep += 1

                # A completed episode: freeze its per-agent return now
                if episode_done:
                    episodes_completed += 1
                    episode_index += 1
                    returns_since_last_log.append(
                        float(np.mean(list(episode_reward_accum.values())))
                    )

                # PPO update when buffer is full
                if trainer.should_update():
                    losses = trainer.compute_gae_and_update(observations)

                    # Log statistics — one CSV row per update, flushed immediately
                    stats = trainer.get_statistics()
                    avg_policy_loss = float(np.mean([l.policy_loss for l in losses.values()]))
                    avg_value_loss = float(np.mean([l.value_loss for l in losses.values()]))
                    avg_entropy = float(np.mean([l.entropy for l in losses.values()]))
                    avg_kl = float(np.mean([l.approx_kl for l in losses.values()]))
                    avg_clip = float(np.mean([l.clip_fraction for l in losses.values()]))
                    evs = [l.explained_variance for l in losses.values()]
                    avg_ev = float(np.nanmean(evs)) if any(not np.isnan(e) for e in evs) else float("nan")

                    if returns_since_last_log:
                        mean_return_since_last_log = float(np.mean(returns_since_last_log))
                    else:
                        mean_return_since_last_log = float("nan")

                    if step_reward_count > 0:
                        mean_step_reward = {
                            name: step_reward_sum[name] / step_reward_count
                            for name in agent_names
                        }
                    else:
                        mean_step_reward = {name: float("nan") for name in agent_names}

                    log_entry = {
                        "global_step": stats["global_step"],
                        "update_count": stats["update_count"],
                        "episodes_completed": episodes_completed,
                        "mean_return_since_last_log": mean_return_since_last_log,
                        "mean_episode_reward": stats["mean_episode_reward"],
                        "mean_episode_length": stats["mean_episode_length"],
                        "policy_loss": avg_policy_loss,
                        "value_loss": avg_value_loss,
                        "entropy": avg_entropy,
                        "approx_kl": avg_kl,
                        "clip_fraction": avg_clip,
                        "explained_variance": avg_ev,
                        "mean_step_reward": mean_step_reward,
                        "wall_time_s": time.time() - start_time,
                    }
                    logger.log(log_entry)

                    # Reset per-update accumulators
                    returns_since_last_log = []
                    step_reward_sum = {name: 0.0 for name in agent_names}
                    step_reward_count = 0

                    if stats["update_count"] % 10 == 0:
                        logger.write_json()
                        elapsed = time.time() - start_time
                        print(
                            f"  Step {stats['global_step']:6d} | "
                            f"Update {stats['update_count']:3d} | "
                            f"Reward {stats['mean_episode_reward']:+8.2f} | "
                            f"pi loss {avg_policy_loss:+.4f} | "
                            f"V loss {avg_value_loss:.4f} | "
                            f"Ent {avg_entropy:.4f} | "
                            f"KL {avg_kl:.4f} | "
                            f"Time {elapsed:.1f}s"
                        )

                # Checkpoint (records episode_index for demand-seed continuity)
                if global_timestep % args.checkpoint_interval == 0 and global_timestep > 0:
                    trainer.save_checkpoint(
                        checkpoint_dir / f"step_{global_timestep}",
                        config={
                            "timesteps": args.timesteps,
                            "seed": args.seed,
                            "episode_index": episode_index,
                        },
                    )

            # Close environment after episode
            if env is not None:
                env.close()
                env = None

    except KeyboardInterrupt:
        print("\n  Training interrupted by user.")
    finally:
        if env is not None:
            env.close()
        # Persist logs even on interrupt / crash
        try:
            logger.write_json()
        finally:
            logger.close()

    # Save final checkpoint
    final_dir = checkpoint_dir / "final"
    trainer.save_checkpoint(
        final_dir,
        config={"timesteps": args.timesteps, "seed": args.seed, "episode_index": episode_index},
    )

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("  Training Complete")
    print("=" * 70)
    print(f"  Total timesteps: {global_timestep}")
    print(f"  Total updates: {trainer.update_count}")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {output_dir / 'training_logs.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
