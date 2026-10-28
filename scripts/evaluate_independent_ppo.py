#!/usr/bin/env python3
"""
Independent PPO Evaluation Script
===================================

Loads a trained checkpoint and evaluates it deterministically against
the fixed-time baseline using the same SUMO network, traffic demand,
and evaluation metrics.

Usage:
    python scripts/evaluate_independent_ppo.py --checkpoint results/independent_ppo/checkpoints/final
    python scripts/evaluate_independent_ppo.py --checkpoint results/independent_ppo/checkpoints/final --episodes 10
"""

from __future__ import annotations

import argparse
import os
import sys
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

from envs.multi_intersection_env import MultiIntersectionEnv
from agents.independent_ppo.trainer import IndependentPPOTrainer


@torch.no_grad()
def evaluate(
    trainer: IndependentPPOTrainer,
    config_path: str,
    num_episodes: int = 10,
    episode_duration: int = 3600,
    seed: int = 42,
) -> dict:
    """
    Evaluate trained agents deterministically.

    Parameters
    ----------
    trainer : IndependentPPOTrainer
        Trained trainer with loaded checkpoint.
    config_path : str
        Path to environment config.
    num_episodes : int
        Number of evaluation episodes.
    episode_duration : int
        Duration of each episode in seconds.
    seed : int
        Random seed for environment.

    Returns
    -------
    dict
        Evaluation results including per-episode and aggregate metrics.
    """
    agent_names = trainer.agent_names
    all_episode_rewards = {name: [] for name in agent_names}
    all_episode_lengths = {name: [] for name in agent_names}
    env_metrics = []

    for ep in range(num_episodes):
        env = MultiIntersectionEnv(
            config_path=config_path,
            episode_duration=episode_duration,
            seed=seed + ep,
        )
        observations, infos = env.reset()
        episode_done = False
        ep_rewards = {name: 0.0 for name in agent_names}
        ep_lengths = {name: 0 for name in agent_names}

        while not episode_done:
            # Get deterministic actions (greedy: argmax of action probs)
            actions = {}
            for name in agent_names:
                obs = observations[name]
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                trainer.networks[name].eval()
                probs = trainer.networks[name].get_action_probs(obs_tensor)
                actions[name] = probs.argmax(dim=-1).item()

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            for name in agent_names:
                ep_rewards[name] += rewards[name]
                ep_lengths[name] += 1

            dones = {
                name: terminations[name] or truncations[name]
                for name in agent_names
            }
            episode_done = any(dones.values())

        # Parse tripinfo BEFORE env.close() — close() deletes the temp dir
        tripinfo_path = env.get_tripinfo_path()
        ep_metrics = _parse_tripinfo(tripinfo_path)
        env.close()
        ep_metrics["episode"] = ep
        ep_metrics["episode_rewards"] = dict(ep_rewards)
        ep_metrics["episode_lengths"] = dict(ep_lengths)
        env_metrics.append(ep_metrics)

        for name in agent_names:
            all_episode_rewards[name].append(ep_rewards[name])
            all_episode_lengths[name].append(ep_lengths[name])

    # Aggregate results
    results = {
        "num_episodes": num_episodes,
        "episode_duration": episode_duration,
        "seed": seed,
        "network_level": _aggregate_network_metrics(env_metrics),
        "per_agent": {
            name: {
                "mean_reward": float(np.mean(all_episode_rewards[name])),
                "std_reward": float(np.std(all_episode_rewards[name])),
                "mean_episode_length": float(np.mean(all_episode_lengths[name])),
            }
            for name in agent_names
        },
        "per_episode": env_metrics,
    }

    return results


def _sanitize_truncated_tripinfo(content: str) -> str:
    """Repair a truncated tripinfo.xml so ElementTree can parse it.

    SUMO writes tripinfo.xml incrementally; while the simulation is running
    the file may lack the closing ``</tripinfos>`` tag and/or end with a
    partially written ``<tripinfo .../>`` record.  During evaluation we parse
    the file BEFORE ``env.close()`` (which deletes the temp dir), so the file
    is always mid-write and XML-invalid at that moment.

    Repair strategy:
    1. Truncate back to the end of the last complete ``<tripinfo ... />`` record
       (cheap substring scan on the closing ``/ >`` of the last record).
    2. Append the closing ``</tripinfos>`` tag.
    Returns the repaired XML text, guaranteed to end with ``</tripinfos>``.
    """
    last_end = content.rfind("/>")
    if last_end != -1:
        content = content[: last_end + len("/>")]
    stripped = content.rstrip()
    if not stripped.endswith("</tripinfos>"):
        content = stripped + "\n</tripinfos>\n"
    return content


def _parse_tripinfo(tripinfo_path) -> dict:
    """Parse tripinfo.xml for evaluation metrics.
    Handles truncated (mid-write) files, which are normal before SUMO exits.
    """
    import xml.etree.ElementTree as ET

    metrics = {
        "avg_waiting_time": 0.0,
        "total_waiting_time": 0.0,
        "avg_travel_time": 0.0,
        "throughput": 0,
        "vehicles_entered": 0,
    }

    if tripinfo_path is None or not tripinfo_path.exists():
        return metrics

    raw = tripinfo_path.read_text(encoding="utf-8", errors="replace")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        try:
            root = ET.fromstring(_sanitize_truncated_tripinfo(raw))
        except ET.ParseError:
            return metrics

    waiting_times = []
    travel_times = []
    waiting_times = []
    travel_times = []

    for trip in root.findall(".//tripinfo"):
        wt = trip.get("waitingTime")
        dur = trip.get("duration")
        if wt is not None:
            waiting_times.append(float(wt))
        if dur is not None:
            travel_times.append(float(dur))

    if waiting_times:
        metrics["total_waiting_time"] = float(np.sum(waiting_times))
        metrics["avg_waiting_time"] = float(np.mean(waiting_times))
    if travel_times:
        metrics["avg_travel_time"] = float(np.mean(travel_times))
    metrics["throughput"] = len(travel_times)

    return metrics


def _aggregate_network_metrics(env_metrics: list) -> dict:
    """Aggregate metrics across episodes."""
    if not env_metrics:
        return {}

    avg_wait = [m["avg_waiting_time"] for m in env_metrics]
    total_wait = [m["total_waiting_time"] for m in env_metrics]
    avg_travel = [m["avg_travel_time"] for m in env_metrics]
    throughput = [m["throughput"] for m in env_metrics]

    return {
        "avg_waiting_time": float(np.mean(avg_wait)),
        "std_waiting_time": float(np.std(avg_wait)),
        "total_waiting_time": float(np.mean(total_wait)),
        "avg_travel_time": float(np.mean(avg_travel)),
        "std_travel_time": float(np.std(avg_travel)),
        "avg_throughput": float(np.mean(throughput)),
        "min_throughput": int(min(throughput)),
        "max_throughput": int(max(throughput)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate independent PPO agents")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--episode-duration", type=int, default=3600, help="Episode duration (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print("  Independent PPO Evaluation")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Episode duration: {args.episode_duration}s")
    print(f"  Seed: {args.seed}")
    print()

    # Load checkpoint
    trainer = IndependentPPOTrainer.load_checkpoint(args.checkpoint)
    print(f"  Loaded {len(trainer.agent_names)} agents")
    print(f"  Training steps: {trainer.global_step}")
    print(f"  Training updates: {trainer.update_count}")
    print()

    # Evaluate
    results = evaluate(
        trainer,
        config_path=args.config,
        num_episodes=args.episodes,
        episode_duration=args.episode_duration,
        seed=args.seed,
    )

    # Print results
    nl = results["network_level"]
    print("=" * 70)
    print("  Evaluation Results")
    print("=" * 70)
    print(f"  Avg waiting time:      {nl['avg_waiting_time']:.2f} s")
    print(f"  Std waiting time:      {nl['std_waiting_time']:.2f} s")
    print(f"  Avg travel time:       {nl['avg_travel_time']:.2f} s")
    print(f"  Avg throughput:        {nl['avg_throughput']:.0f} vehicles")
    print(f"  Throughput range:      {nl['min_throughput']} - {nl['max_throughput']}")
    print()

    print("  Per-Agent Results:")
    for name, agent_r in results["per_agent"].items():
        print(
            f"    {name}: reward={agent_r['mean_reward']:+.2f} ± {agent_r['std_reward']:.2f}, "
            f"ep_len={agent_r['mean_episode_length']:.0f}"
        )
    print()

    # Baseline comparison
    baseline_wait = 18.94  # Reference from Phase 2
    baseline_travel = 47.35
    if nl["avg_throughput"] <= 0:
        print("  Baseline Comparison: SKIPPED — no completed vehicles parsed;")
        print("                         comparison would be meaningless versus zero.")
        print()
        output_path = args.output or str(
            Path(args.checkpoint).parent.parent / "evaluation_results.json"
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {output_path}")
        print("=" * 70)
        return
    print("  Baseline Comparison (Phase 2 fixed-time):")
    print(f"    Baseline avg wait:    {baseline_wait:.2f} s")
    print(f"    PPO avg wait:         {nl['avg_waiting_time']:.2f} s")
    diff = nl["avg_waiting_time"] - baseline_wait
    if diff < 0:
        print(f"    Improvement:          {abs(diff):.2f} s ({abs(diff)/baseline_wait*100:.1f}% better)")
    elif diff > 0:
        print(f"    Regression:           {diff:.2f} s ({diff/baseline_wait*100:.1f}% worse)")
    else:
        print(f"    Same performance")
    print()

    # Save results
    output_path = args.output or str(
        Path(args.checkpoint).parent.parent / "evaluation_results.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
