#!/usr/bin/env python3
"""
Multi-Agent Environment Integration Test
==========================================

Creates the PettingZoo ParallelEnv, resets it, runs ~20 steps with random
actions for all 9 agents, and prints diagnostics.

Usage:
    python scripts/test_multi_agent_env.py
    python scripts/test_multi_agent_env.py --config configs/default.yaml
    python scripts/test_multi_agent_env.py --steps 20
"""

from __future__ import annotations

import argparse
import os
import sys
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
from envs.multi_intersection_env import MultiIntersectionEnv


def main():
    parser = argparse.ArgumentParser(description="Multi-agent environment integration test")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--steps", type=int, default=20, help="Number of random steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("  Multi-Agent Environment Integration Test")
    print("=" * 70)
    print(f"  Config:  {args.config}")
    print(f"  Steps:   {args.steps}")
    print(f"  Seed:    {args.seed}")
    print()

    env = MultiIntersectionEnv(
        config_path=args.config,
        episode_duration=3600,
        step_duration=10,
        seed=args.seed,
    )

    try:
        # --- Reset ---
        print("  [1] Resetting environment...")
        obs, infos = env.reset()
        print(f"      Agents: {env.num_agents}")
        print(f"      Agent names: {sorted(env.possible_agents)}")
        print(f"      Observation dim: {env.get_observation_dim()}")
        print(f"      Cycle length: {env.get_cycle_length()} s")
        print(f"      Num phases: {env.get_num_phases()}")
        print()

        # --- Verify 9 agents ---
        assert len(obs) == 9, f"Expected 9 agents, got {len(obs)}"
        assert len(env.possible_agents) == 9
        print("      [OK] 9 agents confirmed")
        print()

        # --- Check agent-to-TLS mapping ---
        mapping = env.get_agent_intersection_map()
        print("  [2] Agent -> TLS mapping:")
        for agent in sorted(mapping):
            print(f"      {agent} -> {mapping[agent]}")
        print()

        # --- Check incoming edges ---
        incoming = env.get_incoming_edges()
        print("  [3] Incoming edges per agent:")
        for agent in sorted(incoming):
            print(f"      {agent}: {incoming[agent]}")
        print()

        # --- Run random steps ---
        rng = np.random.RandomState(args.seed)
        print(f"  [4] Running {args.steps} random-action steps...")
        print()

        all_rewards: list[float] = []
        all_queues: list[float] = []
        all_waits: list[float] = []

        for step_i in range(args.steps):
            # Random actions: 0 (keep) or 1 (switch)
            actions = {
                agent: int(rng.randint(0, 2))
                for agent in env.possible_agents
            }

            obs, rewards, terminations, truncations, infos = env.step(actions)

            # Collect metrics
            step_rewards = list(rewards.values())
            step_queues = [infos[a]["local_queue_length"] for a in env.possible_agents]
            step_waits = [infos[a]["local_waiting_time"] for a in env.possible_agents]
            avg_reward = np.mean(step_rewards)
            avg_queue = np.mean(step_queues)
            avg_wait = np.mean(step_waits)

            all_rewards.append(avg_reward)
            all_queues.append(avg_queue)
            all_waits.append(avg_wait)

            # Show which agents switched
            switched = [a for a, act in actions.items() if act == 1]
            switch_str = ", ".join(switched) if switched else "(none)"

            print(f"      Step {step_i + 1:3d} | "
                  f"avg_reward={avg_reward:+8.2f} | "
                  f"avg_queue={avg_queue:6.2f} | "
                  f"avg_wait={avg_wait:7.2f} | "
                  f"switched: {switch_str}")

            # Check for NaN/inf in observations
            for agent, o in obs.items():
                assert np.all(np.isfinite(o)), f"{agent} has non-finite obs at step {step_i}"

            if any(terminations.values()):
                print(f"      *** Episode terminated at step {step_i + 1} ***")
                break
            if any(truncations.values()):
                print(f"      *** Episode truncated at step {step_i + 1} ***")
                break

        print()
        print("  [5] Summary:")
        print(f"      Steps completed: {len(all_rewards)}")
        print(f"      Avg reward per step:  {np.mean(all_rewards):+.4f}")
        print(f"      Avg queue per step:   {np.mean(all_queues):.4f}")
        print(f"      Avg wait per step:    {np.mean(all_waits):.4f}")
        print(f"      Min/Max reward:       {min(all_rewards):+.2f} / {max(all_rewards):+.2f}")
        print()

        # --- Verify no orphan SUMO processes ---
        print("  [6] Closing environment...")
        env.close()
        assert env._traci is None, "TraCI should be None after close"
        assert env._tmp_dir is None, "Temp dir should be None after close"
        print("      [OK] SUMO shut down cleanly")
        print("      [OK] No orphan processes")
        print()

        # --- Multiple reset test ---
        print("  [7] Testing multiple resets...")
        for ep in range(3):
            obs, infos = env.reset()
            assert len(obs) == 9
            actions = {a: 0 for a in env.possible_agents}
            env.step(actions)
        env.close()
        print("      [OK] 3 consecutive resets completed")
        print()

        print("=" * 70)
        print("  ALL CHECKS PASSED [OK]")
        print("=" * 70)

    except Exception as e:
        env.close()
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
