"""
Independent PPO — Phase 4
==========================

Each of the 9 signalized intersections has its own independent PPO agent.
There is NO information sharing, NO centralized critic, NO parameter sharing.

Architecture per agent:
    observation (18-dim) → shared feature MLP → actor head (logits) + critic head (V)
"""

from agents.independent_ppo.network import ActorCritic
from agents.independent_ppo.buffer import RolloutBuffer
from agents.independent_ppo.ppo import PPOAlgorithm
from agents.independent_ppo.trainer import IndependentPPOTrainer

__all__ = [
    "ActorCritic",
    "RolloutBuffer",
    "PPOAlgorithm",
    "IndependentPPOTrainer",
]
