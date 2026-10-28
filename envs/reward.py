"""
Reward Function
================

Local reward for each agent (intersection) computed from the change in
traffic conditions between consecutive simulation steps.

Mathematical Definition
------------------------

.. math::

    r_i = \\alpha \\cdot (W_{i}^{t-1} - W_{i}^{t})
        + \\beta  \\cdot (Q_{i}^{t-1} - Q_{i}^{t})

where

    * :math:`r_i`        – reward for intersection *i*
    * :math:`W_{i}^{t}`  – total waiting time of vehicles on incoming
      lanes of intersection *i* at step *t*
    * :math:`Q_{i}^{t}`  – total queue length (halting vehicles) on
      incoming lanes of intersection *i* at step *t*
    * :math:`\\alpha`     – waiting-time weight (default **1.0**)
    * :math:`\\beta`      – queue-length weight  (default **1.0**)

Interpretation
--------------
* **Positive reward** → the intersection *improved* (less waiting / shorter
  queues).
* **Negative reward** → the intersection *worsened*.
* **Zero** → no change.

Properties
----------
* Computed independently for each intersection.
* Uses only local information (no cross-intersection data).
* Does not use future information.
* Reproducible given the same state sequence.

Configuration
-------------
Weights can be set via the ``reward_weights`` section of the YAML config:

    reward_weights:
      waiting_weight: 1.0
      queue_weight: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    """Configurable reward weights."""

    waiting_weight: float = 1.0   # α
    queue_weight: float = 1.0     # β
    # Reward scaling applied AFTER the weighted sum; set reward_scale ≈
    # 1/σ(step reward) to standardise the critic's input range, then clip to
    # ±reward_clip (None = no clipping) so outlier steps cannot dominate a
    # PPO minibatch.
    reward_scale: float = 1.0
    reward_clip: float | None = None
    mode: str = "difference"


def compute_reward(
    prev_waiting_time: float,
    current_waiting_time: float,
    prev_queue_length: int,
    current_queue_length: int,
    config: RewardConfig | None = None,
) -> float:
    """
    Compute the scalar reward for one intersection.

    Parameters
    ----------
    prev_waiting_time : float
        Total waiting time at the previous step.
    current_waiting_time : float
        Total waiting time at the current step.
    prev_queue_length : int
        Total queue length at the previous step.
    current_queue_length : int
        Total queue length at the current step.
    config : RewardConfig, optional
        Weight configuration.  Defaults to equal weights.

    Returns
    -------
    float
        Scalar reward.  Positive = improvement.
    """
    if config is None:
        config = RewardConfig()

    delta_wait = prev_waiting_time - current_waiting_time
    delta_queue = float(prev_queue_length - current_queue_length)

    reward = config.waiting_weight * delta_wait + config.queue_weight * delta_queue
    if config.mode == "queue_cost":
        reward = -config.queue_weight * float(current_queue_length)
    elif config.mode != "difference":
        raise ValueError(f"Unknown reward mode: {config.mode}")

    # Scale, then clip (clip bounds are in the SCALED units).
    reward = reward * config.reward_scale
    if config.reward_clip is not None:
        reward = float(np.clip(reward, -config.reward_clip, config.reward_clip))
    return reward


def reward_from_config(yaml_config: dict) -> RewardConfig:
    """
    Build a ``RewardConfig`` from a YAML config dict.

    Looks for ``reward_weights`` key; missing keys fall back to defaults.
    """
    rw = yaml_config.get("reward_weights", {})
    return RewardConfig(
        waiting_weight=float(rw.get("waiting_weight", 1.0)),
        queue_weight=float(rw.get("queue_weight", 1.0)),
        reward_scale=float(rw.get("reward_scale", 1.0)),
        reward_clip=(
            float(rw["reward_clip"]) if rw.get("reward_clip") is not None else None
        ),
        mode=rw.get("mode", "difference"),
    )
