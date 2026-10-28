"""
Observation Space
==================

Defines the observation space for each agent in the multi-agent
traffic-signal environment.

Observation Vector Layout (per intersection)
---------------------------------------------

Every agent receives an identical, fixed-size flat vector.  For a
3×3 grid with 1 lane per direction the vector has **18** elements:

    Index  Field                     Description
    ─────  ────────────────────────  ──────────────────────────────────────
      0    current_phase              Active TLS phase index (0–5)
      1    time_in_phase              Seconds since the phase started
      2    queue_length[0]            Halting vehicles on incoming lane 0
      3    queue_length[1]            Halting vehicles on incoming lane 1
      4    queue_length[2]            Halting vehicles on incoming lane 2
      5    queue_length[3]            Halting vehicles on incoming lane 3
      6    waiting_vehicles[0]        Waiting vehicles on incoming lane 0
      7    waiting_vehicles[1]        Waiting vehicles on incoming lane 1
      8    waiting_vehicles[2]        Waiting vehicles on incoming lane 2
      9    waiting_vehicles[3]        Waiting vehicles on incoming lane 3
     10    vehicle_count[0]           Vehicles on incoming lane 0
     11    vehicle_count[1]           Vehicles on incoming lane 1
     12    vehicle_count[2]           Vehicles on incoming lane 2
     13    vehicle_count[3]           Vehicles on incoming lane 3
     14    mean_speed[0]              Mean speed on incoming lane 0 (m/s)
     15    mean_speed[1]              Mean speed on incoming lane 1 (m/s)
     16    mean_speed[2]              Mean speed on incoming lane 2 (m/s)
     17    mean_speed[3]              Mean speed on incoming lane 3 (m/s)

Dimension formula
-----------------
    dim = 2 + (num_incoming_edges × num_lanes) × 4

For the default 3×3 grid: 2 + (4 × 1) × 4 = **18**

Note
----
* Observations are **strictly local** — no information from neighbouring
  intersections is included.  This is intentional for the CTDE architecture
  where the critic will receive global information during training, but the
  actor only sees its own local state.
* Fixed per-dim scaling (``scale_observation``) is applied EXACTLY ONCE, at
  the end of ``envs/single_intersection.py::collect_observation`` — the
  vector the networks ever see is already scaled.  The constants are chosen
  from measured network_wide maxima so that most values land in [0, 1] and
  every dim is clipped to [0, 5].
"""

from __future__ import annotations

from gymnasium import spaces
import numpy as np


# Default incoming edges per intersection in a 3×3 grid with 1 lane/dir
DEFAULT_NUM_INCOMING_EDGES = 4
DEFAULT_NUM_LANES = 1

# --- Fixed per-dim scaling constants (applied in ONE place) --------------
# Divisors chosen from network_wide random-policy maxima:
#   phase up to 5 → 5/3 ≈ 1.67; time_in_phase now ≤ ~60 s; queues/waiting
#   up to ~7; counts up to ~15 lanes; speeds up to ~9 m/s.
OBS_SCALE_PHASE = 3.0
OBS_SCALE_TIME_IN_PHASE = 60.0
OBS_SCALE_QUEUE = 8.0
OBS_SCALE_WAITING = 8.0
OBS_SCALE_COUNT = 15.0
OBS_SCALE_SPEED = 12.0
OBS_CLIP_MAX = 5.0
OBS_CLIP_MIN = 0.0


def observation_dim(
    num_incoming_edges: int = DEFAULT_NUM_INCOMING_EDGES,
    num_lanes: int = DEFAULT_NUM_LANES,
) -> int:
    """Return the dimensionality of the observation vector."""
    total_lanes = num_incoming_edges * num_lanes
    return 2 + total_lanes * 4


def observation_space(
    num_incoming_edges: int = DEFAULT_NUM_INCOMING_EDGES,
    num_lanes: int = DEFAULT_NUM_LANES,
) -> spaces.Box:
    """
    Build a ``gymnasium.spaces.Box`` for one agent's observation.

    Returns
    -------
    spaces.Box
        Float32 box of shape ``(dim,)`` with lower bound 0 and upper bound
        ``np.inf`` (unbounded above for speed / count values).
    """
    dim = observation_dim(num_incoming_edges, num_lanes)
    low = np.zeros(dim, dtype=np.float32)
    high = np.full(dim, np.inf, dtype=np.float32)

    # Phase index is discrete [0, 5] — keep low=0, high=inf for simplicity
    # since the Box space doesn't enforce integrality.
    return spaces.Box(low=low, high=high, dtype=np.float32)


def scale_observation(
    obs: list[float] | np.ndarray, num_lanes_total: int | None = None
) -> list[float]:
    """Apply the fixed per-dim scaling and clip every dim to [0, 5].

    Layout of the unscaled vector (see module docstring):
        [0]            phase          ÷ 3
        [1]            time_in_phase  ÷ 60
        [2 : 2+L]      queue lengths  ÷ 8
        [2+L : 2+2L]   waiting count  ÷ 8
        [2+2L : 2+3L]  vehicle count  ÷ 15
        [2+3L : 2+4L]  mean speed     ÷ 12
    where ``L`` = number of lanes (incoming edges × lanes). When the vector
    length does not fit this layout (e.g. padding), remaining values fall
    back to the queue divisor so every dim ends up in [0, 5].

    This is the ONLY place observation scaling happens.
    """
    arr = np.asarray(obs, dtype=np.float64)
    if arr.size == 0:
        return []

    if num_lanes_total is None:
        num_lanes_total = max(0, (arr.size - 2) // 4)

    divisors = np.empty(arr.size, dtype=np.float64)
    divisors[0] = OBS_SCALE_PHASE
    if arr.size > 1:
        divisors[1] = OBS_SCALE_TIME_IN_PHASE
    if num_lanes_total > 0 and arr.size > 2:
        # Block layout: [queue]*L, [waiting]*L, [count]*L, [speed]*L
        lane_divisors = np.repeat(
            np.array(
                [OBS_SCALE_QUEUE, OBS_SCALE_WAITING,
                 OBS_SCALE_COUNT, OBS_SCALE_SPEED],
                dtype=np.float64,
            ),
            num_lanes_total,
        )
        n_fit = min(arr.size - 2, lane_divisors.size)
        divisors[2:2 + n_fit] = lane_divisors[:n_fit]
        divisors[2 + n_fit:] = OBS_SCALE_QUEUE  # padding (zeros stay zero)

    scaled = arr / divisors
    scaled = np.clip(scaled, OBS_CLIP_MIN, OBS_CLIP_MAX)
    return scaled.tolist()
