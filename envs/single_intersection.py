"""
Single Intersection Logic
=========================

Reusable logic for controlling and observing one SUMO traffic-light
intersection within a shared simulation.  Does NOT start its own SUMO
instance — it operates through a live TraCI connection managed by the
multi-agent environment.

Observation vector (17 floats per intersection):
    [0]     current_phase          – integer phase index (0–5)
    [1]     time_in_phase          – seconds since the current phase started
    [2–5]   queue_length × 4       – halting vehicles per incoming lane
    [6–9]   waiting_vehicles × 4   – waiting vehicles per incoming lane
    [10–13] vehicle_count × 4      – vehicles on each incoming lane
    [14–17] mean_speed × 4         – mean speed on each incoming lane

Action space:
    0 = keep current phase
    1 = switch to next phase (subject to min-green constraint)

Reward:
    reward = Δwaiting_time  +  Δqueue_length
    (positive = improvement, i.e. waiting / queue decreased)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from envs.observation import scale_observation


# ---------------------------------------------------------------------------
# Phase information
# ---------------------------------------------------------------------------

# Legacy constants kept for backward-compatible tests.
# The environment dynamically detects phases from SUMO at runtime,
# so these are NOT used for actual TLS control.
GREEN_PHASES = {0, 3}          # legacy: indices of green phases (6-phase cycle)
NUM_PHASES = 6                 # legacy: total number of signal phases
DEFAULT_MIN_GREEN_TIME = 10.0  # seconds


@dataclass
class IntersectionState:
    """Mutable state tracked for one intersection between steps."""

    tls_id: str                              # SUMO TLS id, e.g. "N_1_1"
    incoming_edges: list[str]                # edges feeding INTO this intersection
    num_lanes: int = 1                       # lanes per incoming edge

    # — updated each step —
    current_phase: int = 0
    time_in_phase: float = 0.0
    prev_waiting_time: float = 0.0           # total local waiting time last step
    prev_queue_length: int = 0               # total local queue last step
    last_phase_switch_time: float = 0.0      # sim time at which the CURRENT phase began
    last_observed_phase: int | None = None   # phase seen by collect_observation last step
    # — dynamic TLS info (detected from SUMO at runtime) —
    num_phases: int = 6          # total phases in the loaded TLS program
    green_phases: set[int] = field(default_factory=lambda: {0, 3})
    tls_detected: bool = False   # True once we've queried SUMO for phase info

    # — per-step observation scratch —
    queue_lengths: list[int] = field(default_factory=list)
    waiting_vehicles: list[int] = field(default_factory=list)
    vehicle_counts: list[int] = field(default_factory=list)
    mean_speeds: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dynamic TLS detection
# ---------------------------------------------------------------------------

def detect_tls_phases(traci, state: IntersectionState) -> None:
    """
    Query SUMO for the actual TLS program loaded at this intersection.

    Populates ``state.num_phases`` and ``state.green_phases`` so that
    ``apply_action`` works with the real program rather than hardcoded
    constants.

    A phase is considered "green" if its state string contains at least
    one uppercase ``G`` character (SUMO convention for green signal).
    """
    if state.tls_detected:
        return
    try:
        program = traci.trafficlight.getCompleteRedYellowGreenDefinition(state.tls_id)
        # program is a list of TrafficLightProgram objects;
        # each has a .phases attribute list of phase objects.
        # Each phase object has .state (str) like "GrGr".
        if program and hasattr(program, '__len__') and len(program) > 0:
            first = program[0]
            if hasattr(first, 'phases') and first.phases:
                phases = first.phases
                n = int(len(phases))
                if n > 0:
                    state.num_phases = n
                    state.green_phases = set()
                    for i, phase in enumerate(phases):
                        if hasattr(phase, 'state') and isinstance(phase.state, str):
                            # A phase is green if ANY link has uppercase 'G'
                            if any(ch == 'G' for ch in phase.state):
                                state.green_phases.add(i)
        # Fallback: use getPhaseCount if program parsing didn't work
        if state.num_phases <= 0:
            count = traci.trafficlight.getPhaseCount(state.tls_id)
            if isinstance(count, int) and count > 0:
                state.num_phases = count
                # Heuristic: if only 1 phase, it's always green
                if count <= 1:
                    state.green_phases = {0}
                else:
                    state.green_phases = set()
    except Exception:
        pass  # Keep defaults
    state.tls_detected = True


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def collect_observation(
    traci,
    state: IntersectionState,
    sim_time: float,
    max_dim: int | None = None,
) -> list[float]:
    """
    Collect a fixed-size observation vector for *one* intersection.

    Must be called **after** the simulation step has advanced.

    All observations are padded (zero-filled) to ``max_dim`` so that
    every agent receives the same length vector regardless of its
    number of incoming edges.

    Parameters
    ----------
    traci : module
        Live TraCI module.
    state : IntersectionState
        Mutable state for this intersection.
    sim_time : float
        Current SUMO simulation time (seconds).
    max_dim : int, optional
        Target observation length.  If ``None``, uses the natural length
        (no padding).

    Returns
    -------
    list[float]
        Observation vector of length ``max_dim`` (or natural length).
    """
    # --- Current phase ---
    try:
        phase = traci.trafficlight.getPhase(state.tls_id)
    except Exception:
        phase = 0
    state.current_phase = phase

    # --- Time in phase ---
    # Bug #7 fix: ``last_phase_switch_time`` used to be updated *only* in
    # ``apply_action`` after a successful ``setPhase``.  SUMO's own program
    # cycling (green → yellow → all-red → next green) never touched it, so
    # ``time_in_phase = sim_time - last_phase_switch_time`` silently became
    # "seconds since the episode started" (observed mean ≈ 1635 s).

    # Whenever the phase observed from SUMO changed since the previous step
    # (auto-cycled transition OR an agent-triggered setPhase), the current
    # phase began at some point since the last observation.  Remember the
    # switch time now; ``time_in_phase`` measures from this step onward.
    if (
        state.last_observed_phase is not None
        and state.current_phase != state.last_observed_phase
    ):
        state.last_phase_switch_time = sim_time

    # ``time_in_phase`` = seconds since the CURRENT phase began.  Note the
    # switch was detected at the previous observation, so the value may be
    # up to one decision interval larger than zero right after a transition.
    state.time_in_phase = max(0.0, sim_time - state.last_phase_switch_time)
    state.last_observed_phase = state.current_phase

    # --- Per-lane metrics ---
    queue_lengths: list[int] = []
    waiting_vehicles: list[int] = []
    vehicle_counts: list[int] = []
    mean_speeds: list[float] = []

    for edge_id in state.incoming_edges:
        for lane_idx in range(state.num_lanes):
            lane_id = f"{edge_id}_{lane_idx}"
            try:
                # Queue: vehicles halting (speed < 0.1 m/s) on this lane
                halting = traci.lane.getLastStepHaltingNumber(lane_id)
                queue_lengths.append(halting)

                # Vehicle count on this lane
                v_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                vcount = len(v_ids)
                vehicle_counts.append(vcount)

                # Waiting vehicles: vehicles with accumulated waiting time > 0
                waiting = 0
                speeds: list[float] = []
                for vid in v_ids:
                    wt = traci.vehicle.getWaitingTime(vid)
                    spd = traci.vehicle.getSpeed(vid)
                    if wt > 0:
                        waiting += 1
                    speeds.append(spd)
                waiting_vehicles.append(waiting)

                # Mean speed on this lane
                mean_speed = float(sum(speeds) / len(speeds)) if speeds else 0.0
                mean_speeds.append(mean_speed)

            except Exception:
                queue_lengths.append(0)
                waiting_vehicles.append(0)
                vehicle_counts.append(0)
                mean_speeds.append(0.0)

    # Cache for reward / info
    state.queue_lengths = queue_lengths
    state.waiting_vehicles = waiting_vehicles
    state.vehicle_counts = vehicle_counts
    state.mean_speeds = mean_speeds

    # Build the flat observation vector (unscaled)
    obs: list[float] = [
        float(state.current_phase),
        state.time_in_phase,
    ]
    obs.extend(float(q) for q in queue_lengths)
    obs.extend(float(w) for w in waiting_vehicles)
    obs.extend(float(v) for v in vehicle_counts)
    obs.extend(mean_speeds)

    # Apply fixed per-dim scaling in ONE place (envs/observation.py), BEFORE
    # padding so the per-lane layout stays valid; padded values are zeros.
    obs = scale_observation(obs)

    # Pad or truncate to max_dim so all agents have the same length
    if max_dim is not None:
        while len(obs) < max_dim:
            obs.append(0.0)
        return obs[:max_dim]

    return obs


def get_observation_dim(num_incoming_edges: int, num_lanes: int = 1) -> int:
    """Return the length of the observation vector."""
    total_lanes = num_incoming_edges * num_lanes
    return 2 + total_lanes * 4  # phase + time_in_phase + 4 metrics per lane


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def apply_action(
    traci,
    state: IntersectionState,
    action: int,
    sim_time: float,
    min_green_time: float = DEFAULT_MIN_GREEN_TIME,
) -> None:
    """
    Apply an agent action to the SUMO traffic light.

    Parameters
    ----------
    traci : module
        Live TraCI module.
    state : IntersectionState
        Mutable state for this intersection.
    action : int
        0 = keep current phase, 1 = switch to next phase.
    sim_time : float
        Current SUMO simulation time.
    min_green_time : float
        Minimum time a green phase must be active before a switch is allowed.
    """
    if action == 0:
        return  # keep current phase — nothing to do

    # Ensure we know the actual TLS program
    detect_tls_phases(traci, state)

    num_phases = state.num_phases

    # Single-phase TLS: nothing to switch to
    if num_phases <= 1:
        return

    # action == 1: request phase switch
    current_phase = state.current_phase

    # ``last_phase_switch_time`` is maintained by ``collect_observation`` as
    # the sim time at which the CURRENT phase began (updated on every
    # observed phase change, both SUMO-cycled and agent-triggered), so this
    # gate uses exactly the same value as the reported time_in_phase.
    time_in_phase = sim_time - state.last_phase_switch_time

    # SUMO completes every yellow/all-red interval automatically. A policy
    # may shorten a green phase, but cannot bypass a clearance interval.
    if current_phase not in state.green_phases:
        return
    if time_in_phase < min_green_time:
        return  # too early to switch — ignore action

    # Compute next phase (sequential advancement through the actual cycle)
    next_phase = (current_phase + 1) % num_phases

    try:
        traci.trafficlight.setPhase(state.tls_id, next_phase)
        state.current_phase = next_phase
        state.last_phase_switch_time = sim_time
        state.time_in_phase = 0.0
    except Exception:
        pass  # TLS might not exist (e.g. during testing with mocks)


def is_valid_action(action: int) -> bool:
    """Return True if *action* is a valid discrete action."""
    return action in (0, 1)


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def compute_reward(
    state: IntersectionState,
    current_waiting_time: float,
    current_queue_length: int,
) -> float:
    """
    Compute the local reward for one intersection.

    Reward = decrease in waiting time  +  decrease in queue length

    Positive values mean the intersection improved; negative means it worsened.

    Parameters
    ----------
    state : IntersectionState
        Must contain ``prev_waiting_time`` and ``prev_queue_length`` from the
        previous step.
    current_waiting_time : float
        Total waiting time of vehicles on incoming lanes this step.
    current_queue_length : int
        Total queue length (halting vehicles) on incoming lanes this step.

    Returns
    -------
    float
        Scalar reward.
    """
    delta_wait = state.prev_waiting_time - current_waiting_time
    delta_queue = float(state.prev_queue_length - current_queue_length)

    # Update previous values for next step
    state.prev_waiting_time = current_waiting_time
    state.prev_queue_length = current_queue_length

    return delta_wait + delta_queue


def compute_local_waiting_time(traci, state: IntersectionState) -> float:
    """Sum of waiting times for vehicles on the intersection's incoming lanes."""
    total = 0.0
    for edge_id in state.incoming_edges:
        try:
            v_ids = traci.edge.getLastStepVehicleIDs(edge_id)
            for vid in v_ids:
                total += traci.vehicle.getWaitingTime(vid)
        except Exception:
            pass
    return total


def compute_local_queue_length(state: IntersectionState) -> int:
    """Sum of halting vehicles across all incoming lanes."""
    return sum(state.queue_lengths) if state.queue_lengths else 0


# ---------------------------------------------------------------------------
# Info helpers
# ---------------------------------------------------------------------------

def build_info(
    state: IntersectionState,
    reward: float,
    current_waiting_time: float,
    current_queue_length: int,
    sim_time: float,
) -> dict[str, Any]:
    """Build an info dict for one agent."""
    return {
        "intersection_id": state.tls_id,
        "current_phase": state.current_phase,
        "time_in_phase": state.time_in_phase,
        "local_queue_length": current_queue_length,
        "local_waiting_time": current_waiting_time,
        "local_vehicle_count": sum(state.vehicle_counts) if state.vehicle_counts else 0,
        "local_mean_speed": (
            float(sum(state.mean_speeds) / len(state.mean_speeds))
            if state.mean_speeds else 0.0
        ),
        "reward": reward,
        "sim_time": sim_time,
    }
