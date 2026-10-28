# Phase 3 — Multi-Agent PettingZoo Environment Specification

**Status**: NOT IMPLEMENTED — This is a specification only.

---

## Overview

Create a multi-agent reinforcement learning environment where each signalized intersection is controlled by an independent agent. The environment uses the PettingZoo library (AEC API) to enable independent PPO training in later phases.

---

## Architecture

```
envs/
├── __init__.py
└── sumo_grid_env.py       # Main PettingZoo environment

networks/
├── __init__.py
└── observation_encoder.py  # (Optional) Observation preprocessing
```

---

## Environment Design

### Agent Structure
- **9 agents** (one per intersection): `agent_N_1_1` through `agent_N_3_3`
- **Environment type**: PettingZoo AEC (Agent Environment Cycle) API
- **Parallel API**: PettingZoo parallel wrapper for vectorized training

### Observation Space (per agent)

Each agent observes its local intersection state:

```python
{
    "queue_length": Box(0, inf, shape=(4,)),      # Queue on 4 incoming lanes
    "waiting_time": Box(0, inf, shape=(4,)),       # Cumulative wait per lane
    "current_phase": Discrete(2),                   # 0=NS green, 1=EW green
    "time_in_phase": Box(0, cycle_length, shape=(1,)),  # Seconds since phase start
    "neighbor_phases": Discrete(2, shape=(num_neighbors,)),  # Phase of adjacent intersections
}
```

**Key design decisions**:
- Observations are LOCAL — each agent sees only its own intersection + neighbor phases
- This enables decentralized execution with centralized training (CTDE)
- Neighbor phases enable implicit coordination without explicit message passing

### Action Space (per agent)

```python
Discrete(2)  # 0 = keep current phase, 1 = switch phase
```

**Alternative** (more flexible):
```python
Dict({
    "extend": Box(0, 1, shape=()),      # Probability of extending current phase
    "switch": Box(0, 1, shape=()),      # Probability of switching phase
})
```

### Reward Function

```python
reward = -(
    alpha * local_queue_penalty +
    beta * local_waiting_penalty +
    gamma * throughput_bonus
)

# Default weights
alpha = 1.0   # Queue penalty weight
beta = 0.5    # Waiting time penalty weight
gamma = 0.1   # Throughput bonus weight
```

**Reward components**:
- **Queue penalty**: Sum of queue lengths on incoming lanes (normalized by max queue)
- **Waiting penalty**: Sum of waiting times on incoming lanes
- **Throughput bonus**: Number of vehicles that passed through this step

### Step Timing

- **Control interval**: Every `control_interval` seconds (default: 10s, matching SUMO step)
- **Phase switch**: When agent selects "switch", the TLS transitions to the next phase
- **Minimum green**: Enforce minimum green time (e.g., 10s) to prevent rapid switching
- **Yellow/all-red**: Handled automatically by SUMO TLS

---

## Gymnasium/PettingZoo Interface

### Core Class: `SumoGridEnv`

```python
class SumoGridEnv:
    """
    PettingZoo AEC environment for multi-agent traffic signal control.
    
    Agents interact sequentially within each step:
    1. Reset: Initialize SUMO, observe all agents
    2. For each agent (in order):
       a. Agent observes local state
       b. Agent selects action (keep/switch)
       c. Action is applied to SUMO TLS
       d. Agent receives reward
    3. Advance SUMO simulation by control_interval
    4. Repeat until episode ends
    """
```

### Required Methods

```python
# PettingZoo AEC API
def reset(seed=None, options=None) -> tuple[obs_dict, info_dict]
def step(action: dict[str, int]) -> None
def observe(agent: str) -> dict
def observation_space(agent: str) -> spaces.Dict
def action_space(agent: str) -> spaces.Discrete
def reward(agent: str) -> float
def termination(agent: str) -> bool
def truncation(agent: str) -> bool
def info(agent: str) -> dict

# Lifecycle
def close() -> None
```

### Configuration

```python
env_config = {
    # SUMO settings
    "sumo_config": "configs/default.yaml",
    "use_gui": False,
    "simulation_duration": 3600,      # seconds
    "step_length": 10,                # seconds
    
    # Agent settings
    "num_agents": 9,
    "control_interval": 10,           # seconds (must match step_length)
    "min_green_time": 10,             # seconds
    
    # Reward settings
    "reward_type": "queue_waiting",   # or "travel_time" or "composite"
    "reward_weights": {
        "queue_penalty": 1.0,
        "waiting_penalty": 0.5,
        "throughput_bonus": 0.1,
    },
    
    # Observation settings
    "obs_include_neighbor_phases": True,
    "obs_queue_normalize": True,      # Normalize by max queue
    
    # Reproducibility
    "seed": 42,
}
```

---

## Implementation Steps

### Step 1: Basic Environment Skeleton
- Create `envs/sumo_grid_env.py` with PettingZoo AEC wrapper
- Implement `reset()`, `step()`, `close()`
- Connect to SUMO via TraCI
- Load network and demand from existing files

### Step 2: Observation Space
- Implement `observe()` for each agent
- Build local observation from TraCI data
- Include neighbor phase information

### Step 3: Action Space
- Implement phase switching logic
- Enforce minimum green time
- Handle yellow/all-red transitions

### Step 4: Reward Function
- Implement queue-based reward
- Implement waiting-time-based reward
- Implement throughput bonus

### Step 5: Integration Testing
- Verify environment runs end-to-end
- Verify observations are consistent
- Verify rewards are reasonable
- Compare baseline (fixed-time) vs random policy

### Step 6: Parallel Wrapper
- Create PettingZoo parallel wrapper
- Enable vectorized environments for training

---

## Key Interfaces

### SUMO Integration

The environment must:
1. Start SUMO via `traci.start()` in `reset()`
2. Step SUMO via `traci.simulationStep()` after all agents act
3. Read observations via `traci.vehicle.*`, `traci.edge.*`, `traci.trafficlight.*`
4. Apply actions via `traci.trafficlight.setPhase()`
5. Close SUMO via `traci.close()` in `close()`

### Observation Collection

For each intersection `N_i_j`:
```python
# Get incoming edges from topology (already implemented in run_fixed_time.py)
incoming_edges = grid.incoming_edges[node_id]

# Collect per-lane observations
for edge_id in incoming_edges:
    lane_id = f"{edge_id}_0"  # Single lane
    queue = traci.lane.getLastStepHaltingNumber(lane_id)
    waiting = traci.like.getWaitingTime(lane_id)  # or sum vehicle waits
    speed = traci.lane.getLastStepMeanSpeed(lane_id)

# Get current phase
phase = traci.trafficlight.getPhase(node_id)
time_in_phase = traci.trafficlight.getNextSwitch(node_id) - traci.simulation.getTime()
```

### Action Application

```python
def apply_action(agent_id: str, action: int):
    current_phase = traci.trafficlight.getPhase(agent_id)
    if action == 1:  # Switch
        # Only switch if minimum green time has elapsed
        if time_in_phase >= min_green_time:
            next_phase = (current_phase + 1) % num_phases
            traci.trafficlight.setPhase(agent_id, next_phase)
```

---

## Testing Strategy

### Unit Tests
- Observation space shape and bounds
- Action space validity
- Reward function correctness
- Phase switching logic

### Integration Tests
- Environment reset produces valid initial state
- Single step advances simulation correctly
- Episode terminates after simulation_duration
- Fixed-time agent produces baseline performance

### Validation
- Run 100 episodes with random policy, verify statistics
- Run 100 episodes with fixed-time policy, verify matches Phase 2 baseline
- Verify reproducibility with same seed

---

## Dependencies

Already in `requirements.txt`:
- `gymnasium>=0.29.0`
- `pettingzoo>=1.24.0`
- `numpy>=1.24.0`

May need:
- `stable-baselines3` or `rl_zoo` for PPO training (Phase 4)

---

## File Structure After Phase 3

```
envs/
├── __init__.py
└── sumo_grid_env.py       # PettingZoo AEC environment

tests/
├── test_grid_builder.py    (existing, 28 tests)
├── test_signal_plans.py    (existing, 14 tests)
├── test_traffic_demand.py  (existing, 21 tests)
├── test_fixed_time.py      (existing, 50 tests)
└── test_sumo_grid_env.py   (NEW, ~15-20 tests)
```

---

## Open Questions for Implementation

1. **Observation normalization**: Should queue lengths be normalized by max possible queue, or left raw?
2. **Reward scaling**: Should rewards be normalized per-step, or accumulated?
3. **Action space**: Discrete (keep/switch) or continuous (extend probability)?
4. **Neighbor definition**: 4-connected (N/S/E/W) or include diagonals?
5. **Phase switching**: Should agents be able to skip phases, or only advance sequentially?

---

*This specification defines Phase 3. Do NOT implement until instructed.*
