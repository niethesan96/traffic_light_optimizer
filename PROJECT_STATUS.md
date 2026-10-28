# Multi-Agent Traffic Network Optimization — Project Status

> Updated 2026-10-29: The frozen MAPPO policy achieved **43.75% lower waiting
> time than matched independent PPO** on five unused synthetic stress-test seeds.
> Ray RLlib training, checkpoint reload, and decentralized inference are verified.
> A published Bangalore-count-driven benchmark shows a separate 23.86% reduction;
> actual city geometry and real-world delay validation are not claimed.
> **273 tests pass.** See `COMPLETION_REPORT.md`, `results/acceptance.json`, and
> `results/platoons_test_comparison.json`. All phase notes below are historical.

## Archived Phase 3/4 Status

---

## Phase 3/4 Correction Pass ✅ COMPLETE

### Summary
Fixed 6 critical bugs discovered during Independent PPO validation. The environment now dynamically detects TLS phases from SUMO, the network is regenerated with correct programs, and the evaluation pipeline parses tripinfo correctly.

### Bugs Fixed

| Bug | Problem | Fix |
|-----|---------|-----|
| #1 TLS phase mismatch | Hardcoded 6 phases; network has 4 or 1 | Dynamic `detect_tls_phases()` reads from SUMO at runtime |
| #2 add.xml not in netconvert | TLS file never passed to netconvert | Let netconvert auto-generate TLS; post-process durations |
| #3 signal_state key mismatch | YAML uses `signal_state`, code read `state` | Changed to `p.get("signal_state", p.get("state", ...))` |
| #4 Network not regenerated | Old net.xml had incorrect TLS | Rebuilt; verified 9 TLS with correct phases |
| #5 tripinfo parsed after close | Evaluation deleted tripinfo before parsing | Swapped order: parse first, then close |
| #6 Demand coverage | Only 4/9 intersections get traffic | Added `network_wide` demand type with OD flows |

### Files Modified

| File | Change |
|------|--------|
| `envs/single_intersection.py` | Dynamic TLS detection, updated `apply_action()` |
| `envs/multi_intersection_env.py` | Import `detect_tls_phases`, per-agent phase counts |
| `sumo_network/grid_builder.py` | Absolute paths, removed broken `-i` flag |
| `scripts/build_network.py` | `signal_state` fix, per-intersection state strings, duration post-processing |
| `scripts/evaluate_independent_ppo.py` | Parse tripinfo before `env.close()` |
| `sumo_network/traffic_demand.py` | Added `generate_network_wide_with_grid_builder()` |
| `configs/default.yaml` | Added `demand_type: "boundary"` |

### Files Created

| File | Description |
|------|-------------|
| `tests/test_bugfixes.py` | 23 regression tests for all 6 bugs |

### Test Results

**223 tests passing** (200 original + 23 new regression tests)

### Network State After Rebuild

- 9 TLS programs with correct phases:
  - N_1_1, N_1_2, N_2_1, N_2_2: 4 phases, 4 links (GrGr pattern)
  - N_1_3, N_2_3, N_3_1, N_3_2: 4 phases, 5 links
  - N_3_3: 1 phase (always green)
- Phase durations: 41s green, 3s yellow, 1s all-red (90s cycle)
- All 48 edges present, all route edges valid

### Demand Behavior

- **Boundary-only** (default): 8 entry flows, traffic at N_1_1, N_1_2, N_2_1, N_2_2
- **Network-wide** (new option): OD-based flows covering all 9 intersections
- Original Phase 2 baseline preserved unchanged

### PPO Smoke Test

- 33 steps, 4 PPO updates, ALL 5 CRITICAL CHECKS PASSED
- Non-zero observations, varying rewards, finite losses, non-zero advantages
- Entropy beginning to move (0.6928 from ln(2)=0.6931)

---

## Phase 1 — SUMO Foundation ✅ COMPLETE

### Summary
Built a 3×3 SUMO grid network with configurable signalized intersections.

### Deliverables
- `sumo_network/grid_builder.py` — Configurable grid network builder (nodes, edges, connections)
- `sumo_network/signal_plans.py` — Fixed-time signal plans with offsets
- `sumo_network/traffic_demand.py` — Synthetic traffic demand generator
- `scripts/build_network.py` — CLI to generate SUMO network and demand files
- `scripts/run_simulation.py` — CLI to run SUMO simulation with TraCI
- `configs/default.yaml` — Full YAML configuration
- `data/synthetic/` — Generated network and demand XML files

### Network Statistics
- **Grid**: 3×3 signalized intersections
- **Nodes**: 16 (4×4 grid)
- **Edges**: 48 directed edges (bidirectional)
- **Connections**: 24 signal-controlled connections
- **Signalized Intersections**: 9

---

## Phase 2 — Fixed-Time Baseline ✅ COMPLETE (with correctness fixes)

### Summary
Created a fixed-time traffic signal baseline for comparison against future RL controllers.

### Deliverables
- `evaluation/metrics.py` — Metrics collection, computation, and export
- `scripts/run_fixed_time.py` — Fixed-time baseline simulation runner
- `scripts/plot_baseline.py` — Result visualization
- `tests/test_fixed_time.py` — Comprehensive test suite

### Correctness Fixes Applied (Phase 2 Revisions)

#### Fix 1: Invalid Route Edges (Issue 1)
**Problem**: `netconvert --geometry.remove` merged corner nodes N_0_0, N_0_3, N_3_0 into neighbors, dropping 6 edges from net.xml. Routes referenced edges that didn't exist in the compiled network.

**Root Cause**: `--geometry.remove` removes "geometry nodes" — nodes with no connections. Corner boundary nodes had no explicit connections in .con.xml.

**Fix**: Removed `--geometry.remove` from the netconvert command in `grid_builder.py`. All 48 edges now persist in net.xml.

**Also Fixed**: `identify_boundary_edges()` in `traffic_demand.py` hardcoded `== 0 or == 3` for grid dimensions. Updated to accept `grid_rows` and `grid_cols` parameters.

#### Fix 2: Metric Consistency (Issue 2)
**Problem**: `avg_waiting_time` was computed as `_sum_waiting_times / num_steps` — the average per-step total wait across all vehicles, not the average per-vehicle wait. This double-counted vehicles present across multiple steps, inflating the value.

**Root Cause**: Per-step accumulation summed waiting times of all vehicles at each step, then divided by number of steps. A vehicle present for 10 steps contributed its waiting time 10 times.

**Fix**: `finalize()` now reads per-vehicle `waitingTime` and `duration` from SUMO's tripinfo.xml output file, which provides the authoritative per-vehicle statistics.

**Result**: Avg waiting time corrected from 72.31s → 18.94s (now correctly less than avg travel time of 47.35s).

#### Fix 3: Signal Cycle Consistency (Issue 3)
**Problem**: Config claimed 90-second cycle but phases summed to 88 seconds (40+3+1+40+3+1 = 88).

**Fix**: Updated green phase durations from 40s to 41s each, so the total equals 90s (41+3+1+41+3+1 = 90).

**Verification**: Added tests confirming `cycle_length == sum(phase.durations)` for both config-based and default plans.

#### Fix 4: Per-Intersection Metrics (Issue 4)
**Problem**: `incoming_edges` mapping was built from `.con.xml` connections, but corner intersections (N_1_3, N_3_1, N_3_3) had no explicit connections. SUMO handles them via implicit connections.

**Fix**: Built `incoming_edges` from edge topology (`edge.to_node` matching intersection node ID) instead of from connections. All 9 intersections now have valid incoming edge mappings.

#### Fix 5: Tripinfo Parse Timing
**Problem**: `finalize()` tried to read tripinfo.xml before `traci.close()` wrote it.

**Fix**: Split finalization — first finalize without tripinfo, then re-finalize after `traci.close()` writes the file.

---

## Metric Definitions

| Metric | Definition | Source |
|--------|-----------|--------|
| **total_waiting_time** | Sum of waiting times across all completed vehicles | tripinfo.xml `waitingTime` |
| **avg_waiting_time** | total_waiting_time / num_completed_vehicles | tripinfo.xml |
| **avg_travel_time** | Mean of trip durations | tripinfo.xml `duration` |
| **avg_queue_length** | Mean of per-step total queue length (vehicles with speed < 0.1 m/s on incoming edges) | TraCI per-step |
| **max_queue_length** | Maximum per-step total queue length across all intersections | TraCI per-step |
| **throughput** | Number of vehicles that completed their route | tripinfo.xml count |
| **vehicles_entered** | Total vehicles that departed | TraCI `getDepartedNumber()` |
| **vehicles_remaining** | vehicles_entered − throughput | Computed |

### Per-Intersection Metrics
| Metric | Definition |
|--------|-----------|
| **avg_queue_length** | Mean queue length at intersection across all steps |
| **max_queue_length** | Maximum queue length at intersection |
| **avg_vehicle_count** | Mean number of vehicles on incoming edges |
| **avg_speed** | Mean speed of vehicles on incoming edges |

---

## Signal Timing Definition

**Cycle Length**: 90 seconds

| Phase | Duration | State | Description |
|-------|----------|-------|-------------|
| NS_green | 41s | GGGrrr | North-South green |
| NS_yellow | 3s | yyyrrr | North-South yellow |
| all_red_1 | 1s | rrrrrr | All-red clearance |
| EW_green | 41s | rrrGGG | East-West green |
| EW_yellow | 3s | rrryyy | East-West yellow |
| all_red_2 | 1s | rrrrrr | All-red clearance |

**Verification**: 41 + 3 + 1 + 41 + 3 + 1 = 90 seconds ✓

---

## Current Baseline Results (seed=42, 1-hour, medium demand)

### Network-Level Metrics
| Metric | Value |
|--------|-------|
| Vehicles entered | 856 |
| Vehicles completed | 848 |
| Vehicles remaining | 8 |
| Throughput rate | 99.1% |
| **Avg waiting time/vehicle** | **18.94 s** |
| **Avg travel time** | **47.35 s** |
| Avg queue length | 4.51 |
| Max queue length | 9 |
| Simulation wall time | 6.3 s |

### Per-Intersection Metrics
| Intersection | Avg Queue | Max Queue | Avg Speed (m/s) |
|-------------|-----------|-----------|-----------------|
| N_1_1 | 3.12 | 8 | 2.24 |
| N_2_2 | 1.39 | 6 | 2.20 |
| Others | 0.00 | 0 | 0.00 |

*Note: Most intersections show 0 queue because the boundary-demand pattern concentrates traffic at entry-point intersections (N_1_1, N_2_2). This is expected behavior for the current demand model.*

---

## Phase 3 — Multi-Agent PettingZoo Environment ✅ COMPLETE

### Summary
Created a PettingZoo-compatible ParallelEnv representing 9 signalized intersections. Each intersection is controlled by an independent agent that observes local traffic and selects a discrete action.

### Deliverables
- `envs/multi_intersection_env.py` — PettingZoo ParallelEnv (main environment)
- `envs/single_intersection.py` — Per-intersection observation, action, reward logic
- `envs/observation.py` — Observation space definition and documentation
- `envs/action.py` — Action space definition and documentation
- `envs/reward.py` — Reward function definition and documentation
- `envs/__init__.py` — Package exports
- `scripts/test_multi_agent_env.py` — Integration test script
- `tests/test_multi_agent_env.py` — Comprehensive test suite (54 tests)
- `tests/conftest.py` — SUMO tools path setup for tests

### Environment Architecture

```
envs/
├── __init__.py                  # Exports MultiIntersectionEnv
├── multi_intersection_env.py    # PettingZoo ParallelEnv (9 agents)
├── single_intersection.py       # Per-intersection logic
├── observation.py               # Observation space definition
├── action.py                    # Action space definition
└── reward.py                    # Reward function definition
```

### Agent Naming

| Agent Name | SUMO TLS ID | Grid Position |
|------------|-------------|---------------|
| agent_1_1  | N_1_1       | (1,1)         |
| agent_1_2  | N_1_2       | (1,2)         |
| agent_1_3  | N_1_3       | (1,3)         |
| agent_2_1  | N_2_1       | (2,1)         |
| agent_2_2  | N_2_2       | (2,2)         |
| agent_2_3  | N_2_3       | (2,3)         |
| agent_3_1  | N_3_1       | (3,1)         |
| agent_3_2  | N_3_2       | (3,2)         |
| agent_3_3  | N_3_3       | (3,3)         |

### Observation Definition

Each agent receives an identical, fixed-size flat vector of **18 floats**:

| Index | Field | Description |
|-------|-------|-------------|
| 0 | current_phase | Active TLS phase index (0-5) |
| 1 | time_in_phase | Seconds since the phase started |
| 2-5 | queue_length[0..3] | Halting vehicles per incoming lane |
| 6-9 | waiting_vehicles[0..3] | Waiting vehicles per incoming lane |
| 10-13 | vehicle_count[0..3] | Vehicles on each incoming lane |
| 14-17 | mean_speed[0..3] | Mean speed on each incoming lane (m/s) |

**Dimension**: 2 + (num_incoming_edges x num_lanes) x 4 = 2 + (4 x 1) x 4 = **18**

**Note**: Observations are **strictly local** — no information from neighbouring intersections.
All observations are zero-padded to 18 dimensions for uniformity.

### Action Definition

Discrete action space: `Discrete(2)`

| Action | Value | Description |
|--------|-------|-------------|
| KEEP | 0 | Keep current signal phase unchanged |
| SWITCH | 1 | Switch to next phase (subject to min-green constraint) |

**Constraints**:
- SWITCH only honoured if current phase is not green, or green time >= min_green_time (10s)
- Yellow/all-red phases always run to completion
- Wraps around: phase 5 -> phase 0

### Reward Equation

```
r_i = alpha * (W_i^{t-1} - W_i^t) + beta * (Q_i^{t-1} - Q_i^t)
```

Where:
- `r_i` = reward for intersection i
- `W_i^t` = total waiting time at step t
- `Q_i^t` = total queue length at step t
- `alpha` = waiting weight (default 1.0)
- `beta` = queue weight (default 1.0)

**Positive reward** = improvement (less waiting/queuing)
**Negative reward** = worsening

Configurable via `reward_weights` in YAML config.

### Simulation Timestep

- **Step length**: 10 seconds (matching SUMO step)
- **Cycle length**: 90 seconds (6 phases)
- **Min green time**: 10 seconds

### Episode Duration

- **Default**: 3600 seconds (360 steps)
- **Configurable** via `episode_duration` parameter
- **Termination**: truncation at max_steps (no natural termination)

### Info Dictionary

Each agent receives an info dict with:
- `intersection_id`, `current_phase`, `time_in_phase`
- `local_queue_length`, `local_waiting_time`, `local_vehicle_count`, `local_mean_speed`
- `reward`, `sim_time`

### Known Issues

1. **TLS phase range errors**: Some SUMO TLS may load with fewer phases than expected (e.g., 1 phase instead of 6), causing non-blocking errors when switching. This is a network configuration artifact.

2. **Collision teleport warnings**: Single-lane entry edges cause SUMO to teleport vehicles at high demand. SUMO simulation artifact.

3. **Route file sorting warnings**: SUMO warns about unsorted route files. Non-blocking.

---

## Phase 4 — Independent PPO ✅ COMPLETE

### Summary
Implemented independent PPO training for 9 agents. Each intersection has its own ActorCritic network with NO parameter sharing, NO centralized critic, and NO cross-agent information.

### Deliverables
- `agents/independent_ppo/network.py` — ActorCritic MLP (18→64→64→actor+critic)
- `agents/independent_ppo/buffer.py` — Rollout buffer with GAE computation
- `agents/independent_ppo/ppo.py` — Clipped PPO algorithm
- `agents/independent_ppo/trainer.py` — Independent trainer for 9 agents
- `scripts/train_independent_ppo.py` — Training script with CLI
- `scripts/evaluate_independent_ppo.py` — Evaluation with baseline comparison
- `tests/test_independent_ppo.py` — 33 comprehensive tests
- `configs/default.yaml` — Added `ppo:` and `training:` sections

### PPO Architecture (per agent)

```
observation (18-dim float32)
    ↓
Linear(18, 64) → Tanh
    ↓
Linear(64, 64) → Tanh
    ↓
├── Actor:  Linear(64, 2) → softmax → π(a|s)
└── Critic: Linear(64, 1) → V(s)
```

**Parameters per agent**: 5,571
**Total parameters (9 agents)**: 50,139

### PPO Objective

```
L = -L_CLIP + c₁·L_VF - c₂·H[π]

L_CLIP = E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]
L_VF   = (V(s) - R)²
H[π]   = -Σ π(a|s) log π(a|s)

r_t    = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

### GAE Equation

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t  = δ_t + γ·λ·A_{t+1}
```

### Default Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| learning_rate | 3e-4 | Standard PPO default |
| gamma | 0.99 | Standard for ~360-step episodes |
| gae_lambda | 0.95 | Standard GAE lambda |
| clip_epsilon | 0.2 | Standard PPO clip |
| entropy_coefficient | 0.01 | Small exploration bonus |
| value_loss_coefficient | 0.5 | Balances policy/value gradients |
| update_epochs | 4 | Standard PPO epochs |
| hidden_size | 64 | Appropriate for 18-dim input |
| max_grad_norm | 0.5 | Gradient clipping |
| target_kl | 0.015 | Early stopping threshold |
| rollout_steps | 128 | Steps per rollout |
| minibatch_size | 64 | Minibatch for PPO update |

### Training Command

```bash
python scripts/train_independent_ppo.py \
    --config configs/default.yaml \
    --timesteps 100000 \
    --rollout-steps 128
```

### Evaluation Command

```bash
python scripts/evaluate_independent_ppo.py \
    --checkpoint results/independent_ppo/checkpoints/final \
    --episodes 10
```

### Output Locations

| Output | Location |
|--------|----------|
| Checkpoints | `results/independent_ppo/checkpoints/` |
| Training logs | `results/independent_ppo/training_logs.json` |
| Training CSV | `results/independent_ppo/training_metrics.csv` |
| Evaluation | `results/independent_ppo/evaluation_results.json` |

### Known Issues

1. **TLS phase range errors**: Some SUMO TLS may load with fewer phases than expected. Network configuration artifact.
2. **Collision teleport warnings**: Single-lane entry edges at high demand. SUMO simulation artifact.
3. **Route file sorting warnings**: Non-blocking SUMO warnings.
4. **Untrained PPO expected to underperform**: The PPO agent has not been trained long enough to beat the fixed-time baseline. A 100k+ timestep training run is needed for meaningful comparison.

---

## Reproduction Commands

```bash
# Run all tests (223 tests)
python -m pytest tests/ -v

# Run PPO smoke test (50 timesteps)
python scripts/train_independent_ppo.py --timesteps 50 --episode-duration 60

# Run integration test (20 random steps)
python scripts/test_multi_agent_env.py --steps 20

# Rebuild network and demand (boundary-only)
python scripts/build_network.py --config configs/default.yaml

# Rebuild with network-wide demand
# Edit configs/default.yaml: set demand_type: "network_wide"
python scripts/build_network.py --config configs/default.yaml

# Run full 1-hour baseline
python scripts/run_fixed_time.py --config configs/default.yaml --output-dir results/baseline
```

---

## Remaining Known Issues

1. **Collision teleport warnings**: Single-lane entry edges cause SUMO to teleport vehicles at high demand. This is a SUMO simulation artifact, not a code bug.

2. **Route file sorting warnings**: SUMO warns about unsorted route files. Non-blocking.

3. **Per-intersection metrics concentration** (boundary demand): With default boundary-only demand, most activity is at entry-point intersections. Use `demand_type: "network_wide"` for full coverage.

4. **N_3_3 always green**: Corner intersection N_3_3 has only 1 phase (always green), so agents there cannot switch signals. This is correct behavior for a corner with minimal traffic.

5. **Phase duration matching**: netconvert auto-generates TLS with 4 phases (merging yellow+all_red). Post-processing sets green durations correctly but yellow/all-red durations may not exactly match the 6-phase config. This does not affect environment functionality.

---

## Test Suite

**223 tests passing** across 7 test files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_grid_builder.py` | 28 | Grid topology, XML generation, edge cases |
| `test_signal_plans.py` | 14 | Signal plans, phases, offsets |
| `test_traffic_demand.py` | 21 | Demand generation, boundary edges, reproducibility |
| `test_fixed_time.py` | 50 | Metrics, throughput, queues, files, offsets, edge existence, cycle consistency |
| `test_multi_agent_env.py` | 54 | Environment construction, obs/action spaces, reward, SUMO integration |
| `test_independent_ppo.py` | 33 | Network, PPO, GAE, buffer, checkpoint, independence, smoke test, SUMO integration |
| `test_bugfixes.py` | 23 | Regression tests for all 6 correction-pass bugs |

---

## Files Created/Modified in Phase 4

| File | Change |
|------|--------|
| `agents/independent_ppo/__init__.py` | Created: Package exports |
| `agents/independent_ppo/network.py` | Created: ActorCritic neural network |
| `agents/independent_ppo/buffer.py` | Created: Rollout buffer with GAE |
| `agents/independent_ppo/ppo.py` | Created: Clipped PPO algorithm |
| `agents/independent_ppo/trainer.py` | Created: Independent trainer (9 agents) |
| `scripts/train_independent_ppo.py` | Created: Training script |
| `scripts/evaluate_independent_ppo.py` | Created: Evaluation script |
| `tests/test_independent_ppo.py` | Created: 33 tests |
| `configs/default.yaml` | Modified: Added `ppo:` and `training:` sections |
| `training/__init__.py` | Modified: Updated from empty placeholder |
| `PROJECT_STATUS.md` | Updated with Phase 4 documentation |
| `CHANGELOG.md` | Updated with Phase 4 entry |
| `NEXT_SESSION.md` | Updated: Phase 4 complete, Phase 5 next |

---

*Last updated: Phase 3/4 correction pass complete. 223 tests passing. Next step: real PPO training experiment.*
