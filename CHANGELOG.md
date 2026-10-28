# Changelog

## Phase 3/4 Correction Pass — Environment/Network Bug Fixes

### Bugs Found and Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| #1 TLS phase mismatch | `single_intersection.py` hardcoded `NUM_PHASES=6`, `GREEN_PHASES={0,3}` but network has 4 or 1 phases | Dynamic TLS detection via `detect_tls_phases()` reads actual phases from SUMO at runtime |
| #2 add.xml not passed to netconvert | `grid_builder.py` did not pass `-i` flag with TLS file | Removed `-i` approach (caused link-count mismatches); netconvert now auto-generates TLS with correct link counts |
| #3 signal_state key mismatch | `build_network.py` read `state` but YAML uses `signal_state` | Changed to `p.get("signal_state", p.get("state", ...))` |
| #4 Network not regenerated | Old net.xml had incorrect TLS programs | Rebuilt network; post-processes phase durations from config |
| #5 tripinfo parsed after env.close() | `evaluate_independent_ppo.py` called `env.close()` before parsing tripinfo | Swapped order: parse first, then close |
| #6 Traffic demand coverage | Boundary-only demand only covers 4/9 intersections | Added `network_wide` demand type with OD-based flows covering all intersections |

### Files Modified

| File | Change |
|------|--------|
| `envs/single_intersection.py` | Added `detect_tls_phases()`, dynamic `num_phases`/`green_phases` on `IntersectionState`, updated `apply_action()` |
| `envs/multi_intersection_env.py` | Import `detect_tls_phases`, call during `reset()`, per-agent `get_num_phases()` |
| `sumo_network/grid_builder.py` | Use absolute paths in netconvert, removed broken `-i` flag |
| `scripts/build_network.py` | Fix `signal_state` key, generate per-intersection state strings, add `_set_netxml_phase_durations()` post-processing, support `demand_type` config |
| `scripts/evaluate_independent_ppo.py` | Parse tripinfo before `env.close()` |
| `sumo_network/traffic_demand.py` | Added `generate_network_wide_with_grid_builder()` |
| `configs/default.yaml` | Added `demand_type: "boundary"` config option |

### Files Created

| File | Description |
|------|-------------|
| `tests/test_bugfixes.py` | 23 regression tests covering all 6 bugs |

### Test Results

- **223 tests passing** (200 original + 23 new regression tests)
- All existing tests continue to pass

### Network Regeneration

After fixes, regenerated the SUMO network:
- 16 grid nodes, 48 directed edges
- 9 TLS programs with correct phase counts:
  - N_1_1, N_1_2, N_2_1, N_2_2: 4 phases, 4 links
  - N_1_3, N_2_3, N_3_1, N_3_2: 4 phases, 5 links
  - N_3_3: 1 phase (always green)
- Phase durations set from config (41s green, 3s yellow, 1s all-red)
- All route edges exist in net.xml

### Demand Behavior

- **Boundary-only** (default): 8 entry flows, traffic concentrated at N_1_1, N_1_2, N_2_1, N_2_2
- **Network-wide** (new): OD-based flows covering all 9 intersections
- Original Phase 2 baseline preserved unchanged

### PPO Smoke Test Results

- 33 steps across 3 episodes, 4 PPO updates
- Non-zero observations (21 distinct combos)
- Rewards vary: [-11.00, +11.00], 12/297 nonzero
- Non-zero advantages (8/8 per agent)
- Finite losses: policy ~-0.003, value ~4.7
- Entropy: 0.6928 (moved from ln(2)=0.6931) — beginning to learn
- **ALL 5 CRITICAL CHECKS PASSED**

---

## Phase 4 — Independent PPO

### Files Created

| File | Description |
|------|-------------|
| `agents/independent_ppo/__init__.py` | Package exports |
| `agents/independent_ppo/network.py` | ActorCritic neural network (18-dim → hidden → actor+critic) |
| `agents/independent_ppo/buffer.py` | Rollout buffer with GAE computation |
| `agents/independent_ppo/ppo.py` | PPO algorithm (clipped surrogate, value loss, entropy bonus) |
| `agents/independent_ppo/trainer.py` | Independent trainer managing 9 separate PPO agents |
| `scripts/train_independent_ppo.py` | Training script with CLI arguments |
| `scripts/evaluate_independent_ppo.py` | Evaluation script with baseline comparison |
| `tests/test_independent_ppo.py` | 33 comprehensive tests |

### Files Modified

| File | Change |
|------|--------|
| `configs/default.yaml` | Added `ppo:` and `training:` configuration sections |
| `training/__init__.py` | Updated from empty placeholder |
| `PROJECT_STATUS.md` | Updated with Phase 4 documentation |
| `CHANGELOG.md` | This file |
| `NEXT_SESSION.md` | Updated: Phase 4 complete, Phase 5 next |

### Implementation Details

- **Algorithm**: Standard clipped PPO (Schulman et al., 2017)
- **Architecture**: Per-agent ActorCritic MLP (18→64→64→actor+critic), ~5,571 params per agent
- **Independence**: 9 separate networks with NO parameter sharing, NO centralized critic
- **GAE**: γ=0.99, λ=0.95, with advantage normalization
- **PPO objective**: clipped surrogate with ε=0.2, entropy bonus 0.01, value loss coeff 0.5
- **Update**: 4 epochs, minibatch size 64, target KL 0.015
- **Seeding**: Python, NumPy, PyTorch seeds set for reproducibility
- **Checkpointing**: Save/load per-agent networks + optimizer state + metadata
- **Logs**: JSON + CSV training metrics
- **Evaluation**: Deterministic policy, tripinfo parsing, baseline comparison

### Test Results

- 200 tests total (167 Phase 1-3 + 33 Phase 4)
- All passing

---

## Phase 3 — Multi-Agent PettingZoo Environment

### Files Created

| File | Description |
|------|-------------|
| `envs/multi_intersection_env.py` | PettingZoo ParallelEnv for 9 signalized intersections |
| `envs/single_intersection.py` | Per-intersection observation/action/reward logic |
| `envs/observation.py` | Observation space definition (18-dim Box) |
| `envs/action.py` | Action space definition (Discrete(2)) |
| `envs/reward.py` | Reward function definition (Δwaiting + Δqueue) |
| `tests/conftest.py` | SUMO tools path setup for pytest |
| `tests/test_multi_agent_env.py` | 54 comprehensive tests |
| `scripts/test_multi_agent_env.py` | Integration test script (random-action run) |

### Files Modified

| File | Change |
|------|--------|
| `envs/__init__.py` | Updated from empty placeholder to export `MultiIntersectionEnv` |
| `PROJECT_STATUS.md` | Updated with Phase 3 status and architecture documentation |

### Implementation Details

- **API**: PettingZoo `ParallelEnv` (all agents act simultaneously)
- **Agents**: 9 (`agent_1_1` through `agent_3_3`), one per intersection
- **Observation**: 18-dim flat float vector (phase, time_in_phase, queue×4, waiting×4, count×4, speed×4). Strictly local — no neighbor information.
- **Action**: `Discrete(2)` — 0=KEEP, 1=SWITCH. Minimum green time enforcement (10s).
- **Reward**: `r_i = α(W_i^{t-1} - W_i^t) + β(Q_i^{t-1} - Q_i^t)` with default α=1.0, β=1.0
- **Episode**: truncation at `max_steps = episode_duration / step_length`
- **Tests**: 54 new tests (observation, action, reward, single-intersection helpers, env construction, SUMO integration)

### Test Results

- 167 tests total (113 Phase 1/2 + 54 Phase 3)
- All passing

---

## Phase 2 — Fixed-Time Baseline (Correctness Pass)

### Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `sumo_network/grid_builder.py` | Modified | Removed `--geometry.remove` from netconvert command |
| `sumo_network/traffic_demand.py` | Modified | Added `grid_rows`/`grid_cols` params to `identify_boundary_edges()` and `generate_rou_xml()` |
| `evaluation/metrics.py` | Modified | Fixed avg_waiting_time to use tripinfo; added `compute_waiting_times_from_tripinfo()`; added error handling for empty/malformed tripinfo |
| `scripts/run_fixed_time.py` | Modified | Fixed `incoming_edges` topology (edge-based instead of connection-based); fixed tripinfo parse timing (after `traci.close()`) |
| `configs/default.yaml` | Modified | Fixed green phase duration from 40s to 41s for correct 90s cycle |
| `tests/test_fixed_time.py` | Modified | Added 22 new tests (edge existence, metric correctness, cycle consistency, intersection mappings) |
| `tests/test_traffic_demand.py` | Modified | Updated calls to `identify_boundary_edges` with grid dimensions |

### Files Created

| File | Description |
|------|-------------|
| `PROJECT_STATUS.md` | Comprehensive project status document |
| `PHASE3_SPEC.md` | Phase 3 implementation specification |
| `NEXT_SESSION.md` | Handoff document for next session |
| `CHANGELOG.md` | This file |

### Bug Fixes

1. **Invalid route edges**: Removed `--geometry.remove` from netconvert; all 48 edges now persist in net.xml
2. **Metric inconsistency**: Fixed avg_waiting_time to use per-vehicle tripinfo data instead of per-step accumulation
3. **Signal cycle mismatch**: Updated green phase from 40s to 41s so cycle equals 90s
4. **Per-intersection zeros**: Built incoming_edges from edge topology instead of connections
5. **Tripinfo parse timing**: Fixed to read after `traci.close()` writes the file

### Test Changes

- Added 22 new tests to `test_fixed_time.py`
- Updated 3 existing tests in `test_traffic_demand.py` for new function signatures
- Total: 113 tests (was 98 after Phase 2 initial, 63 after Phase 1)

### Baseline Results (After Fixes)

| Metric | Before | After |
|--------|--------|-------|
| Edge errors | Many "not known" | 0 |
| Avg waiting time | 72.31s | 18.94s |
| Avg travel time | 46.33s | 47.35s |
| Signal cycle | 88s (broken) | 90s (correct) |
| Throughput | 847 | 848 |

---

*Generated: Phase 3/4 correction pass complete*
