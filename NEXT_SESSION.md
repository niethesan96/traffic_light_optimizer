# Next Session Handoff

> Updated 2026-10-29: see `COMPLETION_REPORT.md` for the current handoff.
> The 35% simulation target is exceeded: 43.75% paired reduction on frozen
> test seeds 6001–6005, with matched 25,200-decision budgets. Winning checkpoint:
> `results/platoons_mappo_14400/final.pt`. Ray RLlib training/reload/inference
> and a historical Bangalore-count-driven transfer benchmark are verified.
> Real-city delay validation is not claimed. The old phase gates and suggested
> prompt below are archived context and are superseded by the user's work request.

## Current Project State

**Phase 1**: ✅ COMPLETE — SUMO 3×3 grid network, 9 intersections, 48 edges
**Phase 2**: ✅ COMPLETE — Fixed-time baseline with corrected metrics
**Phase 3**: ✅ COMPLETE — Multi-agent PettingZoo ParallelEnv (9 agents)
**Phase 4**: ✅ COMPLETE — Independent PPO training (9 separate agents)
**Phase 3/4 Correction Pass**: ✅ COMPLETE — Environment/network bug fixes

---

## Correction Pass Summary

### Bugs Found and Fixed

1. **Bug #1 (TLS phase mismatch)**: Environment hardcoded 6 phases but network has 4 or 1. Fixed with dynamic TLS detection via `detect_tls_phases()` that reads actual phases from SUMO at runtime.

2. **Bug #2 (add.xml not in netconvert)**: The intended TLS file was never passed to netconvert. Fixed by letting netconvert auto-generate TLS with correct link counts, then post-processing phase durations from config.

3. **Bug #3 (signal_state key)**: `build_network.py` read `state` but YAML uses `signal_state`. Fixed with `p.get("signal_state", p.get("state", ...))`.

4. **Bug #4 (network regeneration)**: Rebuilt entire network. Verified 16 nodes, 48 edges, 9 TLS programs with correct phase counts.

5. **Bug #5 (tripinfo parsing)**: Evaluation parsed tripinfo after `env.close()` deleted it. Swapped order.

6. **Bug #6 (demand coverage)**: Boundary-only demand only covers 4/9 intersections. Added `network_wide` demand type with OD-based flows.

### Test Count

**223 tests passing** (200 original + 23 regression tests in `tests/test_bugfixes.py`)

### PPO Smoke Test

- 33 steps, 4 PPO updates
- All 5 critical checks passed: non-zero observations, varying rewards, successful updates, non-zero advantages, finite losses
- Entropy beginning to move (0.6928 from ln(2)=0.6931)
- **NOT a claim that PPO learns** — only that the pipeline is functional

---

## What To Do Next

### Immediate: Run Real PPO Training

The environment fixes are verified. The next step is a real training experiment to see if PPO actually learns.

```bash
python scripts/train_independent_ppo.py \
    --config configs/default.yaml \
    --timesteps 10000 \
    --episode-duration 300 \
    --rollout-steps 128
```

### After Training: Evaluate and Compare

```bash
python scripts/evaluate_independent_ppo.py \
    --checkpoint results/independent_ppo/checkpoints/final \
    --episodes 10
```

### Decision Point

After training + evaluation, determine:
1. Is the reward increasing during training?
2. Does PPO beat or match the fixed-time baseline?
3. Are observations non-trivial (vehicles at intersections)?
4. Are rewards non-zero and varying?

If PPO shows learning: consider Phase 5 (MAPPO).
If PPO doesn't show learning: investigate reward shaping, demand model, or hyperparameters.

---

## Do NOT Proceed to Phase 5 Until:
- [ ] A real training run (10k+ timesteps) has been completed
- [ ] Training logs show the reward is changing (learning is happening)
- [ ] Evaluation results are recorded and compared to baseline
- [ ] The user has reviewed the findings and approved moving to Phase 5

---

## Key Files After Correction Pass

| File | Purpose |
|------|---------|
| `envs/single_intersection.py` | Dynamic TLS detection, action logic, reward computation |
| `envs/multi_intersection_env.py` | PettingZoo ParallelEnv with per-agent phase detection |
| `sumo_network/grid_builder.py` | Network builder (absolute paths, no `-i` flag) |
| `scripts/build_network.py` | Network + demand generation with `signal_state` fix |
| `scripts/evaluate_independent_ppo.py` | Evaluation with fixed tripinfo parsing |
| `sumo_network/traffic_demand.py` | Demand generator with `network_wide` preset |
| `tests/test_bugfixes.py` | 23 regression tests |
| `configs/default.yaml` | Added `demand_type: "boundary"` |

---

## Network State

After rebuilding:
- 9 TLS programs with correct phases:
  - 4 intersections: 4 phases, 4 links (GrGr pattern)
  - 4 edge intersections: 4 phases, 5 links
  - N_3_3: 1 phase (always green, never switches)
- Phase durations from config: 41s green, 3s yellow, 1s all-red
- All 48 edges present, all route edges valid

## Demand State

- **Default**: boundary-only (8 entry flows, 4/9 intersections get traffic)
- **New option**: `demand_type: "network_wide"` for OD-based full coverage
- Original Phase 2 baseline preserved unchanged

---

## Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| learning_rate | 3e-4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_epsilon | 0.2 |
| entropy_coefficient | 0.01 |
| value_loss_coefficient | 0.5 |
| update_epochs | 4 |
| hidden_size | 64 |
| rollout_steps | 128 |
| minibatch_size | 64 |

---

## Recommended First Prompt for Next Session

```
We are continuing the Multi-Agent Traffic Network Optimization project.

Phases 1–4 are COMPLETE. Phase 3/4 correction pass is COMPLETE.
223 tests passing.

The environment bugs have been fixed:
- Dynamic TLS phase detection (no more hardcoded 6 phases)
- Network regenerated with correct programs
- Tripinfo parsing fixed
- Network-wide demand option added

The next step is to run a REAL Independent PPO training experiment
and determine whether PPO actually learns a useful policy.

DO NOT implement MAPPO yet.

Steps:
1. Run the existing 223 tests.
2. Run a real PPO training experiment (~10,000 timesteps).
3. Analyze training curves and evaluation results.
4. Compare against the fixed-time baseline.
5. Determine if PPO is actually learning.
6. Report findings.

Fixed-time baseline reference:
- Average waiting time: 18.94 s
- Average travel time: 47.35 s
- Throughput: 848 vehicles

Wait for user approval before proceeding to Phase 5.
```

---

*Handoff prepared: Phase 3/4 correction pass complete. 223 tests passing. Next step: real PPO training experiment.*
