# Multi-Agent Traffic Network Optimization

SUMO traffic control with a fixed-time baseline, nine independent PPO agents,
and MAPPO with neighbor message passing and centralized training.

**Verified result:** 43.75% lower waiting time than matched independent PPO on
five unused short-link/platoon evaluation seeds. See [COMPLETION_REPORT.md](COMPLETION_REPORT.md)
for the frozen protocol, limitations, Ray RLlib verification, and the separate
Bangalore-count-driven transfer benchmark.

## Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe scripts/build_network.py --config configs/coordination.yaml --output-dir data/coordination --network-name grid_coordination
```

Run from the project root. SUMO is provided by `eclipse-sumo`; no separate
SUMO installation is required when its bundled binary is available.

## Verify the trained result in this workspace

```powershell
.\venv\Scripts\python.exe scripts/compare_controllers.py --config configs/platoons.yaml --mappo results/platoons_mappo_14400/final.pt --independent results/platoons_independent_14400/final --seeds 6001 6002 6003 6004 6005 --output results/reproduced_comparison.json
```

The trained checkpoint files and generated networks are present in this
workspace. For a fresh checkout, use the retraining protocol in the completion
report. The wide-grid prototype examples below are a different scenario and
do not reproduce the 43.75% result.

## Prototype wide-grid training

```powershell
.\venv\Scripts\python.exe scripts/train_mappo.py --timesteps 100000 --output-dir results/mappo
.\venv\Scripts\python.exe scripts/train_independent_ppo.py --config configs/coordination.yaml --timesteps 100000 --output-dir results/independent_coordination
.\venv\Scripts\python.exe scripts/compare_controllers.py --mappo results/mappo/final.pt --independent results/independent_coordination/checkpoints/final --output results/comparison.json
.\venv\Scripts\python.exe scripts/report_comparison.py results/comparison.json
.\venv\Scripts\python.exe -m pytest tests -q
```

`--timesteps` counts joint environment decisions, not individual agent samples.
One decision advances SUMO by ten seconds using one-second physics steps.
MAPPO writes `latest.pt`, `final.pt`, `run_config.json`, and per-update JSON
loss/reward logs. Resume parameters, optimizer, and decision count using
`--resume results/mappo/latest.pt --timesteps 200000`. A resume starts a fresh
SUMO episode; it does not replay the exact simulator/RNG state.

Evaluation uses identical demand and simulation seeds for each controller.
It reports completed-trip waiting/travel time, throughput, remaining vehicles,
pending departures, and sampled queues. A no-message inference ablation is
included. XML is parsed after SUMO closes and flushes its output. Waiting-time
reductions must be read alongside throughput and backlog; completing fewer
trips can make completed-trip waiting time look artificially low. Five
held-out evaluation seeds default to 101–105. Train multiple policy seeds for
a research-quality comparison; the default run is a single policy seed.

## Architecture and scenario

`configs/coordination.yaml` adds a boundary ring so all nine signalized
intersections have four approaches and six-phase signal programs. Twelve
bidirectional straight corridor flows cross the grid. Each route traverses
three signals. The scenario is synthetic, with no turning demand. Three lanes
per approach are observed, giving 50 local features per intersection.

Each actor encodes its own 50 features and receives learned encodings from
its immediate north/east/south/west neighbors. Actors share parameters and
have a local intersection identity. `MAPPOModel.actor_logits()` is the
deployment API: it requires only the local observation, four neighbor slots,
and local identity. Non-neighbor state cannot influence its action. The
centralized critic uses all nine observations to estimate one value per agent
during training. PPO uses clipped policy updates, entropy regularization,
gradient clipping, KL stopping, and GAE that bootstraps time limits while
cutting the recursion at episode resets.

The project includes native PyTorch trainers and a tested **Ray RLlib 2.59.0**
new-stack PPO module with a centralized critic. Optional dependencies are in
`requirements-rllib.txt`; PyArrow 23 matches the libsumo Arrow ABI. Legacy
`default.yaml` and `network_wide.yaml` retain their original topology and
observation format for existing checkpoints. Do not evaluate those older
checkpoints on the new scenario; retrain independent PPO with the same config.

## Evidence and limits

The reference image is a target project description. A **35% improvement**,
emergent green waves, and validation on Bangalore traffic are empirical claims,
not properties guaranteed by the algorithm. `data/bangalore/` now contains six
published Bilekahalli daily movement counts, with attribution and provenance.
`configs/bangalore_counts.yaml` projects their proportions and daily-average
flow onto synthetic corridors. This is a real-count-driven benchmark, not a
validated reconstruction of city traffic. Measured delay validation still needs
signal/topology data, calibration, and held-out observed outcomes. A message-ablation result alone
does not establish green waves; corridor travel-time and signal-offset analysis
would be needed.

See `results/coordination_validation.md` for the local validation run.

## Controlled experiments and RLlib

The earlier pilot used different training-demand schedules. Use
`train_controlled.py` for equal budgets and the same sequence of episode seeds.
`configs/cooperative.yaml` uses a persistent queue-cost reward and in-process
SUMO. MAPPO version 2 combines separate local experts with a learned directional
message correction and a centralized critic. A zero initial correction permits
an exact common-policy warm start for both algorithms.

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-rllib.txt
.\venv\Scripts\python.exe scripts/train_controlled.py --algorithm mappo --steps 7200 --initialize-from results/independent_coordination_10k/checkpoints/final --output results/controlled_mappo_7200
.\venv\Scripts\python.exe scripts/train_controlled.py --algorithm independent --steps 7200 --initialize-from results/independent_coordination_10k/checkpoints/final --output results/controlled_independent_7200
.\venv\Scripts\python.exe scripts/compare_controllers.py --config configs/cooperative.yaml --mappo results/controlled_mappo_7200/final.pt --independent results/controlled_independent_7200/final --trace-dir results/controlled_traces --output results/controlled_comparison.json
.\venv\Scripts\python.exe scripts/plot_progression.py results/controlled_traces/mappo_seed101.csv
.\venv\Scripts\python.exe scripts/train_rllib.py --iterations 100 --output results/rllib_mappo
.\venv\Scripts\python.exe scripts/compare_controllers.py --config configs/cooperative.yaml --rllib results/rllib_mappo/module --output results/rllib_comparison.json
.\venv\Scripts\python.exe scripts/prepare_bangalore.py
```

RLlib has its own optimizer/checkpoint format; `--rllib` loads its exported
RLModule for evaluation. The RLlib smoke test completed two PPO iterations,
saved both module and algorithm checkpoints, reloaded the module and evaluated
it in SUMO. This verifies integration, not trained policy quality.

The controlled simple scenario produced a tie at 8.50 seconds per completed
vehicle. Disabling MAPPO messages left its actions' traffic outcome unchanged;
this does not demonstrate a communication benefit. `configs/platoons.yaml`
adds a separate short-link, burst-arrival stress scenario to test anticipation.
Keep its results separate from the original benchmark. Signal-space/time plots
are diagnostic; interpret them alongside stops, throughput, and message ablations.
