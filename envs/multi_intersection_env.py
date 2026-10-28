"""
Multi-Intersection PettingZoo Environment
==========================================

A ``pettingzoo.ParallelEnv`` that wraps a SUMO simulation containing a
3×3 grid of signalized intersections.  Each of the 9 intersections is
controlled by an independent agent that observes local traffic and selects
a discrete action (keep / switch phase).

Lifecycle
---------
1. ``__init__``  – stores configuration, does NOT start SUMO.
2. ``reset``     – starts SUMO, loads network + demand, returns initial
                   observations for all 9 agents.
3. ``step``      – applies actions, advances SUMO by ``step_length`` seconds,
                   collects new observations, computes rewards, returns
                   (obs, rewards, terminations, truncations, infos).
4. ``close``     – shuts down SUMO and cleans up temporary files.

Agent naming
------------
    agent_1_1  …  agent_3_3

mapped to SUMO TLS IDs

    N_1_1  …  N_3_3
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pettingzoo import ParallelEnv
from gymnasium import spaces

# Configure SUMO tools path for traci import (same as run_fixed_time.py)
_sumo_tools = str(Path(__file__).resolve().parent.parent / "venv" / "Lib" / "site-packages" / "sumo" / "tools")
if _sumo_tools not in sys.path:
    sys.path.insert(0, _sumo_tools)

try:
    import sumo as _sumo_pkg
    os.environ.setdefault("SUMO_HOME", str(Path(_sumo_pkg.SUMO_HOME)))
except Exception:
    pass

from envs.single_intersection import (
    IntersectionState,
    NUM_PHASES,
    DEFAULT_MIN_GREEN_TIME,
    collect_observation,
    apply_action as _apply_action,
    compute_local_waiting_time,
    compute_local_queue_length,
    build_info as _build_info,
    detect_tls_phases,
)
from envs.observation import observation_space as _obs_space, observation_dim
from envs.action import action_space as _act_space, ACTION_KEEP, ACTION_SWITCH, NUM_ACTIONS
from envs.reward import RewardConfig, compute_reward as _compute_reward, reward_from_config

from sumo_network.grid_builder import GridBuilder, GridConfig
from sumo_network.signal_plans import SignalPlanGenerator, SignalConfig
from sumo_network.traffic_demand import DemandConfig, TrafficDemandGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_sumo_binary(use_gui: bool = False) -> str:
    """Locate the SUMO binary."""
    import shutil as _shutil

    binary_name = "sumo-gui" if use_gui else "sumo"

    on_path = _shutil.which(binary_name)
    if on_path:
        return on_path

    try:
        import sumo as sumo_pkg
        sumo_home = Path(sumo_pkg.SUMO_HOME)
        suffix = ".exe" if os.name == "nt" else ""
        candidate = sumo_home / "bin" / f"{binary_name}{suffix}"
        if candidate.exists():
            return str(candidate)
    except (ImportError, AttributeError):
        pass

    sumo_home_env = os.environ.get("SUMO_HOME")
    if sumo_home_env:
        suffix = ".exe" if os.name == "nt" else ""
        candidate = Path(sumo_home_env) / "bin" / f"{binary_name}{suffix}"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not find {binary_name}. "
        "Ensure SUMO is installed and SUMO_HOME is set."
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MultiIntersectionEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for 9 signalized intersections in a 3×3 SUMO grid.

    All agents act simultaneously (parallel step).
    """

    metadata = {
        "name": "multi_intersection_v0",
        "render_modes": ["human"],
    }

    def __init__(self, config_path: str = "configs/default.yaml", **overrides):
        """
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file.
        **overrides
            Optional overrides: ``use_gui``, ``render_mode``, ``seed``,
            ``episode_duration``, ``step_duration``, ``min_green_time``,
            ``reward_weights``, ``demand_seed``.

            ``demand_seed``: when given, a per-seed demand .rou.xml is
            generated (and cached) from the YAML ``demand`` section for
            every ``reset()``; without it the static demand file under
            ``paths.demand_dir`` is used as before.
        """
        # --- Load config ---
        with open(config_path, "r") as f:
            self._yaml_config = yaml.safe_load(f)

        self._config_path = config_path

        # --- Grid dimensions ---
        grid_cfg_raw = self._yaml_config.get("network", {}).get("grid", {})
        self._grid_rows = grid_cfg_raw.get("rows", 3)
        self._grid_cols = grid_cfg_raw.get("cols", 3)

        # --- SUMO settings ---
        sumo_cfg = self._yaml_config.get("sumo", {})
        self._step_length: int = overrides.get(
            "step_duration", sumo_cfg.get("step_length", 10)
        )
        # Decision interval: simulated seconds between agent actions. Defaults
        # to step_length. When larger, each env step advances SUMO by multiple
        # simulation sub-steps (physics finer than decisions).
        self._decision_interval: int = overrides.get(
            "decision_interval",
            sumo_cfg.get("decision_interval", self._step_length),
        )
        self._sim_steps_per_decision: int = max(
            1, round(self._decision_interval / self._step_length)
        )
        self._sim_seed: int = overrides.get("seed", sumo_cfg.get("seed", 42))
        self._use_gui: bool = overrides.get("use_gui", sumo_cfg.get("use_gui", False))
        self._backend = overrides.get("backend", sumo_cfg.get("backend", "traci"))
        if self._backend not in ("traci", "libsumo"):
            raise ValueError("backend must be traci or libsumo")
        # Per-episode demand seed (None = static demand file, unchanged)
        self._demand_seed: int | None = overrides.get("demand_seed", None)

        # --- Episode settings ---
        self._episode_duration: int = overrides.get(
            "episode_duration", sumo_cfg.get("end_time", 3600)
        )

        # --- Signal plan ---
        sig_cfg_raw = self._yaml_config.get("signals", {})
        self._cycle_length: float = sig_cfg_raw.get("cycle_length", 90.0)

        # --- Reward config ---
        self._reward_config = RewardConfig()
        if "reward_weights" in overrides:
            rw = overrides["reward_weights"]
            self._reward_config.waiting_weight = float(rw.get("waiting_weight", 1.0))
            self._reward_config.queue_weight = float(rw.get("queue_weight", 1.0))
        elif "reward_weights" in self._yaml_config:
            self._reward_config = reward_from_config(self._yaml_config)

        self._min_green_time: float = overrides.get(
            "min_green_time", DEFAULT_MIN_GREEN_TIME
        )

        # --- Build grid topology (pure Python, no SUMO needed) ---
        grid_config = GridConfig(
            rows=self._grid_rows,
            cols=self._grid_cols,
            block_length=self._yaml_config.get("network", {}).get("block_length", 300.0),
            lanes_per_direction=self._yaml_config.get("network", {}).get("lanes_per_direction", 1),
            speed_limit=self._yaml_config.get("network", {}).get("speed_limit", 8.33),
            bidi=self._yaml_config.get("network", {}).get("bidi", True),
            boundary_ring=self._yaml_config.get("network", {}).get("boundary_ring", False),
        )
        self._grid = GridBuilder(grid_config)

        # --- Incoming edges (edge topology approach, matching run_fixed_time.py) ---
        self._incoming_edges: dict[str, list[str]] = {
            nid: [] for nid in self._grid.intersection_nodes
        }
        for edge in self._grid.edges:
            if edge.to_node in self._incoming_edges:
                if edge.id not in self._incoming_edges[edge.to_node]:
                    self._incoming_edges[edge.to_node].append(edge.id)

        # --- Agent names ---
        self._tls_ids: list[str] = list(self._grid.intersection_nodes)
        self._agent_names: list[str] = [
            self._tls_to_agent(nid) for nid in self._tls_ids
        ]

        self.possible_agents = list(self._agent_names)

        # --- Spaces (identical for all agents) ---
        num_incoming = len(self._tls_ids and self._incoming_edges[self._tls_ids[0]])
        self._observation_lanes = self._yaml_config.get("network", {}).get("observation_lanes", 1)
        self._obs_dim = observation_dim(num_incoming, num_lanes=self._observation_lanes)

        self._max_obs_dim = self._obs_dim  # uniform dimension for all agents

        self.observation_spaces = spaces.Dict({
            agent: _obs_space(num_incoming, num_lanes=self._observation_lanes)
            for agent in self.possible_agents
        })
        self.action_spaces = spaces.Dict({
            agent: _act_space()
            for agent in self.possible_agents
        })

        # --- Runtime state (set in reset) ---
        self._traci = None
        self._sumo_cfg_path: Path | None = None
        self._tmp_dir: Path | None = None
        self._states: dict[str, IntersectionState] = {}
        self._step_count: int = 0
        self._max_steps: int = self._episode_duration // self._decision_interval
        self._episode_done: bool = False
        self.simulation_observer = None
        self._render_mode = overrides.get("render_mode", None)

    # ----- naming helpers -----

    @staticmethod
    def _tls_to_agent(tls_id: str) -> str:
        """N_1_1 → agent_1_1"""
        parts = tls_id.split("_")
        return f"agent_{parts[1]}_{parts[2]}"

    @staticmethod
    def _agent_to_tls(agent: str) -> str:
        """agent_1_1 → N_1_1"""
        parts = agent.split("_")
        return f"N_{parts[1]}_{parts[2]}"

    def _generate_demand_file(self, rou_path: Path) -> None:
        """Generate a cached demand .rou.xml for ``self._demand_seed``.

        Uses the YAML ``demand`` section (flow rate, depart options, demand
        type) with the per-episode-seed replacing the static seed, and the
        current episode duration. The generator's own RNG consumes the seed
        so different seeds produce (slightly) different demand files.
        """
        dem = self._yaml_config.get("demand", {})
        demand_config = DemandConfig(
            duration=self._episode_duration,
            seed=int(self._demand_seed),
            flow_rate=dem.get("flow_rate", 300),
            demand_level=dem.get("demand_level", "medium"),
            demand_presets=dem.get("demand_presets", {
                "low": 100, "medium": 300, "high": 600,
            }),
            depart_position=dem.get("depart_position", "random_free"),
            depart_speed=dem.get("depart_speed", "max"),
            min_gap=dem.get("min_gap", 2.5),
            corridor_weights=dem.get("corridor_weights"),
            platoon_period=dem.get("platoon_period"),
            platoon_on_time=dem.get("platoon_on_time",30),
            demand_jitter=dem.get("demand_jitter",0.1),
        )
        generator = TrafficDemandGenerator(demand_config)
        demand_type = dem.get("demand_type", "boundary")
        if demand_type == "network_wide":
            generator.generate_network_wide_with_grid_builder(self._grid, rou_path)
        else:
            generator.generate_with_grid_builder(self._grid, rou_path)

    # ----- PettingZoo required properties -----

    @property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    # ----- Lifecycle -----

    def reset(self, seed=None, options=None):
        """Start SUMO, return initial observations and infos."""
        if options and "demand_seed" in options:
            self._demand_seed = int(options["demand_seed"])
        # Close any previous SUMO connection
        self.close()

        # --- Reset internal state ---
        self._step_count = 0
        self._episode_done = False

        # --- Prepare SUMO config ---
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="sumo_menv_"))
        self._sumo_cfg_path = self._tmp_dir / "sumo.sumocfg"
        tripinfo_path = self._tmp_dir / "tripinfo.xml"

        network_dir = Path(
            self._yaml_config.get("paths", {}).get(
                "network_dir", "data/synthetic/network"
            )
        )
        demand_dir = Path(
            self._yaml_config.get("paths", {}).get(
                "demand_dir", "data/synthetic/demand"
            )
        )
        network_name = self._yaml_config.get("network", {}).get("name", "grid_3x3")
        net_file = (network_dir / f"{network_name}.net.xml").resolve()

        if self._demand_seed is None:
            rou_file = (demand_dir / "demand.rou.xml").resolve()
        else:
            # Per-episode demand: generate (or reuse) a cached .rou.xml for
            # this demand seed. Cached under <demand_dir>/cache/ so repeated
            # episodes/evaluations with the same seed are deterministic.
            cache_dir = Path(
                self._yaml_config.get("paths", {}).get(
                    "demand_cache_dir", str(demand_dir / "cache")
                )
            )
            rou_file = (
                cache_dir / f"demand_seed{self._demand_seed}_dur{self._episode_duration}.rou.xml"
            ).resolve()
            # Demand seeds alone do not identify the scenario: changing rates,
            # platoon settings, or topology must invalidate a cached route.
            from filelock import FileLock
            cache_dir.mkdir(parents=True,exist_ok=True)
            fingerprint=hashlib.sha256(json.dumps(dict(
                demand=self._yaml_config.get('demand',{}),
                network=self._yaml_config.get('network',{}),
                duration=self._episode_duration,seed=self._demand_seed),
                sort_keys=True).encode()).hexdigest()
            marker=rou_file.with_suffix('.fingerprint')
            with FileLock(str(rou_file)+'.lock',timeout=30):
                if not rou_file.exists() or not marker.exists() or marker.read_text()!=fingerprint:
                    temporary=rou_file.with_name(rou_file.name+f'.{os.getpid()}.tmp')
                    self._generate_demand_file(temporary)
                    temporary.replace(rou_file)
                    marker.write_text(fingerprint)

        if not net_file.exists():
            raise FileNotFoundError(f"Network file not found: {net_file}")
        if not rou_file.exists():
            raise FileNotFoundError(f"Demand file not found: {rou_file}")

        cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{rou_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{self._episode_duration}"/>
        <step-length value="{self._step_length}"/>
    </time>
    <random>
        <seed value="{self._sim_seed}"/>
    </random>
    <output>
        <tripinfo-output value="{tripinfo_path}"/>
    </output>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
"""
        self._sumo_cfg_path.write_text(cfg_content)
        self._tripinfo_path = tripinfo_path

        # --- Start SUMO ---
        if self._backend == "libsumo":
            import libsumo as _traci
        else:
            import traci as _traci

        sumo_bin = _find_sumo_binary(self._use_gui)
        sumo_cmd = [sumo_bin, "-c", str(self._sumo_cfg_path), "--no-step-log"]
        _traci.start(sumo_cmd)
        self._traci = _traci

        # --- Create per-intersection state objects ---
        self._states = {}
        for tls_id in self._tls_ids:
            state = IntersectionState(
                tls_id=tls_id,
                incoming_edges=list(self._incoming_edges[tls_id]),
                num_lanes=self._observation_lanes,
            )
            # Read initial phase from SUMO
            try:
                state.current_phase = _traci.trafficlight.getPhase(tls_id)
            except Exception:
                state.current_phase = 0
            state.last_phase_switch_time = 0.0
            state.last_observed_phase = None
            # Detect actual TLS phases from SUMO
            detect_tls_phases(_traci, state)
            self._states[tls_id] = state

        # Store per-agent phase counts for get_num_phases()
        self._agent_num_phases = {
            tls_id: self._states[tls_id].num_phases for tls_id in self._tls_ids
        }

        # --- Advance one step to populate initial observations ---
        # The very first observation should reflect the initial state.
        # Step once so vehicles spawn and we have something to observe.
        for _ in range(self._sim_steps_per_decision):
            _traci.simulationStep()
        sim_time = self._decision_interval

        # Collect initial observations
        observations: dict[str, np.ndarray] = {}
        infos: dict[str, dict] = {}
        for agent_name, tls_id in zip(self._agent_names, self._tls_ids):
            state = self._states[tls_id]
            obs = collect_observation(_traci, state, sim_time, max_dim=self._max_obs_dim)
            observations[agent_name] = np.array(obs, dtype=np.float32)

            # Compute initial waiting time / queue for reward baseline
            wt = compute_local_waiting_time(_traci, state)
            ql = compute_local_queue_length(state)
            state.prev_waiting_time = wt
            state.prev_queue_length = ql

            infos[agent_name] = _build_info(state, 0.0, wt, ql, sim_time)

        self._step_count = 1

        return observations, infos

    def step(self, actions: dict[str, int]):
        """
        Apply actions for all agents, advance SUMO, return results.

        Parameters
        ----------
        actions : dict[str, int]
            Mapping from agent name to action (0=keep, 1=switch).

        Returns
        -------
        tuple
            (observations, rewards, terminations, truncations, infos)
        """
        if self._episode_done:
            raise RuntimeError("Episode already done. Call reset() first.")

        traci = self._traci

        # --- 1. Apply actions ---
        # Decisions happen on the decision_interval grid (matching the sim_time
        # labels used in observations), NOT on the finer step_length grid —
        # otherwise the min-green gate sees time_in_phase ≈ one step_length
        # and blocks almost every SWITCH in configs with decision_interval
        # > step_length (e.g. network_wide: step 1 s, decide every 10 s).
        sim_time = self._step_count * self._decision_interval
        for agent_name, action in actions.items():
            tls_id = self._agent_to_tls(agent_name)
            state = self._states[tls_id]
            _apply_action(traci, state, action, sim_time, self._min_green_time)

        # --- 2. Advance SUMO (sub-steps when decision interval > step length) ---
        for _ in range(self._sim_steps_per_decision):
            traci.simulationStep()
            if self.simulation_observer is not None:
                self.simulation_observer(traci)
        sim_time = (self._step_count + 1) * self._decision_interval
        self._step_count += 1

        # --- 3. Collect observations, rewards, infos ---
        observations: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        done = self._step_count >= self._max_steps

        for agent_name, tls_id in zip(self._agent_names, self._tls_ids):
            state = self._states[tls_id]

            # Observation
            obs = collect_observation(traci, state, sim_time, max_dim=self._max_obs_dim)
            observations[agent_name] = np.array(obs, dtype=np.float32)

            # Current local metrics
            current_wt = compute_local_waiting_time(traci, state)
            current_ql = compute_local_queue_length(state)

            # Reward
            reward = _compute_reward(
                state.prev_waiting_time, current_wt,
                state.prev_queue_length, current_ql,
                self._reward_config,
            )
            rewards[agent_name] = reward

            # Update previous values
            state.prev_waiting_time = current_wt
            state.prev_queue_length = current_ql

            # Termination / truncation
            terminations[agent_name] = False
            truncations[agent_name] = done

            # Info
            infos[agent_name] = _build_info(
                state, reward, current_wt, current_ql, sim_time
            )

        self._episode_done = done

        return observations, rewards, terminations, truncations, infos

    def close(self):
        """Shut down SUMO and clean up temporary files."""
        if self._traci is not None:
            try:
                self._traci.close()
            except Exception:
                pass
            self._traci = None

        if self._tmp_dir is not None and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

        self._states = {}
        self._episode_done = True

    def render(self):
        """Render is a no-op for headless mode; SUMO-GUI handles its own window."""
        if self._render_mode == "human" and self._use_gui:
            pass  # SUMO-GUI renders itself

    # ----- Utility -----

    def get_agent_intersection_map(self) -> dict[str, str]:
        """Return mapping of agent name → SUMO TLS ID."""
        return dict(zip(self._agent_names, self._tls_ids))

    def get_incoming_edges(self) -> dict[str, list[str]]:
        """Return mapping of agent name → list of incoming edge IDs."""
        return {
            self._tls_to_agent(tls_id): list(edges)
            for tls_id, edges in self._incoming_edges.items()
        }

    def get_cycle_length(self) -> float:
        """Return the signal cycle length in seconds."""
        return self._cycle_length

    def get_num_phases(self, agent_name: str | None = None) -> int:
        """Return the number of signal phases.

        Parameters
        ----------
        agent_name : str, optional
            If given, return the phase count for this specific agent's TLS.
            If None, returns the legacy default (6).
        """
        if agent_name is not None:
            tls_id = self._agent_to_tls(agent_name)
            return self._agent_num_phases.get(tls_id, NUM_PHASES)
        return NUM_PHASES

    def get_observation_dim(self) -> int:
        """Return the observation vector dimensionality."""
        return self._obs_dim

    def get_tripinfo_path(self) -> Path | None:
        """Return path to the tripinfo.xml (available after close)."""
        if hasattr(self, "_tripinfo_path"):
            return self._tripinfo_path
        return None
