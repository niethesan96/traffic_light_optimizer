"""
Tests for Stage 2b — time_in_phase correctness
==============================================

Bug #7: ``last_phase_switch_time`` was only updated in ``apply_action``
after a successful ``setPhase``, so SUMO's own program cycling never
updated it and ``time_in_phase`` tracked total simulation time
(observed mean ≈ 1635 s on network_wide instead of seconds-in-phase).

Fix: ``collect_observation`` detects the phase change from SUMO and resets
``last_phase_switch_time``; the min-green gate in ``apply_action`` uses the
same value.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.single_intersection import (
    IntersectionState,
    collect_observation,
    apply_action,
    DEFAULT_MIN_GREEN_TIME,
)


def _mock_traci(phase_sequence):
    """Mock traci whose getPhase yields *phase_sequence* values in order."""
    mock = MagicMock()
    mock.trafficlight.getPhase.side_effect = list(phase_sequence)
    mock.lane.getLastStepHaltingNumber.return_value = 0
    mock.lane.getLastStepVehicleIDs.return_value = []
    mock.vehicle.getWaitingTime.return_value = 0.0
    mock.vehicle.getSpeed.return_value = 0.0
    return mock


class TestPhaseChangeDetection:
    """collect_observation must reset time_in_phase on every phase change."""

    def test_switch_observed_resets_time_in_phase(self):
        state = IntersectionState(tls_id="N_1_1", incoming_edges=[])
        state.last_phase_switch_time = 0.0
        traci = _mock_traci([0, 0, 3, 3])

        # t=10: first observation, phase 0 unchanged → 10 s since phase start
        obs = collect_observation(traci, state, sim_time=10.0)
        assert state.time_in_phase == pytest.approx(10.0)

        # t=20: still phase 0 → grows by the interval
        obs = collect_observation(traci, state, sim_time=20.0)
        assert state.time_in_phase == pytest.approx(20.0)

        # t=30: SUMO auto-cycled to phase 3 → time_in_phase resets ≤ interval
        obs = collect_observation(traci, state, sim_time=30.0)
        assert state.last_phase_switch_time == pytest.approx(30.0)
        assert state.time_in_phase <= 10.0

        # t=40: phase 3 unchanged → resumes counting from the reset point
        obs = collect_observation(traci, state, sim_time=40.0)
        assert state.time_in_phase == pytest.approx(10.0)

    def test_time_in_phase_never_tracks_sim_time(self):
        """Regression: with auto cycling, time_in_phase must stay small."""
        state = IntersectionState(tls_id="N_1_1", incoming_edges=[])
        state.last_phase_switch_time = 0.0
        # Alternating phase index every other observation (auto cycle)
        phases = [0, 3] * 10
        traci = _mock_traci(phases)
        for t in range(10, 210, 10):
            collect_observation(traci, state, sim_time=float(t))
            assert state.time_in_phase <= 20.0, (
                f"t={t}: time_in_phase={state.time_in_phase} tracks sim time"
            )


class TestMinGreenGateUsesSameValue:
    """The min-green gate must use the phase-start time, like the obs."""

    def _green_state(self, last_phase_switch_time):
        state = IntersectionState(tls_id="N_1_1", incoming_edges=[])
        state.current_phase = 0
        state.num_phases = 4
        state.green_phases = {0, 2}
        state.tls_detected = True
        state.last_observed_phase = 0
        state.last_phase_switch_time = last_phase_switch_time
        return state

    def test_switch_inside_min_green_ignored(self):
        # Phase started at sim time 5 (detected by collect_observation).
        state = self._green_state(last_phase_switch_time=5.0)
        mock = MagicMock()
        apply_action(mock, state, 1, sim_time=8.0, min_green_time=10.0)
        mock.trafficlight.setPhase.assert_not_called()
        assert state.current_phase == 0

    def test_switch_outside_min_green_allowed(self):
        state = self._green_state(last_phase_switch_time=5.0)
        mock = MagicMock()
        apply_action(mock, state, 1, sim_time=26.0, min_green_time=10.0)
        mock.trafficlight.setPhase.assert_called_once_with("N_1_1", 1)
        assert state.current_phase == 1
        # Successful switch restarts the phase clock
        assert state.last_phase_switch_time == pytest.approx(26.0)
        assert state.time_in_phase == pytest.approx(0.0)


class TestTimeInPhaseInSUMO:
    """Integration: forced switch in the live SUMO network_wide env."""

    def test_forced_switch_resets_time_in_phase(self):
        from envs.multi_intersection_env import MultiIntersectionEnv

        env = MultiIntersectionEnv(
            config_path="configs/network_wide.yaml",
            episode_duration=120,
        )
        try:
            obs, infos = env.reset()
            agent = "agent_1_1"
            tls_state = env._states["N_1_1"]

            # Step 1: force a phase switch for agent_1_1, KEEP everywhere else
            actions = {a: 0 for a in env.possible_agents}
            actions[agent] = 1
            phase_before = tls_state.current_phase
            obs, rewards, terms, truncs, infos = env.step(actions)

            # SWITCH must have changed the phase
            assert tls_state.current_phase != phase_before, (
                "SWITCH did not change the phase in SUMO"
            )

            # time_in_phase must be back within one decision interval
            decision_interval = env._decision_interval
            assert tls_state.time_in_phase <= decision_interval, (
                f"after forced switch time_in_phase={tls_state.time_in_phase} "
                f"> decision_interval={decision_interval}"
            )

            # Now run all-KEEP; time_in_phase grows with the interval while
            # the phase stays, but must NEVER look like total sim time.
            actions = {a: 0 for a in env.possible_agents}
            for t in range(3):
                obs, rewards, terms, truncs, infos = env.step(actions)
                sim_time = infos[agent]["sim_time"]
                assert tls_state.time_in_phase <= sim_time, (
                    f"time_in_phase may never exceed sim time"
                )
                assert tls_state.time_in_phase < 100.0, (
                    f"t={sim_time}: time_in_phase={tls_state.time_in_phase} "
                    f"is tracking sim time again"
                )
        finally:
            env.close()
