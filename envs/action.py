"""
Action Space
=============

Discrete action space for each agent (intersection controller).

Actions
-------
    0  KEEP   – keep the current signal phase unchanged
    1  SWITCH – request switch to the next phase in the cycle

The 6-phase cycle is:

    Phase 0  NS_green     41 s   GGrr  (green for north–south)
    Phase 1  NS_yellow     3 s   yyrr
    Phase 2  all_red_1     1 s   rrrr
    Phase 3  EW_green     41 s   rrrG  (green for east–west)
    Phase 4  EW_yellow     3 s   rrry
    Phase 5  all_red_2     1 s   rrrr

Constraints
-----------
* SWITCH is only honoured when the current phase is green and has been active for at least
  ``min_green_time`` seconds (default 10 s).
* Yellow and all-red phases always run to completion — the agent cannot
  interrupt them.
* SUMO wraps from the final clearance phase to phase 0 automatically.

Action-to-phase mapping
-----------------------

    action → next_phase
    ─────────────────────
    KEEP (0)  → current_phase            (no change)
    SWITCH (1)→ (current_phase + 1) % 6

Example
-------
    Current phase = 0 (NS_green), time_in_phase = 15 s, min_green = 10 s
    → SWITCH allowed → next phase = 1 (NS_yellow)

    Current phase = 0 (NS_green), time_in_phase = 5 s, min_green = 10 s
    → SWITCH blocked (green too short) → stays at phase 0
"""

from __future__ import annotations

from gymnasium import spaces


# Actions
ACTION_KEEP = 0
ACTION_SWITCH = 1
NUM_ACTIONS = 2


def action_space() -> spaces.Discrete:
    """
    Build a ``gymnasium.spaces.Discrete`` action space for one agent.

    Returns
    -------
    spaces.Discrete
        Discrete(2): {0=keep, 1=switch}
    """
    return spaces.Discrete(NUM_ACTIONS)


def is_valid_action(action: int) -> bool:
    """Return True if *action* is within the valid range."""
    return action in (ACTION_KEEP, ACTION_SWITCH)


def action_to_description(action: int) -> str:
    """Human-readable description of an action."""
    if action == ACTION_KEEP:
        return "keep"
    elif action == ACTION_SWITCH:
        return "switch"
    return f"invalid({action})"
