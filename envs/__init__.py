"""
Multi-Agent Traffic Signal Environment
=======================================

PettingZoo ParallelEnv for 9 signalized intersections in a 3×3 SUMO grid.

Modules
-------
* ``multi_intersection_env`` – main environment class
* ``single_intersection``    – per-intersection observation / action / reward logic
* ``observation``            – observation space definition and documentation
* ``action``                 – action space definition and documentation
* ``reward``                 – reward function definition and documentation
"""

from envs.multi_intersection_env import MultiIntersectionEnv

__all__ = ["MultiIntersectionEnv"]
