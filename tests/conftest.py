"""Pytest conftest — sets up SUMO tools path for traci import."""

import os
import sys
from pathlib import Path

# Add SUMO tools to sys.path before any test imports traci
_project_root = Path(__file__).resolve().parent.parent
_sumo_tools = str(_project_root / "venv" / "Lib" / "site-packages" / "sumo" / "tools")
if os.path.isdir(_sumo_tools) and _sumo_tools not in sys.path:
    sys.path.insert(0, _sumo_tools)

try:
    import sumo as _sumo_pkg
    os.environ.setdefault("SUMO_HOME", str(Path(_sumo_pkg.SUMO_HOME)))
except Exception:
    pass
