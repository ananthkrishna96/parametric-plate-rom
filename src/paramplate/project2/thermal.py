"""Project-2 thermal compatibility helpers.

These functions intentionally delegate to ``GeneralMultiphysicsSolver`` methods
during the first migration phase. Later refactor phases can move the actual
implementation here without changing external usage.
"""

from __future__ import annotations

from typing import Any


def solve_heat3d(solver: Any, **kwargs):
    """Delegate to ``solver.heat_solve``."""
    return solver.heat_solve(**kwargs)


def make_T1_from_DeltaT(solver: Any, DeltaT3d, **kwargs):
    """Delegate to ``solver.make_T1_from_DeltaT``."""
    return solver.make_T1_from_DeltaT(DeltaT3d, **kwargs)


def enable_plate_thermal_from_heat(solver: Any, **kwargs):
    """Delegate to ``solver.enable_plate_thermal_from_heat``."""
    return solver.enable_plate_thermal_from_heat(**kwargs)
