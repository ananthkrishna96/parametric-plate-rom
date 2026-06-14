"""Project-2 thermomechanical coupling compatibility helpers."""

from __future__ import annotations

from typing import Any


def solve_coupled_thermomechanical(solver: Any, mechanical_mu, thermal_mu=None, **kwargs):
    """Delegate to the legacy coupled solver method.

    Parameters are kept permissive because the legacy solver supports several
    calling conventions accumulated in the notebook.
    """
    if thermal_mu is None:
        return solver.solve_coupled_thermo_mechanical(mechanical_mu, **kwargs)
    return solver.solve_coupled_thermo_mechanical(mechanical_mu, thermal_mu, **kwargs)


def solve_mechanical_given_T1(solver: Any, mechanical_mu, T1, **kwargs):
    """Delegate to ``solver.solve_mechanical_given_T1``."""
    return solver.solve_mechanical_given_T1(mechanical_mu, T1, **kwargs)


def solve_thermal_given_w(solver: Any, w, **kwargs):
    """Delegate to ``solver.solve_thermal_given_w``."""
    return solver.solve_thermal_given_w(w, **kwargs)
