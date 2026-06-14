"""Project-2 coupling compatibility imports."""

from paramplate.project2.coupling import (
    solve_coupled_thermomechanical,
    solve_mechanical_given_T1,
    solve_thermal_given_w,
)

__all__ = [
    "solve_coupled_thermomechanical",
    "solve_mechanical_given_T1",
    "solve_thermal_given_w",
]
