"""Project-2 thermal compatibility imports."""

from paramplate.project2.thermal import (
    enable_plate_thermal_from_heat,
    make_T1_from_DeltaT,
    solve_heat3d,
)

__all__ = ["solve_heat3d", "make_T1_from_DeltaT", "enable_plate_thermal_from_heat"]
