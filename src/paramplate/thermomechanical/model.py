"""Thermomechanical campaign definitions from the final study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paramplate.core.sampling import ParameterRange


@dataclass(frozen=True)
class ThermomechanicalCase:
    case: int
    parameter_names: tuple[str, ...]
    accepted_samples: int
    displacement_rank: int = 6
    thermal_driver_rank: int = 4


CASES: dict[int, ThermomechanicalCase] = {
    2: ThermomechanicalCase(2, ("load_amplitude", "absorbed_solar_flux"), 450),
    3: ThermomechanicalCase(3, ("foundation_stiffness", "ambient_temperature", "substrate_temperature"), 378),
    5: ThermomechanicalCase(
        5,
        (
            "foundation_stiffness",
            "load_amplitude",
            "absorbed_solar_flux",
            "contact_conductance",
            "gap_conductance",
        ),
        500,
    ),
}


def parameter_ranges(case: int) -> tuple[ParameterRange, ...]:
    if int(case) == 2:
        return (
            ParameterRange("load_amplitude", -6500.0, -1500.0, "N/m^2", role="mechanical load"),
            ParameterRange("absorbed_solar_flux", 150.0, 850.0, "W/m^2", role="thermal load"),
        )
    if int(case) == 3:
        return (
            ParameterRange("foundation_stiffness", 1.0e5, 2.0e6, "N/m^3", scale="log"),
            ParameterRange("ambient_temperature", 298.15, 313.15, "K"),
            ParameterRange("substrate_temperature", 295.15, 310.15, "K"),
        )
    if int(case) == 5:
        return (
            ParameterRange("foundation_stiffness", 1.0e5, 3.0e6, "N/m^3", scale="log"),
            ParameterRange("load_amplitude", -9000.0, -2000.0, "N/m^2"),
            ParameterRange("absorbed_solar_flux", 150.0, 850.0, "W/m^2"),
            ParameterRange("contact_conductance", 75.0, 300.0, "W/(m^2 K)", scale="log"),
            ParameterRange("gap_conductance", 3.0, 15.0, "W/(m^2 K)", scale="log"),
        )
    raise ValueError("The final thermomechanical study uses cases 2, 3, and 5.")


def parameter_mapping(case: int, values: np.ndarray) -> dict[str, float]:
    spec = CASES.get(int(case))
    if spec is None:
        raise ValueError("The final thermomechanical study uses cases 2, 3, and 5.")
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != len(spec.parameter_names):
        raise ValueError(f"Case {case} expects {len(spec.parameter_names)} values.")
    return dict(zip(spec.parameter_names, map(float, vector), strict=True))
