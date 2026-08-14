"""Final transient campaign definitions."""

from __future__ import annotations

from dataclasses import dataclass

from paramplate.core.sampling import ParameterRange


@dataclass(frozen=True)
class TransientCampaign:
    name: str
    n_trajectories: int
    parameter_names: tuple[str, ...]
    pod_rank: int
    panel_interfaces: int
    output_dimension: int


CAMPAIGNS: dict[str, TransientCampaign] = {
    "case2_monolithic": TransientCampaign("case2_monolithic", 200, ("load_amplitude",), 7, 0, 20503),
    "case2_one_interface": TransientCampaign("case2_one_interface", 250, ("load_amplitude",), 8, 1, 20375),
    "case3_monolithic": TransientCampaign("case3_monolithic", 250, ("Dx", "load_amplitude"), 8, 0, 20503),
}


def parameter_ranges(campaign: str) -> tuple[ParameterRange, ...]:
    if campaign in {"case2_monolithic", "case2_one_interface"}:
        return (ParameterRange("load_amplitude", 2.0e3, 1.0e4, "N/m^2"),)
    if campaign == "case3_monolithic":
        return (
            ParameterRange("Dx", 4.0e3, 1.4e4, "N m"),
            ParameterRange("load_amplitude", 2.0e3, 1.0e4, "N/m^2"),
        )
    raise ValueError(f"Unknown transient campaign {campaign!r}.")
