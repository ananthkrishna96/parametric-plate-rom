"""Construction of validated solver records from concise workflow mappings."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from paramplate.core.parameters import (
    FoundationParameters,
    LoadParameters,
    OrthotropicRigidity,
    PlateGeometry,
    RayleighDamping,
    SolverTolerances,
    ThermalEnvironment,
)
from paramplate.dynamics.fem import TransientFEMConfig
from paramplate.mechanical.fem import MechanicalFEMConfig
from paramplate.thermomechanical.fem import CouplingConfig, ThermomechanicalFEMConfig


def _mapping(values: Any, label: str) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(values)


def plate_geometry(config: Mapping[str, Any]) -> PlateGeometry:
    values = _mapping(config.get("geometry", config.get("plate")), "geometry")
    aliases = {
        "mesh_resolution": "resolution",
        "n_vert": "n_vertical_interfaces",
        "n_horiz": "n_horizontal_interfaces",
    }
    normalized = {aliases.get(key, key): value for key, value in values.items()}
    return PlateGeometry(**normalized)


def rigidity(config: Mapping[str, Any]) -> OrthotropicRigidity:
    values = _mapping(config.get("rigidity", config.get("material")), "rigidity")
    if not values:
        return OrthotropicRigidity(9.5e3, 9.5e3, 2.85e3, 3.32e3)
    return OrthotropicRigidity(**values)


def foundation(config: Mapping[str, Any]) -> FoundationParameters:
    return FoundationParameters(**_mapping(config.get("foundation"), "foundation"))


def load_parameters(config: Mapping[str, Any]) -> LoadParameters:
    values = _mapping(config.get("load"), "load")
    if "patches" in values:
        values["patches"] = tuple(dict(item) for item in values["patches"])
    return LoadParameters(**values)


def solver_tolerances(config: Mapping[str, Any]) -> SolverTolerances:
    return SolverTolerances(**_mapping(config.get("solver"), "solver"))


def mechanical_config(config: Mapping[str, Any]) -> MechanicalFEMConfig:
    return MechanicalFEMConfig(
        geometry=plate_geometry(config),
        rigidity=rigidity(config),
        foundation=foundation(config),
        load=load_parameters(config),
        boundary_condition=str(config.get("boundary_condition", "free_edge")),
        tolerances=solver_tolerances(config),
        linear_solver=str(config.get("linear_solver", "mumps")),
    )


def thermomechanical_config(config: Mapping[str, Any]) -> ThermomechanicalFEMConfig:
    mechanical = mechanical_config(config)
    thermal = _mapping(config.get("thermal"), "thermal")
    environment_fields = {
        key: value
        for key, value in thermal.items()
        if key in ThermalEnvironment.__dataclass_fields__
    }
    environment = ThermalEnvironment(**environment_fields)
    heat = _mapping(config.get("heat_discretization"), "heat_discretization")
    coupling = CouplingConfig(**_mapping(config.get("coupling"), "coupling"))
    return ThermomechanicalFEMConfig(
        mechanical=mechanical,
        environment=environment,
        heat_nx=int(heat.get("nx", 64)),
        heat_ny=int(heat.get("ny", 32)),
        heat_nz=int(heat.get("nz", 16)),
        heat_degree=int(heat.get("degree", 1)),
        theta_degree=int(heat.get("theta_degree", 1)),
        theta_quadrature_points=int(heat.get("theta_quadrature_points", 20)),
        coupling=coupling,
        reference_temperature=float(thermal.get("reference_temperature", environment.reference_temperature)),
    )


def transient_config(config: Mapping[str, Any]) -> TransientFEMConfig:
    dynamics = _mapping(config.get("dynamics"), "dynamics")
    damping = RayleighDamping(**_mapping(dynamics.get("rayleigh"), "dynamics.rayleigh"))
    return TransientFEMConfig(
        mechanical=mechanical_config(config),
        density=float(dynamics.get("density", 1200.0)),
        damping=damping,
        foundation_branch=str(dynamics.get("foundation_branch", "compression")),
        final_time=float(dynamics.get("final_time", 0.20)),
        time_step=float(dynamics.get("time_step", 1.0e-3)),
        store_every=int(dynamics.get("store_every", 5)),
    )
