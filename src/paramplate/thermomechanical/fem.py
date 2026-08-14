"""Physically coupled steady thermomechanical full-order workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from paramplate.core.parameters import ThermalEnvironment
from paramplate.mechanical.fem import MechanicalFEMConfig, MechanicalPlateFOM, MechanicalSolution

from .coupling import CoupledState, relaxed_partitioned_solve
from .heat import HeatConductionFOM, HeatFEMConfig, HeatSolution


@dataclass(frozen=True)
class CouplingConfig:
    relaxation: float = 0.7
    tolerance: float = 1.0e-5
    maximum_iterations: int = 25


@dataclass(frozen=True)
class ThermomechanicalFEMConfig:
    mechanical: MechanicalFEMConfig = field(default_factory=MechanicalFEMConfig)
    environment: ThermalEnvironment = field(default_factory=ThermalEnvironment)
    heat_nx: int = 64
    heat_ny: int = 32
    heat_nz: int = 16
    heat_degree: int = 1
    theta_degree: int = 1
    theta_quadrature_points: int = 20
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    reference_temperature: float = 293.15


@dataclass
class ThermomechanicalSolution:
    displacement: Any
    thermal_driver: Any
    heat: HeatSolution
    last_mechanical_candidate: MechanicalSolution
    coupled_state: CoupledState[HeatSolution]

    def displacement_vector(self) -> np.ndarray:
        return np.asarray(self.displacement.vector().get_local(), dtype=float)

    def theta_vector(self) -> np.ndarray:
        return np.asarray(self.thermal_driver.vector().get_local(), dtype=float)


class ThermomechanicalPlateFOM:
    """Partitioned 3-D heat / 2-D plate solver used by the steady study."""

    def __init__(self, config: ThermomechanicalFEMConfig):
        self.config = config
        self.base_mechanics = MechanicalPlateFOM(replace(config.mechanical, thermal_driver=None))
        heat_config = HeatFEMConfig(
            geometry=config.mechanical.geometry,
            environment=config.environment,
            nx=config.heat_nx,
            ny=config.heat_ny,
            nz=config.heat_nz,
            degree=config.heat_degree,
            theta_degree=config.theta_degree,
            theta_quadrature_points=config.theta_quadrature_points,
        )
        self.heat_solver = HeatConductionFOM(heat_config, self.base_mechanics.domain.mesh)

    def _function_from_vector(self, values: np.ndarray, *, name: str) -> Any:
        fn = self.base_mechanics.df.Function(self.base_mechanics.V_output, name=name)
        fn.vector().set_local(np.asarray(values, dtype=float))
        fn.vector().apply("insert")
        return fn

    def _relative_l2_update(self, function_space: Any, *, floor: float = 1.0e-14):
        """Return the finite-element L2 update measure used by the coupling loop."""

        df = self.base_mechanics.df
        current = df.Function(function_space)
        previous = df.Function(function_space)
        difference = df.Function(function_space)

        def measure(new: np.ndarray, old: np.ndarray) -> float:
            new_values = np.asarray(new, dtype=float)
            old_values = np.asarray(old, dtype=float)
            if new_values.shape != old_values.shape or new_values.size != function_space.dim():
                raise ValueError("Coupling vectors do not match the finite-element output space.")
            current.vector().set_local(new_values)
            current.vector().apply("insert")
            previous.vector().set_local(old_values)
            previous.vector().apply("insert")
            difference.vector().zero()
            difference.vector().axpy(1.0, current.vector())
            difference.vector().axpy(-1.0, previous.vector())
            difference.vector().apply("insert")
            numerator = float(df.norm(difference, "L2"))
            denominator = max(float(df.norm(current, "L2")), float(floor))
            return numerator / denominator

        return measure

    def solve(self, initial_displacement: Any | None = None) -> ThermomechanicalSolution:
        df = self.base_mechanics.df
        if initial_displacement is None:
            initial_fn = df.Function(self.base_mechanics.V_output, name="initial_displacement")
        else:
            initial_fn = initial_displacement
        initial = np.asarray(initial_fn.vector().get_local(), dtype=float)
        holders: dict[str, Any] = {}
        last_native_guess: np.ndarray | None = None

        def thermal(vector: np.ndarray):
            displacement = self._function_from_vector(vector, name="coupling_displacement")
            heat = self.heat_solver.solve(
                displacement,
                reference_temperature=float(self.config.reference_temperature),
            )
            holders["heat"] = heat
            return heat, heat.theta_vector()

        def mechanical(theta_vector: np.ndarray):
            nonlocal last_native_guess
            theta = df.Function(self.heat_solver.V_theta, name="thermal_driver_theta")
            theta.vector().set_local(theta_vector)
            theta.vector().apply("insert")
            mechanics_config = replace(
                self.config.mechanical,
                thermal_driver=theta,
                thermal_expansion_x=float(self.config.environment.thermal_expansion_x),
                thermal_expansion_y=float(self.config.environment.thermal_expansion_y),
            )
            # The thermal driver is defined on the base plate mesh.  Reusing the
            # same domain avoids an invalid cross-mesh UFL coefficient while still
            # rebuilding the parameter-dependent mechanical forms.
            solver = MechanicalPlateFOM(mechanics_config, domain=self.base_mechanics.domain)
            candidate = solver.solve(initial=last_native_guess)
            last_native_guess = candidate.vector("native")
            holders["mechanical"] = candidate
            return candidate.vector("global")

        c = self.config.coupling
        state = relaxed_partitioned_solve(
            initial,
            thermal_solve=thermal,
            mechanical_solve=mechanical,
            displacement_update_measure=self._relative_l2_update(self.base_mechanics.V_output),
            thermal_driver_update_measure=self._relative_l2_update(self.heat_solver.V_theta),
            relaxation=c.relaxation,
            tolerance=c.tolerance,
            maximum_iterations=c.maximum_iterations,
        )
        if not state.converged:
            raise RuntimeError(
                f"Partitioned thermomechanical solve did not converge in {state.iterations} iterations."
            )
        displacement = self._function_from_vector(state.displacement, name="accepted_displacement")
        theta = df.Function(self.heat_solver.V_theta, name="accepted_thermal_driver")
        theta.vector().set_local(state.thermal_driver)
        theta.vector().apply("insert")
        return ThermomechanicalSolution(
            displacement=displacement,
            thermal_driver=theta,
            heat=state.heat_state,
            last_mechanical_candidate=holders["mechanical"],
            coupled_state=state,
        )
