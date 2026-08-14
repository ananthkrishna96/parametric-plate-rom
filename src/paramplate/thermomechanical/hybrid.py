"""Hybrid thermomechanical workflow with reduced mechanical projection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from paramplate.mechanical.fem import MechanicalPlateFOM
from paramplate.mechanical.intrusive import MechanicalPODGalerkin

from .coupling import CoupledState, relaxed_partitioned_solve
from .fem import ThermomechanicalFEMConfig, ThermomechanicalPlateFOM
from .heat import HeatSolution


@dataclass(frozen=True)
class HybridResult:
    coupled_state: CoupledState[HeatSolution]
    final_coefficients: np.ndarray


class HybridMechanicalProjection:
    """Keep the heat/reduction/assembly stages full order and reduce mechanics only."""

    def __init__(self, config: ThermomechanicalFEMConfig, native_mechanical_basis: np.ndarray):
        self.config = config
        self.parent = ThermomechanicalPlateFOM(config)
        self.basis = np.asarray(native_mechanical_basis, dtype=float)

    def solve(self) -> HybridResult:
        df = self.parent.base_mechanics.df
        initial = np.zeros(self.parent.base_mechanics.n_output_dofs, dtype=float)
        last_coefficients = np.zeros(self.basis.shape[1], dtype=float)

        def thermal(vector: np.ndarray):
            displacement = self.parent._function_from_vector(vector, name="hybrid_displacement")
            heat = self.parent.heat_solver.solve(
                displacement,
                reference_temperature=float(self.config.reference_temperature),
            )
            return heat, heat.theta_vector()

        def mechanical(theta_vector: np.ndarray):
            nonlocal last_coefficients
            theta = df.Function(self.parent.heat_solver.V_theta, name="hybrid_theta")
            theta.vector().set_local(theta_vector)
            theta.vector().apply("insert")
            cfg = replace(
                self.config.mechanical,
                thermal_driver=theta,
                thermal_expansion_x=self.config.environment.thermal_expansion_x,
                thermal_expansion_y=self.config.environment.thermal_expansion_y,
            )
            # Keep the reduced mechanical problem on the plate mesh used by the
            # thermal-to-plate reduction.
            fom = MechanicalPlateFOM(cfg, domain=self.parent.base_mechanics.domain)
            reduced = MechanicalPODGalerkin(fom, self.basis)
            result = reduced.solve(last_coefficients)
            last_coefficients = result.coefficients
            return result.output_vector

        c = self.config.coupling
        state = relaxed_partitioned_solve(
            initial,
            thermal_solve=thermal,
            mechanical_solve=mechanical,
            displacement_update_measure=self.parent._relative_l2_update(
                self.parent.base_mechanics.V_output
            ),
            thermal_driver_update_measure=self.parent._relative_l2_update(
                self.parent.heat_solver.V_theta
            ),
            relaxation=c.relaxation,
            tolerance=c.tolerance,
            maximum_iterations=c.maximum_iterations,
        )
        if not state.converged:
            raise RuntimeError("Hybrid partitioned solve did not converge.")
        return HybridResult(state, last_coefficients.copy())
