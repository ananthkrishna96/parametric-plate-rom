"""Three-dimensional steady heat-conduction stage and plate reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from paramplate.core.fenics import require_fenics
from paramplate.core.parameters import PlateGeometry, SolverTolerances, ThermalEnvironment


@dataclass(frozen=True)
class HeatFEMConfig:
    geometry: PlateGeometry
    environment: ThermalEnvironment
    nx: int = 64
    ny: int = 32
    nz: int = 16
    degree: int = 1
    theta_degree: int = 1
    theta_quadrature_points: int = 20
    tolerances: SolverTolerances = SolverTolerances(1.0e-8, 1.0e-8, 50)

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) < 1:
            raise ValueError("Heat mesh counts must be positive.")
        if self.degree < 1 or self.theta_degree < 1:
            raise ValueError("Heat and thermal-driver degrees must be positive.")


@dataclass
class HeatSolution:
    temperature: Any
    temperature_increment: Any
    thermal_driver: Any
    reference_temperature: float
    contact_conductance: Any

    def theta_vector(self) -> np.ndarray:
        return np.asarray(self.thermal_driver.vector().get_local(), dtype=float)


class HeatConductionFOM:
    """Steady 3-D conduction with top exchange and contact-gap bottom exchange."""

    def __init__(self, config: HeatFEMConfig, plate_mesh: Any):
        self.config = config
        self.df, _ = require_fenics(require_mshr=False)
        self.plate_mesh = plate_mesh
        self._build_mesh()

    def _build_mesh(self) -> None:
        df = self.df
        g = self.config.geometry
        h = float(g.thickness)
        self.mesh = df.BoxMesh(
            df.Point(0.0, 0.0, -0.5 * h),
            df.Point(g.length, g.width, 0.5 * h),
            int(self.config.nx),
            int(self.config.ny),
            int(self.config.nz),
        )
        self.facets = df.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1, 0)

        class Top(df.SubDomain):
            def inside(inner_self, x, on_boundary):  # noqa: ANN001
                return on_boundary and df.near(x[2], 0.5 * h)

        class Bottom(df.SubDomain):
            def inside(inner_self, x, on_boundary):  # noqa: ANN001
                return on_boundary and df.near(x[2], -0.5 * h)

        class Lateral(df.SubDomain):
            def inside(inner_self, x, on_boundary):  # noqa: ANN001
                return on_boundary and (
                    df.near(x[0], 0.0)
                    or df.near(x[0], g.length)
                    or df.near(x[1], 0.0)
                    or df.near(x[1], g.width)
                )

        Top().mark(self.facets, 1)
        Bottom().mark(self.facets, 2)
        Lateral().mark(self.facets, 3)
        self.V = df.FunctionSpace(self.mesh, "CG", int(self.config.degree))
        self.V_theta = df.FunctionSpace(self.plate_mesh, "CG", int(self.config.theta_degree))

    def _lift_displacement(self, displacement: Any | None) -> Any:
        df = self.df
        eta = float(self.config.environment.contact_transition)
        if displacement is None:
            return df.Constant(-10.0 / eta)
        if isinstance(displacement, (float, int, np.floating)):
            return df.Constant(float(displacement))
        try:
            if displacement.function_space().mesh().geometry().dim() == 3:
                return displacement
        except Exception:
            pass

        class Lift(df.UserExpression):
            def __init__(inner_self, field, **kwargs):
                super().__init__(**kwargs)
                inner_self.field = field

            def eval(inner_self, values, x):  # noqa: ANN001
                values[0] = float(inner_self.field(df.Point(float(x[0]), float(x[1]))))

            def value_shape(inner_self):
                return ()

        return Lift(displacement, degree=max(2, int(self.config.degree) + 1))

    def solve(self, displacement: Any | None, *, reference_temperature: float | None = None) -> HeatSolution:
        df = self.df
        env = self.config.environment
        temperature = df.Function(self.V, name="temperature")
        temperature.interpolate(df.Constant(float(env.ambient_temperature)))
        test = df.TestFunction(self.V)
        dx = df.Measure("dx", domain=self.mesh)
        ds = df.Measure("ds", domain=self.mesh, subdomain_data=self.facets)
        conductivity = df.as_tensor(
            (
                (float(env.conductivity_x), 0.0, 0.0),
                (0.0, float(env.conductivity_y), 0.0),
                (0.0, 0.0, float(env.conductivity_z)),
            )
        )
        w3 = self._lift_displacement(displacement)
        switch = 0.5 * (1.0 + df.tanh(float(env.contact_transition) * w3))
        hc = float(env.contact_conductance) + (
            float(env.gap_conductance) - float(env.contact_conductance)
        ) * switch
        sigma = 5.67e-8
        radiation = (
            float(env.emissivity) * sigma * (temperature**4 - float(env.ambient_temperature) ** 4)
            if env.radiation
            else 0.0
        )
        residual = (
            df.inner(conductivity * df.grad(temperature), df.grad(test)) * dx
            + float(env.top_convection) * (temperature - float(env.ambient_temperature)) * test * ds(1)
            + radiation * test * ds(1)
            - float(env.absorbed_solar_flux) * test * ds(1)
            + hc * (temperature - float(env.substrate_temperature)) * test * ds(2)
        )
        jacobian = df.derivative(residual, temperature)
        problem = df.NonlinearVariationalProblem(residual, temperature, [], jacobian)
        solver = df.NonlinearVariationalSolver(problem)
        newton = solver.parameters["newton_solver"]
        newton["absolute_tolerance"] = float(self.config.tolerances.absolute)
        newton["relative_tolerance"] = float(self.config.tolerances.relative)
        newton["maximum_iterations"] = int(self.config.tolerances.maximum_iterations)
        newton["linear_solver"] = "mumps"
        solver.solve()
        if reference_temperature is None:
            volume = float(df.assemble(df.Constant(1.0) * dx))
            tref = float(df.assemble(temperature * dx) / volume)
        else:
            tref = float(reference_temperature)
        delta = df.project(temperature - df.Constant(tref), self.V)
        delta.rename("temperature_increment", "")
        theta = self._reduce_to_plate(delta)
        return HeatSolution(temperature, delta, theta, tref, hc)

    def _reduce_to_plate(self, delta_temperature: Any) -> Any:
        df = self.df
        h = float(self.config.geometry.thickness)
        xi, weights = np.polynomial.legendre.leggauss(int(self.config.theta_quadrature_points))
        try:
            delta_temperature.set_allow_extrapolation(True)
        except Exception:
            pass

        class ThermalDriver(df.UserExpression):
            def eval(inner_self, values, x):  # noqa: ANN001
                integral = 0.0
                for point, weight in zip(xi, weights, strict=True):
                    z = 0.5 * h * float(point)
                    value = float(delta_temperature(df.Point(float(x[0]), float(x[1]), z)))
                    integral += float(weight) * z * value
                integral *= 0.5 * h
                values[0] = 12.0 * integral / h**3

            def value_shape(inner_self):
                return ()

        theta = df.interpolate(
            ThermalDriver(degree=max(2, int(self.config.theta_degree) + 2)),
            self.V_theta,
        )
        theta.rename("thermal_driver_theta", "K_per_m")
        return theta
