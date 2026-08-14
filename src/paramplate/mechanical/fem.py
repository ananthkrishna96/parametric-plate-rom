"""FEniCS 2019 implementation of the static orthotropic plate FOM.

The module keeps the legacy solver dependency lazy.  Importing :mod:`paramplate`
does not require DOLFIN; constructing :class:`MechanicalPlateFOM` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from paramplate.core.fenics import require_fenics
from paramplate.core.parameters import (
    FoundationParameters,
    LoadParameters,
    OrthotropicRigidity,
    PlateGeometry,
    SolverTolerances,
)


@dataclass(frozen=True)
class MechanicalFEMConfig:
    geometry: PlateGeometry = field(default_factory=PlateGeometry)
    rigidity: OrthotropicRigidity = field(
        default_factory=lambda: OrthotropicRigidity(9.5e3, 9.5e3, 2.85e3, 3.32e3)
    )
    foundation: FoundationParameters = field(
        default_factory=lambda: FoundationParameters(1.0e7, 1.0e-3)
    )
    load: LoadParameters = field(default_factory=LoadParameters)
    boundary_condition: str = "free_edge"
    tolerances: SolverTolerances = field(default_factory=SolverTolerances)
    linear_solver: str = "mumps"
    thermal_driver: Any | None = None
    thermal_expansion_x: float = 0.0
    thermal_expansion_y: float = 0.0
    fixed_foundation_factor: float | None = None

    def __post_init__(self) -> None:
        if self.boundary_condition not in {"free_edge", "simply_supported"}:
            raise ValueError("boundary_condition must be free_edge or simply_supported.")
        if self.fixed_foundation_factor is not None and self.fixed_foundation_factor < 0.0:
            raise ValueError("fixed_foundation_factor must be nonnegative when supplied.")


@dataclass(frozen=True)
class NewtonRecord:
    converged: bool
    iterations: int
    initial_residual: float
    final_residual: float
    threshold: float


@dataclass
class MechanicalSolution:
    native: Any
    displacement: Any
    newton: NewtonRecord
    parameters: Mapping[str, float]

    def vector(self, output: str = "global") -> np.ndarray:
        fn = self.displacement if output == "global" else self.native
        return np.asarray(fn.vector().get_local(), dtype=float)


@dataclass
class _PlateDomain:
    mesh: Any
    subdomains: Any
    facets: Any
    interface_pairs: tuple[tuple[int, int, int], ...]
    panel_ids: tuple[int, ...]
    boundary_ids: tuple[int, ...] = (1, 2, 3, 4)


class MechanicalPlateFOM:
    """Nonlinear C0-IPG plate solver with a regularized unilateral foundation.

    The sign convention follows the thesis: displacement and transverse load are
    upward-positive.  Consequently a downward pressure has a negative value, and
    the compression branch of the foundation is active where ``w < 0``.
    """

    def __init__(self, config: MechanicalFEMConfig, *, domain: _PlateDomain | None = None):
        self.config = config
        self.df, self.mshr = require_fenics(require_mshr=True)
        self.domain = self._build_domain() if domain is None else domain
        self._build_spaces()
        self._build_forms()

    @property
    def n_native_dofs(self) -> int:
        return int(self.W.dim())

    @property
    def n_output_dofs(self) -> int:
        return int(self.V_output.dim())

    def _build_domain(self) -> _PlateDomain:
        df, mshr = self.df, self.mshr
        g = self.config.geometry
        rectangle = mshr.Rectangle(df.Point(0.0, 0.0), df.Point(g.length, g.width))
        xs = np.linspace(0.0, g.length, g.n_vertical_interfaces + 2)
        ys = np.linspace(0.0, g.width, g.n_horizontal_interfaces + 2)
        sid = 1
        for ix in range(len(xs) - 1):
            for iy in range(len(ys) - 1):
                rectangle.set_subdomain(
                    sid,
                    mshr.Rectangle(df.Point(xs[ix], ys[iy]), df.Point(xs[ix + 1], ys[iy + 1])),
                )
                sid += 1
        mesh = mshr.generate_mesh(rectangle, int(g.resolution))
        subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
        facets = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

        class CoordinateBoundary(df.SubDomain):
            def __init__(self, axis: int, value: float):
                super().__init__()
                self.axis, self.value = axis, float(value)

            def inside(self, x, on_boundary):  # noqa: ANN001 - DOLFIN callback signature
                return df.near(x[self.axis], self.value)

        CoordinateBoundary(0, 0.0).mark(facets, 1)
        CoordinateBoundary(1, g.width).mark(facets, 2)
        CoordinateBoundary(0, g.length).mark(facets, 3)
        CoordinateBoundary(1, 0.0).mark(facets, 4)
        marker = 5
        for x in xs[1:-1]:
            CoordinateBoundary(0, float(x)).mark(facets, marker)
            marker += 1
        for y in ys[1:-1]:
            CoordinateBoundary(1, float(y)).mark(facets, marker)
            marker += 1

        interface_pairs: set[tuple[int, int, int]] = set()
        if g.n_panels > 1:
            mesh.init(mesh.topology().dim() - 1, mesh.topology().dim())
            cell_ids = np.asarray(subdomains.array(), dtype=int)
            facet_ids = np.asarray(facets.array(), dtype=int)
            for facet in df.facets(mesh):
                if facet.exterior():
                    continue
                cells = facet.entities(mesh.topology().dim())
                if len(cells) != 2:
                    continue
                left, right = int(cell_ids[cells[0]]), int(cell_ids[cells[1]])
                if left != right:
                    interface_pairs.add((min(left, right), max(left, right), int(facet_ids[facet.index()])))
        return _PlateDomain(
            mesh=mesh,
            subdomains=subdomains,
            facets=facets,
            interface_pairs=tuple(sorted(interface_pairs)),
            panel_ids=tuple(range(1, g.n_panels + 1)),
        )

    def _build_spaces(self) -> None:
        df = self.df
        g = self.config.geometry
        cell = self.domain.mesh.ufl_cell()
        element = df.FiniteElement("Lagrange", cell, int(g.degree))
        self.V_output = df.FunctionSpace(self.domain.mesh, element)
        self.V_dg0 = df.FunctionSpace(self.domain.mesh, "DG", 0)
        if g.n_panels == 1:
            self.W = self.V_output
        else:
            self.W = df.FunctionSpace(
                self.domain.mesh,
                df.MixedElement([element] * g.n_panels),
            )
        self.state = df.Function(self.W, name="plate_displacement_native")
        self._indicator = self._build_indicators()

    def _build_indicators(self) -> dict[int, Any]:
        df = self.df
        if self.config.geometry.n_panels == 1:
            return {1: df.Constant(1.0)}
        labels = np.asarray(self.domain.subdomains.array(), dtype=int)
        indicators: dict[int, Any] = {}
        for sid in self.domain.panel_ids:
            chi = df.Function(self.V_dg0, name=f"chi_panel_{sid}")
            chi.vector()[:] = (labels == sid).astype(float)
            indicators[sid] = chi
        return indicators

    @staticmethod
    def _physical_trace(field: Any, indicator: Any) -> Any:
        return indicator("-") * field("-") + indicator("+") * field("+")

    def _physical_gradient_trace(self, field: Any, indicator: Any) -> Any:
        df = self.df
        return indicator("-") * df.grad(field)("-") + indicator("+") * df.grad(field)("+")

    def _load_expression(self, load_parameters: LoadParameters | None = None) -> Any:
        df = self.df
        load = self.config.load if load_parameters is None else load_parameters
        if load.kind == "uniform":
            return df.Constant(float(load.amplitude))
        if load.kind == "patch":
            return df.Expression(
                "((x[0]>=xa)&&(x[0]<=xb)&&(x[1]>=ya)&&(x[1]<=yb)) ? q : 0.0",
                degree=0,
                xa=float(load.x0 - load.size_x / 2.0),
                xb=float(load.x0 + load.size_x / 2.0),
                ya=float(load.y0 - load.size_y / 2.0),
                yb=float(load.y0 + load.size_y / 2.0),
                q=float(load.amplitude),
            )
        if load.kind == "linear_x":
            return df.Expression(
                "q0 + (q1-q0)*x[0]/L",
                degree=1,
                q0=float(load.linear_start),
                q1=float(load.amplitude),
                L=float(self.config.geometry.length),
            )
        if load.kind == "modal_sine":
            return df.Expression(
                "q*sin(pi*x[0]/L)*sin(pi*x[1]/B)",
                degree=4,
                q=float(load.amplitude),
                pi=float(np.pi),
                L=float(self.config.geometry.length),
                B=float(self.config.geometry.width),
            )
        if load.kind == "multi_patch":
            expressions = []
            for patch in load.patches:
                sx = float(patch.get("size_x", patch.get("patch_size_x", load.size_x)))
                sy = float(patch.get("size_y", patch.get("patch_size_y", load.size_y)))
                x0, y0 = float(patch["x0"]), float(patch["y0"])
                expressions.append(
                    df.Expression(
                        "((x[0]>=xa)&&(x[0]<=xb)&&(x[1]>=ya)&&(x[1]<=yb)) ? q : 0.0",
                        degree=0,
                        xa=x0 - sx / 2.0,
                        xb=x0 + sx / 2.0,
                        ya=y0 - sy / 2.0,
                        yb=y0 + sy / 2.0,
                        q=float(patch.get("amplitude", patch.get("load_value", load.amplitude))),
                    )
                )
            return sum(expressions, df.Constant(0.0))
        raise ValueError(f"Unsupported load kind: {load.kind}")

    def _build_forms(self) -> None:
        df = self.df
        g, r, foundation = self.config.geometry, self.config.rigidity, self.config.foundation
        dx = df.Measure("dx", domain=self.domain.mesh, subdomain_data=self.domain.subdomains)
        dS = df.Measure("dS", domain=self.domain.mesh, subdomain_data=self.domain.facets)
        n = df.FacetNormal(self.domain.mesh)
        h_avg = (df.CellDiameter(self.domain.mesh)("+") + df.CellDiameter(self.domain.mesh)("-")) / 2.0
        D_eff = float(r.effective)
        stabilization = float((g.degree + 2) ** 2 * D_eff) / h_avg
        alpha_seam = float(g.degree * D_eff) / h_avg**3
        beta_seam = float((12 ** (g.degree + 1) / g.degree) * D_eff) / h_avg

        components = df.split(self.state) if g.n_panels > 1 else (self.state,)
        energy = 0
        cdg = 0
        support = 0
        load_work = 0
        load = self._load_expression()
        for index, wi in enumerate(components, start=1):
            curvature = df.variable(df.sym(df.grad(df.grad(wi))))
            kx, ky, kxy = curvature[0, 0], curvature[1, 1], curvature[0, 1]
            bending_density = (
                0.5 * r.Dx * kx**2
                + r.Dxy * kx * ky
                + 0.5 * r.Dy * ky**2
                + 2.0 * r.Ds * kxy**2
            )
            local_dx = dx(index) if g.n_panels > 1 else dx
            energy += bending_density * local_dx
            if self.config.thermal_driver is not None:
                theta = self.config.thermal_driver
                mx_thermal = (r.Dx * self.config.thermal_expansion_x + r.Dxy * self.config.thermal_expansion_y) * theta
                my_thermal = (r.Dxy * self.config.thermal_expansion_x + r.Dy * self.config.thermal_expansion_y) * theta
                energy += (mx_thermal * kx + my_thermal * ky) * local_dx
            moment = df.diff(bending_density, curvature)
            normal_moment = df.inner(moment, df.outer(n, n))
            cdg_density = -df.inner(df.jump(df.grad(wi), n), df.avg(normal_moment))
            cdg_density += 0.5 * stabilization * df.inner(df.jump(df.grad(wi), n), df.jump(df.grad(wi), n))
            if g.n_panels > 1:
                chi = self._indicator[index]
                cdg += cdg_density * chi("+") * chi("-") * dS
            else:
                cdg += cdg_density * dS(0)
            branch_factor = (
                df.conditional(wi < 0.0, 1.0, float(foundation.uplift_factor))
                if self.config.fixed_foundation_factor is None
                else df.Constant(float(self.config.fixed_foundation_factor))
            )
            support += 0.5 * float(foundation.stiffness) * branch_factor * wi**2 * local_dx
            load_work += load * wi * local_dx

        seam_energy = 0
        if g.n_panels > 1:
            for i, j, marker in self.domain.interface_pairs:
                wi = self._physical_trace(components[i - 1], self._indicator[i])
                wj = self._physical_trace(components[j - 1], self._indicator[j])
                gi = self._physical_gradient_trace(components[i - 1], self._indicator[i])
                gj = self._physical_gradient_trace(components[j - 1], self._indicator[j])
                jump_w = wi - wj
                jump_rotation = df.inner(gi - gj, n("-"))
                seam_energy += (
                    0.5 * alpha_seam * jump_w**2 + 0.5 * beta_seam * jump_rotation**2
                ) * dS(marker)

        test = df.TestFunction(self.W)
        trial = df.TrialFunction(self.W)
        total_potential = energy + cdg + support + seam_energy - load_work
        self.residual_form = df.derivative(total_potential, self.state, test)
        self.jacobian_form = df.derivative(self.residual_form, self.state, trial)
        self.bcs = self._make_boundary_conditions()

    def _make_boundary_conditions(self) -> list[Any]:
        df = self.df
        if self.config.boundary_condition == "free_edge":
            return []
        if self.config.geometry.n_panels == 1:
            return [
                df.DirichletBC(self.W, df.Constant(0.0), self.domain.facets, marker)
                for marker in self.domain.boundary_ids
            ]
        return [
            df.DirichletBC(self.W.sub(panel), df.Constant(0.0), self.domain.facets, marker)
            for panel in range(self.config.geometry.n_panels)
            for marker in self.domain.boundary_ids
        ]


    def assemble_external_load(self, load_parameters: LoadParameters | None = None) -> Any:
        """Assemble the native-space transverse load vector without stiffness terms."""

        df = self.df
        g = self.config.geometry
        dx = df.Measure("dx", domain=self.domain.mesh, subdomain_data=self.domain.subdomains)
        test = df.TestFunction(self.W)
        components = df.split(test) if g.n_panels > 1 else (test,)
        load = self._load_expression(load_parameters)
        form = sum(
            (load * component * (dx(index) if g.n_panels > 1 else dx)
             for index, component in enumerate(components, start=1)),
            df.Constant(0.0) * dx,
        )
        vector = df.assemble(form)
        for bc in self.bcs:
            bc.apply(vector)
        return vector

    def set_initial_vector(self, values: Sequence[float]) -> None:
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != self.n_native_dofs:
            raise ValueError(f"Expected {self.n_native_dofs} native values, got {vector.size}.")
        self.state.vector().set_local(vector)
        self.state.vector().apply("insert")

    def assemble_residual(self) -> Any:
        vector = self.df.assemble(self.residual_form)
        for bc in self.bcs:
            bc.apply(vector, self.state.vector())
        return vector

    def assemble_jacobian(self) -> Any:
        matrix = self.df.assemble(self.jacobian_form)
        for bc in self.bcs:
            bc.apply(matrix)
        # Mixed panel spaces contain component degrees of freedom outside each
        # component's physical panel. Their residual entries are zero; identity
        # rows keep the Newton system nonsingular without altering active rows.
        matrix.ident_zeros()
        return matrix

    def solve(self, *, initial: Sequence[float] | None = None) -> MechanicalSolution:
        df = self.df
        if initial is not None:
            self.set_initial_vector(initial)
        tol = self.config.tolerances
        residual = self.assemble_residual()
        initial_norm = float(residual.norm("l2"))
        threshold = max(float(tol.absolute), float(tol.relative) * initial_norm)
        final_norm = initial_norm
        converged = final_norm <= threshold
        iteration = 0
        while not converged and iteration < tol.maximum_iterations:
            matrix = self.assemble_jacobian()
            residual *= -1.0
            increment = df.Function(self.W)
            df.solve(matrix, increment.vector(), residual, self.config.linear_solver)
            self.state.vector().axpy(1.0, increment.vector())
            self.state.vector().apply("insert")
            iteration += 1
            residual = self.assemble_residual()
            final_norm = float(residual.norm("l2"))
            converged = final_norm <= threshold
        if not converged:
            raise RuntimeError(
                f"Mechanical Newton solve did not converge: residual={final_norm:.6e}, "
                f"threshold={threshold:.6e}, iterations={iteration}."
            )
        output = self.transfer_to_output(self.state)
        record = NewtonRecord(converged, iteration, initial_norm, final_norm, threshold)
        r, f, q = self.config.rigidity, self.config.foundation, self.config.load
        parameters = {
            "Dx": r.Dx,
            "Dy": r.Dy,
            "Dxy": r.Dxy,
            "Ds": r.Ds,
            "foundation_stiffness": f.stiffness,
            "load_amplitude": q.amplitude,
        }
        return MechanicalSolution(self.state.copy(deepcopy=True), output, record, parameters)

    def transfer_to_output(self, native: Any) -> Any:
        df = self.df
        if self.config.geometry.n_panels == 1:
            result = df.Function(self.V_output, name="displacement")
            result.assign(native)
            return result
        pieces = df.split(native)
        physical = sum(
            (self._indicator[sid] * pieces[sid - 1] for sid in self.domain.panel_ids),
            df.Constant(0.0),
        )
        return df.project(physical, self.V_output, function=self._named_output())

    def _named_output(self) -> Any:
        return self.df.Function(self.V_output, name="displacement")

    def output_metric(self, kind: str = "h1") -> Any:
        df = self.df
        u, v = df.TrialFunction(self.V_output), df.TestFunction(self.V_output)
        form = u * v * df.dx
        if kind.lower() in {"h1", "h1-type", "energy"}:
            form += df.inner(df.grad(u), df.grad(v)) * df.dx
        elif kind.lower() != "l2":
            raise ValueError("kind must be l2 or h1.")
        return df.assemble(form)

    def native_metric(self, kind: str = "h1") -> Any:
        df = self.df
        u, v = df.TrialFunction(self.W), df.TestFunction(self.W)
        form = df.inner(u, v) * df.dx
        if kind.lower() in {"h1", "h1-type", "energy"}:
            form += df.inner(df.grad(u), df.grad(v)) * df.dx
        elif kind.lower() != "l2":
            raise ValueError("kind must be l2 or h1.")
        return df.assemble(form)
