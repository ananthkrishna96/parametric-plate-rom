"""FEniCS assembly and Newmark trajectory generation for transient plates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from paramplate.core.parameters import LoadParameters, RayleighDamping
from paramplate.mechanical.fem import MechanicalFEMConfig, MechanicalPlateFOM

from .active_dofs import ActiveDOFSystem, restrict_active_dofs
from .newmark import NewmarkResult, average_acceleration_newmark


@dataclass(frozen=True)
class TransientFEMConfig:
    mechanical: MechanicalFEMConfig = field(default_factory=MechanicalFEMConfig)
    density: float = 1200.0
    damping: RayleighDamping = field(default_factory=RayleighDamping)
    foundation_branch: str = "compression"
    final_time: float = 0.20
    time_step: float = 1.0e-3
    store_every: int = 5

    def __post_init__(self) -> None:
        if self.density <= 0.0:
            raise ValueError("density must be positive.")
        if self.foundation_branch not in {"compression", "uplift", "none"}:
            raise ValueError("foundation_branch must be compression, uplift, or none.")
        if self.final_time <= 0.0 or self.time_step <= 0.0:
            raise ValueError("Time interval and step must be positive.")
        ratio = self.final_time / self.time_step
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-10):
            raise ValueError("final_time must be an integer multiple of time_step.")


@dataclass
class TransientSolution:
    active_result: NewmarkResult
    stored_times: np.ndarray
    stored_output_snapshots: np.ndarray
    active_system: ActiveDOFSystem
    full_native_dofs: int


class TransientPlateFOM:
    """Branch-frozen C0-IPG transient solver with consistent mass."""

    def __init__(self, config: TransientFEMConfig):
        self.config = config
        factor = {
            "compression": 1.0,
            "uplift": float(config.mechanical.foundation.uplift_factor),
            "none": 0.0,
        }[config.foundation_branch]
        zero_load = replace(config.mechanical.load, amplitude=0.0)
        mechanical = replace(
            config.mechanical,
            load=zero_load,
            fixed_foundation_factor=factor,
            thermal_driver=None,
        )
        self.fom = MechanicalPlateFOM(mechanical)
        self._assemble_system()

    def _petsc_to_csr(self, matrix: Any) -> sparse.csr_matrix:
        if int(self.fom.df.MPI.size(self.fom.df.MPI.comm_world)) != 1:
            raise RuntimeError(
                "The SciPy/Newmark matrix path currently requires a serial DOLFIN run."
            )
        backend = self.fom.df.as_backend_type(matrix).mat()
        indptr, indices, data = backend.getValuesCSR()
        return sparse.csr_matrix((data, indices, indptr), shape=backend.getSize())

    def _assemble_system(self) -> None:
        df = self.fom.df
        K_full = self._petsc_to_csr(self.fom.assemble_jacobian())
        u, v = df.TrialFunction(self.fom.W), df.TestFunction(self.fom.W)
        dx = df.Measure("dx", domain=self.fom.domain.mesh, subdomain_data=self.fom.domain.subdomains)
        if self.config.mechanical.geometry.n_panels == 1:
            mass_form = float(self.config.density * self.config.mechanical.geometry.thickness) * u * v * dx
        else:
            us, vs = df.split(u), df.split(v)
            mass_form = sum(
                float(self.config.density * self.config.mechanical.geometry.thickness)
                * us[index - 1]
                * vs[index - 1]
                * dx(index)
                for index in self.fom.domain.panel_ids
            )
        M_full = self._petsc_to_csr(df.assemble(mass_form))
        constrained: set[int] = set()
        for bc in self.fom.bcs:
            constrained.update(int(index) for index in bc.get_boundary_values())
        self.active = restrict_active_dofs(
            K_full,
            M_full,
            constrained_dofs=np.asarray(sorted(constrained), dtype=np.int64),
        )
        self.damping = (
            float(self.config.damping.alpha_M) * self.active.mass
            + float(self.config.damping.alpha_K) * self.active.stiffness
        ).tocsr()

    def spatial_load(self, load: LoadParameters) -> np.ndarray:
        full = np.asarray(self.fom.assemble_external_load(load).get_local(), dtype=float)
        return full[self.active.active_dofs]

    def time_grid(self) -> np.ndarray:
        steps = int(round(self.config.final_time / self.config.time_step))
        return np.linspace(0.0, self.config.final_time, steps + 1)

    @staticmethod
    def time_factor(kind: str, *, frequency: float | None = None) -> Callable[[float], float]:
        normalized = kind.lower()
        if normalized in {"step", "constant"}:
            return lambda _time: 1.0
        if normalized in {"sine", "harmonic"}:
            if frequency is None:
                raise ValueError("A harmonic load requires an angular frequency.")
            return lambda time: float(np.sin(float(frequency) * time))
        if normalized == "cosine":
            if frequency is None:
                raise ValueError("A cosine load requires an angular frequency.")
            return lambda time: float(np.cos(float(frequency) * time))
        raise ValueError("Supported time profiles are step, sine/harmonic, and cosine.")

    def solve(
        self,
        *,
        load: LoadParameters | None = None,
        time_profile: str = "step",
        excitation_frequency: float | None = None,
    ) -> TransientSolution:
        physical_load = self.config.mechanical.load if load is None else load
        spatial = self.spatial_load(physical_load)
        factor = self.time_factor(time_profile, frequency=excitation_frequency)
        result = average_acceleration_newmark(
            self.active.mass,
            self.damping,
            self.active.stiffness,
            self.time_grid(),
            lambda time: factor(time) * spatial,
        )
        selected = np.unique(
            np.append(np.arange(0, result.times.size, max(1, int(self.config.store_every))), result.times.size - 1)
        )
        outputs = np.asarray([self._active_to_output(row) for row in result.displacement[selected]])
        return TransientSolution(
            active_result=result,
            stored_times=result.times[selected],
            stored_output_snapshots=outputs,
            active_system=self.active,
            full_native_dofs=self.fom.n_native_dofs,
        )

    def _active_to_output(self, active_vector: np.ndarray) -> np.ndarray:
        full = np.zeros(self.fom.n_native_dofs, dtype=float)
        full[self.active.active_dofs] = np.asarray(active_vector, dtype=float)
        self.fom.set_initial_vector(full)
        output = self.fom.transfer_to_output(self.fom.state)
        return np.asarray(output.vector().get_local(), dtype=float)

    def modal_frequencies(self, n_modes: int = 8) -> np.ndarray:
        values = eigsh(
            self.active.stiffness,
            M=self.active.mass,
            k=min(int(n_modes), self.active.stiffness.shape[0] - 1),
            sigma=0.0,
            which="LM",
            return_eigenvectors=False,
        )
        values = np.maximum(np.sort(np.real(values)), 0.0)
        return np.sqrt(values) / (2.0 * np.pi)
