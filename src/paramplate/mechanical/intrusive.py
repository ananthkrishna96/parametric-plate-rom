"""Non-hyper-reduced intrusive POD--Galerkin solver for static mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .fem import MechanicalPlateFOM


@dataclass(frozen=True)
class ReducedNewtonRecord:
    converged: bool
    iterations: int
    initial_residual: float
    final_residual: float
    threshold: float


@dataclass(frozen=True)
class IntrusiveResult:
    coefficients: np.ndarray
    native_vector: np.ndarray
    output_vector: np.ndarray
    newton: ReducedNewtonRecord


class MechanicalPODGalerkin:
    """Project the assembled full residual and Jacobian onto a native POD basis.

    This implementation deliberately retains full-order residual and Jacobian
    assembly.  It is therefore the non-hyper-reduced intrusive method assessed in
    the thesis, not an affine or hyper-reduced online solver.
    """

    def __init__(self, fom: MechanicalPlateFOM, native_basis: np.ndarray):
        self.fom = fom
        basis = np.asarray(native_basis, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != fom.n_native_dofs:
            raise ValueError(
                f"Expected basis shape ({fom.n_native_dofs}, r), got {basis.shape}."
            )
        if basis.shape[1] < 1:
            raise ValueError("At least one reduced basis vector is required.")
        self.basis = basis

    def _reduced_residual(self, coefficients: np.ndarray) -> np.ndarray:
        self.fom.set_initial_vector(self.basis @ coefficients)
        residual = np.asarray(self.fom.assemble_residual().get_local(), dtype=float)
        return self.basis.T @ residual

    def _reduced_jacobian(self) -> np.ndarray:
        df = self.fom.df
        matrix = self.fom.assemble_jacobian()
        reduced = np.empty((self.basis.shape[1], self.basis.shape[1]), dtype=float)
        source = df.Function(self.fom.W)
        image = df.Function(self.fom.W)
        for column in range(self.basis.shape[1]):
            source.vector().set_local(self.basis[:, column])
            source.vector().apply("insert")
            matrix.mult(source.vector(), image.vector())
            reduced[:, column] = self.basis.T @ np.asarray(image.vector().get_local(), dtype=float)
        return reduced

    def solve(
        self,
        initial_coefficients: Sequence[float] | None = None,
        *,
        absolute_tolerance: float = 1.0e-8,
        relative_tolerance: float = 1.0e-8,
        maximum_iterations: int = 25,
    ) -> IntrusiveResult:
        rank = self.basis.shape[1]
        coefficients = (
            np.zeros(rank, dtype=float)
            if initial_coefficients is None
            else np.asarray(initial_coefficients, dtype=float).reshape(-1)
        )
        if coefficients.size != rank:
            raise ValueError(f"Expected {rank} initial coefficients, got {coefficients.size}.")
        residual = self._reduced_residual(coefficients)
        initial = float(np.linalg.norm(residual))
        threshold = max(float(absolute_tolerance), float(relative_tolerance) * initial)
        final = initial
        iteration = 0
        while final > threshold and iteration < int(maximum_iterations):
            jacobian = self._reduced_jacobian()
            increment = np.linalg.solve(jacobian, -residual)
            coefficients += increment
            residual = self._reduced_residual(coefficients)
            final = float(np.linalg.norm(residual))
            iteration += 1
        if final > threshold:
            raise RuntimeError(
                f"Reduced Newton solve did not converge: residual={final:.6e}, "
                f"threshold={threshold:.6e}."
            )
        native = self.basis @ coefficients
        self.fom.set_initial_vector(native)
        output = self.fom.transfer_to_output(self.fom.state)
        return IntrusiveResult(
            coefficients=coefficients.copy(),
            native_vector=native,
            output_vector=np.asarray(output.vector().get_local(), dtype=float),
            newton=ReducedNewtonRecord(True, iteration, initial, final, threshold),
        )
