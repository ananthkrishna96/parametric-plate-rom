"""Mechanical reference and comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paramplate.rom.metrics import relative_metric_errors

from .navier import NavierPlateSolution, solve_ssss_navier


@dataclass(frozen=True)
class FieldComparison:
    relative_l2: float
    maximum_absolute_error: float
    reference_maximum: float


def compare_arrays(reference: np.ndarray, approximation: np.ndarray) -> FieldComparison:
    ref = np.asarray(reference, dtype=float)
    app = np.asarray(approximation, dtype=float)
    if ref.shape != app.shape:
        raise ValueError(f"Field shapes differ: {ref.shape} and {app.shape}.")
    denominator = max(float(np.linalg.norm(ref)), np.finfo(float).eps)
    return FieldComparison(
        relative_l2=float(np.linalg.norm(app - ref) / denominator),
        maximum_absolute_error=float(np.max(np.abs(app - ref))),
        reference_maximum=float(np.max(np.abs(ref))),
    )


def navier_smoke_reference(*, n_terms: int = 21) -> NavierPlateSolution:
    from paramplate.core.parameters import OrthotropicRigidity

    return solve_ssss_navier(
        OrthotropicRigidity(9.5e3, 9.5e3, 2.85e3, 3.32e3),
        foundation_stiffness=1.0e3,
        load_amplitude=-1000.0,
        load="uniform",
        n_terms=n_terms,
        nx=41,
        ny=21,
    )
