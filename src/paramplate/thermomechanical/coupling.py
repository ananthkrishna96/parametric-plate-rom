"""Relaxed partitioned thermomechanical coupling protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np

THeat = TypeVar("THeat")


@dataclass(frozen=True)
class CouplingIteration:
    iteration: int
    displacement_update: float
    thermal_driver_update: float


@dataclass(frozen=True)
class CoupledState(Generic[THeat]):
    displacement: np.ndarray
    thermal_driver: np.ndarray
    heat_state: THeat
    converged: bool
    iterations: int
    history: tuple[CouplingIteration, ...]


def relative_update(new: np.ndarray, old: np.ndarray, *, floor: float = 1.0e-14) -> float:
    new_array = np.asarray(new, dtype=float)
    old_array = np.asarray(old, dtype=float)
    if new_array.shape != old_array.shape:
        raise ValueError("Coupling fields must retain their output-space shapes.")
    return float(np.linalg.norm(new_array - old_array) / max(np.linalg.norm(new_array), floor))


def relaxed_partitioned_solve(
    initial_displacement: np.ndarray,
    *,
    thermal_solve: Callable[[np.ndarray], tuple[THeat, np.ndarray]],
    mechanical_solve: Callable[[np.ndarray], np.ndarray],
    displacement_update_measure: Callable[[np.ndarray, np.ndarray], float] | None = None,
    thermal_driver_update_measure: Callable[[np.ndarray, np.ndarray], float] | None = None,
    relaxation: float = 0.7,
    tolerance: float = 1.0e-5,
    maximum_iterations: int = 25,
) -> CoupledState[THeat]:
    """Alternate heat reduction and mechanics using the thesis acceptance rule.

    On convergence, the accepted displacement is the last *relaxed* iterate and
    the accepted thermal driver is the one computed from the displacement at the
    start of that same coupling iteration.  No unreported post-loop correction is
    performed.
    """

    omega = float(relaxation)
    if not (0.0 < omega <= 1.0):
        raise ValueError("relaxation must lie in (0, 1].")
    if tolerance <= 0.0 or maximum_iterations < 1:
        raise ValueError("Invalid coupling tolerance or maximum_iterations.")
    measure_w = displacement_update_measure or relative_update
    measure_theta = thermal_driver_update_measure or relative_update
    displacement = np.asarray(initial_displacement, dtype=float).copy()
    previous_theta: np.ndarray | None = None
    history: list[CouplingIteration] = []
    last_heat: THeat | None = None
    last_theta: np.ndarray | None = None
    for iteration in range(1, int(maximum_iterations) + 1):
        heat_state, theta = thermal_solve(displacement)
        theta = np.asarray(theta, dtype=float)
        candidate = np.asarray(mechanical_solve(theta), dtype=float)
        if candidate.shape != displacement.shape:
            raise ValueError("The mechanical output changed shape during coupling.")
        relaxed = (1.0 - omega) * displacement + omega * candidate
        error_w = float(measure_w(relaxed, displacement))
        error_theta = np.inf if previous_theta is None else float(measure_theta(theta, previous_theta))
        if not np.isfinite(error_w) or (previous_theta is not None and not np.isfinite(error_theta)):
            raise ValueError("Coupling update measures must return finite nonnegative values.")
        if error_w < 0.0 or (previous_theta is not None and error_theta < 0.0):
            raise ValueError("Coupling update measures must return finite nonnegative values.")
        history.append(CouplingIteration(iteration, error_w, float(error_theta)))
        last_heat, last_theta = heat_state, theta.copy()
        displacement = relaxed
        if previous_theta is not None and max(error_w, error_theta) <= float(tolerance):
            return CoupledState(
                displacement=displacement,
                thermal_driver=last_theta,
                heat_state=last_heat,
                converged=True,
                iterations=iteration,
                history=tuple(history),
            )
        previous_theta = theta.copy()
    assert last_heat is not None and last_theta is not None
    return CoupledState(
        displacement=displacement,
        thermal_driver=last_theta,
        heat_state=last_heat,
        converged=False,
        iterations=int(maximum_iterations),
        history=tuple(history),
    )
