"""Average-acceleration Newmark integration for linear second-order systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class NewmarkResult:
    times: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray
    mechanical_energy: np.ndarray


def average_acceleration_newmark(
    mass,
    damping,
    stiffness,
    times: np.ndarray,
    load: Callable[[float], np.ndarray],
    *,
    initial_displacement: np.ndarray | None = None,
    initial_velocity: np.ndarray | None = None,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> NewmarkResult:
    """Advance ``M u_ddot + C u_dot + K u = f(t)`` on a uniform grid."""

    M = sparse.csr_matrix(mass)
    C = sparse.csr_matrix(damping)
    K = sparse.csr_matrix(stiffness)
    if M.shape != C.shape or M.shape != K.shape or M.shape[0] != M.shape[1]:
        raise ValueError("M, C, and K must have one common square shape.")
    t = np.asarray(times, dtype=float).reshape(-1)
    if t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing and contain at least two values.")
    increments = np.diff(t)
    if not np.allclose(increments, increments[0], rtol=1e-12, atol=1e-14):
        raise ValueError("The current Newmark path requires a uniform time step.")
    if beta <= 0.0 or gamma <= 0.0:
        raise ValueError("Newmark beta and gamma must be positive.")
    dt = float(increments[0])
    n = M.shape[0]
    u = np.zeros(n) if initial_displacement is None else np.asarray(initial_displacement, dtype=float).copy()
    v = np.zeros(n) if initial_velocity is None else np.asarray(initial_velocity, dtype=float).copy()
    if u.shape != (n,) or v.shape != (n,):
        raise ValueError(f"Initial vectors must have shape ({n},).")
    f0 = np.asarray(load(float(t[0])), dtype=float).reshape(-1)
    if f0.shape != (n,):
        raise ValueError("The load callback returned an incompatible vector.")
    a = np.asarray(spsolve(M, f0 - C @ v - K @ u), dtype=float)
    if not np.all(np.isfinite(a)):
        raise RuntimeError("The initial acceleration solve failed.")

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    effective = (K + a0 * M + a1 * C).tocsr()

    displacements = np.empty((t.size, n), dtype=float)
    velocities = np.empty_like(displacements)
    accelerations = np.empty_like(displacements)
    kinetic = np.empty(t.size, dtype=float)
    potential = np.empty(t.size, dtype=float)

    def record(index: int) -> None:
        displacements[index], velocities[index], accelerations[index] = u, v, a
        kinetic[index] = 0.5 * float(v @ (M @ v))
        potential[index] = 0.5 * float(u @ (K @ u))

    record(0)
    for step in range(1, t.size):
        force = np.asarray(load(float(t[step])), dtype=float).reshape(-1)
        if force.shape != (n,):
            raise ValueError("The load callback changed vector size.")
        rhs = force + M @ (a0 * u + a2 * v + a3 * a) + C @ (a1 * u + a4 * v + a5 * a)
        new_u = np.asarray(spsolve(effective, rhs), dtype=float)
        if not np.all(np.isfinite(new_u)):
            raise RuntimeError(f"Newmark effective solve failed at step {step}.")
        new_a = a0 * (new_u - u) - a2 * v - a3 * a
        new_v = v + dt * ((1.0 - gamma) * a + gamma * new_a)
        u, v, a = new_u, new_v, new_a
        record(step)
    return NewmarkResult(
        times=t,
        displacement=displacements,
        velocity=velocities,
        acceleration=accelerations,
        kinetic_energy=kinetic,
        potential_energy=potential,
        mechanical_energy=kinetic + potential,
    )
