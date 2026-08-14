"""Active-degree restriction for panelized transient systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ActiveDOFSystem:
    stiffness: sparse.csr_matrix
    mass: sparse.csr_matrix
    active_dofs: np.ndarray
    constrained_dofs: np.ndarray
    removed_zero_mass_dofs: np.ndarray
    mass_tolerance: float


def restrict_active_dofs(
    stiffness,
    mass,
    *,
    constrained_dofs: np.ndarray | None = None,
    relative_mass_tolerance: float = 1.0e-12,
) -> ActiveDOFSystem:
    """Eliminate essential constraints and zero-mass mixed-space ghost entries."""

    K = sparse.csr_matrix(stiffness)
    M = sparse.csr_matrix(mass)
    if K.shape != M.shape or K.shape[0] != K.shape[1]:
        raise ValueError("Stiffness and mass matrices must be square and shape-compatible.")
    constrained = np.asarray([] if constrained_dofs is None else constrained_dofs, dtype=np.int64)
    constrained = np.unique(constrained)
    all_dofs = np.arange(K.shape[0], dtype=np.int64)
    free = np.setdiff1d(all_dofs, constrained, assume_unique=True)
    Kff = K[free, :][:, free].tocsr()
    Mff = M[free, :][:, free].tocsr()
    diagonal = np.abs(np.asarray(Mff.diagonal(), dtype=float))
    if diagonal.size == 0 or not np.all(np.isfinite(diagonal)):
        raise ValueError("The reduced mass diagonal is empty or non-finite.")
    maximum = float(np.max(diagonal))
    if maximum <= 0.0:
        raise ValueError("All free degrees of freedom have zero mass.")
    tolerance = float(relative_mass_tolerance) * maximum
    active_local = np.flatnonzero(diagonal > tolerance).astype(np.int64)
    inactive_local = np.flatnonzero(diagonal <= tolerance).astype(np.int64)
    removed = free[inactive_local]
    active = free[active_local]
    Kaa = Kff[active_local, :][:, active_local].tocsr()
    Maa = Mff[active_local, :][:, active_local].tocsr()
    if active.size == 0:
        raise ValueError("No active dynamic degrees of freedom remain.")
    if np.min(np.abs(Maa.diagonal())) <= 0.0:
        raise ValueError("Zero mass remains after active-degree restriction.")
    return ActiveDOFSystem(
        stiffness=Kaa,
        mass=Maa,
        active_dofs=active,
        constrained_dofs=np.unique(np.concatenate([constrained, removed])),
        removed_zero_mass_dofs=removed,
        mass_tolerance=tolerance,
    )
