"""Regularized unilateral Winkler law under the upward-positive convention."""

from __future__ import annotations

import numpy as np


def active_foundation_stiffness(
    displacement: np.ndarray | float,
    stiffness: float,
    uplift_factor: float = 1.0e-3,
) -> np.ndarray:
    """Return ``k_s`` in compression (``w<0``) and ``epsilon k_s`` in uplift."""

    w = np.asarray(displacement, dtype=float)
    if stiffness < 0.0 or uplift_factor < 0.0:
        raise ValueError("Foundation stiffness and uplift factor must be nonnegative.")
    return np.where(w < 0.0, float(stiffness), float(uplift_factor * stiffness))


def foundation_reaction(
    displacement: np.ndarray | float,
    stiffness: float,
    uplift_factor: float = 1.0e-3,
) -> np.ndarray:
    """Return the algebraic foundation term ``k_active w``.

    With upward-positive displacement, downward deflection is negative and the
    compression branch is selected. The weak residual uses this signed term.
    """

    w = np.asarray(displacement, dtype=float)
    return active_foundation_stiffness(w, stiffness, uplift_factor) * w


def foundation_energy_density(
    displacement: np.ndarray | float,
    stiffness: float,
    uplift_factor: float = 1.0e-3,
) -> np.ndarray:
    w = np.asarray(displacement, dtype=float)
    return 0.5 * active_foundation_stiffness(w, stiffness, uplift_factor) * w**2


def compression_mask(displacement: np.ndarray | float) -> np.ndarray:
    return np.asarray(displacement, dtype=float) < 0.0
