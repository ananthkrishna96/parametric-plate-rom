"""Analytical and algebraic checks for the transient plate implementation."""

from __future__ import annotations

import numpy as np

from paramplate.core.parameters import OrthotropicRigidity


def navier_angular_frequency(
    m: int,
    n: int,
    *,
    length: float,
    width: float,
    thickness: float,
    density: float,
    rigidity: OrthotropicRigidity,
    foundation_stiffness: float = 0.0,
) -> float:
    """Natural angular frequency of an SSSS aligned-orthotropic mode."""

    if m < 1 or n < 1:
        raise ValueError("Mode indices must be positive.")
    ax = m * np.pi / float(length)
    ay = n * np.pi / float(width)
    numerator = (
        rigidity.Dx * ax**4
        + 2.0 * (rigidity.Dxy + 2.0 * rigidity.Ds) * ax**2 * ay**2
        + rigidity.Dy * ay**4
        + float(foundation_stiffness)
    )
    return float(np.sqrt(numerator / (float(density) * float(thickness))))


def relative_energy_drift(mechanical_energy: np.ndarray) -> float:
    energy = np.asarray(mechanical_energy, dtype=float).reshape(-1)
    reference = max(abs(float(energy[0])), np.finfo(float).eps)
    return float(np.max(np.abs(energy - energy[0])) / reference)
