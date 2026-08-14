"""Mechanical parameter maps used by the static study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paramplate.core.parameters import FoundationParameters, OrthotropicRigidity
from paramplate.core.sampling import ParameterRange


@dataclass(frozen=True)
class MechanicalParameters:
    rigidity: OrthotropicRigidity
    foundation: FoundationParameters
    load_amplitude: float

    @classmethod
    def from_direct_vector(cls, values) -> "MechanicalParameters":
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != 6:
            raise ValueError("Direct mechanical parameters are [Dx, Dy, Dxy, Ds, ks, q].")
        return cls(
            rigidity=OrthotropicRigidity(*vector[:4]),
            foundation=FoundationParameters(vector[4]),
            load_amplitude=float(vector[5]),
        )

    def as_vector(self) -> np.ndarray:
        r = self.rigidity
        return np.asarray([r.Dx, r.Dy, r.Dxy, r.Ds, self.foundation.stiffness, self.load_amplitude])


def direct_parameter_ranges() -> tuple[ParameterRange, ...]:
    """Six-parameter monolithic campaign used in the mechanical notebook."""

    return (
        ParameterRange("Dx", 5.0e3, 2.0e4, "N m", role="mechanical"),
        ParameterRange("Dy", 5.0e3, 2.0e4, "N m", role="mechanical"),
        ParameterRange("Dxy", 1.0e3, 5.0e3, "N m", role="mechanical"),
        ParameterRange("Ds", 1.0e3, 5.0e3, "N m", role="mechanical"),
        ParameterRange("ks", 1.0e6, 1.0e7, "N/m^3", role="foundation"),
        ParameterRange("q", -1.0e4, -2.0e3, "N/m^2", role="load"),
    )


def linked_rigidity(D_eff: float, R_B: float, R_H: float) -> OrthotropicRigidity:
    """Notebook parameterization by effective stiffness and orthotropic ratios."""

    if D_eff <= 0.0 or R_B <= 0.0 or R_H <= 0.0:
        raise ValueError("D_eff, R_B, and R_H must be positive.")
    Dx = float(D_eff)
    Dy = float(R_B * D_eff)
    return OrthotropicRigidity(Dx, Dy, 0.0, 0.5 * R_H * np.sqrt(Dx * Dy))
